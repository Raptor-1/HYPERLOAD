r"""
HYPERLOAD — Astrometric Distortion Mapper
==========================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Measures and corrects full-field optical distortion using SIP polynomial
fitting against the Gaia DR3 reference catalog.

Pipeline:
  1. Star detection via DAOStarFinder          (Stetson 1987)
  2. Gaia DR3 cone-search via astroquery       (Gaia Collaboration 2022)
  3. WCS cross-match: detected ↔ catalog
  4. Residual vector field: Δx, Δy per star
  5. SIP polynomial fit + sigma clipping       (Shupe et al. 2005)
  6. Inverse SIP coefficients (AP/BP)          (Calabretta & Greisen 2002)
  7. Distortion map: quiver + heatmap
  8. Corrected FITS output with SIP header
  9. Optional pixel resampling                 (Anderson & King 2003)

References:
  Shupe et al. 2005        — SIP convention, ADASS XIV Proc.
  Calabretta & Greisen 2002— FITS WCS standard, A&A 395, 1077
  Bertin 2010              — SCAMP astrometric calibration
  Anderson & King 2003     — HST/ACS geometric distortion, PASP 115, 113
  Gaia Collaboration 2022  — Gaia DR3, A&A 649, A1
  Stetson 1987             — DAOPhot, PASP 99, 191

Version: 1.0.0
Project: HYPERLOAD
"""

import sys
import os
import traceback
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ── crash logger ─────────────────────────────────────────────────────────────
def _crash_handler(exc_type, exc_val, exc_tb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\n"
                f"astrometric_distortion_mapper.py\n")
        traceback.print_exception(exc_type, exc_val, exc_tb, file=f)
    print(f"[CRASH] See {log}")
    sys.__excepthook__(exc_type, exc_val, exc_tb)

sys.excepthook = _crash_handler

# ── sirilpy (optional — graceful no-op if not running inside Siril) ──────────
try:
    import sirilpy as s
    s.ensure_installed("PyQt6")
    s.ensure_installed("astropy")
    s.ensure_installed("astroquery")
    s.ensure_installed("photutils")
    s.ensure_installed("scipy")
    s.ensure_installed("matplotlib")
    SIRIL_ENV = True
except ImportError:
    SIRIL_ENV = False

# ── scipy ─────────────────────────────────────────────────────────────────────
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── astropy ───────────────────────────────────────────────────────────────────
try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.stats import sigma_clipped_stats
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

# ── astroquery ────────────────────────────────────────────────────────────────
try:
    from astroquery.gaia import Gaia
    GAIA_OK = True
except ImportError:
    GAIA_OK = False

# ── photutils ─────────────────────────────────────────────────────────────────
try:
    from photutils.detection import DAOStarFinder
    from photutils.background import Background2D, MedianBackground
    PHOTUTILS_OK = True
except ImportError:
    PHOTUTILS_OK = False

# ── PyQt6 ─────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QMessageBox, QSizePolicy, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QFont, QColor, QPixmap, QIcon

# ── matplotlib ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Astrometric Distortion Mapper"

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHM CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class SIPFitter:
    """
    Fit SIP (Simple Imaging Polynomial) distortion coefficients.

    SIP forward transform (Shupe et al. 2005):
        u = x_pixel - CRPIX1,   v = y_pixel - CRPIX2
        x_ideal = u + A(u,v)    where A = Σ A_p_q · u^p · v^q
        y_ideal = v + B(u,v)    where B = Σ B_p_q · u^p · v^q

    Inverse (AP / BP) fitted as approximate polynomial from:
        u_corrected → u_original
        v_corrected → v_original

    Polynomial order 2–7 supported; order 4–5 recommended for most optics.
    Constant and linear terms are intentionally excluded from the SIP
    polynomial (they live in the CD matrix / CRPIX).
    """

    def __init__(self, order: int = 4):
        self.order = order
        self.A_coeffs  = {}   # forward correction in x
        self.B_coeffs  = {}   # forward correction in y
        self.AP_coeffs = {}   # inverse correction in x
        self.BP_coeffs = {}   # inverse correction in y
        self.terms     = []   # (p, q) index pairs built during fit
        self.rms_x     = None
        self.rms_y     = None
        self.rms_total = None
        self.n_stars   = 0
        self.n_clipped = 0

    # ── design matrix ─────────────────────────────────────────────────────────
    def _build_design_matrix(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Build the Vandermonde-style design matrix for SIP fitting.
        Columns: u^p * v^q for all (p,q) with p+q ≥ 1 and p+q ≤ order.
        The (0,0) constant term is excluded — handled by CRPIX/CD matrix.
        """
        cols  = []
        terms = []
        for p in range(self.order + 1):
            for q in range(self.order + 1 - p):
                if p + q < 1:          # skip constant
                    continue
                cols.append(u**p * v**q)
                terms.append((p, q))
        self.terms = terms
        return np.column_stack(cols)   # shape (N, n_terms)

    # ── main fit ──────────────────────────────────────────────────────────────
    def fit(self, u: np.ndarray, v: np.ndarray,
            dx: np.ndarray, dy: np.ndarray,
            sigma_clip: float = 3.0, max_iter: int = 5) -> np.ndarray:
        """
        Fit SIP coefficients with iterative sigma-clipping.

        Parameters
        ----------
        u, v      : pixel offsets from CRPIX  (x - CRPIX1, y - CRPIX2)
        dx, dy    : residuals in pixels  (catalog_pixel - WCS_predicted)
        sigma_clip: rejection threshold in sigma units
        max_iter  : maximum rejection iterations

        Returns
        -------
        inlier_mask : boolean array, True = star used in final fit
        """
        u  = np.asarray(u,  dtype=float)
        v  = np.asarray(v,  dtype=float)
        dx = np.asarray(dx, dtype=float)
        dy = np.asarray(dy, dtype=float)

        mask = np.ones(len(u), dtype=bool)

        for _iteration in range(max_iter):
            A_mat = self._build_design_matrix(u[mask], v[mask])

            # Ordinary least squares — independent for x and y
            cx, *_ = np.linalg.lstsq(A_mat, dx[mask], rcond=None)
            cy, *_ = np.linalg.lstsq(A_mat, dy[mask], rcond=None)

            # Predict over masked subset
            res_dx = dx[mask] - A_mat @ cx
            res_dy = dy[mask] - A_mat @ cy
            dist   = np.sqrt(res_dx**2 + res_dy**2)
            rms    = np.std(dist)

            # Build full rejection mask
            new_mask           = mask.copy()
            new_mask[mask]     = dist < sigma_clip * rms
            n_rejected_this    = np.sum(mask) - np.sum(new_mask)

            mask = new_mask
            if n_rejected_this == 0:
                break   # converged

        # ── final fit on cleaned data ──────────────────────────────────────
        A_final = self._build_design_matrix(u[mask], v[mask])
        cx, *_ = np.linalg.lstsq(A_final, dx[mask], rcond=None)
        cy, *_ = np.linalg.lstsq(A_final, dy[mask], rcond=None)

        # Store as dicts  {(p,q): value}
        self.A_coeffs = {pq: cx[i] for i, pq in enumerate(self.terms)}
        self.B_coeffs = {pq: cy[i] for i, pq in enumerate(self.terms)}

        # ── fit statistics ────────────────────────────────────────────────
        res_dx_f = dx[mask] - A_final @ cx
        res_dy_f = dy[mask] - A_final @ cy
        self.rms_x     = float(np.std(res_dx_f))
        self.rms_y     = float(np.std(res_dy_f))
        self.rms_total = float(np.sqrt(self.rms_x**2 + self.rms_y**2))
        self.n_stars   = int(np.sum(mask))
        self.n_clipped = int(len(mask) - np.sum(mask))

        # ── inverse SIP ──────────────────────────────────────────────────
        self._fit_inverse(u[mask], v[mask], cx, cy)

        return mask

    def _fit_inverse(self, u: np.ndarray, v: np.ndarray,
                     cx: np.ndarray, cy: np.ndarray) -> None:
        """
        Fit approximate inverse SIP (AP, BP) coefficients.

        Forward:  (u, v) → (u + A(u,v),  v + B(u,v)) = (u', v')
        Inverse:   given (u', v'), recover (u, v)
        Approach:  fit polynomial  u = u' + AP(u', v') over the sample
        """
        A_mat      = self._build_design_matrix(u, v)
        u_corr     = u + A_mat @ cx     # = u' ≈ ideal coords
        v_corr     = v + A_mat @ cy

        A_inv = self._build_design_matrix(u_corr, v_corr)
        cap, *_ = np.linalg.lstsq(A_inv, u - u_corr, rcond=None)
        cbp, *_ = np.linalg.lstsq(A_inv, v - v_corr, rcond=None)

        self.AP_coeffs = {pq: cap[i] for i, pq in enumerate(self.terms)}
        self.BP_coeffs = {pq: cbp[i] for i, pq in enumerate(self.terms)}

    # ── evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, u: np.ndarray, v: np.ndarray):
        """
        Evaluate forward SIP distortion at (u, v) = (x - CRPIX1, y - CRPIX2).
        Returns (dx, dy) in pixels.
        """
        dx = np.zeros_like(u, dtype=float)
        dy = np.zeros_like(v, dtype=float)
        for (p, q), c in self.A_coeffs.items():
            dx += c * u**p * v**q
        for (p, q), c in self.B_coeffs.items():
            dy += c * u**p * v**q
        return dx, dy

    # ── FITS header I/O ───────────────────────────────────────────────────────
    def write_to_header(self, header) -> None:
        """
        Inject SIP coefficients into a FITS header (in-place).
        Appends '-SIP' to CTYPE1/CTYPE2 if not already present.
        """
        for key in ('CTYPE1', 'CTYPE2'):
            val = header.get(key, '')
            if '-SIP' not in val:
                header[key] = val + '-SIP'

        header['A_ORDER']  = (self.order, 'SIP forward polynomial order X')
        header['B_ORDER']  = (self.order, 'SIP forward polynomial order Y')
        header['AP_ORDER'] = (self.order, 'SIP inverse polynomial order X')
        header['BP_ORDER'] = (self.order, 'SIP inverse polynomial order Y')

        for (p, q), val in self.A_coeffs.items():
            header[f'A_{p}_{q}']  = float(val)
        for (p, q), val in self.B_coeffs.items():
            header[f'B_{p}_{q}']  = float(val)
        for (p, q), val in self.AP_coeffs.items():
            header[f'AP_{p}_{q}'] = float(val)
        for (p, q), val in self.BP_coeffs.items():
            header[f'BP_{p}_{q}'] = float(val)

        # HYPERLOAD provenance
        header['HIERARCH HYPERLOAD_DIST']   = (True,               'Distortion corrected')
        header['HIERARCH HYPERLOAD_RMS']    = (round(self.rms_total, 5), 'Fit RMS px')
        header['HIERARCH HYPERLOAD_NSTAR']  = (self.n_stars,       'Stars used for SIP fit')
        header['HIERARCH HYPERLOAD_ORDER']  = (self.order,         'SIP polynomial order')
        header['HIERARCH HYPERLOAD_DATE']   = (datetime.utcnow().isoformat()+'Z', 'Correction UTC')

    def summary_dict(self) -> dict:
        """Return fit statistics as a plain dict (for JSON export)."""
        return {
            'order':     self.order,
            'n_stars':   self.n_stars,
            'n_clipped': self.n_clipped,
            'rms_x_px':  self.rms_x,
            'rms_y_px':  self.rms_y,
            'rms_total_px': self.rms_total,
            'A_coeffs':  {str(k): v for k, v in self.A_coeffs.items()},
            'B_coeffs':  {str(k): v for k, v in self.B_coeffs.items()},
        }


# ──────────────────────────────────────────────────────────────────────────────

class StarDetector:
    """
    DAOStarFinder-based star detection with automatic background subtraction.

    Background is estimated on a 64×64-block grid using the median estimator
    (robust to nebulosity patches). Falls back to global sigma-clipped stats
    if Background2D fails (e.g. very small images).
    """

    def __init__(self, fwhm: float = 3.5, threshold_sigma: float = 5.0,
                 sharplo: float = 0.2, sharphi: float = 1.0,
                 roundlo: float = -1.0, roundhi: float = 1.0):
        self.fwhm            = fwhm
        self.threshold_sigma = threshold_sigma
        self.sharplo         = sharplo
        self.sharphi         = sharphi
        self.roundlo         = roundlo
        self.roundhi         = roundhi

    def detect(self, data: np.ndarray, mask: np.ndarray = None):
        """
        Detect stars.  Returns astropy Table with xcentroid, ycentroid, flux,
        or None if nothing found.
        """
        if not PHOTUTILS_OK:
            raise ImportError("photutils is required.  "
                              "pip install photutils")

        data_f = data.astype(float)

        # ── background estimation ─────────────────────────────────────────
        try:
            bkg = Background2D(data_f, (64, 64), filter_size=(3, 3),
                               bkg_estimator=MedianBackground(), mask=mask)
            data_sub  = data_f - bkg.background
            threshold = self.threshold_sigma * bkg.background_rms_median
        except Exception:
            _, median, std = sigma_clipped_stats(data_f, sigma=3.0, maxiters=5)
            data_sub  = data_f - median
            threshold = self.threshold_sigma * std

        finder  = DAOStarFinder(
            fwhm=self.fwhm, threshold=threshold,
            sharplo=self.sharplo, sharphi=self.sharphi,
            roundlo=self.roundlo, roundhi=self.roundhi,
        )
        sources = finder(data_sub, mask=mask)
        return sources


# ──────────────────────────────────────────────────────────────────────────────

class CatalogMatcher:
    """
    Cross-match detected stars against the Gaia DR3 astrometric catalog.

    Query strategy:
    - Determine field centre and bounding radius from WCS + image shape
    - Cone search via astroquery.gaia (async job)
    - Quality filter: phot_g_mean_mag < mag_limit, astrometric_excess_noise < 1.0,
      duplicated_source = False
    - Positional match: nearest-neighbour within match_radius arcsec

    Returns paired pixel arrays (det_x, det_y) ↔ (cat_x, cat_y) and
    residual vectors (dx, dy) in pixels.
    """

    def __init__(self, match_radius_arcsec: float = 3.0,
                 mag_limit: float = 17.0):
        self.match_radius = match_radius_arcsec * u.arcsec
        self.mag_limit    = mag_limit

    # ── Gaia query ────────────────────────────────────────────────────────────
    def query_gaia(self, wcs, shape: tuple, log=None) -> object:
        """
        Cone search Gaia DR3.  Returns an astropy Table.
        log: optional callable(str) for progress messages.
        """
        if not GAIA_OK:
            raise ImportError("astroquery not found.  pip install astroquery")

        ny, nx = shape
        center = wcs.pixel_to_world(nx / 2, ny / 2)

        # Corner coords → bounding radius
        px = [0, nx, 0, nx];  py = [0, 0, ny, ny]
        corners = SkyCoord(*wcs.pixel_to_world_values(px, py),
                           unit=u.deg, frame='icrs')
        radius  = (center.separation(corners).max() * 1.15)

        if log:
            log(f"Gaia DR3 cone search: "
                f"RA={center.ra.deg:.4f}°  Dec={center.dec.deg:+.4f}°  "
                f"r={radius.deg:.3f}°")

        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = 5000

        job   = Gaia.cone_search_async(
            coordinate=center, radius=radius,
            columns=['source_id', 'ra', 'dec', 'phot_g_mean_mag',
                     'astrometric_excess_noise', 'duplicated_source'],
        )
        table = job.get_results()

        filt  = (
            (table['phot_g_mean_mag'] < self.mag_limit) &
            (table['astrometric_excess_noise'] < 1.0)   &
            (~table['duplicated_source'])
        )
        filtered = table[filt]

        if log:
            log(f"Gaia returned {len(table)} raw, "
                f"{len(filtered)} after quality filter  "
                f"(G < {self.mag_limit}, excess_noise < 1)")
        return filtered

    # ── matching ──────────────────────────────────────────────────────────────
    def match(self, detected_sources, catalog_table, wcs, log=None) -> dict:
        """
        Nearest-neighbour match between detected and catalog stars.

        Returns dict with keys:
            det_x, det_y   — detected pixel coords
            cat_x, cat_y   — catalog pixel coords (via WCS)
            dx, dy         — det - cat  pixel residuals
            sep_arcsec     — angular separation of matched pairs
            mag            — Gaia G magnitude of catalog match
            n_matched      — number of pairs within match_radius
        """
        det_ra, det_dec = wcs.all_pix2world(
            np.array(detected_sources['xcentroid']),
            np.array(detected_sources['ycentroid']), 0)
        det_sky = SkyCoord(ra=det_ra*u.deg, dec=det_dec*u.deg)

        cat_sky = SkyCoord(ra=np.array(catalog_table['ra'])*u.deg,
                           dec=np.array(catalog_table['dec'])*u.deg)

        idx, sep2d, _ = det_sky.match_to_catalog_sky(cat_sky)
        good = sep2d < self.match_radius

        det_x_m = np.array(detected_sources['xcentroid'])[good]
        det_y_m = np.array(detected_sources['ycentroid'])[good]

        cat_x_m, cat_y_m = wcs.all_world2pix(
            np.array(catalog_table['ra'])[idx[good]],
            np.array(catalog_table['dec'])[idx[good]], 0)

        dx  = det_x_m - cat_x_m
        dy  = det_y_m - cat_y_m
        mag = np.array(catalog_table['phot_g_mean_mag'])[idx[good]]

        n = int(np.sum(good))
        if log:
            med_sep = float(np.median(sep2d[good].arcsec)) if n else float('nan')
            log(f"Matched {n}/{len(detected_sources)} stars  "
                f"(median sep = {med_sep:.3f}\")")

        return {
            'det_x':      det_x_m, 'det_y':      det_y_m,
            'cat_x':      cat_x_m, 'cat_y':      cat_y_m,
            'dx':         dx,       'dy':         dy,
            'sep_arcsec': sep2d[good].arcsec,
            'mag':        mag,
            'n_matched':  n,
        }


# ──────────────────────────────────────────────────────────────────────────────

class DistortionCorrector:
    """
    Apply SIP distortion correction by remapping pixel values.

    Approach (Anderson & King 2003):
    - For each output pixel (x_out, y_out), compute its source location
      by subtracting the SIP distortion vector:
          x_src = x_out − A(u, v)
          y_src = y_out − B(u, v)
      where u = x_out − CRPIX1, v = y_out − CRPIX2
    - Resample with scipy.ndimage.map_coordinates (bicubic by default)

    Note: this is a forward-model inversion.  For highest precision a
    Newton–Raphson iteration could be used, but for distortions < 20 px
    the single-pass subtraction is accurate to < 0.01 px.
    """

    def __init__(self, fitter: SIPFitter):
        self.fitter = fitter

    def build_distortion_map(self, shape: tuple, crpix: list):
        """Return (dx, dy) arrays covering the full image, one value per pixel."""
        ny, nx = shape
        xx, yy = np.meshgrid(np.arange(nx, dtype=float),
                             np.arange(ny, dtype=float))
        u = xx - crpix[0]
        v = yy - crpix[1]
        return self.fitter.evaluate(u, v)   # (dx, dy)

    def correct_image(self, data: np.ndarray, crpix: list, interp_order: int = 3):
        """
        Remap image data to remove distortion.

        Parameters
        ----------
        data         : 2-D or 3-D (C, H, W) float array
        crpix        : [CRPIX1, CRPIX2]  (0-indexed: header value − 1)
        interp_order : spline order  (3 = bicubic, 1 = bilinear)

        Returns corrected array with same dtype as input.
        """
        if data.ndim == 3:
            shape2d = data.shape[1:]
        else:
            shape2d = data.shape

        dx, dy = self.build_distortion_map(shape2d, crpix)
        ny, nx = shape2d

        xx, yy = np.meshgrid(np.arange(nx, dtype=float),
                             np.arange(ny, dtype=float))
        src_x = np.clip(xx - dx, 0, nx - 1)
        src_y = np.clip(yy - dy, 0, ny - 1)
        coords = np.array([src_y.ravel(), src_x.ravel()])

        orig_dtype = data.dtype

        def _remap_channel(ch):
            return map_coordinates(ch, coords, order=interp_order,
                                   mode='reflect').reshape(ny, nx)

        if data.ndim == 3:
            corrected = np.stack([_remap_channel(data[c].astype(float))
                                  for c in range(data.shape[0])], axis=0)
        else:
            corrected = _remap_channel(data.astype(float))

        return corrected.astype(orig_dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER THREADS
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisWorker(QObject):
    """
    Background thread: detect stars → query Gaia → match → fit SIP.
    Emits progress(pct, msg), finished(result_dict), error(msg).
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, fits_path: str, params: dict):
        super().__init__()
        self.fits_path = fits_path
        self.params    = params
        self._cancel   = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _emit(self, pct: int, msg: str):
        self.progress.emit(pct, msg)

    def _run_pipeline(self):
        p = self.params

        # ── 1. Load FITS ──────────────────────────────────────────────────
        self._emit(5, "Loading FITS…")
        with fits.open(self.fits_path) as hdul:
            header = hdul[0].header.copy()
            raw    = hdul[0].data
            if raw is None:
                # Try extension 1
                raw    = hdul[1].data
                header = hdul[1].header.copy()
            data = raw.astype(float)

        # Extract 2D luminance
        if data.ndim == 3:
            data_2d = data.mean(axis=0) if data.shape[0] <= 4 else data[0]
        else:
            data_2d = data

        # ── 2. Parse WCS ──────────────────────────────────────────────────
        self._emit(10, "Parsing WCS…")
        wcs = WCS(header, naxis=2)
        if not wcs.has_celestial:
            self.error.emit(
                "No valid celestial WCS found in FITS header.\n\n"
                "Solve astrometry first (astrometry.net, ASTAP, or Siril's "
                "built-in plate-solver) then run this tool.")
            return

        crpix = [float(header.get('CRPIX1', data_2d.shape[1] / 2)),
                 float(header.get('CRPIX2', data_2d.shape[0] / 2))]
        # Convert from FITS 1-indexed to 0-indexed
        crpix0 = [crpix[0] - 1.0, crpix[1] - 1.0]

        # ── 3. Star detection ─────────────────────────────────────────────
        self._emit(20, "Detecting stars…")
        detector = StarDetector(
            fwhm=p['fwhm'],
            threshold_sigma=p['threshold'],
            sharplo=p.get('sharplo', 0.2),
            sharphi=p.get('sharphi', 1.0),
        )
        sources = detector.detect(data_2d)

        n_det = len(sources) if sources is not None else 0
        if n_det < 10:
            self.error.emit(
                f"Only {n_det} stars detected.\n"
                "Lower the threshold sigma or increase the FWHM estimate.")
            return
        self._emit(30, f"Detected {n_det} stars")

        if self._cancel: return

        # ── 4. Gaia query ─────────────────────────────────────────────────
        self._emit(40, "Querying Gaia DR3…  (may take 10–30 s)")

        def gaia_log(msg):
            self._emit(-1, msg)   # -1 = log only, no progress bar update

        matcher = CatalogMatcher(
            match_radius_arcsec=p['match_radius'],
            mag_limit=p['mag_limit'],
        )
        try:
            catalog = matcher.query_gaia(wcs, data_2d.shape, log=gaia_log)
        except Exception as ex:
            self.error.emit(
                f"Gaia query failed:\n{ex}\n\n"
                "Check internet connection and astroquery installation.")
            return

        if len(catalog) < 20:
            self.error.emit(
                f"Only {len(catalog)} Gaia stars in field.\n"
                "Try a brighter magnitude limit (e.g. 16–17) or check WCS.")
            return

        if self._cancel: return

        # ── 5. Cross-match ────────────────────────────────────────────────
        self._emit(58, f"Cross-matching {n_det} detected ↔ {len(catalog)} Gaia…")
        match = matcher.match(sources, catalog, wcs, log=gaia_log)

        if match['n_matched'] < p.get('min_matches', 15):
            self.error.emit(
                f"Only {match['n_matched']} pairs within {p['match_radius']}\".\n"
                "Increase match radius, lower mag limit, or reduce threshold.")
            return

        if self._cancel: return

        # ── 6. SIP fit ────────────────────────────────────────────────────
        self._emit(70, f"Fitting SIP order {p['sip_order']}…")
        fitter = SIPFitter(order=p['sip_order'])

        u = match['det_x'] - crpix0[0]
        v = match['det_y'] - crpix0[1]

        inlier_mask = fitter.fit(u, v, match['dx'], match['dy'],
                                 sigma_clip=p['sigma_clip'])

        self._emit(84,
            f"Fit done — RMS {fitter.rms_total:.4f} px  "
            f"({fitter.n_stars} stars, {fitter.n_clipped} clipped)")

        if self._cancel: return

        # ── 7. Build visualisation grids ──────────────────────────────────
        self._emit(90, "Building distortion maps…")
        corrector = DistortionCorrector(fitter)
        ny, nx    = data_2d.shape

        # Coarse quiver grid (≈40×40 arrows)
        step_q = max(1, min(ny, nx) // 40)
        gx_q   = np.arange(0, nx, step_q, dtype=float)
        gy_q   = np.arange(0, ny, step_q, dtype=float)
        GX_q, GY_q = np.meshgrid(gx_q, gy_q)
        DX_q, DY_q = fitter.evaluate(GX_q - crpix0[0], GY_q - crpix0[1])

        # Denser heatmap grid (≈200×200)
        step_h = max(1, min(ny, nx) // 200)
        gx_h   = np.arange(0, nx, step_h, dtype=float)
        gy_h   = np.arange(0, ny, step_h, dtype=float)
        GX_h, GY_h = np.meshgrid(gx_h, gy_h)
        DX_h, DY_h = fitter.evaluate(GX_h - crpix0[0], GY_h - crpix0[1])
        mag_h = np.sqrt(DX_h**2 + DY_h**2)

        # Pixel scale
        try:
            pix_scale = float(wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value)
        except Exception:
            pix_scale = None

        self._emit(100, "Done!")

        self.finished.emit({
            'header':       header,
            'data':         data,
            'data_2d':      data_2d,
            'wcs':          wcs,
            'shape':        (ny, nx),
            'crpix':        crpix,    # 1-indexed (FITS)
            'crpix0':       crpix0,   # 0-indexed (Python)
            'sources':      sources,
            'match':        match,
            'inlier_mask':  inlier_mask,
            'fitter':       fitter,
            'corrector':    corrector,
            'pix_scale':    pix_scale,
            'quiver_grid':  {'GX': GX_q, 'GY': GY_q,
                             'DX': DX_q, 'DY': DY_q},
            'heat_grid':    {'gx': gx_h, 'gy': gy_h, 'mag': mag_h},
        })


# ──────────────────────────────────────────────────────────────────────────────

class SaveWorker(QObject):
    """Save corrected FITS in background thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, result: dict, out_path: str,
                 write_sip: bool, resample: bool):
        super().__init__()
        self.result    = result
        self.out_path  = out_path
        self.write_sip = write_sip
        self.resample  = resample

    def run(self):
        try:
            r      = self.result
            header = r['header'].copy()
            data   = r['data']
            fitter = r['fitter']
            crpix0 = r['crpix0']

            self.progress.emit(15, "Preparing header…")
            if self.write_sip:
                fitter.write_to_header(header)

            if self.resample:
                self.progress.emit(35, "Resampling image (removing distortion)…")
                corrector = r['corrector']
                data = corrector.correct_image(data, crpix0)
                self.progress.emit(80, "Resampling complete.")

            self.progress.emit(88, "Writing FITS…")
            fits.PrimaryHDU(data, header=header).writeto(
                self.out_path, overwrite=True)

            self.progress.emit(100, f"Saved → {Path(self.out_path).name}")
            self.finished.emit(self.out_path)

        except Exception as ex:
            self.error.emit(f"{ex}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI — SMALL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow, QWidget {
    background-color: #11111e;
    color: #d0d0ea;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}
QTabWidget::pane  { border: 1px solid #252545; background: #15152a; }
QTabBar::tab      { background: #1a1a32; color: #7070a0;
                    padding: 6px 18px; border: 1px solid #252545;
                    border-bottom: none; }
QTabBar::tab:selected { background: #20204a; color: #a0a0ff;
                        border-bottom: 2px solid #5050c0; }
QGroupBox         { border: 1px solid #252545; border-radius: 4px;
                    margin-top: 8px; padding-top: 8px;
                    color: #7070a8; font-weight: bold; }
QGroupBox::title  { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton       { background: #1e1e40; color: #b0b0d8;
                    border: 1px solid #3535a0; border-radius: 4px;
                    padding: 5px 14px; }
QPushButton:hover   { background: #28285a; border-color: #5555c0; }
QPushButton:pressed { background: #181838; }
QPushButton:disabled{ color: #383860; border-color: #242444; }
QPushButton#run_btn { background: #28185a; color: #c0a0ff;
                      border: 1px solid #7050c0; font-weight: bold;
                      padding: 7px 20px; }
QPushButton#run_btn:hover { background: #382870; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #18182e; border: 1px solid #252545;
    border-radius: 3px; padding: 3px 6px; color: #d0d0ea; }
QLineEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus   { border-color: #5555c0; }
QTextEdit  { background: #0d0d1c; border: 1px solid #252545;
             color: #88e088; font-family: Consolas, 'Courier New', monospace;
             font-size: 11px; }
QTableWidget { background: #0d0d1c; alternate-background-color: #131326;
               gridline-color: #252545; color: #d0d0ea;
               selection-background-color: #282860; }
QTableWidget QHeaderView::section {
    background: #181830; color: #7070a8;
    border: 1px solid #252545; padding: 3px; font-weight: bold; }
QSplitter::handle  { background: #252545; }
QScrollArea        { border: none; }
QLabel#title_lbl   { font-size: 19px; font-weight: bold; color: #9878ff;
                      padding: 6px; }
QLabel#sub_lbl     { font-size: 11px; color: #545480;
                      padding: 0 8px 6px; }
"""


class MplCanvas(QWidget):
    """Thin wrapper around a matplotlib Figure embedded in Qt."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig    = Figure(figsize=(7, 5), facecolor='#1a1a2e')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def clear(self):
        self.fig.clear()
        self.canvas.draw_idle()

    def redraw(self):
        self.fig.tight_layout()
        self.canvas.draw_idle()


def _cbar(fig, mappable, ax):
    """Attach a slim colorbar without distorting the axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cb  = fig.colorbar(mappable, cax=cax)
    cb.ax.yaxis.set_tick_params(colors='#9090b0', labelsize=7)
    cb.outline.set_edgecolor('#303060')
    return cb


def _ax_style(ax, title='', xlabel='', ylabel=''):
    """Apply uniform dark style to a matplotlib Axes."""
    ax.set_facecolor('#0d0d1c')
    ax.set_title(title, color='#9090ff', fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color='#585888', fontsize=8)
    ax.set_ylabel(ylabel, color='#585888', fontsize=8)
    ax.tick_params(colors='#585888', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#252545')


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#0b0b16; border-top:1px solid #252545;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        self.lbl  = QLabel("Ready")
        self.lbl.setStyleSheet("color:#686898; font-size:10px;")
        self.pbar = QProgressBar()
        self.pbar.setFixedSize(180, 12)
        self.pbar.setVisible(False)
        self.pbar.setStyleSheet("""
            QProgressBar { background:#1a1a2e; border:1px solid #3535a0;
                           border-radius:3px; color:white; font-size:9px; }
            QProgressBar::chunk { background:qlineargradient(
                x1:0,y1:0,x2:1,y2:0, stop:0 #4040c0, stop:1 #9050e0);
                border-radius:2px; }""")
        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.pbar)

    def update(self, pct: int, msg: str):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True)
            self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(2500,
                    lambda: self.pbar.setVisible(False))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class AstrometricDistortionMapper(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1340, 870)
        self.setStyleSheet(DARK)

        # Load logo
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            self.setWindowIcon(QIcon(str(_logo)))

        self._result  = None    # filled by AnalysisWorker.finished
        self._worker  = None
        self._wthread = None

        self._build_ui()
        self._check_deps()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setSpacing(0)
        root_lay.setContentsMargins(0, 0, 0, 0)

        root_lay.addWidget(self._make_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._make_control_panel())
        splitter.addWidget(self._make_view_tabs())
        splitter.setSizes([300, 1040])
        root_lay.addWidget(splitter, 1)

        self.status = StatusFooter()
        root_lay.addWidget(self.status)

    # ── header banner ─────────────────────────────────────────────────────────
    def _make_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setFixedHeight(66)
        hdr.setStyleSheet("background:#0b0b16; border-bottom:1px solid #252545;")
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(12, 0, 12, 0)

        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                50, 50,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)

        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Astrometric Distortion Mapper");  t.setObjectName("title_lbl")
        s = QLabel(
            "SIP polynomial fit · Gaia DR3 reference · "
            "Full-field optical distortion measurement & correction")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s)
        lay.addLayout(v, 1)

        ver = QLabel(f"v{VERSION}  |  HYPERLOAD")
        ver.setStyleSheet("color:#303058; font-size:10px;")
        lay.addWidget(ver)
        return hdr

    # ── left control panel ────────────────────────────────────────────────────
    def _make_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(7)

        # ── input ─────────────────────────────────────────────────────────
        grp = QGroupBox("Input FITS")
        gl  = QVBoxLayout(grp)
        self.file_lbl = QLabel("No file selected")
        self.file_lbl.setStyleSheet("color:#484868; font-size:10px;")
        self.file_lbl.setWordWrap(True)
        btn_open = QPushButton("📂  Open FITS…")
        btn_open.clicked.connect(self._open_fits)
        gl.addWidget(self.file_lbl)
        gl.addWidget(btn_open)
        lay.addWidget(grp)

        # ── star detection ────────────────────────────────────────────────
        grp2 = QGroupBox("Star Detection  (DAOStarFinder)")
        g2   = QGridLayout(grp2)
        g2.addWidget(QLabel("FWHM (px):"),       0, 0)
        self.sp_fwhm = QDoubleSpinBox()
        self.sp_fwhm.setRange(1.0, 30.0); self.sp_fwhm.setValue(3.5)
        self.sp_fwhm.setSingleStep(0.5);  self.sp_fwhm.setDecimals(1)
        g2.addWidget(self.sp_fwhm, 0, 1)

        g2.addWidget(QLabel("Threshold (σ):"),   1, 0)
        self.sp_thresh = QDoubleSpinBox()
        self.sp_thresh.setRange(2.0, 30.0); self.sp_thresh.setValue(5.0)
        self.sp_thresh.setSingleStep(0.5);  self.sp_thresh.setDecimals(1)
        g2.addWidget(self.sp_thresh, 1, 1)

        g2.addWidget(QLabel("Sharpness min:"),   2, 0)
        self.sp_sharplo = QDoubleSpinBox()
        self.sp_sharplo.setRange(0.0, 1.0); self.sp_sharplo.setValue(0.2)
        self.sp_sharplo.setSingleStep(0.05); self.sp_sharplo.setDecimals(2)
        g2.addWidget(self.sp_sharplo, 2, 1)
        lay.addWidget(grp2)

        # ── Gaia catalog ──────────────────────────────────────────────────
        grp3 = QGroupBox("Gaia DR3 Catalog")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("G mag limit:"),     0, 0)
        self.sp_maglim = QDoubleSpinBox()
        self.sp_maglim.setRange(10.0, 21.0); self.sp_maglim.setValue(17.0)
        self.sp_maglim.setSingleStep(0.5);   self.sp_maglim.setDecimals(1)
        g3.addWidget(self.sp_maglim, 0, 1)

        g3.addWidget(QLabel("Match radius (\"):"), 1, 0)
        self.sp_match = QDoubleSpinBox()
        self.sp_match.setRange(0.5, 30.0); self.sp_match.setValue(3.0)
        self.sp_match.setSingleStep(0.5);  self.sp_match.setDecimals(1)
        g3.addWidget(self.sp_match, 1, 1)
        lay.addWidget(grp3)

        # ── SIP fit ───────────────────────────────────────────────────────
        grp4 = QGroupBox("SIP Polynomial Fit")
        g4   = QGridLayout(grp4)
        g4.addWidget(QLabel("Order (2–7):"),    0, 0)
        self.sp_order = QSpinBox()
        self.sp_order.setRange(2, 7); self.sp_order.setValue(4)
        g4.addWidget(self.sp_order, 0, 1)

        g4.addWidget(QLabel("Sigma clip:"),     1, 0)
        self.sp_sigclip = QDoubleSpinBox()
        self.sp_sigclip.setRange(2.0, 5.0); self.sp_sigclip.setValue(3.0)
        self.sp_sigclip.setSingleStep(0.5);  self.sp_sigclip.setDecimals(1)
        g4.addWidget(self.sp_sigclip, 1, 1)
        lay.addWidget(grp4)

        # ── quiver scale (live update) ────────────────────────────────────
        grp5 = QGroupBox("Visualisation")
        g5   = QGridLayout(grp5)
        g5.addWidget(QLabel("Quiver scale ×:"), 0, 0)
        self.sp_qscale = QSpinBox()
        self.sp_qscale.setRange(1, 100); self.sp_qscale.setValue(5)
        self.sp_qscale.valueChanged.connect(self._refresh_distortion_tab)
        g5.addWidget(self.sp_qscale, 0, 1)
        lay.addWidget(grp5)

        # ── output ────────────────────────────────────────────────────────
        grp6 = QGroupBox("Output")
        g6   = QVBoxLayout(grp6)
        self.cb_sip      = QCheckBox("Write SIP coefficients to header")
        self.cb_sip.setChecked(True)
        self.cb_resample = QCheckBox("Resample image (remove distortion pixels)")
        self.cb_resample.setChecked(False)
        self.cb_resample.setToolTip(
            "Remaps every pixel via bicubic interpolation.\n"
            "Slower but produces a geometrically correct image.")
        self.btn_save = QPushButton("💾  Save Corrected FITS…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_corrected)
        self.btn_json = QPushButton("📋  Export Fit Report (JSON)")
        self.btn_json.setEnabled(False)
        self.btn_json.clicked.connect(self._export_json)
        g6.addWidget(self.cb_sip)
        g6.addWidget(self.cb_resample)
        g6.addWidget(self.btn_save)
        g6.addWidget(self.btn_json)
        lay.addWidget(grp6)

        # ── run / cancel ──────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Run Distortion Analysis")
        self.btn_run.setObjectName("run_btn")
        self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)

        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("color:#b05050;")
        self.btn_cancel.clicked.connect(self._cancel)

        lay.addWidget(self.btn_run)
        lay.addWidget(self.btn_cancel)
        lay.addStretch()

        # ── result summary label ──────────────────────────────────────────
        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(
            "color:#70c070; font-size:10px; padding:4px;")
        self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)

        scroll.setWidget(inner)
        return scroll

    # ── right view tabs ───────────────────────────────────────────────────────
    def _make_view_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_detection(),   "⭐  Detection")
        self.tabs.addTab(self._tab_distortion(),  "📐  Distortion Map")
        self.tabs.addTab(self._tab_residuals(),   "📊  Residuals")
        self.tabs.addTab(self._tab_radial(),      "〇  Radial Profile")
        self.tabs.addTab(self._tab_log(),         "📋  Log")
        return self.tabs

    def _tab_detection(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        self.plot_det = MplCanvas()
        lay.addWidget(self.plot_det, 2)

        lay.addWidget(QLabel("Matched star pairs (first 500 shown):"))
        self.match_table = QTableWidget(0, 7)
        self.match_table.setHorizontalHeaderLabels(
            ["Det X", "Det Y", "Cat X", "Cat Y",
             "ΔX px", "ΔY px", "G mag"])
        self.match_table.setAlternatingRowColors(True)
        self.match_table.setFixedHeight(170)
        lay.addWidget(self.match_table)
        return w

    def _tab_distortion(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        cbar = QHBoxLayout()
        cbar.addWidget(QLabel("Display:"))
        self.cmb_dist = QComboBox()
        self.cmb_dist.addItems(
            ["Quiver (vectors)", "Heatmap (magnitude)", "Both"])
        self.cmb_dist.currentIndexChanged.connect(self._refresh_distortion_tab)
        cbar.addWidget(self.cmb_dist)
        cbar.addStretch()
        lay.addLayout(cbar)

        self.plot_dist = MplCanvas()
        lay.addWidget(self.plot_dist)
        return w

    def _tab_residuals(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        self.plot_res = MplCanvas()
        lay.addWidget(self.plot_res)
        return w

    def _tab_radial(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        self.plot_rad = MplCanvas()
        lay.addWidget(self.plot_rad)
        return w

    def _tab_log(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear")
        btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    # ─────────────────────────────────────────────────────────────────────────
    #  Dependency check
    # ─────────────────────────────────────────────────────────────────────────

    def _check_deps(self):
        missing = []
        if not ASTROPY_OK:   missing.append("astropy")
        if not GAIA_OK:      missing.append("astroquery")
        if not PHOTUTILS_OK: missing.append("photutils")
        if missing:
            self._log("⚠  Missing packages: " + ", ".join(missing))
            self._log("   Install with:  pip install " + " ".join(missing))
            self._log("   (Running inside Siril, sirilpy.ensure_installed "
                      "should handle this automatically)")
        else:
            self._log("✓  All dependencies found.")

    # ─────────────────────────────────────────────────────────────────────────
    #  File I/O
    # ─────────────────────────────────────────────────────────────────────────

    def _open_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FITS", "",
            "FITS (*.fits *.fit *.fts);;All (*)")
        if not path:
            return
        self.fits_path = path
        self.file_lbl.setText(Path(path).name)
        self.file_lbl.setStyleSheet("color:#70c070; font-size:10px;")
        self.btn_run.setEnabled(True)
        self._log(f"Loaded: {path}")

    def _save_corrected(self):
        if self._result is None:
            return
        src  = Path(self.fits_path)
        dflt = str(src.parent / (src.stem + "_distcorr.fits"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Corrected FITS", dflt, "FITS (*.fits)")
        if not path:
            return

        self.btn_save.setEnabled(False)
        t = QThread(self)
        w = SaveWorker(self._result, path,
                       self.cb_sip.isChecked(),
                       self.cb_resample.isChecked())
        w.moveToThread(t)
        t.started.connect(w.run)
        w.progress.connect(lambda p, m: self.status.update(p, m))
        w.finished.connect(lambda p: (
            self._log(f"✓  Saved: {p}"),
            self.btn_save.setEnabled(True)))
        w.error.connect(self._on_error)
        t.start()
        self._save_thread = t   # keep reference

    def _export_json(self):
        if self._result is None:
            return
        src  = Path(self.fits_path)
        dflt = str(src.parent / (src.stem + "_distortion_report.json"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Report", dflt, "JSON (*.json)")
        if not path:
            return
        r      = self._result
        fitter = r['fitter']
        report = {
            'file':          str(self.fits_path),
            'date_utc':      datetime.utcnow().isoformat() + 'Z',
            'hyperload_ver': VERSION,
            'fit':           fitter.summary_dict(),
            'match':  {
                'n_matched':  r['match']['n_matched'],
                'median_sep_arcsec': float(np.median(
                    r['match']['sep_arcsec'])),
            },
            'pixel_scale_arcsec': r['pix_scale'],
            'image_shape': list(r['shape']),
            'crpix_fits':  list(r['crpix']),
        }
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        self._log(f"✓  Report saved: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Analysis pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _run(self):
        if not hasattr(self, 'fits_path'):
            return

        params = {
            'fwhm':        self.sp_fwhm.value(),
            'threshold':   self.sp_thresh.value(),
            'sharplo':     self.sp_sharplo.value(),
            'sharphi':     1.0,
            'mag_limit':   self.sp_maglim.value(),
            'match_radius':self.sp_match.value(),
            'sip_order':   self.sp_order.value(),
            'sigma_clip':  self.sp_sigclip.value(),
            'min_matches': 15,
        }

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_json.setEnabled(False)
        self._result = None

        self._wthread = QThread(self)
        self._worker  = AnalysisWorker(self.fits_path, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)
        self.btn_run.setEnabled(True)
        self._log("Cancelled.")

    # ─────────────────────────────────────────────────────────────────────────
    #  Worker callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_progress(self, pct: int, msg: str):
        if pct >= 0:
            self.status.update(pct, msg)
            self._log(f"[{pct:3d}%]  {msg}")
        else:
            self._log(f"        {msg}")

    def _on_error(self, msg: str):
        self._log(f"\n❌  ERROR:\n{msg}")
        self.status.update(0, "Error — see Log tab")
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.critical(self, "Error", msg[:600])

    def _on_finished(self, result: dict):
        self._result = result
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)
        self.btn_json.setEnabled(True)

        f  = result['fitter']
        mr = result['match']
        ps = result['pix_scale']

        rms_as = f"({f.rms_total * ps:.4f}\")" if ps else ""

        self._log("\n" + "=" * 55)
        self._log(f"✓  SIP order {f.order} distortion fit complete")
        self._log(f"   Stars used   : {f.n_stars}")
        self._log(f"   Clipped      : {f.n_clipped}")
        self._log(f"   RMS X        : {f.rms_x:.5f} px")
        self._log(f"   RMS Y        : {f.rms_y:.5f} px")
        self._log(f"   RMS total    : {f.rms_total:.5f} px  {rms_as}")
        self._log(f"   Matched stars: {mr['n_matched']}")
        self._log(f"   Median sep   : {np.median(mr['sep_arcsec']):.3f} \"")
        if ps:
            self._log(f"   Pixel scale  : {ps:.4f} \"/px")
        self._log("=" * 55 + "\n")

        self.summary_lbl.setText(
            f"RMS {f.rms_total:.4f} px  |  "
            f"{f.n_stars} stars  |  order {f.order}")

        # Populate all tabs
        self._draw_detection_tab()
        self._refresh_distortion_tab()
        self._draw_residuals_tab()
        self._draw_radial_tab()
        self._fill_match_table()
        self.tabs.setCurrentIndex(1)   # jump to distortion map

    # ─────────────────────────────────────────────────────────────────────────
    #  Plot: Detection & match overview
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_detection_tab(self):
        r    = self._result
        fig  = self.plot_det.fig
        fig.clear()
        fig.patch.set_facecolor('#1a1a2e')
        ax   = fig.add_subplot(111)
        _ax_style(ax, "Detected Stars & Gaia Cross-matches", "X (px)", "Y (px)")

        d2   = r['data_2d']
        vlo, vhi = np.percentile(d2, [0.5, 99.5])
        ax.imshow(d2, origin='lower', cmap='gray', vmin=vlo, vmax=vhi,
                  aspect='auto', interpolation='nearest')

        src  = r['sources']
        mr   = r['match']
        mask = r['inlier_mask']

        # All detected (faint blue)
        ax.scatter(src['xcentroid'], src['ycentroid'],
                   s=6, c='#3838a0', marker='.', alpha=0.5,
                   label=f"Detected ({len(src)})")

        # Inliers (bright green +)
        ax.scatter(mr['det_x'][mask], mr['det_y'][mask],
                   s=22, c='#55ff88', marker='+', linewidths=0.8,
                   alpha=0.85, label=f"Inliers ({mask.sum()})")

        # Clipped (red ×)
        if not np.all(mask):
            ax.scatter(mr['det_x'][~mask], mr['det_y'][~mask],
                       s=22, c='#ff5040', marker='x', linewidths=0.8,
                       alpha=0.7, label=f"Clipped ({(~mask).sum()})")

        ax.legend(loc='upper right', fontsize=8,
                  facecolor='#1a1a2e', edgecolor='#3030a0',
                  labelcolor='white')
        self.plot_det.redraw()

    # ─────────────────────────────────────────────────────────────────────────
    #  Plot: Distortion map (quiver + heatmap)
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_distortion_tab(self):
        """Redraws distortion plot; called on mode/scale change."""
        if self._result is None:
            return
        mode  = self.cmb_dist.currentIndex()   # 0=quiver 1=heat 2=both
        scale = self.sp_qscale.value()
        r     = self._result
        fig   = self.plot_dist.fig
        fig.clear()
        fig.patch.set_facecolor('#1a1a2e')

        qg = r['quiver_grid']
        hg = r['heat_grid']

        if mode == 0:
            ax = fig.add_subplot(111)
            self._quiver_ax(ax, qg, scale)
        elif mode == 1:
            ax = fig.add_subplot(111)
            self._heat_ax(ax, hg, r['shape'])
        else:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            self._quiver_ax(ax1, qg, scale)
            self._heat_ax(ax2, hg, r['shape'])

        self.plot_dist.redraw()

    def _quiver_ax(self, ax, qg, scale):
        _ax_style(ax, f"Distortion Vector Field  (×{scale})",
                  "X (px)", "Y (px)")
        mag = np.sqrt(qg['DX']**2 + qg['DY']**2)
        q   = ax.quiver(qg['GX'], qg['GY'],
                        qg['DX'] * scale, qg['DY'] * scale,
                        mag, cmap='plasma', alpha=0.88,
                        scale=None, scale_units='xy', angles='xy',
                        width=0.002, headwidth=4, headlength=5)
        cb  = _cbar(ax.get_figure(), q, ax)
        cb.set_label("Distortion (px)", color='#8888b0', fontsize=7)

        qk_val = float(np.percentile(mag, 90))
        ax.quiverkey(q, 0.84, 1.025, qk_val * scale,
                     f'{qk_val:.2f} px', labelpos='E',
                     color='white', labelcolor='#9090b0',
                     fontproperties={'size': 7})

    def _heat_ax(self, ax, hg, shape):
        _ax_style(ax, "Distortion Magnitude Map", "X (px)", "Y (px)")
        im = ax.pcolormesh(hg['gx'], hg['gy'], hg['mag'],
                           cmap='inferno', shading='auto')
        ax.set_aspect('equal')
        cb = _cbar(ax.get_figure(), im, ax)
        cb.set_label("Distortion (px)", color='#8888b0', fontsize=7)

    # ─────────────────────────────────────────────────────────────────────────
    #  Plot: Residuals — 3-panel
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_residuals_tab(self):
        r    = self._result
        mr   = r['match']
        mask = r['inlier_mask']
        f    = r['fitter']
        c0   = r['crpix0']

        fig  = self.plot_res.fig
        fig.clear()
        fig.patch.set_facecolor('#1a1a2e')
        axes = fig.subplots(1, 3)

        # ── panel 1: raw residual scatter ─────────────────────────────────
        ax = axes[0]
        _ax_style(ax, "Raw ΔX / ΔY", "ΔX (px)", "ΔY (px)")
        ax.scatter(mr['dx'][mask],  mr['dy'][mask],
                   s=8, c='#6080ff', alpha=0.55, label='inliers')
        if not np.all(mask):
            ax.scatter(mr['dx'][~mask], mr['dy'][~mask],
                       s=8, c='#ff5030', alpha=0.4, marker='x',
                       label='clipped')
        lim = max(np.abs(mr['dx']).max(), np.abs(mr['dy']).max()) * 1.1
        ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
        ax.axhline(0, color='#3030a0', lw=0.5)
        ax.axvline(0, color='#3030a0', lw=0.5)
        ax.legend(fontsize=7, facecolor='#1a1a2e',
                  edgecolor='#3030a0', labelcolor='white')

        # ── panel 2: separation histogram ─────────────────────────────────
        ax2 = axes[1]
        _ax_style(ax2, "Match Separation", "Sep (arcsec)", "Count")
        sep = mr['sep_arcsec'][mask]
        ax2.hist(sep, bins=30, color='#5050b0', edgecolor='#3030a0',
                 alpha=0.85)
        med = np.median(sep)
        ax2.axvline(med, color='#ff8030', lw=1.5,
                    label=f'Median {med:.3f}"')
        ax2.legend(fontsize=7, facecolor='#1a1a2e',
                   edgecolor='#3030a0', labelcolor='white')

        # ── panel 3: post-fit residuals ────────────────────────────────────
        ax3 = axes[2]
        u   = mr['det_x'] - c0[0]
        v   = mr['det_y'] - c0[1]
        pdx, pdy   = f.evaluate(u, v)
        rdx = mr['dx'] - pdx
        rdy = mr['dy'] - pdy
        rms = np.sqrt(np.std(rdx[mask])**2 + np.std(rdy[mask])**2)
        _ax_style(ax3, f"Post-fit Residuals\nRMS = {rms:.4f} px",
                  "ΔX (px)", "ΔY (px)")
        ax3.scatter(rdx[mask], rdy[mask],
                    s=8, c='#55ff88', alpha=0.55)
        lim3 = max(np.abs(rdx[mask]).max(), np.abs(rdy[mask]).max()) * 1.2
        ax3.set_xlim(-lim3, lim3); ax3.set_ylim(-lim3, lim3)
        ax3.axhline(0, color='#3030a0', lw=0.5)
        ax3.axvline(0, color='#3030a0', lw=0.5)

        self.plot_res.redraw()

    # ─────────────────────────────────────────────────────────────────────────
    #  Plot: Radial profile — 2-panel
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_radial_tab(self):
        r    = self._result
        mr   = r['match']
        mask = r['inlier_mask']
        f    = r['fitter']
        c0   = r['crpix0']
        ny, nx = r['shape']

        fig  = self.plot_rad.fig
        fig.clear()
        fig.patch.set_facecolor('#1a1a2e')
        axes = fig.subplots(1, 2)

        # ── panel 1: scatter measured distortion vs radius ─────────────────
        ax = axes[0]
        _ax_style(ax, "Radial Distortion Profile",
                  "Radius from CRPIX (px)", "Distortion (px)")

        u_s = mr['det_x'][mask] - c0[0]
        v_s = mr['det_y'][mask] - c0[1]
        r_s = np.sqrt(u_s**2 + v_s**2)

        pdx, pdy = f.evaluate(u_s, v_s)
        dmag     = np.sqrt(pdx**2 + pdy**2)

        ax.scatter(r_s, dmag, s=7, c='#9060ff', alpha=0.6)

        # Savitzky-Golay smooth curve
        srt  = np.argsort(r_s)
        n_pts = len(r_s)
        if n_pts >= 11:
            win = min(21, (n_pts // 4) * 2 + 1)
            win = max(win, 5)
            if win % 2 == 0:
                win += 1
            if win <= n_pts:
                smooth = savgol_filter(dmag[srt], win, 3)
                ax.plot(r_s[srt], smooth, c='#ff8030', lw=2,
                        label='Smooth')
                ax.legend(fontsize=7, facecolor='#1a1a2e',
                          edgecolor='#3030a0', labelcolor='white')

        # ── panel 2: model along diagonal vs field fraction ────────────────
        ax2 = axes[1]
        _ax_style(ax2, "Distortion vs Field Fraction\n(diagonal direction)",
                  "Distance from centre (% of half-diagonal)", "Distortion (px)")

        field_r = np.sqrt((nx / 2)**2 + (ny / 2)**2)
        ang     = np.radians(45)
        r_lin   = np.linspace(0, field_r, 300)
        pdx2, pdy2 = f.evaluate(r_lin * np.cos(ang),
                                 r_lin * np.sin(ang))
        dmag2    = np.sqrt(pdx2**2 + pdy2**2)
        frac     = r_lin / field_r * 100

        ax2.plot(frac, dmag2, c='#55ff88', lw=2)
        ax2.fill_between(frac, dmag2, alpha=0.18, color='#55ff88')
        ax2.set_xlim(0, 100)

        # Mark corner
        ax2.axvline(100, color='#ff8030', lw=0.8, ls='--', alpha=0.6)
        ax2.text(98, dmag2[-1] * 0.9, 'corner',
                 color='#ff8030', fontsize=7, ha='right')

        self.plot_rad.redraw()

    # ─────────────────────────────────────────────────────────────────────────
    #  Match table
    # ─────────────────────────────────────────────────────────────────────────

    def _fill_match_table(self):
        r    = self._result
        mr   = r['match']
        mask = r['inlier_mask']
        n    = int(mask.sum())

        self.match_table.setRowCount(min(n, 500))
        indices = np.where(mask)[0]
        for row, i in enumerate(indices[:500]):
            vals = [
                f"{mr['det_x'][i]:.2f}", f"{mr['det_y'][i]:.2f}",
                f"{mr['cat_x'][i]:.2f}", f"{mr['cat_y'][i]:.2f}",
                f"{mr['dx'][i]:+.4f}",  f"{mr['dy'][i]:+.4f}",
                f"{mr['mag'][i]:.2f}",
            ]
            for col, val in enumerate(vals):
                it = QTableWidgetItem(val)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.match_table.setItem(row, col, it)

        self.match_table.resizeColumnsToContents()

    # ─────────────────────────────────────────────────────────────────────────
    #  Log
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.log.append(msg)
        self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AstrometricDistortionMapper()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
