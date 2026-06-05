r"""
HYPERLOAD — Drizzle Combiner
=============================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Sub-pixel image combination using the Variable Pixel Linear Reconstruction
(Drizzle) algorithm.  Combines a dithered stack of aligned frames into a
single output at user-chosen resolution while conserving flux and building
a proper weight/exposure map.

Core algorithm  (Fruchter & Hook 2002):
  For each input frame F_k at sub-pixel offset (Δx_k, Δy_k):
    Each input pixel (i, j) with value V drops a square footprint of
    size pixfrac × pixfrac (input pixel units) onto the output grid:
        x_out = i · scale + Δx_k · scale
        y_out = j · scale + Δy_k · scale
        d     = pixfrac · scale / 2          (half-width, output units)

    For every output pixel (I, J) the footprint overlaps:
        overlap = max(0, min(x_out+d, I+0.5) − max(x_out−d, I−0.5))
                × max(0, min(y_out+d, J+0.5) − max(y_out−d, J−0.5))
        sci[J,I] += overlap · V · w_k
        wht[J,I] += overlap · w_k
    Final:  combined = sci / wht

  Vectorised scatter-accumulation via np.bincount → O(N_in · k²)
  where k = ceil(pixfrac · scale) + 1 ≤ 4.

Shift estimation:
  Fourier phase-correlation with Hann windowing and parabolic
  sub-pixel refinement.  Accurate to ~0.1–0.2 px for typical SNR.

References:
  Fruchter & Hook 2002    — Drizzle, PASP 114, 144
  Gonzaga et al. 2012     — DrizzlePac Handbook (STScI)
  Lauer 1999              — Interlaced CCDs, PASP 111, 227
  Koekemoer et al. 2011   — MultiDrizzle/CANDELS, ApJS 197, 36
  Guizar-Sicairos 2008    — Sub-pixel registration, Opt. Lett. 33, 156

Version: 1.0.0
Project: HYPERLOAD
"""

import sys
import os
import traceback
import json
import math
import numpy as np
from pathlib import Path
from datetime import datetime

# ── crash logger ─────────────────────────────────────────────────────────────
def _crash(exc_type, exc_val, exc_tb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\ndrizzle_combiner.py\n")
        traceback.print_exception(exc_type, exc_val, exc_tb, file=f)
    print(f"[CRASH] See {log}")
    sys.__excepthook__(exc_type, exc_val, exc_tb)

sys.excepthook = _crash

# ── sirilpy (optional) ────────────────────────────────────────────────────────
try:
    import sirilpy as s
    s.ensure_installed("PyQt6")
    s.ensure_installed("astropy")
    s.ensure_installed("scipy")
    s.ensure_installed("matplotlib")
except ImportError:
    pass

# ── scipy ─────────────────────────────────────────────────────────────────────
from scipy.ndimage import shift as ndimage_shift, gaussian_filter, uniform_filter
from scipy.signal import savgol_filter

# ── astropy ───────────────────────────────────────────────────────────────────
try:
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.wcs import WCS
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

# ── PyQt6 ─────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QMessageBox, QSizePolicy, QScrollArea, QFrame, QSlider,
    QAbstractItemView, QHeaderView, QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QFont, QPixmap, QIcon, QColor

# ── matplotlib ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Drizzle Combiner"

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

def _cos_taper_1d(n: int, pct: float = 0.20) -> np.ndarray:
    """
    1-D cosine taper: flat centre, smoothly tapered over `pct` fraction at
    each end.  Less aggressive than a full Hann window — preserves more
    correlation signal while still suppressing edge discontinuities.
    Critical for accurate sub-pixel shift estimation (≤ 0.5 px regime).
    """
    w = np.ones(n)
    t = max(1, int(n * pct))
    taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(t) / t))
    w[:t]  = taper
    w[-t:] = taper[::-1]
    return w


def estimate_shift_fft(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """
    Estimate (dy, dx) such that img2 ≈ ndimage_shift(img1, [dy, dx]).

    Method: Fourier phase-correlation with 20 % cosine-taper window and
    parabolic sub-pixel peak refinement (Guizar-Sicairos 2008).

    The partial taper (vs full Hann) significantly improves accuracy for
    sub-pixel shifts while maintaining robustness for large shifts.

    Returns (dy, dx) in pixels  (float).
    """
    from numpy.fft import fft2, ifft2

    a = img1.astype(float)
    b = img2.astype(float)

    # 20% cosine taper in each axis
    wy  = _cos_taper_1d(a.shape[0])[:, None]
    wx  = _cos_taper_1d(a.shape[1])[None, :]
    win = wy * wx

    F1 = fft2(a * win)
    F2 = fft2(b * win)

    # Cross-power spectrum: conj(F1)·F2  →  peak at shift of img2 w.r.t. img1
    R     = np.conj(F1) * F2
    denom = np.abs(R)
    denom[denom < 1e-10] = 1.0
    R    /= denom

    r     = np.real(ifft2(R))
    ny_s, nx_s = r.shape

    pj, pi = np.unravel_index(np.argmax(r), r.shape)

    # Convert FFT index to signed shift (wrap around Nyquist)
    dy = float(pj if pj < ny_s // 2 else pj - ny_s)
    dx = float(pi if pi < nx_s // 2 else pi - nx_s)

    # Parabolic sub-pixel refinement
    ry = np.array([r[(pj - 1) % ny_s, pi],
                   r[pj, pi],
                   r[(pj + 1) % ny_s, pi]])
    rx = np.array([r[pj, (pi - 1) % nx_s],
                   r[pj, pi],
                   r[pj, (pi + 1) % nx_s]])
    d2y = ry[0] - 2 * ry[1] + ry[2]
    d2x = rx[0] - 2 * rx[1] + rx[2]
    if abs(d2y) > 1e-10:
        dy += 0.5 * (ry[0] - ry[2]) / d2y
    if abs(d2x) > 1e-10:
        dx += 0.5 * (rx[0] - rx[2]) / d2x

    return float(dy), float(dx)


def luminance_from_fits(data: np.ndarray) -> np.ndarray:
    """Return 2-D luminance from any FITS data shape."""
    if data.ndim == 2:
        return data.astype(float)
    if data.ndim == 3:
        # Shape (C, H, W) from Siril monochrome or RGB
        return data.mean(axis=0).astype(float)
    return data.astype(float)


def sigma_clip_mask(data: np.ndarray, sigma: float = 4.0,
                    maxiters: int = 5) -> np.ndarray:
    """Return boolean mask: True = pixel is outlier."""
    med = np.median(data)
    std = np.std(data)
    for _ in range(maxiters):
        mask = np.abs(data - med) > sigma * std
        if not mask.any():
            break
        med  = np.median(data[~mask])
        std  = np.std(data[~mask])
    return mask


# ──────────────────────────────────────────────────────────────────────────────

class DrizzleEngine:
    """
    Accumulate drizzled frames into a single output image.

    Implements the exact Fruchter & Hook 2002 square-kernel drizzle,
    vectorised via np.bincount (no per-pixel Python loops).

    Usage:
        eng = DrizzleEngine(ny_out, nx_out)
        for frame, dx, dy, w in stack:
            eng.add_frame(frame, dx, dy, scale, pixfrac, weight=w)
        sci, wht, exp = eng.finalize()
    """

    def __init__(self, ny_out: int, nx_out: int):
        self.ny_out = ny_out
        self.nx_out = nx_out
        self._sci = np.zeros(ny_out * nx_out, dtype=np.float64)
        self._wht = np.zeros(ny_out * nx_out, dtype=np.float64)
        self._exp = np.zeros(ny_out * nx_out, dtype=np.float64)
        self.n_frames = 0

    def add_frame(self, frame: np.ndarray,
                  dx_input: float, dy_input: float,
                  scale: float, pixfrac: float,
                  weight: float = 1.0,
                  exptime: float = 1.0,
                  bad_mask: np.ndarray = None):
        """
        Drizzle one frame onto the accumulator.

        Parameters
        ----------
        frame     : 2-D float image (ny_in, nx_in)
        dx_input  : x shift of this frame in INPUT pixel units
                    (positive = frame content moved right)
        dy_input  : y shift in input pixel units
        scale     : output/input pixel ratio  (2 = 2× resolution)
        pixfrac   : drop size as fraction of input pixel  (0 < pf ≤ 1)
        weight    : frame weight scalar
        exptime   : exposure time for exposure map (seconds)
        bad_mask  : 2-D bool, True = bad pixel (skipped)
        """
        ny_in, nx_in = frame.shape
        V   = frame.ravel().astype(np.float64)
        N   = len(V)

        xi  = np.arange(nx_in, dtype=np.float64)
        yi  = np.arange(ny_in, dtype=np.float64)
        XI, YI = np.meshgrid(xi, yi)

        # Centre of each input pixel in output coordinates
        # Convention: the (0,0) input pixel maps to (0,0) in output
        # Shift dx_input moves the reference frame right → subtract to align
        xo = XI.ravel() * scale - dx_input * scale
        yo = YI.ravel() * scale - dy_input * scale

        if bad_mask is not None:
            good = ~bad_mask.ravel()
            xo = xo[good]
            yo = yo[good]
            V  = V[good]

        # Drop half-width in output pixel units
        d = pixfrac * scale / 2.0

        # First output pixel (in each axis) the drop can touch
        i0 = np.floor(xo - d + 0.5).astype(np.int32)
        j0 = np.floor(yo - d + 0.5).astype(np.int32)

        # Maximum span of output pixels each drop touches (conservative)
        max_span = max(1, int(math.ceil(2.0 * d + 1.0)))

        N_out = self.ny_out * self.nx_out
        nx_out = self.nx_out

        for dj in range(max_span):
            for di in range(max_span):
                IO = i0 + di
                JO = j0 + dj

                valid = ((IO >= 0) & (IO < self.nx_out) &
                         (JO >= 0) & (JO < self.ny_out))
                if not valid.any():
                    continue

                # Overlap area between [xo-d, xo+d] and output pixel [IO-0.5, IO+0.5]
                ox   = (np.minimum(xo + d, IO + 0.5) -
                        np.maximum(xo - d, IO - 0.5))
                oy   = (np.minimum(yo + d, JO + 0.5) -
                        np.maximum(yo - d, JO - 0.5))
                ovlp = np.maximum(0.0, ox) * np.maximum(0.0, oy)

                sel = valid & (ovlp > 1e-12)
                if not sel.any():
                    continue

                flat = (JO[sel] * nx_out + IO[sel]).astype(np.int64)
                wc   = ovlp[sel] * weight

                # bincount is faster than add.at for non-duplicate-heavy arrays
                self._sci += np.bincount(flat, weights=wc * V[sel],
                                         minlength=N_out)
                self._wht += np.bincount(flat, weights=wc,
                                         minlength=N_out)
                self._exp += np.bincount(flat, weights=ovlp[sel] * exptime,
                                         minlength=N_out)

        self.n_frames += 1

    def finalize(self):
        """
        Return (sci, wht, exp) as 2-D arrays shaped (ny_out, nx_out).

        sci : flux-conserving combined image  (sci_sum / wht_sum)
        wht : accumulated weight map
        exp : accumulated exposure map (seconds)
        """
        safe = self._wht > 0
        sci_2d = np.zeros(self.ny_out * self.nx_out, dtype=np.float64)
        sci_2d[safe] = self._sci[safe] / self._wht[safe]

        sci = sci_2d.reshape(self.ny_out, self.nx_out)
        wht = self._wht.reshape(self.ny_out, self.nx_out)
        exp = self._exp.reshape(self.ny_out, self.nx_out)
        return sci, wht, exp

    def preview(self) -> np.ndarray:
        """Quick 2-D preview of current state (no copy of full arrays)."""
        safe = self._wht > 0
        out  = np.zeros_like(self._sci)
        out[safe] = self._sci[safe] / self._wht[safe]
        return out.reshape(self.ny_out, self.nx_out)

    def coverage_fraction(self) -> float:
        return float((self._wht > 0).sum() / len(self._wht))


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class DrizzleWorker(QObject):
    """
    Background thread: load frames → estimate shifts → drizzle → emit result.
    Emits frame_done after each frame (for live preview), finished at end.
    """
    progress   = pyqtSignal(int, str)           # (pct, message)
    frame_done = pyqtSignal(int, object)        # (frame_idx, preview_array)
    shifts_done= pyqtSignal(list)               # [(dy, dx), …]
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, paths: list, params: dict):
        super().__init__()
        self.paths   = paths
        self.params  = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    # ── internal ──────────────────────────────────────────────────────────────
    def _log(self, pct: int, msg: str):
        self.progress.emit(pct, msg)

    def _load_frame(self, path: str):
        """Load FITS, return (data_2d, header, exptime)."""
        if not ASTROPY_OK:
            raise ImportError("astropy required")
        with fits.open(path) as hdul:
            hdr  = hdul[0].header.copy()
            data = hdul[0].data
            if data is None and len(hdul) > 1:
                hdr  = hdul[1].header.copy()
                data = hdul[1].data
            exp = float(hdr.get('EXPTIME', hdr.get('EXPOSURE', 1.0)))
        return luminance_from_fits(data), hdr, exp

    def _pipeline(self):
        p = self.params
        paths   = self.paths
        n       = len(paths)
        scale   = p['scale']
        pixfrac = p['pixfrac']

        # ── 1. Load all frames ──────────────────────────────────────────────
        frames   = []
        headers  = []
        exptimes = []
        self._log(2, f"Loading {n} frames…")

        for k, path in enumerate(paths):
            if self._cancel: return
            self._log(int(2 + 18 * k / n), f"Loading {k+1}/{n}: {Path(path).name}")
            data, hdr, exp = self._load_frame(path)
            frames.append(data)
            headers.append(hdr)
            exptimes.append(exp)

        ny_in, nx_in = frames[0].shape
        self._log(20, f"Loaded {n} frames  ({ny_in}×{nx_in} px)")

        # ── 2. Estimate shifts ──────────────────────────────────────────────
        shifts = [(0.0, 0.0)]   # first frame is reference
        self._log(22, "Estimating inter-frame shifts (phase correlation)…")

        ref = gaussian_filter(frames[0], 1.0)   # light smooth for robustness
        for k in range(1, n):
            if self._cancel: return
            self._log(int(22 + 15 * (k-1) / max(1, n-1)),
                      f"Shift {k}/{n-1}: {Path(paths[k]).name}")
            tgt = gaussian_filter(frames[k], 1.0)
            dy, dx = estimate_shift_fft(ref, tgt)
            shifts.append((dy, dx))

        self.shifts_done.emit(shifts)
        self._log(37, f"Shifts estimated. Max: "
                  f"{max(abs(dy) for dy,dx in shifts):.2f} px (dy), "
                  f"{max(abs(dx) for dy,dx in shifts):.2f} px (dx)")

        # ── 3. Compute output size ──────────────────────────────────────────
        # Account for maximum frame extent due to dithering
        max_dy_px = max(abs(dy) for dy, dx in shifts)
        max_dx_px = max(abs(dx) for dy, dx in shifts)
        margin_y  = int(math.ceil(max_dy_px * scale)) + 2
        margin_x  = int(math.ceil(max_dx_px * scale)) + 2

        ny_out = int(ny_in * scale) + 2 * margin_y
        nx_out = int(nx_in * scale) + 2 * margin_x

        # Output centre offset (to keep all frames within bounds)
        cy_off = margin_y
        cx_off = margin_x

        self._log(40, f"Output grid: {ny_out}×{nx_out} px  (scale={scale}×)")

        # ── 4. Drizzle ──────────────────────────────────────────────────────
        engine = DrizzleEngine(ny_out, nx_out)

        weighting = p['weighting']   # 'equal', 'exptime', 'snr'

        for k, (frame, (dy, dx), exp) in enumerate(zip(frames, shifts, exptimes)):
            if self._cancel: return
            self._log(int(40 + 50 * k / n),
                      f"Drizzling frame {k+1}/{n}  "
                      f"(shift dy={dy:+.2f} dx={dx:+.2f} px)  "
                      f"exp={exp:.1f}s")

            # Frame weight
            if weighting == 'equal':
                w = 1.0
            elif weighting == 'exptime':
                w = exp
            else:  # snr — use robust std as proxy for noise
                _, _, std = sigma_clipped_stats(frame, sigma=3.0, maxiters=3)
                w = 1.0 / max(std**2, 1e-10)

            # Bad pixel mask (sigma-clip based)
            bad = None
            if p.get('mask_bad', True):
                bad = sigma_clip_mask(frame, sigma=p.get('clip_sigma', 5.0))

            # Derivation:
            #   Input pixel (i,j) from frame k (shifted by dx,dy relative to ref)
            #   represents reference content at (i-dx, j-dy).
            #   On the output grid that maps to:
            #     x_out = (i - dx) * scale + margin_x
            #   Engine formula:  x_out = (i - dx_input) * scale
            #   Matching:        dx_input = dx - margin_x / scale
            dx_out = dx - cx_off / scale
            dy_out = dy - cy_off / scale

            engine.add_frame(frame, dx_out, dy_out,
                             scale, pixfrac, weight=w, exptime=exp,
                             bad_mask=bad)

            # Emit live preview every frame (or every Nth for large stacks)
            preview_interval = max(1, n // 10)
            if k % preview_interval == 0 or k == n - 1:
                self.frame_done.emit(k, engine.preview())

        # ── 5. Finalize ─────────────────────────────────────────────────────
        self._log(91, "Finalizing combined image…")
        sci, wht, exp_map = engine.finalize()

        # Trim zero-weight border
        if p.get('trim_border', True):
            covered = wht > 0
            rows    = np.where(covered.any(axis=1))[0]
            cols    = np.where(covered.any(axis=0))[0]
            if len(rows) and len(cols):
                r0, r1 = rows[0], rows[-1] + 1
                c0, c1 = cols[0], cols[-1] + 1
                sci     = sci[r0:r1, c0:c1]
                wht     = wht[r0:r1, c0:c1]
                exp_map = exp_map[r0:r1, c0:c1]

        # Effective resolution gain estimate
        # Assuming Nyquist-sampled PSF and random dithering:
        #   FWHM_out ≈ FWHM_in / sqrt(scale)  for well-dithered stacks
        fwhm_gain = math.sqrt(scale)
        coverage  = float((wht > 0).mean())

        self._log(100, f"Done!  Output {sci.shape[1]}×{sci.shape[0]} px  "
                       f"coverage={coverage*100:.1f}%")

        self.finished.emit({
            'sci':         sci,
            'wht':         wht,
            'exp_map':     exp_map,
            'shifts':      shifts,
            'n_frames':    n,
            'scale':       scale,
            'pixfrac':     pixfrac,
            'fwhm_gain':   fwhm_gain,
            'coverage':    coverage,
            'headers':     headers,
            'exptimes':    exptimes,
            'paths':       paths,
            'input_shape': (ny_in, nx_in),
        })


class SaveWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, result: dict, out_dir: str, out_stem: str,
                 save_wht: bool, save_exp: bool):
        super().__init__()
        self.result    = result
        self.out_dir   = out_dir
        self.out_stem  = out_stem
        self.save_wht  = save_wht
        self.save_exp  = save_exp

    def run(self):
        try:
            r   = self.result
            hdr = r['headers'][0].copy()   # use first frame header as template

            # Update header provenance
            hdr['HYPERDRZ'] = (True,          'HYPERLOAD Drizzle Combiner')
            hdr['HDRSCALE']  = (r['scale'],    'Drizzle output/input scale')
            hdr['HDRPXFR']   = (r['pixfrac'],  'Drizzle pixfrac')
            hdr['HDRNFRM']   = (r['n_frames'], 'Frames combined')
            hdr['HDRCOVR']   = (round(r['coverage'], 4), 'Pixel coverage fraction')
            hdr['HDRDATE']   = (datetime.utcnow().isoformat()+'Z', 'Drizzle date UTC')
            hdr['EXPTIME']   = float(sum(r['exptimes']))

            # ── science FITS ───────────────────────────────────────────────
            sci_path = str(Path(self.out_dir) / f"{self.out_stem}_drizzle.fits")
            self.progress.emit(30, f"Writing science FITS…")
            fits.PrimaryHDU(r['sci'].astype(np.float32), header=hdr).writeto(
                sci_path, overwrite=True)

            # ── weight map ────────────────────────────────────────────────
            if self.save_wht:
                wht_path = str(Path(self.out_dir) / f"{self.out_stem}_drizzle_wht.fits")
                self.progress.emit(55, "Writing weight map…")
                fits.PrimaryHDU(r['wht'].astype(np.float32)).writeto(
                    wht_path, overwrite=True)

            # ── exposure map ──────────────────────────────────────────────
            if self.save_exp:
                exp_path = str(Path(self.out_dir) / f"{self.out_stem}_drizzle_exp.fits")
                self.progress.emit(75, "Writing exposure map…")
                fits.PrimaryHDU(r['exp_map'].astype(np.float32)).writeto(
                    exp_path, overwrite=True)

            # ── JSON report ───────────────────────────────────────────────
            rep_path = str(Path(self.out_dir) / f"{self.out_stem}_drizzle_report.json")
            report = {
                'date_utc':   datetime.utcnow().isoformat() + 'Z',
                'hyperload':  VERSION,
                'n_frames':   r['n_frames'],
                'scale':      r['scale'],
                'pixfrac':    r['pixfrac'],
                'output_shape': list(r['sci'].shape),
                'input_shape':  list(r['input_shape']),
                'coverage':   r['coverage'],
                'fwhm_gain_estimate': r['fwhm_gain'],
                'total_exptime_s':    float(sum(r['exptimes'])),
                'shifts': [{'frame': Path(p).name, 'dy': float(dy), 'dx': float(dx)}
                           for p, (dy, dx) in zip(r['paths'], r['shifts'])],
            }
            with open(rep_path, 'w') as f:
                json.dump(report, f, indent=2)

            self.progress.emit(100, f"Saved → {self.out_stem}_drizzle.fits")
            self.finished.emit(sci_path)

        except Exception as ex:
            self.error.emit(f"{ex}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow, QWidget {
    background-color: #111120;
    color: #d0d0e8;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}
QTabWidget::pane  { border: 1px solid #26264a; background: #151530; }
QTabBar::tab      { background: #1a1a34; color: #6868a0; padding: 6px 18px;
                    border: 1px solid #26264a; border-bottom: none; }
QTabBar::tab:selected { background: #20204c; color: #9898ff;
                        border-bottom: 2px solid #5050c8; }
QGroupBox         { border: 1px solid #26264a; border-radius: 4px;
                    margin-top: 8px; padding-top: 8px;
                    color: #6868a8; font-weight: bold; }
QGroupBox::title  { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton       { background: #1e1e42; color: #b0b0d8;
                    border: 1px solid #3838a8; border-radius: 4px;
                    padding: 5px 14px; }
QPushButton:hover   { background: #28285e; border-color: #5050c8; }
QPushButton:pressed { background: #181838; }
QPushButton:disabled{ color: #383858; border-color: #242448; }
QPushButton#run_btn { background: #281a5c; color: #c0a0ff;
                      border: 1px solid #7054c8; font-weight: bold;
                      padding: 7px 20px; }
QPushButton#run_btn:hover { background: #382a78; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background: #181832; border: 1px solid #26264a;
    border-radius: 3px; padding: 3px 6px; color: #d0d0e8; }
QTextEdit { background: #0e0e20; border: 1px solid #26264a;
            color: #85e085; font-family: Consolas, monospace; font-size: 11px; }
QTableWidget { background: #0e0e20; alternate-background-color: #131328;
               gridline-color: #26264a; color: #d0d0e8;
               selection-background-color: #28286a; }
QTableWidget QHeaderView::section {
    background: #181834; color: #6868a8;
    border: 1px solid #26264a; padding: 3px; font-weight: bold; }
QSplitter::handle { background: #26264a; }
QScrollArea       { border: none; }
QLabel#title_lbl  { font-size: 19px; font-weight: bold; color: #9878ff; padding: 6px; }
QLabel#sub_lbl    { font-size: 11px; color: #504870; padding: 0 8px 6px; }
QSlider::groove:horizontal {
    height: 4px; background: #26264a; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; border-radius: 7px;
    background: #6050c0; margin: -5px 0; }
QProgressBar { background: #1a1a30; border: 1px solid #3535a0;
               border-radius: 3px; color: white; font-size: 9px; text-align: center; }
QProgressBar::chunk { background: qlineargradient(
    x1:0,y1:0,x2:1,y2:0, stop:0 #4040c0, stop:1 #9050e0);
    border-radius:2px; }
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(6, 4)):
        super().__init__(parent)
        self.fig    = Figure(figsize=figsize, facecolor='#191930')
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
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()


def _ax_dark(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#0d0d1e')
    ax.set_title(title, color='#8888ff', fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color='#545472', fontsize=8)
    ax.set_ylabel(ylabel, color='#545472', fontsize=8)
    ax.tick_params(colors='#545472', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#26264a')


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#0b0b18; border-top:1px solid #26264a;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        self.lbl  = QLabel("Ready")
        self.lbl.setStyleSheet("color:#585878; font-size:10px;")
        self.pbar = QProgressBar()
        self.pbar.setFixedSize(200, 12)
        self.pbar.setVisible(False)
        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.pbar)

    def update(self, pct: int, msg: str):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True)
            self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(2500, lambda: self.pbar.setVisible(False))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class DrizzleCombiner(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1340, 880)
        self.setStyleSheet(DARK)

        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            self.setWindowIcon(QIcon(str(_logo)))

        self._result   = None
        self._worker   = None
        self._wthread  = None
        self._frame_paths: list[str] = []
        self._shifts: list          = []

        self._build_ui()
        self._check_deps()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setSpacing(0)
        rl.setContentsMargins(0, 0, 0, 0)

        rl.addWidget(self._make_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._make_control_panel())
        splitter.addWidget(self._make_view_tabs())
        splitter.setSizes([290, 1050])
        rl.addWidget(splitter, 1)

        self.status = StatusFooter()
        rl.addWidget(self.status)

    def _make_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setFixedHeight(66)
        hdr.setStyleSheet("background:#0b0b18; border-bottom:1px solid #26264a;")
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(12, 0, 12, 0)

        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                50, 50, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)

        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Drizzle Combiner");          t.setObjectName("title_lbl")
        s = QLabel(
            "Variable Pixel Linear Reconstruction  ·  "
            "Fruchter & Hook 2002  ·  "
            "Sub-pixel super-resolution for dithered stacks")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s)
        lay.addLayout(v, 1)

        ver = QLabel(f"v{VERSION}  |  HYPERLOAD")
        ver.setStyleSheet("color:#30305a; font-size:10px;")
        lay.addWidget(ver)
        return hdr

    # ── left control panel ────────────────────────────────────────────────────
    def _make_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(7)

        # ── frames ────────────────────────────────────────────────────────
        grp = QGroupBox("Input Frames")
        gl  = QVBoxLayout(grp)
        btn_row = QHBoxLayout()
        self.btn_add    = QPushButton("📂  Add FITS…")
        self.btn_folder = QPushButton("📁  Add Folder…")
        self.btn_clear  = QPushButton("✖")
        self.btn_clear.setFixedWidth(32)
        self.btn_add.clicked.connect(self._add_files)
        self.btn_folder.clicked.connect(self._add_folder)
        self.btn_clear.clicked.connect(self._clear_frames)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_folder)
        btn_row.addWidget(self.btn_clear)
        gl.addLayout(btn_row)

        self.frame_count_lbl = QLabel("0 frames")
        self.frame_count_lbl.setStyleSheet("color:#585878; font-size:10px;")
        gl.addWidget(self.frame_count_lbl)
        lay.addWidget(grp)

        # ── drizzle parameters ────────────────────────────────────────────
        grp2 = QGroupBox("Drizzle Parameters")
        g2   = QGridLayout(grp2)

        g2.addWidget(QLabel("Output scale:"), 0, 0)
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(["1× (same resolution)",
                                  "2× (2× super-resolution)",
                                  "3× (3× super-resolution)"])
        self.cmb_scale.setCurrentIndex(1)
        g2.addWidget(self.cmb_scale, 0, 1)

        g2.addWidget(QLabel("Pixfrac:"), 1, 0)
        self.sp_pixfrac = QDoubleSpinBox()
        self.sp_pixfrac.setRange(0.01, 1.0)
        self.sp_pixfrac.setValue(0.8)
        self.sp_pixfrac.setSingleStep(0.05)
        self.sp_pixfrac.setDecimals(2)
        self.sp_pixfrac.setToolTip(
            "Drop size as fraction of input pixel.\n"
            "0.7–0.8: best resolution with dithered frames.\n"
            "1.0: no resolution improvement (like weighted mean).\n"
            "< 0.5: gaps appear unless frames are well-dithered.")
        g2.addWidget(self.sp_pixfrac, 1, 1)
        lay.addWidget(grp2)

        # ── weighting ─────────────────────────────────────────────────────
        grp3 = QGroupBox("Frame Weighting")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("Weight by:"), 0, 0)
        self.cmb_weight = QComboBox()
        self.cmb_weight.addItems(["Equal", "Exposure time (EXPTIME)", "SNR² (inverse noise)"])
        g3.addWidget(self.cmb_weight, 0, 1)
        lay.addWidget(grp3)

        # ── masking ───────────────────────────────────────────────────────
        grp4 = QGroupBox("Masking")
        g4   = QVBoxLayout(grp4)
        self.cb_mask     = QCheckBox("Sigma-clip bad pixels per frame")
        self.cb_mask.setChecked(True)
        self.cb_trim     = QCheckBox("Trim zero-weight border")
        self.cb_trim.setChecked(True)
        g4.addWidget(self.cb_mask)

        clip_row = QHBoxLayout()
        clip_row.addWidget(QLabel("  Clip σ:"))
        self.sp_clip = QDoubleSpinBox()
        self.sp_clip.setRange(2.0, 10.0)
        self.sp_clip.setValue(5.0)
        self.sp_clip.setSingleStep(0.5)
        clip_row.addWidget(self.sp_clip)
        clip_row.addStretch()
        g4.addLayout(clip_row)
        g4.addWidget(self.cb_trim)
        lay.addWidget(grp4)

        # ── output ────────────────────────────────────────────────────────
        grp5 = QGroupBox("Output")
        g5   = QVBoxLayout(grp5)

        stem_row = QHBoxLayout()
        stem_row.addWidget(QLabel("File prefix:"))
        self.stem_edit = QLineEdit("stack")
        stem_row.addWidget(self.stem_edit)
        g5.addLayout(stem_row)

        self.cb_wht = QCheckBox("Save weight map (_wht.fits)")
        self.cb_wht.setChecked(True)
        self.cb_exp = QCheckBox("Save exposure map (_exp.fits)")
        self.cb_exp.setChecked(False)
        g5.addWidget(self.cb_wht)
        g5.addWidget(self.cb_exp)

        self.btn_save = QPushButton("💾  Save Output…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        g5.addWidget(self.btn_save)
        lay.addWidget(grp5)

        # ── run / cancel ──────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Drizzle Combine")
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
        self.tabs.addTab(self._tab_frames(),   "📁  Frames")
        self.tabs.addTab(self._tab_live(),     "⚡  Live Drizzle")
        self.tabs.addTab(self._tab_result(),   "🖼  Result")
        self.tabs.addTab(self._tab_stats(),    "📊  Statistics")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_frames(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        self.frame_table = QTableWidget(0, 4)
        self.frame_table.setHorizontalHeaderLabels(
            ["Filename", "Size", "EXPTIME", "Status"])
        self.frame_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.frame_table.setAlternatingRowColors(True)
        self.frame_table.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        lay.addWidget(self.frame_table)

        # Shift table (filled after analysis)
        lay.addWidget(QLabel("Estimated shifts (after combining):"))
        self.shift_table = QTableWidget(0, 3)
        self.shift_table.setHorizontalHeaderLabels(["Filename", "ΔY (px)", "ΔX (px)"])
        self.shift_table.setFixedHeight(140)
        self.shift_table.setAlternatingRowColors(True)
        lay.addWidget(self.shift_table)

        return w

    def _tab_live(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        self.live_plot = MplCanvas(figsize=(8, 6))
        lay.addWidget(self.live_plot)

        self.live_pbar = QProgressBar()
        self.live_pbar.setFixedHeight(14)
        lay.addWidget(self.live_pbar)

        self.live_lbl = QLabel("Waiting for drizzle to start…")
        self.live_lbl.setStyleSheet("color:#585878; font-size:10px;")
        lay.addWidget(self.live_lbl)

        return w

    def _tab_result(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("View:"))
        self.cmb_result_view = QComboBox()
        self.cmb_result_view.addItems(
            ["Science (drizzled)", "Weight map", "Exposure map",
             "Before (first frame)", "Weight histogram"])
        self.cmb_result_view.currentIndexChanged.connect(self._refresh_result_tab)
        ctrl.addWidget(self.cmb_result_view)

        ctrl.addWidget(QLabel("  Stretch:"))
        self.cmb_stretch = QComboBox()
        self.cmb_stretch.addItems(["Linear (0.5–99.5%)", "Sqrt", "Log", "Asinh"])
        self.cmb_stretch.currentIndexChanged.connect(self._refresh_result_tab)
        ctrl.addWidget(self.cmb_stretch)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        self.result_plot = MplCanvas(figsize=(8, 6))
        lay.addWidget(self.result_plot)
        return w

    def _tab_stats(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        self.stats_plot = MplCanvas(figsize=(8, 5))
        lay.addWidget(self.stats_plot)
        self.stats_lbl = QLabel("")
        self.stats_lbl.setWordWrap(True)
        self.stats_lbl.setStyleSheet(
            "color:#a0a0d0; font-family:Consolas,monospace; font-size:11px; padding:6px;")
        lay.addWidget(self.stats_lbl)
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
    #  Frame management
    # ─────────────────────────────────────────────────────────────────────────

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add FITS frames", "",
            "FITS (*.fits *.fit *.fts);;All (*)")
        for p in paths:
            if p not in self._frame_paths:
                self._frame_paths.append(p)
        self._refresh_frame_table()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add FITS folder")
        if not folder:
            return
        for ext in ('*.fits', '*.fit', '*.fts', '*.FITS', '*.FIT', '*.FTS'):
            for p in sorted(Path(folder).glob(ext)):
                if str(p) not in self._frame_paths:
                    self._frame_paths.append(str(p))
        self._refresh_frame_table()

    def _clear_frames(self):
        self._frame_paths.clear()
        self._refresh_frame_table()

    def _refresh_frame_table(self):
        n = len(self._frame_paths)
        self.frame_count_lbl.setText(f"{n} frame{'s' if n!=1 else ''} queued")
        self.btn_run.setEnabled(n >= 2 and ASTROPY_OK)

        self.frame_table.setRowCount(n)
        for row, path in enumerate(self._frame_paths):
            p = Path(path)
            self.frame_table.setItem(row, 0, QTableWidgetItem(p.name))
            size_kb = p.stat().st_size // 1024 if p.exists() else 0
            self.frame_table.setItem(row, 1,
                QTableWidgetItem(f"{size_kb} KB"))
            self.frame_table.setItem(row, 2, QTableWidgetItem("—"))
            self.frame_table.setItem(row, 3, QTableWidgetItem("queued"))
        self.frame_table.resizeColumnsToContents()

    # ─────────────────────────────────────────────────────────────────────────
    #  Dependency check
    # ─────────────────────────────────────────────────────────────────────────

    def _check_deps(self):
        if not ASTROPY_OK:
            self._log("⚠  astropy not found.  pip install astropy")
        else:
            self._log("✓  All dependencies found.")

    # ─────────────────────────────────────────────────────────────────────────
    #  Run / cancel
    # ─────────────────────────────────────────────────────────────────────────

    def _run(self):
        if len(self._frame_paths) < 2:
            QMessageBox.warning(self, "Need Frames",
                                "Add at least 2 FITS frames first.")
            return

        scale_map = {0: 1, 1: 2, 2: 3}
        wt_map    = {0: 'equal', 1: 'exptime', 2: 'snr'}

        params = {
            'scale':      scale_map[self.cmb_scale.currentIndex()],
            'pixfrac':    self.sp_pixfrac.value(),
            'weighting':  wt_map[self.cmb_weight.currentIndex()],
            'mask_bad':   self.cb_mask.isChecked(),
            'clip_sigma': self.sp_clip.value(),
            'trim_border':self.cb_trim.isChecked(),
        }

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._result = None
        self._shifts = []

        self._wthread = QThread(self)
        self._worker  = DrizzleWorker(list(self._frame_paths), params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_done.connect(self._on_frame_done)
        self._worker.shifts_done.connect(self._on_shifts_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

        self.tabs.setCurrentIndex(1)   # switch to Live tab
        self._log("=" * 55)
        self._log(f"Starting drizzle  |  {len(self._frame_paths)} frames  "
                  f"|  scale={params['scale']}×  pixfrac={params['pixfrac']}")

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
        self.status.update(pct, msg)
        self.live_pbar.setValue(pct)
        self.live_lbl.setText(msg)
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_frame_done(self, frame_idx: int, preview: np.ndarray):
        """Render live preview."""
        fig = self.live_plot.fig
        fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax_dark(ax, f"Live preview — {frame_idx+1} frame(s) drizzled")

        lo, hi = np.percentile(preview[preview > 0], [1, 99.5]) \
            if (preview > 0).any() else (0, 1)
        ax.imshow(np.sqrt(np.clip(preview - lo, 0, hi - lo)),
                  origin='lower', cmap='inferno', aspect='auto',
                  interpolation='nearest')
        self.live_plot.redraw()

        # Update frame table status
        if frame_idx < self.frame_table.rowCount():
            self.frame_table.setItem(frame_idx, 3,
                QTableWidgetItem("drizzled ✓"))

    def _on_shifts_done(self, shifts: list):
        self._shifts = shifts
        self.shift_table.setRowCount(len(shifts))
        for row, ((dy, dx), path) in enumerate(
                zip(shifts, self._frame_paths)):
            self.shift_table.setItem(row, 0,
                QTableWidgetItem(Path(path).name))
            self.shift_table.setItem(row, 1,
                QTableWidgetItem(f"{dy:+.3f}"))
            self.shift_table.setItem(row, 2,
                QTableWidgetItem(f"{dx:+.3f}"))
        self.shift_table.resizeColumnsToContents()

    def _on_error(self, msg: str):
        self._log(f"\n❌  ERROR:\n{msg}")
        self.status.update(0, "Error — see Log tab")
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.critical(self, "Drizzle Error", msg[:600])

    def _on_finished(self, result: dict):
        self._result = result
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)

        r = result
        self._log("\n" + "=" * 55)
        self._log(f"✓  Drizzle complete")
        self._log(f"   Frames combined : {r['n_frames']}")
        self._log(f"   Scale           : {r['scale']}×")
        self._log(f"   Pixfrac         : {r['pixfrac']}")
        self._log(f"   Output shape    : {r['sci'].shape[1]}×{r['sci'].shape[0]} px")
        self._log(f"   Coverage        : {r['coverage']*100:.1f}%")
        self._log(f"   FWHM gain est.  : {r['fwhm_gain']:.2f}× "
                  f"(theoretical for well-dithered stack)")
        self._log(f"   Total exp. time : {sum(r['exptimes']):.1f} s")
        self._log("=" * 55 + "\n")

        self.summary_lbl.setText(
            f"{r['n_frames']} frames  |  {r['scale']}×  |  "
            f"pixfrac={r['pixfrac']}  |  "
            f"{r['sci'].shape[1]}×{r['sci'].shape[0]}")

        self._refresh_result_tab()
        self._draw_stats_tab()
        self.tabs.setCurrentIndex(2)   # jump to Result

    # ─────────────────────────────────────────────────────────────────────────
    #  Plots
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_stretch(self, img: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(img[img > 0], [0.5, 99.5]) \
            if (img > 0).any() else (0.0, 1.0)
        img_c = np.clip(img - lo, 0, hi - lo)
        mode  = self.cmb_stretch.currentIndex()
        if mode == 0:  return img_c / max(hi - lo, 1e-10)   # linear
        if mode == 1:  return np.sqrt(img_c / max(hi - lo, 1e-10))
        if mode == 2:  return np.log1p(img_c) / np.log1p(hi - lo)
        # asinh
        return np.arcsinh(img_c / max((hi - lo) * 0.1, 1e-10)) / \
               np.arcsinh(10.0)

    def _refresh_result_tab(self):
        if self._result is None:
            return
        r   = self._result
        idx = self.cmb_result_view.currentIndex()

        fig = self.result_plot.fig
        fig.clear()
        fig.patch.set_facecolor('#191930')
        ax  = fig.add_subplot(111)

        if idx == 0:   # science
            disp = self._apply_stretch(r['sci'])
            _ax_dark(ax, f"Drizzled science  ({r['sci'].shape[1]}×{r['sci'].shape[0]} px)")
            ax.imshow(disp, origin='lower', cmap='inferno',
                      aspect='auto', interpolation='nearest')

        elif idx == 1:  # weight map
            _ax_dark(ax, "Weight map")
            im = ax.imshow(r['wht'], origin='lower', cmap='plasma',
                           aspect='auto', interpolation='nearest')
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="3%", pad=0.05)
            fig.colorbar(im, cax=cax).ax.yaxis.set_tick_params(
                colors='#7070a0', labelsize=7)

        elif idx == 2:  # exposure map
            _ax_dark(ax, "Exposure map (seconds)")
            im = ax.imshow(r['exp_map'], origin='lower', cmap='viridis',
                           aspect='auto', interpolation='nearest')
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="3%", pad=0.05)
            fig.colorbar(im, cax=cax).ax.yaxis.set_tick_params(
                colors='#7070a0', labelsize=7)

        elif idx == 3:  # first frame (before)
            _ax_dark(ax, f"Before — {Path(r['paths'][0]).name}")
            if ASTROPY_OK:
                try:
                    with fits.open(r['paths'][0]) as hdul:
                        raw = luminance_from_fits(hdul[0].data)
                    disp = self._apply_stretch(raw)
                    ax.imshow(disp, origin='lower', cmap='inferno',
                              aspect='auto', interpolation='nearest')
                except Exception as ex:
                    ax.text(0.5, 0.5, f"Cannot load: {ex}",
                            transform=ax.transAxes, ha='center',
                            color='#ff6060')

        elif idx == 4:  # weight histogram
            _ax_dark(ax, "Weight Distribution", "Weight value", "Pixel count")
            wht = r['wht'].ravel()
            wht_pos = wht[wht > 0]
            if wht_pos.size:
                ax.hist(wht_pos, bins=60, color='#5050b8',
                        edgecolor='#3030a0', alpha=0.85)
                ax.axvline(float(np.median(wht_pos)), color='#ff8030',
                           lw=1.5, label=f"Median: {np.median(wht_pos):.2f}")
                ax.legend(fontsize=8, facecolor='#1a1a30',
                          edgecolor='#3030a0', labelcolor='white')

        self.result_plot.redraw()

    def _draw_stats_tab(self):
        if self._result is None:
            return
        r    = self._result
        sci  = r['sci']
        wht  = r['wht']
        good = wht > 0

        fig  = self.stats_plot.fig
        fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1, 3)

        # ── pixel histogram ───────────────────────────────────────────────
        ax = axes[0]
        _ax_dark(ax, "Pixel Value Distribution", "Value", "Count (log)")
        vals = sci[good].ravel()
        lo, hi = np.percentile(vals, [0.5, 99.9])
        ax.hist(vals, bins=80, range=(lo, hi), color='#4848c0',
                edgecolor='#3030a0', alpha=0.85, log=True)
        _, med, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=3) \
            if ASTROPY_OK else (None, float(np.median(vals)), float(np.std(vals)))
        ax.axvline(float(med), color='#ff8030', lw=1.5,
                   label=f"Median: {med:.1f}")
        ax.legend(fontsize=7, facecolor='#1a1a30',
                  edgecolor='#3030a0', labelcolor='white')

        # ── shift scatter ─────────────────────────────────────────────────
        ax2 = axes[1]
        _ax_dark(ax2, "Frame Dither Pattern", "ΔX (px)", "ΔY (px)")
        if r['shifts']:
            dys = [s[0] for s in r['shifts']]
            dxs = [s[1] for s in r['shifts']]
            sc  = ax2.scatter(dxs, dys, c=range(len(dxs)),
                              cmap='plasma', s=40, zorder=3)
            # Label frame numbers
            for i, (dy, dx) in enumerate(zip(dys, dxs)):
                ax2.text(dx, dy, str(i), fontsize=6, color='white',
                         ha='center', va='center', zorder=4)
            ax2.axhline(0, color='#3030a0', lw=0.5)
            ax2.axvline(0, color='#3030a0', lw=0.5)

        # ── weight map cross-section (horizontal) ─────────────────────────
        ax3 = axes[2]
        _ax_dark(ax3, "Weight Cross-section (horizontal)",
                 "X (px)", "Weight")
        mid_y = wht.shape[0] // 2
        row   = wht[mid_y, :]
        ax3.plot(row, c='#5070ff', lw=1.2)
        ax3.fill_between(range(len(row)), row, alpha=0.2, color='#5070ff')
        ax3.set_xlim(0, len(row))

        self.stats_plot.redraw()

        # Text summary
        fwhm_gain = r['fwhm_gain']
        txt = (
            f"Output:          {sci.shape[1]} × {sci.shape[0]} px\n"
            f"Frames combined: {r['n_frames']}\n"
            f"Scale:           {r['scale']}×\n"
            f"Pixfrac:         {r['pixfrac']}\n"
            f"Coverage:        {r['coverage']*100:.1f}%\n"
            f"FWHM gain (est): {fwhm_gain:.2f}× "
            f"≈ {fwhm_gain*100-100:.0f}% improvement\n"
            f"Total exp. time: {sum(r['exptimes']):.1f} s\n"
            f"Max shift ΔY:    {max(abs(s[0]) for s in r['shifts']):.2f} px\n"
            f"Max shift ΔX:    {max(abs(s[1]) for s in r['shifts']):.2f} px\n"
            f"Median weight:   {float(np.median(wht[wht>0])):.3f}\n"
            f"Min/max weight:  {float(wht[wht>0].min()):.3f} / {float(wht.max()):.3f}"
        )
        self.stats_lbl.setText(txt)

    # ─────────────────────────────────────────────────────────────────────────
    #  Save
    # ─────────────────────────────────────────────────────────────────────────

    def _save(self):
        if self._result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return

        stem = self.stem_edit.text().strip() or "stack"

        self.btn_save.setEnabled(False)
        t = QThread(self)
        w = SaveWorker(self._result, folder, stem,
                       self.cb_wht.isChecked(),
                       self.cb_exp.isChecked())
        w.moveToThread(t)
        t.started.connect(w.run)
        w.progress.connect(lambda p, m: self.status.update(p, m))
        w.finished.connect(lambda p: (
            self._log(f"✓  Saved: {p}"),
            self.btn_save.setEnabled(True)))
        w.error.connect(self._on_error)
        t.start()
        self._save_thread = t

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
    win = DrizzleCombiner()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
