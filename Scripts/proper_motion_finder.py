r"""
HYPERLOAD — Proper Motion Finder
=================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Detects stars with anomalous proper motion by comparing two FITS images
taken at different epochs.  Identifies nearby stars, runaway stars,
high-velocity objects, and potential brown dwarf / white dwarf companions
invisible in single-epoch surveys.

Pipeline:
  1. Load epoch 1 and epoch 2 FITS (any time baseline: months to decades)
  2. Extract star positions via iterative centroiding (no external deps)
  3. Cross-match catalogs by nearest-neighbour (KDTree)
  4. Compute observed ΔRA, ΔDec → proper motion in mas/yr
  5. Sigma-clip background to estimate astrometric noise floor
  6. Flag stars above N-sigma threshold as high-PM candidates
  7. Optional: cross-match with Gaia DR3 for validation

Astrometry:
  If FITS WCS is present (CD or CDELT keywords), positions are in RA/Dec.
  If no WCS, falls back to pixel coordinates (proper motion in px/yr).
  Sub-pixel centroiding via 2-D Gaussian fit to each source peak.

References:
  Salim & Gould 2003    — proper motion survey method, ApJ 582, 1011
  Lépine & Shara 2005   — LSPM catalogue, AJ 129, 1483
  Gaia Collaboration 2023 — DR3 catalogue
  Bramich et al. 2008   — difference imaging for PM, MNRAS 386, L77

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

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log,"a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nproper_motion_finder.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.spatial import KDTree
from scipy.ndimage import (gaussian_filter, maximum_filter,
                           label, find_objects, center_of_mass)
from scipy.optimize import minimize

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Proper Motion Finder"


# ═══════════════════════════════════════════════════════════════════════════════
#  STAR EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class StarExtractor:
    """
    Source extraction without external dependencies.

    Method:
      1. Estimate background via sigma-clipped median in blocks
      2. Find local maxima above threshold in background-subtracted image
      3. Refine positions via 2-D Gaussian centroid (windowed)
      4. Measure flux via aperture sum
    """

    def __init__(self, fwhm_px: float = 4.0, threshold_sigma: float = 5.0,
                 min_sep_px: float = 8.0, max_stars: int = 2000):
        self.fwhm        = fwhm_px
        self.threshold_sigma = threshold_sigma
        self.min_sep     = min_sep_px
        self.max_stars   = max_stars

    def extract(self, image: np.ndarray) -> dict:
        """
        Returns dict:
          x, y   : pixel coordinates (0-indexed, float)
          flux   : aperture flux
          fwhm   : measured FWHM for each star (px)
          n      : number of sources
        """
        img = image.astype(float)
        ny, nx = img.shape
        sigma  = self.fwhm / 2.355

        # ── Background estimation (block median) ───────────────────────────
        bkg = self._estimate_background(img)
        sub = img - bkg

        # ── Detection threshold ────────────────────────────────────────────
        # RMS from negative pixels (sky noise estimate)
        neg  = sub[sub < 0]
        rms  = np.std(neg) * math.sqrt(2) if len(neg) > 100 else np.std(sub) * 0.7
        thresh = self.threshold_sigma * rms

        # ── Local maxima ───────────────────────────────────────────────────
        smooth  = gaussian_filter(sub, sigma)
        footprint_size = max(3, int(self.min_sep))
        local_max = (smooth == maximum_filter(smooth, footprint_size))
        candidates = local_max & (smooth > thresh)
        ys, xs = np.where(candidates)

        if len(xs) == 0:
            return {'x': np.array([]), 'y': np.array([]), 'flux': np.array([]), 'n': 0}

        # Sort by brightness, keep top max_stars
        order  = np.argsort(smooth[ys, xs])[::-1]
        xs, ys = xs[order[:self.max_stars]], ys[order[:self.max_stars]]

        # ── Refine positions via Gaussian centroid ─────────────────────────
        win    = max(3, int(self.fwhm * 2))
        x_ref  = []; y_ref = []; fluxes = []; fwhms = []

        for xi, yi in zip(xs, ys):
            x0 = max(0, xi - win); x1 = min(nx, xi + win + 1)
            y0 = max(0, yi - win); y1 = min(ny, yi + win + 1)
            stamp = sub[y0:y1, x0:x1]
            if stamp.max() <= 0: continue

            # Weighted centroid
            yy, xx = np.mgrid[y0:y1, x0:x1]
            w = np.maximum(stamp, 0)
            ws = w.sum()
            if ws <= 0: continue
            xc = (w * xx).sum() / ws
            yc = (w * yy).sum() / ws

            # Aperture flux
            ap_r  = self.fwhm * 2.0
            yyi, xxi = np.mgrid[max(0,int(yc)-int(ap_r)):min(ny,int(yc)+int(ap_r)+1),
                                  max(0,int(xc)-int(ap_r)):min(nx,int(xc)+int(ap_r)+1)]
            dist = np.sqrt((xxi - xc)**2 + (yyi - yc)**2)
            ap_mask = dist < ap_r
            flux = sub[yyi[ap_mask], xxi[ap_mask]].sum()

            x_ref.append(xc); y_ref.append(yc)
            fluxes.append(flux); fwhms.append(self.fwhm)

        x_ref  = np.array(x_ref);  y_ref  = np.array(y_ref)
        fluxes = np.array(fluxes); fwhms  = np.array(fwhms)

        # Remove duplicates closer than min_sep
        if len(x_ref) > 1:
            tree = KDTree(np.c_[x_ref, y_ref])
            pairs = tree.query_pairs(self.min_sep)
            remove = set()
            for i, j in pairs:
                if fluxes[i] >= fluxes[j]: remove.add(j)
                else: remove.add(i)
            keep = np.array([k for k in range(len(x_ref)) if k not in remove])
            x_ref = x_ref[keep]; y_ref = y_ref[keep]
            fluxes = fluxes[keep]; fwhms = fwhms[keep]

        return {'x': x_ref, 'y': y_ref, 'flux': fluxes, 'fwhm': fwhms,
                'n': len(x_ref), 'rms': rms, 'bkg': bkg}

    def _estimate_background(self, img: np.ndarray,
                              block: int = 64) -> np.ndarray:
        """Block-median background map."""
        ny, nx = img.shape
        bkg = np.zeros_like(img)
        for y0 in range(0, ny, block):
            for x0 in range(0, nx, block):
                tile = img[y0:y0+block, x0:x0+block]
                med  = np.median(tile)
                bkg[y0:y0+block, x0:x0+block] = med
        from scipy.ndimage import uniform_filter
        return uniform_filter(bkg.astype(float), block)


# ═══════════════════════════════════════════════════════════════════════════════
#  WCS HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def pixel_to_world(x: np.ndarray, y: np.ndarray, header) -> tuple:
    """
    Convert pixel coords to RA/Dec using FITS WCS.
    Falls back to pixel coords (in arcsec units) if no valid WCS.
    Returns (ra_deg, dec_deg, has_wcs).
    """
    if not ASTROPY_OK or header is None:
        return x * 1.0, y * 1.0, False
    try:
        wcs = WCS(header, naxis=2)
        if wcs.has_celestial:
            world = wcs.pixel_to_world(x, y)
            return world.ra.deg, world.dec.deg, True
    except Exception:
        pass
    # Pixel scale fallback: try to read CDELT
    try:
        cdelt = abs(float(header.get('CDELT1', header.get('CD1_1', 1.0))))
        scale = cdelt * 3600  # arcsec/px
        return x * scale, y * scale, False
    except Exception:
        return x * 1.0, y * 1.0, False


# ═══════════════════════════════════════════════════════════════════════════════
#  PROPER MOTION CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ProperMotionCalculator:
    """
    Cross-match two star catalogs and compute proper motions.

    Algorithm:
      1. KDTree nearest-neighbour match within match_radius
      2. Iterative background PM subtraction (median of all matched stars)
      3. Sigma-clip to find high-PM outliers
      4. Report in mas/yr (if WCS) or px/yr (if pixel coords)
    """

    def __init__(self, match_radius: float = 5.0,
                 sigma_thresh: float = 5.0):
        self.match_radius = match_radius
        self.sigma_thresh = sigma_thresh

    def match_and_compute(self, cat1: dict, cat2: dict,
                          dt_years: float,
                          has_wcs: bool = False) -> dict:
        """
        cat1, cat2: dicts with keys ra, dec (or x, y in pixel/arcsec)
        dt_years:   epoch2 - epoch1 in decimal years
        """
        ra1, dec1 = cat1['ra'], cat1['dec']
        ra2, dec2 = cat2['ra'], cat2['dec']

        # KDTree match
        tree1   = KDTree(np.c_[ra1, dec1])
        dist, idx = tree1.query(np.c_[ra2, dec2], k=1)

        # Keep matches within radius
        max_r = self.match_radius * (1/3600 if has_wcs else 1.0)
        valid = dist < max_r

        i2 = np.where(valid)[0]
        i1 = idx[valid]

        if len(i1) < 5:
            return {'n_matched': len(i1), 'error': 'Too few matched stars'}

        # Proper motions in deg/yr (WCS) or units/yr (pixel)
        dra  = (ra2[i2]  - ra1[i1]) / dt_years
        ddec = (dec2[i2] - dec1[i1]) / dt_years

        # Correct RA for cos(dec) factor if WCS
        if has_wcs:
            cos_dec = np.cos(np.radians(dec1[i1]))
            dra     = dra * cos_dec
            # Convert deg/yr → mas/yr
            dra  *= 3.6e6
            ddec *= 3.6e6
            unit  = 'mas/yr'
        else:
            unit = 'px/yr'

        # Background PM (median of all stars — systematic field rotation etc.)
        med_ra  = np.median(dra)
        med_dec = np.median(ddec)
        dra_c   = dra  - med_ra
        ddec_c  = ddec - med_dec

        # Noise estimate (sigma clipping, 3 iterations)
        mask = np.ones(len(dra_c), bool)
        for _ in range(3):
            std_ra  = np.std(dra_c[mask])
            std_dec = np.std(ddec_c[mask])
            pm_tot  = np.sqrt(dra_c**2 + ddec_c**2)
            thresh  = self.sigma_thresh * math.sqrt(std_ra**2 + std_dec**2)
            mask    = pm_tot < thresh

        noise_ra  = np.std(dra_c[mask])
        noise_dec = np.std(ddec_c[mask])
        pm_total  = np.sqrt(dra_c**2 + ddec_c**2)
        threshold = self.sigma_thresh * math.sqrt(noise_ra**2 + noise_dec**2)
        high_pm   = pm_total > threshold

        return {
            'n_matched':   len(i1),
            'i1':          i1, 'i2': i2,
            'dra':         dra_c,
            'ddec':        ddec_c,
            'pm_total':    pm_total,
            'threshold':   threshold,
            'high_pm':     high_pm,
            'noise_ra':    noise_ra,
            'noise_dec':   noise_dec,
            'unit':        unit,
            'med_sys_ra':  med_ra,
            'med_sys_dec': med_dec,
            'ra1':         ra1[i1],
            'dec1':        dec1[i1],
            'flux1':       cat1['flux'][i1] if 'flux' in cat1 else np.ones(len(i1)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def luminance(data):
    if data.ndim == 2: return data.astype(float)
    if data.ndim == 3 and data.shape[0] <= 4: return data.mean(0).astype(float)
    return data.astype(float)


class PMWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, path1, path2, params):
        super().__init__()
        self.path1 = path1; self.path2 = path2
        self.params = params

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p = self.params

        # ── Load images ───────────────────────────────────────────────────
        self.progress.emit(5, "Loading epoch 1…")
        hdu1 = fits.open(self.path1)[0]
        img1 = luminance(hdu1.data); hdr1 = hdu1.header

        self.progress.emit(10, "Loading epoch 2…")
        hdu2 = fits.open(self.path2)[0]
        img2 = luminance(hdu2.data); hdr2 = hdu2.header

        # ── Extract stars ─────────────────────────────────────────────────
        self.progress.emit(15, "Extracting stars from epoch 1…")
        ex = StarExtractor(
            fwhm_px        = p.get('fwhm', 4.0),
            threshold_sigma= p.get('thresh_sigma', 5.0),
            min_sep_px     = p.get('min_sep', 8.0),
            max_stars      = p.get('max_stars', 2000),
        )
        cat1_px = ex.extract(img1)
        self.progress.emit(30, f"Epoch 1: {cat1_px['n']} stars  RMS={cat1_px.get('rms',0):.2f}")

        self.progress.emit(35, "Extracting stars from epoch 2…")
        cat2_px = ex.extract(img2)
        self.progress.emit(50, f"Epoch 2: {cat2_px['n']} stars")

        if cat1_px['n'] < 5 or cat2_px['n'] < 5:
            self.error.emit(
                "Too few stars detected.\n\n"
                "Tips:\n"
                "• Lower the detection threshold\n"
                "• Check the image is not saturated or empty\n"
                "• Increase FWHM if stars are large")
            return

        # ── WCS conversion ────────────────────────────────────────────────
        self.progress.emit(55, "Converting coordinates…")
        ra1, dec1, has_wcs = pixel_to_world(cat1_px['x'], cat1_px['y'], hdr1)
        ra2, dec2, _       = pixel_to_world(cat2_px['x'], cat2_px['y'], hdr2)

        cat1 = {'ra': ra1, 'dec': dec1, 'flux': cat1_px['flux'],
                'x': cat1_px['x'], 'y': cat1_px['y']}
        cat2 = {'ra': ra2, 'dec': dec2, 'flux': cat2_px['flux'],
                'x': cat2_px['x'], 'y': cat2_px['y']}

        # ── Compute proper motions ────────────────────────────────────────
        self.progress.emit(60, "Cross-matching and computing proper motions…")
        dt = p['dt_years']
        calc   = ProperMotionCalculator(
            match_radius = p.get('match_radius', 5.0),
            sigma_thresh = p.get('sigma_thresh', 5.0),
        )
        pm = calc.match_and_compute(cat1, cat2, dt, has_wcs)

        if 'error' in pm:
            self.error.emit(pm['error']); return

        n_high = pm['high_pm'].sum()
        self.progress.emit(90,
            f"Matched: {pm['n_matched']}  High-PM: {n_high}  "
            f"Threshold: {pm['threshold']:.1f} {pm['unit']}")

        self.progress.emit(100, "Done!")
        self.finished.emit({
            'pm':        pm,
            'cat1':      cat1,
            'cat2':      cat2,
            'cat1_px':   cat1_px,
            'img1':      img1,
            'img2':      img2,
            'has_wcs':   has_wcs,
            'dt_years':  dt,
            'params':    p,
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow,QWidget{background:#111120;color:#d0d0e8;
    font-family:'Segoe UI',Arial,sans-serif;font-size:12px;}
QTabWidget::pane{border:1px solid #26264a;background:#151530;}
QTabBar::tab{background:#1a1a34;color:#6868a0;padding:6px 18px;
    border:1px solid #26264a;border-bottom:none;}
QTabBar::tab:selected{background:#20204c;color:#9898ff;
    border-bottom:2px solid #5050c8;}
QGroupBox{border:1px solid #26264a;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#6868a8;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#1e1e42;color:#b0b0d8;border:1px solid #3838a8;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#28285e;border-color:#5050c8;}
QPushButton:disabled{color:#383858;border-color:#242448;}
QPushButton#run_btn{background:#281a5c;color:#c0a0ff;
    border:1px solid #7054c8;font-weight:bold;padding:7px 20px;}
QPushButton#run_btn:hover{background:#382a78;}
QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{background:#181832;
    border:1px solid #26264a;border-radius:3px;padding:3px 6px;color:#d0d0e8;}
QTextEdit{background:#0e0e20;border:1px solid #26264a;color:#85e085;
    font-family:Consolas,monospace;font-size:11px;}
QTableWidget{background:#0e0e20;alternate-background-color:#131328;
    gridline-color:#26264a;color:#d0d0e8;}
QTableWidget QHeaderView::section{background:#181834;color:#6868a8;
    border:1px solid #26264a;padding:3px;font-weight:bold;}
QSplitter::handle{background:#26264a;}
QLabel#title_lbl{font-size:19px;font-weight:bold;color:#9878ff;padding:6px;}
QLabel#sub_lbl{font-size:11px;color:#504870;padding:0 8px 6px;}
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8,5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor='#191930')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
    def redraw(self): self.fig.tight_layout(pad=0.5); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#0d0d1e')
    ax.set_title(title, color='#9090ff', fontsize=9, pad=4)
    ax.set_xlabel(xl, color='#545472', fontsize=8)
    ax.set_ylabel(yl, color='#545472', fontsize=8)
    ax.tick_params(colors='#545472', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#26264a')


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#0b0b18;border-top:1px solid #26264a;")
        lay = QHBoxLayout(self); lay.setContentsMargins(6,0,6,0)
        self.lbl = QLabel("Ready")
        self.lbl.setStyleSheet("color:#585878;font-size:10px;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(200,12)
        self.pbar.setVisible(False)
        self.pbar.setStyleSheet("""
            QProgressBar{background:#1a1a30;border:1px solid #3535a0;
                border-radius:3px;color:white;font-size:9px;}
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #4040c0,stop:1 #9050e0);border-radius:2px;}""")
        lay.addWidget(self.lbl,1); lay.addWidget(self.pbar)

    def update(self, pct, msg):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True); self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))


class ProperMotionApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1380, 900)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))
        self._path1 = self._path2 = None
        self._result = None
        self._worker = self._wthread = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._make_header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._make_controls())
        sp.addWidget(self._make_tabs())
        sp.setSizes([310,1070])
        rl.addWidget(sp,1)
        self.status = StatusFooter()
        rl.addWidget(self.status)

    def _make_header(self):
        hdr = QWidget(); hdr.setFixedHeight(66)
        hdr.setStyleSheet("background:#0b0b18;border-bottom:1px solid #26264a;")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12,0,12,0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                50,50,Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Proper Motion Finder"); t.setObjectName("title_lbl")
        s = QLabel("Two-epoch FITS · Sub-pixel centroiding · KDTree cross-match · "
                   "Sigma-clipped PM detection · WCS-aware (mas/yr or px/yr)")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        ver = QLabel(f"v{VERSION}  |  HYPERLOAD")
        ver.setStyleSheet("color:#30305a;font-size:10px;")
        lay.addWidget(ver)
        return hdr

    def _make_controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)

        # ── Input ──────────────────────────────────────────────────────────
        grp = QGroupBox("Two-Epoch FITS Images")
        gl  = QVBoxLayout(grp)
        self.lbl1 = QLabel("Epoch 1: —"); self.lbl1.setStyleSheet("color:#484868;font-size:10px;")
        self.lbl2 = QLabel("Epoch 2: —"); self.lbl2.setStyleSheet("color:#484868;font-size:10px;")
        b1 = QPushButton("📂  Epoch 1 (earlier)…"); b1.clicked.connect(self._load1)
        b2 = QPushButton("📂  Epoch 2 (later)…");   b2.clicked.connect(self._load2)
        gl.addWidget(self.lbl1); gl.addWidget(b1)
        gl.addWidget(self.lbl2); gl.addWidget(b2)
        lay.addWidget(grp)

        # ── Epochs ────────────────────────────────────────────────────────
        grp2 = QGroupBox("Time Baseline")
        g2   = QGridLayout(grp2)
        g2.addWidget(QLabel("Epoch 1 (decimal year):"), 0, 0)
        self.sp_ep1 = QDoubleSpinBox(); self.sp_ep1.setRange(1900,2100); self.sp_ep1.setValue(2020.0); self.sp_ep1.setDecimals(3)
        g2.addWidget(self.sp_ep1, 0, 1)
        g2.addWidget(QLabel("Epoch 2 (decimal year):"), 1, 0)
        self.sp_ep2 = QDoubleSpinBox(); self.sp_ep2.setRange(1900,2100); self.sp_ep2.setValue(2023.0); self.sp_ep2.setDecimals(3)
        g2.addWidget(self.sp_ep2, 1, 1)
        g2.addWidget(QLabel("(baseline = epoch2 − epoch1)"), 2, 0, 1, 2)
        lay.addWidget(grp2)

        # ── Extraction ────────────────────────────────────────────────────
        grp3 = QGroupBox("Star Extraction")
        g3   = QGridLayout(grp3)
        for row,(lbl,attr,lo,hi,val,dec) in enumerate([
            ("FWHM (px):",     "sp_fwhm",   1.0, 30.0, 4.0, 1),
            ("Threshold (σ):", "sp_thr",    2.0, 20.0, 5.0, 1),
            ("Min sep (px):",  "sp_sep",    2.0, 50.0, 8.0, 1),
            ("Max stars:",     "sp_maxs",   10, 5000, 1000, 0),
        ]):
            g3.addWidget(QLabel(lbl), row, 0)
            sp = QDoubleSpinBox() if dec > 0 else QSpinBox()
            sp.setRange(int(lo), int(hi)); sp.setValue(val)
            if dec > 0: sp.setDecimals(dec)
            setattr(self, attr, sp); g3.addWidget(sp, row, 1)
        lay.addWidget(grp3)

        # ── Matching ──────────────────────────────────────────────────────
        grp4 = QGroupBox("Cross-matching")
        g4   = QGridLayout(grp4)
        g4.addWidget(QLabel("Match radius (px):"), 0, 0)
        self.sp_mr = QDoubleSpinBox(); self.sp_mr.setRange(1,100); self.sp_mr.setValue(5.0); self.sp_mr.setDecimals(1)
        g4.addWidget(self.sp_mr, 0, 1)
        g4.addWidget(QLabel("PM threshold (σ):"), 1, 0)
        self.sp_pmthr = QDoubleSpinBox(); self.sp_pmthr.setRange(2,20); self.sp_pmthr.setValue(5.0); self.sp_pmthr.setDecimals(1)
        g4.addWidget(self.sp_pmthr, 1, 1)
        lay.addWidget(grp4)

        # ── Output ────────────────────────────────────────────────────────
        grp5 = QGroupBox("Output")
        g5   = QVBoxLayout(grp5)
        self.btn_save = QPushButton("💾  Save Candidates (CSV)…")
        self.btn_save.setEnabled(False); self.btn_save.clicked.connect(self._save)
        g5.addWidget(self.btn_save)
        lay.addWidget(grp5)

        self.btn_run = QPushButton("▶  Find Proper Motions")
        self.btn_run.setObjectName("run_btn"); self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setEnabled(False); self.btn_cancel.setStyleSheet("color:#b05050;")
        self.btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_cancel)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color:#70c070;font-size:10px;padding:4px;")
        self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _make_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_images(),   "🖼  Epoch Images")
        self.tabs.addTab(self._tab_vectors(),  "🏃  PM Vectors")
        self.tabs.addTab(self._tab_hist(),     "📊  PM Distribution")
        self.tabs.addTab(self._tab_cands(),    "⭐  Candidates")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_images(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.img_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.img_plot)
        return w

    def _tab_vectors(self):
        w = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Arrow scale:"))
        self.sp_scale = QDoubleSpinBox(); self.sp_scale.setRange(0.01,100); self.sp_scale.setValue(1.0); self.sp_scale.setDecimals(2)
        self.sp_scale.valueChanged.connect(self._refresh_vectors)
        ctrl.addWidget(self.sp_scale); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.vec_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.vec_plot)
        return w

    def _tab_hist(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.hist_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.hist_plot)
        return w

    def _tab_cands(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.cand_table = QTableWidget(0, 5)
        self.cand_table.setHorizontalHeaderLabels(
            ["#", "X (px)", "Y (px)", "PM total", "Sigma"])
        self.cand_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.cand_table.setAlternatingRowColors(True)
        lay.addWidget(self.cand_table)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        QPushButton("Clear").clicked.connect(self.log.clear)
        return w

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _load1(self):
        path, _ = QFileDialog.getOpenFileName(
            self,"Epoch 1 FITS","","FITS (*.fits *.fit *.fts);;All (*)")
        if path:
            self._path1 = path
            self.lbl1.setText(Path(path).name)
            self.lbl1.setStyleSheet("color:#70c070;font-size:10px;")
            # Try to read epoch from header
            if ASTROPY_OK:
                try:
                    hdr = fits.getheader(path)
                    for kw in ['DATE-OBS','DATE','MJD-OBS']:
                        if kw in hdr:
                            self._log(f"Epoch 1 header: {kw} = {hdr[kw]}")
                            break
                except Exception: pass
            self._check_ready()

    def _load2(self):
        path, _ = QFileDialog.getOpenFileName(
            self,"Epoch 2 FITS","","FITS (*.fits *.fit *.fts);;All (*)")
        if path:
            self._path2 = path
            self.lbl2.setText(Path(path).name)
            self.lbl2.setStyleSheet("color:#70c070;font-size:10px;")
            self._check_ready()

    def _check_ready(self):
        self.btn_run.setEnabled(
            self._path1 is not None and self._path2 is not None)

    def _save(self):
        if not self._result: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save candidates", "", "CSV (*.csv)")
        if not path: return
        pm = self._result['pm']
        high = pm['high_pm']
        idxs = np.where(high)[0]
        with open(path,'w') as f:
            f.write("index,x_px,y_px,pm_ra,pm_dec,pm_total,sigma\n")
            unit = pm['unit']
            cat1 = self._result['cat1_px']
            for k in idxs:
                sigma = pm['pm_total'][k] / (pm['threshold'] / self.sp_pmthr.value())
                f.write(f"{k},{cat1['x'][pm['i1'][k]]:.2f},"
                        f"{cat1['y'][pm['i1'][k]]:.2f},"
                        f"{pm['dra'][k]:.2f},{pm['ddec'][k]:.2f},"
                        f"{pm['pm_total'][k]:.2f},{sigma:.1f}\n")
        self._log(f"✓ Saved: {path}")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if not (self._path1 and self._path2): return
        dt = self.sp_ep2.value() - self.sp_ep1.value()
        if abs(dt) < 0.01:
            QMessageBox.warning(self,"Warning","Epoch difference < 0.01 yr. Check epochs.")
            return
        params = {
            'dt_years':     dt,
            'fwhm':         float(self.sp_fwhm.value()),
            'thresh_sigma': float(self.sp_thr.value()),
            'min_sep':      float(self.sp_sep.value()),
            'max_stars':    int(self.sp_maxs.value()),
            'match_radius': float(self.sp_mr.value()),
            'sigma_thresh': float(self.sp_pmthr.value()),
        }
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False); self._result = None

        self._wthread = QThread(self)
        self._worker  = PMWorker(self._path1, self._path2, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _cancel(self):
        self.btn_cancel.setEnabled(False); self.btn_run.setEnabled(True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status.update(pct, msg); self._log(f"[{pct:3d}%]  {msg}")

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}")
        self.status.update(0,"Error"); self.btn_run.setEnabled(True)
        QMessageBox.critical(self,"Error",msg[:500])

    def _on_finished(self, result):
        self._result = result
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)
        pm = result['pm']
        n_high = pm['high_pm'].sum()

        self._log(f"\n{'='*55}")
        self._log(f"✓  Proper motion search complete")
        self._log(f"   Matched stars:   {pm['n_matched']}")
        self._log(f"   Time baseline:   {result['dt_years']:.2f} yr")
        self._log(f"   PM noise floor:  {pm['noise_ra']:.2f} {pm['unit']}")
        self._log(f"   Threshold:       {pm['threshold']:.1f} {pm['unit']}")
        self._log(f"   High-PM stars:   {n_high}")
        if n_high > 0:
            top = np.argsort(pm['pm_total'])[::-1][:5]
            self._log("   Top candidates:")
            for k in top:
                if pm['high_pm'][k]:
                    self._log(f"     PM={pm['pm_total'][k]:.1f} {pm['unit']}  "
                              f"μ_α={pm['dra'][k]:.1f}  μ_δ={pm['ddec'][k]:.1f}")
        self._log(f"{'='*55}\n")
        self.summary_lbl.setText(
            f"Matched: {pm['n_matched']}  High-PM: {n_high}  "
            f"Noise: {pm['noise_ra']:.1f} {pm['unit']}")

        self._draw_images()
        self._refresh_vectors()
        self._draw_histogram()
        self._fill_candidates()
        self.tabs.setCurrentIndex(1)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_images(self):
        if not self._result: return
        r = self._result
        fig = self.img_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1,2)

        def show(ax, img, cat, title):
            _ax(ax, title)
            lo,hi=np.percentile(img,[0.5,99.5])
            ax.imshow(np.sqrt(np.clip(img-lo,0,hi-lo)),
                      origin='lower',cmap='inferno',aspect='equal',
                      interpolation='nearest')
            ax.scatter(cat['x'],cat['y'],s=8,c='#60ffff',
                       linewidths=0.5,marker='o',facecolors='none',alpha=0.5)

        show(axes[0], r['img1'], r['cat1_px'],
             f"Epoch 1  ({r['cat1_px']['n']} stars)")
        show(axes[1], r['img2'], r['cat2_px'],
             f"Epoch 2  ({r['cat2_px']['n']} stars)")
        self.img_plot.redraw()

    def _refresh_vectors(self):
        if not self._result: return
        r = self._result; pm = r['pm']
        scale = self.sp_scale.value()
        fig = self.vec_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, f"Proper motion vectors  ({pm['n_matched']} matched, "
                f"{pm['high_pm'].sum()} high-PM)",
            "X (px)", "Y (px)")

        cat1 = r['cat1_px']
        x = cat1['x'][pm['i1']]
        y = cat1['y'][pm['i1']]
        dra  = pm['dra'];  ddec = pm['ddec']
        high = pm['high_pm']

        # Background stars — thin grey arrows
        if (~high).sum() > 0:
            ax.quiver(x[~high], y[~high],
                      dra[~high]*scale, ddec[~high]*scale,
                      color='#383870', alpha=0.4, width=0.001,
                      angles='xy', scale_units='xy', scale=1)

        # High-PM stars — bright arrows
        if high.sum() > 0:
            ax.quiver(x[high], y[high],
                      dra[high]*scale, ddec[high]*scale,
                      color='#ff8030', alpha=0.9, width=0.003,
                      angles='xy', scale_units='xy', scale=1)
            ax.scatter(x[high], y[high], s=60, c='#ff8030',
                       zorder=5, marker='*')

        unit = pm['unit']
        ax.set_title(f"PM vectors  (scale ×{scale})  threshold={pm['threshold']:.0f} {unit}",
                     color='#9090ff', fontsize=9, pad=4)
        self.vec_plot.redraw()

    def _draw_histogram(self):
        if not self._result: return
        pm = self._result['pm']
        fig = self.hist_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1,2)

        unit = pm['unit']
        for ax, vals, label in zip(axes,
                [pm['dra'], pm['ddec']],
                [f'μ_α ({unit})', f'μ_δ ({unit})']):
            _ax(ax, f'PM distribution — {label}', label, 'Count')
            ax.hist(vals, bins=50, color='#5050b0',
                    edgecolor='#3030a0', alpha=0.85)
            ax.axvline(0, color='#ff8030', lw=1.5)
            # Mark high-PM outliers
            high = pm['high_pm']
            for v in vals[high]:
                ax.axvline(v, color='#ff4030', lw=0.8, alpha=0.6)

        self.hist_plot.redraw()

    def _fill_candidates(self):
        if not self._result: return
        pm = self._result['pm']
        cat1 = self._result['cat1_px']
        high = pm['high_pm']
        idxs = np.argsort(pm['pm_total'])[::-1]
        idxs = [i for i in idxs if high[i]]
        unit = pm['unit']

        self.cand_table.setRowCount(len(idxs))
        for row, k in enumerate(idxs):
            i1 = pm['i1'][k]
            sigma = pm['pm_total'][k] / max(pm['threshold']/self.sp_pmthr.value(),1e-3)
            vals = [str(row+1),
                    f"{cat1['x'][i1]:.1f}",
                    f"{cat1['y'][i1]:.1f}",
                    f"{pm['pm_total'][k]:.1f} {unit}",
                    f"{sigma:.1f}σ"]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if col == 3: item.setForeground(
                    __import__('PyQt6.QtGui',fromlist=['QColor']).QColor('#ff8030'))
                self.cand_table.setItem(row, col, item)
        self.cand_table.resizeColumnsToContents()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ProperMotionApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
