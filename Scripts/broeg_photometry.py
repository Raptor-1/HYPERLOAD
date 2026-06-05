r"""
HYPERLOAD — Broeg Optimal Photometry
======================================
Companion to variable_star.py.
Place in: C:\Users\Marcell\Desktop\Siril Suites\

Usage (standalone):
    python broeg_photometry.py
    → Load FITS sequence + specify target position
    → Auto-select comp stars, run Broeg weighting, show light curve

Usage (from variable_star.py — add after auto_detect_comp_stars):
    from broeg_photometry import broeg_optimal_weights, broeg_differential_photometry
    weights = broeg_optimal_weights(comp_light_curves)  # comp_lcs shape (N_stars, N_frames)
    lc = broeg_differential_photometry(fits_files, target_x, target_y, comp_positions)

Algorithm (Broeg et al. 2005, AN 326, 134):
  1. Start with N comparison stars, uniform weights w_i = 1/N
  2. For each star k, compute "ensemble without k":
       C_k(t) = Σ_{j≠k} w_j · f_j(t)  /  Σ_{j≠k} w_j
  3. Measure scatter of ratio: σ_k = RMS[ f_k / C_k ]
  4. Update weights: w_k = 1 / σ_k²
  5. Normalise: w ← w / Σ w
  6. Repeat until convergence (typically 5-10 iterations)

  The optimally-weighted ensemble minimises the differential photometry
  scatter by automatically down-weighting variable or noisy comp stars.

Improvement over uniform weighting:
  - Variable comp stars get near-zero weight automatically
  - High-noise (faint) stars get low weight
  - Stable bright stars dominate the ensemble
  - Typical improvement: 10–30% scatter reduction for a mixed comp set

References:
  Broeg et al. 2005, AN 326, 134 — original algorithm
  Tamuz et al. 2005, MNRAS 356   — SYSREM detrending (related)
  Everett & Howell 2001, PASP 113 — differential photometry review

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, math, traceback
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log,"a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nbroeg_photometry.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib","photutils"]: s.ensure_installed(p)
except ImportError:
    pass

try:
    from astropy.io import fits as astropy_fits
    from astropy.stats import sigma_clipped_stats
    from astropy.time import Time
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

try:
    from photutils.detection import DAOStarFinder
    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit, QProgressBar,
    QGroupBox, QSplitter, QMessageBox, QSizePolicy, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Broeg Optimal Photometry"


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS  (importable by variable_star.py)
# ═══════════════════════════════════════════════════════════════════════════════

def broeg_optimal_weights(comp_lcs: np.ndarray,
                           n_iter: int = 30,
                           tol: float = 1e-8) -> np.ndarray:
    """
    Compute optimal comparison star weights using Broeg et al. (2005).

    Parameters
    ----------
    comp_lcs : (N_stars, N_frames) array of raw fluxes (not magnitudes)
               All values should be positive; zeros/negatives are masked.
    n_iter   : maximum iterations
    tol      : convergence tolerance on max weight change

    Returns
    -------
    weights : (N_stars,) array normalised to sum=1
              Variable/noisy stars get near-zero weight.
              Returns uniform weights if < 2 valid stars.

    Notes
    -----
    Stars that are constant in the ensemble get high weight.
    Stars that deviate from the ensemble (variable, noisy) get low weight.
    """
    N_stars, N_frames = comp_lcs.shape

    # Mask invalid stars (any negative/zero/NaN frame)
    valid = np.all(np.isfinite(comp_lcs) & (comp_lcs > 0), axis=1)
    if valid.sum() < 2:
        return np.ones(N_stars) / N_stars

    lc = comp_lcs[valid].copy().astype(float)  # (N_valid, N_frames)
    N_v = lc.shape[0]

    # Initialise: uniform weights
    w = np.ones(N_v) / N_v

    for it in range(n_iter):
        w_old  = w.copy()
        sigma  = np.zeros(N_v)

        for k in range(N_v):
            # Ensemble excluding star k
            w_ex  = w.copy(); w_ex[k] = 0.0
            s_ex  = w_ex.sum()
            if s_ex < 1e-12: continue
            ens_k = (lc.T @ w_ex) / s_ex          # (N_frames,)

            # Ratio star_k / ensemble
            ratio = lc[k] / (ens_k + 1e-10)
            # Normalise to median = 1
            med   = np.median(ratio)
            if abs(med) < 1e-10: continue
            ratio /= med

            # Scatter = RMS deviation from 1
            sigma[k] = float(np.sqrt(np.mean((ratio - 1.0)**2)))

        # Update: w_k ∝ 1 / σ_k²   (stars with σ=0 get very high weight)
        sigma  = np.maximum(sigma, 1e-10)
        w_new  = 1.0 / (sigma**2)
        w_new /= w_new.sum()

        delta = float(np.max(np.abs(w_new - w_old)))
        w     = w_new
        if delta < tol:
            break

    # Map back to full N_stars
    weights        = np.zeros(N_stars)
    weights[valid] = w
    return weights


def broeg_ensemble(comp_lcs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute the weighted ensemble light curve.

    Parameters
    ----------
    comp_lcs : (N_stars, N_frames) flux array
    weights  : (N_stars,) weights (from broeg_optimal_weights)

    Returns
    -------
    ensemble : (N_frames,) weighted sum of comp star fluxes
    """
    w = weights / (weights.sum() + 1e-10)
    return comp_lcs.T @ w  # (N_frames,)


def differential_mag(target_flux: np.ndarray,
                     ensemble_flux: np.ndarray) -> np.ndarray:
    """
    Differential magnitude = -2.5 log10(target / ensemble), median-zeroed.
    """
    ratio = target_flux / (ensemble_flux + 1e-10)
    dm    = -2.5 * np.log10(np.maximum(ratio, 1e-10))
    return dm - float(np.median(dm))


def _measure_star(image: np.ndarray, x: float, y: float,
                  aper_r: float, ann_in: float, ann_out: float) -> tuple[float, float]:
    """Aperture photometry. Returns (flux, SNR)."""
    h, w  = image.shape
    yi, xi = int(round(y)), int(round(x))
    yy, xx = np.ogrid[:h, :w]
    r_sq   = (xx - x)**2 + (yy - y)**2
    aper   = r_sq <= aper_r**2
    ann    = (r_sq >= ann_in**2) & (r_sq <= ann_out**2)
    if ann.sum() < 4: return 0.0, 0.0
    sky    = float(np.median(image[ann]))
    flux   = float((image[aper] - sky).sum())
    noise  = float(np.std(image[ann])) * math.sqrt(float(aper.sum())) + 1e-10
    return flux, float(abs(flux) / noise)


def broeg_differential_photometry(fits_files: list[str],
                                   target_x: float, target_y: float,
                                   comp_positions: list[tuple[float, float]],
                                   aperture_r: float = 8.0,
                                   n_iter: int = 30,
                                   log_cb=None) -> dict:
    """
    Full Broeg-weighted differential photometry pipeline.

    Parameters
    ----------
    fits_files      : sorted FITS file paths
    target_x/y      : target star pixel position
    comp_positions  : list of (x,y) for comparison stars
    aperture_r      : aperture radius in pixels
    n_iter          : Broeg weight iterations

    Returns
    -------
    dict with keys:
        times           : JD array
        diff_mags_broeg : Broeg-weighted differential magnitudes
        diff_mags_uniform : uniform-weighted (for comparison)
        errors          : per-frame photometric uncertainty
        weights         : (N_comp, N_frames) — final Broeg weights per frame
        comp_lcs        : (N_comp, N_frames) raw comp star fluxes
        weight_final    : (N_comp,) final weights from last iteration
        rms_broeg       : RMS scatter of Broeg light curve (mmag)
        rms_uniform     : RMS scatter of uniform light curve (mmag)
        improvement_pct : scatter reduction percentage
    """
    ann_in = aperture_r * 1.5; ann_out = aperture_r * 2.5
    N_comp = len(comp_positions)
    times  = []; target_fluxes = []
    comp_fluxes = [[] for _ in range(N_comp)]
    errors = []

    for i, path in enumerate(fits_files):
        if log_cb and i % max(1, len(fits_files)//10) == 0:
            log_cb(f"  Photometry frame {i+1}/{len(fits_files)}…")
        try:
            data = astropy_fits.getdata(path).astype(float)
            if data.ndim == 3:
                data = data[0] if data.shape[0] <= 4 else \
                       0.299*data[0]+0.587*data[1]+0.114*data[2]
            hdr = astropy_fits.getheader(path)
            try:
                t_jd = float(Time(hdr.get('DATE-OBS',''), format='isot', scale='utc').jd)
            except Exception:
                t_jd = float(i)

            t_flux, t_snr = _measure_star(data, target_x, target_y,
                                           aperture_r, ann_in, ann_out)
            if t_flux <= 0: continue
            times.append(t_jd)
            target_fluxes.append(t_flux)
            errors.append(1.0857 / max(t_snr, 0.1))
            for k, (cx, cy) in enumerate(comp_positions):
                cf, _ = _measure_star(data, cx, cy, aperture_r, ann_in, ann_out)
                comp_fluxes[k].append(max(cf, 1e-5))
        except Exception as e:
            if log_cb: log_cb(f"  Frame {i} failed: {e}")

    if not times:
        return {'times': np.array([]), 'diff_mags_broeg': np.array([])}

    times          = np.array(times)
    target_fluxes  = np.array(target_fluxes)
    errors         = np.array(errors)
    comp_lcs       = np.array([np.array(c) for c in comp_fluxes])  # (N_comp, N_frames)

    # Trim to valid length
    N_frames = min(len(times), comp_lcs.shape[1])
    times         = times[:N_frames]
    target_fluxes = target_fluxes[:N_frames]
    comp_lcs      = comp_lcs[:, :N_frames]
    errors        = errors[:N_frames]

    # ── Broeg weights ─────────────────────────────────────────────────────
    if log_cb: log_cb(f"  Running Broeg optimisation ({N_comp} comp stars, {N_frames} frames)…")
    weight_final = broeg_optimal_weights(comp_lcs, n_iter=n_iter)

    # ── Differential magnitudes ───────────────────────────────────────────
    ens_broeg   = broeg_ensemble(comp_lcs, weight_final)
    ens_uniform = comp_lcs.mean(axis=0)

    dm_broeg   = differential_mag(target_fluxes, ens_broeg)
    dm_uniform = differential_mag(target_fluxes, ens_uniform)

    rms_broeg   = float(np.std(dm_broeg)   * 1000)   # mmag
    rms_uniform = float(np.std(dm_uniform) * 1000)
    improvement = float((1 - rms_broeg / (rms_uniform + 1e-10)) * 100)

    if log_cb:
        log_cb(f"  RMS uniform={rms_uniform:.2f}mmag  Broeg={rms_broeg:.2f}mmag  "
               f"improvement={improvement:.1f}%")
        for k, (w, pos) in enumerate(zip(weight_final, comp_positions)):
            log_cb(f"  Comp {k} ({pos[0]:.0f},{pos[1]:.0f}): weight={w:.4f}")

    return {
        'times':            times,
        'diff_mags_broeg':  dm_broeg,
        'diff_mags_uniform':dm_uniform,
        'errors':           errors,
        'comp_lcs':         comp_lcs,
        'weight_final':     weight_final,
        'rms_broeg':        rms_broeg,
        'rms_uniform':      rms_uniform,
        'improvement_pct':  improvement,
        'n_frames':         N_frames,
        'n_comp':           N_comp,
    }


def auto_detect_comp_stars_broeg(image: np.ndarray, target_x: float,
                                   target_y: float, n_max: int = 12,
                                   min_sep_px: float = 30.0) -> list[tuple[float,float]]:
    """
    Extended comp star selection: more candidates than variable_star.py,
    to give Broeg more to work with. Falls back to uniform if photutils unavailable.
    """
    if not HAS_PHOTUTILS:
        return []
    try:
        _, med, std = sigma_clipped_stats(image, sigma=3.0)
        finder  = DAOStarFinder(fwhm=4.0, threshold=8.*std)
        sources = finder(image - med)
        if sources is None or len(sources) == 0: return []
        cands = []
        for row in sources:
            x, y   = float(row['xcentroid']), float(row['ycentroid'])
            dist   = math.sqrt((x-target_x)**2 + (y-target_y)**2)
            if dist < min_sep_px: continue
            peak = float(row['peak'])
            if peak > 0.92 * float(image.max()): continue   # saturated
            if float(row['flux']) <= 0: continue
            cands.append((x, y, float(row['flux'])))
        cands.sort(key=lambda c: -c[2])
        return [(x, y) for x, y, _ in cands[:n_max]]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class BroegWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, fits_files, tx, ty, comp_pos, aper, n_iter):
        super().__init__()
        self.fits_files = fits_files; self.tx = tx; self.ty = ty
        self.comp_pos   = comp_pos;   self.aper = aper; self.n_iter = n_iter

    def run(self):
        try:
            def log(msg): self.progress.emit(-1, msg)
            self.progress.emit(10, "Starting photometry…")
            result = broeg_differential_photometry(
                self.fits_files, self.tx, self.ty, self.comp_pos,
                self.aper, self.n_iter, log_cb=log)
            self.progress.emit(100,
                f"Done!  RMS Broeg={result.get('rms_broeg',0):.2f}mmag  "
                f"uniform={result.get('rms_uniform',0):.2f}mmag  "
                f"improvement={result.get('improvement_pct',0):.1f}%")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow,QWidget{background:#0e1018;color:#d0d0e8;
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}
QTabWidget::pane{border:1px solid #202040;background:#111128;}
QTabBar::tab{background:#141430;color:#4040a0;padding:6px 16px;
    border:1px solid #202040;border-bottom:none;}
QTabBar::tab:selected{background:#111128;color:#a0a0ff;
    border-bottom:2px solid #6060e0;}
QGroupBox{border:1px solid #202040;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#4040a0;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#141430;color:#a0a0d8;border:1px solid #303090;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#1c1c48;border-color:#5050c0;}
QPushButton#run_btn{background:#0e0e28;color:#a0a0ff;
    border:1px solid #5050c0;font-weight:bold;padding:7px 20px;}
QPushButton#run_btn:hover{background:#141438;}
QSpinBox,QDoubleSpinBox,QComboBox{background:#0c0c1c;
    border:1px solid #202040;border-radius:3px;padding:3px 6px;color:#d0d0e0;}
QTextEdit{background:#080810;border:1px solid #202040;color:#6060e0;
    font-family:Consolas,monospace;font-size:10px;}
QTableWidget{background:#080810;alternate-background-color:#0a0a18;
    gridline-color:#202040;color:#d0d0e0;}
QTableWidget QHeaderView::section{background:#0c0c1c;color:#4040a0;
    border:1px solid #202040;padding:3px;font-weight:bold;}
QSplitter::handle{background:#202040;}
QLabel#title_lbl{font-size:17px;font-weight:bold;color:#a0a0ff;padding:6px;}
QLabel#sub_lbl{font-size:9px;color:#202040;padding:0 8px 4px;}
QProgressBar{background:#0c0c1c;border:1px solid #303090;border-radius:3px;height:8px;}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #203060,stop:1 #5060e0);border-radius:2px;}
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8,5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor='#0c0c18')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
    def redraw(self): self.fig.tight_layout(pad=0.4); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#080810')
    ax.set_title(title, color='#6060d0', fontsize=9, pad=3)
    ax.set_xlabel(xl, color='#282848', fontsize=8)
    ax.set_ylabel(yl, color='#282848', fontsize=8)
    ax.tick_params(colors='#282848', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#202040')


class BroegApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1360, 880)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._fits_files = []
        self._comp_pos   = []
        self._result     = None
        self._worker = self._wthread = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._controls())
        sp.addWidget(self._tabs())
        sp.setSizes([270,1090])
        rl.addWidget(sp,1)
        rl.addWidget(self._footer())

    def _header(self):
        hdr = QWidget(); hdr.setFixedHeight(60)
        hdr.setStyleSheet("background:#080810;border-bottom:1px solid #202040;")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12,0,12,0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                44,44,Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Broeg Optimal Photometry"); t.setObjectName("title_lbl")
        s = QLabel("Broeg (2005) comparison star weighting  ·  Auto down-weight variable/noisy comp stars  "
                   "·  Companion to variable_star.py")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(5)

        # FITS
        grp = QGroupBox("FITS Sequence"); gl = QVBoxLayout(grp)
        self.file_lbl = QLabel("No frames loaded")
        self.file_lbl.setStyleSheet("color:#202040;font-size:10px;")
        b1 = QPushButton("📂  Load FITS frames…"); b1.clicked.connect(self._load_fits)
        b1.setObjectName("run_btn")
        b_ex = QPushButton("📋  Synthetic example"); b_ex.clicked.connect(self._load_example)
        gl.addWidget(self.file_lbl); gl.addWidget(b1); gl.addWidget(b_ex)
        lay.addWidget(grp)

        # Target
        grp2 = QGroupBox("Target Star"); g2 = QGridLayout(grp2)
        g2.addWidget(QLabel("X (px):"),0,0); self.sp_tx=QDoubleSpinBox(); self.sp_tx.setRange(0,9999); g2.addWidget(self.sp_tx,0,1)
        g2.addWidget(QLabel("Y (px):"),1,0); self.sp_ty=QDoubleSpinBox(); self.sp_ty.setRange(0,9999); g2.addWidget(self.sp_ty,1,1)
        b_auto = QPushButton("🔍  Auto-detect comp stars"); b_auto.clicked.connect(self._auto_comp)
        g2.addWidget(b_auto,2,0,1,2)
        lay.addWidget(grp2)

        # Comp stars
        grp3 = QGroupBox("Comparison Stars"); g3 = QVBoxLayout(grp3)
        self.comp_lbl = QLabel("No comp stars set")
        self.comp_lbl.setStyleSheet("color:#202040;font-size:10px;")
        g3.addWidget(self.comp_lbl)
        lay.addWidget(grp3)

        # Parameters
        grp4 = QGroupBox("Parameters"); g4 = QGridLayout(grp4)
        g4.addWidget(QLabel("Aperture (px):"),0,0); self.sp_aper=QDoubleSpinBox(); self.sp_aper.setRange(2,50); self.sp_aper.setValue(8.0); g4.addWidget(self.sp_aper,0,1)
        g4.addWidget(QLabel("Broeg iterations:"),1,0); self.sp_iter=QSpinBox(); self.sp_iter.setRange(5,100); self.sp_iter.setValue(30); g4.addWidget(self.sp_iter,1,1)
        lay.addWidget(grp4)

        self.btn_run = QPushButton("▶  Run Broeg Photometry")
        self.btn_run.setObjectName("run_btn"); self.btn_run.setFixedHeight(36)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        lay.addWidget(self.btn_run)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color:#a0a0ff;font-size:10px;")
        self.summary_lbl.setWordWrap(True); lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_lc(),      "📈  Light Curve")
        self.tabs.addTab(self._tab_weights(), "⚖  Comp Star Weights")
        self.tabs.addTab(self._tab_comp(),    "📊  Comp Diagnostics")
        self.tabs.addTab(self._tab_log(),     "📋  Log")
        return self.tabs

    def _tab_lc(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.lc_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.lc_plot)
        return w

    def _tab_weights(self):
        w = QWidget(); main = QHBoxLayout(w)
        self.w_plot = MplCanvas(figsize=(7,6)); main.addWidget(self.w_plot,1)
        self.w_table = QTableWidget(0,3)
        self.w_table.setHorizontalHeaderLabels(["Comp","Weight","σ (mmag)"])
        self.w_table.setFixedWidth(260)
        self.w_table.setAlternatingRowColors(True)
        self.w_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        main.addWidget(self.w_table)
        return w

    def _tab_comp(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.comp_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.comp_plot)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    def _footer(self):
        foot = QWidget(); foot.setFixedHeight(22)
        foot.setStyleSheet("background:#080810;border-top:1px solid #202040;")
        lay = QHBoxLayout(foot); lay.setContentsMargins(6,0,6,0)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color:#202040;font-size:10px;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(200,12); self.pbar.setVisible(False)
        lay.addWidget(self.status_lbl,1); lay.addWidget(self.pbar)
        return foot

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_fits(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,"Load FITS frames","","FITS (*.fits *.fit);;All (*)")
        if not paths: return
        self._fits_files = sorted(paths)
        self.file_lbl.setText(f"{len(paths)} frames loaded")
        self.file_lbl.setStyleSheet("color:#a0a0ff;font-size:10px;")
        self.btn_run.setEnabled(bool(self._comp_pos))
        self._log(f"Loaded {len(paths)} frames")

    def _load_example(self):
        """Synthetic example: variable target + 8 comp stars (2 secretly variable)."""
        import tempfile; td = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(42)
        N   = 512; N_frames = 80

        # Target: sinusoidal variable with period 50 frames, amplitude 0.1 mag
        base_target = 8000.0
        target_var  = base_target * 10**(-0.1 * 0.05 * np.sin(
            2*np.pi*np.arange(N_frames)/50))

        # 8 comp stars at various positions and fluxes
        comp_configs = [
            (120,150,12000,0.0),  # stable bright
            (300,200,9000,0.0),   # stable
            (180,350,7500,0.0),   # stable
            (400,100,6000,0.08),  # mildly variable
            (80,300,5500,0.0),    # stable
            (350,350,4000,0.15),  # variable (bad comp)
            (200,80,3500,0.0),    # stable faint
            (450,250,2500,0.0),   # stable faint
        ]
        # Airmass trend
        airmass = 1.0 + 0.3*np.linspace(0,1,N_frames)

        self._fits_files = []
        for i in range(N_frames):
            img = np.random.default_rng(i).normal(100, 15, (N,N))
            # Target
            ty, tx = 256, 256
            yy,xx = np.ogrid[:N,:N]
            flux_t = target_var[i] * airmass[i]
            img += flux_t * np.exp(-((xx-tx)**2+(yy-ty)**2)/(2*3.0**2))
            # Comp stars
            for cx,cy,base,amp in comp_configs:
                var = 10**(-0.1*amp*np.sin(2*np.pi*i/30+0.5)) if amp>0 else 1.0
                flux_c = base * var * airmass[i]
                img += flux_c * np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*3.0**2))
            p = td/f"frame_{i:04d}.fits"
            astropy_fits.PrimaryHDU(img.astype(np.float32)).writeto(str(p),overwrite=True)
            self._fits_files.append(str(p))

        self.sp_tx.setValue(256); self.sp_ty.setValue(256)
        self._comp_pos = [(cx,cy) for cx,cy,_,_ in comp_configs]
        self.comp_lbl.setText(f"{len(self._comp_pos)} comp stars  "
                              f"(2 are intentionally variable — Broeg should down-weight them)")
        self.comp_lbl.setStyleSheet("color:#a0a0ff;font-size:10px;")
        self.file_lbl.setText(f"{N_frames} synthetic frames")
        self.file_lbl.setStyleSheet("color:#a0a0ff;font-size:10px;")
        self.btn_run.setEnabled(True)
        self._log(f"Synthetic example: {N_frames} frames, target amplitude=0.1mag")
        self._log(f"Comp stars 3 (amp=0.08) and 5 (amp=0.15) are variable")
        self._log(f"Broeg should assign these near-zero weight")

    def _auto_comp(self):
        if not self._fits_files: return
        try:
            data = astropy_fits.getdata(self._fits_files[0]).astype(float)
            if data.ndim == 3: data = data[0]
            tx, ty = self.sp_tx.value(), self.sp_ty.value()
            self._comp_pos = auto_detect_comp_stars_broeg(data, tx, ty)
            self.comp_lbl.setText(f"{len(self._comp_pos)} comp stars auto-detected")
            self.comp_lbl.setStyleSheet("color:#a0a0ff;font-size:10px;")
            self.btn_run.setEnabled(bool(self._comp_pos and self._fits_files))
            self._log(f"Auto-detected {len(self._comp_pos)} comparison stars")
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if not self._fits_files or not self._comp_pos: return
        self.btn_run.setEnabled(False)
        self._wthread = QThread(self)
        self._worker  = BroegWorker(
            self._fits_files,
            self.sp_tx.value(), self.sp_ty.value(),
            self._comp_pos,
            self.sp_aper.value(),
            self.sp_iter.value())
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _on_progress(self, pct, msg):
        self.status_lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True); self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))
        self._log(msg)

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}"); self.btn_run.setEnabled(True)

    def _on_finished(self, result):
        self._result = result; self.btn_run.setEnabled(True)
        r = result
        self.summary_lbl.setText(
            f"RMS Broeg={r['rms_broeg']:.2f}mmag\n"
            f"RMS uniform={r['rms_uniform']:.2f}mmag\n"
            f"Improvement={r['improvement_pct']:.1f}%")
        self._log(f"\n{'='*45}")
        self._log(f"✓  Broeg photometry complete")
        self._log(f"   RMS Broeg   = {r['rms_broeg']:.3f} mmag")
        self._log(f"   RMS uniform = {r['rms_uniform']:.3f} mmag")
        self._log(f"   Improvement = {r['improvement_pct']:.1f}%")
        for k, w in enumerate(r['weight_final']):
            pos = self._comp_pos[k] if k < len(self._comp_pos) else ('?','?')
            self._log(f"   Comp {k} ({pos[0]:.0f},{pos[1]:.0f}): w={w:.4f}")
        self._log(f"{'='*45}\n")
        self._draw_lc(r); self._draw_weights(r); self._draw_comp(r)
        self.tabs.setCurrentIndex(0)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_lc(self, r):
        fig = self.lc_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c0c18')
        axes = fig.subplots(2,1, sharex=True)

        t = r['times'] - r['times'][0]
        _ax(axes[0], f"Light curve comparison  (Broeg RMS={r['rms_broeg']:.2f}mmag  "
                     f"uniform RMS={r['rms_uniform']:.2f}mmag)","","Δmag")
        axes[0].errorbar(t, r['diff_mags_broeg']*1000, yerr=r['errors']*1000,
                         fmt='o', color='#6060e0', ms=3, lw=1, alpha=0.8,
                         label=f"Broeg ({r['rms_broeg']:.2f}mmag)")
        axes[0].plot(t, r['diff_mags_uniform']*1000, '.', color='#a06060',
                     ms=2, alpha=0.5, label=f"Uniform ({r['rms_uniform']:.2f}mmag)")
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=7, facecolor='#0c0c18',
                       edgecolor='#202040', labelcolor='white')

        _ax(axes[1], "Difference: Broeg − Uniform","Time (days)","Δ (mmag)")
        diff = (r['diff_mags_broeg'] - r['diff_mags_uniform'])*1000
        axes[1].plot(t, diff, '-', color='#40c090', lw=1, alpha=0.8)
        axes[1].axhline(0, color='#303060', lw=0.8)
        self.lc_plot.redraw()

    def _draw_weights(self, r):
        fig = self.w_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c0c18')
        ax = fig.add_subplot(111)
        _ax(ax, "Broeg weights per comp star","Comp star","Weight")
        N = len(r['weight_final'])
        w = r['weight_final']
        colors = ['#6060e0' if wi > 0.1 else '#e06060' if wi < 0.01 else '#c0a030'
                  for wi in w]
        bars = ax.bar(range(N), w, color=colors, edgecolor='#0c0c18')
        ax.axhline(1/N, color='#404060', lw=1, ls='--', label='Uniform (1/N)')
        ax.legend(fontsize=7, facecolor='#0c0c18', edgecolor='#202040',
                  labelcolor='white')
        ax.set_xticks(range(N)); ax.set_xticklabels([f"C{k}" for k in range(N)])
        for bar, wi in zip(bars, w):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{wi:.3f}", ha='center', va='bottom', fontsize=7, color='white')
        self.w_plot.redraw()

        # Fill table
        self.w_table.setRowCount(N)
        comp_lcs = r['comp_lcs']
        for k in range(N):
            # Individual comp scatter relative to ensemble
            ens = broeg_ensemble(comp_lcs, r['weight_final'])
            ratio = comp_lcs[k] / (ens + 1e-10)
            sigma = float(np.std(ratio/ratio.mean())) * 1000
            pos   = self._comp_pos[k] if k<len(self._comp_pos) else ('?','?')
            for col, val in enumerate([f"C{k} ({pos[0]:.0f},{pos[1]:.0f})",
                                        f"{r['weight_final'][k]:.4f}",
                                        f"{sigma:.2f}"]):
                self.w_table.setItem(k, col, QTableWidgetItem(val))

    def _draw_comp(self, r):
        fig = self.comp_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c0c18')
        comp_lcs = r['comp_lcs']
        N = min(comp_lcs.shape[0], 8)
        cols = min(N, 4); rows = math.ceil(N/cols)
        if N == 0: return
        axes = fig.subplots(rows, cols) if N > 1 else [[fig.add_subplot(111)]]
        if rows == 1 and N > 1: axes = [axes]
        ax_flat = [a for row in axes for a in (row if hasattr(row,'__iter__') else [row])]

        t = r['times'] - r['times'][0]
        ens = broeg_ensemble(comp_lcs, r['weight_final'])
        for k, ax in enumerate(ax_flat[:N]):
            w_k = r['weight_final'][k]
            ratio = comp_lcs[k] / (ens + 1e-10); ratio /= ratio.mean()
            dm = (ratio - 1)*1000
            sigma = float(np.std(dm))
            color = '#6060e0' if w_k > 0.1 else '#e06060' if w_k < 0.01 else '#c0a030'
            _ax(ax, f"Comp {k}  w={w_k:.3f}  σ={sigma:.1f}mmag")
            ax.plot(t, dm, '.', color=color, ms=2, alpha=0.7)
            ax.axhline(0, color='#303060', lw=0.8)
        for ax in ax_flat[N:]: ax.set_visible(False)
        self.comp_plot.redraw()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = BroegApp(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
