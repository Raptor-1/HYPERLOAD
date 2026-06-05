r"""
HYPERLOAD — Comet Coma Analyser
================================
Standalone script AND pipeline module.
Place in: C:\Users\Marcell\Desktop\Siril Suites\

Can be used two ways:

  1. STANDALONE — run directly, load a comet-stacked FITS file:
         python comet_coma_analyser.py

  2. PIPELINE INTEGRATION — import and call from comet_pipeline.py:
         from comet_coma_analyser import launch_analyser
         analyser = launch_analyser(
             stack        = comet_stack_array,
             nucleus_pos  = (cy, cx),
             comet_name   = config["comet_name"],
             pixel_scale_arcsec = 0.5,   # optional — enables Afρ in cm
             r_hel_AU     = 1.5,          # optional — heliocentric distance
             delta_AU     = 1.2,          # optional — geocentric distance
         )
         analyser.show()

Algorithms:
  Radial surface brightness profile (azimuthal median — robust to stars)
  Power-law + Gaussian nucleus surface brightness fit
  Degree of condensation (DC, A'Hearn scale 0–9)
  Afρ dust production proxy (A'Hearn et al. 1984, ApJ 278, 849)
    Afρ [cm] = 4 · Δ² [cm] · r² [AU] · F_comet(ρ) / (ρ [cm] · F_sun)
  Growth curve: Afρ vs aperture radius (detects active vs quiescent)
  Coma morphology: azimuthal median subtraction → jet/fan detection
  Jet position angle extraction from azimuthal brightness profile

References:
  A'Hearn et al. 1984 — Afρ diagnostic, ApJ 278, 849
  Farnham et al. 2000 — standard techniques, ApJS 128, 519
  Cochran & Schleicher 1993 — DC scale, Icarus 105, 235

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, traceback, math
import numpy as np
from pathlib import Path
from datetime import datetime

# ── optional pipeline integration ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log,"a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\ncomet_coma_analyser.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import linregress

try:
    from astropy.io import fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QFormLayout, QLineEdit, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon, QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Comet Coma Analyser"

# Physical constants
AU_TO_CM  = 1.49598e13   # 1 AU in cm
AU_TO_KM  = 1.49598e8    # 1 AU in km
M_SUN_V   = -26.74       # Solar V magnitude


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS  (pure functions — importable independently)
# ═══════════════════════════════════════════════════════════════════════════════

def refine_nucleus(img: np.ndarray, cx: float, cy: float,
                   box: int = 15) -> tuple[float, float]:
    """
    Sub-pixel nucleus centroid via flux-weighted centroid in a small box.
    Returns (cx_refined, cy_refined).
    """
    h, w = img.shape
    y0 = max(0, int(cy) - box); y1 = min(h, int(cy) + box + 1)
    x0 = max(0, int(cx) - box); x1 = min(w, int(cx) + box + 1)
    stamp = img[y0:y1, x0:x1].astype(float)
    stamp -= stamp.min()
    total = stamp.sum()
    if total < 1e-10:
        return float(cx), float(cy)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((stamp * xx).sum() / total), float((stamp * yy).sum() / total)


def radial_profile(img: np.ndarray, cx: float, cy: float,
                   max_r: float = None, n_bins: int = 80,
                   use_median: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Azimuthally averaged radial surface brightness profile.
    Uses median per annulus (robust against stars) if use_median=True.

    Returns (r_px, brightness) arrays.
    """
    ny, nx = img.shape
    yy, xx = np.ogrid[:ny, :nx]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2).ravel()
    v = img.ravel()
    if max_r is None:
        max_r = min(cx, cy, nx - cx, ny - cy) - 1
    bins  = np.linspace(0, max_r, n_bins + 1)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    prof  = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if mask.sum() > 0:
            prof[i] = np.median(v[mask]) if use_median else v[mask].mean()
    return r_mid, prof


def fit_surface_brightness(r: np.ndarray, I: np.ndarray,
                           fit_range: tuple = (1, 50)
                           ) -> dict:
    """
    Fit surface brightness profile with nucleus Gaussian + coma power law.

    Model: S(r) = I_nuc · exp(-r²/(2·σ²)) + I_coma · r^(-α)

    Returns dict with fit parameters and power-law-only fit for comparison.
    """
    r1, r2 = fit_range
    mask = (r >= r1) & (r < r2) & (I > 0)
    if mask.sum() < 5:
        return {'success': False}

    def model(r_, I_nuc, sigma, I_coma, alpha):
        nuc  = I_nuc * np.exp(-r_**2 / (2 * sigma**2))
        coma = np.where(r_ > 0.5, I_coma * r_**(-alpha), I_coma)
        return nuc + coma

    try:
        I_peak = float(I[0])
        popt, pcov = curve_fit(
            model, r[mask], I[mask],
            p0=[I_peak * 0.7, 3.0, I_peak * 0.1, 1.0],
            bounds=([0, 0.3, 0, 0.2], [I_peak * 2, 30, I_peak, 3.5]),
            maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        I_fit = model(r[mask], *popt)
        ss_res = float(np.sum((I[mask] - I_fit)**2))
        ss_tot = float(np.sum((I[mask] - I[mask].mean())**2))
        r2_val = 1 - ss_res / (ss_tot + 1e-30)
    except Exception:
        popt  = [float(I[0]), 3.0, float(I[1]), 1.0]
        perr  = [0, 0, 0, 0]; r2_val = 0.0

    # Power-law only fit on outer coma (r > 5px) in log-log space
    outer = (r > 5) & (r < r2) & (I > 0)
    alpha_simple = -1.0
    if outer.sum() > 3:
        slope, _, _, _, _ = linregress(np.log(r[outer]), np.log(I[outer] + 1))
        alpha_simple = -float(slope)

    return {
        'success':    True,
        'I_nuc':      float(popt[0]),
        'sigma_nuc':  float(popt[1]),
        'I_coma':     float(popt[2]),
        'alpha':      float(popt[3]),
        'alpha_err':  float(perr[3]),
        'alpha_simple': alpha_simple,
        'r2':         float(r2_val),
        'model_fn':   lambda r_: model(r_, *popt),
    }


def afro_profile(r_px: np.ndarray, brightness: np.ndarray,
                 pixel_scale_arcsec: float,
                 r_hel_AU: float, delta_AU: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Afρ [cm] as a function of aperture radius.
    (A'Hearn et al. 1984, ApJ 278, 849)

    Afρ = (4 · Δ² [cm²] · r_hel² [AU²]) / (ρ [cm]) · (F_comet / F_sun_at_1AU)

    Since we work with unnormalised pixel values, this returns a relative
    Afρ profile in units of [px·AU²/arcsec].  Scale to absolute [cm] if
    an absolute flux calibration is available.

    For a steady-state coma Afρ is constant with aperture.
    Rising Afρ → outgassing / active source.
    Falling Afρ → radiation pressure / old dust tail.

    Returns (rho_km, afro_relative) arrays.
    """
    # Aperture radius in km at the comet
    arcsec_to_rad = 4.848136e-6
    rho_km = r_px * pixel_scale_arcsec * arcsec_to_rad * delta_AU * AU_TO_KM

    # Cumulative flux within each aperture (approximate: sum annuli × 2πr)
    dr     = np.gradient(r_px)
    annulus_area = 2 * np.pi * r_px * dr
    cum_flux = np.cumsum(brightness * annulus_area)

    # Relative Afρ ∝ cum_flux × Δ² × r² / ρ
    denom = rho_km + 1e-10
    afro  = cum_flux * delta_AU**2 * r_hel_AU**2 / denom
    return rho_km, afro


def degree_of_condensation(r_px: np.ndarray, brightness: np.ndarray) -> float:
    """
    Estimate the degree of condensation (DC) on A'Hearn 0–9 scale.
    Based on brightness ratio inner coma / outer coma.
    DC=0: completely diffuse; DC=9: stellar nucleus.
    """
    if len(brightness) < 10 or brightness.max() < 1e-6:
        return 0.0
    i_inner = min(3, len(brightness) - 1)
    i_outer = min(30, len(brightness) - 1)
    inner = float(brightness[i_inner])
    outer = float(brightness[i_outer])
    if outer < 1e-6:
        return 9.0
    ratio = inner / outer
    dc = min(9.0, max(0.0, math.log10(max(ratio, 1.0)) * 3.0))
    return dc


def coma_morphology(img: np.ndarray, cx: float, cy: float,
                    r_inner: float = 10.0, r_outer: float = 120.0,
                    smooth_sigma: float = 1.5) -> dict:
    """
    Reveal jets and fans by subtracting the azimuthal median at each radius.

    Returns:
      enhanced    : morphology-enhanced image
      azimuthal   : azimuthal profile at r=r_inner..r_outer
      jet_pa_deg  : list of jet position angles (degrees from N)
    """
    ny, nx = img.shape
    yy, xx = np.ogrid[:ny, :nx]
    r   = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    phi = np.degrees(np.arctan2(xx - cx, -(yy - cy))) % 360  # PA from North

    enhanced = img.copy().astype(float)

    for ri in np.arange(r_inner, r_outer, 2.0):
        ann = (r >= ri - 1) & (r < ri + 1)
        if ann.sum() > 3:
            med = np.median(img[ann])
            enhanced[ann] -= med

    enhanced = gaussian_filter(np.maximum(enhanced, 0), smooth_sigma)

    # Azimuthal profile: sum of enhanced signal as function of PA
    n_pa  = 72  # 5° bins
    pa_bins = np.linspace(0, 360, n_pa + 1)
    pa_mid  = 0.5 * (pa_bins[:-1] + pa_bins[1:])
    az_prof = np.zeros(n_pa)
    ann_mask = (r >= r_inner) & (r < r_outer)

    for i in range(n_pa):
        pa_mask = (phi >= pa_bins[i]) & (phi < pa_bins[i+1]) & ann_mask
        if pa_mask.sum() > 0:
            az_prof[i] = np.mean(enhanced[pa_mask])

    # Detect jet PAs: peaks above 1.5× median in azimuthal profile
    med_az = np.median(az_prof)
    std_az = np.std(az_prof)
    jet_pa = [float(pa_mid[i]) for i in range(n_pa)
              if az_prof[i] > med_az + 1.5 * std_az]

    return {
        'enhanced':   enhanced,
        'pa_bins':    pa_mid,
        'az_profile': az_prof,
        'jet_pa_deg': jet_pa,
    }


def luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2: return data.astype(float)
    if data.ndim == 3 and data.shape[0] == 3:
        return (0.299*data[0]+0.587*data[1]+0.114*data[2]).astype(float)
    if data.ndim == 3 and data.shape[2] == 3:
        return (0.299*data[:,:,0]+0.587*data[:,:,1]+0.114*data[:,:,2]).astype(float)
    if data.ndim == 3: return data[0].astype(float)
    return data.astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class ComaWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, image, cx, cy, params):
        super().__init__()
        self.image = image; self.cx = cx; self.cy = cy
        self.params = params

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p = self.params
        img = self.image
        max_r = p.get('max_r', min(self.cx, self.cy,
                                   img.shape[1]-self.cx,
                                   img.shape[0]-self.cy) - 5)

        # 1. Refine nucleus
        self.progress.emit(5, "Refining nucleus centroid…")
        cx, cy = refine_nucleus(img, self.cx, self.cy,
                                box=p.get('refine_box', 15))

        # 2. Radial profile
        self.progress.emit(20, "Computing radial profile…")
        r_px, prof = radial_profile(img, cx, cy,
                                    max_r=max_r,
                                    n_bins=p.get('n_bins', 100))

        # 3. Surface brightness fit
        self.progress.emit(40, "Fitting surface brightness…")
        sb_fit = fit_surface_brightness(r_px, prof,
                                        fit_range=(1, min(60, max_r)))

        # 4. DC
        self.progress.emit(55, "Computing degree of condensation…")
        dc = degree_of_condensation(r_px, prof)

        # 5. Afρ
        self.progress.emit(65, "Computing Afρ profile…")
        afro_result = None
        pix = p.get('pixel_scale_arcsec')
        r_hel = p.get('r_hel_AU'); delta = p.get('delta_AU')
        if pix and r_hel and delta:
            rho_km, afro_vals = afro_profile(r_px, prof, pix, r_hel, delta)
            afro_result = {'rho_km': rho_km, 'afro': afro_vals}

        # 6. Morphology
        self.progress.emit(80, "Extracting coma morphology…")
        r_inner = p.get('morph_r_inner', max(5.0, max_r * 0.05))
        r_outer = p.get('morph_r_outer', min(max_r * 0.6, 150.0))
        morph = coma_morphology(img, cx, cy,
                                r_inner=r_inner, r_outer=r_outer)

        self.progress.emit(100, f"Done!  DC={dc:.1f}  jets={len(morph['jet_pa_deg'])}")
        self.finished.emit({
            'image':      img,
            'cx': cx, 'cy': cy,
            'r_px':       r_px,
            'prof':       prof,
            'sb_fit':     sb_fit,
            'dc':         dc,
            'afro':       afro_result,
            'morph':      morph,
            'max_r':      max_r,
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET  (matches comet_pipeline.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

SIRIL_BG       = "#1e2128"
SIRIL_BG2      = "#252930"
SIRIL_BG3      = "#2d3240"
SIRIL_ACCENT   = "#4a9eff"
SIRIL_ACCENT2  = "#2d6abf"
SIRIL_TEXT     = "#dde3ee"
SIRIL_TEXT_DIM = "#7a8499"
SIRIL_BORDER   = "#3a4055"
SIRIL_SUCCESS  = "#4caf7d"
SIRIL_WARNING  = "#e8a23a"
SIRIL_SECTION  = "#5ba3ff"
SIRIL_ERROR    = "#cc4444"

SIRIL_STYLESHEET = f"""
QMainWindow, QDialog, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}}
QTabWidget::pane {{
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
    background: {SIRIL_BG2};
}}
QTabBar::tab {{
    background: {SIRIL_BG3};
    color: {SIRIL_TEXT_DIM};
    border: 1px solid {SIRIL_BORDER};
    padding: 6px 16px;
    margin-right: 2px;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    background: {SIRIL_BG2};
    color: {SIRIL_ACCENT};
    border-bottom: 2px solid {SIRIL_ACCENT};
}}
QGroupBox {{
    background-color: {SIRIL_BG2};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 5px;
    margin-top: 8px; padding: 8px;
    font-weight: bold; color: {SIRIL_SECTION};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QPushButton {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px; padding: 6px 12px;
}}
QPushButton:hover {{ background-color: {SIRIL_ACCENT2}; border-color: {SIRIL_ACCENT}; color: white; }}
QPushButton#primary {{
    background-color: {SIRIL_ACCENT2}; border-color: {SIRIL_ACCENT};
    color: white; font-weight: bold;
}}
QPushButton#primary:hover {{ background-color: {SIRIL_ACCENT}; }}
QPushButton#export {{
    background-color: {SIRIL_BG3}; border-color: {SIRIL_SUCCESS};
    color: {SIRIL_SUCCESS}; font-weight: bold;
}}
QPushButton#export:hover {{ background-color: #1a3d2a; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {SIRIL_BG3}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 3px; padding: 4px 6px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {SIRIL_ACCENT};
}}
QTextEdit, QPlainTextEdit {{
    background-color: #181c22; color: {SIRIL_SUCCESS};
    font-family: 'Courier New', monospace; font-size: 9pt;
    border: 1px solid {SIRIL_BORDER};
}}
QScrollBar:vertical {{
    background: {SIRIL_BG2}; width: 10px; border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {SIRIL_BORDER}; border-radius: 5px; min-height: 20px;
}}
QProgressBar {{
    background: {SIRIL_BG3}; border: 1px solid {SIRIL_BORDER};
    border-radius: 3px; height: 8px;
}}
QProgressBar::chunk {{ background: {SIRIL_ACCENT}; border-radius: 3px; }}
QLabel#section {{ color: {SIRIL_SECTION}; font-weight: bold; padding-top: 4px; }}
QLabel#dim     {{ color: {SIRIL_TEXT_DIM}; font-size: 9pt; }}
QLabel#ok      {{ color: {SIRIL_SUCCESS}; font-weight: bold; }}
QLabel#err     {{ color: {SIRIL_ERROR};   font-weight: bold; }}
QLabel#warn    {{ color: {SIRIL_WARNING}; font-weight: bold; }}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS
# ═══════════════════════════════════════════════════════════════════════════════

class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(7, 5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor=SIRIL_BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)

    def redraw(self):
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#141820')
    ax.set_title(title, color=SIRIL_SECTION, fontsize=9, pad=3)
    ax.set_xlabel(xl, color=SIRIL_TEXT_DIM, fontsize=8)
    ax.set_ylabel(yl, color=SIRIL_TEXT_DIM, fontsize=8)
    ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(SIRIL_BORDER)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class CometComaApp(QMainWindow):

    def __init__(self, image: np.ndarray = None,
                 nucleus_pos: tuple = None,
                 comet_name: str = '',
                 pixel_scale_arcsec: float = None,
                 r_hel_AU: float = None,
                 delta_AU: float = None):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1380, 900)
        self.setStyleSheet(SIRIL_STYLESHEET)

        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._image    = image
        self._nucleus  = nucleus_pos   # (cy, cx)
        self._result   = None
        self._worker   = self._wthread = None

        self._build_ui(comet_name, pixel_scale_arcsec, r_hel_AU, delta_AU)

        if image is not None and nucleus_pos is not None:
            self._draw_input_preview()
            cy, cx = nucleus_pos
            self.sp_cx.setValue(round(cx, 1))
            self.sp_cy.setValue(round(cy, 1))
            self._log(f"Received from pipeline: {image.shape[1]}×{image.shape[0]}px")
            self._log(f"Nucleus: X={cx:.1f}  Y={cy:.1f}")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self, comet_name, pix, r_hel, delta):
        root = QWidget(); self.setCentralWidget(root)
        rl   = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._make_header(comet_name))
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._make_controls(pix, r_hel, delta))
        sp.addWidget(self._make_tabs())
        sp.setSizes([280, 1100])
        rl.addWidget(sp, 1)
        rl.addWidget(self._make_footer())

    def _make_header(self, comet_name):
        hdr = QWidget(); hdr.setFixedHeight(56)
        hdr.setStyleSheet(f"background:{SIRIL_BG};border-bottom:1px solid {SIRIL_BORDER};")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12, 0, 12, 0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                44, 44, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        title = f"Comet Coma Analyser{f'  —  {comet_name}' if comet_name else ''}"
        t = QLabel(title)
        t.setStyleSheet(f"color:{SIRIL_ACCENT};font-size:17px;font-weight:bold;")
        s = QLabel("Radial profile  ·  Surface brightness fit  ·  "
                   "Degree of condensation  ·  Afρ dust production  ·  Jet morphology")
        s.setStyleSheet(f"color:{SIRIL_TEXT_DIM};font-size:9pt;")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v, 1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _make_controls(self, pix, r_hel, delta):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay   = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)

        # ── Load ──────────────────────────────────────────────────────────
        grp = QGroupBox("Input"); gl = QVBoxLayout(grp)
        self.file_lbl = QLabel("No image loaded")
        self.file_lbl.setStyleSheet(f"color:{SIRIL_TEXT_DIM};font-size:9pt;")
        self.file_lbl.setWordWrap(True)
        b1 = QPushButton("📂  Open comet stack FITS…")
        b1.clicked.connect(self._load_fits)
        b1.setObjectName("primary")
        gl.addWidget(self.file_lbl); gl.addWidget(b1)
        lay.addWidget(grp)

        # ── Nucleus ────────────────────────────────────────────────────────
        grp2 = QGroupBox("Nucleus Position"); g2 = QFormLayout(grp2)
        self.sp_cx = QDoubleSpinBox(); self.sp_cx.setRange(0,9999); self.sp_cx.setDecimals(1)
        self.sp_cy = QDoubleSpinBox(); self.sp_cy.setRange(0,9999); self.sp_cy.setDecimals(1)
        self.cb_refine = QCheckBox("Auto-refine centroid"); self.cb_refine.setChecked(True)
        g2.addRow("X (px):", self.sp_cx); g2.addRow("Y (px):", self.sp_cy)
        g2.addRow(self.cb_refine)
        b_auto = QPushButton("🔍  Auto-detect nucleus"); b_auto.clicked.connect(self._auto_nucleus)
        g2.addRow(b_auto)
        lay.addWidget(grp2)

        # ── Analysis ───────────────────────────────────────────────────────
        grp3 = QGroupBox("Analysis Parameters"); g3 = QFormLayout(grp3)
        self.sp_maxr   = QSpinBox(); self.sp_maxr.setRange(20,1000); self.sp_maxr.setValue(200)
        self.sp_nbins  = QSpinBox(); self.sp_nbins.setRange(20,300); self.sp_nbins.setValue(100)
        self.sp_refbox = QSpinBox(); self.sp_refbox.setRange(5,50);  self.sp_refbox.setValue(15)
        g3.addRow("Max radius (px):", self.sp_maxr)
        g3.addRow("Profile bins:",    self.sp_nbins)
        g3.addRow("Refine box (px):", self.sp_refbox)
        lay.addWidget(grp3)

        # ── Ephemeris ──────────────────────────────────────────────────────
        grp4 = QGroupBox("Ephemeris  (for Afρ in cm)"); g4 = QFormLayout(grp4)
        self.sp_pix   = QDoubleSpinBox(); self.sp_pix.setRange(0,100); self.sp_pix.setDecimals(4)
        self.sp_rh    = QDoubleSpinBox(); self.sp_rh.setRange(0,100);  self.sp_rh.setDecimals(4)
        self.sp_delta = QDoubleSpinBox(); self.sp_delta.setRange(0,100);self.sp_delta.setDecimals(4)
        if pix:   self.sp_pix.setValue(pix)
        if r_hel: self.sp_rh.setValue(r_hel)
        if delta: self.sp_delta.setValue(delta)
        g4.addRow("Pixel scale (″/px):", self.sp_pix)
        g4.addRow("r_hel (AU):",          self.sp_rh)
        g4.addRow("Δ geocentric (AU):",   self.sp_delta)
        eph_note = QLabel("Leave 0 if unknown — Afρ\nwill be in relative units.")
        eph_note.setObjectName("dim"); g4.addRow(eph_note)
        lay.addWidget(grp4)

        # ── Morphology ─────────────────────────────────────────────────────
        grp5 = QGroupBox("Morphology"); g5 = QFormLayout(grp5)
        self.sp_rin  = QDoubleSpinBox(); self.sp_rin.setRange(2,100); self.sp_rin.setValue(10)
        self.sp_rout = QDoubleSpinBox(); self.sp_rout.setRange(10,500);self.sp_rout.setValue(120)
        g5.addRow("Inner r (px):", self.sp_rin)
        g5.addRow("Outer r (px):", self.sp_rout)
        lay.addWidget(grp5)

        # ── Run ────────────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Analyse Coma")
        self.btn_run.setObjectName("primary"); self.btn_run.setFixedHeight(36)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_export = QPushButton("💾  Export report…")
        self.btn_export.setObjectName("export")
        self.btn_export.setEnabled(False); self.btn_export.clicked.connect(self._export)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_cancel)
        lay.addWidget(self.btn_export)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setObjectName("ok")
        self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)

        scroll.setWidget(inner); return scroll

    def _make_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_input(),    "🌠  Image")
        self.tabs.addTab(self._tab_profile(),  "📊  Radial Profile")
        self.tabs.addTab(self._tab_nucleus(),  "🔆  Nucleus & DC")
        self.tabs.addTab(self._tab_afro(),     "☄  Afρ / Activity")
        self.tabs.addTab(self._tab_morph(),    "💨  Morphology")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_input(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.img_plot = MplCanvas(figsize=(8,6)); lay.addWidget(self.img_plot)
        return w

    def _tab_profile(self):
        w = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        self.cb_loglog = QCheckBox("Log-log scale"); self.cb_loglog.setChecked(True)
        self.cb_loglog.stateChanged.connect(self._refresh_profile)
        ctrl.addWidget(self.cb_loglog); ctrl.addStretch(); lay.addLayout(ctrl)
        self.prof_plot = MplCanvas(figsize=(8,6)); lay.addWidget(self.prof_plot)
        return w

    def _tab_nucleus(self):
        w = QWidget(); main = QHBoxLayout(w)
        self.nuc_plot = MplCanvas(figsize=(7,6))
        main.addWidget(self.nuc_plot, 1)
        # Stats panel
        stats = QWidget(); stats.setFixedWidth(260)
        sl = QVBoxLayout(stats); sl.setContentsMargins(8,8,8,8)
        self.nuc_stats = QTextEdit(); self.nuc_stats.setReadOnly(True)
        sl.addWidget(QLabel("Analysis Results:")); sl.addWidget(self.nuc_stats)
        main.addWidget(stats)
        return w

    def _tab_afro(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.afro_plot = MplCanvas(figsize=(8,6)); lay.addWidget(self.afro_plot)
        return w

    def _tab_morph(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.morph_plot = MplCanvas(figsize=(8,6)); lay.addWidget(self.morph_plot)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    def _make_footer(self):
        foot = QWidget(); foot.setFixedHeight(22)
        foot.setStyleSheet(f"background:{SIRIL_BG};border-top:1px solid {SIRIL_BORDER};")
        lay = QHBoxLayout(foot); lay.setContentsMargins(8,0,8,0)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet(f"color:{SIRIL_TEXT_DIM};font-size:9pt;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(180,10)
        self.pbar.setVisible(False)
        lay.addWidget(self.status_lbl,1); lay.addWidget(self.pbar)
        return foot

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open comet stack FITS", "",
            "FITS (*.fits *.fit *.fts);;All (*)")
        if not path: return
        try:
            data = fits.getdata(path)
            self._image = luminance(data)
            ny, nx = self._image.shape
            self.file_lbl.setText(Path(path).name)
            self.file_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS};font-size:9pt;")
            self.sp_cx.setValue(nx/2); self.sp_cy.setValue(ny/2)
            self.sp_maxr.setValue(int(min(nx,ny)/2 - 10))
            self.sp_rout.setValue(min(120, int(min(nx,ny)/4)))
            self._nucleus = (ny/2, nx/2)
            self._auto_nucleus()
            self._draw_input_preview()
            self.btn_run.setEnabled(True)
            self._log(f"Loaded: {path}  {nx}×{ny}px")
        except Exception as e:
            QMessageBox.critical(self,"Load error",str(e))

    def _auto_nucleus(self):
        if self._image is None: return
        from scipy.ndimage import gaussian_filter, label, center_of_mass
        img = self._image.astype(float)
        blur = gaussian_filter(img, sigma=8.0)
        thresh = np.percentile(blur, 70)
        binary = (blur > thresh).astype(int)
        labs, n = label(binary)
        best_lbl = None; best_bright = -1
        for k in range(1, n+1):
            mask = labs==k
            if mask.sum() < 20: continue
            bright = np.sum(blur[mask])
            if bright > best_bright:
                best_bright = bright; best_lbl = k
        if best_lbl:
            mask = labs==best_lbl
            cy,cx = center_of_mass(blur*mask)
            self.sp_cx.setValue(round(float(cx),1))
            self.sp_cy.setValue(round(float(cy),1))
            self._nucleus = (cy,cx)
            self._log(f"Auto-detected nucleus: X={cx:.1f}  Y={cy:.1f}")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if self._image is None:
            QMessageBox.warning(self,"No image","Load an image first."); return
        cx = self.sp_cx.value(); cy = self.sp_cy.value()
        pix   = self.sp_pix.value()   or None
        r_hel = self.sp_rh.value()    or None
        delta = self.sp_delta.value() or None
        params = {
            'max_r':            self.sp_maxr.value(),
            'n_bins':           self.sp_nbins.value(),
            'refine_box':       self.sp_refbox.value(),
            'pixel_scale_arcsec': pix,
            'r_hel_AU':         r_hel,
            'delta_AU':         delta,
            'morph_r_inner':    self.sp_rin.value(),
            'morph_r_outer':    self.sp_rout.value(),
        }
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_export.setEnabled(False)

        self._wthread = QThread(self)
        self._worker  = ComaWorker(self._image, cx, cy, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _cancel(self):
        if self._wthread and self._wthread.isRunning():
            self._wthread.quit()
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status_lbl.setText(msg)
        self.pbar.setVisible(True); self.pbar.setValue(pct)
        if pct >= 100:
            QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self,"Error",msg[:400])

    def _on_finished(self, result):
        self._result = result
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.btn_export.setEnabled(True)

        r    = result
        sb   = r['sb_fit']
        dc   = r['dc']
        jets = r['morph']['jet_pa_deg']
        alpha_str = f"{sb['alpha']:.3f}" if sb.get('success') else 'N/A'
        self.summary_lbl.setText(
            f"α={alpha_str}  DC={dc:.1f}  {len(jets)} jet(s)\n"
            f"nucleus σ={sb.get('sigma_nuc',0):.1f}px")

        self._log(f"\n{'='*50}")
        self._log(f"✓  Analysis complete")
        self._log(f"   Power-law index α  = {alpha_str}")
        self._log(f"   DC                 = {dc:.1f} / 9")
        if sb.get('success'):
            self._log(f"   Nucleus σ          = {sb['sigma_nuc']:.2f} px")
            self._log(f"   Fit R²             = {sb['r2']:.4f}")
        self._log(f"   Jet PA(s)          = {jets}")
        self._log(f"{'='*50}\n")

        self._draw_input_preview()
        self._refresh_profile()
        self._draw_nucleus()
        self._draw_afro()
        self._draw_morphology()
        self.tabs.setCurrentIndex(1)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_input_preview(self):
        if self._image is None: return
        img = self._image
        fig = self.img_plot.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)
        ax = fig.add_subplot(111); _ax(ax, "Comet stack  (log stretch)")
        lo, hi = np.percentile(img[img > 0], [0.5, 99.8]) if img.max() > 0 else (0,1)
        ax.imshow(np.log1p(np.clip(img-lo,0,None)),
                  origin='lower', cmap='inferno',
                  aspect='equal', interpolation='nearest')
        # Nucleus marker
        r = self._result
        cx = float(r['cx']) if r else self.sp_cx.value()
        cy = float(r['cy']) if r else self.sp_cy.value()
        ax.plot(cx, cy, '+', color=SIRIL_SUCCESS,
                ms=18, mew=2, label=f"Nucleus ({cx:.1f},{cy:.1f})")
        if r:
            max_r = r['max_r']
            import matplotlib.patches as mpatches
            ax.add_patch(mpatches.Circle((cx,cy), max_r,
                ec=SIRIL_ACCENT, fc='none', lw=1, ls='--', alpha=0.5))
        ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                  labelcolor=SIRIL_TEXT, fontsize=8, loc='upper right')
        self.img_plot.redraw()

    def _refresh_profile(self):
        if not self._result: return
        r   = self._result
        r_px = r['r_px']; prof = r['prof']; sb = r['sb_fit']
        fig  = self.prof_plot.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)
        ax = fig.add_subplot(111)
        loglog = self.cb_loglog.isChecked()
        _ax(ax, "Surface brightness profile",
            "Radius (px)", "Surface brightness (ADU)")

        good = prof > 0
        if loglog and good.sum() > 3:
            ax.loglog(r_px[good], prof[good],
                      '.-', color=SIRIL_ACCENT, lw=1.5, ms=4, label='Data (median)')
            if sb.get('success') and sb.get('model_fn'):
                r_fine = np.linspace(r_px[1], r_px[good][-1], 200)
                ax.loglog(r_fine, np.maximum(sb['model_fn'](r_fine), 1e-10),
                          '--', color=SIRIL_WARNING, lw=1.5,
                          label=f"Fit  α={sb['alpha']:.3f}  R²={sb['r2']:.3f}")
            # Reference 1/r line
            scale = prof[good][0] * r_px[good][0]
            ax.loglog(r_px[good], scale/r_px[good],
                      ':', color='#666688', lw=1, label='1/r (steady coma)')
        else:
            ax.plot(r_px, prof, '.-', color=SIRIL_ACCENT, lw=1.5, ms=4, label='Data')
            if sb.get('success') and sb.get('model_fn'):
                r_fine = np.linspace(r_px[0], r_px[-1], 300)
                ax.plot(r_fine, np.maximum(sb['model_fn'](r_fine),0),
                        '--', color=SIRIL_WARNING, lw=1.5,
                        label=f"Fit  α={sb['alpha']:.3f}")

        ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                  labelcolor=SIRIL_TEXT, fontsize=8)
        self.prof_plot.redraw()

    def _draw_nucleus(self):
        if not self._result: return
        r = self._result; sb = r['sb_fit']

        # Nucleus stamp (inner region)
        cx, cy   = r['cx'], r['cy']
        img      = r['image']
        box = 40
        ny, nx   = img.shape
        y0 = max(0,int(cy)-box); y1 = min(ny,int(cy)+box+1)
        x0 = max(0,int(cx)-box); x1 = min(nx,int(cx)+box+1)
        stamp = img[y0:y1, x0:x1]

        fig = self.nuc_plot.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)
        axes = fig.subplots(1,2)

        # Left: nucleus image
        _ax(axes[0], "Nucleus (inner 80×80 px)")
        lo,hi = np.percentile(stamp,[0.5,99.5])
        axes[0].imshow(stamp, origin='lower', cmap='hot',
                       vmin=lo, vmax=hi, aspect='equal',
                       interpolation='nearest')
        axes[0].plot(cx-x0, cy-y0, '+', color='cyan', ms=16, mew=2)

        # Right: radial profile close-up
        _ax(axes[1], "Inner profile + Gaussian nucleus",
            "Radius (px)", "SB (ADU)")
        r_px = r['r_px']; prof = r['prof']
        close = r_px < 30
        axes[1].plot(r_px[close], prof[close], '.-',
                     color=SIRIL_ACCENT, lw=1.5, ms=5, label='Data')
        if sb.get('success') and sb.get('model_fn'):
            r_fine = np.linspace(0, 30, 200)
            axes[1].plot(r_fine, np.maximum(sb['model_fn'](r_fine),0),
                         '--', color=SIRIL_WARNING, lw=2,
                         label=f"Fit: σ={sb['sigma_nuc']:.2f}px")
        axes[1].legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                       labelcolor=SIRIL_TEXT, fontsize=8)
        self.nuc_plot.redraw()

        # Stats text
        dc_labels = {
            (0,1): "0–1: Completely diffuse, no condensation",
            (1,3): "1–3: Weakly condensed, broad coma",
            (3,5): "3–5: Moderately condensed",
            (5,7): "5–7: Strongly condensed, bright core",
            (7,9): "7–9: Nearly stellar nucleus",
        }
        dc = r['dc']
        dc_desc = next((v for k,v in dc_labels.items() if k[0]<=dc<k[1]),
                       "9: Stellar nucleus")
        lines = [
            f"Nucleus position:",
            f"  X = {cx:.2f} px",
            f"  Y = {cy:.2f} px",
            f"",
            f"Surface brightness fit:",
            f"  α (power law)  = {sb.get('alpha','N/A'):.3f}" if sb.get('success') else "  Fit: N/A",
            f"  α (log-log)    = {sb.get('alpha_simple',0):.3f}" if sb.get('success') else "",
            f"  Nucleus σ      = {sb.get('sigma_nuc',0):.2f} px" if sb.get('success') else "",
            f"  Fit R²         = {sb.get('r2',0):.4f}" if sb.get('success') else "",
            f"",
            f"Degree of condensation:",
            f"  DC = {dc:.1f} / 9",
            f"  {dc_desc}",
        ]
        self.nuc_stats.setPlainText('\n'.join(lines))

    def _draw_afro(self):
        if not self._result: return
        r = self._result; afro = r['afro']

        fig = self.afro_plot.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)

        if afro is not None:
            ax = fig.add_subplot(111)
            _ax(ax, "Afρ vs aperture radius  (A'Hearn et al. 1984)",
                "ρ (km)", "Afρ (relative)")
            ax.semilogx(afro['rho_km'], afro['afro'],
                        color=SIRIL_ACCENT, lw=2)
            ax.fill_between(afro['rho_km'], afro['afro'],
                            alpha=0.15, color=SIRIL_ACCENT)
            # Annotations
            mid_idx = len(afro['rho_km'])//2
            ax.axhline(afro['afro'][mid_idx], color=SIRIL_WARNING,
                       lw=1, ls='--',
                       label=f"Afρ ≈ {afro['afro'][mid_idx]:.2f} (r.u.)")
            ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                      labelcolor=SIRIL_TEXT, fontsize=8)
            ax.text(0.97, 0.05,
                    "Flat → steady state\nRising → active\nFalling → tail/old dust",
                    transform=ax.transAxes, ha='right', va='bottom',
                    color=SIRIL_TEXT_DIM, fontsize=8)
        else:
            ax = fig.add_subplot(111); _ax(ax, "Afρ — ephemeris not provided")
            ax.text(0.5, 0.5,
                    "Set pixel scale, r_hel and Δ\nin the control panel to compute Afρ",
                    transform=ax.transAxes, ha='center', va='center',
                    color=SIRIL_TEXT_DIM, fontsize=10)

        self.afro_plot.redraw()

    def _draw_morphology(self):
        if not self._result: return
        r    = self._result
        morph = r['morph']

        fig = self.morph_plot.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)
        axes = fig.subplots(1, 3)

        # Original
        _ax(axes[0], "Comet stack (log)")
        img = r['image']
        lo,hi = np.percentile(img,[0.1,99.9])
        axes[0].imshow(np.log1p(np.clip(img-lo,0,None)),
                       origin='lower', cmap='inferno',
                       aspect='equal', interpolation='nearest')
        axes[0].plot(r['cx'],r['cy'],'+',color=SIRIL_SUCCESS,ms=14,mew=2)

        # Enhanced
        _ax(axes[1], "Azimuthal-subtracted (jets)")
        enh = morph['enhanced']
        lo2,hi2 = np.percentile(enh,[1,99])
        axes[1].imshow(enh, origin='lower', cmap='RdBu_r',
                       vmin=lo2, vmax=hi2, aspect='equal',
                       interpolation='nearest')
        axes[1].plot(r['cx'],r['cy'],'+',color='white',ms=14,mew=2)
        for pa in morph['jet_pa_deg']:
            length = min(r['max_r']*0.4, 100)
            pa_r   = math.radians(pa)
            dx, dy = math.sin(pa_r)*length, math.cos(pa_r)*length
            axes[1].annotate('', xy=(r['cx']+dx, r['cy']-dy),
                             xytext=(r['cx'], r['cy']),
                             arrowprops=dict(arrowstyle='->', color=SIRIL_WARNING, lw=1.5))

        # Azimuthal profile
        _ax(axes[2], "Azimuthal profile (jet PA)", "PA (°)", "Mean brightness")
        pa_mid  = morph['pa_bins']
        az_prof = morph['az_profile']
        axes[2].plot(pa_mid, az_prof, color=SIRIL_ACCENT, lw=1.5)
        axes[2].fill_between(pa_mid, az_prof, alpha=0.2, color=SIRIL_ACCENT)
        med = np.median(az_prof); std = np.std(az_prof)
        axes[2].axhline(med + 1.5*std, color=SIRIL_WARNING,
                        lw=1, ls='--', label='1.5σ threshold')
        for pa in morph['jet_pa_deg']:
            axes[2].axvline(pa, color=SIRIL_ERROR, lw=1.2, alpha=0.8)
        if morph['jet_pa_deg']:
            axes[2].legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                           labelcolor=SIRIL_TEXT, fontsize=7)

        self.morph_plot.redraw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        if not self._result: return
        path, _ = QFileDialog.getSaveFileName(
            self,"Export coma analysis","comet_coma_report.txt",
            "Text (*.txt);;All (*)")
        if not path: return
        r = self._result; sb = r['sb_fit']
        lines = [
            "HYPERLOAD — Comet Coma Analysis Report",
            f"Generated: {datetime.now().isoformat()}",
            "="*60,
            f"Nucleus position: X={r['cx']:.2f}  Y={r['cy']:.2f} px",
            "",
            "SURFACE BRIGHTNESS PROFILE",
            f"  Power-law index α   = {sb.get('alpha','N/A'):.4f}" if sb.get('success') else "  Fit failed",
            f"  Power-law (log-log) = {sb.get('alpha_simple',0):.4f}" if sb.get('success') else "",
            f"  Nucleus σ           = {sb.get('sigma_nuc',0):.3f} px" if sb.get('success') else "",
            f"  Fit R²              = {sb.get('r2',0):.5f}" if sb.get('success') else "",
            "",
            f"DEGREE OF CONDENSATION: DC = {r['dc']:.2f} / 9",
            "",
            "RADIAL PROFILE (r_px, brightness):",
        ] + [f"  {rr:.2f}\t{bb:.4f}"
             for rr,bb in zip(r['r_px'], r['prof'])]
        if r['afro']:
            lines += [
                "",
                "Afρ PROFILE (rho_km, afro_relative):",
            ] + [f"  {rk:.1f}\t{av:.4f}"
                 for rk,av in zip(r['afro']['rho_km'], r['afro']['afro'])]
        lines += [
            "",
            f"JET POSITION ANGLES: {r['morph']['jet_pa_deg']}",
        ]
        with open(path,'w') as f:
            f.write('\n'.join(str(l) for l in lines))
        self._log(f"✓ Exported: {path}")

    def _log(self, msg):
        self.log.append(msg)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE INTEGRATION  (call from comet_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════

def launch_analyser(stack: np.ndarray,
                    nucleus_pos: tuple,
                    comet_name: str = '',
                    pixel_scale_arcsec: float = None,
                    r_hel_AU: float = None,
                    delta_AU: float = None,
                    parent=None) -> CometComaApp:
    """
    Launch the Comet Coma Analyser pre-loaded with data from comet_pipeline.py.

    Usage in comet_pipeline.py (add to _on_stacks_ready or _on_finished):

        try:
            from comet_coma_analyser import launch_analyser
            analyser = launch_analyser(
                stack       = comet_stack_array,   # (H,W) float numpy array
                nucleus_pos = (cy, cx),             # from track_nucleus_sequence()
                comet_name  = config['comet_name'],
            )
            analyser.show()
        except ImportError:
            pass

    Parameters
    ----------
    stack               : (H,W) or (3,H,W) float array — comet-registered stack
    nucleus_pos         : (cy, cx) in pixels
    comet_name          : string shown in title bar
    pixel_scale_arcsec  : arcsec/pixel — enables Afρ calculation
    r_hel_AU            : heliocentric distance [AU]
    delta_AU            : geocentric distance [AU]
    parent              : optional parent QWidget

    Returns
    -------
    CometComaApp instance (call .show() to display)
    """
    img = luminance(stack) if stack.ndim == 3 else stack.astype(float)
    win = CometComaApp(
        image              = img,
        nucleus_pos        = nucleus_pos,
        comet_name         = comet_name,
        pixel_scale_arcsec = pixel_scale_arcsec,
        r_hel_AU           = r_hel_AU,
        delta_AU           = delta_AU,
    )
    win.btn_run.setEnabled(True)
    return win


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    win = CometComaApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
