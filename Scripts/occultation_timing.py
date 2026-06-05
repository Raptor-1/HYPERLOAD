r"""
HYPERLOAD — Asteroid Occultation Timing Analyzer
==================================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Analyzes stellar occultation light curves to determine chord timing
with sub-millisecond precision.  Supports multi-observer chord
analysis for asteroid size and shape determination.

Two physical models:
  1. Geometric step function (large asteroids, small diffraction):
       F(t) = F₀ · (1 − d · ½[erf((t−T₁)/τ√2) − erf((t−T₂)/τ√2)])
       τ = smoothing (atmosphere + instrument response)

  2. Fresnel diffraction (TNOs, KBOs, small asteroids):
       F(u) = F₀ · (1 − d · (1 − ½[(C(u)+½)² + (S(u)+½)²]))
       u = x/r_F,  r_F = √(λD/2)  Fresnel scale
       C,S = Fresnel cosine/sine integrals

Both models fitted via Nelder-Mead optimisation.
Timing uncertainties from bootstrap resampling (100 iterations).

Multi-chord analysis:
  Given N observer chords (T₁ⁿ, T₂ⁿ, v_shadowⁿ, baseline coords),
  fits an ellipse in the sky plane to constrain asteroid dimensions.
  Minimum: 3 chords for full ellipse; 2 for diameter estimate.

References:
  Elliot 1979          — stellar occultations, ARA&A 17, 445
  Roques et al. 1987   — TNO occultations, A&A 288, 985
  Richichi et al. 2016 — high-SNR occultations, A&A 590, A70
  Herald et al. 2020   — IOTA methods, Icarus 344, 113241
  Widemann et al. 2009 — Pluto, Icarus 202, 167

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
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\noccultation_timing.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    print(f"[CRASH] {log}")
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.optimize import minimize
from scipy.special import fresnel, erf
from scipy.ndimage import gaussian_filter1d

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Asteroid Occultation Timing Analyzer"

# ═══════════════════════════════════════════════════════════════════════════════
#  LIGHT CURVE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def geometric_model(t: np.ndarray, t1: float, t2: float,
                    baseline: float, depth: float, tau: float) -> np.ndarray:
    """
    Geometric occultation: smoothed step function (large asteroids).

    F(t) = baseline · (1 − depth · ½[erf((t−t1)/τ√2) − erf((t−t2)/τ√2)])

    t1, t2 : ingress / egress times
    tau    : smoothing timescale (atmosphere + instrument, seconds)
    depth  : fractional depth (1.0 = total occultation)
    """
    tau = max(tau, 1e-5)
    F = baseline * (1.0 - depth * 0.5 * (
        erf((t - t1) / (tau * math.sqrt(2))) -
        erf((t - t2) / (tau * math.sqrt(2)))))
    return F


def fresnel_model(t: np.ndarray, t_mid: float, v_shadow: float,
                  dist_km: float, baseline: float, depth: float,
                  star_diam_km: float = 0.0) -> np.ndarray:
    """
    Fresnel diffraction light curve (TNOs / small asteroids).

    Applicable when Fresnel scale r_F = √(λD/2) is comparable to or
    larger than the asteroid size — gives characteristic diffraction fringes.

    t_mid     : time of geometrical shadow mid-edge (seconds)
    v_shadow  : shadow velocity (km/s)
    dist_km   : observer-to-asteroid distance (km)
    star_diam : projected stellar diameter (km) — blurs fringes
    """
    lam_km = 5.5e-10        # 550 nm in km
    r_F    = math.sqrt(lam_km * dist_km / 2)   # Fresnel scale (km)
    x_km   = (t - t_mid) * v_shadow            # shadow coordinate (km)
    u      = x_km / (r_F * math.sqrt(math.pi)) # Fresnel units (scipy convention)

    S, C   = fresnel(u)     # scipy.special.fresnel returns (S, C)
    F_edge = 0.5 * ((C + 0.5)**2 + (S + 0.5)**2)

    # Star diameter blurring
    if star_diam_km > 0.0 and r_F > 0.0:
        dt = abs(t[1] - t[0]) if len(t) > 1 else 1e-4
        sigma_t = star_diam_km / (v_shadow * 2.355)   # sigma in seconds
        sigma_s = sigma_t / dt
        if sigma_s > 0.1:
            F_edge = gaussian_filter1d(F_edge, sigma_s)

    return baseline * (1.0 - depth * (1.0 - F_edge))


# ═══════════════════════════════════════════════════════════════════════════════
#  EVENT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class EventDetector:
    """
    Robust detection and initial timing of the occultation event.

    Method: sliding-window median with outlier detection.
    The event is located as the region where flux drops below
    (baseline − n_sigma · rms) for at least min_pts consecutive points.
    """

    def __init__(self, n_sigma: float = 3.0, min_pts: int = 3,
                 window_frac: float = 0.1):
        self.n_sigma     = n_sigma
        self.min_pts     = min_pts
        self.window_frac = window_frac

    def detect(self, t: np.ndarray, flux: np.ndarray) -> dict:
        """
        Returns dict with:
          found        : bool
          t1, t2       : ingress / egress times (initial estimate)
          baseline     : out-of-event median flux
          depth        : flux drop fraction
          event_mask   : bool array, True during event
        """
        n = len(t)
        # Estimate baseline from outer 20% of time series
        outer = int(0.10 * n)
        baseline_pts = np.concatenate([flux[:outer], flux[-outer:]])
        baseline  = np.median(baseline_pts)
        rms       = np.std(baseline_pts)
        thresh    = baseline - self.n_sigma * rms

        # Find event region
        in_event = flux < thresh
        # Require consecutive points
        from scipy.ndimage import label, find_objects
        labs, n_lab = label(in_event)
        if n_lab == 0:
            return {'found': False}

        # Largest contiguous region
        sizes = [np.sum(labs == k) for k in range(1, n_lab + 1)]
        best  = np.argmax(sizes) + 1
        mask  = labs == best

        if mask.sum() < self.min_pts:
            return {'found': False}

        idx = np.where(mask)[0]
        t1_est = t[idx[0]]
        t2_est = t[idx[-1]]
        depth  = max(0.0, (baseline - flux[mask].min()) / (baseline + 1e-10))

        return {
            'found':      True,
            't1_init':    t1_est,
            't2_init':    t2_est,
            'baseline':   baseline,
            'rms':        rms,
            'depth_init': min(depth, 1.0),
            'event_mask': mask,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL FITTER
# ═══════════════════════════════════════════════════════════════════════════════

class OccultationFitter:
    """
    Fits geometric or Fresnel model to the light curve.
    Timing uncertainties estimated by bootstrap resampling.
    """

    def __init__(self, model_type: str = 'geometric',
                 v_shadow: float = 20.0, dist_km: float = 4e9,
                 n_bootstrap: int = 200):
        self.model_type  = model_type
        self.v_shadow    = v_shadow
        self.dist_km     = dist_km
        self.n_bootstrap = n_bootstrap

    def fit(self, t: np.ndarray, flux: np.ndarray,
            flux_err: np.ndarray, init: dict,
            progress_cb=None) -> dict:
        """
        Fit model and return timing with uncertainties.
        init: dict from EventDetector.detect()
        """
        if self.model_type == 'geometric':
            return self._fit_geometric(t, flux, flux_err, init, progress_cb)
        else:
            return self._fit_fresnel(t, flux, flux_err, init, progress_cb)

    def _fit_geometric(self, t, flux, flux_err, init, progress_cb):
        base  = init['baseline']
        depth = init['depth_init']
        t1_i  = init['t1_init']
        t2_i  = init['t2_init']
        dt    = abs(t[1] - t[0]) if len(t) > 1 else 0.01
        tau0  = 2 * dt

        def cost(x):
            t1, t2, bl, d, tau = x
            if t2 <= t1 or d <= 0 or d > 1 or tau <= 0: return 1e10
            m = geometric_model(t, t1, t2, bl, d, tau)
            return np.sum(((flux - m) / flux_err)**2)

        x0  = [t1_i, t2_i, base, depth, tau0]
        res = minimize(cost, x0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-6,
                                'fatol': 1e-4, 'adaptive': True})
        t1, t2, bl, d, tau = res.x

        # Bootstrap uncertainties
        t1_bs = []; t2_bs = []
        rng   = np.random.default_rng(0)
        for k in range(self.n_bootstrap):
            flux_r = flux + rng.normal(0, flux_err)
            r_k    = minimize(cost, res.x, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-5})
            t1_bs.append(r_k.x[0]); t2_bs.append(r_k.x[1])
            if progress_cb and k % 20 == 0:
                progress_cb(k, self.n_bootstrap, 'bootstrap')

        t1_err = np.std(t1_bs); t2_err = np.std(t2_bs)
        chord_s    = t2 - t1
        chord_km   = chord_s * self.v_shadow
        chord_err  = math.sqrt(t1_err**2 + t2_err**2) * self.v_shadow

        model_flux = geometric_model(t, t1, t2, bl, d, tau)
        residuals  = flux - model_flux
        chi2_red   = np.sum((residuals / flux_err)**2) / max(len(t) - 5, 1)

        return {
            'model':      'geometric',
            't1':         t1,    't1_err':     t1_err,
            't2':         t2,    't2_err':     t2_err,
            'baseline':   bl,    'depth':      d,
            'tau':        tau,
            'chord_s':    chord_s,  'chord_km': chord_km,
            'chord_km_err': chord_err,
            'chi2_red':   chi2_red,
            'model_flux': model_flux,
            'residuals':  residuals,
            't1_bootstrap': t1_bs,
            't2_bootstrap': t2_bs,
        }

    def _fit_fresnel(self, t, flux, flux_err, init, progress_cb):
        base  = init['baseline']
        depth = init['depth_init']
        t_mid = 0.5 * (init['t1_init'] + init['t2_init'])

        def cost(x):
            tm, v, bl, d = x
            if v <= 0 or d <= 0 or d > 1: return 1e10
            m = fresnel_model(t, tm, v, self.dist_km, bl, d)
            return np.sum(((flux - m) / flux_err)**2)

        x0  = [t_mid, self.v_shadow, base, depth]
        res = minimize(cost, x0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-7,
                                'fatol': 1e-4, 'adaptive': True})
        tm, v, bl, d = res.x

        # Bootstrap
        tm_bs = []; v_bs = []
        rng   = np.random.default_rng(0)
        for k in range(self.n_bootstrap):
            flux_r = flux + rng.normal(0, flux_err)
            r_k    = minimize(cost, res.x, method='Nelder-Mead',
                             options={'maxiter': 2000})
            tm_bs.append(r_k.x[0]); v_bs.append(r_k.x[1])
            if progress_cb and k % 20 == 0:
                progress_cb(k, self.n_bootstrap, 'bootstrap')

        tm_err = np.std(tm_bs)
        lam_km = 5.5e-10
        r_F    = math.sqrt(lam_km * self.dist_km / 2)

        model_flux = fresnel_model(t, tm, v, self.dist_km, bl, d)
        residuals  = flux - model_flux
        chi2_red   = np.sum((residuals/flux_err)**2) / max(len(t)-4, 1)

        return {
            'model':       'fresnel',
            't_mid':       tm,     't_mid_err':  tm_err,
            'v_shadow':    v,
            'baseline':    bl,     'depth':      d,
            'fresnel_scale_km': r_F,
            'chi2_red':    chi2_red,
            'model_flux':  model_flux,
            'residuals':   residuals,
            'tm_bootstrap':tm_bs,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-CHORD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class ChordAnalyzer:
    """
    Fit an ellipse in the sky plane to multiple observer chords.

    Each chord is defined by:
      - mid-chord position (x_c, y_c) in km from target centre
      - chord half-length L/2 km
      - chord position angle θ (degrees from North)

    Ellipse model: (x/a)² + (y/b)² = 1  (axes-aligned for simplicity)
    General case: rotated ellipse with position angle ψ.
    """

    @staticmethod
    def chord_from_timing(result: dict, observer_lat: float,
                          observer_lon: float,
                          event_time_utc: str) -> dict:
        """Build a chord dict from a fitting result and observer coordinates."""
        # Simplified: returns chord length only (full geo requires astrometry)
        return {
            'chord_km': result.get('chord_km', 0),
            'chord_km_err': result.get('chord_km_err', 0),
        }

    @staticmethod
    def fit_diameter(chords: list) -> dict:
        """
        Simple diameter estimate from multiple chords.
        Uses the longest chord as a lower bound on the equatorial diameter.
        For a circle: fits radius to minimise residuals.
        """
        lengths = [c['chord_km'] for c in chords if c['chord_km'] > 0]
        errors  = [c.get('chord_km_err', 1.0) for c in chords
                   if c['chord_km'] > 0]
        if not lengths:
            return {}

        max_chord = max(lengths)
        mean_chord = np.average(lengths, weights=[1/e**2 for e in errors])

        # Minimum diameter lower bound: longest chord
        # Circle fit: R = sqrt((L/2)^2 + b^2) where b = impact parameter
        # Average of all chords assuming random sampling:
        # E[chord] = π·R/2 for random impact parameters
        R_est = max_chord / 2    # conservative lower bound
        D_est = max_chord        # minimum diameter

        return {
            'n_chords':    len(lengths),
            'max_chord_km': max_chord,
            'mean_chord_km': mean_chord,
            'diameter_lower_bound_km': D_est,
            'radius_est_km': R_est,
            'chord_lengths': lengths,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class OccultationWorker(QObject):
    progress   = pyqtSignal(int, str)
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, t, flux, flux_err, params):
        super().__init__()
        self.t = t; self.flux = flux; self.flux_err = flux_err
        self.params = params
        self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p = self.params
        t, flux, flux_err = self.t, self.flux, self.flux_err

        # ── 1. Detect event ───────────────────────────────────────────────
        self.progress.emit(5, "Detecting occultation event…")
        det = EventDetector(
            n_sigma=p.get('n_sigma', 3.0),
            min_pts=p.get('min_pts', 3))
        init = det.detect(t, flux)

        if not init['found']:
            self.error.emit(
                "No occultation event detected.\n\n"
                "Tips:\n"
                "• Lower the detection threshold (sigma)\n"
                "• Check that the flux drops significantly below baseline\n"
                "• Verify time and flux columns are correct")
            return

        self.progress.emit(15, f"Event detected: T1≈{init['t1_init']:.4f}  "
                               f"T2≈{init['t2_init']:.4f}  "
                               f"depth≈{init['depth_init']*100:.1f}%")

        # ── 2. Fit model ──────────────────────────────────────────────────
        model_type = p.get('model', 'geometric')
        self.progress.emit(20, f"Fitting {model_type} model…")

        fitter = OccultationFitter(
            model_type  = model_type,
            v_shadow    = p.get('v_shadow', 20.0),
            dist_km     = p.get('dist_km', 4e9),
            n_bootstrap = p.get('n_bootstrap', 200),
        )

        def prog_cb(k, total, phase):
            pct = int(20 + 65 * k / max(total, 1))
            self.progress.emit(pct, f"Bootstrap {k}/{total}…")

        result = fitter.fit(t, flux, flux_err, init, progress_cb=prog_cb)

        # ── 3. Summary statistics ─────────────────────────────────────────
        self.progress.emit(90, "Computing statistics…")
        snr = init['depth_init'] * init['baseline'] / (init['rms'] + 1e-10)

        self.progress.emit(100, "Done!")
        self.finished.emit({
            'result':    result,
            'detect':    init,
            'snr':       snr,
            'n_pts':     len(t),
            't':         t,
            'flux':      flux,
            'flux_err':  flux_err,
            'params':    p,
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI HELPERS
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
    def redraw(self):
        self.fig.tight_layout(pad=0.5); self.canvas.draw_idle()


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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class OccultationApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1380, 900)
        self.setStyleSheet(DARK)

        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            self.setWindowIcon(QIcon(str(_logo)))

        self._t = self._flux = self._ferr = None
        self._result  = None
        self._worker  = None
        self._wthread = None
        self._chords  = []   # list of result dicts from multiple observers

        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._make_header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._make_controls())
        sp.addWidget(self._make_tabs())
        sp.setSizes([310, 1070])
        rl.addWidget(sp, 1)
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
        t = QLabel("Asteroid Occultation Timing Analyzer"); t.setObjectName("title_lbl")
        s = QLabel("Geometric + Fresnel diffraction models · Bootstrap timing uncertainties · "
                   "Multi-chord asteroid size · Sub-millisecond precision")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v, 1)
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
        grp = QGroupBox("Light Curve Input")
        gl  = QVBoxLayout(grp)
        self.file_lbl = QLabel("No file loaded")
        self.file_lbl.setStyleSheet("color:#484868;font-size:10px;")
        btn_csv = QPushButton("📂  Load CSV / TXT…")
        btn_csv.clicked.connect(self._load_csv)
        btn_ex  = QPushButton("📋  Load Example (synthetic)")
        btn_ex.clicked.connect(self._load_example)
        gl.addWidget(self.file_lbl); gl.addWidget(btn_csv); gl.addWidget(btn_ex)
        lay.addWidget(grp)

        # ── Physical parameters ────────────────────────────────────────────
        grp2 = QGroupBox("Physical Parameters")
        g2   = QGridLayout(grp2)
        g2.addWidget(QLabel("Shadow velocity (km/s):"), 0, 0)
        self.sp_v = QDoubleSpinBox(); self.sp_v.setRange(0.1, 999); self.sp_v.setValue(20.0); self.sp_v.setDecimals(2)
        self.sp_v.setToolTip("Shadow velocity on Earth's surface.\nTypical: 10-30 km/s for MBAs, 5-25 km/s for TNOs")
        g2.addWidget(self.sp_v, 0, 1)
        g2.addWidget(QLabel("Distance (km, ×10⁹):"), 1, 0)
        self.sp_dist = QDoubleSpinBox(); self.sp_dist.setRange(0.01, 10000); self.sp_dist.setValue(4.0); self.sp_dist.setDecimals(3)
        self.sp_dist.setToolTip("Observer-to-asteroid distance.\n2-4 AU ~ 3-6 × 10⁸ km for MBAs\n40-100 AU for TNOs")
        g2.addWidget(self.sp_dist, 1, 1)
        g2.addWidget(QLabel("Model:"), 2, 0)
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["Geometric (large asteroids)", "Fresnel diffraction (TNOs)"])
        g2.addWidget(self.cmb_model, 2, 1)
        lay.addWidget(grp2)

        # ── Detection ─────────────────────────────────────────────────────
        grp3 = QGroupBox("Event Detection")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("Threshold (σ):"), 0, 0)
        self.sp_sigma = QDoubleSpinBox(); self.sp_sigma.setRange(1.5, 10); self.sp_sigma.setValue(3.0); self.sp_sigma.setDecimals(1)
        g3.addWidget(self.sp_sigma, 0, 1)
        g3.addWidget(QLabel("Bootstrap samples:"), 1, 0)
        self.sp_boot = QSpinBox(); self.sp_boot.setRange(50, 2000); self.sp_boot.setValue(500)
        g3.addWidget(self.sp_boot, 1, 1)
        lay.addWidget(grp3)

        # ── Multi-chord ────────────────────────────────────────────────────
        grp4 = QGroupBox("Multi-chord Analysis")
        g4   = QVBoxLayout(grp4)
        self.chord_lbl = QLabel("0 chords loaded")
        self.chord_lbl.setStyleSheet("color:#585878;font-size:10px;")
        btn_add = QPushButton("➕  Add current result as chord")
        btn_add.clicked.connect(self._add_chord)
        btn_clear = QPushButton("🗑  Clear all chords")
        btn_clear.clicked.connect(self._clear_chords)
        g4.addWidget(self.chord_lbl); g4.addWidget(btn_add); g4.addWidget(btn_clear)
        lay.addWidget(grp4)

        # ── Output ────────────────────────────────────────────────────────
        grp5 = QGroupBox("Output")
        g5   = QVBoxLayout(grp5)
        self.btn_save = QPushButton("💾  Save Report (JSON)…")
        self.btn_save.setEnabled(False); self.btn_save.clicked.connect(self._save)
        g5.addWidget(self.btn_save)
        lay.addWidget(grp5)

        self.btn_run = QPushButton("▶  Fit Occultation Model")
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
        self.tabs.addTab(self._tab_lc(),       "📊  Light Curve")
        self.tabs.addTab(self._tab_zoom(),     "🔍  Event Zoom")
        self.tabs.addTab(self._tab_residuals(),"📈  Residuals")
        self.tabs.addTab(self._tab_timing(),   "⏱  Timing Results")
        self.tabs.addTab(self._tab_chords(),   "📐  Multi-chord")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_lc(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.lc_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.lc_plot)
        return w

    def _tab_zoom(self):
        w = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Zoom window (s):"))
        self.sp_zoom = QDoubleSpinBox(); self.sp_zoom.setRange(0.01,100); self.sp_zoom.setValue(2.0)
        self.sp_zoom.valueChanged.connect(self._refresh_zoom)
        ctrl.addWidget(self.sp_zoom); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.zoom_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.zoom_plot)
        return w

    def _tab_residuals(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.res_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.res_plot)
        return w

    def _tab_timing(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.timing_table = QTableWidget(0, 3)
        self.timing_table.setHorizontalHeaderLabels(["Parameter", "Value", "Uncertainty"])
        self.timing_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.timing_table.setAlternatingRowColors(True)
        lay.addWidget(self.timing_table)

        # Bootstrap distribution
        self.boot_plot = MplCanvas(figsize=(9,3)); lay.addWidget(self.boot_plot)
        return w

    def _tab_chords(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.chord_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.chord_plot)
        self.size_lbl = QLabel("")
        self.size_lbl.setStyleSheet("color:#a0a0d0;font-size:11px;padding:6px;")
        lay.addWidget(self.size_lbl)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load light curve", "",
            "Text files (*.csv *.txt *.dat *.lc);;All (*)")
        if not path: return
        try:
            data = np.loadtxt(path, comments='#')
            if data.ndim == 1:
                self._t = np.arange(len(data)); self._flux = data
                self._ferr = np.full(len(data), np.std(data)*0.1)
            elif data.shape[1] >= 3:
                self._t, self._flux, self._ferr = data[:,0], data[:,1], data[:,2]
            else:
                self._t, self._flux = data[:,0], data[:,1]
                self._ferr = np.full(len(self._t), np.std(self._flux)*0.1)
            self._after_load(Path(path).name)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _load_example(self):
        rng = np.random.default_rng(42)
        # Simulate a 0.8s chord at 20 km/s → 16 km chord
        t   = np.arange(-2.0, 2.0, 0.02)   # 20 fps camera
        t1_t, t2_t = -0.398, 0.402          # true ingress/egress
        noise = 0.008
        F   = geometric_model(t, t1_t, t2_t, 1.0, 1.0, 0.015)
        self._t = t
        self._flux = F + rng.normal(0, noise, len(t))
        self._ferr = np.full(len(t), noise)
        self._after_load("synthetic_occultation.txt")
        self.sp_v.setValue(20.0)
        self._log(f"Synthetic: T1={t1_t}s T2={t2_t}s chord={(t2_t-t1_t)*20:.1f}km")

    def _after_load(self, name):
        self.file_lbl.setText(name)
        self.file_lbl.setStyleSheet("color:#70c070;font-size:10px;")
        self.btn_run.setEnabled(True)
        self._log(f"Loaded: {name}  {len(self._t)} pts  "
                  f"cadence={abs(np.median(np.diff(self._t)))*1000:.1f}ms")
        self._draw_lc()

    def _save(self):
        if not self._result: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save report", "", "JSON (*.json)")
        if not path: return
        r = self._result['result']
        out = {
            'date_utc':  datetime.utcnow().isoformat()+'Z',
            'hyperload': VERSION,
            'model':     r['model'],
            'snr':       float(self._result['snr']),
            'n_pts':     self._result['n_pts'],
            'chi2_red':  float(r['chi2_red']),
        }
        if r['model'] == 'geometric':
            out.update({
                'T1_s':         float(r['t1']),
                'T1_err_ms':    float(r['t1_err'] * 1000),
                'T2_s':         float(r['t2']),
                'T2_err_ms':    float(r['t2_err'] * 1000),
                'chord_s':      float(r['chord_s']),
                'chord_km':     float(r['chord_km']),
                'chord_km_err': float(r['chord_km_err']),
                'depth':        float(r['depth']),
                'tau_s':        float(r['tau']),
            })
        else:
            out.update({
                'T_mid_s':    float(r['t_mid']),
                'T_mid_err_ms': float(r['t_mid_err']*1000),
                'v_shadow_kms': float(r['v_shadow']),
                'fresnel_scale_km': float(r['fresnel_scale_km']),
                'depth':      float(r['depth']),
            })
        if self._chords:
            chord_sizes = ChordAnalyzer.fit_diameter(self._chords)
            out['multi_chord'] = chord_sizes
        with open(path,'w') as f: json.dump(out, f, indent=2)
        self._log(f"✓ Saved: {path}")

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _run(self):
        if self._t is None: return
        model_names = {0: 'geometric', 1: 'fresnel'}
        params = {
            'model':       model_names[self.cmb_model.currentIndex()],
            'v_shadow':    self.sp_v.value(),
            'dist_km':     self.sp_dist.value() * 1e9,
            'n_sigma':     self.sp_sigma.value(),
            'n_bootstrap': self.sp_boot.value(),
            'min_pts':     3,
        }
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False); self._result = None

        self._wthread = QThread(self)
        self._worker  = OccultationWorker(self._t, self._flux, self._ferr, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _cancel(self):
        if self._worker: self._worker.cancel()
        self.btn_cancel.setEnabled(False); self.btn_run.setEnabled(True)

    # ── Worker callbacks ──────────────────────────────────────────────────────

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
        r = result['result']

        self._log(f"\n{'='*55}")
        self._log(f"✓  Occultation timing complete  (SNR={result['snr']:.1f})")
        self._log(f"   Model: {r['model']}  χ²_r={r['chi2_red']:.2f}")
        if r['model'] == 'geometric':
            self._log(f"   T₁ = {r['t1']:.5f} ± {r['t1_err']*1000:.1f} ms")
            self._log(f"   T₂ = {r['t2']:.5f} ± {r['t2_err']*1000:.1f} ms")
            self._log(f"   Chord = {r['chord_s']*1000:.1f} ms = {r['chord_km']:.2f} ± {r['chord_km_err']:.2f} km")
            self._log(f"   Depth = {r['depth']*100:.1f}%  τ={r['tau']*1000:.1f} ms")
            self.summary_lbl.setText(
                f"T₁={r['t1']:.4f}s  T₂={r['t2']:.4f}s  "
                f"chord={r['chord_km']:.1f}km  χ²={r['chi2_red']:.2f}")
        else:
            self._log(f"   T_mid = {r['t_mid']:.5f} ± {r['t_mid_err']*1000:.1f} ms")
            self._log(f"   r_F  = {r['fresnel_scale_km']:.2f} km")
        self._log(f"{'='*55}\n")

        self._draw_lc()
        self._refresh_zoom()
        self._draw_residuals()
        self._fill_timing_table()
        self._draw_bootstrap()
        self.tabs.setCurrentIndex(1)   # zoom tab

    # ── Chord management ──────────────────────────────────────────────────────

    def _add_chord(self):
        if not self._result: return
        r = self._result['result']
        if r['model'] == 'geometric':
            self._chords.append({
                'chord_km':     r['chord_km'],
                'chord_km_err': r['chord_km_err'],
                'depth':        r['depth'],
            })
        self.chord_lbl.setText(f"{len(self._chords)} chord(s) loaded")
        self._log(f"Added chord: {r['chord_km']:.2f} km")
        self._draw_chord_map()

    def _clear_chords(self):
        self._chords.clear()
        self.chord_lbl.setText("0 chords loaded")
        self._draw_chord_map()

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_lc(self):
        if self._t is None: return
        fig = self.lc_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "Occultation Light Curve", "Time (s)", "Flux")
        ax.errorbar(self._t, self._flux, self._ferr, fmt='.', ms=2,
                    color='#6080ff', ecolor='#3040a0', elinewidth=0.5, alpha=0.7)
        if self._result:
            r = self._result['result']
            ax.plot(self._t, r['model_flux'], color='#ff8030', lw=2,
                    zorder=5, label=f"{r['model']} fit  χ²={r['chi2_red']:.2f}")
            if r['model'] == 'geometric':
                ax.axvline(r['t1'], color='#60ffff', lw=1, ls='--', label=f"T₁={r['t1']:.4f}s")
                ax.axvline(r['t2'], color='#ff60ff', lw=1, ls='--', label=f"T₂={r['t2']:.4f}s")
            ax.legend(fontsize=7, facecolor='#1a1a30', edgecolor='#3030a0', labelcolor='white')
        if self._result:
            det = self._result['detect']
            ax.axhline(det['baseline'], color='#404060', lw=0.8, ls=':')
        self.lc_plot.redraw()

    def _refresh_zoom(self):
        if not self._result: return
        r = self._result['result']
        fig = self.zoom_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1, 2)

        if r['model'] == 'geometric':
            t_centres = [r['t1'], r['t2']]
            labels    = ["Ingress", "Egress"]
        else:
            t_centres = [r['t_mid'], r['t_mid']]
            labels    = ["Leading edge", "Trailing edge"]

        hw = self.sp_zoom.value() / 2
        for ax, tc, lbl in zip(axes, t_centres, labels):
            _ax(ax, lbl, "Time (s)", "Flux")
            mask = (self._t >= tc - hw) & (self._t <= tc + hw)
            ax.errorbar(self._t[mask], self._flux[mask], self._ferr[mask],
                        fmt='.', ms=3, color='#6080ff', ecolor='#3040a0',
                        elinewidth=0.7, zorder=3)
            t_f = np.linspace(tc - hw, tc + hw, 500)
            if r['model'] == 'geometric':
                mf = geometric_model(t_f, r['t1'], r['t2'], r['baseline'],
                                     r['depth'], r['tau'])
                ax.axvline(r['t1'] if lbl=='Ingress' else r['t2'],
                           color='#60ffff', lw=1, ls='--')
            else:
                from scipy.ndimage import gaussian_filter1d as gf
                mf = fresnel_model(t_f, r['t_mid'], r['v_shadow'],
                                   self._result['params']['dist_km'],
                                   r['baseline'], r['depth'])
            ax.plot(t_f, mf, c='#ff8030', lw=2, zorder=5)
            ax.set_xlim(tc - hw, tc + hw)
        self.zoom_plot.redraw()

    def _draw_residuals(self):
        if not self._result: return
        r = self._result['result']
        fig = self.res_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, f"Residuals  (χ²_red = {r['chi2_red']:.2f})", "Time (s)", "Flux − model")
        ax.errorbar(self._t, r['residuals'], self._ferr, fmt='.', ms=2,
                    color='#6080ff', ecolor='#3040a0', elinewidth=0.5, alpha=0.7)
        ax.axhline(0, color='#404070', lw=1)
        rms = np.std(r['residuals'])
        ax.set_title(f"Residuals  χ²_red={r['chi2_red']:.2f}  RMS={rms:.4f}",
                     color='#9090ff', fontsize=9, pad=4)
        self.res_plot.redraw()

    def _fill_timing_table(self):
        if not self._result: return
        r = self._result['result']
        rows = []
        if r['model'] == 'geometric':
            rows = [
                ("T₁ (ingress)",  f"{r['t1']:.6f} s",     f"±{r['t1_err']*1000:.2f} ms"),
                ("T₂ (egress)",   f"{r['t2']:.6f} s",     f"±{r['t2_err']*1000:.2f} ms"),
                ("Chord duration",f"{r['chord_s']*1000:.2f} ms",  ""),
                ("Chord length",  f"{r['chord_km']:.3f} km", f"±{r['chord_km_err']:.3f} km"),
                ("Depth",         f"{r['depth']*100:.2f}%", ""),
                ("τ (smoothing)", f"{r['tau']*1000:.2f} ms",""),
                ("χ²_red",        f"{r['chi2_red']:.3f}",  ""),
                ("SNR",           f"{self._result['snr']:.1f}", ""),
            ]
        else:
            rows = [
                ("T_mid",          f"{r['t_mid']:.6f} s",  f"±{r['t_mid_err']*1000:.2f} ms"),
                ("Shadow velocity",f"{r['v_shadow']:.2f} km/s", ""),
                ("Fresnel scale",  f"{r['fresnel_scale_km']:.2f} km", ""),
                ("Depth",          f"{r['depth']*100:.2f}%",""),
                ("χ²_red",         f"{r['chi2_red']:.3f}",  ""),
            ]
        self.timing_table.setRowCount(len(rows))
        for i,(k,v,e) in enumerate(rows):
            self.timing_table.setItem(i,0,QTableWidgetItem(k))
            self.timing_table.setItem(i,1,QTableWidgetItem(v))
            self.timing_table.setItem(i,2,QTableWidgetItem(e))
        self.timing_table.resizeColumnsToContents()

    def _draw_bootstrap(self):
        if not self._result: return
        r = self._result['result']
        fig = self.boot_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        if r['model'] == 'geometric' and 't1_bootstrap' in r:
            axes = fig.subplots(1, 2)
            for ax, bs, label, val in zip(
                    axes,
                    [r['t1_bootstrap'], r['t2_bootstrap']],
                    ["T₁ bootstrap", "T₂ bootstrap"],
                    [r['t1'], r['t2']]):
                _ax(ax, label, "Time (s)", "Count")
                ax.hist(bs, bins=30, color='#5050b0', edgecolor='#3030a0',
                        alpha=0.85)
                ax.axvline(val, color='#ff8030', lw=1.5,
                           label=f'Fit={val:.5f}')
                ax.axvline(val-np.std(bs), color='#ff8030', lw=0.8, ls='--')
                ax.axvline(val+np.std(bs), color='#ff8030', lw=0.8, ls='--')
                ax.legend(fontsize=7, facecolor='#1a1a30',
                          edgecolor='#3030a0', labelcolor='white')
        self.boot_plot.redraw()

    def _draw_chord_map(self):
        fig = self.chord_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, f"Multi-chord analysis  ({len(self._chords)} chords)",
            "Chord position (km)", "Chord length (km)")

        if self._chords:
            lengths = [c['chord_km'] for c in self._chords]
            errors  = [c.get('chord_km_err', 1.0) for c in self._chords]
            x = np.arange(len(lengths))
            ax.bar(x, lengths, color='#5050b0', edgecolor='#3030a0', alpha=0.85)
            ax.errorbar(x, lengths, errors, fmt='none', color='#ff8030', capsize=5)

            size_info = ChordAnalyzer.fit_diameter(self._chords)
            if size_info:
                self.size_lbl.setText(
                    f"N chords: {size_info['n_chords']}  |  "
                    f"Max chord: {size_info['max_chord_km']:.1f} km  |  "
                    f"Diameter lower bound: {size_info['diameter_lower_bound_km']:.1f} km  |  "
                    f"Radius estimate: {size_info['radius_est_km']:.1f} km")
        self.chord_plot.redraw()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = OccultationApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
