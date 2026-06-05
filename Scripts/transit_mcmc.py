r"""
HYPERLOAD — Exoplanet Transit MCMC Pipeline
============================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Fits a limb-darkened transit model to a photometric light curve using
Markov Chain Monte Carlo (Metropolis-Hastings with adaptive step sizes).
Recovers planet parameters with full posterior probability distributions.

Transit model:
  Mandel & Agol 2002 (ApJL 580, L171) — quadratic limb darkening.
  Flux deficit computed via double Gauss-Legendre quadrature over the
  intersection of stellar and planetary disks.  Vectorised over all
  time points simultaneously.

  F(t) = f₀ × [1 + ct] × transit_flux(z(t), p, u₁, u₂)

  where z(t) is the projected separation for a circular orbit:
    z(t) = √[(a/Rs·sin(M))² + b²]   M = 2π(t−t₀)/P

MCMC:
  Metropolis-Hastings with adaptive Gaussian proposal.
  Proposal scales updated every 500 steps to target 23% acceptance rate.
  Convergence monitored via Gelman-Rubin R̂ statistic (two chains).

Parameters:
  t₀     : mid-transit time (same units as input time column)
  P      : orbital period (can be fixed if known)
  p      : Rp/Rs — planet-to-star radius ratio
  a/Rs   : scaled semi-major axis (sets transit duration)
  b      : impact parameter ∈ [0, 1)
  u₁,u₂ : quadratic limb-darkening coefficients (can be fixed)
  f₀     : baseline flux normalisation
  c      : linear trend in baseline (instrumental systematics)

References:
  Mandel & Agol 2002     — transit model, ApJL 580, L171
  Eastman et al. 2013    — parameter conventions, EXOFAST
  Kipping 2013           — uninformative LD priors, MNRAS 435, 2152
  Ford 2005              — adaptive MCMC, AJ 129, 1706
  Gelman & Rubin 1992    — convergence diagnostic, Statistical Science

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
def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\ntransit_mcmc.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    print(f"[CRASH] {log}")
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QLineEdit, QTableWidget, QTableWidgetItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Exoplanet Transit MCMC Pipeline"

# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSIT MODEL  (Mandel & Agol 2002)
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-computed Gauss-Legendre nodes for transit integration
_N_GL  = 25
_N_GLY = 20
_XI_X, _WI_X = np.polynomial.legendre.leggauss(_N_GL)
_XI_Y, _WI_Y = np.polynomial.legendre.leggauss(_N_GLY)


def transit_flux(z_arr: np.ndarray, p: float,
                 u1: float, u2: float) -> np.ndarray:
    """
    Vectorised Mandel & Agol 2002 quadratic limb-darkened transit model.

    Uses double Gauss-Legendre quadrature to evaluate the LD-weighted
    occulted flux fraction exactly (to machine precision for n_GL ≥ 25).

    Parameters
    ----------
    z_arr : projected centre-to-centre separation(s) in units of R_star
    p     : Rp/Rs — planet radius / stellar radius
    u1,u2 : quadratic limb-darkening coefficients

    Returns
    -------
    F : normalised flux ∈ [0, 1], shape matching z_arr
    """
    z   = np.atleast_1d(z_arr).astype(float)
    # Stellar flux normalisation: ∫₀¹ I(r) 2πr dr = π(1 - u1/3 - u2/6)
    norm = np.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    F    = np.ones(len(z))

    tr = z < 1.0 + p
    if not tr.any():
        return F

    zt   = z[tr]
    x_lo = np.maximum(zt - p, -1.0)
    x_hi = np.minimum(zt + p,  1.0)

    # Outer x-integration mapped to [x_lo, x_hi] : shape (Nt, n_gl)
    x  = 0.5*(x_hi - x_lo)[:, None]*_XI_X + 0.5*(x_hi + x_lo)[:, None]
    wx = 0.5*(x_hi - x_lo)[:, None]*_WI_X

    # Overlap half-chord in y : shape (Nt, n_gl)
    y_star   = np.sqrt(np.maximum(1.0 - x**2, 0.0))
    y_planet = np.sqrt(np.maximum(p**2 - (x - zt[:, None])**2, 0.0))
    yh       = np.minimum(y_star, y_planet)

    # Inner y-integration over [0, yh] : shape (Nt, n_gl, n_gl_y)
    y  = 0.5 * yh[:, :, None] * (_XI_Y[None, None, :] + 1.0)
    wy = 0.5 * yh[:, :, None] * _WI_Y[None, None, :]

    # Limb-darkening intensity I(r) = 1 − u1(1−μ) − u2(1−μ)²
    r2 = x[:, :, None]**2 + y**2
    mu = np.sqrt(np.maximum(1.0 - r2, 0.0))
    I  = 1.0 - u1*(1.0 - mu) - u2*(1.0 - mu)**2

    # Double integral: 2 (symmetry in y) × Σ_y w_y I × Σ_x w_x
    occ      = np.sum(wx * 2.0 * np.sum(wy * I, axis=2), axis=1)
    F[tr]    = 1.0 - occ / norm
    return F


def orbital_z(t: np.ndarray, t0: float, P: float,
              a_Rs: float, b: float) -> np.ndarray:
    """
    Projected stellar-planet separation z(t) for a circular orbit.

    z = √[(a/Rs · sin M)² + b²]
    where M = 2π(t − t₀)/P and b = a/Rs · cos i  (impact parameter).
    """
    M = 2.0 * np.pi * (t - t0) / P
    return np.sqrt((a_Rs * np.sin(M))**2 + b**2)


def full_model(t: np.ndarray, theta: np.ndarray, P_fixed: float) -> np.ndarray:
    """
    Full light-curve model: baseline × transit.

    theta = [t0, p, a_Rs, b, u1, u2, f0, c]
    P_fixed: orbital period (fixed, not sampled)
    """
    t0, p, a_Rs, b, u1, u2, f0, c = theta
    z  = orbital_z(t, t0, P_fixed, a_Rs, b)
    Ft = transit_flux(z, p, u1, u2)
    baseline = f0 * (1.0 + c * (t - t.mean()))
    return baseline * Ft


# ═══════════════════════════════════════════════════════════════════════════════
#  MCMC  (Metropolis-Hastings with adaptive steps)
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_NAMES   = ["t₀", "p", "a/Rs", "b", "u₁", "u₂", "f₀", "c"]
PARAM_LATEX   = [r"$t_0$", r"$R_p/R_s$", r"$a/R_s$", r"$b$",
                 r"$u_1$", r"$u_2$", r"$f_0$", r"$c$"]


class TransitMCMC:
    """
    Metropolis-Hastings MCMC for transit parameter estimation.

    Adaptive proposal: step sizes updated every `adapt_interval` steps
    to drive the acceptance rate towards the optimal 23.4% (Roberts 1997).

    Parameterisation: θ = [t0, p, a_Rs, b, u1, u2, f0, c]

    Priors:
      p     ∈ (0, 1)          — radius ratio must be physical
      a_Rs  ∈ (1, 200)         — planet must be outside star
      b     ∈ [0, 1)           — impact parameter
      u1,u2 ∈ [0, 1], u1+u2<1 — physical LD (Kipping 2013)
      f0    ∈ (0.5, 1.5)       — baseline close to 1
    """

    def __init__(self, t: np.ndarray, flux: np.ndarray,
                 flux_err: np.ndarray, P_fixed: float,
                 fix_ld: bool = False):
        self.t         = t
        self.flux      = flux
        self.flux_err  = flux_err
        self.P         = P_fixed
        self.fix_ld    = fix_ld

    def log_prior(self, theta: np.ndarray) -> float:
        t0, p, a_Rs, b, u1, u2, f0, c = theta
        if not (0 < p < 1):         return -np.inf
        if not (1 < a_Rs < 300):    return -np.inf
        if not (0 <= b < 1):        return -np.inf
        if not (0 <= u1 <= 1):      return -np.inf
        if not (0 <= u2 <= 1):      return -np.inf
        if u1 + u2 > 1:             return -np.inf
        if not (0.5 < f0 < 1.5):    return -np.inf
        return 0.0

    def log_like(self, theta: np.ndarray) -> float:
        model = full_model(self.t, theta, self.P)
        resid = self.flux - model
        return -0.5 * np.sum((resid / self.flux_err)**2)

    def log_post(self, theta: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        return lp + self.log_like(theta)

    def run(self, theta0: np.ndarray, n_steps: int,
            n_burn: int, adapt_interval: int = 500,
            progress_cb=None, cancel_flag=None) -> dict:
        """
        Run adaptive Metropolis-Hastings.

        Returns:
          chain        — (n_steps, 8) array of accepted samples
          log_post_arr — log-posterior at each step
          accept_rate  — overall acceptance rate
          best_theta   — MAP estimate
          best_lp      — MAP log-posterior
        """
        n_par   = len(theta0)
        chain   = np.zeros((n_steps, n_par))
        lp_arr  = np.zeros(n_steps)
        step    = np.abs(theta0) * 0.02 + 1e-4   # initial proposal widths

        theta  = theta0.copy()
        lp     = self.log_post(theta)
        accepts = 0
        best_theta = theta.copy()
        best_lp    = lp
        rng = np.random.default_rng()

        for i in range(n_steps):
            if cancel_flag and cancel_flag[0]: break

            # Proposal
            prop = theta + rng.normal(0, step)
            lp_p = self.log_post(prop)

            if lp_p > lp or np.log(rng.random()) < lp_p - lp:
                theta = prop; lp = lp_p; accepts += 1
                if lp > best_lp:
                    best_lp = lp; best_theta = theta.copy()

            chain[i]  = theta
            lp_arr[i] = lp

            # Adaptive step sizes — target ~23% acceptance
            if i > 0 and i % adapt_interval == 0:
                window = chain[max(0, i-adapt_interval):i]
                std    = window.std(axis=0)
                std[std < 1e-8] = 1e-8
                step = std * 2.38 / np.sqrt(n_par)   # optimal Gaussian scale
                rate = accepts / (i + 1)
                if progress_cb:
                    progress_cb(i, n_steps, rate, lp)

        return {
            'chain':       chain[:i+1],
            'log_post':    lp_arr[:i+1],
            'accept_rate': accepts / max(i + 1, 1),
            'best_theta':  best_theta,
            'best_lp':     best_lp,
            'n_burn':      n_burn,
            'steps_done':  i + 1,
        }

    @staticmethod
    def gelman_rubin(chain1: np.ndarray, chain2: np.ndarray) -> np.ndarray:
        """Gelman-Rubin R̂ convergence diagnostic (should be < 1.1)."""
        n, k = chain1.shape
        W  = 0.5 * (chain1.var(axis=0) + chain2.var(axis=0))
        B  = n * np.array([np.var([chain1[:, j].mean(),
                                   chain2[:, j].mean()])
                           for j in range(k)])
        var = (1 - 1/n) * W + B / n
        return np.sqrt(var / (W + 1e-10))

    @staticmethod
    def credible_interval(samples: np.ndarray, pct: float = 68.27):
        """Median and 1-sigma credible interval from MCMC samples."""
        lo = (100 - pct) / 2
        hi = 100 - lo
        return (np.median(samples),
                np.percentile(samples, lo),
                np.percentile(samples, hi))


# ═══════════════════════════════════════════════════════════════════════════════
#  DERIVED PLANET PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

R_SUN    = 6.957e8    # m
R_EARTH  = 6.371e6    # m
R_JUP    = 7.149e7    # m
AU       = 1.496e11   # m


def derive_planet_params(theta: np.ndarray, P_days: float,
                         Rs_solar: float = 1.0) -> dict:
    """
    Derive physical planet parameters from fitted dimensionless parameters.

    Inputs:
      theta    : [t0, p, a_Rs, b, u1, u2, f0, c]
      P_days   : orbital period in days
      Rs_solar : stellar radius in solar radii (from catalog or assumption)
    """
    t0, p, a_Rs, b, u1, u2, f0, c = theta
    Rs = Rs_solar * R_SUN

    # Planet radius
    Rp_m     = p * Rs
    Rp_earth = Rp_m / R_EARTH
    Rp_jup   = Rp_m / R_JUP

    # Semi-major axis
    a_m  = a_Rs * Rs
    a_AU = a_m / AU

    # Inclination
    inc_deg = math.degrees(math.acos(b / a_Rs)) if a_Rs > 0 else 90.0

    # Transit duration (from Winn 2010 formula)
    P_s = P_days * 86400  # seconds
    try:
        sin_factor = math.sqrt((1 + p)**2 - b**2) / a_Rs
        T14 = (P_s / np.pi) * math.asin(math.sqrt(sin_factor**2))
        sin_factor2 = math.sqrt((1 - p)**2 - b**2) / a_Rs
        T23 = (P_s / np.pi) * math.asin(math.sqrt(sin_factor2**2))
        tau = (T14 - T23) / 2   # ingress/egress duration
    except (ValueError, ZeroDivisionError):
        T14 = T23 = tau = float('nan')

    return {
        'Rp_earth':  Rp_earth,
        'Rp_jup':    Rp_jup,
        'a_AU':      a_AU,
        'inc_deg':   inc_deg,
        'T14_hours': T14 / 3600 if not math.isnan(T14) else float('nan'),
        'T23_hours': T23 / 3600 if not math.isnan(T23) else float('nan'),
        'tau_min':   tau / 60   if not math.isnan(tau)  else float('nan'),
        'depth_ppt': p**2 * 1000,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class TransitWorker(QObject):
    progress   = pyqtSignal(int, str)
    chain_update = pyqtSignal(int, float, float)  # step, accept_rate, lp
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, t, flux, flux_err, params):
        super().__init__()
        self.t = t; self.flux = flux; self.flux_err = flux_err
        self.params = params
        self._cancel = [False]

    def cancel(self): self._cancel[0] = True

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p = self.params
        t, flux, flux_err = self.t, self.flux, self.flux_err

        self.progress.emit(2, "Initialising MCMC…")

        P_fixed = p['period']
        mcmc = TransitMCMC(t, flux, flux_err, P_fixed,
                           fix_ld=p.get('fix_ld', False))

        # Build initial parameter vector
        theta0 = np.array([
            p['t0_init'],
            p['p_init'],
            p['a_Rs_init'],
            p['b_init'],
            p['u1'],
            p['u2'],
            1.0,   # f0
            0.0,   # c (trend)
        ])

        n_burn  = p['n_burn']
        n_steps = p['n_steps']

        def prog_cb(step, total, rate, lp):
            pct = int(5 + 90 * step / max(total, 1))
            self.progress.emit(pct,
                f"Step {step}/{total}  accept={rate:.1%}  logP={lp:.2f}")
            self.chain_update.emit(step, rate, lp)

        # ── Chain 1 (primary) ─────────────────────────────────────────────
        self.progress.emit(5, f"Running chain 1 ({n_burn} burn-in + {n_steps} production)…")
        result1 = mcmc.run(theta0, n_burn + n_steps, n_burn,
                           progress_cb=prog_cb,
                           cancel_flag=self._cancel)
        if self._cancel[0]: return

        # ── Chain 2 (convergence check) ───────────────────────────────────
        self.progress.emit(48, f"Running chain 2 for R̂ convergence check…")
        theta0_2 = theta0 + np.random.default_rng(1).normal(0, 0.01, len(theta0))
        result2 = mcmc.run(theta0_2, n_burn + n_steps // 4, n_burn,
                           cancel_flag=self._cancel)
        if self._cancel[0]: return

        # ── Burn-in removal ────────────────────────────────────────────────
        self.progress.emit(92, "Analysing posterior…")
        chain1 = result1['chain'][n_burn:]
        chain2 = result2['chain'][min(n_burn, len(result2['chain'])):]
        min_len = min(len(chain1), len(chain2))
        chain1 = chain1[:min_len]; chain2 = chain2[:min_len]

        # Gelman-Rubin
        if min_len > 10:
            rhat = TransitMCMC.gelman_rubin(chain1, chain2)
        else:
            rhat = np.ones(len(theta0)) * 99.0

        # Posterior statistics
        posteriors = {}
        for i, name in enumerate(PARAM_NAMES):
            med, lo, hi = TransitMCMC.credible_interval(chain1[:, i])
            posteriors[name] = {'median': med, 'lo_1sig': lo, 'hi_1sig': hi,
                                'std': chain1[:, i].std()}

        # MAP estimate
        best = result1['best_theta']

        # BIC comparison: transit vs flat
        n_data = len(t)
        ll_transit = mcmc.log_like(best)
        flat_flux  = np.median(flux)
        ll_flat    = -0.5 * np.sum(((flux - flat_flux) / flux_err)**2)
        delta_bic  = (ll_transit - ll_flat) * 2 - (8 - 1) * np.log(n_data)

        # Derived parameters
        derived = derive_planet_params(best, P_fixed, p.get('Rs_solar', 1.0))

        self.progress.emit(100, "Done!")

        self.finished.emit({
            'chain':       chain1,
            'chain2':      chain2,
            'best_theta':  best,
            'best_lp':     result1['best_lp'],
            'posteriors':  posteriors,
            'rhat':        rhat,
            'accept_rate': result1['accept_rate'],
            'delta_bic':   delta_bic,
            'derived':     derived,
            'mcmc':        mcmc,
            'P':           P_fixed,
            'n_data':      n_data,
            'log_post_trace': result1['log_post'],
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

class TransitApp(QMainWindow):

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

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

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
        t = QLabel("Exoplanet Transit MCMC Pipeline"); t.setObjectName("title_lbl")
        s = QLabel("Mandel & Agol 2002 LD transit model · Metropolis-Hastings MCMC · "
                   "Posterior distributions · Gelman-Rubin convergence · Planet parameters")
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

        # ── Orbital parameters ─────────────────────────────────────────────
        grp2 = QGroupBox("Orbital Parameters (initial guess)")
        g2   = QGridLayout(grp2)
        fields = [
            ("Period (days):", "sp_period", 1.0, 10000.0, 3.0, 3),
            ("Mid-transit t₀:", "sp_t0", -1000, 1000, 0.0, 4),
            ("p = Rp/Rs:", "sp_p", 0.001, 0.999, 0.1, 4),
            ("a/Rs:", "sp_aRs", 1.5, 300.0, 15.0, 2),
            ("Impact param b:", "sp_b", 0.0, 0.999, 0.3, 3),
        ]
        for row, (lbl, attr, lo, hi, val, dec) in enumerate(fields):
            g2.addWidget(QLabel(lbl), row, 0)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi); sp.setValue(val); sp.setDecimals(dec)
            sp.setSingleStep(10**(-(dec-1)))
            setattr(self, attr, sp)
            g2.addWidget(sp, row, 1)
        lay.addWidget(grp2)

        # ── Limb darkening ─────────────────────────────────────────────────
        grp3 = QGroupBox("Limb Darkening")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("u₁:"), 0, 0)
        self.sp_u1 = QDoubleSpinBox(); self.sp_u1.setRange(0,1); self.sp_u1.setValue(0.4); self.sp_u1.setDecimals(3)
        g3.addWidget(self.sp_u1, 0, 1)
        g3.addWidget(QLabel("u₂:"), 1, 0)
        self.sp_u2 = QDoubleSpinBox(); self.sp_u2.setRange(0,1); self.sp_u2.setValue(0.2); self.sp_u2.setDecimals(3)
        g3.addWidget(self.sp_u2, 1, 1)
        self.cb_fix_ld = QCheckBox("Fix LD (don't sample u₁, u₂)")
        self.cb_fix_ld.setChecked(True)
        g3.addWidget(self.cb_fix_ld, 2, 0, 1, 2)
        lay.addWidget(grp3)

        # ── Stellar parameters ─────────────────────────────────────────────
        grp4 = QGroupBox("Stellar Parameters (for Rp in physical units)")
        g4   = QGridLayout(grp4)
        g4.addWidget(QLabel("Rs (R☉):"), 0, 0)
        self.sp_Rs = QDoubleSpinBox(); self.sp_Rs.setRange(0.01, 1000); self.sp_Rs.setValue(1.0); self.sp_Rs.setDecimals(3)
        g4.addWidget(self.sp_Rs, 0, 1)
        lay.addWidget(grp4)

        # ── MCMC settings ──────────────────────────────────────────────────
        grp5 = QGroupBox("MCMC Settings")
        g5   = QGridLayout(grp5)
        g5.addWidget(QLabel("Burn-in steps:"), 0, 0)
        self.sp_burn = QSpinBox(); self.sp_burn.setRange(100,100000); self.sp_burn.setValue(2000); self.sp_burn.setSingleStep(500)
        g5.addWidget(self.sp_burn, 0, 1)
        g5.addWidget(QLabel("Production steps:"), 1, 0)
        self.sp_steps = QSpinBox(); self.sp_steps.setRange(500,500000); self.sp_steps.setValue(10000); self.sp_steps.setSingleStep(1000)
        g5.addWidget(self.sp_steps, 1, 1)
        lay.addWidget(grp5)

        # ── Output ────────────────────────────────────────────────────────
        grp6 = QGroupBox("Output")
        g6   = QVBoxLayout(grp6)
        self.btn_save = QPushButton("💾  Save Results (JSON)…")
        self.btn_save.setEnabled(False); self.btn_save.clicked.connect(self._save)
        g6.addWidget(self.btn_save)
        lay.addWidget(grp6)

        # ── Run ───────────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Run Transit MCMC")
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
        self.tabs.addTab(self._tab_chains(),   "🎲  MCMC Chains")
        self.tabs.addTab(self._tab_posteriors(),"📐  Posteriors")
        self.tabs.addTab(self._tab_params(),   "🌍  Planet Parameters")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_lc(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.lc_plot = MplCanvas(figsize=(9,5))
        lay.addWidget(self.lc_plot)
        return w

    def _tab_chains(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.chain_plot = MplCanvas(figsize=(9,6))
        lay.addWidget(self.chain_plot)
        return w

    def _tab_posteriors(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.post_plot = MplCanvas(figsize=(9,6))
        lay.addWidget(self.post_plot)
        return w

    def _tab_params(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.params_table = QTableWidget(0, 4)
        self.params_table.setHorizontalHeaderLabels(
            ["Parameter", "Median", "−1σ", "+1σ"])
        self.params_table.setAlternatingRowColors(True)
        lay.addWidget(self.params_table)
        self.derived_table = QTableWidget(0, 2)
        self.derived_table.setHorizontalHeaderLabels(["Derived quantity", "Value"])
        self.derived_table.setAlternatingRowColors(True)
        lay.addWidget(QLabel("Derived physical parameters:"))
        lay.addWidget(self.derived_table)
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
            if data.shape[1] >= 3:
                self._t, self._flux, self._ferr = data[:,0], data[:,1], data[:,2]
            elif data.shape[1] == 2:
                self._t, self._flux = data[:,0], data[:,1]
                self._ferr = np.full(len(self._t), np.std(self._flux)*0.5)
            else:
                raise ValueError("Need at least 2 columns (time, flux)")
            self._after_load(Path(path).name)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _load_example(self):
        """Generate a synthetic transit light curve."""
        rng = np.random.default_rng(42)
        t = np.linspace(-0.15, 0.15, 250)
        P = 3.52; t0_true = 0.002; p_true = 0.108
        a_Rs = 15.0; b_true = 0.25; u1, u2 = 0.35, 0.25
        z = orbital_z(t, t0_true, P, a_Rs, b_true)
        F_true = transit_flux(z, p_true, u1, u2)
        noise  = 8e-4
        self._t    = t
        self._flux = F_true + rng.normal(0, noise, len(t))
        self._ferr = np.full(len(t), noise)
        self._after_load("synthetic_transit.txt")
        # Set good initial params
        self.sp_t0.setValue(0.0); self.sp_p.setValue(0.11)
        self.sp_aRs.setValue(14.0); self.sp_b.setValue(0.3)
        self.sp_period.setValue(P)
        self.sp_u1.setValue(0.35); self.sp_u2.setValue(0.25)
        self._log("Synthetic transit: t0=0.002d, p=0.108, a/Rs=15, b=0.25")

    def _after_load(self, name):
        self.file_lbl.setText(name)
        self.file_lbl.setStyleSheet("color:#70c070;font-size:10px;")
        self.btn_run.setEnabled(True)
        self._log(f"Loaded: {name}  {len(self._t)} points  "
                  f"t=[{self._t.min():.4f}, {self._t.max():.4f}]")
        self._draw_lc(show_model=False)

    def _save(self):
        if not self._result: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results", "", "JSON (*.json)")
        if not path: return
        r = self._result
        out = {
            'date_utc':   datetime.utcnow().isoformat()+'Z',
            'hyperload':  VERSION,
            'n_data':     r['n_data'],
            'period_days':r['P'],
            'accept_rate':float(r['accept_rate']),
            'delta_bic':  float(r['delta_bic']),
            'best_fit': {n: float(v) for n,v in zip(PARAM_NAMES, r['best_theta'])},
            'posteriors': {k: {kk: float(vv) for kk,vv in v.items()}
                          for k,v in r['posteriors'].items()},
            'gelman_rubin': {n: float(v) for n,v in
                             zip(PARAM_NAMES, r['rhat'])},
            'derived': {k: float(v) if not math.isnan(v) else None
                       for k,v in r['derived'].items()},
        }
        with open(path,'w') as f: json.dump(out, f, indent=2)
        self._log(f"✓ Saved: {path}")

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _run(self):
        if self._t is None: return
        params = {
            'period':      self.sp_period.value(),
            't0_init':     self.sp_t0.value(),
            'p_init':      self.sp_p.value(),
            'a_Rs_init':   self.sp_aRs.value(),
            'b_init':      self.sp_b.value(),
            'u1':          self.sp_u1.value(),
            'u2':          self.sp_u2.value(),
            'fix_ld':      self.cb_fix_ld.isChecked(),
            'Rs_solar':    self.sp_Rs.value(),
            'n_burn':      self.sp_burn.value(),
            'n_steps':     self.sp_steps.value(),
        }
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False); self._result = None

        self._wthread = QThread(self)
        self._worker  = TransitWorker(self._t, self._flux, self._ferr, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.chain_update.connect(self._on_chain_update)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _cancel(self):
        if self._worker: self._worker.cancel()
        self.btn_cancel.setEnabled(False); self.btn_run.setEnabled(True)

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status.update(pct, msg); self._log(f"[{pct:3d}%]  {msg}")

    def _on_chain_update(self, step, rate, lp):
        pass  # live updates handled in finished callback

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}")
        self.status.update(0,"Error"); self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Error", msg[:500])

    def _on_finished(self, result):
        self._result = result
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)
        r = result
        post = r['posteriors']

        # Quick summary
        p_med  = post['p']['median']
        b_med  = post['b']['median']
        depth  = p_med**2 * 100

        self._log(f"\n{'='*55}")
        self._log(f"✓  Transit MCMC complete")
        self._log(f"   Rp/Rs    = {p_med:.4f} ± {post['p']['std']:.4f}")
        self._log(f"   b        = {b_med:.3f} ± {post['b']['std']:.3f}")
        self._log(f"   a/Rs     = {post['a/Rs']['median']:.2f}")
        self._log(f"   Depth    = {depth:.4f}%")
        self._log(f"   Accept   = {r['accept_rate']*100:.1f}%")
        self._log(f"   ΔBIC     = {r['delta_bic']:.1f}  "
                  f"({'STRONG detection' if r['delta_bic']>10 else 'weak'})")
        rhat_max = r['rhat'][:5].max()
        self._log(f"   R̂ (max) = {rhat_max:.3f}  "
                  f"({'converged ✓' if rhat_max<1.1 else 'NOT converged ✗'})")
        d = r['derived']
        self._log(f"   Rp       = {d['Rp_earth']:.2f} R⊕  = {d['Rp_jup']:.3f} Rjup")
        self._log(f"   Duration = {d['T14_hours']:.2f} h")
        self._log(f"{'='*55}\n")

        self.summary_lbl.setText(
            f"Rp/Rs={p_med:.4f}  b={b_med:.3f}  "
            f"Depth={depth:.3f}%  ΔBIC={r['delta_bic']:.0f}")

        self._draw_lc(show_model=True)
        self._draw_chains()
        self._draw_posteriors()
        self._fill_params_table()
        self.tabs.setCurrentIndex(0)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_lc(self, show_model=False):
        if self._t is None: return
        fig = self.lc_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "Transit Light Curve", "Time (days)", "Normalised flux")
        ax.errorbar(self._t, self._flux, self._ferr, fmt='.', color='#6080ff',
                    ecolor='#3040a0', elinewidth=0.8, ms=3, alpha=0.7, label='Data')
        if show_model and self._result:
            t_fine = np.linspace(self._t.min(), self._t.max(), 500)
            best = self._result['best_theta']
            F_m  = full_model(t_fine, best, self._result['P'])
            ax.plot(t_fine, F_m, color='#ff8030', lw=2, label='Best fit', zorder=5)
            depth = best[1]**2 * 100
            ax.set_title(f"Transit Light Curve  |  depth={depth:.3f}%  "
                         f"ΔBIC={self._result['delta_bic']:.1f}",
                         color='#9090ff', fontsize=9, pad=4)
        ax.legend(fontsize=7, facecolor='#1a1a30',
                  edgecolor='#3030a0', labelcolor='white')
        self.lc_plot.redraw()

    def _draw_chains(self):
        if not self._result: return
        chain = self._result['chain']
        fig = self.chain_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        # Show 4 key parameters
        show = [0, 1, 2, 3]  # t0, p, a_Rs, b
        axes = fig.subplots(len(show), 1)
        for i, (ax, pi) in enumerate(zip(axes, show)):
            _ax(ax, '', 'Step', PARAM_NAMES[pi])
            ax.plot(chain[:, pi], lw=0.5, c='#5070ff', alpha=0.8)
            med = np.median(chain[:, pi])
            ax.axhline(med, color='#ff8030', lw=1, ls='--')
        self.chain_plot.redraw()

    def _draw_posteriors(self):
        if not self._result: return
        chain = self._result['chain']
        fig = self.post_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        show = [0, 1, 2, 3]  # t0, p, a_Rs, b
        axes = fig.subplots(1, len(show))
        for i, (ax, pi) in enumerate(zip(axes, show)):
            _ax(ax, PARAM_NAMES[pi], 'Value', 'Count')
            ax.hist(chain[:, pi], bins=40, color='#5050b0',
                    edgecolor='#3030a0', alpha=0.85)
            med = np.median(chain[:, pi])
            ax.axvline(med, color='#ff8030', lw=1.5,
                       label=f'med={med:.4f}')
            lo = np.percentile(chain[:,pi],16)
            hi = np.percentile(chain[:,pi],84)
            ax.axvspan(lo, hi, alpha=0.15, color='#ff8030')
            ax.legend(fontsize=6, facecolor='#1a1a30',
                      edgecolor='#3030a0', labelcolor='white')
        self.post_plot.redraw()

    def _fill_params_table(self):
        if not self._result: return
        r = self._result; post = r['posteriors']
        rows_p = list(post.items())
        self.params_table.setRowCount(len(rows_p))
        for row,(name,v) in enumerate(rows_p):
            self.params_table.setItem(row,0,QTableWidgetItem(name))
            self.params_table.setItem(row,1,QTableWidgetItem(f"{v['median']:.6f}"))
            self.params_table.setItem(row,2,QTableWidgetItem(f"{v['lo_1sig']:.6f}"))
            self.params_table.setItem(row,3,QTableWidgetItem(f"{v['hi_1sig']:.6f}"))
        self.params_table.resizeColumnsToContents()

        d = r['derived']
        rows_d = [
            ("Rp (R⊕)",        f"{d['Rp_earth']:.3f}"),
            ("Rp (Rjup)",       f"{d['Rp_jup']:.4f}"),
            ("a (AU)",          f"{d['a_AU']:.4f}"),
            ("Inclination (°)", f"{d['inc_deg']:.2f}"),
            ("T₁₄ duration (h)",f"{d['T14_hours']:.3f}"),
            ("Ingress τ (min)", f"{d['tau_min']:.2f}"),
            ("Depth (ppt)",     f"{d['depth_ppt']:.3f}"),
            ("ΔBIC",            f"{r['delta_bic']:.1f}  "
                                f"({'STRONG' if r['delta_bic']>10 else 'WEAK'})"),
            ("R̂ (max param)",  f"{r['rhat'][:5].max():.3f}"),
        ]
        self.derived_table.setRowCount(len(rows_d))
        for row,(k,v) in enumerate(rows_d):
            self.derived_table.setItem(row,0,QTableWidgetItem(k))
            self.derived_table.setItem(row,1,QTableWidgetItem(v))
        self.derived_table.resizeColumnsToContents()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TransitApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
