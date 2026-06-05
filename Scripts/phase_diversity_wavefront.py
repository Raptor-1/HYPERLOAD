r"""
HYPERLOAD — Phase Diversity Wavefront Sensor
=============================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Measures the optical wavefront error from a focused + slightly defocused
image pair, then applies PSF deconvolution to sharpen the full image.
No AO, no special hardware — just two exposures.

Algorithm (Löfdahl & Scharmer 1994):
  1. Parameterise wavefront as sum of Zernike polynomials Z_n^m
  2. For trial coefficients, simulate PSF_focused and PSF_defocused
  3. Jointly estimate the object O that best explains BOTH images:
         O = (H_f* · F_f + H_d* · F_d) / (|H_f|² + |H_d|²)
  4. Compute residual merit function:
         L = Σ|F_f - H_f·O|² + Σ|F_d - H_d·O|²
  5. Minimise L over Zernike coefficients (Nelder-Mead / L-BFGS-B)
  6. Recovered coefficients → true PSF → Wiener deconvolution of full image

Extended object mode (Mugnier et al. 2006):
  Same as above but the ROI does not need to be a point source.
  The planetary limb or any high-contrast feature can be the reference.

Zernike modes (OSA/ANSI ordering):
  Z1  piston       Z2  tilt-x      Z3  tilt-y
  Z4  defocus      Z5  astig-45°   Z6  astig-0°
  Z7  coma-x       Z8  coma-y      Z9  trefoil-x
  Z10 trefoil-y    Z11 spherical   Z12 2nd-astig

References:
  Löfdahl & Scharmer 1994  — A&AS 107, 243
  Mugnier et al. 2006       — JOSAA 23, 3737 (extended objects)
  Gonsalves 1982            — Opt. Eng. 21, 829 (original PD concept)
  Roddier & Roddier 1993    — JOSAA 10, 2277

Version: 1.0.0
Project: HYPERLOAD
"""

import sys
import os
import math
import traceback
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ── crash logger ─────────────────────────────────────────────────────────────
def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nphase_diversity_wavefront.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    print(f"[CRASH] {log}")
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import convolve, shift as ndshift
from scipy.signal import wiener as scipy_wiener

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
    QScrollArea, QRubberBand, QSlider, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QRect, QPoint, QSize
from PyQt6.QtGui import QPixmap, QIcon, QCursor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Phase Diversity Wavefront Sensor"

# ═══════════════════════════════════════════════════════════════════════════════
#  ZERNIKE BASIS
# ═══════════════════════════════════════════════════════════════════════════════

# OSA/ANSI mode names for the first 21 modes
ZERNIKE_NAMES = [
    "Piston", "Tilt X", "Tilt Y",
    "Defocus", "Astig 45°", "Astig 0°",
    "Coma X", "Coma Y", "Trefoil X", "Trefoil Y",
    "Spherical", "2nd Astig 0°", "2nd Astig 45°",
    "Quad X", "Quad Y",
    "2nd Coma X", "2nd Coma Y", "2nd Trefoil X", "2nd Trefoil Y",
    "Pentafoil X", "Pentafoil Y",
]


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial component of Zernike polynomial Z_n^m."""
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - abs(m)) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s) /
             (math.factorial(s) *
              math.factorial((n + abs(m)) // 2 - s) *
              math.factorial((n - abs(m)) // 2 - s)))
        R += c * rho ** (n - 2 * s)
    return R


class ZernikeBasis:
    """
    Pre-computed Zernike polynomial basis over a circular pupil.

    OSA/ANSI indexing: piston=0, tilt=1,2, defocus=3, astigmatism=4,5, ...
    Basis arrays have shape (n_modes, N, N).
    Pupil mask: True inside unit circle.
    """

    def __init__(self, N: int, n_modes: int = 15):
        self.N       = N
        self.n_modes = n_modes
        self.basis, self.modes, self.pupil = self._build(N, n_modes)

    def _build(self, N: int, n_modes: int):
        y = np.linspace(-1, 1, N)
        x = np.linspace(-1, 1, N)
        XX, YY = np.meshgrid(x, y)
        rho   = np.sqrt(XX ** 2 + YY ** 2)
        theta = np.arctan2(YY, XX)
        pupil = rho <= 1.0

        # OSA/ANSI ordering
        modes = []
        n = 0
        while len(modes) < n_modes:
            for m in range(-n, n + 1, 1 if n == 0 else 2):
                if len(modes) < n_modes:
                    modes.append((n, m))
            n += 1

        basis = np.zeros((n_modes, N, N))
        for i, (nn, mm) in enumerate(modes):
            R = _zernike_radial(nn, mm, rho)
            if mm == 0:
                z = R
            elif mm > 0:
                z = R * np.cos(mm * theta)
            else:
                z = R * np.sin(-mm * theta)
            z[~pupil] = 0.0
            basis[i] = z

        return basis, modes, pupil

    def wavefront(self, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate wavefront from coefficient vector (radians)."""
        WF = np.zeros((self.N, self.N))
        for c, z in zip(coeffs, self.basis):
            WF += c * z
        return WF * self.pupil

    def strehl(self, coeffs: np.ndarray) -> float:
        """Approximate Strehl ratio: exp(-σ²_WF)."""
        WF = self.wavefront(coeffs)
        piston = WF[self.pupil].mean()
        var    = np.var(WF[self.pupil] - piston)
        return float(np.exp(-var))

    def mode_name(self, idx: int) -> str:
        if idx < len(ZERNIKE_NAMES):
            return ZERNIKE_NAMES[idx]
        n, m = self.modes[idx]
        return f"Z{idx} (n={n},m={m})"


# ═══════════════════════════════════════════════════════════════════════════════
#  PSF SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class PSFSimulator:
    """
    Simulate PSF from Zernike wavefront coefficients.

    PSF = |FT{P · exp(i·WF)}|²

    where P = circular pupil amplitude, WF = wavefront error.
    A known defocus term is added for the diversity image.
    """

    def __init__(self, zb: ZernikeBasis, psf_size: int):
        self.zb       = zb
        self.psf_size = psf_size
        self._defocus_z = self._make_defocus_map()

    def _make_defocus_map(self) -> np.ndarray:
        """Defocus Zernike: Z4 = sqrt(3)·(2ρ²-1), index 3 in OSA."""
        N = self.zb.N
        y = np.linspace(-1, 1, N); x = np.linspace(-1, 1, N)
        XX, YY = np.meshgrid(x, y)
        rho = np.sqrt(XX**2 + YY**2)
        defocus = np.sqrt(3) * (2 * rho**2 - 1) * self.zb.pupil
        return defocus

    def psf(self, coeffs: np.ndarray, defocus_waves: float = 0.0) -> np.ndarray:
        """
        Compute normalised PSF for given Zernike coefficients.
        defocus_waves: additional defocus in waves (2π radians per wave).
        """
        WF  = self.zb.wavefront(coeffs)
        WF += defocus_waves * 2 * np.pi * self._defocus_z

        N  = self.zb.N
        pup_c = self.zb.pupil.astype(float) * np.exp(1j * WF)
        amp   = np.abs(np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(pup_c)))) ** 2

        # Crop central PSF region
        c = N // 2; h = self.psf_size // 2
        psf = amp[c - h:c + h, c - h:c + h]
        s   = psf.sum()
        return psf / s if s > 0 else psf

    def otf(self, coeffs: np.ndarray, defocus_waves: float = 0.0,
            full_size: int = None) -> np.ndarray:
        """Optical Transfer Function (complex) at full image size."""
        psf = self.psf(coeffs, defocus_waves)
        sz  = full_size or psf.shape[0]
        pad = np.zeros((sz, sz))
        h   = psf.shape[0] // 2
        c   = sz // 2
        pad[c-h:c+h, c-h:c+h] = psf
        return np.fft.fft2(np.fft.ifftshift(pad))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE DIVERSITY SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseDiversitySolver:
    """
    Joint-estimation phase diversity solver (Löfdahl & Scharmer 1994).

    Given two images of the same object — one focused (D_f) and one
    with known defocus (D_d) — find the wavefront coefficients that
    minimise the joint residual:

        L(a) = Σ|F_f - H_f(a)·Ô|² + Σ|F_d - H_d(a)·Ô|²

    where Ô is the optimal object estimate given H_f and H_d:

        Ô = (H_f*·F_f + H_d*·F_d) / (|H_f|² + |H_d|² + ε)

    This is the joint MAP estimator under Gaussian noise.
    """

    def __init__(self, sim: PSFSimulator, n_modes: int = 10,
                 defocus_waves: float = 1.0, noise_reg: float = 1e-4):
        self.sim          = sim
        self.n_modes      = n_modes
        self.defocus_waves = defocus_waves
        self.noise_reg    = noise_reg
        self._history     = []

    def merit(self, coeffs: np.ndarray,
              roi_focused: np.ndarray,
              roi_defocus: np.ndarray) -> float:
        """Phase diversity merit function L(a)."""
        sz = roi_focused.shape[0]

        # OTFs at full ROI size
        Hf = self.sim.otf(coeffs, 0.0,              sz)
        Hd = self.sim.otf(coeffs, self.defocus_waves, sz)

        Ff = np.fft.fft2(roi_focused.astype(float))
        Fd = np.fft.fft2(roi_defocus.astype(float))

        eps = self.noise_reg * (np.abs(Hf)**2 + np.abs(Hd)**2).max()
        den = np.abs(Hf)**2 + np.abs(Hd)**2 + eps
        O   = (np.conj(Hf) * Ff + np.conj(Hd) * Fd) / den

        L = (np.sum(np.abs(Ff - Hf * O)**2) +
             np.sum(np.abs(Fd - Hd * O)**2))
        return float(np.real(L))

    def solve(self, roi_focused: np.ndarray, roi_defocus: np.ndarray,
              progress_cb=None) -> dict:
        """
        Optimise Zernike coefficients.

        Returns dict with:
          coeffs    — recovered Zernike coefficients (n_modes,)
          strehl    — estimated Strehl ratio
          history   — merit function values per iteration
          psf       — recovered PSF (normalised)
          n_modes   — number of modes fitted
        """
        self._history = []
        call_count    = [0]

        def callback(coeffs):
            v = self.merit(coeffs, roi_focused, roi_defocus)
            self._history.append(v)
            call_count[0] += 1
            if progress_cb and call_count[0] % 20 == 0:
                progress_cb(call_count[0], v)
            return v

        x0 = np.zeros(self.n_modes)

        # Phase 1: coarse global search with differential evolution
        # (avoids local minima in tip/tilt/defocus)
        bounds = [(-2.0, 2.0)] * self.n_modes
        # Reduce bounds for higher modes (less expected power)
        for i in range(6, self.n_modes):
            bounds[i] = (-1.0, 1.0)

        result = minimize(
            callback, x0,
            method='Nelder-Mead',
            options={
                'maxiter':  3000,
                'xatol':    1e-5,
                'fatol':    1e-8,
                'adaptive': True,
            }
        )

        coeffs = result.x

        # Remove piston and tip/tilt (not relevant to image sharpness)
        coeffs[0] = 0.0   # piston
        # Tip/tilt (modes 1,2) kept — they affect centring

        strehl = self.sim.zb.strehl(coeffs)
        psf    = self.sim.psf(coeffs, 0.0)

        return {
            'coeffs':   coeffs,
            'strehl':   strehl,
            'history':  self._history,
            'psf':      psf,
            'n_modes':  self.n_modes,
            'n_iter':   result.nit,
            'converged':result.success,
            'final_merit': float(result.fun),
        }

    def reconstruct_object(self, roi_f: np.ndarray, roi_d: np.ndarray,
                           coeffs: np.ndarray) -> np.ndarray:
        """Recover best object estimate from the two images."""
        sz = roi_f.shape[0]
        Hf = self.sim.otf(coeffs, 0.0,               sz)
        Hd = self.sim.otf(coeffs, self.defocus_waves,  sz)
        Ff = np.fft.fft2(roi_f.astype(float))
        Fd = np.fft.fft2(roi_d.astype(float))
        eps = self.noise_reg * (np.abs(Hf)**2 + np.abs(Hd)**2).max()
        den = np.abs(Hf)**2 + np.abs(Hd)**2 + eps
        O   = (np.conj(Hf)*Ff + np.conj(Hd)*Fd) / den
        return np.real(np.fft.ifft2(O))


class ImageDeconvolver:
    """
    Apply Wiener deconvolution to the full image using the recovered PSF.

    H_full_size: OTF at the full image size (not just the ROI).
    """

    def deconvolve(self, image: np.ndarray, psf: np.ndarray,
                   wiener_k: float = 0.002) -> np.ndarray:
        """
        Wiener deconvolution: O = F* / (|F|² + k·|F|²_max) · I_fft

        Parameters
        ----------
        image    : full 2-D image to deconvolve
        psf      : recovered PSF (any size, will be padded)
        wiener_k : regularisation parameter (noise/signal ratio)
        """
        ny, nx  = image.shape
        psf_pad = np.zeros((ny, nx))
        hp, wp  = psf.shape
        cy, cx  = ny // 2, nx // 2
        hy, hx  = hp // 2, wp // 2
        # Place PSF centre at (0,0) for correct FFT
        psf_pad[cy-hy:cy+hy, cx-hx:cx+hx] = psf
        psf_pad = np.roll(np.roll(psf_pad, -cy, axis=0), -cx, axis=1)

        H  = np.fft.fft2(psf_pad)
        F  = np.fft.fft2(image.astype(float))

        H_sq   = np.abs(H) ** 2
        reg    = wiener_k * H_sq.max()
        O_fft  = np.conj(H) * F / (H_sq + reg)
        return np.real(np.fft.ifft2(O_fft))


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class PDWorker(QObject):
    progress   = pyqtSignal(int, str)
    iter_update= pyqtSignal(int, float)   # iteration, merit value
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, focused: np.ndarray, defocused: np.ndarray,
                 roi: tuple, params: dict):
        super().__init__()
        self.focused   = focused
        self.defocused = defocused
        self.roi       = roi         # (y0, y1, x0, x1) — star/reference ROI
        self.params    = params
        self._cancel   = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p = self.params
        y0, y1, x0, x1 = self.roi

        # ── 1. Extract ROIs ───────────────────────────────────────────────
        self.progress.emit(5, "Extracting ROI…")
        roi_f = self.focused [y0:y1, x0:x1].astype(float)
        roi_d = self.defocused[y0:y1, x0:x1].astype(float)

        # Normalise each ROI to [0,1]
        roi_f -= roi_f.min(); roi_f /= (roi_f.max() + 1e-10)
        roi_d -= roi_d.min(); roi_d /= (roi_d.max() + 1e-10)

        sz = roi_f.shape[0]

        # ── 2. Build Zernike basis ────────────────────────────────────────
        self.progress.emit(10, "Building Zernike basis…")
        n_modes = p.get('n_modes', 10)
        N_pup   = p.get('pupil_size', 128)
        zb      = ZernikeBasis(N_pup, n_modes)

        psf_size = min(sz, N_pup // 2) * 2
        psf_size = max(16, psf_size - psf_size % 2)
        sim     = PSFSimulator(zb, psf_size)
        solver  = PhaseDiversitySolver(
            sim,
            n_modes        = n_modes,
            defocus_waves  = p.get('defocus_waves', 1.0),
            noise_reg      = p.get('noise_reg', 1e-4),
        )

        # ── 3. Solve ──────────────────────────────────────────────────────
        self.progress.emit(15, f"Optimising {n_modes} Zernike modes…")

        def prog_cb(it, val):
            pct = min(90, 15 + int(75 * it / 3000))
            self.progress.emit(pct, f"Iteration {it}  merit={val:.4e}")
            self.iter_update.emit(it, val)

        sol = solver.solve(roi_f, roi_d, progress_cb=prog_cb)

        if self._cancel:
            return

        self.progress.emit(91, f"Converged: {sol['converged']}  "
                               f"Strehl={sol['strehl']*100:.1f}%")

        # ── 4. Reconstruct ROI object ─────────────────────────────────────
        self.progress.emit(93, "Reconstructing object from ROI…")
        roi_recon = solver.reconstruct_object(roi_f, roi_d, sol['coeffs'])

        # ── 5. Deconvolve full image ──────────────────────────────────────
        self.progress.emit(95, "Deconvolving full image…")
        dec    = ImageDeconvolver()
        deconv = dec.deconvolve(
            self.focused, sol['psf'],
            wiener_k=p.get('wiener_k', 0.002))

        self.progress.emit(100, "Done!")

        self.finished.emit({
            'coeffs':    sol['coeffs'],
            'strehl':    sol['strehl'],
            'history':   sol['history'],
            'psf':       sol['psf'],
            'n_modes':   n_modes,
            'n_iter':    sol['n_iter'],
            'converged': sol['converged'],
            'roi_focused':  roi_f,
            'roi_defocus':  roi_d,
            'roi_recon':    roi_recon,
            'full_deconv':  deconv,
            'zb':           zb,
            'roi':          self.roi,
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow, QWidget {
    background:#111120; color:#d0d0e8;
    font-family:'Segoe UI',Arial,sans-serif; font-size:12px;
}
QTabWidget::pane  { border:1px solid #26264a; background:#151530; }
QTabBar::tab      { background:#1a1a34; color:#6868a0; padding:6px 18px;
                    border:1px solid #26264a; border-bottom:none; }
QTabBar::tab:selected { background:#20204c; color:#9898ff;
                        border-bottom:2px solid #5050c8; }
QGroupBox         { border:1px solid #26264a; border-radius:4px;
                    margin-top:8px; padding-top:8px; color:#6868a8;
                    font-weight:bold; }
QGroupBox::title  { subcontrol-origin:margin; left:8px; padding:0 4px; }
QPushButton       { background:#1e1e42; color:#b0b0d8;
                    border:1px solid #3838a8; border-radius:4px; padding:5px 14px; }
QPushButton:hover   { background:#28285e; border-color:#5050c8; }
QPushButton:disabled{ color:#383858; border-color:#242448; }
QPushButton#run_btn { background:#281a5c; color:#c0a0ff;
                      border:1px solid #7054c8; font-weight:bold; padding:7px 20px; }
QPushButton#run_btn:hover { background:#382a78; }
QSpinBox, QDoubleSpinBox, QComboBox, QLabel#val {
    background:#181832; border:1px solid #26264a;
    border-radius:3px; padding:3px 6px; color:#d0d0e8; }
QTextEdit { background:#0e0e20; border:1px solid #26264a;
            color:#85e085; font-family:Consolas,monospace; font-size:11px; }
QSplitter::handle { background:#26264a; }
QScrollArea       { border:none; }
QLabel#title_lbl  { font-size:19px; font-weight:bold; color:#9878ff; padding:6px; }
QLabel#sub_lbl    { font-size:11px; color:#504870; padding:0 8px 6px; }
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(7,5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor='#191930')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)

    def redraw(self):
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()


def _ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#0d0d1e')
    ax.set_title(title, color='#9090ff', fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color='#545472', fontsize=8)
    ax.set_ylabel(ylabel, color='#545472', fontsize=8)
    ax.tick_params(colors='#545472', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#26264a')


def _cbar(fig, m, ax, label=''):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    cb  = fig.colorbar(m, cax=cax)
    cb.ax.yaxis.set_tick_params(colors='#7070a0', labelsize=7)
    if label: cb.set_label(label, color='#7070a0', fontsize=7)


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#0b0b18; border-top:1px solid #26264a;")
        lay = QHBoxLayout(self); lay.setContentsMargins(6,0,6,0)
        self.lbl  = QLabel("Ready")
        self.lbl.setStyleSheet("color:#585878; font-size:10px;")
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


def luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2: return data.astype(float)
    if data.ndim == 3 and data.shape[0] <= 4:
        return data.mean(axis=0).astype(float)
    return data.astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseDiversityApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1380, 900)
        self.setStyleSheet(DARK)

        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            self.setWindowIcon(QIcon(str(_logo)))

        self._focused   = None   # 2-D float array
        self._defocused = None
        self._result    = None
        self._roi       = None   # (y0,y1,x0,x1)
        self._worker    = None
        self._wthread   = None
        self._merit_history = []

        self._build_ui()
        if not ASTROPY_OK:
            self._log("⚠  astropy not found — pip install astropy")

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
                50, 50, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Phase Diversity Wavefront Sensor"); t.setObjectName("title_lbl")
        s = QLabel("Zernike wavefront recovery · Focused + defocused pair · "
                   "Full-image Wiener deconvolution · No reference star required")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        ver = QLabel(f"v{VERSION}  |  HYPERLOAD")
        ver.setStyleSheet("color:#30305a; font-size:10px;")
        lay.addWidget(ver)
        return hdr

    def _make_controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(7)

        # ── Input ──────────────────────────────────────────────────────────
        grp = QGroupBox("Input Images")
        gl  = QGridLayout(grp)
        self.lbl_focused  = QLabel("Focused:  —")
        self.lbl_focused.setStyleSheet("color:#484868;font-size:10px;")
        self.lbl_defocus  = QLabel("Defocused: —")
        self.lbl_defocus.setStyleSheet("color:#484868;font-size:10px;")
        btn_f = QPushButton("📂  Focused FITS…");   btn_f.clicked.connect(self._load_focused)
        btn_d = QPushButton("📂  Defocused FITS…"); btn_d.clicked.connect(self._load_defocused)
        gl.addWidget(self.lbl_focused, 0, 0, 1, 2)
        gl.addWidget(btn_f,            1, 0, 1, 2)
        gl.addWidget(self.lbl_defocus, 2, 0, 1, 2)
        gl.addWidget(btn_d,            3, 0, 1, 2)
        lay.addWidget(grp)

        # ── ROI ────────────────────────────────────────────────────────────
        grp2 = QGroupBox("Reference Region (star or planet limb)")
        g2   = QGridLayout(grp2)
        g2.addWidget(QLabel("ROI centre X:"), 0, 0)
        self.sp_cx = QSpinBox(); self.sp_cx.setRange(0,9999); self.sp_cx.setValue(0)
        g2.addWidget(self.sp_cx, 0, 1)
        g2.addWidget(QLabel("ROI centre Y:"), 1, 0)
        self.sp_cy = QSpinBox(); self.sp_cy.setRange(0,9999); self.sp_cy.setValue(0)
        g2.addWidget(self.sp_cy, 1, 1)
        g2.addWidget(QLabel("ROI size (px):"), 2, 0)
        self.sp_roi = QSpinBox(); self.sp_roi.setRange(16,512)
        self.sp_roi.setValue(64); self.sp_roi.setSingleStep(16)
        g2.addWidget(self.sp_roi, 2, 1)
        btn_auto = QPushButton("🎯  Auto-find star")
        btn_auto.clicked.connect(self._auto_find_star)
        g2.addWidget(btn_auto, 3, 0, 1, 2)
        lay.addWidget(grp2)

        # ── Wavefront ──────────────────────────────────────────────────────
        grp3 = QGroupBox("Wavefront Parameters")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("Zernike modes:"), 0, 0)
        self.sp_modes = QSpinBox(); self.sp_modes.setRange(4,21)
        self.sp_modes.setValue(10)
        self.sp_modes.setToolTip("More modes = more detail but slower\n"
                                  "10-12 recommended for planetary imaging")
        g3.addWidget(self.sp_modes, 0, 1)
        g3.addWidget(QLabel("Pupil grid size:"), 1, 0)
        self.sp_pupil = QSpinBox(); self.sp_pupil.setRange(32,512)
        self.sp_pupil.setValue(128); self.sp_pupil.setSingleStep(32)
        g3.addWidget(self.sp_pupil, 1, 1)
        g3.addWidget(QLabel("Defocus (waves):"), 2, 0)
        self.sp_defocus = QDoubleSpinBox()
        self.sp_defocus.setRange(0.1, 5.0); self.sp_defocus.setValue(1.0)
        self.sp_defocus.setSingleStep(0.1); self.sp_defocus.setDecimals(2)
        self.sp_defocus.setToolTip(
            "How much defocus was applied to the second image.\n"
            "1 wave = one Rayleigh length of defocus (typical).")
        g3.addWidget(self.sp_defocus, 2, 1)
        lay.addWidget(grp3)

        # ── Deconvolution ──────────────────────────────────────────────────
        grp4 = QGroupBox("Full-image Deconvolution")
        g4   = QGridLayout(grp4)
        g4.addWidget(QLabel("Wiener k:"), 0, 0)
        self.sp_wiener = QDoubleSpinBox()
        self.sp_wiener.setRange(1e-5, 0.5); self.sp_wiener.setValue(0.002)
        self.sp_wiener.setDecimals(4); self.sp_wiener.setSingleStep(0.001)
        self.sp_wiener.setToolTip(
            "Noise/signal regularisation.\n"
            "Smaller = sharper but noisier.\n"
            "0.001–0.01 typical.")
        g4.addWidget(self.sp_wiener, 0, 1)
        lay.addWidget(grp4)

        # ── Output ────────────────────────────────────────────────────────
        grp5 = QGroupBox("Output")
        g5   = QVBoxLayout(grp5)
        self.btn_save_deconv = QPushButton("💾  Save Deconvolved FITS…")
        self.btn_save_deconv.setEnabled(False)
        self.btn_save_deconv.clicked.connect(self._save_deconv)
        self.btn_save_report = QPushButton("📋  Save Wavefront Report (JSON)")
        self.btn_save_report.setEnabled(False)
        self.btn_save_report.clicked.connect(self._save_report)
        g5.addWidget(self.btn_save_deconv)
        g5.addWidget(self.btn_save_report)
        lay.addWidget(grp5)

        # ── Run ───────────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Solve Wavefront")
        self.btn_run.setObjectName("run_btn"); self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("color:#b05050;")
        self.btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_cancel)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color:#70c070;font-size:10px;padding:4px;")
        self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)

        scroll.setWidget(inner)
        return scroll

    def _make_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_input(),    "🖼  Input Preview")
        self.tabs.addTab(self._tab_wavefront(),"🌊  Wavefront")
        self.tabs.addTab(self._tab_psf(),      "🔭  PSF")
        self.tabs.addTab(self._tab_deconv(),   "✨  Deconvolution")
        self.tabs.addTab(self._tab_convergence(),"📈  Convergence")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_input(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.input_plot = MplCanvas(figsize=(9,5))
        lay.addWidget(self.input_plot)
        info = QLabel("Load focused + defocused images, then click 'Auto-find star' "
                       "or set ROI manually.")
        info.setStyleSheet("color:#585878;font-size:10px;padding:4px;")
        lay.addWidget(info)
        return w

    def _tab_wavefront(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.wf_plot = MplCanvas(figsize=(9,5))
        lay.addWidget(self.wf_plot)
        return w

    def _tab_psf(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.psf_plot = MplCanvas(figsize=(9,5))
        lay.addWidget(self.psf_plot)
        return w

    def _tab_deconv(self):
        w = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Stretch:"))
        self.cmb_stretch = QComboBox()
        self.cmb_stretch.addItems(["Linear","Sqrt","Log","Asinh"])
        self.cmb_stretch.currentIndexChanged.connect(self._refresh_deconv)
        ctrl.addWidget(self.cmb_stretch); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.deconv_plot = MplCanvas(figsize=(9,5))
        lay.addWidget(self.deconv_plot)
        return w

    def _tab_convergence(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.conv_plot = MplCanvas(figsize=(9,4))
        lay.addWidget(self.conv_plot)
        self.zernike_plot = MplCanvas(figsize=(9,3))
        lay.addWidget(self.zernike_plot)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _load_fits(self, label) -> np.ndarray:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FITS", "", "FITS (*.fits *.fit *.fts);;All (*)")
        if not path: return None
        data = luminance(fits.getdata(path))
        label.setText(Path(path).name)
        label.setStyleSheet("color:#70c070;font-size:10px;")
        self._log(f"Loaded: {path}  shape={data.shape}")
        return data

    def _load_focused(self):
        d = self._load_fits(self.lbl_focused)
        if d is not None:
            self._focused = d
            # Default ROI to image centre
            cy, cx = d.shape[0]//2, d.shape[1]//2
            self.sp_cy.setValue(cy); self.sp_cx.setValue(cx)
            self._refresh_input_preview()
            self._check_ready()

    def _load_defocused(self):
        d = self._load_fits(self.lbl_defocus)
        if d is not None:
            self._defocused = d
            self._refresh_input_preview()
            self._check_ready()

    def _check_ready(self):
        self.btn_run.setEnabled(
            self._focused is not None and self._defocused is not None)

    def _auto_find_star(self):
        if self._focused is None: return
        # Find brightest compact peak via Laplacian-of-Gaussian
        from scipy.ndimage import gaussian_filter, label, find_objects
        img = self._focused.astype(float)
        smooth = gaussian_filter(img, 2.0)
        thresh = np.percentile(smooth, 99)
        mask   = smooth > thresh
        lab, _ = label(mask)
        objs   = find_objects(lab)
        if objs:
            # Pick largest
            obj = max(objs, key=lambda o: (o[0].stop-o[0].start)*(o[1].stop-o[1].start))
            cy = (obj[0].start + obj[0].stop) // 2
            cx = (obj[1].start + obj[1].stop) // 2
            self.sp_cy.setValue(cy); self.sp_cx.setValue(cx)
            self._log(f"Auto-found star at ({cx}, {cy})")
            self._refresh_input_preview()

    def _save_deconv(self):
        if self._result is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Deconvolved FITS", "", "FITS (*.fits)")
        if not path: return
        hdr = fits.Header()
        hdr['HISTORY'] = f'HYPERLOAD Phase Diversity Deconvolution v{VERSION}'
        hdr['STREHL']  = (round(float(self._result['strehl']), 4), 'Estimated Strehl ratio')
        hdr['WIENER_K']= (self.sp_wiener.value(), 'Wiener parameter')
        hdr['N_MODES'] = (self._result['n_modes'], 'Zernike modes fitted')
        fits.PrimaryHDU(self._result['full_deconv'].astype(np.float32),
                        header=hdr).writeto(path, overwrite=True)
        self._log(f"✓ Saved: {path}")

    def _save_report(self):
        if self._result is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "JSON (*.json)")
        if not path: return
        r = self._result
        zb = r['zb']
        report = {
            'date_utc':   datetime.utcnow().isoformat()+'Z',
            'hyperload':  VERSION,
            'n_modes':    r['n_modes'],
            'strehl':     float(r['strehl']),
            'converged':  r['converged'],
            'n_iter':     r['n_iter'],
            'defocus_waves': self.sp_defocus.value(),
            'zernike_coefficients': {
                zb.mode_name(i): float(r['coeffs'][i])
                for i in range(r['n_modes'])
            },
            'rms_wavefront_rad': float(np.std(
                sum(c*z for c,z in zip(r['coeffs'], zb.basis))[zb.pupil])),
        }
        with open(path, 'w') as f: json.dump(report, f, indent=2)
        self._log(f"✓ Report: {path}")

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _run(self):
        if self._focused is None or self._defocused is None: return
        cy = self.sp_cy.value(); cx = self.sp_cx.value()
        sz = self.sp_roi.value() // 2
        y0 = max(0, cy-sz); y1 = min(self._focused.shape[0], cy+sz)
        x0 = max(0, cx-sz); x1 = min(self._focused.shape[1], cx+sz)
        roi = (y0, y1, x0, x1)
        self._roi = roi
        self._merit_history = []

        params = {
            'n_modes':      self.sp_modes.value(),
            'pupil_size':   self.sp_pupil.value(),
            'defocus_waves':self.sp_defocus.value(),
            'noise_reg':    1e-4,
            'wiener_k':     self.sp_wiener.value(),
        }

        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_save_deconv.setEnabled(False)
        self.btn_save_report.setEnabled(False)
        self._result = None

        self._wthread = QThread(self)
        self._worker  = PDWorker(self._focused, self._defocused, roi, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.iter_update.connect(self._on_iter)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()
        self.tabs.setCurrentIndex(4)  # Convergence tab

    def _cancel(self):
        if self._worker: self._worker.cancel()
        self.btn_cancel.setEnabled(False); self.btn_run.setEnabled(True)

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status.update(pct, msg)
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_iter(self, it, val):
        self._merit_history.append(val)
        if len(self._merit_history) % 10 == 0:
            self._draw_convergence_live()

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}")
        self.status.update(0, "Error")
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        QMessageBox.critical(self, "Error", msg[:500])

    def _on_finished(self, result):
        self._result = result
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.btn_save_deconv.setEnabled(True)
        self.btn_save_report.setEnabled(True)

        r = result
        self._log(f"\n{'='*55}")
        self._log(f"✓  Phase diversity solution complete")
        self._log(f"   Strehl ratio:   {r['strehl']*100:.1f}%")
        self._log(f"   Converged:      {r['converged']}")
        self._log(f"   Iterations:     {r['n_iter']}")
        self._log(f"   Modes fitted:   {r['n_modes']}")
        zb = r['zb']
        self._log(f"\n   Zernike coefficients (radians):")
        for i in range(r['n_modes']):
            self._log(f"     {zb.mode_name(i):20s}  {r['coeffs'][i]:+.4f}")
        self._log(f"{'='*55}\n")

        self.summary_lbl.setText(
            f"Strehl {r['strehl']*100:.1f}%  |  "
            f"{r['n_modes']} modes  |  "
            f"{'converged' if r['converged'] else 'max iter'}")

        self._draw_wavefront_tab()
        self._draw_psf_tab()
        self._refresh_deconv()
        self._draw_convergence_live()
        self._draw_zernike_bar()
        self.tabs.setCurrentIndex(3)   # jump to deconv

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _stretch(self, img):
        lo, hi = np.percentile(img[np.isfinite(img)], [0.5, 99.5])
        c = np.clip(img-lo, 0, hi-lo+1e-10)
        m = self.cmb_stretch.currentIndex()
        if m == 0: return c/(hi-lo+1e-10)
        if m == 1: return np.sqrt(c/(hi-lo+1e-10))
        if m == 2: return np.log1p(c)/np.log1p(hi-lo+1)
        return np.arcsinh(c/max((hi-lo)*0.1,1e-10))/np.arcsinh(10)

    def _refresh_input_preview(self):
        if self._focused is None: return
        fig = self.input_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        n = 2 if self._defocused is not None else 1
        axes = fig.subplots(1, n)
        if n == 1: axes = [axes]

        def _show(ax, img, title):
            _ax(ax, title)
            lo,hi = np.percentile(img,[0.5,99.5])
            ax.imshow(np.sqrt(np.clip(img-lo,0,hi-lo)),
                      origin='lower',cmap='inferno',aspect='equal',
                      interpolation='nearest')
            if self._roi:
                y0,y1,x0,x1 = self._roi
                rect = mpatches.Rectangle(
                    (x0,y0), x1-x0, y1-y0,
                    lw=1.5, edgecolor='#60ffff', facecolor='none')
                ax.add_patch(rect)

        _show(axes[0], self._focused, "Focused image")
        if n == 2:
            _show(axes[1], self._defocused, "Defocused image")
        self.input_plot.redraw()

    def _draw_wavefront_tab(self):
        if self._result is None: return
        r = self._result; zb = r['zb']
        WF = zb.wavefront(r['coeffs'])

        fig = self.wf_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1, 3)

        # Wavefront map
        ax = axes[0]; _ax(ax, "Recovered wavefront (radians)")
        wf_m = WF.copy(); wf_m[~zb.pupil] = np.nan
        im = ax.imshow(wf_m, origin='lower', cmap='RdBu_r',
                       vmin=-np.pi, vmax=np.pi, aspect='equal')
        _cbar(fig, im, ax, "rad")

        # Wavefront over pupil — 3D-style colour map
        ax2 = axes[1]; _ax(ax2, "Wavefront error (pupil only)")
        wf_p = np.ma.masked_where(~zb.pupil, WF)
        im2 = ax2.imshow(wf_p, origin='lower', cmap='coolwarm',
                         aspect='equal', interpolation='bilinear')
        _cbar(fig, im2, ax2, "rad")
        rms = np.std(WF[zb.pupil])
        ax2.set_title(f"RMS = {rms:.3f} rad  Strehl = {r['strehl']*100:.1f}%",
                      color='#9090ff', fontsize=9, pad=4)

        # Residual PSF comparison
        ax3 = axes[2]; _ax(ax3, "Recovered PSF")
        ax3.imshow(r['psf'], origin='lower', cmap='inferno',
                   aspect='equal', interpolation='nearest')

        self.wf_plot.redraw()

    def _draw_psf_tab(self):
        if self._result is None: return
        r = self._result; zb = r['zb']

        fig = self.psf_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1, 3)

        # Diffraction-limited PSF (zero aberration)
        zero_coeffs = np.zeros(r['n_modes'])
        sim = PSFSimulator(zb, r['psf'].shape[0])
        psf_dl = sim.psf(zero_coeffs)

        ax = axes[0]; _ax(ax, "Diffraction-limited PSF\n(no aberration)")
        ax.imshow(np.sqrt(psf_dl), origin='lower', cmap='inferno',
                  aspect='equal', interpolation='nearest')

        ax2 = axes[1]; _ax(ax2, "Recovered PSF\n(measured aberration)")
        ax2.imshow(np.sqrt(r['psf']), origin='lower', cmap='inferno',
                   aspect='equal', interpolation='nearest')

        # Radial profiles
        ax3 = axes[2]; _ax(ax3, "Radial PSF profiles", "Radius (px)", "Intensity")
        def radial(psf):
            cy, cx = psf.shape[0]//2, psf.shape[1]//2
            y, x = np.ogrid[:psf.shape[0], :psf.shape[1]]
            r_map = np.sqrt((x-cx)**2+(y-cy)**2)
            r_max = min(cy,cx)
            return np.array([psf[(r_map>=b)&(r_map<b+1)].mean()
                             for b in range(r_max)])

        rp_dl  = radial(psf_dl)
        rp_rec = radial(r['psf'])
        ax3.plot(rp_dl,  c='#4080ff', lw=2, label='Diffraction limit')
        ax3.plot(rp_rec, c='#ff6040', lw=2, label='Measured')
        ax3.legend(fontsize=7, facecolor='#1a1a30',
                   edgecolor='#3030a0', labelcolor='white')
        ax3.set_yscale('log')

        self.psf_plot.redraw()

    def _refresh_deconv(self):
        if self._result is None: return
        r = self._result

        fig = self.deconv_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1, 3)

        ax = axes[0]; _ax(ax, "Original (focused)")
        ax.imshow(self._stretch(self._focused), origin='lower',
                  cmap='inferno', aspect='equal', interpolation='nearest')
        if self._roi:
            y0,y1,x0,x1=self._roi
            ax.add_patch(mpatches.Rectangle(
                (x0,y0),x1-x0,y1-y0,lw=1.5,edgecolor='#60ffff',facecolor='none'))

        ax2 = axes[1]; _ax(ax2, "Deconvolved\n(Wiener, recovered PSF)")
        ax2.imshow(self._stretch(r['full_deconv']), origin='lower',
                   cmap='inferno', aspect='equal', interpolation='nearest')

        ax3 = axes[2]; _ax(ax3, "ROI: joint object estimate")
        ax3.imshow(self._stretch(r['roi_recon']), origin='lower',
                   cmap='inferno', aspect='equal', interpolation='nearest')

        self.deconv_plot.redraw()

    def _draw_convergence_live(self):
        if not self._merit_history: return
        fig = self.conv_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "Merit function convergence", "Iteration", "L (lower = better)")
        h = np.array(self._merit_history)
        ax.semilogy(h, c='#5070ff', lw=1.2)
        ax.fill_between(range(len(h)), h, alpha=0.15, color='#5070ff')
        self.conv_plot.redraw()

    def _draw_zernike_bar(self):
        if self._result is None: return
        r = self._result; zb = r['zb']
        fig = self.zernike_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "Zernike coefficients", "Mode", "Coefficient (radians)")
        idx    = np.arange(r['n_modes'])
        coeffs = r['coeffs']
        colors = ['#ff5040' if abs(c) > 0.3 else '#5070ff' for c in coeffs]
        ax.bar(idx, coeffs, color=colors, edgecolor='#26264a', width=0.7)
        ax.axhline(0, color='#404070', lw=0.8)
        names = [zb.mode_name(i)[:8] for i in range(r['n_modes'])]
        ax.set_xticks(idx); ax.set_xticklabels(names, rotation=45,
                                                 ha='right', fontsize=7,
                                                 color='#7070a0')
        self.zernike_plot.redraw()

    def _draw_convergence_live(self):
        if not self._merit_history: return
        fig = self.conv_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "Merit function (Nelder-Mead)", "Function evaluations", "L")
        h = np.array(self._merit_history)
        ax.semilogy(h, c='#5070ff', lw=1.2)
        ax.fill_between(range(len(h)), h, alpha=0.15, color='#5070ff')
        self.conv_plot.redraw()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PhaseDiversityApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
