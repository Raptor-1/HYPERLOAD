r"""
HYPERLOAD — CCD Fringe Corrector
==================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Removes interference fringes from CCD spectra caused by the etalon effect
in back-illuminated, thinned CCDs.  The CCD silicon layer acts as a
Fabry-Pérot etalon at red/NIR wavelengths (>650 nm), producing sinusoidal
modulations of 2–8% amplitude that are wavelength- and position-dependent.

Works as companion to spectral_extractor.py:
    from ccd_fringe_corrector import FringeCorrectorPipeline
    result = FringeCorrectorPipeline().run(fits_path, wav_solution)

Algorithm
---------
1. CONTINUUM FIT  —  Iterative polynomial fit with sigma-clipping
   isolates the fringes without touching spectral lines.

2. OPD DETECTION  —  FFT of the fringe signal in evenly-sampled
   wavenumber (σ = 1/λ) space.  The dominant peak gives the optical
   path difference OPD = 2·n·d directly.  The frequency axis is in
   cm, so the peak position = OPD in cm → CCD thickness d in μm.

3. SINUSOIDAL FIT  —  Fit A·cos(2π·OPD·σ + φ) at the detected OPD.
   Uses the Hilbert transform to extract a smooth amplitude envelope
   (fringe amplitude is wavelength-dependent: A ∝ λ^{-0.5}).

4. 2D CORRECTION  —  For 2D spectral images, fit fringe phase at each
   spatial row separately.  Phase changes smoothly along the slit due
   to the varying angle of incidence; a polynomial phase map is fitted
   and interpolated to all rows.

5. SCALE FITTING  —  Optional amplitude scale factor optimised to
   minimise RMS of the corrected spectrum.  Useful when the fringe
   amplitude in the science frame differs from the model.

Key parameters (typical back-illuminated CCD):
  CCD thickness d  : 10–25 μm  (silicon)
  Refractive index : 3.5  (Si at 750 nm)
  OPD              : 2 × 3.5 × d = 70–175 μm
  Fringe period    : λ²/(2·n·d) ≈ 4–8 nm  at 750 nm
  Contrast         : 2–8 % amplitude

Companion script connections
-----------------------------
  spectral_extractor.py  →  ccd_fringe_corrector.py  (2D FITS input)
  ccd_fringe_corrector.py →  spectral_extractor.py  (corrected FITS output)

Reference
---------
  Palunas et al. 2000  — CCD fringe modelling (PASP)
  Barden et al. 2000   — Fringe pattern synthesis
  Roth et al. 2015     — MUSE pipeline fringe correction

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, traceback, math
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nccd_fringe_corrector.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6", "astropy", "scipy", "matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import hilbert, savgol_filter

try:
    from astropy.io import fits as astropy_fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QFormLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — CCD Fringe Corrector"

# CCD presets: name → (n_refr, d_range_um, wav_start_nm)
CCD_PRESETS = {
    "Generic back-illuminated":  (3.5,  (10, 25), 650),
    "Sony IMX571/455 (ASI2600)": (3.5,  (5,  15), 680),
    "Kodak KAF-8300":            (3.6,  (12, 22), 650),
    "Hamamatsu (deep depl.)":    (3.45, (40, 80), 700),
    "E2V (thick, deep depl.)":   (3.5,  (50,100), 720),
    "Custom":                    (3.5,  (5, 100), 600),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS  (importable by spectral_extractor.py)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_continuum(wav: np.ndarray, spec: np.ndarray,
                  degree: int = 5, n_iter: int = 7,
                  sigma_clip: float = 2.5) -> np.ndarray:
    """
    Iterative polynomial continuum fit with sigma-clipping.
    Robustly excludes spectral lines and fringe peaks from the fit.
    """
    mask = np.ones(len(wav), dtype=bool)
    coeffs = None
    for _ in range(n_iter):
        wc = wav[mask] - wav.mean()
        sc = spec[mask]
        coeffs = np.polyfit(wc, sc, degree)
        cont   = np.polyval(coeffs, wav - wav.mean())
        rms    = float(np.std((spec - cont)[mask]))
        mask   = np.abs(spec - cont) < sigma_clip * rms
        if mask.sum() < degree + 2:
            mask = np.ones(len(wav), dtype=bool)
            break
    return np.polyval(coeffs, wav - wav.mean())


def detect_opd(wav_nm: np.ndarray, fringe: np.ndarray,
               n_refr: float = 3.5,
               opd_min_um: float = 5.0,
               opd_max_um: float = 150.0) -> dict:
    """
    Detect the dominant CCD fringe OPD via FFT in wavenumber space.

    The fringe signal is sinusoidal in wavenumber σ = 1/λ:
        f(σ) = A·cos(2π·OPD·σ + φ)
    with OPD = 2·n·d  (optical path difference, n=refractive index, d=thickness)

    Returns dict with: opd_cm, d_um, fringe_period_nm, spectrum (FFT power)
    """
    # Resample to evenly-spaced wavenumber grid
    sigma     = 1e7 / wav_nm                # cm⁻¹
    sigma_even= np.linspace(sigma.min(), sigma.max(), len(sigma))
    fringe_rs = np.interp(sigma_even, sigma[::-1], fringe[::-1])
    d_sigma   = sigma_even[1] - sigma_even[0]

    # FFT: frequency axis = OPD in cm
    power  = np.abs(np.fft.rfft(fringe_rs))**2
    freqs  = np.fft.rfftfreq(len(sigma_even), d=d_sigma)   # cm

    # Search in physical OPD range
    opd_min = opd_min_um  * 2 * n_refr * 1e-4   # μm → cm
    opd_max = opd_max_um  * 2 * n_refr * 1e-4
    mask_f  = (freqs >= opd_min) & (freqs <= opd_max)
    if mask_f.sum() < 2:
        return {'opd_cm': 0, 'd_um': 0, 'fringe_period_nm': 0,
                'power': power, 'freqs': freqs}

    peak_idx   = np.argmax(power[mask_f])
    opd_cm     = float(freqs[mask_f][peak_idx])
    d_um       = opd_cm / (2 * n_refr) * 1e4
    wav_mid    = float(wav_nm.mean())
    period_nm  = wav_mid**2 / (2 * n_refr * d_um * 1e3)

    return {
        'opd_cm':          opd_cm,
        'd_um':            d_um,
        'fringe_period_nm':period_nm,
        'power':           power[mask_f],
        'freqs':           freqs[mask_f],
        'peak_snr':        float(power[mask_f][peak_idx] / np.median(power[mask_f])),
    }


def fit_fringe_model(wav_nm: np.ndarray, fringe: np.ndarray,
                     opd_cm: float, n_refr: float = 3.5) -> dict:
    """
    Fit sinusoidal fringe model at detected OPD.
    Uses Hilbert transform for amplitude envelope.

    Returns: dict with template, A_fit, phi_fit, amp_envelope
    """
    sigma = 1e7 / wav_nm

    # Fit phase at fixed OPD
    def model(s, A, phi):
        return A * np.cos(2 * np.pi * opd_cm * s + phi)

    try:
        amp_guess = float(np.std(fringe) * np.sqrt(2))
        popt, _ = curve_fit(
            model, sigma, fringe,
            p0=[amp_guess, 0.5],
            bounds=([0, -np.pi], [1.0, 3 * np.pi]),
            maxfev=8000)
        A_fit, phi_fit = popt
    except Exception:
        A_fit  = float(np.std(fringe) * np.sqrt(2))
        phi_fit= 0.0

    # Hilbert-transform amplitude envelope (captures λ-dependent amplitude)
    analytic      = hilbert(fringe)
    amp_env_raw   = np.abs(analytic)
    amp_envelope  = gaussian_filter1d(amp_env_raw, sigma=max(5, len(wav_nm)//80))

    # Build template
    template = amp_envelope * np.cos(2 * np.pi * opd_cm * sigma + phi_fit)

    return {
        'template':     template,
        'A_fit':        A_fit,
        'phi_fit':      phi_fit,
        'amp_envelope': amp_envelope,
    }


def correct_spectrum_1d(wav_nm: np.ndarray, spec: np.ndarray,
                         n_refr: float = 3.5,
                         degree: int = 5,
                         opd_min_um: float = 5.0,
                         opd_max_um: float = 150.0,
                         optimize_scale: bool = True) -> dict:
    """
    Full 1D fringe correction pipeline for an extracted spectrum.

    Parameters
    ----------
    wav_nm   : wavelength array in nm
    spec     : flux array
    n_refr   : silicon refractive index (~3.5)
    degree   : continuum polynomial degree
    opd_min/max_um : OPD search range in μm

    Returns
    -------
    dict: corrected, continuum, fringe_extracted, template, opd_result, model
    """
    # Restrict to fringe-affected region (>650 nm)
    good = np.isfinite(spec) & (spec > 0)
    if good.sum() < 20:
        return {'corrected': spec.copy(), 'error': 'Too few valid pixels'}

    # Continuum
    continuum = fit_continuum(wav_nm, spec, degree=degree)
    fringe_x  = spec / np.maximum(continuum, 1e-10) - 1.0

    # OPD detection
    opd_result = detect_opd(wav_nm, fringe_x, n_refr, opd_min_um, opd_max_um)
    if opd_result['opd_cm'] == 0:
        return {'corrected': spec.copy(), 'continuum': continuum,
                'fringe_extracted': fringe_x, 'error': 'No fringe OPD detected'}

    # Fringe model
    model = fit_fringe_model(wav_nm, fringe_x, opd_result['opd_cm'], n_refr)

    # Optional: optimise amplitude scale
    scale = 1.0
    if optimize_scale:
        def residual_rms(s):
            corr = spec - continuum * s * model['template']
            return float(np.std(corr / np.maximum(continuum, 1e-10)))
        res = minimize_scalar(residual_rms, bounds=(0.3, 3.0), method='bounded')
        scale = float(res.x)

    corrected = spec - continuum * scale * model['template']

    # Metrics
    rms_before = float(np.std(fringe_x))
    rms_after  = float(np.std(corrected / np.maximum(continuum, 1e-10) - 1.0))
    improvement = rms_before / max(rms_after, 1e-10)

    return {
        'corrected':        corrected,
        'continuum':        continuum,
        'fringe_extracted': fringe_x,
        'template':         model['template'],
        'amp_envelope':     model['amp_envelope'],
        'opd_result':       opd_result,
        'model':            model,
        'scale':            scale,
        'rms_before':       rms_before,
        'rms_after':        rms_after,
        'improvement':      improvement,
        'd_um':             opd_result['d_um'],
    }


def correct_image_2d(image: np.ndarray,
                     wav_nm: np.ndarray,
                     n_refr: float = 3.5,
                     degree: int = 5,
                     row_step: int = 5,
                     opd_min_um: float = 5.0,
                     opd_max_um: float = 150.0,
                     progress_cb=None) -> dict:
    """
    2D fringe correction for a full spectral image.

    For each row (spatial pixel), fits an independent fringe phase.
    Phase varies smoothly along the slit — fit with polynomial and
    interpolate to all rows.

    image : (N_spatial, N_spectral) 2D array
    wav_nm: (N_spectral,) wavelength axis
    """
    N_rows, N_cols = image.shape
    corrected_2d = image.copy().astype(float)

    # First: detect OPD from the median row (most reliable)
    median_row = np.median(image, axis=0)
    cont_med   = fit_continuum(wav_nm, median_row, degree=degree)
    fringe_med = median_row / np.maximum(cont_med, 1e-10) - 1.0
    opd_result = detect_opd(wav_nm, fringe_med, n_refr, opd_min_um, opd_max_um)

    if opd_result['opd_cm'] == 0:
        return {'corrected_2d': corrected_2d,
                'error': 'No fringe OPD found in median row'}

    opd_cm = opd_result['opd_cm']
    sigma  = 1e7 / wav_nm

    # Fit phase at sampled rows
    sample_rows  = list(range(0, N_rows, row_step))
    phase_fitted = []

    for i, row_idx in enumerate(sample_rows):
        if progress_cb:
            progress_cb(int(50 + 40*i/len(sample_rows)),
                        f"Fitting row {row_idx}/{N_rows}…")
        row   = image[row_idx].astype(float)
        cont  = fit_continuum(wav_nm, row, degree=degree)
        fring = row / np.maximum(cont, 1e-10) - 1.0
        mod   = fit_fringe_model(wav_nm, fring, opd_cm, n_refr)
        phase_fitted.append(mod['phi_fit'])

    # Polynomial fit to phase along slit
    phase_arr = np.array(phase_fitted)
    row_arr   = np.array(sample_rows, dtype=float)
    try:
        phase_poly = np.polyfit(row_arr, np.unwrap(phase_arr), 2)
    except Exception:
        phase_poly = np.polyfit(row_arr, phase_arr, 1)
    phase_all  = np.polyval(phase_poly, np.arange(N_rows))

    # Apply correction to each row
    for row_idx in range(N_rows):
        row  = image[row_idx].astype(float)
        cont = fit_continuum(wav_nm, row, degree=degree)
        fring= row / np.maximum(cont, 1e-10) - 1.0
        # Use global amplitude envelope, row-specific phase
        analytic    = hilbert(fring)
        amp_env     = gaussian_filter1d(np.abs(analytic),
                                         sigma=max(5, N_cols//80))
        template_r  = amp_env * np.cos(2*np.pi*opd_cm*sigma + phase_all[row_idx])

        # Optimise scale
        def res(s):
            return float(np.std((row - cont*s*template_r)/np.maximum(cont, 1e-10)))
        r = minimize_scalar(res, bounds=(0.3, 3.0), method='bounded')
        corrected_2d[row_idx] = row - cont * r.x * template_r

    rms_before = float(np.std(image - np.median(image)))
    rms_after  = float(np.std(corrected_2d - np.median(corrected_2d)))

    return {
        'corrected_2d': corrected_2d,
        'opd_result':   opd_result,
        'd_um':         opd_result['d_um'],
        'phase_all':    phase_all,
        'rms_before':   rms_before,
        'rms_after':    rms_after,
        'improvement':  rms_before / max(rms_after, 1e-10),
    }


class FringeCorrectorPipeline:
    """Thin wrapper for import by spectral_extractor.py."""

    def run_1d(self, wav_nm, spec, **kwargs):
        return correct_spectrum_1d(wav_nm, spec, **kwargs)

    def run_2d(self, image, wav_nm, **kwargs):
        return correct_image_2d(image, wav_nm, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class FringeWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, image, wav_nm, params, mode):
        super().__init__()
        self.image  = image
        self.wav_nm = wav_nm
        self.params = params
        self.mode   = mode   # '1d' or '2d'

    def run(self):
        try:
            p = self.params
            self.progress.emit(5, "Fitting continuum…")

            if self.mode == '1d':
                spec = self.image
                result = correct_spectrum_1d(
                    self.wav_nm, spec,
                    n_refr     = p['n_refr'],
                    degree     = p['degree'],
                    opd_min_um = p['opd_min'],
                    opd_max_um = p['opd_max'],
                    optimize_scale=p['optimize_scale'])
                self.progress.emit(90, "Done.")
            else:
                def prog(pct, msg):
                    self.progress.emit(pct, msg)
                result = correct_image_2d(
                    self.image, self.wav_nm,
                    n_refr     = p['n_refr'],
                    degree     = p['degree'],
                    row_step   = p['row_step'],
                    opd_min_um = p['opd_min'],
                    opd_max_um = p['opd_max'],
                    progress_cb= prog)

            if 'error' not in result:
                impr = result.get('improvement', 0)
                d    = result.get('d_um', 0)
                self.progress.emit(100,
                    f"Done!  d={d:.1f}μm  improvement={impr:.1f}×")
            else:
                self.progress.emit(100, f"Warning: {result['error']}")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

BG="#0e0c18"; BG2="#131020"; BG3="#1a1530"; BOR="#251e45"
ACC="#b060ff"; AC2="#6030a0"; TXT="#d8d0f0"; DIM="#352860"
OK="#60d090"; WRN="#d09040"; ERR="#c04040"; SEC="#9060e0"

DARK = f"""
QMainWindow,QWidget{{background:{BG};color:{TXT};
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}}
QTabWidget::pane{{border:1px solid {BOR};background:{BG2};}}
QTabBar::tab{{background:{BG3};color:{DIM};padding:6px 16px;
    border:1px solid {BOR};border-bottom:none;}}
QTabBar::tab:selected{{background:{BG2};color:{ACC};border-bottom:2px solid {ACC};}}
QGroupBox{{border:1px solid {BOR};border-radius:4px;margin-top:8px;
    padding-top:8px;color:{DIM};font-weight:bold;}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}
QPushButton{{background:{BG3};color:{TXT};border:1px solid {BOR};
    border-radius:4px;padding:5px 14px;}}
QPushButton:hover{{background:{AC2};border-color:{ACC};color:white;}}
QPushButton#run{{background:{AC2};border-color:{ACC};
    color:white;font-weight:bold;padding:7px 20px;}}
QSpinBox,QDoubleSpinBox,QComboBox{{background:{BG3};
    border:1px solid {BOR};border-radius:3px;padding:3px 6px;color:{TXT};}}
QTextEdit{{background:#08060e;border:1px solid {BOR};color:{OK};
    font-family:Consolas,monospace;font-size:10px;}}
QProgressBar{{background:{BG3};border:1px solid {BOR};
    border-radius:3px;height:8px;}}
QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {AC2},stop:1 {ACC});border-radius:2px;}}
QSplitter::handle{{background:{BOR};}}
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8, 4)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor=BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def redraw(self): self.fig.tight_layout(pad=0.3); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#07050d')
    ax.set_title(title, color=SEC, fontsize=9, pad=3)
    ax.set_xlabel(xl, color=DIM, fontsize=8)
    ax.set_ylabel(yl, color=DIM, fontsize=8)
    ax.tick_params(colors=DIM, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(BOR)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class FringeApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._image   = None   # 1D or 2D numpy array
        self._wav_nm  = None
        self._result  = None
        self._worker  = self._wthread = None
        self._is_2d   = False
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._controls())
        sp.addWidget(self._tabs())
        sp.setSizes([280, 1120])
        rl.addWidget(sp, 1)
        rl.addWidget(self._footer())

    def _header(self):
        hdr = QWidget(); hdr.setFixedHeight(60)
        hdr.setStyleSheet(f"background:{BG};border-bottom:1px solid {BOR};")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12,0,12,0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(44,44,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("CCD Fringe Corrector")
        t.setStyleSheet(f"color:{ACC};font-size:17px;font-weight:bold;padding:6px;")
        s = QLabel("Physics-based fringe removal for back-illuminated CCDs  ·  "
                   "OPD detection via FFT  ·  Hilbert amplitude envelope  ·  "
                   "1D spectra + 2D images  ·  Companion to spectral_extractor.py")
        s.setStyleSheet(f"color:{DIM};font-size:9pt;padding:0 8px 4px;")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(5)

        # Load
        grp = QGroupBox("Load Spectrum"); gl = QVBoxLayout(grp)
        self.file_lbl = QLabel("No file loaded")
        self.file_lbl.setStyleSheet(f"color:{DIM};font-size:10px;"); self.file_lbl.setWordWrap(True)
        b1 = QPushButton("📂  Load FITS (1D or 2D)…"); b1.setObjectName("run"); b1.clicked.connect(self._load_fits)
        b2 = QPushButton("📋  Synthetic example"); b2.clicked.connect(self._load_example)
        gl.addWidget(self.file_lbl); gl.addWidget(b1); gl.addWidget(b2)
        lay.addWidget(grp)

        # Wavelength axis
        grp_wav = QGroupBox("Wavelength Axis"); gw = QFormLayout(grp_wav)
        self.sp_wav0  = QDoubleSpinBox(); self.sp_wav0.setRange(300,1200); self.sp_wav0.setValue(620)
        self.sp_wav1  = QDoubleSpinBox(); self.sp_wav1.setRange(300,1200); self.sp_wav1.setValue(900)
        self.sp_dispax= QSpinBox(); self.sp_dispax.setRange(0,1); self.sp_dispax.setValue(1)
        gw.addRow("λ start (nm):", self.sp_wav0)
        gw.addRow("λ end (nm):",   self.sp_wav1)
        gw.addRow("Dispersion axis:", self.sp_dispax)
        lay.addWidget(grp_wav)

        # CCD parameters
        grp2 = QGroupBox("CCD Parameters"); g2 = QFormLayout(grp2)
        self.cb_preset = QComboBox(); self.cb_preset.addItems(list(CCD_PRESETS.keys()))
        self.cb_preset.currentTextChanged.connect(self._on_preset)
        self.sp_n    = QDoubleSpinBox(); self.sp_n.setRange(2.0,5.0); self.sp_n.setValue(3.5); self.sp_n.setDecimals(2)
        self.sp_dmin = QDoubleSpinBox(); self.sp_dmin.setRange(1,200); self.sp_dmin.setValue(5.0)
        self.sp_dmax = QDoubleSpinBox(); self.sp_dmax.setRange(5,500); self.sp_dmax.setValue(150.0)
        g2.addRow("CCD preset:", self.cb_preset)
        g2.addRow("n (Si refractive index):", self.sp_n)
        g2.addRow("d min (μm):", self.sp_dmin)
        g2.addRow("d max (μm):", self.sp_dmax)
        lay.addWidget(grp2)

        # Algorithm parameters
        grp3 = QGroupBox("Algorithm"); g3 = QFormLayout(grp3)
        self.sp_deg   = QSpinBox(); self.sp_deg.setRange(2,10); self.sp_deg.setValue(5)
        self.sp_rstep = QSpinBox(); self.sp_rstep.setRange(1,20); self.sp_rstep.setValue(5)
        self.cb_scale = QCheckBox("Optimise amplitude scale"); self.cb_scale.setChecked(True)
        g3.addRow("Continuum degree:", self.sp_deg)
        g3.addRow("Row step (2D only):", self.sp_rstep)
        g3.addRow(self.cb_scale)
        lay.addWidget(grp3)

        self.btn_run = QPushButton("▶  Correct Fringes")
        self.btn_run.setObjectName("run"); self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_exp = QPushButton("💾  Export corrected FITS…")
        self.btn_exp.setEnabled(False); self.btn_exp.clicked.connect(self._export)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_exp)
        lay.addStretch()
        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(f"color:{OK};font-size:10px;"); self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _on_preset(self, name):
        if name in CCD_PRESETS:
            n, (dmin, dmax), _ = CCD_PRESETS[name]
            self.sp_n.setValue(n); self.sp_dmin.setValue(dmin); self.sp_dmax.setValue(dmax)

    def _tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_input(),    "📥  Input")
        self.tabs.addTab(self._tab_analysis(), "🔬  Fringe Analysis")
        self.tabs.addTab(self._tab_result(),   "✨  Corrected")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_input(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.input_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.input_plot)
        return w

    def _tab_analysis(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.analysis_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.analysis_plot)
        return w

    def _tab_result(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.result_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.result_plot)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True); lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear); lay.addWidget(btn)
        return w

    def _footer(self):
        foot = QWidget(); foot.setFixedHeight(22)
        foot.setStyleSheet(f"background:{BG};border-top:1px solid {BOR};")
        lay = QHBoxLayout(foot); lay.setContentsMargins(6,0,6,0)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet(f"color:{DIM};font-size:10px;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(220,12); self.pbar.setVisible(False)
        lay.addWidget(self.status_lbl,1); lay.addWidget(self.pbar)
        return foot

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self,"Load spectrum FITS","","FITS (*.fits *.fit);;All (*)")
        if not path: return
        try:
            data = astropy_fits.getdata(path).astype(float)
            hdr  = astropy_fits.getheader(path)
            # Try to read wavelength from WCS
            if 'CRVAL1' in hdr and 'CDELT1' in hdr:
                npts = data.shape[-1]
                start= float(hdr['CRVAL1']); step=float(hdr['CDELT1'])
                wav  = start + step*np.arange(npts)
                # Convert Angstrom → nm if needed
                if wav.mean() > 5000: wav /= 10.0
                self.sp_wav0.setValue(wav.min()); self.sp_wav1.setValue(wav.max())
            self._set_data(data, path)
        except Exception as e:
            QMessageBox.critical(self,"Load error",str(e))

    def _set_data(self, data, label=""):
        self._is_2d = data.ndim == 2
        self._image = data
        ax = self.sp_dispax.value()
        nspec = data.shape[ax] if self._is_2d else len(data)
        wav0=self.sp_wav0.value(); wav1=self.sp_wav1.value()
        self._wav_nm = np.linspace(wav0, wav1, nspec)
        self.file_lbl.setText(f"{'2D' if self._is_2d else '1D'}  shape={data.shape}\n{Path(label).name if label else 'synthetic'}")
        self.file_lbl.setStyleSheet(f"color:{OK};font-size:10px;")
        self.btn_run.setEnabled(True)
        self._draw_input(); self._log(f"Loaded: {data.shape}  wav {wav0:.0f}-{wav1:.0f}nm")

    def _load_example(self):
        rng2 = np.random.default_rng(1)
        wav  = np.linspace(630, 880, 1500)
        cont = 1.0 + 0.03*(wav-630)/(880-630)
        # Ha absorption
        cont -= 0.2*np.exp(-((wav-656.3)**2)/4)
        # Fringes: d=12μm
        n=3.5; d=12; opd=2*n*d*1e-4
        sigma=1e7/wav; amp=0.035*(700/wav)**0.5
        fringe=1+amp*np.cos(2*np.pi*opd*sigma+0.8)
        spec=(cont*fringe+rng2.normal(0,0.004,len(wav))).astype(np.float32)
        self.sp_wav0.setValue(630); self.sp_wav1.setValue(880)
        self._set_data(spec,"synthetic_spectrum")
        self._log("Synthetic: d=12μm n=3.5 fringe ~3.5% contrast")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _params(self):
        return dict(
            n_refr=self.sp_n.value(), degree=self.sp_deg.value(),
            opd_min=self.sp_dmin.value(), opd_max=self.sp_dmax.value(),
            row_step=self.sp_rstep.value(), optimize_scale=self.cb_scale.isChecked())

    def _run(self):
        if self._image is None: return
        self.btn_run.setEnabled(False); self.btn_exp.setEnabled(False)
        data = self._image
        if self._is_2d and self.sp_dispax.value() == 0:
            data = data.T
        mode = '2d' if (self._is_2d) else '1d'
        self._wthread = QThread(self)
        self._worker  = FringeWorker(data, self._wav_nm, self._params(), mode)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _on_progress(self, pct, msg):
        self.status_lbl.setText(msg)
        self.pbar.setVisible(True); self.pbar.setValue(pct)
        if pct >= 100: QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}"); self.btn_run.setEnabled(True)

    def _on_finished(self, result):
        self._result = result; self.btn_run.setEnabled(True); self.btn_exp.setEnabled(True)
        impr = result.get('improvement', 0); d = result.get('d_um', 0)
        self.summary_lbl.setText(
            f"d = {d:.1f} μm  |  {impr:.1f}× improvement\n"
            f"RMS before: {result.get('rms_before',0)*100:.2f}%  "
            f"→ after: {result.get('rms_after',0)*100:.2f}%")
        self._log(f"\n✓  d={d:.2f}μm  improvement={impr:.1f}×\n{'='*40}")
        self._draw_analysis(); self._draw_result()
        self.tabs.setCurrentIndex(2)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_input(self):
        fig = self.input_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        ax  = fig.add_subplot(111); _ax(ax,"Input spectrum","Wavelength (nm)","Flux")
        if self._image is not None:
            spec = self._image if not self._is_2d else np.median(self._image, axis=0)
            ax.plot(self._wav_nm, spec, color=ACC, lw=1, alpha=0.85)
            ax.axvline(656.3, color=WRN, lw=0.8, ls='--', alpha=0.6, label='Hα')
            ax.legend(fontsize=7, facecolor=BG3, edgecolor=BOR, labelcolor=TXT)
        self.input_plot.redraw()

    def _draw_analysis(self):
        r = self._result
        if not r or 'fringe_extracted' not in r: return
        fig = self.analysis_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        axes = fig.subplots(2,2)
        wav  = self._wav_nm

        # Top-left: extracted fringe
        _ax(axes[0,0],"Extracted fringe","Wavelength (nm)","Relative amplitude")
        axes[0,0].plot(wav, r['fringe_extracted']*100, color=ACC, lw=1)
        axes[0,0].axhline(0, color=BOR, lw=0.5)
        if 'template' in r:
            axes[0,0].plot(wav, r['template']*100, color=WRN, lw=1.5, ls='--', label='Model')
            axes[0,0].legend(fontsize=7,facecolor=BG3,edgecolor=BOR,labelcolor=TXT)

        # Top-right: OPD power spectrum
        opr = r.get('opd_result',{})
        _ax(axes[0,1],"OPD power spectrum","OPD (μm)","Power")
        if 'freqs' in opr and len(opr['freqs'])>0:
            n_r=self.sp_n.value()
            opd_um=opr['freqs']/(2*n_r)*1e4   # cm → μm × 2n already factored
            # freqs is already in cm, convert to effective d in μm
            d_axis=opr['freqs']*1e4/(2*n_r)
            axes[0,1].plot(d_axis, opr['power'], color=SEC, lw=1)
            axes[0,1].axvline(opr['d_um'], color=OK, lw=1.5, ls='--',
                              label=f"d={opr['d_um']:.1f}μm")
            axes[0,1].legend(fontsize=7,facecolor=BG3,edgecolor=BOR,labelcolor=TXT)

        # Bottom-left: amplitude envelope
        _ax(axes[1,0],"Fringe amplitude vs wavelength","Wavelength (nm)","Amplitude (%)")
        if 'amp_envelope' in r:
            axes[1,0].plot(wav, r['amp_envelope']*100, color=OK, lw=1.5)
            axes[1,0].fill_between(wav, 0, r['amp_envelope']*100, color=OK, alpha=0.2)

        # Bottom-right: phase map (2D only)
        _ax(axes[1,1],"Phase map (spatial axis)","Row","Phase (rad)")
        if 'phase_all' in r:
            axes[1,1].plot(r['phase_all'], color=WRN, lw=1.5)
        else:
            if 'model' in r:
                axes[1,1].text(0.5,0.5,f"φ = {r['model']['phi_fit']:.3f} rad\n(1D: single phase)",
                               transform=axes[1,1].transAxes,ha='center',va='center',
                               color=WRN,fontsize=10)

        self.analysis_plot.redraw()

    def _draw_result(self):
        r = self._result; wav = self._wav_nm
        if not r: return
        fig = self.result_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        axes = fig.subplots(2,1)

        # Get original and corrected spectra
        if not self._is_2d:
            orig    = self._image
            corr    = r.get('corrected', orig)
        else:
            orig    = np.median(self._image, axis=0)
            corr_2d = r.get('corrected_2d', self._image)
            corr    = np.median(corr_2d, axis=0)

        _ax(axes[0],"Before vs After","Wavelength (nm)","Flux")
        axes[0].plot(wav, orig, color=ERR,  lw=1,   alpha=0.7, label="Before")
        axes[0].plot(wav, corr, color=OK,   lw=1.2, alpha=0.9, label="After")
        axes[0].legend(fontsize=7,facecolor=BG3,edgecolor=BOR,labelcolor=TXT)

        _ax(axes[1],"Residual fringes after correction","Wavelength (nm)","Residual (%)")
        cont = r.get('continuum', fit_continuum(wav, corr))
        res  = corr / np.maximum(cont, 1e-10) - 1.0
        axes[1].plot(wav, res*100, color=SEC, lw=1)
        axes[1].axhline(0, color=BOR, lw=0.5)
        axes[1].text(0.02,0.92,
            f"Improvement: {r.get('improvement',0):.1f}×  "
            f"d={r.get('d_um',0):.1f}μm",
            transform=axes[1].transAxes, color=OK, fontsize=9)

        self.result_plot.redraw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        r = self._result
        if not r: return
        path, _ = QFileDialog.getSaveFileName(
            self,"Export corrected spectrum","spectrum_fringe_corrected.fits",
            "FITS (*.fits);;All (*)")
        if not path: return
        try:
            key = 'corrected_2d' if self._is_2d else 'corrected'
            data_out = r[key].astype(np.float32)
            hdr = astropy_fits.Header()
            hdr['COMMENT'] = 'CCD fringe corrected by HYPERLOAD ccd_fringe_corrector.py'
            hdr['FRNG_D']  = (r.get('d_um',0), 'CCD thickness μm')
            hdr['FRNG_IMP']= (r.get('improvement',0), 'Fringe reduction factor')
            hdr['CRVAL1']  = float(self._wav_nm[0])
            hdr['CDELT1']  = float(np.diff(self._wav_nm).mean())
            hdr['CTYPE1']  = 'WAVE-WAV'
            astropy_fits.PrimaryHDU(data_out, header=hdr).writeto(path, overwrite=True)
            self._log(f"✓ Exported: {path}")
        except Exception as e:
            QMessageBox.critical(self,"Export error",str(e))

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FringeApp(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
