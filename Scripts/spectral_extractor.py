r"""
HYPERLOAD — Slitless Spectral Extractor
========================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Extracts a calibrated 1D spectrum from a 2D slitless spectroscopy image
(SA100, SA200, LHIRES, or any grating spectrograph).

Pipeline:
  1. Load 2D FITS spectrum
  2. Auto-detect spectral trace (peak of cross-dispersion profile)
  3. Trace the spectrum column-by-column (polynomial tilt correction)
  4. Horne 1986 optimal extraction — maximum SNR weighted sum
  5. Wavelength calibration — polynomial fit to known spectral lines
  6. Continuum normalisation
  7. Line identification and equivalent width measurement

Optimal extraction (Horne 1986):
  f(x) = Σ_y P(y)·D(x,y)/V(x,y)  /  Σ_y P(y)²/V(x,y)
  where P(y) = spatial profile, D(x,y) = data, V(x,y) = variance model.
  Achieves theoretical maximum SNR for sky/read-noise limited spectra.
  Improvement over simple sum: ~1.2–1.5× for typical conditions.

Supported gratings (pre-configured):
  SA100  — ~4.7 nm/px  (100 lines/mm, f/10, 1 arcsec/px)
  SA200  — ~2.3 nm/px  (200 lines/mm)
  DADOS  — ~0.7 nm/px  (900 lines/mm)
  Custom — user-defined dispersion

References:
  Horne 1986      — optimal extraction, PASP 98, 609
  Marsh 1989      — improved optimal extraction, PASP 101, 1032
  Lebrun et al.   — slitless spectroscopy techniques
  Pickles 1998    — spectral library for flux calibration

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, traceback, json, math
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log,"a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nspectral_extractor.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

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
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Slitless Spectral Extractor"

# Pre-configured gratings
GRATINGS = {
    "SA100  (~4.7 nm/px)":  4.70,
    "SA200  (~2.3 nm/px)":  2.30,
    "DADOS 900 (~0.7 nm/px)": 0.72,
    "Custom":               None,
}

# Known spectral lines for wavelength calibration (nm)
SPECTRAL_LINES = {
    "Hα 656.3":  656.28,
    "Hβ 486.1":  486.13,
    "Hγ 434.0":  434.05,
    "Hδ 410.2":  410.17,
    "HeI 587.6": 587.56,
    "HeI 667.8": 667.82,
    "HeII 468.6":468.57,
    "NaI 589.0": 589.00,
    "OI  777.4": 777.42,
    "CaII K 393.4": 393.37,
    "CaII H 396.8": 396.85,
    "MgI 518.4": 518.36,
    "O2  762.0": 762.00,   # telluric
    "O2  687.0": 687.00,   # telluric B band
}


# ═══════════════════════════════════════════════════════════════════════════════
#  TRACE FINDER
# ═══════════════════════════════════════════════════════════════════════════════

class TraceFinder:
    """
    Locates the spectral trace (star position in cross-dispersion direction)
    at each column and fits a polynomial to handle tilt.
    """

    def __init__(self, disp_axis: int = 1, win_half: int = 20,
                 poly_order: int = 2):
        self.disp_axis  = disp_axis   # 1 = spectrum along columns (x)
        self.win_half   = win_half    # half-width of extraction window
        self.poly_order = poly_order

    def find(self, image: np.ndarray) -> dict:
        """
        Returns dict:
          trace_y    : trace centre at each column (array, length = ncols)
          poly_coeff : polynomial coefficients (y vs x)
          win_lo/hi  : extraction window lower/upper bounds
          n_cols     : number of columns used
        """
        ny, nx = image.shape
        # Collapse to get rough trace centre
        collapsed = np.median(image, axis=1)
        rough_y   = int(np.argmax(gaussian_filter(collapsed, 3)))

        # Track trace column by column
        trace_y  = np.full(nx, float(rough_y))
        valid    = np.zeros(nx, bool)
        half_win = max(self.win_half, 5)

        for xi in range(nx):
            col = image[max(0, rough_y-half_win):
                         min(ny, rough_y+half_win+1), xi]
            if col.max() <= 0: continue
            # Weighted centroid
            yy  = np.arange(max(0, rough_y-half_win),
                             min(ny, rough_y+half_win+1))
            wts = np.maximum(col - np.percentile(col, 20), 0)
            if wts.sum() > 0:
                trace_y[xi] = (wts * yy).sum() / wts.sum()
                valid[xi]   = True

        # Polynomial fit to smooth the trace
        xv = np.where(valid)[0]
        if len(xv) < self.poly_order + 1:
            poly = np.polyfit(np.arange(nx), trace_y, 1)
        else:
            poly = np.polyfit(xv, trace_y[xv], self.poly_order)

        trace_smooth = np.polyval(poly, np.arange(nx))

        return {
            'trace_y':    trace_smooth,
            'poly_coeff': poly,
            'win_lo':     np.maximum(0,    (trace_smooth - self.win_half).astype(int)),
            'win_hi':     np.minimum(ny-1, (trace_smooth + self.win_half).astype(int)),
            'rough_y':    rough_y,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIMAL EXTRACTOR  (Horne 1986)
# ═══════════════════════════════════════════════════════════════════════════════

class OptimalExtractor:
    """
    Horne (1986) optimal extraction algorithm.

    The spatial profile P(y) is estimated from the collapsed spectrum
    (smoothed median of all columns) and used to weight the extraction:

      f(x) = Σ_y [P(y) · D(x,y) / V(x,y)] / Σ_y [P(y)² / V(x,y)]

    Variance model: V(x,y) = RON² + sky + |D(x,y)|
    (assumes Poisson statistics + Gaussian read noise)

    This achieves the theoretical maximum SNR per wavelength channel.
    """

    def __init__(self, ron: float = 10.0, sky_sigma: float = 5.0):
        self.ron       = ron
        self.sky_sigma = sky_sigma

    def extract(self, image: np.ndarray, trace: dict) -> dict:
        ny, nx = image.shape
        trace_y = trace['trace_y']

        # ── Background subtraction ─────────────────────────────────────────
        # Estimate sky from regions away from the trace
        sky_bands = []
        sky_off   = trace['win_hi'] - trace['win_lo'] + 20
        for xi in range(nx):
            y0_s = max(0, int(trace_y[xi]) - int(sky_off[xi]) - 10)
            y1_s = max(0, int(trace_y[xi]) - int(sky_off[xi]))
            y0_e = min(ny, int(trace_y[xi]) + int(sky_off[xi]))
            y1_e = min(ny, int(trace_y[xi]) + int(sky_off[xi]) + 10)
            sky_cols = np.concatenate([image[y0_s:y1_s, xi],
                                        image[y0_e:y1_e, xi]])
            sky_bands.append(np.median(sky_cols) if len(sky_cols) > 0 else 0.0)
        sky_arr = np.array(sky_bands)
        # Smooth sky
        sky_smooth = savgol_filter(sky_arr, min(51, nx//4*2+1), 3)

        # Sky-subtracted image
        sub = image - sky_smooth[np.newaxis, :]

        # ── Spatial profile (Horne step 5) ────────────────────────────────
        # Collapse: median of all columns (sky subtracted, trace-centred)
        hw = int(trace['win_hi'].mean() - trace['win_lo'].mean()) // 2
        hw = max(hw, 3)
        profile_stamps = []
        for xi in range(0, nx, max(1, nx//50)):
            yc = int(round(trace_y[xi]))
            y0 = max(0, yc - hw); y1 = min(ny, yc + hw + 1)
            col = sub[y0:y1, xi]
            if col.max() > 0:
                profile_stamps.append(col / (col.max() + 1e-10))
        if profile_stamps:
            min_len = min(len(c) for c in profile_stamps)
            P_1d    = np.median([c[:min_len] for c in profile_stamps], axis=0)
            P_1d    = np.maximum(P_1d, 0); P_1d /= P_1d.sum() + 1e-10
        else:
            P_1d = np.ones(2*hw+1) / (2*hw+1)

        # ── Optimal extraction ─────────────────────────────────────────────
        f_opt   = np.zeros(nx)
        f_err   = np.zeros(nx)
        f_simple= np.zeros(nx)

        for xi in range(nx):
            yc = int(round(trace_y[xi]))
            y0 = max(0, yc - hw); y1 = min(ny, yc + hw + 1)
            nl = y1 - y0
            if nl < 2: continue

            P = P_1d[:nl] if nl <= len(P_1d) else np.pad(P_1d, (0,nl-len(P_1d)))
            P = P / (P.sum() + 1e-10)   # renormalise for window size

            D   = sub[y0:y1, xi]
            V   = self.ron**2 + abs(sky_smooth[xi]) + np.maximum(D, 0)

            num = np.sum(P * D / V)
            den = np.sum(P**2 / V)
            if den > 0:
                f_opt[xi]   = num / den
                f_err[xi]   = math.sqrt(1.0 / den)
            f_simple[xi] = D.sum()

        return {
            'flux':        f_opt,
            'flux_err':    f_err,
            'flux_simple': f_simple,
            'sky':         sky_smooth,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  WAVELENGTH CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

class WavelengthCalibrator:
    """
    Polynomial wavelength solution from known line positions.

    Modes:
      'linear'  : λ(x) = a·x + b  (2 lines required)
      'quadratic': λ(x) = a·x² + b·x + c  (3+ lines recommended)
    """

    def __init__(self, poly_order: int = 1):
        self.poly_order = poly_order
        self.coeff      = None

    def fit(self, pixel_positions: list, wavelengths: list) -> dict:
        """
        pixel_positions : list of pixel x-values for known lines
        wavelengths     : corresponding wavelengths (nm)
        """
        if len(pixel_positions) < self.poly_order + 1:
            return {'success': False,
                    'error': f'Need ≥ {self.poly_order+1} calibration lines'}

        self.coeff = np.polyfit(pixel_positions, wavelengths, self.poly_order)
        wav_fit    = np.polyval(self.coeff, pixel_positions)
        residuals  = np.array(wavelengths) - wav_fit
        rms        = float(np.sqrt(np.mean(residuals**2)))

        return {
            'success':   True,
            'coeff':     self.coeff,
            'rms_nm':    rms,
            'residuals': residuals,
            'dispersion': float(self.coeff[-2]),   # nm/px (linear term)
        }

    def pixel_to_wav(self, pixels: np.ndarray) -> np.ndarray:
        if self.coeff is None: return pixels.astype(float)
        return np.polyval(self.coeff, pixels)

    def wav_to_pixel(self, wav: float) -> float:
        if self.coeff is None: return wav
        # Find root of poly(x) - wav
        from numpy.polynomial.polynomial import polyroots
        c = self.coeff.copy(); c[-1] -= wav
        roots = np.roots(c)
        real_roots = roots[np.abs(roots.imag) < 1e-6].real
        if len(real_roots) == 0: return 0.0
        return float(real_roots[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  SPECTRUM ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

class SpectrumAnalyser:
    """
    Line detection, identification and equivalent width measurement.
    """

    @staticmethod
    def find_lines(wav: np.ndarray, flux: np.ndarray,
                   flux_err: np.ndarray, min_sigma: float = 3.0) -> list:
        """
        Detect emission and absorption lines.
        Returns list of dicts with line properties.
        """
        # Normalise to local continuum
        cont = gaussian_filter(flux, 20)
        norm = flux / (cont + 1e-10)

        # Detect emission peaks
        peaks_em, _ = find_peaks(norm, height=1.0 + min_sigma * flux_err.mean() /
                                  (cont.mean() + 1e-10), distance=3)
        # Detect absorption troughs
        peaks_ab, _ = find_peaks(1 - norm, height=min_sigma * flux_err.mean() /
                                  (cont.mean() + 1e-10), distance=3)

        lines = []
        for px in peaks_em:
            ew = SpectrumAnalyser._equiv_width(wav, flux, cont, px, kind='em')
            lines.append({'wav': wav[px], 'type': 'emission',
                          'strength': norm[px] - 1, 'ew_nm': ew})
        for px in peaks_ab:
            ew = SpectrumAnalyser._equiv_width(wav, flux, cont, px, kind='ab')
            lines.append({'wav': wav[px], 'type': 'absorption',
                          'strength': 1 - norm[px], 'ew_nm': ew})

        return sorted(lines, key=lambda x: -abs(x['ew_nm']))

    @staticmethod
    def _equiv_width(wav, flux, cont, peak_px, kind='em',
                     half_width_px: int = 10) -> float:
        """Equivalent width in nm."""
        lo = max(0, peak_px - half_width_px)
        hi = min(len(wav)-1, peak_px + half_width_px)
        if hi <= lo: return 0.0
        W = wav[lo:hi+1]; F = flux[lo:hi+1]; C = cont[lo:hi+1]
        dw = np.gradient(W)
        ew = np.sum((1 - F / (C + 1e-10)) * dw)
        return float(ew)

    @staticmethod
    def identify_line(wav_nm: float, catalog: dict,
                      tol_nm: float = 3.0) -> str:
        """Match measured wavelength to catalog line."""
        best = min(catalog.items(), key=lambda kv: abs(kv[1] - wav_nm))
        if abs(best[1] - wav_nm) < tol_nm:
            return f"{best[0]} ({best[1]:.2f} nm)"
        return f"Unknown ({wav_nm:.2f} nm)"


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def luminance(data):
    if data.ndim == 2: return data.astype(float)
    if data.ndim == 3 and data.shape[0] <= 4: return data.mean(0).astype(float)
    return data.astype(float)


class SpectralWorker(QObject):
    progress  = pyqtSignal(int, str)
    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, path, params, cal_points):
        super().__init__()
        self.path = path; self.params = params
        self.cal_points = cal_points   # list of (pixel, wavelength_nm) tuples

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p = self.params

        self.progress.emit(5, "Loading 2D spectrum…")
        hdu  = fits.open(self.path)[0]
        img  = luminance(hdu.data)
        hdr  = hdu.header

        # Flip if needed so spectrum runs left-right
        if p.get('flip_x', False): img = img[:, ::-1]
        if p.get('flip_y', False): img = img[::-1, :]
        if p.get('transpose', False): img = img.T

        ny, nx = img.shape
        self.progress.emit(10, f"Image: {nx}×{ny} px")

        # ── Trace ─────────────────────────────────────────────────────────
        self.progress.emit(20, "Finding spectral trace…")
        tf = TraceFinder(win_half=p.get('win_half', 15),
                         poly_order=p.get('trace_poly', 2))
        trace = tf.find(img)
        self.progress.emit(30, f"Trace centre ≈ y={trace['rough_y']}")

        # ── Extract ───────────────────────────────────────────────────────
        self.progress.emit(40, "Optimal extraction (Horne 1986)…")
        oe = OptimalExtractor(ron=p.get('ron', 10.0))
        spec = oe.extract(img, trace)
        self.progress.emit(60, "Extraction complete")

        # ── Wavelength calibration ────────────────────────────────────────
        self.progress.emit(70, "Wavelength calibration…")
        wc = WavelengthCalibrator(poly_order=p.get('cal_poly', 1))
        pixels = np.arange(nx)

        if self.cal_points and len(self.cal_points) >= 2:
            cal_pix = [c[0] for c in self.cal_points]
            cal_wav = [c[1] for c in self.cal_points]
            cal_result = wc.fit(cal_pix, cal_wav)
            wav = wc.pixel_to_wav(pixels)
            self.progress.emit(75, f"Calibration RMS: {cal_result['rms_nm']:.4f} nm")
        else:
            # Fall back to grating dispersion + start wavelength
            disp = p.get('dispersion', 4.7)
            wav0 = p.get('wav0', 380.0)
            wav  = wav0 + pixels * disp
            cal_result = {'success': False, 'rms_nm': 0, 'coeff': [disp, wav0]}

        # ── Continuum normalisation ───────────────────────────────────────
        self.progress.emit(80, "Normalising continuum…")
        continuum = gaussian_filter(spec['flux'], 30)
        norm_flux = spec['flux'] / (continuum + 1e-10)

        # ── Line detection ────────────────────────────────────────────────
        self.progress.emit(90, "Detecting spectral lines…")
        lines = SpectrumAnalyser.find_lines(
            wav, spec['flux'], spec['flux_err'],
            min_sigma=p.get('line_sigma', 3.0))

        # Identify lines
        for line in lines:
            line['id'] = SpectrumAnalyser.identify_line(
                line['wav'], SPECTRAL_LINES, tol_nm=p.get('id_tol_nm', 5.0))

        self.progress.emit(100, f"Done!  {len(lines)} lines detected")

        self.finished.emit({
            'img':        img,
            'trace':      trace,
            'flux':       spec['flux'],
            'flux_err':   spec['flux_err'],
            'flux_simple':spec['flux_simple'],
            'sky':        spec['sky'],
            'wav':        wav,
            'continuum':  continuum,
            'norm_flux':  norm_flux,
            'cal_result': cal_result,
            'cal_points': self.cal_points,
            'lines':      lines,
            'nx':         nx, 'ny': ny,
            'params':     p,
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
    def __init__(self, parent=None, figsize=(8,4)):
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


class SpectralApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1380, 900)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))
        self._path   = None
        self._result = None
        self._worker = self._wthread = None
        self._cal_points = []   # [(pixel, wavelength_nm), ...]
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
        self.status = StatusFooter(); rl.addWidget(self.status)

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
        t = QLabel("Slitless Spectral Extractor"); t.setObjectName("title_lbl")
        s = QLabel("Horne 1986 optimal extraction · Polynomial wavelength calibration · "
                   "Continuum normalisation · Line identification & equivalent width")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD").setStyleSheet("color:#30305a;") or
                      QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _make_controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)

        # ── Input ──────────────────────────────────────────────────────────
        grp = QGroupBox("2D Spectrum FITS")
        gl  = QVBoxLayout(grp)
        self.file_lbl = QLabel("No file loaded")
        self.file_lbl.setStyleSheet("color:#484868;font-size:10px;")
        btn = QPushButton("📂  Open FITS…"); btn.clicked.connect(self._load)
        gl.addWidget(self.file_lbl); gl.addWidget(btn)
        lay.addWidget(grp)

        # ── Orientation ────────────────────────────────────────────────────
        grp0 = QGroupBox("Image Orientation")
        g0   = QVBoxLayout(grp0)
        self.cb_flip_x = QCheckBox("Flip horizontally")
        self.cb_flip_y = QCheckBox("Flip vertically")
        self.cb_transp = QCheckBox("Transpose (spectrum along rows)")
        g0.addWidget(self.cb_flip_x); g0.addWidget(self.cb_flip_y)
        g0.addWidget(self.cb_transp)
        lay.addWidget(grp0)

        # ── Grating ───────────────────────────────────────────────────────
        grp2 = QGroupBox("Grating / Dispersion")
        g2   = QGridLayout(grp2)
        g2.addWidget(QLabel("Grating:"), 0, 0)
        self.cmb_grating = QComboBox()
        self.cmb_grating.addItems(list(GRATINGS.keys()))
        self.cmb_grating.currentIndexChanged.connect(self._on_grating)
        g2.addWidget(self.cmb_grating, 0, 1)
        g2.addWidget(QLabel("Dispersion (nm/px):"), 1, 0)
        self.sp_disp = QDoubleSpinBox(); self.sp_disp.setRange(0.01,50); self.sp_disp.setValue(4.7); self.sp_disp.setDecimals(3)
        g2.addWidget(self.sp_disp, 1, 1)
        g2.addWidget(QLabel("Start wavelength (nm):"), 2, 0)
        self.sp_wav0 = QDoubleSpinBox(); self.sp_wav0.setRange(100,3000); self.sp_wav0.setValue(380.0); self.sp_wav0.setDecimals(1)
        g2.addWidget(self.sp_wav0, 2, 1)
        lay.addWidget(grp2)

        # ── Extraction ────────────────────────────────────────────────────
        grp3 = QGroupBox("Extraction")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("Half-window (px):"), 0, 0)
        self.sp_win = QSpinBox(); self.sp_win.setRange(3,100); self.sp_win.setValue(15)
        g3.addWidget(self.sp_win, 0, 1)
        g3.addWidget(QLabel("Read noise (e⁻):"), 1, 0)
        self.sp_ron = QDoubleSpinBox(); self.sp_ron.setRange(1,100); self.sp_ron.setValue(10.0)
        g3.addWidget(self.sp_ron, 1, 1)
        lay.addWidget(grp3)

        # ── Calibration points ─────────────────────────────────────────────
        grp4 = QGroupBox("Wavelength Calibration")
        g4   = QVBoxLayout(grp4)
        g4.addWidget(QLabel("Add calibration point:"))
        h = QHBoxLayout()
        h.addWidget(QLabel("px:"))
        self.le_px = QLineEdit("0"); self.le_px.setFixedWidth(60)
        h.addWidget(self.le_px)
        h.addWidget(QLabel("nm:"))
        self.cmb_line = QComboBox()
        self.cmb_line.addItems(list(SPECTRAL_LINES.keys()))
        h.addWidget(self.cmb_line)
        g4.addLayout(h)
        b_add = QPushButton("➕ Add"); b_add.clicked.connect(self._add_cal)
        b_clr = QPushButton("🗑 Clear"); b_clr.clicked.connect(self._clear_cal)
        bh = QHBoxLayout(); bh.addWidget(b_add); bh.addWidget(b_clr)
        g4.addLayout(bh)
        self.cal_lbl = QLabel("0 calibration points")
        self.cal_lbl.setStyleSheet("color:#585878;font-size:10px;")
        g4.addWidget(self.cal_lbl)
        lay.addWidget(grp4)

        # ── Output ────────────────────────────────────────────────────────
        grp5 = QGroupBox("Output")
        g5   = QVBoxLayout(grp5)
        self.btn_save_fits = QPushButton("💾  Save Spectrum (FITS)…")
        self.btn_save_fits.setEnabled(False); self.btn_save_fits.clicked.connect(self._save_fits)
        self.btn_save_csv  = QPushButton("💾  Save Spectrum (CSV)…")
        self.btn_save_csv.setEnabled(False);  self.btn_save_csv.clicked.connect(self._save_csv)
        g5.addWidget(self.btn_save_fits); g5.addWidget(self.btn_save_csv)
        lay.addWidget(grp5)

        self.btn_run = QPushButton("▶  Extract Spectrum")
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
        self.tabs.addTab(self._tab_2d(),    "🖼  2D Spectrum")
        self.tabs.addTab(self._tab_1d(),    "📊  1D Spectrum")
        self.tabs.addTab(self._tab_norm(),  "📈  Normalised")
        self.tabs.addTab(self._tab_lines(), "🎯  Line Catalog")
        self.tabs.addTab(self._tab_log(),   "📋  Log")
        return self.tabs

    def _tab_2d(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.plot_2d = MplCanvas(figsize=(9,4)); lay.addWidget(self.plot_2d)
        return w

    def _tab_1d(self):
        w = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Show:"))
        self.cmb_show = QComboBox(); self.cmb_show.addItems(["Optimal","Simple","Sky"])
        self.cmb_show.currentIndexChanged.connect(self._refresh_1d)
        ctrl.addWidget(self.cmb_show); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.plot_1d = MplCanvas(figsize=(9,5)); lay.addWidget(self.plot_1d)
        return w

    def _tab_norm(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.plot_norm = MplCanvas(figsize=(9,5)); lay.addWidget(self.plot_norm)
        return w

    def _tab_lines(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.line_table = QTableWidget(0, 5)
        self.line_table.setHorizontalHeaderLabels(
            ["Wavelength (nm)", "Type", "Identified as", "Strength", "EW (nm)"])
        self.line_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.line_table.setAlternatingRowColors(True)
        lay.addWidget(self.line_table)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    # ── Grating selection ─────────────────────────────────────────────────────

    def _on_grating(self, idx):
        name = list(GRATINGS.keys())[idx]
        val  = GRATINGS[name]
        if val is not None:
            self.sp_disp.setValue(val)
            self.sp_disp.setEnabled(False)
        else:
            self.sp_disp.setEnabled(True)

    # ── Calibration points ────────────────────────────────────────────────────

    def _add_cal(self):
        try:
            px  = float(self.le_px.text())
            wav = SPECTRAL_LINES[self.cmb_line.currentText()]
            self._cal_points.append((px, wav))
            self.cal_lbl.setText(f"{len(self._cal_points)} calibration points: " +
                                  ", ".join(f"{p:.0f}px→{w:.1f}nm"
                                            for p,w in self._cal_points[-3:]))
            self._log(f"Cal: {px:.0f}px → {wav:.2f}nm ({self.cmb_line.currentText()})")
        except Exception as e:
            QMessageBox.warning(self,"Error",str(e))

    def _clear_cal(self):
        self._cal_points.clear(); self.cal_lbl.setText("0 calibration points")

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self,"Open 2D Spectrum FITS","","FITS (*.fits *.fit *.fts);;All (*)")
        if not path: return
        self._path = path
        self.file_lbl.setText(Path(path).name)
        self.file_lbl.setStyleSheet("color:#70c070;font-size:10px;")
        self.btn_run.setEnabled(True)
        self._log(f"Loaded: {path}")

    def _save_fits(self):
        if not self._result: return
        path, _ = QFileDialog.getSaveFileName(
            self,"Save Spectrum","","FITS (*.fits)")
        if not path: return
        r = self._result
        hdr = fits.Header()
        hdr['CTYPE1']='WAVELENGTH'; hdr['CRPIX1']=1
        hdr['CDELT1']=float(np.mean(np.diff(r['wav']))); hdr['CRVAL1']=float(r['wav'][0])
        hdr['CUNIT1']='nm'; hdr['HISTORY']=f'HYPERLOAD SpectralExtractor v{VERSION}'
        data = np.vstack([r['wav'],r['flux'],r['flux_err']]).astype(np.float32)
        fits.PrimaryHDU(data, header=hdr).writeto(path, overwrite=True)
        self._log(f"✓ Saved FITS: {path}")

    def _save_csv(self):
        if not self._result: return
        path, _ = QFileDialog.getSaveFileName(
            self,"Save Spectrum CSV","","CSV (*.csv)")
        if not path: return
        r = self._result
        with open(path,'w') as f:
            f.write("wavelength_nm,flux,flux_err,normalised_flux\n")
            for w,fl,fe,nf in zip(r['wav'],r['flux'],r['flux_err'],r['norm_flux']):
                f.write(f"{w:.4f},{fl:.4f},{fe:.4f},{nf:.6f}\n")
        self._log(f"✓ Saved CSV: {path}")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if not self._path: return
        params = {
            'dispersion':  self.sp_disp.value(),
            'wav0':        self.sp_wav0.value(),
            'win_half':    self.sp_win.value(),
            'ron':         self.sp_ron.value(),
            'flip_x':      self.cb_flip_x.isChecked(),
            'flip_y':      self.cb_flip_y.isChecked(),
            'transpose':   self.cb_transp.isChecked(),
            'cal_poly':    1 if len(self._cal_points) < 3 else 2,
            'line_sigma':  3.0,
            'id_tol_nm':   5.0,
        }
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_save_fits.setEnabled(False); self.btn_save_csv.setEnabled(False)

        self._wthread = QThread(self)
        self._worker  = SpectralWorker(self._path, params, list(self._cal_points))
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
        self.btn_save_fits.setEnabled(True); self.btn_save_csv.setEnabled(True)

        r = result
        self._log(f"\n{'='*55}")
        self._log(f"✓  Spectral extraction complete")
        self._log(f"   Wavelength range: {r['wav'].min():.1f}–{r['wav'].max():.1f} nm")
        self._log(f"   Resolution:       {np.mean(np.diff(r['wav'])):.3f} nm/px")
        cr = r['cal_result']
        if cr.get('success'):
            self._log(f"   Cal RMS:          {cr['rms_nm']:.4f} nm")
        self._log(f"   Lines detected:   {len(r['lines'])}")
        for ln in r['lines'][:5]:
            self._log(f"     {ln['wav']:.2f}nm  {ln['type'][:4]}  {ln['id']}  EW={ln['ew_nm']:.3f}nm")
        self._log(f"{'='*55}\n")

        snr = r['flux'].max() / (r['flux_err'].mean() + 1e-10)
        self.summary_lbl.setText(
            f"{r['wav'].min():.0f}–{r['wav'].max():.0f}nm  |  "
            f"SNR≈{snr:.0f}  |  {len(r['lines'])} lines")

        self._draw_2d(); self._refresh_1d(); self._draw_norm()
        self._fill_lines(); self.tabs.setCurrentIndex(1)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_2d(self):
        if not self._result: return
        r = self._result
        fig = self.plot_2d.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "2D Spectrum", "Column (px)", "Row (px)")
        lo,hi = np.percentile(r['img'],[1,99])
        ax.imshow(r['img'], origin='lower', cmap='inferno',
                  vmin=lo, vmax=hi, aspect='auto', interpolation='nearest')
        # Overlay trace
        nx = r['nx']
        ax.plot(np.arange(nx), r['trace']['trace_y'], c='#60ffff', lw=1, ls='--', alpha=0.7)
        ax.plot(np.arange(nx), r['trace']['win_lo'], c='#ff6040', lw=0.8, ls=':', alpha=0.6)
        ax.plot(np.arange(nx), r['trace']['win_hi'], c='#ff6040', lw=0.8, ls=':', alpha=0.6)
        self.plot_2d.redraw()

    def _refresh_1d(self):
        if not self._result: return
        r = self._result; idx = self.cmb_show.currentIndex()
        fig = self.plot_1d.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "Extracted 1D Spectrum", "Wavelength (nm)", "Flux (ADU)")
        if idx == 0:
            flux = r['flux']; err = r['flux_err']
            ax.plot(r['wav'], flux, c='#5070ff', lw=1)
            ax.fill_between(r['wav'], flux-err, flux+err, alpha=0.2, color='#5070ff')
            ax.plot(r['wav'], r['continuum'], c='#ff8030', lw=1, ls='--', alpha=0.7, label='Continuum')
        elif idx == 1:
            ax.plot(r['wav'], r['flux_simple'], c='#70b070', lw=1, label='Simple sum')
        else:
            ax.plot(r['wav'], r['sky'], c='#a060a0', lw=1, label='Sky')

        # Mark detected lines
        for ln in r['lines']:
            c = '#ff6060' if ln['type']=='absorption' else '#60ff60'
            ax.axvline(ln['wav'], color=c, lw=0.8, alpha=0.5)

        ax.legend(fontsize=7, facecolor='#1a1a30', edgecolor='#3030a0', labelcolor='white')
        self.plot_1d.redraw()

    def _draw_norm(self):
        if not self._result: return
        r = self._result
        fig = self.plot_norm.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "Continuum-normalised Spectrum", "Wavelength (nm)", "Normalised Flux")
        ax.plot(r['wav'], r['norm_flux'], c='#5070ff', lw=1)
        ax.axhline(1.0, c='#404070', lw=0.8, ls='--')
        for ln in r['lines']:
            c = '#ff6060' if ln['type']=='absorption' else '#60ff60'
            ax.axvline(ln['wav'], color=c, lw=0.8, alpha=0.6,
                       label=ln['id'].split('(')[0].strip() if abs(ln['ew_nm'])>0.05 else '')
        handles = [h for h in ax.get_legend_handles_labels()[0] if h.get_label()]
        if handles:
            ax.legend(fontsize=6, facecolor='#1a1a30', edgecolor='#3030a0',
                      labelcolor='white', ncol=3)
        self.plot_norm.redraw()

    def _fill_lines(self):
        if not self._result: return
        lines = self._result['lines']
        self.line_table.setRowCount(len(lines))
        for row, ln in enumerate(lines):
            c = '#ff6060' if ln['type']=='absorption' else '#80ff80'
            vals = [f"{ln['wav']:.3f}", ln['type'], ln['id'],
                    f"{ln['strength']:.3f}", f"{ln['ew_nm']:.4f}"]
            for col,v in enumerate(vals):
                item = QTableWidgetItem(v)
                if col in (0,2): item.setForeground(
                    __import__('PyQt6.QtGui',fromlist=['QColor']).QColor(c))
                self.line_table.setItem(row,col,item)
        self.line_table.resizeColumnsToContents()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SpectralApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
