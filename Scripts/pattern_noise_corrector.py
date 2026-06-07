r"""
HYPERLOAD — Pattern Noise Corrector
=====================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Corrects sensor pattern noise artifacts common in modern BSI CMOS cameras
(ASI2600, ASI2400, QHY268, IMX455 etc.) and older CCDs at high gain.

Four correction modes — apply in sequence for best results:

  1. AMP GLOW / 2D GRADIENT
     Corner or edge illumination from readout electronics leaking
     into the sensor during long exposures. Modelled as a smooth 2D
     Gaussian gradient; iterative sigma-clipping prevents stars from
     biasing the gradient model.

  2. COLUMN BANDING
     Fixed per-column bias offset from column amplifiers reading
     at different operating points. Each column's median (sigma-clipped)
     is computed and subtracted independently.
     Ref: Regnault et al. 2009 (PASP) — column-by-column correction

  3. ROW STRIPING
     Horizontal banding from ADC or readout electronics. Same
     approach as column correction but along rows.

  4. PERIODIC / QUASI-PERIODIC STRIPE REMOVAL  (Münch 2009)
     Electronic interference (50/60 Hz pickup, switching power supply)
     creates periodic or quasi-periodic horizontal or vertical stripes.
     Algorithm: à-trous wavelet decomposition → per-scale Fourier
     suppression of stripe-frequency components → reconstruct.
     Suppresses specific frequency bands without blurring the image.
     Ref: Münch et al. 2009, Optics Express 17(10), 8567

Sigma-clipping throughout: stars and galaxies are excluded from all
background statistics, preventing source contamination of the correction.

This corrects structural (patterned) noise only — it does NOT touch
random shot noise or read noise (those are handled by stacking).

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, traceback
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log,"a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\npattern_noise_corrector.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.ndimage import gaussian_filter, median_filter

try:
    from astropy.io import fits as astropy_fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit, QProgressBar,
    QGroupBox, QSplitter, QMessageBox, QSizePolicy, QScrollArea,
    QFormLayout, QComboBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Pattern Noise Corrector"


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS  (pure functions, importable)
# ═══════════════════════════════════════════════════════════════════════════════

def _sigma_clip_mask(arr: np.ndarray, sigma: float = 3.5) -> np.ndarray:
    """Return boolean mask True where |arr - median| < sigma × MAD × 1.4826."""
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    return np.abs(arr - med) < sigma * max(mad, 1e-10)


def remove_amp_glow(image: np.ndarray,
                    sigma: float = 70.0,
                    clip_sigma: float = 2.5,
                    n_iter: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove smooth 2D amp glow / gradient by iterative Gaussian modelling.

    At each iteration: smooth → measure residuals → sigma-clip bright sources
    → re-smooth with sources filled by current model → repeat.
    Final correction = image - model.

    Parameters
    ----------
    sigma      : Gaussian smoothing scale (pixels). Should be >> star FWHM.
    clip_sigma : Sigma for source masking (lower = more aggressive masking).
    n_iter     : Convergence iterations.

    Returns
    -------
    corrected, model
    """
    img   = image.astype(float)
    model = gaussian_filter(img, sigma=sigma)
    for _ in range(n_iter):
        residual = img - model
        rms      = float(np.std(residual))
        filled   = np.where(np.abs(residual) < clip_sigma * rms, img, model)
        model    = gaussian_filter(filled, sigma=sigma)
    return img - model, model


def correct_columns(image: np.ndarray,
                    clip_sigma: float = 3.5,
                    smooth_sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Column-wise bias correction. Per-column background median with sigma-clipping.

    Excludes compact sources (stars) from column statistics so their
    column flux does not bias the correction.

    smooth_sigma : if > 0, smooth the correction profile across adjacent columns.
    Returns (corrected, correction_profile)
    """
    img  = image.astype(float)
    corr = np.zeros(img.shape[1])
    for c in range(img.shape[1]):
        col  = img[:, c]
        mask = _sigma_clip_mask(col, clip_sigma)
        if mask.sum() > 3:
            corr[c] = float(col[mask].mean())
    corr -= float(corr.mean())  # preserve global offset
    if smooth_sigma > 0:
        corr = gaussian_filter(corr, sigma=smooth_sigma)
    return img - corr[np.newaxis, :], corr


def correct_rows(image: np.ndarray,
                 clip_sigma: float = 3.5,
                 smooth_sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-wise bias correction. Same approach as correct_columns but for rows.
    """
    img  = image.astype(float)
    corr = np.zeros(img.shape[0])
    for r in range(img.shape[0]):
        row  = img[r, :]
        mask = _sigma_clip_mask(row, clip_sigma)
        if mask.sum() > 3:
            corr[r] = float(row[mask].mean())
    corr -= float(corr.mean())
    if smooth_sigma > 0:
        corr = gaussian_filter(corr, sigma=smooth_sigma)
    return img - corr[:, np.newaxis], corr


def munch_destripe(image: np.ndarray,
                   n_scales: int = 4,
                   clip_sigma: float = 3.0,
                   stripe_axis: int = 0) -> np.ndarray:
    """
    Wavelet-Fourier destriping (Münch et al. 2009, Optics Express 17, 8567).

    Suppresses periodic and quasi-periodic stripe artifacts in the image
    without blurring real structures at the same spatial scale.

    Algorithm per scale:
      1. Decompose with à-trous (undecimated) Gaussian wavelet.
      2. In each detail image, take 1-D FFT along the stripe axis.
      3. Identify stripe-dominant frequency components (power > clip_sigma × median).
      4. Zero those frequency components; IFFT back.
      5. Reconstruct.

    Parameters
    ----------
    stripe_axis : 0 = horizontal stripes (row banding, FFT along rows)
                  1 = vertical stripes (column banding, FFT along columns)
    n_scales    : wavelet decomposition depth (4 is usually sufficient)
    clip_sigma  : threshold for stripe frequency detection
    """
    img_f  = image.astype(float)
    N      = img_f.shape[stripe_axis]
    details= []
    approx = img_f.copy()

    # À-trous decomposition: smooth perpendicular to the FFT axis
    perp_axis = 1 - stripe_axis
    for s in range(n_scales):
        sig = [0, 0]
        sig[perp_axis] = 2**s
        smooth  = gaussian_filter(approx, sigma=sig)
        details.append(approx - smooth)
        approx  = smooth

    # Fourier stripe suppression at each scale
    filtered = []
    for detail in details:
        F     = np.fft.fft(detail, axis=stripe_axis)
        power = np.abs(F).mean(axis=1 - stripe_axis)  # marginal power spectrum
        med   = float(np.median(power[1:N//2]))
        # Identify stripe frequencies (skip DC=0 and Nyquist)
        stripe_freqs = np.where(power > clip_sigma * med)[0]
        stripe_freqs = stripe_freqs[(stripe_freqs > 0) & (stripe_freqs < N//2)]
        if stripe_axis == 0:
            F[stripe_freqs, :]          = 0
            F[N - stripe_freqs, :]      = 0
            F[0, :]                     = 0
        else:
            F[:, stripe_freqs]          = 0
            F[:, N - stripe_freqs]      = 0
            F[:, 0]                     = 0
        filtered.append(np.real(np.fft.ifft(F, axis=stripe_axis)))

    return approx + sum(filtered)


def correct_pattern_noise(image: np.ndarray,
                           do_amp_glow:   bool  = True,
                           do_columns:    bool  = True,
                           do_rows:       bool  = True,
                           do_destripe_h: bool  = True,
                           do_destripe_v: bool  = False,
                           amp_sigma:     float = 70.0,
                           amp_clip:      float = 2.5,
                           amp_iter:      int   = 5,
                           col_clip:      float = 3.5,
                           row_clip:      float = 3.5,
                           col_smooth:    float = 0.0,
                           row_smooth:    float = 0.0,
                           destripe_scales: int  = 4,
                           destripe_clip:   float = 3.0,
                           progress_cb = None) -> dict:
    """
    Full pattern noise correction pipeline.
    Apply in order: amp glow → columns → rows → Münch destripe.

    Returns dict with corrected image and all intermediate steps.
    """
    steps   = {}
    current = image.astype(float)
    n_steps = sum([do_amp_glow, do_columns, do_rows, do_destripe_h, do_destripe_v])
    step_n  = 0

    if do_amp_glow:
        if progress_cb: progress_cb(int(100*step_n/max(n_steps,1)), "Removing amp glow…")
        current, model = remove_amp_glow(current, amp_sigma, amp_clip, amp_iter)
        steps['amp_glow_model'] = model
        step_n += 1

    if do_columns:
        if progress_cb: progress_cb(int(100*step_n/max(n_steps,1)), "Correcting columns…")
        current, col_prof = correct_columns(current, col_clip, col_smooth)
        steps['col_profile'] = col_prof
        step_n += 1

    if do_rows:
        if progress_cb: progress_cb(int(100*step_n/max(n_steps,1)), "Correcting rows…")
        current, row_prof = correct_rows(current, row_clip, row_smooth)
        steps['row_profile'] = row_prof
        step_n += 1

    if do_destripe_h:
        if progress_cb: progress_cb(int(100*step_n/max(n_steps,1)), "Destriping (horizontal)…")
        current = munch_destripe(current, destripe_scales, destripe_clip, stripe_axis=0)
        step_n += 1

    if do_destripe_v:
        if progress_cb: progress_cb(int(100*step_n/max(n_steps,1)), "Destriping (vertical)…")
        current = munch_destripe(current, destripe_scales, destripe_clip, stripe_axis=1)
        step_n += 1

    # Quality metrics
    rms_before = float(np.std(image))
    rms_after  = float(np.std(current))
    steps.update({
        'corrected':  current,
        'rms_before': rms_before,
        'rms_after':  rms_after,
    })
    if progress_cb: progress_cb(100, f"Done.  σ before={rms_before:.1f}  after={rms_after:.1f}")
    return steps


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class PatternWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, image, params):
        super().__init__()
        self.image = image; self.params = params

    def run(self):
        try:
            p = self.params
            result = correct_pattern_noise(
                self.image,
                do_amp_glow    = p['do_amp'],
                do_columns     = p['do_cols'],
                do_rows        = p['do_rows'],
                do_destripe_h  = p['do_dsh'],
                do_destripe_v  = p['do_dsv'],
                amp_sigma      = p['amp_sigma'],
                amp_clip       = p['amp_clip'],
                amp_iter       = p['amp_iter'],
                col_clip       = p['col_clip'],
                row_clip       = p['row_clip'],
                col_smooth     = p['col_smooth'],
                row_smooth     = p['row_smooth'],
                destripe_scales= p['ds_scales'],
                destripe_clip  = p['ds_clip'],
                progress_cb    = lambda pct,msg: self.progress.emit(pct,msg))
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

BG="#1e2128";  BG2="#252930"; BG3="#2d3240"; BOR="#3a4055"
ACC="#4a9eff"; AC2="#2d6abf"; TXT="#dde3ee"; DIM="#7a8499"
OK="#4caf7d";  WRN="#e8a23a"; ERR="#cc4444"; SEC="#5ba3ff"

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
QCheckBox::indicator{{width:14px;height:14px;border:1px solid {BOR};
    background:{BG3};border-radius:2px;}}
QCheckBox::indicator:checked{{background:{ACC};}}
QSpinBox,QDoubleSpinBox,QComboBox{{background:{BG3};
    border:1px solid {BOR};border-radius:3px;padding:3px 6px;color:{TXT};}}
QTextEdit{{background:#12141a;border:1px solid {BOR};color:{OK};
    font-family:Consolas,monospace;font-size:10px;}}
QProgressBar{{background:{BG3};border:1px solid {BOR};
    border-radius:3px;height:8px;}}
QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {AC2},stop:1 {ACC});border-radius:2px;}}
QSplitter::handle{{background:{BOR};}}
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8,5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor=BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)

    def redraw(self): self.fig.tight_layout(pad=0.3); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#080c08')
    ax.set_title(title, color=SEC, fontsize=9, pad=3)
    ax.set_xlabel(xl, color=DIM, fontsize=8)
    ax.set_ylabel(yl, color=DIM, fontsize=8)
    ax.tick_params(colors=DIM, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(BOR)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class PatternApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1380, 900)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._image  = None
        self._result = None
        self._worker = self._wthread = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._controls())
        sp.addWidget(self._tabs())
        sp.setSizes([270, 1110])
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
        t = QLabel("Pattern Noise Corrector")
        t.setStyleSheet(f"color:{ACC};font-size:17px;font-weight:bold;padding:6px;")
        s = QLabel("Amp glow  ·  Column banding  ·  Row striping  ·  "
                   "Wavelet-Fourier destriping (Münch 2009)  ·  "
                   "BSI CMOS: ASI2600 / ASI2400 / QHY268 / IMX455")
        s.setStyleSheet(f"color:{DIM};font-size:9pt;padding:0 8px 4px;")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(4)

        # Load
        grp = QGroupBox("Image"); gl = QVBoxLayout(grp)
        self.file_lbl = QLabel("No file")
        self.file_lbl.setStyleSheet(f"color:{DIM};font-size:10px;"); self.file_lbl.setWordWrap(True)
        b1 = QPushButton("📂  Load FITS…"); b1.setObjectName("run"); b1.clicked.connect(self._load)
        b2 = QPushButton("⚗  Synthetic example"); b2.clicked.connect(self._example)
        gl.addWidget(self.file_lbl); gl.addWidget(b1); gl.addWidget(b2)
        lay.addWidget(grp)

        # Pipeline steps
        def section(title, enabled=True):
            g = QGroupBox(title); g.setCheckable(True); g.setChecked(enabled); return g

        # 1. Amp glow
        self.grp_amp = section("① Amp Glow / 2D Gradient", True)
        ga = QFormLayout(self.grp_amp)
        self.sp_amp_sig  = QDoubleSpinBox(); self.sp_amp_sig.setRange(20,200); self.sp_amp_sig.setValue(70)
        self.sp_amp_clip = QDoubleSpinBox(); self.sp_amp_clip.setRange(1,5);   self.sp_amp_clip.setValue(2.5); self.sp_amp_clip.setDecimals(1)
        self.sp_amp_iter = QSpinBox();       self.sp_amp_iter.setRange(2,12);  self.sp_amp_iter.setValue(5)
        ga.addRow("Smooth σ (px):", self.sp_amp_sig)
        ga.addRow("Clip σ:",        self.sp_amp_clip)
        ga.addRow("Iterations:",    self.sp_amp_iter)
        lay.addWidget(self.grp_amp)

        # 2. Columns
        self.grp_col = section("② Column Banding", True)
        gc = QFormLayout(self.grp_col)
        self.sp_col_clip   = QDoubleSpinBox(); self.sp_col_clip.setRange(1,6);  self.sp_col_clip.setValue(3.5); self.sp_col_clip.setDecimals(1)
        self.sp_col_smooth = QDoubleSpinBox(); self.sp_col_smooth.setRange(0,5);self.sp_col_smooth.setValue(0); self.sp_col_smooth.setDecimals(1)
        gc.addRow("Clip σ:",         self.sp_col_clip)
        gc.addRow("Profile smooth:", self.sp_col_smooth)
        lay.addWidget(self.grp_col)

        # 3. Rows
        self.grp_row = section("③ Row Striping", True)
        gr = QFormLayout(self.grp_row)
        self.sp_row_clip   = QDoubleSpinBox(); self.sp_row_clip.setRange(1,6);  self.sp_row_clip.setValue(3.5); self.sp_row_clip.setDecimals(1)
        self.sp_row_smooth = QDoubleSpinBox(); self.sp_row_smooth.setRange(0,5);self.sp_row_smooth.setValue(0); self.sp_row_smooth.setDecimals(1)
        gr.addRow("Clip σ:",         self.sp_row_clip)
        gr.addRow("Profile smooth:", self.sp_row_smooth)
        lay.addWidget(self.grp_row)

        # 4. Destripe
        self.grp_ds = section("④ Periodic Destripe (Münch)", True)
        gd = QFormLayout(self.grp_ds)
        self.sp_ds_scales = QSpinBox();       self.sp_ds_scales.setRange(2,8); self.sp_ds_scales.setValue(4)
        self.sp_ds_clip   = QDoubleSpinBox(); self.sp_ds_clip.setRange(1.5,8); self.sp_ds_clip.setValue(3.0); self.sp_ds_clip.setDecimals(1)
        self.cb_dsh = QCheckBox("Horizontal stripes"); self.cb_dsh.setChecked(True)
        self.cb_dsv = QCheckBox("Vertical stripes");   self.cb_dsv.setChecked(False)
        gd.addRow("Wavelet scales:", self.sp_ds_scales)
        gd.addRow("Detection σ:",    self.sp_ds_clip)
        gd.addRow(self.cb_dsh); gd.addRow(self.cb_dsv)
        lay.addWidget(self.grp_ds)

        # Run
        self.btn_run = QPushButton("▶  Correct")
        self.btn_run.setObjectName("run"); self.btn_run.setFixedHeight(36)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_exp = QPushButton("💾  Export corrected FITS…")
        self.btn_exp.setEnabled(False); self.btn_exp.clicked.connect(self._export)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_exp)
        lay.addStretch()
        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(f"color:{OK};font-size:10px;"); self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_input(), "🖼  Input")
        self.tabs.addTab(self._tab_steps(), "🔍  Steps")
        self.tabs.addTab(self._tab_result(),"✨  Result")
        self.tabs.addTab(self._tab_log(),   "📋  Log")
        return self.tabs

    def _tab_input(self):
        w=QWidget(); lay=QVBoxLayout(w)
        self.input_plot=MplCanvas(figsize=(9,6)); lay.addWidget(self.input_plot)
        return w

    def _tab_steps(self):
        w=QWidget(); lay=QVBoxLayout(w)
        self.steps_plot=MplCanvas(figsize=(9,6)); lay.addWidget(self.steps_plot)
        return w

    def _tab_result(self):
        w=QWidget(); lay=QVBoxLayout(w)
        self.result_plot=MplCanvas(figsize=(9,6)); lay.addWidget(self.result_plot)
        return w

    def _tab_log(self):
        w=QWidget(); lay=QVBoxLayout(w)
        self.log=QTextEdit(); self.log.setReadOnly(True); lay.addWidget(self.log)
        btn=QPushButton("Clear"); btn.clicked.connect(self.log.clear); lay.addWidget(btn)
        return w

    def _footer(self):
        foot=QWidget(); foot.setFixedHeight(22)
        foot.setStyleSheet(f"background:{BG};border-top:1px solid {BOR};")
        lay=QHBoxLayout(foot); lay.setContentsMargins(6,0,6,0)
        self.status_lbl=QLabel("Ready"); self.status_lbl.setStyleSheet(f"color:{DIM};font-size:10px;")
        self.pbar=QProgressBar(); self.pbar.setFixedSize(220,12); self.pbar.setVisible(False)
        lay.addWidget(self.status_lbl,1); lay.addWidget(self.pbar)
        return foot

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        path,_=QFileDialog.getOpenFileName(self,"Load FITS","","FITS (*.fits *.fit);;All (*)")
        if not path: return
        try:
            data=astropy_fits.getdata(path).astype(float)
            if data.ndim==3: data=data[0]
            self._set_image(data, Path(path).name)
        except Exception as e:
            QMessageBox.critical(self,"Load error",str(e))

    def _example(self):
        rng2=np.random.default_rng(42); N=512
        bg=np.zeros((N,N))+300.0
        yy,xx=np.mgrid[:N,:N]
        bg+=800*np.exp(-((xx-256)**2+(yy-256)**2)/5000)
        for _ in range(60):
            sx=rng2.integers(10,N-10); sy=rng2.integers(10,N-10)
            bg+=rng2.lognormal(5,1.5)*np.exp(-((xx-sx)**2+(yy-sy)**2)/2.0)
        col_bias=rng2.normal(0,18,N)
        row_bias=rng2.normal(0,10,N)
        amp_glow=500*np.exp(-((xx-(N-1))**2+(yy-(N-1))**2)/8000)
        hf=12*np.sin(2*np.pi*np.arange(N)/6.3)
        img=(bg+col_bias[np.newaxis,:]+row_bias[:,np.newaxis]
             +amp_glow+hf[:,np.newaxis]+rng2.normal(0,10,(N,N)))
        self._set_image(img.astype(float),"synthetic (ASI2600-like)")
        self._log("Synthetic: amp glow + col banding + row striping + 6.3px periodic stripes")

    def _set_image(self, data, label=""):
        self._image=data
        self.file_lbl.setText(f"{label}\n{data.shape[1]}×{data.shape[0]}px")
        self.file_lbl.setStyleSheet(f"color:{OK};font-size:10px;")
        self.btn_run.setEnabled(True)
        self._draw_input(); self._log(f"Loaded: {label}  {data.shape}")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _params(self):
        return dict(
            do_amp=self.grp_amp.isChecked(), do_cols=self.grp_col.isChecked(),
            do_rows=self.grp_row.isChecked(), do_dsh=self.grp_ds.isChecked() and self.cb_dsh.isChecked(),
            do_dsv=self.grp_ds.isChecked() and self.cb_dsv.isChecked(),
            amp_sigma=self.sp_amp_sig.value(), amp_clip=self.sp_amp_clip.value(),
            amp_iter=self.sp_amp_iter.value(), col_clip=self.sp_col_clip.value(),
            row_clip=self.sp_row_clip.value(), col_smooth=self.sp_col_smooth.value(),
            row_smooth=self.sp_row_smooth.value(), ds_scales=self.sp_ds_scales.value(),
            ds_clip=self.sp_ds_clip.value())

    def _run(self):
        if self._image is None: return
        self.btn_run.setEnabled(False); self.btn_exp.setEnabled(False)
        self._wthread=QThread(self)
        self._worker=PatternWorker(self._image, self._params())
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _on_progress(self,pct,msg):
        self.status_lbl.setText(msg); self.pbar.setVisible(True); self.pbar.setValue(pct)
        if pct>=100: QTimer.singleShot(3000,lambda:self.pbar.setVisible(False))
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_error(self,msg):
        self._log(f"\n❌  {msg}"); self.btn_run.setEnabled(True)

    def _on_finished(self, result):
        self._result=result; self.btn_run.setEnabled(True); self.btn_exp.setEnabled(True)
        rb=result['rms_before']; ra=result['rms_after']
        self.summary_lbl.setText(f"σ before: {rb:.1f}  →  after: {ra:.1f}\nReduction: {(1-ra/rb)*100:.1f}%")
        self._log(f"\n✓  σ {rb:.1f} → {ra:.1f}  ({(1-ra/rb)*100:.1f}% reduction)")
        self._draw_steps(); self._draw_result()
        self.tabs.setCurrentIndex(2)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _imshow(self, ax, img, title='', cmap='gray'):
        lo,hi=np.nanpercentile(img[np.isfinite(img)],[0.5,99.5])
        ax.imshow(img,origin='lower',cmap=cmap,vmin=lo,vmax=hi,
                  aspect='equal',interpolation='bicubic')
        _ax(ax,title); ax.set_xticks([]); ax.set_yticks([])

    def _draw_input(self):
        if self._image is None: return
        fig=self.input_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        axes=fig.subplots(1,2)
        self._imshow(axes[0], self._image, "Raw image")
        _ax(axes[1],"Pixel histogram","ADU","Count")
        axes[1].hist(self._image.ravel(), bins=200,
                     color=ACC, edgecolor='none', alpha=0.8, density=True)
        self.input_plot.redraw()

    def _draw_steps(self):
        r=self._result
        if not r: return
        fig=self.steps_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        axes=fig.subplots(2,3)
        ax=axes.flat

        self._imshow(ax[0], self._image, "Raw input")

        if 'amp_glow_model' in r:
            self._imshow(ax[1], r['amp_glow_model'], "Amp glow model", 'hot')
        else:
            ax[1].text(0.5,0.5,"(disabled)",ha='center',va='center',
                       transform=ax[1].transAxes,color=DIM); _ax(ax[1],"Amp glow")

        if 'col_profile' in r:
            _ax(ax[2],"Column correction profile","Column","ADU")
            ax[2].plot(r['col_profile'],color=ACC,lw=0.8)
            ax[2].axhline(0,color=BOR,lw=0.5)
        else:
            ax[2].text(0.5,0.5,"(disabled)",ha='center',va='center',
                       transform=ax[2].transAxes,color=DIM); _ax(ax[2],"Column profile")

        if 'row_profile' in r:
            _ax(ax[3],"Row correction profile","ADU","Row")
            ax[3].plot(r['row_profile'],np.arange(len(r['row_profile'])),color=WRN,lw=0.8)
            ax[3].axvline(0,color=BOR,lw=0.5)
        else:
            ax[3].text(0.5,0.5,"(disabled)",ha='center',va='center',
                       transform=ax[3].transAxes,color=DIM); _ax(ax[3],"Row profile")

        self._imshow(ax[4], r['corrected'], "Corrected")

        # Difference map
        diff=r['corrected']-self._image
        _ax(ax[5],"Correction map (removed noise)")
        vd=np.nanstd(diff)*3
        ax[5].imshow(diff,origin='lower',cmap='RdBu_r',vmin=-vd,vmax=vd,
                     aspect='equal',interpolation='nearest')
        ax[5].set_xticks([]); ax[5].set_yticks([])

        self.steps_plot.redraw()

    def _draw_result(self):
        r=self._result
        if not r: return
        fig=self.result_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        axes=fig.subplots(1,2)
        self._imshow(axes[0],self._image,"Raw  (before)")
        self._imshow(axes[1],r['corrected'],
                     f"Corrected  (σ {r['rms_before']:.1f}→{r['rms_after']:.1f}  "
                     f"{(1-r['rms_after']/r['rms_before'])*100:.0f}% reduction)")
        self.result_plot.redraw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        r=self._result
        if not r: return
        path,_=QFileDialog.getSaveFileName(self,"Export corrected FITS",
                                            "pattern_corrected.fits","FITS (*.fits);;All (*)")
        if not path: return
        try:
            hdr=astropy_fits.Header()
            hdr['COMMENT']='Pattern noise corrected by HYPERLOAD pattern_noise_corrector.py'
            hdr['PNC_RMSb']=(r['rms_before'],'RMS before correction')
            hdr['PNC_RMSa']=(r['rms_after'], 'RMS after correction')
            astropy_fits.PrimaryHDU(r['corrected'].astype(np.float32),header=hdr).writeto(
                path,overwrite=True)
            self._log(f"✓ Exported: {path}")
        except Exception as e:
            QMessageBox.critical(self,"Export error",str(e))

    def _log(self,msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app=QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win=PatternApp(); win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
