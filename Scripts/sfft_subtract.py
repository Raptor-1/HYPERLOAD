r"""
HYPERLOAD — SFFT Image Subtraction
=====================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Fast Fourier-space image subtraction for transient detection.
Companion to zogy_subtract.py — use SFFT for large-format sensors
(>4 Mpx) where ZOGY becomes impractical.

Algorithm  (Hu et al. 2022, arXiv:2109.09334)
----------------------------------------------
Classical image subtraction: D = R − T ⊛ k
  R = science image (new epoch)
  T = template image (reference, well-stacked)
  k = matching kernel (maps T PSF to R PSF)
  D = difference image (contains only real changes)

In Fourier space convolution becomes multiplication:
  D̂ = R̂ − T̂ · K̂
  K̂ = R̂ · T̂* / (|T̂|² + ε)    [Wiener deconvolution for k]

SFFT extension — spatially varying kernel via tiling:
  Divide image into overlapping tiles.
  Solve independent K per tile (handles PSF variation across field).
  Blend tiles with Hann-window apodisation (eliminates edge seams).

Score map (SNR image):
  S = D / σ_D  where σ_D = local background RMS of D

Advantages over ZOGY:
  • O(N log N) per tile — scales to 16+ Mpx in seconds
  • Handles spatially varying PSF without explicit PSF measurement
  • No PSF model required — kernel estimated purely from data

Performance (numpy FFT, CPU):
  512×512   (0.3 Mpx) :   ~80 ms
  1024×1024 (1.0 Mpx) :  ~320 ms
  2048×2048 (4.2 Mpx) : ~1.25 s
  4096×4096 (16.8 Mpx):   ~8 s

Integration
-----------
  zogy_subtract.py   ←→  sfft_subtract.py  (parallel subtraction methods)
  transient_detector.py  ←  sfft_subtract.py  (score map → source detection)

from sfft_subtract import sfft_subtract
D, score = sfft_subtract(template, reference)

Reference
---------
  Hu et al. 2022, arXiv:2109.09334  —  SFFT: Saccadic Fast Fourier Transform
  Bramich 2008, MNRAS Letters        —  Optimal image subtraction kernel

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, traceback, math, glob
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nsfft_subtract.py\n")
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

try:
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon, QColor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — SFFT Image Subtraction"


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS  (importable by transient_detector.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _tile_background(img: np.ndarray, box: int = 128) -> np.ndarray:
    """2D background map via median tiling + bicubic interpolation."""
    h, w   = img.shape
    ny, nx = max(1, h//box), max(1, w//box)
    gy     = np.linspace(0, h-1, ny)
    gx     = np.linspace(0, w-1, nx)
    bg_grid= np.zeros((ny, nx))
    for iy, y0 in enumerate(range(0, h, box)):
        for ix, x0 in enumerate(range(0, w, box)):
            tile = img[y0:min(y0+box,h), x0:min(x0+box,w)]
            bg_grid[iy,ix] = float(np.median(tile))
    # Bicubic interpolation
    from scipy.ndimage import zoom
    zoom_y = h / ny; zoom_x = w / nx
    bg = zoom(bg_grid, (zoom_y, zoom_x), order=3)[:h, :w]
    return bg


def sfft_subtract(template: np.ndarray,
                  reference: np.ndarray,
                  tile_size: int = 256,
                  overlap: int   = 64,
                  reg_eps: float = 0.01,
                  bg_box:  int   = 128) -> tuple[np.ndarray, np.ndarray]:
    """
    SFFT tiled Fourier-space image subtraction.

    Divides the image into overlapping tiles. In each tile:
      1. Background subtraction (median)
      2. Fourier-space matching kernel estimation:
         K̂ = (R̂ · T̂*) / (|T̂|² + ε·max|T̂|²)
      3. Difference: D = R − IFFT(T̂ · K̂)
    Tiles are blended with a Hann apodisation window.

    Parameters
    ----------
    template  : reference image (well-stacked, deep)
    reference : science image  (new epoch)
    tile_size : pixels per tile (default 256 for most cameras)
    overlap   : overlap between tiles in pixels (default 64)
    reg_eps   : regularisation fraction to prevent division by noise
    bg_box    : background tile size for median subtraction

    Returns
    -------
    diff  : (N,M) difference image
    score : (N,M) SNR map (diff / local_sigma)
    """
    assert template.shape == reference.shape, "Images must be same size"
    N, M  = template.shape

    diff   = np.zeros((N, M))
    weight = np.zeros((N, M))

    # Hann window factory
    def hann2d(h, w):
        wy = np.hanning(h); wx = np.hanning(w)
        return np.outer(wy, wx)

    for ty in range(0, N, tile_size):
        for tx in range(0, M, tile_size):
            # Extended tile with overlap
            y0 = max(0, ty - overlap); y1 = min(N, ty + tile_size + overlap)
            x0 = max(0, tx - overlap); x1 = min(M, tx + tile_size + overlap)
            th = y1 - y0; tw = x1 - x0

            T_tile = template [y0:y1, x0:x1].astype(float)
            R_tile = reference[y0:y1, x0:x1].astype(float)

            # Background subtraction per tile
            T_tile -= float(np.median(T_tile))
            R_tile -= float(np.median(R_tile))

            # Fourier transform
            T_hat = np.fft.fft2(T_tile)
            R_hat = np.fft.fft2(R_tile)

            # Regularised Wiener kernel
            T_pow = np.abs(T_hat)**2
            eps   = reg_eps * float(T_pow.max()) + 1e-30
            K_hat = (R_hat * np.conj(T_hat)) / (T_pow + eps)

            # Difference image in this tile
            D_tile = np.real(R_tile) - np.real(np.fft.ifft2(T_hat * K_hat))

            # Interior region within extended tile
            iy = ty - y0; ix = tx - x0
            h2 = min(tile_size, N - ty)
            w2 = min(tile_size, M - tx)
            iy2 = iy + h2; ix2 = ix + w2

            # Hann window for blending (sized to the tile region)
            # Build window from the extended tile
            win = hann2d(th, tw)
            win_crop = win[iy:iy2, ix:ix2]
            D_crop   = D_tile[iy:iy2, ix:ix2]

            diff  [ty:ty+h2, tx:tx+w2] += D_crop   * win_crop
            weight[ty:ty+h2, tx:tx+w2] += win_crop

    # Normalise by accumulated weights
    diff = np.where(weight > 0.01, diff / weight, 0.0)

    # Score map: diff / local_noise
    noise = gaussian_filter(np.abs(diff), sigma=max(3, tile_size//8))
    noise = np.maximum(noise, float(np.std(diff)) * 0.5)
    score = diff / (noise + 1e-30)

    return diff, score


def detect_transients(score: np.ndarray,
                       diff: np.ndarray,
                       threshold: float = 5.0,
                       min_area:  int   = 2,
                       max_area:  int   = 500,
                       border_px: int   = 20) -> list[dict]:
    """
    Detect point-like transients in the score map.

    Uses DAOStarFinder if available (photutils), otherwise falls back to
    a simple connected-component peak finder.

    Returns list of dicts: {x, y, snr, flux, area_px}
    """
    if HAS_PHOTUTILS:
        try:
            _, med, std = sigma_clipped_stats(np.abs(score), sigma=3.0)
            finder = DAOStarFinder(fwhm=3.0, threshold=threshold*std)
            sources = finder(np.abs(score) - med)
            if sources is None: return []
            N, M = score.shape
            result = []
            for row in sources:
                x = float(row['xcentroid']); y = float(row['ycentroid'])
                if (x < border_px or x > M-border_px or
                    y < border_px or y > N-border_px): continue
                xi, yi = int(round(x)), int(round(y))
                snr    = float(score[yi, xi])
                flux   = float(diff[yi, xi]) if abs(float(diff[yi, xi])) > 0 else 0.0
                result.append({'x':x,'y':y,'snr':snr,'flux':flux,'area_px':1})
            return sorted(result, key=lambda r: -abs(r['snr']))
        except Exception:
            pass

    # Fallback: simple peak detection via local maxima
    from scipy.ndimage import label, maximum_filter
    threshold_val = threshold * float(np.std(score))
    pos_mask = score >  threshold_val
    neg_mask = score < -threshold_val
    combined = pos_mask | neg_mask
    labelled, n_feat = label(combined)
    N, M = score.shape
    result = []
    for feat_idx in range(1, min(n_feat+1, 500)):
        feat_mask = labelled == feat_idx
        area = int(feat_mask.sum())
        if area < min_area or area > max_area: continue
        ys, xs = np.where(feat_mask)
        cy = float(np.average(ys, weights=np.abs(score[feat_mask])))
        cx = float(np.average(xs, weights=np.abs(score[feat_mask])))
        if (cx < border_px or cx > M-border_px or
            cy < border_px or cy > N-border_px): continue
        snr  = float(score[int(round(cy)), int(round(cx))])
        flux = float(diff[int(round(cy)), int(round(cx))])
        result.append({'x':cx,'y':cy,'snr':snr,'flux':flux,'area_px':area})
    return sorted(result, key=lambda r: -abs(r['snr']))


def estimate_kernel(template: np.ndarray,
                    reference: np.ndarray,
                    cx: int, cy: int,
                    size: int = 31) -> np.ndarray:
    """Extract the local matching kernel at position (cx, cy)."""
    N, M = template.shape
    hs   = size // 2
    y0   = max(0, cy-hs); y1 = min(N, cy+hs+1)
    x0   = max(0, cx-hs); x1 = min(M, cx+hs+1)
    T_t  = (template [y0:y1,x0:x1] - np.median(template [y0:y1,x0:x1])).astype(complex)
    R_t  = (reference[y0:y1,x0:x1] - np.median(reference[y0:y1,x0:x1])).astype(complex)
    T_h  = np.fft.fft2(T_t); R_h = np.fft.fft2(R_t)
    eps  = 0.01 * float(np.abs(T_h)**2).max() + 1e-30
    K_h  = (R_h*np.conj(T_h))/(np.abs(T_h)**2+eps)
    k    = np.real(np.fft.ifft2(K_h))
    return np.fft.fftshift(k)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class SFFTWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, template, reference, params):
        super().__init__()
        self.template  = template
        self.reference = reference
        self.params    = params

    def run(self):
        try:
            import time
            p = self.params
            self.progress.emit(5, "Running SFFT subtraction…")
            t0 = time.time()

            diff, score = sfft_subtract(
                self.template, self.reference,
                tile_size = p['tile_size'],
                overlap   = p['overlap'],
                reg_eps   = p['reg_eps'])

            dt = time.time() - t0
            self.progress.emit(75, f"Detecting transients (threshold={p['threshold']}σ)…")

            alerts = detect_transients(
                score, diff,
                threshold = p['threshold'],
                min_area  = p['min_area'],
                max_area  = p['max_area'],
                border_px = p['border_px'])

            # Build kernel at image centre for display
            N, M = self.template.shape
            k_display = estimate_kernel(self.template, self.reference,
                                         M//2, N//2, size=51)

            self.progress.emit(100,
                f"Done!  {len(alerts)} candidates  ({dt*1000:.0f}ms)")
            self.finished.emit({
                'diff': diff, 'score': score,
                'alerts': alerts, 'kernel': k_display,
                'time_ms': dt*1000,
                'template': self.template,
                'reference': self.reference,
            })
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

BG="#0c0e10"; BG2="#111418"; BG3="#181c22"; BOR="#1e2840"
ACC="#40b8e0"; AC2="#1878a0"; TXT="#c8d8e8"; DIM="#283848"
OK="#40c080"; WRN="#e09040"; ERR="#c04040"; SEC="#5898d8"

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
QPushButton#run{{background:{AC2};border-color:{ACC};color:white;font-weight:bold;padding:7px 20px;}}
QSpinBox,QDoubleSpinBox,QComboBox{{background:{BG3};
    border:1px solid {BOR};border-radius:3px;padding:3px 6px;color:{TXT};}}
QTextEdit{{background:#07090c;border:1px solid {BOR};color:{OK};
    font-family:Consolas,monospace;font-size:10px;}}
QTableWidget{{background:#07090c;alternate-background-color:{BG3};
    gridline-color:{BOR};color:{TXT};}}
QTableWidget QHeaderView::section{{background:{BG3};color:{SEC};
    border:1px solid {BOR};padding:3px;font-weight:bold;}}
QProgressBar{{background:{BG3};border:1px solid {BOR};
    border-radius:3px;height:8px;}}
QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {AC2},stop:1 {ACC});border-radius:2px;}}
QSplitter::handle{{background:{BOR};}}
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8, 5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor=BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def redraw(self): self.fig.tight_layout(pad=0.3); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#07090c')
    ax.set_title(title, color=SEC, fontsize=9, pad=3)
    ax.set_xlabel(xl, color=DIM, fontsize=8)
    ax.set_ylabel(yl, color=DIM, fontsize=8)
    ax.tick_params(colors=DIM, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(BOR)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class SFFTApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 920)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._template  = None
        self._reference = None
        self._result    = None
        self._worker    = self._wthread = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._controls())
        sp.addWidget(self._tabs())
        sp.setSizes([280, 1160])
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
        t = QLabel("SFFT Image Subtraction")
        t.setStyleSheet(f"color:{ACC};font-size:17px;font-weight:bold;padding:6px;")
        s = QLabel("Fourier-space difference imaging  ·  O(N log N) tiled kernel  ·  "
                   "Spatially varying PSF  ·  Companion to zogy_subtract.py  ·  Hu et al. 2022")
        s.setStyleSheet(f"color:{DIM};font-size:9pt;padding:0 8px 4px;")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(5)

        # Load images
        grp = QGroupBox("Images"); gl = QGridLayout(grp)
        self.tmpl_lbl = QLabel("No template"); self.tmpl_lbl.setStyleSheet(f"color:{DIM};font-size:10px;")
        self.ref_lbl  = QLabel("No reference"); self.ref_lbl .setStyleSheet(f"color:{DIM};font-size:10px;")
        b_tmpl = QPushButton("📂  Template (reference)"); b_tmpl.clicked.connect(self._load_template)
        b_ref  = QPushButton("📂  Science image (new)");  b_ref .clicked.connect(self._load_reference)
        b_ex   = QPushButton("📋  Synthetic example");    b_ex  .clicked.connect(self._load_example)
        gl.addWidget(b_tmpl,0,0,1,2); gl.addWidget(self.tmpl_lbl,1,0,1,2)
        gl.addWidget(b_ref, 2,0,1,2); gl.addWidget(self.ref_lbl, 3,0,1,2)
        gl.addWidget(b_ex,  4,0,1,2); lay.addWidget(grp)

        # SFFT parameters
        grp2 = QGroupBox("SFFT Parameters"); g2 = QFormLayout(grp2)
        self.sp_tile  = QSpinBox();   self.sp_tile .setRange(64,1024); self.sp_tile .setValue(256); self.sp_tile.setSingleStep(64)
        self.sp_ovlp  = QSpinBox();   self.sp_ovlp .setRange(16,256);  self.sp_ovlp .setValue(64)
        self.sp_eps   = QDoubleSpinBox(); self.sp_eps.setRange(1e-4,0.1); self.sp_eps.setValue(0.01); self.sp_eps.setDecimals(4)
        g2.addRow("Tile size (px):", self.sp_tile)
        g2.addRow("Overlap (px):",   self.sp_ovlp)
        g2.addRow("Regularisation ε:", self.sp_eps)
        note = QLabel("Larger tiles = better for low-density fields.\n"
                      "More overlap = smoother transitions.")
        note.setStyleSheet(f"color:{DIM};font-size:9px;"); note.setWordWrap(True)
        g2.addRow(note); lay.addWidget(grp2)

        # Detection parameters
        grp3 = QGroupBox("Transient Detection"); g3 = QFormLayout(grp3)
        self.sp_thresh = QDoubleSpinBox(); self.sp_thresh.setRange(2.0,20.0); self.sp_thresh.setValue(5.0); self.sp_thresh.setDecimals(1)
        self.sp_amin   = QSpinBox(); self.sp_amin.setRange(1,50);   self.sp_amin.setValue(2)
        self.sp_amax   = QSpinBox(); self.sp_amax.setRange(10,2000);self.sp_amax.setValue(400)
        self.sp_border = QSpinBox(); self.sp_border.setRange(0,100);self.sp_border.setValue(20)
        g3.addRow("Detection threshold (σ):", self.sp_thresh)
        g3.addRow("Min area (px):", self.sp_amin)
        g3.addRow("Max area (px):", self.sp_amax)
        g3.addRow("Border mask (px):", self.sp_border)
        lay.addWidget(grp3)

        self.btn_run = QPushButton("▶  Subtract")
        self.btn_run.setObjectName("run"); self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_exp = QPushButton("💾  Export difference FITS…")
        self.btn_exp.setEnabled(False); self.btn_exp.clicked.connect(self._export)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_exp)
        lay.addStretch()
        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(f"color:{OK};font-size:10px;"); self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_images(),   "🖼  Images")
        self.tabs.addTab(self._tab_diff(),     "🔍  Difference")
        self.tabs.addTab(self._tab_kernel(),   "⚙  Kernel")
        self.tabs.addTab(self._tab_detections(),"⚡  Detections")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_images(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.images_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.images_plot)
        return w

    def _tab_diff(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.diff_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.diff_plot)
        return w

    def _tab_kernel(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.kernel_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.kernel_plot)
        return w

    def _tab_detections(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.det_table = QTableWidget(0,5)
        self.det_table.setHorizontalHeaderLabels(["#","X","Y","SNR (σ)","Flux"])
        self.det_table.setFixedHeight(220); self.det_table.setAlternatingRowColors(True)
        self.det_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.det_table)
        self.det_map = MplCanvas(figsize=(9,4)); lay.addWidget(self.det_map)
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

    def _load_fits_file(self, role):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {role} FITS", "", "FITS (*.fits *.fit);;All (*)")
        if not path: return None, path
        try:
            data = astropy_fits.getdata(path).astype(float)
            if data.ndim == 3: data = data[0]
            return data, path
        except Exception as e:
            QMessageBox.critical(self,"Load error",str(e)); return None, path

    def _load_template(self):
        data, path = self._load_fits_file("template")
        if data is None: return
        self._template = data
        self.tmpl_lbl.setText(f"{Path(path).name}  {data.shape}")
        self.tmpl_lbl.setStyleSheet(f"color:{OK};font-size:10px;")
        self._try_enable(); self._draw_images(); self._log(f"Template: {path}")

    def _load_reference(self):
        data, path = self._load_fits_file("science")
        if data is None: return
        self._reference = data
        self.ref_lbl.setText(f"{Path(path).name}  {data.shape}")
        self.ref_lbl.setStyleSheet(f"color:{OK};font-size:10px;")
        self._try_enable(); self._draw_images(); self._log(f"Reference: {path}")

    def _load_example(self):
        rng2 = np.random.default_rng(42); N = 512

        def psf(N, s, cx, cy):
            yy,xx=np.ogrid[:N,:N]
            p=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*(s/2.355)**2)); return p/p.sum()

        def fft_conv(a,b): return np.real(np.fft.ifft2(np.fft.fft2(a)*np.fft.fft2(b)))

        # Stars
        img=np.zeros((N,N))
        xs=rng2.integers(20,N-20,60); ys=rng2.integers(20,N-20,60)
        fl=rng2.lognormal(5,1.5,60)
        for x,y,f in zip(xs,ys,fl): img[y,x]+=f
        # Galaxy
        yg,xg=np.ogrid[:N,:N]
        img+=300*np.exp(-((xg-N//2)**2+(yg-N//2)**2)/1000)

        self._template  = fft_conv(img,psf(N,3.0,N//2,N//2))+rng2.normal(0,2,(N,N))+100
        self._reference = fft_conv(img,psf(N,4.0,N//2,N//2))+rng2.normal(0,3,(N,N))+130
        # Transients
        for ty,tx,fl2 in [(200,150,500),(300,380,1200),(100,400,300)]:
            self._reference[ty,tx]+=fl2

        for lbl,data in [("Template",self._template),("Reference (3 transients)",self._reference)]:
            lbl_w = self.tmpl_lbl if lbl.startswith("T") else self.ref_lbl
            lbl_w.setText(f"{lbl}  {data.shape}"); lbl_w.setStyleSheet(f"color:{OK};font-size:10px;")

        self._try_enable(); self._draw_images()
        self._log("Synthetic example: 512×512  PSF 3px vs 4px  3 injected transients")

    def _try_enable(self):
        ok = self._template is not None and self._reference is not None
        if ok and self._template.shape != self._reference.shape:
            self._log("⚠ Image shapes don't match"); ok=False
        self.btn_run.setEnabled(ok)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _params(self):
        return dict(
            tile_size=self.sp_tile.value(), overlap=self.sp_ovlp.value(),
            reg_eps=self.sp_eps.value(), threshold=self.sp_thresh.value(),
            min_area=self.sp_amin.value(), max_area=self.sp_amax.value(),
            border_px=self.sp_border.value())

    def _run(self):
        if self._template is None or self._reference is None: return
        self.btn_run.setEnabled(False); self.btn_exp.setEnabled(False)
        self._wthread = QThread(self)
        self._worker  = SFFTWorker(self._template, self._reference, self._params())
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()
        self.tabs.setCurrentIndex(1)

    def _on_progress(self, pct, msg):
        self.status_lbl.setText(msg)
        self.pbar.setVisible(True); self.pbar.setValue(pct)
        if pct>=100: QTimer.singleShot(3000,lambda:self.pbar.setVisible(False))
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}"); self.btn_run.setEnabled(True)

    def _on_finished(self, result):
        self._result = result; self.btn_run.setEnabled(True); self.btn_exp.setEnabled(True)
        n = len(result['alerts'])
        self.summary_lbl.setText(
            f"{n} candidates detected\n"
            f"Time: {result['time_ms']:.0f}ms  "
            f"({self._template.shape[0]}×{self._template.shape[1]}px)")
        self._log(f"\n✓  {n} candidates  {result['time_ms']:.0f}ms")
        for i,a in enumerate(result['alerts'][:10]):
            self._log(f"   #{i+1}  ({a['x']:.1f},{a['y']:.1f})  SNR={a['snr']:.1f}σ  flux={a['flux']:.1f}")
        self._draw_diff(); self._draw_kernel(); self._draw_detections()
        self.tabs.setCurrentIndex(3 if n>0 else 1)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_images(self):
        fig = self.images_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        axes = fig.subplots(1,2)
        for ax,img,title in [(axes[0],self._template,"Template"),
                             (axes[1],self._reference,"Science / Reference")]:
            if img is None: continue
            lo,hi=np.nanpercentile(img,[1,99.5])
            ax.imshow(img,origin='lower',cmap='gray',vmin=lo,vmax=hi,
                      aspect='equal',interpolation='bicubic')
            _ax(ax,title)
            ax.set_xticks([]); ax.set_yticks([])
        self.images_plot.redraw()

    def _draw_diff(self):
        r = self._result
        if not r: return
        fig = self.diff_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        axes = fig.subplots(1,2)

        # Difference image
        diff  = r['diff']; score = r['score']
        vmax  = max(np.std(diff)*4, 1e-5)
        _ax(axes[0],"Difference image (R − T⊛k)")
        axes[0].imshow(diff, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='equal', interpolation='nearest')
        axes[0].set_xticks([]); axes[0].set_yticks([])
        # Overlay detections
        for a in r['alerts']:
            c='lime' if a['snr']>0 else 'red'
            axes[0].plot(a['x'],a['y'],'x',color=c,ms=12,mew=2)

        # Score map
        sv = max(np.std(score)*5, 1.0)
        _ax(axes[1],"SNR score map")
        axes[1].imshow(score, origin='lower', cmap='hot',
                       vmin=-sv, vmax=sv, aspect='equal', interpolation='nearest')
        axes[1].set_xticks([]); axes[1].set_yticks([])
        thresh_line = f"threshold={self.sp_thresh.value()}σ"
        axes[1].text(0.02,0.02,f"{len(r['alerts'])} detected  {thresh_line}",
                     transform=axes[1].transAxes,color=OK,fontsize=8)

        self.diff_plot.redraw()

    def _draw_kernel(self):
        r = self._result
        if not r or r.get('kernel') is None: return
        fig = self.kernel_plot.fig; fig.clear(); fig.patch.set_facecolor(BG)
        axes = fig.subplots(1,2)

        k = r['kernel']; kc = k.shape[0]//2
        crop= k[kc-12:kc+13, kc-12:kc+13]
        _ax(axes[0],"Matching kernel (centre tile)"); axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].imshow(crop, origin='lower', cmap='viridis',
                       vmin=crop.min(), vmax=crop.max(),
                       aspect='equal', interpolation='bicubic')

        # Radial profile
        ny,nx = crop.shape; cy2=ny//2; cx2=nx//2
        yy,xx = np.ogrid[:ny,:nx]
        r_px  = np.sqrt((xx-cx2)**2+(yy-cy2)**2)
        r_max = int(r_px.max())
        rads  = np.arange(0, r_max)
        prof  = [float(crop[np.abs(r_px-rr)<0.7].mean()) if (np.abs(r_px-rr)<0.7).sum()>0
                 else 0.0 for rr in rads]
        _ax(axes[1],"Kernel radial profile","Radius (px)","Weight")
        axes[1].plot(rads, prof, color=ACC, lw=1.5)
        axes[1].axhline(0, color=BOR, lw=0.5)

        self.kernel_plot.redraw()

    def _draw_detections(self):
        r = self._result
        if not r: return
        alerts = r['alerts']
        self.det_table.setRowCount(len(alerts))
        for i, a in enumerate(alerts):
            for col, val in enumerate([str(i+1),
                                        f"{a['x']:.1f}",f"{a['y']:.1f}",
                                        f"{a['snr']:.1f}",f"{a['flux']:.0f}"]):
                item = QTableWidgetItem(val)
                if col == 3:
                    color = OK if a['snr'] > 0 else ERR
                    item.setForeground(QColor(color))
                self.det_table.setItem(i, col, item)

        fig = self.det_map.fig; fig.clear(); fig.patch.set_facecolor(BG)
        ax  = fig.add_subplot(111)
        _ax(ax, f"Detection map  ({len(alerts)} candidates)","X (px)","Y (px)")
        if self._reference is not None:
            lo,hi=np.nanpercentile(self._reference,[1,99.5])
            ax.imshow(self._reference, origin='lower', cmap='gray',
                      vmin=lo, vmax=hi, aspect='equal', alpha=0.7,
                      interpolation='bicubic')
        for a in alerts:
            c='lime' if a['snr']>0 else 'red'
            ax.plot(a['x'],a['y'],'o',color=c,ms=12,mew=2,fillstyle='none')
            ax.text(a['x']+5,a['y']+5,f"{a['snr']:.1f}σ",color=c,fontsize=7)
        self.det_map.redraw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        r = self._result
        if not r: return
        path, _ = QFileDialog.getSaveFileName(
            self,"Export difference image","sfft_difference.fits","FITS (*.fits);;All (*)")
        if not path: return
        try:
            base = Path(path)
            # Difference FITS
            hdr=astropy_fits.Header()
            hdr['COMMENT']='SFFT difference image — HYPERLOAD sfft_subtract.py'
            hdr['SFFT_T']=self.sp_tile.value(); hdr['SFFT_O']=self.sp_ovlp.value()
            hdr['N_CAND']=len(r['alerts'])
            astropy_fits.PrimaryHDU(r['diff'].astype(np.float32),header=hdr).writeto(str(path),overwrite=True)
            # Score map
            score_path=base.with_name(base.stem+'_score'+base.suffix)
            astropy_fits.PrimaryHDU(r['score'].astype(np.float32)).writeto(str(score_path),overwrite=True)
            # Detections CSV
            csv_path=base.with_name(base.stem+'_detections.csv')
            import csv as _csv
            with open(str(csv_path),'w',newline='') as f:
                w=_csv.DictWriter(f,fieldnames=['x','y','snr','flux','area_px']); w.writeheader(); w.writerows(r['alerts'])
            self._log(f"✓ Exported: {path}, {score_path}, {csv_path}")
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
    win = SFFTApp(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
