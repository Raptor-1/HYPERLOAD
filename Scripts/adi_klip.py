r"""
HYPERLOAD — ADI / KLIP High-Contrast Imaging
=============================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Angular Differential Imaging with KLIP PSF subtraction for direct imaging
of close companions (exoplanets, brown dwarfs, tight binaries).

The star's PSF is static in the pupil-stabilised (ADI) frame while the
companion rotates with the sky.  By modelling the PSF from reference frames
at different parallactic angles and subtracting, the companion is revealed.

Two algorithms:
  cADI (Marois et al. 2006):
    PSF model = temporal median of all frames (excluding nearby PAs).
    Fast, works well with large field rotation (> 1× FWHM at companion sep).

  KLIP (Soummer et al. 2012):
    PSF model = projection of target onto top-K Karhunen-Loève vectors
    computed from the reference library via SVD.
    Optimal for small rotation, high-contrast.  Excludes frames within
    exclusion_angle degrees of target to avoid companion self-subtraction.

Output:
  • ADI / KLIP combined image
  • S/N map (noise estimated per annulus, Mawet et al. 2014)
  • 5-sigma contrast curve (detection limits vs angular separation)
  • Candidate companion table

References:
  Marois et al. 2006    — ADI, ApJ 641, 556
  Lafrenière et al. 2007— LOCI, ApJ 660, 770
  Soummer et al. 2012   — KLIP, ApJL 755, L28
  Mawet et al. 2014     — SNR in ADI, ApJ 792, 97 (t-test correction)
  Amara & Quanz 2012    — PynPoint/contrast curves, MNRAS 427, 948

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, traceback, json, math
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nadi_klip.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.ndimage import rotate as nd_rotate, gaussian_filter, shift as nd_shift
from scipy.signal import find_peaks

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
    QLineEdit, QPlainTextEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — ADI / KLIP High-Contrast Imaging"

# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSICAL ADI  (Marois et al. 2006)
# ═══════════════════════════════════════════════════════════════════════════════

def cadi(frames: np.ndarray, pa_angles: np.ndarray,
         exclusion_angle: float = 1.0,
         combine: str = 'median') -> dict:
    """
    Classical ADI — median PSF subtraction.

    PSF model for frame i = median of all frames j where
    |PA_j − PA_i| > exclusion_angle × FWHM / (2π × separation).
    For simplicity: exclusion_angle in degrees directly.

    combine: 'median' | 'mean' | 'weighted_mean'
    """
    N_fr, ny, nx = frames.shape
    subtracted   = np.zeros_like(frames)

    for i in range(N_fr):
        pa_diff   = np.abs(pa_angles - pa_angles[i])
        ref_mask  = pa_diff > exclusion_angle
        ref_mask[i] = False

        if ref_mask.sum() < 3:
            ref_mask  = np.ones(N_fr, bool); ref_mask[i] = False

        psf_model       = np.median(frames[ref_mask], axis=0)
        subtracted[i]   = frames[i] - psf_model

    # Derotate
    derotated = np.zeros_like(subtracted)
    for i, pa in enumerate(pa_angles):
        derotated[i] = nd_rotate(subtracted[i], pa,
                                  reshape=False, order=3, cval=0.0)

    if combine == 'median':
        combined = np.median(derotated, axis=0)
    elif combine == 'mean':
        combined = np.mean(derotated, axis=0)
    else:
        # Weighted mean by frame variance
        weights  = 1.0 / (np.var(subtracted, axis=(1,2)) + 1e-10)
        combined = np.average(derotated, weights=weights, axis=0)

    return {
        'image':     combined,
        'subtracted': subtracted,
        'derotated': derotated,
        'method':    'cADI',
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  KLIP  (Soummer et al. 2012)
# ═══════════════════════════════════════════════════════════════════════════════

def klip(frames: np.ndarray, pa_angles: np.ndarray,
         n_modes: int = 10, exclusion_angle: float = 20.0,
         inner_mask_px: float = 3.0,
         progress_cb=None) -> dict:
    """
    Karhunen-Loève Image Processing (KLIP).

    For each target frame:
      1. Build reference library: frames with |ΔPA| > exclusion_angle
      2. SVD of mean-subtracted reference library → KL basis vectors
      3. Project target onto top-K KL vectors → PSF estimate
      4. Subtract PSF estimate
    Derotate and median-combine.

    exclusion_angle: in degrees.  Larger → less self-subtraction but
                     fewer reference frames → noisier PSF model.
                     Rule of thumb: exclusion_angle ≈ FWHM / sep (rad) × (180/π)
    """
    N_fr, ny, nx = frames.shape
    subtracted   = np.zeros_like(frames)

    # Build annular mask (avoid using pixels interior to inner_mask_px)
    yy, xx = np.ogrid[:ny, :nx]
    rr     = np.sqrt((xx - nx//2)**2 + (yy - ny//2)**2)
    good_px = rr > inner_mask_px

    for i in range(N_fr):
        if progress_cb and i % max(1, N_fr//20) == 0:
            progress_cb(i, N_fr)

        target    = frames[i]
        pa_diff   = np.abs(pa_angles - pa_angles[i])
        ref_mask  = pa_diff > exclusion_angle
        ref_mask[i] = False

        if ref_mask.sum() < max(n_modes, 3):
            ref_mask = np.ones(N_fr, bool); ref_mask[i] = False

        R      = frames[ref_mask].reshape(ref_mask.sum(), -1).astype(float)
        t      = target.reshape(-1).astype(float)
        R_mean = R.mean(axis=0)
        R_c    = R - R_mean
        t_c    = t - R_mean

        # Economy SVD
        try:
            _, S, Vt = np.linalg.svd(R_c, full_matrices=False)
        except np.linalg.LinAlgError:
            subtracted[i] = target; continue

        K      = min(n_modes, len(S), ref_mask.sum())
        KL     = Vt[:K]                       # (K, N_px)
        coeff  = KL @ t_c                     # (K,)
        psf_est= coeff @ KL + R_mean          # (N_px,)  coeff(K)@KL(K,Npx)

        res    = (t_c - (psf_est - R_mean)).reshape(ny, nx)
        res[~good_px] = 0.0
        subtracted[i] = res

    # Derotate
    derotated = np.zeros_like(subtracted)
    for i, pa in enumerate(pa_angles):
        derotated[i] = nd_rotate(subtracted[i], pa,
                                  reshape=False, order=3, cval=0.0)

    combined = np.median(derotated, axis=0)

    return {
        'image':      combined,
        'subtracted': subtracted,
        'derotated':  derotated,
        'method':    f'KLIP (K={n_modes})',
        'n_modes':    n_modes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  S/N MAP  (Mawet et al. 2014)
# ═══════════════════════════════════════════════════════════════════════════════

def snr_map(image: np.ndarray, fwhm: float = 4.0) -> np.ndarray:
    """
    Per-pixel S/N map using the small sample statistics correction.
    Mawet et al. 2014 — annular noise estimation with t-test correction.

    For each pixel at separation r:
      - Signal = pixel value
      - Noise  = std of resolution elements in same annulus (width=FWHM)
      - Correction: multiply by sqrt(1 + 1/N_res) for small sample bias

    Returns S/N array, same shape as image.
    """
    ny, nx = image.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr     = np.sqrt((xx - nx//2)**2 + (yy - ny//2)**2)
    snr    = np.zeros_like(image)

    r_max = min(ny, nx) // 2
    dr    = fwhm   # annulus width = 1 FWHM

    for r0 in np.arange(fwhm, r_max - dr, dr/2):
        ann  = (rr >= r0) & (rr < r0 + dr)
        vals = image[ann]
        if len(vals) < 3: continue

        n_res  = max(int(2 * np.pi * r0 / fwhm), 2)  # resolution elements
        noise  = vals.std()
        # Small sample t-test correction (Mawet+2014 Eq. 9)
        corr   = math.sqrt(1.0 + 1.0 / n_res)
        snr[ann] = image[ann] / (noise * corr + 1e-10)

    return snr


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTRAST CURVE
# ═══════════════════════════════════════════════════════════════════════════════

def contrast_curve(image: np.ndarray, fwhm: float = 4.0,
                   sigma: float = 5.0) -> dict:
    """
    5-sigma detection limit vs angular separation.

    At each separation r, the noise in the annulus gives the
    minimum detectable flux for the given sigma threshold.
    Normalized to the peak stellar flux.
    """
    ny, nx = image.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr     = np.sqrt((xx - nx//2)**2 + (yy - ny//2)**2)
    star_peak = image[ny//2 - 3:ny//2 + 4, nx//2 - 3:nx//2 + 4].max()
    if abs(star_peak) < 1e-10: star_peak = 1.0

    r_max = min(ny, nx) // 2
    seps  = []; contrasts = []

    for r0 in np.arange(fwhm, r_max - fwhm, fwhm/2):
        ann  = (rr >= r0) & (rr < r0 + fwhm)
        vals = image[ann]
        if len(vals) < 3: continue
        n_res = max(int(2*np.pi*r0/fwhm), 2)
        noise = vals.std() * math.sqrt(1 + 1/n_res)
        seps.append(r0)
        contrasts.append(sigma * noise / abs(star_peak))

    return {'sep_px': np.array(seps), 'contrast': np.array(contrasts)}


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPANION FINDER
# ═══════════════════════════════════════════════════════════════════════════════

def find_companions(snr_img: np.ndarray, fwhm: float = 4.0,
                    snr_thresh: float = 5.0) -> list:
    """
    Detect companion candidates above SNR threshold.
    Returns list of dicts: sep_px, pa_deg, snr, x, y.
    """
    ny, nx = snr_img.shape
    yy, xx = np.ogrid[:ny, :nx]
    rr     = np.sqrt((xx - nx//2)**2 + (yy - ny//2)**2)

    # Mask inner region (star)
    inner = rr < fwhm * 2
    img_m = snr_img.copy(); img_m[inner] = 0

    # Find peaks
    from scipy.ndimage import label, find_objects
    above = img_m > snr_thresh
    labs, n_lab = label(above)
    candidates = []

    for k in range(1, n_lab + 1):
        reg = np.where(labs == k)
        if len(reg[0]) < 1: continue
        peak_idx = snr_img[reg].argmax()
        py, px   = reg[0][peak_idx], reg[1][peak_idx]
        sep      = float(rr[py, px])
        pa       = float(math.degrees(math.atan2(px - nx//2,
                                                   -(py - ny//2))) % 360)
        candidates.append({
            'x': int(px), 'y': int(py),
            'sep_px': sep, 'pa_deg': pa,
            'snr': float(snr_img[py, px]),
        })

    return sorted(candidates, key=lambda c: -c['snr'])


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def luminance(data):
    if data.ndim == 2:  return data.astype(float)
    if data.ndim == 3 and data.shape[0] <= 4: return data.mean(0).astype(float)
    if data.ndim == 3 and data.shape[2] <= 4: return data.mean(2).astype(float)
    return data.astype(float)


class ADIWorker(QObject):
    progress    = pyqtSignal(int, str)
    frame_update= pyqtSignal(int, int)
    finished    = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, paths, pa_angles, params):
        super().__init__()
        self.paths     = paths
        self.pa_angles = pa_angles
        self.params    = params
        self._cancel   = False

    def cancel(self): self._cancel = True

    def run(self):
        try: self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p = self.params

        # ── Load frames ───────────────────────────────────────────────────
        self.progress.emit(5, f"Loading {len(self.paths)} frames…")
        frames_list = []
        for i, path in enumerate(self.paths):
            if self._cancel: return
            try:
                data = fits.getdata(str(path))
                frames_list.append(luminance(data))
            except Exception as e:
                self.error.emit(f"Could not load {path}: {e}"); return
            if i % max(1, len(self.paths)//10) == 0:
                self.progress.emit(int(5 + 20*i/len(self.paths)),
                                   f"Loading frame {i+1}/{len(self.paths)}")

        frames = np.array(frames_list, dtype=float)
        N_fr, ny, nx = frames.shape
        self.progress.emit(25, f"Loaded {N_fr} frames  {nx}×{ny}")

        # ── Centre frames on star ─────────────────────────────────────────
        self.progress.emit(28, "Centring on star…")
        if p.get('auto_centre', True):
            frames = self._centre_frames(frames)

        pa = np.asarray(self.pa_angles, float)

        # ── PSF subtraction ───────────────────────────────────────────────
        method = p.get('method', 'KLIP')
        fwhm   = p.get('fwhm', 4.0)

        if method == 'cADI':
            self.progress.emit(35, "Running cADI…")
            result = cadi(frames, pa,
                          exclusion_angle=p.get('exclusion_angle', 10.0),
                          combine=p.get('combine', 'median'))
        else:
            self.progress.emit(35, f"Running KLIP (K={p.get('n_modes',10)})…")
            def prog(i, total):
                pct = int(35 + 45*i/total)
                self.progress.emit(pct, f"KLIP frame {i+1}/{total}")
                self.frame_update.emit(i, total)
            result = klip(frames, pa,
                          n_modes=p.get('n_modes', 10),
                          exclusion_angle=p.get('exclusion_angle', 20.0),
                          inner_mask_px=p.get('inner_mask', fwhm),
                          progress_cb=prog)

        if self._cancel: return

        # ── S/N map ───────────────────────────────────────────────────────
        self.progress.emit(82, "Computing S/N map…")
        snr  = snr_map(result['image'], fwhm)

        # ── Contrast curve ────────────────────────────────────────────────
        self.progress.emit(88, "Computing contrast curve…")
        # Use median-collapsed unsubtracted frames as star proxy
        star_frame = np.median(frames, axis=0)
        cc         = contrast_curve(result['image'], fwhm,
                                    sigma=p.get('snr_thresh', 5.0))

        # ── Companion detection ───────────────────────────────────────────
        self.progress.emit(94, "Detecting companions…")
        candidates = find_companions(snr, fwhm,
                                     snr_thresh=p.get('snr_thresh', 5.0))

        self.progress.emit(100, f"Done!  {len(candidates)} candidate(s) found")

        self.finished.emit({
            'adi':        result['image'],
            'snr':        snr,
            'contrast':   cc,
            'candidates': candidates,
            'frames':     frames,
            'pa_angles':  pa,
            'star_frame': star_frame,
            'method':     result['method'],
            'fwhm':       fwhm,
            'N_fr':       N_fr,
        })

    def _centre_frames(self, frames: np.ndarray) -> np.ndarray:
        N_fr, ny, nx = frames.shape
        ref_fft = np.fft.fft2(frames[0])
        centred = [frames[0]]
        for i in range(1, N_fr):
            F   = np.fft.fft2(frames[i])
            R   = np.conj(ref_fft) * F
            d   = np.abs(R); d[d < 1e-10] = 1.0; R /= d
            r   = np.real(np.fft.ifft2(R))
            pj, pi = np.unravel_index(r.argmax(), r.shape)
            dy  = float(pj if pj < ny//2 else pj - ny)
            dx  = float(pi if pi < nx//2 else pi - nx)
            centred.append(nd_shift(frames[i], [-dy, -dx],
                                    order=1, mode='constant', cval=0))
        return np.array(centred)


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
QTextEdit,QPlainTextEdit{background:#0e0e20;border:1px solid #26264a;color:#85e085;
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
                border-radius:3px;}
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #4040c0,stop:1 #9050e0);border-radius:2px;}""")
        lay.addWidget(self.lbl,1); lay.addWidget(self.pbar)

    def update(self, pct, msg):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True); self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))


class ADIApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 920)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))
        self._paths    = []
        self._pa_angles= []
        self._result   = None
        self._worker   = self._wthread = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._make_header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._make_controls())
        sp.addWidget(self._make_tabs())
        sp.setSizes([320,1080])
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
        t = QLabel("ADI / KLIP High-Contrast Imaging"); t.setObjectName("title_lbl")
        s = QLabel("Angular Differential Imaging · Karhunen-Loève PSF subtraction · "
                   "S/N map (Mawet 2014) · 5-sigma contrast curve · Companion detection")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _make_controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)

        # ── Frame loading ──────────────────────────────────────────────────
        grp = QGroupBox("ADI Frame Cube")
        gl  = QVBoxLayout(grp)
        self.file_lbl = QLabel("No frames loaded")
        self.file_lbl.setStyleSheet("color:#484868;font-size:10px;")
        b_fits = QPushButton("📂  Load FITS series…")
        b_fits.clicked.connect(self._load_fits_series)
        b_cube = QPushButton("📦  Load FITS cube (3D)…")
        b_cube.clicked.connect(self._load_fits_cube)
        b_ex   = QPushButton("📋  Load synthetic example")
        b_ex.clicked.connect(self._load_example)
        gl.addWidget(self.file_lbl); gl.addWidget(b_fits)
        gl.addWidget(b_cube); gl.addWidget(b_ex)
        lay.addWidget(grp)

        # ── Parallactic angles ─────────────────────────────────────────────
        grp2 = QGroupBox("Parallactic Angles")
        g2   = QVBoxLayout(grp2)
        g2.addWidget(QLabel("One angle per line (degrees), or start:stop:N:"))
        self.pa_edit = QPlainTextEdit()
        self.pa_edit.setFixedHeight(80)
        self.pa_edit.setPlainText("0:60:30")
        g2.addWidget(self.pa_edit)
        g2.addWidget(QLabel("Or: from FITS header keyword:"))
        h2 = QHBoxLayout()
        self.le_pa_kw = QLineEdit("PA"); self.le_pa_kw.setFixedWidth(80)
        b_kw = QPushButton("Read from headers")
        b_kw.clicked.connect(self._read_pa_headers)
        h2.addWidget(self.le_pa_kw); h2.addWidget(b_kw); h2.addStretch()
        g2.addLayout(h2)
        lay.addWidget(grp2)

        # ── Algorithm ─────────────────────────────────────────────────────
        grp3 = QGroupBox("Algorithm")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("Method:"), 0, 0)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["KLIP (Soummer 2012)", "cADI (Marois 2006)"])
        self.cmb_method.currentIndexChanged.connect(self._on_method)
        g3.addWidget(self.cmb_method, 0, 1)
        g3.addWidget(QLabel("KL modes (KLIP):"), 1, 0)
        self.sp_modes = QSpinBox(); self.sp_modes.setRange(1,100); self.sp_modes.setValue(10)
        self.sp_modes.setToolTip(
            "Number of KL basis modes.\n"
            "More modes = better PSF model but more self-subtraction.\n"
            "Typical: 5-20.  Start with 10.")
        g3.addWidget(self.sp_modes, 1, 1)
        g3.addWidget(QLabel("Exclusion angle (°):"), 2, 0)
        self.sp_excl = QDoubleSpinBox(); self.sp_excl.setRange(0,180); self.sp_excl.setValue(20.0)
        self.sp_excl.setDecimals(1)
        self.sp_excl.setToolTip(
            "Minimum PA separation for reference frames.\n"
            "Prevents companion self-subtraction.\n"
            "Rule: ≥ FWHM / (2π × sep_px) × 180°")
        g3.addWidget(self.sp_excl, 2, 1)
        g3.addWidget(QLabel("FWHM (px):"), 3, 0)
        self.sp_fwhm = QDoubleSpinBox(); self.sp_fwhm.setRange(1,50); self.sp_fwhm.setValue(4.0); self.sp_fwhm.setDecimals(1)
        g3.addWidget(self.sp_fwhm, 3, 1)
        g3.addWidget(QLabel("Inner mask (px):"), 4, 0)
        self.sp_mask = QDoubleSpinBox(); self.sp_mask.setRange(0,50); self.sp_mask.setValue(4.0); self.sp_mask.setDecimals(1)
        g3.addWidget(self.sp_mask, 4, 1)
        g3.addWidget(QLabel("Combine:"), 5, 0)
        self.cmb_comb = QComboBox()
        self.cmb_comb.addItems(["median","mean","weighted_mean"])
        g3.addWidget(self.cmb_comb, 5, 1)
        lay.addWidget(grp3)

        # ── Detection ─────────────────────────────────────────────────────
        grp4 = QGroupBox("Companion Detection")
        g4   = QGridLayout(grp4)
        g4.addWidget(QLabel("SNR threshold:"), 0, 0)
        self.sp_snrth = QDoubleSpinBox(); self.sp_snrth.setRange(1,20); self.sp_snrth.setValue(5.0); self.sp_snrth.setDecimals(1)
        g4.addWidget(self.sp_snrth, 0, 1)
        self.cb_centre = QCheckBox("Auto-centre frames")
        self.cb_centre.setChecked(True)
        g4.addWidget(self.cb_centre, 1, 0, 1, 2)
        lay.addWidget(grp4)

        # ── Output ────────────────────────────────────────────────────────
        grp5 = QGroupBox("Output")
        g5   = QVBoxLayout(grp5)
        self.btn_save = QPushButton("💾  Save ADI image (FITS)…")
        self.btn_save.setEnabled(False); self.btn_save.clicked.connect(self._save)
        g5.addWidget(self.btn_save)
        lay.addWidget(grp5)

        self.btn_run = QPushButton("▶  Run ADI / KLIP")
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
        self.tabs.addTab(self._tab_preview(), "🎞  Frame Preview")
        self.tabs.addTab(self._tab_adi(),     "🔭  ADI Image")
        self.tabs.addTab(self._tab_snr(),     "📊  S/N Map")
        self.tabs.addTab(self._tab_contrast(),"📈  Contrast Curve")
        self.tabs.addTab(self._tab_cands(),   "🪐  Candidates")
        self.tabs.addTab(self._tab_log(),     "📋  Log")
        return self.tabs

    def _tab_preview(self):
        w = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Frame:"))
        self.sp_frame_idx = QSpinBox(); self.sp_frame_idx.setRange(0,0)
        self.sp_frame_idx.valueChanged.connect(self._refresh_preview)
        ctrl.addWidget(self.sp_frame_idx); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.prev_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.prev_plot)
        return w

    def _tab_adi(self):
        w = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Stretch:"))
        self.cmb_stretch = QComboBox()
        self.cmb_stretch.addItems(["Linear","Sqrt","Log","Asinh"])
        self.cmb_stretch.currentIndexChanged.connect(self._refresh_adi)
        ctrl.addWidget(self.cmb_stretch); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.adi_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.adi_plot)
        return w

    def _tab_snr(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.snr_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.snr_plot)
        return w

    def _tab_contrast(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.cc_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.cc_plot)
        return w

    def _tab_cands(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.cand_table = QTableWidget(0, 5)
        self.cand_table.setHorizontalHeaderLabels(
            ["#","Sep (px)","PA (°)","S/N","Notes"])
        self.cand_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch)
        self.cand_table.setAlternatingRowColors(True)
        lay.addWidget(self.cand_table)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_fits_series(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,"Select ADI frames (in order)","",
            "FITS (*.fits *.fit *.fts);;All (*)")
        if not paths: return
        self._paths = [Path(p) for p in sorted(paths)]
        self._after_load(f"{len(self._paths)} FITS files")

    def _load_fits_cube(self):
        path, _ = QFileDialog.getOpenFileName(
            self,"Select 3D FITS cube","","FITS (*.fits *.fit);;All (*)")
        if not path: return
        try:
            data = fits.getdata(path)
            if data.ndim != 3:
                QMessageBox.warning(self,"Warning",
                    f"Expected 3D cube, got {data.ndim}D. Loading as single frame.")
            # Save each slice as a temp array reference
            import tempfile; td = tempfile.mkdtemp()
            self._paths = []
            for i in range(data.shape[0]):
                p = Path(td)/f"slice_{i:04d}.fits"
                fits.PrimaryHDU(data[i]).writeto(str(p), overwrite=True)
                self._paths.append(p)
            self._after_load(f"3D cube: {data.shape[0]} slices")
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

    def _load_example(self):
        """Generate synthetic ADI dataset and save as temp FITS."""
        import tempfile; td = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(42)
        N=64; N_FR=30
        y,x = np.ogrid[-N//2:N//2,-N//2:N//2]
        r   = np.sqrt(x**2+y**2)
        psf = 1e4*np.exp(-r**2/(2*2.5**2))
        pa  = np.linspace(0,60,N_FR)
        self._paths = []
        for i,pa_i in enumerate(pa):
            frame = psf.copy()
            cx = N//2+12*np.sin(np.radians(45+pa_i))
            cy = N//2-12*np.cos(np.radians(45+pa_i))
            yy,xx=np.mgrid[:N,:N]
            frame+=10*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*1.5**2))
            frame+=rng.normal(0,1,(N,N))
            p = td/f"adi_{i:04d}.fits"
            fits.PrimaryHDU(frame.astype(np.float32)).writeto(str(p),overwrite=True)
            self._paths.append(p)
        self.pa_edit.setPlainText("0:60:30")
        self._after_load(f"Synthetic: {N_FR} frames  companion at 12px, PA=45°, contrast=1e-3")
        self._log("Injected companion: sep=12px, PA=45°, contrast=10/10000=1e-3")

    def _after_load(self, desc):
        self.file_lbl.setText(desc)
        self.file_lbl.setStyleSheet("color:#70c070;font-size:10px;")
        self.btn_run.setEnabled(True)
        self.sp_frame_idx.setRange(0, max(0, len(self._paths)-1))
        self._log(f"Loaded: {desc}")
        if self._paths:
            try:
                d = fits.getdata(str(self._paths[0]))
                self._draw_single_frame(luminance(d), 0)
            except Exception: pass

    def _read_pa_headers(self):
        kw = self.le_pa_kw.text().strip()
        if not self._paths or not kw: return
        angles = []
        for p in self._paths:
            try:
                hdr = fits.getheader(str(p))
                angles.append(float(hdr.get(kw, 0.0)))
            except Exception:
                angles.append(0.0)
        self.pa_edit.setPlainText("\n".join(f"{a:.4f}" for a in angles))
        self._log(f"Read {len(angles)} PA values from header keyword '{kw}'")

    def _parse_pa_angles(self) -> list:
        text = self.pa_edit.toPlainText().strip()
        if ':' in text:
            parts = text.split(':')
            if len(parts) == 3:
                start,stop,n = float(parts[0]),float(parts[1]),int(parts[2])
                return list(np.linspace(start,stop,n))
        angles = []
        for line in text.splitlines():
            line=line.strip()
            if line:
                try: angles.append(float(line))
                except ValueError: pass
        return angles

    def _on_method(self, idx):
        is_klip = (idx == 0)
        self.sp_modes.setEnabled(is_klip)

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self):
        if not self._result: return
        path,_ = QFileDialog.getSaveFileName(self,"Save ADI FITS","","FITS (*.fits)")
        if not path: return
        r = self._result
        hdr = fits.Header()
        hdr['HISTORY'] = f'HYPERLOAD ADI/KLIP v{VERSION}'
        hdr['METHOD']  = r['method']
        hdr['NFRAMES'] = r['N_fr']
        fits.PrimaryHDU(r['adi'].astype(np.float32),header=hdr).writeto(
            path,overwrite=True)
        self._log(f"✓ Saved: {path}")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if not self._paths: return
        pa = self._parse_pa_angles()
        if len(pa) != len(self._paths):
            if len(pa) < 2:
                QMessageBox.warning(self,"PA error",
                    f"Got {len(pa)} PA values for {len(self._paths)} frames.\n"
                    "Use 'start:stop:N' format or one value per line.")
                return
            # Interpolate
            pa = list(np.interp(np.linspace(0,1,len(self._paths)),
                                 np.linspace(0,1,len(pa)), pa))

        method_names = {0:'KLIP',1:'cADI'}
        params = {
            'method':          method_names[self.cmb_method.currentIndex()],
            'n_modes':         self.sp_modes.value(),
            'exclusion_angle': self.sp_excl.value(),
            'fwhm':            self.sp_fwhm.value(),
            'inner_mask':      self.sp_mask.value(),
            'combine':         self.cmb_comb.currentText(),
            'snr_thresh':      self.sp_snrth.value(),
            'auto_centre':     self.cb_centre.isChecked(),
        }
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False); self._result = None

        self._wthread = QThread(self)
        self._worker  = ADIWorker(self._paths, pa, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_update.connect(self._on_frame_update)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _cancel(self):
        if self._worker: self._worker.cancel()
        self.btn_cancel.setEnabled(False); self.btn_run.setEnabled(True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status.update(pct, msg); self._log(f"[{pct:3d}%]  {msg}")

    def _on_frame_update(self, i, total):
        pass   # Could show live derotated preview here

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}")
        self.status.update(0,"Error"); self.btn_run.setEnabled(True)
        QMessageBox.critical(self,"Error",msg[:500])

    def _on_finished(self, result):
        self._result = result
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)
        r = result
        cands = r['candidates']

        self._log(f"\n{'='*55}")
        self._log(f"✓  ADI/KLIP complete  ({r['method']})")
        self._log(f"   Frames:        {r['N_fr']}")
        self._log(f"   Candidates:    {len(cands)}")
        for i,c in enumerate(cands[:5]):
            self._log(f"     #{i+1}  sep={c['sep_px']:.1f}px  PA={c['pa_deg']:.1f}°  SNR={c['snr']:.1f}")
        self._log(f"{'='*55}\n")

        n = len(cands)
        self.summary_lbl.setText(
            f"{r['method']}  |  {r['N_fr']} frames  |  "
            f"{n} candidate{'s' if n!=1 else ''}")

        self._refresh_adi(); self._draw_snr(); self._draw_contrast()
        self._fill_candidates(); self.tabs.setCurrentIndex(1)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _refresh_preview(self):
        if not self._paths: return
        idx = self.sp_frame_idx.value()
        try:
            d = fits.getdata(str(self._paths[idx]))
            self._draw_single_frame(luminance(d), idx)
        except Exception: pass

    def _draw_single_frame(self, img, idx):
        fig = self.prev_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, f"Frame {idx}  ({img.shape[1]}×{img.shape[0]})")
        lo,hi = np.percentile(img,[0.5,99.5])
        ax.imshow(img, origin='lower', cmap='inferno',
                  vmin=lo, vmax=hi, aspect='equal', interpolation='nearest')
        self.prev_plot.redraw()

    def _stretch(self, img):
        lo,hi = np.percentile(img[np.isfinite(img)],[0.5,99.8])
        c = np.clip(img-lo, 0, hi-lo+1e-10)
        m = self.cmb_stretch.currentIndex()
        if m == 0: return c/(hi-lo+1e-10)
        if m == 1: return np.sqrt(c/(hi-lo+1e-10))
        if m == 2: return np.log1p(c)/np.log1p(hi-lo+1)
        return np.arcsinh(c/max((hi-lo)*0.1,1e-10))/np.arcsinh(10)

    def _refresh_adi(self):
        if not self._result: return
        r = self._result
        fig = self.adi_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, f"ADI image  ({r['method']}  N={r['N_fr']})")
        img = self._stretch(r['adi'])
        ax.imshow(img, origin='lower', cmap='RdBu_r',
                  aspect='equal', interpolation='nearest')
        # Mark candidates
        ny, nx = r['adi'].shape
        for c in r['candidates']:
            ax.plot(c['x'], c['y'], 'o', ms=10, mec='#ffff30',
                    mfc='none', mew=1.5)
            ax.annotate(f"SNR={c['snr']:.1f}", (c['x'], c['y']),
                        color='#ffff60', fontsize=7,
                        xytext=(5,5), textcoords='offset points')
        self.adi_plot.redraw()

    def _draw_snr(self):
        if not self._result: return
        r = self._result
        fig = self.snr_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "S/N map  (Mawet et al. 2014 annular noise)")
        snr_clip = np.clip(r['snr'], -5, 10)
        im = ax.imshow(snr_clip, origin='lower', cmap='RdYlGn',
                       vmin=-3, vmax=8, aspect='equal', interpolation='nearest')
        div = make_axes_locatable(ax)
        cax = div.append_axes("right",size="4%",pad=0.05)
        cb  = fig.colorbar(im,cax=cax)
        cb.ax.yaxis.set_tick_params(colors='#7070a0',labelsize=7)
        cb.set_label("S/N", color='#7070a0', fontsize=7)
        # Threshold contour
        try:
            ax.contour(snr_clip, levels=[self.sp_snrth.value()],
                       colors='#ffff30', linewidths=0.8, alpha=0.6)
        except Exception: pass
        # Candidates
        for c in r['candidates']:
            ax.plot(c['x'],c['y'],'*', ms=12, color='#ffff00', zorder=5)
        self.snr_plot.redraw()

    def _draw_contrast(self):
        if not self._result: return
        r = self._result; cc = r['contrast']
        fwhm = r['fwhm']
        fig = self.cc_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, f"5-sigma contrast curve  ({r['method']})",
            "Separation (FWHM)", "5σ Contrast (flux ratio)")
        if len(cc['sep_px']) > 0:
            sep_fwhm = cc['sep_px'] / fwhm
            ax.semilogy(sep_fwhm, cc['contrast'], c='#5070ff', lw=2)
            ax.fill_between(sep_fwhm, cc['contrast'],
                            np.ones_like(cc['contrast']), alpha=0.1, color='#5070ff')
            # Mark candidates
            ny,nx = r['adi'].shape
            for c in r['candidates']:
                ax.axvline(c['sep_px']/fwhm, color='#ff8030', lw=1, ls='--',
                           label=f"Candidate SNR={c['snr']:.1f}")
            if r['candidates']:
                ax.legend(fontsize=7, facecolor='#1a1a30',
                          edgecolor='#3030a0', labelcolor='white')
        self.cc_plot.redraw()

    def _fill_candidates(self):
        if not self._result: return
        cands = self._result['candidates']
        self.cand_table.setRowCount(len(cands))
        for row, c in enumerate(cands):
            snr_s = c['snr']
            note = "STRONG" if snr_s>7 else ("MARGINAL" if snr_s>5 else "WEAK")
            vals = [str(row+1), f"{c['sep_px']:.1f}", f"{c['pa_deg']:.1f}",
                    f"{snr_s:.2f}", note]
            for col,v in enumerate(vals):
                item = QTableWidgetItem(v)
                if col==3:
                    from PyQt6.QtGui import QColor
                    c_col = '#ff8030' if snr_s>7 else '#ffff60' if snr_s>5 else '#a0a0a0'
                    item.setForeground(QColor(c_col))
                self.cand_table.setItem(row,col,item)
        self.cand_table.resizeColumnsToContents()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ADIApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
