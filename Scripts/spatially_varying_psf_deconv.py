r"""
HYPERLOAD — Spatially Varying PSF Deconvolution
=================================================
Position-dependent Richardson-Lucy deconvolution for wide-field images.

Every existing deconvolution tool (blind_deconv, imagemm, phase_diversity)
uses a SINGLE PSF for the whole frame. For modern full-frame sensors
(ASI2600, QHY268, IMX455) and refractors with field curvature, the PSF
in the corners can be 2–3× broader than at centre. A single PSF produces
under-deconvolved corners and over-deconvolved centre.

This script reads the psf_heatmap.py output (a per-position FWHM map)
and applies a position-dependent Richardson-Lucy deconvolution via tiling:
  1. Divide image into overlapping tiles
  2. Each tile uses the local PSF FWHM (median over the tile)
  3. Run standard RL deconvolution within each tile
  4. Blend tiles with Hann-window apodisation (no seam artifacts)

Optionally accepts an externally measured PSF FITS stack (one PSF per
grid position, as exported by psf_heatmap.py) instead of a Gaussian model.

Import API:
    from spatially_varying_psf_deconv import sv_rl_deconv, sv_rl_deconv_psf_stack

Reads from: psf_heatmap.py (FWHM map or PSF stack export)

References:
    Lucy 1974, AJ 79, 745              — Richardson-Lucy algorithm
    Richardson 1972, JOSA 62, 55       — original formulation
    Nagy & O'Leary 1998, SIAM J.Sci.  — spatially variant deblurring
    Lauer 2002, PASP 114, 1173        — space-variant PSF in astronomy

Version: 1.0.0
Project: HYPERLOAD
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

# ── crash logger ──────────────────────────────────────────────────────────────
def _crash(exc_type, exc_val, exc_tb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nspatially_varying_psf_deconv.py\n")
        traceback.print_exception(exc_type, exc_val, exc_tb, file=f)
    sys.__excepthook__(exc_type, exc_val, exc_tb)


sys.excepthook = _crash

try:
    import sirilpy as s
    s.ensure_installed("PyQt6")
    s.ensure_installed("numpy")
    s.ensure_installed("scipy")
    s.ensure_installed("matplotlib")
    s.ensure_installed("astropy")
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
VERSION = "1.0.0"
APP_TITLE = "HYPERLOAD — Spatially Varying PSF Deconvolution"
ACCENT = "#40d0a0"
ACCENT_DARK = "#1a7050"

DARK = """
QMainWindow,QWidget{background:#0e1118;color:#d0d8e8;
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}
QTabWidget::pane{border:1px solid #1e2840;background:#111828;}
QTabBar::tab{background:#131a28;color:#303850;padding:6px 16px;
    border:1px solid #1e2840;border-bottom:none;}
QTabBar::tab:selected{background:#111828;color:#40d0a0;
    border-bottom:2px solid #40d0a0;}
QGroupBox{border:1px solid #1e2840;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#303850;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#131a28;color:#c0d0e8;border:1px solid #1e2840;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#1a7050;border-color:#40d0a0;color:white;}
QPushButton:disabled{color:#283040;border-color:#131a28;}
QPushButton#run{background:#1a7050;border-color:#40d0a0;
    color:white;font-weight:bold;padding:7px 20px;}
QPushButton#run:hover{background:#40d0a0;}
QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{background:#0e1420;
    border:1px solid #1e2840;border-radius:3px;padding:3px 6px;color:#c0d0e8;}
QCheckBox::indicator{width:14px;height:14px;border:1px solid #1e2840;
    background:#131a28;border-radius:2px;}
QCheckBox::indicator:checked{background:#40d0a0;}
QTextEdit{background:#080c14;border:1px solid #1e2840;
    color:#40c080;font-family:Consolas,monospace;font-size:10px;}
QProgressBar{background:#0e1420;border:1px solid #1e2840;
    border-radius:3px;height:8px;}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #1a7050,stop:1 #40d0a0);border-radius:2px;}
QSplitter::handle{background:#1e2840;}
QLabel#title_lbl{font-size:15px;font-weight:bold;color:#40d0a0;}
QLabel#sub_lbl{font-size:10px;color:#505868;}
QLabel#summary_lbl{font-family:Consolas,monospace;font-size:10px;color:#40d0a0;}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


def rl_deconv_gaussian(
    image: np.ndarray,
    psf_sigma: float,
    n_iter: int = 15,
    reg_eps: float = 1e-6,
) -> np.ndarray:
    """Standard Richardson-Lucy deconvolution with Gaussian PSF."""
    est = np.maximum(image.astype(float), reg_eps)
    for _ in range(n_iter):
        conv = gaussian_filter(est, psf_sigma)
        ratio = image / (conv + reg_eps)
        corr = gaussian_filter(ratio, psf_sigma)
        est = np.maximum(est * corr, reg_eps)
    return est


def rl_deconv_psf(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 15,
    reg_eps: float = 1e-6,
) -> np.ndarray:
    """Richardson-Lucy deconvolution with arbitrary PSF."""
    psf_n = psf / (psf.sum() + 1e-30)
    psf_flp = psf_n[::-1, ::-1]
    est = np.maximum(image.astype(float), reg_eps)
    for _ in range(n_iter):
        conv = np.real(
            np.fft.ifft2(np.fft.fft2(est) * np.fft.fft2(psf_n, s=est.shape))
        )
        ratio = image / (conv + reg_eps)
        corr = np.real(
            np.fft.ifft2(np.fft.fft2(ratio) * np.fft.fft2(psf_flp, s=est.shape))
        )
        est = np.maximum(est * corr, reg_eps)
    return est


def sv_rl_deconv(
    image: np.ndarray,
    fwhm_map: np.ndarray,
    n_iter: int = 15,
    tile_size: int = 128,
    overlap: int = 32,
    reg_eps: float = 1e-6,
    progress_cb=None,
) -> np.ndarray:
    """Spatially varying Richardson-Lucy deconvolution."""
    H, W = image.shape
    result = np.zeros((H, W), dtype=float)
    weight = np.zeros((H, W), dtype=float)

    ts_win = tile_size + 2 * overlap
    win_1d = np.hanning(ts_win)
    win_2d = np.outer(win_1d, win_1d)

    n_tiles_y = max(1, (H + tile_size - 1) // tile_size)
    n_tiles_x = max(1, (W + tile_size - 1) // tile_size)
    n_total = n_tiles_y * n_tiles_x
    tile_idx = 0

    for ty in range(0, H, tile_size):
        for tx in range(0, W, tile_size):
            y0 = max(0, ty - overlap)
            y1 = min(H, ty + tile_size + overlap)
            x0 = max(0, tx - overlap)
            x1 = min(W, tx + tile_size + overlap)

            tile = image[y0:y1, x0:x1].astype(float)
            fwhm_tile = float(np.median(fwhm_map[y0:y1, x0:x1]))
            sig_tile = max(fwhm_tile / 2.355, 0.3)

            tile_deconv = rl_deconv_gaussian(tile, sig_tile, n_iter, reg_eps)

            iy = ty - y0
            ix = tx - x0
            h2 = min(tile_size, H - ty)
            w2 = min(tile_size, W - tx)

            wc = win_2d[iy : iy + h2, ix : ix + w2]
            result[ty : ty + h2, tx : tx + w2] += tile_deconv[iy : iy + h2, ix : ix + w2] * wc
            weight[ty : ty + h2, tx : tx + w2] += wc

            tile_idx += 1
            if progress_cb and tile_idx % max(1, n_total // 10) == 0:
                progress_cb(int(90 * tile_idx / n_total), f"Tile {tile_idx}/{n_total}…")

    return np.where(weight > 0.01, result / weight, image.astype(float))


def sv_rl_deconv_psf_stack(
    image: np.ndarray,
    psf_stack: np.ndarray,
    psf_grid_xy: list[tuple[float, float]],
    n_iter: int = 15,
    tile_size: int = 128,
    overlap: int = 32,
    progress_cb=None,
) -> np.ndarray:
    """Spatially varying RL using measured PSF stack."""
    from scipy.spatial import cKDTree

    H, W = image.shape
    result = np.zeros((H, W), dtype=float)
    weight = np.zeros((H, W), dtype=float)
    ts_win = tile_size + 2 * overlap
    win_2d = np.outer(np.hanning(ts_win), np.hanning(ts_win))

    grid_pts = np.array(psf_grid_xy)
    tree = cKDTree(grid_pts)

    n_tiles_y = max(1, (H + tile_size - 1) // tile_size)
    n_tiles_x = max(1, (W + tile_size - 1) // tile_size)
    n_total = n_tiles_y * n_tiles_x
    tile_idx = 0

    for ty in range(0, H, tile_size):
        for tx in range(0, W, tile_size):
            y0 = max(0, ty - overlap)
            y1 = min(H, ty + tile_size + overlap)
            x0 = max(0, tx - overlap)
            x1 = min(W, tx + tile_size + overlap)
            tile = image[y0:y1, x0:x1].astype(float)

            cx_tile = tx + min(tile_size, W - tx) / 2
            cy_tile = ty + min(tile_size, H - ty) / 2
            _, psf_idx = tree.query([cx_tile, cy_tile])
            psf = psf_stack[psf_idx]

            tile_deconv = rl_deconv_psf(tile, psf, n_iter)

            iy = ty - y0
            ix = tx - x0
            h2 = min(tile_size, H - ty)
            w2 = min(tile_size, W - tx)
            wc = win_2d[iy : iy + h2, ix : ix + w2]
            result[ty : ty + h2, tx : tx + w2] += (
                tile_deconv[iy : iy + h2, ix : ix + w2] * wc
            )
            weight[ty : ty + h2, tx : tx + w2] += wc

            tile_idx += 1
            if progress_cb and tile_idx % max(1, n_total // 10) == 0:
                progress_cb(int(90 * tile_idx / n_total), f"Tile {tile_idx}/{n_total}…")

    return np.where(weight > 0.01, result / weight, image.astype(float))


def build_gaussian_fwhm_map(
    shape: tuple, fwhm_centre: float, fwhm_corner: float
) -> np.ndarray:
    """Synthetic radial FWHM map (centre to corner)."""
    H, W = shape
    yy, xx = np.mgrid[:H, :W].astype(float)
    cx, cy = W / 2, H / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_d = np.sqrt(cx**2 + cy**2)
    return fwhm_centre + (fwhm_corner - fwhm_centre) * (dist / max_d)


def build_fwhm_map_pattern(
    shape: tuple,
    fwhm_centre: float,
    fwhm_corner: float,
    pattern: str = "Radial (circular)",
) -> np.ndarray:
    """Extended synthetic FWHM maps for GUI builder."""
    H, W = shape
    yy, xx = np.mgrid[:H, :W].astype(float)
    cx, cy = W / 2, H / 2

    if pattern == "Uniform":
        return np.full((H, W), fwhm_centre, dtype=float)

    if pattern.startswith("Radial"):
        return build_gaussian_fwhm_map(shape, fwhm_centre, fwhm_corner)

    if pattern.startswith("Astigmatism"):
        dx = (xx - cx) / max(cx, 1)
        dy = (yy - cy) / max(cy, 1)
        fwhm = fwhm_centre + (fwhm_corner - fwhm_centre) * (
            0.6 * np.abs(dx) + 0.4 * np.abs(dy)
        )
        return fwhm.astype(float)

    # Coma — one corner worse
    corner_x, corner_y = W * 0.95, H * 0.05
    dist = np.sqrt((xx - corner_x) ** 2 + (yy - corner_y) ** 2)
    max_d = np.sqrt(W**2 + H**2)
    return (fwhm_centre + (fwhm_corner - fwhm_centre) * (dist / max_d)).astype(float)


def load_fwhm_map_from_psf_heatmap(fits_path: str) -> np.ndarray | None:
    """Load FWHM map exported by psf_heatmap.py."""
    try:
        from astropy.io import fits as f

        data = f.getdata(fits_path).astype(float)
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            return data[0]
        return None
    except Exception:
        return None


def load_fits_2d(path: str) -> np.ndarray:
    """Load 2D frame; 3D RGB → luminance."""
    from astropy.io import fits as af

    with af.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if data.shape[0] in (3, 4):
            planes = data[:3]
        elif data.shape[-1] in (3, 4):
            planes = np.moveaxis(data[..., :3], -1, 0)
        else:
            return np.asarray(data[0], dtype=np.float64)
        r, g, b = planes[0], planes[1], planes[2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    raise ValueError(f"Unsupported FITS shape {data.shape}")


def load_psf_stack_fits(path: str) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Load PSF stack and grid positions from FITS header keywords."""
    from astropy.io import fits as af

    with af.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
    if data.ndim != 3:
        raise ValueError("PSF stack must be 3D (N, P, P)")
    n_psf = data.shape[0]
    grid = []
    for i in range(n_psf):
        xk = f"PSF_X_{i}"
        yk = f"PSF_Y_{i}"
        if xk in hdr and yk in hdr:
            grid.append((float(hdr[xk]), float(hdr[yk])))
        else:
            # fallback: regular grid guess
            side = int(np.ceil(np.sqrt(n_psf)))
            row, col = divmod(i, side)
            grid.append((float(col), float(row)))
    return data, grid


def make_synthetic_starfield(seed: int = 42) -> dict:
    """Validation synthetic image + FWHM map."""
    rng = np.random.default_rng(seed)
    n = 256
    fwhm_centre = 3.0
    fwhm_corner = 6.0
    fwhm_map = build_gaussian_fwhm_map((n, n), fwhm_centre, fwhm_corner)
    star_positions = [
        (50, 50),
        (50, 200),
        (200, 50),
        (200, 200),
        (128, 128),
        (80, 170),
        (170, 80),
    ]
    blurred = np.zeros((n, n))
    for sy, sx in star_positions:
        fwhm_loc = float(fwhm_map[sy, sx])
        sig_loc = fwhm_loc / 2.355
        stamp = np.zeros((n, n))
        stamp[sy, sx] = 1000.0
        blurred += gaussian_filter(stamp, sig_loc)
    blurred += rng.normal(0, 3, (n, n))
    return {
        "image": blurred,
        "fwhm_map": fwhm_map,
        "star_positions": star_positions,
        "filename": "synthetic_starfield.fits",
    }


def star_peak_ratio(
    before: np.ndarray, after: np.ndarray, positions: list[tuple[int, int]]
) -> float:
    peaks_before = [float(before[sy, sx]) for sy, sx in positions]
    peaks_after = [float(after[sy, sx]) for sy, sx in positions]
    return float(np.mean(peaks_after) / (np.mean(peaks_before) + 1e-10))


def _validate_algorithms() -> None:
    import time

    rng = np.random.default_rng(42)
    n = 256
    fwhm_centre = 3.0
    fwhm_corner = 6.0
    fwhm_map = build_gaussian_fwhm_map((n, n), fwhm_centre, fwhm_corner)
    star_positions = [
        (50, 50),
        (50, 200),
        (200, 50),
        (200, 200),
        (128, 128),
        (80, 170),
        (170, 80),
    ]
    blurred = np.zeros((n, n))
    for sy, sx in star_positions:
        fwhm_loc = float(fwhm_map[sy, sx])
        sig_loc = fwhm_loc / 2.355
        stamp = np.zeros((n, n))
        stamp[sy, sx] = 1000.0
        blurred += gaussian_filter(stamp, sig_loc)
    blurred += rng.normal(0, 3, (n, n))

    t0 = time.time()
    deconvolved = sv_rl_deconv(
        blurred, fwhm_map, n_iter=10, tile_size=64, overlap=16
    )
    dt = time.time() - t0
    ratio = star_peak_ratio(blurred, deconvolved, star_positions)
    if ratio <= 1.2:
        raise RuntimeError(f"Peak flux ratio too low: {ratio:.2f}")

    uniform_map = np.full((n, n), 4.0)
    deconv_sv = sv_rl_deconv(blurred, uniform_map, n_iter=8, tile_size=64)
    deconv_std = rl_deconv_gaussian(blurred, 4.0 / 2.355, n_iter=8)
    diff = float(np.abs(deconv_sv - deconv_std).mean()) / (float(blurred.mean()) + 1)
    if diff >= 0.15:
        raise RuntimeError(f"Uniform map mismatch: {diff:.3f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--validate":
    _validate_algorithms()
    print("SV-PSF DECONV VALIDATED OK")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PyQt6 GUI
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8, 5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor="#0c1018")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def redraw(self):
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()


def _ax(ax, title="", xl="", yl="", accent=ACCENT):
    ax.set_facecolor("#080c14")
    ax.set_title(title, color=accent, fontsize=9, pad=3)
    ax.set_xlabel(xl, color="#303850", fontsize=8)
    ax.set_ylabel(yl, color="#303850", fontsize=8)
    ax.tick_params(colors="#303850", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2840")


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#080c14;border-top:1px solid #1e2840;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        self.lbl = QLabel("Ready")
        self.lbl.setStyleSheet("color:#283040;font-size:10px;")
        self.pbar = QProgressBar()
        self.pbar.setFixedSize(200, 12)
        self.pbar.setVisible(False)
        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.pbar)

    def update(self, pct, msg):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True)
            self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))


class FwhmMapBuilderDialog(QDialog):
    """Non-modal dialog to build synthetic FWHM map."""

    map_ready = pyqtSignal(object)

    def __init__(self, shape: tuple, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build synthetic FWHM map")
        self.setMinimumWidth(360)
        self._shape = shape
        lay = QVBoxLayout(self)

        form = QFormLayout()
        self.sp_centre = QDoubleSpinBox()
        self.sp_centre.setRange(0.5, 20)
        self.sp_centre.setValue(3.0)
        self.sp_corner = QDoubleSpinBox()
        self.sp_corner.setRange(0.5, 30)
        self.sp_corner.setValue(6.0)
        self.cmb_pattern = QComboBox()
        self.cmb_pattern.addItems(
            [
                "Radial (circular)",
                "Astigmatism (oval)",
                "Coma (one-sided)",
                "Uniform",
            ]
        )
        form.addRow("Centre FWHM (px):", self.sp_centre)
        form.addRow("Corner FWHM (px):", self.sp_corner)
        form.addRow("Pattern:", self.cmb_pattern)
        lay.addLayout(form)

        self.preview = MplCanvas(figsize=(3, 2))
        self.preview.setFixedHeight(150)
        lay.addWidget(self.preview)

        for w in (self.sp_centre, self.sp_corner):
            w.valueChanged.connect(self._update_preview)
        self.cmb_pattern.currentIndexChanged.connect(self._update_preview)

        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply)
        lay.addWidget(btn_apply)
        self._update_preview()

    def _current_map(self) -> np.ndarray:
        return build_fwhm_map_pattern(
            self._shape,
            self.sp_centre.value(),
            self.sp_corner.value(),
            self.cmb_pattern.currentText(),
        )

    def _update_preview(self):
        fmap = self._current_map()
        fig = self.preview.fig
        fig.clear()
        ax = fig.add_subplot(111)
        im = ax.imshow(fmap, origin="lower", cmap="inferno")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _ax(ax, "FWHM preview", "X", "Y")
        self.preview.redraw()

    def _apply(self):
        self.map_ready.emit(self._current_map())
        self.accept()


class SVDeconvWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            import time

            p = self.params
            img = p["image"].astype(float)
            self.progress.emit(2, "Starting deconvolution…")

            def pcb(pct, msg):
                self.progress.emit(pct, msg)

            t0 = time.time()
            mode = p["mode"]
            if mode == "stack":
                out = sv_rl_deconv_psf_stack(
                    img,
                    p["psf_stack"],
                    p["psf_grid"],
                    n_iter=p["n_iter"],
                    tile_size=p["tile_size"],
                    overlap=p["overlap"],
                    progress_cb=pcb,
                )
            else:
                out = sv_rl_deconv(
                    img,
                    p["fwhm_map"],
                    n_iter=p["n_iter"],
                    tile_size=p["tile_size"],
                    overlap=p["overlap"],
                    reg_eps=p["reg_eps"],
                    progress_cb=pcb,
                )
            elapsed = time.time() - t0

            if p.get("clip_negative"):
                out = np.maximum(out, 0)
            if p.get("preserve_sky"):
                sky_in = float(np.median(img))
                sky_out = float(np.median(out))
                out = out - sky_out + sky_in

            ratio = 1.0
            if p.get("star_positions"):
                ratio = star_peak_ratio(img, out, p["star_positions"])

            self.progress.emit(100, "Deconvolution complete")
            self.finished.emit(
                {
                    "deconvolved": out,
                    "elapsed": elapsed,
                    "peak_ratio": ratio,
                    "input": img,
                }
            )
        except Exception as ex:
            self.error.emit(f"{ex}\n\n{traceback.format_exc()}")


class SVPSFDeconvWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 920)
        self.setStyleSheet(DARK)
        logo = SCRIPT_DIR / "logo.png"
        if logo.exists():
            self.setWindowIcon(QIcon(str(logo)))

        self._image: np.ndarray | None = None
        self._fwhm_map: np.ndarray | None = None
        self._psf_stack = None
        self._psf_grid: list[tuple[float, float]] = []
        self._deconvolved: np.ndarray | None = None
        self._filename = ""
        self._star_positions: list[tuple[int, int]] = []
        self._worker = None
        self._wthread = None
        self._fwhm_dialog: FwhmMapBuilderDialog | None = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setSpacing(0)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self._make_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._make_controls())
        splitter.addWidget(self._make_tabs())
        splitter.setSizes([280, 1120])
        rl.addWidget(splitter, 1)

        self.status = StatusFooter()
        rl.addWidget(self.status)

    def _make_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setFixedHeight(66)
        hdr.setStyleSheet("background:#080c14;border-bottom:1px solid #1e2840;")
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(12, 0, 12, 0)
        logo = SCRIPT_DIR / "logo.png"
        if logo.exists():
            lbl = QLabel()
            lbl.setPixmap(
                QPixmap(str(logo)).scaled(
                    50,
                    50,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            lay.addWidget(lbl)
        v = QVBoxLayout()
        t = QLabel("Spatially Varying PSF Deconvolution")
        t.setObjectName("title_lbl")
        s = QLabel("Lucy 1974 · Nagy & O'Leary 1998 · tiled Hann-blended RL")
        s.setObjectName("sub_lbl")
        v.addWidget(t)
        v.addWidget(s)
        lay.addLayout(v, 1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _make_controls(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        grp = QGroupBox("Image")
        gl = QVBoxLayout(grp)
        btn_load = QPushButton("📂 Load FITS…")
        btn_load.clicked.connect(self._load_fits)
        btn_syn = QPushButton("⚗ Synthetic example")
        btn_syn.clicked.connect(self._load_synthetic)
        gl.addWidget(btn_load)
        gl.addWidget(btn_syn)
        self.lbl_file = QLabel("No image loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#505868;font-size:10px;")
        gl.addWidget(self.lbl_file)
        lay.addWidget(grp)

        grp_psf = QGroupBox("PSF Source")
        vl = QVBoxLayout(grp_psf)
        self.cmb_psf_mode = QComboBox()
        self.cmb_psf_mode.addItems(
            [
                "Gaussian model (from FWHM map)",
                "Measured PSF stack (FITS)",
                "Uniform Gaussian (single FWHM)",
            ]
        )
        self.cmb_psf_mode.currentIndexChanged.connect(self._psf_mode_changed)
        vl.addWidget(self.cmb_psf_mode)

        self.stack_psf = QStackedWidget()
        # Page 0: FWHM map
        w0 = QWidget()
        l0 = QVBoxLayout(w0)
        btn_fwhm = QPushButton("📂 Load FWHM map FITS (psf_heatmap output)…")
        btn_fwhm.clicked.connect(self._load_fwhm_fits)
        btn_build = QPushButton("📐 Build synthetic FWHM map")
        btn_build.clicked.connect(self._open_fwhm_builder)
        l0.addWidget(btn_fwhm)
        l0.addWidget(btn_build)
        self.lbl_fwhm_range = QLabel("FWHM range: —")
        self.lbl_fwhm_range.setStyleSheet("color:#505868;font-size:10px;")
        l0.addWidget(self.lbl_fwhm_range)
        self.stack_psf.addWidget(w0)

        # Page 1: PSF stack
        w1 = QWidget()
        l1 = QVBoxLayout(w1)
        btn_stack = QPushButton("📂 Load PSF stack FITS…")
        btn_stack.clicked.connect(self._load_psf_stack)
        l1.addWidget(btn_stack)
        self.lbl_stack = QLabel("N_psf PSFs, P×P px each")
        self.lbl_stack.setStyleSheet("color:#505868;font-size:10px;")
        l1.addWidget(self.lbl_stack)
        self.stack_psf.addWidget(w1)

        # Page 2: uniform
        w2 = QWidget()
        f2 = QFormLayout(w2)
        self.sp_uniform_fwhm = QDoubleSpinBox()
        self.sp_uniform_fwhm.setRange(0.5, 20)
        self.sp_uniform_fwhm.setValue(3.0)
        self.sp_uniform_fwhm.valueChanged.connect(self._on_uniform_fwhm_changed)
        f2.addRow("FWHM (px):", self.sp_uniform_fwhm)
        hint = QLabel("Equivalent to single-PSF RL deconvolution")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#505868;font-size:9px;")
        f2.addRow(hint)
        self.stack_psf.addWidget(w2)

        vl.addWidget(self.stack_psf)
        lay.addWidget(grp_psf)

        grp_d = QGroupBox("Deconvolution Parameters")
        fd = QFormLayout(grp_d)
        self.sp_iter = QSpinBox()
        self.sp_iter.setRange(3, 50)
        self.sp_iter.setValue(15)
        self.sp_tile = QSpinBox()
        self.sp_tile.setRange(32, 256)
        self.sp_tile.setSingleStep(32)
        self.sp_tile.setValue(128)
        self.sp_overlap = QSpinBox()
        self.sp_overlap.setRange(16, 128)
        self.sp_overlap.setValue(32)
        self.sp_eps = QDoubleSpinBox()
        self.sp_eps.setDecimals(8)
        self.sp_eps.setRange(1e-8, 1e-3)
        self.sp_eps.setValue(1e-6)
        fd.addRow("RL iterations:", self.sp_iter)
        fd.addRow("Tile size (px):", self.sp_tile)
        fd.addRow("Tile overlap (px):", self.sp_overlap)
        fd.addRow("Regularisation ε:", self.sp_eps)
        lay.addWidget(grp_d)

        grp_o = QGroupBox("Output")
        fo = QFormLayout(grp_o)
        self.chk_clip = QCheckBox("Clip negative values to 0")
        self.chk_clip.setChecked(True)
        self.chk_sky = QCheckBox("Preserve original sky level")
        self.chk_sky.setChecked(True)
        self.cmb_stretch = QComboBox()
        self.cmb_stretch.addItems(["Linear", "Log", "Sqrt"])
        fo.addRow(self.chk_clip)
        fo.addRow(self.chk_sky)
        fo.addRow("Preview scale:", self.cmb_stretch)
        lay.addWidget(grp_o)

        self.btn_run = QPushButton("▶ Deconvolve")
        self.btn_run.setObjectName("run")
        self.btn_run.setFixedHeight(38)
        self.btn_run.clicked.connect(self._run_deconv)
        lay.addWidget(self.btn_run)

        self.btn_export = QPushButton("💾 Export deconvolved FITS…")
        self.btn_export.clicked.connect(self._export_fits)
        lay.addWidget(self.btn_export)

        self.lbl_summary = QLabel("Sharpness improvement: —")
        self.lbl_summary.setObjectName("summary_lbl")
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)

        lay.addStretch()
        scroll.setWidget(inner)
        self._psf_mode_changed(0)
        return scroll

    def _make_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        self.canvas_prev = MplCanvas(figsize=(9, 6))
        self.canvas_map = MplCanvas(figsize=(9, 5))
        self.canvas_cmp = MplCanvas(figsize=(9, 6))
        self.canvas_qual = MplCanvas(figsize=(9, 5))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        tabs.addTab(self.canvas_prev, "🖼 Preview")
        tabs.addTab(self.canvas_map, "🗺 PSF Map")
        tabs.addTab(self.canvas_cmp, "🔬 Comparison")
        tabs.addTab(self.canvas_qual, "📈 Quality Metrics")
        tabs.addTab(self.log, "📋 Log")
        return tabs

    def _psf_mode_changed(self, idx: int):
        self.stack_psf.setCurrentIndex(idx)
        if self._image is not None and idx == 2:
            self._set_uniform_fwhm_map()

    def _on_uniform_fwhm_changed(self):
        if self.cmb_psf_mode.currentIndex() == 2:
            self._set_uniform_fwhm_map()

    def _log(self, msg: str):
        self.log.append(msg)
        self.status.update(-1, msg)

    def _update_fwhm_label(self):
        if self._fwhm_map is None:
            self.lbl_fwhm_range.setText("FWHM range: —")
            return
        lo, hi = float(self._fwhm_map.min()), float(self._fwhm_map.max())
        self.lbl_fwhm_range.setText(f"FWHM range: {lo:.1f} – {hi:.1f} px")

    def _set_uniform_fwhm_map(self):
        if self._image is None:
            return
        fwhm = self.sp_uniform_fwhm.value()
        self._fwhm_map = np.full(self._image.shape, fwhm)
        self._update_fwhm_label()
        self._plot_psf_map()

    def _load_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FITS", str(SCRIPT_DIR), "FITS (*.fits *.fit *.fts)"
        )
        if not path:
            return
        try:
            img = load_fits_2d(path)
            self._image = img
            self._filename = Path(path).name
            self.lbl_file.setText(f"{self._filename}\n{img.shape[1]}×{img.shape[0]}  {img.dtype}")
            self._log(f"Loaded {self._filename}")
            if self.cmb_psf_mode.currentIndex() == 2:
                self._set_uniform_fwhm_map()
            elif self._fwhm_map is None or self._fwhm_map.shape != img.shape:
                self._fwhm_map = build_gaussian_fwhm_map(img.shape, 3.0, 6.0)
                self._update_fwhm_label()
        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))

    def _load_synthetic(self):
        syn = make_synthetic_starfield()
        self._image = syn["image"]
        self._fwhm_map = syn["fwhm_map"]
        self._star_positions = syn["star_positions"]
        self._filename = syn["filename"]
        h, w = self._image.shape
        self.lbl_file.setText(f"{self._filename}\n{w}×{h}")
        self._update_fwhm_label()
        self._log("Synthetic starfield loaded (varying PSF)")

    def _load_fwhm_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FWHM map", str(SCRIPT_DIR), "FITS (*.fits)"
        )
        if not path:
            return
        fmap = load_fwhm_map_from_psf_heatmap(path)
        if fmap is None:
            QMessageBox.warning(self, "FWHM map", "Could not read FWHM map FITS.")
            return
        if self._image is not None and fmap.shape != self._image.shape:
            QMessageBox.warning(
                self,
                "FWHM map",
                f"Shape mismatch: map {fmap.shape} vs image {self._image.shape}",
            )
            return
        self._fwhm_map = fmap
        self._update_fwhm_label()
        self._log(f"FWHM map loaded: {Path(path).name}")
        self._plot_psf_map()

    def _open_fwhm_builder(self):
        if self._image is None:
            QMessageBox.warning(self, "FWHM map", "Load an image first.")
            return
        dlg = FwhmMapBuilderDialog(self._image.shape, self)
        dlg.map_ready.connect(self._on_fwhm_built)
        dlg.show()

    def _on_fwhm_built(self, fmap: np.ndarray):
        self._fwhm_map = fmap
        self._update_fwhm_label()
        self._plot_psf_map()
        self._log("Synthetic FWHM map applied")

    def _load_psf_stack(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load PSF stack", str(SCRIPT_DIR), "FITS (*.fits)"
        )
        if not path:
            return
        try:
            stack, grid = load_psf_stack_fits(path)
            self._psf_stack = stack
            self._psf_grid = grid
            p = stack.shape[-1]
            self.lbl_stack.setText(f"{stack.shape[0]} PSFs, {p}×{p} px each")
            self._log(f"PSF stack: {Path(path).name}")
        except Exception as ex:
            QMessageBox.critical(self, "PSF stack", str(ex))

    def _get_deconv_params(self) -> dict | None:
        if self._image is None:
            return None
        mode_idx = self.cmb_psf_mode.currentIndex()
        if mode_idx == 0:
            if self._fwhm_map is None:
                return None
            mode = "map"
        elif mode_idx == 1:
            if self._psf_stack is None:
                return None
            mode = "stack"
        else:
            self._set_uniform_fwhm_map()
            mode = "map"

        p = {
            "image": self._image,
            "mode": mode,
            "fwhm_map": self._fwhm_map,
            "psf_stack": self._psf_stack,
            "psf_grid": self._psf_grid,
            "n_iter": self.sp_iter.value(),
            "tile_size": self.sp_tile.value(),
            "overlap": self.sp_overlap.value(),
            "reg_eps": self.sp_eps.value(),
            "clip_negative": self.chk_clip.isChecked(),
            "preserve_sky": self.chk_sky.isChecked(),
            "star_positions": self._star_positions,
        }
        return p

    def _run_deconv(self):
        params = self._get_deconv_params()
        if params is None:
            QMessageBox.warning(self, "Deconvolve", "Load image and PSF data first.")
            return
        self.btn_run.setEnabled(False)
        self._worker = SVDeconvWorker(params)
        self._wthread = QThread(self)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda p, m: self.status.update(p, m))
        self._worker.finished.connect(self._on_deconv_done)
        self._worker.error.connect(self._on_deconv_error)
        self._worker.finished.connect(self._wthread.quit)
        self._worker.error.connect(self._wthread.quit)
        self._wthread.finished.connect(self._cleanup_thread)
        self._wthread.start()

    def _cleanup_thread(self):
        self.btn_run.setEnabled(True)
        if self._wthread:
            self._wthread.deleteLater()
            self._wthread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_deconv_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Deconvolution failed", msg[:500])
        self._cleanup_thread()

    def _on_deconv_done(self, data: dict):
        self._deconvolved = data["deconvolved"]
        ratio = data["peak_ratio"]
        elapsed = data["elapsed"]
        self.lbl_summary.setText(
            f"Sharpness improvement: {ratio:.2f}x\nTime: {elapsed:.1f}s"
        )
        self._log(f"Done in {elapsed:.1f}s  peak ratio={ratio:.2f}x")
        self._plot_all(data)
        self._cleanup_thread()

    def _export_fits(self):
        if self._deconvolved is None:
            QMessageBox.warning(self, "Export", "Run deconvolution first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export FITS",
            str(SCRIPT_DIR / "deconvolved.fits"),
            "FITS (*.fits)",
        )
        if not path:
            return
        try:
            from astropy.io import fits as af

            af.writeto(path, self._deconvolved.astype(np.float32), overwrite=True)
            self._log(f"Saved {path}")
            QMessageBox.information(self, "Export", f"Saved:\n{path}")
        except Exception as ex:
            QMessageBox.critical(self, "Export", str(ex))

    def _display_norm(self, data: np.ndarray):
        stretch = self.cmb_stretch.currentText()
        d = np.maximum(data, 1e-6)
        if stretch == "Log":
            return mcolors.LogNorm(vmin=np.percentile(d, 1), vmax=np.percentile(d, 99.5))
        if stretch == "Sqrt":
            return mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=np.percentile(d, 99.5))
        return None

    def _plot_all(self, data: dict):
        self._plot_preview(data)
        self._plot_psf_map()
        self._plot_comparison(data)
        self._plot_quality(data)

    def _plot_preview(self, data: dict):
        inp = data["input"]
        out = data["deconvolved"]
        fig = self.canvas_prev.fig
        fig.clear()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        norm = self._display_norm(inp)
        ax1.imshow(inp, origin="lower", cmap="gray", norm=norm)
        _ax(ax1, "Input", "X", "Y")
        ax2.imshow(out, origin="lower", cmap="gray", norm=norm)
        _ax(ax2, f"Deconvolved ({data['peak_ratio']:.2f}x)", "X", "Y")
        self.canvas_prev.redraw()

    def _plot_psf_map(self):
        if self._fwhm_map is None:
            return
        fig = self.canvas_map.fig
        fig.clear()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        im = ax1.imshow(self._fwhm_map, origin="lower", cmap="coolwarm")
        fig.colorbar(im, ax=ax1, label="FWHM (px)")
        _ax(ax1, "PSF FWHM map (px)", "X", "Y")

        h, w = self._fwhm_map.shape
        yy, xx = np.mgrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
        fwhm_r = self._fwhm_map.ravel()
        bins = np.linspace(0, dist.max(), 40)
        prof = [
            float(np.median(fwhm_r[(dist >= bins[i]) & (dist < bins[i + 1])]))
            for i in range(len(bins) - 1)
        ]
        r_mid = (bins[:-1] + bins[1:]) / 2
        ax2.plot(r_mid, prof, "-", color=ACCENT)
        _ax(ax2, "Radial FWHM profile", "Distance from centre (px)", "FWHM (px)")
        self.canvas_map.redraw()

    def _plot_comparison(self, data: dict):
        inp = data["input"]
        out = data["deconvolved"]
        h, w = inp.shape
        corners = [
            ("TL", 8, h - 40),
            ("TR", w - 40, h - 40),
            ("Centre", w // 2 - 16, h // 2 - 16),
            ("BL", 8, 8),
        ]
        fig = self.canvas_cmp.fig
        fig.clear()
        size = 32
        for i, (name, x0, y0) in enumerate(corners):
            ax = fig.add_subplot(2, 2, i + 1)
            sl_in = inp[y0 : y0 + size, x0 : x0 + size]
            sl_out = out[y0 : y0 + size, x0 : x0 + size]
            comb = np.zeros((size, size * 2))
            comb[:, :size] = sl_in
            comb[:, size:] = sl_out
            ax.imshow(comb, origin="lower", cmap="gray")
            fwhm_loc = float(self._fwhm_map[y0, x0]) if self._fwhm_map is not None else 0
            _ax(ax, f"{name}  FWHM≈{fwhm_loc:.1f}px", "", "")
        self.canvas_cmp.redraw()

    def _plot_quality(self, data: dict):
        inp = data["input"]
        out = data["deconvolved"]
        fig = self.canvas_qual.fig
        fig.clear()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        def power_spec(img):
            f = np.fft.fftshift(np.fft.fft2(img))
            return np.log10(np.abs(f) + 1e-10)

        ax1.plot(power_spec(inp).mean(axis=0), color="#606878", label="Input", lw=0.8)
        ax1.plot(power_spec(out).mean(axis=0), color=ACCENT, label="Deconvolved", lw=0.8)
        _ax(ax1, "Mean power spectrum (log10)", "Spatial freq bin", "Power")
        ax1.legend(fontsize=7, facecolor="#0c1018")

        from scipy.ndimage import laplace

        lap_in = np.abs(laplace(inp))
        lap_out = np.abs(laplace(out))
        h, w = inp.shape
        yy, xx = np.mgrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
        ratio = (lap_out / (lap_in + 1e-10)).ravel()
        bins = np.linspace(0, dist.max(), 30)
        prof = []
        for i in range(len(bins) - 1):
            m = (dist >= bins[i]) & (dist < bins[i + 1])
            prof.append(float(np.median(ratio[m])) if m.any() else 1.0)
        r_mid = (bins[:-1] + bins[1:]) / 2
        ax2.plot(r_mid, prof, "-", color=ACCENT)
        ax2.axhline(1, color="#303850", lw=0.5)
        _ax(ax2, "Laplacian sharpness ratio vs radius", "Radius (px)", "deconv / input")
        fig.text(
            0.5,
            0.01,
            f"Mean improvement: {data['peak_ratio']:.2f}x",
            ha="center",
            color=ACCENT,
            fontsize=8,
        )
        self.canvas_qual.redraw()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SVPSFDeconvWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
