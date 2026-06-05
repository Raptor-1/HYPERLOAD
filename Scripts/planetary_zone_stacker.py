r"""
planetary_zone_stacker.py  —  HYPERLOAD Planetary Zone Stacker
===============================================================
Zone-based (patch-based) lucky imaging stacker for planets, Moon and Sun.

Unlike simple global lucky imaging (AutoStakkert basic mode), this script:
  • Divides the planet disk into an adaptive grid of overlapping zones
  • Independently aligns each zone in every frame using sub-pixel
    phase-correlation (Guizar-Sicairos et al. 2008, Optics Letters)
  • Scores each zone in each frame using a per-zone quality metric
  • Selects the best N% of frames PER ZONE (not globally)
  • Stacks each zone independently using any HyperLoad algorithm
  • Feather-blends all zones into a seamless final image using
    Gaussian distance-weighted blending
  • Optional: Hybrid multi-scale mode (Wang, Li & Zhang 2021, RAA 21, 55)
    — low spatial frequencies selected in Fourier domain
    — high spatial frequencies selected in spatial domain
    — wavelet-combined for optimal resolution

This means a seeing spike that ruins the north pole of Jupiter can still
contribute a perfect south equatorial belt — impossible with global stacking.

References:
  Zone stacking:      PlanetarySystemStacker (Hempel 2020, github.com/Rolf-Hempel)
  Sub-pixel align:    Guizar-Sicairos, Thurman & Fienup 2008, Opt. Lett. 33, 156
  Hybrid LI:          Wang, Li & Zhang 2021, RAA 21(5), 55  (arXiv:2012.05480)
  Lucky Fourier:      Garrel, Guyon & Baudoz 2012, PASP 124, 861
  Phase correlation:  Kuglin & Hines 1975, IEEE ICASSP, 163
  Feather blending:   Szeliski 2006, IEEE TPAMI 28(9), 1409

Place in: Siril Suites folder
Run via:  Siril → Scripts → planetary_zone_stacker
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("scipy")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")

import os
import sys
import glob
import struct
import threading
import traceback
from datetime import datetime

import numpy as np
from scipy.ndimage import (gaussian_filter, map_coordinates,
                           uniform_filter, label, center_of_mass)
from scipy.optimize import curve_fit
from scipy.fft import fft2, ifft2, fftshift
from astropy.io import fits as astropy_fits

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox,
    QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QTabWidget, QSizePolicy, QSplitter,
    QSpinBox, QDoubleSpinBox, QSlider, QScrollArea, QFrame,
    QRadioButton, QButtonGroup, QStackedWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QTimer
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QPixmap, QImage

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

SIRIL_BG        = "#1e2128"
SIRIL_BG2       = "#252930"
SIRIL_BG3       = "#2d3340"
SIRIL_ACCENT    = "#4a9eff"
SIRIL_ACCENT2   = "#2d6abf"
SIRIL_TEXT      = "#dde3ee"
SIRIL_TEXT_DIM  = "#7a8499"
SIRIL_BORDER    = "#3a4055"
SIRIL_SUCCESS   = "#4caf7d"
SIRIL_WARNING   = "#e8c46a"
SIRIL_ERROR     = "#cc4444"
SIRIL_NOVA      = "#ff6b35"
SIRIL_SECTION   = "#9db4d0"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: "Segoe UI", sans-serif;
    font-size: 9pt;
}}
QGroupBox {{
    border: 1px solid {SIRIL_BORDER};
    border-radius: 5px;
    margin-top: 8px;
    padding: 6px;
    font-weight: bold;
    color: {SIRIL_SECTION};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QTabWidget::pane {{ border: 1px solid {SIRIL_BORDER}; background: {SIRIL_BG}; }}
QTabBar::tab {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT_DIM};
    padding: 6px 14px; border: 1px solid {SIRIL_BORDER};
    border-bottom: none; border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{ background: {SIRIL_BG}; color: {SIRIL_ACCENT}; font-weight: bold; }}
QPushButton {{
    background: {SIRIL_BG3}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    padding: 5px 12px; min-height: 22px;
}}
QPushButton:hover {{ background: {SIRIL_ACCENT2}; border-color: {SIRIL_ACCENT}; }}
QPushButton[objectName="primary"] {{
    background: {SIRIL_ACCENT2}; color: white; font-weight: bold;
    border-color: {SIRIL_ACCENT};
}}
QPushButton[objectName="danger"] {{
    background: #4a1e1e; color: #e07070; border-color: #803030;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px; padding: 3px 6px;
}}
QProgressBar {{
    background: {SIRIL_BG2}; border: 1px solid {SIRIL_BORDER};
    border-radius: 4px; color: {SIRIL_TEXT}; text-align: center;
}}
QProgressBar::chunk {{ background: {SIRIL_ACCENT}; border-radius: 3px; }}
QPlainTextEdit, QTextEdit {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    font-family: "Courier New", monospace; font-size: 8pt;
}}
QLabel[objectName="dim"] {{ color: {SIRIL_TEXT_DIM}; }}
QLabel[objectName="ok"]  {{ color: {SIRIL_SUCCESS}; }}
QLabel[objectName="warn"] {{ color: {SIRIL_WARNING}; }}
QCheckBox {{ spacing: 6px; }}
QSlider::groove:horizontal {{
    height: 4px; background: {SIRIL_BG3}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {SIRIL_ACCENT}; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}}
QScrollArea {{ border: none; }}
"""

# ─────────────────────────────────────────────────────────────────────────────
# SER FILE I/O
# ─────────────────────────────────────────────────────────────────────────────

SER_HEADER_SIZE = 178

def read_ser_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(SER_HEADER_SIZE)
    file_id   = raw[0:14].decode("ascii", errors="ignore").strip("\x00")
    lu_id     = struct.unpack_from("<i", raw, 14)[0]
    color_id  = struct.unpack_from("<i", raw, 18)[0]
    endian    = struct.unpack_from("<i", raw, 22)[0]
    w         = struct.unpack_from("<i", raw, 26)[0]
    h         = struct.unpack_from("<i", raw, 30)[0]
    depth     = struct.unpack_from("<i", raw, 34)[0]
    n_frames  = struct.unpack_from("<i", raw, 38)[0]
    bytes_pp  = (depth + 7) // 8
    is_color  = color_id not in (0, 100)
    channels  = 3 if is_color else 1
    return {
        "width": w, "height": h, "depth": depth, "frame_count": n_frames,
        "bytes_per_pixel": bytes_pp, "channels": channels,
        "is_color": is_color, "color_id": color_id,
        "frame_size": w * h * bytes_pp * channels,
        "header_size": SER_HEADER_SIZE, "endian": endian,
    }

def read_ser_frame(f, header: dict, frame_index: int) -> np.ndarray:
    depth = int(header.get("pixel_depth") or header.get("depth") or header.get("bit_depth") or 8)
    color_id = int(header.get("color_id") or header.get("color") or header.get("colour_id") or 0)
    is_color = color_id >= 100
    bpp = 1 if depth <= 8 else 2
    n_ch = 3 if is_color else 1
    w = header["width"]
    h = header["height"]
    frame_size = w * h * bpp * n_ch
    offset = 178 + frame_index * frame_size
    f.seek(offset)
    raw = f.read(frame_size)
    if len(raw) < frame_size:
        raise EOFError(f"Unexpected end of file at frame {frame_index}")
    dtype = np.uint8 if bpp == 1 else np.uint16
    arr = np.frombuffer(raw, dtype=dtype)
    if is_color:
        arr = arr.reshape(h, w, 3)
        if color_id == 101:
            arr = arr[:, :, ::-1]
        lum = (0.2126 * arr[:, :, 0].astype(np.float32)
               + 0.7152 * arr[:, :, 1].astype(np.float32)
               + 0.0722 * arr[:, :, 2].astype(np.float32))
    else:
        lum = arr.reshape(h, w).astype(np.float32)
    return lum
def load_fits_mono(path: str) -> np.ndarray:
    data = astropy_fits.getdata(path).astype(np.float32)
    if data.ndim == 3:
        if data.shape[0] == 3:
            return 0.299*data[0] + 0.587*data[1] + 0.114*data[2]
        return data[0]
    return data

def save_fits(data: np.ndarray, path: str, header_comment: str = ""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hdu = astropy_fits.PrimaryHDU(data=data.astype(np.float32))
    if header_comment:
        header_comment = header_comment.encode('ascii', errors='replace').decode('ascii')
        hdu.header["HISTORY"] = header_comment
    hdu.header["STACKALG"] = "HyperLoad Zone Stacking"
    hdu.writeto(path, overwrite=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLANET DISK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_planet_disk(frame: np.ndarray, min_fill: float = 0.02) -> dict:
    """
    Detect planet disk via Otsu thresholding + circular fit.
    Returns: {"cx": float, "cy": float, "radius": float, "found": bool}
    """
    f = frame if frame.ndim == 2 else (
        0.299*frame[0] + 0.587*frame[1] + 0.114*frame[2]
        if frame.shape[0] == 3 else frame[0])
    f = gaussian_filter(f, sigma=3.0)

    # Otsu threshold
    hist, edges = np.histogram(f, bins=256)
    total = f.size
    best_t, best_var = 0.0, 0.0
    w0 = 0.0; sum0 = 0.0; total_sum = np.sum(hist * edges[:-1])
    for i, (count, edge) in enumerate(zip(hist, edges[:-1])):
        w0 += count / total
        if w0 == 0 or w0 == 1: continue
        sum0 += edge * count
        m0 = sum0 / (w0 * total)
        m1 = (total_sum - sum0) / ((1 - w0) * total + 1e-10)
        var = w0 * (1 - w0) * (m0 - m1)**2
        if var > best_var:
            best_var = var; best_t = edge

    mask = f > best_t
    labeled, n_comp = label(mask)
    if n_comp == 0:
        h, w = f.shape
        return {"cx": w/2, "cy": h/2, "radius": min(h, w)*0.35, "found": False}

    # Largest connected component
    sizes = [np.sum(labeled == i) for i in range(1, n_comp+1)]
    largest = np.argmax(sizes) + 1
    planet_mask = labeled == largest

    if planet_mask.sum() < min_fill * f.size:
        h, w = f.shape
        return {"cx": w/2, "cy": h/2, "radius": min(h, w)*0.35, "found": False}

    cy, cx = center_of_mass(planet_mask)
    ys, xs = np.where(planet_mask)
    radius = np.sqrt(((xs - cx)**2 + (ys - cy)**2)).mean() * 1.1
    return {"cx": cx, "cy": cy, "radius": radius, "found": True}

# ─────────────────────────────────────────────────────────────────────────────
# ZONE GRID GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_zones(disk: dict, h: int, w: int, zone_size: int,
                   overlap: float, disk_only: bool = True) -> list:
    """
    Generate overlapping zone patches covering the planet disk.
    Returns list of {"x0","y0","x1","y1","cx","cy","weight_map"}
    """
    step = max(1, int(zone_size * (1.0 - overlap)))
    zones = []
    cx_disk, cy_disk, r_disk = disk["cx"], disk["cy"], disk["radius"]
    margin = zone_size // 2

    y0 = max(0, int(cy_disk - r_disk) - margin)
    y1 = min(h, int(cy_disk + r_disk) + margin)
    x0 = max(0, int(cx_disk - r_disk) - margin)
    x1 = min(w, int(cx_disk + r_disk) + margin)

    for yr in range(y0, y1, step):
        for xr in range(x0, x1, step):
            zy0 = yr; zy1 = min(h, yr + zone_size)
            zx0 = xr; zx1 = min(w, xr + zone_size)
            if zy1 - zy0 < zone_size // 2: continue
            if zx1 - zx0 < zone_size // 2: continue
            zcx = (zx0 + zx1) / 2.0
            zcy = (zy0 + zy1) / 2.0
            if disk_only:
                dist = np.sqrt((zcx - cx_disk)**2 + (zcy - cy_disk)**2)
                if dist > r_disk + zone_size * 1.0: continue
            # Gaussian weight map for feather blending
            zh = zy1 - zy0; zw = zx1 - zx0
            ys = np.linspace(-1, 1, zh); xs = np.linspace(-1, 1, zw)
            xx, yy = np.meshgrid(xs, ys)
            wmap = np.exp(-2.0 * (xx**2 + yy**2))
            zones.append({
                "x0": zx0, "y0": zy0, "x1": zx1, "y1": zy1,
                "cx": zcx, "cy": zcy, "weight_map": wmap,
            })
    return zones

# ─────────────────────────────────────────────────────────────────────────────
# SUB-PIXEL PHASE CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def phase_correlate_subpixel(ref: np.ndarray, img: np.ndarray,
                              upsample: int = 10) -> tuple:
    """
    Sub-pixel shift estimation via upsampled DFT.
    Reference: Guizar-Sicairos, Thurman & Fienup 2008, Opt. Lett. 33, 156.
    Returns (dy, dx) in pixels (sub-pixel precision 1/upsample).
    """
    # Normalize
    ref_n = (ref - ref.mean()) / (ref.std() + 1e-10)
    img_n = (img - img.mean()) / (img.std() + 1e-10)

    R = fft2(ref_n)
    M = fft2(img_n)
    prod = R * np.conj(M)
    denom = np.abs(prod) + 1e-10
    cross_power = prod / denom

    # Pixel-level peak
    corr = np.abs(ifft2(cross_power))
    hy, hx = np.array(corr.shape) // 2
    corr = np.roll(np.roll(corr, hy, axis=0), hx, axis=1)
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    dy0 = peak_y - hy; dx0 = peak_x - hx

    if upsample <= 1:
        return float(dy0), float(dx0)

    # Upsampled DFT around the pixel peak for sub-pixel refinement
    h, w = ref.shape
    up_region = 1.5
    dft_shift_y = np.arange(-up_region, up_region, 1.0/upsample)
    dft_shift_x = np.arange(-up_region, up_region, 1.0/upsample)
    kernel_y = np.exp(-2j * np.pi * (dy0 + dft_shift_y[:, None]) *
                      np.arange(h)[None, :] / h)
    kernel_x = np.exp(-2j * np.pi * (dx0 + dft_shift_x[None, :]) *
                      np.arange(w)[:, None] / w)
    upsampled = kernel_y @ cross_power @ kernel_x
    sub_y, sub_x = np.unravel_index(np.argmax(np.abs(upsampled)), upsampled.shape)
    dy = dy0 + dft_shift_y[sub_y]
    dx = dx0 + dft_shift_x[sub_x]
    return float(dy), float(dx)

def apply_shift(frame: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Apply sub-pixel shift via sinc interpolation (scipy map_coordinates)."""
    h, w = frame.shape
    ys = np.arange(h) + dy
    xs = np.arange(w) + dx
    xx, yy = np.meshgrid(xs, ys)
    return map_coordinates(frame, [yy.ravel(), xx.ravel()],
                           order=3, mode="reflect").reshape(h, w)

# ─────────────────────────────────────────────────────────────────────────────
# ZONE QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def zone_quality(patch: np.ndarray, metric: str = "laplacian") -> float:
    """Compute quality score for a single zone patch."""
    if patch.size == 0: return 0.0
    if metric == "laplacian":
        from scipy.ndimage import laplace
        return float(np.var(laplace(patch)))
    elif metric == "gradient":
        gy, gx = np.gradient(patch)
        return float(np.mean(gx**2 + gy**2))
    elif metric == "peak_pixel":
        return float(patch.max())
    elif metric == "normalized_variance":
        mu = patch.mean()
        if mu == 0: return 0.0
        return float(patch.var() / mu)
    elif metric == "tenengrad":
        from scipy.ndimage import sobel
        gx = sobel(patch, axis=1)
        gy = sobel(patch, axis=0)
        return float(np.mean(gx**2 + gy**2))
    elif metric == "vollath":
        # Vollath's F4 autocorrelation measure
        return float(np.mean(patch[:-1] * patch[1:]) -
                     np.mean(patch[:-2] * patch[2:]))
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# PER-ZONE STACKING ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

def zone_mean(patches: np.ndarray) -> np.ndarray:
    return np.mean(patches, axis=0)

def zone_median(patches: np.ndarray) -> np.ndarray:
    return np.median(patches, axis=0)

def zone_winsorized(patches: np.ndarray, sigma: float = 3.0,
                    n_iter: int = 5) -> np.ndarray:
    S = patches.copy()
    for _ in range(n_iter):
        med = np.median(S, axis=0)
        mad = 1.4826 * np.median(np.abs(S - med[np.newaxis]), axis=0)
        mad = np.where(mad == 0, 1e-10, mad)
        S = np.clip(S, med - sigma*mad, med + sigma*mad)
    return np.mean(S, axis=0)

def zone_trimmed(patches: np.ndarray, trim: float = 0.1) -> np.ndarray:
    N = patches.shape[0]
    n_trim = max(0, int(np.floor(N * trim)))
    if n_trim == 0: return np.mean(patches, axis=0)
    S = np.sort(patches, axis=0)
    return np.mean(S[n_trim:N-n_trim], axis=0)

def zone_biweight(patches: np.ndarray, c: float = 6.0,
                  n_iter: int = 10) -> np.ndarray:
    loc = np.median(patches, axis=0)
    for _ in range(n_iter):
        mad = 1.4826 * np.median(np.abs(patches - loc[np.newaxis]), axis=0)
        mad = np.where(mad == 0, 1e-10, mad)
        u = (patches - loc[np.newaxis]) / (c * mad[np.newaxis])
        w = np.where(np.abs(u) < 1.0, (1 - u**2)**2, 0.0)
        ws = w.sum(axis=0); ws = np.where(ws == 0, 1e-10, ws)
        loc = (w * patches).sum(axis=0) / ws
    return loc

ZONE_STACKERS = {
    "Mean":              zone_mean,
    "Median":            zone_median,
    "Winsorized σ-clip": zone_winsorized,
    "Trimmed Mean":      zone_trimmed,
    "Biweight (Tukey)":  zone_biweight,
}

# ─────────────────────────────────────────────────────────────────────────────
# LUCKY FOURIER (low-frequency Fourier-domain selection)
# ─────────────────────────────────────────────────────────────────────────────

def lucky_fourier_low_freq(frames: list, keep_pct: float,
                            lf_radius_frac: float = 0.1) -> np.ndarray:
    """
    Garrel, Guyon & Baudoz 2012 / Mackay 2013 Lucky Fourier algorithm.
    Selects best frames per spatial frequency in Fourier domain.
    Applied only to low spatial frequencies (radius < lf_radius_frac × Nyquist).
    """
    N = len(frames)
    h, w = frames[0].shape
    ffts = [fft2(f) for f in frames]
    # Build amplitude stacks per frequency
    amps = np.stack([np.abs(ft) for ft in ffts], axis=0)  # [N, H, W]
    # Frequency mask for low-frequency region
    fy = fftshift(np.fft.fftfreq(h))
    fx = fftshift(np.fft.fftfreq(w))
    FX, FY = np.meshgrid(fx, fy)
    freq_r = np.sqrt(FX**2 + FY**2)
    lf_mask = freq_r < lf_radius_frac
    n_keep = max(1, int(N * keep_pct / 100))
    # Per-frequency best-amplitude selection
    result_ft = np.zeros((h, w), dtype=complex)
    result_count = np.zeros((h, w))
    amps_shifted = np.stack([fftshift(a) for a in amps], axis=0)
    ffts_shifted = np.stack([fftshift(ft) for ft in ffts], axis=0)
    for yi in range(h):
        for xi in range(w):
            if not lf_mask[yi, xi]: continue
            pixel_amps = amps_shifted[:, yi, xi]
            top_idx = np.argsort(pixel_amps)[::-1][:n_keep]
            result_ft[yi, xi] = np.mean(ffts_shifted[top_idx, yi, xi])
            result_count[yi, xi] = 1
    # Fill non-LF with simple mean
    for yi in range(h):
        for xi in range(w):
            if not lf_mask[yi, xi]:
                result_ft[yi, xi] = np.mean(ffts_shifted[:, yi, xi])
    from scipy.fft import ifftshift
    return np.abs(ifft2(ifftshift(result_ft))).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ZONE STACKING WORKER
# ─────────────────────────────────────────────────────────────────────────────

class ZoneStackWorker(QThread):
    """
    Main processing worker. Runs the full zone-based pipeline:
    1. Load all frames from SER or FITS folder
    2. Detect planet disk from mean of first N frames
    3. Generate zone grid
    4. For each zone: score all frames, select best, align, stack
    5. Feather-blend all zones into final image
    6. Optional: Hybrid multi-scale (LF Fourier + HF spatial)
    7. Save result
    """
    progress      = pyqtSignal(int, int, str)       # step, total, msg
    log_line      = pyqtSignal(str)
    zone_preview  = pyqtSignal(np.ndarray)          # current assembled image
    disk_detected = pyqtSignal(dict, int, int)      # disk, H, W
    finished      = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event,
                 parent=None):
        super().__init__(parent)
        self.cfg    = config
        self._cancel = cancel_event

    def run(self):
        try:
            self._run()
        except Exception as e:
            tb = traceback.format_exc()
            self.log_line.emit(f"CRITICAL ERROR: {e}\n{tb}")
            self.finished.emit({"success": False, "error": str(e)})

    def _run(self):
        cfg = self.cfg
        self.log_line.emit("═══ HyperLoad Planetary Zone Stacker ═══")
        self.log_line.emit(f"  Mode:          {cfg['input_type']}")
        self.log_line.emit(f"  Zone size:     {cfg['zone_size']} px")
        self.log_line.emit(f"  Zone overlap:  {cfg['zone_overlap']*100:.0f}%")
        self.log_line.emit(f"  Best frames:   {cfg['keep_pct']}% per zone")
        self.log_line.emit(f"  Zone stacker:  {cfg['zone_stacker']}")
        self.log_line.emit(f"  Quality metric:{cfg['quality_metric']}")
        self.log_line.emit(f"  Hybrid mode:   {'ON' if cfg['hybrid_mode'] else 'OFF'}")
        self.log_line.emit("")

        # ── STEP 1: Load frames ───────────────────────────────────────────────
        self.progress.emit(0, 100, "Loading frames…")
        frames, h, w = self._load_frames()
        if not frames:
            self.finished.emit({"success": False, "error": "No frames loaded"})
            return
        N = len(frames)
        self.log_line.emit(f"Loaded: {N} frames  {w}×{h} px")

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── STEP 2: Detect planet disk ────────────────────────────────────────
        self.progress.emit(5, 100, "Detecting planet disk…")
        mean_frame = np.mean(np.stack(frames, axis=0), axis=0)
        disk = detect_planet_disk(mean_frame)
        self.log_line.emit(
            f"Disk: center=({disk['cx']:.1f}, {disk['cy']:.1f})  "
            f"radius={disk['radius']:.1f} px  "
            f"{'✓ found' if disk['found'] else '⚠ estimated'}")
        self.disk_detected.emit(disk, h, w)

        # ── STEP 3: Generate zones ────────────────────────────────────────────
        self.progress.emit(8, 100, "Generating zone grid…")
        zones = generate_zones(disk, h, w,
                               cfg["zone_size"],
                               cfg["zone_overlap"],
                               cfg["disk_only"])
        self.log_line.emit(f"Zones: {len(zones)} patches on planet disk")
        if not zones:
            self.finished.emit({"success": False, "error": "No zones generated"})
            return

        # ── STEP 4: Compute reference frame ───────────────────────────────────
        self.progress.emit(10, 100, "Computing reference frame…")
        # Use the frame with highest global quality as reference
        global_scores = [zone_quality(f, cfg["quality_metric"])
                         for f in frames]
        ref_idx  = int(np.argmax(global_scores))
        ref_frame = frames[ref_idx]
        self.log_line.emit(
            f"Reference frame: #{ref_idx} "
            f"(score={global_scores[ref_idx]:.4f})")

        # ── STEP 5: Build global alignment (coarse, whole-disk) ───────────────
        self.progress.emit(12, 100, "Global coarse alignment…")
        global_shifts = []
        for i, frame in enumerate(frames):
            if self._cancel.is_set(): break
            dy, dx = phase_correlate_subpixel(ref_frame, frame,
                                              upsample=cfg["upsample"])
            global_shifts.append((dy, dx))
        self.log_line.emit(
            f"Global shifts computed: max drift "
            f"dy={max(abs(s[0]) for s in global_shifts):.2f}  "
            f"dx={max(abs(s[1]) for s in global_shifts):.2f} px")
        self.progress.emit(13, 100, "Pre-applying global shifts…")
        shifted_frames = []
        for i, (frame, (gdy, gdx)) in enumerate(zip(frames, global_shifts)):
            shifted_frames.append(apply_shift(frame, -gdy, -gdx))
            if i % 500 == 0:
                self.progress.emit(13, 100, f"Pre-shifting {i}/{N}…")
        self.log_line.emit(f"Pre-shift complete.")    

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── STEP 6: Zone-by-zone processing ───────────────────────────────────
        result_image  = np.zeros((h, w), dtype=np.float64)
        weight_accum  = np.zeros((h, w), dtype=np.float64)
        n_zones = len(zones)
        stacker_fn = ZONE_STACKERS.get(cfg["zone_stacker"], zone_winsorized)

        self.log_line.emit(f"\nProcessing {n_zones} zones…")

        for zi, zone in enumerate(zones):
            if self._cancel.is_set(): break

            pct = 12 + int(85 * zi / n_zones)
            self.progress.emit(pct, 100,
                               f"Zone {zi+1}/{n_zones} — align + stack")

            zy0, zy1 = zone["y0"], zone["y1"]
            zx0, zx1 = zone["x0"], zone["x1"]
            zh = zy1 - zy0; zw = zx1 - zx0
            wmap = zone["weight_map"]
            if wmap.shape != (zh, zw):
                wmap = gaussian_filter(np.ones((zh, zw)), sigma=zh*0.3)

            # Extract reference patch
            ref_patch = ref_frame[zy0:zy1, zx0:zx1]

            # Score, align and collect patches for this zone
            zone_scores = []
            zone_patches = []

            for i, shifted_frame in enumerate(shifted_frames):
                if self._cancel.is_set(): break

                # Apply global shift first
               
                patch = shifted_frame[zy0:zy1, zx0:zx1]

                # Local fine alignment within the zone
                if cfg["local_align"] and patch.shape == ref_patch.shape:
                    try:
                        ldy, ldx = phase_correlate_subpixel(
                            ref_patch, patch, upsample=cfg["upsample"])
                        # Limit local shift to avoid over-correction
                        max_local = cfg["zone_size"] * 0.25
                        ldy = np.clip(ldy, -max_local, max_local)
                        ldx = np.clip(ldx, -max_local, max_local)
                        patch = apply_shift(patch, -ldy, -ldx)
                    except Exception:
                        pass

                # Quality score for this zone in this frame
                score = zone_quality(patch, cfg["quality_metric"])
                zone_scores.append(score)
                zone_patches.append(patch)

            if not zone_patches: continue

            # Select best N% of frames for this zone
            n_keep = max(1, int(len(zone_patches) * cfg["keep_pct"] / 100))
            sorted_idx = np.argsort(zone_scores)[::-1][:n_keep]
            best_patches = np.stack([zone_patches[i] for i in sorted_idx],
                                    axis=0)

            # Stack the selected patches
            stacked_patch = stacker_fn(best_patches)
            # Normalize zone brightness to reference to eliminate seams
            ref_mean = ref_frame[zy0:zy1, zx0:zx1].mean()
            stack_mean = stacked_patch.mean()
            if stack_mean > 1e-6:
                stacked_patch *= ref_mean / stack_mean

            # Accumulate with feather weight
            result_image[zy0:zy1, zx0:zx1] += stacked_patch * wmap
            weight_accum[zy0:zy1, zx0:zx1] += wmap

            # Emit live preview every 10 zones
            if zi % max(1, n_zones // 10) == 0:
                preview = np.where(weight_accum > 0,
                                   result_image / (weight_accum + 1e-10),
                                   0).astype(np.float32)
                self.zone_preview.emit(preview)
                self.log_line.emit(
                    f"  Zone {zi+1:4d}/{n_zones}  "
                    f"kept {n_keep}/{N} frames  "
                    f"best_score={zone_scores[sorted_idx[0]]:.4f}")

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"})
            return

        # Normalise by accumulated weights
        weight_accum = np.where(weight_accum == 0, 1.0, weight_accum)
        result = (result_image / weight_accum).astype(np.float32)

        # Fill any un-zoned pixels with the mean stack
        no_coverage = weight_accum == 1.0  # only where we set it to avoid div0
        if np.any(no_coverage):
            mean_stack = np.mean(np.stack(frames, axis=0), axis=0)
            result = np.where(no_coverage, mean_stack, result)

        # ── STEP 7: Hybrid multi-scale (optional) ────────────────────────────
        if cfg["hybrid_mode"]:
            self.progress.emit(97, 100, "Hybrid LF Fourier combination…")
            self.log_line.emit("\nHybrid mode: computing LF Fourier stack…")
            try:
                lf_result = lucky_fourier_low_freq(
                    frames, cfg["keep_pct"],
                    lf_radius_frac=cfg["lf_radius_frac"])
                # Combine: LF from Fourier stack, HF from zone stack
                # Use à trous wavelet split (Gaussian scale-space)
                sigma_split = cfg["zone_size"] * 0.5
                lf_zones = gaussian_filter(result, sigma_split)
                hf_zones = result - lf_zones
                lf_fourier = gaussian_filter(lf_result, sigma_split)
                result = lf_fourier + hf_zones
                self.log_line.emit(
                    "  Hybrid: LF from Fourier-domain, HF from zone stack")
            except Exception as e:
                self.log_line.emit(f"  Hybrid mode failed ({e}), using zone result")

        # ── STEP 8: Output normalisation ─────────────────────────────────────
        if cfg["output_norm"]:
            mn = result.min(); mx = result.max()
            if mx > mn: result = (result - mn) / (mx - mn)

        # ── STEP 9: Save ─────────────────────────────────────────────────────
        self.progress.emit(99, 100, "Saving…")
        out_dir  = cfg["output_dir"]
        out_name = cfg["output_name"]
        out_path = os.path.join(out_dir, f"{out_name}.fit")
        save_fits(result, out_path,
                  f"HyperLoad Zone Stacking  zones={n_zones}  "
                  f"N={N}  keep={cfg['keep_pct']}%  "
                  f"stacker={cfg['zone_stacker']}")
        self.progress.emit(100, 100, "Done")
        self.log_line.emit(f"\n✓ Saved: {out_path}")
        self.log_line.emit(
            f"  Zones processed: {n_zones}  "
            f"Frames used (max per zone): {n_keep}/{N}")
        self.zone_preview.emit(result)
        self.finished.emit({
            "success":    True,
            "output_path": out_path,
            "n_zones":    n_zones,
            "n_frames":   N,
        })

    def _load_frames(self) -> tuple:
        """Load frames from SER or FITS folder. Returns (frames list, H, W)."""
        cfg = self.cfg
        frames = []; h = 0; w = 0
        max_frames = cfg.get("max_frames") or 99999

        if cfg["input_type"] == "ser":
            path = cfg["ser_path"]
            if not os.path.isfile(path):
                self.log_line.emit(f"SER not found: {path}"); return [], 0, 0
            hdr = read_ser_header(path)
            h, w = hdr["height"], hdr["width"]
            n = min(hdr["frame_count"], max_frames)
            self.log_line.emit(f"SER: {n}/{hdr['frame_count']} frames  {w}×{h}")
            with open(path, "rb") as f:
                for i in range(n):
                    if self._cancel.is_set(): break
                    frame = read_ser_frame(f, hdr, i)
                    if frame.ndim == 3:
                        if frame.shape[0] == 3:
                            frame = (0.299*frame[0] +
                                     0.587*frame[1] +
                                     0.114*frame[2])
                        else:
                            frame = frame[0]
                    frames.append(frame.astype(np.float32))
                    if i % 200 == 0:
                        self.progress.emit(
                            int(4 * i / n), 100, f"Loading SER frame {i}/{n}")

        else:  # FITS folder
            folder = cfg["fits_dir"]
            files  = sorted(
                glob.glob(os.path.join(folder, "*.fit"))  +
                glob.glob(os.path.join(folder, "*.fits")) +
                glob.glob(os.path.join(folder, "*.fts")))[:max_frames]
            if not files:
                self.log_line.emit(f"No FITS files in {folder}"); return [], 0, 0
            self.log_line.emit(f"FITS: {len(files)} files")
            for i, f in enumerate(files):
                if self._cancel.is_set(): break
                try:
                    frame = load_fits_mono(f)
                    if h == 0: h, w = frame.shape
                    frames.append(frame)
                except Exception as e:
                    self.log_line.emit(f"  Skip {os.path.basename(f)}: {e}")
                if i % 50 == 0:
                    self.progress.emit(
                        int(4 * i / len(files)), 100,
                        f"Loading FITS {i}/{len(files)}")

        return frames, h, w


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class PlanetPreviewCanvas(FigureCanvasQTAgg):
    """2-panel canvas: left = reference frame with zone overlay, right = result."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), facecolor=SIRIL_BG)
        self.ax_ref    = self.fig.add_subplot(1, 2, 1)
        self.ax_result = self.fig.add_subplot(1, 2, 2)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)
        self._ref_frame   = None
        self._disk        = None
        self._zones       = None
        self._h = self._w = 0

    def _style(self):
        for ax, title in [(self.ax_ref, "Reference frame + zones"),
                          (self.ax_result, "Zone stack result")]:
            ax.set_facecolor(SIRIL_BG2)
            ax.set_title(title, color=SIRIL_ACCENT, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color(SIRIL_BORDER)

    def show_reference(self, frame: np.ndarray, disk: dict, zones: list):
        self._ref_frame = frame; self._disk = disk; self._zones = zones
        self._h, self._w = frame.shape
        self.ax_ref.clear(); self.ax_ref.set_facecolor(SIRIL_BG2)
        display = np.clip(frame, *np.percentile(frame, [0.5, 99.5]))
        display = (display - display.min()) / (display.max() - display.min() + 1e-10)
        self.ax_ref.imshow(display, cmap="gray", origin="lower",
                           aspect="equal", interpolation="nearest")
        # Draw disk circle
        if disk["found"]:
            circle = __import__("matplotlib").patches.Circle(
                (disk["cx"], disk["cy"]), disk["radius"],
                fill=False, edgecolor="#ff6b35", linewidth=1.5, linestyle="--")
            self.ax_ref.add_patch(circle)
        # Draw zones
        if zones:
            for zone in zones[::max(1, len(zones)//40)]:
                rect = __import__("matplotlib").patches.Rectangle(
                    (zone["x0"], zone["y0"]),
                    zone["x1"]-zone["x0"], zone["y1"]-zone["y0"],
                    fill=False, edgecolor="#4a9eff", linewidth=0.4, alpha=0.5)
                self.ax_ref.add_patch(rect)
        self.ax_ref.set_title(
            f"Reference + {len(zones)} zones", color=SIRIL_ACCENT, fontsize=9)
        self.ax_ref.tick_params(left=False, bottom=False,
                                labelleft=False, labelbottom=False)
        for sp in self.ax_ref.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_result(self, result: np.ndarray):
        self.ax_result.clear(); self.ax_result.set_facecolor(SIRIL_BG2)
        display = np.clip(result, *np.percentile(result, [0.1, 99.9]))
        display = (display - display.min()) / (display.max() - display.min() + 1e-10)
        self.ax_result.imshow(display, cmap="gray", origin="lower",
                              aspect="equal", interpolation="nearest")
        self.ax_result.set_title("Zone stack result", color=SIRIL_SUCCESS,
                                 fontsize=9)
        self.ax_result.tick_params(left=False, bottom=False,
                                   labelleft=False, labelbottom=False)
        for sp in self.ax_result.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# ZONE MAP CANVAS  (top-down quality heat map of zones)
# ─────────────────────────────────────────────────────────────────────────────

class ZoneMapCanvas(FigureCanvasQTAgg):
    """Shows per-zone quality heat map after analysis."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor=SIRIL_BG)
        self.ax  = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(SIRIL_BG2)
        super().__init__(self.fig)
        self.setParent(parent)

    def show_zone_qualities(self, zones: list, qualities: dict,
                            h: int, w: int):
        """qualities: {zone_idx: mean_quality_score}"""
        self.ax.clear(); self.ax.set_facecolor(SIRIL_BG2)
        hmap = np.zeros((h, w))
        for zi, zone in enumerate(zones):
            q = qualities.get(zi, 0)
            hmap[zone["y0"]:zone["y1"], zone["x0"]:zone["x1"]] += q
        if hmap.max() > 0:
            hmap /= hmap.max()
        self.ax.imshow(hmap, cmap="plasma", origin="lower",
                       aspect="equal", interpolation="bilinear")
        self.ax.set_title("Zone quality heat map", color=SIRIL_ACCENT,
                          fontsize=9)
        self.ax.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for sp in self.ax.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.2)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planetary Zone Stacker  —  HYPERLOAD  —  Siril")
        self.resize(1400, 900)
        self._worker        = None
        self._cancel_event  = threading.Event()
        self._preview_zones = []
        self._disk          = {}
        self._h = self._w   = 0
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QWidget(); header.setFixedHeight(56)
        header.setStyleSheet(
            f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon  = QLabel("🪐")
        lbl_icon.setStyleSheet("font-size:20pt; background:transparent;")
        lbl_title = QLabel("Planetary Zone Stacker")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_NOVA}; font-size:14pt; font-weight:bold; "
            f"background:transparent;")
        lbl_sub = QLabel(
            "Zone-based lucky imaging  ·  Sub-pixel alignment  ·  "
            "Hybrid Fourier+spatial  ·  Feather blending")
        lbl_sub.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title)
        hl.addWidget(lbl_sub); hl.addStretch()
        root.addWidget(header)

        # ── Main tabs ─────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_tab_input(),  "📁  Input")
        self._tabs.addTab(self._build_tab_zones(),  "⚙  Zone Settings")
        self._tabs.addTab(self._build_tab_preview(),"🔬  Preview")
        self._tabs.addTab(self._build_tab_output(), "💾  Output")
        self._tabs.addTab(self._build_tab_log(),    "📋  Log")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom = QWidget(); bottom.setFixedHeight(50)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 6, 10, 6)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(10); self._progress.setValue(0)
        bl.addWidget(self._progress, 1)
        self._btn_run = QPushButton("▶  Run Zone Stack")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumWidth(160); self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run)
        bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(34); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bl.addWidget(self._btn_cancel)
        self._status = QLabel("Ready — load a SER or FITS sequence")
        self._status.setObjectName("dim")
        self._status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status, 1)
        root.addWidget(bottom)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_tab_input(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        # Source type
        grp_src = QGroupBox("Input source")
        sl = QVBoxLayout(grp_src)
        self._rb_ser  = QRadioButton("SER file  (planetary video — recommended)")
        self._rb_fits = QRadioButton("FITS folder  (pre-debayered sequence)")
        self._rb_ser.setChecked(True)
        self._bg_src = QButtonGroup()
        self._bg_src.addButton(self._rb_ser); self._bg_src.addButton(self._rb_fits)
        self._rb_ser.toggled.connect(self._on_src_type)
        sl.addWidget(self._rb_ser); sl.addWidget(self._rb_fits)
        lay.addWidget(grp_src)

        # SER input
        self._grp_ser = QGroupBox("SER file")
        ser_row = QHBoxLayout(self._grp_ser)
        self._edit_ser = QLineEdit(); self._edit_ser.setPlaceholderText("Path to .ser file…")
        btn_ser = QPushButton("Browse…"); btn_ser.setFixedWidth(80)
        btn_ser.clicked.connect(self._browse_ser)
        ser_row.addWidget(self._edit_ser); ser_row.addWidget(btn_ser)
        lay.addWidget(self._grp_ser)

        # FITS folder input
        self._grp_fits = QGroupBox("FITS folder")
        fits_row = QHBoxLayout(self._grp_fits)
        self._edit_fits = QLineEdit()
        self._edit_fits.setPlaceholderText("Folder with .fit / .fits files…")
        btn_fits = QPushButton("Browse…"); btn_fits.setFixedWidth(80)
        btn_fits.clicked.connect(self._browse_fits)
        fits_row.addWidget(self._edit_fits); fits_row.addWidget(btn_fits)
        self._grp_fits.setVisible(False)
        lay.addWidget(self._grp_fits)

        # Frame limit
        grp_limit = QGroupBox("Frame limit")
        lf = QFormLayout(grp_limit)
        self._spin_max = QSpinBox()
        self._spin_max.setRange(0, 999999); self._spin_max.setValue(0)
        self._spin_max.setSpecialValueText("All frames")
        lf.addRow("Max frames (0=all):", self._spin_max)
        lay.addWidget(grp_limit)

        # Disk detection preview
        grp_disk = QGroupBox("Planet disk detection")
        dl = QVBoxLayout(grp_disk)
        lbl_disk_hint = QLabel(
            "The disk is auto-detected using Otsu thresholding + centroid fitting.\n"
            "If detection fails, zones still cover the full frame.")
        lbl_disk_hint.setObjectName("dim"); lbl_disk_hint.setWordWrap(True)
        dl.addWidget(lbl_disk_hint)
        self._chk_disk_only = QCheckBox(
            "Restrict zones to disk area (recommended for planets)")
        self._chk_disk_only.setChecked(True)
        dl.addWidget(self._chk_disk_only)
        lay.addWidget(grp_disk)
        lay.addStretch()
        return w

    def _build_tab_zones(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        # Zone grid
        grp_grid = QGroupBox("Zone grid")
        gf = QFormLayout(grp_grid)
        self._spin_zone_size = QSpinBox()
        self._spin_zone_size.setRange(16, 512); self._spin_zone_size.setValue(64)
        self._spin_zone_size.setSingleStep(8); self._spin_zone_size.setSuffix(" px")
        gf.addRow("Zone size:", self._spin_zone_size)
        self._spin_overlap = QSpinBox()
        self._spin_overlap.setRange(0, 90); self._spin_overlap.setValue(50)
        self._spin_overlap.setSuffix(" %")
        gf.addRow("Overlap:", self._spin_overlap)
        lbl_g = QLabel("Larger zones = more stable alignment but less local correction.\n"
                        "More overlap = smoother blending but slower.")
        lbl_g.setObjectName("dim"); lbl_g.setWordWrap(True); gf.addRow(lbl_g)
        lay.addWidget(grp_grid)

        # Frame selection
        grp_sel = QGroupBox("Frame selection per zone")
        sf = QFormLayout(grp_sel)
        self._spin_keep = QSpinBox()
        self._spin_keep.setRange(1, 100); self._spin_keep.setValue(30)
        self._spin_keep.setSuffix(" %")
        sf.addRow("Keep best:", self._spin_keep)
        self._cmb_quality = QComboBox()
        self._cmb_quality.addItems([
            "laplacian", "tenengrad", "gradient",
            "normalized_variance", "vollath", "peak_pixel"])
        sf.addRow("Quality metric:", self._cmb_quality)
        lbl_q = QLabel("laplacian — best for sharpness\n"
                        "tenengrad — Sobel gradient, very robust\n"
                        "vollath — autocorrelation, good for low contrast")
        lbl_q.setObjectName("dim"); lbl_q.setWordWrap(True); sf.addRow(lbl_q)
        lay.addWidget(grp_sel)

        # Alignment
        grp_align = QGroupBox("Sub-pixel alignment")
        af = QFormLayout(grp_align)
        self._chk_local_align = QCheckBox(
            "Local zone alignment (in addition to global)")
        self._chk_local_align.setChecked(True)
        af.addRow(self._chk_local_align)
        self._spin_upsample = QSpinBox()
        self._spin_upsample.setRange(1, 100); self._spin_upsample.setValue(10)
        af.addRow("Sub-pixel precision (1/N):", self._spin_upsample)
        lbl_a = QLabel("10 → 0.1px precision  ·  50 → 0.02px  ·  higher = slower")
        lbl_a.setObjectName("dim"); af.addRow(lbl_a)
        lay.addWidget(grp_align)

        # Stacking algorithm
        grp_stack = QGroupBox("Zone stacking algorithm")
        sf2 = QFormLayout(grp_stack)
        self._cmb_stacker = QComboBox()
        self._cmb_stacker.addItems(list(ZONE_STACKERS.keys()))
        self._cmb_stacker.setCurrentText("Winsorized σ-clip")
        sf2.addRow("Algorithm:", self._cmb_stacker)
        lay.addWidget(grp_stack)

        # Hybrid mode
        grp_hybrid = QGroupBox("Hybrid multi-scale mode (Wang, Li & Zhang 2021)")
        hf = QVBoxLayout(grp_hybrid)
        self._chk_hybrid = QCheckBox(
            "Enable hybrid mode\n"
            "Low frequencies: Fourier-domain frame selection (Garrel 2012)\n"
            "High frequencies: spatial zone stacking\n"
            "Combined via à-trous wavelet decomposition")
        self._chk_hybrid.setChecked(False)
        hf.addWidget(self._chk_hybrid)
        lf_row = QFormLayout()
        self._spin_lf_radius = QDoubleSpinBox()
        self._spin_lf_radius.setRange(0.01, 0.3); self._spin_lf_radius.setValue(0.1)
        self._spin_lf_radius.setSingleStep(0.01); self._spin_lf_radius.setDecimals(2)
        lf_row.addRow("LF radius (fraction of Nyquist):", self._spin_lf_radius)
        lbl_h = QLabel("Typical: 0.05–0.15  ·  Lower = more of the image from Fourier stack")
        lbl_h.setObjectName("dim"); lf_row.addRow(lbl_h)
        hf.addLayout(lf_row)
        lay.addWidget(grp_hybrid)
        lay.addStretch()
        return w

    def _build_tab_preview(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._preview_canvas = PlanetPreviewCanvas()
        self._preview_canvas.setMinimumHeight(400)
        lay.addWidget(self._preview_canvas, 2)
        self._zone_map = ZoneMapCanvas()
        self._zone_map.setMinimumHeight(220)
        lay.addWidget(self._zone_map, 1)
        return w

    def _build_tab_output(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        grp_out = QGroupBox("Output")
        of = QFormLayout(grp_out)
        out_row = QHBoxLayout()
        self._edit_out_dir = QLineEdit()
        self._edit_out_dir.setPlaceholderText("Output folder…")
        btn_od = QPushButton("Browse…"); btn_od.setFixedWidth(80)
        btn_od.clicked.connect(self._browse_out)
        out_row.addWidget(self._edit_out_dir); out_row.addWidget(btn_od)
        of.addRow("Output folder:", out_row)
        self._edit_out_name = QLineEdit("zone_stack_result")
        of.addRow("Filename (no ext):", self._edit_out_name)
        self._chk_norm = QCheckBox("Normalise output to [0, 1]")
        self._chk_norm.setChecked(True); of.addRow(self._chk_norm)
        lay.addWidget(grp_out)

        grp_ref = QGroupBox("References")
        rl = QVBoxLayout(grp_ref)
        refs = [
            "Zone-based stacking:  Hempel 2020, PlanetarySystemStacker (github)",
            "Sub-pixel alignment:  Guizar-Sicairos et al. 2008, Opt. Lett. 33, 156",
            "Lucky Fourier:        Garrel, Guyon & Baudoz 2012, PASP 124, 861",
            "Hybrid LI algorithm:  Wang, Li & Zhang 2021, RAA 21(5), 55",
            "Phase correlation:    Kuglin & Hines 1975, IEEE ICASSP, 163",
            "Feather blending:     Szeliski 2006, IEEE TPAMI 28(9), 1409",
        ]
        for r in refs:
            lbl = QLabel(r); lbl.setObjectName("dim"); rl.addWidget(lbl)
        lay.addWidget(grp_ref)
        lay.addStretch()
        return w

    def _build_tab_log(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._log.clear)
        lay.addWidget(btn_clr); lay.addWidget(self._log)
        return w

    # ── Browse / input helpers ─────────────────────────────────────────────────

    def _on_src_type(self, checked: bool):
        self._grp_ser.setVisible(self._rb_ser.isChecked())
        self._grp_fits.setVisible(self._rb_fits.isChecked())

    def _browse_ser(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SER file", "", "SER files (*.ser);;All (*)")
        if path:
            self._edit_ser.setText(path)
            auto_out = os.path.splitext(path)[0] + "_zone_stacked"
            if not self._edit_out_dir.text():
                self._edit_out_dir.setText(os.path.dirname(path))

    def _browse_fits(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self._edit_fits.setText(d)
            if not self._edit_out_dir.text():
                self._edit_out_dir.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self._edit_out_dir.setText(d)

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _get_config(self) -> dict | None:
        if self._rb_ser.isChecked():
            ser = self._edit_ser.text().strip()
            if not ser or not os.path.isfile(ser):
                QMessageBox.warning(self, "No SER", "Select a valid SER file.")
                return None
            input_type = "ser"; src = ser
        else:
            fits = self._edit_fits.text().strip()
            if not fits or not os.path.isdir(fits):
                QMessageBox.warning(self, "No folder",
                    "Select a valid FITS folder.")
                return None
            input_type = "fits"; src = fits

        out_dir = self._edit_out_dir.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output", "Set an output folder.")
            return None

        return {
            "input_type":    input_type,
            "ser_path":      src if input_type == "ser" else "",
            "fits_dir":      src if input_type == "fits" else "",
            "max_frames":    self._spin_max.value() or None,
            "disk_only":     self._chk_disk_only.isChecked(),
            "zone_size":     self._spin_zone_size.value(),
            "zone_overlap":  self._spin_overlap.value() / 100.0,
            "keep_pct":      self._spin_keep.value(),
            "quality_metric":self._cmb_quality.currentText(),
            "local_align":   self._chk_local_align.isChecked(),
            "upsample":      self._spin_upsample.value(),
            "zone_stacker":  self._cmb_stacker.currentText(),
            "hybrid_mode":   self._chk_hybrid.isChecked(),
            "lf_radius_frac":self._spin_lf_radius.value(),
            "output_dir":    out_dir,
            "output_name":   self._edit_out_name.text().strip() or "zone_stack_result",
            "output_norm":   self._chk_norm.isChecked(),
        }

    def _run(self):
        cfg = self._get_config()
        if cfg is None: return
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._set_status("Running zone stacker…", SIRIL_ACCENT)
        self._worker = ZoneStackWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.zone_preview.connect(self._preview_canvas.show_result)
        self._worker.disk_detected.connect(self._on_disk_detected)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._tabs.setCurrentIndex(4)  # show log

    def _cancel(self):
        self._cancel_event.set()
        self._btn_cancel.setEnabled(False)
        self._set_status("Cancelling…", SIRIL_WARNING)
        self._log.appendPlainText("⚠ Cancel requested…")

    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(f"[{step}/{total}]  {msg}", SIRIL_ACCENT)

    def _on_disk_detected(self, disk: dict, h: int, w: int):
        self._disk = disk; self._h = h; self._w = w
        # Generate zones to show on preview
        from threading import Thread
        zones = generate_zones(disk, h, w,
                               self._spin_zone_size.value(),
                               self._spin_overlap.value() / 100.0,
                               self._chk_disk_only.isChecked())
        self._preview_zones = zones

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        if result.get("success"):
            out  = result.get("output_path", "")
            n_z  = result.get("n_zones", 0)
            n_f  = result.get("n_frames", 0)
            msg  = f"✓ Done — {n_z} zones  ·  {n_f} frames  →  {os.path.basename(out)}"
            self._set_status(msg, SIRIL_SUCCESS)
            self._log.appendPlainText(f"\n{msg}")
            self._tabs.setCurrentIndex(2)  # show preview
        else:
            err = result.get("error", "Unknown error")
            self._set_status(f"✗ Failed: {err}", SIRIL_ERROR)
            self._log.appendPlainText(f"\n✗ FAILED: {err}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status.setText(msg)
        self._status.setStyleSheet(f"color:{color}; font-size:9pt;")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import traceback as _tb
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "crash_log.txt")
    def crash_handler(exc_type, exc_value, exc_tb):
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\nCRASH  {datetime.now()}\n{msg}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = crash_handler

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
