"""
thresher_stack.py — Script #14
The Thresher: PSF-Blind Lucky Imaging Stack
Based on Hitchcock et al. 2022, MNRAS 511, 5372 (arXiv: 2202.04686)
Online multi-frame blind deconvolution using PyTorch SGD.
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")

import os
import sys
import glob
import threading
import time
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    HAS_CUDA  = torch.cuda.is_available()
    DEVICE    = torch.device("cuda" if HAS_CUDA else "cpu")
except ImportError:
    HAS_TORCH = False
    HAS_CUDA  = False
    DEVICE    = None

from astropy.io import fits as astropy_fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QFrame, QSizePolicy, QDialog, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ══════════════════════════════════════════════════════════════════
# SIRIL THEME
# ══════════════════════════════════════════════════════════════════

SIRIL_BG       = "#1e2128"
SIRIL_BG2      = "#252930"
SIRIL_BG3      = "#2d3240"
SIRIL_ACCENT   = "#4a9eff"
SIRIL_ACCENT2  = "#2d6abf"
SIRIL_TEXT     = "#dde3ee"
SIRIL_TEXT_DIM = "#7a8499"
SIRIL_BORDER   = "#3a4055"
SIRIL_SUCCESS  = "#4caf7d"
SIRIL_WARNING  = "#e8a23a"
SIRIL_SECTION  = "#5ba3ff"
SIRIL_ERROR    = "#cc4444"

SIRIL_STYLESHEET = f"""
QMainWindow, QDialog, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}}
QTabWidget::pane {{
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
    background: {SIRIL_BG2};
}}
QTabBar::tab {{
    background: {SIRIL_BG3};
    color: {SIRIL_TEXT_DIM};
    border: 1px solid {SIRIL_BORDER};
    padding: 6px 16px;
    margin-right: 2px;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    background: {SIRIL_BG2};
    color: {SIRIL_ACCENT};
    border-bottom: 2px solid {SIRIL_ACCENT};
}}
QGroupBox {{
    background-color: {SIRIL_BG2};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 5px;
    margin-top: 8px;
    padding: 8px;
    font-weight: bold;
    color: {SIRIL_SECTION};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QTableWidget {{
    background-color: {SIRIL_BG2};
    alternate-background-color: {SIRIL_BG3};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
    gridline-color: {SIRIL_BORDER};
    selection-background-color: {SIRIL_ACCENT2};
    selection-color: {SIRIL_TEXT};
}}
QTableWidget::item {{ padding: 4px 8px; border: none; }}
QHeaderView::section {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_ACCENT};
    border: 1px solid {SIRIL_BORDER};
    padding: 5px 8px;
    font-weight: bold;
}}
QPushButton {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
    padding: 6px 12px;
    text-align: left;
}}
QPushButton:hover {{
    background-color: {SIRIL_ACCENT2};
    border-color: {SIRIL_ACCENT};
    color: white;
}}
QPushButton:pressed {{ background-color: {SIRIL_ACCENT}; }}
QPushButton#primary {{
    background-color: {SIRIL_ACCENT2};
    border-color: {SIRIL_ACCENT};
    color: white;
    font-weight: bold;
    text-align: center;
}}
QPushButton#primary:hover {{ background-color: {SIRIL_ACCENT}; }}
QPushButton#danger:hover {{
    background-color: #8b2020;
    border-color: {SIRIL_ERROR};
}}
QPushButton#export {{
    background-color: {SIRIL_BG3};
    border-color: {SIRIL_SUCCESS};
    color: {SIRIL_SUCCESS};
    text-align: center;
    font-weight: bold;
}}
QPushButton#export:hover {{ background-color: #1a3d2a; }}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {SIRIL_ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {SIRIL_ACCENT};
}}
QPlainTextEdit[readOnly="true"] {{
    background-color: #181c22;
    color: {SIRIL_SUCCESS};
    font-family: 'Courier New', monospace;
    font-size: 9pt;
}}
QComboBox::drop-down {{ border: none; padding-right: 4px; }}
QComboBox QAbstractItemView {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT};
    selection-background-color: {SIRIL_ACCENT2};
    border: 1px solid {SIRIL_BORDER};
}}
QScrollBar:vertical {{
    background: {SIRIL_BG2}; width: 10px; border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {SIRIL_BORDER}; border-radius: 5px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {SIRIL_ACCENT2}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{
    background: {SIRIL_BG3};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    height: 8px;
}}
QProgressBar::chunk {{
    background: {SIRIL_ACCENT};
    border-radius: 3px;
}}
QSlider::groove:horizontal {{
    border: 1px solid {SIRIL_BORDER};
    height: 4px;
    background: {SIRIL_BG3};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {SIRIL_ACCENT};
    border: 1px solid {SIRIL_ACCENT2};
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {SIRIL_ACCENT2};
    border-radius: 2px;
}}
QLabel#section  {{ color: {SIRIL_SECTION}; font-weight: bold; padding-top: 6px; }}
QLabel#dim      {{ color: {SIRIL_TEXT_DIM}; font-size: 9pt; }}
QLabel#ok       {{ color: {SIRIL_SUCCESS}; font-weight: bold; }}
QLabel#err      {{ color: {SIRIL_ERROR};   font-weight: bold; }}
QLabel#warn     {{ color: {SIRIL_WARNING}; font-weight: bold; }}
QLabel#gpu      {{
    color: {SIRIL_SUCCESS};
    font-weight: bold;
    padding: 2px 6px;
    border: 1px solid {SIRIL_SUCCESS};
    border-radius: 3px;
}}
QCheckBox {{ color: {SIRIL_TEXT}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {SIRIL_BORDER};
    border-radius: 2px;
    background: {SIRIL_BG3};
}}
QCheckBox::indicator:checked {{
    background: {SIRIL_ACCENT};
    border-color: {SIRIL_ACCENT};
}}
QScrollArea {{ border: none; background: transparent; }}
QFrame#separator {{ background-color: {SIRIL_BORDER}; max-height: 1px; }}
QSplitter::handle {{ background: {SIRIL_BORDER}; width: 2px; }}
"""

# ══════════════════════════════════════════════════════════════════
# ALGORITHM CORE
# ══════════════════════════════════════════════════════════════════

# ── SER format constants ──────────────────────────────────────────
SER_MONO        = 0
SER_BAYER_RGGB  = 8
SER_BAYER_GRBG  = 9
SER_BAYER_GBRG  = 10
SER_BAYER_BGGR  = 11
SER_RGB         = 100
SER_BGR         = 101

def read_ser_header(ser_path):
    """
    Parse the 178-byte SER file header.
    Returns dict with: width, height, pixel_depth, n_frames, color_id, date_time
    SER spec: https://www.grischa-hahn.homepage.t-online.de/astro/ser/SER%20Doc%20V3b.pdf
    """
    with open(ser_path, "rb") as f:
        raw = f.read(178)
    if len(raw) < 178:
        raise ValueError("File too short to be a valid SER file")
    file_id = raw[:14].rstrip(b'\x00').decode("ascii", errors="ignore")
    if not file_id.startswith("LUCAM-RECORDER"):
        # Many recorders omit the exact ID — continue anyway
        pass
    lu_id      = int.from_bytes(raw[14:18],  "little")
    color_id   = int.from_bytes(raw[18:22],  "little")
    little_end = int.from_bytes(raw[22:26],  "little")
    width      = int.from_bytes(raw[26:30],  "little")
    height     = int.from_bytes(raw[30:34],  "little")
    pixel_depth= int.from_bytes(raw[34:38],  "little")
    n_frames   = int.from_bytes(raw[38:42],  "little")
    observer   = raw[42:82].rstrip(b'\x00').decode("utf-8", errors="ignore")
    instrument = raw[82:122].rstrip(b'\x00').decode("utf-8", errors="ignore")
    telescope  = raw[122:162].rstrip(b'\x00').decode("utf-8", errors="ignore")
    date_time  = int.from_bytes(raw[162:170], "little")  # Windows FILETIME
    return {
        "width":       width,
        "height":      height,
        "pixel_depth": pixel_depth,
        "n_frames":    n_frames,
        "color_id":    color_id,
        "little_endian": little_end,
        "observer":    observer,
        "instrument":  instrument,
        "telescope":   telescope,
    }


def load_ser_frame(ser_path, frame_index, header=None):
    """
    Load a single frame from a SER file as float32 numpy array (H, W).
    For color frames: returns luminance (mean of channels).
    Uses direct seek — no need to load the whole file.
    """
    if header is None:
        header = read_ser_header(ser_path)
    w          = header["width"]
    h          = header["height"]
    depth      = header["pixel_depth"]
    color_id   = header["color_id"]
    little_end = header["little_endian"]

    bytes_per_pixel = 1 if depth <= 8 else 2
    is_color        = color_id in (SER_RGB, SER_BGR)
    channels        = 3 if is_color else 1
    frame_bytes     = w * h * channels * bytes_per_pixel
    header_size     = 178

    offset = header_size + frame_index * frame_bytes

    with open(ser_path, "rb") as f:
        f.seek(offset)
        raw = f.read(frame_bytes)

    if len(raw) < frame_bytes:
        raise ValueError(f"SER frame {frame_index}: unexpected EOF")

    dtype = np.uint8 if bytes_per_pixel == 1 else (
        np.uint16 if little_end else np.dtype(np.uint16).newbyteorder(">"))

    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    if is_color:
        arr = arr.reshape(h, w, 3)
        frame = arr.mean(axis=2)   # luminance
    else:
        frame = arr.reshape(h, w)

    return frame


def build_spool_from_ser(ser_path, output_spool_path,
                          max_frames=None, log_callback=None):
    """
    Convert a SER file into a FITS spool cube (N, H, W).
    Returns number of frames written.
    """
    hdr      = read_ser_header(ser_path)
    n_total  = hdr["n_frames"]
    n        = min(n_total, max_frames) if max_frames else n_total
    w, h     = hdr["width"], hdr["height"]

    if log_callback:
        log_callback(
            f"SER: {n_total} frames  {w}×{h}px  "
            f"{hdr['pixel_depth']}-bit  "
            f"color={'yes' if hdr['color_id'] in (SER_RGB,SER_BGR) else 'no'}")
        log_callback(f"Building spool: {n} frames")

    spool = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        try:
            spool[i] = load_ser_frame(ser_path, i, hdr)
            if log_callback and i % 200 == 0:
                log_callback(f"  Loaded frame {i+1}/{n}")
        except Exception as e:
            if log_callback:
                log_callback(f"  WARNING: frame {i} failed: {e}")

    hdu = astropy_fits.PrimaryHDU(data=spool)
    hdu.header["ORIGIN"] = "SER"
    hdu.header["SERFILE"] = os.path.basename(ser_path)
    hdu.writeto(output_spool_path, overwrite=True)
    if log_callback:
        log_callback(f"Spool saved: {output_spool_path}")
    return n


def export_spool_to_fits_sequence(spool_path, out_dir,
                                   prefix="frame", log_callback=None):
    """
    Export every frame of a FITS spool cube to individual FITS files.
    Useful for feeding results into other Siril scripts.
    Files named: prefix_000001.fit, prefix_000002.fit, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    with astropy_fits.open(spool_path, memmap=True) as hdul:
        n = hdul[0].data.shape[0]
        if log_callback:
            log_callback(f"Exporting {n} frames to {out_dir} ...")
        for i in range(n):
            frame = hdul[0].data[i].astype(np.float32)
            out_path = os.path.join(out_dir, f"{prefix}_{i+1:06d}.fit")
            h = astropy_fits.PrimaryHDU(data=frame)
            h.writeto(out_path, overwrite=True)
            if log_callback and i % 100 == 0:
                log_callback(f"  Exported {i+1}/{n}")
    if log_callback:
        log_callback(f"Export done: {n} files in {out_dir}")
    return n


class LiveFolderWatcher:
    """
    Watches a folder for new FITS or SER frames arriving in real time
    (e.g. SharpCap / FireCapture dropping files during capture).
    Maintains an ordered list of known files; poll() returns new ones since
    the last call.  Thread-safe.
    """
    def __init__(self, watch_dir, extensions=(".fit", ".fits", ".fts", ".ser")):
        self.watch_dir  = watch_dir
        self.extensions = extensions
        self._seen      = set()
        self._lock      = threading.Lock()
        self._refresh()   # snapshot existing files — don't re-process old ones

    def _refresh(self):
        found = set()
        for ext in self.extensions:
            found.update(glob.glob(os.path.join(self.watch_dir, f"*{ext}")))
        return found

    def mark_existing(self):
        """Call once before starting optimization to skip pre-existing files."""
        with self._lock:
            self._seen = self._refresh()

    def poll(self):
        """Return list of newly appeared files since last poll, sorted by mtime."""
        with self._lock:
            current = self._refresh()
            new     = sorted(current - self._seen,
                             key=lambda p: os.path.getmtime(p))
            self._seen = current
        return new


def build_spool(fits_dir, output_spool_path, max_frames=None, log_callback=None):
    files = sorted(
        glob.glob(os.path.join(fits_dir, "*.fit"))  +
        glob.glob(os.path.join(fits_dir, "*.fits")) +
        glob.glob(os.path.join(fits_dir, "*.fts"))
    )
    if max_frames:
        files = files[:max_frames]
    if not files:
        raise ValueError(f"No FITS files found in {fits_dir}")
    first = astropy_fits.getdata(files[0]).astype(np.float32)
    if first.ndim == 3:
        first = first[0]
    h, w = first.shape
    n    = len(files)
    if log_callback:
        log_callback(f"Building spool: {n} frames × {w}×{h} px")
    spool = np.zeros((n, h, w), dtype=np.float32)
    for i, path in enumerate(files):
        try:
            d = astropy_fits.getdata(path).astype(np.float32)
            if d.ndim == 3:
                d = d[0]
            spool[i] = d
            if log_callback and i % 100 == 0:
                log_callback(f"  Loaded frame {i+1}/{n}")
        except Exception as e:
            if log_callback:
                log_callback(f"  WARNING: frame {i} failed: {e}")
    hdu = astropy_fits.PrimaryHDU(data=spool)
    hdu.writeto(output_spool_path, overwrite=True)
    if log_callback:
        log_callback(f"Spool saved: {output_spool_path}")
    return n


def load_spool_frame(spool_path, frame_index):
    with astropy_fits.open(spool_path, memmap=True) as hdul:
        frame = hdul[0].data[frame_index].astype(np.float32)
    return frame


def gaussian_log_likelihood(model, data, sigma=1.0):
    residuals = data - model
    return -0.5 * torch.sum(residuals ** 2) / (sigma ** 2)


def emccd_log_likelihood(model, data, readout_noise=60.0, f=1.4,
                          em_gain=300.0, c=0.0, q=1.0):
    signal_electrons = model * q + c
    variance = (f**2 * em_gain**2 * signal_electrons.clamp(min=1e-6) +
                readout_noise**2)
    residuals = data - model * em_gain
    return -0.5 * torch.sum(residuals**2 / variance + torch.log(variance))


def ccd_log_likelihood(model, data, readout_noise_adu=1.36, gain=7.7):
    shot_variance  = model.clamp(min=1e-6) / gain
    total_variance = shot_variance + readout_noise_adu**2
    residuals      = data - model
    return -0.5 * torch.sum(residuals**2 / total_variance +
                             torch.log(total_variance))


def make_gaussian_init_kernel(kernel_size, sigma=1.5):
    half = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half+1, dtype=torch.float32),
        torch.arange(-half, half+1, dtype=torch.float32),
        indexing="ij"
    )
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def subtract_sky_background(frame):
    """Sigma-clipped sky estimate from image corners."""
    h, w   = frame.shape
    cs     = max(8, min(32, h // 8, w // 8))
    corners = np.concatenate([
        frame[:cs, :cs].ravel(), frame[:cs, -cs:].ravel(),
        frame[-cs:, :cs].ravel(), frame[-cs:, -cs:].ravel()
    ])
    med  = np.median(corners)
    std  = np.std(corners)
    mask = np.abs(corners - med) < 3 * std
    sky  = np.mean(corners[mask]) if mask.any() else med
    return (frame - sky).clip(0, None)


class ThresherModel:
    def __init__(self, init_scene, kernel_size, device,
                 noise_model="gaussian", noise_params=None,
                 lr_scene=1e-3, lr_kernel=1e-3,
                 l1_kernel=1e-3, proportional_clip=5e-3):
        self.device           = device
        self.noise_model      = noise_model
        self.noise_params     = noise_params or {}
        self.kernel_size      = kernel_size
        self.l1_kernel        = l1_kernel
        self.proportional_clip = proportional_clip
        self.n_updates        = 0

        scene_t = torch.tensor(
            np.log(np.clip(init_scene.astype(np.float32), 1e-8, None)),
            dtype=torch.float32, device=device)
        self.log_scene    = torch.nn.Parameter(scene_t)
        self.scene_optim  = torch.optim.Adam([self.log_scene], lr=lr_scene)
        self.init_kernel  = make_gaussian_init_kernel(kernel_size).to(device)
        self.lr_kernel    = lr_kernel

    @property
    def scene(self):
        return torch.exp(self.log_scene)

    def update(self, frame):
        pad = self.kernel_size // 2
        d   = torch.tensor(frame, dtype=torch.float32,
                            device=self.device).unsqueeze(0).unsqueeze(0)

        log_k = torch.log(self.init_kernel.clone().detach().clamp(min=1e-8))
        log_k.requires_grad_(True)
        k_optim = torch.optim.Adam([log_k], lr=self.lr_kernel)

        scene_detached = self.scene.detach().unsqueeze(0).unsqueeze(0)
        k_loss_final   = 0.0

        for _ in range(5):
            k_optim.zero_grad()
            kernel = torch.exp(log_k)
            kernel = kernel / (kernel.sum() + 1e-8)
            model  = F.conv2d(scene_detached, kernel, padding=pad)
            if model.shape != d.shape:
                model = model[:, :, :d.shape[2], :d.shape[3]]
            nll   = -self._log_likelihood(model, d)
            l1_k  = self.l1_kernel * torch.sum(torch.abs(torch.exp(log_k)))
            loss  = nll + l1_k
            loss.backward()
            k_optim.step()
            k_loss_final = float(loss)

        kernel_detached = torch.exp(log_k).detach()
        kernel_detached = kernel_detached / (kernel_detached.sum() + 1e-8)
        scene_before    = self.scene.detach().clone()

        self.scene_optim.zero_grad()
        model = F.conv2d(self.scene.unsqueeze(0).unsqueeze(0),
                          kernel_detached, padding=pad)
        if model.shape != d.shape:
            model = model[:, :, :d.shape[2], :d.shape[3]]
        nll    = -self._log_likelihood(model, d)
        s_loss = float(nll)
        nll.backward()
        self.scene_optim.step()

        with torch.no_grad():
            ratio   = self.scene / (scene_before + 1e-8)
            c_max   = 1.0 + self.proportional_clip
            c_min   = 1.0 / c_max
            corrected = scene_before * torch.clamp(ratio, c_min, c_max)
            self.log_scene.data = torch.log(corrected.clamp(min=1e-8))

        self.n_updates += 1
        return {"kernel_loss": k_loss_final, "scene_loss": s_loss}

    def _log_likelihood(self, model, data):
        p = self.noise_params
        if self.noise_model == "emccd":
            return emccd_log_likelihood(
                model, data,
                readout_noise=p.get("readout_noise", 60.0),
                f=p.get("f", 1.4),
                em_gain=p.get("em_gain", 300.0),
                c=p.get("c", 0.0),
                q=p.get("q", 1.0))
        elif self.noise_model == "ccd":
            return ccd_log_likelihood(
                model, data,
                readout_noise_adu=p.get("readout_noise_adu", 1.36),
                gain=p.get("gain", 7.7))
        else:
            return gaussian_log_likelihood(
                model, data,
                sigma=p.get("sigma", 1.0))

    def get_scene_numpy(self):
        return self.scene.detach().cpu().numpy().astype(np.float32)

    def save_scene(self, output_path, update_num):
        scene = self.get_scene_numpy()
        hdu   = astropy_fits.PrimaryHDU(data=scene)
        hdu.header["HISTORY"] = "The Thresher blind deconvolution"
        hdu.header["NUPDATE"] = update_num
        hdu.writeto(output_path, overwrite=True)


# ══════════════════════════════════════════════════════════════════
# WORKER THREAD
# ══════════════════════════════════════════════════════════════════

def estimate_eta(update_num, total_updates, start_time):
    elapsed = time.time() - start_time
    if update_num <= 0:
        return "ETA: calculating..."
    rate      = update_num / max(elapsed, 1e-6)
    remaining = (total_updates - update_num) / max(rate, 1e-6)
    mins      = int(remaining // 60)
    secs      = int(remaining % 60)
    if mins == 0:
        return f"ETA: {secs}s"
    return f"ETA: {mins}m {secs}s"


class ThresherWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    scene_updated = pyqtSignal(object)
    loss_point    = pyqtSignal(int, float)
    finished      = pyqtSignal(dict)

    def __init__(self, config, cancel_event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg    = self.config
            device = DEVICE

            if not HAS_TORCH:
                self.finished.emit({
                    "success": False,
                    "error":   "PyTorch not available. Install from https://pytorch.org"
                })
                return

            gpu_name = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU"
            self.log_line.emit(f"Device: {'GPU (' + gpu_name + ')' if HAS_CUDA else 'CPU only'}")
            self.log_line.emit(f"Started: {datetime.now().strftime('%H:%M:%S')}")

            # ── Step 1: Prepare spool ──────────────────────────────────────────
            self.progress.emit(0, 6, "Preparing spool...")
            input_type = cfg.get("input_type", "fits_dir")
            out_dir    = cfg.get("output_dir", ".")
            os.makedirs(out_dir, exist_ok=True)
            spool_path = cfg.get("spool_path", "")
            self._watcher = None   # live folder watcher (live mode only)

            if input_type == "fits_dir":
                spool_path = os.path.join(out_dir, "thresher_spool.fits")
                n_frames   = build_spool(
                    cfg["fits_dir"], spool_path,
                    max_frames=cfg.get("max_frames") or None,
                    log_callback=self.log_line.emit)

            elif input_type == "ser":
                spool_path = os.path.join(out_dir, "thresher_spool.fits")
                n_frames   = build_spool_from_ser(
                    cfg["ser_path"], spool_path,
                    max_frames=cfg.get("max_frames") or None,
                    log_callback=self.log_line.emit)

            else:   # pre-built FITS spool
                with astropy_fits.open(spool_path) as hdul:
                    shape = hdul[0].data.shape
                n_frames = shape[0]
                self.log_line.emit(
                    f"Spool: {n_frames} frames  {shape[2]}×{shape[1]}")

            if self._cancel.is_set():
                self._abort(); return


            # ── Live watch setup (works with any source type) ────────────────
            if cfg.get("live_watch", False):
                watch_dir = cfg.get("live_watch_dir", "").strip()
                if not watch_dir:
                    if input_type == "fits_dir":
                        watch_dir = cfg["fits_dir"]
                    elif input_type == "ser":
                        watch_dir = os.path.dirname(cfg["ser_path"])
                    else:
                        watch_dir = out_dir
                self._watcher = LiveFolderWatcher(watch_dir)
                self._watcher.mark_existing()
                self.log_line.emit(
                    f"🟢 Live watch ON — folder: {watch_dir}")

            # ── Step 2: Load init image ──────────────────────────────────────────
            self.progress.emit(1, 6, "Loading initialization image...")
            init_path = cfg.get("init_path", "")
            if init_path and os.path.isfile(init_path):
                init_scene = astropy_fits.getdata(init_path).astype(np.float32)
                if init_scene.ndim == 3:
                    init_scene = init_scene[0]
                self.log_line.emit(f"Init image: {init_path}")
            else:
                self.log_line.emit("Computing mean of first 10 frames as init...")
                frames_for_init = [
                    load_spool_frame(spool_path, i)
                    for i in range(min(10, n_frames))
                ]
                init_scene = np.mean(frames_for_init, axis=0)

            if cfg.get("subtract_sky", True):
                init_scene = subtract_sky_background(init_scene)

            if init_scene.max() > 0:
                init_scene = init_scene / init_scene.max()
            init_scene = init_scene.clip(1e-6, None)
            self.scene_updated.emit(init_scene.copy())

            if self._cancel.is_set():
                self._abort(); return

            # ── Step 3: Build model ──────────────────────────────────────────
            self.progress.emit(2, 6, "Initializing model...")
            kernel_size = cfg.get("kernel_size", 21)
            if kernel_size % 2 == 0:
                kernel_size += 1

            model = ThresherModel(
                init_scene       = init_scene,
                kernel_size      = kernel_size,
                device           = device,
                noise_model      = cfg.get("noise_model", "gaussian"),
                noise_params     = cfg.get("noise_params", {}),
                lr_scene         = cfg.get("lr_scene", 1e-3),
                lr_kernel        = cfg.get("lr_kernel", 1e-3),
                l1_kernel        = cfg.get("l1_kernel", 1e-3),
                proportional_clip = cfg.get("proportional_clip", 5e-3)
            )
            self.log_line.emit(
                f"Model ready  kernel={kernel_size}×{kernel_size}  "
                f"scene={init_scene.shape[1]}×{init_scene.shape[0]}  "
                f"noise={cfg.get('noise_model','gaussian')}")

            # ── Step 4: SGD ──────────────────────────────────────────────────
            self.progress.emit(3, 6, "Running SGD optimization...")
            iterations   = cfg.get("iterations", 1)
            save_every   = cfg.get("save_every", 100)
            out_dir      = cfg.get("output_dir", ".")
            os.makedirs(out_dir, exist_ok=True)
            fname        = cfg.get("output_name", "thresher_scene")
            out_path     = os.path.join(out_dir, f"{fname}.fit")
            total_updates = n_frames * iterations
            update_num    = 0
            start_time    = time.time()

            self.log_line.emit(
                f"SGD: {n_frames} frames × {iterations} iter "
                f"= {total_updates} updates")

            for iteration in range(iterations):
                if self._cancel.is_set():
                    break
                self.log_line.emit(f"=== Iteration {iteration+1}/{iterations} ===")
                frame_order = np.random.permutation(n_frames)

                for idx, frame_i in enumerate(frame_order):
                    if self._cancel.is_set():
                        break

                    frame = load_spool_frame(spool_path, int(frame_i))

                    if cfg.get("subtract_sky", True):
                        frame = subtract_sky_background(frame)
                    if frame.max() > 0:
                        frame = frame / frame.max()
                    frame = frame.clip(0, None)

                    losses     = model.update(frame)
                    update_num += 1
                    self.progress.emit(update_num, total_updates,
                                        f"Update {update_num}/{total_updates}")

                    total_loss = losses["scene_loss"] + losses["kernel_loss"]
                    self.loss_point.emit(update_num, total_loss)

                    if update_num % 50 == 0:
                        eta = estimate_eta(update_num, total_updates, start_time)
                        self.log_line.emit(
                            f"  [{update_num}/{total_updates}]  "
                            f"scene_loss={losses['scene_loss']:.4f}  "
                            f"kernel_loss={losses['kernel_loss']:.4f}  {eta}")

                    if update_num % save_every == 0:
                        model.save_scene(out_path, update_num)
                        self.scene_updated.emit(model.get_scene_numpy())
                        self.log_line.emit(f"  Saved checkpoint at update {update_num}")

                    # ── Live mode: absorb new frames that arrived since last poll ───────
                    if self._watcher and update_num % 25 == 0:
                        new_files = self._watcher.poll()
                        for nf in new_files:
                            if self._cancel.is_set():
                                break
                            try:
                                live_frame = astropy_fits.getdata(nf).astype(np.float32)
                                if live_frame.ndim == 3:
                                    live_frame = live_frame[0]
                                if cfg.get("subtract_sky", True):
                                    live_frame = subtract_sky_background(live_frame)
                                if live_frame.max() > 0:
                                    live_frame = live_frame / live_frame.max()
                                live_frame = live_frame.clip(0, None)
                                losses_live = model.update(live_frame)
                                update_num += 1
                                n_frames   += 1
                                total_updates += 1
                                tl = losses_live["scene_loss"] + losses_live["kernel_loss"]
                                self.loss_point.emit(update_num, tl)
                                self.log_line.emit(
                                    f"  ▶ Live frame absorbed: {os.path.basename(nf)}  "
                                    f"total_frames={n_frames}")
                            except Exception as live_err:
                                self.log_line.emit(
                                    f"  Live frame warning ({os.path.basename(nf)}): {live_err}")

            # ── Step 5: Final save ───────────────────────────────────────────
            self.progress.emit(5, 6, "Saving final result...")
            final_path = os.path.join(out_dir, f"{fname}_final.fit")
            model.save_scene(final_path, update_num)
            self.scene_updated.emit(model.get_scene_numpy())
            self.log_line.emit(f"Final scene saved: {final_path}")

            # ── FITS sequence export (optional) ──────────────────────────────
            if cfg.get("export_fits_seq", False) and os.path.isfile(spool_path):
                seq_dir = os.path.join(out_dir, fname + "_frames")
                self.log_line.emit(f"Exporting FITS sequence to {seq_dir} ...")
                try:
                    export_spool_to_fits_sequence(
                        spool_path, seq_dir,
                        prefix=fname,
                        log_callback=self.log_line.emit)
                except Exception as ex:
                    self.log_line.emit(f"  FITS export warning: {ex}")

            # Optionally delete spool
            auto_delete = (
                input_type in ("fits_dir", "ser") and
                not cfg.get("keep_spool", False) and
                os.path.isfile(spool_path))
            if auto_delete:
                os.remove(spool_path)
                self.log_line.emit("Temp spool deleted.")

            # ── Step 6: Load in Siril ────────────────────────────────────────
            self.progress.emit(6, 6, "Done")
            if cfg.get("load_in_siril", True):
                try:
                    siril = s.SirilInterface()
                    siril.connect()
                    siril.cmd("load", final_path)
                    siril.disconnect()
                    self.log_line.emit("Loaded in Siril ✓")
                except Exception as e:
                    self.log_line.emit(f"Siril load warning: {e}")

            elapsed_total = time.time() - start_time
            self.finished.emit({
                "success":     True,
                "output_path": final_path,
                "n_frames":    n_frames,
                "n_updates":   update_num,
                "iterations":  iterations,
                "elapsed_sec": elapsed_total,
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})


# ══════════════════════════════════════════════════════════════════
# CANVASES
# ══════════════════════════════════════════════════════════════════

class SceneCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig      = Figure(figsize=(8, 4), facecolor=SIRIL_BG)
        self.ax_init  = self.fig.add_subplot(1, 2, 1)
        self.ax_scene = self.fig.add_subplot(1, 2, 2)
        self._init_data  = None
        self._update_num = 0
        super().__init__(self.fig)
        self.setParent(parent)
        self._style_axes()

    def _style_axes(self):
        for ax, title, color in [
            (self.ax_init,  "Initialization",         SIRIL_TEXT_DIM),
            (self.ax_scene, "Thresher scene (live)",  SIRIL_ACCENT),
        ]:
            ax.set_facecolor(SIRIL_BG)
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)

    def _disp(self, arr):
        p_lo = np.percentile(arr, 0.5)
        p_hi = np.percentile(arr, 99.9)
        return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)

    def _ds(self, arr, max_px=400):
        h, w   = arr.shape
        factor = max(1, max(h, w) // max_px)
        return arr[::factor, ::factor]

    def show_init(self, init):
        self._init_data = init
        self.ax_init.clear()
        self.ax_init.imshow(self._ds(self._disp(init)),
                             cmap="gray", origin="lower",
                             aspect="equal", interpolation="nearest")
        self.ax_init.set_title("Initialization", color=SIRIL_TEXT_DIM, fontsize=9)
        for spine in self.ax_init.spines.values():
            spine.set_color(SIRIL_BORDER)
        self.ax_init.tick_params(left=False, bottom=False,
                                  labelleft=False, labelbottom=False)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_scene(self, scene, update_num=0):
        self._update_num = update_num
        self.ax_scene.clear()
        self.ax_scene.imshow(self._ds(self._disp(scene)),
                              cmap="gray", origin="lower",
                              aspect="equal", interpolation="nearest")
        self.ax_scene.set_title(
            f"Thresher scene — update {update_num}",
            color=SIRIL_ACCENT, fontsize=9)
        for spine in self.ax_scene.spines.values():
            spine.set_color(SIRIL_BORDER)
        self.ax_scene.tick_params(left=False, bottom=False,
                                   labelleft=False, labelbottom=False)
        self.fig.tight_layout(pad=0.3)
        self.draw()


class LossCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig  = Figure(figsize=(6, 3), facecolor=SIRIL_BG)
        self.ax   = self.fig.add_subplot(111)
        self._xs  = []
        self._ys  = []
        super().__init__(self.fig)
        self.setParent(parent)
        self._style()

    def _style(self):
        self.ax.set_facecolor(SIRIL_BG2)
        self.ax.tick_params(colors=SIRIL_TEXT, labelsize=7)
        for spine in ["bottom", "left"]:
            self.ax.spines[spine].set_color(SIRIL_BORDER)
        for spine in ["top", "right"]:
            self.ax.spines[spine].set_visible(False)
        self.ax.set_xlabel("SGD update", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_ylabel("Loss",       color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_title("Optimization convergence",
                           color=SIRIL_SECTION, fontsize=9)

    def add_point(self, update_num, loss):
        self._xs.append(update_num)
        self._ys.append(loss)
        if len(self._xs) % 10 == 0:
            self._redraw()

    def _redraw(self):
        self.ax.clear()
        self._style()
        if self._xs:
            ys     = np.array(self._ys)
            window = min(20, len(ys))
            smooth = np.convolve(ys, np.ones(window)/window, mode="valid")
            xs_s   = self._xs[window-1:]
            self.ax.plot(self._xs, self._ys,
                          color=SIRIL_BORDER, linewidth=0.5, alpha=0.5)
            self.ax.plot(xs_s, smooth, color=SIRIL_ACCENT, linewidth=1.5)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def reset(self):
        self._xs = []
        self._ys = []
        self.ax.clear()
        self._style()
        self.draw()

    def get_convergence_status(self):
        if len(self._ys) < 100:
            return "Accumulating data..."
        recent   = np.mean(self._ys[-50:])
        previous = np.mean(self._ys[-100:-50])
        delta    = (previous - recent) / (abs(previous) + 1e-10)
        if delta > 0.01:
            return f"Converging... (Δ={delta*100:.1f}%)"
        elif delta > -0.005:
            return "Converged ✓"
        else:
            return "⚠ Diverging — reduce gradient clip"


# ══════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Thresher — Siril")
        self.setMinimumSize(1100, 720)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._update_num   = 0
        self._total_updates = 0
        self._start_time   = None
        self._init_scene   = None

        self.setStyleSheet(SIRIL_STYLESHEET)
        self._build_ui()
        self._check_torch()

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root    = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(8, 6, 8, 6)

        # Title bar
        title_row = QHBoxLayout()
        lbl_title = QLabel("🔬  The Thresher  —  PSF-Blind Lucky Imaging  (Hitchcock et al. 2022)")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold;")
        title_row.addWidget(lbl_title)
        title_row.addStretch()

        if HAS_TORCH and HAS_CUDA:
            name = torch.cuda.get_device_name(0)
            self.gpu_badge = QLabel(f"  GPU: {name}  ")
            self.gpu_badge.setObjectName("gpu")
        else:
            self.gpu_badge = QLabel("  CPU only  ")
            self.gpu_badge.setObjectName("warn")
        title_row.addWidget(self.gpu_badge)
        root.addLayout(title_row)

        # Separator
        sep = QFrame(); sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # LEFT: config panel inside scroll area
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(400)
        left_inner  = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(4, 4, 4, 4)

        left_layout.addWidget(self._build_input_group())
        left_layout.addWidget(self._build_init_group())
        left_layout.addWidget(self._build_noise_group())
        left_layout.addWidget(self._build_sgd_group())
        left_layout.addWidget(self._build_output_group())
        left_layout.addStretch()
        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)

        # RIGHT: tabs
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([400, 700])
        root.addWidget(splitter, stretch=1)

        # Bottom row: progress + buttons
        bottom = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)
        bottom.addWidget(self.progress_bar, stretch=1)

        self.btn_run    = QPushButton("▶  Run Thresher")
        self.btn_run.setObjectName("primary")
        self.btn_run.setFixedWidth(160)
        self.btn_run.clicked.connect(self._on_run)
        bottom.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setFixedWidth(110)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        bottom.addWidget(self.btn_cancel)
        root.addLayout(bottom)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("dim")
        root.addWidget(self.status_label)

    def _build_input_group(self):
        grp    = QGroupBox("Input")
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Source:"))
        self.combo_input_type = QComboBox()
        self.combo_input_type.addItems([
            "FITS sequence folder",
            "FITS spool file (N×H×W cube)",
            "SER file (.ser)"
        ])
        self.combo_input_type.currentIndexChanged.connect(self._update_input_hint)
        row.addWidget(self.combo_input_type)
        layout.addLayout(row)

        path_row = QHBoxLayout()
        self.edit_input_path = QLineEdit()
        self.edit_input_path.setPlaceholderText("Path to folder or spool FITS…")
        path_row.addWidget(self.edit_input_path)
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(70)
        btn_browse.clicked.connect(self._browse_input)
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)

        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Max frames:"))
        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(0, 99999)
        self.spin_max_frames.setValue(0)
        self.spin_max_frames.setSpecialValueText("All")
        frames_row.addWidget(self.spin_max_frames)
        layout.addLayout(frames_row)

        self.lbl_input_hint = QLabel("0 = use all frames")
        self.lbl_input_hint.setObjectName("dim")
        layout.addWidget(self.lbl_input_hint)

        btn_inspect = QPushButton("🔍  Load & inspect")
        btn_inspect.clicked.connect(self._inspect_input)
        layout.addWidget(btn_inspect)

        self.lbl_input_info = QLabel("")
        self.lbl_input_info.setObjectName("dim")
        self.lbl_input_info.setWordWrap(True)
        layout.addWidget(self.lbl_input_info)

        # separator
        sep = QFrame(); sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        # Live watch checkbox
        self.chk_live_watch = QCheckBox(
            "🟢  Live folder watch (absorb new frames during run)")
        self.chk_live_watch.setChecked(False)
        self.chk_live_watch.setToolTip(
            "When enabled, the script will monitor the input folder and "
            "automatically absorb any new FITS frames that arrive while "
            "the optimization is running. Ideal for real-time capture with "
            "SharpCap or FireCapture.")
        self.chk_live_watch.toggled.connect(self._toggle_live_watch_ui)
        layout.addWidget(self.chk_live_watch)

        # Watch folder row (only shown when live watch is on AND input is not a folder)
        self.live_watch_folder_widget = QWidget()
        lw_row = QHBoxLayout(self.live_watch_folder_widget)
        lw_row.setContentsMargins(0, 0, 0, 0)
        lw_row.addWidget(QLabel("Watch folder:"))
        self.edit_live_watch_dir = QLineEdit()
        self.edit_live_watch_dir.setPlaceholderText(
            "Folder to watch (defaults to input folder)")
        lw_row.addWidget(self.edit_live_watch_dir)
        btn_lw = QPushButton("Browse")
        btn_lw.setFixedWidth(70)
        btn_lw.clicked.connect(self._browse_live_watch)
        lw_row.addWidget(btn_lw)
        self.live_watch_folder_widget.setVisible(False)
        layout.addWidget(self.live_watch_folder_widget)

        lbl_live_hint = QLabel(
            "Polling interval: every 25 SGD updates. "
            "New frames are processed as they arrive.")
        lbl_live_hint.setObjectName("dim")
        lbl_live_hint.setWordWrap(True)
        self._lbl_live_hint = lbl_live_hint
        lbl_live_hint.setVisible(False)
        layout.addWidget(lbl_live_hint)

        return grp

    def _build_init_group(self):
        grp    = QGroupBox("Initialization image")
        layout = QVBoxLayout(grp)

        self.combo_init = QComboBox()
        self.combo_init.addItems([
            "Auto (mean of first 10 frames)",
            "Load FITS file"
        ])
        self.combo_init.currentIndexChanged.connect(self._toggle_init_path)
        layout.addWidget(self.combo_init)

        self.init_path_widget = QWidget()
        row = QHBoxLayout(self.init_path_widget)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit_init_path = QLineEdit()
        self.edit_init_path.setPlaceholderText("Path to init FITS…")
        row.addWidget(self.edit_init_path)
        btn_b = QPushButton("Browse")
        btn_b.setFixedWidth(70)
        btn_b.clicked.connect(self._browse_init)
        row.addWidget(btn_b)
        self.init_path_widget.setVisible(False)
        layout.addWidget(self.init_path_widget)

        hint = QLabel("Tip: use lucky imaging result or best single frame")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return grp

    def _build_noise_group(self):
        grp    = QGroupBox("Noise model")
        layout = QVBoxLayout(grp)

        self.combo_noise = QComboBox()
        self.combo_noise.addItems(["Gaussian (simple)", "EMCCD", "CCD"])
        self.combo_noise.currentIndexChanged.connect(self._toggle_noise_params)
        layout.addWidget(self.combo_noise)

        # Gaussian params
        self.noise_gaussian_widget = QWidget()
        f = QFormLayout(self.noise_gaussian_widget)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 100.0)
        self.spin_sigma.setValue(1.0)
        f.addRow("Sigma:", self.spin_sigma)
        layout.addWidget(self.noise_gaussian_widget)

        # EMCCD params
        self.noise_emccd_widget = QWidget()
        f = QFormLayout(self.noise_emccd_widget)
        self.spin_rn      = QDoubleSpinBox(); self.spin_rn.setRange(0, 1000); self.spin_rn.setValue(60.0)
        self.spin_f       = QDoubleSpinBox(); self.spin_f.setRange(1.0, 3.0); self.spin_f.setValue(1.4); self.spin_f.setSingleStep(0.05)
        self.spin_emgain  = QDoubleSpinBox(); self.spin_emgain.setRange(1, 5000); self.spin_emgain.setValue(300.0)
        self.spin_cic     = QDoubleSpinBox(); self.spin_cic.setRange(0, 10); self.spin_cic.setValue(0.0)
        self.spin_qe      = QDoubleSpinBox(); self.spin_qe.setRange(0.01, 1.0); self.spin_qe.setValue(1.0); self.spin_qe.setSingleStep(0.05)
        f.addRow("Readout noise (e⁻):", self.spin_rn)
        f.addRow("Excess noise f:", self.spin_f)
        f.addRow("EM gain:", self.spin_emgain)
        f.addRow("CIC rate c:", self.spin_cic)
        f.addRow("Quantum eff. q:", self.spin_qe)
        lbl_emccd_hint = QLabel("Typical: readnoise=60, f=1.4, gain=300")
        lbl_emccd_hint.setObjectName("dim")
        self.noise_emccd_widget.layout().addRow(lbl_emccd_hint)
        self.noise_emccd_widget.setVisible(False)
        layout.addWidget(self.noise_emccd_widget)

        # CCD params
        self.noise_ccd_widget = QWidget()
        f = QFormLayout(self.noise_ccd_widget)
        self.spin_rn_adu  = QDoubleSpinBox(); self.spin_rn_adu.setRange(0, 100); self.spin_rn_adu.setValue(1.36); self.spin_rn_adu.setSingleStep(0.01)
        self.spin_gain_ccd = QDoubleSpinBox(); self.spin_gain_ccd.setRange(0.1, 100); self.spin_gain_ccd.setValue(7.7); self.spin_gain_ccd.setSingleStep(0.1)
        f.addRow("Readout noise (ADU):", self.spin_rn_adu)
        f.addRow("Gain (e⁻/ADU):", self.spin_gain_ccd)
        self.noise_ccd_widget.setVisible(False)
        layout.addWidget(self.noise_ccd_widget)
        return grp

    def _build_sgd_group(self):
        grp    = QGroupBox("SGD settings")
        layout = QFormLayout(grp)

        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(11, 51)
        self.spin_kernel.setValue(21)
        self.spin_kernel.setSingleStep(2)
        layout.addRow("PSF kernel size (px):", self.spin_kernel)
        lbl_k = QLabel("Should be ~5× typical seeing FWHM")
        lbl_k.setObjectName("dim")
        layout.addRow("", lbl_k)

        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(1, 5)
        self.spin_iterations.setValue(1)
        layout.addRow("Iterations over spool:", self.spin_iterations)
        lbl_i = QLabel("More iterations = better convergence, slower")
        lbl_i.setObjectName("dim")
        layout.addRow("", lbl_i)

        # Gradient clip slider (log scale 1e-3 to 1e-1)
        clip_widget = QWidget()
        clip_row    = QHBoxLayout(clip_widget)
        clip_row.setContentsMargins(0, 0, 0, 0)
        self.slider_clip = QSlider(Qt.Orientation.Horizontal)
        self.slider_clip.setRange(0, 100)
        self.slider_clip.setValue(35)   # ≈ 5e-3 in log scale
        self.lbl_clip_val = QLabel("5e-3")
        self.lbl_clip_val.setFixedWidth(50)
        self.slider_clip.valueChanged.connect(self._update_clip_label)
        clip_row.addWidget(self.slider_clip)
        clip_row.addWidget(self.lbl_clip_val)
        layout.addRow("Gradient clip:", clip_widget)
        lbl_cl = QLabel("Max fractional pixel change per update")
        lbl_cl.setObjectName("dim")
        layout.addRow("", lbl_cl)

        self.spin_lr_scene = QDoubleSpinBox()
        self.spin_lr_scene.setRange(1e-4, 1e-2)
        self.spin_lr_scene.setValue(1e-3)
        self.spin_lr_scene.setSingleStep(1e-4)
        self.spin_lr_scene.setDecimals(4)
        layout.addRow("Scene learning rate:", self.spin_lr_scene)

        self.spin_lr_kernel = QDoubleSpinBox()
        self.spin_lr_kernel.setRange(1e-4, 1e-2)
        self.spin_lr_kernel.setValue(1e-3)
        self.spin_lr_kernel.setSingleStep(1e-4)
        self.spin_lr_kernel.setDecimals(4)
        layout.addRow("Kernel learning rate:", self.spin_lr_kernel)

        self.spin_l1 = QDoubleSpinBox()
        self.spin_l1.setRange(1e-4, 0.1)
        self.spin_l1.setValue(1e-3)
        self.spin_l1.setSingleStep(1e-4)
        self.spin_l1.setDecimals(4)
        layout.addRow("L1 kernel reg.:", self.spin_l1)
        lbl_l1 = QLabel("Guards against noise in PSF kernel")
        lbl_l1.setObjectName("dim")
        layout.addRow("", lbl_l1)

        self.spin_save_every = QSpinBox()
        self.spin_save_every.setRange(10, 500)
        self.spin_save_every.setValue(100)
        layout.addRow("Save every N updates:", self.spin_save_every)

        self.chk_sky = QCheckBox("Subtract sky background per frame")
        self.chk_sky.setChecked(True)
        layout.addRow("", self.chk_sky)

        return grp

    def _build_output_group(self):
        grp    = QGroupBox("Output")
        layout = QFormLayout(grp)

        out_row = QHBoxLayout()
        self.edit_out_dir = QLineEdit()
        self.edit_out_dir.setPlaceholderText("Output folder…")
        out_row.addWidget(self.edit_out_dir)
        btn_out = QPushButton("Browse")
        btn_out.setFixedWidth(70)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(btn_out)
        layout.addRow("Output folder:", out_row)

        self.edit_fname = QLineEdit("thresher_scene")
        layout.addRow("Filename prefix:", self.edit_fname)

        self.chk_load_siril = QCheckBox("Load final result in Siril")
        self.chk_load_siril.setChecked(True)
        layout.addRow("", self.chk_load_siril)

        self.chk_keep_spool = QCheckBox("Keep spool FITS after processing")
        self.chk_keep_spool.setChecked(False)
        layout.addRow("", self.chk_keep_spool)

        self.chk_export_fits_seq = QCheckBox("Export spool as FITS sequence folder")
        self.chk_export_fits_seq.setChecked(False)
        self.chk_export_fits_seq.setToolTip(
            "Saves every frame from the spool as individual .fit files. "
            "Useful for feeding into other Siril scripts (speckle, DIPLI, etc.)")
        layout.addRow("", self.chk_export_fits_seq)

        return grp

    def _build_right_panel(self):
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1: Scene preview
        scene_tab = QWidget()
        sl = QVBoxLayout(scene_tab)
        self.scene_canvas = SceneCanvas()
        sl.addWidget(self.scene_canvas)
        self.lbl_update_counter = QLabel("Waiting to start…")
        self.lbl_update_counter.setObjectName("dim")
        self.lbl_update_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(self.lbl_update_counter)
        self.tabs.addTab(scene_tab, "🔬  Scene")

        # Tab 2: Loss curve
        loss_tab = QWidget()
        ll = QVBoxLayout(loss_tab)
        self.loss_canvas = LossCanvas()
        ll.addWidget(self.loss_canvas)

        stats_grp  = QGroupBox("Convergence stats")
        stats_form = QFormLayout(stats_grp)
        self.lbl_current_loss = QLabel("—")
        self.lbl_best_loss    = QLabel("—")
        self.lbl_conv_status  = QLabel("—")
        stats_form.addRow("Current loss:", self.lbl_current_loss)
        stats_form.addRow("Best loss:",    self.lbl_best_loss)
        stats_form.addRow("Status:",       self.lbl_conv_status)
        ll.addWidget(stats_grp)
        self.tabs.addTab(loss_tab, "📈  Loss")

        # Tab 3: Log
        log_tab = QWidget()
        llt = QVBoxLayout(log_tab)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Log output will appear here…")
        llt.addWidget(self.log_text)
        self.tabs.addTab(log_tab, "📋  Log")

        layout.addWidget(self.tabs)
        return panel

    # ── Signals and slots ────────────────────────────────────────

    def _update_clip_label(self, val):
        # Map slider 0-100 → log scale 1e-3 to 1e-1
        log_val = -3.0 + val / 100.0 * 2.0   # -3 to -1
        clip    = 10 ** log_val
        self.lbl_clip_val.setText(f"{clip:.2e}")

    def _get_clip_value(self):
        val     = self.slider_clip.value()
        log_val = -3.0 + val / 100.0 * 2.0
        return 10 ** log_val

    def _toggle_init_path(self, idx):
        self.init_path_widget.setVisible(idx == 1)

    def _toggle_noise_params(self, idx):
        self.noise_gaussian_widget.setVisible(idx == 0)
        self.noise_emccd_widget.setVisible(idx == 1)
        self.noise_ccd_widget.setVisible(idx == 2)

    def _update_input_hint(self, idx):
        hints = [
            "Select a folder containing .fit/.fits/.fts files",
            "Select a FITS cube file with shape (N_frames, H, W)",
            "Select a .ser file (SharpCap, FireCapture, Lucam Recorder)"
        ]
        self.lbl_input_hint.setText(hints[idx])

    def _browse_input(self):
        idx = self.combo_input_type.currentIndex()
        if idx == 0:
            path = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        elif idx == 1:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select spool FITS", "", "FITS files (*.fit *.fits *.fts)")
        else:  # SER
            path, _ = QFileDialog.getOpenFileName(
                self, "Select SER file", "", "SER files (*.ser);;All files (*)")
        if path:
            self.edit_input_path.setText(path)

    def _toggle_live_watch_ui(self, checked):
        # Only show the explicit watch-folder row when input is NOT a folder type
        # (for FITS folder input the watch dir IS the input dir automatically)
        is_folder_input = self.combo_input_type.currentIndex() == 0
        self.live_watch_folder_widget.setVisible(checked and not is_folder_input)
        self._lbl_live_hint.setVisible(checked)

    def _browse_live_watch(self):
        path = QFileDialog.getExistingDirectory(self, "Select live-watch folder")
        if path:
            self.edit_live_watch_dir.setText(path)

    def _browse_init(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select init FITS", "", "FITS files (*.fit *.fits *.fts)")
        if path:
            self.edit_init_path.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.edit_out_dir.setText(path)

    def _inspect_input(self):
        path = self.edit_input_path.text().strip()
        if not path:
            self.lbl_input_info.setText("No path set.")
            return
        try:
            if self.combo_input_type.currentIndex() == 0:
                files = sorted(
                    glob.glob(os.path.join(path, "*.fit"))  +
                    glob.glob(os.path.join(path, "*.fits")) +
                    glob.glob(os.path.join(path, "*.fts"))
                )
                if not files:
                    self.lbl_input_info.setText("No FITS files found.")
                    return
                d  = astropy_fits.getdata(files[0]).astype(np.float32)
                if d.ndim == 3: d = d[0]
                n  = len(files)
                h, w = d.shape
                # ETA estimate: ~1 update/sec on CPU, ~50 on GPU
                rate = 50 if HAS_CUDA else 2
                iter_val = self.spin_iterations.value()
                eta_s = n * iter_val / rate
                eta_m = int(eta_s // 60)
                self.lbl_input_info.setText(
                    f"{n} frames  |  {w}×{h} px  |  "
                    f"Est. {eta_m}m ({'GPU' if HAS_CUDA else 'CPU'})")
            elif idx == 1:  # FITS spool
                with astropy_fits.open(path) as hdul:
                    shape = hdul[0].data.shape
                if len(shape) < 3:
                    self.lbl_input_info.setText("Not a valid spool cube.")
                    return
                n, h, w = shape[0], shape[1], shape[2]
                rate = 50 if HAS_CUDA else 2
                eta_s = n * self.spin_iterations.value() / rate
                eta_m = int(eta_s // 60)
                self.lbl_input_info.setText(
                    f"{n} frames  |  {w}×{h} px  |  "
                    f"Est. {eta_m}m ({'GPU' if HAS_CUDA else 'CPU'})")
            elif idx == 2:  # SER file
                hdr  = read_ser_header(path)
                n    = hdr["n_frames"]
                w, h = hdr["width"], hdr["height"]
                depth = hdr["pixel_depth"]
                color = "color" if hdr["color_id"] in (SER_RGB, SER_BGR) else "mono"
                rate  = 50 if HAS_CUDA else 2
                eta_s = n * self.spin_iterations.value() / rate
                eta_m = int(eta_s // 60)
                self.lbl_input_info.setText(
                    f"{n} frames  |  {w}×{h} px  |  {depth}-bit {color}  |  "
                    f"Est. {eta_m}m ({'GPU' if HAS_CUDA else 'CPU'})")
        except Exception as e:
            self.lbl_input_info.setText(f"Error: {e}")

    def _on_run(self):
        if not HAS_TORCH:
            self._show_torch_dialog()
            return

        input_path = self.edit_input_path.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Input required", "Please set an input path.")
            return

        out_dir = self.edit_out_dir.text().strip()
        if not out_dir:
            # Default alongside input
            if self.combo_input_type.currentIndex() == 0:
                out_dir = input_path
            else:
                out_dir = os.path.dirname(input_path)
            self.edit_out_dir.setText(out_dir)

        # Build config
        noise_model  = ["gaussian", "emccd", "ccd"][self.combo_noise.currentIndex()]
        noise_params = {}
        if noise_model == "gaussian":
            noise_params = {"sigma": self.spin_sigma.value()}
        elif noise_model == "emccd":
            noise_params = {
                "readout_noise": self.spin_rn.value(),
                "f":             self.spin_f.value(),
                "em_gain":       self.spin_emgain.value(),
                "c":             self.spin_cic.value(),
                "q":             self.spin_qe.value(),
            }
        else:
            noise_params = {
                "readout_noise_adu": self.spin_rn_adu.value(),
                "gain":              self.spin_gain_ccd.value(),
            }

        idx = self.combo_input_type.currentIndex()
        if idx == 0:
            cfg = {
                "input_type":        "fits_dir",
                "fits_dir":          input_path,
                "max_frames":        self.spin_max_frames.value(),
            }
        elif idx == 2:
            cfg = {
                "input_type":        "ser",
                "ser_path":          input_path,
                "max_frames":        self.spin_max_frames.value(),
            }
        else:
            cfg = {
                "input_type":        "spool",
                "spool_path":        input_path,
            }

        init_path = ""
        if self.combo_init.currentIndex() == 1:
            init_path = self.edit_init_path.text().strip()

        cfg.update({
            "init_path":        init_path,
            "noise_model":      noise_model,
            "noise_params":     noise_params,
            "kernel_size":      self.spin_kernel.value(),
            "iterations":       self.spin_iterations.value(),
            "proportional_clip": self._get_clip_value(),
            "lr_scene":         self.spin_lr_scene.value(),
            "lr_kernel":        self.spin_lr_kernel.value(),
            "l1_kernel":        self.spin_l1.value(),
            "save_every":       self.spin_save_every.value(),
            "subtract_sky":     self.chk_sky.isChecked(),
            "output_dir":       out_dir,
            "output_name":      self.edit_fname.text().strip() or "thresher_scene",
            "load_in_siril":    self.chk_load_siril.isChecked(),
            "keep_spool":       self.chk_keep_spool.isChecked(),
            "export_fits_seq":  self.chk_export_fits_seq.isChecked(),
            "live_watch":       self.chk_live_watch.isChecked(),
            "live_watch_dir":   self.edit_live_watch_dir.text().strip(),
        })

        # Reset UI
        self.log_text.clear()
        self.loss_canvas.reset()
        self.lbl_current_loss.setText("—")
        self.lbl_best_loss.setText("—")
        self.lbl_conv_status.setText("—")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._update_num   = 0
        self._best_loss    = float("inf")
        self._start_time   = time.time()

        self._cancel_event.clear()
        self._worker = ThresherWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log)
        self._worker.scene_updated.connect(self._on_scene_updated)
        self._worker.loss_point.connect(self._on_loss_point)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_cancel(self):
        self._cancel_event.set()
        self.btn_cancel.setEnabled(False)
        self.status_label.setText("Cancelling…")

    def _on_progress(self, current, total, desc):
        if total > 0:
            self._update_num   = current
            self._total_updates = total
            pct = int(100 * current / total)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

            if self._start_time and current > 0:
                eta = estimate_eta(current, total, self._start_time)
                loss_str = f"Loss: {self._last_loss:.4f}  |  " if hasattr(self, "_last_loss") else ""
                self.status_label.setText(
                    f"Update {current}/{total} ({pct}%)  |  {loss_str}{eta}")
            self.lbl_update_counter.setText(
                f"Update: {current} / {total}  ({pct}%)")

    def _on_log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {text}")
        self.log_text.ensureCursorVisible()

    def _on_scene_updated(self, scene_np):
        if self._init_scene is None:
            self._init_scene = scene_np
            self.scene_canvas.show_init(scene_np)
        else:
            self.scene_canvas.show_scene(scene_np, self._update_num)

    def _on_loss_point(self, update_num, loss):
        self._last_loss = loss
        self.loss_canvas.add_point(update_num, loss)
        self.lbl_current_loss.setText(f"{loss:.4f}")
        if loss < self._best_loss:
            self._best_loss = loss
            self.lbl_best_loss.setText(f"{loss:.4f}  @ update {update_num}")
        status = self.loss_canvas.get_convergence_status()
        self.lbl_conv_status.setText(status)

    def _on_finished(self, result):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

        if result.get("success"):
            n      = result.get("n_updates", 0)
            path   = result.get("output_path", "")
            elapsed = result.get("elapsed_sec", 0)
            mins   = int(elapsed // 60)
            secs   = int(elapsed % 60)
            self.status_label.setText(
                f"✓  Done — {n} updates in {mins}m {secs}s  |  Saved: {path}")
            self.scene_canvas.show_scene(
                self.scene_canvas.ax_scene.images[0].get_array()
                if self.scene_canvas.ax_scene.images else np.zeros((10,10)),
                n)
            QMessageBox.information(self, "Complete",
                f"The Thresher finished.\n\n"
                f"Updates:  {n}\n"
                f"Time:     {mins}m {secs}s\n"
                f"Output:   {path}")
        else:
            err = result.get("error", "Unknown error")
            self.status_label.setText(f"✗  Failed: {err}")
            if "PyTorch" not in err:
                QMessageBox.critical(self, "Error", f"Processing failed:\n{err}")

    def _check_torch(self):
        if not HAS_TORCH:
            self._show_torch_dialog()

    def _show_torch_dialog(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("PyTorch Required")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            "<b>PyTorch is required for The Thresher.</b><br><br>"
            "Install from <a href='https://pytorch.org/get-started/locally/'>"
            "https://pytorch.org/get-started/locally/</a><br><br>"
            "<b>CPU only:</b><br>"
            "<code>pip install torch torchvision</code><br><br>"
            "<b>CUDA 12.1 (GPU):</b><br>"
            "<code>pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121</code><br><br>"
            "<b>Performance:</b><br>"
            "• CPU: ~1–5 updates/sec (OK for small images)<br>"
            "• GPU: ~50–500 updates/sec (recommended)"
        )
        msg.exec()


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
