"""
wavelet_stretch.py — Siril Script #7
Wavelet-based per-layer stretch for linear FITS images.
Uses A-trous ("with holes") wavelet decomposition.
Requires: PyQt6, astropy, matplotlib, scipy
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")

import os
import sys
import json
import threading
from datetime import datetime

import numpy as np
from scipy.ndimage import convolve1d
from astropy.io import fits
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import sirilpy as s
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QScrollArea, QSizePolicy, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ============================================================
# SIRIL THEME
# ============================================================

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
QPushButton#preset {{
    background-color: {SIRIL_BG3};
    border-color: {SIRIL_SECTION};
    color: {SIRIL_SECTION};
    text-align: center;
    border-radius: 3px;
    padding: 4px 8px;
}}
QPushButton#preset:hover {{
    background-color: #1a2540;
    border-color: {SIRIL_ACCENT};
}}
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
QSplitter::handle {{
    background: {SIRIL_BORDER};
    width: 2px;
}}
"""

# ============================================================
# A-TROUS WAVELET ALGORITHM
# ============================================================

B3_KERNEL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def atrous_decompose(image: np.ndarray, levels: int = 6) -> list:
    layers  = []
    current = image.astype(np.float64).copy()
    for level in range(levels):
        step   = 2 ** level
        k_size = step * (len(B3_KERNEL) - 1) + 1
        kernel = np.zeros(k_size)
        kernel[::step] = B3_KERNEL
        smoothed = convolve1d(current, kernel, axis=0, mode='reflect')
        smoothed = convolve1d(smoothed, kernel, axis=1, mode='reflect')
        layers.append(current - smoothed)
        current = smoothed
    layers.append(current)
    return layers


def atrous_reconstruct(layers: list) -> np.ndarray:
    result = np.zeros_like(layers[0], dtype=np.float64)
    for layer in layers:
        result += layer
    return result


def verify_reconstruction(image: np.ndarray, levels: int = 6) -> float:
    layers = atrous_decompose(image, levels)
    recon  = atrous_reconstruct(layers)
    return float(np.max(np.abs(image - recon)))


# ============================================================
# STRETCH FUNCTIONS
# ============================================================

def stretch_linear(layer: np.ndarray, strength: float) -> np.ndarray:
    return layer * strength


def stretch_asinh(layer: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return layer
    return np.arcsinh(layer * strength) / np.arcsinh(strength + 1e-10)


def stretch_log(layer: np.ndarray, strength: float) -> np.ndarray:
    positive = np.clip(layer, 0, None)
    if strength <= 0:
        return positive
    return np.log1p(positive * strength) / np.log1p(strength)


def stretch_ghs(layer: np.ndarray, strength: float,
                sp: float = 0.0, hp: float = 1.0) -> np.ndarray:
    if strength <= 0:
        return layer
    stretched = stretch_asinh(layer, strength)
    shadow_mask = layer < sp
    stretched[shadow_mask] = layer[shadow_mask]
    return stretched


def stretch_none(layer: np.ndarray, strength: float = 1.0) -> np.ndarray:
    return layer


STRETCH_FUNCTIONS = {
    "Linear": stretch_linear,
    "Asinh":  stretch_asinh,
    "Log":    stretch_log,
    "GHS":    stretch_ghs,
    "None":   stretch_none,
}

# ============================================================
# PRESETS
# ============================================================

PRESET_NEBULA = {
    "name": "Nebula",
    "description": "Protect stars, aggressively stretch diffuse emission",
    "levels": 6,
    "layers": [
        {"method": "Linear", "strength": 0.8,  "enabled": True},
        {"method": "Linear", "strength": 1.0,  "enabled": True},
        {"method": "Asinh",  "strength": 2.0,  "enabled": True},
        {"method": "Asinh",  "strength": 5.0,  "enabled": True},
        {"method": "Asinh",  "strength": 10.0, "enabled": True},
        {"method": "Asinh",  "strength": 15.0, "enabled": True},
        {"method": "Linear", "strength": 1.0,  "enabled": True},
    ]
}

PRESET_GALAXY = {
    "name": "Galaxy",
    "description": "Balance core detail with outer arm visibility",
    "levels": 6,
    "layers": [
        {"method": "Linear", "strength": 1.0,  "enabled": True},
        {"method": "Asinh",  "strength": 1.5,  "enabled": True},
        {"method": "Asinh",  "strength": 3.0,  "enabled": True},
        {"method": "Asinh",  "strength": 6.0,  "enabled": True},
        {"method": "Asinh",  "strength": 8.0,  "enabled": True},
        {"method": "Asinh",  "strength": 10.0, "enabled": True},
        {"method": "Linear", "strength": 1.0,  "enabled": True},
    ]
}

PRESET_NARROWBAND = {
    "name": "Narrowband",
    "description": "Very aggressive coarse stretch for low-SNR narrowband",
    "levels": 6,
    "layers": [
        {"method": "Linear", "strength": 0.7,  "enabled": True},
        {"method": "Linear", "strength": 0.9,  "enabled": True},
        {"method": "Asinh",  "strength": 3.0,  "enabled": True},
        {"method": "Asinh",  "strength": 8.0,  "enabled": True},
        {"method": "Asinh",  "strength": 20.0, "enabled": True},
        {"method": "Asinh",  "strength": 30.0, "enabled": True},
        {"method": "Linear", "strength": 1.0,  "enabled": True},
    ]
}

PRESET_STAR_FIELD = {
    "name": "Stars",
    "description": "Balanced stretch, protect star sizes",
    "levels": 6,
    "layers": [
        {"method": "None",   "strength": 1.0,  "enabled": True},
        {"method": "Linear", "strength": 1.2,  "enabled": True},
        {"method": "Asinh",  "strength": 2.0,  "enabled": True},
        {"method": "Asinh",  "strength": 4.0,  "enabled": True},
        {"method": "Asinh",  "strength": 6.0,  "enabled": True},
        {"method": "Asinh",  "strength": 8.0,  "enabled": True},
        {"method": "Linear", "strength": 1.0,  "enabled": True},
    ]
}

PRESET_GENTLE = {
    "name": "Gentle",
    "description": "Conservative stretch for well-exposed data",
    "levels": 6,
    "layers": [
        {"method": "Linear", "strength": 1.0,  "enabled": True},
        {"method": "Linear", "strength": 1.0,  "enabled": True},
        {"method": "Asinh",  "strength": 1.5,  "enabled": True},
        {"method": "Asinh",  "strength": 3.0,  "enabled": True},
        {"method": "Asinh",  "strength": 5.0,  "enabled": True},
        {"method": "Asinh",  "strength": 7.0,  "enabled": True},
        {"method": "Linear", "strength": 1.0,  "enabled": True},
    ]
}

ALL_PRESETS = [PRESET_NEBULA, PRESET_GALAXY, PRESET_NARROWBAND,
               PRESET_STAR_FIELD, PRESET_GENTLE]

# ============================================================
# LINEARITY DETECTION
# ============================================================

def check_linearity(data: np.ndarray) -> tuple:
    flat = data.flatten()
    mn   = float(np.min(flat))
    mx   = float(np.max(flat))
    if mx <= mn:
        return True, 0.0
    med  = float(np.median(flat))
    frac = (med - mn) / (mx - mn)
    return frac < 0.25, frac

# ============================================================
# FULL STRETCH PIPELINE
# ============================================================

def apply_wavelet_stretch(image: np.ndarray,
                          layer_configs: list,
                          levels: int = 6) -> np.ndarray:
    assert image.ndim == 2, "Input must be 2D (single channel)"
    assert len(layer_configs) == levels + 1, \
        f"Need {levels+1} layer configs, got {len(layer_configs)}"

    img_min = image.min()
    img_max = image.max()
    if img_max <= img_min:
        return image.copy()
    normalized = (image - img_min) / (img_max - img_min)

    layers = atrous_decompose(normalized, levels)

    stretched_layers = []
    for i, (layer, config) in enumerate(zip(layers, layer_configs)):
        if not config.get("enabled", True):
            stretched_layers.append(layer)
            continue
        method   = config.get("method", "Linear")
        strength = float(config.get("strength", 1.0))
        fn       = STRETCH_FUNCTIONS.get(method, stretch_linear)
        stretched_layers.append(fn(layer, strength))

    result = atrous_reconstruct(stretched_layers)
    return np.clip(result, 0.0, 1.0)


def apply_wavelet_stretch_rgb(data: np.ndarray,
                              layer_configs: list,
                              levels: int = 6,
                              mode: str = "luminance") -> np.ndarray:
    if data.ndim == 2:
        return apply_wavelet_stretch(data, layer_configs, levels)

    if data.shape[0] == 3:
        R = data[0].astype(np.float64)
        G = data[1].astype(np.float64)
        B = data[2].astype(np.float64)
        layout = "CHW"
    else:
        R = data[:,:,0].astype(np.float64)
        G = data[:,:,1].astype(np.float64)
        B = data[:,:,2].astype(np.float64)
        layout = "HWC"

    def norm(ch):
        mn, mx = ch.min(), ch.max()
        return (ch - mn) / (mx - mn + 1e-10), mn, mx

    R_n, Rmin, Rmax = norm(R)
    G_n, Gmin, Gmax = norm(G)
    B_n, Bmin, Bmax = norm(B)

    if mode == "luminance":
        lum = 0.299 * R_n + 0.587 * G_n + 0.114 * B_n
        lum_stretched = apply_wavelet_stretch(lum, layer_configs, levels)
        ratio = np.where(lum > 1e-10, lum_stretched / (lum + 1e-10), 1.0)
        ratio = np.clip(ratio, 0, 10)
        R_out = np.clip(R_n * ratio, 0, 1)
        G_out = np.clip(G_n * ratio, 0, 1)
        B_out = np.clip(B_n * ratio, 0, 1)
    else:  # per_channel or combined
        R_out = apply_wavelet_stretch(R_n, layer_configs, levels)
        G_out = apply_wavelet_stretch(G_n, layer_configs, levels)
        B_out = apply_wavelet_stretch(B_n, layer_configs, levels)

    if layout == "CHW":
        return np.stack([R_out, G_out, B_out], axis=0).astype(np.float32)
    else:
        return np.stack([R_out, G_out, B_out], axis=2).astype(np.float32)

# ============================================================
# WORKER THREAD
# ============================================================

class StretchWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    preview_ready = pyqtSignal(object)
    finished      = pyqtSignal(dict)

    def __init__(self, fits_path: str, layer_configs: list,
                 levels: int, rgb_mode: str,
                 output_path: str, cancel_event: threading.Event,
                 original_data=None):
        super().__init__()
        self.fits_path     = fits_path
        self.layer_configs = layer_configs
        self.levels        = levels
        self.rgb_mode      = rgb_mode
        self.output_path   = output_path
        self._cancel       = cancel_event
        self.original_data = original_data

    def run(self):
        try:
            self.log_line.emit(f"[{datetime.now().strftime('%H:%M:%S')}] Loading: {self.fits_path}")
            self.progress.emit(1, 5, "Loading FITS file...")
            data, header = self._load_fits()

            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"})
                return

            self.log_line.emit(f"Image shape: {data.shape}, dtype: {data.dtype}")

            # Reconstruction verification
            self.progress.emit(2, 5, "Verifying reconstruction accuracy...")
            if data.ndim == 2:
                test_ch = data.astype(np.float64)
            elif data.shape[0] == 3:
                test_ch = data[0].astype(np.float64)
            else:
                test_ch = data[:,:,0].astype(np.float64)

            mn, mx = test_ch.min(), test_ch.max()
            if mx > mn:
                test_norm = (test_ch - mn) / (mx - mn)
                err = verify_reconstruction(test_norm, self.levels)
                self.log_line.emit(f"Reconstruction error: {err:.2e} (must be < 1e-8)")
                if err >= 1e-8:
                    self.finished.emit({
                        "success": False,
                        "error": f"Reconstruction verification failed (error={err:.2e}). "
                                  "Platform numerical issue detected."
                    })
                    return

            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"})
                return

            self.log_line.emit(f"Decomposing into {self.levels} wavelet levels...")
            self.progress.emit(3, 5, "Applying wavelet decomposition & stretch...")

            if data.ndim == 2:
                result = apply_wavelet_stretch(
                    data.astype(np.float64), self.layer_configs, self.levels)
            else:
                result = apply_wavelet_stretch_rgb(
                    data.astype(np.float64), self.layer_configs,
                    self.levels, self.rgb_mode)

            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"})
                return

            self.progress.emit(4, 5, "Generating preview...")
            self.preview_ready.emit(result)

            self.progress.emit(5, 5, "Saving output FITS...")
            self._save_fits(result, header)

            self.log_line.emit(f"[{datetime.now().strftime('%H:%M:%S')}] Saved: {self.output_path}")
            self.finished.emit({"success": True, "output": self.output_path})

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _load_fits(self):
        with fits.open(self.fits_path) as hdul:
            data   = hdul[0].data.astype(np.float64)
            header = hdul[0].header.copy()
        return data, header

    def _save_fits(self, result: np.ndarray, header):
        out_data = result.astype(np.float32)
        hdu = fits.PrimaryHDU(data=out_data, header=header)
        hdu.header["HISTORY"] = "Wavelet stretch applied by wavelet_stretch.py"
        hdu.writeto(self.output_path, overwrite=True)

# ============================================================
# PREVIEW CANVAS
# ============================================================

class PreviewCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4), facecolor="#1e2128")
        self.ax_orig    = self.fig.add_subplot(1, 2, 1)
        self.ax_stretch = self.fig.add_subplot(1, 2, 2)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self.original_data  = None
        self.stretched_data = None

    def _style_axes(self):
        for ax in [self.ax_orig, self.ax_stretch]:
            ax.set_facecolor("#1e2128")
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")
        self.ax_orig.set_title("Original (linear)",
                               color="#7a8499", fontsize=9)
        self.ax_stretch.set_title("Wavelet stretched",
                                  color="#4a9eff", fontsize=9)

    def show_original(self, data: np.ndarray):
        self.original_data = data
        self.ax_orig.clear()
        self._style_axes()
        display = self._prepare_display(self.downsample(data))
        self.ax_orig.imshow(display, cmap="gray", origin="lower",
                            aspect="equal", interpolation="nearest")
        self.ax_orig.set_title("Original (linear)", color="#7a8499", fontsize=9)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def show_stretched(self, data: np.ndarray):
        self.stretched_data = data
        self.ax_stretch.clear()
        self._style_axes()
        display = self._prepare_display(self.downsample(data))
        self.ax_stretch.imshow(display, cmap="gray", origin="lower",
                               aspect="equal", interpolation="nearest")
        self.ax_stretch.set_title("Wavelet stretched", color="#4a9eff", fontsize=9)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def _prepare_display(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            if data.shape[0] == 3:
                display = np.transpose(data, (1, 2, 0))
            else:
                display = data
            display = (display - display.min()) / (display.max() - display.min() + 1e-10)
            return np.clip(display, 0, 1)
        else:
            p_low  = np.percentile(data, 0.5)
            p_high = np.percentile(data, 99.5)
            display = (data - p_low) / (p_high - p_low + 1e-10)
            return np.clip(display, 0, 1)

    def downsample(self, data: np.ndarray, max_px: int = 800) -> np.ndarray:
        if data.ndim == 2:
            h, w = data.shape
        elif data.shape[0] == 3:
            _, h, w = data.shape
        else:
            h, w, _ = data.shape
        factor = max(1, max(h, w) // max_px)
        if factor == 1:
            return data
        if data.ndim == 2:
            return data[::factor, ::factor]
        elif data.shape[0] == 3:
            return data[:, ::factor, ::factor]
        else:
            return data[::factor, ::factor, :]

# ============================================================
# LAYER ROW WIDGET
# ============================================================

class LayerRow(QWidget):
    changed = pyqtSignal()

    def __init__(self, layer_idx: int, is_residual: bool = False, parent=None):
        super().__init__(parent)
        self.layer_idx   = layer_idx
        self.is_residual = is_residual
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(4)

        self.enabled_cb = QCheckBox()
        self.enabled_cb.setChecked(True)
        self.enabled_cb.setFixedWidth(18)
        self.enabled_cb.stateChanged.connect(self.changed.emit)
        layout.addWidget(self.enabled_cb)

        if self.is_residual:
            label_text = "Residual  (background)"
        else:
            scale = 2 ** self.layer_idx
            label_text = f"L{self.layer_idx}  ({scale}px scale)"
        self.lbl = QLabel(label_text)
        self.lbl.setFixedWidth(110)
        self.lbl.setObjectName("dim")
        layout.addWidget(self.lbl)

        self.method_cb = QComboBox()
        self.method_cb.addItems(list(STRETCH_FUNCTIONS.keys()))
        self.method_cb.setFixedWidth(72)
        self.method_cb.currentTextChanged.connect(self.changed.emit)
        layout.addWidget(self.method_cb)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 300)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider, 1)

        self.strength_lbl = QLabel("1.0")
        self.strength_lbl.setFixedWidth(28)
        self.strength_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.strength_lbl)

    def _on_slider(self, val):
        self.strength_lbl.setText(f"{val/10:.1f}")
        self.changed.emit()

    def get_config(self) -> dict:
        return {
            "method":   self.method_cb.currentText(),
            "strength": self.slider.value() / 10.0,
            "enabled":  self.enabled_cb.isChecked(),
        }

    def set_config(self, cfg: dict):
        self.enabled_cb.setChecked(cfg.get("enabled", True))
        method = cfg.get("method", "Linear")
        idx = self.method_cb.findText(method)
        if idx >= 0:
            self.method_cb.setCurrentIndex(idx)
        strength = cfg.get("strength", 1.0)
        self.slider.setValue(int(strength * 10))

# ============================================================
# PRESET SAVE/LOAD
# ============================================================

PRESETS_FILE = os.path.join(SCRIPT_DIR, "wavelet_presets.json")

def load_custom_presets() -> list:
    if not os.path.exists(PRESETS_FILE):
        return []
    try:
        with open(PRESETS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_custom_presets(presets: list):
    try:
        with open(PRESETS_FILE, "w") as f:
            json.dump(presets, f, indent=2)
    except Exception as e:
        print(f"Failed to save presets: {e}")

# ============================================================
# SIRIL INTEGRATION
# ============================================================

def load_in_siril(fits_path: str):
    siril = s.SirilInterface()
    siril.connect()
    try:
        siril.cmd("load", fits_path)
    except Exception as e:
        print(f"Siril load error: {e}")
    finally:
        siril.disconnect()

# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wavelet Stretch — Siril")
        self.resize(1280, 800)

        self._fits_data     = None    # raw loaded data (float64)
        self._fits_path     = ""
        self._is_rgb        = False
        self._worker        = None
        self._cancel_event  = threading.Event()
        self._layer_rows    = []
        self._current_preset_name = ""
        self._custom_presets = load_custom_presets()

        self._build_ui()
        self._rebuild_layer_rows(6)

    # -------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(36)
        title_bar.setStyleSheet(f"background:{SIRIL_BG3}; border-bottom:1px solid {SIRIL_BORDER};")
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(12, 0, 12, 0)
        lbl_title = QLabel("〜  Wavelet Stretch  —  Siril")
        lbl_title.setStyleSheet(f"color:{SIRIL_ACCENT}; font-weight:bold; font-size:11pt;")
        lbl_ver   = QLabel("v1.0  |  Linear input only")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        tb_layout.addWidget(lbl_title)
        tb_layout.addStretch()
        tb_layout.addWidget(lbl_ver)
        root_layout.addWidget(title_bar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)
        root_layout.addWidget(splitter, 1)

        # ---------- LEFT PANEL ----------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(380)
        left_scroll.setStyleSheet(f"background:{SIRIL_BG}; border:none;")

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        left_scroll.setWidget(left_widget)

        # Input group
        grp_input = QGroupBox("Input file")
        g_in = QVBoxLayout(grp_input)

        row_path = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Path to linear FITS file...")
        row_path.addWidget(self.input_edit)
        btn_browse_in = QPushButton("Browse…")
        btn_browse_in.setFixedWidth(70)
        btn_browse_in.clicked.connect(self._browse_input)
        row_path.addWidget(btn_browse_in)
        g_in.addLayout(row_path)

        btn_load = QPushButton("Load & show original")
        btn_load.setObjectName("primary")
        btn_load.clicked.connect(self._load_fits)
        g_in.addWidget(btn_load)

        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("dim")
        self.lbl_info.setWordWrap(True)
        g_in.addWidget(self.lbl_info)

        self.lbl_linearity = QLabel("")
        self.lbl_linearity.setWordWrap(True)
        g_in.addWidget(self.lbl_linearity)

        left_layout.addWidget(grp_input)

        # Preset group
        grp_preset = QGroupBox("Preset")
        g_pre = QVBoxLayout(grp_preset)

        preset_btn_row = QHBoxLayout()
        preset_btn_row.setSpacing(3)
        preset_names = [p["name"] for p in ALL_PRESETS]
        self._preset_btns = []
        for p in ALL_PRESETS:
            btn = QPushButton(p["name"])
            btn.setObjectName("preset")
            btn.clicked.connect(lambda checked, preset=p: self._apply_preset(preset))
            preset_btn_row.addWidget(btn)
            self._preset_btns.append(btn)
        g_pre.addLayout(preset_btn_row)

        self.lbl_preset_desc = QLabel("Select a preset to begin")
        self.lbl_preset_desc.setObjectName("dim")
        self.lbl_preset_desc.setWordWrap(True)
        g_pre.addWidget(self.lbl_preset_desc)

        # Save/load custom
        custom_row = QHBoxLayout()
        btn_save_preset = QPushButton("💾  Save current as preset…")
        btn_save_preset.clicked.connect(self._save_preset_dialog)
        custom_row.addWidget(btn_save_preset)
        if self._custom_presets:
            self._custom_preset_cb = QComboBox()
            self._custom_preset_cb.addItem("Custom presets…")
            for cp in self._custom_presets:
                self._custom_preset_cb.addItem(cp["name"])
            self._custom_preset_cb.currentIndexChanged.connect(self._load_custom_preset)
            custom_row.addWidget(self._custom_preset_cb)
        else:
            self._custom_preset_cb = None
        g_pre.addLayout(custom_row)

        left_layout.addWidget(grp_preset)

        # Wavelet levels group
        grp_levels = QGroupBox("Wavelet levels")
        g_lev = QHBoxLayout(grp_levels)
        g_lev.addWidget(QLabel("Levels:"))
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(4, 8)
        self.levels_spin.setValue(6)
        self.levels_spin.setFixedWidth(50)
        self.levels_spin.valueChanged.connect(self._on_levels_changed)
        g_lev.addWidget(self.levels_spin)
        lbl_lvl_hint = QLabel("More levels = finer control, slower")
        lbl_lvl_hint.setObjectName("dim")
        g_lev.addWidget(lbl_lvl_hint)
        g_lev.addStretch()
        left_layout.addWidget(grp_levels)

        # Layer settings group
        grp_layers = QGroupBox("Layer settings")
        g_lay = QVBoxLayout(grp_layers)

        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        for txt, w in [("En", 18), ("Layer", 110), ("Method", 72), ("Strength", -1), ("Val", 28)]:
            lbl = QLabel(txt)
            lbl.setObjectName("dim")
            if w > 0:
                lbl.setFixedWidth(w)
            else:
                lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            hdr_row.addWidget(lbl)
        g_lay.addLayout(hdr_row)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{SIRIL_BORDER};")
        g_lay.addWidget(sep)

        self._layer_container = QWidget()
        self._layer_layout    = QVBoxLayout(self._layer_container)
        self._layer_layout.setContentsMargins(0, 0, 0, 0)
        self._layer_layout.setSpacing(2)
        g_lay.addWidget(self._layer_container)
        left_layout.addWidget(grp_layers)

        # RGB mode (hidden for mono)
        self.grp_rgb = QGroupBox("RGB mode")
        g_rgb = QVBoxLayout(self.grp_rgb)
        self.rgb_cb = QComboBox()
        self.rgb_cb.addItems(["Luminance (recommended)", "Per channel", "Combined"])
        g_rgb.addWidget(self.rgb_cb)
        lbl_rgb_hint = QLabel("Luminance preserves color ratios (avoids hue shifts)")
        lbl_rgb_hint.setObjectName("dim")
        lbl_rgb_hint.setWordWrap(True)
        g_rgb.addWidget(lbl_rgb_hint)
        self.grp_rgb.setVisible(False)
        left_layout.addWidget(self.grp_rgb)

        # Output group
        grp_out = QGroupBox("Output")
        g_out = QVBoxLayout(grp_out)

        row_out = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output FITS path (auto-filled on load)")
        row_out.addWidget(self.output_edit)
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.setFixedWidth(70)
        btn_browse_out.clicked.connect(self._browse_output)
        row_out.addWidget(btn_browse_out)
        g_out.addLayout(row_out)

        self.load_in_siril_cb = QCheckBox("Load result in Siril after processing")
        self.load_in_siril_cb.setChecked(True)
        g_out.addWidget(self.load_in_siril_cb)
        left_layout.addWidget(grp_out)

        # Action buttons
        self.btn_apply = QPushButton("▶   Apply stretch")
        self.btn_apply.setObjectName("primary")
        self.btn_apply.setMinimumHeight(36)
        self.btn_apply.clicked.connect(self._start_stretch)
        left_layout.addWidget(self.btn_apply)

        self.btn_cancel = QPushButton("✕   Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setMinimumHeight(32)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_stretch)
        left_layout.addWidget(self.btn_cancel)

        left_layout.addStretch()

        splitter.addWidget(left_scroll)

        # ---------- RIGHT PANEL ----------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        # Tab widget (Preview + Log)
        self.tabs = QTabWidget()

        # Preview tab
        preview_tab = QWidget()
        pt_layout = QVBoxLayout(preview_tab)
        pt_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PreviewCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pt_layout.addWidget(self.canvas)
        lbl_preview_note = QLabel(
            "Preview downsampled for speed. Output FITS is full resolution.")
        lbl_preview_note.setObjectName("dim")
        lbl_preview_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pt_layout.addWidget(lbl_preview_note)
        self.tabs.addTab(preview_tab, "Preview")

        # Log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(4, 4, 4, 4)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setProperty("readOnly", True)
        log_layout.addWidget(self.log_view)
        self.tabs.addTab(log_tab, "Log")

        right_layout.addWidget(self.tabs, 1)
        splitter.addWidget(right_widget)

        splitter.setSizes([380, 900])

        # ---------- BOTTOM BAR ----------
        bottom = QWidget()
        bottom.setFixedHeight(28)
        bottom.setStyleSheet(f"background:{SIRIL_BG3}; border-top:1px solid {SIRIL_BORDER};")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 0, 8, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("dim")
        bottom_layout.addWidget(self.status_lbl)

        root_layout.addWidget(bottom)

    # -------------------------------------------------------
    def _rebuild_layer_rows(self, levels: int):
        # Save existing configs
        old_configs = []
        for row in self._layer_rows:
            old_configs.append(row.get_config())

        # Clear
        for row in self._layer_rows:
            row.setParent(None)
            row.deleteLater()
        self._layer_rows = []

        # Rebuild
        n_total = levels + 1  # layers 0..levels-1 + residual
        for i in range(n_total):
            is_residual = (i == levels)
            row = LayerRow(i, is_residual)
            row.changed.connect(self._on_layer_changed)
            self._layer_layout.addWidget(row)
            self._layer_rows.append(row)

            # Restore old config if available
            if i < len(old_configs):
                row.set_config(old_configs[i])

    def _on_levels_changed(self, val):
        self._rebuild_layer_rows(val)

    def _on_layer_changed(self):
        pass  # could live-update preview in future

    # -------------------------------------------------------
    def _apply_preset(self, preset: dict):
        levels = preset.get("levels", 6)
        self.levels_spin.setValue(levels)
        self._rebuild_layer_rows(levels)

        layer_cfgs = preset.get("layers", [])
        for i, row in enumerate(self._layer_rows):
            if i < len(layer_cfgs):
                row.set_config(layer_cfgs[i])

        self.lbl_preset_desc.setText(preset.get("description", ""))
        self._current_preset_name = preset["name"].lower().replace(" ", "_").replace("/", "_")
        self._update_output_path()
        self._log(f"Preset loaded: {preset['name']}")

    def _load_custom_preset(self, idx):
        if self._custom_preset_cb is None or idx <= 0:
            return
        cp = self._custom_presets[idx - 1]
        self._apply_preset(cp)

    def _save_preset_dialog(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        configs = [row.get_config() for row in self._layer_rows]
        new_preset = {
            "name":        name.strip(),
            "description": f"Custom preset saved {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "levels":      self.levels_spin.value(),
            "layers":      configs,
        }
        self._custom_presets.append(new_preset)
        save_custom_presets(self._custom_presets)
        self._log(f"Preset saved: {name.strip()}")
        QMessageBox.information(self, "Preset saved",
                                f"Preset '{name.strip()}' saved to wavelet_presets.json")

    # -------------------------------------------------------
    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FITS file", "",
            "FITS files (*.fit *.fits *.fts);;All files (*.*)")
        if path:
            self.input_edit.setText(path)
            self._fits_path = path
            self._update_output_path()

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save output FITS", self.output_edit.text(),
            "FITS files (*.fit *.fits);;All files (*.*)")
        if path:
            self.output_edit.setText(path)

    def _update_output_path(self):
        in_path = self.input_edit.text().strip()
        if not in_path:
            return
        base, ext = os.path.splitext(in_path)
        if not ext:
            ext = ".fit"
        suffix = "_wavelet"
        if self._current_preset_name:
            suffix += f"_{self._current_preset_name}"
        self.output_edit.setText(base + suffix + ext)

    # -------------------------------------------------------
    def _load_fits(self):
        path = self.input_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Please select a FITS file first.")
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "Not found", f"File not found:\n{path}")
            return

        self._fits_path = path
        self._set_status("Loading FITS…")
        try:
            with fits.open(path) as hdul:
                data   = hdul[0].data.astype(np.float64)
                header = hdul[0].header
                naxis  = header.get("NAXIS", data.ndim)

            self._fits_data = data

            # Determine mono vs RGB
            if data.ndim == 3 and (data.shape[0] == 3 or data.shape[2] == 3):
                self._is_rgb = True
                h = data.shape[1] if data.shape[0] == 3 else data.shape[0]
                w = data.shape[2] if data.shape[0] == 3 else data.shape[1]
                info = f"RGB  {w}×{h}  |  dtype: {data.dtype}"
            else:
                self._is_rgb = False
                if data.ndim == 2:
                    h, w = data.shape
                else:
                    h, w = data.shape[-2], data.shape[-1]
                info = f"Mono  {w}×{h}  |  dtype: {data.dtype}"

            self.grp_rgb.setVisible(self._is_rgb)

            # Linearity check
            flat = data if data.ndim == 2 else (data[0] if data.shape[0] == 3 else data[:,:,0])
            is_linear, frac = check_linearity(flat)
            if is_linear:
                self.lbl_linearity.setObjectName("ok")
                self.lbl_linearity.setText(
                    f"✓ Image appears linear — median at {frac*100:.1f}% of dynamic range")
            else:
                self.lbl_linearity.setObjectName("warn")
                self.lbl_linearity.setText(
                    f"⚠ Median at {frac*100:.0f}% of dynamic range. "
                    f"Image may already be stretched. Results may be unpredictable.")
            # Force style refresh
            self.lbl_linearity.style().unpolish(self.lbl_linearity)
            self.lbl_linearity.style().polish(self.lbl_linearity)

            self.lbl_info.setText(info)
            self._update_output_path()

            # Show original in preview
            self.canvas.show_original(data)
            self.tabs.setCurrentIndex(0)
            self._set_status(f"Loaded: {os.path.basename(path)}")
            self._log(f"Loaded: {path}")
            self._log(info)
            self._log(f"Linearity fraction: {frac:.3f} ({'linear' if is_linear else 'possibly stretched'})")

        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self._set_status("Load failed")

    # -------------------------------------------------------
    def _get_layer_configs(self) -> list:
        return [row.get_config() for row in self._layer_rows]

    def _get_rgb_mode(self) -> str:
        txt = self.rgb_cb.currentText()
        if "Luminance" in txt:
            return "luminance"
        elif "Per" in txt:
            return "per_channel"
        return "combined"

    def _start_stretch(self):
        fits_path = self.input_edit.text().strip()
        if not fits_path or not os.path.exists(fits_path):
            QMessageBox.warning(self, "No input", "Please load a FITS file first.")
            return
        out_path = self.output_edit.text().strip()
        if not out_path:
            QMessageBox.warning(self, "No output", "Please specify an output path.")
            return

        layer_configs = self._get_layer_configs()
        levels        = self.levels_spin.value()

        if len(layer_configs) != levels + 1:
            QMessageBox.critical(self, "Config error",
                                 f"Expected {levels+1} layer configs, got {len(layer_configs)}")
            return

        self._cancel_event.clear()
        self.btn_apply.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._worker = StretchWorker(
            fits_path=fits_path,
            layer_configs=layer_configs,
            levels=levels,
            rgb_mode=self._get_rgb_mode(),
            output_path=out_path,
            cancel_event=self._cancel_event,
            original_data=self._fits_data,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log)
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._set_status("Processing…")
        self.tabs.setCurrentIndex(1)  # show log

    def _cancel_stretch(self):
        if self._worker:
            self._cancel_event.set()
            self._log("Cancel requested…")
            self._set_status("Cancelling…")

    def _on_progress(self, current: int, total: int, desc: str):
        pct = int(current / total * 100)
        self.progress_bar.setValue(pct)
        self._set_status(desc)

    def _on_preview_ready(self, result):
        self.canvas.show_stretched(result)

    def _on_finished(self, result: dict):
        self.btn_apply.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._worker = None

        if result.get("success"):
            self._set_status(f"Done: {os.path.basename(result['output'])}")
            self._log(f"✓ Stretch complete: {result['output']}")
            self.tabs.setCurrentIndex(0)  # show preview
            if self.load_in_siril_cb.isChecked():
                try:
                    load_in_siril(result["output"])
                    self._log("Result loaded in Siril.")
                except Exception as e:
                    self._log(f"Siril load error: {e}")
            QMessageBox.information(self, "Done",
                                    f"Wavelet stretch complete!\n\n{result['output']}")
        else:
            err = result.get("error", "Unknown error")
            if err == "Cancelled":
                self._set_status("Cancelled")
                self._log("Processing cancelled.")
            else:
                self._set_status("Error")
                self._log(f"✗ Error: {err}")
                QMessageBox.critical(self, "Processing error", err)

    # -------------------------------------------------------
    def _log(self, msg: str):
        self.log_view.appendPlainText(msg)

    def _set_status(self, msg: str):
        self.status_lbl.setText(msg)

# ============================================================
# ENTRY POINT
# ============================================================

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
