r"""
mask_builder.py  —  Advanced Mask Builder for Siril  v1.0
Script #16 — Advanced Masking System

Place in: C:\Users\Marcell\Desktop\Siril New Scripts\
Run via:  Siril → Scripts menu → mask_builder
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

# ─── Standard imports ────────────────────────────────────────────────────────
import os, sys, json, uuid, threading, copy
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, laplace, zoom
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QComboBox,
    QSlider, QScrollArea, QSizePolicy, QSplitter, QFrame,
    QListWidget, QListWidgetItem, QStackedWidget, QDialog,
    QDialogButtonBox
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

# ─── Siril theme ─────────────────────────────────────────────────────────────
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
QListWidget {{
    background-color: {SIRIL_BG2};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
    alternate-background-color: {SIRIL_BG3};
}}
QListWidget::item {{
    padding: 5px 8px;
    border-bottom: 1px solid {SIRIL_BORDER};
}}
QListWidget::item:selected {{
    background: {SIRIL_ACCENT2};
    color: white;
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
QPushButton#small {{
    padding: 3px 8px;
    font-size: 9pt;
    text-align: center;
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

# ─── Constants ────────────────────────────────────────────────────────────────
MASK_TYPES = {
    "range":      "Range (luminance)",
    "star":       "Star Mask",
    "edge":       "Edge / Structure",
    "nebulosity": "Adaptive Nebulosity",
    "color":      "Color / Hue",
    "gradient":   "Gradient",
}

MASK_ICONS = {
    "range":      "▭",
    "star":       "★",
    "edge":       "⊞",
    "nebulosity": "〰",
    "color":      "◑",
    "gradient":   "▽",
}

OPERATORS = ["AND", "OR", "ADD", "SUBTRACT", "DIFFERENCE"]

# ─── Mask math ────────────────────────────────────────────────────────────────

def to_luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        arr = data.astype(np.float64)
    elif data.ndim == 3 and data.shape[0] == 3:
        arr = (0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]).astype(np.float64)
    elif data.ndim == 3 and data.shape[2] == 3:
        arr = (0.299 * data[:,:,0] + 0.587 * data[:,:,1] + 0.114 * data[:,:,2]).astype(np.float64)
    else:
        arr = data[0].astype(np.float64) if data.ndim == 3 else data.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)


def make_range_mask(lum, lo=0.1, hi=0.8, feather_px=20.0):
    mask = np.zeros_like(lum)
    mask[(lum >= lo) & (lum <= hi)] = 1.0
    if feather_px > 0:
        mask = gaussian_filter(mask.astype(float), sigma=feather_px)
    return np.clip(mask, 0.0, 1.0)


def make_star_mask(lum, fwhm_guess=5.0, threshold_sigma=5.0,
                   growth_factor=2.0, feather_px=5.0, max_stars=500):
    mean, median, std = sigma_clipped_stats(lum, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm_guess,
                            threshold=threshold_sigma * std,
                            sharplo=0.2, sharphi=1.0)
    sources = daofind(lum - median)
    mask = np.zeros_like(lum)
    if sources is None or len(sources) == 0:
        return mask
    h, w = lum.shape
    if len(sources) > max_stars:
        sources.sort("peak")
        sources.reverse()
        sources = sources[:max_stars]
    y_grid, x_grid = np.ogrid[:h, :w]
    for src in sources:
        x0 = float(src["xcentroid"])
        y0 = float(src["ycentroid"])
        peak = float(src["peak"])
        radius = fwhm_guess * growth_factor * (1.0 + 0.5 * np.log1p(peak / (median + 1e-10)))
        radius = max(fwhm_guess, min(radius, fwhm_guess * growth_factor * 5))
        dist_sq = (x_grid - x0)**2 + (y_grid - y0)**2
        mask[dist_sq <= radius**2] = 1.0
    if feather_px > 0:
        mask = gaussian_filter(mask, sigma=feather_px)
    return np.clip(mask, 0.0, 1.0)


def make_edge_mask(lum, method="sobel", pre_blur=1.0, post_blur=3.0, normalize=True):
    smoothed = gaussian_filter(lum, sigma=pre_blur) if pre_blur > 0 else lum
    if method == "laplacian":
        edges = np.abs(laplace(smoothed))
    else:
        gx = sobel(smoothed, axis=1)
        gy = sobel(smoothed, axis=0)
        edges = np.sqrt(gx**2 + gy**2)
    if normalize:
        edges = edges / (np.percentile(edges, 99) + 1e-10)
        edges = np.clip(edges, 0, 1)
    if post_blur > 0:
        edges = gaussian_filter(edges, sigma=post_blur)
    return np.clip(edges, 0.0, 1.0)


def make_nebulosity_mask(lum, tile_size=64, sigma_above=2.0, feather_px=10.0):
    h, w = lum.shape
    th = max(1, h // tile_size)
    tw = max(1, w // tile_size)
    bg_map  = np.zeros((th, tw))
    std_map = np.zeros((th, tw))
    for i in range(th):
        for j in range(tw):
            y1 = i * tile_size; y2 = min(h, y1 + tile_size)
            x1 = j * tile_size; x2 = min(w, x1 + tile_size)
            tile = lum[y1:y2, x1:x2]
            med = np.median(tile)
            bg_map[i, j]  = med
            std_map[i, j] = np.median(np.abs(tile - med)) * 1.4826
    zoom_y = h / th; zoom_x = w / tw
    bg_full  = zoom(bg_map,  (zoom_y, zoom_x), order=1)[:h, :w]
    std_full = zoom(std_map, (zoom_y, zoom_x), order=1)[:h, :w]
    threshold = bg_full + sigma_above * std_full
    mask = np.clip((lum - threshold) / (std_full + 1e-10), 0, 1)
    if feather_px > 0:
        mask = gaussian_filter(mask, sigma=feather_px)
    return np.clip(mask, 0.0, 1.0)


def make_color_mask(data, hue_center=300.0, hue_range=30.0,
                    saturation_min=0.15, feather_px=5.0):
    if data.ndim == 2:
        return np.zeros_like(data)
    if data.shape[0] == 3:
        R = data[0].astype(float); G = data[1].astype(float); B = data[2].astype(float)
    else:
        R = data[:,:,0].astype(float); G = data[:,:,1].astype(float); B = data[:,:,2].astype(float)
    for ch in [R, G, B]:
        mn, mx = ch.min(), ch.max()
        if mx > mn: ch[:] = (ch - mn) / (mx - mn)
    cmax = np.maximum(np.maximum(R, G), B)
    cmin = np.minimum(np.minimum(R, G), B)
    delta = cmax - cmin
    hue = np.zeros_like(R)
    mask_r = (delta > 0) & (cmax == R)
    mask_g = (delta > 0) & (cmax == G)
    mask_b = (delta > 0) & (cmax == B)
    hue[mask_r] = (60.0 * ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = (60.0 * ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = (60.0 * ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4)
    hue = hue % 360.0
    sat = np.where(cmax > 0, delta / cmax, 0.0)
    lo = (hue_center - hue_range) % 360.0
    hi = (hue_center + hue_range) % 360.0
    if lo < hi:
        in_hue = (hue >= lo) & (hue <= hi)
    else:
        in_hue = (hue >= lo) | (hue <= hi)
    result = np.where(in_hue & (sat >= saturation_min), 1.0, 0.0)
    if feather_px > 0:
        result = gaussian_filter(result.astype(float), sigma=feather_px)
    return np.clip(result, 0.0, 1.0)


def make_gradient_mask(shape, mode="radial", cx=0.5, cy=0.5,
                        radius=0.5, angle=0.0, invert=False):
    h, w = shape
    y_idx = np.linspace(0, 1, h)
    x_idx = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x_idx, y_idx)
    if mode == "radial":
        dist = np.sqrt(((xx - cx) * w/h)**2 + (yy - cy)**2)
        mask = np.clip(1.0 - dist / (radius + 1e-10), 0.0, 1.0)
    else:
        rad  = np.radians(angle)
        proj = (xx - cx) * np.cos(rad) + (yy - cy) * np.sin(rad)
        mn, mx = proj.min(), proj.max()
        mask = (proj - mn) / (mx - mn + 1e-10)
    if invert:
        mask = 1.0 - mask
    return np.clip(mask, 0.0, 1.0)


def combine_masks(mask_a, mask_b, operator="AND", opacity_a=1.0, opacity_b=1.0):
    a = np.clip(mask_a * opacity_a, 0, 1)
    b = np.clip(mask_b * opacity_b, 0, 1)
    if operator == "AND":
        result = a * b
    elif operator == "OR":
        result = 1.0 - (1.0 - a) * (1.0 - b)
    elif operator == "ADD":
        result = np.clip(a + b, 0.0, 1.0)
    elif operator == "SUBTRACT":
        result = np.clip(a - b, 0.0, 1.0)
    elif operator == "DIFFERENCE":
        result = np.abs(a - b)
    else:
        result = a
    return np.clip(result, 0.0, 1.0)


def _compute_single_layer(layer, lum, data):
    t = layer["type"]
    p = layer.get("params", {})
    if t == "range":
        return make_range_mask(lum, lo=p.get("lo", 0.1), hi=p.get("hi", 0.7),
                                feather_px=p.get("feather", 20.0))
    elif t == "star":
        return make_star_mask(lum, fwhm_guess=p.get("fwhm_guess", 5.0),
                               threshold_sigma=p.get("threshold_sigma", 5.0),
                               growth_factor=p.get("growth_factor", 2.0),
                               feather_px=p.get("feather", 5.0),
                               max_stars=p.get("max_stars", 500))
    elif t == "edge":
        return make_edge_mask(lum, method=p.get("method", "sobel"),
                               pre_blur=p.get("pre_blur", 1.0),
                               post_blur=p.get("post_blur", 3.0))
    elif t == "nebulosity":
        return make_nebulosity_mask(lum, tile_size=p.get("tile_size", 64),
                                     sigma_above=p.get("sigma_above", 2.0),
                                     feather_px=p.get("feather", 10.0))
    elif t == "color":
        return make_color_mask(data, hue_center=p.get("hue_center", 300.0),
                                hue_range=p.get("hue_range", 30.0),
                                saturation_min=p.get("saturation_min", 0.15),
                                feather_px=p.get("feather", 5.0))
    elif t == "gradient":
        h, w = lum.shape
        return make_gradient_mask((h, w), mode=p.get("mode", "radial"),
                                   cx=p.get("cx", 0.5), cy=p.get("cy", 0.5),
                                   radius=p.get("radius", 0.5),
                                   angle=p.get("angle", 0.0),
                                   invert=p.get("invert", False))
    return np.ones_like(lum)


def compute_final_mask(layers, data):
    lum = to_luminance(data)
    h, w = lum.shape
    result = None
    first = True
    for layer in layers:
        if not layer.get("enabled", True):
            continue
        if layer.get("cached_mask") is None:
            layer["cached_mask"] = _compute_single_layer(layer, lum, data)
        mask = layer["cached_mask"].copy()
        if layer.get("invert", False):
            mask = 1.0 - mask
        mask = mask * layer.get("opacity", 1.0)
        if first:
            result = mask
            first = False
        else:
            op = layer.get("operator", "AND")
            result = combine_masks(result, mask, op)
    if result is None:
        result = np.ones((h, w), dtype=np.float64)
    return np.clip(result, 0.0, 1.0)


# ─── Presets ──────────────────────────────────────────────────────────────────

def _new_layer_id():
    return uuid.uuid4().hex[:8]

def _make_layer(name, mtype, operator, opacity, invert, params):
    return {
        "id": _new_layer_id(), "name": name, "type": mtype,
        "enabled": True, "opacity": opacity, "operator": operator,
        "invert": invert, "params": params, "cached_mask": None
    }

ALL_PRESETS = [
    {
        "name": "Nebula + Star protection",
        "description": "Aggressive nebula selection, stars excluded",
        "layers": [
            _make_layer("Nebulosity", "nebulosity", "AND", 1.0, False,
                        {"tile_size": 64, "sigma_above": 2.0, "feather": 15.0}),
            _make_layer("Exclude stars", "star", "SUBTRACT", 1.0, False,
                        {"fwhm_guess": 5.0, "threshold_sigma": 5.0,
                         "growth_factor": 1.5, "feather": 5.0}),
        ]
    },
    {
        "name": "Star halos",
        "description": "Target only the halos around bright stars",
        "layers": [
            _make_layer("Large star mask", "star", "AND", 1.0, False,
                        {"growth_factor": 3.5, "feather": 10.0}),
            _make_layer("Exclude star cores", "star", "SUBTRACT", 1.0, False,
                        {"growth_factor": 1.2, "feather": 3.0}),
        ]
    },
    {
        "name": "Structure sharpening",
        "description": "Target only edges and fine structure",
        "layers": [
            _make_layer("Edge mask", "edge", "AND", 1.0, False,
                        {"method": "sobel", "pre_blur": 1.0, "post_blur": 4.0}),
        ]
    },
    {
        "name": "Midtones only",
        "description": "Protect shadows and highlights, target midtones",
        "layers": [
            _make_layer("Midtone range", "range", "AND", 1.0, False,
                        {"lo": 0.15, "hi": 0.65, "feather": 30.0}),
        ]
    },
    {
        "name": "Magenta star correction",
        "description": "Select only magenta-colored stars for hue correction",
        "layers": [
            _make_layer("Magenta hue", "color", "AND", 1.0, False,
                        {"hue_center": 300.0, "hue_range": 35.0,
                         "saturation_min": 0.12, "feather": 4.0}),
            _make_layer("Stars only", "star", "AND", 1.0, False,
                        {"growth_factor": 1.8, "feather": 5.0}),
        ]
    },
]


# ─── Worker thread ────────────────────────────────────────────────────────────

class MaskWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    preview_ready = pyqtSignal(object, object)
    finished      = pyqtSignal(dict)

    def __init__(self, layers, data, output_path, cancel_event,
                 single_layer_idx=None):
        super().__init__()
        self.layers            = layers
        self.data              = data
        self.output_path       = output_path
        self._cancel           = cancel_event
        self.single_layer_idx  = single_layer_idx  # preview one layer only

    def run(self):
        try:
            lum = to_luminance(self.data)

            if self.single_layer_idx is not None:
                # Preview a single layer
                idx = self.single_layer_idx
                if 0 <= idx < len(self.layers):
                    layer = self.layers[idx]
                    self.log_line.emit(f"Computing preview: {layer['name']}")
                    layer["cached_mask"] = _compute_single_layer(layer, lum, self.data)
                    mask = layer["cached_mask"].copy()
                    if layer.get("invert", False):
                        mask = 1.0 - mask
                    overlay = self._build_overlay(lum, mask)
                    self.preview_ready.emit(mask, overlay)
                    self.finished.emit({"success": True, "output": None,
                                        "mask_mean": float(np.mean(mask)),
                                        "mask_coverage": float(np.mean(mask > 0.5))})
                return

            # Full computation
            active = [l for l in self.layers if l.get("enabled", True)]
            total  = len(active)
            done   = 0

            for layer in self.layers:
                if self._cancel.is_set():
                    self.finished.emit({"success": False, "error": "Cancelled"})
                    return
                if not layer.get("enabled", True):
                    continue
                self.log_line.emit(f"Computing: {layer['name']}")
                self.progress.emit(done, total, layer["name"])
                layer["cached_mask"] = _compute_single_layer(layer, lum, self.data)
                done += 1

            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"})
                return

            self.log_line.emit("Combining layers...")
            final_mask = compute_final_mask(self.layers, self.data)
            overlay    = self._build_overlay(lum, final_mask)
            self.preview_ready.emit(final_mask, overlay)

            if self.output_path:
                self._save_mask(final_mask)
                self.log_line.emit(f"Saved: {self.output_path}")

            self.finished.emit({
                "success":       True,
                "output":        self.output_path,
                "mask_mean":     float(np.mean(final_mask)),
                "mask_coverage": float(np.mean(final_mask > 0.5)),
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _build_overlay(self, lum, mask):
        p_lo = np.percentile(lum, 0.5)
        p_hi = np.percentile(lum, 99.5)
        disp = np.clip((lum - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)
        R = disp * (1.0 - mask * 0.5) + mask * 0.8
        G = disp * (1.0 - mask * 0.7)
        B = disp * (1.0 - mask * 0.7)
        return np.stack([np.clip(R, 0, 1), np.clip(G, 0, 1), np.clip(B, 0, 1)], axis=2)

    def _save_mask(self, mask):
        from astropy.io import fits as _fits
        hdu = _fits.PrimaryHDU(data=mask.astype(np.float32))
        hdu.header["HISTORY"]  = "Mask created by mask_builder.py"
        hdu.header["MASKTYPE"] = "COMBINED"
        hdu.writeto(self.output_path, overwrite=True)

    def cancel(self):
        self._cancel.set()


# ─── Preview canvas ───────────────────────────────────────────────────────────

class MaskPreviewCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 4), facecolor="#1e2128")
        self.ax_orig    = self.fig.add_subplot(1, 3, 1)
        self.ax_mask    = self.fig.add_subplot(1, 3, 2)
        self.ax_overlay = self.fig.add_subplot(1, 3, 3)
        self._style_all()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _style_all(self):
        titles = ["Original", "Mask", "Overlay (red = affected)"]
        for ax, title in zip([self.ax_orig, self.ax_mask, self.ax_overlay], titles):
            ax.set_facecolor("#1e2128")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")
            ax.set_title(title, color="#7a8499", fontsize=8)

    def _downsample(self, arr, max_px=600):
        if arr.ndim == 2:
            h, w = arr.shape
        else:
            h, w = arr.shape[:2]
        factor = max(1, max(h, w) // max_px)
        if factor == 1:
            return arr
        if arr.ndim == 2:
            return arr[::factor, ::factor]
        return arr[::factor, ::factor, :]

    def show_original(self, data):
        lum  = to_luminance(data)
        p_lo = np.percentile(lum, 0.5)
        p_hi = np.percentile(lum, 99.5)
        disp = np.clip((lum - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)
        self.ax_orig.clear()
        self.ax_orig.imshow(self._downsample(disp), cmap="gray",
                            origin="lower", aspect="equal", interpolation="nearest")
        self.ax_orig.set_title("Original", color="#7a8499", fontsize=8)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_mask_and_overlay(self, mask, overlay):
        self.ax_mask.clear()
        self.ax_overlay.clear()
        self.ax_mask.imshow(self._downsample(mask), cmap="gray",
                            origin="lower", vmin=0, vmax=1,
                            aspect="equal", interpolation="nearest")
        self.ax_mask.set_title("Mask", color="#7a8499", fontsize=8)
        self.ax_overlay.imshow(self._downsample(overlay),
                               origin="lower", aspect="equal", interpolation="nearest")
        self.ax_overlay.set_title("Overlay (red = affected)", color="#7a8499", fontsize=8)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def clear_all(self):
        for ax in [self.ax_orig, self.ax_mask, self.ax_overlay]:
            ax.clear()
            ax.set_facecolor("#1e2128")
        self._style_all()
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ─── Parameter panels ─────────────────────────────────────────────────────────

class RangePanel(QWidget):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setSpacing(8)

        # Lo slider
        lo_row = QHBoxLayout()
        self.lo_slider = QSlider(Qt.Orientation.Horizontal)
        self.lo_slider.setRange(0, 100); self.lo_slider.setValue(10)
        self.lo_label  = QLabel("0.10")
        lo_row.addWidget(self.lo_slider); lo_row.addWidget(self.lo_label)
        form.addRow("Lo threshold:", lo_row)

        # Hi slider
        hi_row = QHBoxLayout()
        self.hi_slider = QSlider(Qt.Orientation.Horizontal)
        self.hi_slider.setRange(0, 100); self.hi_slider.setValue(70)
        self.hi_label  = QLabel("0.70")
        hi_row.addWidget(self.hi_slider); hi_row.addWidget(self.hi_label)
        form.addRow("Hi threshold:", hi_row)

        self.feather = QSpinBox(); self.feather.setRange(0, 100); self.feather.setValue(20)
        form.addRow("Feather (px):", self.feather)

        self.lo_slider.valueChanged.connect(lambda v: (self.lo_label.setText(f"{v/100:.2f}"), self.params_changed.emit()))
        self.hi_slider.valueChanged.connect(lambda v: (self.hi_label.setText(f"{v/100:.2f}"), self.params_changed.emit()))
        self.feather.valueChanged.connect(lambda _: self.params_changed.emit())

    def get_params(self):
        return {"lo": self.lo_slider.value() / 100,
                "hi": self.hi_slider.value() / 100,
                "feather": float(self.feather.value())}

    def set_params(self, p):
        self.lo_slider.setValue(int(p.get("lo", 0.1) * 100))
        self.hi_slider.setValue(int(p.get("hi", 0.7) * 100))
        self.feather.setValue(int(p.get("feather", 20)))


class StarPanel(QWidget):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setSpacing(8)

        self.threshold = QDoubleSpinBox(); self.threshold.setRange(1.0, 20.0)
        self.threshold.setValue(5.0); self.threshold.setSingleStep(0.5)
        form.addRow("Detection σ:", self.threshold)

        self.fwhm = QDoubleSpinBox(); self.fwhm.setRange(2.0, 20.0)
        self.fwhm.setValue(5.0); self.fwhm.setSingleStep(0.5)
        form.addRow("FWHM guess (px):", self.fwhm)

        self.growth = QDoubleSpinBox(); self.growth.setRange(1.0, 5.0)
        self.growth.setValue(2.0); self.growth.setSingleStep(0.1)
        form.addRow("Growth factor:", self.growth)
        hint = QLabel("1.0=tight star  3.0=include halos"); hint.setObjectName("dim")
        form.addRow("", hint)

        self.feather = QSpinBox(); self.feather.setRange(0, 30); self.feather.setValue(5)
        form.addRow("Feather (px):", self.feather)

        self.max_stars = QSpinBox(); self.max_stars.setRange(50, 2000)
        self.max_stars.setValue(500); self.max_stars.setSingleStep(50)
        form.addRow("Max stars:", self.max_stars)

        for w in [self.threshold, self.fwhm, self.growth, self.feather, self.max_stars]:
            w.valueChanged.connect(lambda _: self.params_changed.emit())

    def get_params(self):
        return {"fwhm_guess": self.fwhm.value(),
                "threshold_sigma": self.threshold.value(),
                "growth_factor": self.growth.value(),
                "feather": float(self.feather.value()),
                "max_stars": self.max_stars.value()}

    def set_params(self, p):
        self.threshold.setValue(p.get("threshold_sigma", 5.0))
        self.fwhm.setValue(p.get("fwhm_guess", 5.0))
        self.growth.setValue(p.get("growth_factor", 2.0))
        self.feather.setValue(int(p.get("feather", 5)))
        self.max_stars.setValue(p.get("max_stars", 500))


class EdgePanel(QWidget):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setSpacing(8)

        self.method = QComboBox()
        self.method.addItems(["Sobel (directional)", "Laplacian (isotropic)"])
        form.addRow("Method:", self.method)

        self.pre_blur = QDoubleSpinBox(); self.pre_blur.setRange(0.0, 5.0)
        self.pre_blur.setValue(1.0); self.pre_blur.setSingleStep(0.5)
        form.addRow("Pre-blur:", self.pre_blur)
        h1 = QLabel("Smoothing before edge detection"); h1.setObjectName("dim")
        form.addRow("", h1)

        self.post_blur = QDoubleSpinBox(); self.post_blur.setRange(0.0, 20.0)
        self.post_blur.setValue(3.0); self.post_blur.setSingleStep(0.5)
        form.addRow("Post-blur:", self.post_blur)
        h2 = QLabel("Widens edge selection area"); h2.setObjectName("dim")
        form.addRow("", h2)

        self.method.currentIndexChanged.connect(lambda _: self.params_changed.emit())
        self.pre_blur.valueChanged.connect(lambda _: self.params_changed.emit())
        self.post_blur.valueChanged.connect(lambda _: self.params_changed.emit())

    def get_params(self):
        method = "sobel" if self.method.currentIndex() == 0 else "laplacian"
        return {"method": method, "pre_blur": self.pre_blur.value(),
                "post_blur": self.post_blur.value()}

    def set_params(self, p):
        m = p.get("method", "sobel")
        self.method.setCurrentIndex(0 if m == "sobel" else 1)
        self.pre_blur.setValue(p.get("pre_blur", 1.0))
        self.post_blur.setValue(p.get("post_blur", 3.0))


class NebulosityPanel(QWidget):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setSpacing(8)

        self.tile_size = QComboBox()
        self.tile_size.addItems(["32", "64", "128", "256"])
        self.tile_size.setCurrentIndex(1)
        form.addRow("Tile size (px):", self.tile_size)
        h1 = QLabel("Smaller = more local, more sensitive"); h1.setObjectName("dim")
        form.addRow("", h1)

        self.sigma = QDoubleSpinBox(); self.sigma.setRange(0.5, 10.0)
        self.sigma.setValue(2.0); self.sigma.setSingleStep(0.5)
        form.addRow("Sigma above bg:", self.sigma)
        h2 = QLabel("Sigma above local background"); h2.setObjectName("dim")
        form.addRow("", h2)

        self.feather = QSpinBox(); self.feather.setRange(0, 50); self.feather.setValue(10)
        form.addRow("Feather (px):", self.feather)

        self.tile_size.currentIndexChanged.connect(lambda _: self.params_changed.emit())
        self.sigma.valueChanged.connect(lambda _: self.params_changed.emit())
        self.feather.valueChanged.connect(lambda _: self.params_changed.emit())

    def get_params(self):
        return {"tile_size": int(self.tile_size.currentText()),
                "sigma_above": self.sigma.value(),
                "feather": float(self.feather.value())}

    def set_params(self, p):
        ts = str(p.get("tile_size", 64))
        idx = self.tile_size.findText(ts)
        if idx >= 0: self.tile_size.setCurrentIndex(idx)
        self.sigma.setValue(p.get("sigma_above", 2.0))
        self.feather.setValue(int(p.get("feather", 10)))


class ColorPanel(QWidget):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setSpacing(8)

        # Hue center
        hue_row = QHBoxLayout()
        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(0, 360); self.hue_slider.setValue(300)
        self.hue_label  = QLabel("300°")
        self.hue_swatch = QLabel("  ")
        self.hue_swatch.setMinimumWidth(30)
        hue_row.addWidget(self.hue_slider)
        hue_row.addWidget(self.hue_label)
        hue_row.addWidget(self.hue_swatch)
        form.addRow("Hue center:", hue_row)

        # Hue range
        range_row = QHBoxLayout()
        self.range_slider = QSlider(Qt.Orientation.Horizontal)
        self.range_slider.setRange(5, 90); self.range_slider.setValue(30)
        self.range_label  = QLabel("±30°")
        range_row.addWidget(self.range_slider); range_row.addWidget(self.range_label)
        form.addRow("Hue range ±:", range_row)

        self.sat_min = QDoubleSpinBox(); self.sat_min.setRange(0.0, 1.0)
        self.sat_min.setValue(0.15); self.sat_min.setSingleStep(0.05)
        form.addRow("Min saturation:", self.sat_min)

        self.feather = QSpinBox(); self.feather.setRange(0, 20); self.feather.setValue(5)
        form.addRow("Feather (px):", self.feather)

        hint = QLabel("Requires color (RGB) input image"); hint.setObjectName("warn")
        form.addRow("", hint)

        # Presets
        preset_row = QHBoxLayout()
        for name, hue in [("Magenta", 300), ("Green", 120), ("Ha Red", 0)]:
            btn = QPushButton(name); btn.setObjectName("small")
            btn.clicked.connect(lambda checked, h=hue: self._set_hue(h))
            preset_row.addWidget(btn)
        form.addRow("Presets:", preset_row)

        self.hue_slider.valueChanged.connect(self._on_hue_changed)
        self.range_slider.valueChanged.connect(lambda v: (self.range_label.setText(f"±{v}°"), self.params_changed.emit()))
        self.sat_min.valueChanged.connect(lambda _: self.params_changed.emit())
        self.feather.valueChanged.connect(lambda _: self.params_changed.emit())
        self._update_swatch(300)

    def _set_hue(self, hue):
        self.hue_slider.setValue(hue)

    def _on_hue_changed(self, v):
        self.hue_label.setText(f"{v}°")
        self._update_swatch(v)
        self.params_changed.emit()

    def _update_swatch(self, hue):
        self.hue_swatch.setStyleSheet(
            f"background-color: hsl({hue}, 80%, 50%); border: 1px solid #3a4055; border-radius: 2px;")

    def get_params(self):
        return {"hue_center": float(self.hue_slider.value()),
                "hue_range": float(self.range_slider.value()),
                "saturation_min": self.sat_min.value(),
                "feather": float(self.feather.value())}

    def set_params(self, p):
        self.hue_slider.setValue(int(p.get("hue_center", 300)))
        self.range_slider.setValue(int(p.get("hue_range", 30)))
        self.sat_min.setValue(p.get("saturation_min", 0.15))
        self.feather.setValue(int(p.get("feather", 5)))


class GradientPanel(QWidget):
    params_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setSpacing(8)

        self.mode = QComboBox()
        self.mode.addItems(["Radial (from center)", "Linear (directional)"])
        form.addRow("Mode:", self.mode)

        self.cx = QDoubleSpinBox(); self.cx.setRange(0.0, 1.0)
        self.cx.setValue(0.5); self.cx.setSingleStep(0.05)
        form.addRow("Center X:", self.cx)

        self.cy = QDoubleSpinBox(); self.cy.setRange(0.0, 1.0)
        self.cy.setValue(0.5); self.cy.setSingleStep(0.05)
        form.addRow("Center Y:", self.cy)

        self.radius = QDoubleSpinBox(); self.radius.setRange(0.1, 2.0)
        self.radius.setValue(0.5); self.radius.setSingleStep(0.05)
        form.addRow("Radius:", self.radius)

        self.angle = QSpinBox(); self.angle.setRange(0, 360); self.angle.setValue(0)
        form.addRow("Angle (°):", self.angle)

        self.invert_chk = QCheckBox("Invert gradient")
        form.addRow("", self.invert_chk)

        self.mode.currentIndexChanged.connect(self._on_mode_change)
        for w in [self.cx, self.cy, self.radius, self.angle]:
            w.valueChanged.connect(lambda _: self.params_changed.emit())
        self.invert_chk.stateChanged.connect(lambda _: self.params_changed.emit())
        self._on_mode_change(0)

    def _on_mode_change(self, idx):
        radial = (idx == 0)
        self.cx.setVisible(radial); self.cy.setVisible(radial)
        self.radius.setVisible(radial); self.angle.setVisible(not radial)
        self.params_changed.emit()

    def get_params(self):
        mode = "radial" if self.mode.currentIndex() == 0 else "linear"
        return {"mode": mode, "cx": self.cx.value(), "cy": self.cy.value(),
                "radius": self.radius.value(), "angle": float(self.angle.value()),
                "invert": self.invert_chk.isChecked()}

    def set_params(self, p):
        m = p.get("mode", "radial")
        self.mode.setCurrentIndex(0 if m == "radial" else 1)
        self.cx.setValue(p.get("cx", 0.5)); self.cy.setValue(p.get("cy", 0.5))
        self.radius.setValue(p.get("radius", 0.5))
        self.angle.setValue(int(p.get("angle", 0)))
        self.invert_chk.setChecked(p.get("invert", False))


# ─── Presets dialog ───────────────────────────────────────────────────────────

class PresetsDialog(QDialog):
    preset_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Preset")
        self.setMinimumSize(420, 340)
        layout = QVBoxLayout(self)

        lbl = QLabel("Select a preset mask combination:")
        lbl.setObjectName("section")
        layout.addWidget(lbl)

        self.list_w = QListWidget()
        for p in ALL_PRESETS:
            item = QListWidgetItem(p["name"])
            item.setToolTip(p["description"])
            self.list_w.addItem(item)
        layout.addWidget(self.list_w)

        self.desc_lbl = QLabel("")
        self.desc_lbl.setObjectName("dim")
        self.desc_lbl.setWordWrap(True)
        layout.addWidget(self.desc_lbl)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.list_w.currentRowChanged.connect(self._on_row_changed)
        self.list_w.setCurrentRow(0)

    def _on_row_changed(self, row):
        if 0 <= row < len(ALL_PRESETS):
            self.desc_lbl.setText(ALL_PRESETS[row]["description"])

    def _on_accept(self):
        row = self.list_w.currentRow()
        if 0 <= row < len(ALL_PRESETS):
            # Deep copy so preset originals are preserved
            preset = copy.deepcopy(ALL_PRESETS[row])
            # Re-assign fresh IDs
            for layer in preset["layers"]:
                layer["id"] = _new_layer_id()
                layer["cached_mask"] = None
            self.preset_selected.emit(preset)
        self.accept()


# ─── Main window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⬛  Advanced Mask Builder  —  Siril  v1.0")
        self.setMinimumSize(1200, 700)

        self.image_data   = None   # loaded FITS data
        self.is_color     = False
        self.layers       = []     # list of layer dicts
        self.worker       = None
        self._cancel_evt  = threading.Event()
        self._suppress_signals = False

        self._build_ui()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root    = QVBoxLayout(central); root.setContentsMargins(6, 6, 6, 4)

        # Splitter: left panel | right preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        root.addWidget(splitter, 1)

        # ── Left panel ──────────────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(430)

        left_container = QWidget()
        left_layout    = QVBoxLayout(left_container)
        left_layout.setSpacing(8)
        left_scroll.setWidget(left_container)

        # Source image
        src_grp = QGroupBox("Source image")
        src_lay = QVBoxLayout(src_grp)
        src_row = QHBoxLayout()
        self.src_path = QLineEdit(); self.src_path.setPlaceholderText("Path to FITS file...")
        self.src_browse = QPushButton("Browse...")
        src_row.addWidget(self.src_path); src_row.addWidget(self.src_browse)
        src_lay.addLayout(src_row)
        self.load_btn = QPushButton("⬇  Load image"); self.load_btn.setObjectName("primary")
        src_lay.addWidget(self.load_btn)
        self.img_info = QLabel("No image loaded"); self.img_info.setObjectName("dim")
        src_lay.addWidget(self.img_info)
        left_layout.addWidget(src_grp)

        # Layer stack
        layers_grp = QGroupBox("Mask layers")
        layers_lay = QVBoxLayout(layers_grp)
        self.layer_list = QListWidget()
        self.layer_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.layer_list.setAlternatingRowColors(True)
        self.layer_list.setMinimumHeight(120)
        layers_lay.addWidget(self.layer_list)

        btn_row = QHBoxLayout()
        self.add_btn  = QPushButton("+ Add"); self.add_btn.setObjectName("small")
        self.up_btn   = QPushButton("↑ Up");  self.up_btn.setObjectName("small")
        self.dn_btn   = QPushButton("↓ Down");self.dn_btn.setObjectName("small")
        self.del_btn  = QPushButton("✕ Del"); self.del_btn.setObjectName("small")
        self.pre_btn  = QPushButton("⚙ Presets"); self.pre_btn.setObjectName("small")
        for b in [self.add_btn, self.up_btn, self.dn_btn, self.del_btn, self.pre_btn]:
            btn_row.addWidget(b)
        layers_lay.addLayout(btn_row)
        left_layout.addWidget(layers_grp)

        # Layer properties
        props_grp = QGroupBox("Layer properties")
        props_lay = QVBoxLayout(props_grp)
        form      = QFormLayout(); form.setSpacing(6)

        self.lyr_name = QLineEdit()
        form.addRow("Name:", self.lyr_name)

        self.lyr_type = QComboBox()
        for key, label in MASK_TYPES.items():
            self.lyr_type.addItem(label, key)
        form.addRow("Type:", self.lyr_type)

        self.lyr_op = QComboBox(); self.lyr_op.addItems(OPERATORS)
        form.addRow("Operator:", self.lyr_op)
        op_hint = QLabel("How to combine with layer below"); op_hint.setObjectName("dim")
        form.addRow("", op_hint)

        op_row = QHBoxLayout()
        self.lyr_opacity = QSlider(Qt.Orientation.Horizontal)
        self.lyr_opacity.setRange(0, 100); self.lyr_opacity.setValue(100)
        self.lyr_op_label = QLabel("100%")
        op_row.addWidget(self.lyr_opacity); op_row.addWidget(self.lyr_op_label)
        form.addRow("Opacity:", op_row)

        self.lyr_invert = QCheckBox("Invert this layer")
        form.addRow("", self.lyr_invert)

        props_lay.addLayout(form)

        sep = QFrame(); sep.setObjectName("separator"); sep.setFrameShape(QFrame.Shape.HLine)
        props_lay.addWidget(sep)

        # Stacked param panels
        self.param_stack = QStackedWidget()
        self.panel_range  = RangePanel();       self.param_stack.addWidget(self.panel_range)
        self.panel_star   = StarPanel();         self.param_stack.addWidget(self.panel_star)
        self.panel_edge   = EdgePanel();         self.param_stack.addWidget(self.panel_edge)
        self.panel_nebul  = NebulosityPanel();   self.param_stack.addWidget(self.panel_nebul)
        self.panel_color  = ColorPanel();        self.param_stack.addWidget(self.panel_color)
        self.panel_grad   = GradientPanel();     self.param_stack.addWidget(self.panel_grad)
        props_lay.addWidget(self.param_stack)

        self._panels = {
            "range": (0, self.panel_range),
            "star":  (1, self.panel_star),
            "edge":  (2, self.panel_edge),
            "nebulosity": (3, self.panel_nebul),
            "color": (4, self.panel_color),
            "gradient": (5, self.panel_grad),
        }

        # Preview this layer button
        self.preview_layer_btn = QPushButton("👁  Preview this layer")
        props_lay.addWidget(self.preview_layer_btn)

        left_layout.addWidget(props_grp)

        # Output
        out_grp = QGroupBox("Output")
        out_lay = QVBoxLayout(out_grp)
        out_row = QHBoxLayout()
        self.out_path = QLineEdit(); self.out_path.setPlaceholderText("output_mask.fit")
        self.out_browse = QPushButton("Browse...")
        out_row.addWidget(self.out_path); out_row.addWidget(self.out_browse)
        out_lay.addLayout(out_row)

        self.compute_btn = QPushButton("▶  Compute & Export mask")
        self.compute_btn.setObjectName("primary")
        out_lay.addWidget(self.compute_btn)

        self.load_siril_btn = QPushButton("📋  Load mask in Siril")
        self.load_siril_btn.setObjectName("export")
        out_lay.addWidget(self.load_siril_btn)

        self.copy_cmd_btn = QPushButton("⎘  Copy Siril command")
        self.copy_cmd_btn.setObjectName("small")
        out_lay.addWidget(self.copy_cmd_btn)

        left_layout.addWidget(out_grp)

        # Save/load layer stack
        stack_row = QHBoxLayout()
        self.save_stack_btn = QPushButton("💾 Save layer stack"); self.save_stack_btn.setObjectName("small")
        self.load_stack_btn = QPushButton("📂 Load layer stack"); self.load_stack_btn.setObjectName("small")
        stack_row.addWidget(self.save_stack_btn); stack_row.addWidget(self.load_stack_btn)
        left_layout.addLayout(stack_row)

        left_layout.addStretch()
        splitter.addWidget(left_scroll)

        # ── Right panel: preview ─────────────────────────────────────────────
        right_w  = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self.canvas = MaskPreviewCanvas(right_w)
        right_lay.addWidget(self.canvas, 1)

        hint_lbl = QLabel("White = fully affected  |  Black = fully protected  |  Red overlay = masked region")
        hint_lbl.setObjectName("dim")
        hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_lay.addWidget(hint_lbl)

        splitter.addWidget(right_w)
        splitter.setSizes([430, 770])

        # ── Bottom bar ───────────────────────────────────────────────────────
        bot_row = QHBoxLayout()
        self.recompute_btn = QPushButton("↺  Recompute preview"); self.recompute_btn.setObjectName("small")
        self.cancel_btn    = QPushButton("✕  Cancel");            self.cancel_btn.setObjectName("small")
        bot_row.addWidget(self.recompute_btn); bot_row.addWidget(self.cancel_btn)
        bot_row.addStretch()
        root.addLayout(bot_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setObjectName("dim")
        root.addWidget(self.status_lbl)

        self._set_props_enabled(False)

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.src_browse.clicked.connect(self._browse_src)
        self.load_btn.clicked.connect(self._load_image)
        self.out_browse.clicked.connect(self._browse_out)
        self.compute_btn.clicked.connect(self._start_compute)
        self.load_siril_btn.clicked.connect(self._load_in_siril)
        self.copy_cmd_btn.clicked.connect(self._copy_cmd)
        self.cancel_btn.clicked.connect(self._cancel)
        self.recompute_btn.clicked.connect(self._start_compute)

        self.add_btn.clicked.connect(self._add_layer)
        self.up_btn.clicked.connect(self._move_up)
        self.dn_btn.clicked.connect(self._move_down)
        self.del_btn.clicked.connect(self._delete_layer)
        self.pre_btn.clicked.connect(self._open_presets)

        self.layer_list.currentRowChanged.connect(self._on_layer_selected)
        self.layer_list.model().rowsMoved.connect(self._on_rows_moved)

        self.lyr_name.textChanged.connect(self._on_name_changed)
        self.lyr_type.currentIndexChanged.connect(self._on_type_changed)
        self.lyr_op.currentIndexChanged.connect(self._on_operator_changed)
        self.lyr_opacity.valueChanged.connect(self._on_opacity_changed)
        self.lyr_invert.stateChanged.connect(self._on_invert_changed)
        self.preview_layer_btn.clicked.connect(self._preview_current_layer)

        self.save_stack_btn.clicked.connect(self._save_stack)
        self.load_stack_btn.clicked.connect(self._load_stack)

        for _, (_, panel) in self._panels.items():
            panel.params_changed.connect(self._on_params_changed)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _browse_src(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FITS image", "",
            "FITS files (*.fit *.fits *.fts);;All files (*)")
        if path:
            self.src_path.setText(path)
            # Auto-set output path
            base, _ = os.path.splitext(path)
            self.out_path.setText(base + "_mask.fit")

    def _load_image(self):
        path = self.src_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Error", "Please select a valid FITS file.")
            return
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
            if data is None:
                QMessageBox.warning(self, "Error", "FITS file has no image data.")
                return
            self.image_data = data.astype(np.float64)

            # Determine color
            if self.image_data.ndim == 2:
                self.is_color = False
                h, w = self.image_data.shape
                info = f"{w} × {h}  |  Mono"
            elif self.image_data.ndim == 3:
                if self.image_data.shape[0] == 3:
                    self.is_color = True
                    _, h, w = self.image_data.shape
                    info = f"{w} × {h}  |  RGB (3 channels)"
                elif self.image_data.shape[2] == 3:
                    self.is_color = True
                    h, w, _ = self.image_data.shape
                    info = f"{w} × {h}  |  RGB (HWC)"
                else:
                    self.is_color = False
                    h, w = self.image_data.shape[1], self.image_data.shape[2]
                    info = f"{w} × {h}  |  Mono (stacked)"
            else:
                QMessageBox.warning(self, "Error", "Unsupported FITS data shape.")
                return

            # Large image note
            if max(h, w) > 4000:
                info += "  |  ⚠ >4000px: preview at 50%"

            self.img_info.setText(info)
            self._set_status(f"Loaded: {os.path.basename(path)}")
            self.canvas.show_original(self.image_data)
            self._update_color_mask_availability()

        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _update_color_mask_availability(self):
        """Disable color mask type when image is mono."""
        # Find index of color type in combo
        for i in range(self.lyr_type.count()):
            if self.lyr_type.itemData(i) == "color":
                model = self.lyr_type.model()
                item  = model.item(i)
                if item:
                    item.setEnabled(self.is_color)
                break

    # ── Layer management ──────────────────────────────────────────────────────

    def _add_layer(self):
        layer = _make_layer(
            f"Layer {len(self.layers)+1}", "range", "AND", 1.0, False,
            {"lo": 0.1, "hi": 0.7, "feather": 20.0}
        )
        self.layers.append(layer)
        self._rebuild_list()
        self.layer_list.setCurrentRow(len(self.layers) - 1)

    def _delete_layer(self):
        row = self.layer_list.currentRow()
        if row < 0 or row >= len(self.layers):
            return
        self.layers.pop(row)
        self._rebuild_list()
        if self.layers:
            self.layer_list.setCurrentRow(min(row, len(self.layers)-1))
        else:
            self._set_props_enabled(False)

    def _move_up(self):
        row = self.layer_list.currentRow()
        if row <= 0: return
        self.layers.insert(row-1, self.layers.pop(row))
        self._rebuild_list()
        self.layer_list.setCurrentRow(row-1)

    def _move_down(self):
        row = self.layer_list.currentRow()
        if row < 0 or row >= len(self.layers)-1: return
        self.layers.insert(row+1, self.layers.pop(row))
        self._rebuild_list()
        self.layer_list.setCurrentRow(row+1)

    def _on_rows_moved(self, *_):
        """Sync layers list after drag-drop reorder."""
        new_order = []
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            idx  = item.data(Qt.ItemDataRole.UserRole)
            # idx is original position, find by id
            layer_id = item.data(Qt.ItemDataRole.UserRole + 1)
            for lyr in self.layers:
                if lyr["id"] == layer_id:
                    new_order.append(lyr)
                    break
        self.layers = new_order
        self._rebuild_list()

    def _rebuild_list(self):
        self._suppress_signals = True
        self.layer_list.clear()
        for layer in self.layers:
            icon = MASK_ICONS.get(layer["type"], "?")
            op   = layer.get("operator", "AND")
            chk  = "✓" if layer.get("enabled", True) else "✗"
            text = f"{chk}  {icon}  {layer['name']}  [{op}]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole + 1, layer["id"])
            if not layer.get("enabled", True):
                item.setForeground(QColor(SIRIL_TEXT_DIM))
            self.layer_list.addItem(item)
        self._suppress_signals = False

    def _on_layer_selected(self, row):
        if self._suppress_signals: return
        if row < 0 or row >= len(self.layers):
            self._set_props_enabled(False)
            return
        self._set_props_enabled(True)
        self._load_layer_props(self.layers[row])

    def _load_layer_props(self, layer):
        self._suppress_signals = True
        self.lyr_name.setText(layer.get("name", ""))

        # Set type combo
        type_key = layer.get("type", "range")
        for i in range(self.lyr_type.count()):
            if self.lyr_type.itemData(i) == type_key:
                self.lyr_type.setCurrentIndex(i)
                break

        # Set operator
        op = layer.get("operator", "AND")
        idx = self.lyr_op.findText(op)
        if idx >= 0: self.lyr_op.setCurrentIndex(idx)

        # Opacity
        self.lyr_opacity.setValue(int(layer.get("opacity", 1.0) * 100))
        self.lyr_invert.setChecked(layer.get("invert", False))

        # Load panel params
        panel_idx, panel = self._panels.get(type_key, (0, self.panel_range))
        self.param_stack.setCurrentIndex(panel_idx)
        panel.set_params(layer.get("params", {}))

        self._suppress_signals = False

    def _set_props_enabled(self, enabled):
        for w in [self.lyr_name, self.lyr_type, self.lyr_op, self.lyr_opacity,
                  self.lyr_invert, self.param_stack, self.preview_layer_btn]:
            w.setEnabled(enabled)

    # ── Layer property change handlers ────────────────────────────────────────

    def _current_layer(self):
        row = self.layer_list.currentRow()
        if 0 <= row < len(self.layers):
            return self.layers[row]
        return None

    def _invalidate_current(self):
        layer = self._current_layer()
        if layer:
            layer["cached_mask"] = None

    def _on_name_changed(self, text):
        if self._suppress_signals: return
        layer = self._current_layer()
        if layer:
            layer["name"] = text
            self._rebuild_list()
            self.layer_list.setCurrentRow(
                next((i for i, l in enumerate(self.layers) if l["id"] == layer["id"]), 0))

    def _on_type_changed(self, idx):
        if self._suppress_signals: return
        layer = self._current_layer()
        if not layer: return
        type_key = self.lyr_type.itemData(idx)
        layer["type"] = type_key
        layer["cached_mask"] = None
        panel_idx, panel = self._panels.get(type_key, (0, self.panel_range))
        self.param_stack.setCurrentIndex(panel_idx)
        panel.set_params(layer.get("params", {}))
        self._rebuild_list()

    def _on_operator_changed(self, idx):
        if self._suppress_signals: return
        layer = self._current_layer()
        if layer:
            layer["operator"] = self.lyr_op.currentText()
            self._rebuild_list()

    def _on_opacity_changed(self, val):
        if self._suppress_signals: return
        self.lyr_op_label.setText(f"{val}%")
        layer = self._current_layer()
        if layer:
            layer["opacity"] = val / 100.0

    def _on_invert_changed(self, state):
        if self._suppress_signals: return
        layer = self._current_layer()
        if layer:
            layer["invert"] = bool(state)

    def _on_params_changed(self):
        if self._suppress_signals: return
        layer = self._current_layer()
        if not layer: return
        type_key = layer["type"]
        _, panel = self._panels.get(type_key, (0, self.panel_range))
        layer["params"] = panel.get_params()
        layer["cached_mask"] = None

    # ── Presets ───────────────────────────────────────────────────────────────

    def _open_presets(self):
        dlg = PresetsDialog(self)
        dlg.preset_selected.connect(self._apply_preset)
        dlg.exec()

    def _apply_preset(self, preset):
        self.layers = preset["layers"]
        self._rebuild_list()
        if self.layers:
            self.layer_list.setCurrentRow(0)
        self._set_status(f"Preset loaded: {preset['name']}")

    # ── Output / Siril integration ────────────────────────────────────────────

    def _browse_out(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mask as", "",
            "FITS files (*.fit *.fits);;All files (*)")
        if path:
            self.out_path.setText(path)

    def _load_in_siril(self):
        path = self.out_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Error",
                                "Mask file not found. Compute and export first.")
            return
        try:
            siril = s.SirilInterface()
            siril.connect()
            try:
                siril.cmd("load_mask", path)
                self._set_status(f"Mask loaded in Siril: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.information(
                    self, "Siril info",
                    f"Could not auto-load (may not be supported in your Siril version):\n{e}\n\n"
                    f"Manual: Image → Load Mask → select the file, or paste into Siril console:\n"
                    f"load_mask {path}")
            finally:
                siril.disconnect()
        except Exception as e:
            QMessageBox.critical(self, "Siril connection error", str(e))

    def _copy_cmd(self):
        path = self.out_path.text().strip()
        if not path:
            self._set_status("No output path set.")
            return
        cmd = f"load_mask {path}"
        QApplication.clipboard().setText(cmd)
        self._set_status(f"Copied to clipboard: {cmd}")

    # ── Compute ───────────────────────────────────────────────────────────────

    def _start_compute(self, single_layer_idx=None):
        if self.image_data is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return
        if not self.layers:
            QMessageBox.warning(self, "Error", "Add at least one mask layer.")
            return
        if self.worker and self.worker.isRunning():
            return

        out_path = self.out_path.text().strip() if single_layer_idx is None else None
        self._cancel_evt.clear()
        self.worker = MaskWorker(
            layers=copy.deepcopy(self.layers),
            data=self.image_data,
            output_path=out_path,
            cancel_event=self._cancel_evt,
            single_layer_idx=single_layer_idx,
        )
        self.worker.log_line.connect(self._log)
        self.worker.progress.connect(self._on_progress)
        self.worker.preview_ready.connect(self._on_preview_ready)
        self.worker.finished.connect(self._on_finished)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.compute_btn.setEnabled(False)
        self.worker.start()

    def _preview_current_layer(self):
        row = self.layer_list.currentRow()
        if row >= 0:
            # Push current params into layer before preview
            self._on_params_changed()
            self._start_compute(single_layer_idx=row)

    def _cancel(self):
        self._cancel_evt.set()
        if self.worker:
            self.worker.cancel()
        self._set_status("Cancelling...")

    def _on_progress(self, done, total, name):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        self._set_status(f"Computing: {name} ({done}/{total})")

    def _on_preview_ready(self, mask, overlay):
        self.canvas.show_mask_and_overlay(mask, overlay)

    def _on_finished(self, result):
        self.progress_bar.setVisible(False)
        self.compute_btn.setEnabled(True)
        if result.get("success"):
            coverage = result.get("mask_coverage", 0) * 100
            mean_val = result.get("mask_mean", 0) * 100
            msg = f"Coverage: {coverage:.1f}% fully affected (>50%)  |  Mean: {mean_val:.1f}%"
            if result.get("output"):
                msg += f"  |  Saved: {os.path.basename(result['output'])}"
            self._set_status(msg)
        else:
            err = result.get("error", "Unknown error")
            self._set_status(f"Error: {err}")

    def _log(self, line):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[mask_builder {ts}] {line}")

    def _set_status(self, msg):
        self.status_lbl.setText(msg)

    # ── Save / load layer stack ───────────────────────────────────────────────

    def _save_stack(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save layer stack", SCRIPT_DIR,
            "JSON files (*.json);;All files (*)")
        if not path: return
        exportable = []
        for layer in self.layers:
            l = {k: v for k, v in layer.items() if k != "cached_mask"}
            exportable.append(l)
        try:
            with open(path, "w") as f:
                json.dump(exportable, f, indent=2)
            self._set_status(f"Layer stack saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _load_stack(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load layer stack", SCRIPT_DIR,
            "JSON files (*.json);;All files (*)")
        if not path: return
        try:
            with open(path) as f:
                layers = json.load(f)
            for l in layers:
                l["cached_mask"] = None
                if "id" not in l:
                    l["id"] = _new_layer_id()
            self.layers = layers
            self._rebuild_list()
            if self.layers:
                self.layer_list.setCurrentRow(0)
            self._set_status(f"Layer stack loaded: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
