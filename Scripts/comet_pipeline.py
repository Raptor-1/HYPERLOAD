r"""
comet_pipeline.py — Comet Dual-Stack Pipeline for Siril
Script #8 | Fully automated comet+star dual-stack composite
Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
"""

import sirilpy as s

s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import glob
import json
import threading
import copy
from datetime import datetime

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    center_of_mass,
    label,
    shift as ndimage_shift,
    zoom,
)
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ── PyQt6 ─────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QScrollArea, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# ── sirilpy ───────────────────────────────────────────────────────────────────
import sirilpy as siril_module

# ── optional local modules ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

try:
    from mask_builder import (
        make_nebulosity_mask, make_star_mask,
        combine_masks, to_luminance,
    )
    HAS_MASK_BUILDER = True
except ImportError:
    HAS_MASK_BUILDER = False


# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════

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
}}
QListWidget::item {{ padding: 4px 8px; }}
QListWidget::item:selected {{ background: {SIRIL_ACCENT2}; color: white; }}
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
QPushButton#nucleus {{
    background-color: #1a2d1a;
    border-color: {SIRIL_SUCCESS};
    color: {SIRIL_SUCCESS};
    text-align: center;
    font-weight: bold;
}}
QPushButton#nucleus:hover {{ background-color: #223322; }}
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
QSplitter::handle {{ background: {SIRIL_BORDER}; width: 2px; }}
"""


# ══════════════════════════════════════════════════════════════════════════════
# NUCLEUS DETECTION & TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def detect_comet_nucleus_auto(
    frame: np.ndarray,
    search_region: tuple = None,
    blur_sigma: float = 8.0,
    min_area_px: int = 20,
) -> tuple | None:
    """
    Detect comet nucleus by suppressing stars with heavy Gaussian blur,
    then finding the brightest extended blob.
    Returns (y, x) centroid or None.
    """
    h, w = frame.shape
    if search_region:
        y1, y2, x1, x2 = search_region
        region = frame[y1:y2, x1:x2].astype(float)
    else:
        region = frame.astype(float)
        y1, x1 = 0, 0

    mn, mx = region.min(), region.max()
    if mx <= mn:
        return None
    norm = (region - mn) / (mx - mn)

    smoothed = gaussian_filter(norm, sigma=blur_sigma)
    threshold = np.percentile(smoothed, 70)
    binary = (smoothed > threshold).astype(int)

    labeled, num_features = label(binary)
    if num_features == 0:
        return None

    best_label = None
    best_brightness = -1
    for lbl in range(1, num_features + 1):
        blob_mask = labeled == lbl
        area = np.sum(blob_mask)
        if area < min_area_px:
            continue
        brightness = np.sum(smoothed[blob_mask])
        if brightness > best_brightness:
            best_brightness = brightness
            best_label = lbl

    if best_label is None:
        return None

    blob_mask = labeled == best_label
    cy, cx = center_of_mass(smoothed * blob_mask)
    return (cy + y1, cx + x1)


def track_nucleus_sequence(
    frames: list,
    first_pos: tuple,
    search_radius: int = 80,
    blur_sigma: float = 8.0,
    log_callback=None,
) -> list:
    """Track nucleus across all frames. Returns list of (y, x) tuples."""
    positions = [first_pos]
    interpolated_flags = [False]
    h, w = frames[0].shape

    for i, frame in enumerate(frames[1:], 1):
        prev_y, prev_x = positions[-1]
        y1 = max(0, int(prev_y) - search_radius)
        y2 = min(h, int(prev_y) + search_radius)
        x1 = max(0, int(prev_x) - search_radius)
        x2 = min(w, int(prev_x) + search_radius)

        pos = detect_comet_nucleus_auto(
            frame,
            search_region=(y1, y2, x1, x2),
            blur_sigma=blur_sigma,
        )

        if pos is None:
            if len(positions) >= 2:
                dy = positions[-1][0] - positions[-2][0]
                dx = positions[-1][1] - positions[-2][1]
                pos = (positions[-1][0] + dy, positions[-1][1] + dx)
            else:
                pos = positions[-1]
            interpolated_flags.append(True)
            if log_callback:
                log_callback(f"  Frame {i}: not detected, interpolated → ({pos[1]:.1f}, {pos[0]:.1f})")
        else:
            interpolated_flags.append(False)
            if log_callback:
                log_callback(f"  Frame {i}: nucleus at ({pos[1]:.1f}, {pos[0]:.1f})")

        positions.append(pos)

    return positions, interpolated_flags


def compute_comet_shifts(positions: list) -> list:
    """Per-frame shift vectors relative to first frame nucleus."""
    ref_y, ref_x = positions[0]
    return [(ref_y - y, ref_x - x) for y, x in positions]


# ══════════════════════════════════════════════════════════════════════════════
# FRAME SHIFTING & STACKING
# ══════════════════════════════════════════════════════════════════════════════

def shift_frame(
    frame: np.ndarray,
    dy: float,
    dx: float,
    fill_value: float = 0.0,
) -> np.ndarray:
    return ndimage_shift(
        frame.astype(float),
        shift=(dy, dx),
        mode="constant",
        cval=fill_value,
    ).astype(frame.dtype)


def stack_frames_mean(
    frames: list,
    shifts: list = None,
    rejection: str = "sigma",
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    log_callback=None,
) -> np.ndarray:
    n = len(frames)
    if n == 0:
        raise ValueError("No frames to stack")

    h, w = frames[0].shape
    stack = np.zeros((n, h, w), dtype=np.float64)

    for i, frame in enumerate(frames):
        if shifts is not None:
            dy, dx = shifts[i]
            stack[i] = shift_frame(frame, dy, dx)
        else:
            stack[i] = frame.astype(np.float64)

    if rejection == "sigma" and n >= 5:
        median = np.median(stack, axis=0)
        std = np.std(stack, axis=0)
        result = np.zeros((h, w), dtype=np.float64)
        count = np.zeros((h, w), dtype=np.float64)
        for i in range(n):
            diff = np.abs(stack[i] - median)
            good = (diff <= sigma_high * std) & (diff <= sigma_low * std)
            result += stack[i] * good
            count += good
        result = np.where(count > 0, result / count, median)
    else:
        result = np.mean(stack, axis=0)

    if log_callback:
        log_callback(f"  Stacked {n} frames → shape {result.shape}")

    return result.astype(np.float32)


def stack_color_frames_mean(
    frames_list: list,        # list of (3, H, W) arrays
    shifts: list = None,
    rejection: str = "sigma",
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    log_callback=None,
) -> np.ndarray:
    """Stack color (3-channel) frames."""
    channels = []
    for c in range(3):
        ch_frames = [f[c] for f in frames_list]
        ch_shifts = shifts  # same spatial shifts for all channels
        ch_stack = stack_frames_mean(
            ch_frames, shifts=ch_shifts,
            rejection=rejection,
            sigma_low=sigma_low, sigma_high=sigma_high,
            log_callback=log_callback if c == 0 else None,
        )
        channels.append(ch_stack)
    return np.stack(channels, axis=0)  # (3, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# COMET MASK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _nebulosity_mask_inline(
    comet_stack: np.ndarray,
    radial_mask: np.ndarray,
    sigma_above: float,
    feather_px: float,
) -> np.ndarray:
    h, w = comet_stack.shape
    mn, mx = comet_stack.min(), comet_stack.max()
    lum = (comet_stack.astype(float) - mn) / (mx - mn + 1e-10)
    tile_size = 64
    th = max(1, h // tile_size)
    tw = max(1, w // tile_size)
    bg_map = np.zeros((th, tw))
    std_map = np.zeros((th, tw))
    for i in range(th):
        for j in range(tw):
            tile = lum[i * tile_size:(i + 1) * tile_size,
                       j * tile_size:(j + 1) * tile_size]
            med = np.median(tile)
            bg_map[i, j] = med
            std_map[i, j] = np.median(np.abs(tile - med)) * 1.4826
    bg_full = zoom(bg_map, (h / th, w / tw), order=1)[:h, :w]
    std_full = zoom(std_map, (h / th, w / tw), order=1)[:h, :w]
    neb = np.clip(
        (lum - bg_full - sigma_above * std_full) / (std_full + 1e-10), 0, 1
    )
    if feather_px > 0:
        neb = gaussian_filter(neb, sigma=feather_px)
    return np.clip(1 - (1 - radial_mask) * (1 - neb), 0, 1)


def make_comet_mask(
    comet_stack: np.ndarray,
    nucleus_pos: tuple,
    mask_radius_px: float = 150.0,
    feather_px: float = 50.0,
    use_nebulosity: bool = True,
    sigma_above: float = 2.0,
) -> np.ndarray:
    h, w = comet_stack.shape
    cy, cx = nucleus_pos
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    radial_mask = np.clip(
        1.0 - (dist - mask_radius_px) / (feather_px + 1e-10), 0.0, 1.0
    )
    radial_mask[dist <= mask_radius_px] = 1.0

    if use_nebulosity and HAS_MASK_BUILDER:
        lum = to_luminance(comet_stack)
        neb_mask = make_nebulosity_mask(
            lum, tile_size=64, sigma_above=sigma_above, feather_px=feather_px
        )
        combined = combine_masks(radial_mask, neb_mask, operator="OR")
    elif use_nebulosity:
        combined = _nebulosity_mask_inline(
            comet_stack, radial_mask, sigma_above, feather_px
        )
    else:
        combined = radial_mask

    return np.clip(combined, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE
# ══════════════════════════════════════════════════════════════════════════════

def match_backgrounds(
    stars_stack: np.ndarray,
    comet_stack: np.ndarray,
) -> tuple:
    h, w = stars_stack.shape
    margin = min(h, w) // 8

    def corner_median(arr):
        return np.median(np.concatenate([
            arr[:margin, :margin].ravel(),
            arr[:margin, -margin:].ravel(),
            arr[-margin:, :margin].ravel(),
            arr[-margin:, -margin:].ravel(),
        ]))

    s_bg = corner_median(stars_stack.astype(float))
    c_bg = corner_median(comet_stack.astype(float))
    offset = s_bg - c_bg
    adjusted = np.clip(comet_stack.astype(float) + offset, 0, None)
    return stars_stack, adjusted.astype(np.float32)


def composite_stacks(
    stars_stack: np.ndarray,
    comet_stack: np.ndarray,
    comet_mask: np.ndarray,
) -> np.ndarray:
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else arr.copy()

    s_norm = normalize(stars_stack.astype(float))
    c_norm = normalize(comet_stack.astype(float))
    m = comet_mask.astype(float)

    if m.shape != s_norm.shape:
        zy = s_norm.shape[0] / m.shape[0]
        zx = s_norm.shape[1] / m.shape[1]
        m = zoom(m, (zy, zx), order=1)
        m = np.clip(m, 0, 1)

    result = c_norm * m + s_norm * (1.0 - m)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CANVASES
# ══════════════════════════════════════════════════════════════════════════════

class NucleusPickerCanvas(FigureCanvasQTAgg):
    nucleus_selected = pyqtSignal(float, float)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), facecolor=SIRIL_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(SIRIL_BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.nucleus_pos = None
        self.mpl_connect("button_press_event", self._on_click)
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor(SIRIL_BG)
        self.ax.text(
            0.5, 0.5,
            "Scan frames and auto-detect,\nor load first frame for manual click",
            color=SIRIL_TEXT_DIM, ha="center", va="center",
            fontsize=9, transform=self.ax.transAxes,
        )
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_frame(self, frame: np.ndarray, auto_pos: tuple = None):
        self.ax.clear()
        self.ax.set_facecolor(SIRIL_BG)
        p_lo = np.percentile(frame, 0.5)
        p_hi = np.percentile(frame, 99.5)
        disp = np.clip((frame.astype(float) - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)
        self.ax.imshow(disp, cmap="gray", origin="lower",
                       aspect="equal", interpolation="nearest")
        if auto_pos is not None:
            cy, cx = auto_pos
            self.ax.plot(cx, cy, "+", color=SIRIL_SUCCESS,
                         markersize=20, markeredgewidth=2,
                         label=f"Auto: ({cx:.1f}, {cy:.1f})")
            self.ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                           labelcolor=SIRIL_TEXT, fontsize=8, loc="upper right")
        self.ax.set_title("Click on the comet nucleus to confirm position",
                          color=SIRIL_SECTION, fontsize=9)
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None:
            return
        self.nucleus_pos = (cy, cx)
        self.ax.set_title(
            f"Nucleus set at ({cx:.1f}, {cy:.1f}) — click again to adjust",
            color=SIRIL_SUCCESS, fontsize=9,
        )
        for line in list(self.ax.lines):
            line.remove()
        self.ax.plot(cx, cy, "+", color=SIRIL_WARNING,
                     markersize=24, markeredgewidth=2.5)
        self.draw()
        self.nucleus_selected.emit(cy, cx)


class TrackingCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), facecolor=SIRIL_BG)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor(SIRIL_BG)
        self.ax.text(
            0.5, 0.5, "Tracking path shown after pipeline runs",
            color=SIRIL_TEXT_DIM, ha="center", va="center",
            fontsize=9, transform=self.ax.transAxes,
        )
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_tracking(
        self,
        first_frame: np.ndarray,
        positions: list,
        interpolated_flags: list = None,
    ):
        self.ax.clear()
        self.ax.set_facecolor(SIRIL_BG)
        p_lo = np.percentile(first_frame, 0.5)
        p_hi = np.percentile(first_frame, 99.5)
        disp = np.clip(
            (first_frame.astype(float) - p_lo) / (p_hi - p_lo + 1e-10), 0, 1
        )
        self.ax.imshow(disp, cmap="gray", origin="lower",
                       aspect="equal", interpolation="nearest")

        if positions:
            ys = [p[0] for p in positions]
            xs = [p[1] for p in positions]
            n = len(positions)
            flags = interpolated_flags or [False] * n

            for i in range(n - 1):
                frac = i / max(n - 1, 1)
                color = (frac, 1.0 - frac, 0.0)
                ls = "--" if flags[i + 1] else "-"
                self.ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                             color=color, linewidth=1.5, alpha=0.8, linestyle=ls)

            self.ax.plot(xs[0], ys[0], "o", color=SIRIL_SUCCESS,
                         markersize=8, label="Frame 1")
            self.ax.plot(xs[-1], ys[-1], "o", color=SIRIL_ERROR,
                         markersize=8, label=f"Frame {n}")

            dy = ys[-1] - ys[0]
            dx = xs[-1] - xs[0]
            drift = np.sqrt(dy ** 2 + dx ** 2)
            interp_pct = sum(flags) / max(n, 1) * 100
            title_color = SIRIL_WARNING if interp_pct > 20 else SIRIL_SECTION
            self.ax.set_title(
                f"Tracking: {n} frames | Drift: {drift:.1f}px | Interp: {interp_pct:.0f}%",
                color=title_color, fontsize=9,
            )
            self.ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                           labelcolor=SIRIL_TEXT, fontsize=8)

        for spine in self.ax.spines.values():
            spine.set_color(SIRIL_BORDER)
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.fig.tight_layout(pad=0.3)
        self.draw()


class ResultPreviewCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), facecolor=SIRIL_BG)
        self.axes = [self.fig.add_subplot(2, 2, i + 1) for i in range(4)]
        self._titles = ["Stars stack", "Comet stack", "Comet mask", "Final composite"]
        super().__init__(self.fig)
        self.setParent(parent)
        self._draw_placeholder()

    def _style_all(self):
        colors = [SIRIL_TEXT_DIM, SIRIL_TEXT_DIM, SIRIL_TEXT_DIM, SIRIL_ACCENT]
        for ax, title, color in zip(self.axes, self._titles, colors):
            ax.set_facecolor(SIRIL_BG)
            ax.set_title(title, color=color, fontsize=8)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color(SIRIL_BORDER)

    def _draw_placeholder(self):
        for ax, title in zip(self.axes, self._titles):
            ax.clear()
            ax.set_facecolor(SIRIL_BG)
            ax.text(0.5, 0.5, title, color=SIRIL_TEXT_DIM,
                    ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def _disp(self, arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
        if is_mask:
            return np.clip(arr.astype(float), 0, 1)
        p_lo = np.percentile(arr, 0.5)
        p_hi = np.percentile(arr, 99.5)
        return np.clip((arr.astype(float) - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)

    def _ds(self, arr: np.ndarray, max_px: int = 400) -> np.ndarray:
        factor = max(1, max(arr.shape[0], arr.shape[1]) // max_px)
        return arr[::factor, ::factor]

    def update_all(
        self,
        stars: np.ndarray,
        comet: np.ndarray,
        mask: np.ndarray,
        composite: np.ndarray,
    ):
        arrays = [stars, comet, mask, composite]
        is_mask = [False, False, True, False]
        for ax, arr, im in zip(self.axes, arrays, is_mask):
            ax.clear()
            d = self._ds(self._disp(arr, im))
            ax.imshow(d, cmap="gray", origin="lower",
                      aspect="equal", interpolation="nearest")
        self._style_all()
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class CometWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log_line = pyqtSignal(str)
    tracking_done = pyqtSignal(list, list, object)   # positions, flags, first_frame
    stacks_ready = pyqtSignal(object, object, object, object)
    finished = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config = config
        self._cancel = cancel_event
        self._siril = None

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_line.emit(f"[{ts}] {msg}")

    def _check_cancel(self) -> bool:
        if self._cancel.is_set():
            self._log("⚠ Cancelled by user.")
            self.finished.emit({"success": False, "error": "Cancelled by user"})
            return True
        return False

    def run(self):
        try:
            cfg = self.config
            lights_dir = cfg["lights_dir"]
            out_dir = cfg.get("output_dir") or lights_dir
            os.makedirs(out_dir, exist_ok=True)

            self._siril = siril_module.SirilInterface()
            self._siril.connect()

            # ── Step 1: Star registration ────────────────────────────────────
            if not cfg.get("pre_registered", False):
                self.progress.emit(1, 10, "Star registration…")
                self._log("Running Siril star registration…")
                self._siril.cmd("cd", lights_dir)
                self._siril.cmd("convert", "light", "-out=lights_")
                self._siril.cmd("register", "lights_")
                if self._check_cancel():
                    return

                # ── Step 2: Star stack ───────────────────────────────────────
                self.progress.emit(2, 10, "Star-aligned stack…")
                self._log("Creating star-aligned stack…")
                sl = str(cfg.get("sigma_low", 3.0))
                sh = str(cfg.get("sigma_high", 3.0))
                self._siril.cmd(
                    "stack", "lights_", "rej", "w", sl, sh,
                    f"-output={os.path.join(out_dir, 'master_stars')}"
                )
            else:
                self.progress.emit(2, 10, "Skipping registration (pre-registered)…")
                self._log("Pre-registered sequence: skipping Siril registration.")

            if self._check_cancel():
                return

            # ── Load master_stars ────────────────────────────────────────────
            stars_path = os.path.join(out_dir, "master_stars.fit")
            if not os.path.exists(stars_path):
                # Try .fits
                stars_path = os.path.join(out_dir, "master_stars.fits")
            self._log(f"Loading star stack: {stars_path}")
            stars_raw = fits.getdata(stars_path).astype(np.float32)
            stars_is_color = stars_raw.ndim == 3 and stars_raw.shape[0] == 3
            if stars_is_color:
                stars_lum = (
                    0.299 * stars_raw[0] + 0.587 * stars_raw[1] + 0.114 * stars_raw[2]
                ).astype(np.float32)
            elif stars_raw.ndim == 3:
                stars_lum = stars_raw[0].astype(np.float32)
            else:
                stars_lum = stars_raw

            # ── Step 3: Load registered frames ───────────────────────────────
            self.progress.emit(3, 10, "Loading frames…")
            self._log("Scanning for registered frames…")
            patterns = [
                os.path.join(lights_dir, "lights_*.fit"),
                os.path.join(lights_dir, "lights_*.fits"),
                os.path.join(lights_dir, "lights_*.fts"),
            ]
            frame_files = sorted(
                fp for pat in patterns for fp in glob.glob(pat)
            )
            if not frame_files:
                self._log("⚠ No lights_*.fit files found. Looking for *.fit...")
                frame_files = sorted(
                    glob.glob(os.path.join(lights_dir, "*.fit")) +
                    glob.glob(os.path.join(lights_dir, "*.fits")) +
                    glob.glob(os.path.join(lights_dir, "*.fts"))
                )

            max_in_mem = cfg.get("max_frames_memory", 50)
            if len(frame_files) > max_in_mem:
                self._log(
                    f"⚠ {len(frame_files)} frames found; loading first {max_in_mem} "
                    f"(max_frames_memory={max_in_mem})"
                )
                frame_files = frame_files[:max_in_mem]

            self._log(f"Loading {len(frame_files)} frames…")
            frames_lum = []
            frames_color = []
            first_is_color = False

            for i, fp in enumerate(frame_files):
                d = fits.getdata(fp).astype(np.float32)
                if d.ndim == 3 and d.shape[0] == 3:
                    first_is_color = True
                    frames_color.append(d)
                    frames_lum.append(
                        (0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2]).astype(np.float32)
                    )
                elif d.ndim == 3:
                    frames_lum.append(d[0])
                    frames_color.append(None)
                else:
                    frames_lum.append(d)
                    frames_color.append(None)

                if i % 10 == 0:
                    self._log(f"  Loaded frame {i+1}/{len(frame_files)}")

            self._log(f"Loaded {len(frames_lum)} frames (color: {first_is_color})")
            if self._check_cancel():
                return

            # ── Step 4: Nucleus detection ─────────────────────────────────────
            self.progress.emit(4, 10, "Detecting nucleus in first frame…")
            if cfg.get("manual_nucleus_pos"):
                first_pos = cfg["manual_nucleus_pos"]
                self._log(
                    f"Using manual nucleus: ({first_pos[1]:.1f}, {first_pos[0]:.1f})"
                )
            else:
                self._log("Auto-detecting nucleus in first frame…")
                first_pos = detect_comet_nucleus_auto(
                    frames_lum[0], blur_sigma=cfg.get("blur_sigma", 8.0)
                )
                if first_pos is None:
                    self.finished.emit({
                        "success": False,
                        "error": (
                            "Nucleus not auto-detected in first frame.\n"
                            "Switch to Manual mode and click the nucleus."
                        ),
                    })
                    return
                self._log(
                    f"Nucleus detected at ({first_pos[1]:.1f}, {first_pos[0]:.1f})"
                )

            # ── Step 5: Track nucleus ─────────────────────────────────────────
            self.progress.emit(5, 10, "Tracking nucleus…")
            self._log("Tracking nucleus across all frames…")
            positions, interp_flags = track_nucleus_sequence(
                frames_lum,
                first_pos,
                search_radius=cfg.get("search_radius", 80),
                blur_sigma=cfg.get("blur_sigma", 8.0),
                log_callback=self._log,
            )
            self.tracking_done.emit(positions, interp_flags, frames_lum[0])

            interp_pct = sum(interp_flags) / max(len(interp_flags), 1) * 100
            if interp_pct > 20:
                self._log(
                    f"⚠ {interp_pct:.0f}% of frames required interpolation. "
                    "Consider manual nucleus mode or larger search radius."
                )

            if self._check_cancel():
                return

            # ── Step 6: Compute shifts ─────────────────────────────────────────
            self.progress.emit(6, 10, "Computing comet shifts…")
            shifts = compute_comet_shifts(positions)

            # ── Step 7: Comet-aligned stack ───────────────────────────────────
            self.progress.emit(7, 10, "Building comet-aligned stack…")
            self._log("Stacking frames on comet nucleus…")
            rejection = "sigma" if cfg.get("rejection", "sigma") == "sigma" else "none"

            if first_is_color and all(f is not None for f in frames_color):
                self._log("  Color frames detected — stacking per channel…")
                comet_color = stack_color_frames_mean(
                    frames_color, shifts=shifts,
                    rejection=rejection,
                    sigma_low=float(cfg.get("sigma_low", 3.0)),
                    sigma_high=float(cfg.get("sigma_high", 3.0)),
                    log_callback=self._log,
                )
                comet_lum = (
                    0.299 * comet_color[0] +
                    0.587 * comet_color[1] +
                    0.114 * comet_color[2]
                ).astype(np.float32)
            else:
                comet_lum = stack_frames_mean(
                    frames_lum, shifts=shifts,
                    rejection=rejection,
                    sigma_low=float(cfg.get("sigma_low", 3.0)),
                    sigma_high=float(cfg.get("sigma_high", 3.0)),
                    log_callback=self._log,
                )

            # Crop border artifacts if requested
            if cfg.get("crop_overlap", False):
                max_dy = int(max(abs(s[0]) for s in shifts)) + 1
                max_dx = int(max(abs(s[1]) for s in shifts)) + 1
                h, w = comet_lum.shape
                if max_dy * 2 < h and max_dx * 2 < w:
                    comet_lum = comet_lum[max_dy:h - max_dy, max_dx:w - max_dx]
                    stars_lum = stars_lum[max_dy:h - max_dy, max_dx:w - max_dx]
                    self._log(
                        f"  Cropped to valid overlap: {comet_lum.shape} "
                        f"(removed {max_dy}px top/bottom, {max_dx}px left/right)"
                    )

            if self._check_cancel():
                return

            # ── Step 8: Comet mask ────────────────────────────────────────────
            self.progress.emit(8, 10, "Generating comet mask…")
            self._log("Generating comet region mask…")
            nuc_in_comet = detect_comet_nucleus_auto(
                comet_lum, blur_sigma=cfg.get("blur_sigma", 8.0)
            ) or (comet_lum.shape[0] // 2, comet_lum.shape[1] // 2)
            self._log(
                f"  Nucleus in comet stack: ({nuc_in_comet[1]:.1f}, {nuc_in_comet[0]:.1f})"
            )
            comet_mask = make_comet_mask(
                comet_lum,
                nuc_in_comet,
                mask_radius_px=float(cfg.get("mask_radius", 150)),
                feather_px=float(cfg.get("mask_feather", 50)),
                use_nebulosity=cfg.get("use_nebulosity_mask", True),
                sigma_above=float(cfg.get("nebulosity_sigma", 2.0)),
            )

            # ── Step 9: Composite ─────────────────────────────────────────────
            self.progress.emit(9, 10, "Compositing stacks…")
            self._log("Matching backgrounds and blending…")
            stars_matched, comet_matched = match_backgrounds(stars_lum, comet_lum)
            composite = composite_stacks(stars_matched, comet_matched, comet_mask)

            # ── Step 10: Save ─────────────────────────────────────────────────
            self.progress.emit(10, 10, "Saving outputs…")
            comet_name = cfg.get("comet_name", "").strip()
            prefix = f"{comet_name}_" if comet_name else ""

            def save_fit(arr, name, history=""):
                path = os.path.join(out_dir, name)
                hdu = fits.PrimaryHDU(data=arr.astype(np.float32))
                if history:
                    hdu.header["HISTORY"] = history
                hdu.header["CREATOR"] = "comet_pipeline.py"
                hdu.writeto(path, overwrite=True)
                self._log(f"Saved: {path}")
                return path

            save_fit(stars_lum, f"{prefix}master_stars.fit",
                     "Star-aligned stack — comet_pipeline.py")
            comet_path = save_fit(comet_lum, f"{prefix}master_comet.fit",
                                  "Comet-aligned stack — comet_pipeline.py")
            save_fit(comet_mask, f"{prefix}comet_mask.fit",
                     "Comet region mask — comet_pipeline.py")
            comp_path = save_fit(composite, f"{prefix}final_composite.fit",
                                 "Comet composite — comet_pipeline.py")

            self.stacks_ready.emit(stars_lum, comet_lum, comet_mask, composite)

            if cfg.get("load_in_siril", True):
                self._log(f"Loading composite in Siril: {comp_path}")
                self._siril.cmd("load", comp_path)

            drift_px = float(np.sqrt(
                (positions[-1][0] - positions[0][0]) ** 2 +
                (positions[-1][1] - positions[0][1]) ** 2
            ))
            self._log(f"✓ Pipeline complete. Total drift: {drift_px:.1f} px")
            self.finished.emit({
                "success": True,
                "composite_path": comp_path,
                "comet_path": comet_path,
                "nucleus_pos": first_pos,
                "total_drift_px": drift_px,
                "interp_pct": interp_pct,
            })

        except Exception as exc:
            import traceback
            self._log(f"ERROR: {exc}")
            self._log(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(exc)})
        finally:
            if self._siril:
                try:
                    self._siril.disconnect()
                except Exception:
                    pass

    def cancel(self):
        self._cancel.set()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_separator() -> QFrame:
    f = QFrame()
    f.setObjectName("separator")
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    return f


def _label(text: str, role: str = "") -> QLabel:
    lbl = QLabel(text)
    if role:
        lbl.setObjectName(role)
    return lbl


def _browse_folder(parent, line_edit: QLineEdit, title: str = "Select folder"):
    folder = QFileDialog.getExistingDirectory(parent, title)
    if folder:
        line_edit.setText(folder)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("☄  Comet Dual-Stack Pipeline  —  Siril")
        self.setMinimumSize(900, 720)
        self.resize(960, 780)

        self._worker: CometWorker | None = None
        self._cancel_event = threading.Event()
        self._first_frame: np.ndarray | None = None
        self._auto_nucleus_pos: tuple | None = None
        self._manual_nucleus_pos: tuple | None = None
        self._tracking_positions: list = []
        self._tracking_flags: list = []

        self._build_ui()

    # ── UI BUILD ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Title bar
        title_row = QHBoxLayout()
        title_lbl = QLabel("☄  Comet Dual-Stack Pipeline")
        title_lbl.setObjectName("section")
        f = title_lbl.font()
        f.setPointSize(13)
        f.setBold(True)
        title_lbl.setFont(f)
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        title_row.addWidget(_label("v1.0", "dim"))
        root.addLayout(title_row)
        root.addWidget(_make_separator())

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)
        self._build_tab_input()
        self._build_tab_settings()
        self._build_tab_run()
        self._build_tab_results()
        self._build_tab_log()

        root.addWidget(_make_separator())

        # Bottom bar
        bottom = QHBoxLayout()
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setFixedWidth(120)
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_cancel.setEnabled(False)
        bottom.addWidget(self._btn_cancel)
        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setObjectName("dim")
        bottom.addWidget(self._status_lbl, 1)
        root.addLayout(bottom)

    # ── TAB 1: INPUT & NUCLEUS ───────────────────────────────────────────────

    def _build_tab_input(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # Lights folder group
        grp_folder = QGroupBox("Lights folder")
        fl = QVBoxLayout(grp_folder)

        folder_row = QHBoxLayout()
        self._edit_lights = QLineEdit()
        self._edit_lights.setPlaceholderText("Path to calibrated light frames…")
        folder_row.addWidget(self._edit_lights)
        btn_browse_lights = QPushButton("Browse…")
        btn_browse_lights.setFixedWidth(80)
        btn_browse_lights.clicked.connect(
            lambda: _browse_folder(self, self._edit_lights, "Select lights folder")
        )
        folder_row.addWidget(btn_browse_lights)
        fl.addLayout(folder_row)

        scan_row = QHBoxLayout()
        btn_scan = QPushButton("🔍  Scan frames")
        btn_scan.setObjectName("primary")
        btn_scan.setFixedWidth(140)
        btn_scan.clicked.connect(self._on_scan_frames)
        scan_row.addWidget(btn_scan)
        self._lbl_frame_info = QLabel("No frames scanned.")
        self._lbl_frame_info.setObjectName("dim")
        scan_row.addWidget(self._lbl_frame_info, 1)
        fl.addLayout(scan_row)

        # Input type toggle
        input_type_row = QHBoxLayout()
        input_type_row.addWidget(QLabel("Input type:"))
        self._combo_input_type = QComboBox()
        self._combo_input_type.addItems([
            "Raw lights (will register in Siril)",
            "Pre-registered sequence (skip registration)",
        ])
        self._combo_input_type.setFixedWidth(280)
        input_type_row.addWidget(self._combo_input_type)
        input_type_row.addStretch()
        fl.addLayout(input_type_row)

        # Comet name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Comet name (optional):"))
        self._edit_comet_name = QLineEdit()
        self._edit_comet_name.setPlaceholderText("e.g. C2025A1  →  C2025A1_master_stars.fit")
        self._edit_comet_name.setFixedWidth(220)
        name_row.addWidget(self._edit_comet_name)
        name_row.addStretch()
        fl.addLayout(name_row)

        lay.addWidget(grp_folder)

        # Nucleus detection group
        grp_nucleus = QGroupBox("Nucleus detection")
        nl = QVBoxLayout(grp_nucleus)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._combo_nucleus_mode = QComboBox()
        self._combo_nucleus_mode.addItems(["Automatic", "Manual (click on canvas)"])
        self._combo_nucleus_mode.setFixedWidth(220)
        self._combo_nucleus_mode.currentIndexChanged.connect(self._on_nucleus_mode_changed)
        mode_row.addWidget(self._combo_nucleus_mode)
        mode_row.addStretch()
        nl.addLayout(mode_row)

        # Automatic params
        self._auto_params_widget = QWidget()
        auto_form = QFormLayout(self._auto_params_widget)
        auto_form.setContentsMargins(0, 0, 0, 0)

        self._spin_blur = QDoubleSpinBox()
        self._spin_blur.setRange(3.0, 20.0)
        self._spin_blur.setValue(8.0)
        self._spin_blur.setSingleStep(0.5)
        self._spin_blur.setToolTip("Larger = better star suppression, slower")
        auto_form.addRow("Blur sigma:", self._spin_blur)

        self._spin_search_radius = QSpinBox()
        self._spin_search_radius.setRange(20, 200)
        self._spin_search_radius.setValue(80)
        self._spin_search_radius.setSuffix(" px")
        self._spin_search_radius.setToolTip("Max comet movement between frames")
        auto_form.addRow("Search radius:", self._spin_search_radius)

        detect_row = QHBoxLayout()
        btn_auto_detect = QPushButton("🔍  Auto-detect in first frame")
        btn_auto_detect.setObjectName("nucleus")
        btn_auto_detect.clicked.connect(self._on_auto_detect)
        detect_row.addWidget(btn_auto_detect)
        self._lbl_auto_result = QLabel("Not detected yet.")
        self._lbl_auto_result.setObjectName("dim")
        detect_row.addWidget(self._lbl_auto_result, 1)
        auto_form.addRow("", detect_row)

        nl.addWidget(self._auto_params_widget)

        # Manual params
        self._manual_params_widget = QWidget()
        man_lay = QVBoxLayout(self._manual_params_widget)
        man_lay.setContentsMargins(0, 0, 0, 0)
        man_lay.addWidget(_label("Click on the comet nucleus in the preview below.", "dim"))
        btn_load_first = QPushButton("Load first frame for clicking")
        btn_load_first.setObjectName("nucleus")
        btn_load_first.setFixedWidth(220)
        btn_load_first.clicked.connect(self._on_load_first_frame)
        man_lay.addWidget(btn_load_first)
        self._manual_params_widget.setVisible(False)
        nl.addWidget(self._manual_params_widget)

        self._lbl_nucleus_pos = _label("Nucleus: not set", "dim")
        nl.addWidget(self._lbl_nucleus_pos)

        lay.addWidget(grp_nucleus)

        # Nucleus picker canvas
        self._canvas_picker = NucleusPickerCanvas()
        self._canvas_picker.setMinimumHeight(300)
        self._canvas_picker.nucleus_selected.connect(self._on_nucleus_clicked)
        lay.addWidget(self._canvas_picker, 1)

        self._tabs.addTab(tab, "📁  Input & Nucleus")

    # ── TAB 2: STACK SETTINGS ────────────────────────────────────────────────

    def _build_tab_settings(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # Stacking group
        grp_stack = QGroupBox("Stacking")
        sf = QFormLayout(grp_stack)

        self._combo_rejection = QComboBox()
        self._combo_rejection.addItems(["Winsorized sigma", "None"])
        sf.addRow("Rejection method:", self._combo_rejection)

        self._spin_sigma_low = QDoubleSpinBox()
        self._spin_sigma_low.setRange(1.0, 5.0)
        self._spin_sigma_low.setValue(3.0)
        self._spin_sigma_low.setSingleStep(0.1)
        sf.addRow("Sigma low:", self._spin_sigma_low)

        self._spin_sigma_high = QDoubleSpinBox()
        self._spin_sigma_high.setRange(1.0, 5.0)
        self._spin_sigma_high.setValue(3.0)
        self._spin_sigma_high.setSingleStep(0.1)
        sf.addRow("Sigma high:", self._spin_sigma_high)

        self._spin_max_frames = QSpinBox()
        self._spin_max_frames.setRange(10, 500)
        self._spin_max_frames.setValue(50)
        self._spin_max_frames.setToolTip(
            "Max frames loaded into RAM at once.\n"
            "100 frames @ 4K ≈ 4.7 GB — reduce if memory is tight."
        )
        sf.addRow("Max frames in memory:", self._spin_max_frames)

        lay.addWidget(grp_stack)

        # Comet mask group
        grp_mask = QGroupBox("Comet mask")
        mf = QFormLayout(grp_mask)

        self._spin_mask_radius = QSpinBox()
        self._spin_mask_radius.setRange(50, 500)
        self._spin_mask_radius.setValue(150)
        self._spin_mask_radius.setSuffix(" px")
        self._spin_mask_radius.setToolTip("Radius of the primary radial mask around nucleus")
        mf.addRow("Mask radius:", self._spin_mask_radius)

        self._spin_feather = QSpinBox()
        self._spin_feather.setRange(10, 150)
        self._spin_feather.setValue(50)
        self._spin_feather.setSuffix(" px")
        self._spin_feather.setToolTip("Gaussian feather at mask edge for smooth blending")
        mf.addRow("Feather:", self._spin_feather)

        self._chk_nebulosity = QCheckBox("Use nebulosity detection (adaptive coma shape)")
        self._chk_nebulosity.setChecked(True)
        self._chk_nebulosity.setToolTip(
            "Also detect coma extent from local background.\n"
            "Recommended for comets with extended coma."
        )
        mf.addRow("", self._chk_nebulosity)

        self._spin_neb_sigma = QDoubleSpinBox()
        self._spin_neb_sigma.setRange(1.0, 5.0)
        self._spin_neb_sigma.setValue(2.0)
        self._spin_neb_sigma.setSingleStep(0.1)
        self._spin_neb_sigma.setToolTip("Detection threshold above local background")
        mf.addRow("Nebulosity sigma:", self._spin_neb_sigma)

        lay.addWidget(grp_mask)

        # Output group
        grp_out = QGroupBox("Output")
        of = QVBoxLayout(grp_out)

        out_row = QHBoxLayout()
        self._edit_output = QLineEdit()
        self._edit_output.setPlaceholderText("Default: same as lights folder")
        out_row.addWidget(self._edit_output)
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.setFixedWidth(80)
        btn_browse_out.clicked.connect(
            lambda: _browse_folder(self, self._edit_output, "Select output folder")
        )
        out_row.addWidget(btn_browse_out)
        of.addLayout(out_row)

        self._chk_load_siril = QCheckBox("Load composite in Siril after pipeline")
        self._chk_load_siril.setChecked(True)
        of.addWidget(self._chk_load_siril)

        self._chk_crop_overlap = QCheckBox("Crop to valid overlap region (removes border artifacts)")
        self._chk_crop_overlap.setChecked(False)
        self._chk_crop_overlap.setToolTip(
            "For fast comets: removes black borders left by large shifts."
        )
        of.addWidget(self._chk_crop_overlap)

        lay.addWidget(grp_out)
        lay.addStretch()

        self._tabs.addTab(tab, "⚙️  Stack Settings")

    # ── TAB 3: RUN & TRACK ──────────────────────────────────────────────────

    def _build_tab_run(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self._btn_run = QPushButton("▶  Run Full Pipeline")
        self._btn_run.setObjectName("primary")
        self._btn_run.setFixedHeight(40)
        f = self._btn_run.font()
        f.setPointSize(11)
        f.setBold(True)
        self._btn_run.setFont(f)
        self._btn_run.clicked.connect(self._on_run)
        lay.addWidget(self._btn_run)

        # Tracking preview
        grp_track = QGroupBox("Tracking preview")
        tl = QVBoxLayout(grp_track)
        self._canvas_tracking = TrackingCanvas()
        self._canvas_tracking.setMinimumHeight(280)
        tl.addWidget(self._canvas_tracking)
        self._lbl_drift = _label("No tracking data yet.", "dim")
        tl.addWidget(self._lbl_drift)
        lay.addWidget(grp_track, 1)

        # Progress
        grp_prog = QGroupBox("Progress")
        pl = QVBoxLayout(grp_prog)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 10)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(12)
        pl.addWidget(self._progress_bar)
        self._lbl_step = _label("Idle.", "dim")
        pl.addWidget(self._lbl_step)
        lay.addWidget(grp_prog)

        self._tabs.addTab(tab, "▶  Run & Track")

    # ── TAB 4: RESULTS ──────────────────────────────────────────────────────

    def _build_tab_results(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(4, 4, 4, 4)
        self._canvas_results = ResultPreviewCanvas()
        lay.addWidget(self._canvas_results)
        self._tabs.addTab(tab, "🖼  Results")

    # ── TAB 5: LOG ──────────────────────────────────────────────────────────

    def _build_tab_log(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(6, 6, 6, 6)

        self._log_edit = QPlainTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setProperty("readOnly", True)
        self._log_edit.setPlaceholderText("Pipeline log output appears here…")
        lay.addWidget(self._log_edit, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_save_log = QPushButton("Save log…")
        btn_save_log.setObjectName("export")
        btn_save_log.setFixedWidth(120)
        btn_save_log.clicked.connect(self._on_save_log)
        btn_row.addWidget(btn_save_log)
        lay.addLayout(btn_row)

        self._tabs.addTab(tab, "📋  Log")

    # ── SLOTS ─────────────────────────────────────────────────────────────────

    def _on_nucleus_mode_changed(self, idx: int):
        is_auto = idx == 0
        self._auto_params_widget.setVisible(is_auto)
        self._manual_params_widget.setVisible(not is_auto)

    def _on_scan_frames(self):
        folder = self._edit_lights.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Please enter a valid lights folder path.")
            return
        patterns = ["*.fit", "*.fits", "*.fts"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        if not files:
            self._lbl_frame_info.setText("⚠ No FITS files found in folder.")
            self._lbl_frame_info.setObjectName("warn")
            return
        files.sort()
        # Peek at first file for shape
        try:
            d = fits.getdata(files[0]).astype(np.float32)
            shape = d.shape
            mem_bytes = len(files) * d.nbytes
            mem_gb = mem_bytes / 1e9
            info = (
                f"{len(files)} frames  |  shape: {shape}  |  "
                f"Est. memory: {mem_gb:.1f} GB"
            )
            self._lbl_frame_info.setText(info)
            self._lbl_frame_info.setObjectName("ok")
            self._log_append(f"Scanned: {info}")
            self._first_frame = d if d.ndim == 2 else (
                (0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2]).astype(np.float32)
                if d.shape[0] == 3 else d[0]
            )
        except Exception as exc:
            self._lbl_frame_info.setText(f"Error reading first frame: {exc}")

    def _on_auto_detect(self):
        if self._first_frame is None:
            folder = self._edit_lights.text().strip()
            if folder and os.path.isdir(folder):
                self._on_scan_frames()
            if self._first_frame is None:
                QMessageBox.warning(self, "No frame", "Scan frames first.")
                return
        blur = self._spin_blur.value()
        pos = detect_comet_nucleus_auto(self._first_frame, blur_sigma=blur)
        if pos is None:
            self._lbl_auto_result.setText("⚠ Not detected — try Manual mode.")
            self._lbl_auto_result.setObjectName("warn")
            self._auto_nucleus_pos = None
        else:
            cy, cx = pos
            self._auto_nucleus_pos = pos
            self._lbl_auto_result.setText(f"✓ Found at X={cx:.1f}, Y={cy:.1f}")
            self._lbl_auto_result.setObjectName("ok")
            self._lbl_nucleus_pos.setText(f"Nucleus: X={cx:.1f}, Y={cy:.1f} (auto)")
            self._canvas_picker.show_frame(self._first_frame, auto_pos=pos)
            self._log_append(f"Auto-detected nucleus at ({cx:.1f}, {cy:.1f})")

    def _on_load_first_frame(self):
        folder = self._edit_lights.text().strip()
        if not folder:
            QMessageBox.warning(self, "No folder", "Set the lights folder first.")
            return
        files = sorted(
            glob.glob(os.path.join(folder, "*.fit")) +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts"))
        )
        if not files:
            QMessageBox.warning(self, "No files", "No FITS files found in folder.")
            return
        try:
            d = fits.getdata(files[0]).astype(np.float32)
            if d.ndim == 3:
                d = (0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2]).astype(np.float32) \
                    if d.shape[0] == 3 else d[0]
            self._first_frame = d
            self._canvas_picker.show_frame(d)
            self._log_append("First frame loaded for manual nucleus selection.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load frame:\n{exc}")

    def _on_nucleus_clicked(self, cy: float, cx: float):
        self._manual_nucleus_pos = (cy, cx)
        self._lbl_nucleus_pos.setText(f"Nucleus: X={cx:.1f}, Y={cy:.1f} (manual)")
        self._log_append(f"Manual nucleus set: X={cx:.1f}, Y={cy:.1f}")

    def _on_run(self):
        lights = self._edit_lights.text().strip()
        if not lights or not os.path.isdir(lights):
            QMessageBox.warning(self, "Input error", "Please set a valid lights folder.")
            return

        # Resolve nucleus position
        mode = self._combo_nucleus_mode.currentIndex()
        if mode == 0:  # Auto
            manual_pos = None
        else:  # Manual
            manual_pos = self._manual_nucleus_pos
            if manual_pos is None:
                QMessageBox.warning(
                    self, "No nucleus",
                    "Manual mode selected but nucleus not clicked.\n"
                    "Load the first frame and click the nucleus.",
                )
                return

        rejection = (
            "sigma" if self._combo_rejection.currentIndex() == 0 else "none"
        )
        config = {
            "lights_dir": lights,
            "output_dir": self._edit_output.text().strip() or lights,
            "pre_registered": self._combo_input_type.currentIndex() == 1,
            "comet_name": self._edit_comet_name.text().strip(),
            "manual_nucleus_pos": manual_pos,
            "blur_sigma": self._spin_blur.value(),
            "search_radius": self._spin_search_radius.value(),
            "rejection": rejection,
            "sigma_low": self._spin_sigma_low.value(),
            "sigma_high": self._spin_sigma_high.value(),
            "max_frames_memory": self._spin_max_frames.value(),
            "mask_radius": self._spin_mask_radius.value(),
            "mask_feather": self._spin_feather.value(),
            "use_nebulosity_mask": self._chk_nebulosity.isChecked(),
            "nebulosity_sigma": self._spin_neb_sigma.value(),
            "load_in_siril": self._chk_load_siril.isChecked(),
            "crop_overlap": self._chk_crop_overlap.isChecked(),
        }

        self._cancel_event.clear()
        self._worker = CometWorker(config, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log_append)
        self._worker.tracking_done.connect(self._on_tracking_done)
        self._worker.stacks_ready.connect(self._on_stacks_ready)
        self._worker.finished.connect(self._on_finished)

        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._status_lbl.setText("Running pipeline…")
        self._progress_bar.setValue(0)
        self._tabs.setCurrentIndex(2)  # jump to Run tab
        self._worker.start()

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._log_append("⚠ Cancel requested…")
            self._status_lbl.setText("Cancelling…")

    def _on_progress(self, step: int, total: int, desc: str):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(step)
        self._lbl_step.setText(f"Step {step}/{total}: {desc}")
        self._status_lbl.setText(desc)

    def _on_tracking_done(self, positions: list, flags: list, first_frame: object):
        self._tracking_positions = positions
        self._tracking_flags = flags
        if first_frame is not None and len(first_frame.shape) == 2:
            self._canvas_tracking.show_tracking(first_frame, positions, flags)
        n = len(positions)
        if n > 1:
            dy = positions[-1][0] - positions[0][0]
            dx = positions[-1][1] - positions[0][1]
            drift = np.sqrt(dy ** 2 + dx ** 2)
            interp_pct = sum(flags) / max(n, 1) * 100
            drift_txt = (
                f"Total drift: {drift:.1f} px over {n} frames  |  "
                f"Interpolated: {interp_pct:.0f}%"
            )
            self._lbl_drift.setText(drift_txt)
            if interp_pct > 20:
                self._lbl_drift.setObjectName("warn")
            else:
                self._lbl_drift.setObjectName("ok")

    def _on_stacks_ready(self, stars, comet, mask, composite):
        self._canvas_results.update_all(stars, comet, mask, composite)
        self._tabs.setCurrentIndex(3)  # jump to Results tab

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        if result.get("success"):
            drift = result.get("total_drift_px", 0.0)
            comp = result.get("composite_path", "")
            self._status_lbl.setText(
                f"✓  Done  |  Drift: {drift:.1f} px  |  Saved: {comp}"
            )
            self._status_lbl.setObjectName("ok")
            self._log_append(
                f"✓ Pipeline complete. Drift: {drift:.1f} px. Output: {comp}"
            )
            # Jump to Results
            self._tabs.setCurrentIndex(3)
        else:
            err = result.get("error", "Unknown error")
            self._status_lbl.setText(f"✗  Error: {err}")
            self._status_lbl.setObjectName("err")
            self._log_append(f"✗ Pipeline failed: {err}")
            QMessageBox.critical(self, "Pipeline Error", err)

    def _on_save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save log", "", "Text files (*.txt);;All files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log_edit.toPlainText())
            self._log_append(f"Log saved to: {path}")

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _log_append(self, msg: str):
        self._log_edit.appendPlainText(msg)
        sb = self._log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
