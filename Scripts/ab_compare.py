r"""
ab_compare.py — A/B Comparator for Siril
Script #46 — Interactive side-by-side FITS image comparison tool.
Place in: C:\Users\Marcell\Desktop\Siril New Scripts\
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")

import os
import sys
import time
from datetime import datetime

import numpy as np
from astropy.io import fits as astropy_fits
from astropy.stats import sigma_clipped_stats
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QButtonGroup, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QKeySequence, QShortcut

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ── Siril Theme ────────────────────────────────────────────────────────────────

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
QPushButton#a_btn {{
    background-color: #1a2d40;
    border-color: {SIRIL_ACCENT};
    color: {SIRIL_ACCENT};
    font-weight: bold;
    text-align: center;
}}
QPushButton#a_btn:hover {{ background-color: #223850; }}
QPushButton#b_btn {{
    background-color: #2d1a2d;
    border-color: #cc44cc;
    color: #cc44cc;
    font-weight: bold;
    text-align: center;
}}
QPushButton#b_btn:hover {{ background-color: #3a2040; }}
QPushButton#export {{
    background-color: {SIRIL_BG3};
    border-color: {SIRIL_SUCCESS};
    color: {SIRIL_SUCCESS};
    text-align: center;
    font-weight: bold;
}}
QPushButton#export:hover {{ background-color: #1a3d2a; }}
QPushButton#mode_btn {{
    background-color: {SIRIL_BG3};
    border: 1px solid {SIRIL_BORDER};
    color: {SIRIL_TEXT_DIM};
    font-size: 9pt;
    padding: 4px 10px;
    text-align: center;
}}
QPushButton#mode_btn:checked {{
    background-color: {SIRIL_ACCENT2};
    border-color: {SIRIL_ACCENT};
    color: white;
    font-weight: bold;
}}
QPushButton#mode_btn:hover {{
    background-color: {SIRIL_BG3};
    border-color: {SIRIL_ACCENT};
    color: {SIRIL_TEXT};
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
QLabel#section {{ color: {SIRIL_SECTION}; font-weight: bold; padding-top: 6px; }}
QLabel#dim    {{ color: {SIRIL_TEXT_DIM}; font-size: 9pt; }}
QLabel#ok     {{ color: {SIRIL_SUCCESS}; font-weight: bold; }}
QLabel#err    {{ color: {SIRIL_ERROR};   font-weight: bold; }}
QLabel#warn   {{ color: {SIRIL_WARNING}; font-weight: bold; }}
QLabel#a_label {{
    color: {SIRIL_ACCENT};
    font-weight: bold;
    font-size: 11pt;
    padding: 2px 8px;
    background: #1a2d40;
    border: 1px solid {SIRIL_ACCENT};
    border-radius: 3px;
}}
QLabel#b_label {{
    color: #cc44cc;
    font-weight: bold;
    font-size: 11pt;
    padding: 2px 8px;
    background: #2d1a2d;
    border: 1px solid #cc44cc;
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

# ── Image Processing Functions ─────────────────────────────────────────────────

def load_fits_for_display(path: str) -> tuple:
    """Load a FITS file and prepare for display (luminance 2D)."""
    with astropy_fits.open(path) as hdul:
        data   = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()

    is_rgb = (data.ndim == 3 and
              (data.shape[0] == 3 or data.shape[2] == 3))

    if data.ndim == 3:
        if data.shape[0] == 3:
            lum = (0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2])
        elif data.shape[2] == 3:
            lum = (0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] +
                   0.114 * data[:, :, 2])
        else:
            lum = data[0]
    else:
        lum = data

    return lum.astype(np.float32), header, is_rgb, data


def auto_stretch(data: np.ndarray, lo_pct: float = 0.5,
                 hi_pct: float = 99.5) -> tuple:
    """Compute percentile-based stretch parameters."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 0.0, 1.0
    bp = float(np.percentile(finite, lo_pct))
    wp = float(np.percentile(finite, hi_pct))
    if wp <= bp:
        wp = bp + 1e-6
    return bp, wp


def apply_stretch(data: np.ndarray, black_point: float,
                  white_point: float) -> np.ndarray:
    """Apply linear stretch, return float32 in [0, 1]."""
    return np.clip(
        (data.astype(np.float64) - black_point) /
        (white_point - black_point + 1e-10),
        0, 1
    ).astype(np.float32)


def compute_difference(a: np.ndarray, b: np.ndarray,
                       scale_sigma: float = 5.0) -> np.ndarray:
    """Compute signed difference (A-B) scaled to ±scale_sigma×noise."""
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    diff = a.astype(np.float64) - b.astype(np.float64)
    _, _, std = sigma_clipped_stats(diff, sigma=3.0)
    scale = max(scale_sigma * std, 1e-10)
    return np.clip(diff / scale, -1, 1).astype(np.float32)


def downsample(arr: np.ndarray, max_px: int = 1400) -> np.ndarray:
    """Downsample for display while preserving aspect ratio."""
    h, w   = arr.shape
    factor = max(1, max(h, w) // max_px)
    return arr[::factor, ::factor]


def compute_image_stats(data: np.ndarray, label: str) -> dict:
    """Compute per-image display stats."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return {}
    _, med, std = sigma_clipped_stats(finite, sigma=3.0)
    return {
        "label":   label,
        "min":     round(float(finite.min()), 4),
        "max":     round(float(finite.max()), 4),
        "median":  round(float(med), 4),
        "std":     round(float(std), 4),
        "snr_est": round(float(med) / max(float(std), 1e-10), 1),
        "shape":   f"{data.shape[1]}×{data.shape[0]}",
    }


def compute_comparison_stats(data_a: np.ndarray,
                              data_b: np.ndarray) -> dict:
    """Compute A vs B comparison stats."""
    if data_a.shape != data_b.shape:
        h = min(data_a.shape[0], data_b.shape[0])
        w = min(data_a.shape[1], data_b.shape[1])
        data_a = data_a[:h, :w]
        data_b = data_b[:h, :w]

    diff = data_a.astype(np.float64) - data_b.astype(np.float64)
    _, _, std_a = sigma_clipped_stats(data_a, sigma=3.0)
    _, _, std_b = sigma_clipped_stats(data_b, sigma=3.0)

    rms_diff  = float(np.sqrt(np.mean(diff ** 2)))
    mean_diff = float(np.mean(diff))
    snr_a     = float(np.median(data_a)) / max(std_a, 1e-10)
    snr_b     = float(np.median(data_b)) / max(std_b, 1e-10)
    snr_gain  = snr_b / max(snr_a, 1e-10)

    return {
        "rms_diff":         round(rms_diff, 4),
        "mean_diff":        round(mean_diff, 4),
        "snr_a":            round(snr_a, 2),
        "snr_b":            round(snr_b, 2),
        "snr_gain_pct":     round((snr_gain - 1.0) * 100, 1),
        "noise_a":          round(std_a, 4),
        "noise_b":          round(std_b, 4),
        "noise_change_pct": round((std_b / max(std_a, 1e-10) - 1.0) * 100, 1),
    }


# ── CompareCanvas ──────────────────────────────────────────────────────────────

class CompareCanvas(FigureCanvasQTAgg):
    """Main comparison canvas supporting four display modes."""

    COLORS = {
        "A": "#4a9eff",
        "B": "#cc44cc",
    }

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(11, 6), facecolor="#1e2128")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        # Data
        self.data_a      = None
        self.data_b      = None
        self.disp_a      = None
        self.disp_b      = None
        self.diff_ab     = None
        self.bp_a = self.wp_a = None
        self.bp_b = self.wp_b = None

        # Mode
        self.mode         = "side_by_side"
        self.wipe_frac    = 0.5
        self.blink_state  = "A"
        self.linked_stretch = True

        # Axes
        self.ax_a = None
        self.ax_b = None

        # Zoom/pan
        self._pan_start   = None
        self._xlim_a      = None
        self._ylim_a      = None
        self._zoom_factor = 1.2
        self._syncing     = False

        # Wipe drag
        self._wipe_dragging = False

        # Connect matplotlib events
        self.mpl_connect("scroll_event",         self._on_scroll)
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    # ── Data ──────────────────────────────────────────────────────────────────

    def set_images(self, data_a, data_b,
                   bp_a=None, wp_a=None,
                   bp_b=None, wp_b=None):
        self.data_a = data_a
        self.data_b = data_b

        if bp_a is None or wp_a is None:
            bp_a, wp_a = auto_stretch(data_a)
        if bp_b is None or wp_b is None:
            if self.linked_stretch:
                bp_b, wp_b = bp_a, wp_a
            else:
                bp_b, wp_b = auto_stretch(data_b)

        self.bp_a, self.wp_a = bp_a, wp_a
        self.bp_b, self.wp_b = bp_b, wp_b

        self.disp_a  = downsample(apply_stretch(data_a, bp_a, wp_a))
        self.disp_b  = downsample(apply_stretch(data_b, bp_b, wp_b))
        self.diff_ab = compute_difference(data_a, data_b)
        self._redraw()

    def update_stretch(self, bp_a=None, wp_a=None, bp_b=None, wp_b=None):
        if self.data_a is None:
            return
        if bp_a is not None:
            self.bp_a, self.wp_a = bp_a, wp_a
        if bp_b is not None:
            self.bp_b, self.wp_b = bp_b, wp_b
        self.disp_a = downsample(apply_stretch(self.data_a, self.bp_a, self.wp_a))
        self.disp_b = downsample(apply_stretch(self.data_b, self.bp_b, self.wp_b))
        self._redraw()

    def set_mode(self, mode: str):
        self.mode = mode
        self._redraw()

    def set_wipe_fraction(self, frac: float):
        self.wipe_frac = float(np.clip(frac, 0.02, 0.98))
        if self.mode == "wipe":
            self._redraw()

    def set_blink_state(self, state: str):
        self.blink_state = state
        if self.mode == "blink":
            self._redraw()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw(self):
        self.fig.clear()
        self.ax_a = None
        self.ax_b = None
        if self.disp_a is None or self.disp_b is None:
            self.draw()
            return

        if self.mode == "side_by_side":
            self._draw_side_by_side()
        elif self.mode == "wipe":
            self._draw_wipe()
        elif self.mode == "difference":
            self._draw_difference()
        elif self.mode == "blink":
            self._draw_blink()

        self.fig.tight_layout(pad=0.3)
        self.draw()

    def _style_ax(self, ax, label: str, color: str):
        ax.set_facecolor("#1e2128")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color("#3a4055")
        ax.text(0.02, 0.97, label,
                transform=ax.transAxes,
                color=color, fontsize=14, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#1e2128", alpha=0.75,
                          edgecolor=color))

    def _draw_side_by_side(self):
        self.ax_a = self.fig.add_subplot(1, 2, 1)
        self.ax_b = self.fig.add_subplot(1, 2, 2)

        for ax, disp, label, color in [
            (self.ax_a, self.disp_a, "A", self.COLORS["A"]),
            (self.ax_b, self.disp_b, "B", self.COLORS["B"]),
        ]:
            ax.imshow(disp, cmap="gray", origin="lower",
                      aspect="equal", interpolation="nearest")
            self._style_ax(ax, label, color)

        def _sync_a_to_b(*args):
            if self._syncing or self.ax_b is None:
                return
            self._syncing = True
            self.ax_b.set_xlim(self.ax_a.get_xlim())
            self.ax_b.set_ylim(self.ax_a.get_ylim())
            self._syncing = False
            self.fig.canvas.draw_idle()

        def _sync_b_to_a(*args):
            if self._syncing or self.ax_a is None:
                return
            self._syncing = True
            self.ax_a.set_xlim(self.ax_b.get_xlim())
            self.ax_a.set_ylim(self.ax_b.get_ylim())
            self._syncing = False
            self.fig.canvas.draw_idle()

        self.ax_a.callbacks.connect("xlim_changed", _sync_a_to_b)
        self.ax_a.callbacks.connect("ylim_changed", _sync_a_to_b)
        self.ax_b.callbacks.connect("xlim_changed", _sync_b_to_a)
        self.ax_b.callbacks.connect("ylim_changed", _sync_b_to_a)

    def _draw_wipe(self):
        ax = self.fig.add_subplot(1, 1, 1)
        self.ax_a = ax

        h_a, w_a = self.disp_a.shape
        h_b, w_b = self.disp_b.shape
        h = min(h_a, h_b)
        w = min(w_a, w_b)

        composite = np.zeros((h, w), dtype=np.float32)
        split_px  = int(w * self.wipe_frac)
        split_px  = max(1, min(split_px, w - 1))
        composite[:, :split_px] = self.disp_a[:h, :split_px]
        composite[:, split_px:] = self.disp_b[:h, split_px:]

        ax.imshow(composite, cmap="gray", origin="lower",
                  aspect="equal", interpolation="nearest")
        ax.set_facecolor("#1e2128")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color("#3a4055")

        ax.axvline(split_px, color="#ffffff",
                   linewidth=1.5, linestyle="--", alpha=0.85)

        a_frac = split_px * 0.5 / w if w > 0 else 0.25
        b_frac = (split_px + (w - split_px) * 0.5) / w if w > 0 else 0.75

        ax.text(a_frac, 0.97, "A",
                transform=ax.transAxes,
                color=self.COLORS["A"], fontsize=14, fontweight="bold",
                va="top", ha="center",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#1e2128", alpha=0.75,
                          edgecolor=self.COLORS["A"]))
        ax.text(b_frac, 0.97, "B",
                transform=ax.transAxes,
                color=self.COLORS["B"], fontsize=14, fontweight="bold",
                va="top", ha="center",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#1e2128", alpha=0.75,
                          edgecolor=self.COLORS["B"]))

    def _draw_difference(self):
        ax = self.fig.add_subplot(1, 1, 1)
        self.ax_a = ax

        diff_ds = downsample(self.diff_ab)
        im = ax.imshow(diff_ds, cmap="RdBu_r", origin="lower",
                       aspect="equal", interpolation="nearest",
                       vmin=-1, vmax=1)
        cb = self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Difference (±5σ)", color=SIRIL_TEXT_DIM, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=SIRIL_TEXT_DIM)
        for lbl in cb.ax.get_yticklabels():
            lbl.set_color(SIRIL_TEXT_DIM)
            lbl.set_fontsize(7)

        ax.set_facecolor("#1e2128")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color("#3a4055")

        rms  = float(np.sqrt(np.mean(self.diff_ab ** 2)))
        mn   = float(np.mean(self.diff_ab))
        mx   = float(np.max(np.abs(self.diff_ab)))
        ax.set_title(
            f"A − B   |   RMS={rms:.3f}σ   mean={mn:+.3f}σ   "
            f"max|diff|={mx:.2f}σ",
            color="#5ba3ff", fontsize=9, pad=4)

    def _draw_blink(self):
        ax = self.fig.add_subplot(1, 1, 1)
        self.ax_a = ax

        if self.blink_state == "A":
            disp, label, color = self.disp_a, "A", self.COLORS["A"]
        else:
            disp, label, color = self.disp_b, "B", self.COLORS["B"]

        ax.imshow(disp, cmap="gray", origin="lower",
                  aspect="equal", interpolation="nearest")
        self._style_ax(ax, label, color)

    # ── Zoom / Pan ────────────────────────────────────────────────────────────

    def _get_primary_ax(self):
        return self.ax_a

    def _on_scroll(self, event):
        ax = self._get_primary_ax()
        if ax is None or event.inaxes not in (self.ax_a, self.ax_b):
            return
        factor = 1.0 / self._zoom_factor \
                 if event.button == "up" else self._zoom_factor
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        # Apply to clicked axis (sync propagates to the other)
        clicked_ax = event.inaxes
        xlim = clicked_ax.get_xlim()
        ylim = clicked_ax.get_ylim()
        clicked_ax.set_xlim([xdata + (x - xdata) * factor for x in xlim])
        clicked_ax.set_ylim([ydata + (y - ydata) * factor for y in ylim])
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.button == 2 or event.button == 3:
            # Check for wipe drag
            if self.mode == "wipe" and self.ax_a is not None and \
               event.inaxes == self.ax_a and event.xdata is not None:
                h_a, w_a = self.disp_a.shape
                h_b, w_b = self.disp_b.shape
                w = min(w_a, w_b)
                split_px = int(w * self.wipe_frac)
                if abs(event.xdata - split_px) < 15:
                    self._wipe_dragging = True
                    return

            self._pan_start = (event.xdata, event.ydata)
            clicked_ax = event.inaxes
            if clicked_ax:
                self._xlim_a = clicked_ax.get_xlim()
                self._ylim_a = clicked_ax.get_ylim()
                self._pan_ax = clicked_ax
        elif event.button == 1 and self.mode == "wipe":
            if self.ax_a is not None and event.inaxes == self.ax_a \
               and event.xdata is not None:
                h_a, w_a = self.disp_a.shape
                h_b, w_b = self.disp_b.shape
                w = min(w_a, w_b)
                split_px = int(w * self.wipe_frac)
                if abs(event.xdata - split_px) < 15:
                    self._wipe_dragging = True

    def _on_motion(self, event):
        # Wipe drag
        if self._wipe_dragging and event.xdata is not None \
           and self.ax_a is not None and self.disp_a is not None:
            h_a, w_a = self.disp_a.shape
            h_b, w_b = self.disp_b.shape
            w = min(w_a, w_b)
            if w > 0:
                new_frac = float(np.clip(event.xdata / w, 0.02, 0.98))
                self.wipe_frac = new_frac
                self._redraw()
                # Notify parent via wipe_fraction_changed if set
                if hasattr(self, "_wipe_frac_callback") and \
                   self._wipe_frac_callback:
                    self._wipe_frac_callback(new_frac)
            return

        if self._pan_start is None:
            return
        clicked_ax = getattr(self, "_pan_ax", None)
        if clicked_ax is None or event.xdata is None:
            return
        dx = event.xdata - self._pan_start[0]
        dy = event.ydata - self._pan_start[1]
        clicked_ax.set_xlim([x - dx for x in self._xlim_a])
        clicked_ax.set_ylim([y - dy for y in self._ylim_a])
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._pan_start    = None
        self._wipe_dragging = False

    def reset_zoom(self):
        for ax in [self.ax_a, self.ax_b]:
            if ax is not None:
                ax.autoscale()
        self.fig.canvas.draw_idle()

    def zoom_by(self, factor: float):
        """Zoom in/out centered on image centre."""
        ax = self._get_primary_ax()
        if ax is None:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cx   = (xlim[0] + xlim[1]) / 2
        cy   = (ylim[0] + ylim[1]) / 2
        ax.set_xlim([cx + (x - cx) * factor for x in xlim])
        ax.set_ylim([cy + (y - cy) * factor for y in ylim])
        self.fig.canvas.draw_idle()


# ── Statistics Tab Widget ──────────────────────────────────────────────────────

class StatsWidget(QWidget):
    """Three-column stats display: A | Δ | B."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Column A
        col_a = QWidget()
        col_a.setStyleSheet(f"background: #1a2535; border: 1px solid {SIRIL_ACCENT}; "
                            f"border-radius: 4px;")
        la = QVBoxLayout(col_a)
        la.setContentsMargins(8, 6, 8, 6)
        lbl_a = QLabel("A")
        lbl_a.setObjectName("a_label")
        lbl_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        la.addWidget(lbl_a)
        self._form_a = QFormLayout()
        self._form_a.setSpacing(3)
        la.addLayout(self._form_a)
        la.addStretch()
        layout.addWidget(col_a)

        # Column Δ
        col_d = QWidget()
        col_d.setStyleSheet(f"background: {SIRIL_BG2}; border: 1px solid {SIRIL_BORDER}; "
                            f"border-radius: 4px;")
        ld = QVBoxLayout(col_d)
        ld.setContentsMargins(8, 6, 8, 6)
        lbl_d = QLabel("A vs B")
        lbl_d.setObjectName("section")
        lbl_d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ld.addWidget(lbl_d)
        self._form_d = QFormLayout()
        self._form_d.setSpacing(3)
        ld.addLayout(self._form_d)
        note = QLabel("Positive SNR gain = B is better")
        note.setObjectName("dim")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setWordWrap(True)
        ld.addWidget(note)
        ld.addStretch()
        layout.addWidget(col_d)

        # Column B
        col_b = QWidget()
        col_b.setStyleSheet("background: #251a35; border: 1px solid #cc44cc; "
                            "border-radius: 4px;")
        lb = QVBoxLayout(col_b)
        lb.setContentsMargins(8, 6, 8, 6)
        lbl_b = QLabel("B")
        lbl_b.setObjectName("b_label")
        lbl_b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lb.addWidget(lbl_b)
        self._form_b = QFormLayout()
        self._form_b.setSpacing(3)
        lb.addLayout(self._form_b)
        lb.addStretch()
        layout.addWidget(col_b)

        self._a_fields = {}
        self._b_fields = {}
        self._d_fields = {}
        self._init_forms()

    def _make_val_label(self) -> QLabel:
        lbl = QLabel("—")
        lbl.setStyleSheet(f"color: {SIRIL_TEXT}; font-size: 9pt;")
        return lbl

    def _init_forms(self):
        for key, text in [("file", "File"), ("shape", "Size"),
                          ("median", "Median"), ("std", "Noise"),
                          ("snr_est", "SNR (est.)")]:
            lbl_a = self._make_val_label()
            lbl_b = self._make_val_label()
            row_lbl = QLabel(text + ":")
            row_lbl.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 9pt;")
            self._form_a.addRow(row_lbl, lbl_a)
            row_lbl2 = QLabel(text + ":")
            row_lbl2.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 9pt;")
            self._form_b.addRow(row_lbl2, lbl_b)
            self._a_fields[key] = lbl_a
            self._b_fields[key] = lbl_b

        for key, text in [("rms_diff", "RMS diff"), ("mean_diff", "Mean diff"),
                          ("snr_gain_pct", "SNR gain"),
                          ("noise_change_pct", "Noise Δ")]:
            lbl_d = self._make_val_label()
            row_lbl = QLabel(text + ":")
            row_lbl.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 9pt;")
            self._form_d.addRow(row_lbl, lbl_d)
            self._d_fields[key] = lbl_d

    def update_stats(self, stats_a: dict, stats_b: dict,
                     cmp: dict, path_a: str, path_b: str):
        def _set(fields, key, val, color=None):
            lbl = fields.get(key)
            if lbl is None:
                return
            lbl.setText(str(val))
            if color:
                lbl.setStyleSheet(f"color: {color}; font-size: 9pt; "
                                  f"font-weight: bold;")
            else:
                lbl.setStyleSheet(f"color: {SIRIL_TEXT}; font-size: 9pt;")

        _set(self._a_fields, "file",   os.path.basename(path_a) if path_a else "—")
        _set(self._a_fields, "shape",  stats_a.get("shape", "—"))
        _set(self._a_fields, "median", stats_a.get("median", "—"))
        _set(self._a_fields, "std",    stats_a.get("std", "—"))
        _set(self._a_fields, "snr_est", stats_a.get("snr_est", "—"))

        _set(self._b_fields, "file",   os.path.basename(path_b) if path_b else "—")
        _set(self._b_fields, "shape",  stats_b.get("shape", "—"))
        _set(self._b_fields, "median", stats_b.get("median", "—"))
        _set(self._b_fields, "std",    stats_b.get("std", "—"))
        _set(self._b_fields, "snr_est", stats_b.get("snr_est", "—"))

        _set(self._d_fields, "rms_diff",   cmp.get("rms_diff", "—"))
        _set(self._d_fields, "mean_diff",  f"{cmp.get('mean_diff', 0):+.4f}"
                                            if cmp else "—")

        snr_pct = cmp.get("snr_gain_pct", None)
        if snr_pct is not None:
            c = SIRIL_SUCCESS if snr_pct >= 0 else SIRIL_ERROR
            _set(self._d_fields, "snr_gain_pct",
                 f"{snr_pct:+.1f}%", color=c)
        else:
            _set(self._d_fields, "snr_gain_pct", "—")

        noise_pct = cmp.get("noise_change_pct", None)
        if noise_pct is not None:
            c = SIRIL_SUCCESS if noise_pct <= 0 else SIRIL_ERROR
            _set(self._d_fields, "noise_change_pct",
                 f"{noise_pct:+.1f}%", color=c)
        else:
            _set(self._d_fields, "noise_change_pct", "—")


# ── Stretch Tab Widget ─────────────────────────────────────────────────────────

class StretchWidget(QWidget):
    """Dynamic stretch controls based on active mode and stretch preset."""

    stretch_changed = pyqtSignal(float, float, float, float)  # bp_a, wp_a, bp_b, wp_b
    wipe_changed    = pyqtSignal(float)
    blink_speed_changed = pyqtSignal(int)   # ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(6)

        self._mode         = "side_by_side"
        self._stretch_mode = "linked"
        self._data_a       = None
        self._data_b       = None

        self._placeholder = QLabel("Load images to see stretch controls.")
        self._placeholder.setObjectName("dim")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._placeholder)

    def set_context(self, mode: str, stretch_mode: str,
                    data_a=None, data_b=None):
        self._mode         = mode
        self._stretch_mode = stretch_mode
        self._data_a       = data_a
        self._data_b       = data_b
        self._rebuild()

    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _rebuild(self):
        self._clear()

        if self._data_a is None:
            self._placeholder = QLabel("Load images to see controls.")
            self._placeholder.setObjectName("dim")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._layout.addWidget(self._placeholder)
            return

        row = QHBoxLayout()
        row.setSpacing(20)

        if self._stretch_mode == "manual":
            for which in ("A", "B"):
                grp = QGroupBox(f"Image {which} — Manual Stretch")
                gl  = QFormLayout(grp)
                gl.setSpacing(4)

                bp_slider = QSlider(Qt.Orientation.Horizontal)
                bp_slider.setRange(0, 1000)
                bp_slider.setValue(0)
                bp_val    = QLabel("0.0000")
                bp_val.setObjectName("dim")
                wp_slider = QSlider(Qt.Orientation.Horizontal)
                wp_slider.setRange(0, 1000)
                wp_slider.setValue(1000)
                wp_val    = QLabel("1.0000")
                wp_val.setObjectName("dim")

                data = self._data_a if which == "A" else self._data_b
                if data is not None:
                    finite = data[np.isfinite(data)]
                    dmin   = float(finite.min()) if len(finite) else 0.0
                    dmax   = float(finite.max()) if len(finite) else 1.0
                    drange = dmax - dmin
                else:
                    dmin, dmax, drange = 0.0, 1.0, 1.0

                def _make_cb(slider, val_lbl, dmin_, drange_, other_slider,
                              other_val_lbl, other_dmin, other_drange,
                              is_bp, wh):
                    def cb(v):
                        actual = dmin_ + (v / 1000.0) * drange_
                        val_lbl.setText(f"{actual:.4f}")
                        # Emit stretch for both A and B
                        if wh == "A":
                            bp_a_v = dmin_ + (self._bp_a_sl.value() / 1000) * drange_
                            wp_a_v = dmin_ + (self._wp_a_sl.value() / 1000) * drange_
                            bp_b_v = other_dmin + (self._bp_b_sl.value() / 1000) * other_drange
                            wp_b_v = other_dmin + (self._wp_b_sl.value() / 1000) * other_drange
                        else:
                            bp_a_v = (self._data_a[np.isfinite(self._data_a)].min()
                                      if self._data_a is not None else 0.0)
                            wp_a_v = (self._data_a[np.isfinite(self._data_a)].max()
                                      if self._data_a is not None else 1.0)
                            bp_a_v = dmin_ + (self._bp_a_sl.value() / 1000) * (
                                (self._data_a[np.isfinite(self._data_a)].max()
                                 - self._data_a[np.isfinite(self._data_a)].min())
                                if self._data_a is not None else 1.0
                            )
                            bp_b_v = other_dmin + (self._bp_b_sl.value() / 1000) * other_drange
                            wp_b_v = other_dmin + (self._wp_b_sl.value() / 1000) * other_drange
                            bp_a_v = (self._data_a_dmin +
                                      (self._bp_a_sl.value() / 1000) * self._data_a_range
                                      if hasattr(self, "_data_a_dmin") else 0.0)
                            wp_a_v = (self._data_a_dmin +
                                      (self._wp_a_sl.value() / 1000) * self._data_a_range
                                      if hasattr(self, "_data_a_dmin") else 1.0)
                            bp_b_v = actual if is_bp else (
                                other_dmin + (other_slider.value() / 1000) * other_drange)
                            wp_b_v = actual if not is_bp else (
                                other_dmin + (other_slider.value() / 1000) * other_drange)
                        self.stretch_changed.emit(bp_a_v, wp_a_v, bp_b_v, wp_b_v)
                    return cb

                bp_h = QHBoxLayout()
                bp_h.addWidget(bp_slider)
                bp_h.addWidget(bp_val)
                wp_h = QHBoxLayout()
                wp_h.addWidget(wp_slider)
                wp_h.addWidget(wp_val)
                gl.addRow("Black point:", bp_h)  # type: ignore
                gl.addRow("White point:", wp_h)  # type: ignore

                if which == "A":
                    self._bp_a_sl  = bp_slider
                    self._wp_a_sl  = wp_slider
                    self._bp_a_val = bp_val
                    self._wp_a_val = wp_val
                    self._data_a_dmin  = dmin
                    self._data_a_range = drange
                else:
                    self._bp_b_sl  = bp_slider
                    self._wp_b_sl  = wp_slider
                    self._bp_b_val = bp_val
                    self._wp_b_val = wp_val
                    self._data_b_dmin  = dmin
                    self._data_b_range = drange

                row.addWidget(grp)

            # Wire up sliders now that both exist
            def _emit_stretch():
                bp_a = self._data_a_dmin + (self._bp_a_sl.value() / 1000) * self._data_a_range
                wp_a = self._data_a_dmin + (self._wp_a_sl.value() / 1000) * self._data_a_range
                bp_b = self._data_b_dmin + (self._bp_b_sl.value() / 1000) * self._data_b_range
                wp_b = self._data_b_dmin + (self._wp_b_sl.value() / 1000) * self._data_b_range
                wp_a = max(wp_a, bp_a + 1e-6)
                wp_b = max(wp_b, bp_b + 1e-6)
                self._bp_a_val.setText(f"{bp_a:.4f}")
                self._wp_a_val.setText(f"{wp_a:.4f}")
                self._bp_b_val.setText(f"{bp_b:.4f}")
                self._wp_b_val.setText(f"{wp_b:.4f}")
                self.stretch_changed.emit(bp_a, wp_a, bp_b, wp_b)

            self._bp_a_sl.valueChanged.connect(_emit_stretch)
            self._wp_a_sl.valueChanged.connect(_emit_stretch)
            self._bp_b_sl.valueChanged.connect(_emit_stretch)
            self._wp_b_sl.valueChanged.connect(_emit_stretch)

            self._layout.addLayout(row)

        elif self._mode == "wipe":
            lbl = QLabel("Wipe Position")
            lbl.setObjectName("section")
            self._layout.addWidget(lbl)
            h = QHBoxLayout()
            self._wipe_slider = QSlider(Qt.Orientation.Horizontal)
            self._wipe_slider.setRange(0, 100)
            self._wipe_slider.setValue(50)
            self._wipe_val = QLabel("50%")
            self._wipe_val.setObjectName("dim")
            self._wipe_val.setFixedWidth(40)
            h.addWidget(self._wipe_slider)
            h.addWidget(self._wipe_val)
            self._layout.addLayout(h)
            self._wipe_slider.valueChanged.connect(self._on_wipe_slider)
            self._layout.addStretch()

        elif self._mode == "blink":
            lbl = QLabel("Blink Speed")
            lbl.setObjectName("section")
            self._layout.addWidget(lbl)
            h = QHBoxLayout()
            self._blink_slider = QSlider(Qt.Orientation.Horizontal)
            self._blink_slider.setRange(1, 30)
            self._blink_slider.setValue(5)   # 0.5s
            self._blink_val = QLabel("0.5s per frame")
            self._blink_val.setObjectName("dim")
            h.addWidget(self._blink_slider)
            h.addWidget(self._blink_val)
            self._layout.addLayout(h)
            self._blink_slider.valueChanged.connect(self._on_blink_slider)
            self._layout.addStretch()
        else:
            note = QLabel("Stretch mode: " + self._stretch_mode.replace("_", " ").title() +
                          "\nAdjust via the Stretch dropdown in the toolbar.")
            note.setObjectName("dim")
            note.setAlignment(Qt.AlignmentFlag.AlignCenter)
            note.setWordWrap(True)
            self._layout.addWidget(note)
            self._layout.addStretch()

    def _on_wipe_slider(self, v: int):
        pct = v / 100.0
        self._wipe_val.setText(f"{v}%")
        self.wipe_changed.emit(pct)

    def _on_blink_slider(self, v: int):
        ms = int(v * 100)
        self._blink_val.setText(f"{ms / 1000:.1f}s per frame")
        self.blink_speed_changed.emit(ms)

    def set_wipe_value(self, frac: float):
        """Update wipe slider from canvas drag."""
        if hasattr(self, "_wipe_slider"):
            self._wipe_slider.blockSignals(True)
            self._wipe_slider.setValue(int(frac * 100))
            self._wipe_slider.blockSignals(False)
            self._wipe_val.setText(f"{int(frac * 100)}%")


# ── Main Window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍  A/B Comparator  —  Siril")
        self.setMinimumSize(1100, 700)
        self.resize(1300, 820)

        # State
        self.data_a = self.data_b = None
        self.header_a = self.header_b = None
        self.is_rgb_a = self.is_rgb_b = False
        self.raw_a   = self.raw_b    = None
        self.path_a  = self.path_b   = None
        self._current_mode    = "side_by_side"
        self._stretch_mode    = "linked"
        self._blink_state     = "A"
        self._blink_interval  = 500

        # Blink timer
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_tick)

        self._build_ui()
        self._setup_shortcuts()
        self.log("A/B Comparator ready.  Load image A and image B to begin.")

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        root.addWidget(self._build_title_bar())

        sep = QFrame(); sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        root.addLayout(self._build_top_bar())

        sep2 = QFrame(); sep2.setObjectName("separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep2)

        # Splitter: canvas top, tabs bottom
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._canvas = CompareCanvas(self)
        self._canvas._wipe_frac_callback = self._on_canvas_wipe_drag
        self._splitter.addWidget(self._canvas)
        self._splitter.addWidget(self._build_bottom_tabs())
        self._splitter.setSizes([500, 200])
        root.addWidget(self._splitter, stretch=1)

        self._status_bar = QLabel("No images loaded")
        self._status_bar.setObjectName("dim")
        self._status_bar.setContentsMargins(4, 2, 4, 2)
        root.addWidget(self._status_bar)

        # Size warning
        self._size_warn = QLabel(
            "⚠  Images have different sizes — wipe and difference modes "
            "crop to smallest common area")
        self._size_warn.setObjectName("warn")
        self._size_warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._size_warn.hide()
        root.addWidget(self._size_warn)

    def _build_title_bar(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 2, 4, 2)
        title = QLabel("🔍  A/B Comparator  —  Siril")
        title.setStyleSheet(f"color: {SIRIL_ACCENT}; font-size: 12pt; "
                            f"font-weight: bold;")
        h.addWidget(title)
        h.addStretch()
        ver = QLabel("v1.0")
        ver.setObjectName("dim")
        h.addWidget(ver)
        return w

    def _build_top_bar(self):
        bar = QHBoxLayout()
        bar.setSpacing(6)

        # A button + filename
        btn_a = QPushButton("[ A ]  Browse…")
        btn_a.setObjectName("a_btn")
        btn_a.setFixedWidth(120)
        btn_a.clicked.connect(lambda: self._browse_image("A"))
        bar.addWidget(btn_a)

        self._lbl_a = QLabel("no file")
        self._lbl_a.setObjectName("dim")
        self._lbl_a.setFixedWidth(180)
        self._lbl_a.setToolTip("")
        bar.addWidget(self._lbl_a)

        # B button + filename
        btn_b = QPushButton("[ B ]  Browse…")
        btn_b.setObjectName("b_btn")
        btn_b.setFixedWidth(120)
        btn_b.clicked.connect(lambda: self._browse_image("B"))
        bar.addWidget(btn_b)

        self._lbl_b = QLabel("no file")
        self._lbl_b.setObjectName("dim")
        self._lbl_b.setFixedWidth(180)
        bar.addWidget(self._lbl_b)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setObjectName("separator"); sep.setFixedWidth(2)
        bar.addWidget(sep)

        # Mode buttons
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        modes = [
            ("Side-by-side", "side_by_side"),
            ("Wipe",         "wipe"),
            ("Difference",   "difference"),
            ("Blink",        "blink"),
        ]
        self._mode_btns = {}
        for label, key in modes:
            btn = QPushButton(label)
            btn.setObjectName("mode_btn")
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            if key == "side_by_side":
                btn.setChecked(True)
            self._mode_group.addButton(btn)
            btn.clicked.connect(lambda _=False, k=key: self._set_mode(k))
            bar.addWidget(btn)
            self._mode_btns[key] = btn

        # Separator
        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setObjectName("separator"); sep2.setFixedWidth(2)
        bar.addWidget(sep2)

        # Stretch combo
        self._stretch_combo = QComboBox()
        self._stretch_combo.addItems([
            "Linked auto",
            "Independent auto",
            "Match A → B",
            "Manual",
        ])
        self._stretch_combo.setFixedWidth(148)
        self._stretch_combo.currentIndexChanged.connect(self._on_stretch_mode_changed)
        bar.addWidget(self._stretch_combo)

        # Reset zoom
        btn_reset = QPushButton("⊞ Reset zoom")
        btn_reset.setObjectName("primary")
        btn_reset.setFixedHeight(28)
        btn_reset.clicked.connect(lambda: self._canvas.reset_zoom())
        bar.addWidget(btn_reset)

        bar.addStretch()
        return bar

    def _build_bottom_tabs(self):
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Stats tab
        self._stats_widget = StatsWidget()
        self._tabs.addTab(self._stats_widget, "📊 Statistics")

        # Stretch tab
        self._stretch_widget = StretchWidget()
        self._stretch_widget.stretch_changed.connect(self._on_manual_stretch)
        self._stretch_widget.wipe_changed.connect(self._canvas.set_wipe_fraction)
        self._stretch_widget.blink_speed_changed.connect(self._set_blink_speed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._stretch_widget)
        self._tabs.addTab(scroll, "🎛  Stretch")

        # Log tab
        self._log_edit = QPlainTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFont(QFont("Courier New", 9))
        self._log_edit.setPlaceholderText("Script log…")
        self._tabs.addTab(self._log_edit, "📋 Log")

        return self._tabs

    # ── Shortcuts ─────────────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        shortcuts = {
            "A":            lambda: self._set_mode("side_by_side"),
            "W":            lambda: self._set_mode("wipe"),
            "D":            lambda: self._set_mode("difference"),
            "B":            lambda: self._set_mode("blink"),
            "R":            self._canvas.reset_zoom,
            "S":            self._swap_images,
            "+":            lambda: self._canvas.zoom_by(0.8),
            "-":            lambda: self._canvas.zoom_by(1.25),
            "Ctrl+O":       lambda: self._browse_image("A"),
            "Ctrl+Shift+O": lambda: self._browse_image("B"),
            "Left":         lambda: self._wipe_step(-0.05),
            "Right":        lambda: self._wipe_step(+0.05),
        }
        for seq, fn in shortcuts.items():
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(fn)

    # ── Event Handlers ────────────────────────────────────────────────────────

    def _browse_image(self, which: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Open Image {which}", "",
            "FITS files (*.fits *.fit *.fts);;All files (*.*)"
        )
        if path:
            self._load_image(which, path)

    def _load_image(self, which: str, path: str):
        self.log(f"Loading image {which}: {path}")
        t0 = time.time()
        try:
            data, header, is_rgb, raw = load_fits_for_display(path)
        except Exception as e:
            self.log(f"ERROR loading {which}: {e}")
            QMessageBox.critical(self, f"Load error (image {which})", str(e))
            return

        dt = time.time() - t0
        name = os.path.basename(path)
        shape_str = f"{data.shape[1]}×{data.shape[0]}"
        self.log(f"  {which} loaded: {name}  {shape_str}  "
                 f"({'RGB' if is_rgb else 'mono'})  {dt:.2f}s")

        if which == "A":
            self.data_a = data; self.header_a = header
            self.is_rgb_a = is_rgb; self.raw_a = raw; self.path_a = path
            self._lbl_a.setText(name[:26] + "…" if len(name) > 26 else name)
            self._lbl_a.setToolTip(path)
        else:
            self.data_b = data; self.header_b = header
            self.is_rgb_b = is_rgb; self.raw_b = raw; self.path_b = path
            self._lbl_b.setText(name[:26] + "…" if len(name) > 26 else name)
            self._lbl_b.setToolTip(path)

        if self.data_a is not None and self.data_b is not None:
            self._update_canvas()
            self._update_stats()
            self._check_size_mismatch()

        self._update_status()

    def _update_canvas(self):
        if self.data_a is None or self.data_b is None:
            return

        stretch = self._stretch_combo.currentText()
        bp_a, wp_a = auto_stretch(self.data_a)
        bp_b, wp_b = auto_stretch(self.data_b)

        if stretch == "Linked auto":
            bp_b, wp_b = bp_a, wp_a
            self._canvas.linked_stretch = True
        elif stretch == "Independent auto":
            self._canvas.linked_stretch = False
        elif stretch == "Match A → B":
            bp_b, wp_b = bp_a, wp_a
            self._canvas.linked_stretch = False
        elif stretch == "Manual":
            self._canvas.linked_stretch = False

        self._canvas.set_images(self.data_a, self.data_b,
                                bp_a, wp_a, bp_b, wp_b)
        self._stretch_widget.set_context(
            self._current_mode, stretch.lower().replace(" ", "_"),
            self.data_a, self.data_b
        )

    def _update_stats(self):
        if self.data_a is None or self.data_b is None:
            return
        try:
            stats_a = compute_image_stats(self.data_a, "A")
            stats_b = compute_image_stats(self.data_b, "B")
            cmp     = compute_comparison_stats(self.data_a, self.data_b)
            self._stats_widget.update_stats(
                stats_a, stats_b, cmp, self.path_a, self.path_b)
        except Exception as e:
            self.log(f"Stats error: {e}")

    def _check_size_mismatch(self):
        if self.data_a is not None and self.data_b is not None:
            if self.data_a.shape != self.data_b.shape:
                self._size_warn.show()
            else:
                self._size_warn.hide()

    def _set_mode(self, mode: str):
        self._current_mode = mode
        # Update button checked state
        for k, btn in self._mode_btns.items():
            btn.setChecked(k == mode)

        # Blink timer
        if mode == "blink":
            self._blink_timer.start(self._blink_interval)
        else:
            self._blink_timer.stop()

        self._canvas.set_mode(mode)
        self._update_stretch_tab()
        self._update_status()

    def _on_stretch_mode_changed(self, _idx: int):
        self._update_canvas()

    def _on_manual_stretch(self, bp_a, wp_a, bp_b, wp_b):
        if self.data_a is None or self.data_b is None:
            return
        self._canvas.update_stretch(bp_a, wp_a, bp_b, wp_b)

    def _blink_tick(self):
        self._blink_state = "B" if self._blink_state == "A" else "A"
        self._canvas.set_blink_state(self._blink_state)

    def _set_blink_speed(self, ms: int):
        self._blink_interval = ms
        if self._blink_timer.isActive():
            self._blink_timer.setInterval(ms)

    def _on_canvas_wipe_drag(self, frac: float):
        """Called when user drags wipe line on canvas."""
        self._stretch_widget.set_wipe_value(frac)

    def _wipe_step(self, delta: float):
        if self._current_mode == "wipe":
            new_frac = float(np.clip(self._canvas.wipe_frac + delta, 0.02, 0.98))
            self._canvas.set_wipe_fraction(new_frac)
            self._stretch_widget.set_wipe_value(new_frac)

    def _swap_images(self):
        if self.data_a is None or self.data_b is None:
            self.log("Cannot swap: both images must be loaded.")
            return
        self.data_a,   self.data_b   = self.data_b,   self.data_a
        self.header_a, self.header_b = self.header_b, self.header_a
        self.is_rgb_a, self.is_rgb_b = self.is_rgb_b, self.is_rgb_a
        self.raw_a,    self.raw_b    = self.raw_b,    self.raw_a
        self.path_a,   self.path_b   = self.path_b,   self.path_a

        a_name = os.path.basename(self.path_a) if self.path_a else "—"
        b_name = os.path.basename(self.path_b) if self.path_b else "—"
        self._lbl_a.setText(a_name[:26] + "…" if len(a_name) > 26 else a_name)
        self._lbl_b.setText(b_name[:26] + "…" if len(b_name) > 26 else b_name)
        self.log(f"Swapped: A={a_name}  B={b_name}")

        self._update_canvas()
        self._update_stats()
        self._update_status()

    def _update_stretch_tab(self):
        stretch_text = self._stretch_combo.currentText()
        self._stretch_widget.set_context(
            self._current_mode,
            stretch_text.lower().replace(" ", "_"),
            self.data_a, self.data_b
        )

    def _update_status(self):
        a = os.path.basename(self.path_a) if self.path_a else "none"
        b = os.path.basename(self.path_b) if self.path_b else "none"
        mode_map = {
            "side_by_side": "Side-by-side",
            "wipe":         "Wipe",
            "difference":   "Difference",
            "blink":        "Blink",
        }
        mode = mode_map.get(self._current_mode, self._current_mode)
        self._status_bar.setText(
            f"A: {a}  |  B: {b}  |  Mode: {mode}  |  "
            f"Stretch: {self._stretch_combo.currentText()}")

    def log(self, msg: str):
        ts  = datetime.now().strftime("%H:%M:%S")
        self._log_edit.appendPlainText(f"[{ts}] {msg}")

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left and self._current_mode == "wipe":
            self._wipe_step(-0.05)
        elif key == Qt.Key.Key_Right and self._current_mode == "wipe":
            self._wipe_step(+0.05)
        else:
            super().keyPressEvent(event)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
