# local_norm.py  --  Script #5: Local Normalization
# Siril astrophotography pipeline tool
# Place in: C:/Users/Marcell/Desktop/Siril New Scripts/
#
# PixInsight-equivalent Local Normalization for multi-session DSO imaging.
# Normalizes brightness/contrast locally per tile so multi-night stacks
# blend seamlessly without session banding.

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")

# ─────────────────────────── imports ────────────────────────────────────────
import os, sys, glob, json, threading, shutil
from datetime import datetime

import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from astropy.io import fits as astropy_fits

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QFrame, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ─────────────────────────── theme ──────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════════
# CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════

def local_normalize_frame(frame: np.ndarray,
                           reference: np.ndarray,
                           tile_size: int = 64,
                           overlap: float = 0.5,
                           use_mad: bool = True) -> np.ndarray:
    """
    Tile-based local normalization.  Each overlapping tile is independently
    matched (background level + scale) to the corresponding reference tile,
    then blended with a Gaussian weight kernel so boundaries are invisible.
    """
    assert frame.shape == reference.shape, \
        f"Frame {frame.shape} != reference {reference.shape}"

    h, w   = frame.shape
    step   = max(1, int(tile_size * (1.0 - overlap)))
    output = np.zeros((h, w), dtype=np.float64)
    wts    = np.zeros((h, w), dtype=np.float64)

    weight_1d = np.exp(
        -0.5 * ((np.arange(tile_size) - tile_size / 2.0) /
                 (tile_size / 3.0)) ** 2
    )
    weight_2d = np.outer(weight_1d, weight_1d)

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            f_tile = frame    [y:y+tile_size, x:x+tile_size].astype(np.float64)
            r_tile = reference[y:y+tile_size, x:x+tile_size].astype(np.float64)

            f_med = float(np.median(f_tile))
            r_med = float(np.median(r_tile))

            if use_mad:
                f_scale = float(np.median(np.abs(f_tile - f_med))) * 1.4826
                r_scale = float(np.median(np.abs(r_tile - r_med))) * 1.4826
            else:
                f_scale = float(np.std(f_tile))
                r_scale = float(np.std(r_tile))

            if f_scale < 1e-10:
                norm_tile = f_tile - f_med + r_med
            else:
                norm_tile = (f_tile - f_med) / f_scale * r_scale + r_med

            output[y:y+tile_size, x:x+tile_size] += norm_tile * weight_2d
            wts   [y:y+tile_size, x:x+tile_size] += weight_2d

    uncovered = wts < 1e-10
    if np.any(uncovered):
        f_med_g = float(np.median(frame))
        r_med_g = float(np.median(reference))
        f_mad_g = float(np.median(np.abs(frame - f_med_g))) * 1.4826
        r_mad_g = float(np.median(np.abs(reference - r_med_g))) * 1.4826
        if f_mad_g > 1e-10:
            global_norm = (frame - f_med_g) / f_mad_g * r_mad_g + r_med_g
        else:
            global_norm = frame - f_med_g + r_med_g
        output[uncovered] = global_norm[uncovered]
        wts[uncovered]    = 1.0

    return (output / wts).astype(np.float32)


def build_normalization_map(frame: np.ndarray,
                             reference: np.ndarray,
                             tile_size: int = 64) -> tuple:
    """Return (scale_map, offset_map) showing per-pixel correction applied."""
    h, w = frame.shape
    step = max(1, tile_size // 2)
    th   = max(1, (h - tile_size) // step + 1)
    tw   = max(1, (w - tile_size) // step + 1)

    scale_grid  = np.ones( (th, tw), dtype=np.float32)
    offset_grid = np.zeros((th, tw), dtype=np.float32)

    for ti, y in enumerate(range(0, h - tile_size + 1, step)):
        for tj, x in enumerate(range(0, w - tile_size + 1, step)):
            if ti >= th or tj >= tw:
                continue
            f_tile = frame    [y:y+tile_size, x:x+tile_size].astype(float)
            r_tile = reference[y:y+tile_size, x:x+tile_size].astype(float)
            f_med  = float(np.median(f_tile))
            r_med  = float(np.median(r_tile))
            f_mad  = float(np.median(np.abs(f_tile - f_med))) * 1.4826
            r_mad  = float(np.median(np.abs(r_tile - r_med))) * 1.4826
            scale_grid [ti, tj] = r_mad / max(f_mad, 1e-10)
            offset_grid[ti, tj] = r_med - f_med * (r_mad / max(f_mad, 1e-10))

    zoom_y = h / th
    zoom_x = w / tw
    scale_map  = gaussian_filter(
        zoom(scale_grid,  (zoom_y, zoom_x), order=1)[:h, :w], sigma=3)
    offset_map = gaussian_filter(
        zoom(offset_grid, (zoom_y, zoom_x), order=1)[:h, :w], sigma=3)

    return scale_map, offset_map


def find_best_reference(fits_files: list, log_callback=None) -> str:
    """Select frame with highest SNR proxy (median / MAD)."""
    best_path = fits_files[0]
    best_snr  = -1.0
    for path in fits_files:
        try:
            data = astropy_fits.getdata(path).astype(float)
            if data.ndim == 3:
                data = data[0]
            med = float(np.median(data))
            mad = float(np.median(np.abs(data - med))) * 1.4826
            snr = med / max(mad, 1e-10)
            if snr > best_snr:
                best_snr, best_path = snr, path
        except Exception:
            continue
    if log_callback:
        log_callback(f"Auto reference: {os.path.basename(best_path)} "
                     f"(SNR proxy: {best_snr:.1f})")
    return best_path


def stack_normalized_in_siril(output_dir, stack_method, sigma,
                               output_name, log_callback=None):
    """Optionally stack normalized files using Siril after normalization."""
    siril = s.SirilInterface()
    siril.connect()
    try:
        siril.cmd("cd", output_dir)
        siril.cmd("convert", "light", "-out=norm_lights_")
        siril.cmd("register", "norm_lights_")
        m = {"winsorized": "w", "linear": "l", "sigma": "l"}.get(
            stack_method, "w")
        siril.cmd("stack", "norm_lights_",
                  "rej", m, str(sigma), str(sigma),
                  f"-output={output_name}")
        if log_callback:
            log_callback(f"Stacked → {output_name}.fit in {output_dir}")
    except Exception as e:
        if log_callback:
            log_callback(f"Siril stack error: {e}")
    finally:
        siril.disconnect()


# ═══════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════

class NormWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    map_ready     = pyqtSignal(object, object)
    preview_ready = pyqtSignal(object, object)
    finished      = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg       = self.config
            files     = cfg["fits_files"]
            ref_path  = cfg.get("reference_path")
            out_dir   = cfg["output_dir"]
            tile_size = cfg.get("tile_size", 64)
            overlap   = cfg.get("overlap", 0.5)
            use_mad   = cfg.get("use_mad", True)
            n         = len(files)

            self.log_line.emit(
                f"Local normalization: {n} files | "
                f"tile={tile_size}px | overlap={int(overlap*100)}%")

            # ── load reference ───────────────────────────────────────────
            if ref_path and os.path.isfile(ref_path):
                ref_data = astropy_fits.getdata(ref_path).astype(np.float32)
                self.log_line.emit(f"Reference: {os.path.basename(ref_path)}")
            else:
                ref_data = astropy_fits.getdata(files[0]).astype(np.float32)
                self.log_line.emit(
                    f"Reference: {os.path.basename(files[0])} (first file)")

            if ref_data.ndim == 3:
                ref_data = ref_data[0]

            os.makedirs(out_dir, exist_ok=True)
            output_paths = []

            for i, fits_path in enumerate(files):
                if self._cancel.is_set():
                    self._abort(); return

                self.progress.emit(i + 1, n,
                    f"Normalizing {i+1}/{n}: {os.path.basename(fits_path)}")
                self.log_line.emit(
                    f"[{i+1}/{n}] {os.path.basename(fits_path)}")

                try:
                    with astropy_fits.open(fits_path) as hdul:
                        data   = hdul[0].data.astype(np.float32)
                        header = hdul[0].header.copy()

                    is_rgb = (data.ndim == 3)

                    if is_rgb:
                        if data.shape[0] == 3:
                            channels = [data[c] for c in range(3)]
                            layout   = "CHW"
                        else:
                            channels = [data[:,:,c] for c in range(3)]
                            layout   = "HWC"

                        norm_channels = []
                        for ch in channels:
                            nc = local_normalize_frame(
                                ch, ref_data, tile_size, overlap, use_mad)
                            norm_channels.append(nc)

                        if layout == "CHW":
                            result = np.stack(norm_channels, axis=0)
                        else:
                            result = np.stack(norm_channels, axis=2)
                        frame_for_map = (0.299 * channels[0] +
                                         0.587 * channels[1] +
                                         0.114 * channels[2])
                    else:
                        result        = local_normalize_frame(
                            data, ref_data, tile_size, overlap, use_mad)
                        frame_for_map = data

                    # ── emit preview for first non-reference frame ───────
                    if i == 0 and fits_path != ref_path:
                        s_map, o_map = build_normalization_map(
                            frame_for_map, ref_data, tile_size)
                        self.map_ready.emit(s_map, o_map)
                        after_frame = (result[0] if is_rgb and result.ndim == 3
                                       else result)
                        self.preview_ready.emit(frame_for_map, after_frame)

                    # ── save ─────────────────────────────────────────────
                    out_name = os.path.basename(fits_path)
                    out_path = os.path.join(out_dir, out_name)
                    hdu = astropy_fits.PrimaryHDU(data=result, header=header)
                    hdu.header["HISTORY"] = "Local normalization — local_norm.py"
                    hdu.header["LOCNORM"] = (
                        f"tile={tile_size} overlap={overlap:.2f}")
                    hdu.writeto(out_path, overwrite=True)
                    output_paths.append(out_path)
                    self.log_line.emit(f"  → saved: {out_name}")

                except Exception as e:
                    import traceback
                    self.log_line.emit(
                        f"  ERROR on {os.path.basename(fits_path)}: {e}")
                    self.log_line.emit(traceback.format_exc())

            # ── optional Siril stack ─────────────────────────────────────
            if cfg.get("siril_stack") and output_paths:
                self.log_line.emit("Stacking in Siril…")
                stack_normalized_in_siril(
                    out_dir,
                    cfg.get("stack_method", "winsorized"),
                    cfg.get("stack_sigma", 3.0),
                    cfg.get("stack_output", "master_normalized"),
                    log_callback=lambda m: self.log_line.emit(m),
                )

            self.finished.emit({
                "success":      True,
                "n_processed":  len(output_paths),
                "n_total":      n,
                "output_dir":   out_dir,
                "output_paths": output_paths,
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"FATAL ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _abort(self):
        self.log_line.emit("Cancelled by user.")
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()


# ═══════════════════════════════════════════════════════════════════════════
# CANVASES
# ═══════════════════════════════════════════════════════════════════════════

class PreviewCanvas(FigureCanvasQTAgg):
    """Before / After comparison panel."""

    def __init__(self, parent=None):
        self.fig    = Figure(figsize=(9, 4), facecolor="#1e2128")
        self.ax_bef = self.fig.add_subplot(1, 2, 1)
        self.ax_aft = self.fig.add_subplot(1, 2, 2)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def _style_axes(self):
        for ax, title, color in [
            (self.ax_bef, "Before normalization", SIRIL_WARNING),
            (self.ax_aft, "After normalization",  SIRIL_SUCCESS),
        ]:
            ax.set_facecolor(SIRIL_BG)
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color(SIRIL_BORDER)

    def _disp(self, arr):
        p_lo = np.percentile(arr, 0.5)
        p_hi = np.percentile(arr, 99.5)
        return np.clip((arr.astype(float) - p_lo) /
                       (p_hi - p_lo + 1e-10), 0, 1)

    def _ds(self, arr, max_px=500):
        h, w   = arr.shape[:2]
        factor = max(1, max(h, w) // max_px)
        return arr[::factor, ::factor]

    def show_before(self, before):
        self.ax_bef.clear()
        self.ax_bef.imshow(self._ds(self._disp(before)),
                           cmap="gray", origin="lower",
                           aspect="equal", interpolation="nearest")
        self.ax_bef.set_title("Before normalization",
                               color=SIRIL_WARNING, fontsize=9)
        for sp in self.ax_bef.spines.values():
            sp.set_color(SIRIL_BORDER)
        self.ax_bef.tick_params(left=False, bottom=False,
                                 labelleft=False, labelbottom=False)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_after(self, after):
        self.ax_aft.clear()
        self.ax_aft.imshow(self._ds(self._disp(after)),
                           cmap="gray", origin="lower",
                           aspect="equal", interpolation="nearest")
        self.ax_aft.set_title("After normalization",
                               color=SIRIL_SUCCESS, fontsize=9)
        for sp in self.ax_aft.spines.values():
            sp.set_color(SIRIL_BORDER)
        self.ax_aft.tick_params(left=False, bottom=False,
                                 labelleft=False, labelbottom=False)
        self.fig.tight_layout(pad=0.3)
        self.draw()


class MapCanvas(FigureCanvasQTAgg):
    """Scale / Offset correction map panel with live statistics."""

    def __init__(self, parent=None):
        self.fig      = Figure(figsize=(9, 4), facecolor="#1e2128")
        self.ax_scale = self.fig.add_subplot(1, 2, 1)
        self.ax_off   = self.fig.add_subplot(1, 2, 2)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._last_scale  = None
        self._last_offset = None

    def _style_axes(self):
        for ax, title in [
            (self.ax_scale, "Scale correction map"),
            (self.ax_off,   "Offset correction map"),
        ]:
            ax.set_facecolor(SIRIL_BG)
            ax.set_title(title, color=SIRIL_SECTION, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color(SIRIL_BORDER)

    def show_maps(self, scale_map, offset_map):
        self._last_scale  = scale_map
        self._last_offset = offset_map

        h, w   = scale_map.shape
        factor = max(1, max(h, w) // 400)
        s_ds   = scale_map [::factor, ::factor]
        o_ds   = offset_map[::factor, ::factor]

        s_range = max(abs(s_ds.max() - 1.0),
                      abs(s_ds.min() - 1.0), 0.01)
        im1 = self.ax_scale.imshow(
            s_ds, cmap="RdBu_r", origin="lower",
            aspect="equal", interpolation="nearest",
            vmin=1.0 - s_range, vmax=1.0 + s_range)
        self.fig.colorbar(im1, ax=self.ax_scale, fraction=0.046)
        self.ax_scale.set_title("Scale correction (1.0 = none)",
                                 color=SIRIL_SECTION, fontsize=8)

        o_range = max(abs(o_ds).max(), 0.001)
        im2 = self.ax_off.imshow(
            o_ds, cmap="RdBu_r", origin="lower",
            aspect="equal", interpolation="nearest",
            vmin=-o_range, vmax=o_range)
        self.fig.colorbar(im2, ax=self.ax_off, fraction=0.046)
        self.ax_off.set_title("Offset correction (0 = none)",
                               color=SIRIL_SECTION, fontsize=8)

        for ax in [self.ax_scale, self.ax_off]:
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color(SIRIL_BORDER)

        self.fig.tight_layout(pad=0.3)
        self.draw()

    def get_stats(self):
        """Return dict of map statistics, or None if no maps loaded."""
        if self._last_scale is None:
            return None
        s = self._last_scale
        o = self._last_offset
        return {
            "scale_min":    float(s.min()),
            "scale_max":    float(s.max()),
            "offset_range": float(np.abs(o).max()),
            "max_scale_pct": float(abs(s.mean() - 1.0) * 100),
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚖  Local Normalization  —  Siril")
        self.resize(1280, 780)
        self._fits_files  = []
        self._cancel_evt  = threading.Event()
        self._worker      = None
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(4)
        vbox.setContentsMargins(6, 6, 6, 4)

        # title bar
        title_bar = QWidget()
        tl = QHBoxLayout(title_bar)
        tl.setContentsMargins(4, 2, 4, 2)
        lbl_title = QLabel("⚖  Local Normalization  —  Siril")
        lbl_title.setObjectName("section")
        lbl_ver   = QLabel("v1.0  |  PixInsight Local Normalization equivalent")
        lbl_ver.setObjectName("dim")
        tl.addWidget(lbl_title)
        tl.addStretch()
        tl.addWidget(lbl_ver)
        vbox.addWidget(title_bar)

        sep = QFrame(); sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        vbox.addWidget(sep)

        # main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        vbox.addWidget(splitter, stretch=1)

        # ── left panel ───────────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(380)
        left_inner  = QWidget()
        left_vbox   = QVBoxLayout(left_inner)
        left_vbox.setSpacing(8)
        left_vbox.setContentsMargins(6, 6, 6, 6)
        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)

        # Input sequence
        gb_in = QGroupBox("Input sequence")
        f_in  = QFormLayout(gb_in)
        self.edit_folder  = QLineEdit()
        self.edit_folder.setPlaceholderText("Folder of FITS files…")
        btn_browse_folder = QPushButton("Browse…")
        btn_browse_folder.clicked.connect(self._browse_folder)
        row_folder = QHBoxLayout()
        row_folder.addWidget(self.edit_folder)
        row_folder.addWidget(btn_browse_folder)
        f_in.addRow("Folder:", row_folder)

        btn_scan = QPushButton("🔍  Scan")
        btn_scan.clicked.connect(self._scan_folder)
        f_in.addRow("", btn_scan)

        self.lbl_file_info = QLabel("No files scanned")
        self.lbl_file_info.setObjectName("dim")
        f_in.addRow("", self.lbl_file_info)

        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Filename", "DATE-OBS"])
        self.combo_sort.currentIndexChanged.connect(self._apply_sort)
        f_in.addRow("Sort by:", self.combo_sort)
        left_vbox.addWidget(gb_in)

        # Reference frame
        gb_ref = QGroupBox("Reference frame")
        f_ref  = QFormLayout(gb_ref)
        self.combo_ref = QComboBox()
        self.combo_ref.addItems([
            "First file in folder",
            "Best frame (highest SNR, auto)",
            "Choose specific file",
        ])
        self.combo_ref.currentIndexChanged.connect(self._ref_mode_changed)
        f_ref.addRow("Mode:", self.combo_ref)

        self.ref_specific_widget = QWidget()
        rs_layout = QHBoxLayout(self.ref_specific_widget)
        rs_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_ref_file = QLineEdit()
        self.edit_ref_file.setPlaceholderText("Reference FITS path…")
        btn_ref_browse = QPushButton("Browse…")
        btn_ref_browse.clicked.connect(self._browse_ref)
        rs_layout.addWidget(self.edit_ref_file)
        rs_layout.addWidget(btn_ref_browse)
        self.ref_specific_widget.setVisible(False)
        f_ref.addRow("File:", self.ref_specific_widget)

        lbl_ref_dim = QLabel("Reference = target statistics for all other frames")
        lbl_ref_dim.setObjectName("dim")
        lbl_ref_dim.setWordWrap(True)
        f_ref.addRow("", lbl_ref_dim)
        left_vbox.addWidget(gb_ref)

        # Normalization parameters
        gb_params = QGroupBox("Normalization parameters")
        f_params  = QFormLayout(gb_params)

        self.combo_tile = QComboBox()
        self.combo_tile.addItems([
            "32px (fine — complex gradients)",
            "64px (recommended)",
            "128px (coarse — session differences)",
            "256px (large gradient only)",
        ])
        self.combo_tile.setCurrentIndex(1)
        f_params.addRow("Tile size:", self.combo_tile)
        lbl_tile_dim = QLabel("Smaller = more local, better for complex gradients")
        lbl_tile_dim.setObjectName("dim")
        lbl_tile_dim.setWordWrap(True)
        f_params.addRow("", lbl_tile_dim)

        overlap_row = QHBoxLayout()
        self.slider_overlap = QSlider(Qt.Orientation.Horizontal)
        self.slider_overlap.setRange(0, 9)
        self.slider_overlap.setValue(5)
        self.slider_overlap.setTickPosition(
            QSlider.TickPosition.TicksBelow)
        self.lbl_overlap_val = QLabel("50%")
        self.lbl_overlap_val.setFixedWidth(36)
        self.slider_overlap.valueChanged.connect(
            lambda v: self.lbl_overlap_val.setText(f"{v*10}%"))
        overlap_row.addWidget(self.slider_overlap)
        overlap_row.addWidget(self.lbl_overlap_val)
        f_params.addRow("Tile overlap:", overlap_row)
        lbl_ov_dim = QLabel("Higher = smoother blending, slower processing")
        lbl_ov_dim.setObjectName("dim")
        lbl_ov_dim.setWordWrap(True)
        f_params.addRow("", lbl_ov_dim)

        self.combo_estimator = QComboBox()
        self.combo_estimator.addItems([
            "MAD (robust, recommended)",
            "Std deviation",
        ])
        f_params.addRow("Scale estimator:", self.combo_estimator)
        lbl_mad_dim = QLabel("MAD ignores stars and cosmic rays")
        lbl_mad_dim.setObjectName("dim")
        f_params.addRow("", lbl_mad_dim)
        left_vbox.addWidget(gb_params)

        # Output
        gb_out = QGroupBox("Output")
        f_out  = QFormLayout(gb_out)
        out_row = QHBoxLayout()
        self.edit_out_folder = QLineEdit()
        self.edit_out_folder.setPlaceholderText("Output folder…")
        btn_out_browse = QPushButton("Browse…")
        btn_out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(self.edit_out_folder)
        out_row.addWidget(btn_out_browse)
        f_out.addRow("Output folder:", out_row)

        self.chk_overwrite = QCheckBox("Overwrite if output files already exist")
        self.chk_overwrite.setChecked(True)
        f_out.addRow("", self.chk_overwrite)

        self.chk_copy_ref = QCheckBox("Copy reference unchanged to output folder")
        self.chk_copy_ref.setChecked(True)
        f_out.addRow("", self.chk_copy_ref)
        left_vbox.addWidget(gb_out)

        # Siril integration
        gb_siril = QGroupBox("Siril integration (after normalization)")
        f_siril  = QFormLayout(gb_siril)
        self.chk_stack = QCheckBox("Stack normalized files in Siril")
        self.chk_stack.setChecked(False)
        self.chk_stack.toggled.connect(self._siril_stack_toggled)
        f_siril.addRow("", self.chk_stack)

        self.siril_stack_opts = QWidget()
        ss_layout = QFormLayout(self.siril_stack_opts)
        self.combo_stack_method = QComboBox()
        self.combo_stack_method.addItems(["winsorized", "linear", "sigma"])
        ss_layout.addRow("Stack method:", self.combo_stack_method)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, 10.0)
        self.spin_sigma.setValue(3.0)
        self.spin_sigma.setSingleStep(0.5)
        ss_layout.addRow("Sigma:", self.spin_sigma)
        self.edit_stack_name = QLineEdit("master_normalized")
        ss_layout.addRow("Output name:", self.edit_stack_name)
        self.siril_stack_opts.setVisible(False)
        f_siril.addRow("", self.siril_stack_opts)
        left_vbox.addWidget(gb_siril)

        left_vbox.addStretch()

        # ── right panel ──────────────────────────────────────────────────
        right_widget = QWidget()
        right_vbox   = QVBoxLayout(right_widget)
        right_vbox.setContentsMargins(4, 0, 4, 0)
        splitter.addWidget(right_widget)
        splitter.setSizes([380, 900])

        self.tabs = QTabWidget()
        right_vbox.addWidget(self.tabs, stretch=1)

        # Tab 1: Before / After
        tab_preview = QWidget()
        tpv = QVBoxLayout(tab_preview)
        self.preview_canvas = PreviewCanvas()
        tpv.addWidget(self.preview_canvas)
        lbl_prev_dim = QLabel(
            "Showing first normalized frame vs reference (luminance)")
        lbl_prev_dim.setObjectName("dim")
        tpv.addWidget(lbl_prev_dim)
        self.tabs.addTab(tab_preview, "🖼  Before / After")

        # Tab 2: Correction Maps
        tab_maps = QWidget()
        tmv = QVBoxLayout(tab_maps)
        self.map_canvas = MapCanvas()
        tmv.addWidget(self.map_canvas)
        lbl_map_dim = QLabel(
            "Uniform maps = consistent data quality   "
            "Patchy maps = variable transparency or gradients")
        lbl_map_dim.setObjectName("dim")
        tmv.addWidget(lbl_map_dim)

        gb_map_stats = QGroupBox("Map statistics")
        map_stats_layout = QHBoxLayout(gb_map_stats)
        self.lbl_scale_range  = QLabel("Scale range: —")
        self.lbl_offset_range = QLabel("Offset range: —")
        self.lbl_max_corr     = QLabel("Max correction: —")
        for l in [self.lbl_scale_range, self.lbl_offset_range,
                  self.lbl_max_corr]:
            l.setObjectName("dim")
            map_stats_layout.addWidget(l)
        tmv.addWidget(gb_map_stats)
        self.tabs.addTab(tab_maps, "🗺  Correction Maps")

        # Tab 3: Log
        tab_log = QWidget()
        tlv = QVBoxLayout(tab_log)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Courier New", 9))
        tlv.addWidget(self.log_edit)
        self.tabs.addTab(tab_log, "📋  Log")

        # ── bottom row ───────────────────────────────────────────────────
        bottom = QWidget()
        bottom_h = QHBoxLayout(bottom)
        bottom_h.setContentsMargins(4, 4, 4, 4)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)

        self.btn_run = QPushButton("▶  Normalize")
        self.btn_run.setObjectName("primary")
        self.btn_run.setFixedHeight(34)
        self.btn_run.clicked.connect(self._run)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setFixedHeight(34)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)

        bottom_h.addWidget(self.progress_bar, stretch=1)
        bottom_h.addWidget(self.btn_run)
        bottom_h.addWidget(self.btn_cancel)
        vbox.addWidget(bottom)

        # status bar
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setObjectName("dim")
        self.lbl_status.setContentsMargins(6, 2, 6, 2)
        vbox.addWidget(self.lbl_status)

    # ── slots ────────────────────────────────────────────────────────────

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder")
        if d:
            self.edit_folder.setText(d)
            # auto-fill output
            if not self.edit_out_folder.text():
                self.edit_out_folder.setText(d.rstrip("/\\") + "_normalized")
            self._scan_folder()

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.edit_out_folder.setText(d)

    def _browse_ref(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select reference FITS", filter="FITS (*.fits *.fit *.fts)")
        if f:
            self.edit_ref_file.setText(f)

    def _scan_folder(self):
        folder = self.edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self.lbl_file_info.setText("⚠  Folder not found")
            self._fits_files = []
            return
        patterns = ["*.fits", "*.fit", "*.fts",
                    "*.FITS", "*.FIT", "*.FTS"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        files = sorted(set(files))
        self._fits_files = files

        if not files:
            self.lbl_file_info.setText("No FITS files found")
            return

        # get image size from first file
        try:
            d = astropy_fits.getdata(files[0])
            shape_str = f"{d.shape[-1]}×{d.shape[-2]}"
        except Exception:
            shape_str = "unknown size"

        self.lbl_file_info.setText(
            f"✓  {len(files)} FITS files  |  {shape_str}")
        self._apply_sort()
        self._log(f"Scanned: {len(files)} files in {folder}")

    def _apply_sort(self):
        if not self._fits_files:
            return
        mode = self.combo_sort.currentText()
        if mode == "DATE-OBS":
            def key(p):
                try:
                    hdr = astropy_fits.getheader(p)
                    return hdr.get("DATE-OBS", "")
                except Exception:
                    return ""
            self._fits_files.sort(key=key)
        else:
            self._fits_files.sort(key=os.path.basename)

    def _ref_mode_changed(self, idx):
        self.ref_specific_widget.setVisible(idx == 2)

    def _siril_stack_toggled(self, checked):
        self.siril_stack_opts.setVisible(checked)

    def _log(self, msg: str):
        ts  = datetime.now().strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{ts}] {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet(f"color: {color};")

    # ── run / cancel ─────────────────────────────────────────────────────

    def _run(self):
        if not self._fits_files:
            QMessageBox.warning(self, "No files",
                                "Scan a folder with FITS files first.")
            return

        out_dir = self.edit_out_folder.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output folder",
                                "Please specify an output folder.")
            return

        # resolve reference
        ref_mode = self.combo_ref.currentIndex()
        ref_path = None

        if ref_mode == 0:
            ref_path = self._fits_files[0]
        elif ref_mode == 1:
            self._log("Finding best reference (SNR)…")
            ref_path = find_best_reference(
                self._fits_files, log_callback=self._log)
        else:
            ref_path = self.edit_ref_file.text().strip()
            if not os.path.isfile(ref_path):
                QMessageBox.warning(self, "Reference not found",
                                    "Specified reference file does not exist.")
                return

        # tile size
        tile_map = {0: 32, 1: 64, 2: 128, 3: 256}
        tile_size = tile_map.get(self.combo_tile.currentIndex(), 64)

        overlap  = self.slider_overlap.value() * 0.10
        use_mad  = (self.combo_estimator.currentIndex() == 0)

        # copy reference to output
        if self.chk_copy_ref.isChecked() and ref_path:
            os.makedirs(out_dir, exist_ok=True)
            ref_out = os.path.join(out_dir, os.path.basename(ref_path))
            if not os.path.exists(ref_out):
                try:
                    shutil.copy2(ref_path, ref_out)
                    self._log(f"Reference copied → {os.path.basename(ref_out)}")
                except Exception as e:
                    self._log(f"WARNING: could not copy reference: {e}")

        config = {
            "fits_files":   self._fits_files,
            "reference_path": ref_path,
            "output_dir":   out_dir,
            "tile_size":    tile_size,
            "overlap":      overlap,
            "use_mad":      use_mad,
            "siril_stack":  self.chk_stack.isChecked(),
            "stack_method": self.combo_stack_method.currentText(),
            "stack_sigma":  self.spin_sigma.value(),
            "stack_output": self.edit_stack_name.text().strip()
                             or "master_normalized",
        }

        self._cancel_evt.clear()
        self._worker = NormWorker(config, self._cancel_evt)
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.map_ready.connect(self._on_map_ready)
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.finished.connect(self._on_finished)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self._fits_files))
        self.progress_bar.setValue(0)
        self._set_status("Normalizing…", SIRIL_ACCENT)

        self.tabs.setCurrentIndex(2)  # show log while running
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)
        self._set_status("Cancelling…", SIRIL_WARNING)

    # ── worker signals ───────────────────────────────────────────────────

    def _on_progress(self, cur, total, msg):
        self.progress_bar.setValue(cur)
        self._set_status(msg, SIRIL_ACCENT)

    def _on_map_ready(self, scale_map, offset_map):
        self.map_canvas.show_maps(scale_map, offset_map)
        stats = self.map_canvas.get_stats()
        if stats:
            self.lbl_scale_range.setText(
                f"Scale range: {stats['scale_min']:.3f} – {stats['scale_max']:.3f}")
            self.lbl_offset_range.setText(
                f"Offset range: ±{stats['offset_range']:.4f}")
            self.lbl_max_corr.setText(
                f"Max correction: {stats['max_scale_pct']:.1f}% (scale), "
                f"{stats['offset_range']:.4f} (offset)")

    def _on_preview_ready(self, before, after):
        self.preview_canvas.show_before(before)
        self.preview_canvas.show_after(after)

    def _on_finished(self, result):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

        if result.get("success"):
            n    = result["n_processed"]
            odir = result["output_dir"]
            msg  = f"✓  {n} files normalized  |  Output: {odir}"
            self._set_status(msg, SIRIL_SUCCESS)
            self._log(msg)
        else:
            err = result.get("error", "Unknown error")
            self._set_status(f"✗  {err}", SIRIL_ERROR)
            self._log(f"FAILED: {err}")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
