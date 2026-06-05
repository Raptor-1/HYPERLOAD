r"""
image_processing_suite.py  —  Script Merger: B5 Image Processing Suite
=======================================================================
Merged launcher for 5 standalone image processing tools:
    〜  Wavelet Stretch        (wavelet_stretch.py)
    ⚖  Local Normalization    (local_norm.py)
    🌫  Dark Channel Dehaze   (dark_channel.py)
    🌈  Narrowband Palette    (nb_palette.py)
    🎭  Mask Builder          (mask_builder.py)

Architecture: Tool Selector (B5 spec)
    - Shared FITS input bar — set once, all tools pre-filled
    - Each tool is a self-contained tab with full UI
    - Chain mode: output of one tool auto-fills the next tool's input
    - No sequential pipeline dependency — each tool is independently runnable

Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:  Siril → Scripts → image_processing_suite
"""

from pathlib import Path
import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

import os
import sys
import json
import glob
import threading
import shutil
from datetime import datetime

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# ── Child scripts location ────────────────────────────────────────────────────
# Put the 7 suite launchers in their OWN folder (e.g. "Siril Suites\").
# Put the 20+ small child scripts in a SEPARATE folder (e.g. "Siril Scripts\").
# Set CHILDREN_DIR to wherever the small scripts live.
# The launchers never have to be in the same folder as the children.
CHILDREN_DIR = str(Path(__file__).parent.parent / "Scripts")
sys.path.insert(0, CHILDREN_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── Import workers + headless functions from child scripts ────────────────────

from wavelet_stretch import (
    StretchWorker,
    PreviewCanvas as WaveletPreviewCanvas,
    LayerRow,
    # Algorithms
    atrous_decompose,
    atrous_reconstruct,
    apply_wavelet_stretch,
    apply_wavelet_stretch_rgb,
    check_linearity,
    STRETCH_FUNCTIONS,
    ALL_PRESETS as WAVELET_PRESETS,
    load_custom_presets,
    save_custom_presets,
    # Theme
    SIRIL_STYLESHEET,
    SIRIL_BG, SIRIL_BG2, SIRIL_BG3,
    SIRIL_ACCENT, SIRIL_ACCENT2,
    SIRIL_TEXT, SIRIL_TEXT_DIM,
    SIRIL_BORDER, SIRIL_SUCCESS,
    SIRIL_WARNING, SIRIL_SECTION,
    SIRIL_ERROR,
)

from local_norm import (
    NormWorker,
    PreviewCanvas as NormPreviewCanvas,
    MapCanvas,
    # Algorithms
    local_normalize_frame,
    build_normalization_map,
    find_best_reference,
    stack_normalized_in_siril,
)

from dark_channel import (
    DehazeWorker,
    # Algorithms
    dehaze_image,
    compute_dark_channel,
    estimate_atmospheric_light,
    estimate_transmission_map,
    refine_transmission_guided_filter,
    recover_scene,
    quick_haze_estimate,
)

from nb_palette import (
    PaletteWorker,
    PaletteSwatch,
    # Algorithms
    detect_filter_masters,
    get_eligible_palettes,
    apply_palette_in_siril,
    build_copy_commands,
    validate_custom_palette,
    PALETTES,
    ALL_FILTERS,
    FILTER_ALIASES,
    VAR_MAP,
)

from mask_builder import (
    MaskWorker,
    MaskPreviewCanvas,
    RangePanel,
    # Algorithms
    to_luminance,
    make_range_mask,
    make_star_mask,
    make_edge_mask,
    make_nebulosity_mask,
    make_color_mask,
    make_gradient_mask,
    combine_masks,
    compute_final_mask,
    ALL_PRESETS as MASK_PRESETS,
    MASK_TYPES,
    MASK_ICONS,
    OPERATORS,
    _make_layer,
    _new_layer_id,
)

import numpy as np
from astropy.io import fits as astropy_fits

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QScrollArea, QSizePolicy, QSplitter, QFrame,
    QAbstractItemView, QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QRect
from PyQt6.QtGui import QFont, QColor, QPainter

# ─────────────────────────────────────────────────────────────────────────────
# SHARED FITS LOADER  (used by all panels for the shared input bar)
# ─────────────────────────────────────────────────────────────────────────────

def load_fits_data(path: str) -> tuple:
    """Load FITS file. Returns (data: np.ndarray, header, info_str: str)."""
    with astropy_fits.open(path) as hdul:
        data   = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    if data.ndim == 3 and data.shape[0] == 3:
        h, w = data.shape[1], data.shape[2]
        info = f"RGB  {w}×{h}  {data.dtype}"
    elif data.ndim == 3:
        h, w = data.shape[1], data.shape[2]
        info = f"Cube ({data.shape[0]} channels)  {w}×{h}  {data.dtype}"
    else:
        h, w = data.shape
        info = f"Mono  {w}×{h}  {data.dtype}"
    return data, header, info


def _auto_output_path(input_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(input_path)
    if not ext: ext = ".fit"
    return base + suffix + ext
# ─────────────────────────────────────────────────────────────────────────────
# WAVELET CROP CANVAS  (full-resolution crop preview)
# ─────────────────────────────────────────────────────────────────────────────

class WaveletCropCanvas(FigureCanvasQTAgg):
    """
    Full-resolution crop preview.
    Top row: downsampled overview (original | stretched) with draggable selection box.
    Bottom row: full-res crop of selected area (original | stretched).
    Click and drag on either top image to select the crop region.
    """
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), facecolor="#1e2128")
        self.ax_orig    = self.fig.add_subplot(2, 2, 1)
        self.ax_stretch = self.fig.add_subplot(2, 2, 2)
        self.ax_crop_orig    = self.fig.add_subplot(2, 2, 3)
        self.ax_crop_stretch = self.fig.add_subplot(2, 2, 4)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)

        self.original_data  = None
        self.stretched_data = None
        self._rect_patch    = None
        self._crop_x0 = self._crop_x1 = None
        self._crop_y0 = self._crop_y1 = None
        self._press   = None

        # Mouse events for rubber-band selection on top axes
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def _style_axes(self):
        titles = [
            (self.ax_orig,         "Original (linear)",      "#7a8499"),
            (self.ax_stretch,      "Wavelet stretched",       "#4a9eff"),
            (self.ax_crop_orig,    "Original — crop (full res)",  "#7a8499"),
            (self.ax_crop_stretch, "Stretched — crop (full res)", "#4a9eff"),
        ]
        for ax, title, color in titles:
            ax.set_facecolor("#1e2128")
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")

    def _prepare(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 3:
            d = np.transpose(data, (1,2,0)) if data.shape[0] == 3 else data
            d = (d - d.min()) / (d.max() - d.min() + 1e-10)
            return np.clip(d, 0, 1)
        p_lo = np.percentile(data, 0.5)
        p_hi = np.percentile(data, 99.5)
        return np.clip((data - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)

    def _ds(self, data: np.ndarray, max_px: int = 800) -> np.ndarray:
        h = data.shape[-2] if data.ndim == 3 else data.shape[0]
        w = data.shape[-1] if data.ndim == 3 else data.shape[1]
        f = max(1, max(h, w) // max_px)
        if data.ndim == 2: return data[::f, ::f]
        if data.shape[0] == 3: return data[:, ::f, ::f]
        return data[::f, ::f, :]

    def show_original(self, data: np.ndarray):
        self.original_data = data
        self.ax_orig.clear()
        self.ax_orig.set_facecolor("#1e2128")
        self.ax_orig.imshow(self._prepare(self._ds(data)), cmap="gray",
                            origin="lower", aspect="equal", interpolation="nearest")
        self.ax_orig.set_title("Original (linear)", color="#7a8499", fontsize=9)
        self.ax_orig.tick_params(left=False, bottom=False,
                                  labelleft=False, labelbottom=False)
        lbl = self.ax_crop_orig.set_title(
            "Original — crop (full res)  |  drag on image above to select",
            color="#7a8499", fontsize=9)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def show_stretched(self, data: np.ndarray):
        self.stretched_data = data
        self.ax_stretch.clear()
        self.ax_stretch.set_facecolor("#1e2128")
        self.ax_stretch.imshow(self._prepare(self._ds(data)), cmap="gray",
                               origin="lower", aspect="equal", interpolation="nearest")
        self.ax_stretch.set_title("Wavelet stretched", color="#4a9eff", fontsize=9)
        self.ax_stretch.tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)
        self.fig.tight_layout(pad=0.5)
        self.draw()
        # Refresh crop if selection already exists
        if self._crop_x0 is not None:
            self._update_crops()

    def _on_press(self, event):
        if event.inaxes not in (self.ax_orig, self.ax_stretch): return
        self._press = (event.xdata, event.ydata)
        self._crop_x0, self._crop_y0 = event.xdata, event.ydata
        self._crop_x1, self._crop_y1 = event.xdata, event.ydata
        if self._rect_patch:
            self._rect_patch.remove()
            self._rect_patch = None

    def _on_motion(self, event):
        if self._press is None: return
        if event.inaxes not in (self.ax_orig, self.ax_stretch): return
        self._crop_x1, self._crop_y1 = event.xdata, event.ydata
        # Draw rectangle on both overview axes
        from matplotlib.patches import Rectangle
        for ax in (self.ax_orig, self.ax_stretch):
            # Remove old patches
            for p in ax.patches: p.remove()
            x0 = min(self._crop_x0, self._crop_x1)
            y0 = min(self._crop_y0, self._crop_y1)
            w  = abs(self._crop_x1 - self._crop_x0)
            h  = abs(self._crop_y1 - self._crop_y0)
            ax.add_patch(Rectangle((x0, y0), w, h,
                         linewidth=1.5, edgecolor="#ff6b35",
                         facecolor="none", linestyle="--"))
        self.draw()

    def _on_release(self, event):
        if self._press is None: return
        self._press = None
        if event.xdata is not None:
            self._crop_x1, self._crop_y1 = event.xdata, event.ydata
        self._update_crops()

    def _update_crops(self):
        if self.original_data is None: return
        if self._crop_x0 is None: return

        # Convert display (downsampled) coords → full-res pixel coords
        data = self.original_data
        if data.ndim == 2:
            full_h, full_w = data.shape
        elif data.shape[0] == 3:
            _, full_h, full_w = data.shape
        else:
            full_h, full_w, _ = data.shape

        ds_data = self._ds(data)
        if ds_data.ndim == 2:
            ds_h, ds_w = ds_data.shape
        elif ds_data.shape[0] == 3:
            _, ds_h, ds_w = ds_data.shape
        else:
            ds_h, ds_w, _ = ds_data.shape

        scale_x = full_w / ds_w
        scale_y = full_h / ds_h

        x0 = int(min(self._crop_x0, self._crop_x1) * scale_x)
        x1 = int(max(self._crop_x0, self._crop_x1) * scale_x)
        y0 = int(min(self._crop_y0, self._crop_y1) * scale_y)
        y1 = int(max(self._crop_y0, self._crop_y1) * scale_y)

        # Clamp to image bounds
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(full_w, x1); y1 = min(full_h, y1)

        if x1 - x0 < 4 or y1 - y0 < 4: return  # Too small

        # Extract full-res crops
        def _crop(d):
            if d.ndim == 2:       return d[y0:y1, x0:x1]
            elif d.shape[0] == 3: return d[:, y0:y1, x0:x1]
            else:                 return d[y0:y1, x0:x1, :]

        crop_orig = _crop(data)
        for ax, crop, title, color in [
            (self.ax_crop_orig,    crop_orig,                        "Original — crop (full res)",   "#7a8499"),
            (self.ax_crop_stretch, _crop(self.stretched_data) if self.stretched_data is not None
                                   else crop_orig,                   "Stretched — crop (full res)",  "#4a9eff"),
        ]:
            ax.clear()
            ax.set_facecolor("#1e2128")
            ax.imshow(self._prepare(crop), cmap="gray", origin="lower",
                      aspect="equal", interpolation="nearest")
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")

        self.fig.tight_layout(pad=0.5)
        self.draw()

# ─────────────────────────────────────────────────────────────────────────────
# WAVELET PANEL
# ─────────────────────────────────────────────────────────────────────────────

class WaveletPanel(QWidget):
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fits_data    = None
        self._fits_path    = ""
        self._is_rgb       = False
        self._worker       = None
        self._cancel_event = threading.Event()
        self._layer_rows   = []
        self._custom_presets = load_custom_presets()
        self._current_preset_name = ""
        self._build_ui()
        self._rebuild_layer_rows(6)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Left settings panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(400)
        left_w = QWidget(); ll = QVBoxLayout(left_w); ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)
        left_scroll.setWidget(left_w)

        # Preset group
        grp_preset = QGroupBox("Preset")
        gp = QVBoxLayout(grp_preset)
        preset_row = QHBoxLayout(); preset_row.setSpacing(3)
        for p in WAVELET_PRESETS:
            btn = QPushButton(p["name"])
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda checked, preset=p: self._apply_preset(preset))
            preset_row.addWidget(btn)
        gp.addLayout(preset_row)
        self._lbl_preset_desc = QLabel("Select a preset")
        self._lbl_preset_desc.setObjectName("dim"); self._lbl_preset_desc.setWordWrap(True)
        gp.addWidget(self._lbl_preset_desc)
        ll.addWidget(grp_preset)

        # Linearity check
        self._lbl_linearity = QLabel("")
        self._lbl_linearity.setObjectName("dim"); self._lbl_linearity.setWordWrap(True)
        ll.addWidget(self._lbl_linearity)

        # Levels
        grp_levels = QGroupBox("Wavelet levels")
        glev = QHBoxLayout(grp_levels)
        glev.addWidget(QLabel("Levels:"))
        self._levels_spin = QSpinBox()
        self._levels_spin.setRange(4, 8); self._levels_spin.setValue(6)
        self._levels_spin.setFixedWidth(50)
        self._levels_spin.valueChanged.connect(self._on_levels_changed)
        glev.addWidget(self._levels_spin)
        lbl_hint = QLabel("More = finer control, slower")
        lbl_hint.setObjectName("dim"); glev.addWidget(lbl_hint); glev.addStretch()
        ll.addWidget(grp_levels)

        # Layer settings
        grp_layers = QGroupBox("Layer settings")
        g_lay = QVBoxLayout(grp_layers)
        hdr = QHBoxLayout()
        for txt, w in [("En", 18), ("Layer", 110), ("Method", 72), ("Strength", -1), ("Val", 28)]:
            lbl = QLabel(txt); lbl.setObjectName("dim")
            if w > 0: lbl.setFixedWidth(w)
            else: lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            hdr.addWidget(lbl)
        g_lay.addLayout(hdr)
        sep = QFrame(); sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine); sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{SIRIL_BORDER};"); g_lay.addWidget(sep)
        self._layer_container = QWidget()
        self._layer_layout    = QVBoxLayout(self._layer_container)
        self._layer_layout.setContentsMargins(0, 0, 0, 0); self._layer_layout.setSpacing(2)
        g_lay.addWidget(self._layer_container)
        ll.addWidget(grp_layers)

        # RGB mode
        self._grp_rgb = QGroupBox("RGB mode")
        gr = QVBoxLayout(self._grp_rgb)
        self._rgb_cb = QComboBox()
        self._rgb_cb.addItems(["Luminance (recommended)", "Per channel", "Combined"])
        gr.addWidget(self._rgb_cb)
        lbl_rgb = QLabel("Luminance preserves color ratios")
        lbl_rgb.setObjectName("dim"); gr.addWidget(lbl_rgb)
        self._grp_rgb.setVisible(False)
        ll.addWidget(self._grp_rgb)

        # Output
        grp_out = QGroupBox("Output")
        go = QVBoxLayout(grp_out)
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output FITS path…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(70)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit); out_row.addWidget(btn_out)
        go.addLayout(out_row)
        self._load_siril_chk = QCheckBox("Load result in Siril")
        self._load_siril_chk.setChecked(True); go.addWidget(self._load_siril_chk)
        ll.addWidget(grp_out)
        ll.addStretch()

        # Run button
        self._btn_run = QPushButton("▶  Apply wavelet stretch")
        self._btn_run.setObjectName("primary"); self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        ll.addWidget(self._btn_run); ll.addWidget(self._btn_cancel)
        root.addWidget(left_scroll)

        # Right: preview + log
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(4, 0, 0, 0)
        self._tabs = QTabWidget()
        prev_tab = QWidget(); ptl = QVBoxLayout(prev_tab)
        self._canvas = WaveletCropCanvas()
        ptl.addWidget(self._canvas)
        lbl_note = QLabel("Top row: downsampled overview — drag to select crop area.  Bottom row: full-res crop.")
        lbl_note.setObjectName("dim"); lbl_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ptl.addWidget(lbl_note)
        self._tabs.addTab(prev_tab, "🖼  Preview")
        log_tab = QWidget(); ltl = QVBoxLayout(log_tab)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        ltl.addWidget(self._log)
        self._tabs.addTab(log_tab, "📋  Log")
        rv.addWidget(self._tabs, 1)
        self._progress = QProgressBar()
        self._progress.setVisible(False); self._progress.setFixedHeight(6)
        rv.addWidget(self._progress)
        self._status_lbl = QLabel("Ready."); self._status_lbl.setObjectName("dim")
        rv.addWidget(self._status_lbl)
        root.addWidget(right, 1)

    def set_input(self, path: str):
        self._fits_path = path
        auto = _auto_output_path(path, "_wavelet")
        if not self._out_edit.text(): self._out_edit.setText(auto)
        try:
            data, _, info = load_fits_data(path)
            self._fits_data = data
            self._is_rgb = (data.ndim == 3 and (data.shape[0] == 3 or data.shape[2] == 3))
            self._grp_rgb.setVisible(self._is_rgb)
            is_lin, frac = check_linearity(data if data.ndim == 2 else
                (data[0] if data.shape[0] == 3 else data[:,:,0]))
            self._lbl_linearity.setText(
                f"✓ Linear ({frac*100:.1f}% of range)" if is_lin
                else f"⚠ Median at {frac*100:.0f}% — may already be stretched")
            self._lbl_linearity.setObjectName("ok" if is_lin else "warn")
            self._canvas.show_original(data)
            self._log.appendPlainText(f"Input: {info}")
        except Exception as e:
            self._log.appendPlainText(f"Warning: could not pre-load {path}: {e}")

    def get_output_path(self) -> str:
        return self._out_edit.text().strip()

    def _rebuild_layer_rows(self, levels: int):
        old = [r.get_config() for r in self._layer_rows]
        for r in self._layer_rows:
            r.setParent(None); r.deleteLater()
        self._layer_rows = []
        for i in range(levels + 1):
            row = LayerRow(i, is_residual=(i == levels))
            row.changed.connect(lambda: None)
            self._layer_layout.addWidget(row)
            self._layer_rows.append(row)
            if i < len(old): row.set_config(old[i])

    def _on_levels_changed(self, val): self._rebuild_layer_rows(val)

    def _apply_preset(self, preset: dict):
        levels = preset.get("levels", 6)
        self._levels_spin.setValue(levels)
        self._rebuild_layer_rows(levels)
        for i, (row, cfg) in enumerate(zip(self._layer_rows, preset["layers"])):
            row.set_config(cfg)
        self._current_preset_name = preset["name"]
        self._lbl_preset_desc.setText(preset.get("description", ""))
        if self._fits_path:
            base, ext = os.path.splitext(self._fits_path)
            if not ext: ext = ".fit"
            self._out_edit.setText(f"{base}_wavelet_{preset['name'].lower().replace(' ','_')}{ext}")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save output FITS",
            self._out_edit.text(), "FITS (*.fit *.fits);;All (*)")
        if path: self._out_edit.setText(path)

    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)
        self.log_line.emit(f"[WAVELET]  {msg}")

    def _run(self):
        if not self._fits_path or not os.path.isfile(self._fits_path):
            QMessageBox.warning(self, "No input", "Load a FITS file first."); return
        out = self._out_edit.text().strip()
        if not out:
            QMessageBox.warning(self, "No output", "Set an output path."); return
        layer_configs = [r.get_config() for r in self._layer_rows]
        levels        = self._levels_spin.value()
        if len(layer_configs) != levels + 1:
            QMessageBox.critical(self, "Config error",
                f"Expected {levels+1} layer configs, got {len(layer_configs)}"); return
        rgb_map = {"Luminance (recommended)": "luminance",
                   "Per channel": "per_channel", "Combined": "combined"}
        rgb_mode = rgb_map.get(self._rgb_cb.currentText(), "luminance")
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True); self._progress.setValue(0)
        self._status_lbl.setText("Processing…")
        self._worker = StretchWorker(self._fits_path, layer_configs, levels,
                                     rgb_mode, out, self._cancel_event, self._fits_data)
        self._worker.progress.connect(lambda c, t, d: (
            self._progress.setValue(int(c/t*100)), self._status_lbl.setText(d)))
        self._worker.log_line.connect(self._log_msg)
        self._worker.preview_ready.connect(self._canvas.show_stretched)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        self._cancel_event.set(); self._btn_cancel.setEnabled(False)

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        if result.get("success"):
            out = result["output"]
            self._status_lbl.setText(f"✓ Done: {os.path.basename(out)}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS};")
            self._log_msg(f"✓ Saved: {out}")
            if self._load_siril_chk.isChecked():
                try:
                    siril = s.SirilInterface(); siril.connect()
                    siril.cmd("load", out); siril.disconnect()
                    self._log_msg("Loaded in Siril.")
                except Exception as e:
                    self._log_msg(f"Siril load: {e}")
            self.finished.emit({"success": True, "output_path": out})
        else:
            err = result.get("error", "Unknown")
            self._status_lbl.setText(f"✗ {err}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_ERROR};")
            self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL NORM PANEL
# ─────────────────────────────────────────────────────────────────────────────

class LocalNormPanel(QWidget):
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fits_files  = []
        self._cancel_event = threading.Event()
        self._worker      = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        # Input group
        grp_in = QGroupBox("Input files (folder of FITS)")
        gl = QVBoxLayout(grp_in)
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Folder with FITS files to normalize…")
        btn_browse = QPushButton("Browse…"); btn_browse.setFixedWidth(70)
        btn_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit); folder_row.addWidget(btn_browse)
        gl.addLayout(folder_row)
        btn_scan = QPushButton("🔍  Scan folder"); btn_scan.clicked.connect(self._scan)
        gl.addWidget(btn_scan)
        self._lbl_scan = QLabel("No files scanned.")
        self._lbl_scan.setObjectName("dim"); gl.addWidget(self._lbl_scan)
        root.addWidget(grp_in)

        # Reference
        grp_ref = QGroupBox("Reference frame")
        rf = QHBoxLayout(grp_ref)
        self._ref_combo = QComboBox()
        self._ref_combo.addItems(["Auto (highest SNR)", "First file", "Manual selection"])
        rf.addWidget(self._ref_combo)
        self._ref_edit = QLineEdit(); self._ref_edit.setPlaceholderText("Manual reference path…")
        self._ref_edit.setVisible(False)
        btn_ref = QPushButton("Browse…"); btn_ref.setFixedWidth(70); btn_ref.setVisible(False)
        btn_ref.clicked.connect(self._browse_ref)
        self._ref_combo.currentTextChanged.connect(
            lambda t: (self._ref_edit.setVisible(t == "Manual selection"),
                       btn_ref.setVisible(t == "Manual selection")))
        rf.addWidget(self._ref_edit); rf.addWidget(btn_ref)
        root.addWidget(grp_ref)

        # Parameters
        grp_params = QGroupBox("Normalization parameters")
        pf = QFormLayout(grp_params)
        self._tile_spin = QSpinBox()
        self._tile_spin.setRange(16, 512); self._tile_spin.setValue(64); self._tile_spin.setSuffix(" px")
        pf.addRow("Tile size:", self._tile_spin)
        self._overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self._overlap_slider.setRange(0, 90); self._overlap_slider.setValue(50)
        self._overlap_lbl = QLabel("50%")
        self._overlap_lbl.setFixedWidth(35)
        self._overlap_slider.valueChanged.connect(lambda v: self._overlap_lbl.setText(f"{v}%"))
        ol_row = QHBoxLayout()
        ol_row.addWidget(self._overlap_slider); ol_row.addWidget(self._overlap_lbl)
        pf.addRow("Tile overlap:", ol_row)
        self._mad_chk = QCheckBox("Use MAD scale (robust, recommended)")
        self._mad_chk.setChecked(True); pf.addRow(self._mad_chk)
        root.addWidget(grp_params)

        # Output
        grp_out = QGroupBox("Output")
        of = QVBoxLayout(grp_out)
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit(); self._out_edit.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(70)
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(self._out_edit); out_row.addWidget(btn_out)
        of.addLayout(out_row)
        self._stack_chk = QCheckBox("Auto-stack in Siril after normalization")
        of.addWidget(self._stack_chk)
        root.addWidget(grp_out)

        # Preview canvas
        self._preview = NormPreviewCanvas(); self._preview.setMinimumHeight(200)
        root.addWidget(self._preview, 1)

        # Bottom
        bot = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run normalization")
        self._btn_run.setObjectName("primary"); self._btn_run.clicked.connect(self._run)
        bot.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bot.addWidget(self._btn_cancel)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        self._progress.setFixedHeight(8); bot.addWidget(self._progress, 1)
        root.addLayout(bot)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        self._log.setMaximumHeight(80); root.addWidget(self._log)
        self._status_lbl = QLabel("Ready."); self._status_lbl.setObjectName("dim")
        root.addWidget(self._status_lbl)

    def set_input(self, path: str):
        """Single file → set its parent folder."""
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        self._folder_edit.setText(folder)
        if not self._out_edit.text():
            self._out_edit.setText(os.path.join(folder, "normalized"))

    def get_output_path(self) -> str:
        return self._out_edit.text().strip()

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d: self._folder_edit.setText(d)

    def _browse_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select reference FITS", "",
            "FITS (*.fit *.fits *.fts);;All (*)")
        if path: self._ref_edit.setText(path)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self._out_edit.setText(d)

    def _scan(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Select a valid FITS folder."); return
        files = sorted(
            glob.glob(os.path.join(folder, "*.fit")) +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts")))
        self._fits_files = files
        self._lbl_scan.setText(f"{len(files)} FITS files found")
        if files and not self._out_edit.text():
            self._out_edit.setText(os.path.join(folder, "normalized"))

    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)
        self.log_line.emit(f"[LOCAL_NORM]  {msg}")

    def _run(self):
        if not self._fits_files:
            QMessageBox.warning(self, "No files", "Scan a folder first."); return
        ref_mode = self._ref_combo.currentText()
        ref_path = None
        if ref_mode == "Manual selection":
            ref_path = self._ref_edit.text().strip()
        elif ref_mode == "First file":
            ref_path = self._fits_files[0]

        out_dir = self._out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output", "Set an output folder."); return

        cfg = {
            "fits_files":    self._fits_files,
            "reference_path":ref_path,
            "output_dir":    out_dir,
            "tile_size":     self._tile_spin.value(),
            "overlap":       self._overlap_slider.value() / 100.0,
            "use_mad":       self._mad_chk.isChecked(),
            "auto_stack":    self._stack_chk.isChecked(),
            "stack_method":  "winsorized",
            "stack_sigma":   3.0,
            "stack_output":  "master_normalized",
        }
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._worker = NormWorker(cfg, self._cancel_event)
        self._worker.progress.connect(lambda c, t, m: (
            self._progress.setRange(0, t), self._progress.setValue(c),
            self._status_lbl.setText(m)))
        self._worker.log_line.connect(self._log_msg)
        self._worker.preview_ready.connect(
            lambda bef, aft: (self._preview.show_before(bef),
                              self._preview.show_after(aft)))
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        self._cancel_event.set()

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        if result.get("success"):
            n = result.get("n_processed", 0)
            self._status_lbl.setText(f"✓ Done: {n} files normalized")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS};")
            self.finished.emit({"success": True, "output_path": result.get("output_dir", "")})
        else:
            self._status_lbl.setText(f"✗ {result.get('error','')}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_ERROR};")
            self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# DARK CHANNEL PANEL
# ─────────────────────────────────────────────────────────────────────────────

class DarkChannelPanel(QWidget):
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self); root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)

        # Left settings
        left = QWidget(); left.setFixedWidth(320)
        ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 4, 0); ll.setSpacing(6)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); il = QVBoxLayout(inner); il.setSpacing(8); il.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(inner); ll.addWidget(scroll, 1)

        # Mode
        grp_mode = QGroupBox("Mode")
        mf = QFormLayout(grp_mode)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Single image", "Batch folder"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mf.addRow("Mode:", self._mode_combo)
        il.addWidget(grp_mode)

        # Single image input
        self._grp_single = QGroupBox("Single image")
        sf = QVBoxLayout(self._grp_single)
        in_row = QHBoxLayout()
        self._single_edit = QLineEdit(); self._single_edit.setPlaceholderText("FITS file…")
        btn_si = QPushButton("Browse…"); btn_si.setFixedWidth(70)
        btn_si.clicked.connect(self._browse_single)
        in_row.addWidget(self._single_edit); in_row.addWidget(btn_si)
        sf.addLayout(in_row)
        self._lbl_haze = QLabel(""); self._lbl_haze.setObjectName("dim")
        sf.addWidget(self._lbl_haze)
        btn_estimate = QPushButton("⚡  Quick haze estimate")
        btn_estimate.clicked.connect(self._quick_estimate)
        sf.addWidget(btn_estimate)
        il.addWidget(self._grp_single)

        # Batch input
        self._grp_batch = QGroupBox("Batch folder")
        bf = QVBoxLayout(self._grp_batch)
        batch_row = QHBoxLayout()
        self._batch_edit = QLineEdit(); self._batch_edit.setPlaceholderText("FITS folder…")
        btn_bf = QPushButton("Browse…"); btn_bf.setFixedWidth(70)
        btn_bf.clicked.connect(self._browse_batch)
        batch_row.addWidget(self._batch_edit); batch_row.addWidget(btn_bf)
        bf.addLayout(batch_row)
        self._grp_batch.setVisible(False)
        il.addWidget(self._grp_batch)

        # DCP parameters
        grp_dcp = QGroupBox("Dark Channel Prior parameters")
        df = QFormLayout(grp_dcp)

        self._patch_spin = QSpinBox()
        self._patch_spin.setRange(5, 51); self._patch_spin.setValue(15)
        self._patch_spin.setSingleStep(2); self._patch_spin.setSuffix(" px")
        df.addRow("Patch size:", self._patch_spin)

        self._omega_spin = QDoubleSpinBox()
        self._omega_spin.setRange(0.5, 1.0); self._omega_spin.setValue(0.95)
        self._omega_spin.setSingleStep(0.05); self._omega_spin.setDecimals(2)
        df.addRow("Omega (haze strength):", self._omega_spin)

        self._tmin_spin = QDoubleSpinBox()
        self._tmin_spin.setRange(0.01, 0.5); self._tmin_spin.setValue(0.1)
        self._tmin_spin.setSingleStep(0.01); self._tmin_spin.setDecimals(2)
        df.addRow("t_min (lower bound):", self._tmin_spin)

        self._refine_chk = QCheckBox("Refine transmission (guided filter)")
        self._refine_chk.setChecked(True); df.addRow(self._refine_chk)

        self._guided_radius_spin = QSpinBox()
        self._guided_radius_spin.setRange(5, 120); self._guided_radius_spin.setValue(40)
        df.addRow("Guided filter radius:", self._guided_radius_spin)

        il.addWidget(grp_dcp)

        # Output
        grp_out = QGroupBox("Output")
        of = QFormLayout(grp_out)
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit(); self._out_edit.setPlaceholderText("Same as input folder")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(70)
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(self._out_edit); out_row.addWidget(btn_out)
        of.addRow("Output folder:", out_row)
        self._load_siril_chk = QCheckBox("Load result in Siril")
        self._load_siril_chk.setChecked(True); of.addRow(self._load_siril_chk)
        il.addWidget(grp_out)
        il.addStretch()

        # Run
        self._btn_run = QPushButton("▶  Dehaze")
        self._btn_run.setObjectName("primary"); self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        self._progress.setFixedHeight(8)
        ll.addWidget(self._btn_run); ll.addWidget(self._btn_cancel)
        ll.addWidget(self._progress)
        root.addWidget(left)

        # Right: preview canvas
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(4, 0, 0, 0)
        self._tabs = QTabWidget()
        prev_tab = QWidget(); ptl = QVBoxLayout(prev_tab)
        self._prev_canvas = _DehazePrevCanvas()
        ptl.addWidget(self._prev_canvas)
        self._tabs.addTab(prev_tab, "🖼  Preview")
        log_tab = QWidget(); ltl = QVBoxLayout(log_tab)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        ltl.addWidget(self._log)
        self._tabs.addTab(log_tab, "📋  Log")
        rv.addWidget(self._tabs, 1)
        self._status_lbl = QLabel("Ready."); self._status_lbl.setObjectName("dim")
        rv.addWidget(self._status_lbl)
        root.addWidget(right, 1)

    def set_input(self, path: str):
        self._single_edit.setText(path)
        out = os.path.dirname(path) if os.path.isfile(path) else path
        if not self._out_edit.text(): self._out_edit.setText(out)

    def get_output_path(self) -> str:
        p = self._single_edit.text().strip()
        if p and os.path.isfile(p):
            base = os.path.splitext(os.path.basename(p))[0]
            out  = self._out_edit.text().strip() or os.path.dirname(p)
            return os.path.join(out, f"{base}_dehazed.fit")
        return self._out_edit.text().strip()

    def _on_mode_changed(self, t: str):
        self._grp_single.setVisible(t == "Single image")
        self._grp_batch.setVisible(t == "Batch folder")

    def _browse_single(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select FITS", "",
            "FITS (*.fit *.fits *.fts);;All (*)")
        if path: self._single_edit.setText(path)

    def _browse_batch(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d: self._batch_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self._out_edit.setText(d)

    def _quick_estimate(self):
        path = self._single_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file", "Select a FITS file first."); return
        try:
            haze = quick_haze_estimate(path, patch_size=self._patch_spin.value())
            pct  = int(haze * 100)
            if pct < 5:
                msg = f"Haze estimate: ~{pct}%  — image appears clean"
                color = SIRIL_SUCCESS
            elif pct < 20:
                msg = f"Haze estimate: ~{pct}%  — mild haze"
                color = SIRIL_WARNING
            else:
                msg = f"Haze estimate: ~{pct}%  — significant haze detected"
                color = SIRIL_ERROR
            self._lbl_haze.setText(msg)
            self._lbl_haze.setStyleSheet(f"color:{color}; font-weight:bold;")
        except Exception as e:
            self._lbl_haze.setText(f"Estimate failed: {e}")

    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)
        self.log_line.emit(f"[DEHAZE]  {msg}")

    def _run(self):
        mode = self._mode_combo.currentText()
        if mode == "Single image":
            path = self._single_edit.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "No file", "Select a FITS file."); return
            cfg = {
                "mode":                "single",
                "fits_path":           path,
                "output_dir":          self._out_edit.text().strip() or os.path.dirname(path),
                "patch_size":          self._patch_spin.value(),
                "omega":               self._omega_spin.value(),
                "t_min":               self._tmin_spin.value(),
                "refine_transmission": self._refine_chk.isChecked(),
                "guided_radius":       self._guided_radius_spin.value(),
                "guided_epsilon":      1e-3,
                "top_fraction":        0.001,
                "load_in_siril":       self._load_siril_chk.isChecked(),
            }
        else:
            folder = self._batch_edit.text().strip()
            if not folder or not os.path.isdir(folder):
                QMessageBox.warning(self, "No folder", "Select a FITS folder."); return
            files = sorted(glob.glob(os.path.join(folder, "*.fit")) +
                           glob.glob(os.path.join(folder, "*.fits")) +
                           glob.glob(os.path.join(folder, "*.fts")))
            cfg = {
                "mode":     "batch",
                "fits_files":files,
                "output_dir":self._out_edit.text().strip() or folder,
                "patch_size":self._patch_spin.value(),
                "omega":     self._omega_spin.value(),
                "t_min":     self._tmin_spin.value(),
                "refine_transmission": self._refine_chk.isChecked(),
                "guided_radius": self._guided_radius_spin.value(),
                "guided_epsilon": 1e-3,
                "top_fraction": 0.001,
            }
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True); self._progress.setValue(0)
        self._worker = DehazeWorker(cfg, self._cancel_event)
        self._worker.progress.connect(lambda c, t, m: (
            self._progress.setRange(0, t), self._progress.setValue(c),
            self._status_lbl.setText(m)))
        self._worker.log_line.connect(self._log_msg)
        self._worker.preview_ready.connect(self._prev_canvas.show_result)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        self._cancel_event.set()

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        if result.get("success"):
            self._status_lbl.setText("✓ Dehaze complete")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS};")
            out = result.get("output_path", self.get_output_path())
            self.finished.emit({"success": True, "output_path": out})
        else:
            err = result.get("error", "Unknown")
            self._status_lbl.setText(f"✗ {err}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_ERROR};")
            self.finished.emit(result)


class _DehazePrevCanvas(FigureCanvasQTAgg):
    """3-panel preview: original lum / dehazed lum / transmission map."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 4), facecolor=SIRIL_BG)
        self.ax_orig = self.fig.add_subplot(1, 3, 1)
        self.ax_deh  = self.fig.add_subplot(1, 3, 2)
        self.ax_t    = self.fig.add_subplot(1, 3, 3)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style(self):
        titles = [("Original", SIRIL_TEXT_DIM),
                  ("Dehazed",  SIRIL_SUCCESS),
                  ("Transmission map", SIRIL_ACCENT)]
        for ax, (title, color) in zip([self.ax_orig, self.ax_deh, self.ax_t], titles):
            ax.set_facecolor(SIRIL_BG)
            ax.set_title(title, color=color, fontsize=8)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values(): sp.set_color(SIRIL_BORDER)

    def _disp(self, arr):
        p_lo = np.percentile(arr, 0.5); p_hi = np.percentile(arr, 99.5)
        return np.clip((arr.astype(float) - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)

    def _ds(self, arr, max_px=400):
        h, w = arr.shape[:2]; factor = max(1, max(h, w) // max_px)
        return arr[::factor, ::factor]

    def show_result(self, dehazed_lum, t_map, orig_lum):
        for ax, arr, cmap in [
            (self.ax_orig, orig_lum,    "gray"),
            (self.ax_deh,  dehazed_lum, "gray"),
            (self.ax_t,    t_map,       "viridis"),
        ]:
            ax.clear(); ax.set_facecolor(SIRIL_BG)
            ax.imshow(self._ds(self._disp(arr)), cmap=cmap,
                      origin="lower", aspect="equal", interpolation="nearest")
        self._style(); self.fig.tight_layout(pad=0.3); self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# NARROWBAND PALETTE PANEL
# ─────────────────────────────────────────────────────────────────────────────

class NarrowbandPanel(QWidget):
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_paths: dict = {}
        self._selected_palette   = None
        self._worker             = None
        self._cancel_event       = threading.Event()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._build_tab_input()
        self._build_tab_palette()
        self._build_tab_log()

        bot = QHBoxLayout()
        self._btn_apply = QPushButton("▶  Apply palette")
        self._btn_apply.setObjectName("primary"); self._btn_apply.setMinimumHeight(34)
        self._btn_apply.clicked.connect(self._apply)
        bot.addWidget(self._btn_apply, 1)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bot.addWidget(self._btn_cancel)
        self._progress = QProgressBar(); self._progress.setRange(0, 3)
        self._progress.setVisible(False); self._progress.setMaximumWidth(140)
        bot.addWidget(self._progress)
        root.addLayout(bot)
        self._status_lbl = QLabel("Ready — detect filters and choose a palette")
        self._status_lbl.setObjectName("dim"); root.addWidget(self._status_lbl)

    def _build_tab_input(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(8)

        grp_folder = QGroupBox("Master FITS folder")
        ff = QVBoxLayout(grp_folder)
        frow = QHBoxLayout()
        self._folder_edit = QLineEdit(); self._folder_edit.setPlaceholderText("Folder with master FITS…")
        btn_f = QPushButton("Browse…"); btn_f.setFixedWidth(70)
        btn_f.clicked.connect(self._browse_folder)
        frow.addWidget(self._folder_edit); frow.addWidget(btn_f)
        ff.addLayout(frow)
        btn_detect = QPushButton("🔍  Auto-detect filters")
        btn_detect.clicked.connect(self._detect_filters); ff.addWidget(btn_detect)

        # Filter table
        self._filter_table = QTableWidget(len(ALL_FILTERS), 3)
        self._filter_table.setHorizontalHeaderLabels(["Filter", "File", "Status"])
        self._filter_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._filter_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._filter_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._filter_table.setAlternatingRowColors(True)
        self._filter_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._filter_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._filter_table.setMaximumHeight(220)
        self._init_filter_table()
        ff.addWidget(self._filter_table)
        self._lbl_filter_summary = QLabel("No filters detected")
        self._lbl_filter_summary.setObjectName("dim"); ff.addWidget(self._lbl_filter_summary)
        lay.addWidget(grp_folder)

        grp_out = QGroupBox("Output")
        of = QFormLayout(grp_out)
        orow = QHBoxLayout()
        self._out_edit = QLineEdit(); self._out_edit.setPlaceholderText("Defaults to input folder")
        btn_o = QPushButton("Browse…"); btn_o.setFixedWidth(70)
        btn_o.clicked.connect(self._browse_out)
        orow.addWidget(self._out_edit); orow.addWidget(btn_o)
        of.addRow("Output folder:", orow)
        self._prefix_edit = QLineEdit("palette"); of.addRow("Filename prefix:", self._prefix_edit)
        lay.addWidget(grp_out)
        lay.addStretch()
        self._tabs.addTab(w, "📁  Input")

    def _init_filter_table(self):
        for row, filt in enumerate(ALL_FILTERS):
            self._filter_table.setItem(row, 0, QTableWidgetItem(filt))
            self._filter_table.setItem(row, 1, QTableWidgetItem(""))
            item = QTableWidgetItem("—")
            item.setForeground(QColor(SIRIL_TEXT_DIM))
            self._filter_table.setItem(row, 2, item)

    def _build_tab_palette(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(6)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); il = QVBoxLayout(inner); il.setSpacing(4)
        scroll.setWidget(inner)

        self._palette_btns = []
        for p in PALETTES:
            btn_row = QHBoxLayout()
            swatch = PaletteSwatch(*p.get("colour_tags", ["#888","#888","#888"]))
            btn = QPushButton(f"  {p['name']}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, pal=p: self._select_palette(pal))
            notes_lbl = QLabel(p.get("description", ""))
            notes_lbl.setObjectName("dim"); notes_lbl.setFixedWidth(200)
            notes_lbl.setWordWrap(True)
            btn_row.addWidget(swatch); btn_row.addWidget(btn, 1); btn_row.addWidget(notes_lbl)
            il.addLayout(btn_row)
            self._palette_btns.append((btn, p))

        il.addStretch()
        lay.addWidget(scroll, 1)

        # Custom palette fields
        grp_custom = QGroupBox("Custom palette (when 'Custom' selected)")
        cf = QFormLayout(grp_custom)
        self._custom_r = QLineEdit(); cf.addRow("R formula:", self._custom_r)
        self._custom_g = QLineEdit(); cf.addRow("G formula:", self._custom_g)
        self._custom_b = QLineEdit(); cf.addRow("B formula:", self._custom_b)
        hint = QLabel("Use $Ha, $OIII, $SII, $L, $Hb, $NII  e.g. 0.8*$SII + 0.2*$Ha")
        hint.setObjectName("dim"); hint.setWordWrap(True); cf.addRow(hint)
        lay.addWidget(grp_custom)
        self._tabs.addTab(w, "🌈  Palettes")

    def _build_tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); lay.addWidget(self._log)
        self._tabs.addTab(w, "📋  Log")

    def set_input(self, path: str):
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        self._folder_edit.setText(folder)
        if not self._out_edit.text(): self._out_edit.setText(folder)

    def get_output_path(self) -> str:
        out = self._out_edit.text().strip()
        prefix = self._prefix_edit.text().strip() or "palette"
        if self._selected_palette:
            return os.path.join(out, f"{prefix}_{self._selected_palette['id']}.fit")
        return out

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d: self._folder_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self._out_edit.setText(d)

    def _detect_filters(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Select a FITS folder first."); return
        detected = detect_filter_masters(folder)
        self._filter_paths = detected
        for row, filt in enumerate(ALL_FILTERS):
            if filt in detected:
                self._filter_table.setItem(row, 1,
                    QTableWidgetItem(os.path.basename(detected[filt])))
                item = QTableWidgetItem("✓")
                item.setForeground(QColor(SIRIL_SUCCESS))
                self._filter_table.setItem(row, 2, item)
            else:
                self._filter_table.setItem(row, 1, QTableWidgetItem(""))
                item = QTableWidgetItem("—")
                item.setForeground(QColor(SIRIL_TEXT_DIM))
                self._filter_table.setItem(row, 2, item)
        n = len(detected)
        filters_str = ", ".join(detected.keys()) if detected else "none"
        self._lbl_filter_summary.setText(
            f"{n} filter(s) detected: {filters_str}")
        if not self._out_edit.text(): self._out_edit.setText(folder)

    def _select_palette(self, palette: dict):
        for btn, p in self._palette_btns:
            btn.setChecked(p["id"] == palette["id"])
        self._selected_palette = palette
        self._status_lbl.setText(f"Selected: {palette['name']}  |  "
                                  f"Required: {', '.join(palette.get('required', []))}")

    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)
        self.log_line.emit(f"[PALETTE]  {msg}")

    def _apply(self):
        if not self._filter_paths:
            QMessageBox.warning(self, "No filters", "Auto-detect filters first."); return
        if not self._selected_palette:
            QMessageBox.warning(self, "No palette", "Select a palette first."); return
        out_dir = self._out_edit.text().strip() or self._folder_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output", "Set an output folder."); return

        palette = self._selected_palette.copy()
        if palette["id"] == "Custom":
            palette["formula_r"] = self._custom_r.text().strip()
            palette["formula_g"] = self._custom_g.text().strip()
            palette["formula_b"] = self._custom_b.text().strip()
            ok, err = validate_custom_palette(
                palette["formula_r"], palette["formula_g"],
                palette["formula_b"], self._filter_paths)
            if not ok:
                QMessageBox.critical(self, "Invalid formula", err); return

        req = palette.get("required", [])
        missing = [f for f in req if f not in self._filter_paths]
        if missing:
            QMessageBox.warning(self, "Missing filters",
                f"Palette '{palette['name']}' requires: {', '.join(missing)}\n"
                f"Available: {', '.join(self._filter_paths.keys())}"); return

        filter_paths = {k: v for k, v in self._filter_paths.items()
                        if k in (req + palette.get("optional", []))}
        prefix = self._prefix_edit.text().strip() or "palette"
        output_name = f"{prefix}_{palette['id']}"

        cfg = {"palette": palette, "filter_paths": filter_paths,
               "output_dir": out_dir, "output_name": output_name}
        self._cancel_event.clear()
        self._btn_apply.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True); self._progress.setValue(0)
        self._worker = PaletteWorker(cfg, self._cancel_event)
        self._worker.progress.connect(lambda c, t, m: (
            self._progress.setValue(c), self._status_lbl.setText(m)))
        self._worker.log_line.connect(self._log_msg)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        self._cancel_event.set()

    def _on_finished(self, result: dict):
        self._btn_apply.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        if result.get("success"):
            out = result.get("output_path", "")
            self._status_lbl.setText(f"✓ Done: {os.path.basename(out)}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS};")
            self.finished.emit({"success": True, "output_path": out})
        else:
            self._status_lbl.setText(f"✗ {result.get('error','')}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_ERROR};")
            self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# MASK BUILDER PANEL
# ─────────────────────────────────────────────────────────────────────────────

class MaskBuilderPanel(QWidget):
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data         = None
        self._fits_path    = ""
        self._layers       = []
        self._worker       = None
        self._cancel_event = threading.Event()
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self); root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)

        # Left: layer list + controls
        left = QWidget(); left.setFixedWidth(340)
        ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 4, 0); ll.setSpacing(6)

        # Input
        grp_in = QGroupBox("Input FITS")
        gi = QVBoxLayout(grp_in)
        in_row = QHBoxLayout()
        self._in_edit = QLineEdit(); self._in_edit.setPlaceholderText("FITS file…")
        btn_in = QPushButton("Browse…"); btn_in.setFixedWidth(70)
        btn_in.clicked.connect(self._browse_input)
        in_row.addWidget(self._in_edit); in_row.addWidget(btn_in)
        gi.addLayout(in_row)
        btn_load = QPushButton("Load & show original")
        btn_load.clicked.connect(self._load_fits); gi.addWidget(btn_load)
        self._lbl_info = QLabel(""); self._lbl_info.setObjectName("dim")
        gi.addWidget(self._lbl_info)
        ll.addWidget(grp_in)

        # Preset
        grp_preset = QGroupBox("Quick presets")
        gp = QVBoxLayout(grp_preset)
        for preset in MASK_PRESETS:
            btn = QPushButton(f"{preset['name']}")
            btn.clicked.connect(lambda checked, p=preset: self._apply_preset(p))
            gp.addWidget(btn)
        ll.addWidget(grp_preset)

        # Layer list
        grp_layers = QGroupBox("Mask layers")
        glay = QVBoxLayout(grp_layers)
        self._layer_list = QListWidget()
        self._layer_list.setMinimumHeight(120)
        glay.addWidget(self._layer_list)
        add_row = QHBoxLayout()
        self._type_combo = QComboBox()
        for k, v in MASK_TYPES.items():
            self._type_combo.addItem(f"{MASK_ICONS[k]}  {v}", k)
        btn_add = QPushButton("➕  Add"); btn_add.clicked.connect(self._add_layer)
        btn_del = QPushButton("🗑  Remove"); btn_del.clicked.connect(self._remove_layer)
        add_row.addWidget(self._type_combo, 1); add_row.addWidget(btn_add); add_row.addWidget(btn_del)
        glay.addLayout(add_row)
        btn_preview_layer = QPushButton("👁  Preview selected layer")
        btn_preview_layer.clicked.connect(self._preview_single_layer)
        glay.addWidget(btn_preview_layer)
        ll.addWidget(grp_layers, 1)

        # Output
        grp_out = QGroupBox("Output")
        go = QVBoxLayout(grp_out)
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit(); self._out_edit.setPlaceholderText("Output mask FITS…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(70)
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(self._out_edit); out_row.addWidget(btn_out)
        go.addLayout(out_row)
        ll.addWidget(grp_out)

        # Run
        self._btn_run = QPushButton("▶  Build mask")
        self._btn_run.setObjectName("primary"); self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        self._progress.setFixedHeight(8)
        ll.addWidget(self._btn_run); ll.addWidget(self._btn_cancel); ll.addWidget(self._progress)
        root.addWidget(left)

        # Right: preview
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(4, 0, 0, 0)
        self._tabs = QTabWidget()
        prev_tab = QWidget(); ptl = QVBoxLayout(prev_tab)
        self._canvas = MaskPreviewCanvas(); ptl.addWidget(self._canvas)
        self._tabs.addTab(prev_tab, "🎭  Preview")
        log_tab = QWidget(); ltl = QVBoxLayout(log_tab)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); ltl.addWidget(self._log)
        self._tabs.addTab(log_tab, "📋  Log")
        rv.addWidget(self._tabs, 1)
        self._status_lbl = QLabel("Load a FITS and add layers.")
        self._status_lbl.setObjectName("dim"); rv.addWidget(self._status_lbl)
        root.addWidget(right, 1)

    def set_input(self, path: str):
        self._in_edit.setText(path)
        auto = _auto_output_path(path, "_mask")
        if not self._out_edit.text(): self._out_edit.setText(auto)

    def get_output_path(self) -> str:
        return self._out_edit.text().strip()

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select FITS", "",
            "FITS (*.fit *.fits *.fts);;All (*)")
        if path: self._in_edit.setText(path)

    def _browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save mask FITS",
            self._out_edit.text(), "FITS (*.fit *.fits);;All (*)")
        if path: self._out_edit.setText(path)

    def _load_fits(self):
        path = self._in_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file", "Select a FITS file first."); return
        try:
            data, _, info = load_fits_data(path)
            self._data = data; self._fits_path = path
            self._lbl_info.setText(info)
            self._canvas.show_original(data)
            self._log.appendPlainText(f"Loaded: {info}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _apply_preset(self, preset: dict):
        self._layers = [dict(l) for l in preset["layers"]]
        for l in self._layers: l["cached_mask"] = None
        self._refresh_layer_list()

    def _add_layer(self):
        mtype = self._type_combo.currentData()
        name  = MASK_TYPES.get(mtype, mtype)
        op    = "AND" if self._layers else "AND"
        layer = _make_layer(name, mtype, op, 1.0, False, self._default_params(mtype))
        self._layers.append(layer)
        self._refresh_layer_list()

    def _default_params(self, mtype: str) -> dict:
        defaults = {
            "range":      {"lo": 0.1, "hi": 0.8, "feather": 20.0},
            "star":       {"fwhm_guess": 5.0, "threshold_sigma": 5.0, "growth_factor": 2.0, "feather": 5.0},
            "edge":       {"method": "sobel", "pre_blur": 1.0, "post_blur": 3.0},
            "nebulosity": {"tile_size": 64, "sigma_above": 2.0, "feather": 10.0},
            "color":      {"hue_center": 300.0, "hue_range": 30.0, "saturation_min": 0.15, "feather": 5.0},
            "gradient":   {"mode": "radial", "cx": 0.5, "cy": 0.5, "radius": 0.5, "angle": 0.0, "invert": False},
        }
        return defaults.get(mtype, {})

    def _remove_layer(self):
        row = self._layer_list.currentRow()
        if 0 <= row < len(self._layers):
            self._layers.pop(row)
            self._refresh_layer_list()

    def _refresh_layer_list(self):
        self._layer_list.clear()
        for i, layer in enumerate(self._layers):
            icon = MASK_ICONS.get(layer["type"], "◆")
            op   = layer.get("operator", "AND") if i > 0 else ""
            text = f"{icon}  {layer['name']}"
            if op: text = f"[{op}]  " + text
            if not layer.get("enabled", True): text = f"(off)  " + text
            item = QListWidgetItem(text)
            if not layer.get("enabled", True):
                item.setForeground(QColor(SIRIL_TEXT_DIM))
            self._layer_list.addItem(item)

    def _preview_single_layer(self):
        if not self._data is not None:
            QMessageBox.warning(self, "No image", "Load a FITS file first."); return
        row = self._layer_list.currentRow()
        if row < 0 or row >= len(self._layers):
            QMessageBox.warning(self, "No layer", "Select a layer first."); return
        if self._data is None:
            QMessageBox.warning(self, "No image", "Load a FITS file first."); return
        self._cancel_event.clear()
        self._btn_run.setEnabled(False)
        worker = MaskWorker(self._layers, self._data, None,
                            self._cancel_event, single_layer_idx=row)
        worker.log_line.connect(self._log_msg)
        worker.preview_ready.connect(self._canvas.show_mask_and_overlay)
        worker.finished.connect(lambda r: self._btn_run.setEnabled(True))
        worker.start()
        self._status_lbl.setText(f"Previewing layer {row+1}…")

    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)
        self.log_line.emit(f"[MASK]  {msg}")

    def _run(self):
        if self._data is None:
            QMessageBox.warning(self, "No image", "Load a FITS file first."); return
        if not self._layers:
            QMessageBox.warning(self, "No layers", "Add at least one layer."); return
        out = self._out_edit.text().strip()
        if not out: QMessageBox.warning(self, "No output", "Set output path."); return
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        worker = MaskWorker(self._layers, self._data, out, self._cancel_event)
        worker.progress.connect(lambda c, t, m: (
            self._progress.setRange(0, max(t, 1)), self._progress.setValue(c),
            self._status_lbl.setText(f"Computing: {m}…")))
        worker.log_line.connect(self._log_msg)
        worker.preview_ready.connect(self._canvas.show_mask_and_overlay)
        worker.finished.connect(self._on_finished)
        self._worker = worker; worker.start()

    def _cancel(self):
        self._cancel_event.set()

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        if result.get("success"):
            cov = result.get("mask_coverage", 0)
            self._status_lbl.setText(
                f"✓ Done — coverage: {cov*100:.1f}%  → {result.get('output','')}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS};")
            self.finished.emit({"success": True, "output_path": result.get("output", "")})
        else:
            self._status_lbl.setText(f"✗ {result.get('error','')}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_ERROR};")
            self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Suite  —  Siril")
        self.resize(1400, 860)
        self._last_output = ""
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QWidget(); header.setFixedHeight(52)
        header.setStyleSheet(f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon  = QLabel("🎨"); lbl_icon.setStyleSheet("font-size:18pt; background:transparent;")
        lbl_title = QLabel("Image Processing Suite")
        lbl_title.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold; background:transparent;")
        lbl_ver   = QLabel("v1.0  |  5 scripts  |  Wavelet · Local Norm · Dehaze · Narrowband · Masks")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title); hl.addStretch(); hl.addWidget(lbl_ver)
        root.addWidget(header)

        # ── Shared input bar ──────────────────────────────────────────────────
        shared = QWidget()
        shared.setStyleSheet(f"background:{SIRIL_BG3}; border-bottom:1px solid {SIRIL_BORDER};")
        sl = QHBoxLayout(shared); sl.setContentsMargins(12, 6, 12, 6); sl.setSpacing(8)
        lbl_shared = QLabel("Shared input FITS:")
        lbl_shared.setStyleSheet(f"color:{SIRIL_SECTION}; font-weight:bold; background:transparent;")
        sl.addWidget(lbl_shared)
        self._shared_input_edit = QLineEdit()
        self._shared_input_edit.setPlaceholderText("Load any FITS — auto-fills all tools…")
        sl.addWidget(self._shared_input_edit, 1)
        btn_browse_shared = QPushButton("Browse…"); btn_browse_shared.setFixedWidth(80)
        btn_browse_shared.clicked.connect(self._browse_shared_input)
        sl.addWidget(btn_browse_shared)
        btn_apply_shared = QPushButton("→  Apply to all tools")
        btn_apply_shared.clicked.connect(self._apply_shared_input)
        btn_apply_shared.setStyleSheet(
            f"background:{SIRIL_ACCENT2}; color:white; font-weight:bold; border-radius:4px; padding:4px 12px;")
        sl.addWidget(btn_apply_shared)
        btn_chain = QPushButton("🔗  Chain output to next tool")
        btn_chain.setToolTip(
            "After each tool completes, its output is automatically\n"
            "set as the input for the next tool.")
        btn_chain.setCheckable(True); btn_chain.setChecked(False)
        self._chain_btn = btn_chain
        sl.addWidget(btn_chain)
        root.addWidget(shared)

        # ── Tool tabs ─────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        # Shared pipeline log
        self._pipeline_log = QPlainTextEdit(); self._pipeline_log.setReadOnly(True)

        # Tool panels
        self._wavelet_panel   = WaveletPanel()
        self._localnorm_panel = LocalNormPanel()
        self._dehaze_panel    = DarkChannelPanel()
        self._nb_panel        = NarrowbandPanel()
        self._mask_panel      = MaskBuilderPanel()

        for panel, label in [
            (self._wavelet_panel,   "〜  Wavelet Stretch"),
            (self._localnorm_panel, "⚖  Local Norm"),
            (self._dehaze_panel,    "🌫  Dehaze"),
            (self._nb_panel,        "🌈  Narrowband"),
            (self._mask_panel,      "🎭  Mask Builder"),
        ]:
            self._tabs.addTab(panel, label)

        # Pipeline log tab
        log_wrap = QWidget(); lw = QVBoxLayout(log_wrap); lw.setContentsMargins(6,6,6,6)
        lbl_log = QLabel("Combined log from all tools"); lbl_log.setObjectName("dim")
        lw.addWidget(lbl_log)
        btn_clr = QPushButton("Clear log"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._pipeline_log.clear)
        lw.addWidget(btn_clr); lw.addWidget(self._pipeline_log)
        self._tabs.addTab(log_wrap, "📋  Log")

        # Connect signals
        TOOL_ORDER = [
            (self._wavelet_panel,   self._localnorm_panel),
            (self._localnorm_panel, self._dehaze_panel),
            (self._dehaze_panel,    self._nb_panel),
            (self._nb_panel,        self._mask_panel),
            (self._mask_panel,      None),
        ]
        for i, (panel, next_panel) in enumerate(TOOL_ORDER):
            tab_idx = i + 1  # tab 0 would be overview if we had one
            panel.finished.connect(
                lambda r, np_=next_panel, ti=i: self._on_tool_done(r, np_, ti))
            panel.log_line.connect(self._pipeline_log.appendPlainText)

        # ── Bottom status bar ─────────────────────────────────────────────────
        bottom = QWidget(); bottom.setFixedHeight(32)
        bottom.setStyleSheet(f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(12, 0, 12, 0)
        self._status_lbl = QLabel("Ready — set shared input or open each tool individually.")
        self._status_lbl.setObjectName("dim")
        bl.addWidget(self._status_lbl, 1)
        root.addWidget(bottom)

        # Startup log
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] Image Processing Suite — ready")
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "Tools: Wavelet Stretch · Local Norm · Dark Channel Dehaze · "
            "Narrowband Palette · Mask Builder")

    # ── Shared input management ───────────────────────────────────────────────

    def _browse_shared_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FITS file", "",
            "FITS files (*.fit *.fits *.fts);;All files (*)")
        if path:
            self._shared_input_edit.setText(path)
            self._apply_shared_input()

    def _apply_shared_input(self):
        path = self._shared_input_edit.text().strip()
        if not path:
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "File not found",
                f"Cannot find:\n{path}"); return
        self._wavelet_panel.set_input(path)
        self._localnorm_panel.set_input(path)
        self._dehaze_panel.set_input(path)
        self._nb_panel.set_input(path)
        self._mask_panel.set_input(path)
        self._status_lbl.setText(
            f"Input applied to all tools: {os.path.basename(path)}")
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Shared input set: {path}")

    # ── Chain mode ────────────────────────────────────────────────────────────

    def _on_tool_done(self, result: dict, next_panel, tool_idx: int):
        tool_names = ["Wavelet", "Local Norm", "Dehaze", "Narrowband", "Mask"]
        name = tool_names[tool_idx] if tool_idx < len(tool_names) else f"Tool {tool_idx+1}"

        if result.get("success"):
            out = result.get("output_path", "")
            self._last_output = out
            msg = f"✓ {name} done"
            if out: msg += f" → {os.path.basename(out)}"
            self._status_lbl.setText(msg)
            self._status_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS};")
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {name}: {out}")

            # Chain mode: pipe output into next tool's input
            if self._chain_btn.isChecked() and next_panel is not None and out:
                if os.path.isfile(out):
                    next_panel.set_input(out)
                    self._pipeline_log.appendPlainText(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Chain: {os.path.basename(out)} → {tool_names[tool_idx+1] if tool_idx+1 < len(tool_names) else '?'}")
                    # Advance to next tab
                    self._tabs.setCurrentIndex(tool_idx + 1)
        else:
            err = result.get("error", "Unknown")
            self._status_lbl.setText(f"✗ {name} failed: {err}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_ERROR};")
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✗ {name} FAILED: {err}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  (Section A mandatory pattern)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import traceback
    import tempfile

    log_path = os.path.join(SCRIPT_DIR, "crash_log.txt")

    def crash_handler(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"CRASH  {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(msg)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = crash_handler

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
