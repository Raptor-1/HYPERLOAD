r"""
solar_system_suite.py  —  Script Merger: B7 Solar System Science
=================================================================
Merged launcher for:
    ☄  MODE A — Comet Dual-Stack Pipeline   (comet_pipeline.py)
    🪨  MODE B — Moving Object Detector      (moving_object.py)

Architecture: Section A merger pattern (two independent modes, not sequential).
The Overview tab shows a large mode badge and switches to the right sub-tool.
Each mode is fully operable standalone; no piped handoff between them (B7 spec).

Shared elements:
    - Same input folder widget (pre-filled in both tabs when set in Overview)
    - Equipment profile pixel scale used by both modes
    - Pipeline Log combined from both workers

Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:  Siril → Scripts → solar_system_suite
"""

from pathlib import Path
import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

try:
    s.ensure_installed("astroquery")
except Exception:
    pass

import os
import sys
import glob
import threading
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
# Never import MainWindow from children — only workers + pure functions.

from comet_pipeline import (
    CometWorker,
    NucleusPickerCanvas,
    TrackingCanvas,
    ResultPreviewCanvas,
    detect_comet_nucleus_auto,
    track_nucleus_sequence,
    compute_comet_shifts,
    stack_frames_mean,
    stack_color_frames_mean,
    make_comet_mask,
    match_backgrounds,
    composite_stacks,
    SIRIL_STYLESHEET,
    SIRIL_BG, SIRIL_BG2, SIRIL_BG3,
    SIRIL_ACCENT, SIRIL_ACCENT2,
    SIRIL_TEXT, SIRIL_TEXT_DIM,
    SIRIL_BORDER, SIRIL_SUCCESS,
    SIRIL_WARNING, SIRIL_SECTION,
    SIRIL_ERROR,
)

from moving_object import (
    MoverWorker,
    TrailCanvas,
    BlinkCanvas,
    detect_sources_in_frame,
    match_sources_across_frames,
    filter_moving_objects,
    fit_linear_motion,
    pixel_to_radec,
    estimate_pixel_scale,
    crossmatch_mpc,
    export_mpc_report,
    format_mpc_oneline,
    get_jd_from_header,
    HAS_ASTROQUERY,
    SIRIL_ASTEROID,
)

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QScrollArea, QFrame, QSizePolicy, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPen

# ── Merge stylesheet: add asteroid colour to base comet stylesheet ────────────
MERGED_STYLESHEET = SIRIL_STYLESHEET + f"""
QPushButton#asteroid {{
    background-color: #2d2000;
    border-color: {SIRIL_ASTEROID};
    color: {SIRIL_ASTEROID};
    font-weight: bold;
    text-align: center;
}}
QPushButton#asteroid:hover {{ background-color: #3d2d00; }}
QPushButton#nucleus {{
    background-color: #1a2d1a;
    border-color: {SIRIL_SUCCESS};
    color: {SIRIL_SUCCESS};
    text-align: center;
    font-weight: bold;
}}
QPushButton#nucleus:hover {{ background-color: #223322; }}
QPushButton#mpc_submit {{
    background-color: #1a2000;
    border-color: {SIRIL_ASTEROID};
    color: {SIRIL_ASTEROID};
    font-weight: bold;
    text-align: center;
}}
QPushButton#mpc_submit:hover {{ background-color: #2d3300; }}
QLabel#comet   {{ color: #4cff99; font-size: 28pt; font-weight: bold; }}
QLabel#asteroid_badge {{ color: {SIRIL_ASTEROID}; font-size: 28pt; font-weight: bold; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMET PANEL  (full Comet Pipeline UI, extracted from comet_pipeline.MainWindow)
# ─────────────────────────────────────────────────────────────────────────────

def _browse_folder_into(parent, line_edit: QLineEdit, title: str = "Select folder"):
    folder = QFileDialog.getExistingDirectory(parent, title)
    if folder:
        line_edit.setText(folder)


class CometPanel(QWidget):
    """
    Complete Comet Dual-Stack Pipeline UI.
    Exposes: get_config(), run(cancel_event), cancel()
    Signals: finished(dict), log_line(str)
    """
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker:             CometWorker | None = None
        self._cancel_event        = threading.Event()
        self._first_frame: np.ndarray | None = None
        self._auto_nucleus_pos:   tuple | None = None
        self._manual_nucleus_pos: tuple | None = None
        self._tracking_positions: list = []
        self._tracking_flags:     list = []
        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._build_tab_input()
        self._build_tab_settings()
        self._build_tab_run()
        self._build_tab_results()

        # Bottom bar
        bot = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run comet pipeline")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._on_run)
        bot.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(34)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        bot.addWidget(self._btn_cancel)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setValue(0)
        bot.addWidget(self._progress_bar, 1)

        root.addLayout(bot)

        self._status_lbl = QLabel("Ready. Set lights folder and run.")
        self._status_lbl.setObjectName("dim")
        root.addWidget(self._status_lbl)

    def _build_tab_input(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        grp_folder = QGroupBox("Lights folder")
        fl = QVBoxLayout(grp_folder)
        folder_row = QHBoxLayout()
        self._edit_lights = QLineEdit()
        self._edit_lights.setPlaceholderText("Path to calibrated light frames…")
        folder_row.addWidget(self._edit_lights)
        btn_browse = QPushButton("Browse…"); btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(
            lambda: _browse_folder_into(self, self._edit_lights, "Select lights folder"))
        folder_row.addWidget(btn_browse)
        fl.addLayout(folder_row)

        scan_row = QHBoxLayout()
        btn_scan = QPushButton("🔍  Scan frames")
        btn_scan.setObjectName("primary"); btn_scan.setFixedWidth(140)
        btn_scan.clicked.connect(self._on_scan_frames)
        scan_row.addWidget(btn_scan)
        self._lbl_frame_info = QLabel("No frames scanned.")
        self._lbl_frame_info.setObjectName("dim")
        scan_row.addWidget(self._lbl_frame_info, 1)
        fl.addLayout(scan_row)

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

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Comet name (optional):"))
        self._edit_comet_name = QLineEdit()
        self._edit_comet_name.setPlaceholderText("e.g. C2025A1")
        self._edit_comet_name.setFixedWidth(220)
        name_row.addWidget(self._edit_comet_name)
        name_row.addStretch()
        fl.addLayout(name_row)
        lay.addWidget(grp_folder)

        grp_nucleus = QGroupBox("Nucleus detection")
        nl = QVBoxLayout(grp_nucleus)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._combo_nucleus_mode = QComboBox()
        self._combo_nucleus_mode.addItems(["Automatic", "Manual (click on canvas)"])
        self._combo_nucleus_mode.setFixedWidth(220)
        self._combo_nucleus_mode.currentIndexChanged.connect(self._on_nucleus_mode_changed)
        mode_row.addWidget(self._combo_nucleus_mode); mode_row.addStretch()
        nl.addLayout(mode_row)

        self._auto_params_widget = QWidget()
        auto_form = QFormLayout(self._auto_params_widget)
        auto_form.setContentsMargins(0, 0, 0, 0)
        self._spin_blur = QDoubleSpinBox()
        self._spin_blur.setRange(3.0, 20.0); self._spin_blur.setValue(8.0)
        self._spin_blur.setSingleStep(0.5)
        auto_form.addRow("Blur sigma:", self._spin_blur)
        self._spin_search_radius = QSpinBox()
        self._spin_search_radius.setRange(20, 200); self._spin_search_radius.setValue(80)
        self._spin_search_radius.setSuffix(" px")
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

        self._manual_widget = QWidget()
        mv = QVBoxLayout(self._manual_widget)
        mv.setContentsMargins(0, 0, 0, 0)
        btn_load_frame = QPushButton("Load first frame for manual click")
        btn_load_frame.clicked.connect(self._on_load_first_frame)
        mv.addWidget(btn_load_frame)
        self._lbl_nucleus_pos = QLabel("No nucleus set.")
        self._lbl_nucleus_pos.setObjectName("dim")
        mv.addWidget(self._lbl_nucleus_pos)
        self._canvas_picker = NucleusPickerCanvas()
        self._canvas_picker.nucleus_selected.connect(self._on_nucleus_clicked)
        self._canvas_picker.setMinimumHeight(300)
        mv.addWidget(self._canvas_picker)
        self._manual_widget.setVisible(False)
        nl.addWidget(self._manual_widget)
        lay.addWidget(grp_nucleus)
        lay.addStretch()
        self._tabs.addTab(tab, "📂  Input & Nucleus")

    def _build_tab_settings(self):
        tab = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(8)
        scroll.setWidget(inner)
        outer = QVBoxLayout(tab); outer.setContentsMargins(0,0,0,0)
        outer.addWidget(scroll)

        grp_stack = QGroupBox("Stacking")
        sf = QFormLayout(grp_stack)
        self._combo_rejection = QComboBox()
        self._combo_rejection.addItems(["Sigma clip (recommended)", "None (mean)"])
        sf.addRow("Rejection:", self._combo_rejection)
        self._spin_sigma_low = QDoubleSpinBox()
        self._spin_sigma_low.setRange(1.0, 5.0); self._spin_sigma_low.setValue(3.0)
        self._spin_sigma_low.setSingleStep(0.5)
        sf.addRow("Sigma low:", self._spin_sigma_low)
        self._spin_sigma_high = QDoubleSpinBox()
        self._spin_sigma_high.setRange(1.0, 5.0); self._spin_sigma_high.setValue(3.0)
        self._spin_sigma_high.setSingleStep(0.5)
        sf.addRow("Sigma high:", self._spin_sigma_high)
        self._spin_max_frames = QSpinBox()
        self._spin_max_frames.setRange(5, 500); self._spin_max_frames.setValue(50)
        sf.addRow("Max frames in memory:", self._spin_max_frames)
        lay.addWidget(grp_stack)

        grp_mask = QGroupBox("Comet mask")
        mf = QFormLayout(grp_mask)
        self._spin_mask_radius = QSpinBox()
        self._spin_mask_radius.setRange(50, 500); self._spin_mask_radius.setValue(150)
        self._spin_mask_radius.setSuffix(" px")
        mf.addRow("Mask radius:", self._spin_mask_radius)
        self._spin_feather = QSpinBox()
        self._spin_feather.setRange(10, 150); self._spin_feather.setValue(50)
        self._spin_feather.setSuffix(" px")
        mf.addRow("Feather:", self._spin_feather)
        self._chk_nebulosity = QCheckBox("Use nebulosity detection (adaptive coma shape)")
        self._chk_nebulosity.setChecked(True)
        mf.addRow("", self._chk_nebulosity)
        self._spin_neb_sigma = QDoubleSpinBox()
        self._spin_neb_sigma.setRange(1.0, 5.0); self._spin_neb_sigma.setValue(2.0)
        self._spin_neb_sigma.setSingleStep(0.1)
        mf.addRow("Nebulosity sigma:", self._spin_neb_sigma)
        lay.addWidget(grp_mask)

        grp_out = QGroupBox("Output")
        of = QVBoxLayout(grp_out)
        out_row = QHBoxLayout()
        self._edit_output = QLineEdit()
        self._edit_output.setPlaceholderText("Default: same as lights folder")
        out_row.addWidget(self._edit_output)
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(80)
        btn_out.clicked.connect(
            lambda: _browse_folder_into(self, self._edit_output, "Select output folder"))
        out_row.addWidget(btn_out)
        of.addLayout(out_row)
        self._chk_load_siril = QCheckBox("Load composite in Siril after pipeline")
        self._chk_load_siril.setChecked(True)
        of.addWidget(self._chk_load_siril)
        self._chk_crop_overlap = QCheckBox("Crop to valid overlap region (removes border artifacts)")
        self._chk_crop_overlap.setChecked(False)
        of.addWidget(self._chk_crop_overlap)
        lay.addWidget(grp_out)
        lay.addStretch()
        self._tabs.addTab(tab, "⚙  Settings")

    def _build_tab_run(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        self._lbl_step = QLabel("Not started.")
        self._lbl_step.setObjectName("dim")
        lay.addWidget(self._lbl_step)

        self._lbl_drift = QLabel("")
        self._lbl_drift.setObjectName("dim")
        self._lbl_drift.setWordWrap(True)
        lay.addWidget(self._lbl_drift)

        self._canvas_tracking = TrackingCanvas()
        self._canvas_tracking.setMinimumHeight(300)
        lay.addWidget(self._canvas_tracking, 1)
        self._tabs.addTab(tab, "▶  Run")

    def _build_tab_results(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(4, 4, 4, 4)
        self._canvas_results = ResultPreviewCanvas()
        lay.addWidget(self._canvas_results)
        self._tabs.addTab(tab, "🖼  Results")

    # ── Config and run ─────────────────────────────────────────────────────────

    def set_lights_folder(self, folder: str):
        """Called by Overview to pre-fill the shared folder."""
        if not self._edit_lights.text().strip():
            self._edit_lights.setText(folder)

    def get_config(self) -> dict:
        lights = self._edit_lights.text().strip()
        mode   = self._combo_nucleus_mode.currentIndex()
        manual_pos = None if mode == 0 else self._manual_nucleus_pos
        rejection  = "sigma" if self._combo_rejection.currentIndex() == 0 else "none"
        return {
            "lights_dir":          lights,
            "output_dir":          self._edit_output.text().strip() or lights,
            "pre_registered":      self._combo_input_type.currentIndex() == 1,
            "comet_name":          self._edit_comet_name.text().strip(),
            "manual_nucleus_pos":  manual_pos,
            "blur_sigma":          self._spin_blur.value(),
            "search_radius":       self._spin_search_radius.value(),
            "rejection":           rejection,
            "sigma_low":           self._spin_sigma_low.value(),
            "sigma_high":          self._spin_sigma_high.value(),
            "max_frames_memory":   self._spin_max_frames.value(),
            "mask_radius":         self._spin_mask_radius.value(),
            "mask_feather":        self._spin_feather.value(),
            "use_nebulosity_mask": self._chk_nebulosity.isChecked(),
            "nebulosity_sigma":    self._spin_neb_sigma.value(),
            "load_in_siril":       self._chk_load_siril.isChecked(),
            "crop_overlap":        self._chk_crop_overlap.isChecked(),
        }

    def run(self, cancel_event: threading.Event | None = None):
        """External trigger (from Overview or internal run button)."""
        self._on_run(external_cancel=cancel_event)

    def cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()

    # ── Internal actions ──────────────────────────────────────────────────────

    def _on_nucleus_mode_changed(self, idx: int):
        self._auto_params_widget.setVisible(idx == 0)
        self._manual_widget.setVisible(idx == 1)

    def _on_scan_frames(self):
        folder = self._edit_lights.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Set a valid lights folder first.")
            return
        patterns = ["*.fit", "*.fits", "*.fts"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        if not files:
            self._lbl_frame_info.setText("⚠ No FITS files found.")
            return
        files.sort()
        try:
            from astropy.io import fits as _fits
            d = _fits.getdata(files[0]).astype(np.float32)
            shape = d.shape
            mem_gb = len(files) * d.nbytes / 1e9
            info = f"{len(files)} frames  |  shape: {shape}  |  Est. memory: {mem_gb:.1f} GB"
            self._lbl_frame_info.setText(info)
            self._lbl_frame_info.setObjectName("ok")
            self._first_frame = d if d.ndim == 2 else (
                (0.299*d[0]+0.587*d[1]+0.114*d[2]).astype(np.float32)
                if d.shape[0] == 3 else d[0])
        except Exception as exc:
            self._lbl_frame_info.setText(f"Error reading first frame: {exc}")

    def _on_auto_detect(self):
        if self._first_frame is None:
            self._on_scan_frames()
            if self._first_frame is None:
                QMessageBox.warning(self, "No frame", "Scan frames first."); return
        pos = detect_comet_nucleus_auto(self._first_frame,
                                         blur_sigma=self._spin_blur.value())
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

    def _on_load_first_frame(self):
        folder = self._edit_lights.text().strip()
        if not folder:
            QMessageBox.warning(self, "No folder", "Set the lights folder first."); return
        files = sorted(
            glob.glob(os.path.join(folder, "*.fit")) +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts"))
        )
        if not files:
            QMessageBox.warning(self, "No files", "No FITS files found."); return
        try:
            from astropy.io import fits as _fits
            d = _fits.getdata(files[0]).astype(np.float32)
            if d.ndim == 3:
                d = (0.299*d[0]+0.587*d[1]+0.114*d[2]).astype(np.float32) \
                    if d.shape[0] == 3 else d[0]
            self._first_frame = d
            self._canvas_picker.show_frame(d)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load frame:\n{exc}")

    def _on_nucleus_clicked(self, cy: float, cx: float):
        self._manual_nucleus_pos = (cy, cx)
        self._lbl_nucleus_pos.setText(f"Nucleus: X={cx:.1f}, Y={cy:.1f} (manual)")

    def _on_run(self, external_cancel: threading.Event | None = None):
        cfg = self.get_config()
        lights = cfg["lights_dir"]
        if not lights or not os.path.isdir(lights):
            QMessageBox.warning(self, "Input error", "Set a valid lights folder.")
            return
        if self._combo_nucleus_mode.currentIndex() == 1 and \
                self._manual_nucleus_pos is None:
            QMessageBox.warning(self, "No nucleus",
                "Manual mode: load the first frame and click the nucleus.")
            return

        cancel_ev = external_cancel or threading.Event()
        cancel_ev.clear()
        self._cancel_event = cancel_ev

        self._worker = CometWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log)
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
        self._btn_cancel.setEnabled(False)
        self._status_lbl.setText("Cancelling…")

    def _on_progress(self, step: int, total: int, desc: str):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(step)
        self._lbl_step.setText(f"Step {step}/{total}: {desc}")
        self._status_lbl.setText(desc)

    def _on_log(self, msg: str):
        self.log_line.emit(f"[COMET]  {msg}")

    def _on_tracking_done(self, positions, flags, first_frame):
        self._tracking_positions = positions
        self._tracking_flags = flags
        if first_frame is not None and len(first_frame.shape) == 2:
            self._canvas_tracking.show_tracking(first_frame, positions, flags)
        n = len(positions)
        if n > 1:
            dy = positions[-1][0] - positions[0][0]
            dx = positions[-1][1] - positions[0][1]
            drift = np.sqrt(dy**2 + dx**2)
            interp_pct = sum(flags) / max(n, 1) * 100
            self._lbl_drift.setText(
                f"Total drift: {drift:.1f} px over {n} frames  |  "
                f"Interpolated: {interp_pct:.0f}%")
            self._lbl_drift.setObjectName(
                "warn" if interp_pct > 20 else "ok")

    def _on_stacks_ready(self, stars, comet, mask, composite):
        self._canvas_results.update_all(stars, comet, mask, composite)
        self._tabs.setCurrentIndex(3)

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        if result.get("success"):
            drift = result.get("total_drift_px", 0.0)
            comp  = result.get("composite_path", "")
            self._status_lbl.setText(
                f"✓  Done  |  Drift: {drift:.1f} px  |  {comp}")
            self._status_lbl.setObjectName("ok")
            self._tabs.setCurrentIndex(3)
        else:
            err = result.get("error", "Unknown error")
            self._status_lbl.setText(f"✗  Error: {err}")
            self._status_lbl.setObjectName("err")
        self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# MOVING OBJECT PANEL  (full Moving Object Detector UI)
# ─────────────────────────────────────────────────────────────────────────────

class MovingObjectPanel(QWidget):
    """
    Complete Moving Object Detector UI.
    Exposes: get_config(), run(cancel_event), cancel()
    Signals: finished(dict), log_line(str)
    """
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker      = None
        self._cancel_event = threading.Event()
        self._movers      = []
        self._fits_files  = []
        self._first_frame = None

        self._blink_timer    = QTimer()
        self._blink_interval = 500
        self._blink_playing  = False
        self._blink_timer.timeout.connect(self._blink_advance)

        self._build_ui()
        self.trail_canvas.show_placeholder()
        self.blink_canvas.show_placeholder()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        root.addWidget(splitter, 1)

        # Left settings panel
        left = QWidget()
        left.setMinimumWidth(290); left.setMaximumWidth(350)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(6)
        self._build_left_panel(left_lay)
        splitter.addWidget(left)

        # Right results panel
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(6)
        self._build_right_panel(right_lay)
        splitter.addWidget(right)
        splitter.setSizes([310, 760])

        # Bottom bar
        bot = QWidget()
        bot.setStyleSheet(f"background:{SIRIL_BG3}; border-top:1px solid {SIRIL_BORDER};")
        bot_lay = QHBoxLayout(bot)
        bot_lay.setContentsMargins(10, 6, 10, 6); bot_lay.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False); self.progress_bar.setFixedHeight(8)
        bot_lay.addWidget(self.progress_bar, 1)

        self.btn_run = QPushButton("▶  Detect movers")
        self.btn_run.setObjectName("primary"); self.btn_run.setFixedWidth(160)
        self.btn_run.clicked.connect(self._start_detection)
        bot_lay.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger"); self.btn_cancel.setFixedWidth(100)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_detection)
        bot_lay.addWidget(self.btn_cancel)
        root.addWidget(bot)

        self.status_label = QLabel("Ready — load a FITS sequence folder and scan.")
        self.status_label.setStyleSheet(
            f"background:{SIRIL_BG2}; color:{SIRIL_TEXT_DIM}; "
            f"padding:3px 10px; font-size:9pt;")
        self.status_label.setFixedHeight(22)
        root.addWidget(self.status_label)

    def _build_left_panel(self, lay):
        grp_input = QGroupBox("Input sequence")
        g_lay = QVBoxLayout(grp_input)
        row_folder = QHBoxLayout()
        self.edit_folder = QLineEdit()
        self.edit_folder.setPlaceholderText("FITS sequence folder…")
        btn_browse = QPushButton("Browse"); btn_browse.setFixedWidth(60)
        btn_browse.clicked.connect(self._browse_folder)
        row_folder.addWidget(self.edit_folder); row_folder.addWidget(btn_browse)
        g_lay.addLayout(row_folder)
        lbl_hint = QLabel("Frames must be time-sorted with DATE-OBS headers")
        lbl_hint.setObjectName("dim"); g_lay.addWidget(lbl_hint)
        btn_scan = QPushButton("🔍  Scan"); btn_scan.clicked.connect(self._scan_folder)
        g_lay.addWidget(btn_scan)
        self.lbl_scan_result = QLabel("No folder selected.")
        self.lbl_scan_result.setObjectName("dim"); self.lbl_scan_result.setWordWrap(True)
        g_lay.addWidget(self.lbl_scan_result)
        self.lbl_no_ts = QLabel("⚠ No timestamps — frame order used as time proxy")
        self.lbl_no_ts.setObjectName("warn"); self.lbl_no_ts.setWordWrap(True)
        self.lbl_no_ts.setVisible(False); g_lay.addWidget(self.lbl_no_ts)
        lay.addWidget(grp_input)

        grp_det = QGroupBox("Detection settings")
        f_lay = QFormLayout(grp_det)
        f_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(3.0, 15.0); self.spin_threshold.setValue(5.0)
        self.spin_threshold.setSingleStep(0.5); self.spin_threshold.setSuffix(" σ")
        f_lay.addRow("Detection threshold:", self.spin_threshold)
        self.spin_match_radius = QDoubleSpinBox()
        self.spin_match_radius.setRange(3.0, 30.0); self.spin_match_radius.setValue(8.0)
        self.spin_match_radius.setSuffix(" px")
        f_lay.addRow("Match radius:", self.spin_match_radius)
        self.spin_min_frames = QSpinBox()
        self.spin_min_frames.setRange(2, 99); self.spin_min_frames.setValue(3)
        f_lay.addRow("Min frames:", self.spin_min_frames)
        self.spin_min_motion = QDoubleSpinBox()
        self.spin_min_motion.setRange(1.0, 50.0); self.spin_min_motion.setValue(3.0)
        self.spin_min_motion.setSuffix(" px")
        f_lay.addRow("Min motion:", self.spin_min_motion)
        self.spin_max_speed = QDoubleSpinBox()
        self.spin_max_speed.setRange(5.0, 500.0); self.spin_max_speed.setValue(200.0)
        self.spin_max_speed.setSuffix(" px/f")
        f_lay.addRow("Max speed:", self.spin_max_speed)
        self.spin_r2 = QDoubleSpinBox()
        self.spin_r2.setRange(0.70, 1.00); self.spin_r2.setValue(0.90)
        self.spin_r2.setSingleStep(0.01); self.spin_r2.setDecimals(2)
        f_lay.addRow("R² threshold:", self.spin_r2)
        lay.addWidget(grp_det)

        grp_mpc = QGroupBox("MPC cross-match")
        m_lay = QVBoxLayout(grp_mpc)
        self.chk_mpc = QCheckBox("Cross-match MPC catalog")
        self.chk_mpc.setChecked(HAS_ASTROQUERY)
        if not HAS_ASTROQUERY:
            lbl_aq = QLabel("⚠ astroquery not installed — MPC disabled")
            lbl_aq.setObjectName("warn"); m_lay.addWidget(lbl_aq)
        m_lay.addWidget(self.chk_mpc)
        mf_lay = QFormLayout()
        self.spin_mpc_radius = QDoubleSpinBox()
        self.spin_mpc_radius.setRange(5.0, 120.0); self.spin_mpc_radius.setValue(30.0)
        self.spin_mpc_radius.setSuffix("\"")
        mf_lay.addRow("Search radius:", self.spin_mpc_radius)
        self.edit_obs_code = QLineEdit("500"); self.edit_obs_code.setMaxLength(3)
        self.edit_obs_code.setFixedWidth(60)
        mf_lay.addRow("Observatory code:", self.edit_obs_code)
        lbl_obs = QLabel("500 = geocenter if unknown"); lbl_obs.setObjectName("dim")
        mf_lay.addRow("", lbl_obs)
        m_lay.addLayout(mf_lay)
        lay.addWidget(grp_mpc)

        grp_out = QGroupBox("Output")
        o_lay = QVBoxLayout(grp_out)
        row_out = QHBoxLayout()
        self.edit_outdir = QLineEdit()
        self.edit_outdir.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse"); btn_out.setFixedWidth(60)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self.edit_outdir); row_out.addWidget(btn_out)
        o_lay.addLayout(row_out)
        self.chk_export_mpc = QCheckBox("Export MPC report"); self.chk_export_mpc.setChecked(True)
        o_lay.addWidget(self.chk_export_mpc)
        self.chk_export_csv = QCheckBox("Export CSV"); self.chk_export_csv.setChecked(True)
        o_lay.addWidget(self.chk_export_csv)
        lay.addWidget(grp_out)
        lay.addStretch()

    def _build_right_panel(self, lay):
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        tab_trail = QWidget()
        tl = QVBoxLayout(tab_trail); tl.setContentsMargins(4, 4, 4, 4)
        self.trail_canvas = TrailCanvas()
        tl.addWidget(self.trail_canvas, 1)
        self.tabs.addTab(tab_trail, "☄  Trail Map")

        tab_det = QWidget()
        dl = QVBoxLayout(tab_det); dl.setContentsMargins(6, 6, 6, 6)
        badge_row = QHBoxLayout()
        self.lbl_total   = QLabel("Total: 0")
        self.lbl_known   = QLabel("Known: 0")
        self.lbl_unknown = QLabel("Unknown: 0")
        self.lbl_known.setStyleSheet(f"color:{SIRIL_ASTEROID}; font-weight:bold;")
        self.lbl_unknown.setStyleSheet(f"color:#ff6b22; font-weight:bold;")
        badge_row.addWidget(self.lbl_total)
        badge_row.addWidget(self.lbl_known)
        badge_row.addWidget(self.lbl_unknown)
        badge_row.addStretch()
        dl.addLayout(badge_row)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "#", "Status", "Designation", "X_start", "Y_start",
            "Motion(px)", "Speed(px/f)", "R²", "RA", "Dec"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_table_double_click)
        dl.addWidget(self.table, 1)
        btn_row = QHBoxLayout()
        btn_csv = QPushButton("💾  Export CSV"); btn_csv.setObjectName("export")
        btn_csv.clicked.connect(self._manual_export_csv)
        btn_mpc_btn = QPushButton("📡  Export MPC report"); btn_mpc_btn.setObjectName("asteroid")
        btn_mpc_btn.clicked.connect(self._manual_export_mpc)
        btn_row.addWidget(btn_csv); btn_row.addWidget(btn_mpc_btn); btn_row.addStretch()
        dl.addLayout(btn_row)
        # MPC submission helper (B7 spec)
        self._btn_mpc_portal = QPushButton("🌐  Open MPC submission portal")
        self._btn_mpc_portal.setObjectName("mpc_submit")
        self._btn_mpc_portal.setEnabled(False)
        self._btn_mpc_portal.clicked.connect(self._open_mpc_portal)
        self._btn_copy_mpc_report = QPushButton("📋  Copy MPC report to clipboard")
        self._btn_copy_mpc_report.setEnabled(False)
        self._btn_copy_mpc_report.clicked.connect(self._copy_mpc_to_clipboard)
        portal_row = QHBoxLayout()
        portal_row.addWidget(self._btn_mpc_portal)
        portal_row.addWidget(self._btn_copy_mpc_report)
        portal_row.addStretch()
        dl.addLayout(portal_row)
        self.tabs.addTab(tab_det, "📋  Detections")

        tab_blink = QWidget()
        bl = QVBoxLayout(tab_blink); bl.setContentsMargins(4, 4, 4, 4)
        self.blink_canvas = BlinkCanvas()
        bl.addWidget(self.blink_canvas, 1)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Speed:"))
        self.slider_blink = QSlider(Qt.Orientation.Horizontal)
        self.slider_blink.setRange(100, 3000); self.slider_blink.setValue(500)
        self.slider_blink.valueChanged.connect(self._set_blink_speed)
        ctrl_row.addWidget(self.slider_blink, 1)
        self.lbl_blink_speed = QLabel("0.5s/frame")
        ctrl_row.addWidget(self.lbl_blink_speed)
        self.btn_blink = QPushButton("▶  Play"); self.btn_blink.setFixedWidth(90)
        self.btn_blink.clicked.connect(self._toggle_blink)
        ctrl_row.addWidget(self.btn_blink)
        bl.addLayout(ctrl_row)
        lbl_blink_hint = QLabel("Double-click a row in Detections to load object")
        lbl_blink_hint.setObjectName("dim")
        lbl_blink_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(lbl_blink_hint)
        self.tabs.addTab(tab_blink, "🎬  Blink")

        tab_log = QWidget()
        ll = QVBoxLayout(tab_log); ll.setContentsMargins(4, 4, 4, 4)
        self.log_edit = QPlainTextEdit(); self.log_edit.setReadOnly(True)
        ll.addWidget(self.log_edit)
        self.tabs.addTab(tab_log, "📋  Log")

    # ── Public interface ──────────────────────────────────────────────────────

    def set_folder(self, folder: str):
        if not self.edit_folder.text().strip():
            self.edit_folder.setText(folder)
            if not self.edit_outdir.text():
                self.edit_outdir.setText(folder)

    def get_config(self) -> dict:
        return {
            "fits_files":        self._fits_files,
            "threshold_sigma":   self.spin_threshold.value(),
            "fwhm_guess":        5.0,
            "max_sources":       500,
            "match_radius_px":   self.spin_match_radius.value(),
            "min_frames":        self.spin_min_frames.value(),
            "min_motion_px":     self.spin_min_motion.value(),
            "max_motion_px":     self.spin_max_speed.value(),
            "min_r2":            self.spin_r2.value(),
            "crossmatch_mpc":    self.chk_mpc.isChecked(),
            "mpc_search_arcsec": self.spin_mpc_radius.value(),
            "observatory_code":  self.edit_obs_code.text().strip() or "500",
            "export_mpc":        self.chk_export_mpc.isChecked(),
            "export_csv":        self.chk_export_csv.isChecked(),
            "output_dir":        self.edit_outdir.text().strip() or ".",
            "blink_cutout_size": 80,
        }

    def run(self, cancel_event: threading.Event | None = None):
        self._start_detection(external_cancel=cancel_event)

    def cancel(self):
        if self._worker:
            self._worker.cancel()

    # ── Detection actions ─────────────────────────────────────────────────────

    def _start_detection(self, external_cancel: threading.Event | None = None):
        if not self._fits_files:
            QMessageBox.warning(self, "No files", "Scan a folder first."); return
        if len(self._fits_files) < 3:
            QMessageBox.warning(self, "Too few frames",
                                "Need at least 3 frames for motion detection."); return
        self.table.setRowCount(0)
        self._movers = []
        self._update_badges()
        self.log_edit.clear()
        self.trail_canvas.show_placeholder()
        self.blink_canvas.show_placeholder()
        self._btn_mpc_portal.setEnabled(False)
        self._btn_copy_mpc_report.setEnabled(False)

        cancel_ev = external_cancel or threading.Event()
        cancel_ev.clear()
        self._cancel_event = cancel_ev

        self._worker = MoverWorker(self.get_config(), self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log)
        self._worker.mover_found.connect(self._on_mover_found)
        self._worker.finished.connect(self._on_finished)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._set_status("Running detection…")
        self._worker.start()

    def _cancel_detection(self):
        if self._worker:
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)
        self._set_status("Cancelling…")

    def _on_progress(self, val, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(val)
        self._set_status(msg)

    def _on_log(self, line: str):
        self.log_edit.appendPlainText(line)
        self.log_line.emit(f"[ASTEROID]  {line}")

    def _on_mover_found(self, data: dict):
        track  = data["track"]
        fit    = data["fit"]
        mpc    = data["mpc_match"]
        ra     = data["ra"]
        dec    = data["dec"]
        self._movers.append((track, fit, mpc, ra, dec))
        self._update_badges()
        row = self.table.rowCount()
        self.table.insertRow(row)
        vals = [
            str(row + 1),
            "Known ★" if mpc else "Unknown ?",
            mpc["designation"] if mpc else "",
            f"{fit['x_start']:.1f}", f"{fit['y_start']:.1f}",
            f"{fit['total_motion_px']:.1f}", f"{fit['speed_px_per_frame']:.3f}",
            f"{fit['r2_mean']:.3f}",
            f"{ra:.4f}" if ra is not None else "—",
            f"{dec:.4f}" if dec is not None else "—",
        ]
        color = QColor("#332800") if mpc else QColor("#2a1500")
        for col, val in enumerate(vals):
            item = QTableWidgetItem(val)
            item.setBackground(color)
            self.table.setItem(row, col, item)

    def _on_finished(self, result: dict):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if not result["success"]:
            self._set_status(f"✗  Error: {result.get('error', 'Unknown')}")
            self.log_edit.appendPlainText(f"[FAILED] {result.get('error','')}")
            self.finished.emit(result)
            return

        n   = result["n_movers"]
        n_k = result["n_known"]
        n_u = result["n_unknown"]
        self._set_status(
            f"✓  {n} movers found  |  Known: {n_k}  |  Unknown: {n_u}")
        if n_u > 0:
            self._btn_mpc_portal.setEnabled(True)
            self._btn_copy_mpc_report.setEnabled(True)

        if self._fits_files and n > 0:
            try:
                from astropy.io import fits as _fits
                with _fits.open(self._fits_files[0]) as hdul:
                    frame = hdul[0].data.astype(np.float32)
                if frame.ndim == 3:
                    frame = frame[0] if frame.shape[0] <= 4 else \
                            0.299*frame[0]+0.587*frame[1]+0.114*frame[2]
                self.trail_canvas.show_trails(frame, self._movers)
                self.tabs.setTabText(0, f"☄  Trail Map  [{n}]")
            except Exception as e:
                self.log_edit.appendPlainText(f"Trail map error: {e}")

        self.tabs.setTabText(1, f"📋  Detections  [{n}]")
        self.log_edit.appendPlainText(f"\n[DONE] {n} movers  |  Known={n_k}  Unknown={n_u}")
        if not result.get("pixel_scale"):
            self.log_edit.appendPlainText(
                "\n⚠ No WCS found — MPC cross-match unavailable.\n"
                "  Tip: Plate-solve your frames in Siril before running.")
        self.finished.emit(result)

    def _on_table_double_click(self, index):
        row = index.row()
        if row >= len(self._movers): return
        track, fit, mpc, ra, dec = self._movers[row]
        self.blink_canvas.load_mover_cutouts(track, self._fits_files, cutout_size=80)
        self.tabs.setCurrentIndex(2)
        if not self._blink_playing:
            self._toggle_blink()

    def _manual_export_csv(self):
        if not self._movers:
            QMessageBox.information(self, "No data", "No movers detected yet."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "moving_objects.csv", "CSV (*.csv)")
        if path:
            w = MoverWorker.__new__(MoverWorker)
            w._export_csv(self._movers, path, None)
            self._set_status(f"CSV saved: {path}")

    def _manual_export_mpc(self):
        if not self._movers:
            QMessageBox.information(self, "No data", "No movers detected yet."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save MPC report", "mpc_report.txt", "Text (*.txt)")
        if path:
            obs = self.edit_obs_code.text().strip() or "500"
            first_hdr = None
            if self._fits_files:
                try:
                    from astropy.io import fits as _fits
                    with _fits.open(self._fits_files[0]) as hdul:
                        first_hdr = hdul[0].header.copy()
                except Exception:
                    pass
            export_mpc_report(self._movers, 1.0, obs, path, first_hdr,
                              log_callback=self._on_log)
            self._set_status(f"MPC report saved: {path}")

    def _open_mpc_portal(self):
        import subprocess, sys
        url = "https://www.minorplanetcenter.net/iau/info/ObsNote.html"
        try:
            if sys.platform == "win32":
                subprocess.Popen(["start", url], shell=True)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["xdg-open", url])
        except Exception as e:
            QMessageBox.information(self, "MPC Portal",
                f"Open this URL in your browser:\n{url}\n\n(Auto-open failed: {e})")

    def _copy_mpc_to_clipboard(self):
        """Build a quick MPC report in memory and copy to clipboard."""
        if not self._movers:
            return
        import io
        obs = self.edit_obs_code.text().strip() or "500"
        lines = [
            f"COD {obs}",
            "OBS Unknown",
            "COM Generated by Siril solar_system_suite.py",
            "---",
        ]
        first_hdr = None
        if self._fits_files:
            try:
                from astropy.io import fits as _fits
                with _fits.open(self._fits_files[0]) as hdul:
                    first_hdr = hdul[0].header.copy()
            except Exception:
                pass
        for i, (track, fit, mpc, ra, dec) in enumerate(self._movers):
            if mpc is not None or ra is None:
                continue
            src0 = track[0]
            jd   = src0.get("jd", 0.0)
            try:
                line = format_mpc_oneline(
                    ra, dec, jd, None, "V", obs,
                    temp_designation=f"TMP{i+1:04d} ")
                lines.append(line)
            except Exception:
                continue
        lines.append("---")
        report = "\n".join(lines)
        QApplication.clipboard().setText(report)
        self._set_status(f"Copied MPC report ({len(lines)-5} observations) to clipboard")

    # ── Folder and blink helpers ───────────────────────────────────────────────

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self.edit_folder.setText(d)
            if not self.edit_outdir.text():
                self.edit_outdir.setText(d)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.edit_outdir.setText(d)

    def _scan_folder(self):
        folder = self.edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self.lbl_scan_result.setText("Invalid folder."); return
        patterns = ["*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        files = sorted(set(files))
        if not files:
            self.lbl_scan_result.setText("No FITS files found."); return

        has_ts = False
        jd_list = []
        for path in files[:min(5, len(files))]:
            try:
                from astropy.io import fits as _fits
                with _fits.open(path) as hdul:
                    jd = get_jd_from_header(hdul[0].header)
                    if jd > 0:
                        jd_list.append(jd); has_ts = True
            except Exception:
                pass

        self._fits_files = files
        if not self.edit_outdir.text():
            self.edit_outdir.setText(folder)

        if has_ts and len(jd_list) >= 2:
            span_min = (max(jd_list) - min(jd_list)) * 1440
            self.lbl_scan_result.setText(
                f"{len(files)} FITS files found\n"
                f"Time span ≈ {span_min:.1f} min")
        else:
            self.lbl_scan_result.setText(f"{len(files)} FITS files found")

        self.lbl_no_ts.setVisible(not has_ts)
        self.spin_min_frames.setMaximum(len(files))
        self.spin_min_frames.setValue(max(3, len(files) - 1))
        self._set_status(f"Scanned: {len(files)} frames in {folder}")

    def _toggle_blink(self):
        if self._blink_playing:
            self._blink_timer.stop(); self._blink_playing = False
            self.btn_blink.setText("▶  Play")
        else:
            self._blink_timer.start(self._blink_interval); self._blink_playing = True
            self.btn_blink.setText("⏸  Pause")

    def _set_blink_speed(self, ms: int):
        self._blink_interval = ms
        self.lbl_blink_speed.setText(f"{ms/1000:.1f}s/frame")
        if self._blink_playing:
            self._blink_timer.setInterval(ms)

    def _blink_advance(self):
        self.blink_canvas.advance()

    def _update_badges(self):
        n_k = sum(1 for _, _, m, _, _ in self._movers if m is not None)
        n_u = sum(1 for _, _, m, _, _ in self._movers if m is None)
        n   = len(self._movers)
        self.lbl_total.setText(f"Total: {n}")
        self.lbl_known.setText(f"Known: {n_k}")
        self.lbl_unknown.setText(f"Unknown: {n_u}")

    def _set_status(self, msg: str):
        self.status_label.setText(msg)


# ─────────────────────────────────────────────────────────────────────────────
# MODE BADGE WIDGET  (B7 spec: large prominent mode indicator)
# ─────────────────────────────────────────────────────────────────────────────

class ModeBadge(QWidget):
    """
    Large mode indicator. Shows ☄ COMET MODE or 🪨 ASTEROID MODE
    with a coloured border. Clicking toggles the mode.
    """
    clicked = pyqtSignal(str)  # emits "comet" or "asteroid"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "comet"
        self.setFixedHeight(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_ui()

    def _build_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 4, 16, 4)

        self._icon_lbl = QLabel("☄")
        self._icon_lbl.setStyleSheet("font-size:36pt; background:transparent;")
        lay.addWidget(self._icon_lbl)

        text_col = QVBoxLayout()
        self._mode_lbl = QLabel("COMET MODE")
        self._mode_lbl.setStyleSheet(
            f"font-size:16pt; font-weight:bold; color:#4cff99; background:transparent;")
        self._desc_lbl = QLabel("Dual-stack pipeline: separate comet + star composites")
        self._desc_lbl.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        text_col.addWidget(self._mode_lbl)
        text_col.addWidget(self._desc_lbl)
        lay.addLayout(text_col, 1)

        hint = QLabel("(click to switch mode)")
        hint.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt; background:transparent;")
        lay.addWidget(hint)

    def set_mode(self, mode: str):
        self._mode = mode
        if mode == "comet":
            self._icon_lbl.setText("☄")
            self._mode_lbl.setText("COMET MODE")
            self._mode_lbl.setStyleSheet(
                "font-size:16pt; font-weight:bold; color:#4cff99; background:transparent;")
            self._desc_lbl.setText("Dual-stack pipeline: separate comet + star composites")
            self.setStyleSheet(
                f"ModeBadge {{ background:{SIRIL_BG2}; border:2px solid #4cff99; "
                f"border-radius:8px; }}")
        else:
            self._icon_lbl.setText("🪨")
            self._mode_lbl.setText("ASTEROID MODE")
            self._mode_lbl.setStyleSheet(
                f"font-size:16pt; font-weight:bold; color:{SIRIL_ASTEROID}; background:transparent;")
            self._desc_lbl.setText("Source detection → cross-frame motion → MPC cross-match")
            self.setStyleSheet(
                f"ModeBadge {{ background:{SIRIL_BG2}; border:2px solid {SIRIL_ASTEROID}; "
                f"border-radius:8px; }}")

    def mousePressEvent(self, event):
        new_mode = "asteroid" if self._mode == "comet" else "comet"
        self.set_mode(new_mode)
        self.clicked.emit(new_mode)


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW TAB  (B7 spec: mode selector, shared input, summaries)
# ─────────────────────────────────────────────────────────────────────────────

class OverviewTab(QWidget):
    mode_changed   = pyqtSignal(str)
    run_requested  = pyqtSignal(str)   # emits "comet" or "asteroid"
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: dict | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(12)

        # Mode badge
        self._badge = ModeBadge()
        self._badge.set_mode("comet")
        self._badge.clicked.connect(self._on_badge_click)
        root.addWidget(self._badge)

        # Mode toggle buttons
        mode_row = QHBoxLayout()
        self._btn_comet = QPushButton("☄  Comet Mode")
        self._btn_comet.setObjectName("primary")
        self._btn_comet.setMinimumHeight(36)
        self._btn_comet.clicked.connect(lambda: self._set_mode("comet"))
        self._btn_asteroid = QPushButton("🪨  Asteroid Mode")
        self._btn_asteroid.setMinimumHeight(36)
        self._btn_asteroid.clicked.connect(lambda: self._set_mode("asteroid"))
        mode_row.addWidget(self._btn_comet)
        mode_row.addWidget(self._btn_asteroid)
        root.addLayout(mode_row)

        # Shared input folder
        grp_shared = QGroupBox("Shared input folder")
        sl = QVBoxLayout(grp_shared)
        lbl_shared = QLabel(
            "Set here to pre-fill the active mode's folder. "
            "Each mode also has its own folder field.")
        lbl_shared.setObjectName("dim"); lbl_shared.setWordWrap(True)
        sl.addWidget(lbl_shared)
        shared_row = QHBoxLayout()
        self._edit_shared_folder = QLineEdit()
        self._edit_shared_folder.setPlaceholderText("FITS folder (fills both modes)…")
        btn_shared = QPushButton("Browse"); btn_shared.setFixedWidth(70)
        btn_shared.clicked.connect(self._browse_shared_folder)
        shared_row.addWidget(self._edit_shared_folder); shared_row.addWidget(btn_shared)
        sl.addLayout(shared_row)
        self._btn_apply_folder = QPushButton("Apply to current mode")
        self._btn_apply_folder.clicked.connect(self._apply_shared_folder)
        sl.addWidget(self._btn_apply_folder)
        root.addWidget(grp_shared)

        # Result summary (shown after run)
        self._grp_result = QGroupBox("Last result")
        self._grp_result.setVisible(False)
        rl = QVBoxLayout(self._grp_result)
        self._lbl_result = QLabel("")
        self._lbl_result.setWordWrap(True)
        rl.addWidget(self._lbl_result)
        root.addWidget(self._grp_result)

        # No-WCS note for asteroid mode
        self._lbl_wcs_note = QLabel(
            "💡 Tip: Plate-solve frames in Siril before asteroid mode for MPC cross-match.")
        self._lbl_wcs_note.setObjectName("dim")
        self._lbl_wcs_note.setWordWrap(True)
        self._lbl_wcs_note.setVisible(False)
        root.addWidget(self._lbl_wcs_note)

        root.addStretch()

        # Action row
        action_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(36); self._btn_run.setMinimumWidth(130)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(36)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self.cancel_requested)
        action_row.addWidget(self._btn_run)
        action_row.addWidget(self._btn_cancel)
        action_row.addStretch()
        root.addLayout(action_row)

    def get_mode(self) -> str:
        return self._badge._mode

    def set_running(self, running: bool):
        self._btn_run.setEnabled(not running)
        self._btn_comet.setEnabled(not running)
        self._btn_asteroid.setEnabled(not running)
        self._btn_cancel.setEnabled(running)

    def show_result(self, mode: str, result: dict):
        self._grp_result.setVisible(True)
        if mode == "comet":
            drift = result.get("total_drift_px", 0.0)
            comp  = result.get("composite_path", "—")
            interp = result.get("interp_pct", 0.0)
            self._lbl_result.setText(
                f"☄ Comet pipeline complete\n"
                f"Total drift: {drift:.1f} px  |  Interpolated: {interp:.0f}%\n"
                f"Composite saved: {comp}")
        else:
            n   = result.get("n_movers", 0)
            n_k = result.get("n_known", 0)
            n_u = result.get("n_unknown", 0)
            self._lbl_result.setText(
                f"🪨 Asteroid detection complete\n"
                f"Found {n} moving objects  |  Known: {n_k}  |  "
                f"Unknown: {n_u}"
                + (" ★ — candidates for MPC submission" if n_u > 0 else ""))
        self._grp_result.setTitle(
            f"Last result — {'☄ Comet' if mode == 'comet' else '🪨 Asteroid'}")

    def _on_badge_click(self, mode: str):
        self._set_mode(mode)

    def _set_mode(self, mode: str):
        self._badge.set_mode(mode)
        self._lbl_wcs_note.setVisible(mode == "asteroid")
        is_comet = mode == "comet"
        self._btn_comet.setObjectName("primary" if is_comet else "")
        self._btn_asteroid.setObjectName("" if is_comet else "primary")
        self._btn_comet.style().unpolish(self._btn_comet)
        self._btn_comet.style().polish(self._btn_comet)
        self._btn_asteroid.style().unpolish(self._btn_asteroid)
        self._btn_asteroid.style().polish(self._btn_asteroid)
        mode_label = "☄ Comet" if is_comet else "🪨 Asteroid"
        self._btn_run.setText(f"▶  Run {mode_label}")
        self.mode_changed.emit(mode)

    def _browse_shared_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self._edit_shared_folder.setText(d)

    def _apply_shared_folder(self):
        folder = self._edit_shared_folder.text().strip()
        if folder:
            self.mode_changed.emit(f"apply_folder:{folder}")

    def _on_run(self):
        self.run_requested.emit(self.get_mode())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solar System Science Suite  —  Siril")
        self.resize(1380, 860)

        self._current_worker = None
        self._cancel_event   = threading.Event()
        self._current_mode   = "comet"

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ─────────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(
            f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon  = QLabel("🌌")
        lbl_icon.setStyleSheet("font-size:18pt; background:transparent;")
        lbl_title = QLabel("Solar System Science Suite")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ASTEROID}; font-size:13pt; font-weight:bold; background:transparent;")
        lbl_ver   = QLabel("v1.0  |  2 scripts  |  Comet Pipeline + Moving Object Detector")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon)
        hl.addWidget(lbl_title)
        hl.addStretch()
        hl.addWidget(lbl_ver)
        root.addWidget(header)

        # ── Tab widget ─────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        # Shared pipeline log
        self._pipeline_log = QPlainTextEdit()
        self._pipeline_log.setReadOnly(True)

        # Tab 0 — Overview
        self._overview = OverviewTab()
        self._overview.mode_changed.connect(self._on_mode_changed)
        self._overview.run_requested.connect(self._run_mode)
        self._overview.cancel_requested.connect(self._cancel)
        self._tabs.addTab(self._overview, "🌌  Overview")

        # Tab 1 — Comet Pipeline
        self._comet_panel = CometPanel()
        self._comet_panel.finished.connect(
            lambda r: self._on_mode_finished("comet", r))
        self._comet_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._comet_panel, "☄  Comet Pipeline")

        # Tab 2 — Moving Object Detector
        self._asteroid_panel = MovingObjectPanel()
        self._asteroid_panel.finished.connect(
            lambda r: self._on_mode_finished("asteroid", r))
        self._asteroid_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._asteroid_panel, "🪨  Moving Object Detector")

        # Tab 3 — Pipeline Log
        log_wrap = QWidget()
        lw = QVBoxLayout(log_wrap)
        lw.setContentsMargins(6, 6, 6, 6)
        lbl_log = QLabel("Combined log from all pipeline steps")
        lbl_log.setObjectName("dim")
        lw.addWidget(lbl_log)
        btn_clear = QPushButton("Clear log"); btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._pipeline_log.clear)
        lw.addWidget(btn_clear)
        lw.addWidget(self._pipeline_log)
        self._tabs.addTab(log_wrap, "📋  Pipeline Log")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom = QWidget()
        bottom.setFixedHeight(46)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(10, 4, 10, 4); bl.setSpacing(8)

        self._global_progress = QProgressBar()
        self._global_progress.setFixedHeight(8)
        self._global_progress.setRange(0, 100)
        self._global_progress.setValue(0)
        bl.addWidget(self._global_progress, 1)

        self._btn_run = QPushButton("▶  Run")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(32); self._btn_run.setMinimumWidth(130)
        self._btn_run.clicked.connect(lambda: self._run_mode(self._current_mode))
        bl.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(32)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bl.addWidget(self._btn_cancel)

        self._status_lbl = QLabel("Ready — choose Comet or Asteroid mode.")
        self._status_lbl.setObjectName("dim")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status_lbl, 1)
        root.addWidget(bottom)

        # Startup log
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "Solar System Science Suite — ready")
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "Mode A: Comet dual-stack pipeline  |  Mode B: Moving object / asteroid detection")

    # ── Mode orchestration ────────────────────────────────────────────────────

    def _on_mode_changed(self, payload: str):
        if payload.startswith("apply_folder:"):
            folder = payload[len("apply_folder:"):]
            self._comet_panel.set_lights_folder(folder)
            self._asteroid_panel.set_folder(folder)
            self._set_status(f"Folder applied to both modes: {folder}")
            return
        mode = payload
        self._current_mode = mode
        self._btn_run.setText(
            "▶  Run ☄ Comet" if mode == "comet" else "▶  Run 🪨 Asteroid")
        # Switch to the relevant tab to hint the user
        target_tab = 1 if mode == "comet" else 2
        self._tabs.setCurrentIndex(target_tab)

    def _run_mode(self, mode: str):
        self._cancel_event.clear()
        self._current_mode = mode
        self._set_running(True)

        if mode == "comet":
            cfg = self._comet_panel.get_config()
            lights = cfg.get("lights_dir", "")
            if not lights or not os.path.isdir(lights):
                QMessageBox.warning(self, "Missing input",
                    "Set the lights folder in the Comet Pipeline tab.")
                self._tabs.setCurrentIndex(1)
                self._set_running(False); return
            self._pipeline_log.appendPlainText(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                "═══ ☄ COMET MODE ═══")
            self._set_status("Running comet pipeline…", SIRIL_SUCCESS)
            self._tabs.setCurrentIndex(1)
            self._comet_panel.run(self._cancel_event)
        else:
            cfg = self._asteroid_panel.get_config()
            if not cfg["fits_files"]:
                QMessageBox.warning(self, "Missing input",
                    "Scan a FITS folder in the Moving Object Detector tab.")
                self._tabs.setCurrentIndex(2)
                self._set_running(False); return
            self._pipeline_log.appendPlainText(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                "═══ 🪨 ASTEROID MODE ═══")
            self._set_status("Running asteroid detection…", SIRIL_ASTEROID)
            self._tabs.setCurrentIndex(2)
            self._asteroid_panel.run(self._cancel_event)

    def _on_mode_finished(self, mode: str, result: dict):
        self._set_running(False)
        if result.get("success"):
            self._overview.show_result(mode, result)
            self._global_progress.setValue(100)
            if mode == "comet":
                drift = result.get("total_drift_px", 0.0)
                msg   = f"☄ Comet done — drift {drift:.1f} px"
            else:
                n   = result.get("n_movers", 0)
                n_u = result.get("n_unknown", 0)
                msg = f"🪨 Asteroid done — {n} movers, {n_u} unknown"
            self._set_status(msg, SIRIL_SUCCESS)
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {msg}")
        else:
            err = result.get("error", "Unknown")
            self._set_status(f"✗ {err}", SIRIL_ERROR)
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✗ FAILED: {err}")

    def _cancel(self):
        if self._current_mode == "comet":
            self._comet_panel.cancel()
        else:
            self._asteroid_panel.cancel()
        self._cancel_event.set()
        self._set_status("Cancelled.", SIRIL_WARNING)
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] CANCELLED")
        self._set_running(False)

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _set_running(self, running: bool):
        self._btn_run.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._overview.set_running(running)
        if not running:
            self._global_progress.setValue(0)

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  (Section A — mandatory pattern)
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
