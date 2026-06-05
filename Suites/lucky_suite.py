r"""
lucky_suite.py  —  Script Merger: B6 Lucky Imaging Suite
=========================================================
Merged launcher for:
    1. ⚡ Lucky Preprocessor    (lucky_preprocessor.py)
    2. 🪐 Planet Derotation     (planet_derotation.py)
    3a. 🌟 Speckle Holography   (speckle_holography.py)
    3b. 🔥 The Thresher          (thresher_stack.py)
    3c. ➕ Shift-and-Add         (built-in simple mean stack)

Pipeline (Section A6 sequential):
    PREPROCESS → DEROTATE → RECONSTRUCT (user choice of 3a/3b/3c/all-three)

In-memory handoffs (Section A4):
    preprocess → derotate:    {"frame_files": list, "quality_scores": list,
                               "ser_path": str, "fits_dir": str}
    derotate   → reconstruct: {"derotated_files": list,
                               "derotated_ser": str,
                               "frame_count": int, "session_min": float}

References:
    Speckle holography: Schödel et al. 2013, A&A 558, A33
    The Thresher:       Hitchcock et al. 2022, MNRAS 511, 5372 (arXiv:2202.04686)

Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:  Siril → Scripts → lucky_suite
"""

from pathlib import Path
import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")

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
# Never import MainWindow from children.

from lucky_preprocessor import (
    ScanWorker,
    ConvertWorker,
    AnalysisWorker,
    ExportWorker,
    HistogramCanvas,
    # Algorithms
    load_master_dark,
    apply_dark,
    laplacian_variance,
    gradient_energy,
    peak_pixel,
    normalized_variance,
    fwhm_estimate,
    adaptive_elbow,
    METRIC_FUNCTIONS,
    METRIC_DESCRIPTIONS,
    OUTPUT_DESCRIPTIONS,
    # SER I/O
    read_ser_header,
    read_ser_frame,
    write_ser_file,
    get_ffmpeg_path,
    _load_local_settings,
    _save_local_settings,
    # Theme
    SIRIL_STYLESHEET,
    SIRIL_BG, SIRIL_BG2, SIRIL_BG3,
    SIRIL_ACCENT, SIRIL_ACCENT2,
    SIRIL_TEXT, SIRIL_TEXT_DIM,
    SIRIL_BORDER, SIRIL_SUCCESS,
    SIRIL_WARNING, SIRIL_SECTION,
    SIRIL_ERROR,
)

from planet_derotation import (
    DerotationWorker,
    PreviewWorker,
    RotationPreviewCanvas,
    # Algorithms
    find_planet_center,
    derotate_sequence,
    derotate_frame,
    compute_rotation_angles_offline,
    read_ser_timestamps,
    timestamps_from_manual,
    session_duration_from_timestamps,
    estimate_blur_without_derotation,
    _load_frames,
    _compute_angles,
    _fill_value,
    _save_output,
    # Data
    PLANETS,
    HAS_ASTROQUERY as DEROT_HAS_ASTROQUERY,
)

from speckle_holography import (
    SpeckleWorker,
    StarPickerCanvas,
    # Algorithms
    speckle_holography_reconstruct,
    detect_reference_stars,
    track_stars_across_sequence,
    estimate_strehl,
    compute_mean_stack,
    load_ser_as_frames,
    load_fits_sequence,
)

from thresher_stack import (
    ThresherWorker,
    SceneCanvas,
    LossCanvas,
    # Algorithms
    build_spool,
    build_spool_from_ser,
    load_spool_frame,
    subtract_sky_background,
    ThresherModel,
    HAS_TORCH,
    HAS_CUDA,
    DEVICE,
)

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox,
    QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QTabWidget, QSizePolicy, QListWidget,
    QListWidgetItem, QSlider, QSpinBox, QDoubleSpinBox, QSplitter,
    QAbstractItemView, QScrollArea, QFrame, QRadioButton, QButtonGroup,
    QStackedWidget,
)
from PyQt6.QtCore import Qt, QThread, QRectF, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPen

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATUS WIDGET  (Section A5)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStatusWidget(QWidget):
    STATUSES = {
        "idle":    ("○", "#3a4055"),
        "running": ("◉", "#4a9eff"),
        "done":    ("✓", "#4caf7d"),
        "error":   ("✗", "#cc4444"),
        "skipped": ("−", "#7a8499"),
    }

    def __init__(self, steps: list, parent=None):
        super().__init__(parent)
        self.steps    = steps
        self.statuses = {s["key"]: "idle" for s in steps}
        self.setMinimumHeight(80)

    def set_status(self, key: str, status: str):
        self.statuses[key] = status
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n   = len(self.steps)
        W, H = self.width(), self.height()
        bw  = min(150, (W - 40) // n - 14)
        bh  = 50
        gap = (W - n * bw) // (n + 1)
        by  = (H - bh) // 2

        for i, step in enumerate(self.steps):
            bx     = gap + i * (bw + gap)
            key    = step["key"]
            status = self.statuses.get(key, "idle")
            icon, color = self.STATUSES.get(status, ("○", "#3a4055"))

            p.setBrush(QColor("#252930"))
            p.setPen(QPen(QColor(color), 1.5))
            p.drawRoundedRect(QRectF(bx, by, bw, bh), 6, 6)

            p.setPen(QColor(color))
            f = QFont("Segoe UI", 13); f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(bx + 4, by, 26, bh),
                       Qt.AlignmentFlag.AlignCenter, icon)

            p.setPen(QColor("#dde3ee"))
            f2 = QFont("Segoe UI", 7)
            p.setFont(f2)
            p.drawText(QRectF(bx + 30, by, bw - 34, bh),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       step["name"])

            if i < n - 1:
                ax = bx + bw + 4
                ay = by + bh // 2
                p.setPen(QPen(QColor("#3a4055"), 1.5))
                p.drawLine(int(ax), int(ay), int(ax + gap - 8), int(ay))
                p.drawLine(int(ax+gap-8), int(ay), int(ax+gap-14), int(ay-5))
                p.drawLine(int(ax+gap-8), int(ay), int(ax+gap-14), int(ay+5))
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS PANEL  (Lucky Preprocessor UI)
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessPanel(QWidget):
    """Full Lucky Preprocessor UI. Exposes get_config(), run_pipeline()."""
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ffmpeg      = None
        self._file_infos  = []
        self._all_scores  = []
        self._selected    = []
        self._dark        = None
        self._cancel_event = threading.Event()
        self._scan_worker = self._analysis_worker = self._export_worker = None
        self._convert_worker = self._analysis_worker2 = None
        self._pipeline_ser_paths = []
        self._pipeline_out_dir   = ""
        self._pipeline_output_mode = ""

        saved = _load_local_settings().get("lucky_ffmpeg_path", "")
        if saved and os.path.isfile(saved):
            self._ffmpeg = saved
        else:
            self._ffmpeg = get_ffmpeg_path()

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ffmpeg row
        ffmpeg_row = QHBoxLayout()
        status_txt = "✓ ffmpeg found" if self._ffmpeg else "⚠ ffmpeg not found"
        self._ffmpeg_lbl = QLabel(status_txt)
        self._ffmpeg_lbl.setStyleSheet(
            f"color:{'#4caf7d' if self._ffmpeg else SIRIL_WARNING}; font-size:8pt;")
        ffmpeg_row.addWidget(self._ffmpeg_lbl)
        ffmpeg_row.addStretch()
        self._ffmpeg_edit = QLineEdit()
        self._ffmpeg_edit.setPlaceholderText("ffmpeg path (leave blank to use PATH)")
        self._ffmpeg_edit.setStyleSheet("font-size:8pt;")
        if self._ffmpeg:
            self._ffmpeg_edit.setText(self._ffmpeg)
        self._ffmpeg_edit.textChanged.connect(self._on_ffmpeg_changed)
        self._ffmpeg_edit.setMaximumWidth(280)
        ffmpeg_row.addWidget(self._ffmpeg_edit)
        btn_ff = QPushButton("Browse…"); btn_ff.setFixedWidth(72)
        btn_ff.clicked.connect(self._browse_ffmpeg)
        ffmpeg_row.addWidget(btn_ff)
        root.addLayout(ffmpeg_row)

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._build_tab_input()
        self._build_tab_quality()
        self._build_tab_output()
        self._build_tab_log()

        # Progress + buttons
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        root.addWidget(self._progress)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run Preprocessor")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run_pipeline)
        self._cancel_btn = QPushButton("✕  Cancel")
        self._cancel_btn.setObjectName("danger")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        self._open_btn = QPushButton("📂 Open Output")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_output)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._open_btn)
        root.addLayout(btn_row)

        self._status = QLabel("Ready — select a project folder to begin.")
        self._status.setObjectName("dim")
        root.addWidget(self._status)

    def _build_tab_input(self):
        tab = QWidget(); lay = QVBoxLayout(tab); lay.setSpacing(8)
        self._tabs.addTab(tab, "📁  Input")

        grp_proj = QGroupBox("Project folder")
        pf = QHBoxLayout(grp_proj)
        self._proj_edit = QLineEdit()
        self._proj_edit.setPlaceholderText("e.g. /home/user/Jupiter_2025-05-22/")
        pf.addWidget(self._proj_edit, 3)
        btn_browse = QPushButton("Browse…"); btn_browse.clicked.connect(self._browse_project)
        pf.addWidget(btn_browse)
        btn_scan = QPushButton("🔍 Scan"); btn_scan.setObjectName("primary")
        btn_scan.clicked.connect(self._scan)
        pf.addWidget(btn_scan)
        lay.addWidget(grp_proj)

        grp_files = QGroupBox("Detected files")
        fl = QVBoxLayout(grp_files)
        self._file_list = QListWidget()
        self._file_list.setMinimumHeight(100)
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        fl.addWidget(self._file_list)
        self._files_summary = QLabel("No files scanned yet.")
        self._files_summary.setObjectName("dim")
        fl.addWidget(self._files_summary)
        lay.addWidget(grp_files)

        grp_dark = QGroupBox("Dark calibration (optional)")
        dl = QVBoxLayout(grp_dark)
        self._dark_check = QCheckBox("Apply dark subtraction")
        drow = QHBoxLayout()
        self._dark_edit = QLineEdit()
        self._dark_edit.setPlaceholderText("Path to dark SER")
        self._dark_edit.setEnabled(False)
        dark_browse = QPushButton("Browse dark…"); dark_browse.clicked.connect(self._browse_dark)
        drow.addWidget(self._dark_edit, 3); drow.addWidget(dark_browse)
        self._dark_check.toggled.connect(self._dark_edit.setEnabled)
        self._dark_check.toggled.connect(dark_browse.setEnabled)
        dark_browse.setEnabled(False)
        dl.addWidget(self._dark_check); dl.addLayout(drow)
        lay.addWidget(grp_dark)
        lay.addStretch()

    def _build_tab_quality(self):
        tab = QWidget(); lay = QVBoxLayout(tab); lay.setSpacing(8)
        self._tabs.addTab(tab, "📊  Quality")

        grp_metric = QGroupBox("Quality metric")
        mf = QFormLayout(grp_metric)
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(list(METRIC_FUNCTIONS.keys()))
        self._metric_combo.setCurrentText("laplacian_variance")
        self._metric_desc = QLabel(METRIC_DESCRIPTIONS["laplacian_variance"])
        self._metric_desc.setObjectName("dim"); self._metric_desc.setWordWrap(True)
        self._metric_combo.currentTextChanged.connect(
            lambda t: self._metric_desc.setText(METRIC_DESCRIPTIONS.get(t, "")))
        mf.addRow("Metric:", self._metric_combo)
        mf.addRow("", self._metric_desc)
        lay.addWidget(grp_metric)

        grp_sel = QGroupBox("Frame selection")
        sl = QVBoxLayout(grp_sel)
        sel_row = QHBoxLayout()
        self._sel_mode_combo = QComboBox()
        for m in ("Fixed percent", "Fixed count", "Adaptive elbow", "Threshold"):
            self._sel_mode_combo.addItem(m)
        sel_row.addWidget(QLabel("Mode:")); sel_row.addWidget(self._sel_mode_combo); sel_row.addStretch()
        sl.addLayout(sel_row)
        slider_row = QHBoxLayout()
        self._sel_slider  = QSlider(Qt.Orientation.Horizontal)
        self._sel_slider.setRange(1, 100); self._sel_slider.setValue(20)
        self._sel_spinbox = QSpinBox()
        self._sel_spinbox.setRange(1, 100000); self._sel_spinbox.setValue(20)
        self._sel_spinbox.setMinimumWidth(70)
        self._sel_slider.valueChanged.connect(self._sel_spinbox.setValue)
        self._sel_spinbox.valueChanged.connect(self._sel_slider.setValue)
        self._sel_spinbox.valueChanged.connect(self._update_selection_preview)
        self._sel_unit_lbl = QLabel("%")
        slider_row.addWidget(self._sel_slider, 3)
        slider_row.addWidget(self._sel_spinbox)
        slider_row.addWidget(self._sel_unit_lbl)
        sl.addLayout(slider_row)
        self._sel_preview = QLabel("Will keep: — / — frames")
        self._sel_preview.setObjectName("dim")
        sl.addWidget(self._sel_preview)
        self._sel_mode_combo.currentTextChanged.connect(self._on_sel_mode_changed)
        self._on_sel_mode_changed("Fixed percent")
        lay.addWidget(grp_sel)

        analyze_row = QHBoxLayout()
        self._analyze_btn = QPushButton("▶  Analyze frames"); self._analyze_btn.setObjectName("primary")
        self._analyze_btn.clicked.connect(self._run_analysis)
        self._analyze_progress = QProgressBar(); self._analyze_progress.setVisible(False)
        analyze_row.addWidget(self._analyze_btn); analyze_row.addWidget(self._analyze_progress, 2)
        lay.addLayout(analyze_row)

        grp_hist = QGroupBox("Score distribution")
        hl = QVBoxLayout(grp_hist)
        self._histogram = HistogramCanvas(); hl.addWidget(self._histogram)
        lay.addWidget(grp_hist)

    def _build_tab_output(self):
        tab = QWidget(); lay = QVBoxLayout(tab); lay.setSpacing(8)
        self._tabs.addTab(tab, "💾  Output")

        grp_mode = QGroupBox("Output format")
        of = QFormLayout(grp_mode)
        self._out_mode_combo = QComboBox()
        for k in OUTPUT_DESCRIPTIONS: self._out_mode_combo.addItem(k)
        self._out_mode_desc = QLabel("")
        self._out_mode_desc.setWordWrap(True); self._out_mode_desc.setObjectName("dim")
        self._out_mode_combo.setCurrentText("filtered_ser")
        self._out_mode_combo.currentTextChanged.connect(
            lambda t: self._out_mode_desc.setText(OUTPUT_DESCRIPTIONS.get(t, "")))
        self._out_mode_desc.setText(OUTPUT_DESCRIPTIONS["filtered_ser"])
        of.addRow("Mode:", self._out_mode_combo)
        of.addRow("", self._out_mode_desc)
        lay.addWidget(grp_mode)

        grp_out = QGroupBox("Output location")
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Defaults to project folder if blank")
        out_browse = QPushButton("Browse…"); out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit, 3); out_row.addWidget(out_browse)
        grp_out.setLayout(out_row)
        lay.addWidget(grp_out)

        grp_clean = QGroupBox("Cleanup")
        cl = QVBoxLayout(grp_clean)
        self._del_originals_check = QCheckBox("Delete original .mov/.avi/.mp4 after conversion")
        self._del_unfiltered_check = QCheckBox("Delete unfiltered .ser after filtering")
        warn_lbl = QLabel("⚠  Deletion cannot be undone.")
        warn_lbl.setStyleSheet(f"color:{SIRIL_WARNING}; font-size:8pt;")
        cl.addWidget(self._del_originals_check); cl.addWidget(self._del_unfiltered_check)
        cl.addWidget(warn_lbl)
        lay.addWidget(grp_clean)

        grp_siril = QGroupBox("Siril handoff")
        si = QFormLayout(grp_siril)
        self._auto_stack_check = QCheckBox("Auto-stack in Siril after export")
        self._stack_method_combo = QComboBox()
        for m in ("winsorized", "linear"): self._stack_method_combo.addItem(m)
        si.addRow(self._auto_stack_check)
        si.addRow("Stack rejection:", self._stack_method_combo)
        lay.addWidget(grp_siril)
        lay.addStretch()

    def _build_tab_log(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        clear_btn = QPushButton("Clear log"); clear_btn.setObjectName("secondary")
        clear_btn.clicked.connect(self._log.clear)
        lay.addWidget(self._log); lay.addWidget(clear_btn)
        self._tabs.addTab(tab, "📋  Log")

    # ── Public interface ──────────────────────────────────────────────────────

    def get_config(self) -> dict:
        proj = self._proj_edit.text().strip()
        return {
            "project_dir":   proj,
            "output_dir":    self._out_edit.text().strip() or proj,
            "output_mode":   self._out_mode_combo.currentText(),
            "metric":        self._metric_combo.currentText(),
            "sel_mode":      self._sel_mode_combo.currentText(),
            "sel_value":     self._sel_spinbox.value(),
            "del_originals": self._del_originals_check.isChecked(),
            "del_unfiltered":self._del_unfiltered_check.isChecked(),
            "auto_stack":    self._auto_stack_check.isChecked(),
            "stack_method":  self._stack_method_combo.currentText(),
            "use_dark":      self._dark_check.isChecked(),
            "dark_path":     self._dark_edit.text().strip(),
        }

    def get_selected_files(self) -> list:
        """Return list of (path, frame_idx, score) for selected frames."""
        return list(self._selected)

    def get_output_dir(self) -> str:
        proj = self._proj_edit.text().strip()
        return self._out_edit.text().strip() or proj

    def cancel(self):
        self._cancel_event.set()

    def set_enabled(self, enabled: bool):
        self._run_btn.setEnabled(enabled)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_line(self, msg: str):
        self._log.appendPlainText(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
        self.log_line.emit(f"[PREPROCESS]  {msg}")

    def _set_status(self, text: str, color: str = SIRIL_TEXT_DIM):
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{color}; font-size:8pt; padding:2px;")

    def _set_busy(self, busy: bool, label: str = "Processing…"):
        self._run_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        self._analyze_btn.setEnabled(not busy)
        self._progress.setVisible(busy)
        if busy:
            self._cancel_event.clear()
            self._set_status(label, SIRIL_ACCENT)

    def _on_sel_mode_changed(self, mode: str):
        is_elbow = (mode == "Adaptive elbow")
        self._sel_slider.setEnabled(not is_elbow)
        self._sel_spinbox.setEnabled(not is_elbow)
        if mode == "Fixed percent":
            self._sel_slider.setRange(1, 100); self._sel_spinbox.setRange(1, 100)
            self._sel_spinbox.setValue(20); self._sel_unit_lbl.setText("%")
        elif mode == "Fixed count":
            self._sel_slider.setRange(1, 5000); self._sel_spinbox.setRange(1, 500000)
            self._sel_spinbox.setValue(500); self._sel_unit_lbl.setText("frames")
        elif mode == "Adaptive elbow":
            self._sel_unit_lbl.setText("(auto)")
        elif mode == "Threshold":
            self._sel_slider.setRange(0, 10000); self._sel_spinbox.setRange(0, 10000000)
            self._sel_spinbox.setValue(100); self._sel_unit_lbl.setText("score")
        self._update_selection_preview()

    def _compute_threshold(self) -> float:
        if not self._all_scores: return 0.0
        scores = [sc for _, _, sc in self._all_scores]
        mode   = self._sel_mode_combo.currentText()
        val    = self._sel_spinbox.value()
        if mode == "Fixed percent":
            cutoff = max(1, int(len(scores) * val / 100))
            return sorted(scores, reverse=True)[cutoff - 1]
        elif mode == "Fixed count":
            cutoff = min(val, len(scores))
            return sorted(scores, reverse=True)[cutoff - 1]
        elif mode == "Adaptive elbow":
            return adaptive_elbow(scores)
        elif mode == "Threshold":
            return float(val)
        return 0.0

    def _update_selection_preview(self):
        if not self._all_scores:
            self._sel_preview.setText("Will keep: — / — frames  (analyze first)"); return
        threshold = self._compute_threshold()
        selected  = [(p, i, sc) for p, i, sc in self._all_scores if sc >= threshold]
        total     = len(self._all_scores)
        pct       = 100 * len(selected) / total if total else 0
        self._sel_preview.setText(
            f"Will keep: {len(selected):,} / {total:,} frames  ({pct:.1f}%)")
        self._histogram.update_histogram([sc for _, _, sc in self._all_scores], threshold)

    # ── ffmpeg ────────────────────────────────────────────────────────────────

    def _browse_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Locate ffmpeg", "",
            "Executables (*.exe);;All files (*)")
        if path: self._ffmpeg_edit.setText(path)

    def _on_ffmpeg_changed(self, text: str):
        import shutil
        path = text.strip()
        if not path:
            self._ffmpeg = get_ffmpeg_path()
        elif os.path.isfile(path):
            self._ffmpeg = path
        else:
            found = shutil.which(path)
            self._ffmpeg = found if found else None
        if self._ffmpeg:
            self._ffmpeg_lbl.setText(f"✓ ffmpeg: {self._ffmpeg}")
            self._ffmpeg_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS}; font-size:8pt;")
            _save_local_settings({"lucky_ffmpeg_path": self._ffmpeg})
        else:
            self._ffmpeg_lbl.setText("⚠ ffmpeg not found")
            self._ffmpeg_lbl.setStyleSheet(f"color:{SIRIL_WARNING}; font-size:8pt;")

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _browse_project(self):
        d = QFileDialog.getExistingDirectory(self, "Select project folder")
        if d: self._proj_edit.setText(d)

    def _browse_dark(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select dark SER", "",
            "SER files (*.ser);;All files (*)")
        if path: self._dark_edit.setText(path)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self._out_edit.setText(d)

    def _open_output(self):
        out_dir = self.get_output_dir()
        if out_dir and os.path.isdir(out_dir):
            import subprocess
            try:
                if sys.platform == "win32": subprocess.Popen(["explorer", out_dir])
                elif sys.platform == "darwin": subprocess.Popen(["open", out_dir])
                else: subprocess.Popen(["xdg-open", out_dir])
            except Exception: pass

    # ── Scan ──────────────────────────────────────────────────────────────────

    def _scan(self):
        proj = self._proj_edit.text().strip()
        if not proj or not os.path.isdir(proj):
            QMessageBox.warning(self, "No folder", "Select a valid project folder."); return
        self._file_list.clear(); self._file_infos.clear()
        self._set_busy(True, "Scanning…")
        self._scan_worker = ScanWorker(proj)
        self._scan_worker.log_line.connect(self._log_line)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_scan_done(self, result: dict):
        self._set_busy(False)
        if not result.get("success"):
            self._set_status("Scan failed.", SIRIL_ERROR); return
        self._file_infos = result["files"]
        dark_files = result.get("dark_files", [])
        for fi in self._file_infos:
            label = os.path.basename(fi["path"])
            if fi.get("frames"):
                label += f"  [{fi['frames']} frames  {fi['width']}×{fi['height']}]"
            label += f"  {fi['size']/1e6:.1f} MB"
            item = QListWidgetItem(label)
            item.setForeground(QColor(SIRIL_SUCCESS if fi["ext"] == ".ser" else SIRIL_WARNING))
            self._file_list.addItem(item)
        self._files_summary.setText(
            f"{len(self._file_infos)} file(s)  |  {result['total_size']/1e9:.2f} GB total")
        if dark_files and not self._dark_edit.text():
            self._dark_edit.setText("; ".join(dark_files)); self._dark_check.setChecked(True)
        self._set_status(f"Scan complete: {len(self._file_infos)} file(s) found.", SIRIL_SUCCESS)

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._file_infos:
            QMessageBox.warning(self, "No files", "Scan a project folder first."); return
        ser_paths = [fi["path"] for fi in self._file_infos if fi["ext"] == ".ser"]
        if not ser_paths:
            QMessageBox.warning(self, "No SER files",
                "No SER files to analyze. Run Pipeline to convert videos first."); return
        dark = None
        if self._dark_check.isChecked():
            dark_paths = [p.strip() for p in self._dark_edit.text().split(";") if p.strip()]
            if dark_paths: dark = load_master_dark(dark_paths)
        self._dark = dark
        self._all_scores = []; self._histogram.clear_plot()
        self._analyze_progress.setVisible(True)
        self._set_busy(True, "Analyzing frames…")
        self._analysis_worker = AnalysisWorker(
            ser_paths, self._metric_combo.currentText(), dark, self._cancel_event)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.log_line.connect(self._log_line)
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.start()

    def _on_analysis_progress(self, done, total, fname):
        self._analyze_progress.setMaximum(total); self._analyze_progress.setValue(done)
        self._set_status(f"Analyzing {fname}… {done}/{total}")

    def _on_analysis_done(self, result: dict):
        self._analyze_progress.setVisible(False); self._set_busy(False)
        if not result.get("success"): self._set_status("Analysis failed.", SIRIL_ERROR); return
        self._all_scores = result["scores"]
        self._log_line(f"Analysis complete: {len(self._all_scores):,} frames scored.")
        self._update_selection_preview()
        self._tabs.setCurrentIndex(1)
        self._set_status(f"Analysis done: {len(self._all_scores):,} frames.", SIRIL_SUCCESS)

    # ── Pipeline run ──────────────────────────────────────────────────────────

    def _run_pipeline(self):
        proj = self._proj_edit.text().strip()
        if not proj or not os.path.isdir(proj):
            QMessageBox.warning(self, "No folder", "Select a valid project folder."); return
        if not self._file_infos:
            QMessageBox.warning(self, "Not scanned", "Scan the project folder first."); return
        out_dir = self._out_edit.text().strip() or proj
        os.makedirs(out_dir, exist_ok=True)
        if self._all_scores:
            threshold      = self._compute_threshold()
            self._selected = [(p, i, sc) for p, i, sc in self._all_scores if sc >= threshold]
            if not self._selected:
                QMessageBox.warning(self, "Empty selection",
                    "Current threshold selects 0 frames. Adjust selection settings."); return
        else:
            self._selected = []
        output_mode = self._out_mode_combo.currentText()
        if not self._all_scores:
            self._run_full_pipeline(proj, out_dir, output_mode)
        else:
            self._run_export_only(out_dir, output_mode)

    def run_for_pipeline(self, cancel_event: threading.Event):
        """Called by MainWindow pipeline. Uses shared cancel_event."""
        self._cancel_event = cancel_event
        self._run_pipeline()

    def _run_full_pipeline(self, proj, out_dir, output_mode):
        has_non_ser = any(fi["ext"] != ".ser" for fi in self._file_infos)
        ser_ready   = [fi for fi in self._file_infos if fi["ext"] == ".ser"]
        if not has_non_ser or not self._ffmpeg:
            ser_paths = [fi["path"] for fi in ser_ready]
            self._pipeline_ser_paths   = ser_paths
            self._pipeline_out_dir     = out_dir
            self._pipeline_output_mode = output_mode
            self._start_analysis_step()
            return
        self._set_busy(True, "Converting videos…")
        self._log_line("=== STEP 1: Convert videos → SER ===")
        self._pipeline_out_dir     = out_dir
        self._pipeline_output_mode = output_mode
        self._convert_worker = ConvertWorker(self._file_infos, self._ffmpeg, self._cancel_event)
        self._convert_worker.progress.connect(
            lambda d, t: (self._progress.setMaximum(max(t,1)), self._progress.setValue(d)))
        self._convert_worker.log_line.connect(self._log_line)
        self._convert_worker.finished.connect(self._on_convert_done)
        self._convert_worker.start()

    def _on_convert_done(self, result: dict):
        self._pipeline_ser_paths = result.get("ser_paths", [])
        self._log_line(f"Conversion done: {len(self._pipeline_ser_paths)} SER file(s).")
        self._start_analysis_step()

    def _start_analysis_step(self):
        self._log_line("=== STEP 2: Analyze frame quality ===")
        self._set_status("Analyzing frames…", SIRIL_ACCENT)
        dark = None
        if self._dark_check.isChecked():
            dark_paths = [p.strip() for p in self._dark_edit.text().split(";") if p.strip()]
            if dark_paths: dark = load_master_dark(dark_paths)
        self._dark = dark
        self._analysis_worker2 = AnalysisWorker(
            self._pipeline_ser_paths, self._metric_combo.currentText(),
            dark, self._cancel_event)
        self._analysis_worker2.progress.connect(
            lambda d, t, f: (self._progress.setMaximum(max(t,1)),
                             self._progress.setValue(d),
                             self._set_status(f"Analyzing {f}… {d}/{t}")))
        self._analysis_worker2.log_line.connect(self._log_line)
        self._analysis_worker2.finished.connect(self._on_pipeline_analysis_done)
        self._analysis_worker2.start()

    def _on_pipeline_analysis_done(self, result: dict):
        self._all_scores = result.get("scores", [])
        self._update_selection_preview()
        threshold      = self._compute_threshold()
        self._selected = [(p, i, sc) for p, i, sc in self._all_scores if sc >= threshold]
        self._log_line(f"Selected: {len(self._selected):,} / {len(self._all_scores):,} frames.")
        if not self._selected:
            self._set_busy(False); self._set_status("No frames selected.", SIRIL_WARNING); return
        self._run_export_only(self._pipeline_out_dir, self._pipeline_output_mode)

    def _run_export_only(self, out_dir, output_mode):
        self._log_line(f"=== STEP 3: Export ({output_mode}) ===")
        self._set_status("Exporting frames…", SIRIL_ACCENT)
        self._export_worker = ExportWorker(
            self._selected, output_mode, out_dir, self._dark,
            self._auto_stack_check.isChecked(),
            self._stack_method_combo.currentText(),
            self._cancel_event)
        self._export_worker.progress.connect(
            lambda d, t: (self._progress.setMaximum(max(t,1)), self._progress.setValue(d)))
        self._export_worker.log_line.connect(self._log_line)
        self._export_worker.finished.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_done(self, result: dict):
        self._set_busy(False)
        if not result.get("success"):
            self._set_status("Export failed.", SIRIL_ERROR)
            self.finished.emit({"success": False, "error": "Export failed"})
            return
        self._open_btn.setEnabled(True)
        self._set_status(
            f"✓ Done — {len(self._selected):,} frames exported.", SIRIL_SUCCESS)
        out_dir   = self.get_output_dir()
        ser_files = sorted(glob.glob(os.path.join(out_dir, "*.ser")))
        fits_files = sorted(
            glob.glob(os.path.join(out_dir, "*.fit")) +
            glob.glob(os.path.join(out_dir, "*.fits")))
        self.finished.emit({
            "success":        True,
            "frame_files":    fits_files or ser_files,
            "quality_scores": [(p, i, sc) for p, i, sc in self._selected],
            "ser_path":       ser_files[0] if ser_files else "",
            "fits_dir":       out_dir if fits_files else "",
            "output_dir":     out_dir,
            "n_selected":     len(self._selected),
        })
        self._tabs.setCurrentIndex(3)

    def _cancel(self):
        self._cancel_event.set()
        self._log_line("⚠ Cancel requested…")
        self._set_status("Cancelling…", SIRIL_WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# DEROTATE PANEL  (Planet Derotation UI)
# ─────────────────────────────────────────────────────────────────────────────

class DerotatePanel(QWidget):
    """Full Planet Derotation UI. Exposes get_config(), run_pipeline()."""
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker         = None
        self._preview_worker = None
        self._cancel_event   = threading.Event()
        self._angles         = []
        self._timestamps     = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._build_tab_input()
        self._build_tab_derotation()
        self._build_tab_output()
        self._build_tab_log()

        # Bottom controls
        bot = QHBoxLayout()
        self._btn_preview = QPushButton("👁  Preview rotation")
        self._btn_preview.clicked.connect(self._run_preview)
        bot.addWidget(self._btn_preview)
        self._btn_run = QPushButton("▶  Run Derotation")
        self._btn_run.setObjectName("primary")
        self._btn_run.clicked.connect(self._run_derotation)
        bot.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bot.addWidget(self._btn_cancel)
        self._progress = QProgressBar()
        self._progress.setVisible(False); self._progress.setFixedHeight(8)
        bot.addWidget(self._progress, 1)
        root.addLayout(bot)

        self._status_lbl = QLabel("Ready — configure planet and input.")
        self._status_lbl.setObjectName("dim")
        root.addWidget(self._status_lbl)

    def _build_tab_input(self):
        tab = QWidget(); lay = QVBoxLayout(tab); lay.setSpacing(8)
        self._tabs.addTab(tab, "🪐  Planet & Input")

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); il = QVBoxLayout(inner); il.setSpacing(8)
        scroll.setWidget(inner); lay.addWidget(scroll)

        grp_planet = QGroupBox("Planet selector")
        pf = QFormLayout(grp_planet)
        self._planet_combo = QComboBox()
        for name in PLANETS: self._planet_combo.addItem(f"{PLANETS[name]['emoji']} {name}")
        self._planet_combo.addItem("🌙 Moon"); self._planet_combo.addItem("✏ Custom")
        self._planet_combo.currentTextChanged.connect(self._on_planet_changed)
        pf.addRow("Planet:", self._planet_combo)
        self._system_combo = QComboBox(); pf.addRow("System:", self._system_combo)
        self._lbl_safe_session = QLabel("")
        self._lbl_safe_session.setObjectName("dim"); pf.addRow("Safe session:", self._lbl_safe_session)
        il.addWidget(grp_planet)
        self._on_planet_changed(self._planet_combo.currentText())

        grp_input = QGroupBox("Input frames")
        gi = QFormLayout(grp_input)
        self._input_type_combo = QComboBox()
        self._input_type_combo.addItems(["SER file", "FITS sequence folder"])
        gi.addRow("Type:", self._input_type_combo)
        in_row = QHBoxLayout()
        self._input_path_edit = QLineEdit()
        self._input_path_edit.setPlaceholderText("SER file or FITS folder…")
        btn_in = QPushButton("Browse…"); btn_in.setFixedWidth(70)
        btn_in.clicked.connect(self._browse_input)
        in_row.addWidget(self._input_path_edit); in_row.addWidget(btn_in)
        gi.addRow("Path:", in_row)
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 1000.0); self._fps_spin.setValue(25.0)
        gi.addRow("Frame rate (if no timestamps):", self._fps_spin)
        il.addWidget(grp_input)

        grp_time = QGroupBox("Manual timing (if no SER timestamps)")
        tf = QFormLayout(grp_time)
        self._use_horizons = QCheckBox("Use JPL Horizons (precise, requires internet)")
        if not DEROT_HAS_ASTROQUERY:
            self._use_horizons.setEnabled(False)
            self._use_horizons.setToolTip("astroquery not installed")
        self._use_horizons.setChecked(False)
        tf.addRow(self._use_horizons)
        self._observer_edit = QLineEdit("500")
        tf.addRow("Observer location:", self._observer_edit)
        il.addWidget(grp_time)
        il.addStretch()

    def _build_tab_derotation(self):
        tab = QWidget(); lay = QVBoxLayout(tab); lay.setSpacing(8)
        self._tabs.addTab(tab, "⚙  Derotation")

        grp_center = QGroupBox("Planet center")
        cf = QFormLayout(grp_center)
        self._center_combo = QComboBox()
        self._center_combo.addItems(["centroid", "circle_fit"])
        cf.addRow("Method:", self._center_combo)
        self._fill_combo = QComboBox()
        self._fill_combo.addItems(["black (zero)", "background"])
        cf.addRow("Border fill:", self._fill_combo)
        lay.addWidget(grp_center)

        grp_out_type = QGroupBox("Output type")
        ot = QFormLayout(grp_out_type)
        self._out_type_combo = QComboBox()
        self._out_type_combo.addItems(["ser", "fits_sequence"])
        ot.addRow("Format:", self._out_type_combo)
        out_row = QHBoxLayout()
        self._derot_out_edit = QLineEdit()
        self._derot_out_edit.setPlaceholderText("Output folder…")
        btn_od = QPushButton("Browse…"); btn_od.setFixedWidth(70)
        btn_od.clicked.connect(lambda: self._derot_out_edit.setText(
            QFileDialog.getExistingDirectory(self, "Select output folder") or
            self._derot_out_edit.text()))
        out_row.addWidget(self._derot_out_edit); out_row.addWidget(btn_od)
        ot.addRow("Output folder:", out_row)
        self._chk_load_siril = QCheckBox("Load output folder in Siril")
        self._chk_load_siril.setChecked(True)
        ot.addRow(self._chk_load_siril)
        lay.addWidget(grp_out_type)

        # Preview canvas
        self._rotation_canvas = RotationPreviewCanvas()
        self._rotation_canvas.setMinimumHeight(280)
        lay.addWidget(self._rotation_canvas, 1)

    def _build_tab_output(self):
        tab = QWidget(); lay = QVBoxLayout(tab); lay.setSpacing(8)
        self._tabs.addTab(tab, "📊  Results")
        self._result_lbl = QLabel("Run derotation to see results.")
        self._result_lbl.setObjectName("dim"); self._result_lbl.setWordWrap(True)
        lay.addWidget(self._result_lbl)
        lay.addStretch()

    def _build_tab_log(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        lay.addWidget(self._log)
        self._tabs.addTab(tab, "📋  Log")

    # ── Public interface ──────────────────────────────────────────────────────

    def set_input_from_preprocess(self, data: dict):
        """Pipeline handoff from preprocessor."""
        ser  = data.get("ser_path", "")
        fits = data.get("fits_dir", "")
        if ser and os.path.isfile(ser):
            self._input_type_combo.setCurrentText("SER file")
            self._input_path_edit.setText(ser)
        elif fits and os.path.isdir(fits):
            self._input_type_combo.setCurrentText("FITS sequence folder")
            self._input_path_edit.setText(fits)
        out = data.get("output_dir", "")
        if out and not self._derot_out_edit.text():
            self._derot_out_edit.setText(out)

    def get_config(self) -> dict:
        planet_text = self._planet_combo.currentText()
        planet_name = planet_text.split(" ", 1)[-1] if " " in planet_text else planet_text
        # Handle Moon/Custom (not in PLANETS dict)
        if planet_name not in PLANETS:
            planet_name = "Jupiter"
        input_type = "ser" if self._input_type_combo.currentIndex() == 0 else "fits"
        fill_mode  = "background" if self._fill_combo.currentIndex() == 1 else "zero"
        return {
            "planet":           planet_name,
            "system":           self._system_combo.currentText(),
            "input_type":       input_type,
            "input_path":       self._input_path_edit.text().strip(),
            "frame_rate_fps":   self._fps_spin.value(),
            "use_horizons":     self._use_horizons.isChecked(),
            "observer_location":self._observer_edit.text().strip() or "500",
            "center_method":    self._center_combo.currentText(),
            "fill_mode":        fill_mode,
            "output_type":      self._out_type_combo.currentText(),
            "output_dir":       self._derot_out_edit.text().strip(),
            "load_in_siril":    self._chk_load_siril.isChecked(),
        }

    def get_output_dir(self) -> str:
        d = self._derot_out_edit.text().strip()
        if not d:
            d = os.path.dirname(self._input_path_edit.text().strip()) or "."
        return d

    def cancel(self):
        self._cancel_event.set()
        if self._worker: self._worker._cancel.set()
        if self._preview_worker: self._preview_worker._cancel.set()

    def set_enabled(self, enabled: bool):
        self._btn_run.setEnabled(enabled)

    # ── Planet selector ───────────────────────────────────────────────────────

    def _on_planet_changed(self, text: str):
        planet_name = text.split(" ", 1)[-1] if " " in text else text
        self._system_combo.clear()
        if planet_name in PLANETS:
            p = PLANETS[planet_name]
            for sys_name in p["systems"]:
                self._system_combo.addItem(sys_name)
            self._system_combo.setCurrentText(p["default_system"])
            safe = p.get("safe_session_min", 0)
            self._lbl_safe_session.setText(
                f"≤ {safe} min recommended to avoid noticeable rotation blur")
        else:
            self._system_combo.addItem("Custom")
            self._lbl_safe_session.setText("—")

    def _browse_input(self):
        if self._input_type_combo.currentIndex() == 0:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select SER file", "", "SER files (*.ser);;All files (*)")
            if path: self._input_path_edit.setText(path)
        else:
            d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
            if d: self._input_path_edit.setText(d)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _log_line(self, msg: str):
        self._log.appendPlainText(msg)
        self.log_line.emit(f"[DEROTATE]  {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")

    def _run_preview(self):
        cfg = self.get_config()
        if not cfg["input_path"]:
            QMessageBox.warning(self, "No input", "Set the input SER or FITS folder."); return
        self._cancel_event.clear()
        self._btn_preview.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._set_status("Computing preview…", SIRIL_ACCENT)
        self._preview_worker = PreviewWorker(cfg, self._cancel_event)
        self._preview_worker.log_line.connect(self._log_line)
        self._preview_worker.angles_ready.connect(
            lambda ang, ts: self._rotation_canvas.show_angles(ang, ts))
        self._preview_worker.preview_ready.connect(
            lambda f, lr, ld: self._rotation_canvas.show_frames(f, lr, ld))
        self._preview_worker.finished.connect(self._on_preview_done)
        self._preview_worker.start()

    def _on_preview_done(self, result: dict):
        self._btn_preview.setEnabled(True); self._btn_run.setEnabled(True)
        if result.get("success"):
            self._set_status("Preview ready — review rotation angles.", SIRIL_SUCCESS)
            self._tabs.setCurrentIndex(1)
        else:
            self._set_status(f"Preview failed: {result.get('error','')}", SIRIL_ERROR)

    def _run_derotation(self, cancel_event: threading.Event | None = None):
        cfg = self.get_config()
        if not cfg["input_path"]:
            QMessageBox.warning(self, "No input", "Set the input file/folder."); return
        self._cancel_event = cancel_event or threading.Event()
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.show(); self._progress.setValue(0)
        self._set_status("Running derotation…", SIRIL_ACCENT)
        self._worker = DerotationWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log_line)
        self._worker.angles_ready.connect(
            lambda ang, ts: (setattr(self, "_angles", ang),
                             setattr(self, "_timestamps", ts),
                             self._rotation_canvas.show_angles(ang, ts)))
        self._worker.preview_ready.connect(
            lambda f, lr, ld: self._rotation_canvas.show_frames(f, lr, ld))
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def run_pipeline(self, piped_data: dict, cancel_event: threading.Event):
        """Pipeline mode: auto-fill input from preprocessor then run."""
        if piped_data:
            self.set_input_from_preprocess(piped_data)
        self._run_derotation(cancel_event)

    def _cancel(self):
        self._cancel_event.set()
        self._btn_cancel.setEnabled(False); self._set_status("Cancelling…", SIRIL_WARNING)

    def _on_progress(self, step, total, msg):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(f"Step {step}/{total}: {msg}", SIRIL_ACCENT)

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.hide()
        if result.get("success"):
            rot   = result.get("total_rotation", 0.0)
            blur  = result.get("blur_prevented", 0.0)
            out   = result.get("output_path", "—")
            msg   = (f"✓ Derotation complete\n"
                     f"Total rotation: {rot:.2f}°  |  Blur prevented: {blur:.1f}px\n"
                     f"Saved: {out}")
            self._result_lbl.setText(msg)
            self._set_status(
                f"✓ Done — {rot:.2f}° corrected, {blur:.1f}px blur prevented", SIRIL_SUCCESS)
            # Build pipeline handoff
            out_path = result.get("output_path", "")
            if out_path.endswith(os.sep):  # FITS sequence dir
                fits_files = sorted(
                    glob.glob(os.path.join(out_path, "*.fit")) +
                    glob.glob(os.path.join(out_path, "*.fits")))
                self.finished.emit({
                    "success": True,
                    "derotated_files": fits_files,
                    "derotated_ser": "",
                    "frame_count": result.get("frame_count", 0),
                    "session_min": result.get("session_min", 0.0),
                    "output_dir": out_path,
                })
            else:
                self.finished.emit({
                    "success": True,
                    "derotated_files": [],
                    "derotated_ser": out_path,
                    "frame_count": result.get("frame_count", 0),
                    "session_min": result.get("session_min", 0.0),
                    "output_dir": os.path.dirname(out_path),
                })
            self._tabs.setCurrentIndex(2)
        else:
            err = result.get("error", "Unknown error")
            self._result_lbl.setText(f"✗ Failed: {err}")
            self._set_status(f"✗ {err}", SIRIL_ERROR)
            self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# SPECKLE PANEL  (Speckle Holography UI)
# ─────────────────────────────────────────────────────────────────────────────

class SpecklePanel(QWidget):
    """Speckle Holography UI."""
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._cancel_event = threading.Event()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._build_tab_settings()
        self._build_tab_log()

        bot = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run Speckle Holography")
        self._btn_run.setObjectName("primary")
        self._btn_run.clicked.connect(self._run)
        bot.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bot.addWidget(self._btn_cancel)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        self._progress.setFixedHeight(8); bot.addWidget(self._progress, 1)
        root.addLayout(bot)
        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setObjectName("dim"); root.addWidget(self._status_lbl)

    def _build_tab_settings(self):
        tab = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); il = QVBoxLayout(inner); il.setSpacing(8)
        scroll.setWidget(inner); lay = QVBoxLayout(tab); lay.addWidget(scroll)

        grp_in = QGroupBox("Input frames")
        gi = QFormLayout(grp_in)
        self._input_type_combo = QComboBox()
        self._input_type_combo.addItems(["FITS sequence folder", "SER file"])
        gi.addRow("Type:", self._input_type_combo)
        in_row = QHBoxLayout()
        self._input_edit = QLineEdit(); self._input_edit.setPlaceholderText("Path…")
        btn_in = QPushButton("Browse…"); btn_in.setFixedWidth(70)
        btn_in.clicked.connect(self._browse_input)
        in_row.addWidget(self._input_edit); in_row.addWidget(btn_in)
        gi.addRow("Path:", in_row)
        self._max_frames_spin = QSpinBox()
        self._max_frames_spin.setRange(0, 100000); self._max_frames_spin.setValue(0)
        self._max_frames_spin.setSpecialValueText("All frames")
        gi.addRow("Max frames (0=all):", self._max_frames_spin)
        il.addWidget(grp_in)

        grp_psf = QGroupBox("PSF & reconstruction")
        pf = QFormLayout(grp_psf)
        self._n_stars_spin = QSpinBox()
        self._n_stars_spin.setRange(1, 20); self._n_stars_spin.setValue(3)
        pf.addRow("Reference stars:", self._n_stars_spin)
        self._psf_box_spin = QSpinBox()
        self._psf_box_spin.setRange(16, 256); self._psf_box_spin.setValue(64)
        self._psf_box_spin.setSingleStep(16)
        pf.addRow("PSF box size:", self._psf_box_spin)
        self._star_thresh_spin = QDoubleSpinBox()
        self._star_thresh_spin.setRange(3.0, 30.0); self._star_thresh_spin.setValue(8.0)
        self._star_thresh_spin.setSingleStep(0.5); self._star_thresh_spin.setSuffix(" σ")
        pf.addRow("Star detection threshold:", self._star_thresh_spin)
        self._epsilon_spin = QDoubleSpinBox()
        self._epsilon_spin.setRange(0.001, 1.0); self._epsilon_spin.setValue(0.01)
        self._epsilon_spin.setDecimals(3); self._epsilon_spin.setSingleStep(0.005)
        pf.addRow("Wiener epsilon:", self._epsilon_spin)
        self._track_stars_chk = QCheckBox("Track stars across frames")
        self._track_stars_chk.setChecked(True); pf.addRow(self._track_stars_chk)
        il.addWidget(grp_psf)

        grp_out = QGroupBox("Output")
        of = QFormLayout(grp_out)
        out_row = QHBoxLayout()
        self._out_dir_edit = QLineEdit(); self._out_dir_edit.setPlaceholderText("Same as input")
        btn_od = QPushButton("Browse…"); btn_od.setFixedWidth(70)
        btn_od.clicked.connect(lambda: self._out_dir_edit.setText(
            QFileDialog.getExistingDirectory(self, "Select output folder") or
            self._out_dir_edit.text()))
        out_row.addWidget(self._out_dir_edit); out_row.addWidget(btn_od)
        of.addRow("Output folder:", out_row)
        self._out_name_edit = QLineEdit("speckle_holography_result.fit")
        of.addRow("Filename:", self._out_name_edit)
        self._save_mean_chk = QCheckBox("Also save mean stack (for comparison)")
        of.addRow(self._save_mean_chk)
        self._load_siril_chk = QCheckBox("Load result in Siril")
        self._load_siril_chk.setChecked(False); of.addRow(self._load_siril_chk)
        il.addWidget(grp_out)
        il.addStretch()
        self._tabs.addTab(tab, "⚙  Settings")

    def _build_tab_log(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); lay.addWidget(self._log)
        self._tabs.addTab(tab, "📋  Log")

    def _browse_input(self):
        if self._input_type_combo.currentIndex() == 0:
            d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
            if d: self._input_edit.setText(d)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select SER", "",
                "SER files (*.ser);;All files (*)")
            if path: self._input_edit.setText(path)

    def set_input_from_derotate(self, data: dict):
        fits = data.get("derotated_files", [])
        ser  = data.get("derotated_ser", "")
        out  = data.get("output_dir", "")
        if fits:
            d = os.path.dirname(fits[0])
            self._input_type_combo.setCurrentText("FITS sequence folder")
            self._input_edit.setText(d)
        elif ser and os.path.isfile(ser):
            self._input_type_combo.setCurrentText("SER file")
            self._input_edit.setText(ser)
        if out and not self._out_dir_edit.text():
            self._out_dir_edit.setText(out)

    def get_config(self) -> dict:
        input_type = ("fits_dir" if self._input_type_combo.currentIndex() == 0
                      else "ser")
        return {
            "input_type":    input_type,
            "input_path":    self._input_edit.text().strip(),
            "max_frames":    self._max_frames_spin.value() or None,
            "n_psf_stars":   self._n_stars_spin.value(),
            "psf_box_size":  self._psf_box_spin.value(),
            "star_threshold":self._star_thresh_spin.value(),
            "epsilon":       self._epsilon_spin.value(),
            "track_stars":   self._track_stars_chk.isChecked(),
            "output_dir":    self._out_dir_edit.text().strip() or
                             os.path.dirname(self._input_edit.text().strip() or "."),
            "output_name":   self._out_name_edit.text().strip(),
            "save_mean_stack":self._save_mean_chk.isChecked(),
            "load_in_siril": self._load_siril_chk.isChecked(),
        }

    def cancel(self):
        self._cancel_event.set()
        if self._worker: self._worker.cancel()

    def set_enabled(self, enabled: bool):
        self._btn_run.setEnabled(enabled)

    def _log_line(self, msg: str):
        self._log.appendPlainText(msg)
        self.log_line.emit(f"[SPECKLE]  {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")

    def _run(self, cancel_event: threading.Event | None = None):
        cfg = self.get_config()
        if not cfg["input_path"]:
            QMessageBox.warning(self, "No input", "Set the input path."); return
        self._cancel_event = cancel_event or threading.Event()
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.show(); self._progress.setValue(0)
        self._set_status("Running Speckle Holography…", SIRIL_ACCENT)
        self._worker = SpeckleWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def run_pipeline(self, piped_data: dict, cancel_event: threading.Event):
        if piped_data:
            self.set_input_from_derotate(piped_data)
        self._run(cancel_event)

    def _cancel(self):
        self.cancel(); self._btn_cancel.setEnabled(False)

    def _on_progress(self, step, total, msg):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(f"Step {step}/{total}: {msg}", SIRIL_ACCENT)

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.hide()
        if result.get("success"):
            n     = result.get("n_frames_used", 0)
            n_tot = result.get("n_frames_total", 0)
            best  = result.get("best_strehl", 0)
            out   = result.get("output_path", "—")
            self._set_status(
                f"✓ Done — {n}/{n_tot} frames used  best Strehl={best:.3f}  → {out}",
                SIRIL_SUCCESS)
        else:
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)
        self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHER PANEL  (The Thresher UI)
# ─────────────────────────────────────────────────────────────────────────────

class ThresherPanel(QWidget):
    """The Thresher SGD blind deconvolution UI."""
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._cancel_event = threading.Event()
        self._update_count = 0
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        # PyTorch warning
        if not HAS_TORCH:
            warn = QLabel(
                "⚠  PyTorch not installed — The Thresher requires it.\n"
                "Install: pip install torch --index-url https://download.pytorch.org/whl/cpu")
            warn.setStyleSheet(f"color:{SIRIL_WARNING}; padding:6px; font-weight:bold;")
            warn.setWordWrap(True)
            root.addWidget(warn)
        else:
            device_info = f"{'GPU: ' + __import__('torch').cuda.get_device_name(0) if HAS_CUDA else 'CPU only'}"
            info = QLabel(f"✓ PyTorch ready  |  Device: {device_info}")
            info.setStyleSheet(f"color:{SIRIL_SUCCESS}; font-size:8pt;")
            root.addWidget(info)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._build_tab_settings()
        self._build_tab_live()
        self._build_tab_log()

        bot = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run Thresher")
        self._btn_run.setObjectName("primary")
        self._btn_run.clicked.connect(self._run)
        if not HAS_TORCH: self._btn_run.setEnabled(False)
        bot.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bot.addWidget(self._btn_cancel)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        self._progress.setFixedHeight(8); bot.addWidget(self._progress, 1)
        root.addLayout(bot)
        self._status_lbl = QLabel("Ready."); self._status_lbl.setObjectName("dim")
        root.addWidget(self._status_lbl)

    def _build_tab_settings(self):
        tab = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); il = QVBoxLayout(inner); il.setSpacing(8)
        scroll.setWidget(inner); lay = QVBoxLayout(tab); lay.addWidget(scroll)

        grp_in = QGroupBox("Input frames")
        gi = QFormLayout(grp_in)
        self._input_type_combo = QComboBox()
        self._input_type_combo.addItems(["FITS sequence folder", "SER file", "Pre-built FITS spool"])
        gi.addRow("Type:", self._input_type_combo)
        in_row = QHBoxLayout()
        self._input_edit = QLineEdit(); self._input_edit.setPlaceholderText("Path…")
        btn_in = QPushButton("Browse…"); btn_in.setFixedWidth(70)
        btn_in.clicked.connect(self._browse_input)
        in_row.addWidget(self._input_edit); in_row.addWidget(btn_in)
        gi.addRow("Path:", in_row)
        self._max_frames_spin = QSpinBox()
        self._max_frames_spin.setRange(0, 100000); self._max_frames_spin.setValue(0)
        self._max_frames_spin.setSpecialValueText("All frames")
        gi.addRow("Max frames (0=all):", self._max_frames_spin)
        il.addWidget(grp_in)

        grp_model = QGroupBox("Model & optimization")
        mf = QFormLayout(grp_model)
        self._iterations_spin = QSpinBox()
        self._iterations_spin.setRange(1, 100); self._iterations_spin.setValue(1)
        mf.addRow("SGD iterations:", self._iterations_spin)
        self._kernel_size_spin = QSpinBox()
        self._kernel_size_spin.setRange(5, 101); self._kernel_size_spin.setValue(21)
        self._kernel_size_spin.setSingleStep(2)
        mf.addRow("Kernel size (odd):", self._kernel_size_spin)
        self._noise_model_combo = QComboBox()
        self._noise_model_combo.addItems(["gaussian", "ccd", "emccd"])
        mf.addRow("Noise model:", self._noise_model_combo)
        self._lr_scene_spin = QDoubleSpinBox()
        self._lr_scene_spin.setRange(1e-5, 0.1); self._lr_scene_spin.setValue(1e-3)
        self._lr_scene_spin.setDecimals(5); self._lr_scene_spin.setSingleStep(1e-4)
        mf.addRow("LR scene:", self._lr_scene_spin)
        self._lr_kernel_spin = QDoubleSpinBox()
        self._lr_kernel_spin.setRange(1e-5, 0.1); self._lr_kernel_spin.setValue(1e-3)
        self._lr_kernel_spin.setDecimals(5); self._lr_kernel_spin.setSingleStep(1e-4)
        mf.addRow("LR kernel:", self._lr_kernel_spin)
        self._subtract_sky_chk = QCheckBox("Subtract sky background per frame")
        self._subtract_sky_chk.setChecked(True); mf.addRow(self._subtract_sky_chk)
        il.addWidget(grp_model)

        grp_out = QGroupBox("Output")
        of = QFormLayout(grp_out)
        out_row = QHBoxLayout()
        self._out_dir_edit = QLineEdit(); self._out_dir_edit.setPlaceholderText("Same as input")
        btn_od = QPushButton("Browse…"); btn_od.setFixedWidth(70)
        btn_od.clicked.connect(lambda: self._out_dir_edit.setText(
            QFileDialog.getExistingDirectory(self, "Select output folder") or
            self._out_dir_edit.text()))
        out_row.addWidget(self._out_dir_edit); out_row.addWidget(btn_od)
        of.addRow("Output folder:", out_row)
        self._out_name_edit = QLineEdit("thresher_scene")
        of.addRow("Base filename:", self._out_name_edit)
        self._save_every_spin = QSpinBox()
        self._save_every_spin.setRange(10, 10000); self._save_every_spin.setValue(100)
        of.addRow("Save every N updates:", self._save_every_spin)
        self._keep_spool_chk = QCheckBox("Keep temp spool (large file)")
        of.addRow(self._keep_spool_chk)
        self._load_siril_chk = QCheckBox("Load final scene in Siril")
        self._load_siril_chk.setChecked(True); of.addRow(self._load_siril_chk)
        il.addWidget(grp_out)
        il.addStretch()
        self._tabs.addTab(tab, "⚙  Settings")

    def _build_tab_live(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self._scene_canvas = SceneCanvas(); lay.addWidget(self._scene_canvas, 1)
        self._loss_canvas  = LossCanvas();  lay.addWidget(self._loss_canvas)
        self._tabs.addTab(tab, "📈  Live Scene")

    def _build_tab_log(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True); lay.addWidget(self._log)
        self._tabs.addTab(tab, "📋  Log")

    def _browse_input(self):
        idx = self._input_type_combo.currentIndex()
        if idx == 0:
            d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
            if d: self._input_edit.setText(d)
        elif idx == 1:
            path, _ = QFileDialog.getOpenFileName(self, "Select SER", "",
                "SER files (*.ser);;All files (*)")
            if path: self._input_edit.setText(path)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select spool FITS", "",
                "FITS (*.fit *.fits);;All files (*)")
            if path: self._input_edit.setText(path)

    def set_input_from_derotate(self, data: dict):
        fits = data.get("derotated_files", [])
        ser  = data.get("derotated_ser", "")
        out  = data.get("output_dir", "")
        if fits:
            d = os.path.dirname(fits[0])
            self._input_type_combo.setCurrentIndex(0)
            self._input_edit.setText(d)
        elif ser and os.path.isfile(ser):
            self._input_type_combo.setCurrentIndex(1)
            self._input_edit.setText(ser)
        if out and not self._out_dir_edit.text():
            self._out_dir_edit.setText(out)

    def get_config(self) -> dict:
        type_map = {0: "fits_dir", 1: "ser", 2: "spool"}
        input_type = type_map[self._input_type_combo.currentIndex()]
        path = self._input_edit.text().strip()
        cfg = {
            "input_type":       input_type,
            "iterations":       self._iterations_spin.value(),
            "kernel_size":      self._kernel_size_spin.value(),
            "noise_model":      self._noise_model_combo.currentText(),
            "lr_scene":         self._lr_scene_spin.value(),
            "lr_kernel":        self._lr_kernel_spin.value(),
            "subtract_sky":     self._subtract_sky_chk.isChecked(),
            "output_dir":       self._out_dir_edit.text().strip() or
                                (os.path.dirname(path) if not os.path.isdir(path) else path),
            "output_name":      self._out_name_edit.text().strip() or "thresher_scene",
            "save_every":       self._save_every_spin.value(),
            "keep_spool":       self._keep_spool_chk.isChecked(),
            "load_in_siril":    self._load_siril_chk.isChecked(),
            "max_frames":       self._max_frames_spin.value() or None,
        }
        if input_type == "fits_dir":
            cfg["fits_dir"] = path
        elif input_type == "ser":
            cfg["ser_path"] = path
        else:
            cfg["spool_path"] = path
        return cfg

    def cancel(self):
        self._cancel_event.set()

    def set_enabled(self, enabled: bool):
        self._btn_run.setEnabled(enabled and HAS_TORCH)

    def _log_line(self, msg: str):
        self._log.appendPlainText(msg)
        self.log_line.emit(f"[THRESHER]  {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")

    def _run(self, cancel_event: threading.Event | None = None):
        if not HAS_TORCH:
            QMessageBox.warning(self, "PyTorch missing",
                "The Thresher requires PyTorch. Install from https://pytorch.org"); return
        cfg = self.get_config()
        path = self._input_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No input", "Set the input path."); return
        self._cancel_event = cancel_event or threading.Event()
        self._cancel_event.clear()
        self._update_count = 0
        self._loss_canvas.reset()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.show(); self._progress.setValue(0)
        self._set_status("Building spool…", SIRIL_ACCENT)
        self._worker = ThresherWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log_line)
        self._worker.scene_updated.connect(self._on_scene_updated)
        self._worker.loss_point.connect(self._loss_canvas.add_point)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def run_pipeline(self, piped_data: dict, cancel_event: threading.Event):
        if piped_data:
            self.set_input_from_derotate(piped_data)
        self._run(cancel_event)

    def _cancel(self):
        self.cancel(); self._btn_cancel.setEnabled(False)

    def _on_progress(self, step, total, msg):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(f"{msg}", SIRIL_ACCENT)

    def _on_scene_updated(self, scene: np.ndarray):
        self._update_count += 1
        if self._update_count == 1:
            self._scene_canvas.show_init(scene)
        else:
            self._scene_canvas.show_scene(scene, self._update_count)
        self._tabs.setCurrentIndex(1)

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(HAS_TORCH); self._btn_cancel.setEnabled(False)
        self._progress.hide()
        if result.get("success"):
            n     = result.get("n_frames", 0)
            upd   = result.get("n_updates", 0)
            out   = result.get("output_path", "—")
            elapsed = result.get("elapsed_sec", 0)
            self._set_status(
                f"✓ Done — {n} frames, {upd} updates, {elapsed/60:.1f} min  → {out}",
                SIRIL_SUCCESS)
        else:
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)
        self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# SHIFT-AND-ADD PANEL  (built-in simple mean stack for comparison)
# ─────────────────────────────────────────────────────────────────────────────

class ShiftAddPanel(QWidget):
    """Simple mean stack baseline — no external dependency."""
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._cancel_event = threading.Event()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(8, 8, 8, 8)
        info = QLabel(
            "Shift-and-Add is a simple mean stack of the top X% frames.\n"
            "It is the standard baseline. Use Speckle or Thresher for better results.")
        info.setObjectName("dim"); info.setWordWrap(True); root.addWidget(info)

        grp_in = QGroupBox("Input")
        gi = QFormLayout(grp_in)
        self._type_combo = QComboBox()
        self._type_combo.addItems(["FITS sequence folder", "SER file"])
        gi.addRow("Type:", self._type_combo)
        in_row = QHBoxLayout()
        self._input_edit = QLineEdit(); self._input_edit.setPlaceholderText("Path…")
        btn = QPushButton("Browse…"); btn.setFixedWidth(70); btn.clicked.connect(self._browse)
        in_row.addWidget(self._input_edit); in_row.addWidget(btn)
        gi.addRow("Path:", in_row)
        self._top_pct_spin = QSpinBox()
        self._top_pct_spin.setRange(1, 100); self._top_pct_spin.setValue(100)
        self._top_pct_spin.setSuffix("%")
        gi.addRow("Use top % of frames:", self._top_pct_spin)
        root.addWidget(grp_in)

        grp_out = QGroupBox("Output")
        of = QFormLayout(grp_out)
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit(); self._out_edit.setPlaceholderText("Output folder…")
        btn_od = QPushButton("Browse…"); btn_od.setFixedWidth(70)
        btn_od.clicked.connect(lambda: self._out_edit.setText(
            QFileDialog.getExistingDirectory(self, "Select output folder") or self._out_edit.text()))
        out_row.addWidget(self._out_edit); out_row.addWidget(btn_od)
        of.addRow("Folder:", out_row)
        root.addWidget(grp_out)

        root.addStretch()

        bot = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run Shift-and-Add")
        self._btn_run.setObjectName("primary"); self._btn_run.clicked.connect(self._run)
        bot.addWidget(self._btn_run)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        self._progress.setFixedHeight(8); bot.addWidget(self._progress, 1)
        root.addLayout(bot)
        self._status_lbl = QLabel("Ready."); self._status_lbl.setObjectName("dim")
        root.addWidget(self._status_lbl)

    def _browse(self):
        if self._type_combo.currentIndex() == 0:
            d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
            if d: self._input_edit.setText(d)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select SER", "",
                "SER files (*.ser);;All files (*)")
            if path: self._input_edit.setText(path)

    def set_input_from_derotate(self, data: dict):
        fits = data.get("derotated_files", [])
        ser  = data.get("derotated_ser", "")
        out  = data.get("output_dir", "")
        if fits:
            self._type_combo.setCurrentIndex(0)
            self._input_edit.setText(os.path.dirname(fits[0]))
        elif ser:
            self._type_combo.setCurrentIndex(1); self._input_edit.setText(ser)
        if out and not self._out_edit.text():
            self._out_edit.setText(out)

    def get_config(self) -> dict:
        return {
            "input_type": "fits" if self._type_combo.currentIndex() == 0 else "ser",
            "input_path": self._input_edit.text().strip(),
            "top_pct":    self._top_pct_spin.value(),
            "output_dir": self._out_edit.text().strip(),
        }

    def cancel(self):
        self._cancel_event.set()

    def set_enabled(self, enabled: bool):
        self._btn_run.setEnabled(enabled)

    def _run(self, cancel_event: threading.Event | None = None):
        cfg = self.get_config()
        path = cfg["input_path"]
        if not path:
            QMessageBox.warning(self, "No input", "Set the input path."); return
        self._cancel_event = cancel_event or threading.Event()
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._progress.show()
        self._status_lbl.setText("Loading frames…")

        worker = _ShiftAddWorker(cfg, self._cancel_event)
        worker.progress.connect(self._on_progress)
        worker.log_line.connect(lambda m: self.log_line.emit(f"[SHIFT-ADD]  {m}"))
        worker.finished.connect(self._on_finished)
        self._worker = worker; worker.start()

    def run_pipeline(self, piped_data: dict, cancel_event: threading.Event):
        if piped_data: self.set_input_from_derotate(piped_data)
        self._run(cancel_event)

    def _on_progress(self, step, total, msg):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._status_lbl.setText(msg)

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._progress.hide()
        if result.get("success"):
            self._status_lbl.setText(
                f"✓ Done — {result.get('n_frames',0)} frames → {result.get('output_path','')}")
        else:
            self._status_lbl.setText(f"✗ {result.get('error','')}")
        self.finished.emit(result)


class _ShiftAddWorker(QThread):
    """Built-in simple mean stack worker."""
    progress = pyqtSignal(int, int, str)
    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, config, cancel_event):
        super().__init__(); self.config = config; self._cancel = cancel_event

    def run(self):
        import traceback
        try:
            cfg  = self.config
            path = cfg["input_path"]
            top  = cfg["top_pct"]

            if cfg["input_type"] == "fits":
                files = sorted(
                    glob.glob(os.path.join(path, "*.fit")) +
                    glob.glob(os.path.join(path, "*.fits")) +
                    glob.glob(os.path.join(path, "*.fts")))
                n_use = max(1, int(len(files) * top / 100))
                files = files[:n_use]
                self.log_line.emit(f"Mean stacking {n_use} FITS files…")
                self.progress.emit(0, n_use, "Loading…")
                frames = []
                for i, f in enumerate(files):
                    if self._cancel.is_set(): break
                    try:
                        from astropy.io import fits as _fits
                        d = _fits.getdata(f).astype(np.float32)
                        if d.ndim == 3: d = d[0]
                        frames.append(d)
                    except Exception as e:
                        self.log_line.emit(f"  Warning: {f}: {e}")
                    self.progress.emit(i+1, n_use, f"Loaded {i+1}/{n_use}")
            else:
                from lucky_preprocessor import read_ser_header as _rsh, read_ser_frame as _rsf
                hdr = _rsh(path)
                n_total = hdr["frame_count"]
                n_use   = max(1, int(n_total * top / 100))
                self.log_line.emit(f"Mean stacking {n_use}/{n_total} SER frames…")
                frames  = []
                with open(path, "rb") as f:
                    for i in range(n_use):
                        if self._cancel.is_set(): break
                        frames.append(_rsf(f, hdr, i).astype(np.float32))
                        if i % 50 == 0:
                            self.progress.emit(i, n_use, f"Loaded {i}/{n_use}")

            if not frames:
                self.finished.emit({"success": False, "error": "No frames loaded"}); return

            self.progress.emit(n_use, n_use, "Stacking…")
            stack = np.mean(np.stack(frames, axis=0), axis=0)
            mn, mx = stack.min(), stack.max()
            if mx > mn: stack = (stack - mn) / (mx - mn)

            out_dir = cfg.get("output_dir") or (
                os.path.dirname(path) if not os.path.isdir(path) else path)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "shift_and_add_result.fit")
            from astropy.io import fits as _fits
            hdu = _fits.PrimaryHDU(data=stack.astype(np.float32))
            hdu.header["HISTORY"] = "Shift-and-add mean stack — lucky_suite.py"
            hdu.header["NFRAMES"] = len(frames)
            hdu.writeto(out_path, overwrite=True)
            self.log_line.emit(f"Saved: {out_path}")
            self.finished.emit({"success": True, "output_path": out_path,
                                "n_frames": len(frames)})
        except Exception as e:
            self.log_line.emit(f"ERROR: {e}\n{traceback.format_exc()}")
            self.finished.emit({"success": False, "error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW TAB  (B6 spec: planet selector, method selector, frame budget)
# ─────────────────────────────────────────────────────────────────────────────

class OverviewTab(QWidget):
    run_pipeline_requested   = pyqtSignal()
    run_step_requested       = pyqtSignal(str)  # "preprocess","derotate","reconstruct"
    cancel_requested         = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10); root.setSpacing(10)

        # Pipeline diagram
        diag_grp = QGroupBox("Pipeline")
        dl = QVBoxLayout(diag_grp)
        self._pipeline_widget = PipelineStatusWidget([
            {"name": "Preprocess\n(Quality gate)", "key": "preprocess"},
            {"name": "Derotate\n(Planet rotation)", "key": "derotate"},
            {"name": "Reconstruct\n(Lucky imaging)",  "key": "reconstruct"},
        ])
        self._pipeline_widget.setMinimumHeight(90); dl.addWidget(self._pipeline_widget)
        step_row = QHBoxLayout()
        for key, label in [("preprocess", "▶ Preprocess only"),
                            ("derotate",   "▶ Derotate only"),
                            ("reconstruct","▶ Reconstruct only")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, k=key: self.run_step_requested.emit(k))
            step_row.addWidget(btn)
        step_row.addStretch()
        dl.addLayout(step_row)
        root.addWidget(diag_grp)

        # Planet selector (shared shortcut — mirrors DerotatePanel)
        planet_grp = QGroupBox("Planet (shortcut — also configurable in Derotate tab)")
        pl = QHBoxLayout(planet_grp)
        pl.addWidget(QLabel("Planet:"))
        self._planet_combo = QComboBox()
        for name in PLANETS:
            self._planet_combo.addItem(f"{PLANETS[name]['emoji']} {name}")
        self._planet_combo.addItem("🌙 Moon"); self._planet_combo.addItem("✏ Custom")
        pl.addWidget(self._planet_combo)
        self._lbl_session_warning = QLabel("")
        self._lbl_session_warning.setStyleSheet(f"color:{SIRIL_WARNING}; font-weight:bold;")
        pl.addWidget(self._lbl_session_warning)
        pl.addStretch()
        self._planet_combo.currentTextChanged.connect(self._on_planet_changed)
        self._on_planet_changed(self._planet_combo.currentText())
        root.addWidget(planet_grp)

        # Reconstruction method selector (B6 spec)
        recon_grp = QGroupBox("Reconstruction method")
        rl = QVBoxLayout(recon_grp)
        self._recon_btn_grp = QButtonGroup(self)
        methods = [
            ("🌟  Speckle Holography  (all frames, Fourier domain — fast)",       "speckle"),
            ("🔥  The Thresher  (all frames, SGD blind deconvolution — GPU recommended)",  "thresher"),
            ("➕  Shift-and-Add  (top X%, simple mean — fast baseline)",           "shift_add"),
            ("🔀  All three  (compare results — runs speckle, thresher, shift-add)", "all_three"),
        ]
        self._recon_radios: dict[str, QRadioButton] = {}
        for label, key in methods:
            rb = QRadioButton(label)
            self._recon_btn_grp.addButton(rb)
            rl.addWidget(rb)
            self._recon_radios[key] = rb
        self._recon_radios["speckle"].setChecked(True)

        # PyTorch warning
        if not HAS_TORCH:
            self._torch_warn = QLabel(
                "⚠ The Thresher requires PyTorch — not installed.\n"
                "Install: pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu")
            self._torch_warn.setStyleSheet(f"color:{SIRIL_WARNING}; font-size:8pt;")
            self._torch_warn.setWordWrap(True); rl.addWidget(self._torch_warn)
        root.addWidget(recon_grp)

        # Frame budget display (B6 spec)
        self._budget_grp = QGroupBox("Frame budget")
        bl = QVBoxLayout(self._budget_grp)
        self._lbl_total_frames   = QLabel("Total frames: —")
        self._lbl_selected_frames = QLabel("Selected: —")
        self._lbl_est_derot_time = QLabel("Estimated derotation time: —")
        self._lbl_est_recon_time = QLabel("Estimated reconstruction time: —")
        for lbl in [self._lbl_total_frames, self._lbl_selected_frames,
                    self._lbl_est_derot_time, self._lbl_est_recon_time]:
            lbl.setObjectName("dim"); bl.addWidget(lbl)
        self._budget_grp.setVisible(False)
        root.addWidget(self._budget_grp)

        # Result summary
        self._result_grp = QGroupBox("Last result")
        self._result_grp.setVisible(False)
        rl2 = QVBoxLayout(self._result_grp)
        self._result_lbl = QLabel(""); self._result_lbl.setWordWrap(True)
        rl2.addWidget(self._result_lbl)
        root.addWidget(self._result_grp)

        root.addStretch()

        # Action row
        action_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run full pipeline")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(36); self._btn_run.setMinimumWidth(160)
        self._btn_run.clicked.connect(self.run_pipeline_requested)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(36); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self.cancel_requested)
        action_row.addWidget(self._btn_run); action_row.addWidget(self._btn_cancel)
        action_row.addStretch()
        root.addLayout(action_row)

    def get_recon_method(self) -> str:
        for key, rb in self._recon_radios.items():
            if rb.isChecked(): return key
        return "speckle"

    def get_planet_name(self) -> str:
        t = self._planet_combo.currentText()
        return t.split(" ", 1)[-1] if " " in t else t

    def set_step_status(self, key: str, status: str):
        self._pipeline_widget.set_status(key, status)

    def set_running(self, running: bool):
        self._btn_run.setEnabled(not running)
        self._btn_cancel.setEnabled(running)

    def update_frame_budget(self, total: int, selected: int):
        self._budget_grp.setVisible(True)
        self._lbl_total_frames.setText(f"Total frames: {total:,}")
        self._lbl_selected_frames.setText(
            f"Selected top {selected:,} / {total:,} frames  "
            f"({100*selected/max(total,1):.1f}%)")
        # Very rough estimates
        derot_est = total * 0.002
        recon_est = selected * 0.05
        self._lbl_est_derot_time.setText(
            f"Estimated derotation time: ~{derot_est:.0f}s")
        self._lbl_est_recon_time.setText(
            f"Estimated reconstruction time: ~{recon_est:.0f}s")

    def show_result(self, step: str, result: dict):
        self._result_grp.setVisible(True)
        if step == "reconstruct":
            out = result.get("output_path", "—")
            n   = result.get("n_frames_used", result.get("n_frames", 0))
            self._result_lbl.setText(
                f"✓ Reconstruction complete\n"
                f"Frames used: {n}  |  Output: {out}")
            self._result_grp.setTitle("Last result — Reconstruct")
        elif step == "derotate":
            rot = result.get("total_rotation", 0.0)
            out = result.get("output_path", "—")
            self._result_lbl.setText(
                f"✓ Derotation complete\n"
                f"Total rotation corrected: {rot:.2f}°  |  Output: {out}")
            self._result_grp.setTitle("Last result — Derotate")

    def _on_planet_changed(self, text: str):
        name = text.split(" ", 1)[-1] if " " in text else text
        if name in PLANETS:
            safe = PLANETS[name].get("safe_session_min", 0)
            if safe <= 5:
                self._lbl_session_warning.setText(
                    f"⚠ {name}: session ≤ {safe} min recommended")
            else:
                self._lbl_session_warning.setText(
                    f"Safe session: ≤ {safe} min")
        else:
            self._lbl_session_warning.setText("")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lucky Imaging Suite  —  Siril")
        self.resize(1400, 880)

        # Pipeline state (A6)
        self._pipeline_step   = 0
        self._pipeline_data   = {}
        self._current_worker  = None
        self._cancel_event    = threading.Event()
        self._recon_queue     = []   # for "all three" mode

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(
            f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon  = QLabel("⚡"); lbl_icon.setStyleSheet("font-size:18pt; background:transparent;")
        lbl_title = QLabel("Lucky Imaging Suite")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold; background:transparent;")
        lbl_ver   = QLabel("v1.0  |  4 scripts  |  Preprocess → Derotate → Reconstruct")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title); hl.addStretch(); hl.addWidget(lbl_ver)
        root.addWidget(header)

        # ── Tab widget ─────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        # Shared pipeline log
        self._pipeline_log = QPlainTextEdit(); self._pipeline_log.setReadOnly(True)

        # Tab 0 — Overview
        self._overview = OverviewTab()
        self._overview.run_pipeline_requested.connect(self._run_pipeline)
        self._overview.run_step_requested.connect(self._run_step)
        self._overview.cancel_requested.connect(self._cancel_pipeline)
        self._tabs.addTab(self._overview, "🚀  Overview")

        # Tab 1 — Preprocess
        self._preprocess_panel = PreprocessPanel()
        self._preprocess_panel.finished.connect(
            lambda r: self._on_step_done("preprocess", r))
        self._preprocess_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._preprocess_panel, "⚡  Preprocess")

        # Tab 2 — Derotate
        self._derotate_panel = DerotatePanel()
        self._derotate_panel.finished.connect(
            lambda r: self._on_step_done("derotate", r))
        self._derotate_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._derotate_panel, "🪐  Derotate")

        # Tab 3 — Speckle Holography
        self._speckle_panel = SpecklePanel()
        self._speckle_panel.finished.connect(
            lambda r: self._on_step_done("reconstruct", r))
        self._speckle_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._speckle_panel, "🌟  Speckle")

        # Tab 4 — The Thresher
        self._thresher_panel = ThresherPanel()
        self._thresher_panel.finished.connect(
            lambda r: self._on_step_done("reconstruct", r))
        self._thresher_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._thresher_panel, "🔥  Thresher")

        # Tab 5 — Shift-and-Add
        self._shift_add_panel = ShiftAddPanel()
        self._shift_add_panel.finished.connect(
            lambda r: self._on_step_done("reconstruct", r))
        self._shift_add_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._shift_add_panel, "➕  Shift-Add")

        # Tab 6 — Pipeline Log
        log_wrap = QWidget(); lw = QVBoxLayout(log_wrap)
        lw.setContentsMargins(6, 6, 6, 6)
        lbl_log = QLabel("Combined log from all pipeline steps"); lbl_log.setObjectName("dim")
        lw.addWidget(lbl_log)
        btn_clear = QPushButton("Clear log"); btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._pipeline_log.clear)
        lw.addWidget(btn_clear); lw.addWidget(self._pipeline_log)
        self._tabs.addTab(log_wrap, "📋  Pipeline Log")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom = QWidget()
        bottom.setFixedHeight(46)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 4, 10, 4); bl.setSpacing(8)
        self._global_progress = QProgressBar()
        self._global_progress.setFixedHeight(8); self._global_progress.setRange(0, 3)
        self._global_progress.setValue(0); bl.addWidget(self._global_progress, 1)
        self._btn_run = QPushButton("▶  Run pipeline")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(32); self._btn_run.setMinimumWidth(140)
        self._btn_run.clicked.connect(self._run_pipeline); bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(32); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_pipeline); bl.addWidget(self._btn_cancel)
        self._status_lbl = QLabel("Ready — configure each tab, then run.")
        self._status_lbl.setObjectName("dim")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status_lbl, 1)
        root.addWidget(bottom)

        # Startup log
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] Lucky Imaging Suite — ready")
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "Pipeline: Preprocess → Derotate → Reconstruct (Speckle / Thresher / Shift-Add)")

    # ── Pipeline orchestration (Section A6) ───────────────────────────────────

    def _run_pipeline(self):
        # Validate step 1 input
        proj = self._preprocess_panel._proj_edit.text().strip()
        if not proj or not os.path.isdir(proj):
            QMessageBox.warning(self, "Missing input",
                "Set the project folder in the Preprocess tab.")
            self._tabs.setCurrentIndex(1); return
        if not self._preprocess_panel._file_infos:
            QMessageBox.warning(self, "Not scanned",
                "Scan the project folder in the Preprocess tab first.")
            self._tabs.setCurrentIndex(1); return

        self._cancel_event.clear()
        self._pipeline_step = 0
        self._pipeline_data = {}
        self._recon_queue   = []
        for key in ("preprocess", "derotate", "reconstruct"):
            self._overview.set_step_status(key, "idle")
        self._set_running(True)
        self._global_progress.setValue(0)
        self._run_preprocess_step()

    def _run_preprocess_step(self):
        self._overview.set_step_status("preprocess", "running")
        self._set_status("Running preprocessor…", SIRIL_ACCENT)
        self._log_pipeline("═══ STEP 1: Lucky Preprocessor ═══")
        self._tabs.setCurrentIndex(1)
        self._preprocess_panel.run_for_pipeline(self._cancel_event)

    def _run_derotate_step(self, piped: dict):
        self._overview.set_step_status("derotate", "running")
        self._global_progress.setValue(1)
        self._set_status("Running derotation…", SIRIL_ACCENT)
        self._log_pipeline("═══ STEP 2: Planet Derotation ═══")
        if piped:
            self._log_pipeline(
                f"In-memory handoff: {piped.get('n_selected',0)} frames → Derotate")

        # Sync planet from overview to derotate panel
        planet_text = self._overview._planet_combo.currentText()
        planet_name = planet_text.split(" ", 1)[-1] if " " in planet_text else planet_text
        if planet_name in PLANETS:
            # Find matching text in derotate combo
            for i in range(self._derotate_panel._planet_combo.count()):
                if planet_name in self._derotate_panel._planet_combo.itemText(i):
                    self._derotate_panel._planet_combo.setCurrentIndex(i)
                    break

        self._tabs.setCurrentIndex(2)
        self._derotate_panel.run_pipeline(piped, self._cancel_event)

    def _run_reconstruct_step(self, piped: dict):
        method = self._overview.get_recon_method()
        self._overview.set_step_status("reconstruct", "running")
        self._global_progress.setValue(2)
        self._set_status(f"Reconstructing ({method})…", SIRIL_ACCENT)
        self._log_pipeline(f"═══ STEP 3: Reconstruct ({method}) ═══")
        if piped:
            n = piped.get("frame_count", 0)
            self._log_pipeline(f"In-memory handoff: {n} derotated frames → Reconstruct")

        if method == "all_three":
            self._recon_queue = ["speckle", "thresher", "shift_add"]
            self._run_next_recon(piped)
        else:
            self._dispatch_recon(method, piped)

    def _run_next_recon(self, piped: dict):
        if not self._recon_queue:
            self._on_pipeline_complete()
            return
        method = self._recon_queue.pop(0)
        self._log_pipeline(f"─── Reconstruct: {method} ───")
        self._dispatch_recon(method, piped)

    def _dispatch_recon(self, method: str, piped: dict):
        tab_map = {"speckle": (self._speckle_panel, 3),
                   "thresher": (self._thresher_panel, 4),
                   "shift_add": (self._shift_add_panel, 5)}
        if method not in tab_map:
            self._log_pipeline(f"Unknown method: {method}"); self._on_pipeline_complete(); return
        panel, tab_idx = tab_map[method]
        self._tabs.setCurrentIndex(tab_idx)
        panel.run_pipeline(piped, self._cancel_event)

    def _on_step_done(self, step: str, result: dict):
        if result.get("success"):
            if step == "preprocess":
                self._overview.set_step_status("preprocess", "done")
                n_sel = result.get("n_selected", 0)
                n_tot = result.get("n_selected", 0)  # best we have
                self._overview.update_frame_budget(n_tot, n_sel)
                self._pipeline_data.update(result)
                self._log_pipeline(
                    f"✓ Preprocess done — {n_sel} frames selected")
                self._run_derotate_step(result)

            elif step == "derotate":
                self._overview.set_step_status("derotate", "done")
                self._pipeline_data.update(result)
                self._overview.show_result("derotate", result)
                n = result.get("frame_count", 0)
                self._log_pipeline(
                    f"✓ Derotate done — {n} frames, "
                    f"{result.get('session_min',0):.1f} min session")
                self._run_reconstruct_step(result)

            elif step == "reconstruct":
                if self._recon_queue:
                    # More reconstruction modes to run (all_three)
                    self._log_pipeline(
                        f"✓ Reconstruction step done → {self._recon_queue[0]} next")
                    self._run_next_recon(self._pipeline_data)
                else:
                    self._overview.set_step_status("reconstruct", "done")
                    self._overview.show_result("reconstruct", result)
                    self._on_pipeline_complete()
        else:
            err = result.get("error", "Unknown")
            self._overview.set_step_status(step, "error")
            self._set_status(f"✗ {step.title()} failed: {err}", SIRIL_ERROR)
            self._log_pipeline(f"✗ {step.upper()} FAILED: {err}")
            self._set_running(False)

    def _on_pipeline_complete(self):
        self._global_progress.setValue(3)
        self._set_status("✓ Pipeline complete!", SIRIL_SUCCESS)
        self._log_pipeline("✓ Lucky Imaging Suite pipeline complete!")
        self._set_running(False)

    # ── Standalone step runners (A8) ──────────────────────────────────────────

    def _run_step(self, step: str):
        self._cancel_event.clear()
        self._set_running(True)
        self._pipeline_data = {}
        self._recon_queue   = []

        if step == "preprocess":
            proj = self._preprocess_panel._proj_edit.text().strip()
            if not proj or not os.path.isdir(proj):
                QMessageBox.warning(self, "Missing input",
                    "Set the project folder in the Preprocess tab.")
                self._tabs.setCurrentIndex(1); self._set_running(False); return
            self._log_pipeline("─── Standalone: Preprocess ───")
            self._overview.set_step_status("preprocess", "running")
            self._tabs.setCurrentIndex(1)
            self._preprocess_panel.run_for_pipeline(self._cancel_event)

        elif step == "derotate":
            path = self._derotate_panel._input_path_edit.text().strip()
            if not path:
                QMessageBox.warning(self, "Missing input",
                    "Set the input file in the Derotate tab.")
                self._tabs.setCurrentIndex(2); self._set_running(False); return
            self._log_pipeline("─── Standalone: Derotate ───")
            self._overview.set_step_status("derotate", "running")
            self._tabs.setCurrentIndex(2)
            self._derotate_panel.run_pipeline(None, self._cancel_event)

        elif step == "reconstruct":
            method = self._overview.get_recon_method()
            # Validate input in the relevant panel
            panel_map = {
                "speckle":   (self._speckle_panel, 3),
                "thresher":  (self._thresher_panel, 4),
                "shift_add": (self._shift_add_panel, 5),
                "all_three": (self._speckle_panel, 3),
            }
            _, tab_idx = panel_map.get(method, (self._speckle_panel, 3))
            self._log_pipeline(f"─── Standalone: Reconstruct ({method}) ───")
            self._overview.set_step_status("reconstruct", "running")
            if method == "all_three":
                self._recon_queue = ["speckle", "thresher", "shift_add"]
                self._run_next_recon({})
            else:
                self._tabs.setCurrentIndex(tab_idx)
                self._dispatch_recon(method, {})

    # ── Cancel (A9) ───────────────────────────────────────────────────────────

    def _cancel_pipeline(self):
        self._cancel_event.set()
        self._recon_queue = []
        for panel in [self._preprocess_panel, self._derotate_panel,
                      self._speckle_panel, self._thresher_panel, self._shift_add_panel]:
            try: panel.cancel()
            except Exception: pass
        for key in ("preprocess", "derotate", "reconstruct"):
            if self._overview._pipeline_widget.statuses.get(key) == "running":
                self._overview.set_step_status(key, "error")
        self._set_status("Pipeline cancelled.", SIRIL_WARNING)
        self._log_pipeline("CANCELLED")
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

    def _log_pipeline(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._pipeline_log.appendPlainText(f"[{ts}] {msg}")


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
