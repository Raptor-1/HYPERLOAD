r"""
stacking_suite.py  —  Script Merger: B4 Stacking Suite
=======================================================
Merged launcher for:
    🏆  Subframe Auto-Ranking     (subframe_ranking.py)
    ⚡  Winsorized Sigma Stack    (VectorisedWinsorizedSigmaStack.py)

Architecture: Section A sequential pipeline (2 steps)
    RANK → STACK

In-memory handoff (Section A4):
    Ranker emits approved frame paths → Stacker receives them, writes a
    temp folder of approved-only frames (symlinks or copies), then imports
    the sequence into Siril for winsorized stacking.
    No intermediate CSV needed.

Pipeline flow:
    1. Load FITS folder in Ranker → measure FWHM/Ecc/SNR/BG/Stars
    2. Score + threshold → approved frame list
    3. Export approved frames → temp folder
    4. Auto-import as Siril sequence
    5. Run vectorized winsorized sigma-clip stack → master.fit

References:
    Winsorized sigma clipping: Rousseeuw & Croux (1993), JASA 88(424):1273-1283
    SubframeSelector equivalent: PixInsight documentation

Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:  Siril → Scripts → stacking_suite
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
import csv
import glob
import shutil
import json
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

from subframe_ranking import (
    RankingWorker,
    ScoreCanvas,
    ThumbnailDialog,
    # Algorithms
    measure_frame_fwhm,
    measure_frame_snr,
    measure_background,
    measure_star_count,
    compute_population_stats,
    score_frame,
    find_elbow_threshold,
    apply_threshold_and_export,
    _write_ranking_csv,
    # Constants
    METRIC_KEYS,
    METRIC_LABELS,
    DEFAULT_WEIGHTS,
    TABLE_COLS,
    OUTPUT_MODES,
    OUTPUT_MODE_KEYS,
    # Theme
    SIRIL_STYLESHEET,
    SIRIL_BG, SIRIL_BG2, SIRIL_BG3,
    SIRIL_ACCENT, SIRIL_ACCENT2,
    SIRIL_TEXT, SIRIL_TEXT_DIM,
    SIRIL_BORDER, SIRIL_SUCCESS,
    SIRIL_WARNING, SIRIL_SECTION,
    SIRIL_ERROR,
)

from VectorisedWinsorizedSigmaStack import (
    StackWorker,
    ImportWorker,
    WinsorizedStackUI,
    VERSION as STACK_VERSION,
)

import numpy as np
from astropy.io import fits as astropy_fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QTabWidget, QComboBox,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QDialog, QScrollArea, QFrame,
    QTextEdit, QGridLayout,
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QTextCursor

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
        bw  = min(200, (W - 40) // n - 20)
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
            f2 = QFont("Segoe UI", 8)
            p.setFont(f2)
            p.drawText(QRectF(bx + 30, by, bw - 34, bh),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       step["name"])

            if i < n - 1:
                ax = bx + bw + 4; ay = by + bh // 2
                p.setPen(QPen(QColor("#3a4055"), 1.5))
                p.drawLine(int(ax), int(ay), int(ax + gap - 8), int(ay))
                p.drawLine(int(ax+gap-8), int(ay), int(ax+gap-14), int(ay-5))
                p.drawLine(int(ax+gap-8), int(ay), int(ax+gap-14), int(ay+5))
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# RANKING PANEL  (full Subframe Ranking UI)
# ─────────────────────────────────────────────────────────────────────────────

class RankingPanel(QWidget):
    """
    Full Subframe Auto-Ranking UI.
    Signals: ranking_complete(list) — emits approved frame paths for handoff
             finished(dict)
             log_line(str)
    """
    ranking_complete = pyqtSignal(list)  # approved_paths list for pipeline
    finished         = pyqtSignal(dict)
    log_line         = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_results       = []
        self._fits_files          = []
        self._cancel_event        = threading.Event()
        self._worker              = None
        self._weights_updating    = False
        self._threshold_updating  = False
        self._thumbnail_dialogs   = []
        self._threshold           = 70
        self._elbow_threshold     = None
        self._build_ui()
        self._set_running_state(False)
        self._set_ranked_state(False)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._make_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 870])
        root.addWidget(splitter, 1)
        root.addWidget(self._make_bottom_row())

        self._status = QLabel("Ready — select a folder and scan for FITS files")
        self._status.setObjectName("dim")
        root.addWidget(self._status)

    def _make_left_panel(self) -> QWidget:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.setFixedWidth(340)
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(8)
        vl.addWidget(self._make_group_input())
        vl.addWidget(self._make_group_pixscale())
        vl.addWidget(self._make_group_weights())
        vl.addWidget(self._make_group_threshold())
        vl.addWidget(self._make_group_output())
        vl.addStretch()
        outer.setWidget(container)
        return outer

    def _make_group_input(self) -> QGroupBox:
        grp = QGroupBox("Input Sequence")
        vl  = QVBoxLayout(grp); vl.setSpacing(5)
        hl = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("FITS folder…")
        btn_browse = QPushButton("Browse"); btn_browse.setFixedWidth(70)
        btn_browse.clicked.connect(self._browse_input)
        hl.addWidget(self._folder_edit); hl.addWidget(btn_browse)
        vl.addLayout(hl)
        ext_row = QHBoxLayout()
        ext_row.addWidget(QLabel("Extension:"))
        self._ext_combo = QComboBox()
        self._ext_combo.addItems([".fit", ".fits", ".fts", "All FITS"])
        ext_row.addWidget(self._ext_combo); ext_row.addStretch()
        vl.addLayout(ext_row)
        self._btn_scan = QPushButton("🔍  Scan for FITS")
        self._btn_scan.setObjectName("primary")
        self._btn_scan.clicked.connect(self._scan_folder)
        vl.addWidget(self._btn_scan)
        self._lbl_file_info = QLabel("No files loaded")
        self._lbl_file_info.setObjectName("dim")
        vl.addWidget(self._lbl_file_info)
        return grp

    def _make_group_pixscale(self) -> QGroupBox:
        grp = QGroupBox("Pixel Scale (optional)")
        vl  = QVBoxLayout(grp); vl.setSpacing(5)
        self._pixscale_mode = QComboBox()
        self._pixscale_mode.addItems(["Manual entry", "Skip (use pixels)"])
        self._pixscale_mode.currentIndexChanged.connect(self._on_pixscale_mode)
        vl.addWidget(self._pixscale_mode)
        self._pixscale_spin = QDoubleSpinBox()
        self._pixscale_spin.setRange(0.01, 20.0); self._pixscale_spin.setValue(1.0)
        self._pixscale_spin.setSuffix(" arcsec/px"); self._pixscale_spin.setDecimals(3)
        vl.addWidget(self._pixscale_spin)
        hint = QLabel("Used to display FWHM in arcsec instead of pixels")
        hint.setObjectName("dim"); hint.setWordWrap(True)
        vl.addWidget(hint)
        return grp

    def _make_group_weights(self) -> QGroupBox:
        grp = QGroupBox("Metric Weights")
        vl  = QVBoxLayout(grp); vl.setSpacing(4)
        hint = QLabel("Sliders auto-normalise to 100%")
        hint.setObjectName("dim"); vl.addWidget(hint)
        self._weight_sliders = []
        self._weight_labels  = []
        for i, (key, label, default) in enumerate(
                zip(METRIC_KEYS, METRIC_LABELS, DEFAULT_WEIGHTS)):
            row = QHBoxLayout()
            lbl_name = QLabel(label); lbl_name.setFixedWidth(90)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 100); sl.setValue(default)
            lbl_pct = QLabel(f"{default}%")
            lbl_pct.setFixedWidth(36)
            lbl_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl_pct.setStyleSheet(f"color: {SIRIL_ACCENT};")
            sl.valueChanged.connect(lambda val, idx=i: self._on_weight_changed(idx))
            row.addWidget(lbl_name); row.addWidget(sl); row.addWidget(lbl_pct)
            vl.addLayout(row)
            self._weight_sliders.append(sl); self._weight_labels.append(lbl_pct)
        row_btns = QHBoxLayout()
        btn_reset = QPushButton("Reset to defaults")
        btn_reset.clicked.connect(self._reset_weights); row_btns.addWidget(btn_reset)
        self._btn_recompute = QPushButton("Recompute scores")
        self._btn_recompute.clicked.connect(self._recompute_scores)
        self._btn_recompute.setEnabled(False); row_btns.addWidget(self._btn_recompute)
        vl.addLayout(row_btns)
        return grp

    def _make_group_threshold(self) -> QGroupBox:
        grp = QGroupBox("Rejection Threshold")
        vl  = QVBoxLayout(grp); vl.setSpacing(5)
        vl.addWidget(QLabel("Reject frames scoring below:"))
        thr_row = QHBoxLayout()
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 100); self._threshold_slider.setValue(70)
        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(0, 100); self._threshold_spin.setValue(70)
        self._threshold_spin.setFixedWidth(55)
        thr_row.addWidget(self._threshold_slider); thr_row.addWidget(self._threshold_spin)
        vl.addLayout(thr_row)
        self._threshold_slider.valueChanged.connect(
            lambda v: self._on_threshold_changed(v))
        self._threshold_spin.valueChanged.connect(
            lambda v: self._on_threshold_changed(v))
        self._lbl_approval = QLabel("No frames ranked yet"); self._lbl_approval.setObjectName("warn")
        vl.addWidget(self._lbl_approval)
        self._btn_use_elbow = QPushButton("Use Auto (elbow) threshold")
        self._btn_use_elbow.clicked.connect(self._use_elbow_threshold)
        self._btn_use_elbow.setEnabled(False); vl.addWidget(self._btn_use_elbow)
        hint = QLabel("Drag the orange line in the chart to adjust")
        hint.setObjectName("dim"); vl.addWidget(hint)
        return grp

    def _make_group_output(self) -> QGroupBox:
        grp = QGroupBox("Export (standalone mode)")
        vl  = QVBoxLayout(grp); vl.setSpacing(5)
        vl.addWidget(QLabel("Output mode:"))
        self._output_mode_combo = QComboBox()
        self._output_mode_combo.addItems(OUTPUT_MODES)
        vl.addWidget(self._output_mode_combo)
        vl.addWidget(QLabel("Output folder:"))
        out_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        self._output_folder_edit.setPlaceholderText("Default: [input]_ranked/")
        btn_out = QPushButton("Browse"); btn_out.setFixedWidth(70)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(self._output_folder_edit); out_row.addWidget(btn_out)
        vl.addLayout(out_row)
        self._chk_csv = QCheckBox("Export ranking CSV"); self._chk_csv.setChecked(True)
        vl.addWidget(self._chk_csv)
        return grp

    def _make_right_panel(self) -> QWidget:
        outer = QWidget()
        vl    = QVBoxLayout(outer)
        vl.setContentsMargins(2, 0, 2, 0); vl.setSpacing(4)
        self._tabs = QTabWidget()

        # Tab 0 — Distribution
        dist_tab = QWidget()
        dt = QVBoxLayout(dist_tab); dt.setContentsMargins(4, 4, 4, 4)
        self._score_canvas = ScoreCanvas()
        self._score_canvas.threshold_changed.connect(self._on_threshold_changed)
        dt.addWidget(self._score_canvas, 1)
        dt.addWidget(self._make_stats_group())
        self._tabs.addTab(dist_tab, "📊  Distribution")

        # Tab 1 — Rankings table
        rank_tab = QWidget()
        rt = QVBoxLayout(rank_tab); rt.setContentsMargins(4, 4, 4, 4)
        self._table = QTableWidget()
        self._table.setColumnCount(len(TABLE_COLS))
        self._table.setHorizontalHeaderLabels(TABLE_COLS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        rt.addWidget(self._table, 1)
        tbl_btns = QHBoxLayout()
        self._btn_export_csv = QPushButton("💾  Export CSV")
        self._btn_export_csv.setObjectName("export")
        self._btn_export_csv.clicked.connect(self._export_csv_only)
        self._btn_copy_approved = QPushButton("📋  Copy approved paths")
        self._btn_copy_approved.clicked.connect(self._copy_approved_paths)
        self._btn_open_output = QPushButton("📂  Open output")
        self._btn_open_output.clicked.connect(self._open_output_folder)
        for btn in [self._btn_export_csv, self._btn_copy_approved, self._btn_open_output]:
            tbl_btns.addWidget(btn)
        tbl_btns.addStretch()
        rt.addLayout(tbl_btns)
        self._tabs.addTab(rank_tab, "🏆  Rankings")

        # Tab 2 — Log
        log_tab = QWidget()
        ll = QVBoxLayout(log_tab); ll.setContentsMargins(4, 4, 4, 4)
        self._log_view = QPlainTextEdit(); self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Courier New", 9))
        ll.addWidget(self._log_view)
        btn_clear = QPushButton("Clear log"); btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._log_view.clear)
        ll.addWidget(btn_clear)
        self._tabs.addTab(log_tab, "📋  Log")

        vl.addWidget(self._tabs)
        return outer

    def _make_stats_group(self) -> QGroupBox:
        grp = QGroupBox("Summary Statistics")
        hl  = QHBoxLayout(grp); hl.setSpacing(20)
        self._stat_labels = {}
        for key, caption in [("fwhm_med", "Median FWHM:"), ("ecc_med", "Median Ecc:"),
                              ("snr_med", "Median SNR:"), ("best", "Best score:"),
                              ("worst", "Worst score:")]:
            col = QVBoxLayout()
            cap = QLabel(caption); cap.setObjectName("dim")
            val = QLabel("—")
            val.setStyleSheet(f"color: {SIRIL_ACCENT}; font-weight: bold; font-size: 11pt;")
            col.addWidget(cap); col.addWidget(val)
            hl.addLayout(col); self._stat_labels[key] = val
        hl.addStretch()
        return grp

    def _make_bottom_row(self) -> QWidget:
        bar = QWidget(); bar.setFixedHeight(40)
        hl = QHBoxLayout(bar); hl.setContentsMargins(0, 2, 0, 2); hl.setSpacing(6)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(8); self._progress.setVisible(False)
        hl.addWidget(self._progress, 1)
        self._btn_rank = QPushButton("▶  Rank Frames")
        self._btn_rank.setObjectName("primary"); self._btn_rank.setFixedHeight(30)
        self._btn_rank.clicked.connect(self._start_ranking); hl.addWidget(self._btn_rank)
        self._btn_apply = QPushButton("✓  Export approved frames")
        self._btn_apply.setObjectName("export"); self._btn_apply.setFixedHeight(30)
        self._btn_apply.clicked.connect(self._apply_and_export); hl.addWidget(self._btn_apply)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setFixedHeight(30)
        self._btn_cancel.clicked.connect(self._cancel); hl.addWidget(self._btn_cancel)
        return bar

    # ── State helpers ─────────────────────────────────────────────────────────

    def _set_running_state(self, running: bool):
        self._btn_rank.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._btn_scan.setEnabled(not running)
        self._progress.setVisible(running)

    def _set_ranked_state(self, ranked: bool):
        self._btn_apply.setEnabled(ranked)
        self._btn_export_csv.setEnabled(ranked)
        self._btn_copy_approved.setEnabled(ranked)
        self._btn_open_output.setEnabled(ranked)
        self._btn_recompute.setEnabled(ranked)
        self._btn_use_elbow.setEnabled(ranked)

    # ── Public interface ──────────────────────────────────────────────────────

    def get_approved_paths(self) -> list:
        """Return paths of all frames currently above threshold."""
        return [f["path"] for f in self._frame_results
                if f["score"] >= self._threshold]

    def get_frame_results(self) -> list:
        return list(self._frame_results)

    def get_threshold(self) -> float:
        return float(self._threshold)

    def get_output_dir(self) -> str:
        out = self._output_folder_edit.text().strip()
        if not out:
            folder = self._folder_edit.text().strip()
            out = (folder.rstrip("/\\") + "_ranked") if folder else "ranked"
        return out

    # ── Browse / scan ─────────────────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select FITS folder", "")
        if folder:
            self._folder_edit.setText(folder)
            default_out = folder.rstrip("/\\") + "_ranked"
            self._output_folder_edit.setText(default_out)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", "")
        if folder: self._output_folder_edit.setText(folder)

    def _on_pixscale_mode(self, idx):
        self._pixscale_spin.setVisible(idx == 0)

    def _scan_folder(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Please enter a valid FITS folder path.")
            return
        ext_text = self._ext_combo.currentText()
        if ext_text == "All FITS":
            patterns = ["*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS"]
        else:
            patterns = [f"*{ext_text}", f"*{ext_text.upper()}"]
        files = []
        for pat in patterns: files += glob.glob(os.path.join(folder, pat))
        files = sorted(set(files))
        if not files:
            self._lbl_file_info.setText("No FITS files found"); return
        self._fits_files = files
        try:
            with astropy_fits.open(files[0]) as hdul:
                d = hdul[0].data
                shape_str = f"{d.shape[-1]}×{d.shape[-2]}"
        except Exception:
            shape_str = "?"
        msg = f"{len(files)} FITS files  |  {shape_str} px"
        self._lbl_file_info.setText(msg)
        self._lbl_file_info.setStyleSheet(f"color: {SIRIL_SUCCESS};")
        self._log(f"Scanned: {len(files)} files in {folder}")
        if not self._output_folder_edit.text().strip():
            self._output_folder_edit.setText(folder.rstrip("/\\") + "_ranked")

    # ── Weight sliders ────────────────────────────────────────────────────────

    def _on_weight_changed(self, changed_idx: int):
        if self._weights_updating: return
        self._weights_updating = True
        try:
            vals = [sl.value() for sl in self._weight_sliders]
            new_val = vals[changed_idx]
            others  = [i for i in range(len(vals)) if i != changed_idx]
            sum_others_old = sum(vals[i] for i in others)
            sum_others_new = 100 - new_val
            if sum_others_old > 0:
                for i in others:
                    vals[i] = int(round(vals[i] * sum_others_new / sum_others_old))
            elif sum_others_new > 0 and others:
                per = sum_others_new // len(others)
                for i in others: vals[i] = per
            else:
                for i in others: vals[i] = 0
            total = sum(vals); diff = 100 - total
            if diff != 0 and others:
                biggest = max(others, key=lambda i: vals[i])
                vals[biggest] += diff
            for i, (sl, lbl) in enumerate(zip(self._weight_sliders, self._weight_labels)):
                sl.setValue(vals[i]); lbl.setText(f"{vals[i]}%")
        finally:
            self._weights_updating = False

    def _reset_weights(self):
        self._weights_updating = True
        for sl, lbl, default in zip(self._weight_sliders, self._weight_labels, DEFAULT_WEIGHTS):
            sl.setValue(default); lbl.setText(f"{default}%")
        self._weights_updating = False

    def _get_weights_dict(self) -> dict:
        vals  = [sl.value() for sl in self._weight_sliders]
        total = max(sum(vals), 1)
        return {k: v / total for k, v in zip(METRIC_KEYS, vals)}

    # ── Threshold ─────────────────────────────────────────────────────────────

    def _on_threshold_changed(self, value):
        if self._threshold_updating: return
        self._threshold_updating = True
        try:
            v = int(round(float(value))); v = max(0, min(100, v))
            self._threshold = v
            self._threshold_slider.setValue(v); self._threshold_spin.setValue(v)
            self._update_approval_label()
            if self._frame_results:
                self._score_canvas.update_threshold(v)
                self._update_table_colors()
        finally:
            self._threshold_updating = False

    def _update_approval_label(self):
        if not self._frame_results:
            self._lbl_approval.setText("No frames ranked yet"); return
        n_total = len(self._frame_results)
        n_app   = sum(1 for f in self._frame_results if f["score"] >= self._threshold)
        pct     = n_app / max(n_total, 1) * 100
        self._lbl_approval.setText(f"Approving {n_app}/{n_total} frames  ({pct:.0f}%)")
        self._lbl_approval.setStyleSheet(
            f"color: {SIRIL_SUCCESS};" if pct >= 50 else f"color: {SIRIL_WARNING};")

    def _use_elbow_threshold(self):
        if self._elbow_threshold is not None:
            self._on_threshold_changed(self._elbow_threshold)

    # ── Ranking worker ─────────────────────────────────────────────────────────

    def _start_ranking(self):
        if not self._fits_files:
            QMessageBox.warning(self, "No files", "Please scan a folder first."); return
        self._cancel_event.clear()
        self._log("=" * 60)
        self._log(f"Starting ranking — {datetime.now().strftime('%H:%M:%S')}")
        pixel_scale = None if "Skip" in self._pixscale_mode.currentText() \
                      else self._pixscale_spin.value()
        config = {
            "fits_files":         self._fits_files,
            "pixel_scale_arcsec": pixel_scale,
            "weights":            self._get_weights_dict(),
        }
        self._progress.setRange(0, len(self._fits_files)); self._progress.setValue(0)
        self._set_running_state(True); self._set_ranked_state(False)
        self._frame_results = []
        self._tabs.setCurrentIndex(2)
        self._worker = RankingWorker(config, self._cancel_event)
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(
            lambda done, total, msg: (self._progress.setMaximum(total),
                                     self._progress.setValue(done),
                                     self._status.setText(f"  {msg}  [{done}/{total}]")))
        self._worker.all_scored.connect(self._on_all_scored)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._cancel_event.set(); self._worker.cancel()
            self._log("Cancelling…")

    def _on_all_scored(self, frame_results: list):
        self._frame_results = frame_results
        self._populate_table(frame_results)
        self._update_approval_label()
        scores = [f["score"] for f in frame_results]
        self._elbow_threshold = find_elbow_threshold(scores)
        self._score_canvas.show_scores(frame_results, self._threshold,
                                       elbow=self._elbow_threshold)
        self._update_stats(frame_results)

    def _on_finished(self, result: dict):
        self._set_running_state(False)
        if result.get("success"):
            self._set_ranked_state(True)
            n     = result["n_frames"]
            n_app = sum(1 for f in self._frame_results if f["score"] >= self._threshold)
            self._status.setText(
                f"✓  {n} frames ranked  |  "
                f"Threshold: {self._threshold}  |  "
                f"Approving {n_app}/{n}")
            self._log(f"── Done  |  Auto-threshold: {self._elbow_threshold:.1f} ──")
            self._tabs.setCurrentIndex(0)
            # Emit approved paths for pipeline handoff
            approved = self.get_approved_paths()
            self.ranking_complete.emit(approved)
            self.log_line.emit(
                f"[RANKING]  ✓ {n} ranked  |  {n_app} approved above threshold {self._threshold}")
            self.finished.emit({"success": True, "n_approved": n_app, "n_frames": n,
                                "approved_paths": approved,
                                "threshold": self._threshold})
        else:
            err = result.get("error", "Unknown")
            self._status.setText(f"✗  Failed: {err}")
            self.finished.emit(result)

    # ── Recompute ─────────────────────────────────────────────────────────────

    def _recompute_scores(self):
        if not self._frame_results: return
        weights    = self._get_weights_dict()
        all_metrics = [f.get("metrics", {}) for f in self._frame_results]
        population  = compute_population_stats(all_metrics)
        for frame, metrics in zip(self._frame_results, all_metrics):
            frame["score"] = score_frame(metrics, population, weights)
        self._frame_results.sort(key=lambda f: f["score"], reverse=True)
        for rank, frame in enumerate(self._frame_results, 1):
            frame["rank"] = rank
        self._populate_table(self._frame_results)
        scores = [f["score"] for f in self._frame_results]
        self._elbow_threshold = find_elbow_threshold(scores)
        self._score_canvas.show_scores(self._frame_results, self._threshold,
                                       elbow=self._elbow_threshold)
        self._update_approval_label()
        self._update_stats(self._frame_results)

    # ── Table ─────────────────────────────────────────────────────────────────

    def _populate_table(self, frame_results: list):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        self._table.setRowCount(len(frame_results))
        for row, fr in enumerate(frame_results):
            m     = fr.get("metrics", {})
            score = fr.get("score", 0)
            unit  = fr.get("fwhm_unit", "px")
            vals  = [
                str(fr.get("rank", row + 1)),
                f"{score:.1f}",
                "✓" if score >= self._threshold else "✗",
                fr.get("filename", ""),
                f"{m.get('fwhm', 0):.2f} {unit}",
                f"{m.get('eccentricity', 0):.3f}",
                f"{m.get('snr', 0):.1f}",
                f"{m.get('background', 0):.0f}",
                str(int(m.get("star_count", 0))),
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, fr["path"])
                self._table.setItem(row, col, item)
            self._colour_row(row, score)
        self._table.setSortingEnabled(True)

    def _colour_row(self, row: int, score: float):
        bg = QColor("#2a1e1e") if score < self._threshold else QColor(SIRIL_BG2)
        fg = QColor(SIRIL_TEXT_DIM) if score < self._threshold else QColor(SIRIL_TEXT)
        for col in range(self._table.columnCount()):
            item = self._table.item(row, col)
            if item: item.setBackground(bg); item.setForeground(fg)

    def _update_table_colors(self):
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 1)
            if item:
                try:
                    score = float(item.text())
                except ValueError:
                    score = 0.0
                self._colour_row(row, score)
                check_item = self._table.item(row, 2)
                if check_item:
                    check_item.setText("✓" if score >= self._threshold else "✗")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _update_stats(self, frame_results: list):
        if not frame_results: return
        mets  = [f["metrics"] for f in frame_results]
        fwhms = [m.get("fwhm", 0) for m in mets if m.get("fwhm", 0) > 0]
        eccs  = [m.get("eccentricity", 0) for m in mets]
        snrs  = [m.get("snr", 0) for m in mets if m.get("snr", 0) > 0]
        unit  = frame_results[0].get("fwhm_unit", "px")
        self._stat_labels["fwhm_med"].setText(f"{np.median(fwhms):.2f} {unit}" if fwhms else "—")
        self._stat_labels["ecc_med"].setText(f"{np.median(eccs):.3f}" if eccs else "—")
        self._stat_labels["snr_med"].setText(f"{np.median(snrs):.1f}" if snrs else "—")
        best  = frame_results[0]; worst = frame_results[-1]
        def _fmt(fr): return f"{fr['score']:.1f}  ({fr['filename'][:20]})"
        self._stat_labels["best"].setText(_fmt(best))
        self._stat_labels["worst"].setText(_fmt(worst))

    # ── Thumbnails ────────────────────────────────────────────────────────────

    def _on_row_double_clicked(self, row: int, _col: int):
        item = self._table.item(row, 0)
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path: return
        fr = next((f for f in self._frame_results if f["path"] == path), None)
        if not fr: return
        dlg = ThumbnailDialog(fr, parent=None)
        dlg.setStyleSheet(SIRIL_STYLESHEET)
        dlg.show()
        self._thumbnail_dialogs.append(dlg)
        self._thumbnail_dialogs = [d for d in self._thumbnail_dialogs if d.isVisible()]

    # ── Export ────────────────────────────────────────────────────────────────

    def _apply_and_export(self):
        if not self._frame_results:
            QMessageBox.warning(self, "No data", "No frames ranked yet."); return
        mode_idx   = self._output_mode_combo.currentIndex()
        mode_key   = OUTPUT_MODE_KEYS[mode_idx]
        output_dir = self.get_output_dir()
        self._log(f"Exporting — {OUTPUT_MODES[mode_idx]}  →  {output_dir}")
        result = apply_threshold_and_export(
            self._frame_results, float(self._threshold), mode_key, output_dir,
            export_csv=self._chk_csv.isChecked(), log_callback=self._log)
        n_app = result["n_approved"]; n_rej = result["n_rejected"]
        self._status.setText(f"✓  {n_app} frames exported  ({n_rej} rejected)  →  {output_dir}")
        self._log(f"Export complete: {n_app} approved, {n_rej} rejected.")
        # Also emit for pipeline
        self.ranking_complete.emit(self.get_approved_paths())

    def _export_csv_only(self):
        if not self._frame_results: return
        out_dir = self.get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "frame_ranking.csv")
        _write_ranking_csv(self._frame_results, float(self._threshold), csv_path)
        self._log(f"CSV: {csv_path}")
        self._status.setText(f"✓  CSV saved: {csv_path}")

    def _copy_approved_paths(self):
        if not self._frame_results: return
        approved = self.get_approved_paths()
        QApplication.clipboard().setText("\n".join(approved))
        self._log(f"Copied {len(approved)} approved paths to clipboard.")

    def _open_output_folder(self):
        import subprocess
        out_dir = self.get_output_dir()
        if os.path.isdir(out_dir):
            if sys.platform == "win32":
                subprocess.Popen(["explorer", os.path.normpath(out_dir)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", out_dir])
            else:
                subprocess.Popen(["xdg-open", out_dir])

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_view.appendPlainText(line)
        sb = self._log_view.verticalScrollBar(); sb.setValue(sb.maximum())
        self.log_line.emit(f"[RANKING]  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# STACKING PANEL  (Winsorized Sigma Stack UI)
# ─────────────────────────────────────────────────────────────────────────────

class StackingPanel(QWidget):
    """
    Winsorized Sigma-Clipped Stack UI.
    In pipeline mode: receives approved frames, copies them to a temp folder,
    imports as Siril sequence, then stacks.
    Exposes: set_approved_frames(paths), run_pipeline(cancel_event)
    Signals: finished(dict), log_line(str)
    """
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._siril          = None
        self._worker         = None
        self._thread         = None
        self._imp_worker     = None
        self._imp_thread     = None
        self._cancel         = False
        self._approved_paths = []
        self._pipeline_mode  = False
        self._build_ui()
        self._try_connect_siril()

    def _try_connect_siril(self):
        try:
            self._siril = s.SirilInterface()
            self._siril.connect()
            self._lbl_siril.setText("✓ Siril connected")
            self._lbl_siril.setStyleSheet(f"color:{SIRIL_SUCCESS}; font-size:9pt;")
        except Exception as e:
            self._siril = None
            self._lbl_siril.setText(f"⚠ Siril not connected: {e}")
            self._lbl_siril.setStyleSheet(f"color:{SIRIL_WARNING}; font-size:9pt;")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Siril connection status
        conn_row = QHBoxLayout()
        self._lbl_siril = QLabel("Connecting to Siril…")
        self._lbl_siril.setObjectName("dim")
        conn_row.addWidget(self._lbl_siril)
        btn_reconnect = QPushButton("↺ Reconnect")
        btn_reconnect.setFixedWidth(100); btn_reconnect.clicked.connect(self._try_connect_siril)
        conn_row.addWidget(btn_reconnect); conn_row.addStretch()
        root.addLayout(conn_row)

        # Pipeline handoff info
        self._grp_pipeline_info = QGroupBox("Pipeline Handoff (from Ranker)")
        pi = QVBoxLayout(self._grp_pipeline_info)
        self._lbl_handoff = QLabel("No frames received from Ranker yet.\n"
                                   "Run the pipeline or set a sequence manually below.")
        self._lbl_handoff.setObjectName("dim"); self._lbl_handoff.setWordWrap(True)
        pi.addWidget(self._lbl_handoff)
        self._grp_pipeline_info.setVisible(False)
        root.addWidget(self._grp_pipeline_info)

        # Import sequence
        imp_grp = QGroupBox("Import Sequence")
        imp_layout = QGridLayout(imp_grp); imp_layout.setColumnStretch(1, 1)
        imp_layout.addWidget(QLabel("Folder:"), 0, 0)
        self._txt_imp_folder = QLineEdit()
        self._txt_imp_folder.setPlaceholderText("Directory containing images…")
        imp_layout.addWidget(self._txt_imp_folder, 0, 1)
        btn_browse = QPushButton("…"); btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_import_folder)
        imp_layout.addWidget(btn_browse, 0, 2)
        imp_layout.addWidget(QLabel("Basename:"), 1, 0)
        self._txt_imp_basename = QLineEdit("lights")
        imp_layout.addWidget(self._txt_imp_basename, 1, 1, 1, 2)
        imp_layout.addWidget(QLabel("Format:"), 2, 0)
        self._cmb_imp_fmt = QComboBox()
        self._cmb_imp_fmt.addItems([
            "Individual images → FITS sequence",
            "Individual images → Single FITSEQ (-fitseq)",
            "Existing .seq file",
            "SER (.ser)",
        ])
        imp_layout.addWidget(self._cmb_imp_fmt, 2, 1, 1, 2)
        self._chk_imp_debayer = QCheckBox("Debayer (CFA / OSC cameras)")
        imp_layout.addWidget(self._chk_imp_debayer, 3, 0, 1, 2)
        self._btn_import = QPushButton("⇩  Convert & Load")
        self._btn_import.setObjectName("importButton")
        self._btn_import.clicked.connect(self._on_import)
        imp_layout.addWidget(self._btn_import, 3, 2)
        root.addWidget(imp_grp)

        # Sequence status
        seq_grp = QGroupBox("Sequence")
        seq_layout = QGridLayout(seq_grp); seq_layout.setColumnStretch(1, 1)
        seq_layout.addWidget(QLabel("Loaded:"), 0, 0)
        self._lbl_seq = QLabel("—"); self._lbl_seq.setObjectName("statusLabel")
        seq_layout.addWidget(self._lbl_seq, 0, 1)
        seq_layout.addWidget(QLabel("Frames:"), 1, 0)
        self._lbl_frames = QLabel("—"); self._lbl_frames.setObjectName("statusLabel")
        seq_layout.addWidget(self._lbl_frames, 1, 1)
        btn_refresh = QPushButton("↺  Refresh"); btn_refresh.setFixedWidth(90)
        btn_refresh.clicked.connect(self._detect_sequence)
        seq_layout.addWidget(btn_refresh, 0, 2, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(seq_grp)

        # Rejection parameters
        rej_grp = QGroupBox("Rejection Algorithm")
        rej_layout = QGridLayout(rej_grp); rej_layout.setColumnStretch(1, 1)
        algo_lbl = QLabel("Vectorized Winsorized Sigma Clip")
        algo_lbl.setStyleSheet("color: #6ec96e; font-style: italic;")
        rej_layout.addWidget(QLabel("Algorithm:"), 0, 0); rej_layout.addWidget(algo_lbl, 0, 1, 1, 3)
        rej_layout.addWidget(QLabel("σ Low:"), 1, 0)
        self._spn_sigma_low = QDoubleSpinBox()
        self._spn_sigma_low.setRange(0.1, 10.0); self._spn_sigma_low.setValue(3.0)
        self._spn_sigma_low.setSingleStep(0.1); self._spn_sigma_low.setSuffix("  σ")
        rej_layout.addWidget(self._spn_sigma_low, 1, 1)
        rej_layout.addWidget(QLabel("σ High:"), 1, 2)
        self._spn_sigma_high = QDoubleSpinBox()
        self._spn_sigma_high.setRange(0.1, 10.0); self._spn_sigma_high.setValue(3.0)
        self._spn_sigma_high.setSingleStep(0.1); self._spn_sigma_high.setSuffix("  σ")
        rej_layout.addWidget(self._spn_sigma_high, 1, 3)
        rej_layout.addWidget(QLabel("Iterations:"), 2, 0)
        self._spn_iterations = QSpinBox()
        self._spn_iterations.setRange(1, 20); self._spn_iterations.setValue(5)
        rej_layout.addWidget(self._spn_iterations, 2, 1)
        self._chk_save_rej = QCheckBox("Save rejection map")
        rej_layout.addWidget(self._chk_save_rej, 2, 2, 1, 2)
        root.addWidget(rej_grp)

        # Stacking method
        stack_grp = QGroupBox("Stacking Method")
        stack_layout = QGridLayout(stack_grp); stack_layout.setColumnStretch(1, 1)
        stack_layout.addWidget(QLabel("Method:"), 0, 0)
        self._cmb_method = QComboBox(); self._cmb_method.addItems(["Mean", "Median", "Sum"])
        stack_layout.addWidget(self._cmb_method, 0, 1, 1, 3)
        stack_layout.addWidget(QLabel("Normalization:"), 1, 0)
        self._cmb_norm = QComboBox()
        self._cmb_norm.addItems(["None", "Additive", "Multiplicative", "Additive + Scaling"])
        stack_layout.addWidget(self._cmb_norm, 1, 1, 1, 3)
        self._chk_prenorm = QCheckBox("Pre-normalize sequence (seqnorm)")
        stack_layout.addWidget(self._chk_prenorm, 2, 0, 1, 4)
        self._chk_output_norm = QCheckBox("Output normalization (scale result to [0,1])")
        stack_layout.addWidget(self._chk_output_norm, 3, 0, 1, 4)
        root.addWidget(stack_grp)

        # Frame weighting
        wt_grp = QGroupBox("Frame Weighting")
        wt_layout = QGridLayout(wt_grp); wt_layout.setColumnStretch(1, 1)
        self._chk_weights = QCheckBox("Enable frame weighting")
        self._chk_weights.toggled.connect(
            lambda e: (self._cmb_weight_metric.setEnabled(e),
                       self._spn_weight_power.setEnabled(e)))
        wt_layout.addWidget(self._chk_weights, 0, 0, 1, 4)
        wt_layout.addWidget(QLabel("Metric:"), 1, 0)
        self._cmb_weight_metric = QComboBox()
        self._cmb_weight_metric.addItems([
            "wFWHM  (FWHM × star count — recommended)",
            "FWHM   (raw FWHM only)", "Noise  (background RMS)", "# Stars",
        ])
        self._cmb_weight_metric.setEnabled(False)
        wt_layout.addWidget(self._cmb_weight_metric, 1, 1, 1, 3)
        wt_layout.addWidget(QLabel("Power:"), 2, 0)
        self._spn_weight_power = QDoubleSpinBox()
        self._spn_weight_power.setRange(0.1, 10.0); self._spn_weight_power.setValue(2.0)
        self._spn_weight_power.setDecimals(1); self._spn_weight_power.setEnabled(False)
        wt_layout.addWidget(self._spn_weight_power, 2, 1)
        wt_layout.addWidget(QLabel("(1=linear · 2=quadratic · 0.5=gentle)"), 2, 2, 1, 2)
        root.addWidget(wt_grp)

        # Output
        out_grp = QGroupBox("Output")
        out_layout = QGridLayout(out_grp); out_layout.setColumnStretch(1, 1)
        out_layout.addWidget(QLabel("Filename:"), 0, 0)
        self._txt_output = QLineEdit("stacked_result")
        self._txt_output.setPlaceholderText("Output filename (no extension)")
        out_layout.addWidget(self._txt_output, 0, 1)
        out_layout.addWidget(QLabel(".fit"), 0, 2)
        root.addWidget(out_grp)

        # Log
        log_grp = QGroupBox("Log")
        log_layout = QVBoxLayout(log_grp)
        self._log_view = QTextEdit(); self._log_view.setReadOnly(True)
        self._log_view.setFixedHeight(120)
        log_layout.addWidget(self._log_view)
        root.addWidget(log_grp)

        # Progress + buttons
        self._progress_bar = QProgressBar(); self._progress_bar.setValue(0)
        root.addWidget(self._progress_bar)

        btn_layout = QHBoxLayout()
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("cancelButton"); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._btn_cancel); btn_layout.addStretch()
        self._btn_run = QPushButton("▶  Run Stack")
        self._btn_run.setObjectName("runButton"); self._btn_run.setMinimumWidth(130)
        self._btn_run.clicked.connect(self._on_run)
        btn_layout.addWidget(self._btn_run)
        root.addLayout(btn_layout)
        self._log_msg("Winsorized Sigma-Clipped Stack engine ready.", "ok")

    # ── Public interface ──────────────────────────────────────────────────────

    def set_approved_frames(self, approved_paths: list):
        """
        Receive approved frames from Ranker (pipeline handoff, Section A4).
        Creates a temp folder with the approved files and pre-fills the import folder.
        """
        if not approved_paths:
            return
        self._approved_paths = approved_paths

        # Use the parent folder of the first approved file + "_approved_temp"
        base_dir    = os.path.dirname(approved_paths[0])
        temp_folder = os.path.join(base_dir, "_stacking_suite_approved")
        os.makedirs(temp_folder, exist_ok=True)

        # Remove stale files from previous runs
        for old in glob.glob(os.path.join(temp_folder, "*.fit")) + \
                   glob.glob(os.path.join(temp_folder, "*.fits")):
            try:
                os.remove(old)
            except Exception:
                pass

        # Symlink or copy approved frames
        n_linked = 0
        for src in approved_paths:
            dst = os.path.join(temp_folder, os.path.basename(src))
            try:
                if sys.platform == "win32" or not hasattr(os, "symlink"):
                    shutil.copy2(src, dst)
                else:
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.symlink(src, dst)
                n_linked += 1
            except Exception as e:
                self._log_msg(f"  Warning: could not link {os.path.basename(src)}: {e}", "warn")

        self._txt_imp_folder.setText(temp_folder)
        self._txt_imp_basename.setText("approved")

        self._grp_pipeline_info.setVisible(True)
        self._lbl_handoff.setText(
            f"✓ Received {len(approved_paths)} approved frames from Ranker\n"
            f"  Staged in: {temp_folder}\n"
            f"  Ready to import ({n_linked} files linked)")
        self._lbl_handoff.setStyleSheet(f"color:{SIRIL_SUCCESS};")

        self._log_msg(
            f"Pipeline handoff: {len(approved_paths)} approved frames "
            f"staged → {temp_folder}", "ok")

    def get_config(self) -> dict:
        return {
            "method":         self._cmb_method.currentText(),
            "sigma_low":      self._spn_sigma_low.value(),
            "sigma_high":     self._spn_sigma_high.value(),
            "iterations":     self._spn_iterations.value(),
            "norm":           self._cmb_norm.currentText(),
            "prenormalize":   self._chk_prenorm.isChecked(),
            "output_norm":    self._chk_output_norm.isChecked(),
            "use_weights":    self._chk_weights.isChecked(),
            "weight_metric":  self._cmb_weight_metric.currentText().split()[0],
            "weight_power":   self._spn_weight_power.value(),
            "save_rejection": self._chk_save_rej.isChecked(),
            "output_name":    self._txt_output.text().strip() or "stacked_result",
        }

    def cancel(self):
        self._cancel = True
        if self._worker: self._worker.cancel()

    # ── Import helpers ────────────────────────────────────────────────────────

    def _browse_import_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select image folder", os.path.expanduser("~"),
            QFileDialog.Option.ShowDirsOnly)
        if folder:
            self._txt_imp_folder.setText(folder)
            if not self._txt_imp_basename.text().strip():
                self._txt_imp_basename.setText(os.path.basename(folder).lower() or "lights")

    def _on_import(self):
        if not self._siril:
            QMessageBox.warning(self, "Not connected", "Siril is not connected. Click Reconnect.")
            return
        folder   = self._txt_imp_folder.text().strip()
        basename = self._txt_imp_basename.text().strip()
        fmt      = self._cmb_imp_fmt.currentText()
        debayer  = self._chk_imp_debayer.isChecked()
        if not folder or not os.path.isdir(folder):
            self._log_msg("Please select an image folder first.", "warn"); return
        if not basename:
            self._log_msg("Please enter a sequence basename.", "warn"); return
        self._btn_import.setEnabled(False)
        self._log_msg(f"Starting import from {folder}…", "info")
        self._imp_thread = QThread()
        self._imp_worker = ImportWorker(self._siril, folder, basename, fmt, debayer)
        self._imp_worker.moveToThread(self._imp_thread)
        self._imp_thread.started.connect(self._imp_worker.run)
        self._imp_worker.log_message.connect(self._log_msg)
        self._imp_worker.finished.connect(self._on_import_finished)
        self._imp_worker.finished.connect(self._imp_thread.quit)
        self._imp_thread.start()

    def _on_import_finished(self, success: bool, seq_basename: str):
        self._btn_import.setEnabled(True)
        if success:
            self._log_msg(f"Import complete — '{seq_basename}' ready.", "ok")
            self._detect_sequence()
        else:
            self._log_msg("Import failed.", "error")

    def _detect_sequence(self):
        if not self._siril: return
        try:
            seq_name = self._siril.get_seq_name()
            if seq_name:
                self._lbl_seq.setText(os.path.basename(seq_name))
                try:
                    info     = self._siril.get_sequence_info()
                    total    = getattr(info, "number", "?")
                    included = getattr(info, "selnum", "?")
                    self._lbl_frames.setText(f"{included} selected / {total} total")
                except Exception:
                    self._lbl_frames.setText("(info unavailable)")
                self._log_msg(f"Sequence: {os.path.basename(seq_name)}", "ok")
            else:
                self._lbl_seq.setText("No sequence loaded")
                self._lbl_frames.setText("—")
        except Exception:
            self._lbl_seq.setText("No sequence loaded")
            self._lbl_frames.setText("—")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _on_run(self):
        if not self._siril:
            QMessageBox.warning(self, "Not connected",
                "Siril is not connected. Click Reconnect."); return
        self._cancel = False
        self._progress_bar.setValue(0)
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._thread = QThread()
        self._worker = StackWorker(self._siril, self.get_config())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log_message.connect(self._log_msg)
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def run_pipeline(self, cancel_event: threading.Event):
        """Start stacking in pipeline mode."""
        self._on_run()

    def _on_cancel(self):
        if self._worker: self._worker.cancel()
        self._btn_cancel.setEnabled(False)
        self._log_msg("Cancellation requested…", "warn")

    def _on_finished(self, success: bool, message: str):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        if success:
            self._log_msg(f"Done: {message}", "ok")
            self._progress_bar.setFormat("Complete ✓")
            self.log_line.emit(f"[STACK]  ✓ {message}")
            self.finished.emit({"success": True, "output_name": message})
        else:
            self._log_msg(f"Failed: {message}", "error")
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("Failed ✗")
            self.finished.emit({"success": False, "error": message})

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log_msg(self, msg: str, level: str = "info"):
        color_map = {"info": "#b0b0b0", "ok": "#6ec96e", "warn": "#e8c46a", "error": "#e06c75"}
        color  = color_map.get(level, "#b0b0b0")
        prefix = {"ok": "✓ ", "warn": "⚠ ", "error": "✗ ", "info": "  "}.get(level, "  ")
        self._log_view.append(f'<span style="color:{color}">{prefix}{msg}</span>')
        self._log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_line.emit(f"[STACK]  {msg}")
        try:
            if self._siril: self._siril.log(msg)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW TAB  (pipeline orchestration)
# ─────────────────────────────────────────────────────────────────────────────

class OverviewTab(QWidget):
    run_pipeline_requested  = pyqtSignal()
    run_step_requested      = pyqtSignal(str)
    cancel_requested        = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(10)

        # Pipeline diagram
        diag_grp = QGroupBox("Pipeline")
        dl = QVBoxLayout(diag_grp)
        self._pipeline_widget = PipelineStatusWidget([
            {"name": "Rank Frames\n(5 quality metrics)", "key": "rank"},
            {"name": "Stack\n(Winsorized σ-clip)",       "key": "stack"},
        ])
        self._pipeline_widget.setMinimumHeight(90)
        dl.addWidget(self._pipeline_widget)
        step_row = QHBoxLayout()
        btn_rank  = QPushButton("▶ Rank only")
        btn_rank.clicked.connect(lambda: self.run_step_requested.emit("rank"))
        btn_stack = QPushButton("▶ Stack only")
        btn_stack.clicked.connect(lambda: self.run_step_requested.emit("stack"))
        step_row.addWidget(btn_rank); step_row.addWidget(btn_stack); step_row.addStretch()
        dl.addLayout(step_row)
        root.addWidget(diag_grp)

        # Handoff summary (shown after ranking)
        self._handoff_grp = QGroupBox("Ranking → Stacking handoff")
        self._handoff_grp.setVisible(False)
        hl = QVBoxLayout(self._handoff_grp)
        self._lbl_handoff_summary = QLabel(""); self._lbl_handoff_summary.setWordWrap(True)
        hl.addWidget(self._lbl_handoff_summary)
        root.addWidget(self._handoff_grp)

        # Result summary
        self._result_grp = QGroupBox("Last result"); self._result_grp.setVisible(False)
        rl = QVBoxLayout(self._result_grp)
        self._result_lbl = QLabel(""); self._result_lbl.setWordWrap(True)
        rl.addWidget(self._result_lbl)
        root.addWidget(self._result_grp)

        root.addStretch()

        # Action row
        action_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run pipeline  (Rank → Stack)")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(36); self._btn_run.setMinimumWidth(200)
        self._btn_run.clicked.connect(self.run_pipeline_requested)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(36); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self.cancel_requested)
        action_row.addWidget(self._btn_run); action_row.addWidget(self._btn_cancel)
        action_row.addStretch()
        root.addLayout(action_row)

    def set_step_status(self, key: str, status: str):
        self._pipeline_widget.set_status(key, status)

    def set_running(self, running: bool):
        self._btn_run.setEnabled(not running)
        self._btn_cancel.setEnabled(running)

    def show_handoff(self, n_approved: int, n_total: int, threshold: float):
        self._handoff_grp.setVisible(True)
        pct = 100 * n_approved / max(n_total, 1)
        self._lbl_handoff_summary.setText(
            f"✓ {n_approved} / {n_total} frames approved ({pct:.0f}%)  "
            f"|  Threshold: {threshold:.0f}\n"
            f"→ These frames will be staged and imported for stacking")
        self._lbl_handoff_summary.setStyleSheet(f"color:{SIRIL_SUCCESS};")

    def show_result(self, step: str, result: dict):
        self._result_grp.setVisible(True)
        if step == "stack":
            self._result_lbl.setText(
                f"✓ Stack complete\n{result.get('output_name','')}")
            self._result_grp.setTitle("Last result — ⚡ Stack")
        elif step == "rank":
            n   = result.get("n_approved", 0)
            tot = result.get("n_frames", 0)
            self._result_lbl.setText(
                f"✓ Ranking complete — {n}/{tot} frames approved")
            self._result_grp.setTitle("Last result — 🏆 Rank")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stacking Suite  —  Siril")
        self.resize(1400, 860)

        self._cancel_event     = threading.Event()
        self._pipeline_data    = {}

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────────────
        header = QWidget(); header.setFixedHeight(52)
        header.setStyleSheet(f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon  = QLabel("⚡"); lbl_icon.setStyleSheet("font-size:18pt; background:transparent;")
        lbl_title = QLabel("Stacking Suite")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold; background:transparent;")
        lbl_ver   = QLabel(
            f"v1.0  |  2 scripts  |  Subframe Ranking → Winsorized σ-clip Stack  "
            f"|  Stack v{STACK_VERSION}")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title); hl.addStretch(); hl.addWidget(lbl_ver)
        root.addWidget(header)

        # ── Tab widget ──────────────────────────────────────────────────────
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

        # Tab 1 — Ranking
        self._ranking_panel = RankingPanel()
        self._ranking_panel.ranking_complete.connect(self._on_ranking_complete)
        self._ranking_panel.finished.connect(
            lambda r: self._on_step_done("rank", r))
        self._ranking_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._ranking_panel, "🏆  Subframe Ranking")

        # Tab 2 — Stacking
        self._stacking_panel = StackingPanel()
        self._stacking_panel.finished.connect(
            lambda r: self._on_step_done("stack", r))
        self._stacking_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._stacking_panel, "⚡  Winsorized Stack")

        # Tab 3 — Pipeline Log
        log_wrap = QWidget(); lw = QVBoxLayout(log_wrap)
        lw.setContentsMargins(6, 6, 6, 6)
        lbl_log = QLabel("Combined log"); lbl_log.setObjectName("dim")
        lw.addWidget(lbl_log)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._pipeline_log.clear)
        lw.addWidget(btn_clr); lw.addWidget(self._pipeline_log)
        self._tabs.addTab(log_wrap, "📋  Pipeline Log")

        # ── Bottom bar ──────────────────────────────────────────────────────
        bottom = QWidget(); bottom.setFixedHeight(46)
        bottom.setStyleSheet(f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 4, 10, 4); bl.setSpacing(8)
        self._global_progress = QProgressBar()
        self._global_progress.setFixedHeight(8); self._global_progress.setRange(0, 2)
        bl.addWidget(self._global_progress, 1)
        self._btn_run = QPushButton("▶  Run pipeline")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(32); self._btn_run.setMinimumWidth(140)
        self._btn_run.clicked.connect(self._run_pipeline); bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(32); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_pipeline); bl.addWidget(self._btn_cancel)
        self._status_lbl = QLabel("Ready — scan a FITS folder in the Ranking tab.")
        self._status_lbl.setObjectName("dim")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status_lbl, 1)
        root.addWidget(bottom)

        # Startup log
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] Stacking Suite — ready")
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "Pipeline: Subframe Ranking → Winsorized Sigma-Clip Stack")

    # ── Pipeline orchestration (Section A6) ───────────────────────────────

    def _run_pipeline(self):
        if not self._ranking_panel._fits_files:
            QMessageBox.warning(self, "Missing input",
                "Scan a FITS folder in the Subframe Ranking tab first.")
            self._tabs.setCurrentIndex(1); return

        self._cancel_event.clear()
        self._overview.set_step_status("rank",  "idle")
        self._overview.set_step_status("stack", "idle")
        self._set_running(True)
        self._global_progress.setValue(0)
        self._run_rank_step()

    def _run_rank_step(self):
        self._overview.set_step_status("rank", "running")
        self._set_status("Ranking frames…", SIRIL_ACCENT)
        self._log_pipeline("═══ STEP 1: Subframe Ranking ═══")
        self._tabs.setCurrentIndex(1)
        self._ranking_panel._start_ranking()

    def _on_ranking_complete(self, approved_paths: list):
        """Intercept ranking results and pipe to stacker (Section A4)."""
        if not approved_paths:
            return
        n_total    = len(self._ranking_panel._frame_results)
        n_approved = len(approved_paths)
        threshold  = self._ranking_panel.get_threshold()

        self._overview.show_handoff(n_approved, n_total, threshold)
        self._log_pipeline(
            f"In-memory handoff: {n_approved}/{n_total} approved frames "
            f"→ Stacking Panel (no CSV)")
        self._stacking_panel.set_approved_frames(approved_paths)

    def _run_stack_step(self, approved_paths: list):
        self._overview.set_step_status("stack", "running")
        self._global_progress.setValue(1)
        self._set_status("Stacking…", SIRIL_ACCENT)
        self._log_pipeline("═══ STEP 2: Winsorized Sigma-Clip Stack ═══")
        self._tabs.setCurrentIndex(2)
        self._stacking_panel.run_pipeline(self._cancel_event)

    def _on_step_done(self, step: str, result: dict):
        if result.get("success"):
            self._overview.set_step_status(step, "done")
            self._overview.show_result(step, result)

            if step == "rank":
                n   = result.get("n_approved", 0)
                tot = result.get("n_frames", 0)
                self._log_pipeline(f"✓ Ranking done — {n}/{tot} approved")
                approved = result.get("approved_paths", [])
                if approved:
                    self._run_stack_step(approved)
                else:
                    self._set_status("No frames approved — adjust threshold.", SIRIL_WARNING)
                    self._set_running(False)

            elif step == "stack":
                self._global_progress.setValue(2)
                out = result.get("output_name", "—")
                self._set_status(f"✓ Pipeline complete → {out}", SIRIL_SUCCESS)
                self._log_pipeline(f"✓ Pipeline complete — {out}")
                self._set_running(False)
        else:
            err = result.get("error", "Unknown")
            self._overview.set_step_status(step, "error")
            self._set_status(f"✗ {step.title()} failed: {err}", SIRIL_ERROR)
            self._log_pipeline(f"✗ {step.upper()} FAILED: {err}")
            self._set_running(False)

    # ── Standalone step runners (A8) ──────────────────────────────────────

    def _run_step(self, step: str):
        self._cancel_event.clear()
        self._set_running(True)
        if step == "rank":
            if not self._ranking_panel._fits_files:
                QMessageBox.warning(self, "No files", "Scan a FITS folder first.")
                self._tabs.setCurrentIndex(1); self._set_running(False); return
            self._overview.set_step_status("rank", "running")
            self._log_pipeline("─── Standalone: Rank ───")
            self._tabs.setCurrentIndex(1)
            self._ranking_panel._start_ranking()
        elif step == "stack":
            self._overview.set_step_status("stack", "running")
            self._log_pipeline("─── Standalone: Stack ───")
            self._tabs.setCurrentIndex(2)
            self._stacking_panel.run_pipeline(self._cancel_event)

    # ── Cancel (A9) ───────────────────────────────────────────────────────

    def _cancel_pipeline(self):
        self._cancel_event.set()
        self._ranking_panel._cancel()
        self._stacking_panel.cancel()
        for key in ("rank", "stack"):
            if self._overview._pipeline_widget.statuses.get(key) == "running":
                self._overview.set_step_status(key, "error")
        self._set_status("Cancelled.", SIRIL_WARNING)
        self._log_pipeline("CANCELLED")
        self._set_running(False)

    # ── UI helpers ─────────────────────────────────────────────────────────

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
