r"""
session_qa_suite.py  —  Script Merger: Session / QA Suite
==========================================================
Merged launcher for:
    📁  Multi-Session Manager    (multi_session_manager.py)
    🔍  A/B Comparator           (ab_compare.py)
    👁  Seeing Estimator         (seeing_estimator.py)

Architecture: Tool Selector (B5 pattern — no sequential dependency)
    - Each tool is a self-contained panel in its own tab
    - Shared session header: project root auto-fills FITS folder in each tool
    - Cross-tool handoffs:
        * Session Manager scan → Seeing Estimator pre-fills FITS folder
        * Seeing Estimator results → A/B Comparator (best/worst frame pre-fill)
    - All three tools independently runnable

Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:  Siril → Scripts → session_qa_suite
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
import datetime
from datetime import datetime as dt

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

from multi_session_manager import (
    OrganizeWorker,
    DispatchWorker,
    ProjectTab,
    OrganizeTab,
    DispatchTab,
    # Algorithms
    find_fits,
    scan_project,
    build_manifest,
    save_manifest,
    load_manifest,
    manifest_path,
    manifest_summary,
    _cal_source,
    # Helpers
    ci, dim, sep, ts,
    # Constants
    SKIP_FOLDERS, FITS_EXTS, MANIFEST_NAME,
    STACKER_OPTIONS, VERSION as MSM_VERSION,
    HAS_EM,
    # Theme
    SIRIL_STYLESHEET,
    SIRIL_BG, SIRIL_BG2, SIRIL_BG3,
    SIRIL_ACCENT, SIRIL_ACCENT2,
    SIRIL_TEXT, SIRIL_TEXT_DIM,
    SIRIL_BORDER, SIRIL_SUCCESS,
    SIRIL_WARNING, SIRIL_SECTION,
    SIRIL_ERROR,
)

from ab_compare import (
    CompareCanvas,
    StatsWidget,
    StretchWidget,
    # Algorithms
   
    auto_stretch,
    apply_stretch,
    compute_difference,
    downsample,
    compute_image_stats,
    compute_comparison_stats,
)

from seeing_estimator import (
    SeeingWorker,
    SeeingHistogram,
    BadgeWidget,
    # Algorithms
    detect_stars,
    fit_star_fwhm,
    fwhm_arcsec_to_r0_cm,
    r0_to_quality_label,
    analyze_frame,
    
    WAVELENGTHS,
    HAS_EQUIPMENT_MANAGER as SEEING_HAS_EM,
)

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QFrame, QSizePolicy, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton,
    QButtonGroup, QTreeWidget, QTreeWidgetItem, QTextBrowser,
    
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QFont, QShortcut, QKeySequence


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SESSION MANAGER PANEL  (wraps the three child tabs from the script)
# ─────────────────────────────────────────────────────────────────────────────

class SessionManagerPanel(QWidget):
    """
    Full Multi-Session Manager UI.
    Wraps ProjectTab, OrganizeTab, DispatchTab from child script.
    Exposes: get_project_root(), get_fits_files()
    Signals: project_scanned(str), log_line(str)
    """
    project_scanned = pyqtSignal(str)  # project root path
    log_line        = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_event  = threading.Event()
        self._org_worker    = None
        self._disp_worker   = None
        self._structure     = {}
        self._manifest      = {}
        self._manifest_file = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        inner_tabs = QTabWidget()
        root.addWidget(inner_tabs, 1)

        # ── Tab: Project (scan) ──────────────────────────────────────────────
        self._project_tab = ProjectTab()
        self._project_tab.scanned.connect(self._on_project_scanned)
        inner_tabs.addTab(self._project_tab, "📁  Project")

        # ── Tab: Organize ────────────────────────────────────────────────────
        self._organize_tab = OrganizeTab()
        self._organize_tab.b_build.clicked.connect(self._build_manifest_preview)
        self._organize_tab.b_organize.clicked.connect(self._do_organize)
        inner_tabs.addTab(self._organize_tab, "📂  Organize")

        # ── Tab: Dispatch ────────────────────────────────────────────────────
        self._dispatch_tab = DispatchTab()
        self._dispatch_tab.b_dispatch.clicked.connect(self._do_dispatch)
        self._dispatch_tab.b_resume.clicked.connect(self._do_dispatch)
        inner_tabs.addTab(self._dispatch_tab, "▶  Dispatch")

        # ── Tab: Help ────────────────────────────────────────────────────────
        help_tab = QWidget(); hl = QVBoxLayout(help_tab)
        from multi_session_manager import HELP_HTML
        browser = QTextBrowser(); browser.setHtml(HELP_HTML)
        hl.addWidget(browser)
        inner_tabs.addTab(help_tab, "❓  Help")

        # ── Log ──────────────────────────────────────────────────────────────
        log_tab = QWidget(); ll = QVBoxLayout(log_tab)
        self._log_view = QPlainTextEdit(); self._log_view.setReadOnly(True)
        ll.addWidget(self._log_view)
        inner_tabs.addTab(log_tab, "📋  Log")

        # Bottom: cancel + status
        bot = QHBoxLayout()
        bot.setContentsMargins(8, 4, 8, 4)
        self._cancel_btn = QPushButton("✕  Cancel current operation")
        self._cancel_btn.setObjectName("danger")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        bot.addWidget(self._cancel_btn)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(8)
        self._progress.setVisible(False)
        bot.addWidget(self._progress, 1)
        self._status_lbl = QLabel("Ready — scan a project folder to begin")
        self._status_lbl.setObjectName("dim")
        bot.addWidget(self._status_lbl)
        root.addLayout(bot)

    # ── Public interface ──────────────────────────────────────────────────────

    def get_project_root(self) -> str:
        return self._project_tab.get_root()

    def get_fits_files(self) -> list:
        """Return all lights FITS files from current structure for cross-panel use."""
        files = []
        for flt in self._structure.get("filters", []):
            for night in self._structure.get("nights", {}).get(flt, []):
                batch = self._structure["batches"][flt][night]
                files.extend(batch.get("lights", []))
        return sorted(set(files))

    def get_structure(self) -> dict:
        return self._structure

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _on_project_scanned(self, structure: dict):
        self._structure = structure
        self._organize_tab.set_structure(structure)
        root = structure.get("project_root", "")
        n_filters = len(structure.get("filters", []))
        n_nights  = sum(len(v) for v in structure.get("nights", {}).values())
        n_frames  = sum(
            len(b["lights"])
            for f in structure.get("batches", {}).values()
            for b in f.values()
        )
        self._log(
            f"Scanned: {n_filters} filter(s), {n_nights} night(s), {n_frames} frames")
        if root:
            self.project_scanned.emit(root)
        self._set_status(
            f"✓ {n_filters} filter(s)  {n_nights} night(s)  {n_frames} frames",
            SIRIL_SUCCESS)

    def _build_manifest_preview(self):
        if not self._structure.get("filters"):
            QMessageBox.warning(self, "No project", "Scan a project first."); return
        settings = self._organize_tab.get_settings()
        self._manifest = build_manifest(self._structure, settings)
        self._organize_tab._manifest = self._manifest
        self._organize_tab._build_manifest_preview()

    def _do_organize(self):
        if not self._manifest:
            QMessageBox.warning(self, "No manifest",
                "Build a manifest first (📋 Build manifest button)."); return
        root = self._project_root_safe()
        mf_path = manifest_path(root)
        save_manifest(self._manifest, mf_path)
        self._manifest_file = mf_path
        self._dispatch_tab.set_manifest(self._manifest, mf_path)
        self._log(f"Manifest saved: {mf_path}")

        self._cancel_event.clear()
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._set_status("Organizing files…", SIRIL_ACCENT)

        self._org_worker = OrganizeWorker(self._manifest, self._cancel_event)
        self._org_worker.log_line.connect(self._log)
        self._org_worker.progress.connect(lambda c, t, d: (
            self._progress.setRange(0, max(t,1)), self._progress.setValue(c),
            self._set_status(d, SIRIL_ACCENT)))
        self._org_worker.finished.connect(self._on_org_done)
        self._org_worker.start()

    def _on_org_done(self, result: dict):
        self._cancel_btn.setEnabled(False); self._progress.setVisible(False)
        if result.get("success"):
            self._set_status("✓ Organization complete.", SIRIL_SUCCESS)
            self._log("Organization complete.")
        else:
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)

    def _do_dispatch(self):
        manifest = self._dispatch_tab.get_manifest()
        mf_file  = self._dispatch_tab.get_manifest_file()
        if not manifest:
            QMessageBox.warning(self, "No manifest",
                "Load or build a manifest first."); return
        stacker_key = self._dispatch_tab.get_stacker_key()
        manifest["stacker"] = stacker_key
        save_manifest(manifest, mf_file)

        self._cancel_event.clear()
        self._dispatch_tab.set_dispatching(True)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._set_status("Dispatching to stacker…", SIRIL_ACCENT)

        self._disp_worker = DispatchWorker(manifest, mf_file, self._cancel_event)
        self._disp_worker.log_line.connect(self._log)
        self._disp_worker.progress.connect(
            lambda c, t, d: (
                self._dispatch_tab.update_progress(c, t, d),
                self._progress.setRange(0, max(t,1)),
                self._progress.setValue(c),
                self._set_status(d, SIRIL_ACCENT)))
        self._disp_worker.finished.connect(self._on_disp_done)
        self._disp_worker.start()

    def _on_disp_done(self, result: dict):
        self._dispatch_tab.set_dispatching(False)
        self._cancel_btn.setEnabled(False); self._progress.setVisible(False)
        self._dispatch_tab.refresh_table()
        if result.get("success"):
            self._set_status("✓ Dispatch complete.", SIRIL_SUCCESS)
        else:
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)

    def _cancel(self):
        self._cancel_event.set()
        self._cancel_btn.setEnabled(False)
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _project_root_safe(self) -> str:
        return self._structure.get("project_root", self._project_tab.get_root())

    def _log(self, msg: str):
        t = dt.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{t}] {msg}")
        self.log_line.emit(f"[MSM]  {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")


# ─────────────────────────────────────────────────────────────────────────────
# A/B COMPARATOR PANEL
# ─────────────────────────────────────────────────────────────────────────────

class ABComparePanel(QWidget):
    """
    Full A/B Comparator UI.
    Exposes: set_file_a(path), set_file_b(path)
    Signals: log_line(str)
    """
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path_a  = ""
        self._path_b  = ""
        self._data_a  = None
        self._data_b  = None
        self._mode    = "side_by_side"
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_tick)
        self._blink_state = "A"
        self._build_ui()
        self._setup_shortcuts()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── File load bar ─────────────────────────────────────────────────────
        load_grp = QGroupBox("Images")
        lg = QHBoxLayout(load_grp); lg.setSpacing(6)

        # A
        lg.addWidget(QLabel("A:"))
        self._edit_a = QLineEdit(); self._edit_a.setPlaceholderText("Image A path…")
        lg.addWidget(self._edit_a, 1)
        btn_browse_a = QPushButton("Browse…"); btn_browse_a.setFixedWidth(70)
        btn_browse_a.clicked.connect(lambda: self._browse("A"))
        lg.addWidget(btn_browse_a)

        lg.addWidget(QFrame())

        # B
        lg.addWidget(QLabel("B:"))
        self._edit_b = QLineEdit(); self._edit_b.setPlaceholderText("Image B path…")
        lg.addWidget(self._edit_b, 1)
        btn_browse_b = QPushButton("Browse…"); btn_browse_b.setFixedWidth(70)
        btn_browse_b.clicked.connect(lambda: self._browse("B"))
        lg.addWidget(btn_browse_b)

        btn_load = QPushButton("⚡  Load & Compare")
        btn_load.setObjectName("primary"); btn_load.setMinimumWidth(140)
        btn_load.clicked.connect(self._load_and_compare)
        lg.addWidget(btn_load)
        root.addWidget(load_grp)

        # ── Mode controls ─────────────────────────────────────────────────────
        mode_grp = QGroupBox("Display mode")
        mg = QHBoxLayout(mode_grp); mg.setSpacing(10)

        self._mode_btns = {}
        for label, mode, shortcut in [
            ("Side-by-Side [S]", "side_by_side", "S"),
            ("Wipe [W]",         "wipe",          "W"),
            ("Blink [B]",        "blink",         "B"),
            ("Difference [D]",   "difference",    "D"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setObjectName("mode_btn")
            btn.clicked.connect(lambda checked, m=mode: self._set_mode(m))
            mg.addWidget(btn)
            self._mode_btns[mode] = btn
        self._mode_btns["side_by_side"].setChecked(True)

        mg.addWidget(QFrame())

        # Zoom controls
        mg.addWidget(QLabel("Zoom:"))
        btn_zoom_in  = QPushButton("＋"); btn_zoom_in.setFixedWidth(30)
        btn_zoom_out = QPushButton("－"); btn_zoom_out.setFixedWidth(30)
        btn_reset    = QPushButton("↺"); btn_reset.setFixedWidth(30)
        btn_zoom_in.clicked.connect(lambda: self._canvas.zoom_by(0.8))
        btn_zoom_out.clicked.connect(lambda: self._canvas.zoom_by(1.25))
        btn_reset.clicked.connect(lambda: self._canvas.reset_zoom())
        mg.addWidget(btn_zoom_in); mg.addWidget(btn_zoom_out); mg.addWidget(btn_reset)

        # Blink speed
        mg.addWidget(QLabel("Blink speed:"))
        self._blink_spin = QSpinBox()
        self._blink_spin.setRange(100, 5000); self._blink_spin.setValue(500)
        self._blink_spin.setSuffix(" ms"); self._blink_spin.setFixedWidth(100)
        self._blink_spin.setEnabled(False)
        mg.addWidget(self._blink_spin)
        mg.addStretch()
        root.addWidget(mode_grp)

        # ── Stretch mode ──────────────────────────────────────────────────────
        stretch_grp = QGroupBox("Stretch")
        sg = QHBoxLayout(stretch_grp); sg.setSpacing(8)
        sg.addWidget(QLabel("Mode:"))
        self._stretch_mode_combo = QComboBox()
        self._stretch_mode_combo.addItems(["Linked (A drives B)", "Independent", "Auto"])
        self._stretch_mode_combo.currentTextChanged.connect(self._on_stretch_mode_changed)
        sg.addWidget(self._stretch_mode_combo)

        sg.addWidget(QLabel("Black point:"))
        self._bp_slider = QSlider(Qt.Orientation.Horizontal)
        self._bp_slider.setRange(0, 500); self._bp_slider.setValue(0)
        self._bp_slider.valueChanged.connect(self._on_stretch_changed)
        sg.addWidget(self._bp_slider)

        sg.addWidget(QLabel("White point:"))
        self._wp_slider = QSlider(Qt.Orientation.Horizontal)
        self._wp_slider.setRange(500, 1000); self._wp_slider.setValue(1000)
        self._wp_slider.valueChanged.connect(self._on_stretch_changed)
        sg.addWidget(self._wp_slider)

        sg.addWidget(QLabel("Wipe:"))
        self._wipe_slider = QSlider(Qt.Orientation.Horizontal)
        self._wipe_slider.setRange(2, 98); self._wipe_slider.setValue(50)
        self._wipe_slider.setEnabled(False)
        self._wipe_slider.valueChanged.connect(
            lambda v: self._canvas.set_wipe_fraction(v / 100.0))
        sg.addWidget(self._wipe_slider)
        sg.addStretch()
        root.addWidget(stretch_grp)

        # ── Main area: canvas + stats tabs ────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(3)

        self._canvas = CompareCanvas()
        self._canvas.setMinimumHeight(400)
        self._canvas._wipe_frac_callback = lambda f: self._wipe_slider.setValue(int(f*100))
        splitter.addWidget(self._canvas)

        lower_tabs = QTabWidget()
        self._stats_widget = StatsWidget()
        lower_tabs.addTab(self._stats_widget, "📊  Statistics")
        log_w = QWidget(); ll = QVBoxLayout(log_w)
        self._log_view = QPlainTextEdit(); self._log_view.setReadOnly(True)
        ll.addWidget(self._log_view)
        lower_tabs.addTab(log_w, "📋  Log")
        splitter.addWidget(lower_tabs)
        splitter.setSizes([500, 200])
        root.addWidget(splitter, 1)

        self._status_lbl = QLabel("Load two FITS images to compare.")
        self._status_lbl.setObjectName("dim"); root.addWidget(self._status_lbl)

    def _setup_shortcuts(self):
        for key, mode in [("S", "side_by_side"), ("W", "wipe"),
                          ("B", "blink"), ("D", "difference")]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda m=mode: self._set_mode(m))

    # ── Public interface ──────────────────────────────────────────────────────

    def set_file_a(self, path: str):
        self._edit_a.setText(path)

    def set_file_b(self, path: str):
        self._edit_b.setText(path)

    def load_paths(self, path_a: str, path_b: str):
        self._edit_a.setText(path_a)
        self._edit_b.setText(path_b)
        self._load_and_compare()

    # ── Browse / load ─────────────────────────────────────────────────────────

    def _browse(self, which: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select Image {which}", "",
            "FITS (*.fit *.fits *.fts);;All (*)")
        if not path: return
        if which == "A":
            self._edit_a.setText(path)
        else:
            self._edit_b.setText(path)

    def _load_and_compare(self):
        path_a = self._edit_a.text().strip()
        path_b = self._edit_b.text().strip()
        if not path_a or not os.path.isfile(path_a):
            QMessageBox.warning(self, "Missing image A",
                "Select a valid FITS file for image A."); return
        if not path_b or not os.path.isfile(path_b):
            QMessageBox.warning(self, "Missing image B",
                "Select a valid FITS file for image B."); return
        try:
            self._status_lbl.setText("Loading images…")
            lum_a, hdr_a, is_rgb_a, full_a = load_fits_luminance(path_a)
            lum_b, hdr_b, is_rgb_b, full_b = load_fits_luminance(path_b)
            self._data_a = lum_a; self._data_b = lum_b
            self._path_a = path_a; self._path_b = path_b
            bp_a, wp_a = auto_stretch(lum_a)
            bp_b, wp_b = auto_stretch(lum_b)
            self._canvas.linked_stretch = (
                self._stretch_mode_combo.currentIndex() == 0)
            self._canvas.set_images(lum_a, lum_b, bp_a, wp_a, bp_b, wp_b)
            stats_a  = compute_image_stats(lum_a, "A")
            stats_b  = compute_image_stats(lum_b, "B")
            cmp      = compute_comparison_stats(lum_a, lum_b)
            self._stats_widget.update_stats(stats_a, stats_b, cmp,
                                             path_a, path_b)
            snr_gain = cmp.get("snr_gain_pct", 0)
            rms      = cmp.get("rms_diff", 0)
            self._status_lbl.setText(
                f"Loaded  |  A: {stats_a['shape']}  B: {stats_b['shape']}  "
                f"|  SNR gain: {snr_gain:+.1f}%  RMS diff: {rms:.4f}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_SUCCESS}; font-size:9pt;")
            self._log(f"Loaded A: {os.path.basename(path_a)}")
            self._log(f"Loaded B: {os.path.basename(path_b)}")
            self._log(f"SNR gain B vs A: {snr_gain:+.1f}%  RMS diff: {rms:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            self._status_lbl.setText(f"✗ {e}")
            self._status_lbl.setStyleSheet(f"color:{SIRIL_ERROR};")

    # ── Mode / stretch ────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self._mode = mode
        for m, btn in self._mode_btns.items():
            btn.setChecked(m == mode)
        self._canvas.set_mode(mode)

        is_wipe  = (mode == "wipe")
        is_blink = (mode == "blink")
        self._wipe_slider.setEnabled(is_wipe)
        self._blink_spin.setEnabled(is_blink)

        if is_blink:
            self._blink_timer.start(self._blink_spin.value())
        else:
            self._blink_timer.stop()

    def _blink_tick(self):
        if self._data_a is None: return
        self._blink_state = "B" if self._blink_state == "A" else "A"
        self._canvas.set_blink_state(self._blink_state)
        interval = self._blink_spin.value()
        if self._blink_timer.interval() != interval:
            self._blink_timer.setInterval(interval)

    def _on_stretch_mode_changed(self, text: str):
        if self._data_a is None: return
        linked = ("Linked" in text)
        self._canvas.linked_stretch = linked
        if "Auto" in text:
            if self._data_a is not None:
                bp_a, wp_a = auto_stretch(self._data_a)
                bp_b, wp_b = auto_stretch(self._data_b) if not linked else (bp_a, wp_a)
                self._canvas.update_stretch(bp_a, wp_a, bp_b, wp_b)

    def _on_stretch_changed(self):
        if self._data_a is None: return
        bp_raw = self._bp_slider.value() / 1000.0
        wp_raw = self._wp_slider.value() / 1000.0
        if self._data_a is not None:
            finite_a = self._data_a[np.isfinite(self._data_a)]
            dmin_a   = float(finite_a.min()) if len(finite_a) else 0.0
            dmax_a   = float(finite_a.max()) if len(finite_a) else 1.0
            dr_a     = dmax_a - dmin_a
        else:
            dmin_a, dr_a = 0.0, 1.0
        bp_a = dmin_a + bp_raw * dr_a
        wp_a = dmin_a + wp_raw * dr_a
        if self._canvas.linked_stretch:
            self._canvas.update_stretch(bp_a, wp_a, bp_a, wp_a)
        else:
            self._canvas.update_stretch(bp_a=bp_a, wp_a=wp_a)

    def _log(self, msg: str):
        t = dt.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{t}] {msg}")
        self.log_line.emit(f"[AB_COMPARE]  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# SEEING ESTIMATOR PANEL
# ─────────────────────────────────────────────────────────────────────────────

class SeeingPanel(QWidget):
    """
    Full Seeing Estimator UI.
    Exposes: set_fits_folder(path)
    Signals: analysis_done(dict), log_line(str)
    Cross-panel signal: best_worst_ready(str, str) — best/worst frame paths
    """
    analysis_done    = pyqtSignal(dict)
    best_worst_ready = pyqtSignal(str, str)  # best_path, worst_path
    log_line         = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fits_files   = []
        self._results      = []
        self._summary      = {}
        self._worker       = None
        self._cancel_event = threading.Event()
        self._build_ui()
        self._load_profiles()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        self._inner_tabs = QTabWidget()
        root.addWidget(self._inner_tabs, 1)

        self._inner_tabs.addTab(self._build_input_tab(),   "📁  Input")
        self._inner_tabs.addTab(self._build_results_tab(), "📊  Results")
        self._inner_tabs.addTab(self._build_export_tab(),  "💾  Export")

        log_tab = QWidget(); ll = QVBoxLayout(log_tab)
        self._log_view = QPlainTextEdit(); self._log_view.setReadOnly(True)
        ll.addWidget(self._log_view)
        self._inner_tabs.addTab(log_tab, "📋  Log")

        self._progress = QProgressBar()
        self._progress.setFixedHeight(10); self._progress.setVisible(False)
        root.addWidget(self._progress)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(8, 4, 8, 4)
        self._btn_analyze = QPushButton("▶  Analyze")
        self._btn_analyze.setObjectName("primary"); self._btn_analyze.setMinimumHeight(34)
        self._btn_analyze.clicked.connect(self._start_analysis)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setMinimumHeight(34)
        self._btn_cancel.setEnabled(False); self._btn_cancel.clicked.connect(self._cancel)
        btn_row.addWidget(self._btn_analyze, 3); btn_row.addWidget(self._btn_cancel, 1)
        root.addLayout(btn_row)

        self._status = QLabel("Ready. Select a FITS folder and click Analyze.")
        self._status.setObjectName("dim"); self._status.setContentsMargins(4, 2, 4, 2)
        root.addWidget(self._status)

    def _build_input_tab(self) -> QWidget:
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); lay = QVBoxLayout(inner)
        lay.setSpacing(10); lay.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(inner)

        grp_folder = QGroupBox("FITS Sequence Folder")
        g_lay = QVBoxLayout(grp_folder)
        row1 = QHBoxLayout()
        self._edit_folder = QLineEdit()
        self._edit_folder.setPlaceholderText("Path to folder with .fit / .fits files…")
        btn_browse = QPushButton("Browse…"); btn_browse.clicked.connect(self._browse_folder)
        btn_scan   = QPushButton("🔍  Scan"); btn_scan.clicked.connect(self._scan_folder)
        row1.addWidget(self._edit_folder, 4); row1.addWidget(btn_browse); row1.addWidget(btn_scan)
        g_lay.addLayout(row1)
        self._lbl_file_count = QLabel("No folder selected"); self._lbl_file_count.setObjectName("dim")
        g_lay.addWidget(self._lbl_file_count)
        lay.addWidget(grp_folder)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample every N frames:"))
        self._spin_sample = QSpinBox()
        self._spin_sample.setRange(1, 100); self._spin_sample.setValue(1); self._spin_sample.setFixedWidth(70)
        sample_row.addWidget(self._spin_sample); sample_row.addStretch()
        lay.addLayout(sample_row)

        grp_scale = QGroupBox("Pixel Scale")
        s_lay = QVBoxLayout(grp_scale)
        self._combo_profile = QComboBox(); self._combo_profile.addItem("Manual entry")
        self._combo_profile.currentIndexChanged.connect(self._on_profile_changed)
        s_lay.addWidget(QLabel("Equipment profile:")); s_lay.addWidget(self._combo_profile)
        row_ps = QHBoxLayout()
        row_ps.addWidget(QLabel("Pixel scale (arcsec/px):"))
        self._spin_scale = QDoubleSpinBox()
        self._spin_scale.setRange(0.01, 20.0); self._spin_scale.setValue(1.0); self._spin_scale.setDecimals(4)
        row_ps.addWidget(self._spin_scale); row_ps.addStretch()
        s_lay.addLayout(row_ps)
        self._lbl_profile_scale = QLabel(""); self._lbl_profile_scale.setObjectName("dim")
        self._lbl_profile_scale.setVisible(False); s_lay.addWidget(self._lbl_profile_scale)
        lay.addWidget(grp_scale)

        grp_obs = QGroupBox("Observation Settings")
        obs_form = QFormLayout(grp_obs); obs_form.setSpacing(8)
        self._combo_wavelength = QComboBox()
        for name in WAVELENGTHS: self._combo_wavelength.addItem(name)
        obs_form.addRow("Wavelength / filter:", self._combo_wavelength)
        self._spin_threshold_sigma = QDoubleSpinBox()
        self._spin_threshold_sigma.setRange(1.0, 20.0); self._spin_threshold_sigma.setValue(5.0); self._spin_threshold_sigma.setSingleStep(0.5)
        obs_form.addRow("Detection threshold (σ):", self._spin_threshold_sigma)
        self._spin_min_stars = QSpinBox()
        self._spin_min_stars.setRange(1, 20); self._spin_min_stars.setValue(3)
        obs_form.addRow("Min stars per frame:", self._spin_min_stars)
        self._combo_box_size = QComboBox()
        for sz in ["16px", "32px", "64px"]: self._combo_box_size.addItem(sz)
        self._combo_box_size.setCurrentIndex(1)
        obs_form.addRow("PSF box size:", self._combo_box_size)
        self._combo_color_handling = QComboBox()
        self._combo_color_handling.addItems(["Luminance", "Green channel"])
        obs_form.addRow("Color FITS handling:", self._combo_color_handling)
        lay.addWidget(grp_obs)

        grp_thresh = QGroupBox("Quality Threshold")
        t_lay = QVBoxLayout(grp_thresh)
        t_lay.addWidget(QLabel("Mark frames as poor if FWHM exceeds:"))
        slider_row = QHBoxLayout()
        self._slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self._slider_threshold.setRange(5, 100); self._slider_threshold.setValue(30)
        self._slider_threshold.valueChanged.connect(self._on_threshold_changed)
        self._lbl_threshold_val = QLabel('3.0"'); self._lbl_threshold_val.setMinimumWidth(50)
        slider_row.addWidget(self._slider_threshold); slider_row.addWidget(self._lbl_threshold_val)
        t_lay.addLayout(slider_row)
        lay.addWidget(grp_thresh)
        lay.addStretch()
        return scroll

    def _build_results_tab(self) -> QWidget:
        w   = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        grp_summary = QGroupBox("Session Summary")
        badge_row   = QHBoxLayout(grp_summary); badge_row.setSpacing(8)
        self._badge_best   = BadgeWidget("Best Seeing",   'arcsec')
        self._badge_median = BadgeWidget("Median Seeing", 'arcsec')
        self._badge_r0     = BadgeWidget("Best r0",       'cm')
        self._badge_usable = BadgeWidget("Usable Frames", '')
        for b in [self._badge_best, self._badge_median, self._badge_r0, self._badge_usable]:
            badge_row.addWidget(b)
        lay.addWidget(grp_summary)
        self._histogram = SeeingHistogram(); lay.addWidget(self._histogram)
        grp_table = QGroupBox("Per-Frame Results")
        t_lay = QVBoxLayout(grp_table)
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["#", "Filename", "FWHM (px)", 'FWHM (")', "r0 (cm)", "Stars", "Quality"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True); self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t_lay.addWidget(self._table)
        lay.addWidget(grp_table, 1)

        # Cross-panel button
        btn_ab = QPushButton("📋  Send best & worst frame to A/B Comparator")
        btn_ab.setObjectName("primary"); btn_ab.clicked.connect(self._send_to_ab)
        lay.addWidget(btn_ab)
        return w

    def _build_export_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(10)
        grp_opts = QGroupBox("Export Options")
        o_lay    = QVBoxLayout(grp_opts)
        self._chk_csv       = QCheckBox("Export CSV report"); self._chk_csv.setChecked(True)
        self._chk_rejection = QCheckBox("Export rejection list for Siril")
        self._chk_plot      = QCheckBox("Save seeing distribution plot as PNG")
        for c in [self._chk_csv, self._chk_rejection, self._chk_plot]: o_lay.addWidget(c)
        out_row = QHBoxLayout()
        self._edit_out_folder = QLineEdit()
        self._edit_out_folder.setPlaceholderText("Output folder (default: FITS folder)")
        btn_ob = QPushButton("Browse…"); btn_ob.clicked.connect(self._browse_out_folder)
        out_row.addWidget(QLabel("Output folder:")); out_row.addWidget(self._edit_out_folder, 3)
        out_row.addWidget(btn_ob); o_lay.addLayout(out_row)
        lay.addWidget(grp_opts)
        btn_export = QPushButton("💾  Export")
        btn_export.setObjectName("primary"); btn_export.setMinimumHeight(36)
        btn_export.clicked.connect(self._do_export); lay.addWidget(btn_export)
        grp_siril = QGroupBox("Siril Integration"); si_lay = QVBoxLayout(grp_siril)
        si_lay.addWidget(QLabel("Flag poor frames in the currently loaded Siril sequence:"))
        self._btn_flag = QPushButton("🚩  Flag Poor Frames in Siril")
        self._btn_flag.clicked.connect(self._flag_in_siril); si_lay.addWidget(self._btn_flag)
        lbl_fh = QLabel("Marks frames with FWHM > threshold as excluded in the Siril sequence")
        lbl_fh.setObjectName("dim"); si_lay.addWidget(lbl_fh)
        lay.addWidget(grp_siril); lay.addStretch()
        return w

    # ── Public interface ──────────────────────────────────────────────────────

    def set_fits_folder(self, folder: str):
        if folder and os.path.isdir(folder):
            self._edit_folder.setText(folder)
            self._scan_folder()

    def get_best_worst_paths(self) -> tuple:
        if not self._results: return ("", "")
        by_fwhm  = sorted(self._results, key=lambda r: r["fwhm_arcsec"])
        best_r   = by_fwhm[0]
        worst_r  = by_fwhm[-1]
        return best_r["path"], worst_r["path"]

    # ── Input helpers ─────────────────────────────────────────────────────────

    def _load_profiles(self):
        if not SEEING_HAS_EM: return
        try:
            from equipment_manager import load_all_profiles
            profiles = load_all_profiles()
            for p in profiles:
                name = p if isinstance(p, str) else p.get("name", str(p))
                self._combo_profile.addItem(name)
        except Exception:
            pass

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d: self._edit_folder.setText(d)

    def _browse_out_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self._edit_out_folder.setText(d)

    def _scan_folder(self):
        folder = self._edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Select a valid FITS folder."); return
        files = sorted(
            glob.glob(os.path.join(folder, "*.fit")) +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts")))
        self._fits_files = files
        self._lbl_file_count.setText(f"{len(files)} FITS files found")
        if not self._edit_out_folder.text():
            self._edit_out_folder.setText(folder)

    def _on_threshold_changed(self, v: int):
        thresh = v / 10.0
        self._lbl_threshold_val.setText(f'{thresh:.1f}"')
        threshold = self._slider_threshold.value() / 10.0
        fwhm_vals = [r["fwhm_arcsec"] for r in self._results]
        if fwhm_vals:
            self._histogram.update_plot(fwhm_vals, threshold)

    def _on_profile_changed(self, idx: int):
        if idx == 0:
            self._lbl_profile_scale.setVisible(False); return
        if not SEEING_HAS_EM:
            self._lbl_profile_scale.setVisible(False); return
        name = self._combo_profile.currentText()
        try:
            from equipment_manager import get_profile
            profile = get_profile(name)
            ps = profile.get("pixel_scale_arcsec_per_px",
                             profile.get("pixel_scale", None))
            if ps:
                self._spin_scale.setValue(float(ps))
                self._lbl_profile_scale.setText(f"→ Using {float(ps):.4f} arcsec/px from profile")
                self._lbl_profile_scale.setVisible(True)
        except Exception:
            pass

    def _get_config(self) -> dict:
        box_map = {"16px": 16, "32px": 32, "64px": 64}
        return {
            "pixel_scale_arcsec_per_px": self._spin_scale.value(),
            "wavelength_nm":            WAVELENGTHS.get(
                self._combo_wavelength.currentText(), 550.0),
            "threshold_sigma":          self._spin_threshold_sigma.value(),
            "fwhm_guess_px":            5.0,
            "box_size":                 box_map.get(self._combo_box_size.currentText(), 32),
            "min_stars":                self._spin_min_stars.value(),
            "color_handling":           self._combo_color_handling.currentText(),
            "sample_every":             self._spin_sample.value(),
            "seeing_threshold_arcsec":  self._slider_threshold.value() / 10.0,
        }

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _start_analysis(self):
        if not self._fits_files:
            QMessageBox.warning(self, "No files",
                "Scan a FITS folder first."); return
        self._cancel_event.clear()
        self._results = []
        self._table.setRowCount(0)
        self._btn_analyze.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._fits_files))
        self._worker = SeeingWorker(self._fits_files, self._get_config(), self._cancel_event)
        self._worker.progress.connect(lambda c, t, m: (
            self._progress.setValue(c),
            self._status.setText(f"{m}  [{c}/{t}]")))
        self._worker.result.connect(self._on_frame_result)
        self._worker.log_line.connect(self._log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._status.setText(f"Analyzing {len(self._fits_files)} frames…")

    def _cancel(self):
        self._cancel_event.set(); self._btn_cancel.setEnabled(False)

    def _on_frame_result(self, r: dict):
        row = self._table.rowCount(); self._table.insertRow(row)
        threshold = self._slider_threshold.value() / 10.0
        is_poor   = r["fwhm_arcsec"] > threshold
        vals = [str(row+1), r["filename"],
                f"{r['fwhm_px']:.2f}", f"{r['fwhm_arcsec']:.3f}",
                f"{r['r0_cm']:.1f}", str(r["star_count"]), r["quality"]]
        for col, v in enumerate(vals):
            item = QTableWidgetItem(v)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_poor:
                item.setForeground(QColor(SIRIL_ERROR))
            elif r["quality"] == "Excellent":
                item.setForeground(QColor(SIRIL_SUCCESS))
            self._table.setItem(row, col, item)
        self._results.append(r)
        fwhm_vals = [rx["fwhm_arcsec"] for rx in self._results]
        self._histogram.update_plot(fwhm_vals, threshold)

    def _on_finished(self, summary: dict):
        self._btn_analyze.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        self._summary = summary
        if summary.get("success"):
            bfwhm = summary["best_fwhm"]; mfwhm = summary["median_fwhm"]
            r0    = summary["best_r0"];   usable = summary["usable_count"]
            n_tot = summary["total_analyzed"]
            q, col = r0_to_quality_label(summary["median_r0"])
            self._badge_best.set_value(f"{bfwhm:.3f}",   SIRIL_SUCCESS)
            self._badge_median.set_value(f"{mfwhm:.3f}",  col)
            self._badge_r0.set_value(f"{r0:.1f}",         col)
            self._badge_usable.set_value(f"{usable}/{n_tot}", col)
            self._status.setText(
                f"✓  Median FWHM: {mfwhm:.3f}\"  Best r0: {r0:.1f}cm  ({q})")
            self._status.setStyleSheet(f"color:{col}; font-size:9pt;")
            self._inner_tabs.setCurrentIndex(1)
            self.analysis_done.emit(summary)
        else:
            self._status.setText(f"✗ {summary.get('error','')}")
            self._status.setStyleSheet(f"color:{SIRIL_ERROR};")

    # ── Cross-panel ───────────────────────────────────────────────────────────

    def _send_to_ab(self):
        best, worst = self.get_best_worst_paths()
        if not best:
            QMessageBox.warning(self, "No results",
                "Run analysis first."); return
        self.best_worst_ready.emit(best, worst)
        self._log(f"Sent to A/B Comparator: A={os.path.basename(best)}, "
                  f"B={os.path.basename(worst)}")

    # ── Export / Siril ────────────────────────────────────────────────────────

    def _do_export(self):
        if not self._results:
            QMessageBox.warning(self, "No data", "Run analysis first."); return
        out = self._edit_out_folder.text().strip() or self._edit_folder.text().strip()
        if not out: return
        os.makedirs(out, exist_ok=True)
        threshold = self._slider_threshold.value() / 10.0
        if self._chk_csv.isChecked():
            import csv as _csv
            path = os.path.join(out, "seeing_results.csv")
            with open(path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["#","filename","fwhm_px","fwhm_arcsec","r0_cm","stars","quality"])
                for i, r in enumerate(self._results, 1):
                    w.writerow([i, r["filename"], r["fwhm_px"], r["fwhm_arcsec"],
                                r["r0_cm"], r["star_count"], r["quality"]])
            self._log(f"CSV saved: {path}")
        if self._chk_rejection.isChecked():
            path = os.path.join(out, "poor_frames.txt")
            poor = [r["path"] for r in self._results if r["fwhm_arcsec"] > threshold]
            with open(path, "w") as f:
                for p in poor: f.write(p + "\n")
            self._log(f"Rejection list saved: {path} ({len(poor)} frames)")

    def _flag_in_siril(self):
        if not self._results:
            QMessageBox.warning(self, "No data", "Run analysis first."); return
        threshold = self._slider_threshold.value() / 10.0
        poor = [r["filename"] for r in self._results if r["fwhm_arcsec"] > threshold]
        try:
            siril = s.SirilInterface(); siril.connect()
            for fname in poor:
                try:
                    siril.cmd("seqinfo", fname, "-exclude")
                except Exception:
                    pass
            siril.disconnect()
            self._log(f"Flagged {len(poor)} poor frames in Siril.")
        except Exception as e:
            QMessageBox.critical(self, "Siril error", str(e))

    def _log(self, msg: str):
        t = dt.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{t}] {msg}")
        self.log_line.emit(f"[SEEING]  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Session / QA Suite  —  Siril")
        self.resize(1440, 900)
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
        lbl_icon = QLabel("🔬")
        lbl_icon.setStyleSheet("font-size:18pt; background:transparent;")
        lbl_title = QLabel("Session / QA Suite")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold; background:transparent;")
        lbl_ver = QLabel(
            f"v1.0  |  3 scripts  |  "
            f"Multi-Session Manager v{MSM_VERSION}  ·  A/B Comparator  ·  Seeing Estimator")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title); hl.addStretch(); hl.addWidget(lbl_ver)
        root.addWidget(header)

        # ── Shared project header ─────────────────────────────────────────────
        shared = QWidget()
        shared.setStyleSheet(f"background:{SIRIL_BG3}; border-bottom:1px solid {SIRIL_BORDER};")
        sl = QHBoxLayout(shared); sl.setContentsMargins(12, 6, 12, 6); sl.setSpacing(8)
        lbl_s = QLabel("Session project root:")
        lbl_s.setStyleSheet(f"color:{SIRIL_SECTION}; font-weight:bold; background:transparent;")
        sl.addWidget(lbl_s)
        self._shared_root_edit = QLineEdit()
        self._shared_root_edit.setPlaceholderText(
            "Set your project root — auto-fills FITS folder in Seeing Estimator…")
        sl.addWidget(self._shared_root_edit, 1)
        btn_browse_root = QPushButton("Browse…"); btn_browse_root.setFixedWidth(80)
        btn_browse_root.clicked.connect(self._browse_shared_root)
        sl.addWidget(btn_browse_root)
        btn_apply_root = QPushButton("→  Apply to all tools")
        btn_apply_root.setStyleSheet(
            f"background:{SIRIL_ACCENT2}; color:white; font-weight:bold; "
            f"border-radius:4px; padding:4px 12px;")
        btn_apply_root.clicked.connect(self._apply_shared_root)
        sl.addWidget(btn_apply_root)
        root.addWidget(shared)

        # ── Tool tabs ─────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        # Shared pipeline log
        self._pipeline_log = QPlainTextEdit(); self._pipeline_log.setReadOnly(True)

        # Session Manager
        self._session_panel = SessionManagerPanel()
        self._session_panel.project_scanned.connect(self._on_project_scanned)
        self._session_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._session_panel, "📁  Session Manager")

        # A/B Comparator
        self._ab_panel = ABComparePanel()
        self._ab_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._ab_panel, "🔍  A/B Comparator")

        # Seeing Estimator
        self._seeing_panel = SeeingPanel()
        self._seeing_panel.best_worst_ready.connect(self._on_best_worst_ready)
        self._seeing_panel.analysis_done.connect(self._on_seeing_done)
        self._seeing_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._seeing_panel, "👁  Seeing Estimator")

        # Log tab
        log_wrap = QWidget(); lw = QVBoxLayout(log_wrap)
        lw.setContentsMargins(6, 6, 6, 6)
        lbl_log = QLabel("Combined log from all tools"); lbl_log.setObjectName("dim")
        lw.addWidget(lbl_log)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._pipeline_log.clear)
        lw.addWidget(btn_clr); lw.addWidget(self._pipeline_log)
        self._tabs.addTab(log_wrap, "📋  Log")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom = QWidget(); bottom.setFixedHeight(32)
        bottom.setStyleSheet(f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(12, 0, 12, 0)
        self._status_lbl = QLabel(
            "Ready — set a project root or open each tool individually")
        self._status_lbl.setObjectName("dim")
        bl.addWidget(self._status_lbl, 1)
        root.addWidget(bottom)

        # Startup log
        self._pipeline_log.appendPlainText(
            f"[{dt.now().strftime('%H:%M:%S')}] Session / QA Suite — ready")
        self._pipeline_log.appendPlainText(
            f"[{dt.now().strftime('%H:%M:%S')}] "
            "Tools: Multi-Session Manager  ·  A/B Comparator  ·  Seeing Estimator")

    # ── Shared root management ────────────────────────────────────────────────

    def _browse_shared_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select project root folder")
        if d:
            self._shared_root_edit.setText(d)
            self._apply_shared_root()

    def _apply_shared_root(self):
        root = self._shared_root_edit.text().strip()
        if not root or not os.path.isdir(root): return
        # Pre-fill session manager project root
        self._session_panel._project_tab.path_edit.setText(root)
        # Pre-fill seeing estimator with lights subfolder or root itself
        lights_folders = sorted(
            glob.glob(os.path.join(root, "**", "lights"), recursive=True))
        if lights_folders:
            # Use the folder with the most FITS files
            best_folder = max(lights_folders,
                              key=lambda f: len(find_fits(f)))
            self._seeing_panel.set_fits_folder(best_folder)
        else:
            self._seeing_panel.set_fits_folder(root)
        self._status_lbl.setText(
            f"Project root applied: {os.path.basename(root)}")
        self._pipeline_log.appendPlainText(
            f"[{dt.now().strftime('%H:%M:%S')}] "
            f"Shared root set: {root}")

    # ── Cross-tool signals ────────────────────────────────────────────────────

    def _on_project_scanned(self, root: str):
        """Session Manager scan → update shared root + Seeing Estimator."""
        self._shared_root_edit.setText(root)
        # Collect all lights FITS files for seeing estimator
        fits_files = self._session_panel.get_fits_files()
        if fits_files:
            folder = os.path.dirname(fits_files[0])
            self._seeing_panel.set_fits_folder(folder)
            self._pipeline_log.appendPlainText(
                f"[{dt.now().strftime('%H:%M:%S')}] "
                f"Cross-tool: {len(fits_files)} FITS files → Seeing Estimator")
        self._status_lbl.setText(
            f"✓ Project scanned: {os.path.basename(root)}")

    def _on_best_worst_ready(self, best_path: str, worst_path: str):
        """Seeing Estimator best/worst → A/B Comparator."""
        self._ab_panel.load_paths(best_path, worst_path)
        self._tabs.setCurrentIndex(1)   # Switch to A/B tab
        self._pipeline_log.appendPlainText(
            f"[{dt.now().strftime('%H:%M:%S')}] "
            f"Cross-tool: best/worst frame → A/B Comparator\n"
            f"  A (best):  {os.path.basename(best_path)}\n"
            f"  B (worst): {os.path.basename(worst_path)}")
        self._status_lbl.setText(
            "Best/worst frame loaded in A/B Comparator")

    def _on_seeing_done(self, summary: dict):
        median_fwhm = summary.get("median_fwhm", 0)
        best_r0     = summary.get("best_r0", 0)
        q, _        = r0_to_quality_label(best_r0)
        self._pipeline_log.appendPlainText(
            f"[{dt.now().strftime('%H:%M:%S')}] "
            f"Seeing analysis complete: "
            f"median FWHM {median_fwhm:.3f}\"  best r0 {best_r0:.1f}cm  ({q})")
        self._status_lbl.setText(
            f"✓ Seeing: {median_fwhm:.3f}\"  r0={best_r0:.1f}cm  ({q})"
            "  |  Click 'Send best & worst to A/B' in Results tab")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
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
