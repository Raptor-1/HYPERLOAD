r"""
variable_star_suite.py  —  Script Merger: Variable Star Suite
=============================================================
Merged launcher for:
    ⭐  Variable Star Pipeline    (variable_star.py)
    〜  Periodicity Analysis      (period_finder.py)
    🌫  Extinction Map            (extinction_map.py)

Architecture: Section A sequential pipeline (3 steps)
    PHOTOMETRY → PERIOD ANALYSIS → EXTINCTION CALIBRATION

In-memory handoffs (Section A4):
    Step 1 → Step 2: {"times": np.array, "diff_mags": np.array,
                       "errors": np.array}   (light curve arrays, no file I/O)
    Step 2 → Step 3: {"fits_files": list, "k_correction": float}
                      (approved FITS list + optional extinction coefficient)

References:
    Bouguer's Law:          Young 1974, ApJ 189, 587
    Lomb-Scargle:           Lomb 1976, Ap&SS 39, 447; Scargle 1982, ApJ 263, 835
    ACF period detection:   McQuillan et al. 2013, ApJ 775, L11
    Morlet wavelet:         Torrence & Compo 1998, BAMS 79, 61

Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:  Siril → Scripts → variable_star_suite
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

from variable_star import (
    VarStarWorker,
    LCCanvas,
    FieldCanvas,
    # Algorithms
    aperture_photometry_target,
    build_variable_star_lightcurve,
    classify_variable,
    export_aavso_extended,
    compute_ephemeris,
    crossmatch_vsx,
    crossmatch_simbad,
    get_wcs_coords,
    save_session,
    load_session,
    # Fallbacks
    _fallback_lomb_scargle,
    _fallback_normalize,
    _fallback_phase_fold,
    # Constants
    VARIABILITY_TYPES,
    HAS_PERIOD_FINDER,
    HAS_ASTROQUERY,
    HAS_TRANSIENT_DETECTOR,
    SIRIL_NOVA,
    # Theme
    SIRIL_STYLESHEET,
    SIRIL_BG, SIRIL_BG2, SIRIL_BG3,
    SIRIL_ACCENT, SIRIL_ACCENT2,
    SIRIL_TEXT, SIRIL_TEXT_DIM,
    SIRIL_BORDER, SIRIL_SUCCESS,
    SIRIL_WARNING, SIRIL_SECTION,
    SIRIL_ERROR,
)

from period_finder import (
    PeriodWorker,
    PeriodogramCanvas,
    WaveletCanvas,
    LightCurveCanvas,
    # Algorithms
    run_lomb_scargle,
    run_acf,
    run_wavelet,
    cross_validate_periods,
    phase_fold,
    normalize_lightcurve,
    load_csv_lightcurve,
    compute_from_fits_sequence,
    parse_manual_data,
    check_harmonics,
    save_results_csv,
)

from extinction_map import (
    ExtinctionWorker,
    # Algorithms
    compute_airmass_hardie,
    compute_airmass_simple,
    get_altitude_from_header,
    measure_instrumental_magnitudes,
    detect_reference_stars,
    fit_bouguer_law,
    compute_per_frame_transparency,
    extinction_from_single_frame,
    HAS_ASTROQUERY as EXT_HAS_ASTROQUERY,
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
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSizePolicy, QScrollArea, QStackedWidget, QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QRectF
from PyQt6.QtGui import QFont, QColor, QDesktopServices, QPainter, QPen


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
        n    = len(self.steps)
        W, H = self.width(), self.height()
        bw   = min(170, (W - 40) // n - 14)
        bh   = 50
        gap  = (W - n * bw) // (n + 1)
        by   = (H - bh) // 2

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
                ax = bx + bw + 4; ay = by + bh // 2
                p.setPen(QPen(QColor("#3a4055"), 1.5))
                p.drawLine(int(ax), int(ay), int(ax + gap - 8), int(ay))
                p.drawLine(int(ax + gap - 8), int(ay), int(ax + gap - 14), int(ay - 5))
                p.drawLine(int(ax + gap - 8), int(ay), int(ax + gap - 14), int(ay + 5))
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# VARIABLE STAR PANEL
# ─────────────────────────────────────────────────────────────────────────────

class VarStarPanel(QWidget):
    """
    Full Variable Star Pipeline UI.
    Exposes: get_config(), run_pipeline(cancel_event)
    Signals: lc_ready(dict), finished(dict), log_line(str)
    """
    lc_ready = pyqtSignal(dict)   # light curve for pipeline handoff
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._fits_files   = []
        self._first_header = None
        self._lc_result    = None
        self._final_result = None
        self._session_loaded = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left: settings panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)
        self._build_left(left_layout)
        left_layout.addStretch()

        # Right: tabs
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self._tabs = QTabWidget()
        right_layout.addWidget(self._tabs)
        splitter.addWidget(right)
        self._build_right_tabs()
        splitter.setSizes([360, 880])

        # Bottom
        bot = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 6); self._progress.setValue(0)
        self._progress.setVisible(False); self._progress.setFixedHeight(8)
        bot.addWidget(self._progress, 1)

        self._btn_run = QPushButton("▶  Run Variable Star pipeline")
        self._btn_run.setObjectName("primary")
        self._btn_run.clicked.connect(self._run)
        bot.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bot.addWidget(self._btn_cancel)
        root.addLayout(bot)

        self._status_lbl = QLabel("Ready — load a FITS sequence or CSV")
        self._status_lbl.setObjectName("dim")
        root.addWidget(self._status_lbl)

    def _build_left(self, layout):
        # Data source
        grp_src = QGroupBox("Data source")
        vl_src  = QVBoxLayout(grp_src)
        self._combo_src = QComboBox()
        self._combo_src.addItems([
            "FITS sequence (aperture photometry)",
            "CSV light curve",
            "Transient detector output",
        ])
        self._combo_src.currentIndexChanged.connect(self._on_source_changed)
        vl_src.addWidget(self._combo_src)

        # FITS widgets
        self._fits_widget = QWidget()
        fl = QVBoxLayout(self._fits_widget); fl.setContentsMargins(0,0,0,0); fl.setSpacing(4)
        row_folder = QHBoxLayout()
        self._edit_folder = QLineEdit(); self._edit_folder.setPlaceholderText("FITS folder…")
        btn_browse_folder = QPushButton("…"); btn_browse_folder.setFixedWidth(28)
        btn_browse_folder.clicked.connect(self._browse_folder)
        row_folder.addWidget(self._edit_folder); row_folder.addWidget(btn_browse_folder)
        fl.addLayout(row_folder)
        btn_load_seq = QPushButton("🔍  Load sequence & show field")
        btn_load_seq.clicked.connect(self._load_sequence)
        fl.addWidget(btn_load_seq)
        self._lbl_seq_info = QLabel("—"); self._lbl_seq_info.setObjectName("dim")
        fl.addWidget(self._lbl_seq_info)
        lbl_hint = QLabel("Left-click: target   Right-click: comp stars")
        lbl_hint.setObjectName("dim"); lbl_hint.setWordWrap(True)
        fl.addWidget(lbl_hint)
        self._lbl_target = QLabel("Target: not selected"); self._lbl_target.setObjectName("dim")
        fl.addWidget(self._lbl_target)
        self._lbl_comps  = QLabel("Comp stars: 0"); self._lbl_comps.setObjectName("dim")
        fl.addWidget(self._lbl_comps)
        form_aper = QFormLayout()
        self._spin_aper = QDoubleSpinBox()
        self._spin_aper.setRange(3, 20); self._spin_aper.setValue(8.0); self._spin_aper.setSuffix(" px")
        form_aper.addRow("Aperture:", self._spin_aper)
        fl.addLayout(form_aper)
        vl_src.addWidget(self._fits_widget)

        # CSV widgets
        self._csv_widget = QWidget()
        cl = QVBoxLayout(self._csv_widget); cl.setContentsMargins(0,0,0,0)
        row_csv = QHBoxLayout()
        self._edit_csv = QLineEdit(); self._edit_csv.setPlaceholderText("light curve CSV…")
        btn_browse_csv = QPushButton("…"); btn_browse_csv.setFixedWidth(28)
        btn_browse_csv.clicked.connect(self._browse_csv)
        row_csv.addWidget(self._edit_csv); row_csv.addWidget(btn_browse_csv)
        cl.addLayout(row_csv)
        btn_load_csv = QPushButton("Load CSV"); btn_load_csv.clicked.connect(self._load_csv)
        cl.addWidget(btn_load_csv)
        self._csv_widget.setVisible(False)
        vl_src.addWidget(self._csv_widget)
        layout.addWidget(grp_src)

        # Session
        grp_sess = QGroupBox("Session")
        sl = QHBoxLayout(grp_sess)
        btn_load_session = QPushButton("📂  Load previous session")
        btn_load_session.clicked.connect(self._load_session_dialog)
        sl.addWidget(btn_load_session)
        layout.addWidget(grp_sess)

        # Period search
        grp_period = QGroupBox("Period search")
        pl = QVBoxLayout(grp_period)
        self._chk_ls  = QCheckBox("Lomb-Scargle");  self._chk_ls.setChecked(True)
        self._chk_acf = QCheckBox("ACF");            self._chk_acf.setChecked(True)
        self._chk_wav = QCheckBox("Wavelet");        self._chk_wav.setChecked(True)
        for c in [self._chk_ls, self._chk_acf, self._chk_wav]:
            pl.addWidget(c)
        form_p = QFormLayout()
        self._spin_min_p = QDoubleSpinBox(); self._spin_min_p.setRange(0,10000); self._spin_min_p.setValue(0); self._spin_min_p.setSpecialValueText("auto"); self._spin_min_p.setSuffix(" d")
        self._spin_max_p = QDoubleSpinBox(); self._spin_max_p.setRange(0,10000); self._spin_max_p.setValue(0); self._spin_max_p.setSpecialValueText("auto"); self._spin_max_p.setSuffix(" d")
        form_p.addRow("Min period:", self._spin_min_p)
        form_p.addRow("Max period:", self._spin_max_p)
        pl.addLayout(form_p)
        if not HAS_PERIOD_FINDER:
            lbl_pf = QLabel("⚠ period_finder not found — LS only")
            lbl_pf.setObjectName("warn"); lbl_pf.setWordWrap(True)
            pl.addWidget(lbl_pf)
        layout.addWidget(grp_period)

        # Target info
        grp_target = QGroupBox("Target info")
        tl = QFormLayout(grp_target)
        self._edit_target_name = QLineEdit(); self._edit_target_name.setPlaceholderText("e.g. V* MY Cyg")
        self._edit_obs_code    = QLineEdit(); self._edit_obs_code.setPlaceholderText("AAVSO code")
        self._combo_filter     = QComboBox(); self._combo_filter.addItems(["V","B","R","I","CV","TG","Vis"])
        tl.addRow("Target name:",   self._edit_target_name)
        tl.addRow("Observer code:", self._edit_obs_code)
        tl.addRow("Filter:",        self._combo_filter)
        layout.addWidget(grp_target)

        # Comp star mags
        grp_comp_mags = QGroupBox("Comparison star magnitudes")
        cml = QFormLayout(grp_comp_mags)
        self._spin_comp1 = QDoubleSpinBox(); self._spin_comp1.setRange(0,25); self._spin_comp1.setValue(0); self._spin_comp1.setSpecialValueText("unknown")
        self._spin_comp2 = QDoubleSpinBox(); self._spin_comp2.setRange(0,25); self._spin_comp2.setValue(0); self._spin_comp2.setSpecialValueText("unknown")
        cml.addRow("Comp 1 mag:", self._spin_comp1)
        cml.addRow("Comp 2 mag:", self._spin_comp2)
        lbl_ch = QLabel("From AAVSO VSP chart. Leave 0 for differential only.")
        lbl_ch.setObjectName("dim"); lbl_ch.setWordWrap(True); cml.addRow(lbl_ch)
        layout.addWidget(grp_comp_mags)

        # Catalog
        grp_cat = QGroupBox("Catalog cross-match")
        cat_l = QVBoxLayout(grp_cat)
        self._chk_crossmatch = QCheckBox("Cross-match VSX / SIMBAD")
        self._chk_crossmatch.setChecked(True)
        if not HAS_ASTROQUERY:
            self._chk_crossmatch.setEnabled(False)
            self._chk_crossmatch.setToolTip("astroquery not installed")
        cat_l.addWidget(self._chk_crossmatch)
        layout.addWidget(grp_cat)

        # Export
        grp_export = QGroupBox("Export")
        el = QVBoxLayout(grp_export)
        row_out = QHBoxLayout()
        self._edit_out = QLineEdit(); self._edit_out.setPlaceholderText("Output folder…")
        btn_out = QPushButton("…"); btn_out.setFixedWidth(28)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self._edit_out); row_out.addWidget(btn_out)
        el.addLayout(row_out)
        self._chk_aavso     = QCheckBox("AAVSO Extended Format report"); self._chk_aavso.setChecked(True)
        self._chk_sess_json = QCheckBox("Session JSON"); self._chk_sess_json.setChecked(True)
        self._chk_lc_csv    = QCheckBox("Light curve CSV"); self._chk_lc_csv.setChecked(True)
        for c in [self._chk_aavso, self._chk_sess_json, self._chk_lc_csv]:
            el.addWidget(c)
        layout.addWidget(grp_export)

    def _build_right_tabs(self):
        self._field_canvas = FieldCanvas()
        self._field_canvas.target_changed.connect(self._on_target_selected)
        self._field_canvas.comps_changed.connect(self._on_comps_changed)
        self._tabs.addTab(self._field_canvas, "🌟  Field")

        self._lc_canvas = LCCanvas()
        self._tabs.addTab(self._lc_canvas, "📈  Light Curve")

        cls_widget = QWidget()
        cls_layout = QVBoxLayout(cls_widget)
        self._lbl_period_big = QLabel("P = —")
        self._lbl_period_big.setStyleSheet(
            f"color:{SIRIL_NOVA}; font-size:18pt; font-weight:bold; padding:4px;")
        self._lbl_type = QLabel("—")
        self._lbl_type.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:13pt;")
        self._lbl_amp  = QLabel("Amplitude: —")
        self._lbl_amp.setObjectName("dim")
        self._lbl_conf = QLabel(""); self._lbl_conf.setWordWrap(True)
        self._txt_reasoning = QPlainTextEdit(); self._txt_reasoning.setReadOnly(True)
        self._txt_reasoning.setFixedHeight(80)
        for w in [self._lbl_period_big, self._lbl_type, self._lbl_amp,
                  self._lbl_conf, self._txt_reasoning]:
            cls_layout.addWidget(w)

        grp_catalog = QGroupBox("Catalog cross-match")
        cl = QVBoxLayout(grp_catalog)
        self._lbl_vsx    = QLabel("VSX: —");    cl.addWidget(self._lbl_vsx)
        self._lbl_simbad = QLabel("SIMBAD: —"); cl.addWidget(self._lbl_simbad)
        cls_layout.addWidget(grp_catalog)

        grp_ephem = QGroupBox("Next predicted events")
        el = QVBoxLayout(grp_ephem)
        self._tbl_ephem = QTableWidget(0, 2)
        self._tbl_ephem.setHorizontalHeaderLabels(["JD", "ISO"])
        self._tbl_ephem.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._tbl_ephem.setMaximumHeight(100)
        el.addWidget(self._tbl_ephem)
        cls_layout.addWidget(grp_ephem)

        row_btns = QHBoxLayout()
        btn_aavso  = QPushButton("💾  Save AAVSO report"); btn_aavso.clicked.connect(self._save_aavso_dialog)
        btn_vsp    = QPushButton("🌐  Open VSP chart");    btn_vsp.clicked.connect(self._open_vsp)
        for b in [btn_aavso, btn_vsp]: row_btns.addWidget(b)
        row_btns.addStretch()
        cls_layout.addLayout(row_btns)

        self._txt_aavso = QPlainTextEdit(); self._txt_aavso.setReadOnly(True)
        self._txt_aavso.setMaximumHeight(100)
        cls_layout.addWidget(self._txt_aavso)
        cls_layout.addStretch()
        self._tabs.addTab(cls_widget, "⭐  Classification")

        log_tab = QWidget(); ll = QVBoxLayout(log_tab)
        self._log_view = QPlainTextEdit(); self._log_view.setReadOnly(True)
        ll.addWidget(self._log_view)
        self._tabs.addTab(log_tab, "📋  Log")

    # ── Public interface ──────────────────────────────────────────────────────

    def get_config(self) -> dict:
        src_idx = self._combo_src.currentIndex()
        lc_source = ["fits_sequence", "csv", "transient"][src_idx]
        cfg = {
            "lc_source":     lc_source,
            "fits_files":    self._fits_files,
            "csv_path":      self._edit_csv.text().strip() if hasattr(self, "_edit_csv") else "",
            "target_x":      getattr(self._field_canvas.target_pos, "__getitem__", lambda i: 0)(0) if self._field_canvas.target_pos else 0,
            "target_y":      getattr(self._field_canvas.target_pos, "__getitem__", lambda i: 0)(1) if self._field_canvas.target_pos else 0,
            "comp_positions":self._field_canvas.comp_positions,
            "aperture_r":    self._spin_aper.value(),
            "run_ls":        self._chk_ls.isChecked(),
            "run_acf":       self._chk_acf.isChecked(),
            "run_wavelet":   self._chk_wav.isChecked(),
            "min_period":    self._spin_min_p.value() or None,
            "max_period":    self._spin_max_p.value() or None,
            "target_name":   self._edit_target_name.text().strip() or "TARGET",
            "observer_code": self._edit_obs_code.text().strip() or "XXX",
            "filter_name":   self._combo_filter.currentText(),
            "crossmatch":    self._chk_crossmatch.isChecked(),
            "export_aavso":  self._chk_aavso.isChecked(),
            "output_dir":    self._edit_out.text().strip() or ".",
            "comp_mags": ({"COMP1": self._spin_comp1.value()}
                          if self._spin_comp1.value() > 0 else {}),
            "fits_header":   self._first_header,
        }
        if self._field_canvas.target_pos:
            cfg["target_x"] = self._field_canvas.target_pos[0]
            cfg["target_y"] = self._field_canvas.target_pos[1]
        return cfg

    def get_light_curve(self) -> dict | None:
        return self._lc_result

    def get_fits_files(self) -> list:
        return list(self._fits_files)

    def set_enabled(self, enabled: bool):
        self._btn_run.setEnabled(enabled)

    def cancel(self):
        self._cancel_event.set()
        if self._worker: self._worker.cancel()

    # ── Browse / load helpers ─────────────────────────────────────────────────

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d: self._edit_folder.setText(d)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "",
            "CSV (*.csv *.txt);;All (*)")
        if path: self._edit_csv.setText(path)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self._edit_out.setText(d)

    def _on_source_changed(self, idx: int):
        self._fits_widget.setVisible(idx == 0)
        self._csv_widget.setVisible(idx == 1)

    def _load_sequence(self):
        folder = self._edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Select a FITS folder first.")
            return
        files = sorted(
            glob.glob(os.path.join(folder, "*.fit")) +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts")))
        if not files:
            QMessageBox.warning(self, "No FITS", "No FITS files found."); return
        self._fits_files = files
        self._lbl_seq_info.setText(f"{len(files)} FITS files")
        try:
            with astropy_fits.open(files[0]) as h:
                data = h[0].data.astype(np.float32)
                self._first_header = h[0].header.copy()
            if data.ndim == 3:
                data = data[0] if data.shape[0] <= 4 else (
                    0.299*data[0] + 0.587*data[1] + 0.114*data[2])
            self._field_canvas.show_frame(data)
            self._tabs.setCurrentIndex(0)
        except Exception as e:
            self._log(f"Preview failed: {e}")

    def _load_csv(self):
        path = self._edit_csv.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file", "Select a CSV file first."); return
        try:
            if HAS_PERIOD_FINDER:
                times, fluxes, errors = load_csv_lightcurve(path)
            else:
                import csv as _csv
                rows = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"): continue
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try: rows.append([float(p) for p in parts[:3]])
                            except ValueError: pass
                arr = np.array(rows)
                times  = arr[:, 0]
                fluxes = arr[:, 1]
                errors = arr[:, 2] if arr.shape[1] > 2 else np.full_like(fluxes, 0.01)
            lc = {"times": times, "diff_mags": fluxes,
                  "errors": errors if errors is not None else np.full_like(fluxes, 0.01),
                  "n_frames": len(times), "n_failed": 0}
            self._lc_result = lc
            self._lc_canvas.show_lightcurve(lc)
            self._log(f"CSV loaded: {len(times)} points from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _load_session_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load session JSON", "",
            "JSON (*.json);;All (*)")
        if not path: return
        try:
            session = load_session(path)
            self._session_loaded = session
            lc = session.get("light_curve", {})
            if lc:
                self._lc_result = lc
                self._lc_canvas.show_lightcurve(lc)
            self._edit_target_name.setText(session.get("target_name", ""))
            self._log(f"Session loaded: {path}")
            self._log(
                f"  Period: {session.get('best_period','?')}  "
                f"Type: {session.get('var_type','?')}")
        except Exception as e:
            QMessageBox.critical(self, "Session error", str(e))

    def _on_target_selected(self, x: float, y: float):
        self._lbl_target.setText(f"Target: ({x:.1f}, {y:.1f})")

    def _on_comps_changed(self, positions: list):
        self._lbl_comps.setText(f"Comp stars: {len(positions)}")

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _run(self):
        src_idx = self._combo_src.currentIndex()
        if src_idx == 0:
            if not self._fits_files:
                QMessageBox.warning(self, "No sequence",
                    "Load a FITS sequence first."); return
            if not self._field_canvas.target_pos:
                QMessageBox.warning(self, "No target",
                    "Left-click the target star on the field image."); return
            if not self._field_canvas.comp_positions:
                QMessageBox.warning(self, "No comp stars",
                    "Right-click to add comparison stars."); return
        elif src_idx == 1:
            if not self._lc_result:
                QMessageBox.warning(self, "No LC", "Load a CSV light curve first."); return
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True); self._progress.setValue(0)
        cfg = self.get_config()
        if self._session_loaded and self._lc_result:
            cfg["lc_source"] = "csv"
            cfg["_preloaded_lc"] = self._lc_result
        self._worker = VarStarWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log)
        self._worker.lc_ready.connect(self._on_lc_ready)
        self._worker.period_ready.connect(lambda d: None)
        self._worker.fold_ready.connect(self._on_fold_ready)
        self._worker.class_ready.connect(self._on_class_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def run_pipeline(self, cancel_event: threading.Event):
        self._cancel_event = cancel_event
        self._run()

    def _cancel(self):
        self.cancel()
        self._btn_cancel.setEnabled(False)

    # ── Worker signal handlers ─────────────────────────────────────────────────

    def _on_progress(self, step, total, msg):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(f"Step {step}/{total}: {msg}", SIRIL_ACCENT)

    def _on_lc_ready(self, lc: dict):
        if self._session_loaded and "light_curve" in self._session_loaded:
            old = self._session_loaded["light_curve"]
            merged_t = np.concatenate([np.array(old.get("times", [])),    lc["times"]])
            merged_m = np.concatenate([np.array(old.get("diff_mags", [])),lc["diff_mags"]])
            merged_e = np.concatenate([np.array(old.get("errors", [])),   lc["errors"]])
            idx = np.argsort(merged_t)
            lc["times"]    = merged_t[idx]; lc["diff_mags"] = merged_m[idx]
            lc["errors"]   = merged_e[idx]; lc["n_frames"]  = len(merged_t)
        self._lc_result = lc
        self._lc_canvas.show_lightcurve(lc)
        if self._chk_lc_csv.isChecked():
            out  = self._edit_out.text().strip() or "."
            name = self._edit_target_name.text().strip() or "TARGET"
            csv_path = os.path.join(out, f"{name}_lightcurve.csv")
            try:
                os.makedirs(out, exist_ok=True)
                with open(csv_path, "w") as f:
                    f.write("# JD,diff_mag,error\n")
                    for t, m, e in zip(lc["times"], lc["diff_mags"], lc["errors"]):
                        f.write(f"{t:.6f},{m:.4f},{e:.4f}\n")
                self._log(f"Light curve CSV: {csv_path}")
            except Exception as ex:
                self._log(f"LC CSV write failed: {ex}")
        self.lc_ready.emit(lc)

    def _on_fold_ready(self, fold: dict):
        self._lc_canvas.show_phase_fold(fold)

    def _on_class_ready(self, var_type: str, conf: float, notes: list):
        vt = VARIABILITY_TYPES.get(var_type, {})
        self._lbl_type.setText(vt.get("name", var_type))
        if conf >= 0.7:
            badge = f"HIGH confidence ({conf*100:.0f}%)"
            col   = SIRIL_SUCCESS
        elif conf >= 0.4:
            badge = f"MODERATE confidence ({conf*100:.0f}%)"
            col   = SIRIL_WARNING
        else:
            badge = f"LOW confidence ({conf*100:.0f}%) — more data needed"
            col   = SIRIL_ERROR
        self._lbl_conf.setText(badge)
        self._lbl_conf.setStyleSheet(f"color:{col}; font-weight:bold; font-size:10pt;")
        self._txt_reasoning.setPlainText(
            vt.get("description", "") + "\n\n" + "\n".join(f"• {n}" for n in notes))
        self._tabs.setCurrentIndex(2)

    def _on_finished(self, result: dict):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._final_result = result
        if not result.get("success"):
            err = result.get("error", "Unknown")
            self._set_status(f"✗ Failed: {err}", SIRIL_ERROR)
            self.finished.emit(result)
            return
        p   = result.get("best_period", 0)
        vt  = VARIABILITY_TYPES.get(result.get("var_type","UNK"), {})
        con = result.get("confidence", 0)
        amp = result.get("amplitude", 0)
        self._lbl_period_big.setText(f"P = {p:.6f} d")
        self._lbl_amp.setText(f"Amplitude: {amp:.3f} mag")
        vsx = result.get("vsx_match")
        if vsx:
            self._lbl_vsx.setText(
                f"VSX: {vsx['name']}  type={vsx['type']}  "
                f"P={vsx.get('period','?')}  [{vsx.get('mag_max','?')}–{vsx.get('mag_min','?')}]")
            self._lbl_vsx.setStyleSheet(f"color:{SIRIL_SUCCESS};")
        simbad = result.get("simbad_match")
        if simbad:
            self._lbl_simbad.setText(f"SIMBAD: {simbad['name']}  ({simbad['otype']})")
            self._lbl_simbad.setStyleSheet(f"color:{SIRIL_ACCENT};")
        ephem = result.get("ephemeris", [])
        self._tbl_ephem.setRowCount(len(ephem))
        for row, (jd, iso) in enumerate(ephem):
            self._tbl_ephem.setItem(row, 0, QTableWidgetItem(str(jd)))
            self._tbl_ephem.setItem(row, 1, QTableWidgetItem(iso))
        aavso_path = result.get("aavso_path")
        if aavso_path and os.path.isfile(aavso_path):
            try:
                with open(aavso_path) as f:
                    self._txt_aavso.setPlainText(f.read()[:2000])
            except Exception:
                pass
        self._set_status(
            f"✓ P={p:.6f} d  Type: {vt.get('name','?')}  Conf: {con*100:.0f}%",
            SIRIL_SUCCESS)
        self._log("Variable Star pipeline complete.")
        self.log_line.emit(f"[VARSTAR]  ✓ P={p:.6f} d  Type={result.get('var_type','?')}  conf={con:.2f}")
        self.finished.emit(result)

    # ── Dialogs ───────────────────────────────────────────────────────────────

    def _save_aavso_dialog(self):
        if not self._lc_result:
            QMessageBox.warning(self, "No data", "Run the pipeline first."); return
        target_name = self._edit_target_name.text().strip() or "TARGET"
        path, _ = QFileDialog.getSaveFileName(self, "Save AAVSO report",
            f"{target_name}_aavso.csv", "CSV (*.csv);;All (*)")
        if not path: return
        comp_mags = {}
        if self._spin_comp1.value() > 0:
            comp_mags["COMP1"] = self._spin_comp1.value()
        try:
            export_aavso_extended(
                self._lc_result, target_name,
                self._edit_obs_code.text().strip() or "XXX",
                comp_mags, self._combo_filter.currentText(), path)
            self._log(f"AAVSO report saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed:\n{e}")

    def _open_vsp(self):
        if self._final_result and self._final_result.get("ra") is not None:
            ra  = self._final_result["ra"]
            dec = self._final_result["dec"]
            url = f"https://www.aavso.org/apps/vsp/?ra={ra:.5f}&dec={dec:.5f}&fov=60"
        else:
            url = "https://www.aavso.org/apps/vsp/"
        QDesktopServices.openUrl(QUrl(url))

    # ── Log / status ──────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{ts}] {msg}")
        sb = self._log_view.verticalScrollBar(); sb.setValue(sb.maximum())
        self.log_line.emit(f"[VARSTAR]  {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")


# ─────────────────────────────────────────────────────────────────────────────
# PERIOD FINDER PANEL
# ─────────────────────────────────────────────────────────────────────────────

class PeriodPanel(QWidget):
    """
    Full Periodicity Analysis UI.
    In pipeline mode: receives LC dict from VarStarPanel.
    Signals: finished(dict), log_line(str)
    """
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._times   = None
        self._fluxes  = None
        self._errors  = None
        self._worker  = None
        self._cancel  = threading.Event()
        self._last_result = {}
        self._show_aliases = True
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)
        root.addWidget(splitter, 1)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([340, 1040])

        bottom = QWidget(); bottom.setFixedHeight(44)
        bottom.setStyleSheet(f"background:{SIRIL_BG3}; border-top:1px solid {SIRIL_BORDER};")
        b_lay = QHBoxLayout(bottom); b_lay.setContentsMargins(12, 6, 12, 6)
        self.prog_bar = QProgressBar(); self.prog_bar.setRange(0, 5)
        self.prog_bar.setFixedHeight(8); self.prog_bar.hide()
        b_lay.addWidget(self.prog_bar, 1)
        self.btn_analyze = QPushButton("▶  Analyze"); self.btn_analyze.setObjectName("primary")
        self.btn_analyze.setFixedWidth(120); self.btn_analyze.clicked.connect(self._run_analysis)
        self.btn_cancel  = QPushButton("✕  Cancel"); self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setFixedWidth(100); self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_analysis)
        b_lay.addWidget(self.btn_analyze); b_lay.addWidget(self.btn_cancel)
        root.addWidget(bottom)

        self.status_lbl = QLabel("Ready — load data or receive from Variable Star pipeline")
        self.status_lbl.setStyleSheet(
            f"background:{SIRIL_BG2}; color:{SIRIL_TEXT_DIM}; "
            f"padding:3px 12px; font-size:9pt; border-top:1px solid {SIRIL_BORDER};")
        self.status_lbl.setFixedHeight(24)
        root.addWidget(self.status_lbl)

    def _build_left_panel(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget(); lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        # Pipeline handoff display
        self._grp_piped = QGroupBox("Light curve from Variable Star pipeline")
        self._grp_piped.setVisible(False)
        pl = QVBoxLayout(self._grp_piped)
        self._lbl_piped = QLabel("—"); self._lbl_piped.setWordWrap(True)
        pl.addWidget(self._lbl_piped)
        lay.addWidget(self._grp_piped)

        # Data input
        grp_input = QGroupBox("Data Input"); g_lay = QVBoxLayout(grp_input)
        src_lay = QHBoxLayout(); src_lay.addWidget(QLabel("Source:"))
        self.cmb_source = QComboBox()
        self.cmb_source.addItems(["CSV file", "FITS sequence", "Paste manually"])
        self.cmb_source.currentIndexChanged.connect(self._on_source_changed)
        src_lay.addWidget(self.cmb_source, 1); g_lay.addLayout(src_lay)
        self.stk_source = QStackedWidget()

        # CSV page
        csv_page = QWidget(); cl = QVBoxLayout(csv_page); cl.setContentsMargins(0,0,0,0)
        csv_row = QHBoxLayout()
        self.le_csv = QLineEdit(); self.le_csv.setPlaceholderText("Path to CSV…")
        btn_bc = QPushButton("…"); btn_bc.setFixedWidth(30); btn_bc.clicked.connect(self._browse_csv)
        csv_row.addWidget(self.le_csv, 1); csv_row.addWidget(btn_bc); cl.addLayout(csv_row)
        btn_load_csv = QPushButton("Load CSV"); btn_load_csv.clicked.connect(self._load_csv)
        cl.addWidget(btn_load_csv)
        self.lbl_csv_info = QLabel(""); self.lbl_csv_info.setObjectName("dim"); self.lbl_csv_info.setWordWrap(True)
        cl.addWidget(self.lbl_csv_info); self.stk_source.addWidget(csv_page)

        # FITS page
        fits_page = QWidget(); fl = QVBoxLayout(fits_page); fl.setContentsMargins(0,0,0,0)
        fits_row = QHBoxLayout()
        self.le_fits = QLineEdit(); self.le_fits.setPlaceholderText("FITS folder…")
        btn_bf = QPushButton("…"); btn_bf.setFixedWidth(30); btn_bf.clicked.connect(self._browse_fits)
        fits_row.addWidget(self.le_fits, 1); fits_row.addWidget(btn_bf); fl.addLayout(fits_row)
        fl.addWidget(QLabel("Target star position (pixels):"))
        xy_row = QHBoxLayout()
        xy_row.addWidget(QLabel("X:")); self.spn_star_x = QDoubleSpinBox(); self.spn_star_x.setRange(0, 99999); xy_row.addWidget(self.spn_star_x)
        xy_row.addWidget(QLabel("Y:")); self.spn_star_y = QDoubleSpinBox(); self.spn_star_y.setRange(0, 99999); xy_row.addWidget(self.spn_star_y)
        fl.addLayout(xy_row)
        aper_row = QFormLayout()
        self.spn_aperture = QDoubleSpinBox(); self.spn_aperture.setRange(1,100); self.spn_aperture.setValue(8.0)
        aper_row.addRow("Aperture (px):", self.spn_aperture); fl.addLayout(aper_row)
        btn_phot = QPushButton("Run aperture photometry"); btn_phot.clicked.connect(self._run_photometry)
        fl.addWidget(btn_phot); self.stk_source.addWidget(fits_page)

        # Manual page
        manual_page = QWidget(); ml = QVBoxLayout(manual_page); ml.setContentsMargins(0,0,0,0)
        ml.addWidget(QLabel("One point per line: time flux [error]"))
        self.txt_manual = QPlainTextEdit(); self.txt_manual.setFixedHeight(120)
        ml.addWidget(self.txt_manual)
        btn_parse = QPushButton("Parse data"); btn_parse.clicked.connect(self._parse_manual)
        ml.addWidget(btn_parse); self.stk_source.addWidget(manual_page)

        g_lay.addWidget(self.stk_source); lay.addWidget(grp_input)

        # Normalization
        grp_norm = QGroupBox("Normalization"); n_lay = QVBoxLayout(grp_norm)
        self.cmb_norm = QComboBox(); self.cmb_norm.addItems(["Median (robust)","Mean","Min-Max","None"])
        n_lay.addWidget(self.cmb_norm); lay.addWidget(grp_norm)

        # Period range
        grp_range = QGroupBox("Period Search Range"); r_lay = QFormLayout(grp_range)
        self.spn_min_period = QDoubleSpinBox(); self.spn_min_period.setRange(0,1e9); self.spn_min_period.setDecimals(6); self.spn_min_period.setValue(0)
        self.spn_max_period = QDoubleSpinBox(); self.spn_max_period.setRange(0,1e9); self.spn_max_period.setDecimals(6); self.spn_max_period.setValue(0)
        r_lay.addRow("Min period:", self.spn_min_period); r_lay.addRow("Max period:", self.spn_max_period)
        r_lay.addRow(QLabel("0 = auto from data baseline")); lay.addWidget(grp_range)

        # Methods
        grp_meth = QGroupBox("Methods"); m_lay = QVBoxLayout(grp_meth)
        self.chk_ls      = QCheckBox("Lomb-Scargle");       self.chk_ls.setChecked(True)
        self.chk_acf     = QCheckBox("Autocorrelation (ACF)"); self.chk_acf.setChecked(True)
        self.chk_wavelet = QCheckBox("Wavelet (Morlet)");    self.chk_wavelet.setChecked(True)
        for chk in [self.chk_ls, self.chk_acf, self.chk_wavelet]: m_lay.addWidget(chk)
        tol_row = QFormLayout()
        self.spn_tolerance = QDoubleSpinBox(); self.spn_tolerance.setRange(0.01, 0.2); self.spn_tolerance.setValue(0.05); self.spn_tolerance.setSuffix(" frac")
        tol_row.addRow("CV tolerance:", self.spn_tolerance); m_lay.addLayout(tol_row)
        self.chk_aliases = QCheckBox("Show 1-day / 1-year alias lines"); self.chk_aliases.setChecked(True)
        self.chk_aliases.stateChanged.connect(lambda s: setattr(self, "_show_aliases", bool(s)))
        m_lay.addWidget(self.chk_aliases); lay.addWidget(grp_meth)

        # Output
        grp_out = QGroupBox("Output"); o_lay = QVBoxLayout(grp_out)
        self.chk_save_csv = QCheckBox("Save results CSV"); self.chk_save_csv.setChecked(True)
        self.chk_save_png = QCheckBox("Save plots as PNG")
        for chk in [self.chk_save_csv, self.chk_save_png]: o_lay.addWidget(chk)
        out_row = QHBoxLayout()
        self.le_out_dir = QLineEdit(); self.le_out_dir.setPlaceholderText("Output folder")
        btn_bout = QPushButton("…"); btn_bout.setFixedWidth(30)
        btn_bout.clicked.connect(lambda: self.le_out_dir.setText(
            QFileDialog.getExistingDirectory(self, "Output folder") or self.le_out_dir.text()))
        out_row.addWidget(self.le_out_dir, 1); out_row.addWidget(btn_bout); o_lay.addLayout(out_row)
        lay.addWidget(grp_out)
        lay.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _build_right_panel(self):
        outer = QWidget(); vl = QVBoxLayout(outer)
        vl.setContentsMargins(2, 0, 2, 0); vl.setSpacing(0)
        self._rtabs = QTabWidget()

        # LC tab
        lc_tab = QWidget(); lc_lay = QVBoxLayout(lc_tab)
        self._lc_canvas = LightCurveCanvas(); lc_lay.addWidget(self._lc_canvas)
        self._rtabs.addTab(lc_tab, "📈  Light Curve")

        # Periodogram tab
        pg_tab = QWidget(); pg_lay = QVBoxLayout(pg_tab)
        self._periodogram_canvas = PeriodogramCanvas()
        pg_lay.addWidget(self._periodogram_canvas)
        self._rtabs.addTab(pg_tab, "〜  Periodogram")

        # Wavelet tab
        wav_tab = QWidget(); wav_lay = QVBoxLayout(wav_tab)
        self._wavelet_canvas = WaveletCanvas(); wav_lay.addWidget(self._wavelet_canvas)
        self._rtabs.addTab(wav_tab, "🌊  Wavelet")

        # Results table tab
        res_tab = QWidget(); res_lay = QVBoxLayout(res_tab)
        self._results_table = QTableWidget(0, 7)
        self._results_table.setHorizontalHeaderLabels(
            ["Rank","Period","LS Power","FAP","ACF","Wavelet","Notes"])
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        res_lay.addWidget(self._results_table)
        self._rtabs.addTab(res_tab, "📊  Results")

        # Consensus tab
        con_tab = QWidget(); con_lay = QVBoxLayout(con_tab)
        self._lbl_best_period = QLabel("Best period: —")
        self._lbl_best_period.setStyleSheet(f"color:{SIRIL_NOVA}; font-size:16pt; font-weight:bold;")
        self._lbl_confidence = QLabel("Confidence: —")
        self._lbl_confidence.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:11pt;")
        self._lbl_agreements = QLabel("Methods: —"); self._lbl_agreements.setObjectName("dim")
        self._lbl_harmonics  = QLabel("Harmonics: —"); self._lbl_harmonics.setObjectName("dim")
        for w in [self._lbl_best_period, self._lbl_confidence,
                  self._lbl_agreements, self._lbl_harmonics]:
            con_lay.addWidget(w)
        con_lay.addStretch()
        self._rtabs.addTab(con_tab, "✓  Consensus")

        # Log tab
        log_tab = QWidget(); ll = QVBoxLayout(log_tab)
        self._log_view = QPlainTextEdit(); self._log_view.setReadOnly(True)
        ll.addWidget(self._log_view)
        self._rtabs.addTab(log_tab, "📋  Log")

        vl.addWidget(self._rtabs)
        return outer

    # ── Public interface ──────────────────────────────────────────────────────

    def set_light_curve(self, lc: dict):
        """Pipeline handoff from VarStarPanel (Section A4)."""
        t = lc.get("times", np.array([]))
        m = lc.get("diff_mags", np.array([]))
        e = lc.get("errors", None)
        if len(t) < 3: return
        self._times  = t
        self._fluxes = m
        self._errors = e
        self._lc_canvas.show_lightcurve(t, m, e)
        self._grp_piped.setVisible(True)
        self._lbl_piped.setText(
            f"✓ Received {len(t)} data points from Variable Star pipeline\n"
            f"  Baseline: {t.max()-t.min():.3f}  "
            f"Amplitude: {np.percentile(m,95)-np.percentile(m,5):.3f} mag")
        self._lbl_piped.setStyleSheet(f"color:{SIRIL_SUCCESS};")
        self.status_lbl.setText(
            f"Light curve received: {len(t)} points — click Analyze")
        self.log_line.emit(f"[PERIOD]  Received LC: {len(t)} points from Variable Star panel")

    def cancel(self):
        self._cancel.set()

    def run_pipeline(self, cancel_event: threading.Event):
        self._cancel = cancel_event
        if self._times is not None and len(self._times) > 2:
            self._run_analysis()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _on_source_changed(self, idx: int):
        self.stk_source.setCurrentIndex(idx)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "",
            "CSV (*.csv *.txt);;All (*)");
        if path: self.le_csv.setText(path)

    def _browse_fits(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d: self.le_fits.setText(d)

    def _load_csv(self):
        path = self.le_csv.text().strip()
        if not path or not os.path.isfile(path): return
        try:
            t, f, e = load_csv_lightcurve(path)
            self._times = t; self._fluxes = f; self._errors = e
            self._lc_canvas.show_lightcurve(t, f, e)
            self.lbl_csv_info.setText(f"{len(t)} points  baseline={t.max()-t.min():.4f}")
        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))

    def _run_photometry(self):
        folder = self.le_fits.text().strip()
        if not folder or not os.path.isdir(folder): return
        x = self.spn_star_x.value(); y = self.spn_star_y.value()
        t, f, e = compute_from_fits_sequence(
            folder, x, y, self.spn_aperture.value(),
            log_callback=lambda m: self.log_line.emit(f"[PERIOD]  {m}"))
        if len(t) < 3:
            QMessageBox.warning(self, "Too few points", "Photometry returned < 3 data points.")
            return
        self._times = t; self._fluxes = f; self._errors = e
        self._lc_canvas.show_lightcurve(t, f, e)
        self.status_lbl.setText(f"Photometry complete: {len(t)} points")

    def _parse_manual(self):
        text = self.txt_manual.toPlainText()
        t, f, e = parse_manual_data(text)
        if len(t) < 3:
            QMessageBox.warning(self, "Too few points", "Need at least 3 data points."); return
        self._times = t; self._fluxes = f; self._errors = e
        self._lc_canvas.show_lightcurve(t, f, e)
        self.status_lbl.setText(f"Parsed {len(t)} points")

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if self._times is None or len(self._times) < 3:
            QMessageBox.warning(self, "No data",
                "Load a light curve first."); return
        norm_map = {"Median (robust)": "median", "Mean": "mean",
                    "Min-Max": "minmax", "None": None}
        norm = norm_map.get(self.cmb_norm.currentText())
        if norm:
            t, f, e = normalize_lightcurve(self._times, self._fluxes, self._errors, method=norm)
        else:
            t, f, e = self._times, self._fluxes, self._errors
        cfg = {
            "run_ls":      self.chk_ls.isChecked(),
            "run_acf":     self.chk_acf.isChecked(),
            "run_wavelet": self.chk_wavelet.isChecked(),
            "min_period":  self.spn_min_period.value() or None,
            "max_period":  self.spn_max_period.value() or None,
            "tolerance":   self.spn_tolerance.value(),
            "n_peaks":     10,
            "phase_bins":  50,
            "wavelet_periods": 100,
        }
        self._cancel.clear()
        self.btn_analyze.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.prog_bar.show(); self.prog_bar.setValue(0)
        self._worker = PeriodWorker(cfg, t, f, e, self._cancel)
        self._worker.progress.connect(lambda s,n,m: (self.prog_bar.setValue(s),
                                                      self.status_lbl.setText(m)))
        self._worker.log_line.connect(self._log)
        self._worker.ls_ready.connect(lambda r: self._periodogram_canvas.show_lomb_scargle(
            r, show_aliases=self._show_aliases))
        self._worker.acf_ready.connect(self._periodogram_canvas.show_acf)
        self._worker.wav_ready.connect(self._wavelet_canvas.show_wavelet)
        self._worker.fold_ready.connect(self._lc_canvas.show_phase_fold)
        self._worker.consensus.connect(self._on_consensus)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_analysis(self):
        self._cancel.set()
        self.btn_cancel.setEnabled(False)

    def _on_consensus(self, cv: dict):
        p    = cv.get("best_period", 0)
        conf = cv.get("confidence", "none")
        agree = ", ".join(f"{m1}+{m2}" for m1, m2 in cv.get("agreements", []))
        self._lbl_best_period.setText(f"Best period: {p:.8f}")
        col = SIRIL_SUCCESS if conf == "confirmed" else (SIRIL_WARNING if conf == "candidate" else SIRIL_ERROR)
        self._lbl_confidence.setText(f"Confidence: {conf}")
        self._lbl_confidence.setStyleSheet(f"color:{col}; font-size:11pt;")
        self._lbl_agreements.setText(f"Agreements: {agree or 'none'}")
        self._rtabs.setCurrentIndex(4)

    def _on_finished(self, result: dict):
        self.btn_analyze.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.prog_bar.hide()
        if not result.get("success"):
            self.status_lbl.setText(f"✗ {result.get('error','')}")
            self.finished.emit(result); return

        p         = result.get("best_period", 0)
        conf      = result.get("confidence", "none")
        harmonics = result.get("harmonics", [])
        top_peaks = result.get("top_peaks", [])
        self._last_result = result

        # Populate results table
        self._results_table.setRowCount(0)
        acf_p = (result.get("acf_result") or {}).get("best_period", 0)
        wav_p = (result.get("wav_result") or {}).get("best_period", 0)
        tol   = self.spn_tolerance.value()
        h_set = {round(h["period"], 6) for h in harmonics}
        for rank, peak in enumerate(top_peaks, 1):
            pk = peak["period"]
            acf_c = "✓" if acf_p > 0 and abs(pk - acf_p) / max(pk, 1e-10) < tol else "—"
            wav_c = "✓" if wav_p > 0 and abs(pk - wav_p) / max(pk, 1e-10) < tol else "—"
            notes = "CONFIRMED" if (acf_c == "✓" or wav_c == "✓") else "candidate"
            if round(pk, 6) in h_set: notes += " (harmonic?)"
            fap = peak.get("fap", float("nan"))
            row = self._results_table.rowCount()
            self._results_table.insertRow(row)
            for col, val in enumerate([
                str(rank), f"{pk:.8f}",
                f"{peak['power']:.6f}",
                f"{fap:.2e}" if not (isinstance(fap, float) and np.isnan(fap)) else "N/A",
                acf_c, wav_c, notes
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._results_table.setItem(row, col, item)

        self._lbl_harmonics.setText(
            f"Harmonics detected: {', '.join(str(h['label']) for h in harmonics) or 'none'}")

        if self.chk_save_csv.isChecked():
            out_dir = self.le_out_dir.text().strip() or os.path.expanduser("~/Desktop")
            try:
                ls_r = result.get("ls_result"); acf_r = result.get("acf_result")
                t = self._times if self._times is not None else np.array([])
                f = self._fluxes if self._fluxes is not None else np.array([])
                csv_path = save_results_csv(out_dir, result, top_peaks, t, f,
                                             ls_result=ls_r, acf_result=acf_r,
                                             harmonics=harmonics)
                self._log(f"Results CSV: {csv_path}")
            except Exception as ex:
                self._log(f"CSV save failed: {ex}")

        self.status_lbl.setText(f"✓  Best period: {p:.8f}  |  confidence: {conf}")
        self._log(f"Analysis complete: P={p:.8f}  conf={conf}")
        self.log_line.emit(f"[PERIOD]  ✓ P={p:.8f}  conf={conf}")
        self.finished.emit(result)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{ts}] {msg}")
        self.log_line.emit(f"[PERIOD]  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# EXTINCTION MAP PANEL
# ─────────────────────────────────────────────────────────────────────────────

class ExtinctionCanvas(FigureCanvasQTAgg):
    """Bouguer's Law fit + transparency timeline."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4), facecolor=SIRIL_BG)
        self.ax_bouguer = self.fig.add_subplot(1, 2, 1)
        self.ax_trans   = self.fig.add_subplot(1, 2, 2)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style_axes(self):
        for ax, title in [(self.ax_bouguer, "Bouguer's Law fit"),
                          (self.ax_trans,   "Transparency timeline")]:
            ax.set_facecolor(SIRIL_BG2)
            ax.tick_params(colors=SIRIL_TEXT, labelsize=8)
            ax.set_title(title, color=SIRIL_SECTION, fontsize=9)
            for sp in ["bottom","left"]:  ax.spines[sp].set_color(SIRIL_BORDER)
            for sp in ["top","right"]:    ax.spines[sp].set_visible(False)

    def show_bouguer(self, airmasses, mags, k, m_true, quality: str):
        self.ax_bouguer.clear(); self.ax_bouguer.set_facecolor(SIRIL_BG2)
        X = np.array(airmasses); m = np.array(mags)
        color = {
            "photometric": SIRIL_SUCCESS,
            "good":        SIRIL_ACCENT,
            "variable":    SIRIL_WARNING,
            "poor":        SIRIL_ERROR,
        }.get(quality, SIRIL_TEXT_DIM)
        self.ax_bouguer.scatter(X, m, color=SIRIL_ACCENT, s=20, alpha=0.8, zorder=3)
        if k != 0:
            x_fit = np.linspace(X.min(), X.max(), 100)
            self.ax_bouguer.plot(x_fit, m_true + k * x_fit,
                                  color=color, linewidth=2, zorder=4,
                                  label=f"k={k:.4f} ({quality})")
            self.ax_bouguer.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                                    labelcolor=SIRIL_TEXT, fontsize=8)
        self.ax_bouguer.set_xlabel("Airmass X", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_bouguer.set_ylabel("Instrumental mag", color=SIRIL_TEXT_DIM, fontsize=8)
        for sp in ["bottom","left"]:  self.ax_bouguer.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top","right"]:    self.ax_bouguer.spines[sp].set_visible(False)
        self.ax_bouguer.tick_params(colors=SIRIL_TEXT, labelsize=8)
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def show_transparency(self, jds, transparency):
        self.ax_trans.clear(); self.ax_trans.set_facecolor(SIRIL_BG2)
        t = np.array(jds); tr = np.array(transparency)
        self.ax_trans.plot(t, tr, "-o", color=SIRIL_ACCENT, markersize=4, linewidth=1.2)
        self.ax_trans.axhline(85, color=SIRIL_SUCCESS, linewidth=0.8, linestyle="--",
                               alpha=0.7, label="Photometric (≥85)")
        self.ax_trans.axhline(70, color=SIRIL_WARNING, linewidth=0.8, linestyle="--",
                               alpha=0.7, label="Good (≥70)")
        self.ax_trans.set_ylim(0, 105)
        self.ax_trans.set_xlabel("JD", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_trans.set_ylabel("Transparency score", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_trans.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                              labelcolor=SIRIL_TEXT, fontsize=7)
        for sp in ["bottom","left"]:  self.ax_trans.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top","right"]:    self.ax_trans.spines[sp].set_visible(False)
        self.ax_trans.tick_params(colors=SIRIL_TEXT, labelsize=8)
        self.fig.tight_layout(pad=0.4)
        self.draw()


class ExtinctionPanel(QWidget):
    """
    Extinction Map (Bouguer's Law) UI.
    In pipeline mode: receives fits_files list from VarStarPanel.
    Signals: finished(dict), log_line(str)
    """
    finished = pyqtSignal(dict)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._fits_files   = []
        self._airmasses    = []
        self._mags         = []
        self._jds          = []
        self._transparency = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._build_tab_settings()
        self._build_tab_results()
        self._build_tab_log()

        bot = QHBoxLayout()
        self._btn_run = QPushButton("▶  Measure extinction")
        self._btn_run.setObjectName("primary"); self._btn_run.clicked.connect(self._run)
        bot.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel"); self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setEnabled(False); self._btn_cancel.clicked.connect(self._cancel)
        bot.addWidget(self._btn_cancel)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        self._progress.setFixedHeight(8); bot.addWidget(self._progress, 1)
        root.addLayout(bot)

        self._status_lbl = QLabel("Ready — configure site & load FITS sequence")
        self._status_lbl.setObjectName("dim"); root.addWidget(self._status_lbl)

    def _build_tab_settings(self):
        tab = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); il = QVBoxLayout(inner); il.setSpacing(8); il.setContentsMargins(6,6,6,6)
        scroll.setWidget(inner); lay = QVBoxLayout(tab); lay.addWidget(scroll)

        # Pipeline handoff info
        self._grp_piped = QGroupBox("FITS sequence from Variable Star pipeline")
        self._grp_piped.setVisible(False)
        pl = QVBoxLayout(self._grp_piped)
        self._lbl_piped = QLabel("—"); self._lbl_piped.setWordWrap(True)
        pl.addWidget(self._lbl_piped); il.addWidget(self._grp_piped)

        # Mode
        grp_mode = QGroupBox("Analysis mode")
        mf = QFormLayout(grp_mode)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Time-series (sequence of frames)",
                                    "Single-frame (WCS stars at different altitudes)"])
        mf.addRow("Mode:", self._mode_combo); il.addWidget(grp_mode)

        # Input
        grp_in = QGroupBox("Input FITS")
        gi = QVBoxLayout(grp_in)
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit(); self._folder_edit.setPlaceholderText("FITS folder…")
        btn_f = QPushButton("Browse…"); btn_f.setFixedWidth(70)
        btn_f.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit); folder_row.addWidget(btn_f)
        gi.addLayout(folder_row)
        btn_scan = QPushButton("🔍 Scan folder"); btn_scan.clicked.connect(self._scan)
        gi.addWidget(btn_scan)
        self._lbl_scan = QLabel("No files loaded"); self._lbl_scan.setObjectName("dim")
        gi.addWidget(self._lbl_scan); il.addWidget(grp_in)

        # Site
        grp_site = QGroupBox("Observer location")
        sf = QFormLayout(grp_site)
        self._lat_spin  = QDoubleSpinBox(); self._lat_spin.setRange(-90, 90); self._lat_spin.setDecimals(4); self._lat_spin.setValue(48.0)
        self._lon_spin  = QDoubleSpinBox(); self._lon_spin.setRange(-180,180); self._lon_spin.setDecimals(4); self._lon_spin.setValue(16.0)
        self._elev_spin = QDoubleSpinBox(); self._elev_spin.setRange(0, 5000); self._elev_spin.setValue(200); self._elev_spin.setSuffix(" m")
        sf.addRow("Latitude (°):",  self._lat_spin)
        sf.addRow("Longitude (°):", self._lon_spin)
        sf.addRow("Elevation:",     self._elev_spin)
        lbl_note = QLabel("Used to compute altitude if not in FITS header")
        lbl_note.setObjectName("dim"); lbl_note.setWordWrap(True); sf.addRow(lbl_note)
        il.addWidget(grp_site)

        # Target
        grp_tgt = QGroupBox("Target RA/Dec (optional — for altitude)")
        tf = QFormLayout(grp_tgt)
        self._ra_edit  = QLineEdit(); self._ra_edit.setPlaceholderText("RA in degrees, e.g. 83.82")
        self._dec_edit = QLineEdit(); self._dec_edit.setPlaceholderText("Dec in degrees, e.g. -5.39")
        tf.addRow("RA:", self._ra_edit); tf.addRow("Dec:", self._dec_edit)
        il.addWidget(grp_tgt)

        # Detection params
        grp_det = QGroupBox("Star detection")
        df = QFormLayout(grp_det)
        self._n_stars_spin = QSpinBox(); self._n_stars_spin.setRange(3, 100); self._n_stars_spin.setValue(20)
        self._thresh_spin  = QDoubleSpinBox(); self._thresh_spin.setRange(3.0, 30.0); self._thresh_spin.setValue(10.0); self._thresh_spin.setSuffix(" σ")
        self._aper_spin    = QDoubleSpinBox(); self._aper_spin.setRange(3, 30); self._aper_spin.setValue(8.0); self._aper_spin.setSuffix(" px")
        df.addRow("Max stars:", self._n_stars_spin)
        df.addRow("Detection threshold:", self._thresh_spin)
        df.addRow("Aperture radius:", self._aper_spin)
        il.addWidget(grp_det)

        # Output
        grp_out = QGroupBox("Output")
        of = QVBoxLayout(grp_out)
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit(); self._out_edit.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(70)
        btn_out.clicked.connect(lambda: self._out_edit.setText(
            QFileDialog.getExistingDirectory(self, "Output folder") or self._out_edit.text()))
        out_row.addWidget(self._out_edit); out_row.addWidget(btn_out); of.addLayout(out_row)
        self._chk_apply_correction = QCheckBox(
            "Pass k-coefficient to Variable Star pipeline for photometric correction")
        self._chk_apply_correction.setChecked(True)
        of.addWidget(self._chk_apply_correction)
        il.addWidget(grp_out)
        il.addStretch()
        self._tabs.addTab(tab, "⚙  Settings")

    def _build_tab_results(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self._ext_canvas = ExtinctionCanvas()
        self._ext_canvas.setMinimumHeight(280)
        lay.addWidget(self._ext_canvas, 1)
        self._result_grp = QGroupBox("Fit results")
        rl = QFormLayout(self._result_grp)
        self._lbl_k       = QLabel("k = —")
        self._lbl_r2      = QLabel("R² = —")
        self._lbl_quality = QLabel("Quality: —")
        self._lbl_n_pts   = QLabel("Points: —")
        self._lbl_interp  = QLabel("Interpretation: —"); self._lbl_interp.setWordWrap(True)
        for label, widget in [("Extinction coeff.:", self._lbl_k), ("R²:", self._lbl_r2),
                               ("Quality:", self._lbl_quality), ("N frames:", self._lbl_n_pts),
                               ("Interpretation:", self._lbl_interp)]:
            rl.addRow(label, widget)
        lay.addWidget(self._result_grp)
        self._tabs.addTab(tab, "📊  Results")

    def _build_tab_log(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self._log_view = QPlainTextEdit(); self._log_view.setReadOnly(True)
        lay.addWidget(self._log_view)
        self._tabs.addTab(tab, "📋  Log")

    # ── Public interface ──────────────────────────────────────────────────────

    def set_fits_files(self, fits_files: list):
        """Pipeline handoff from VarStarPanel."""
        if not fits_files: return
        self._fits_files = fits_files
        self._folder_edit.setText(os.path.dirname(fits_files[0]))
        self._lbl_scan.setText(f"{len(fits_files)} files (from Variable Star pipeline)")
        self._lbl_scan.setStyleSheet(f"color:{SIRIL_SUCCESS};")
        self._grp_piped.setVisible(True)
        self._lbl_piped.setText(
            f"✓ Received {len(fits_files)} FITS files from Variable Star pipeline\n"
            f"  First: {os.path.basename(fits_files[0])}")
        self._lbl_piped.setStyleSheet(f"color:{SIRIL_SUCCESS};")
        self.log_line.emit(f"[EXTINCTION]  Received {len(fits_files)} FITS files from pipeline")

    def get_k_coefficient(self) -> float:
        """Return k from last run, for photometric correction handoff."""
        return float(self._lbl_k.text().split("=")[-1].strip().split("±")[0].strip()
                     if "=" in self._lbl_k.text() else "0")

    def cancel(self):
        self._cancel_event.set()

    # ── Browse / scan ─────────────────────────────────────────────────────────

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d: self._folder_edit.setText(d)

    def _scan(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Select a FITS folder."); return
        files = sorted(
            glob.glob(os.path.join(folder, "*.fit")) +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts")))
        self._fits_files = files
        self._lbl_scan.setText(f"{len(files)} FITS files found")
        if not self._out_edit.text():
            self._out_edit.setText(folder)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{ts}] {msg}")
        self.log_line.emit(f"[EXTINCTION]  {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")

    def _run(self):
        if not self._fits_files:
            QMessageBox.warning(self, "No files",
                "Scan a FITS folder or run the pipeline from step 1."); return
        out_dir = self._out_edit.text().strip() or os.path.dirname(self._fits_files[0])
        try:
            ra_val  = float(self._ra_edit.text())  if self._ra_edit.text().strip()  else None
            dec_val = float(self._dec_edit.text()) if self._dec_edit.text().strip() else None
        except ValueError:
            ra_val = dec_val = None

        mode = "single_frame" if self._mode_combo.currentIndex() == 1 else "time_series"
        cfg = {
            "mode":             mode,
            "fits_files":       self._fits_files,
            "output_dir":       out_dir,
            "n_stars":          self._n_stars_spin.value(),
            "threshold_sigma":  self._thresh_spin.value(),
            "aperture_r":       self._aper_spin.value(),
            "annulus_r_in":     self._aper_spin.value() * 1.5,
            "annulus_r_out":    self._aper_spin.value() * 2.25,
            "target_ra":        ra_val,
            "target_dec":       dec_val,
            "lat_deg":          self._lat_spin.value(),
            "lon_deg":          self._lon_spin.value(),
            "elev_m":           self._elev_spin.value(),
        }
        if mode == "single_frame":
            cfg["fits_path"] = self._fits_files[0]

        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True); self._airmasses = []; self._mags = []
        self._jds = []; self._transparency = []

        self._worker = ExtinctionWorker(cfg, self._cancel_event)
        self._worker.progress.connect(lambda c, t, m: (
            self._progress.setRange(0, max(t,1)), self._progress.setValue(c),
            self._set_status(m, SIRIL_ACCENT)))
        self._worker.log_line.connect(self._log)
        self._worker.frame_done.connect(self._on_frame_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        self.cancel(); self._btn_cancel.setEnabled(False)

    def _on_frame_done(self, frame_data: dict):
        self._airmasses.append(frame_data.get("airmass", 0))
        self._mags.append(frame_data.get("mag", 0))
        self._jds.append(frame_data.get("jd", 0))

    def run_pipeline(self, cancel_event: threading.Event):
        self._cancel_event = cancel_event
        if self._fits_files:
            self._run()

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        if not result.get("success"):
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)
            self.finished.emit(result); return

        k     = result.get("k", 0)
        k_err = result.get("k_err", 0)
        r2    = result.get("r2", 0)
        qual  = result.get("quality", "—")
        n     = result.get("n_frames", 0)

        self._lbl_k.setText(f"k = {k:.4f} ± {k_err:.4f} mag/airmass")
        self._lbl_r2.setText(f"R² = {r2:.4f}")
        col = {
            "photometric": SIRIL_SUCCESS, "good": SIRIL_ACCENT,
            "variable":    SIRIL_WARNING, "poor": SIRIL_ERROR,
        }.get(qual, SIRIL_TEXT_DIM)
        self._lbl_quality.setText(qual)
        self._lbl_quality.setStyleSheet(f"color:{col}; font-weight:bold;")
        self._lbl_n_pts.setText(str(n))

        if qual == "photometric":
            interp = f"✓ Photometric night — excellent conditions (k={k:.4f})"
        elif qual == "good":
            interp = f"Good conditions — minor extinction variation (k={k:.4f})"
        elif qual == "variable":
            interp = "⚠ Variable transparency — mild cirrus or dust"
        else:
            interp = "✗ Poor fit — likely significant cloud cover or high aerosol"
        self._lbl_interp.setText(interp)

        if self._airmasses:
            m_true = result.get("m_true_est", 0)
            self._ext_canvas.show_bouguer(self._airmasses, self._mags, k, m_true, qual)
            if result.get("transparency"):
                self._transparency = result["transparency"]
                self._ext_canvas.show_transparency(self._jds, self._transparency)
        self._tabs.setCurrentIndex(1)
        self._set_status(f"✓ k={k:.4f} ± {k_err:.4f}  R²={r2:.4f}  {qual}", SIRIL_SUCCESS)
        self._log(f"Extinction complete: k={k:.4f}  R²={r2:.4f}  quality={qual}")
        self.log_line.emit(f"[EXTINCTION]  ✓ k={k:.4f}  R²={r2:.4f}  quality={qual}")
        self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW TAB
# ─────────────────────────────────────────────────────────────────────────────

class OverviewTab(QWidget):
    run_pipeline_requested = pyqtSignal()
    run_step_requested     = pyqtSignal(str)
    cancel_requested       = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10); root.setSpacing(10)

        diag_grp = QGroupBox("Pipeline")
        dl = QVBoxLayout(diag_grp)
        self._pipeline_widget = PipelineStatusWidget([
            {"name": "Variable Star\n(photometry + classify)", "key": "varstar"},
            {"name": "Period Finder\n(LS + ACF + Wavelet)",    "key": "period"},
            {"name": "Extinction Map\n(Bouguer's Law)",        "key": "extinction"},
        ])
        self._pipeline_widget.setMinimumHeight(90)
        dl.addWidget(self._pipeline_widget)
        step_row = QHBoxLayout()
        for key, label in [("varstar",    "▶ Variable Star only"),
                            ("period",    "▶ Period only"),
                            ("extinction","▶ Extinction only")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, k=key: self.run_step_requested.emit(k))
            step_row.addWidget(btn)
        step_row.addStretch()
        dl.addLayout(step_row)
        root.addWidget(diag_grp)

        self._result_grp = QGroupBox("Last result")
        self._result_grp.setVisible(False)
        rl = QVBoxLayout(self._result_grp)
        self._result_lbl = QLabel(""); self._result_lbl.setWordWrap(True)
        rl.addWidget(self._result_lbl)
        root.addWidget(self._result_grp)

        root.addStretch()

        action_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run full pipeline  (VarStar → Period → Extinction)")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(36); self._btn_run.setMinimumWidth(300)
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

    def show_result(self, step: str, result: dict):
        self._result_grp.setVisible(True)
        if step == "varstar":
            p   = result.get("best_period", 0)
            vt  = result.get("var_type", "?")
            con = result.get("confidence", 0)
            self._result_lbl.setText(
                f"✓ Variable Star: P={p:.6f} d  type={vt}  conf={con*100:.0f}%")
            self._result_grp.setTitle("Last result — ⭐ Variable Star")
        elif step == "period":
            p    = result.get("best_period", 0)
            conf = result.get("confidence", "—")
            self._result_lbl.setText(
                f"✓ Period Analysis: P={p:.8f}  confidence={conf}")
            self._result_grp.setTitle("Last result — 〜 Period Finder")
        elif step == "extinction":
            k    = result.get("k", 0)
            qual = result.get("quality", "—")
            self._result_lbl.setText(
                f"✓ Extinction: k={k:.4f} mag/airmass  quality={qual}")
            self._result_grp.setTitle("Last result — 🌫 Extinction Map")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Variable Star Suite  —  Siril")
        self.resize(1440, 900)
        self._cancel_event = threading.Event()
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # Header
        header = QWidget(); header.setFixedHeight(52)
        header.setStyleSheet(f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon  = QLabel("⭐"); lbl_icon.setStyleSheet("font-size:18pt; background:transparent;")
        lbl_title = QLabel("Variable Star Suite")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold; background:transparent;")
        lbl_ver = QLabel("v1.0  |  3 scripts  |  Photometry → Period Analysis → Extinction Map  |  AAVSO ready")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title); hl.addStretch(); hl.addWidget(lbl_ver)
        root.addWidget(header)

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._pipeline_log = QPlainTextEdit(); self._pipeline_log.setReadOnly(True)

        # Overview
        self._overview = OverviewTab()
        self._overview.run_pipeline_requested.connect(self._run_pipeline)
        self._overview.run_step_requested.connect(self._run_step)
        self._overview.cancel_requested.connect(self._cancel_pipeline)
        self._tabs.addTab(self._overview, "🚀  Overview")

        # Variable Star panel
        self._varstar_panel = VarStarPanel()
        self._varstar_panel.lc_ready.connect(self._on_lc_ready)
        self._varstar_panel.finished.connect(lambda r: self._on_step_done("varstar", r))
        self._varstar_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._varstar_panel, "⭐  Variable Star")

        # Period panel
        self._period_panel = PeriodPanel()
        self._period_panel.finished.connect(lambda r: self._on_step_done("period", r))
        self._period_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._period_panel, "〜  Period Finder")

        # Extinction panel
        self._extinction_panel = ExtinctionPanel()
        self._extinction_panel.finished.connect(lambda r: self._on_step_done("extinction", r))
        self._extinction_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._extinction_panel, "🌫  Extinction Map")

        # Pipeline log
        log_wrap = QWidget(); lw = QVBoxLayout(log_wrap)
        lw.setContentsMargins(6, 6, 6, 6)
        lbl_log = QLabel("Combined pipeline log"); lbl_log.setObjectName("dim")
        lw.addWidget(lbl_log)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._pipeline_log.clear)
        lw.addWidget(btn_clr); lw.addWidget(self._pipeline_log)
        self._tabs.addTab(log_wrap, "📋  Pipeline Log")

        # Bottom bar
        bottom = QWidget(); bottom.setFixedHeight(46)
        bottom.setStyleSheet(f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 4, 10, 4); bl.setSpacing(8)
        self._global_progress = QProgressBar()
        self._global_progress.setFixedHeight(8); self._global_progress.setRange(0, 3)
        bl.addWidget(self._global_progress, 1)
        self._btn_run = QPushButton("▶  Run pipeline")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(32); self._btn_run.setMinimumWidth(140)
        self._btn_run.clicked.connect(self._run_pipeline); bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(32); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_pipeline); bl.addWidget(self._btn_cancel)
        self._status_lbl = QLabel("Ready — configure Variable Star tab, then run.")
        self._status_lbl.setObjectName("dim")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status_lbl, 1)
        root.addWidget(bottom)

        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] Variable Star Suite — ready")
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "Pipeline: Variable Star → Period Finder → Extinction Map")

    # ── Pipeline orchestration (Section A6) ───────────────────────────────────

    def _run_pipeline(self):
        fits_files = self._varstar_panel.get_fits_files()
        if not fits_files:
            src_idx = self._varstar_panel._combo_src.currentIndex()
            if src_idx == 0:
                QMessageBox.warning(self, "Missing input",
                    "Load a FITS sequence in the Variable Star tab first.")
                self._tabs.setCurrentIndex(1); return

        self._cancel_event.clear()
        for key in ("varstar", "period", "extinction"):
            self._overview.set_step_status(key, "idle")
        self._set_running(True)
        self._global_progress.setValue(0)
        self._run_varstar_step()

    def _run_varstar_step(self):
        self._overview.set_step_status("varstar", "running")
        self._set_status("Running Variable Star pipeline…", SIRIL_ACCENT)
        self._log_pipeline("═══ STEP 1: Variable Star Pipeline ═══")
        self._tabs.setCurrentIndex(1)
        self._varstar_panel.run_pipeline(self._cancel_event)

    def _on_lc_ready(self, lc: dict):
        """In-memory handoff: LC dict → Period panel (Section A4)."""
        n = len(lc.get("times", []))
        self._log_pipeline(
            f"In-memory handoff: {n} LC points → Period Finder (no file I/O)")
        self._period_panel.set_light_curve(lc)

        # Also pipe FITS files to extinction panel
        fits_files = self._varstar_panel.get_fits_files()
        if fits_files:
            self._log_pipeline(
                f"In-memory handoff: {len(fits_files)} FITS paths → Extinction Map")
            self._extinction_panel.set_fits_files(fits_files)

    def _run_period_step(self):
        self._overview.set_step_status("period", "running")
        self._global_progress.setValue(1)
        self._set_status("Running period analysis…", SIRIL_ACCENT)
        self._log_pipeline("═══ STEP 2: Period Finder ═══")
        self._tabs.setCurrentIndex(2)
        self._period_panel.run_pipeline(self._cancel_event)

    def _run_extinction_step(self):
        self._overview.set_step_status("extinction", "running")
        self._global_progress.setValue(2)
        self._set_status("Running extinction measurement…", SIRIL_ACCENT)
        self._log_pipeline("═══ STEP 3: Extinction Map ═══")
        self._tabs.setCurrentIndex(3)
        self._extinction_panel.run_pipeline(self._cancel_event)

    def _on_step_done(self, step: str, result: dict):
        if result.get("success"):
            self._overview.set_step_status(step, "done")
            self._overview.show_result(step, result)
            if step == "varstar":
                self._log_pipeline(
                    f"✓ Variable Star done — P={result.get('best_period',0):.6f} d  "
                    f"type={result.get('var_type','?')}")
                self._run_period_step()
            elif step == "period":
                p = result.get("best_period", 0)
                self._log_pipeline(
                    f"✓ Period Finder done — P={p:.8f}  conf={result.get('confidence','?')}")
                self._run_extinction_step()
            elif step == "extinction":
                k = result.get("k", 0)
                self._log_pipeline(
                    f"✓ Extinction done — k={k:.4f} mag/airmass  "
                    f"quality={result.get('quality','?')}")
                self._global_progress.setValue(3)
                self._set_status(
                    f"✓ Pipeline complete — k={k:.4f} mag/airmass", SIRIL_SUCCESS)
                self._set_running(False)
        else:
            err = result.get("error", "Unknown")
            self._overview.set_step_status(step, "error")
            self._set_status(f"✗ {step} failed: {err}", SIRIL_ERROR)
            self._log_pipeline(f"✗ {step.upper()} FAILED: {err}")
            self._set_running(False)

    # ── Standalone step runners (A8) ──────────────────────────────────────────

    def _run_step(self, step: str):
        self._cancel_event.clear()
        self._set_running(True)
        if step == "varstar":
            fits_files = self._varstar_panel.get_fits_files()
            if not fits_files and self._varstar_panel._combo_src.currentIndex() == 0:
                QMessageBox.warning(self, "No files",
                    "Load a FITS sequence in the Variable Star tab."); self._set_running(False); return
            self._overview.set_step_status("varstar", "running")
            self._log_pipeline("─── Standalone: Variable Star ───")
            self._tabs.setCurrentIndex(1)
            self._varstar_panel.run_pipeline(self._cancel_event)
        elif step == "period":
            self._overview.set_step_status("period", "running")
            self._log_pipeline("─── Standalone: Period Finder ───")
            self._tabs.setCurrentIndex(2)
            self._period_panel.run_pipeline(self._cancel_event)
        elif step == "extinction":
            if not self._extinction_panel._fits_files:
                QMessageBox.warning(self, "No files",
                    "Scan a FITS folder in the Extinction Map tab."); self._set_running(False); return
            self._overview.set_step_status("extinction", "running")
            self._log_pipeline("─── Standalone: Extinction Map ───")
            self._tabs.setCurrentIndex(3)
            self._extinction_panel.run_pipeline(self._cancel_event)

    # ── Cancel (A9) ───────────────────────────────────────────────────────────

    def _cancel_pipeline(self):
        self._cancel_event.set()
        self._varstar_panel.cancel()
        self._period_panel.cancel()
        self._extinction_panel.cancel()
        for key in ("varstar", "period", "extinction"):
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
        if not running: self._global_progress.setValue(0)

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")

    def _log_pipeline(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._pipeline_log.appendPlainText(f"[{ts}] {msg}")


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
