r"""
transient_pipeline.py  —  Script Merger: B8 Transient & Variable Science Engine
================================================================================
Merged launcher for:
    1. ZOGY Image Subtraction       (zogy_subtract.py)
    2. Transient Detection          (transient_detector.py)

Architecture: Section A merger pattern
    - Wraps both scripts as importable modules
    - Tabbed launcher GUI (Overview / ZOGY / Transient Detector / Pipeline Log)
    - In-memory handoff: ZOGY result → Transient Detector (no intermediate CSV)
    - Pipeline "Run all" chains both workers sequentially
    - Each step also runnable standalone

References:
    ZOGY: Zackay, Ofek, Gal-Yam (2016), ApJ 830, 27 — arXiv:1601.02655
    Transient detection: 3-layer pipeline (frame diff / ZOGY / photometric)

Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:  Siril → Scripts → transient_pipeline
"""

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
import threading
from datetime import datetime

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# ── Child scripts location ────────────────────────────────────────────────────
# Put the 7 suite launchers in their OWN folder (e.g. "Siril Suites\").
# Put the 20+ small child scripts in a SEPARATE folder (e.g. "Siril Scripts\").
# Set CHILDREN_DIR to wherever the small scripts live.
# The launchers never have to be in the same folder as the children.
CHILDREN_DIR = r"C:\Users\Marcell\Desktop\Siril New Scripts"
sys.path.insert(0, CHILDREN_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── Import workers and functions from child scripts ───────────────────────────
# We import ONLY workers + headless functions, never MainWindow from children.

from zogy_subtract import (
    ZOGYWorker,
    SurveyDownloadWorker,
    ZOGYPreviewCanvas,
    DetectionTable,
    zogy_subtract,
    detect_transients,
    align_images,
    background_match,
    estimate_fwhm_from_stars,
    estimate_noise,
    estimate_psf_gaussian,
    fetch_reference_from_survey,
    SURVEY_OPTIONS,
    SIRIL_STYLESHEET,
    SIRIL_BG, SIRIL_BG2, SIRIL_BG3,
    SIRIL_ACCENT, SIRIL_ACCENT2,
    SIRIL_TEXT, SIRIL_TEXT_DIM,
    SIRIL_BORDER, SIRIL_SUCCESS,
    SIRIL_WARNING, SIRIL_SECTION,
    SIRIL_ERROR,
)

from transient_detector import (
    DetectionWorker,
    LightCurveCanvas,
    CutoutCanvas,
    frame_difference_detect,
    zogy_layer_detect,
    build_light_curves,
    detect_photometric_transients,
    crossmatch_alert,
    export_alerts_csv,
    export_tns_format,
    export_aavso_report,
    deduplicate_alerts,
    make_alert_id,
    SIRIL_NOVA,
)

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QFrame, QScrollArea, QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPen


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATUS WIDGET  (Section A5)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStatusWidget(QWidget):
    """
    Horizontal pipeline flow diagram.
    Shows each step as a rounded rect with status icon and arrows between.
    """
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
        W   = self.width()
        H   = self.height()
        bw  = min(160, (W - 40) // n - 20)
        bh  = 50
        gap = (W - n * bw) // (n + 1)
        by  = (H - bh) // 2

        for i, step in enumerate(self.steps):
            bx     = gap + i * (bw + gap)
            key    = step["key"]
            status = self.statuses.get(key, "idle")
            icon, color = self.STATUSES.get(status, ("○", "#3a4055"))

            # Box
            p.setBrush(QColor("#252930"))
            p.setPen(QPen(QColor(color), 1.5))
            p.drawRoundedRect(QRectF(bx, by, bw, bh), 6, 6)

            # Status icon
            p.setPen(QColor(color))
            f = QFont("Segoe UI", 13)
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(bx + 4, by, 26, bh),
                       Qt.AlignmentFlag.AlignCenter, icon)

            # Step name
            p.setPen(QColor("#dde3ee"))
            f2 = QFont("Segoe UI", 8)
            p.setFont(f2)
            p.drawText(QRectF(bx + 30, by, bw - 34, bh),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       step["name"])

            # Arrow to next
            if i < n - 1:
                ax = bx + bw + 4
                ay = by + bh // 2
                p.setPen(QPen(QColor("#3a4055"), 1.5))
                p.drawLine(int(ax), int(ay),
                           int(ax + gap - 8), int(ay))
                p.drawLine(int(ax + gap - 8), int(ay),
                           int(ax + gap - 14), int(ay - 5))
                p.drawLine(int(ax + gap - 8), int(ay),
                           int(ax + gap - 14), int(ay + 5))
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED ZOGY WORKER  (adds set_input_data / result_ready for pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineZOGYWorker(ZOGYWorker):
    """
    Thin subclass of ZOGYWorker that adds in-memory pipeline support.
    In pipeline mode the result is also stored on self for handoff.
    The parent's result_ready signal already emits the full dict — we just
    capture it so MainWindow can forward it to the transient step.
    """
    pipeline_result = pyqtSignal(dict)   # emitted after finished(success=True)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__(config, cancel_event)
        self._last_full_result: dict | None = None
        # Connect own result_ready to our capturer
        self.result_ready.connect(self._capture_result)

    def _capture_result(self, r: dict):
        self._last_full_result = r

    def run(self):
        super().run()
        # After run completes, if successful, emit pipeline_result
        if self._last_full_result is not None:
            self.pipeline_result.emit(self._last_full_result)


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED DETECTION WORKER  (adds set_input_data for pipeline handoff)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineDetectionWorker(DetectionWorker):
    """
    Thin subclass of DetectionWorker.
    When in pipeline mode, receives ZOGY arrays in memory and injects
    them as the Layer 2 ZOGY result, bypassing file I/O.
    """

    def set_input_data(self, data: dict):
        """
        Receive ZOGY outputs from the previous pipeline step.
        data keys: diff_image (D), sig_image (S), ref_image, sci_image,
                   detections (list of dicts from ZOGY step)
        """
        self._piped_data = data

    def run(self):
        """
        If piped data is present, inject the ZOGY difference + significance
        maps directly into Layer 2 processing rather than re-running ZOGY.
        """
        if hasattr(self, "_piped_data") and self._piped_data:
            self._inject_piped_zogy()
        else:
            # Fall back to normal standalone operation
            super().run()

    def _inject_piped_zogy(self):
        """Run detection with in-memory ZOGY results for Layer 2."""
        import traceback
        try:
            cfg        = self.config
            piped      = self._piped_data
            all_alerts = []

            # Layer 1 — frame difference (still uses files)
            if cfg.get("layer1_enabled", True):
                self._run_layer1(cfg, all_alerts)
                if self._cancel.is_set():
                    self._abort(); return

            # Layer 2 — use in-memory ZOGY result (no re-run)
            if cfg.get("layer2_enabled", True):
                self.log_line.emit("Layer 2: ZOGY (in-memory, piped from ZOGY step)")
                S   = piped.get("sig_image")   # significance map
                D   = piped.get("diff_image")  # difference image
                if S is not None and D is not None:
                    from zogy_subtract import detect_transients as _detect
                    threshold = cfg.get("layer2_threshold", 5.0)
                    raw = _detect(S, D, threshold_sigma=threshold,
                                  min_separation_px=5.0, border_margin=20)
                    for d in raw:
                        a = {
                            "x": d["x"], "y": d["y"], "snr": d["snr"],
                            "delta_flux": d["flux_D"], "area_px": 1,
                            "type": d["type"], "layer": 2,
                            "method": "zogy_piped",
                            "fits_file": cfg.get("new_path", ""),
                            "frame_idx": 0,
                            "alert_id": make_alert_id(0),
                            "timestamp_utc": "",
                            "ra": None, "dec": None,
                            "delta_mag_est": None,
                            "classification": "unknown",
                            "verified": False, "notes": "",
                            "mpc_match": None, "vsx_match": None, "tns_match": None,
                        }
                        all_alerts.append(a)
                        self.alert_found.emit(a)
                    self.log_line.emit(
                        f"  Layer 2 (piped): {len(raw)} detections above {threshold}σ")
                else:
                    self.log_line.emit("  Layer 2 skipped — no sig/diff arrays in piped data")

            # Layer 3 — photometric monitoring (uses files)
            if cfg.get("layer3_enabled", False):
                self._run_layer3(cfg, all_alerts)
                if self._cancel.is_set():
                    self._abort(); return

            # Deduplication
            before     = len(all_alerts)
            all_alerts = deduplicate_alerts(all_alerts)
            if len(all_alerts) < before:
                self.log_line.emit(
                    f"Deduplication: {before} → {len(all_alerts)} alerts")

            if cfg.get("crossmatch_enabled", False) and all_alerts:
                self._run_crossmatch(cfg, all_alerts)

            out_dir  = cfg.get("output_dir", cfg.get("fits_dir", "."))
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "transient_alerts.csv")
            export_alerts_csv(all_alerts, csv_path)
            self.log_line.emit(f"Saved {len(all_alerts)} alerts → {csv_path}")

            if cfg.get("export_tns", False):
                tns_path = os.path.join(out_dir, "tns_report.txt")
                export_tns_format(all_alerts, tns_path,
                                   cfg.get("observer_name", ""),
                                   cfg.get("instrument", ""))
                self.log_line.emit(f"TNS report → {tns_path}")

            if cfg.get("export_aavso", False):
                aavso_path = os.path.join(out_dir, "aavso_report.txt")
                export_aavso_report(all_alerts, aavso_path,
                                     cfg.get("observer_code", "XXX"),
                                     cfg.get("target_name", "UNKNOWN"))
                self.log_line.emit(f"AAVSO report → {aavso_path}")

            unknown_count = sum(1 for a in all_alerts
                                if a.get("classification") == "unknown")
            self.finished.emit({
                "success": True,
                "total_alerts": len(all_alerts),
                "unknown_alerts": unknown_count,
                "csv_path": csv_path,
                "alerts": all_alerts,
            })

        except Exception as e:
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# ZOGY PANEL  (full ZOGY settings UI lifted from zogy_subtract MainWindow)
# ─────────────────────────────────────────────────────────────────────────────

class ZOGYPanel(QWidget):
    """
    Complete ZOGY settings + preview panel, extracted from zogy_subtract.
    Provides get_config() and can host run/cancel independently.
    Signals: run_requested, result_ready(dict), finished(dict)
    """
    run_requested  = pyqtSignal()
    result_ready   = pyqtSignal(dict)
    finished       = pyqtSignal(dict)
    log_line       = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)   # (message, color)

    def __init__(self, shared_log: QPlainTextEdit, parent=None):
        super().__init__(parent)
        self._shared_log    = shared_log
        self._worker        = None
        self._survey_worker = None
        self._cancel_event  = threading.Event()
        self._last_result   = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Left panel: settings ──────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(400)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_w = QWidget()
        sl = QVBoxLayout(scroll_w)
        sl.setSpacing(6)
        sl.setContentsMargins(0, 2, 14, 4)
        scroll.setWidget(scroll_w)
        lv.addWidget(scroll, 1)

        # Input images
        grp_input = QGroupBox("Input images")
        gi = QVBoxLayout(grp_input)
        gi.setSpacing(4)
        gi.addWidget(QLabel("New image (tonight):"))
        rn = QHBoxLayout()
        self.edit_new = QLineEdit()
        self.edit_new.setPlaceholderText("Tonight's stacked image…")
        btn_new = QPushButton("Browse"); btn_new.setFixedWidth(56)
        btn_new.clicked.connect(lambda: self._browse_fits(self.edit_new))
        rn.addWidget(self.edit_new); rn.addWidget(btn_new)
        gi.addLayout(rn)
        gi.addWidget(QLabel("Reference image:"))
        rr = QHBoxLayout()
        self.edit_ref = QLineEdit()
        self.edit_ref.setPlaceholderText("Manual FITS or use survey below…")
        btn_ref = QPushButton("Browse"); btn_ref.setFixedWidth(56)
        btn_ref.clicked.connect(lambda: self._browse_fits(self.edit_ref))
        rr.addWidget(self.edit_ref); rr.addWidget(btn_ref)
        gi.addLayout(rr)
        btn_preview = QPushButton("Load & preview inputs")
        btn_preview.clicked.connect(self._load_and_preview)
        gi.addWidget(btn_preview)
        self.lbl_sizes    = QLabel("")
        self.lbl_sizes.setObjectName("dim")
        self.lbl_size_warn = QLabel("")
        self.lbl_size_warn.setObjectName("warn")
        self.lbl_size_warn.hide()
        gi.addWidget(self.lbl_sizes)
        gi.addWidget(self.lbl_size_warn)
        sl.addWidget(grp_input)

        # Sky survey
        grp_sv = QGroupBox("Sky survey reference")
        sv = QVBoxLayout(grp_sv)
        sv.setSpacing(4)
        lbl_sv = QLabel("New image must be plate-solved.\nDownloads reference matching your FOV.")
        lbl_sv.setObjectName("dim")
        lbl_sv.setWordWrap(True)
        sv.addWidget(lbl_sv)
        self.combo_survey = QComboBox()
        self.combo_survey.addItems(SURVEY_OPTIONS)
        sv.addWidget(self.combo_survey)
        self.btn_download = QPushButton("Download reference from survey")
        self.btn_download.setObjectName("warning_btn")
        self.btn_download.clicked.connect(self._download_reference)
        sv.addWidget(self.btn_download)
        self.lbl_survey_status = QLabel("")
        self.lbl_survey_status.setObjectName("dim")
        self.lbl_survey_status.setWordWrap(True)
        sv.addWidget(self.lbl_survey_status)
        sl.addWidget(grp_sv)

        # Pre-processing
        grp_pre = QGroupBox("Pre-processing")
        gp = QVBoxLayout(grp_pre)
        gp.setSpacing(4)
        gp.addWidget(QLabel("Alignment:"))
        self.combo_align = QComboBox()
        self.combo_align.addItems([
            "Phase correlation (sub-pixel)",
            "Cross-correlation (integer px)",
            "None (pre-aligned)",
        ])
        gp.addWidget(self.combo_align)
        self.chk_bg_match = QCheckBox("Match backgrounds")
        self.chk_bg_match.setChecked(True)
        gp.addWidget(self.chk_bg_match)
        sl.addWidget(grp_pre)

        # PSF settings
        grp_psf = QGroupBox("PSF settings")
        gpsf = QVBoxLayout(grp_psf)
        gpsf.setSpacing(4)
        for attr, lbl_text in [("spin_fwhm_new", "New FWHM (px):"),
                                ("spin_fwhm_ref", "Ref FWHM (px):")]:
            row = QHBoxLayout()
            lbl = QLabel(lbl_text); lbl.setFixedWidth(110)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 20.0); spin.setValue(0.0)
            spin.setSingleStep(0.5);  spin.setDecimals(1)
            setattr(self, attr, spin)
            row.addWidget(lbl); row.addWidget(spin)
            gpsf.addLayout(row)
        lbl_auto = QLabel("0 = auto-estimate from stars")
        lbl_auto.setObjectName("dim")
        gpsf.addWidget(lbl_auto)
        sl.addWidget(grp_psf)

        # Detection
        grp_det = QGroupBox("Detection")
        gd = QVBoxLayout(grp_det)
        gd.setSpacing(4)
        gd.addWidget(QLabel("Detection threshold:"))
        row_thresh = QHBoxLayout()
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(30, 100)
        self.slider_thresh.setValue(50)
        self.lbl_thresh_val = QLabel("5.0 σ")
        self.lbl_thresh_val.setStyleSheet(f"color:{SIRIL_ACCENT}; font-weight:bold;")
        self.lbl_thresh_val.setFixedWidth(42)
        self.slider_thresh.valueChanged.connect(
            lambda v: self.lbl_thresh_val.setText(f"{v/10:.1f} σ"))
        row_thresh.addWidget(self.slider_thresh)
        row_thresh.addWidget(self.lbl_thresh_val)
        gd.addLayout(row_thresh)
        lbl_hint = QLabel("5σ recommended — lower = more false positives")
        lbl_hint.setObjectName("dim"); lbl_hint.setWordWrap(True)
        gd.addWidget(lbl_hint)
        row_sep = QHBoxLayout()
        lbl_sep = QLabel("Min separation (px):"); lbl_sep.setFixedWidth(140)
        self.spin_min_sep = QSpinBox(); self.spin_min_sep.setRange(1, 30); self.spin_min_sep.setValue(5)
        row_sep.addWidget(lbl_sep); row_sep.addWidget(self.spin_min_sep)
        gd.addLayout(row_sep)
        row_brd = QHBoxLayout()
        lbl_brd = QLabel("Border margin (px):"); lbl_brd.setFixedWidth(140)
        self.spin_border = QSpinBox(); self.spin_border.setRange(5, 100); self.spin_border.setValue(20)
        row_brd.addWidget(lbl_brd); row_brd.addWidget(self.spin_border)
        gd.addLayout(row_brd)
        sl.addWidget(grp_det)

        # Output
        grp_out = QGroupBox("Output")
        go = QVBoxLayout(grp_out)
        go.setSpacing(4)
        go.addWidget(QLabel("Output folder:"))
        row_out = QHBoxLayout()
        self.edit_outdir = QLineEdit()
        self.edit_outdir.setPlaceholderText("Same folder as new image")
        btn_od = QPushButton("Browse"); btn_od.setFixedWidth(56)
        btn_od.clicked.connect(lambda: self._browse_dir(self.edit_outdir))
        row_out.addWidget(self.edit_outdir); row_out.addWidget(btn_od)
        go.addLayout(row_out)
        self.chk_save_D   = QCheckBox("Save D  (difference image)"); self.chk_save_D.setChecked(True)
        self.chk_save_S   = QCheckBox("Save S  (significance image)"); self.chk_save_S.setChecked(True)
        self.chk_save_csv = QCheckBox("Save candidates CSV"); self.chk_save_csv.setChecked(True)
        self.chk_load_D   = QCheckBox("Load D in Siril after run")
        for chk in [self.chk_save_D, self.chk_save_S, self.chk_save_csv, self.chk_load_D]:
            go.addWidget(chk)
        sl.addWidget(grp_out)
        sl.addStretch()

        # Run / Cancel
        self.btn_run = QPushButton("▶  Run ZOGY subtraction")
        self.btn_run.setObjectName("primary")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self._run)
        lv.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setMinimumHeight(28)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.hide()
        lv.addWidget(self.btn_cancel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 8)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.hide()
        lv.addWidget(self.progress_bar)
        root.addWidget(left)

        # ── Right panel: tabs ─────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        self.tabs = QTabWidget()
        rv.addWidget(self.tabs)

        # Tab: Preview
        tab_prev = QWidget()
        tv = QVBoxLayout(tab_prev)
        tv.setContentsMargins(2, 2, 2, 2)
        self.canvas = ZOGYPreviewCanvas()
        tv.addWidget(self.canvas)
        self.tabs.addTab(tab_prev, "🖼 Preview")

        # Tab: Candidates
        tab_cand = QWidget()
        cv = QVBoxLayout(tab_cand)
        cv.setContentsMargins(6, 6, 6, 6)
        badge_row = QHBoxLayout()
        self.badge_total = self._make_badge("Total",      "0")
        self.badge_pos   = self._make_badge("▲ Positive", "0", SIRIL_SUCCESS)
        self.badge_neg   = self._make_badge("▼ Negative", "0", SIRIL_ERROR)
        for b in [self.badge_total, self.badge_pos, self.badge_neg]:
            badge_row.addWidget(b)
        badge_row.addStretch()
        cv.addLayout(badge_row)
        self.det_table = DetectionTable()
        cv.addWidget(self.det_table, 1)
        btn_exp = QPushButton("Export CSV"); btn_exp.setObjectName("export")
        btn_exp.clicked.connect(self._export_csv)
        cv.addWidget(btn_exp)
        self.tabs.addTab(tab_cand, "📋 Candidates")

        # Tab: Log (shared from pipeline)
        log_wrap = QWidget()
        lw = QVBoxLayout(log_wrap)
        lw.setContentsMargins(4, 4, 4, 4)
        lbl_log_hint = QLabel("Full log available in Pipeline Log tab.")
        lbl_log_hint.setObjectName("dim")
        lw.addWidget(lbl_log_hint)
        self.local_log = QPlainTextEdit()
        self.local_log.setReadOnly(True)
        lw.addWidget(self.local_log)
        self.tabs.addTab(log_wrap, "📋 ZOGY Log")

        self.status_lbl = QLabel("Load images and run ZOGY subtraction.")
        self.status_lbl.setObjectName("dim")
        rv.addWidget(self.status_lbl)
        root.addWidget(right, 1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_badge(self, label: str, value: str, color: str = SIRIL_ACCENT) -> QWidget:
        w = QWidget()
        w.setStyleSheet(
            f"background:{SIRIL_BG3}; border:1px solid {SIRIL_BORDER}; "
            f"border-radius:4px; padding:4px 10px;")
        hb = QHBoxLayout(w)
        hb.setContentsMargins(6, 4, 6, 4); hb.setSpacing(6)
        lbl_l = QLabel(label); lbl_l.setObjectName("dim")
        lbl_v = QLabel(value)
        lbl_v.setStyleSheet(f"color:{color}; font-size:14pt; font-weight:bold;")
        hb.addWidget(lbl_l); hb.addWidget(lbl_v)
        w._val = lbl_v
        return w

    def _set_badge(self, badge, val: str):
        badge._val.setText(val)

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.local_log.appendPlainText(f"[{ts}] {msg}")
        self.log_line.emit(f"[ZOGY]  {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet(f"color:{color}; padding:4px 8px; font-size:9pt;")

    def _browse_fits(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FITS image", "",
            "FITS files (*.fit *.fits *.fts);;All files (*)")
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            edit.setText(path)

    def get_config(self) -> dict:
        align_map = {
            "Phase correlation (sub-pixel)":   "phase_correlation",
            "Cross-correlation (integer px)":  "cross_correlation",
            "None (pre-aligned)":              "none",
        }
        return {
            "new_path":            self.edit_new.text().strip(),
            "ref_path":            self.edit_ref.text().strip(),
            "align_method":        align_map[self.combo_align.currentText()],
            "match_background":    self.chk_bg_match.isChecked(),
            "fwhm_new_px":         self.spin_fwhm_new.value() or None,
            "fwhm_ref_px":         self.spin_fwhm_ref.value() or None,
            "detection_threshold": self.slider_thresh.value() / 10.0,
            "min_separation_px":   float(self.spin_min_sep.value()),
            "border_margin":       self.spin_border.value(),
            "output_dir":          self.edit_outdir.text().strip() or None,
            "save_D":              self.chk_save_D.isChecked(),
            "save_S":              self.chk_save_S.isChecked(),
            "save_csv":            self.chk_save_csv.isChecked(),
            "load_D_in_siril":     self.chk_load_D.isChecked(),
        }

    def get_new_path(self) -> str:
        return self.edit_new.text().strip()

    def get_output_dir(self) -> str:
        p = self.edit_outdir.text().strip()
        new_p = self.edit_new.text().strip()
        return p or (os.path.dirname(new_p) if new_p else "")

    def set_enabled(self, enabled: bool):
        self.btn_run.setEnabled(enabled)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _load_and_preview(self):
        from astropy.io import fits
        new_p = self.edit_new.text().strip()
        ref_p = self.edit_ref.text().strip()
        if not new_p or not os.path.isfile(new_p):
            QMessageBox.warning(self, "Missing file", "Select a valid new image.")
            return
        if not ref_p or not os.path.isfile(ref_p):
            QMessageBox.warning(self, "Missing file", "Select a valid reference image.")
            return
        try:
            with fits.open(new_p) as h:
                nd = h[0].data
                if nd.ndim == 3:
                    nd = nd[0] if nd.shape[0] != 3 else 0.299*nd[0]+0.587*nd[1]+0.114*nd[2]
            with fits.open(ref_p) as h:
                rd = h[0].data
                if rd.ndim == 3:
                    rd = rd[0] if rd.shape[0] != 3 else 0.299*rd[0]+0.587*rd[1]+0.114*rd[2]
            self.lbl_sizes.setText(f"New: {nd.shape[1]}×{nd.shape[0]}  |  Ref: {rd.shape[1]}×{rd.shape[0]}")
            if nd.shape != rd.shape:
                self.lbl_size_warn.setText("⚠ Sizes differ — resample/align in Siril before ZOGY")
                self.lbl_size_warn.show()
            else:
                self.lbl_size_warn.hide()
            self.canvas.show_inputs(nd.astype(np.float32), rd.astype(np.float32))
            self.tabs.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _download_reference(self):
        new_p = self.edit_new.text().strip()
        if not new_p or not os.path.isfile(new_p):
            QMessageBox.warning(self, "Missing file",
                "Select the new (science) image first — it needs WCS for survey alignment.")
            return
        out_dir = self.get_output_dir() or os.path.dirname(new_p)
        survey  = self.combo_survey.currentText()
        self.btn_download.setEnabled(False)
        self.lbl_survey_status.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:9pt;")
        self.lbl_survey_status.setText("Downloading…")
        self._survey_worker = SurveyDownloadWorker(new_p, survey, out_dir)
        self._survey_worker.log_line.connect(self._log)
        self._survey_worker.finished.connect(self._on_survey_done)
        self._survey_worker.start()

    def _on_survey_done(self, result: dict):
        self.btn_download.setEnabled(True)
        if result.get("success"):
            path = result["path"]
            self.edit_ref.setText(path)
            self.lbl_survey_status.setStyleSheet(f"color:{SIRIL_SUCCESS}; font-size:9pt;")
            self.lbl_survey_status.setText(f"Saved: {os.path.basename(path)}")
            self._log(f"Reference downloaded: {path}")
        else:
            err = result.get("error", "Unknown error")
            self.lbl_survey_status.setStyleSheet(f"color:{SIRIL_ERROR}; font-size:9pt;")
            self.lbl_survey_status.setText(f"Failed: {err}")

    def run_standalone(self):
        """Trigger a standalone run (not from pipeline)."""
        self._run()

    def run_pipeline(self, cancel_event: threading.Event) -> PipelineZOGYWorker:
        """
        Start ZOGY in pipeline mode. Returns the worker so caller can
        connect signals. cancel_event is shared with the pipeline.
        """
        cfg = self.get_config()
        if not cfg["new_path"] or not os.path.isfile(cfg["new_path"]):
            raise ValueError("New image path missing or not found.")
        if not cfg["ref_path"] or not os.path.isfile(cfg["ref_path"]):
            raise ValueError("Reference image path missing or not found.")
        self._cancel_event = cancel_event
        worker = PipelineZOGYWorker(cfg, self._cancel_event)
        worker.log_line.connect(self._log)
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_result_ready)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self.btn_run.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        worker.start()
        return worker

    def _run(self):
        cfg = self.get_config()
        if not cfg["new_path"] or not os.path.isfile(cfg["new_path"]):
            QMessageBox.warning(self, "Missing file", "Select a valid new image.")
            return
        if not cfg["ref_path"] or not os.path.isfile(cfg["ref_path"]):
            QMessageBox.warning(self, "Missing file", "Select a valid reference image.")
            return
        self._cancel_event.clear()
        self.btn_run.setEnabled(False)
        self.btn_cancel.show()
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self._set_status("Running ZOGY subtraction…", SIRIL_ACCENT)
        self._log("=" * 50)
        self._log(f"ZOGY run started [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        self._worker = PipelineZOGYWorker(cfg, self._cancel_event)
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self._log("Cancellation requested…")
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _on_progress(self, step: int, total: int, msg: str):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(step)
        self._set_status(f"Step {step}/{total}: {msg}", SIRIL_ACCENT)

    def _on_result_ready(self, result: dict):
        self._last_result = result
        self.canvas.show_inputs(result["new_data"], result["ref_data"])
        self.canvas.show_results(
            result["D"], result["S"],
            result["detections"],
            self.slider_thresh.value() / 10.0)
        dets  = result["detections"]
        n_pos = sum(1 for d in dets if d["type"] == "positive")
        n_neg = len(dets) - n_pos
        self._set_badge(self.badge_total, str(len(dets)))
        self._set_badge(self.badge_pos,   str(n_pos))
        self._set_badge(self.badge_neg,   str(n_neg))
        self.det_table.load_detections(dets)
        self.result_ready.emit(result)

    def _on_finished(self, result: dict):
        self.btn_run.setEnabled(True)
        self.btn_cancel.hide()
        self.progress_bar.hide()
        if result.get("success"):
            n   = result["n_detections"]
            msg = (f"✓ {n} candidates  "
                   f"(▲ {result['n_positive']}  ▼ {result['n_negative']})  "
                   f"D saved: {result['d_path']}")
            self._set_status(msg, SIRIL_SUCCESS)
            self._log(f"Done. {n} candidates.")
        else:
            err = result.get("error", "Unknown error")
            self._set_status(f"✗ Error: {err}", SIRIL_ERROR)
            self._log(f"FAILED: {err}")
        self.finished.emit(result)

    def _export_csv(self):
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "transient_candidates.csv", "CSV files (*.csv)")
        if path:
            import csv
            dets = self._last_result.get("detections", [])
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["rank", "x", "y", "snr", "flux_D", "type"])
                writer.writeheader()
                for i, d in enumerate(dets, 1):
                    writer.writerow({"rank": i, **d})


# ─────────────────────────────────────────────────────────────────────────────
# TRANSIENT PANEL  (full Transient Detector UI)
# ─────────────────────────────────────────────────────────────────────────────

class TransientPanel(QWidget):
    """
    Complete Transient Detector settings + results panel.
    Provides get_config(), run_pipeline(piped_data, cancel_event).
    Signals: finished(dict), log_line(str)
    """
    finished  = pyqtSignal(dict)
    log_line  = pyqtSignal(str)

    def __init__(self, shared_log: QPlainTextEdit, parent=None):
        super().__init__(parent)
        self._shared_log  = shared_log
        self._worker      = None
        self._cancel_ev   = threading.Event()
        self._alerts      = []
        self._lc_data     = {}
        self._fits_files  = []
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Left panel ─────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(400)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(4, 4, 4, 4)
        il.setSpacing(8)
        scroll.setWidget(inner)
        lv.addWidget(scroll)

        # Input
        grp_input = QGroupBox("Input sequence")
        gl = QVBoxLayout(grp_input)
        fl = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("FITS folder path…")
        btn_browse = QPushButton("…"); btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_folder)
        fl.addWidget(self._folder_edit); fl.addWidget(btn_browse)
        gl.addLayout(fl)
        sr = QHBoxLayout()
        btn_scan = QPushButton("🔍  Scan")
        btn_scan.clicked.connect(self._scan_folder)
        self._lbl_file_count = QLabel("No files loaded")
        self._lbl_file_count.setObjectName("dim")
        sr.addWidget(btn_scan); sr.addWidget(self._lbl_file_count, stretch=1)
        gl.addLayout(sr)
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Filename", "DATE-OBS header"])
        sort_row.addWidget(self._sort_combo, stretch=1)
        gl.addLayout(sort_row)
        il.addWidget(grp_input)

        # Detection layers
        grp_layers = QGroupBox("Detection layers")
        ll = QVBoxLayout(grp_layers)
        self._chk_l1 = QCheckBox("Layer 1 — Frame difference")
        self._chk_l1.setChecked(True)
        ll.addWidget(self._chk_l1)
        l1p = QHBoxLayout()
        l1p.addWidget(QLabel("Threshold:"))
        self._sl_l1 = QSlider(Qt.Orientation.Horizontal)
        self._sl_l1.setRange(20, 150); self._sl_l1.setValue(60)
        self._lbl_l1 = QLabel("6.0 σ")
        self._lbl_l1.setStyleSheet(f"color:{SIRIL_ACCENT}; font-weight:bold;")
        self._lbl_l1.setFixedWidth(42)
        self._sl_l1.valueChanged.connect(lambda v: self._lbl_l1.setText(f"{v/10:.1f} σ"))
        l1p.addWidget(self._sl_l1); l1p.addWidget(self._lbl_l1)
        ll.addLayout(l1p)

        self._chk_l2 = QCheckBox("Layer 2 — ZOGY subtraction (uses piped result in pipeline)")
        self._chk_l2.setChecked(True)
        ll.addWidget(self._chk_l2)
        self._ref_edit = QLineEdit()
        self._ref_edit.setPlaceholderText("Standalone: reference FITS (piped from ZOGY in pipeline)")
        rr = QHBoxLayout()
        btn_ref = QPushButton("…"); btn_ref.setFixedWidth(32)
        btn_ref.clicked.connect(self._browse_reference)
        rr.addWidget(self._ref_edit); rr.addWidget(btn_ref)
        ll.addLayout(rr)
        l2p = QHBoxLayout()
        l2p.addWidget(QLabel("ZOGY threshold:"))
        self._sl_l2 = QSlider(Qt.Orientation.Horizontal)
        self._sl_l2.setRange(20, 100); self._sl_l2.setValue(50)
        self._lbl_l2 = QLabel("5.0 σ")
        self._lbl_l2.setStyleSheet(f"color:{SIRIL_ACCENT}; font-weight:bold;")
        self._lbl_l2.setFixedWidth(42)
        self._sl_l2.valueChanged.connect(lambda v: self._lbl_l2.setText(f"{v/10:.1f} σ"))
        l2p.addWidget(self._sl_l2); l2p.addWidget(self._lbl_l2)
        ll.addLayout(l2p)

        self._chk_l3 = QCheckBox("Layer 3 — Photometric monitoring")
        self._chk_l3.setChecked(False)
        ll.addWidget(self._chk_l3)
        l3p = QHBoxLayout()
        l3p.addWidget(QLabel("Phot threshold:"))
        self._sl_l3 = QSlider(Qt.Orientation.Horizontal)
        self._sl_l3.setRange(20, 100); self._sl_l3.setValue(40)
        self._lbl_l3 = QLabel("4.0 σ")
        self._lbl_l3.setStyleSheet(f"color:{SIRIL_ACCENT}; font-weight:bold;")
        self._lbl_l3.setFixedWidth(42)
        self._sl_l3.valueChanged.connect(lambda v: self._lbl_l3.setText(f"{v/10:.1f} σ"))
        l3p.addWidget(self._sl_l3); l3p.addWidget(self._lbl_l3)
        ll.addLayout(l3p)

        minpx_row = QHBoxLayout()
        minpx_row.addWidget(QLabel("Min blob pixels:"))
        self._sp_minpx = QSpinBox(); self._sp_minpx.setRange(1, 20); self._sp_minpx.setValue(3)
        minpx_row.addWidget(self._sp_minpx)
        ll.addLayout(minpx_row)
        il.addWidget(grp_layers)

        # Photometry
        grp_phot = QGroupBox("Photometry (Layer 3)")
        pl = QFormLayout(grp_phot)
        self._sp_refstars  = QSpinBox(); self._sp_refstars.setRange(5, 100); self._sp_refstars.setValue(20)
        self._dsp_aperture = QDoubleSpinBox(); self._dsp_aperture.setRange(2.0, 30.0); self._dsp_aperture.setValue(8.0)
        pl.addRow("Reference stars:", self._sp_refstars)
        pl.addRow("Aperture (px):",   self._dsp_aperture)
        il.addWidget(grp_phot)

        # Catalog cross-match
        grp_cat = QGroupBox("Catalog cross-match")
        cl = QVBoxLayout(grp_cat)
        self._chk_crossmatch = QCheckBox("Cross-match vs VSX + MPC")
        cl.addWidget(self._chk_crossmatch)
        cat_params = QHBoxLayout()
        cat_params.addWidget(QLabel("Search radius (arcsec):"))
        self._dsp_radius = QDoubleSpinBox(); self._dsp_radius.setRange(1, 60); self._dsp_radius.setValue(10)
        cat_params.addWidget(self._dsp_radius)
        cl.addLayout(cat_params)
        self._lbl_no_wcs = QLabel("⚠ Images need plate-solved WCS for catalog matching")
        self._lbl_no_wcs.setObjectName("warn")
        self._lbl_no_wcs.setVisible(False)
        cl.addWidget(self._lbl_no_wcs)
        self._chk_crossmatch.toggled.connect(self._lbl_no_wcs.setVisible)
        il.addWidget(grp_cat)

        # Export
        grp_exp = QGroupBox("Export")
        el = QVBoxLayout(grp_exp)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self._out_edit = QLineEdit()
        btn_out = QPushButton("…"); btn_out.setFixedWidth(32)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit); out_row.addWidget(btn_out)
        el.addLayout(out_row)
        self._chk_csv   = QCheckBox("Export CSV"); self._chk_csv.setChecked(True)
        self._chk_tns   = QCheckBox("Export TNS-format report")
        self._chk_aavso = QCheckBox("Export AAVSO report")
        for chk in [self._chk_csv, self._chk_tns, self._chk_aavso]:
            el.addWidget(chk)
        obs_form = QFormLayout()
        self._obs_name  = QLineEdit()
        self._obs_code  = QLineEdit()
        self._inst_edit = QLineEdit()
        self._target_edit = QLineEdit()
        obs_form.addRow("Observer name:", self._obs_name)
        obs_form.addRow("AAVSO code:",    self._obs_code)
        obs_form.addRow("Instrument:",    self._inst_edit)
        obs_form.addRow("Target name:",   self._target_edit)
        self._obs_widget = QWidget()
        self._obs_widget.setLayout(obs_form)
        self._obs_widget.setVisible(False)
        el.addWidget(self._obs_widget)
        def _obs_vis():
            self._obs_widget.setVisible(
                self._chk_tns.isChecked() or self._chk_aavso.isChecked())
        self._chk_tns.toggled.connect(_obs_vis)
        self._chk_aavso.toggled.connect(_obs_vis)
        il.addWidget(grp_exp)
        il.addStretch()

        # Run / Cancel
        self._btn_run = QPushButton("▶  Run detection")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run_standalone)
        lv.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setMinimumHeight(28)
        self._btn_cancel.clicked.connect(self._cancel)
        lv.addWidget(self._btn_cancel)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(8)
        self._progress.setVisible(False)
        lv.addWidget(self._progress)

        root.addWidget(left)

        # ── Right panel: tabs ─────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        self._tabs = QTabWidget()
        rv.addWidget(self._tabs)

        # Alert table tab
        alert_tab = QWidget()
        atl = QVBoxLayout(alert_tab)
        badge_row = QHBoxLayout()
        self._badge_total  = self._make_badge("Total: 0",    SIRIL_TEXT)
        self._badge_l1     = self._make_badge("Layer 1: 0",  SIRIL_TEXT_DIM)
        self._badge_l2     = self._make_badge("Layer 2: 0",  SIRIL_ACCENT)
        self._badge_l3     = self._make_badge("Layer 3: 0",  SIRIL_SUCCESS)
        self._badge_unknwn = self._make_badge("★ Unknown: 0", SIRIL_NOVA)
        for b in [self._badge_total, self._badge_l1, self._badge_l2,
                  self._badge_l3, self._badge_unknwn]:
            badge_row.addWidget(b)
        badge_row.addStretch()
        atl.addLayout(badge_row)
        self._alert_table = QTableWidget(0, 9)
        self._alert_table.setHorizontalHeaderLabels(
            ["#", "Time", "File", "X", "Y", "SNR", "Type", "Layer", "Status"])
        self._alert_table.setAlternatingRowColors(True)
        self._alert_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._alert_table.horizontalHeader().setStretchLastSection(True)
        self._alert_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._alert_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._alert_table.doubleClicked.connect(self._on_alert_dblclick)
        atl.addWidget(self._alert_table)
        btn_row = QHBoxLayout()
        btn_exp_csv = QPushButton("Export CSV"); btn_exp_csv.setObjectName("export")
        btn_exp_csv.clicked.connect(self._export_csv)
        btn_copy = QPushButton("Copy unknowns"); btn_copy.setObjectName("alert")
        btn_copy.clicked.connect(self._copy_unknowns)
        btn_fp = QPushButton("Filter false positives")
        btn_fp.clicked.connect(self._filter_fp)
        btn_verified = QPushButton("Mark verified")
        btn_verified.clicked.connect(self._mark_verified)
        for b in [btn_exp_csv, btn_copy, btn_fp, btn_verified]:
            btn_row.addWidget(b)
        btn_row.addStretch()
        atl.addLayout(btn_row)
        self._tabs.addTab(alert_tab, "⚡  Alerts  [0]")

        # Light curves
        lc_tab = QWidget()
        ltl = QVBoxLayout(lc_tab)
        self._lc_canvas = LightCurveCanvas()
        ltl.addWidget(self._lc_canvas)
        ltl.addWidget(QLabel("Red lines = alerted stars"))
        self._tabs.addTab(lc_tab, "📈  Light Curves")

        # Cutout preview
        cut_tab = QWidget()
        ctl = QVBoxLayout(cut_tab)
        self._cutout_canvas = CutoutCanvas()
        ctl.addWidget(self._cutout_canvas)
        self._lbl_cutout_detail = QLabel("")
        self._lbl_cutout_detail.setObjectName("dim")
        self._lbl_cutout_detail.setWordWrap(True)
        ctl.addWidget(self._lbl_cutout_detail)
        self._tabs.addTab(cut_tab, "🖼  Cutout Preview")

        # Local log
        log_tab = QWidget()
        logl = QVBoxLayout(log_tab)
        self._log_pane = QPlainTextEdit()
        self._log_pane.setReadOnly(True)
        logl.addWidget(self._log_pane)
        self._tabs.addTab(log_tab, "📋  Transient Log")

        self._status_bar = QLabel("Ready — scan a folder and run detection.")
        self._status_bar.setObjectName("dim")
        rv.addWidget(self._status_bar)
        root.addWidget(right, 1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_badge(self, text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"background:{SIRIL_BG3}; color:{color}; "
            f"border:1px solid {SIRIL_BORDER}; border-radius:4px; "
            f"padding:3px 8px; font-weight:bold;")
        return lbl

    def _log(self, msg: str):
        self._log_pane.appendPlainText(msg)
        self.log_line.emit(f"[TRANSIENT]  {msg}")

    def set_enabled(self, enabled: bool):
        self._btn_run.setEnabled(enabled)

    def get_config(self) -> dict:
        return {
            "fits_files":               self._fits_files,
            "fits_dir":                 self._folder_edit.text().strip(),
            "output_dir":               self._out_edit.text().strip()
                                        or self._folder_edit.text().strip(),
            "layer1_enabled":           self._chk_l1.isChecked(),
            "layer1_threshold":         self._sl_l1.value() / 10,
            "min_pixels":               self._sp_minpx.value(),
            "border_margin":            20,
            "layer2_enabled":           self._chk_l2.isChecked(),
            "layer2_threshold":         self._sl_l2.value() / 10,
            "reference_path":           self._ref_edit.text().strip(),
            "layer3_enabled":           self._chk_l3.isChecked(),
            "layer3_threshold":         self._sl_l3.value() / 10,
            "n_ref_stars":              self._sp_refstars.value(),
            "aperture_radius":          self._dsp_aperture.value(),
            "fwhm_px":                  5.0,
            "crossmatch_enabled":       self._chk_crossmatch.isChecked(),
            "crossmatch_radius_arcsec": self._dsp_radius.value(),
            "export_csv":               self._chk_csv.isChecked(),
            "export_tns":               self._chk_tns.isChecked(),
            "export_aavso":             self._chk_aavso.isChecked(),
            "observer_name":            self._obs_name.text().strip(),
            "observer_code":            self._obs_code.text().strip() or "XXX",
            "instrument":               self._inst_edit.text().strip(),
            "target_name":              self._target_edit.text().strip() or "UNKNOWN",
        }

    def sync_output_from_zogy(self, output_dir: str):
        """Called by pipeline to sync the output folder from ZOGY settings."""
        if not self._out_edit.text().strip() and output_dir:
            self._out_edit.setText(output_dir)

    # ── Run logic ─────────────────────────────────────────────────────────────

    def run_pipeline(self, piped_data: dict | None,
                     cancel_event: threading.Event) -> PipelineDetectionWorker:
        """
        Start detection in pipeline mode.
        If piped_data has ZOGY arrays, Layer 2 uses them in memory.
        Returns the worker.
        """
        cfg = self.get_config()
        if not cfg["fits_files"]:
            raise ValueError("No FITS files loaded. Scan a folder first.")
        self._cancel_ev = cancel_event
        self._alerts    = []
        self._alert_table.setRowCount(0)
        self._update_badges()

        worker = PipelineDetectionWorker(cfg, self._cancel_ev)
        if piped_data:
            worker.set_input_data(piped_data)
        worker.log_line.connect(self._log)
        worker.progress.connect(self._on_progress)
        worker.alert_found.connect(self._on_alert_found)
        worker.lc_ready.connect(self._on_lc_ready)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._status_bar.setText("⚡ Running detection…")
        worker.start()
        return worker

    def _run_standalone(self):
        cfg = self.get_config()
        if not cfg["fits_files"]:
            QMessageBox.warning(self, "No files", "Scan a folder first.")
            return
        self._alerts = []
        self._alert_table.setRowCount(0)
        self._update_badges()
        self._cancel_ev = threading.Event()
        worker = PipelineDetectionWorker(cfg, self._cancel_ev)
        worker.log_line.connect(self._log)
        worker.progress.connect(self._on_progress)
        worker.alert_found.connect(self._on_alert_found)
        worker.lc_ready.connect(self._on_lc_ready)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._status_bar.setText("⚡ Running detection…")
        worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self._btn_cancel.setEnabled(False)
        self._status_bar.setText("Cancelling…")

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_progress(self, current, total, msg):
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(current)
        self._status_bar.setText(f"⚡ {msg}")

    def _on_alert_found(self, alert: dict):
        self._alerts.append(alert)
        self._add_table_row(alert)
        self._update_badges()
        n = len(self._alerts)
        self._tabs.setTabText(0, f"⚡  Alerts  [{n}]")

    def _on_lc_ready(self, lc: dict):
        self._lc_data = lc
        alerted = {a["star_id"] for a in self._alerts if a.get("layer") == 3}
        self._lc_canvas.show_light_curves(lc, alerted)

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        if result.get("success"):
            unk   = result.get("unknown_alerts", 0)
            total = result.get("total_alerts", 0)
            msg   = f"✓  Complete — {total} alerts, {unk} unidentified"
            self._status_bar.setText(msg)
            self._status_bar.setStyleSheet(
                f"color:{SIRIL_NOVA if unk else SIRIL_SUCCESS};")
        else:
            err = result.get("error", "Unknown error")
            self._status_bar.setText(f"✗  {err}")
            self._status_bar.setStyleSheet(f"color:{SIRIL_ERROR};")
        self.finished.emit(result)

    # ── Alert table helpers ───────────────────────────────────────────────────

    def _add_table_row(self, a: dict):
        row = self._alert_table.rowCount()
        self._alert_table.insertRow(row)
        layer_colors = {1: None, 2: QColor(45, 106, 191, 30),
                        3: QColor(76, 175, 125, 30)}
        values = [
            str(row + 1),
            str(a.get("timestamp_utc", ""))[:19],
            a.get("fits_file", ""),
            f"{a.get('x', 0):.1f}",
            f"{a.get('y', 0):.1f}",
            f"{a.get('snr', 0):.2f}",
            a.get("type", ""),
            str(a.get("layer", "")),
            a.get("classification", "unknown"),
        ]
        bg   = layer_colors.get(a.get("layer"))
        nova = a.get("classification") == "unknown"
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if bg:
                item.setBackground(bg)
            if nova:
                item.setForeground(QColor(SIRIL_NOVA))
            self._alert_table.setItem(row, col, item)

    def _update_badges(self):
        total = len(self._alerts)
        l1    = sum(1 for a in self._alerts if a.get("layer") == 1)
        l2    = sum(1 for a in self._alerts if a.get("layer") == 2)
        l3    = sum(1 for a in self._alerts if a.get("layer") == 3)
        unk   = sum(1 for a in self._alerts if a.get("classification") == "unknown")
        self._badge_total.setText(f"Total: {total}")
        self._badge_l1.setText(f"Layer 1: {l1}")
        self._badge_l2.setText(f"Layer 2: {l2}")
        self._badge_l3.setText(f"Layer 3: {l3}")
        self._badge_unknwn.setText(f"★ Unknown: {unk}")

    def _on_alert_dblclick(self, index):
        row = index.row()
        if row >= len(self._alerts):
            return
        alert = self._alerts[row]
        self._tabs.setCurrentIndex(2)
        self._cutout_canvas.show_cutout(alert, self._fits_files)
        self._lbl_cutout_detail.setText(
            f"Alert #{row+1}  |  {alert.get('alert_id','')}  |  "
            f"({alert.get('x',0):.1f}, {alert.get('y',0):.1f})  |  "
            f"SNR={alert.get('snr',0):.2f}  |  {alert.get('type','')}  |  "
            f"Layer {alert.get('layer','')}  |  "
            f"Classification: {alert.get('classification','?')}")

    def _export_csv(self):
        if not self._alerts:
            QMessageBox.information(self, "No alerts", "No alerts to export.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save CSV", filter="CSV (*.csv)")
        if out:
            export_alerts_csv(self._alerts, out)
            self._log(f"Exported {len(self._alerts)} alerts → {out}")

    def _copy_unknowns(self):
        unknowns = [a for a in self._alerts if a.get("classification") == "unknown"]
        if not unknowns:
            QMessageBox.information(self, "No unknowns", "No unidentified alerts.")
            return
        lines = [
            f"{a['alert_id']}  RA={a.get('ra','N/A')}  Dec={a.get('dec','N/A')}  "
            f"SNR={a['snr']:.2f}  {a['type']}  {a.get('fits_file','')}"
            for a in unknowns
        ]
        QApplication.clipboard().setText("\n".join(lines))
        self._log(f"Copied {len(unknowns)} unknown alerts to clipboard")

    def _filter_fp(self):
        before = len(self._alerts)
        self._alerts = [
            a for a in self._alerts
            if not (a.get("layer") == 1 and a.get("area_px", 0) == 1)
        ]
        removed = before - len(self._alerts)
        self._rebuild_table()
        self._update_badges()
        self._tabs.setTabText(0, f"⚡  Alerts  [{len(self._alerts)}]")
        self._log(f"False-positive filter: removed {removed} single-pixel detections")

    def _mark_verified(self):
        for idx in self._alert_table.selectedItems():
            row = idx.row()
            if row < len(self._alerts):
                self._alerts[row]["verified"] = True
                item = self._alert_table.item(row, 8)
                if item:
                    item.setText("✓ verified")
                    item.setForeground(QColor(SIRIL_SUCCESS))

    def _rebuild_table(self):
        self._alert_table.setRowCount(0)
        for a in self._alerts:
            self._add_table_row(a)

    # ── Folder helpers ─────────────────────────────────────────────────────────

    def _browse_folder(self):
        import glob
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self._folder_edit.setText(d)
            if not self._out_edit.text():
                self._out_edit.setText(d)

    def _browse_reference(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select reference FITS", filter="FITS (*.fit *.fits *.fts)")
        if f:
            self._ref_edit.setText(f)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self._out_edit.setText(d)

    def _scan_folder(self):
        import glob
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Select a valid FITS folder.")
            return
        patterns = ["*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        files = list(set(files))
        if self._sort_combo.currentIndex() == 0:
            files.sort()
        else:
            from astropy.io import fits as astropy_fits_local
            def _sort_key(fp):
                try:
                    return astropy_fits_local.getheader(fp).get("DATE-OBS", "")
                except Exception:
                    return ""
            files.sort(key=_sort_key)
        self._fits_files = files
        n = len(files)
        if n:
            from astropy.io import fits as _fits
            try:
                sample = _fits.getdata(files[0])
                size_str = f"{sample.shape[-1]}×{sample.shape[-2]}" if sample.ndim >= 2 else "?"
            except Exception:
                size_str = "?"
            self._lbl_file_count.setText(f"{n} files  |  {size_str} px")
            self._log(f"Scanned: {n} FITS files in {folder}")
        else:
            self._lbl_file_count.setText("No FITS files found")


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW / PIPELINE TAB
# ─────────────────────────────────────────────────────────────────────────────

class OverviewTab(QWidget):
    """
    Tab 0 — Pipeline Overview.
    Science mode selector, pipeline flow diagram, global controls,
    summary panel after each step.
    """
    run_pipeline_requested  = pyqtSignal()
    run_zogy_only_requested = pyqtSignal()
    run_transient_only_requested = pyqtSignal()
    cancel_requested        = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)

        # Science mode selector
        mode_grp = QGroupBox("Science mode")
        ml = QVBoxLayout(mode_grp)
        self._btn_grp = QButtonGroup(self)
        modes = [
            ("🌟  Transient search  (new/disappeared objects via ZOGY + layer detection)",
             "transient"),
            ("🔭  Full pipeline  (ZOGY subtraction → multi-layer transient detection)",
             "full"),
            ("📈  Transient only  (skip ZOGY, load sequence direct into detector)",
             "detector_only"),
        ]
        self._mode_radios = {}
        for label, key in modes:
            rb = QRadioButton(label)
            self._btn_grp.addButton(rb)
            ml.addWidget(rb)
            self._mode_radios[key] = rb
        self._mode_radios["full"].setChecked(True)
        root.addWidget(mode_grp)

        # Pipeline flow diagram
        diag_grp = QGroupBox("Pipeline")
        dl = QVBoxLayout(diag_grp)
        self._pipeline_widget = PipelineStatusWidget([
            {"name": "ZOGY Subtraction",  "key": "zogy"},
            {"name": "Transient Detector", "key": "transient"},
        ])
        self._pipeline_widget.setMinimumHeight(90)
        dl.addWidget(self._pipeline_widget)

        step_btns = QHBoxLayout()
        self._btn_run_zogy = QPushButton("▶ Run ZOGY only")
        self._btn_run_zogy.clicked.connect(self.run_zogy_only_requested)
        self._btn_run_transient = QPushButton("▶ Run Transient only")
        self._btn_run_transient.clicked.connect(self.run_transient_only_requested)
        step_btns.addWidget(self._btn_run_zogy)
        step_btns.addWidget(self._btn_run_transient)
        step_btns.addStretch()
        dl.addLayout(step_btns)
        root.addWidget(diag_grp)

        # ZOGY summary (shown after ZOGY completes)
        self._zogy_summary = QGroupBox("ZOGY Result")
        self._zogy_summary.setVisible(False)
        zsl = QVBoxLayout(self._zogy_summary)
        self._lbl_zogy_fwhm   = QLabel("")
        self._lbl_zogy_cands  = QLabel("")
        self._lbl_zogy_saved  = QLabel("")
        for lbl in [self._lbl_zogy_fwhm, self._lbl_zogy_cands, self._lbl_zogy_saved]:
            lbl.setObjectName("dim")
            zsl.addWidget(lbl)
        root.addWidget(self._zogy_summary)

        # Transient summary (shown after detection completes)
        self._transient_summary = QGroupBox("Detection Result")
        self._transient_summary.setVisible(False)
        tsl = QVBoxLayout(self._transient_summary)
        self._lbl_total_alerts  = QLabel("")
        self._lbl_unknown_alerts = QLabel("")
        self._btn_copy_unknowns = QPushButton("📋 Copy unknown alerts to clipboard")
        self._btn_copy_unknowns.setObjectName("alert")
        self._btn_copy_unknowns.setEnabled(False)
        self._lbl_total_alerts.setObjectName("dim")
        self._lbl_unknown_alerts.setObjectName("dim")
        tsl.addWidget(self._lbl_total_alerts)
        tsl.addWidget(self._lbl_unknown_alerts)
        tsl.addWidget(self._btn_copy_unknowns)
        root.addWidget(self._transient_summary)

        # Warning label for standalone detector mode
        self._lbl_standalone_warn = QLabel(
            "⚠ Run ZOGY first, or provide reference image in the Transient tab for Layer 2.")
        self._lbl_standalone_warn.setObjectName("warn")
        self._lbl_standalone_warn.setVisible(False)
        root.addWidget(self._lbl_standalone_warn)

        root.addStretch()

        # Bottom action row
        action_row = QHBoxLayout()
        self._btn_run_all = QPushButton("▶  Run pipeline")
        self._btn_run_all.setObjectName("primary")
        self._btn_run_all.setMinimumHeight(36)
        self._btn_run_all.setMinimumWidth(160)
        self._btn_run_all.clicked.connect(self.run_pipeline_requested)
        self._btn_cancel_pipeline = QPushButton("✕  Cancel")
        self._btn_cancel_pipeline.setObjectName("danger")
        self._btn_cancel_pipeline.setMinimumHeight(36)
        self._btn_cancel_pipeline.setEnabled(False)
        self._btn_cancel_pipeline.clicked.connect(self.cancel_requested)
        action_row.addWidget(self._btn_run_all)
        action_row.addWidget(self._btn_cancel_pipeline)
        action_row.addStretch()
        root.addLayout(action_row)

    def get_science_mode(self) -> str:
        for key, rb in self._mode_radios.items():
            if rb.isChecked():
                return key
        return "full"

    def set_running(self, running: bool):
        self._btn_run_all.setEnabled(not running)
        self._btn_run_zogy.setEnabled(not running)
        self._btn_run_transient.setEnabled(not running)
        self._btn_cancel_pipeline.setEnabled(running)

    def set_step_status(self, key: str, status: str):
        self._pipeline_widget.set_status(key, status)

    def show_zogy_result(self, result: dict):
        self._lbl_zogy_fwhm.setText(
            f"FWHM new={result.get('fwhm_new', 0):.2f} px  |  "
            f"FWHM ref={result.get('fwhm_ref', 0):.2f} px")
        n = result.get("n_detections", 0)
        self._lbl_zogy_cands.setText(
            f"Candidates: {n}  "
            f"(▲ {result.get('n_positive',0)}  ▼ {result.get('n_negative',0)})")
        self._lbl_zogy_saved.setText(
            f"D saved: {result.get('d_path', '—')}")
        self._zogy_summary.setVisible(True)

    def show_transient_result(self, result: dict, copy_cb):
        total = result.get("total_alerts", 0)
        unk   = result.get("unknown_alerts", 0)
        self._lbl_total_alerts.setText(f"Total alerts: {total}")
        self._lbl_unknown_alerts.setText(
            f"★ Unidentified: {unk}" + (" — candidates for nova/SN reporting" if unk else ""))
        if unk:
            self._lbl_unknown_alerts.setStyleSheet(f"color:{SIRIL_NOVA}; font-weight:bold;")
        self._btn_copy_unknowns.setEnabled(unk > 0)
        self._btn_copy_unknowns.clicked.connect(copy_cb)
        self._transient_summary.setVisible(True)

    def show_standalone_warning(self, show: bool):
        self._lbl_standalone_warn.setVisible(show)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transient & Variable Science Engine  —  Siril")
        self.resize(1400, 860)

        # Pipeline state (Section A6)
        self._pipeline_step   = 0
        self._pipeline_data   = {}
        self._current_worker  = None
        self._cancel_event    = threading.Event()
        self._zogy_full_result: dict | None = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

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

        lbl_icon  = QLabel("🔭")
        lbl_icon.setStyleSheet("font-size:18pt; background:transparent;")
        lbl_title = QLabel("Transient & Variable Science Engine")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_NOVA}; font-size:13pt; font-weight:bold; background:transparent;")
        lbl_ver   = QLabel("v1.0  |  2 scripts  |  ZOGY → Transient Detection")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")

        hl.addWidget(lbl_icon)
        hl.addWidget(lbl_title)
        hl.addStretch()
        hl.addWidget(lbl_ver)
        root.addWidget(header)

        # ── Tab widget ─────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(False)
        root.addWidget(self._tabs, 1)

        # Shared pipeline log
        self._pipeline_log = QPlainTextEdit()
        self._pipeline_log.setReadOnly(True)

        # Tab 0 — Overview / Pipeline
        self._overview_tab = OverviewTab()
        self._overview_tab.run_pipeline_requested.connect(self._run_pipeline)
        self._overview_tab.run_zogy_only_requested.connect(self._run_zogy_only)
        self._overview_tab.run_transient_only_requested.connect(self._run_transient_only)
        self._overview_tab.cancel_requested.connect(self._cancel_pipeline)
        self._tabs.addTab(self._overview_tab, "🚀  Overview")

        # Tab 1 — ZOGY
        self._zogy_panel = ZOGYPanel(self._pipeline_log)
        self._zogy_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._zogy_panel, "🔭  ZOGY Subtraction")

        # Tab 2 — Transient Detector
        self._transient_panel = TransientPanel(self._pipeline_log)
        self._transient_panel.log_line.connect(self._pipeline_log.appendPlainText)
        self._tabs.addTab(self._transient_panel, "⚡  Transient Detector")

        # Tab 3 — Pipeline Log
        log_wrap = QWidget()
        lw = QVBoxLayout(log_wrap)
        lw.setContentsMargins(6, 6, 6, 6)
        lbl_log_title = QLabel("Combined log from all pipeline steps")
        lbl_log_title.setObjectName("dim")
        lw.addWidget(lbl_log_title)
        btn_clear = QPushButton("Clear log")
        btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._pipeline_log.clear)
        lw.addWidget(btn_clear)
        lw.addWidget(self._pipeline_log)
        self._tabs.addTab(log_wrap, "📋  Pipeline Log")

        # ── Bottom bar (always visible) ────────────────────────────────────────
        bottom = QWidget()
        bottom.setFixedHeight(46)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(10, 4, 10, 4)
        bl.setSpacing(8)

        self._global_progress = QProgressBar()
        self._global_progress.setFixedHeight(8)
        self._global_progress.setRange(0, 100)
        self._global_progress.setValue(0)
        bl.addWidget(self._global_progress, stretch=1)

        self._btn_run_pipeline = QPushButton("▶  Run pipeline")
        self._btn_run_pipeline.setObjectName("primary")
        self._btn_run_pipeline.setMinimumHeight(32)
        self._btn_run_pipeline.setMinimumWidth(140)
        self._btn_run_pipeline.clicked.connect(self._run_pipeline)
        bl.addWidget(self._btn_run_pipeline)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(32)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_pipeline)
        bl.addWidget(self._btn_cancel)

        self._status_lbl = QLabel("Ready — configure ZOGY and Transient tabs, then run.")
        self._status_lbl.setObjectName("dim")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status_lbl, stretch=1)

        root.addWidget(bottom)

        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "Transient & Variable Science Engine — ready")
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "ZOGY ref: Zackay, Ofek, Gal-Yam (2016), ApJ 830, 27 | arXiv:1601.02655")

    # ── Pipeline orchestration (Section A6) ───────────────────────────────────

    def _run_pipeline(self):
        mode = self._overview_tab.get_science_mode()

        # Validate before starting
        if mode in ("full", "transient"):
            cfg = self._zogy_panel.get_config()
            if not cfg["new_path"] or not os.path.isfile(cfg["new_path"]):
                QMessageBox.warning(self, "ZOGY: Missing input",
                    "Set the New image in the ZOGY Subtraction tab before running.")
                self._tabs.setCurrentIndex(1)
                return
            if not cfg["ref_path"] or not os.path.isfile(cfg["ref_path"]):
                QMessageBox.warning(self, "ZOGY: Missing reference",
                    "Set the Reference image in the ZOGY Subtraction tab, "
                    "or download one from a sky survey.")
                self._tabs.setCurrentIndex(1)
                return

        td_cfg = self._transient_panel.get_config()
        if mode == "detector_only" and not td_cfg["fits_files"]:
            QMessageBox.warning(self, "Transient: No files",
                "Scan a FITS folder in the Transient Detector tab first.")
            self._tabs.setCurrentIndex(2)
            return

        # Reset
        self._cancel_event.clear()
        self._pipeline_step  = 0
        self._pipeline_data  = {}
        self._zogy_full_result = None
        self._overview_tab.set_step_status("zogy",     "idle")
        self._overview_tab.set_step_status("transient","idle")
        self._set_running(True)

        if mode == "detector_only":
            self._overview_tab.show_standalone_warning(True)
            self._run_transient_step(piped_data=None)
        else:
            self._overview_tab.show_standalone_warning(False)
            self._run_zogy_step()

    def _run_zogy_step(self):
        self._overview_tab.set_step_status("zogy", "running")
        self._set_status("Running ZOGY subtraction…", SIRIL_ACCENT)
        self._global_progress.setValue(10)
        self._pipeline_log.appendPlainText(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
            "═══ PIPELINE STEP 1: ZOGY Subtraction ═══")
        try:
            worker = self._zogy_panel.run_pipeline(self._cancel_event)
            worker.finished.connect(self._on_zogy_step_done)
            worker.pipeline_result.connect(self._capture_zogy_result)
            self._current_worker = worker
        except ValueError as e:
            self._overview_tab.set_step_status("zogy", "error")
            self._set_status(f"ZOGY setup error: {e}", SIRIL_ERROR)
            self._set_running(False)

    def _capture_zogy_result(self, result: dict):
        """Capture the full ZOGY result dict for handoff to transient step."""
        self._zogy_full_result = result

    def _on_zogy_step_done(self, result: dict):
        if result.get("success"):
            self._overview_tab.set_step_status("zogy", "done")
            self._overview_tab.show_zogy_result(result)
            self._global_progress.setValue(50)
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"ZOGY complete — {result.get('n_detections',0)} candidates")

            # Sync output dir from ZOGY to transient
            self._transient_panel.sync_output_from_zogy(
                self._zogy_panel.get_output_dir())

            # Build in-memory handoff: ZOGY → Transient (Section A4)
            # Pass the significance and difference images in memory
            piped = None
            if self._zogy_full_result:
                piped = {
                    "diff_image":  self._zogy_full_result.get("D"),
                    "sig_image":   self._zogy_full_result.get("S"),
                    "ref_image":   self._zogy_full_result.get("ref_data"),
                    "sci_image":   self._zogy_full_result.get("new_data"),
                    "detections":  self._zogy_full_result.get("detections", []),
                    "new_path":    self._zogy_panel.get_new_path(),
                }
                self._pipeline_log.appendPlainText(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "In-memory handoff: D+S arrays → Transient Detector (no file I/O)")

            self._run_transient_step(piped_data=piped)
        else:
            self._overview_tab.set_step_status("zogy", "error")
            err = result.get("error", "Unknown")
            self._set_status(f"ZOGY failed: {err}", SIRIL_ERROR)
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✗ ZOGY FAILED: {err}")
            self._set_running(False)

    def _run_transient_step(self, piped_data: dict | None):
        self._overview_tab.set_step_status("transient", "running")
        self._set_status("Running transient detection…", SIRIL_ACCENT)
        self._global_progress.setValue(55)
        self._pipeline_log.appendPlainText(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
            "═══ PIPELINE STEP 2: Transient Detector ═══")
        if piped_data:
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                "Layer 2 will use in-memory ZOGY arrays (no disk re-read)")
        try:
            td_cfg = self._transient_panel.get_config()
            if not td_cfg["fits_files"]:
                raise ValueError(
                    "No FITS files loaded in Transient tab. "
                    "Scan a folder in the Transient Detector tab.")
            worker = self._transient_panel.run_pipeline(
                piped_data, self._cancel_event)
            worker.finished.connect(self._on_transient_step_done)
            self._current_worker = worker
        except ValueError as e:
            self._overview_tab.set_step_status("transient", "error")
            self._set_status(f"Transient setup error: {e}", SIRIL_ERROR)
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"⚠ {e}")
            self._set_running(False)

    def _on_transient_step_done(self, result: dict):
        if result.get("success"):
            self._overview_tab.set_step_status("transient", "done")
            self._global_progress.setValue(100)
            total = result.get("total_alerts", 0)
            unk   = result.get("unknown_alerts", 0)
            msg   = (f"✓ Pipeline complete — {total} alerts"
                     + (f", {unk} unidentified ★" if unk else ""))
            self._set_status(msg, SIRIL_SUCCESS if not unk else SIRIL_NOVA)
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Pipeline complete — {total} alerts, {unk} unidentified")
            self._overview_tab.show_transient_result(
                result,
                copy_cb=self._transient_panel._copy_unknowns)
        else:
            self._overview_tab.set_step_status("transient", "error")
            err = result.get("error", "Unknown")
            self._set_status(f"Transient failed: {err}", SIRIL_ERROR)
            self._pipeline_log.appendPlainText(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✗ TRANSIENT FAILED: {err}")
        self._set_running(False)

    # ── Standalone step runners ───────────────────────────────────────────────

    def _run_zogy_only(self):
        cfg = self._zogy_panel.get_config()
        if not cfg["new_path"] or not os.path.isfile(cfg["new_path"]):
            QMessageBox.warning(self, "Missing input",
                "Set the New image in the ZOGY Subtraction tab.")
            self._tabs.setCurrentIndex(1)
            return
        if not cfg["ref_path"] or not os.path.isfile(cfg["ref_path"]):
            QMessageBox.warning(self, "Missing reference",
                "Set the Reference image in the ZOGY Subtraction tab.")
            self._tabs.setCurrentIndex(1)
            return
        self._cancel_event.clear()
        self._overview_tab.set_step_status("zogy", "running")
        self._set_running(True)
        self._pipeline_log.appendPlainText(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
            "─── STANDALONE: ZOGY only ───")
        try:
            worker = self._zogy_panel.run_pipeline(self._cancel_event)
            worker.pipeline_result.connect(self._capture_zogy_result)
            worker.finished.connect(lambda r: (
                self._overview_tab.set_step_status(
                    "zogy", "done" if r.get("success") else "error"),
                self._overview_tab.show_zogy_result(r) if r.get("success") else None,
                self._set_status(
                    f"✓ ZOGY done — {r.get('n_detections',0)} candidates" if r.get("success")
                    else f"✗ {r.get('error','')}",
                    SIRIL_SUCCESS if r.get("success") else SIRIL_ERROR),
                self._set_running(False),
            ))
            self._current_worker = worker
        except ValueError as e:
            self._set_status(f"Error: {e}", SIRIL_ERROR)
            self._set_running(False)

    def _run_transient_only(self):
        self._overview_tab.show_standalone_warning(True)
        td_cfg = self._transient_panel.get_config()
        if not td_cfg["fits_files"]:
            QMessageBox.warning(self, "No files",
                "Scan a folder in the Transient Detector tab first.")
            self._tabs.setCurrentIndex(2)
            return
        self._cancel_event.clear()
        self._overview_tab.set_step_status("transient", "running")
        self._set_running(True)
        self._pipeline_log.appendPlainText(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
            "─── STANDALONE: Transient Detector only ───")
        try:
            worker = self._transient_panel.run_pipeline(None, self._cancel_event)
            worker.finished.connect(lambda r: (
                self._overview_tab.set_step_status(
                    "transient", "done" if r.get("success") else "error"),
                self._set_status(
                    f"✓ Detection done — {r.get('total_alerts',0)} alerts" if r.get("success")
                    else f"✗ {r.get('error','')}",
                    SIRIL_SUCCESS if r.get("success") else SIRIL_ERROR),
                self._set_running(False),
            ))
            self._current_worker = worker
        except ValueError as e:
            self._set_status(f"Error: {e}", SIRIL_ERROR)
            self._set_running(False)

    # ── Cancel (Section A9) ───────────────────────────────────────────────────

    def _cancel_pipeline(self):
        if self._current_worker:
            try:
                self._current_worker.cancel()
            except Exception:
                pass
        self._cancel_event.set()
        self._pipeline_step = 999  # block next step
        for key in ("zogy", "transient"):
            if self._overview_tab._pipeline_widget.statuses.get(key) == "running":
                self._overview_tab.set_step_status(key, "error")
        self._set_status("Pipeline cancelled.", SIRIL_WARNING)
        self._pipeline_log.appendPlainText(
            f"[{datetime.now().strftime('%H:%M:%S')}] CANCELLED")
        self._set_running(False)

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _set_running(self, running: bool):
        self._btn_run_pipeline.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._overview_tab.set_running(running)

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(
            f"color:{color}; font-size:9pt;")


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
