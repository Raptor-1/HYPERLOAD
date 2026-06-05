#!/usr/bin/env python3
"""
Equipment Profile Manager for Siril — #27
Siril-themed PyQt6 GUI | Shared profile store for other scripts

Other scripts can access profiles like this:
    from equipment_manager import load_all_profiles, PROFILE_FILE
    profiles = load_all_profiles()
    p = profiles.get("Newton 900mm + ASI294")
"""

import json
import os
import sys
from datetime import datetime

import sirilpy as s
s.ensure_installed("PyQt6")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QFormLayout, QComboBox,
    QMessageBox, QScrollArea, QFrame, QHeaderView, QFileDialog,
    QSizePolicy, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPalette

# ── Shared profile file — importable by other scripts ─────────────────────────
PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "equipment_profiles.json")

# ── Siril dark theme colors ───────────────────────────────────────────────────
SIRIL_BG        = "#1e2128"   # main background
SIRIL_BG2       = "#252930"   # slightly lighter panels
SIRIL_BG3       = "#2d3240"   # input fields
SIRIL_ACCENT    = "#4a9eff"   # Siril blue accent
SIRIL_ACCENT2   = "#2d6abf"   # darker blue for hover
SIRIL_TEXT      = "#dde3ee"   # main text
SIRIL_TEXT_DIM  = "#7a8499"   # dimmed text / labels
SIRIL_BORDER    = "#3a4055"   # borders
SIRIL_SUCCESS   = "#4caf7d"   # green for ok
SIRIL_WARNING   = "#e8a23a"   # orange for warnings
SIRIL_SECTION   = "#5ba3ff"   # section headers

SIRIL_STYLESHEET = f"""
QMainWindow, QDialog, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
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
QTableWidget::item {{
    padding: 4px 8px;
    border: none;
}}
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
QPushButton:pressed {{
    background-color: {SIRIL_ACCENT};
}}
QPushButton#primary {{
    background-color: {SIRIL_ACCENT2};
    border-color: {SIRIL_ACCENT};
    color: white;
    font-weight: bold;
}}
QPushButton#primary:hover {{
    background-color: {SIRIL_ACCENT};
}}
QPushButton#danger:hover {{
    background-color: #8b2020;
    border-color: #cc3333;
}}
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {SIRIL_ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {SIRIL_ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 4px;
}}
QComboBox QAbstractItemView {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT};
    selection-background-color: {SIRIL_ACCENT2};
    border: 1px solid {SIRIL_BORDER};
}}
QScrollBar:vertical {{
    background: {SIRIL_BG2};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {SIRIL_BORDER};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {SIRIL_ACCENT2};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QLabel#section {{
    color: {SIRIL_SECTION};
    font-weight: bold;
    padding-top: 6px;
}}
QLabel#dim {{
    color: {SIRIL_TEXT_DIM};
    font-size: 9pt;
}}
QCheckBox {{
    color: {SIRIL_TEXT};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {SIRIL_BORDER};
    border-radius: 2px;
    background: {SIRIL_BG3};
}}
QCheckBox::indicator:checked {{
    background: {SIRIL_ACCENT};
    border-color: {SIRIL_ACCENT};
}}
QDialogButtonBox QPushButton {{
    min-width: 80px;
    text-align: center;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QFrame#separator {{
    background-color: {SIRIL_BORDER};
    max-height: 1px;
}}
"""

# ── Shared data helpers — importable by other scripts ─────────────────────────

def load_all_profiles() -> dict:
    """
    Load all equipment profiles.
    Importable by other Siril scripts:
        from equipment_manager import load_all_profiles
        p = load_all_profiles().get("My Setup")
    """
    if not os.path.exists(PROFILE_FILE):
        return {}
    with open(PROFILE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_all_profiles(profiles: dict):
    """Save all profiles to the shared JSON file."""
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


def get_profile(name: str) -> dict | None:
    """
    Get a single profile by name.
    Importable by other Siril scripts:
        from equipment_manager import get_profile
        p = get_profile("Newton 900mm + ASI294")
        focal = p["focal_length_effective"]
    Returns None if not found.
    """
    return load_all_profiles().get(name)


def get_siril_vars(name: str) -> dict:
    """
    Get Siril-ready variables for a named profile.
    Returns a dict with focal, pixelsize, solve_radius etc.
    Importable by other scripts for direct use in Siril commands.

    Example:
        from equipment_manager import get_siril_vars
        v = get_siril_vars("Newton 900mm + ASI294")
        siril.cmd("set", "focal", str(v["focal"]))
    """
    p = get_profile(name)
    if not p:
        return {}
    return {
        "focal":            p.get("focal_length_effective") or p.get("focal_length_nominal") or 0,
        "pixelsize":        p.get("pixel_size_um") or 0,
        "solve_radius":     p.get("solve_radius_deg", 5.0),
        "distortion_order": p.get("distortion_order", 3),
        "blind_solve":      p.get("blind_solve", False),
        "bayer":            p.get("bayer_pattern", "RGGB"),
        "gain":             p.get("gain_default"),
        "drizzle":          p.get("drizzle_factor", 1.0),
        "stack_rejection":  p.get("stack_rejection", "winsorized"),
        "stack_sigma_low":  p.get("stack_sigma_low", 3.0),
        "stack_sigma_high": p.get("stack_sigma_high", 3.0),
        "filters":          p.get("filters", []),
        "spcc_sensor":      p.get("spcc_sensor_name", ""),
        "fov_w":            p.get("fov_width_arcmin"),
        "fov_h":            p.get("fov_height_arcmin"),
        "resolution":       p.get("resolution_arcsec_px"),
    }


def update_solve_result(name: str, success: bool,
                        fov: str = None, focal: float = None):
    """
    Update solve statistics for self-learning.
    Call this from other scripts after a plate solve:
        from equipment_manager import update_solve_result
        update_solve_result("Newton 900mm", success=True, focal=887.3)
    """
    profiles = load_all_profiles()
    if name not in profiles:
        return
    p = profiles[name]
    p["solve_attempt_count"] = p.get("solve_attempt_count", 0) + 1
    if success:
        p["solve_success_count"] = p.get("solve_success_count", 0) + 1
        p["last_verified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        if fov:
            p["last_solve_fov"] = fov
        if focal:
            p["last_solve_focal"] = round(focal, 1)
    profiles[name] = p
    save_all_profiles(profiles)


def calculate_derived_fields(p: dict) -> dict:
    fl = p.get("focal_length_effective") or p.get("focal_length_nominal")
    px = p.get("pixel_size_um")
    w  = p.get("sensor_width_px")
    h  = p.get("sensor_height_px")
    rf = p.get("reducer_factor", 1.0) or 1.0
    fl_eff = (fl * rf) if fl else None
    if fl_eff and px:
        arcsec = (px / fl_eff) * 206.265
        p["resolution_arcsec_px"] = round(arcsec, 3)
        if w:
            p["fov_width_arcmin"]  = round((arcsec * w) / 60, 2)
        if h:
            p["fov_height_arcmin"] = round((arcsec * h) / 60, 2)
    return p


EMPTY_PROFILE = {
    "name": "",
    "focal_length_nominal": None,
    "focal_length_effective": None,
    "aperture_mm": None,
    "reducer_factor": 1.0,
    "camera_name": "",
    "pixel_size_um": None,
    "sensor_width_px": None,
    "sensor_height_px": None,
    "bayer_pattern": "RGGB",
    "bit_depth": 16,
    "gain_default": None,
    "read_noise_e": None,
    "fov_width_arcmin": None,
    "fov_height_arcmin": None,
    "resolution_arcsec_px": None,
    "solve_radius_deg": 5.0,
    "distortion_order": 3,
    "blind_solve": False,
    "solve_success_count": 0,
    "solve_attempt_count": 0,
    "spcc_sensor_name": "",
    "filters": [],
    "stack_rejection": "winsorized",
    "stack_sigma_low": 3.0,
    "stack_sigma_high": 3.0,
    "drizzle_factor": 1.0,
    "notes": "",
    "created": "",
    "last_verified": "",
    "last_solve_fov": None,
    "last_solve_focal": None,
}


# ── Add / Edit dialog ─────────────────────────────────────────────────────────

class ProfileDialog(QDialog):

    def __init__(self, parent=None, existing: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Add Profile" if existing is None else "Edit Profile")
        self.setMinimumWidth(540)
        self.setMinimumHeight(600)
        self.profile = dict(EMPTY_PROFILE)
        if existing:
            self.profile.update(existing)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        form  = QFormLayout(inner)
        form.setSpacing(7)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        def dbl(val, dec=2, max_val=100000):
            w = QDoubleSpinBox()
            w.setDecimals(dec)
            w.setRange(0, max_val)
            w.setSpecialValueText("–")
            w.setValue(val if val else 0)
            return w

        def intbox(val, lo=0, hi=99999):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setSpecialValueText("–")
            w.setValue(val if val else 0)
            return w

        def txt(val):
            w = QLineEdit()
            w.setText(str(val) if val else "")
            return w

        def section(title):
            lbl = QLabel(title)
            lbl.setObjectName("section")
            return lbl

        # ── Profile name ──────────────────────────────────────────────────────
        self.f_name = txt(self.profile["name"])
        self.f_name.setPlaceholderText("e.g.  Newton 900mm + ASI294MC Pro")
        form.addRow(section("Profile name *"), self.f_name)
        self._separator(form)

        # ── Optics ────────────────────────────────────────────────────────────
        form.addRow(section("▸  Optics"), QLabel(""))
        self.f_focal_nom = dbl(self.profile["focal_length_nominal"])
        self.f_focal_eff = dbl(self.profile["focal_length_effective"])
        self.f_aperture  = dbl(self.profile["aperture_mm"])
        self.f_reducer   = dbl(self.profile["reducer_factor"] or 1.0)
        form.addRow("Nominal focal (mm)", self.f_focal_nom)
        form.addRow("Effective focal (mm)", self.f_focal_eff)
        form.addRow("Aperture (mm)", self.f_aperture)
        form.addRow("Reducer factor", self.f_reducer)
        self._separator(form)

        # ── Camera ────────────────────────────────────────────────────────────
        form.addRow(section("▸  Camera"), QLabel(""))
        self.f_cam_name = txt(self.profile["camera_name"])
        self.f_cam_name.setPlaceholderText("e.g.  ZWO ASI294MC Pro")
        self.f_pixel    = dbl(self.profile["pixel_size_um"], dec=3)
        self.f_width    = intbox(self.profile["sensor_width_px"])
        self.f_height   = intbox(self.profile["sensor_height_px"])
        self.f_bayer    = QComboBox()
        self.f_bayer.addItems(["RGGB", "BGGR", "GRBG", "GBRG", "None (mono)"])
        idx = self.f_bayer.findText(self.profile.get("bayer_pattern", "RGGB"))
        self.f_bayer.setCurrentIndex(idx if idx >= 0 else 0)
        self.f_bit      = QComboBox()
        self.f_bit.addItems(["8", "12", "14", "16", "32"])
        self.f_bit.setCurrentText(str(self.profile.get("bit_depth", 16)))
        self.f_gain     = intbox(self.profile["gain_default"])
        self.f_rnoise   = dbl(self.profile["read_noise_e"])
        form.addRow("Camera name", self.f_cam_name)
        form.addRow("Pixel size (µm)", self.f_pixel)
        form.addRow("Sensor width (px)", self.f_width)
        form.addRow("Sensor height (px)", self.f_height)
        form.addRow("Bayer pattern", self.f_bayer)
        form.addRow("Bit depth", self.f_bit)
        form.addRow("Default gain", self.f_gain)
        form.addRow("Read noise (e⁻)", self.f_rnoise)
        self._separator(form)

        # ── Astrometry ────────────────────────────────────────────────────────
        form.addRow(section("▸  Astrometry  (empirical values)"), QLabel(""))
        hint = QLabel("Enter values that actually work — not theoretical specs")
        hint.setObjectName("dim")
        form.addRow("", hint)
        self.f_radius = dbl(self.profile["solve_radius_deg"])
        self.f_dist   = intbox(self.profile["distortion_order"], lo=1, hi=5)
        self.f_blind  = QCheckBox("Enable blind solve by default")
        self.f_blind.setChecked(self.profile.get("blind_solve", False))
        form.addRow("Search radius (°)", self.f_radius)
        form.addRow("Distortion order", self.f_dist)
        form.addRow("", self.f_blind)
        self._separator(form)

        # ── SPCC / Filters ────────────────────────────────────────────────────
        form.addRow(section("▸  SPCC / Filters"), QLabel(""))
        self.f_spcc    = txt(self.profile["spcc_sensor_name"])
        self.f_filters = txt(", ".join(self.profile.get("filters", [])))
        self.f_filters.setPlaceholderText("e.g.  L, R, G, B, Ha_6nm, OIII_6nm")
        form.addRow("SPCC sensor name", self.f_spcc)
        form.addRow("Filters (comma-sep.)", self.f_filters)
        self._separator(form)

        # ── Stack defaults ────────────────────────────────────────────────────
        form.addRow(section("▸  Stack defaults"), QLabel(""))
        self.f_rejection = QComboBox()
        self.f_rejection.addItems(["winsorized", "linear", "percentile", "sigma", "none"])
        self.f_rejection.setCurrentText(self.profile.get("stack_rejection", "winsorized"))
        self.f_sigma_lo = dbl(self.profile["stack_sigma_low"])
        self.f_sigma_hi = dbl(self.profile["stack_sigma_high"])
        self.f_drizzle  = QComboBox()
        self.f_drizzle.addItems(["1.0", "1.5", "2.0"])
        self.f_drizzle.setCurrentText(str(self.profile.get("drizzle_factor", 1.0)))
        form.addRow("Rejection method", self.f_rejection)
        form.addRow("Sigma low", self.f_sigma_lo)
        form.addRow("Sigma high", self.f_sigma_hi)
        form.addRow("Drizzle factor", self.f_drizzle)
        self._separator(form)

        # ── Notes ─────────────────────────────────────────────────────────────
        form.addRow(section("▸  Notes"), QLabel(""))
        self.f_notes = QTextEdit()
        self.f_notes.setFixedHeight(65)
        self.f_notes.setPlaceholderText(
            "e.g.  TS-flattener 55mm backfocus, reducer changes focal to 630mm")
        self.f_notes.setPlainText(self.profile.get("notes", ""))
        form.addRow("", self.f_notes)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Save profile")
        btns.button(QDialogButtonBox.StandardButton.Ok).setObjectName("primary")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _separator(self, form):
        line = QFrame()
        line.setObjectName("separator")
        line.setFrameShape(QFrame.Shape.HLine)
        form.addRow(line)

    def _val(self, spinbox):
        v = spinbox.value()
        return v if v > 0 else None

    def get_profile(self) -> dict:
        p = dict(self.profile)
        p["name"]                   = self.f_name.text().strip()
        p["focal_length_nominal"]   = self._val(self.f_focal_nom)
        p["focal_length_effective"] = self._val(self.f_focal_eff)
        p["aperture_mm"]            = self._val(self.f_aperture)
        p["reducer_factor"]         = self.f_reducer.value() or 1.0
        p["camera_name"]            = self.f_cam_name.text().strip()
        p["pixel_size_um"]          = self._val(self.f_pixel)
        p["sensor_width_px"]        = self._val(self.f_width)
        p["sensor_height_px"]       = self._val(self.f_height)
        bayer = self.f_bayer.currentText()
        p["bayer_pattern"]          = None if "mono" in bayer else bayer
        p["bit_depth"]              = int(self.f_bit.currentText())
        p["gain_default"]           = self._val(self.f_gain)
        p["read_noise_e"]           = self._val(self.f_rnoise)
        p["solve_radius_deg"]       = self.f_radius.value()
        p["distortion_order"]       = self.f_dist.value()
        p["blind_solve"]            = self.f_blind.isChecked()
        p["spcc_sensor_name"]       = self.f_spcc.text().strip()
        p["filters"]                = [x.strip() for x in
                                       self.f_filters.text().split(",") if x.strip()]
        p["stack_rejection"]        = self.f_rejection.currentText()
        p["stack_sigma_low"]        = self.f_sigma_lo.value()
        p["stack_sigma_high"]       = self.f_sigma_hi.value()
        p["drizzle_factor"]         = float(self.f_drizzle.currentText())
        p["notes"]                  = self.f_notes.toPlainText().strip()
        if not p.get("created"):
            p["created"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        p["last_verified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return calculate_derived_fields(p)


# ── Profile detail viewer ─────────────────────────────────────────────────────

class ProfileDetailDialog(QDialog):

    def __init__(self, parent, profile: dict):
        super().__init__(parent)
        self.setWindowTitle(f"Profile — {profile.get('name','')}")
        self.setMinimumSize(520, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 8)

        # Header bar
        header = QLabel(f"  🔭  {profile.get('name','')}")
        header.setStyleSheet(f"""
            background-color: {SIRIL_BG3};
            color: {SIRIL_ACCENT};
            font-size: 12pt;
            font-weight: bold;
            padding: 8px 12px;
            border-radius: 4px;
            border: 1px solid {SIRIL_BORDER};
        """)
        layout.addWidget(header)

        # Calculated badge row
        badge_row = QHBoxLayout()
        p = profile
        res  = p.get("resolution_arcsec_px","–")
        fovw = p.get("fov_width_arcmin","–")
        fovh = p.get("fov_height_arcmin","–")
        for label, val in [
            ("Resolution", f'{res} "/px'),
            ("FOV", f"{fovw}′ × {fovh}′"),
            ("Focal", f'{p.get("focal_length_effective") or p.get("focal_length_nominal","–")} mm'),
        ]:
            badge = QLabel(f"{label}\n{val}")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(f"""
                background-color: {SIRIL_BG3};
                color: {SIRIL_TEXT};
                border: 1px solid {SIRIL_ACCENT2};
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 9pt;
            """)
            badge_row.addWidget(badge)
        layout.addLayout(badge_row)

        # Detail text
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Courier New", 9))
        text.setStyleSheet(f"""
            background-color: {SIRIL_BG2};
            color: {SIRIL_TEXT};
            border: 1px solid {SIRIL_BORDER};
            border-radius: 4px;
        """)
        succ  = p.get("solve_success_count", 0)
        total = p.get("solve_attempt_count", 0)
        rate  = f"{(succ/total*100):.0f}%" if total > 0 else "–"

        lines = [
            "── Optics ──────────────────────────────────────────",
            f"  Nominal focal    : {p.get('focal_length_nominal','–')} mm",
            f"  Effective focal  : {p.get('focal_length_effective','–')} mm",
            f"  Aperture         : {p.get('aperture_mm','–')} mm",
            f"  Reducer factor   : {p.get('reducer_factor',1.0)}×",
            "",
            "── Camera ──────────────────────────────────────────",
            f"  Camera           : {p.get('camera_name','–')}",
            f"  Pixel size       : {p.get('pixel_size_um','–')} µm",
            f"  Sensor           : {p.get('sensor_width_px','–')} × {p.get('sensor_height_px','–')} px",
            f"  Bayer            : {p.get('bayer_pattern','–')}",
            f"  Bit depth        : {p.get('bit_depth','–')}",
            f"  Gain             : {p.get('gain_default','–')}",
            f"  Read noise       : {p.get('read_noise_e','–')} e⁻",
            "",
            "── Astrometry ──────────────────────────────────────",
            f"  Search radius    : {p.get('solve_radius_deg','–')}°",
            f"  Distortion order : {p.get('distortion_order','–')}",
            f"  Blind solve      : {'yes' if p.get('blind_solve') else 'no'}",
            f"  Success rate     : {succ}/{total}  ({rate})",
            f"  Last solve focal : {p.get('last_solve_focal','–')} mm",
            f"  Last solve FOV   : {p.get('last_solve_fov','–')}",
            "",
            "── SPCC / Filters ──────────────────────────────────",
            f"  SPCC sensor      : {p.get('spcc_sensor_name','–')}",
            f"  Filters          : {', '.join(p.get('filters',[])) or '–'}",
            "",
            "── Stack defaults ──────────────────────────────────",
            f"  Rejection        : {p.get('stack_rejection','–')}",
            f"  Sigma low/high   : {p.get('stack_sigma_low','–')} / {p.get('stack_sigma_high','–')}",
            f"  Drizzle          : {p.get('drizzle_factor',1.0)}×",
            "",
            "── Notes ───────────────────────────────────────────",
            f"  {p.get('notes','–')}",
            "",
            f"  Created          : {p.get('created','–')}",
            f"  Last verified    : {p.get('last_verified','–')}",
            "",
            "── Siril script variables ──────────────────────────",
            f"  set focal        {p.get('focal_length_effective') or p.get('focal_length_nominal',0)}",
            f"  set pixelsize    {p.get('pixel_size_um',0)}",
            f"  # solve_radius:     {p.get('solve_radius_deg',5.0)}°",
            f"  # distortion_order: {p.get('distortion_order',3)}",
            f"  # bayer:            {p.get('bayer_pattern','RGGB')}",
        ]
        text.setPlainText("\n".join(lines))
        layout.addWidget(text)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


# ── Main window ───────────────────────────────────────────────────────────────

class EquipmentManager(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Equipment Profile Manager  —  Siril")
        self.setMinimumSize(860, 500)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = QWidget()
        title_bar.setStyleSheet(f"""
            background-color: {SIRIL_BG3};
            border-radius: 5px;
            border: 1px solid {SIRIL_BORDER};
        """)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(12, 8, 12, 8)

        logo = QLabel("🔭  Equipment Profile Manager")
        logo.setStyleSheet(f"color: {SIRIL_ACCENT}; font-size: 13pt; font-weight: bold;"
                           f" background: transparent; border: none;")
        tb_layout.addWidget(logo)
        tb_layout.addStretch()

        subtitle = QLabel(f"Profiles stored in:  {PROFILE_FILE}")
        subtitle.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 8pt;"
                               f" background: transparent; border: none;")
        tb_layout.addWidget(subtitle)
        root.addWidget(title_bar)

        # ── Body ──────────────────────────────────────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(10)

        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Profile name", "Focal (mm)", "Pixel (µm)", "FOV (arcmin)", "Solve %"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 5):
            self.table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(self.view_profile)
        body.addWidget(self.table)

        # Button panel
        btn_panel = QWidget()
        btn_panel.setFixedWidth(170)
        btn_panel.setStyleSheet(f"""
            QWidget {{
                background-color: {SIRIL_BG2};
                border: 1px solid {SIRIL_BORDER};
                border-radius: 5px;
            }}
        """)
        btn_layout = QVBoxLayout(btn_panel)
        btn_layout.setContentsMargins(10, 12, 10, 12)
        btn_layout.setSpacing(5)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        def section_lbl(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 8pt;"
                              f" padding-top: 8px; background: transparent; border: none;")
            return lbl

        def btn(icon, label, slot, obj_name=""):
            b = QPushButton(f"  {icon}  {label}")
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if obj_name:
                b.setObjectName(obj_name)
            b.clicked.connect(slot)
            return b

        btn_layout.addWidget(section_lbl("PROFILES"))
        btn_layout.addWidget(btn("＋", "Add profile",  self.add_profile,  "primary"))
        btn_layout.addWidget(btn("✎", "Edit profile",  self.edit_profile))
        btn_layout.addWidget(btn("◉", "View / Load",   self.view_profile))
        btn_layout.addWidget(btn("✕", "Delete",         self.delete_profile, "danger"))
        btn_layout.addWidget(section_lbl("IMPORT / EXPORT"))
        btn_layout.addWidget(btn("↑", "Export JSON",   self.export_profile))
        btn_layout.addWidget(btn("↓", "Import JSON",   self.import_profile))
        btn_layout.addWidget(section_lbl(""))
        btn_layout.addWidget(btn("↺", "Refresh",       self.refresh_table))
        btn_layout.addStretch()

        # Usage hint at bottom of panel
        hint = QLabel("Double-click a row\nto view full details")
        hint.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 8pt;"
                           f" background: transparent; border: none;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(hint)

        body.addWidget(btn_panel)
        root.addLayout(body)

        # ── Status bar ────────────────────────────────────────────────────────
        self.status = QLabel("Ready")
        self.status.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 8pt; padding: 2px;")
        root.addWidget(self.status)

        self.refresh_table()

    def set_status(self, msg: str, ok: bool = True):
        color = SIRIL_SUCCESS if ok else SIRIL_WARNING
        self.status.setStyleSheet(f"color: {color}; font-size: 8pt; padding: 2px;")
        self.status.setText(msg)

    def refresh_table(self):
        self.profiles = load_all_profiles()
        self.table.setRowCount(0)
        for name, p in self.profiles.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            fl   = p.get("focal_length_effective") or p.get("focal_length_nominal") or "–"
            px   = p.get("pixel_size_um") or "–"
            fovw = p.get("fov_width_arcmin", "–")
            fovh = p.get("fov_height_arcmin", "–")
            fov  = f"{fovw}′ × {fovh}′" if fovw != "–" else "–"
            succ  = p.get("solve_success_count", 0)
            total = p.get("solve_attempt_count", 0)
            rate  = f"{(succ/total*100):.0f}%" if total > 0 else "–"
            vals  = [name, str(fl), str(px), fov, rate]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                align = (Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                         if col == 0 else Qt.AlignmentFlag.AlignCenter)
                item.setTextAlignment(align)
                if col == 4 and total > 0:
                    pct = succ / total
                    item.setForeground(QColor(
                        SIRIL_SUCCESS if pct >= 0.8 else
                        SIRIL_WARNING if pct >= 0.5 else "#cc4444"))
                self.table.setItem(row, col, item)
        count = len(self.profiles)
        self.set_status(f"{count} profile{'s' if count != 1 else ''} loaded  —  {PROFILE_FILE}")

    def _selected_name(self) -> str | None:
        if self.table.currentRow() < 0:
            QMessageBox.information(self, "No selection",
                                    "Please select a profile from the list first.")
            return None
        return self.table.item(self.table.currentRow(), 0).text()

    def add_profile(self):
        dlg = ProfileDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.get_profile()
        if not p["name"]:
            QMessageBox.warning(self, "Error", "Profile name cannot be empty.")
            return
        profiles = load_all_profiles()
        if p["name"] in profiles:
            ans = QMessageBox.question(self, "Overwrite?",
                f"'{p['name']}' already exists. Overwrite it?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        profiles[p["name"]] = p
        save_all_profiles(profiles)
        self.refresh_table()
        self.set_status(f"Profile '{p['name']}' saved.")

    def edit_profile(self):
        name = self._selected_name()
        if not name:
            return
        profiles = load_all_profiles()
        dlg = ProfileDialog(self, existing=profiles[name])
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        p = dlg.get_profile()
        if not p["name"]:
            QMessageBox.warning(self, "Error", "Profile name cannot be empty.")
            return
        if p["name"] != name:
            del profiles[name]
        profiles[p["name"]] = p
        save_all_profiles(profiles)
        self.refresh_table()
        self.set_status(f"Profile '{p['name']}' updated.")

    def view_profile(self):
        name = self._selected_name()
        if not name:
            return
        ProfileDetailDialog(self, load_all_profiles()[name]).exec()

    def delete_profile(self):
        name = self._selected_name()
        if not name:
            return
        ans = QMessageBox.question(self, "Delete profile",
            f"Delete '{name}'?\nThis cannot be undone.")
        if ans != QMessageBox.StandardButton.Yes:
            return
        profiles = load_all_profiles()
        del profiles[name]
        save_all_profiles(profiles)
        self.refresh_table()
        self.set_status(f"Profile '{name}' deleted.", ok=False)

    def export_profile(self):
        name = self._selected_name()
        if not name:
            return
        p = dict(load_all_profiles()[name])
        for k in ("solve_success_count", "solve_attempt_count",
                  "last_solve_fov", "last_solve_focal"):
            p[k] = 0 if "count" in k else None
        p["notes"] = ""
        filename = os.path.join(
            os.path.dirname(PROFILE_FILE),
            name.replace(" ", "_").replace("/", "-") + "_profile.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, "Exported",
            f"Exported to:\n{filename}\n\nShareable with other Siril users.")
        self.set_status(f"Exported: {filename}")

    def import_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Profile JSON", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                p = json.load(f)
            if not p.get("name"):
                raise ValueError("Missing 'name' field in JSON")
            profiles = load_all_profiles()
            profiles[p["name"]] = calculate_derived_fields(p)
            save_all_profiles(profiles)
            self.refresh_table()
            self.set_status(f"Imported: '{p['name']}'")
        except Exception as e:
            QMessageBox.critical(self, "Import failed", str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = EquipmentManager()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()