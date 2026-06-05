r"""
nb_palette.py — Script #38
Narrowband Palette System for Siril
One-click narrowband palette combination (SHO, HOO, Foraxx, etc.)
Place in: C:\Users\Marcell\Desktop\Siril New Scripts\
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import glob
import shutil
import threading
from datetime import datetime

import sirilpy as s
from astropy.io import fits as astropy_fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QCheckBox, QPlainTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QTabWidget, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QGridLayout, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QTimer
from PyQt6.QtGui import QPainter, QColor, QFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile, get_siril_vars
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ─────────────────────────────────────────────────────────────────────────────
# SIRIL THEME
# ─────────────────────────────────────────────────────────────────────────────
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
QPushButton#palette {{
    background-color: {SIRIL_BG3};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
    padding: 8px 12px;
    text-align: left;
    font-size: 10pt;
}}
QPushButton#palette:hover {{
    border-color: {SIRIL_ACCENT};
    background-color: #1a2540;
}}
QPushButton#palette:checked {{
    background-color: {SIRIL_ACCENT2};
    border-color: {SIRIL_ACCENT};
    color: white;
    font-weight: bold;
}}
QPushButton#copy {{
    background-color: {SIRIL_BG3};
    border-color: {SIRIL_SUCCESS};
    color: {SIRIL_SUCCESS};
    text-align: center;
    font-weight: bold;
}}
QPushButton#copy:hover {{ background-color: #1a3d2a; }}
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
QLabel#section  {{ color: {SIRIL_SECTION}; font-weight: bold; padding-top: 6px; }}
QLabel#dim      {{ color: {SIRIL_TEXT_DIM}; font-size: 9pt; }}
QLabel#ok       {{ color: {SIRIL_SUCCESS}; font-weight: bold; }}
QLabel#err      {{ color: {SIRIL_ERROR};   font-weight: bold; }}
QLabel#warn     {{ color: {SIRIL_WARNING}; font-weight: bold; }}
QLabel#formula  {{
    background-color: #181c22;
    color: {SIRIL_SUCCESS};
    font-family: 'Courier New', monospace;
    font-size: 9pt;
    padding: 6px 8px;
    border: 1px solid {SIRIL_BORDER};
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
QProgressBar {{
    background-color: {SIRIL_BG3};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    text-align: center;
    color: {SIRIL_TEXT};
}}
QProgressBar::chunk {{
    background-color: {SIRIL_ACCENT2};
    border-radius: 3px;
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# FILTER ALIASES
# ─────────────────────────────────────────────────────────────────────────────
FILTER_ALIASES = {
    "Ha": "Ha", "HA": "Ha", "ha": "Ha",
    "H-a": "Ha", "H_a": "Ha", "Halpha": "Ha",
    "H-alpha": "Ha", "Hα": "Ha",
    "OIII": "OIII", "oiii": "OIII", "O3": "OIII",
    "O-III": "OIII", "OIII_500": "OIII",
    "SII": "SII", "sii": "SII", "S2": "SII",
    "S-II": "SII",
    "L": "L", "Lum": "L", "Lum_": "L",
    "Hb": "Hb", "HBeta": "Hb",
    "NII": "NII", "N2": "NII",
    "R": "R", "G": "G", "B": "B",
}

VAR_MAP = {
    "Ha":   "ha",
    "OIII": "oiii",
    "SII":  "sii",
    "L":    "lum",
    "Hb":   "hb",
    "NII":  "nii",
    "R":    "red",
    "G":    "grn",
    "B":    "blu",
}

ALL_FILTERS = ["Ha", "OIII", "SII", "L", "Hb", "NII"]

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
PALETTES = [
    {
        "name":        "SHO  (Hubble Palette)",
        "id":          "SHO",
        "required":    ["Ha", "OIII", "SII"],
        "optional":    [],
        "description": "SII=R  Ha=G  OIII=B  —  Hubble Space Telescope classic",
        "colour_tags": ["#cc3333", "#aacc44", "#4488ff"],
        "formula_r":   "$SII",
        "formula_g":   "$Ha",
        "formula_b":   "$OIII",
        "notes":       "The iconic Hubble palette. Produces rich golds and teals.\n"
                       "Sulphur-II maps to red, ionised hydrogen to green,\n"
                       "doubly-ionised oxygen to blue.\n"
                       "Best for emission nebulae with all three filters.",
    },
    {
        "name":        "HOO  (2-filter popular)",
        "id":          "HOO",
        "required":    ["Ha", "OIII"],
        "optional":    [],
        "description": "Ha=R  OIII=G  OIII=B  —  most popular 2-filter palette",
        "colour_tags": ["#cc3333", "#44aacc", "#44aacc"],
        "formula_r":   "$Ha",
        "formula_g":   "$OIII",
        "formula_b":   "$OIII",
        "notes":       "The most widely used 2-filter palette.\n"
                       "Ha emission regions appear red/orange.\n"
                       "OIII emission regions appear cyan/teal.\n"
                       "Perfect when you only have Ha and OIII.",
    },
    {
        "name":        "HOS",
        "id":          "HOS",
        "required":    ["Ha", "OIII", "SII"],
        "optional":    [],
        "description": "Ha=R  OIII=G  SII=B  —  natural-looking colours",
        "colour_tags": ["#cc3333", "#44aacc", "#cc6622"],
        "formula_r":   "$Ha",
        "formula_g":   "$OIII",
        "formula_b":   "$SII",
        "notes":       "More natural-looking than SHO for some objects.\n"
                       "Ha in red feels intuitive.\n"
                       "Good for nebulae where OIII is the dominant emission.",
    },
    {
        "name":        "OHS",
        "id":          "OHS",
        "required":    ["Ha", "OIII", "SII"],
        "optional":    [],
        "description": "OIII=R  Ha=G  SII=B  —  cool blue dominant",
        "colour_tags": ["#44aacc", "#aacc44", "#cc6622"],
        "formula_r":   "$OIII",
        "formula_g":   "$Ha",
        "formula_b":   "$SII",
        "notes":       "OIII-dominant palette. Blue/teal objects stand out.\n"
                       "Less common but striking for OIII-bright nebulae.",
    },
    {
        "name":        "Foraxx  (modified SHO)",
        "id":          "Foraxx",
        "required":    ["Ha", "OIII", "SII"],
        "optional":    [],
        "description": "Modified SHO with boosted greens — more natural look",
        "colour_tags": ["#cc4422", "#66cc66", "#3399cc"],
        "formula_r":   "0.8*$SII + 0.2*$Ha",
        "formula_g":   "0.7*$Ha + 0.3*$OIII",
        "formula_b":   "$OIII",
        "notes":       "The Foraxx palette was developed by astrophotographer\n"
                       "Travis Rector to produce more natural-looking colours\n"
                       "than the standard SHO Hubble palette.\n"
                       "Mixes SII+Ha in red, Ha+OIII in green.\n"
                       "Produces rich greens and avoids the 'cartoon' look\n"
                       "of pure SHO on some objects.",
    },
    {
        "name":        "HOO+Ha  (enhanced red)",
        "id":          "HOO_Ha",
        "required":    ["Ha", "OIII"],
        "optional":    [],
        "description": "Ha=R  OIII=G  0.5*Ha+0.5*OIII=B  —  warmer HOO",
        "colour_tags": ["#cc3333", "#44aacc", "#7755aa"],
        "formula_r":   "$Ha",
        "formula_g":   "$OIII",
        "formula_b":   "0.5*$Ha + 0.5*$OIII",
        "notes":       "HOO variant with a blue channel that blends both filters.\n"
                       "Produces warmer, more varied colours than pure HOO.\n"
                       "Purple/violet tones in transition zones between\n"
                       "Ha and OIII emission.",
    },
    {
        "name":        "CFHT  (3-filter natural)",
        "id":          "CFHT",
        "required":    ["Ha", "OIII", "SII"],
        "optional":    [],
        "description": "SII=R  0.5*Ha+0.5*SII=G  OIII=B  —  CFHT variant",
        "colour_tags": ["#cc4422", "#aa8844", "#4488ff"],
        "formula_r":   "$SII",
        "formula_g":   "0.5*$Ha + 0.5*$SII",
        "formula_b":   "$OIII",
        "notes":       "Canada-France-Hawaii Telescope palette variant.\n"
                       "Less saturated than SHO, smoother colour transitions.\n"
                       "Good for wide-field Milky Way and ISM work.",
    },
    {
        "name":        "Ha+OIII  BiColor",
        "id":          "BiColor",
        "required":    ["Ha", "OIII"],
        "optional":    [],
        "description": "Ha=R+0.3G  OIII=G+B  —  classic bicolour",
        "colour_tags": ["#cc4422", "#44aacc", "#44aacc"],
        "formula_r":   "$Ha",
        "formula_g":   "0.3*$Ha + 0.7*$OIII",
        "formula_b":   "$OIII",
        "notes":       "Classic bicolour palette popular for Ha+OIII imaging.\n"
                       "Mixes a small amount of Ha into the green channel\n"
                       "to reduce the abrupt transition between red and cyan.\n"
                       "Produces warm orange-red Ha regions and cool blue OIII.",
    },
    {
        "name":        "SHO + Luminance",
        "id":          "SHO_L",
        "required":    ["Ha", "OIII", "SII"],
        "optional":    ["L"],
        "description": "SHO palette with luminance boost (LRGB blend)",
        "colour_tags": ["#cc3333", "#aacc44", "#4488ff"],
        "formula_r":   "0.7*$SII + 0.3*$L",
        "formula_g":   "0.7*$Ha  + 0.3*$L",
        "formula_b":   "0.7*$OIII + 0.3*$L",
        "notes":       "SHO palette with luminance channel blended in.\n"
                       "Boosts overall sharpness and dynamic range.\n"
                       "Luminance channel adds star detail and fine structure\n"
                       "while preserving narrowband colour mapping.\n"
                       "Requires a separate L (luminance/broadband) master.",
    },
    {
        "name":        "Custom",
        "id":          "Custom",
        "required":    [],
        "optional":    ["Ha", "OIII", "SII", "L", "Hb", "NII"],
        "description": "Define your own R, G, B expressions",
        "colour_tags": ["#888888", "#888888", "#888888"],
        "formula_r":   "",
        "formula_g":   "",
        "formula_b":   "",
        "notes":       "Enter your own Siril pixelmath expressions.\n"
                       "Use $Ha, $OIII, $SII, $L, $Hb, $NII\n"
                       "as variable names for each filter master.\n"
                       "Example: 0.8*$Ha + 0.2*$SII",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# FILTER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_filter_masters(folder: str) -> dict:
    files = (glob.glob(os.path.join(folder, "*.fit"))  +
             glob.glob(os.path.join(folder, "*.fits")) +
             glob.glob(os.path.join(folder, "*.fts"))  +
             glob.glob(os.path.join(folder, "*.FIT"))  +
             glob.glob(os.path.join(folder, "*.FITS")))
    detected = {}
    for path in files:
        filename = os.path.splitext(os.path.basename(path))[0]
        matched = None
        for alias, canonical in FILTER_ALIASES.items():
            if alias.lower() in filename.lower():
                matched = canonical
                break
        if matched is None:
            try:
                hdr = astropy_fits.getheader(path)
                filter_kw = (hdr.get("FILTER", "") or
                             hdr.get("FILTNAM", "") or
                             hdr.get("FILTER1", "")).strip()
                if filter_kw:
                    for alias, canonical in FILTER_ALIASES.items():
                        if alias.lower() == filter_kw.lower():
                            matched = canonical
                            break
            except Exception:
                pass
        if matched and matched not in detected:
            detected[matched] = path
    return detected


def get_filters_from_profile(profile_name: str) -> list:
    if not HAS_EQUIPMENT_MANAGER:
        return []
    p = get_profile(profile_name)
    if not p:
        return []
    raw_filters = p.get("filters", [])
    canonical = []
    for f in raw_filters:
        for alias, canon in FILTER_ALIASES.items():
            if alias.lower() == f.lower() or alias.lower() in f.lower():
                if canon not in canonical:
                    canonical.append(canon)
                break
    return canonical


# ─────────────────────────────────────────────────────────────────────────────
# PALETTE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_eligible_palettes(available_filters: list) -> list:
    eligible = []
    available = set(available_filters)
    for palette in PALETTES:
        if palette["id"] == "Custom":
            eligible.append(palette)
            continue
        required = set(palette["required"])
        if required.issubset(available):
            eligible.append(palette)
    return eligible


def substitute_vars(expr: str) -> str:
    """Replace canonical $FilterName with $varname in a formula."""
    for canon, var in VAR_MAP.items():
        expr = expr.replace(f"${canon}", f"${var}")
    return expr


def build_copy_commands(palette: dict, filter_paths: dict, output_name: str) -> str:
    lines = [
        f"# {palette['name']} palette — generated by nb_palette.py",
        "# Paste into Siril command window or save as .ssf script",
        "",
    ]
    for canon, path in filter_paths.items():
        var = VAR_MAP.get(canon, canon.lower())
        lines.append(f'load "{path}"')
    r_expr = substitute_vars(palette["formula_r"])
    g_expr = substitute_vars(palette["formula_g"])
    b_expr = substitute_vars(palette["formula_b"])
    pm_expr = f"rgb({r_expr}, {g_expr}, {b_expr})"
    lines.append(f'pm "{pm_expr}"')
    lines.append(f'save "{output_name}"')
    return "\n".join(lines)


def validate_custom_palette(r_expr: str, g_expr: str, b_expr: str,
                             filter_paths: dict) -> tuple:
    """Returns (is_valid: bool, error_msg: str)"""
    available_vars = {VAR_MAP.get(k, k.lower()) for k in filter_paths}
    for label, expr in [("R", r_expr), ("G", g_expr), ("B", b_expr)]:
        if not expr.strip():
            return False, f"{label} expression is empty"
        import re
        refs = re.findall(r'\$([a-zA-Z]+)', expr)
        for ref in refs:
            if ref not in available_vars:
                return False, f"${ref} in {label} expression — filter not loaded"
        try:
            depth = 0
            for ch in expr:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth < 0:
                    return False, f"{label}: unmatched parenthesis"
            if depth != 0:
                return False, f"{label}: unmatched parenthesis"
        except Exception as e:
            return False, str(e)
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# SIRIL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def apply_palette_in_siril(palette: dict, filter_paths: dict,
                            output_dir: str, output_name: str,
                            log_callback=None) -> str:
    temp_dir = os.path.join(output_dir, "_nb_palette_temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        siril = s.SirilInterface()
        siril.connect()
        siril.cmd("cd", temp_dir)

        for canon, src_path in filter_paths.items():
            var  = VAR_MAP.get(canon, canon.lower())
            dest = os.path.join(temp_dir, f"{var}.fit")
            shutil.copy2(src_path, dest)
            siril.cmd("load", f"{var}.fit")
            if log_callback:
                log_callback(f"  Loaded ${var} ← {os.path.basename(src_path)}")

        r_expr = substitute_vars(palette["formula_r"])
        g_expr = substitute_vars(palette["formula_g"])
        b_expr = substitute_vars(palette["formula_b"])

        pm_expr = f"rgb({r_expr}, {g_expr}, {b_expr})"
        if log_callback:
            log_callback(f"  pm: {pm_expr}")
        siril.cmd("pm", f'"{pm_expr}"')

        out_path = os.path.join(output_dir, output_name)
        siril.cmd("save", out_path)
        siril.cmd("load", out_path + ".fit")

        if log_callback:
            log_callback(f"  Saved: {out_path}.fit")

        return out_path + ".fit"

    finally:
        try:
            siril.disconnect()
        except Exception:
            pass
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────
class PaletteWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg          = self.config
            palette      = cfg["palette"]
            filter_paths = cfg["filter_paths"]
            output_dir   = cfg["output_dir"]
            output_name  = cfg.get("output_name", f"palette_{palette['id']}")

            self.log_line.emit(f"─── Applying palette: {palette['name']} ───")
            self.log_line.emit(f"  Filters: {list(filter_paths.keys())}")
            self.log_line.emit(f"  Output: {output_name}")
            self.progress.emit(1, 3, "Loading filter masters...")

            if self._cancel.is_set():
                self._abort()
                return

            self.progress.emit(2, 3, f"Applying {palette['name']}...")

            out_path = apply_palette_in_siril(
                palette      = palette,
                filter_paths = filter_paths,
                output_dir   = output_dir,
                output_name  = output_name,
                log_callback = self.log_line.emit
            )

            self.progress.emit(3, 3, "Done")
            self.log_line.emit(f"✓ Result: {out_path}")
            self.finished.emit({
                "success":     True,
                "output_path": out_path,
                "palette":     palette["name"],
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()


# ─────────────────────────────────────────────────────────────────────────────
# PALETTE SWATCH WIDGET
# ─────────────────────────────────────────────────────────────────────────────
class PaletteSwatch(QWidget):
    def __init__(self, r_hex: str, g_hex: str, b_hex: str, parent=None):
        super().__init__(parent)
        self.r_color = QColor(r_hex)
        self.g_color = QColor(g_hex)
        self.b_color = QColor(b_hex)
        self.setFixedSize(48, 18)
        self.setToolTip(f"R: {r_hex}  G: {g_hex}  B: {b_hex}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        bar_w = w // 3
        painter.fillRect(QRect(0,       0, bar_w,       h), self.r_color)
        painter.fillRect(QRect(bar_w,   0, bar_w,       h), self.g_color)
        painter.fillRect(QRect(2*bar_w, 0, w - 2*bar_w, h), self.b_color)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌈  Narrowband Palette System  —  Siril")
        self.setMinimumSize(780, 680)

        # State
        self.filter_paths: dict  = {}      # {canonical: path}
        self.selected_palette    = None
        self._worker             = None
        self._cancel_event       = threading.Event()
        self._palette_buttons    = []      # list of (QPushButton, palette_dict)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Title bar ──
        title_bar = QHBoxLayout()
        lbl_title = QLabel("🌈  Narrowband Palette System  —  Siril")
        lbl_title.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:12pt; font-weight:bold;")
        lbl_ver   = QLabel("v1.0")
        lbl_ver.setObjectName("dim")
        title_bar.addWidget(lbl_title)
        title_bar.addStretch()
        title_bar.addWidget(lbl_ver)
        root.addLayout(title_bar)

        # ── Tab widget ──
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self._build_tab_input()
        self._build_tab_palette()
        self._build_tab_log()

        # ── Bottom controls ──
        bottom = QHBoxLayout()
        self.btn_apply = QPushButton("▶  Apply palette")
        self.btn_apply.setObjectName("primary")
        self.btn_apply.setMinimumHeight(34)
        self.btn_apply.clicked.connect(self._on_apply)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 3)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(160)

        bottom.addWidget(self.btn_apply, stretch=1)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.progress_bar)
        root.addLayout(bottom)

        # ── Status bar ──
        self.status_label = QLabel("Ready — select input folder and detect filters")
        self.status_label.setObjectName("dim")
        root.addWidget(self.status_label)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: INPUT
    # ──────────────────────────────────────────────────────────────────────────
    def _build_tab_input(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ── Equipment profile group ──
        grp_profile = QGroupBox("Equipment profile (optional)")
        profile_layout = QVBoxLayout(grp_profile)

        if HAS_EQUIPMENT_MANAGER:
            row = QHBoxLayout()
            self.combo_profile = QComboBox()
            try:
                profiles = load_all_profiles()
                for pname in profiles:
                    self.combo_profile.addItem(pname)
            except Exception:
                pass
            btn_load_profile = QPushButton("Load filters from profile")
            btn_load_profile.clicked.connect(self._on_load_profile)
            row.addWidget(QLabel("Profile:"))
            row.addWidget(self.combo_profile, stretch=1)
            row.addWidget(btn_load_profile)
            profile_layout.addLayout(row)
            self.lbl_profile_filters = QLabel("")
            self.lbl_profile_filters.setObjectName("dim")
            profile_layout.addWidget(self.lbl_profile_filters)
        else:
            warn = QLabel("equipment_manager.py not found — profile loading unavailable")
            warn.setObjectName("warn")
            profile_layout.addWidget(warn)

        layout.addWidget(grp_profile)

        # ── Master FITS group ──
        grp_fits = QGroupBox("Master FITS files")
        fits_layout = QVBoxLayout(grp_fits)

        folder_row = QHBoxLayout()
        self.edit_folder = QLineEdit()
        self.edit_folder.setPlaceholderText("Folder containing master FITS files...")
        btn_browse_folder = QPushButton("Browse...")
        btn_browse_folder.clicked.connect(self._on_browse_folder)
        folder_row.addWidget(self.edit_folder, stretch=1)
        folder_row.addWidget(btn_browse_folder)
        fits_layout.addLayout(folder_row)

        btn_detect = QPushButton("🔍  Auto-detect filters")
        btn_detect.clicked.connect(self._on_detect_filters)
        fits_layout.addWidget(btn_detect)

        # Filter table
        self.filter_table = QTableWidget(len(ALL_FILTERS), 3)
        self.filter_table.setHorizontalHeaderLabels(["Filter", "File", "Status"])
        self.filter_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.filter_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.filter_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents)
        self.filter_table.setAlternatingRowColors(True)
        self.filter_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.filter_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.filter_table.setMaximumHeight(230)
        self._init_filter_table()
        fits_layout.addWidget(self.filter_table)

        self.lbl_filter_summary = QLabel("No filters detected yet")
        self.lbl_filter_summary.setObjectName("dim")
        fits_layout.addWidget(self.lbl_filter_summary)

        layout.addWidget(grp_fits)

        # ── Output group ──
        grp_out = QGroupBox("Output")
        out_layout = QFormLayout(grp_out)

        out_folder_row = QHBoxLayout()
        self.edit_out_folder = QLineEdit()
        self.edit_out_folder.setPlaceholderText("Defaults to input folder")
        btn_out_browse = QPushButton("Browse...")
        btn_out_browse.clicked.connect(self._on_browse_out_folder)
        out_folder_row.addWidget(self.edit_out_folder, stretch=1)
        out_folder_row.addWidget(btn_out_browse)

        self.edit_prefix = QLineEdit("palette")

        out_layout.addRow("Output folder:", out_folder_row)
        out_layout.addRow("Filename prefix:", self.edit_prefix)

        layout.addWidget(grp_out)
        layout.addStretch()

        self.tabs.addTab(widget, "📁  Input")

    def _init_filter_table(self):
        for row, fname in enumerate(ALL_FILTERS):
            # Filter name cell
            item_name = QTableWidgetItem(fname)
            item_name.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_name.setForeground(QColor(SIRIL_TEXT))
            self.filter_table.setItem(row, 0, item_name)

            # File path cell — editable via button
            item_file = QTableWidgetItem("—")
            item_file.setForeground(QColor(SIRIL_TEXT_DIM))
            self.filter_table.setItem(row, 1, item_file)

            # Status cell
            item_status = QTableWidgetItem("— Not found")
            item_status.setForeground(QColor(SIRIL_TEXT_DIM))
            item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.filter_table.setItem(row, 2, item_status)

    def _refresh_filter_table(self):
        for row, fname in enumerate(ALL_FILTERS):
            if fname in self.filter_paths:
                path = self.filter_paths[fname]
                basename = os.path.basename(path)
                self.filter_table.item(row, 1).setText(basename)
                self.filter_table.item(row, 1).setToolTip(path)
                self.filter_table.item(row, 1).setForeground(QColor(SIRIL_TEXT))
                self.filter_table.item(row, 2).setText("✓ Found")
                self.filter_table.item(row, 2).setForeground(QColor(SIRIL_SUCCESS))
            else:
                self.filter_table.item(row, 1).setText("—")
                self.filter_table.item(row, 1).setToolTip("")
                self.filter_table.item(row, 1).setForeground(QColor(SIRIL_TEXT_DIM))
                self.filter_table.item(row, 2).setText("— Not found")
                self.filter_table.item(row, 2).setForeground(QColor(SIRIL_TEXT_DIM))

        found    = [f for f in ALL_FILTERS if f in self.filter_paths]
        missing  = [f for f in ALL_FILTERS if f not in self.filter_paths]
        found_str   = ", ".join(found)   if found   else "none"
        missing_str = ", ".join(missing) if missing else "none"
        self.lbl_filter_summary.setText(
            f"Found: {found_str}   |   Missing: {missing_str}")

        self._refresh_palette_tab()

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: PALETTE
    # ──────────────────────────────────────────────────────────────────────────
    def _build_tab_palette(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.lbl_palette_status = QLabel(
            "Available palettes with current filters:")
        self.lbl_palette_status.setObjectName("section")
        layout.addWidget(self.lbl_palette_status)

        # Palette button grid container
        self.palette_grid_widget = QWidget()
        self.palette_grid_layout = QGridLayout(self.palette_grid_widget)
        self.palette_grid_layout.setSpacing(6)
        layout.addWidget(self.palette_grid_widget)

        # ── Details group ──
        grp_details = QGroupBox("Selected palette details")
        details_layout = QVBoxLayout(grp_details)

        self.lbl_palette_name = QLabel("—")
        self.lbl_palette_name.setStyleSheet(
            f"color:{SIRIL_TEXT}; font-size:11pt; font-weight:bold;")
        self.lbl_palette_desc = QLabel("")
        self.lbl_palette_desc.setObjectName("dim")
        self.lbl_palette_notes = QLabel("")
        self.lbl_palette_notes.setObjectName("dim")
        self.lbl_palette_notes.setWordWrap(True)

        details_layout.addWidget(self.lbl_palette_name)
        details_layout.addWidget(self.lbl_palette_desc)
        details_layout.addWidget(self.lbl_palette_notes)

        # Formula rows
        form_layout = QFormLayout()
        self.lbl_r_formula = QLabel("")
        self.lbl_r_formula.setObjectName("formula")
        self.lbl_r_formula.setWordWrap(True)
        self.lbl_g_formula = QLabel("")
        self.lbl_g_formula.setObjectName("formula")
        self.lbl_g_formula.setWordWrap(True)
        self.lbl_b_formula = QLabel("")
        self.lbl_b_formula.setObjectName("formula")
        self.lbl_b_formula.setWordWrap(True)

        r_lbl = QLabel("R =")
        r_lbl.setStyleSheet(f"color:#cc4444; font-weight:bold;")
        g_lbl = QLabel("G =")
        g_lbl.setStyleSheet(f"color:#44aa44; font-weight:bold;")
        b_lbl = QLabel("B =")
        b_lbl.setStyleSheet(f"color:#4488ff; font-weight:bold;")

        form_layout.addRow(r_lbl, self.lbl_r_formula)
        form_layout.addRow(g_lbl, self.lbl_g_formula)
        form_layout.addRow(b_lbl, self.lbl_b_formula)
        details_layout.addLayout(form_layout)

        self.btn_copy = QPushButton("📋  Copy Siril commands")
        self.btn_copy.setObjectName("copy")
        self.btn_copy.clicked.connect(self._on_copy_commands)
        details_layout.addWidget(self.btn_copy)

        layout.addWidget(grp_details)

        # ── Custom palette group ──
        self.grp_custom = QGroupBox("Custom palette")
        custom_layout = QFormLayout(self.grp_custom)

        self.edit_custom_r = QLineEdit()
        self.edit_custom_r.setPlaceholderText("e.g. $SII")
        self.edit_custom_g = QLineEdit()
        self.edit_custom_g.setPlaceholderText("e.g. 0.7*$Ha + 0.3*$OIII")
        self.edit_custom_b = QLineEdit()
        self.edit_custom_b.setPlaceholderText("e.g. $OIII")

        custom_layout.addRow("R expression:", self.edit_custom_r)
        custom_layout.addRow("G expression:", self.edit_custom_g)
        custom_layout.addRow("B expression:", self.edit_custom_b)

        hint = QLabel("Use $Ha  $OIII  $SII  $L  $Hb  $NII")
        hint.setObjectName("dim")
        custom_layout.addRow("", hint)

        self.lbl_custom_error = QLabel("")
        self.lbl_custom_error.setObjectName("err")
        custom_layout.addRow("", self.lbl_custom_error)

        self.grp_custom.setVisible(False)
        layout.addWidget(self.grp_custom)
        layout.addStretch()

        scroll.setWidget(widget)
        self.tabs.addTab(scroll, "🌈  Palette")

        # Build initial palette grid
        self._refresh_palette_tab()

    def _refresh_palette_tab(self):
        available = list(self.filter_paths.keys())
        eligible  = get_eligible_palettes(available)

        # Clear existing buttons
        for btn, _ in self._palette_buttons:
            btn.setParent(None)
        self._palette_buttons.clear()

        # Add all palettes (enabled if eligible, disabled if not)
        all_shown = []
        for p in PALETTES:
            required = set(p["required"])
            is_available = required.issubset(set(available))
            all_shown.append((p, is_available))

        row_idx = 0
        col_idx = 0
        for p, is_available in all_shown:
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(6)

            swatch = PaletteSwatch(*p["colour_tags"])
            cell_layout.addWidget(swatch)

            btn = QPushButton(f"{p['name']}\n{p['description']}")
            btn.setObjectName("palette")
            btn.setCheckable(True)
            btn.setEnabled(is_available)
            if not is_available:
                missing = set(p["required"]) - set(available)
                btn.setToolTip(f"Missing: {', '.join(missing)}")
                btn.setStyleSheet(f"color: {SIRIL_TEXT_DIM};")
            btn.clicked.connect(lambda checked, pal=p: self._on_palette_selected(pal))
            cell_layout.addWidget(btn, stretch=1)

            self.palette_grid_layout.addWidget(cell, row_idx, col_idx)
            self._palette_buttons.append((btn, p))

            col_idx += 1
            if col_idx >= 2:
                col_idx = 0
                row_idx += 1

        # If currently selected palette is no longer available, deselect
        if self.selected_palette:
            still_ok = any(
                p["id"] == self.selected_palette["id"] and avail
                for p, avail in all_shown
            )
            if not still_ok:
                self.selected_palette = None
                self.lbl_palette_name.setText("—")
                self.lbl_palette_desc.setText("")
                self.lbl_palette_notes.setText("")
                self.lbl_r_formula.setText("")
                self.lbl_g_formula.setText("")
                self.lbl_b_formula.setText("")

        n = len(available)
        self.lbl_palette_status.setText(
            f"Available palettes with current filters  ({n} filter(s) loaded):")

    def _on_palette_selected(self, palette: dict):
        self.selected_palette = palette

        # Update checkmarks — only one checked at a time
        for btn, p in self._palette_buttons:
            btn.setChecked(p["id"] == palette["id"])

        # Update details panel
        self.lbl_palette_name.setText(palette["name"])
        self.lbl_palette_desc.setText(palette["description"])
        self.lbl_palette_notes.setText(palette["notes"])
        self.lbl_r_formula.setText(palette["formula_r"] or "(enter below)")
        self.lbl_g_formula.setText(palette["formula_g"] or "(enter below)")
        self.lbl_b_formula.setText(palette["formula_b"] or "(enter below)")

        # Show/hide custom group
        self.grp_custom.setVisible(palette["id"] == "Custom")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: LOG
    # ──────────────────────────────────────────────────────────────────────────
    def _build_tab_log(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Processing log will appear here...")
        layout.addWidget(self.log_view)

        self.tabs.addTab(widget, "📋  Log")

    def _append_log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}]  {text}")
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ──────────────────────────────────────────────────────────────────────────
    # BUTTON HANDLERS
    # ──────────────────────────────────────────────────────────────────────────
    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder with master FITS files")
        if folder:
            self.edit_folder.setText(folder)
            if not self.edit_out_folder.text():
                self.edit_out_folder.setText(folder)

    def _on_browse_out_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder")
        if folder:
            self.edit_out_folder.setText(folder)

    def _on_detect_filters(self):
        folder = self.edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self._set_status("⚠  Please enter a valid input folder first", SIRIL_WARNING)
            return
        self._append_log(f"Scanning: {folder}")
        detected = detect_filter_masters(folder)
        self.filter_paths = detected
        self._refresh_filter_table()
        n = len(detected)
        self._append_log(f"Found {n} filter master(s): {list(detected.keys())}")
        self._set_status(f"Detected {n} filters: {', '.join(detected.keys())}")

        if not self.edit_out_folder.text():
            self.edit_out_folder.setText(folder)

    def _on_load_profile(self):
        if not HAS_EQUIPMENT_MANAGER:
            return
        pname = self.combo_profile.currentText()
        filters = get_filters_from_profile(pname)
        self.lbl_profile_filters.setText(
            f"Profile filters: {', '.join(filters) if filters else 'none found'}")
        self._append_log(
            f"Profile '{pname}' filters: {filters}")

    def _on_copy_commands(self):
        if not self.selected_palette:
            return

        palette = self._get_active_palette()
        if palette is None:
            return

        fp = {k: v for k, v in self.filter_paths.items()
              if k in palette["required"] + palette["optional"]}
        prefix = self.edit_prefix.text().strip() or "palette"
        out_name = f"{prefix}_{palette['id']}"
        text = build_copy_commands(palette, fp, out_name)
        QApplication.clipboard().setText(text)

        orig = self.btn_copy.text()
        self.btn_copy.setText("✓  Copied!")
        QTimer.singleShot(1500, lambda: self.btn_copy.setText(orig))

    def _on_apply(self):
        # Validate inputs
        if not self.selected_palette:
            self._set_status("⚠  Please select a palette first", SIRIL_WARNING)
            return

        palette = self._get_active_palette()
        if palette is None:
            return

        folder = self.edit_folder.text().strip()
        if not folder:
            folder = self.edit_out_folder.text().strip()
        out_dir = self.edit_out_folder.text().strip() or folder
        if not out_dir:
            self._set_status("⚠  Please set an output folder", SIRIL_WARNING)
            return

        required_filters = palette["required"]
        missing = [f for f in required_filters if f not in self.filter_paths]
        if missing:
            self._set_status(
                f"⚠  Missing filters: {', '.join(missing)}", SIRIL_WARNING)
            return

        prefix = self.edit_prefix.text().strip() or "palette"
        out_name = f"{prefix}_{palette['id']}"

        # Build filter_paths subset
        needed = palette["required"] + palette.get("optional", [])
        filter_paths = {k: v for k, v in self.filter_paths.items()
                        if k in needed}

        config = {
            "palette":      palette,
            "filter_paths": filter_paths,
            "output_dir":   out_dir,
            "output_name":  out_name,
        }

        self._cancel_event.clear()
        self._worker = PaletteWorker(config, self._cancel_event)
        self._worker.log_line.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)

        self.btn_apply.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.tabs.setCurrentIndex(2)  # switch to log tab
        self._set_status("Processing…")

        self._worker.start()

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._append_log("⚠  Cancel requested…")
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, val: int, total: int, msg: str):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(val)
        self._set_status(msg)

    def _on_done(self, result: dict):
        self.btn_apply.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

        if result.get("success"):
            out = result.get("output_path", "")
            self._set_status(
                f"✓  Palette applied  |  Result loaded in Siril  |  {os.path.basename(out)}",
                SIRIL_SUCCESS)
            self._append_log(f"✓ Done — {result.get('palette')} → {out}")
        else:
            err = result.get("error", "Unknown error")
            self._set_status(f"✗  Error: {err}", SIRIL_ERROR)

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self.status_label.setText(msg)
        self.status_label.setStyleSheet(f"color: {color};")

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────
    def _get_active_palette(self):
        """Return the effective palette dict, patching custom formulas if needed."""
        palette = dict(self.selected_palette)

        if palette["id"] == "Custom":
            r = self.edit_custom_r.text().strip()
            g = self.edit_custom_g.text().strip()
            b = self.edit_custom_b.text().strip()
            valid, err = validate_custom_palette(r, g, b, self.filter_paths)
            if not valid:
                self.lbl_custom_error.setText(f"⚠  {err}")
                self._set_status(f"⚠  Custom palette error: {err}", SIRIL_WARNING)
                return None
            self.lbl_custom_error.setText("")
            palette["formula_r"] = r
            palette["formula_g"] = g
            palette["formula_b"] = b
            # Update display labels
            self.lbl_r_formula.setText(r)
            self.lbl_g_formula.setText(g)
            self.lbl_b_formula.setText(b)

        return palette


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
