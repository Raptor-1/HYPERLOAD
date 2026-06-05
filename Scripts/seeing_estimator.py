"""
seeing_estimator.py — Script #18
Siril Atmospheric Seeing / Fried Parameter Estimator
Measures atmospheric seeing quality from a FITS image sequence.
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("photutils")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")

import os
import sys
import glob
import csv
import math
import threading
import warnings
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QPushButton, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QPlainTextEdit, QFileDialog, QScrollArea, QFormLayout,
    QSizePolicy, QFrame, QMessageBox
)
from PyQt6.QtGui import QColor, QFont

# ── Equipment manager (optional) ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ── Theme ────────────────────────────────────────────────────────────────────
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
    text-align: center;
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
QLabel#section {{ color: {SIRIL_SECTION}; font-weight: bold; padding-top: 6px; }}
QLabel#dim     {{ color: {SIRIL_TEXT_DIM}; font-size: 9pt; }}
QLabel#ok      {{ color: {SIRIL_SUCCESS}; font-weight: bold; }}
QLabel#err     {{ color: {SIRIL_ERROR};   font-weight: bold; }}
QLabel#warn    {{ color: {SIRIL_WARNING}; font-weight: bold; }}
QLabel#value   {{
    color: {SIRIL_ACCENT};
    font-size: 18pt;
    font-weight: bold;
    padding: 4px;
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
"""

# ── Physics constants ─────────────────────────────────────────────────────────
WAVELENGTHS = {
    "Broadband (550nm)": 550.0,
    "Ha (656nm)":        656.0,
    "OIII (501nm)":      501.0,
    "SII (672nm)":       672.0,
    "Lum (550nm)":       550.0,
    "R (640nm)":         640.0,
    "G (540nm)":         540.0,
    "B (460nm)":         460.0,
}

# ── Core science functions ────────────────────────────────────────────────────

def detect_stars(image: np.ndarray,
                 fwhm_guess: float = 5.0,
                 threshold_sigma: float = 5.0,
                 border_margin: int = 20) -> list:
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    if std <= 0:
        return []
    daofind = DAOStarFinder(
        fwhm=fwhm_guess,
        threshold=threshold_sigma * std,
        sharplo=0.2, sharphi=1.0,
        roundlo=-1.0, roundhi=1.0
    )
    sources = daofind(image - median)
    if sources is None or len(sources) == 0:
        return []

    h, w = image.shape
    result = []
    for src in sources:
        x, y = float(src["xcentroid"]), float(src["ycentroid"])
        if (x < border_margin or x > w - border_margin or
                y < border_margin or y > h - border_margin):
            continue
        bit_depth = 16
        if float(src["peak"]) > 0.95 * (2**bit_depth - 1):
            continue
        result.append((x, y, float(src["peak"]), float(src["flux"])))
    return result


def fit_star_fwhm(image: np.ndarray,
                  x: float, y: float,
                  box_size: int = 32) -> float | None:
    half = box_size // 2
    y0, x0 = int(round(y)), int(round(x))
    h, w = image.shape

    y1 = max(0, y0 - half)
    y2 = min(h, y0 + half + 1)
    x1 = max(0, x0 - half)
    x2 = min(w, x0 + half + 1)
    cutout = image[y1:y2, x1:x2].astype(float)

    if cutout.shape[0] < 5 or cutout.shape[1] < 5:
        return None

    corners = np.concatenate([
        cutout[:3, :3].ravel(), cutout[:3, -3:].ravel(),
        cutout[-3:, :3].ravel(), cutout[-3:, -3:].ravel()
    ])
    bg = np.median(corners)
    cutout = np.clip(cutout - bg, 0, None)

    peak_row = np.argmax(np.max(cutout, axis=1))
    profile  = cutout[peak_row, :].astype(float)
    if profile.max() <= 0:
        return None

    def gaussian(x, amplitude, center, sigma):
        return amplitude * np.exp(-(x - center)**2 / (2.0 * sigma**2))

    xs  = np.arange(len(profile), dtype=float)
    p0  = [profile.max(), float(np.argmax(profile)), 3.0]
    bounds = ([0, 0, 0.5], [profile.max() * 2, len(profile), half])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(gaussian, xs, profile,
                                p0=p0, bounds=bounds, maxfev=500)
        sigma = abs(popt[2])
        fwhm  = 2.355 * sigma
        if fwhm < 1.0 or fwhm > box_size * 0.8:
            return None
        residuals = profile - gaussian(xs, *popt)
        if np.std(residuals) > 0.2 * popt[0]:
            return None
        return fwhm
    except (RuntimeError, ValueError):
        return None


def fwhm_arcsec_to_r0_cm(fwhm_arcsec: float,
                           wavelength_nm: float = 550.0) -> float:
    wavelength_m = wavelength_nm * 1e-9
    fwhm_rad = fwhm_arcsec * math.pi / (180.0 * 3600.0)
    if fwhm_rad <= 0:
        return 0.0
    r0_m = 0.98 * wavelength_m / fwhm_rad
    return r0_m * 100.0


def r0_to_quality_label(r0_cm: float) -> tuple:
    if r0_cm >= 15.0:
        return "Excellent", SIRIL_SUCCESS
    elif r0_cm >= 10.0:
        return "Good",      SIRIL_ACCENT
    elif r0_cm >= 5.0:
        return "Average",   SIRIL_WARNING
    else:
        return "Poor",      SIRIL_ERROR


def analyze_frame(fits_path: str, pixel_scale: float,
                  wavelength_nm: float = 550.0,
                  fwhm_guess: float = 5.0,
                  threshold_sigma: float = 5.0,
                  box_size: int = 32,
                  min_stars: int = 3,
                  color_handling: str = "Luminance") -> dict | None:
    try:
        data = fits.getdata(fits_path)
    except Exception:
        return None

    if data.ndim == 3:
        if data.shape[0] == 3:
            if color_handling == "Green channel":
                data = data[1].astype(float)
            else:
                data = (0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2])
        else:
            data = data[0]

    data = data.astype(float)
    stars = detect_stars(data, fwhm_guess, threshold_sigma)
    if len(stars) < min_stars:
        return None

    fwhm_values = []
    for x, y, peak, flux in stars:
        fwhm_px = fit_star_fwhm(data, x, y, box_size)
        if fwhm_px is not None:
            fwhm_values.append(fwhm_px)

    if len(fwhm_values) < min_stars:
        return None

    arr  = np.array(fwhm_values)
    med  = np.median(arr)
    mad  = np.median(np.abs(arr - med))
    good = arr[np.abs(arr - med) < 2.0 * mad * 1.4826] if mad > 0 else arr
    if len(good) < 1:
        good = arr

    median_fwhm_px     = float(np.median(good))
    median_fwhm_arcsec = median_fwhm_px * pixel_scale
    r0_cm              = fwhm_arcsec_to_r0_cm(median_fwhm_arcsec, wavelength_nm)
    quality, color     = r0_to_quality_label(r0_cm)

    return {
        "path":          fits_path,
        "filename":      os.path.basename(fits_path),
        "fwhm_px":       round(median_fwhm_px, 2),
        "fwhm_arcsec":   round(median_fwhm_arcsec, 3),
        "r0_cm":         round(r0_cm, 1),
        "star_count":    len(good),
        "quality":       quality,
        "quality_color": color,
    }


# ── Worker thread ─────────────────────────────────────────────────────────────

class SeeingWorker(QThread):
    progress = pyqtSignal(int, int, str)
    result   = pyqtSignal(dict)
    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, fits_files: list, config: dict,
                 cancel_event: threading.Event):
        super().__init__()
        self.fits_files = fits_files
        self.config     = config
        self._cancel    = cancel_event

    def run(self):
        results = []
        total   = len(self.fits_files)

        pixel_scale      = self.config["pixel_scale_arcsec_per_px"]
        wavelength_nm    = self.config["wavelength_nm"]
        threshold_sigma  = self.config.get("threshold_sigma", 5.0)
        fwhm_guess       = self.config.get("fwhm_guess_px", 5.0)
        box_size         = self.config.get("box_size", 32)
        min_stars        = self.config.get("min_stars", 3)
        color_handling   = self.config.get("color_handling", "Luminance")
        sample_every     = max(1, self.config.get("sample_every", 1))

        sampled_files = self.fits_files[::sample_every]
        total_sampled = len(sampled_files)
        skipped       = 0

        self.log_line.emit(
            f"Starting analysis: {total_sampled} frames "
            f"(sampling every {sample_every})"
        )
        self.log_line.emit(
            f"Pixel scale: {pixel_scale:.4f} arcsec/px  "
            f"Wavelength: {wavelength_nm:.0f}nm"
        )
        self.log_line.emit("-" * 60)

        for i, fits_path in enumerate(sampled_files):
            if self._cancel.is_set():
                self.log_line.emit("--- Cancelled by user ---")
                break

            fname = os.path.basename(fits_path)
            self.progress.emit(i + 1, total_sampled, fname)
            self.log_line.emit(f"[{i+1}/{total_sampled}] {fname}")

            r = analyze_frame(
                fits_path, pixel_scale, wavelength_nm,
                fwhm_guess, threshold_sigma, box_size, min_stars,
                color_handling
            )

            if r is None:
                self.log_line.emit("  → No stars detected or fit failed — skipped")
                skipped += 1
                continue

            self.log_line.emit(
                f"  → FWHM: {r['fwhm_arcsec']:.3f}\"  "
                f"r0: {r['r0_cm']:.1f}cm  "
                f"({r['quality']})  "
                f"stars: {r['star_count']}"
            )
            results.append(r)
            self.result.emit(r)

        # Warn if most frames skipped
        if total_sampled > 0 and skipped / total_sampled > 0.5:
            self.log_line.emit("")
            self.log_line.emit(
                "⚠  WARNING: More than 50% of frames had no detectable stars."
            )
            self.log_line.emit(
                "   Try lowering the detection threshold or check image quality."
            )

        self.log_line.emit("-" * 60)

        if results:
            fwhms     = [r["fwhm_arcsec"] for r in results]
            r0s       = [r["r0_cm"]       for r in results]
            threshold = self.config.get("seeing_threshold_arcsec", 3.0)
            usable    = [r for r in results if r["fwhm_arcsec"] <= threshold]

            summary = {
                "success":        True,
                "total_analyzed": len(results),
                "total_files":    total_sampled,
                "skipped":        skipped,
                "best_fwhm":      round(min(fwhms), 3),
                "worst_fwhm":     round(max(fwhms), 3),
                "median_fwhm":    round(float(np.median(fwhms)), 3),
                "best_r0":        round(max(r0s), 1),
                "median_r0":      round(float(np.median(r0s)), 1),
                "usable_count":   len(usable),
                "usable_pct":     round(len(usable) / len(results) * 100, 1),
                "results":        results,
            }
            q, _ = r0_to_quality_label(summary["median_r0"])
            self.log_line.emit(
                f"✓  Done: {len(results)} frames analyzed, {skipped} skipped"
            )
            self.log_line.emit(
                f"   Best FWHM: {summary['best_fwhm']:.3f}\"  "
                f"Best r0: {summary['best_r0']:.1f}cm"
            )
            self.log_line.emit(
                f"   Median FWHM: {summary['median_fwhm']:.3f}\"  "
                f"Median r0: {summary['median_r0']:.1f}cm  "
                f"({q})"
            )
            self.log_line.emit(
                f"   Usable frames: {len(usable)}/{len(results)} "
                f"({summary['usable_pct']:.1f}%)"
            )
        else:
            summary = {
                "success": False,
                "error":   "No frames could be analyzed",
                "results": []
            }
            self.log_line.emit("✗  No frames could be analyzed")

        self.finished.emit(summary)

    def cancel(self):
        self._cancel.set()


# ── Matplotlib histogram ──────────────────────────────────────────────────────

class SeeingHistogram(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3), facecolor=SIRIL_BG)
        self.ax  = self.fig.add_subplot(111)
        self._style_ax()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def _style_ax(self):
        self.ax.set_facecolor(SIRIL_BG2)
        self.ax.tick_params(colors=SIRIL_TEXT, labelsize=8)
        for spine in ["bottom", "left"]:
            self.ax.spines[spine].set_color(SIRIL_BORDER)
        for spine in ["top", "right"]:
            self.ax.spines[spine].set_visible(False)

    def update_plot(self, fwhm_values: list, threshold: float):
        self.ax.clear()
        self._style_ax()

        if not fwhm_values:
            self.ax.text(0.5, 0.5, "No data yet",
                         ha="center", va="center",
                         color=SIRIL_TEXT_DIM, transform=self.ax.transAxes)
            self.draw()
            return

        arr  = np.array(fwhm_values)
        good = arr[arr <= threshold]
        bad  = arr[arr >  threshold]
        bins = max(10, len(arr) // 5)

        if len(good) > 0:
            self.ax.hist(good, bins=bins, color=SIRIL_SUCCESS,
                         alpha=0.85, label=f"Usable ({len(good)})")
        if len(bad) > 0:
            self.ax.hist(bad,  bins=bins, color=SIRIL_ERROR,
                         alpha=0.65, label=f"Poor ({len(bad)})")

        self.ax.axvline(threshold, color=SIRIL_WARNING,
                        linestyle="--", linewidth=1.5,
                        label=f'Threshold ({threshold}")')

        self.ax.set_xlabel("FWHM (arcsec)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_ylabel("Frames",        color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_title("Seeing Distribution",
                           color=SIRIL_SECTION, fontsize=9)
        self.ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                       labelcolor=SIRIL_TEXT, fontsize=7)
        self.fig.tight_layout(pad=1.0)
        self.draw()


# ── Badge widget ──────────────────────────────────────────────────────────────

class BadgeWidget(QWidget):
    def __init__(self, title: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {SIRIL_BG3};
                border: 1px solid {SIRIL_ACCENT2};
                border-radius: 4px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)

        self._title_lbl = QLabel(title)
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_lbl.setObjectName("dim")

        self._value_lbl = QLabel("—")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value_lbl.setObjectName("value")

        self._unit_lbl = QLabel(unit)
        self._unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._unit_lbl.setObjectName("dim")

        lay.addWidget(self._title_lbl)
        lay.addWidget(self._value_lbl)
        lay.addWidget(self._unit_lbl)

    def set_value(self, val: str, color: str = SIRIL_ACCENT):
        self._value_lbl.setText(val)
        self._value_lbl.setStyleSheet(
            f"color: {color}; font-size: 18pt; font-weight: bold; padding: 4px;"
        )


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("👁  Seeing / Fried Parameter Estimator  —  Siril")
        self.setMinimumSize(820, 680)
        self.resize(900, 760)

        self._fits_files: list     = []
        self._results:    list     = []
        self._summary:    dict     = {}
        self._worker:     SeeingWorker | None = None
        self._cancel_ev:  threading.Event     = threading.Event()

        self._build_ui()
        self._load_profiles()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Title bar
        title_row = QHBoxLayout()
        lbl_title = QLabel("👁  Seeing / Fried Parameter Estimator  —  Siril")
        lbl_title.setStyleSheet(
            f"color: {SIRIL_ACCENT}; font-size: 12pt; font-weight: bold;"
        )
        lbl_ver = QLabel("v1.0")
        lbl_ver.setObjectName("dim")
        title_row.addWidget(lbl_title)
        title_row.addStretch()
        title_row.addWidget(lbl_ver)
        root.addLayout(title_row)

        # Tab widget
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._tabs.addTab(self._build_input_tab(),   "📁  Input")
        self._tabs.addTab(self._build_results_tab(), "📊  Results")
        self._tabs.addTab(self._build_export_tab(),  "💾  Export")
        self._tabs.addTab(self._build_log_tab(),     "📋  Log")

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(10)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # Bottom button row
        btn_row = QHBoxLayout()
        self._btn_analyze = QPushButton("▶  Analyze")
        self._btn_analyze.setObjectName("primary")
        self._btn_analyze.setMinimumHeight(34)
        self._btn_analyze.clicked.connect(self._start_analysis)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(34)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_analysis)

        btn_row.addWidget(self._btn_analyze, 3)
        btn_row.addWidget(self._btn_cancel,  1)
        root.addLayout(btn_row)

        # Status bar
        self._status = QLabel("Ready. Select a FITS folder and click Analyze.")
        self._status.setObjectName("dim")
        self._status.setContentsMargins(2, 2, 2, 2)
        root.addWidget(self._status)

    # ── Input tab ────────────────────────────────────────────────────────────

    def _build_input_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner  = QWidget()
        lay    = QVBoxLayout(inner)
        lay.setSpacing(10)
        lay.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(inner)

        # Folder group
        grp_folder = QGroupBox("FITS Sequence Folder")
        g_lay = QVBoxLayout(grp_folder)

        row1 = QHBoxLayout()
        self._edit_folder = QLineEdit()
        self._edit_folder.setPlaceholderText("Path to folder containing .fit / .fits files…")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_folder)
        btn_scan   = QPushButton("🔍  Scan")
        btn_scan.clicked.connect(self._scan_folder)
        row1.addWidget(self._edit_folder, 4)
        row1.addWidget(btn_browse)
        row1.addWidget(btn_scan)
        g_lay.addLayout(row1)

        self._lbl_file_count = QLabel("No folder selected")
        self._lbl_file_count.setObjectName("dim")
        g_lay.addWidget(self._lbl_file_count)
        lay.addWidget(grp_folder)

        # Sample every N
        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample every N frames:"))
        self._spin_sample = QSpinBox()
        self._spin_sample.setRange(1, 100)
        self._spin_sample.setValue(1)
        self._spin_sample.setFixedWidth(70)
        self._spin_sample.setToolTip(
            "1 = analyze all frames. Higher values skip frames for speed."
        )
        sample_row.addWidget(self._spin_sample)
        sample_row.addStretch()
        lay.addLayout(sample_row)

        # Pixel scale group
        grp_scale = QGroupBox("Pixel Scale")
        s_lay = QVBoxLayout(grp_scale)

        self._combo_profile = QComboBox()
        self._combo_profile.addItem("Manual entry")
        self._combo_profile.currentIndexChanged.connect(self._on_profile_changed)
        s_lay.addWidget(QLabel("Equipment profile:"))
        s_lay.addWidget(self._combo_profile)

        row_ps = QHBoxLayout()
        row_ps.addWidget(QLabel("Pixel scale (arcsec/px):"))
        self._spin_scale = QDoubleSpinBox()
        self._spin_scale.setRange(0.01, 20.0)
        self._spin_scale.setSingleStep(0.01)
        self._spin_scale.setValue(1.0)
        self._spin_scale.setDecimals(4)
        row_ps.addWidget(self._spin_scale)
        row_ps.addStretch()
        s_lay.addLayout(row_ps)

        self._lbl_profile_scale = QLabel("")
        self._lbl_profile_scale.setObjectName("dim")
        self._lbl_profile_scale.setVisible(False)
        s_lay.addWidget(self._lbl_profile_scale)

        if not HAS_EQUIPMENT_MANAGER:
            lbl_no_em = QLabel("ℹ  equipment_manager.py not found — using manual entry")
            lbl_no_em.setObjectName("warn")
            s_lay.addWidget(lbl_no_em)

        lay.addWidget(grp_scale)

        # Observation settings group
        grp_obs = QGroupBox("Observation Settings")
        obs_form = QFormLayout(grp_obs)
        obs_form.setSpacing(8)

        self._combo_wavelength = QComboBox()
        for name in WAVELENGTHS:
            self._combo_wavelength.addItem(name)
        obs_form.addRow("Wavelength / filter:", self._combo_wavelength)

        self._spin_threshold_sigma = QDoubleSpinBox()
        self._spin_threshold_sigma.setRange(1.0, 20.0)
        self._spin_threshold_sigma.setValue(5.0)
        self._spin_threshold_sigma.setSingleStep(0.5)
        obs_form.addRow("Detection threshold (σ):", self._spin_threshold_sigma)

        self._spin_min_stars = QSpinBox()
        self._spin_min_stars.setRange(1, 20)
        self._spin_min_stars.setValue(3)
        obs_form.addRow("Min stars per frame:", self._spin_min_stars)

        self._combo_box_size = QComboBox()
        for sz in ["16px", "32px", "64px"]:
            self._combo_box_size.addItem(sz)
        self._combo_box_size.setCurrentIndex(1)  # default 32px
        obs_form.addRow("PSF box size:", self._combo_box_size)

        self._combo_color_handling = QComboBox()
        self._combo_color_handling.addItems(["Luminance", "Green channel"])
        obs_form.addRow("Color FITS handling:", self._combo_color_handling)

        lay.addWidget(grp_obs)

        # Quality threshold group
        grp_thresh = QGroupBox("Quality Threshold")
        t_lay = QVBoxLayout(grp_thresh)
        t_lay.addWidget(QLabel("Mark frames as poor if FWHM exceeds:"))

        slider_row = QHBoxLayout()
        self._slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self._slider_threshold.setRange(5, 100)   # 0.5 to 10.0 (×10)
        self._slider_threshold.setValue(30)        # default 3.0 arcsec
        self._slider_threshold.setTickInterval(5)
        self._slider_threshold.valueChanged.connect(self._on_threshold_changed)

        self._lbl_threshold_val = QLabel('3.0"')
        self._lbl_threshold_val.setMinimumWidth(50)
        slider_row.addWidget(self._slider_threshold)
        slider_row.addWidget(self._lbl_threshold_val)
        t_lay.addLayout(slider_row)

        lbl_hint = QLabel("Frames above threshold shown in red in results table")
        lbl_hint.setObjectName("dim")
        t_lay.addWidget(lbl_hint)
        lay.addWidget(grp_thresh)

        lay.addStretch()
        return scroll

    # ── Results tab ──────────────────────────────────────────────────────────

    def _build_results_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Summary badges
        grp_summary = QGroupBox("Session Summary")
        badge_row   = QHBoxLayout(grp_summary)
        badge_row.setSpacing(8)

        self._badge_best   = BadgeWidget("Best Seeing",   'arcsec')
        self._badge_median = BadgeWidget("Median Seeing", 'arcsec')
        self._badge_r0     = BadgeWidget("Best r0",       'cm')
        self._badge_usable = BadgeWidget("Usable Frames", '')

        for b in [self._badge_best, self._badge_median,
                  self._badge_r0, self._badge_usable]:
            badge_row.addWidget(b)

        lay.addWidget(grp_summary)

        # Histogram
        self._histogram = SeeingHistogram()
        lay.addWidget(self._histogram)

        # Per-frame table
        grp_table = QGroupBox("Per-Frame Results")
        t_lay = QVBoxLayout(grp_table)

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "#", "Filename", "FWHM (px)", 'FWHM (")', "r0 (cm)",
            "Stars", "Quality"
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        t_lay.addWidget(self._table)
        lay.addWidget(grp_table, 1)

        return w

    # ── Export tab ───────────────────────────────────────────────────────────

    def _build_export_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        grp_opts = QGroupBox("Export Options")
        o_lay    = QVBoxLayout(grp_opts)

        self._chk_csv = QCheckBox("Export CSV report")
        self._chk_csv.setChecked(True)
        o_lay.addWidget(self._chk_csv)

        self._chk_rejection = QCheckBox("Export rejection list for Siril")
        self._chk_rejection.setToolTip(
            "Text file listing filenames above the FWHM threshold"
        )
        o_lay.addWidget(self._chk_rejection)

        self._chk_plot = QCheckBox("Save seeing distribution plot as PNG")
        o_lay.addWidget(self._chk_plot)

        out_row = QHBoxLayout()
        self._edit_out_folder = QLineEdit()
        self._edit_out_folder.setPlaceholderText("Output folder (default: same as FITS folder)")
        btn_out_browse = QPushButton("Browse…")
        btn_out_browse.clicked.connect(self._browse_out_folder)
        out_row.addWidget(QLabel("Output folder:"))
        out_row.addWidget(self._edit_out_folder, 3)
        out_row.addWidget(btn_out_browse)
        o_lay.addLayout(out_row)

        lay.addWidget(grp_opts)

        btn_export = QPushButton("💾  Export")
        btn_export.setObjectName("export")
        btn_export.setMinimumHeight(36)
        btn_export.clicked.connect(self._do_export)
        lay.addWidget(btn_export)

        # Siril integration
        grp_siril = QGroupBox("Siril Integration")
        si_lay    = QVBoxLayout(grp_siril)
        si_lay.addWidget(QLabel(
            "Flag poor frames in the currently loaded Siril sequence:"
        ))
        self._btn_flag = QPushButton("🚩  Flag Poor Frames in Siril")
        self._btn_flag.clicked.connect(self._flag_in_siril)
        si_lay.addWidget(self._btn_flag)

        lbl_flag_hint = QLabel(
            "Marks frames with FWHM > threshold as excluded in the Siril sequence"
        )
        lbl_flag_hint.setObjectName("dim")
        si_lay.addWidget(lbl_flag_hint)
        lay.addWidget(grp_siril)

        lay.addStretch()
        return w

    # ── Log tab ──────────────────────────────────────────────────────────────

    def _build_log_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._log.document().setMaximumBlockCount(5000)
        lay.addWidget(self._log)

        btn_save_log = QPushButton("💾  Save Log…")
        btn_save_log.clicked.connect(self._save_log)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_save_log)
        lay.addLayout(btn_row)

        return w

    # ── Profile loading ───────────────────────────────────────────────────────

    def _load_profiles(self):
        if not HAS_EQUIPMENT_MANAGER:
            return
        try:
            profiles = load_all_profiles()
            for name in profiles:
                self._combo_profile.addItem(name)
        except Exception as e:
            self._log_append(f"Could not load equipment profiles: {e}")

    def _on_profile_changed(self, idx: int):
        if idx == 0:
            # Manual
            self._spin_scale.setEnabled(True)
            self._lbl_profile_scale.setVisible(False)
            return

        if not HAS_EQUIPMENT_MANAGER:
            return

        name = self._combo_profile.currentText()
        try:
            p = get_profile(name)
        except Exception:
            return

        scale = None
        if p.get("resolution_arcsec_px"):
            scale = p["resolution_arcsec_px"]
        elif p.get("focal_length_effective") and p.get("pixel_size_um"):
            scale = (p["pixel_size_um"] / p["focal_length_effective"]) * 206.265

        if scale:
            self._spin_scale.setValue(scale)
            self._spin_scale.setEnabled(False)
            self._lbl_profile_scale.setText(
                f"Calculated from profile: {scale:.4f} arcsec/px"
            )
            self._lbl_profile_scale.setVisible(True)

            # Auto-set FWHM guess based on scale and ~2 arcsec typical seeing
            # This is just the DAOStarFinder starting hint
            suggested_fwhm = max(3.0, 2.0 / scale)
            self._log_append(
                f"Profile '{name}' loaded: scale={scale:.4f} arcsec/px, "
                f"suggested FWHM guess={suggested_fwhm:.1f}px"
            )
        else:
            self._spin_scale.setEnabled(True)
            self._lbl_profile_scale.setVisible(False)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select FITS Folder")
        if path:
            self._edit_folder.setText(path)
            self._scan_folder()

    def _browse_out_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._edit_out_folder.setText(path)

    def _scan_folder(self):
        folder = self._edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self._lbl_file_count.setText("⚠  Invalid folder path")
            return
        patterns = ["*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS"]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(folder, pat)))
        files = sorted(set(files))
        self._fits_files = files
        n = len(files)
        if n == 0:
            self._lbl_file_count.setText("⚠  No FITS files found in this folder")
        else:
            self._lbl_file_count.setText(f"✓  {n} FITS file{'s' if n != 1 else ''} found")
        self._log_append(f"Scanned: {folder}  →  {n} files")

    def _on_threshold_changed(self, val: int):
        threshold = val / 10.0
        self._lbl_threshold_val.setText(f'{threshold:.1f}"')

    def _get_threshold(self) -> float:
        return self._slider_threshold.value() / 10.0

    def _get_wavelength_nm(self) -> float:
        name = self._combo_wavelength.currentText()
        return WAVELENGTHS.get(name, 550.0)

    def _get_box_size(self) -> int:
        text = self._combo_box_size.currentText()
        return int(text.replace("px", ""))

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _start_analysis(self):
        if not self._fits_files:
            self._scan_folder()
            if not self._fits_files:
                self._set_status("No FITS files found. Please select a folder.", SIRIL_WARNING)
                return

        pixel_scale = self._spin_scale.value()
        if pixel_scale <= 0:
            self._set_status("Invalid pixel scale.", SIRIL_ERROR)
            return

        config = {
            "pixel_scale_arcsec_per_px":  pixel_scale,
            "wavelength_nm":              self._get_wavelength_nm(),
            "threshold_sigma":            self._spin_threshold_sigma.value(),
            "fwhm_guess_px":              max(3.0, 2.0 / pixel_scale),
            "box_size":                   self._get_box_size(),
            "min_stars":                  self._spin_min_stars.value(),
            "seeing_threshold_arcsec":    self._get_threshold(),
            "color_handling":             self._combo_color_handling.currentText(),
            "sample_every":               self._spin_sample.value(),
        }

        # Reset UI
        self._results = []
        self._summary = {}
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        self._table.setSortingEnabled(True)
        self._badge_best.set_value("—")
        self._badge_median.set_value("—")
        self._badge_r0.set_value("—")
        self._badge_usable.set_value("—")
        self._histogram.update_plot([], self._get_threshold())
        self._log.clear()

        self._btn_analyze.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._fits_files))
        self._progress.setValue(0)
        self._tabs.setCurrentIndex(3)   # switch to Log tab

        self._cancel_ev.clear()
        self._worker = SeeingWorker(self._fits_files, config, self._cancel_ev)
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_frame_result)
        self._worker.log_line.connect(self._log_append)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

        self._set_status(f"Analyzing {len(self._fits_files)} frames…")

    def _cancel_analysis(self):
        if self._worker:
            self._worker.cancel()
        self._btn_cancel.setEnabled(False)
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _on_progress(self, current: int, total: int, fname: str):
        self._progress.setValue(current)
        self._progress.setMaximum(total)
        self._set_status(f"[{current}/{total}] {fname}")

    def _on_frame_result(self, r: dict):
        self._results.append(r)
        self._add_table_row(r)

        # Live update histogram
        fwhms = [x["fwhm_arcsec"] for x in self._results]
        self._histogram.update_plot(fwhms, self._get_threshold())

        # Live update badges
        r0s = [x["r0_cm"] for x in self._results]
        threshold = self._get_threshold()
        usable = [x for x in self._results if x["fwhm_arcsec"] <= threshold]

        self._badge_best.set_value(
            f"{min(fwhms):.3f}", SIRIL_SUCCESS
        )
        self._badge_median.set_value(
            f"{float(np.median(fwhms)):.3f}", SIRIL_ACCENT
        )
        self._badge_r0.set_value(
            f"{max(r0s):.1f}", SIRIL_SUCCESS
        )
        self._badge_usable.set_value(
            f"{len(usable)}/{len(self._results)}",
            SIRIL_SUCCESS if len(usable) > len(self._results) * 0.7 else SIRIL_WARNING
        )

    def _on_finished(self, summary: dict):
        self._summary = summary
        self._btn_analyze.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)

        if summary.get("success"):
            q, qc = r0_to_quality_label(summary["median_r0"])
            self._set_status(
                f"Done: {summary['total_analyzed']} frames  |  "
                f"Median: {summary['median_fwhm']:.3f}\"  |  "
                f"r0: {summary['median_r0']:.1f}cm  |  "
                f"{q}",
                qc
            )
            self._tabs.setCurrentIndex(1)  # switch to Results
        else:
            self._set_status(
                summary.get("error", "Analysis failed"), SIRIL_ERROR
            )

    def _add_table_row(self, r: dict):
        self._table.setSortingEnabled(False)
        row = self._table.rowCount()
        self._table.insertRow(row)

        threshold = self._get_threshold()
        q = r["quality"]
        poor = r["fwhm_arcsec"] > threshold or q == "Poor"

        if q == "Excellent":
            bg = QColor("#1a3d2a")
        elif q == "Good":
            bg = QColor("#1a2a40")
        elif poor:
            bg = QColor("#3d1a1a")
        else:
            bg = QColor(SIRIL_BG2)

        def cell(text, align=Qt.AlignmentFlag.AlignCenter):
            item = QTableWidgetItem(str(text))
            item.setTextAlignment(align)
            item.setBackground(bg)
            return item

        self._table.setItem(row, 0, cell(row + 1))
        name_item = QTableWidgetItem(r["filename"])
        name_item.setBackground(bg)
        name_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._table.setItem(row, 1, name_item)
        self._table.setItem(row, 2, cell(f"{r['fwhm_px']:.2f}"))
        self._table.setItem(row, 3, cell(f"{r['fwhm_arcsec']:.3f}"))
        self._table.setItem(row, 4, cell(f"{r['r0_cm']:.1f}"))
        self._table.setItem(row, 5, cell(str(r["star_count"])))

        q_item = QTableWidgetItem(q)
        q_item.setForeground(QColor(r["quality_color"]))
        q_item.setBackground(bg)
        q_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 6, q_item)

        self._table.setSortingEnabled(True)

    # ── Export ────────────────────────────────────────────────────────────────

    def _do_export(self):
        if not self._results:
            QMessageBox.warning(self, "No Data", "Run the analysis first.")
            return

        out_folder = self._edit_out_folder.text().strip()
        if not out_folder:
            src = self._edit_folder.text().strip()
            out_folder = src if src and os.path.isdir(src) else os.path.expanduser("~")

        os.makedirs(out_folder, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        threshold = self._get_threshold()
        wl_name   = self._combo_wavelength.currentText()
        wl_nm     = self._get_wavelength_nm()
        scale     = self._spin_scale.value()
        exported  = []

        # CSV
        if self._chk_csv.isChecked():
            csv_path = os.path.join(out_folder, f"seeing_report_{ts}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_number", "filename", "fwhm_px", "fwhm_arcsec",
                    "r0_cm", "star_count", "quality", "usable"
                ])
                for i, r in enumerate(self._results, 1):
                    usable = r["fwhm_arcsec"] <= threshold
                    writer.writerow([
                        i, r["filename"],
                        r["fwhm_px"], r["fwhm_arcsec"],
                        r["r0_cm"], r["star_count"],
                        r["quality"], usable
                    ])

                s = self._summary
                if s.get("success"):
                    q, _ = r0_to_quality_label(s["median_r0"])
                    bq, _ = r0_to_quality_label(s["best_r0"])
                    writer.writerow([])
                    writer.writerow(["# Session summary"])
                    writer.writerow([f"# Total analyzed: {s['total_analyzed']}"])
                    writer.writerow([
                        f"# Best FWHM: {s['best_fwhm']:.3f} arcsec "
                        f"(r0={s['best_r0']:.1f}cm) - {bq}"
                    ])
                    writer.writerow([
                        f"# Median FWHM: {s['median_fwhm']:.3f} arcsec "
                        f"(r0={s['median_r0']:.1f}cm) - {q}"
                    ])
                    writer.writerow([
                        f"# Usable frames: {s['usable_count']} / "
                        f"{s['total_analyzed']} ({s['usable_pct']:.1f}%)"
                    ])
                    writer.writerow([f"# Threshold: {threshold:.1f} arcsec"])
                    writer.writerow([f"# Wavelength: {wl_name}  ({wl_nm:.0f}nm)"])
                    writer.writerow([f"# Pixel scale: {scale:.4f} arcsec/px"])

            exported.append(csv_path)
            self._log_append(f"CSV exported: {csv_path}")

        # Rejection list
        if self._chk_rejection.isChecked():
            poor = [r["filename"] for r in self._results
                    if r["fwhm_arcsec"] > threshold]
            rej_path = os.path.join(out_folder, f"rejection_list_{ts}.txt")
            with open(rej_path, "w", encoding="utf-8") as f:
                f.write(f"# Rejection list — FWHM > {threshold:.1f} arcsec\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                for fn in poor:
                    f.write(fn + "\n")
            exported.append(rej_path)
            self._log_append(f"Rejection list: {rej_path} ({len(poor)} frames)")

        # PNG plot
        if self._chk_plot.isChecked():
            png_path = os.path.join(out_folder, f"seeing_plot_{ts}.png")
            try:
                fwhms = [r["fwhm_arcsec"] for r in self._results]
                self._histogram.update_plot(fwhms, threshold)
                self._histogram.fig.savefig(
                    png_path, dpi=150, facecolor=SIRIL_BG,
                    bbox_inches="tight"
                )
                exported.append(png_path)
                self._log_append(f"Plot saved: {png_path}")
            except Exception as e:
                self._log_append(f"Could not save plot: {e}")

        if exported:
            msg = "Exported:\n" + "\n".join(exported)
            QMessageBox.information(self, "Export Complete", msg)
        else:
            QMessageBox.warning(self, "Nothing Selected",
                                "Select at least one export option.")

    # ── Siril integration ─────────────────────────────────────────────────────

    def _flag_in_siril(self):
        if not self._results:
            QMessageBox.warning(self, "No Data", "Run the analysis first.")
            return

        threshold = self._get_threshold()
        poor      = [r["filename"] for r in self._results
                     if r["fwhm_arcsec"] > threshold]
        if not poor:
            QMessageBox.information(self, "All Good",
                                    "No frames exceed the FWHM threshold.")
            return

        siril_iface = s.SirilInterface()
        try:
            siril_iface.connect()
            flagged = 0
            failed  = 0
            for fn in poor:
                # Strip extension for Siril frame name
                frame_name = os.path.splitext(fn)[0]
                try:
                    siril_iface.cmd("seqsetexcluded", frame_name)
                    flagged += 1
                except Exception:
                    failed += 1

            siril_iface.disconnect()
            msg = f"Flagged {flagged} frames as excluded in Siril."
            if failed:
                msg += f"\n{failed} frames could not be flagged (seqsetexcluded may not be available)."
                # Fallback: write .ssf script
                self._write_ssf_fallback(poor)
            self._log_append(msg)
            QMessageBox.information(self, "Siril Integration", msg)

        except Exception as e:
            try:
                siril_iface.disconnect()
            except Exception:
                pass
            self._log_append(f"Siril integration error: {e}")
            self._write_ssf_fallback(poor)
            QMessageBox.warning(
                self, "Siril Integration",
                f"Could not connect to Siril or command unavailable:\n{e}\n\n"
                "A fallback .ssf script has been saved in the FITS folder."
            )

    def _write_ssf_fallback(self, poor_filenames: list):
        folder = self._edit_folder.text().strip()
        if not folder:
            return
        ssf_path = os.path.join(folder, "flag_poor_frames.ssf")
        with open(ssf_path, "w", encoding="utf-8") as f:
            f.write("# Siril script — generated by seeing_estimator.py\n")
            f.write("# Run this in Siril to exclude poor-seeing frames\n\n")
            for fn in poor_filenames:
                frame_name = os.path.splitext(fn)[0]
                f.write(f"seqsetexcluded {frame_name}\n")
        self._log_append(f"Fallback SSF script saved: {ssf_path}")

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log_append(self, text: str):
        self._log.appendPlainText(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "", "Text files (*.txt);;All files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log.toPlainText())

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status.setText(msg)
        self._status.setStyleSheet(f"color: {color}; font-size: 9pt;")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
