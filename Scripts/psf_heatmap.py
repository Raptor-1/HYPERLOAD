"""
psf_heatmap.py  —  Script #22
PSF Heatmap — Optical quality analysis across the field of view
Siril Python Script | PyQt6 UI | QThread worker | Dark theme
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

# ── standard library ──────────────────────────────────────────────────────────
import os
import sys
import csv
import threading
from datetime import datetime

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from astropy.io import fits as astropy_fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSplitter, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ── equipment manager (optional) ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False


# ══════════════════════════════════════════════════════════════════════════════
# SIRIL THEME
# ══════════════════════════════════════════════════════════════════════════════

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
}}
QProgressBar::chunk {{
    background: {SIRIL_ACCENT};
    border-radius: 3px;
}}
QLabel#section  {{ color: {SIRIL_SECTION}; font-weight: bold; padding-top: 6px; }}
QLabel#dim      {{ color: {SIRIL_TEXT_DIM}; font-size: 9pt; }}
QLabel#ok       {{ color: {SIRIL_SUCCESS}; font-weight: bold; }}
QLabel#err      {{ color: {SIRIL_ERROR};   font-weight: bold; }}
QLabel#warn     {{ color: {SIRIL_WARNING}; font-weight: bold; }}
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
QSplitter::handle {{ background: {SIRIL_BORDER}; width: 2px; }}
"""


# ══════════════════════════════════════════════════════════════════════════════
# PSF MEASUREMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_stars_for_psf(image: np.ndarray,
                          fwhm_guess: float = 5.0,
                          threshold_sigma: float = 8.0,
                          max_stars: int = 2000,
                          border_margin: int = 40) -> list:
    """
    Detect stars suitable for PSF measurement.
    Returns list of (x, y, peak, flux) tuples sorted by brightness.
    """
    h, w = image.shape
    _, median, std = sigma_clipped_stats(image, sigma=3.0)
    if std <= 0:
        return []

    daofind = DAOStarFinder(
        fwhm=fwhm_guess,
        threshold=threshold_sigma * std,
        sharplo=0.2, sharphi=0.9,
        roundlo=-0.7, roundhi=0.7
    )
    sources = daofind(image - median)
    if sources is None or len(sources) == 0:
        return []

    sources.sort("peak")
    sources.reverse()

    sat_limit = 0.92 * float(np.percentile(image, 99.99))
    results   = []

    for src in sources:
        x, y = float(src["xcentroid"]), float(src["ycentroid"])
        if (x < border_margin or x > w - border_margin or
                y < border_margin or y > h - border_margin):
            continue
        if float(src["peak"]) + median > sat_limit:
            continue
        results.append((x, y, float(src["peak"]), float(src["flux"])))
        if len(results) >= max_stars:
            break

    return results


def fit_psf_2d_gaussian(image: np.ndarray,
                         x: float, y: float,
                         box_size: int = 24) -> dict | None:
    """
    Fit a 2D elliptical Gaussian to a star cutout.
    Returns dict with fwhm_major, fwhm_minor, fwhm_mean, ellipticity,
    angle_deg, peak, x_fit, y_fit — or None if fit fails.
    """
    h, w   = image.shape
    ix, iy = int(round(x)), int(round(y))
    half   = box_size // 2

    if (iy - half < 0 or iy + half >= h or
            ix - half < 0 or ix + half >= w):
        return None

    cutout = image[iy-half : iy+half,
                   ix-half : ix+half].astype(np.float64)

    corners = np.concatenate([
        cutout[:4, :4].ravel(), cutout[:4, -4:].ravel(),
        cutout[-4:, :4].ravel(), cutout[-4:, -4:].ravel()
    ])
    bg     = float(np.median(corners))
    cutout = cutout - bg
    cutout = np.clip(cutout, 0, None)

    if cutout.max() < 10 or cutout.sum() <= 0:
        return None

    def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x_arr, y_arr = xy
        ct, st = np.cos(theta), np.sin(theta)
        a = (ct**2 / (2*sigma_x**2)) + (st**2 / (2*sigma_y**2))
        b = (-np.sin(2*theta) / (4*sigma_x**2)) + (np.sin(2*theta) / (4*sigma_y**2))
        c = (st**2 / (2*sigma_x**2)) + (ct**2 / (2*sigma_y**2))
        dx, dy = x_arr - x0, y_arr - y0
        return amplitude * np.exp(-(a*dx**2 + 2*b*dx*dy + c*dy**2))

    size   = box_size
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    z_flat = cutout.ravel()

    peak_idx = np.unravel_index(np.argmax(cutout), cutout.shape)
    x0_guess = float(peak_idx[1])
    y0_guess = float(peak_idx[0])
    amp      = float(cutout.max())
    sig_guess = 2.0

    p0 = [amp, x0_guess, y0_guess, sig_guess, sig_guess, 0.0]
    bounds = (
        [0,   0,    0,    0.5, 0.5, -np.pi/2],
        [amp*2, size, size, half, half, np.pi/2]
    )

    try:
        popt, _ = curve_fit(
            gaussian_2d, (x_grid.ravel(), y_grid.ravel()),
            z_flat, p0=p0, bounds=bounds, maxfev=800
        )
    except (RuntimeError, ValueError):
        return None

    amplitude, x0, y0, sigma_x, sigma_y, theta = popt

    if sigma_x < sigma_y:
        sigma_x, sigma_y = sigma_y, sigma_x
        theta += np.pi / 2

    fwhm_x = 2.355 * abs(sigma_x)
    fwhm_y = 2.355 * abs(sigma_y)

    if fwhm_x < 0.8 or fwhm_x > box_size * 0.7:
        return None
    if fwhm_y < 0.5 or fwhm_y > fwhm_x:
        return None

    fwhm_major  = max(fwhm_x, fwhm_y)
    fwhm_minor  = min(fwhm_x, fwhm_y)
    ellipticity = (fwhm_major - fwhm_minor) / (fwhm_major + 1e-10)
    angle_deg   = float(np.degrees(theta)) % 180.0

    return {
        "fwhm_major":  fwhm_major,
        "fwhm_minor":  fwhm_minor,
        "fwhm_mean":   (fwhm_major + fwhm_minor) / 2.0,
        "ellipticity": float(ellipticity),
        "angle_deg":   angle_deg,
        "peak":        float(amplitude),
        "x_fit":       float(ix - half + x0),
        "y_fit":       float(iy - half + y0),
    }


def measure_psf_across_field(image: np.ndarray,
                               pixel_scale_arcsec: float = None,
                               fwhm_guess: float = 5.0,
                               threshold_sigma: float = 8.0,
                               max_stars: int = 1000,
                               box_size: int = 24,
                               log_callback=None,
                               cancel_event=None) -> list:
    """
    Measure PSF parameters for all suitable stars across the field.
    Returns list of measurement dicts.
    """
    stars = detect_stars_for_psf(image, fwhm_guess, threshold_sigma, max_stars)

    if log_callback:
        log_callback(f"  Detected {len(stars)} candidate stars")

    measurements = []
    for i, (sx, sy, peak, flux) in enumerate(stars):
        if cancel_event and cancel_event.is_set():
            break

        result = fit_psf_2d_gaussian(image, sx, sy, box_size)
        if result is None:
            continue

        meas = {
            "x":           sx,
            "y":           sy,
            "fwhm_px":     result["fwhm_mean"],
            "fwhm_arcsec": (result["fwhm_mean"] * pixel_scale_arcsec
                            if pixel_scale_arcsec else None),
            "ellipticity": result["ellipticity"],
            "angle_deg":   result["angle_deg"],
            "peak":        result["peak"],
        }
        measurements.append(meas)

        if log_callback and i % 50 == 0:
            log_callback(
                f"  Measured {len(measurements)} stars "
                f"({i+1}/{len(stars)} processed)...")

    if log_callback:
        log_callback(f"  PSF measurements complete: {len(measurements)} stars")

    return measurements


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP INTERPOLATION
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_to_grid(measurements: list,
                         image_shape: tuple,
                         metric: str,
                         grid_resolution: int = 100,
                         smooth_sigma: float = 5.0) -> np.ndarray:
    """
    Interpolate scattered star measurements onto a regular grid.
    Returns 2D numpy array; NaN outside the convex hull of star positions.
    """
    h, w = image_shape
    xs     = np.array([m["x"]    for m in measurements])
    ys     = np.array([m["y"]    for m in measurements])
    values = np.array([m[metric] for m in measurements], dtype=np.float64)

    valid  = np.isfinite(values)
    xs, ys, values = xs[valid], ys[valid], values[valid]

    if len(xs) < 4:
        return np.full((grid_resolution, grid_resolution), np.nan)

    aspect = h / w
    gw     = grid_resolution
    gh     = max(4, int(grid_resolution * aspect))

    xi      = np.linspace(0, w, gw)
    yi      = np.linspace(0, h, gh)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    grid = griddata(
        points    = np.column_stack([xs, ys]),
        values    = values,
        xi        = (xi_grid, yi_grid),
        method    = "cubic",
        fill_value = np.nan
    )

    mask = np.isfinite(grid)
    if mask.any() and smooth_sigma > 0:
        filled         = np.where(mask, grid, 0.0)
        weights        = mask.astype(float)
        smoothed_vals  = gaussian_filter(filled,  sigma=smooth_sigma)
        smoothed_wts   = gaussian_filter(weights, sigma=smooth_sigma)
        grid = np.where(smoothed_wts > 0.01,
                        smoothed_vals / smoothed_wts,
                        np.nan)

    return grid


# ══════════════════════════════════════════════════════════════════════════════
# AUTOMATIC OPTICAL DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════

def generate_diagnosis(stats: dict, measurements: list,
                        image_shape: tuple) -> list:
    """
    Generate diagnostic observations from PSF measurements.
    Returns list of (severity, message) tuples.
    severity: "ok", "warn", "bad"
    """
    h, w   = image_shape
    diags  = []

    fwhm_vals = np.array([m["fwhm_px"]     for m in measurements])
    ell_vals  = np.array([m["ellipticity"] for m in measurements])
    ang_vals  = np.array([m["angle_deg"]   for m in measurements])
    xs        = np.array([m["x"]           for m in measurements])
    ys        = np.array([m["y"]           for m in measurements])

    cx, cy      = w / 2, h / 2
    dist        = np.sqrt(((xs - cx) / w)**2 + ((ys - cy) / h)**2)
    centre_idx  = dist < 0.25
    edge_idx    = dist > 0.45

    if centre_idx.sum() >= 3 and edge_idx.sum() >= 3:
        fwhm_centre = float(np.median(fwhm_vals[centre_idx]))
        fwhm_edge   = float(np.median(fwhm_vals[edge_idx]))
        ratio       = fwhm_edge / max(fwhm_centre, 0.1)

        if ratio < 1.15:
            diags.append(("ok",
                "FWHM uniform across field — excellent flatness"))
        elif ratio < 1.4:
            diags.append(("warn",
                f"FWHM {ratio:.1f}× worse at edges vs centre — "
                "possible mild field curvature or tilt"))
        else:
            diags.append(("bad",
                f"FWHM {ratio:.1f}× worse at edges — "
                "significant field curvature, tilt, or focus gradient"))

    ell_med = float(np.median(ell_vals))
    if ell_med < 0.05:
        diags.append(("ok", "Excellent star roundness across field"))
    elif ell_med < 0.12:
        diags.append(("warn",
            f"Moderate ellipticity (median {ell_med:.3f}) — "
            "possible seeing, mild collimation issue, or tracking"))
    else:
        diags.append(("bad",
            f"High ellipticity (median {ell_med:.3f}) — "
            "check collimation, tracking, or focus"))

    if edge_idx.sum() >= 5:
        ell_edge = float(np.median(ell_vals[edge_idx]))
        ell_cen  = float(np.median(ell_vals[centre_idx])) if centre_idx.sum() >= 3 else ell_edge
        if ell_edge > ell_cen * 1.5 and ell_edge > 0.08:
            edge_angs = ang_vals[edge_idx]
            expected  = np.degrees(np.arctan2(
                ys[edge_idx] - cy, xs[edge_idx] - cx)) % 180
            ang_diff  = np.abs(((edge_angs - expected) + 90) % 180 - 90)
            if np.median(ang_diff) < 30:
                diags.append(("bad",
                    "Radial elongation at edges → coma detected. "
                    "Check collimation and reducer spacing."))
            else:
                diags.append(("warn",
                    "Higher ellipticity at edges vs centre. "
                    "Possible off-axis aberrations."))

    ang_std = float(np.std(ang_vals % 90))
    if ell_med > 0.08 and ang_std < 20:
        diags.append(("warn",
            f"Stars elongated in consistent direction "
            f"({float(np.median(ang_vals)):.0f}°) — "
            "possible astigmatism or focuser tilt"))

    fwhm_med = stats["fwhm_median"]
    unit     = stats["fwhm_unit"]
    if unit == "arcsec":
        if fwhm_med < 2.0:
            diags.append(("ok",
                f"Excellent seeing/focus: {fwhm_med:.2f}\""))
        elif fwhm_med < 3.5:
            diags.append(("ok",
                f"Good seeing/focus: {fwhm_med:.2f}\""))
        elif fwhm_med < 5.0:
            diags.append(("warn",
                f"Average seeing/focus: {fwhm_med:.2f}\""))
        else:
            diags.append(("bad",
                f"Poor seeing/focus: {fwhm_med:.2f}\" — "
                "consider re-focusing or discarding this session"))

    if not diags:
        diags.append(("ok", "Insufficient stars for detailed diagnosis"))

    return diags


# ══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class PSFWorker(QThread):
    progress   = pyqtSignal(int, int, str)
    log_line   = pyqtSignal(str)
    maps_ready = pyqtSignal(dict)
    finished   = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg = self.config

            # Step 1: Load image
            self.progress.emit(1, 4, "Loading FITS image...")
            fits_path = cfg["fits_path"]
            with astropy_fits.open(fits_path) as hdul:
                data   = hdul[0].data.astype(np.float32)
                header = hdul[0].header

            if data.ndim == 3:
                if data.shape[0] == 3:
                    data = (0.299*data[0] + 0.587*data[1] + 0.114*data[2])
                else:
                    data = data[0]

            h, w = data.shape
            self.log_line.emit(
                f"Loaded: {w}×{h} px  from {os.path.basename(fits_path)}")

            if self._cancel.is_set():
                self._abort(); return

            # Step 2: Measure PSF across field
            self.progress.emit(2, 4, "Measuring PSF across field...")
            pixel_scale  = cfg.get("pixel_scale_arcsec")
            measurements = measure_psf_across_field(
                image              = data,
                pixel_scale_arcsec = pixel_scale,
                fwhm_guess         = cfg.get("fwhm_guess", 5.0),
                threshold_sigma    = cfg.get("threshold_sigma", 8.0),
                max_stars          = cfg.get("max_stars", 800),
                box_size           = cfg.get("box_size", 24),
                log_callback       = self.log_line.emit,
                cancel_event       = self._cancel
            )

            if len(measurements) < 5:
                self.finished.emit({
                    "success": False,
                    "error":   f"Only {len(measurements)} stars measured — "
                               "try lowering detection threshold"
                })
                return

            if self._cancel.is_set():
                self._abort(); return

            # Step 3: Build interpolated heatmaps
            self.progress.emit(3, 4, "Building heatmaps...")
            res      = cfg.get("grid_resolution", 80)
            smth     = cfg.get("smooth_sigma", 4.0)
            fwhm_key = "fwhm_arcsec" if pixel_scale else "fwhm_px"

            grids = {}
            for metric in [fwhm_key, "ellipticity", "angle_deg", "peak"]:
                if self._cancel.is_set():
                    self._abort(); return
                grids[metric] = interpolate_to_grid(
                    measurements, (h, w), metric,
                    grid_resolution=res, smooth_sigma=smth)
                self.log_line.emit(
                    f"  Built {metric} heatmap ({res}×{int(res*h/w)} grid)")

            # Step 4: Compute statistics
            self.progress.emit(4, 4, "Computing statistics...")
            fwhm_vals = np.array([m[fwhm_key] for m in measurements
                                   if m.get(fwhm_key) is not None])
            ell_vals  = np.array([m["ellipticity"] for m in measurements])

            stats = {
                "n_stars":      len(measurements),
                "fwhm_median":  float(np.median(fwhm_vals))  if len(fwhm_vals) else 0,
                "fwhm_best":    float(np.percentile(fwhm_vals, 10)) if len(fwhm_vals) else 0,
                "fwhm_worst":   float(np.percentile(fwhm_vals, 90)) if len(fwhm_vals) else 0,
                "ellip_median": float(np.median(ell_vals))   if len(ell_vals) else 0,
                "ellip_worst":  float(np.percentile(ell_vals, 90)) if len(ell_vals) else 0,
                "fwhm_unit":    "arcsec" if pixel_scale else "px",
                "image_shape":  (h, w),
            }
            self.log_line.emit(
                f"Median FWHM: {stats['fwhm_median']:.2f} {stats['fwhm_unit']}")
            self.log_line.emit(
                f"Median ellipticity: {stats['ellip_median']:.3f}")
            self.log_line.emit(f"Stars measured: {stats['n_stars']}")

            result_payload = {
                "grids":        grids,
                "measurements": measurements,
                "stats":        stats,
                "image_shape":  (h, w),
                "fwhm_key":     fwhm_key,
                "image_data":   data,
            }
            self.maps_ready.emit(result_payload)

            if cfg.get("export_csv") and cfg.get("output_dir"):
                csv_path = self._export_csv(measurements, cfg, fwhm_key)
                self.log_line.emit(f"CSV saved: {csv_path}")

            self.finished.emit({
                "success":      True,
                "n_stars":      len(measurements),
                "fwhm_median":  stats["fwhm_median"],
                "fwhm_unit":    stats["fwhm_unit"],
                "ellip_median": stats["ellip_median"],
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _export_csv(self, measurements: list, cfg: dict, fwhm_key: str) -> str:
        path = os.path.join(cfg["output_dir"], "psf_measurements.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["x", "y", "fwhm_px", "fwhm_arcsec",
                                "ellipticity", "angle_deg", "peak"],
                extrasaction="ignore")
            writer.writeheader()
            writer.writerows(measurements)
        return path

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP CANVAS (4-PANEL MATPLOTLIB)
# ══════════════════════════════════════════════════════════════════════════════

class HeatmapCanvas(FigureCanvasQTAgg):
    """Four-panel PSF heatmap: FWHM, Ellipticity, Orientation, Peak brightness."""

    def __init__(self, parent=None):
        self.fig  = Figure(figsize=(11, 8), facecolor="#1e2128")
        self.axes = [self.fig.add_subplot(2, 2, i+1) for i in range(4)]
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self._last_result = None

    def _style_axes(self):
        for ax in self.axes:
            ax.set_facecolor("#1e2128")
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")

    def show_heatmaps(self, result: dict):
        self._last_result = result
        grids        = result["grids"]
        measurements = result["measurements"]
        stats        = result["stats"]
        fwhm_key     = result["fwhm_key"]
        image_data   = result["image_data"]
        ih, iw       = result["image_shape"]

        factor  = max(1, max(ih, iw) // 300)
        bg      = image_data[::factor, ::factor]
        p_lo    = np.percentile(bg, 1)
        p_hi    = np.percentile(bg, 99)
        bg_disp = np.clip((bg - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)

        star_xs = np.array([m["x"] for m in measurements])
        star_ys = np.array([m["y"] for m in measurements])

        panel_configs = [
            {
                "grid":   grids[fwhm_key],
                "title":  (f"FWHM ({stats['fwhm_unit']})  "
                           f"median={stats['fwhm_median']:.2f}"),
                "cmap":   "YlOrRd",
                "vmin":   None,
                "vmax":   None,
                "values": [m[fwhm_key] for m in measurements
                           if m.get(fwhm_key) is not None],
            },
            {
                "grid":   grids["ellipticity"],
                "title":  f"Ellipticity  median={stats['ellip_median']:.3f}",
                "cmap":   "YlOrRd",
                "vmin":   0.0,
                "vmax":   0.5,
                "values": [m["ellipticity"] for m in measurements],
            },
            {
                "grid":   grids["angle_deg"],
                "title":  "Elongation orientation (°)",
                "cmap":   "hsv",
                "vmin":   0.0,
                "vmax":   180.0,
                "values": [m["angle_deg"] for m in measurements],
            },
            {
                "grid":   grids["peak"],
                "title":  "Peak brightness",
                "cmap":   "Blues",
                "vmin":   None,
                "vmax":   None,
                "values": [m["peak"] for m in measurements],
            },
        ]

        for ax, pcfg in zip(self.axes, panel_configs):
            ax.clear()
            ax.set_facecolor("#1e2128")

            ax.imshow(bg_disp, cmap="gray", origin="lower",
                      extent=[0, iw, 0, ih],
                      aspect="auto", alpha=0.35,
                      interpolation="nearest")

            grid = pcfg["grid"]
            if grid is not None and np.any(np.isfinite(grid)):
                vals = np.array(pcfg["values"])
                fin  = vals[np.isfinite(vals)]
                vmin = pcfg["vmin"] if pcfg["vmin"] is not None \
                       else (float(np.percentile(fin, 2))  if len(fin) else 0)
                vmax = pcfg["vmax"] if pcfg["vmax"] is not None \
                       else (float(np.percentile(fin, 98)) if len(fin) else 1)

                im = ax.imshow(
                    grid,
                    cmap   = pcfg["cmap"],
                    origin = "lower",
                    extent = [0, iw, 0, ih],
                    aspect = "auto",
                    alpha  = 0.72,
                    vmin   = vmin,
                    vmax   = vmax,
                    interpolation = "bilinear"
                )
                self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

            vals_for_color = np.array(pcfg["values"])
            if len(vals_for_color) > 0 and np.any(np.isfinite(vals_for_color)):
                fin_c  = vals_for_color[np.isfinite(vals_for_color)]
                vmin_s = float(np.percentile(fin_c, 2))
                vmax_s = float(np.percentile(fin_c, 98))
                norm_c = Normalize(vmin=vmin_s, vmax=vmax_s)
                cmap_f = cm.get_cmap(pcfg["cmap"])
                colors = [cmap_f(norm_c(v)) for v in vals_for_color]
                ax.scatter(star_xs, star_ys,
                           c=colors, s=8, linewidths=0,
                           alpha=0.7, zorder=5)

            ax.set_xlim(0, iw)
            ax.set_ylim(0, ih)
            ax.set_title(pcfg["title"], color="#5ba3ff", fontsize=8, pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")

        self.fig.suptitle(
            f"PSF Analysis  —  {stats['n_stars']} stars measured",
            color="#dde3ee", fontsize=10, y=0.98)
        self.fig.tight_layout(rect=[0, 0, 1, 0.97], pad=0.5)
        self.draw()

    def save_png(self, output_path: str, dpi: int = 150):
        self.fig.savefig(output_path, dpi=dpi,
                         facecolor="#1e2128",
                         bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS TAB — diagnosis display
# ══════════════════════════════════════════════════════════════════════════════

class StatisticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Field summary ───────────────────────────────────────────────────
        self._summary_box = QGroupBox("Field summary")
        self._summary_layout = QVBoxLayout(self._summary_box)
        self._summary_labels = {}
        for key in ["Stars measured", "Median FWHM", "Best FWHM (10th pct)",
                    "Worst FWHM (90th pct)", "Median ellipticity",
                    "Worst ellipticity (90th pct)"]:
            row = QHBoxLayout()
            name_lbl = QLabel(key + ":")
            name_lbl.setFixedWidth(200)
            val_lbl  = QLabel("—")
            val_lbl.setObjectName("ok")
            row.addWidget(name_lbl)
            row.addWidget(val_lbl)
            row.addStretch()
            self._summary_layout.addLayout(row)
            self._summary_labels[key] = val_lbl
        layout.addWidget(self._summary_box)

        # ── Diagnosis ────────────────────────────────────────────────────────
        self._diag_box = QGroupBox("Optical diagnosis")
        self._diag_layout = QVBoxLayout(self._diag_box)
        placeholder = QLabel("Run analysis to see diagnosis.")
        placeholder.setObjectName("dim")
        self._diag_layout.addWidget(placeholder)
        layout.addWidget(self._diag_box)

        layout.addStretch()

    def update_stats(self, stats: dict, measurements: list, image_shape: tuple):
        u = stats["fwhm_unit"]
        self._summary_labels["Stars measured"].setText(str(stats["n_stars"]))
        self._summary_labels["Median FWHM"].setText(
            f"{stats['fwhm_median']:.2f} {u}")
        self._summary_labels["Best FWHM (10th pct)"].setText(
            f"{stats['fwhm_best']:.2f} {u}")
        self._summary_labels["Worst FWHM (90th pct)"].setText(
            f"{stats['fwhm_worst']:.2f} {u}")
        self._summary_labels["Median ellipticity"].setText(
            f"{stats['ellip_median']:.3f}")
        self._summary_labels["Worst ellipticity (90th pct)"].setText(
            f"{stats['ellip_worst']:.3f}")

        # Clear old diagnosis
        while self._diag_layout.count():
            item = self._diag_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        diags = generate_diagnosis(stats, measurements, image_shape)
        icon_map  = {"ok": "✓", "warn": "⚠", "bad": "✕"}
        style_map = {"ok": "ok", "bad": "err", "warn": "warn"}
        for severity, msg in diags:
            lbl = QLabel(f"{icon_map[severity]}  {msg}")
            lbl.setObjectName(style_map[severity])
            lbl.setWordWrap(True)
            self._diag_layout.addWidget(lbl)

        if not diags:
            self._diag_layout.addWidget(QLabel("No diagnosis available."))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📡  PSF Heatmap  —  Siril")
        self.resize(1280, 820)

        self._worker        = None
        self._cancel_event  = threading.Event()
        self._last_result   = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        root.addWidget(self._make_title_bar())

        # Main splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._splitter, stretch=1)

        self._splitter.addWidget(self._make_left_panel())
        self._splitter.addWidget(self._make_right_panel())
        self._splitter.setSizes([300, 980])
        self._splitter.setHandleWidth(3)

        # Bottom row
        root.addWidget(self._make_bottom_row())

        # Status bar
        self._status = QLabel("Ready — select a FITS file and click Analyze")
        self._status.setObjectName("dim")
        self._status.setContentsMargins(8, 3, 8, 3)
        root.addWidget(self._status)

        # Initial UI state
        self._set_idle()

    # ── title bar ─────────────────────────────────────────────────────────────
    def _make_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(42)
        bar.setStyleSheet(f"background:{SIRIL_BG3}; border-bottom:1px solid {SIRIL_BORDER};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)
        left = QLabel("📡  PSF Heatmap  —  Siril")
        left.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:12pt; font-weight:bold;")
        right = QLabel("v1.0  |  Optical quality analysis")
        right.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        layout.addWidget(left)
        layout.addStretch()
        layout.addWidget(right)
        return bar

    # ── left panel ────────────────────────────────────────────────────────────
    def _make_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._make_input_group())
        layout.addWidget(self._make_pixel_scale_group())
        layout.addWidget(self._make_detection_group())
        layout.addWidget(self._make_heatmap_group())
        layout.addWidget(self._make_output_group())
        layout.addStretch()

        scroll.setWidget(container)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(340)
        return scroll

    def _make_input_group(self) -> QGroupBox:
        grp    = QGroupBox("Input image")
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._fits_edit = QLineEdit()
        self._fits_edit.setPlaceholderText("Path to FITS file…")
        browse = QPushButton("Browse")
        browse.setFixedWidth(70)
        browse.clicked.connect(self._browse_fits)
        row.addWidget(self._fits_edit)
        row.addWidget(browse)
        layout.addLayout(row)

        load_btn = QPushButton("Load & show info")
        load_btn.clicked.connect(self._load_fits_info)
        layout.addWidget(load_btn)

        self._fits_info = QLabel("")
        self._fits_info.setObjectName("dim")
        self._fits_info.setWordWrap(True)
        layout.addWidget(self._fits_info)

        return grp

    def _make_pixel_scale_group(self) -> QGroupBox:
        grp    = QGroupBox("Pixel scale")
        layout = QVBoxLayout(grp)

        self._scale_mode = QComboBox()
        options = ["Manual entry"]
        if HAS_EQUIPMENT_MANAGER:
            options.append("From equipment profile")
        self._scale_mode.addItems(options)
        self._scale_mode.currentIndexChanged.connect(self._on_scale_mode_changed)
        layout.addWidget(self._scale_mode)

        # Manual entry
        self._manual_scale_widget = QWidget()
        mw = QFormLayout(self._manual_scale_widget)
        mw.setContentsMargins(0, 0, 0, 0)
        self._pixel_scale_spin = QDoubleSpinBox()
        self._pixel_scale_spin.setRange(0.01, 20.0)
        self._pixel_scale_spin.setSingleStep(0.01)
        self._pixel_scale_spin.setValue(1.0)
        self._pixel_scale_spin.setSpecialValueText("–")
        self._pixel_scale_spin.setToolTip("0 = skip (show in pixels)")
        mw.addRow("Pixel scale (arcsec/px):", self._pixel_scale_spin)
        layout.addWidget(self._manual_scale_widget)

        # Profile entry (optional)
        self._profile_scale_widget = QWidget()
        pw = QVBoxLayout(self._profile_scale_widget)
        pw.setContentsMargins(0, 0, 0, 0)
        self._profile_combo = QComboBox()
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self._profile_info  = QLabel("")
        self._profile_info.setObjectName("dim")
        self._profile_info.setWordWrap(True)
        pw.addWidget(self._profile_combo)
        pw.addWidget(self._profile_info)
        self._profile_scale_widget.setVisible(False)
        layout.addWidget(self._profile_scale_widget)

        if HAS_EQUIPMENT_MANAGER:
            self._load_profiles()

        return grp

    def _make_detection_group(self) -> QGroupBox:
        grp    = QGroupBox("Detection settings")
        layout = QFormLayout(grp)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(3.0, 20.0)
        self._threshold_spin.setValue(8.0)
        self._threshold_spin.setSingleStep(0.5)
        layout.addRow("Detection threshold (σ):", self._threshold_spin)

        self._fwhm_guess_spin = QDoubleSpinBox()
        self._fwhm_guess_spin.setRange(1.0, 20.0)
        self._fwhm_guess_spin.setValue(5.0)
        self._fwhm_guess_spin.setSingleStep(0.5)
        layout.addRow("FWHM guess (px):", self._fwhm_guess_spin)

        self._max_stars_spin = QSpinBox()
        self._max_stars_spin.setRange(50, 2000)
        self._max_stars_spin.setValue(800)
        self._max_stars_spin.setSingleStep(50)
        layout.addRow("Max stars:", self._max_stars_spin)

        self._box_combo = QComboBox()
        self._box_combo.addItems(["16px", "24px", "32px", "48px"])
        self._box_combo.setCurrentText("24px")
        layout.addRow("PSF fit box:", self._box_combo)

        return grp

    def _make_heatmap_group(self) -> QGroupBox:
        grp    = QGroupBox("Heatmap settings")
        layout = QFormLayout(grp)

        self._grid_spin = QSpinBox()
        self._grid_spin.setRange(30, 200)
        self._grid_spin.setValue(80)
        layout.addRow("Grid resolution:", self._grid_spin)

        dim1 = QLabel("Higher = finer map, slower")
        dim1.setObjectName("dim")
        layout.addRow("", dim1)

        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.0, 15.0)
        self._smooth_spin.setValue(4.0)
        self._smooth_spin.setSingleStep(0.5)
        layout.addRow("Smoothing (σ):", self._smooth_spin)

        dim2 = QLabel("0 = no smoothing (noisy)")
        dim2.setObjectName("dim")
        layout.addRow("", dim2)

        return grp

    def _make_output_group(self) -> QGroupBox:
        grp    = QGroupBox("Output")
        layout = QVBoxLayout(grp)

        self._export_png_chk = QCheckBox("Export PNG heatmap")
        self._export_csv_chk = QCheckBox("Export star CSV")
        self._export_png_chk.stateChanged.connect(self._on_export_toggle)
        self._export_csv_chk.stateChanged.connect(self._on_export_toggle)
        layout.addWidget(self._export_png_chk)
        layout.addWidget(self._export_csv_chk)

        self._output_dir_widget = QWidget()
        ow = QHBoxLayout(self._output_dir_widget)
        ow.setContentsMargins(0, 0, 0, 0)
        self._output_dir_edit = QLineEdit()
        self._output_dir_edit.setPlaceholderText("Output folder…")
        browse_out = QPushButton("Browse")
        browse_out.setFixedWidth(70)
        browse_out.clicked.connect(self._browse_output_dir)
        ow.addWidget(self._output_dir_edit)
        ow.addWidget(browse_out)
        self._output_dir_widget.setVisible(False)
        layout.addWidget(self._output_dir_widget)

        return grp

    # ── right panel ───────────────────────────────────────────────────────────
    def _make_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()

        # Heatmaps tab
        heatmap_tab = QWidget()
        hm_layout   = QVBoxLayout(heatmap_tab)
        hm_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = HeatmapCanvas()
        hm_layout.addWidget(self._canvas)
        self._tabs.addTab(heatmap_tab, "📡  Heatmaps")

        # Statistics tab
        self._stats_widget = StatisticsWidget()
        self._tabs.addTab(self._stats_widget, "📊  Statistics")

        # Log tab
        log_tab    = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(4, 4, 4, 4)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        log_layout.addWidget(self._log)
        self._tabs.addTab(log_tab, "📋  Log")

        layout.addWidget(self._tabs)
        return widget

    # ── bottom row ────────────────────────────────────────────────────────────
    def _make_bottom_row(self) -> QWidget:
        widget = QWidget()
        widget.setFixedHeight(46)
        widget.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self._progress = QProgressBar()
        self._progress.setRange(0, 4)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(8)
        layout.addWidget(self._progress, stretch=1)

        self._analyze_btn = QPushButton("▶  Analyze")
        self._analyze_btn.setObjectName("primary")
        self._analyze_btn.setFixedWidth(120)
        self._analyze_btn.clicked.connect(self._start_analysis)
        layout.addWidget(self._analyze_btn)

        self._save_btn = QPushButton("💾  Save PNG")
        self._save_btn.setObjectName("export")
        self._save_btn.setFixedWidth(120)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_png)
        layout.addWidget(self._save_btn)

        self._cancel_btn = QPushButton("✕  Cancel")
        self._cancel_btn.setObjectName("danger")
        self._cancel_btn.setFixedWidth(100)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_analysis)
        layout.addWidget(self._cancel_btn)

        return widget

    # ── state helpers ─────────────────────────────────────────────────────────
    def _set_idle(self):
        self._analyze_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._progress.setValue(0)

    def _set_running(self):
        self._analyze_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)

    # ── equipment profile helpers ─────────────────────────────────────────────
    def _load_profiles(self):
        try:
            profiles = load_all_profiles()
            self._profile_combo.clear()
            for name in profiles:
                self._profile_combo.addItem(name)
            self._on_profile_changed()
        except Exception:
            pass

    def _on_profile_changed(self):
        if not HAS_EQUIPMENT_MANAGER:
            return
        name = self._profile_combo.currentText()
        try:
            p    = get_profile(name)
            res  = p.get("resolution_arcsec_px")
            if res:
                self._profile_info.setText(f"{res:.4f} arcsec/px (from profile)")
            else:
                self._profile_info.setText("Profile has no pixel scale — using manual")
        except Exception:
            self._profile_info.setText("Profile not found")

    def _on_scale_mode_changed(self):
        manual = self._scale_mode.currentText() == "Manual entry"
        self._manual_scale_widget.setVisible(manual)
        self._profile_scale_widget.setVisible(not manual)

    # ── output toggles ────────────────────────────────────────────────────────
    def _on_export_toggle(self):
        show = self._export_png_chk.isChecked() or self._export_csv_chk.isChecked()
        self._output_dir_widget.setVisible(show)

    # ── browse ────────────────────────────────────────────────────────────────
    def _browse_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FITS file", "",
            "FITS files (*.fits *.fit *.fts);;All files (*)")
        if path:
            self._fits_edit.setText(path)
            self._load_fits_info()

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._output_dir_edit.setText(path)

    def _load_fits_info(self):
        path = self._fits_edit.text().strip()
        if not path or not os.path.isfile(path):
            self._fits_info.setText("File not found.")
            return
        try:
            with astropy_fits.open(path) as hdul:
                d = hdul[0].data
                if d is None:
                    self._fits_info.setText("No image data in primary HDU.")
                    return
                shape = d.shape
                fsize = os.path.getsize(path) / 1e6
                ch    = "mono" if d.ndim == 2 else f"{d.shape[0]}ch"
                dims  = f"{shape[-1]}×{shape[-2]}" if d.ndim >= 2 else str(shape)
                self._fits_info.setText(
                    f"✓  {dims} px  |  {ch}  |  {fsize:.1f} MB")
        except Exception as e:
            self._fits_info.setText(f"Error: {e}")

    # ── get pixel scale ───────────────────────────────────────────────────────
    def _get_pixel_scale(self) -> float | None:
        if self._scale_mode.currentText() == "Manual entry":
            v = self._pixel_scale_spin.value()
            return v if v >= 0.01 else None
        if HAS_EQUIPMENT_MANAGER:
            name = self._profile_combo.currentText()
            try:
                p   = get_profile(name)
                res = p.get("resolution_arcsec_px")
                return float(res) if res else None
            except Exception:
                return None
        return None

    # ── analysis ──────────────────────────────────────────────────────────────
    def _start_analysis(self):
        fits_path = self._fits_edit.text().strip()
        if not fits_path or not os.path.isfile(fits_path):
            QMessageBox.warning(self, "No file", "Please select a valid FITS file.")
            return

        box_px = int(self._box_combo.currentText().replace("px", ""))

        cfg = {
            "fits_path":         fits_path,
            "pixel_scale_arcsec": self._get_pixel_scale(),
            "threshold_sigma":   self._threshold_spin.value(),
            "fwhm_guess":        self._fwhm_guess_spin.value(),
            "max_stars":         self._max_stars_spin.value(),
            "box_size":          box_px,
            "grid_resolution":   self._grid_spin.value(),
            "smooth_sigma":      self._smooth_spin.value(),
            "export_csv":        self._export_csv_chk.isChecked(),
            "export_png":        self._export_png_chk.isChecked(),
            "output_dir":        self._output_dir_edit.text().strip() or None,
        }

        self._cancel_event.clear()
        self._set_running()
        self._log.clear()
        self._append_log(
            f"[{datetime.now().strftime('%H:%M:%S')}] Starting PSF analysis…")
        self._status.setText("Analyzing…")

        self._worker = PSFWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._append_log)
        self._worker.maps_ready.connect(self._on_maps_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_analysis(self):
        if self._worker and self._worker.isRunning():
            self._cancel_event.set()
            self._status.setText("Cancelling…")

    # ── worker callbacks ──────────────────────────────────────────────────────
    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setMaximum(total)
        self._progress.setValue(step)
        self._status.setText(f"Step {step}/{total}: {msg}")
        self._append_log(f"  [{step}/{total}] {msg}")

    def _append_log(self, text: str):
        self._log.appendPlainText(text)

    def _on_maps_ready(self, result: dict):
        self._last_result = result
        self._canvas.show_heatmaps(result)
        self._stats_widget.update_stats(
            result["stats"],
            result["measurements"],
            result["image_shape"]
        )
        self._tabs.setCurrentIndex(0)

    def _on_finished(self, result: dict):
        self._set_idle()
        if result.get("success"):
            n   = result["n_stars"]
            fm  = result["fwhm_median"]
            fu  = result["fwhm_unit"]
            em  = result["ellip_median"]
            self._status.setText(
                f"✓  {n} stars  |  Median FWHM: {fm:.2f} {fu}  |  "
                f"Ellipticity: {em:.3f}")
            self._save_btn.setEnabled(True)

            # Auto-export PNG if requested
            if (self._export_png_chk.isChecked() and
                    self._last_result and
                    self._output_dir_edit.text().strip()):
                base = os.path.splitext(
                    os.path.basename(self._fits_edit.text().strip()))[0]
                out  = os.path.join(
                    self._output_dir_edit.text().strip(),
                    f"psf_heatmap_{base}.png")
                try:
                    self._canvas.save_png(out, dpi=150)
                    self._append_log(f"PNG saved: {out}")
                except Exception as e:
                    self._append_log(f"PNG save error: {e}")
        else:
            err = result.get("error", "Unknown error")
            self._status.setText(f"✕  {err}")
            self._append_log(f"FAILED: {err}")

    # ── save PNG ──────────────────────────────────────────────────────────────
    def _save_png(self):
        if not self._last_result:
            return

        base = os.path.splitext(
            os.path.basename(self._fits_edit.text().strip()))[0]
        suggest = f"psf_heatmap_{base}.png"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save heatmap PNG", suggest,
            "PNG images (*.png)")
        if not path:
            return

        # DPI selector
        dpi_combo = QComboBox()
        dpi_combo.addItems(["150 dpi (screen)", "300 dpi (print)"])

        dlg = QWidget(self, Qt.WindowType.Dialog)
        dlg.setWindowTitle("Save PNG")
        dlg.setWindowFlags(Qt.WindowType.Dialog)
        dlg.resize(280, 100)
        dv  = QVBoxLayout(dlg)
        dv.addWidget(QLabel("Select resolution:"))
        dv.addWidget(dpi_combo)
        btn_row = QHBoxLayout()
        ok_btn  = QPushButton("Save")
        ok_btn.setObjectName("primary")
        can_btn = QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(can_btn)
        dv.addLayout(btn_row)
        dlg.show()

        chosen_dpi = [150]

        def do_save():
            chosen_dpi[0] = 150 if "150" in dpi_combo.currentText() else 300
            dlg.close()
            try:
                self._canvas.save_png(path, dpi=chosen_dpi[0])
                self._append_log(f"Saved: {path} ({chosen_dpi[0]} dpi)")
                self._status.setText(f"PNG saved: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save error", str(e))

        ok_btn.clicked.connect(do_save)
        can_btn.clicked.connect(dlg.close)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
