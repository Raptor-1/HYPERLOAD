"""
blind_deconv.py - Blind PSF Deconvolution for Siril
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

# ── Standard library ──────────────────────────────────────────────────────────
import os, sys, csv, json, threading
from datetime import datetime

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ── PyQt6 ─────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QFrame, QSizePolicy, QStackedWidget,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ── Optional integrations ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

try:
    from psf_heatmap import (
        detect_stars_for_psf,
        fit_psf_2d_gaussian,
        measure_psf_across_field,
    )
    HAS_PSF_HEATMAP = True
except ImportError:
    HAS_PSF_HEATMAP = False


# ══════════════════════════════════════════════════════════════════════════════
# THEME
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
# PSF FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def estimate_psf_from_image(image: np.ndarray,
                              n_stars: int = 20,
                              psf_size: int = 64,
                              threshold_sigma: float = 10.0,
                              log_callback=None) -> np.ndarray | None:
    """Estimate the image PSF by averaging cutouts around bright isolated stars."""
    h, w = image.shape
    half  = psf_size // 2
    border = half + 5

    _, median, std = sigma_clipped_stats(image, sigma=3.0)
    if std <= 0:
        return None

    daofind = DAOStarFinder(
        fwhm=5.0,
        threshold=threshold_sigma * std,
        sharpness_range=(0.3, 0.9),
        roundness_range=(-0.4, 0.4),
    )
    sources = daofind(image - median)
    if sources is None or len(sources) == 0:
        if log_callback:
            log_callback("  No stars found for PSF estimation")
        return None

    sources.sort("peak")
    sources.reverse()

    sat_limit = 0.9 * float(np.percentile(image, 99.99))
    psf_stack = []

    # Detect x/y column names - these changed across photutils versions:
    # < 0.7: 'x', 'y'
    # 0.7+:  'xcentroid', 'ycentroid'
    # some builds: 'x_mean', 'y_mean'  or  'x_0', 'y_0'
    col_names = sources.colnames
    xcol = ycol = None
    for xc in ('x_centroid', 'xcentroid', 'x_mean', 'x_0', 'x'):
        if xc in col_names:
            xcol = xc
            break
    for yc in ('y_centroid', 'ycentroid', 'y_mean', 'y_0', 'y'):
        if yc in col_names:
            ycol = yc
            break
    if xcol is None or ycol is None:
        if log_callback:
            log_callback(f"  Cannot find x/y columns. Available: {col_names}")
        return None
    if log_callback:
        log_callback(f"  Star columns: x={xcol}, y={ycol}")

    for src in sources:
        x0 = int(round(float(src[xcol])))
        y0 = int(round(float(src[ycol])))

        if (x0 < border or x0 >= w - border or
                y0 < border or y0 >= h - border):
            continue
        if float(src["peak"]) + median > sat_limit:
            continue

        cutout = image[y0-half:y0+half, x0-half:x0+half].astype(np.float64)
        if cutout.shape != (psf_size, psf_size):
            continue

        corners = np.concatenate([
            cutout[:4,  :4].ravel(), cutout[:4,  -4:].ravel(),
            cutout[-4:, :4].ravel(), cutout[-4:, -4:].ravel(),
        ])
        bg = float(np.median(corners))
        cutout -= bg
        cutout  = np.clip(cutout, 0, None)

        total = cutout.sum()
        if total <= 0:
            continue

        psf_stack.append(cutout / total)
        if len(psf_stack) >= n_stars:
            break

    if not psf_stack:
        if log_callback:
            log_callback("  PSF estimation failed - no suitable stars found")
        return None

    if log_callback:
        log_callback(f"  PSF estimated from {len(psf_stack)} stars")

    avg_psf  = np.median(psf_stack, axis=0)
    avg_psf /= avg_psf.sum()

    psf_fft_conv = np.roll(np.roll(avg_psf, -half, axis=0), -half, axis=1)
    return psf_fft_conv.astype(np.float32)


def make_gaussian_psf(fwhm_px: float,
                       image_shape: tuple,
                       ellipticity: float = 0.0,
                       angle_deg: float = 0.0) -> np.ndarray:
    """Build an analytical Gaussian PSF of given FWHM."""
    h, w     = image_shape
    sigma_m  = fwhm_px / 2.355
    sigma_mi = sigma_m * (1.0 - ellipticity)
    theta    = np.radians(angle_deg)

    y, x = np.ogrid[:h, :w]
    y = np.where(y > h // 2, y - h, y).astype(float)
    x = np.where(x > w // 2, x - w, x).astype(float)

    ct, st = np.cos(theta), np.sin(theta)
    xr =  ct * x + st * y
    yr = -st * x + ct * y

    psf  = np.exp(-0.5 * (xr**2 / sigma_m**2 + yr**2 / sigma_mi**2))
    psf /= psf.sum()
    return psf.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DECONVOLUTION ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def wiener_deconvolve(image: np.ndarray,
                       psf: np.ndarray,
                       epsilon: float = 0.01) -> np.ndarray:
    """Wiener deconvolution in Fourier space."""
    F = fft2(image.astype(np.float64))
    H = fft2(psf.astype(np.float64))

    H_abs2     = np.abs(H) ** 2
    eps_scaled = epsilon * H_abs2.max()

    W      = np.conj(H) / (H_abs2 + eps_scaled + 1e-30)
    result = np.real(ifft2(F * W))
    return np.clip(result, 0, None).astype(np.float32)


def richardson_lucy(image: np.ndarray,
                     psf: np.ndarray,
                     iterations: int = 30,
                     acceleration: float = 1.0,
                     convergence_tol: float = 1e-4,
                     log_callback=None,
                     cancel_event=None) -> np.ndarray:
    """Richardson-Lucy iterative deconvolution."""
    obs      = image.astype(np.float64)
    obs      = np.clip(obs, 1e-10, None)
    H        = fft2(psf.astype(np.float64))
    H_flip   = np.conj(H)

    estimate = obs.copy()
    prev_estimate = None

    for i in range(iterations):
        if cancel_event and cancel_event.is_set():
            break

        model    = np.real(ifft2(fft2(estimate) * H))
        model    = np.clip(model, 1e-10, None)

        ratio    = obs / model
        correction = np.real(ifft2(fft2(ratio) * H_flip))
        correction = np.clip(correction, 1e-10, None)

        if acceleration != 1.0:
            estimate = estimate * (correction ** acceleration)
        else:
            estimate = estimate * correction

        estimate = np.clip(estimate, 0, None)

        if i > 0 and i % 5 == 0 and prev_estimate is not None:
            rel_change = (np.mean(np.abs(estimate - prev_estimate)) /
                          (np.mean(prev_estimate) + 1e-10))
            if log_callback and i % 10 == 0:
                log_callback(
                    f"  RL iteration {i+1}/{iterations}  Δ={rel_change:.2e}")
            if rel_change < convergence_tol:
                if log_callback:
                    log_callback(
                        f"  Converged at iteration {i+1} "
                        f"(Δ={rel_change:.2e} < {convergence_tol})")
                break

        if i % 5 == 0:
            prev_estimate = estimate.copy()

    return estimate.astype(np.float32)


def tikhonov_deconvolve(image: np.ndarray,
                         psf: np.ndarray,
                         lam: float = 0.01) -> np.ndarray:
    """Tikhonov (gradient-regularised) deconvolution."""
    h, w = image.shape

    y_freq = np.fft.fftfreq(h) * 2 * np.pi
    x_freq = np.fft.fftfreq(w) * 2 * np.pi
    Y, X   = np.meshgrid(y_freq, x_freq, indexing="ij")
    L2     = (2*np.cos(X) - 2 + 2*np.cos(Y) - 2) ** 2

    F      = fft2(image.astype(np.float64))
    H      = fft2(psf.astype(np.float64))
    H_abs2 = np.abs(H) ** 2

    W      = np.conj(H) / (H_abs2 + lam * L2 + 1e-30)
    result = np.real(ifft2(F * W))
    return np.clip(result, 0, None).astype(np.float32)


def spatially_varying_deconvolve(image: np.ndarray,
                                   measurements: list,
                                   algorithm: str = "wiener",
                                   epsilon: float = 0.01,
                                   iterations: int = 30,
                                   n_zones: int = 9,
                                   log_callback=None,
                                   cancel_event=None) -> np.ndarray:
    """Apply deconvolution with spatially-varying PSF."""
    h, w         = image.shape
    zone_h       = h // n_zones
    zone_w       = w // n_zones
    overlap_frac = 0.5
    result       = np.zeros((h, w), dtype=np.float64)
    weights      = np.zeros((h, w), dtype=np.float64)

    total_zones = n_zones * n_zones
    zone_idx    = 0

    for zi in range(n_zones):
        for zj in range(n_zones):
            if cancel_event and cancel_event.is_set():
                return image.copy()

            zone_idx += 1
            if log_callback and zone_idx % 3 == 1:
                log_callback(f"  Zone {zone_idx}/{total_zones} ...")

            cy = int((zi + 0.5) * zone_h)
            cx = int((zj + 0.5) * zone_w)

            hs  = int(zone_h * (1 + overlap_frac) / 2)
            ws  = int(zone_w * (1 + overlap_frac) / 2)
            y1  = max(0, cy - hs)
            y2  = min(h, cy + hs)
            x1  = max(0, cx - ws)
            x2  = min(w, cx + ws)
            zone_data = image[y1:y2, x1:x2]

            search_r = max(zone_h, zone_w) * 2
            nearby   = [m for m in measurements
                        if ((m["x"] - cx)**2 + (m["y"] - cy)**2)**0.5 < search_r]

            if len(nearby) >= 3:
                fwhm_med  = float(np.median([m["fwhm_px"]     for m in nearby]))
                ell_med   = float(np.median([m["ellipticity"] for m in nearby]))
                ang_med   = float(np.median([m["angle_deg"]   for m in nearby]))
                local_psf = make_gaussian_psf(fwhm_med, zone_data.shape, ell_med, ang_med)
            elif measurements:
                fwhm_med  = float(np.median([m["fwhm_px"]     for m in measurements]))
                ell_med   = float(np.median([m["ellipticity"] for m in measurements]))
                ang_med   = float(np.median([m["angle_deg"]   for m in measurements]))
                local_psf = make_gaussian_psf(fwhm_med, zone_data.shape, ell_med, ang_med)
            else:
                local_psf = make_gaussian_psf(3.0, zone_data.shape)

            if algorithm == "wiener":
                zone_result = wiener_deconvolve(zone_data, local_psf, epsilon)
            elif algorithm == "richardson_lucy":
                zone_result = richardson_lucy(
                    zone_data, local_psf, iterations,
                    cancel_event=cancel_event)
            else:
                zone_result = tikhonov_deconvolve(zone_data, local_psf, epsilon)

            gy  = np.exp(-0.5 * ((np.arange(y2-y1) - (y2-y1)/2) /
                                   ((y2-y1)/3))**2)
            gx  = np.exp(-0.5 * ((np.arange(x2-x1) - (x2-x1)/2) /
                                   ((x2-x1)/3))**2)
            w2d = np.outer(gy, gx)

            result [y1:y2, x1:x2] += zone_result.astype(np.float64) * w2d
            weights[y1:y2, x1:x2] += w2d

    mask   = weights > 1e-10
    output = np.where(mask, result / (weights + 1e-10), image)
    return np.clip(output, 0, None).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def check_linearity(data: np.ndarray) -> tuple:
    """Check if image appears linear or already stretched."""
    flat = data.flatten()
    mn   = float(np.min(flat))
    mx   = float(np.max(flat))
    if mx <= mn:
        return True, 0.0
    med  = float(np.median(flat))
    frac = (med - mn) / (mx - mn)
    return frac < 0.25, frac


def load_psf_measurements_csv(csv_path: str) -> list:
    """Load per-star PSF measurements saved by psf_heatmap.py."""
    measurements = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                measurements.append({
                    "x":           float(row["x"]),
                    "y":           float(row["y"]),
                    "fwhm_px":     float(row["fwhm_px"]),
                    "ellipticity": float(row["ellipticity"]),
                    "angle_deg":   float(row["angle_deg"]),
                    "peak":        float(row.get("peak", 0)),
                })
            except (ValueError, KeyError):
                continue
    return measurements


def estimate_psf_fwhm(psf: np.ndarray) -> float:
    """Estimate FWHM in pixels from a PSF array (simple radial profile)."""
    h, w = psf.shape
    cy, cx = h // 2, w // 2
    peak = float(psf.max())
    if peak <= 0:
        return 0.0
    half = peak / 2.0
    # Scan outward from centre along x-axis
    for r in range(1, min(cx, cy)):
        if psf[cy, cx + r] < half:
            return r * 2.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class DeconvWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    psf_ready     = pyqtSignal(object)
    preview_ready = pyqtSignal(object, object)
    finished      = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg = self.config

            # Step 1: Load image
            self.progress.emit(1, 5, "Loading image...")
            with fits.open(cfg["fits_path"]) as hdul:
                data   = hdul[0].data.astype(np.float32)
                header = hdul[0].header.copy()

            is_rgb = (data.ndim == 3)
            if is_rgb:
                if data.shape[0] == 3:
                    channels = [data[c] for c in range(3)]
                    layout   = "CHW"
                else:
                    channels = [data[:,:,c] for c in range(3)]
                    layout   = "HWC"
                lum = (0.299*channels[0] + 0.587*channels[1] +
                       0.114*channels[2])
            else:
                lum    = data
                layout = "mono"

            h, w = lum.shape
            self.log_line.emit(
                f"Loaded: {w}×{h} px  {os.path.basename(cfg['fits_path'])}")

            is_linear, frac = check_linearity(lum)
            if not is_linear:
                self.log_line.emit(
                    f"⚠ WARNING: image median at {frac*100:.0f}% of range - "
                    "may already be stretched. Deconvolution works best on linear data.")

            if self._cancel.is_set():
                self._abort(); return

            # Step 2: Build PSF
            self.progress.emit(2, 5, "Estimating PSF...")
            psf_mode = cfg.get("psf_mode", "blind")

            if psf_mode == "blind":
                psf_small = estimate_psf_from_image(
                    lum,
                    n_stars         = cfg.get("n_psf_stars", 20),
                    psf_size        = cfg.get("psf_size", 64),
                    threshold_sigma = cfg.get("threshold_sigma", 10.0),
                    log_callback    = self.log_line.emit
                )
                if psf_small is None:
                    self.finished.emit({
                        "success": False,
                        "error":   "PSF estimation failed - no suitable stars found. "
                                   "Try lowering detection threshold."
                    })
                    return
                psf_full = np.zeros((h, w), dtype=np.float32)
                ph, pw   = psf_small.shape
                psf_full[:ph, :pw] = psf_small

            elif psf_mode == "gaussian":
                fwhm_px  = cfg.get("fwhm_px", 3.0)
                ell      = cfg.get("ellipticity", 0.0)
                ang      = cfg.get("angle_deg", 0.0)
                psf_full = make_gaussian_psf(fwhm_px, (h, w), ell, ang)
                self.log_line.emit(
                    f"PSF: Gaussian FWHM={fwhm_px:.1f}px  "
                    f"ellipticity={ell:.3f}  angle={ang:.1f}deg")

            else:  # load from FITS
                psf_data = fits.getdata(cfg["psf_fits_path"]).astype(np.float32)
                if psf_data.ndim == 3:
                    psf_data = psf_data[0]
                psf_full = np.zeros((h, w), dtype=np.float32)
                ph, pw   = psf_data.shape
                psf_full[:ph, :pw] = psf_data
                self.log_line.emit(f"PSF loaded from: {cfg['psf_fits_path']}")

            psf_display_size = min(128, cfg.get("psf_size", 64) * 2)
            psf_disp = psf_full[:psf_display_size, :psf_display_size].copy()
            self.psf_ready.emit(psf_disp)

            if self._cancel.is_set():
                self._abort(); return

            # Step 3: Deconvolve
            self.progress.emit(3, 5, "Deconvolving...")
            algorithm = cfg.get("algorithm", "wiener")
            sv_mode   = cfg.get("spatially_varying", False)

            def deconv_channel(ch: np.ndarray) -> np.ndarray:
                if sv_mode and HAS_PSF_HEATMAP and cfg.get("measurements"):
                    return spatially_varying_deconvolve(
                        ch, cfg["measurements"],
                        algorithm    = algorithm,
                        epsilon      = cfg.get("epsilon", 0.01),
                        iterations   = cfg.get("iterations", 30),
                        n_zones      = cfg.get("n_zones", 9),
                        log_callback = self.log_line.emit,
                        cancel_event = self._cancel
                    )
                elif algorithm == "wiener":
                    return wiener_deconvolve(ch, psf_full, cfg.get("epsilon", 0.01))
                elif algorithm == "richardson_lucy":
                    return richardson_lucy(
                        ch, psf_full,
                        iterations   = cfg.get("iterations", 30),
                        acceleration = cfg.get("acceleration", 1.0),
                        log_callback = self.log_line.emit,
                        cancel_event = self._cancel)
                else:
                    return tikhonov_deconvolve(ch, psf_full, cfg.get("lam", 0.01))

            if is_rgb:
                result_channels = []
                for ci, ch in enumerate(channels):
                    if self._cancel.is_set():
                        self._abort(); return
                    self.log_line.emit(f"  Processing channel {ci+1}/3...")
                    result_channels.append(deconv_channel(ch))
                if layout == "CHW":
                    result = np.stack(result_channels, axis=0)
                else:
                    result = np.stack(result_channels, axis=2)
                result_lum = (0.299*result_channels[0] +
                               0.587*result_channels[1] +
                               0.114*result_channels[2])
            else:
                result     = deconv_channel(lum)
                result_lum = result

            self.preview_ready.emit(lum, result_lum)

            if self._cancel.is_set():
                self._abort(); return

            # Step 4: Save
            self.progress.emit(4, 5, "Saving result...")
            out_dir  = cfg.get("output_dir", os.path.dirname(cfg["fits_path"]))
            os.makedirs(out_dir, exist_ok=True)
            base     = os.path.splitext(os.path.basename(cfg["fits_path"]))[0]
            algo_tag = {"wiener": "W", "richardson_lucy": "RL",
                        "tikhonov": "T"}.get(algorithm, "D")
            out_path = os.path.join(out_dir, f"{base}_deconv_{algo_tag}.fit")

            hdu = fits.PrimaryHDU(data=result.astype(np.float32), header=header)
            hdu.header["HISTORY"] = (
                f"Blind deconvolution ({algorithm}) - blind_deconv.py")
            hdu.writeto(out_path, overwrite=True)
            self.log_line.emit(f"Saved: {out_path}")

            # Step 5: Load in Siril
            self.progress.emit(5, 5, "Done")
            if cfg.get("load_in_siril", True):
                try:
                    siril = s.SirilInterface()
                    siril.connect()
                    siril.cmd("load", out_path)
                    siril.disconnect()
                except Exception as e:
                    self.log_line.emit(f"Siril load warning: {e}")

            self.finished.emit({
                "success":     True,
                "output_path": out_path,
                "algorithm":   algorithm,
                "psf_mode":    psf_mode,
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


class PSFPreviewWorker(QThread):
    """Lightweight worker: only estimates PSF, no deconvolution."""
    log_line  = pyqtSignal(str)
    psf_ready = pyqtSignal(object, float)
    finished  = pyqtSignal(bool, str)

    def __init__(self, fits_path: str, n_stars: int, psf_size: int,
                 threshold: float, cancel_event: threading.Event):
        super().__init__()
        self._path      = fits_path
        self._n_stars   = n_stars
        self._psf_size  = psf_size
        self._threshold = threshold
        self._cancel    = cancel_event

    def run(self):
        try:
            with fits.open(self._path) as hdul:
                data = hdul[0].data.astype(np.float32)
            if data.ndim == 3:
                if data.shape[0] == 3:
                    lum = (0.299*data[0] + 0.587*data[1] + 0.114*data[2])
                else:
                    lum = (0.299*data[:,:,0] + 0.587*data[:,:,1] + 0.114*data[:,:,2])
            else:
                lum = data

            self.log_line.emit("Estimating PSF from stars...")
            psf_small = estimate_psf_from_image(
                lum,
                n_stars         = self._n_stars,
                psf_size        = self._psf_size,
                threshold_sigma = self._threshold,
                log_callback    = self.log_line.emit
            )
            if psf_small is None:
                self.finished.emit(False, "No suitable stars found")
                return

            # Shift centre to centre of array for display
            half = self._psf_size // 2
            psf_display = np.roll(np.roll(psf_small, half, axis=0), half, axis=1)
            fwhm_est = estimate_psf_fwhm(psf_display)
            self.log_line.emit(f"  Estimated FWHM ≈ {fwhm_est:.1f} px")
            self.psf_ready.emit(psf_display, fwhm_est)
            self.finished.emit(True, "")
        except Exception as e:
            import traceback
            self.log_line.emit(f"PSF preview error: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit(False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CANVASES
# ══════════════════════════════════════════════════════════════════════════════

class ResultCanvas(FigureCanvasQTAgg):
    """Side-by-side: original vs deconvolved."""
    def __init__(self, parent=None):
        self.fig    = Figure(figsize=(9, 4), facecolor="#1e2128")
        self.ax_in  = self.fig.add_subplot(1, 2, 1)
        self.ax_out = self.fig.add_subplot(1, 2, 2)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(280)

    def _style(self):
        for ax, title, color in [
            (self.ax_in,  "Original",    "#e8a23a"),
            (self.ax_out, "Deconvolved", "#4caf7d"),
        ]:
            ax.set_facecolor("#1e2128")
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")

    def _disp(self, arr, vmin=None, vmax=None):
        if vmin is None:
            vmin = float(np.percentile(arr, 0.5))
        if vmax is None:
            vmax = float(np.percentile(arr, 99.5))
        return np.clip((arr.astype(float) - vmin) / (vmax - vmin + 1e-10), 0, 1)

    def _ds(self, arr, max_px=500):
        h, w   = arr.shape
        factor = max(1, max(h, w) // max_px)
        return arr[::factor, ::factor]

    def show_pair(self, original: np.ndarray, deconvolved: np.ndarray):
        p_lo = float(np.percentile(original, 0.5))
        p_hi = float(np.percentile(original, 99.5))
        for ax, data, title, color in [
            (self.ax_in,  original,    "Original",    "#e8a23a"),
            (self.ax_out, deconvolved, "Deconvolved", "#4caf7d"),
        ]:
            ax.clear()
            ax.imshow(self._ds(self._disp(data, p_lo, p_hi)),
                      cmap="gray", origin="lower",
                      aspect="equal", interpolation="nearest")
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_placeholder(self):
        for ax, title, color in [
            (self.ax_in,  "Original",    "#e8a23a"),
            (self.ax_out, "Deconvolved", "#4caf7d"),
        ]:
            ax.clear()
            ax.set_facecolor("#1e2128")
            ax.text(0.5, 0.5, "Run deconvolution\nto see result",
                    ha="center", va="center",
                    color=SIRIL_TEXT_DIM, fontsize=9,
                    transform=ax.transAxes)
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")
        self.fig.tight_layout(pad=0.3)
        self.draw()


class PSFCanvas(FigureCanvasQTAgg):
    """Shows estimated/specified PSF with log-stretch for faint wings."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 3), facecolor="#1e2128")
        self.ax  = self.fig.add_subplot(111)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(220)

    def _style(self):
        self.ax.set_facecolor("#1e2128")
        self.ax.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_color("#3a4055")
        self.ax.set_title("Estimated PSF (log stretch)",
                           color=SIRIL_SECTION, fontsize=8)

    def show_psf(self, psf: np.ndarray):
        self.ax.clear()
        self._style()
        safe    = np.clip(psf.astype(float), 1e-6, None)
        log_psf = np.log10(safe)
        log_psf -= log_psf.min()
        self.ax.imshow(log_psf, cmap="hot", origin="lower",
                       aspect="equal", interpolation="nearest")
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_placeholder(self):
        self.ax.clear()
        self._style()
        self.ax.text(0.5, 0.5, "PSF will appear\nafter estimation",
                     ha="center", va="center",
                     color=SIRIL_TEXT_DIM, fontsize=9,
                     transform=self.ax.transAxes)
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blind PSF Deconvolution - Siril")
        self.resize(1200, 760)

        self._worker        = None
        self._psf_worker    = None
        self._cancel_event  = threading.Event()
        self._measurements  = []
        self._loaded_path   = ""

        self._build_ui()
        self._connect_signals()
        self._on_psf_mode_changed(0)
        self._on_algo_changed(0)

    # ── UI BUILD ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Title bar
        title_bar = self._make_title_bar()
        root_layout.addWidget(title_bar)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(340)
        left_scroll.setMaximumWidth(420)

        left_panel = self._build_left_panel()
        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter, 1)

        # Bottom bar
        bottom = self._build_bottom_bar()
        root_layout.addWidget(bottom)

        # Status bar
        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("dim")
        self._status_label.setContentsMargins(8, 2, 8, 4)
        root_layout.addWidget(self._status_label)

        self.setCentralWidget(root)

    def _make_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet(
            f"background-color: {SIRIL_BG3}; "
            f"border-bottom: 1px solid {SIRIL_BORDER};")
        bar.setFixedHeight(38)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 0, 12, 0)

        lbl_left = QLabel("🔭  Blind PSF Deconvolution  -  Siril")
        lbl_left.setStyleSheet(
            f"color: {SIRIL_ACCENT}; font-weight: bold; font-size: 11pt;")
        lay.addWidget(lbl_left)
        lay.addStretch()

        lbl_right = QLabel("v1.0  |  Wiener · RL · Tikhonov")
        lbl_right.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 9pt;")
        lay.addWidget(lbl_right)
        return bar

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        lay.addWidget(self._build_input_group())
        lay.addWidget(self._build_psf_group())
        lay.addWidget(self._build_algo_group())
        lay.addWidget(self._build_sv_group())
        lay.addWidget(self._build_output_group())
        lay.addStretch()
        return panel

    # ── INPUT GROUP ────────────────────────────────────────────────────────────

    def _build_input_group(self) -> QGroupBox:
        grp = QGroupBox("Input image")
        lay = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._fits_edit = QLineEdit()
        self._fits_edit.setPlaceholderText("Path to FITS file...")
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(64)
        btn_browse.clicked.connect(self._browse_fits)
        row.addWidget(self._fits_edit)
        row.addWidget(btn_browse)
        lay.addLayout(row)

        self._btn_check = QPushButton("🔍  Load & check linearity")
        self._btn_check.clicked.connect(self._check_linearity)
        lay.addWidget(self._btn_check)

        self._lbl_image_info = QLabel("")
        self._lbl_image_info.setObjectName("dim")
        self._lbl_image_info.setWordWrap(True)
        lay.addWidget(self._lbl_image_info)

        self._lbl_stretch_warn = QLabel(
            "⚠ Image may be stretched. Use linear FITS for best results.")
        self._lbl_stretch_warn.setObjectName("warn")
        self._lbl_stretch_warn.setWordWrap(True)
        self._lbl_stretch_warn.setVisible(False)
        lay.addWidget(self._lbl_stretch_warn)

        return grp

    # ── PSF GROUP ──────────────────────────────────────────────────────────────

    def _build_psf_group(self) -> QGroupBox:
        grp = QGroupBox("PSF mode")
        lay = QVBoxLayout(grp)

        self._psf_mode_combo = QComboBox()
        self._psf_mode_combo.addItems([
            "Blind (estimate from stars)  - recommended",
            "Gaussian (specify FWHM manually)",
            "Load PSF from FITS file",
        ])
        lay.addWidget(self._psf_mode_combo)

        self._psf_stack = QStackedWidget()

        # Page 0: Blind
        blind_page = QWidget()
        blind_lay  = QVBoxLayout(blind_page)
        blind_lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setSpacing(4)

        self._n_stars_spin = QSpinBox()
        self._n_stars_spin.setRange(5, 50)
        self._n_stars_spin.setValue(20)
        form.addRow("Stars for PSF:", self._n_stars_spin)

        self._psf_size_combo = QComboBox()
        self._psf_size_combo.addItems(["32 px", "64 px", "128 px"])
        self._psf_size_combo.setCurrentIndex(1)
        form.addRow("PSF array size:", self._psf_size_combo)

        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(5.0, 20.0)
        self._thresh_spin.setValue(10.0)
        self._thresh_spin.setSingleStep(0.5)
        self._thresh_spin.setSuffix(" σ")
        form.addRow("Detection thresh:", self._thresh_spin)

        blind_lay.addLayout(form)

        self._btn_preview_psf = QPushButton("🔬  Preview estimated PSF")
        self._btn_preview_psf.clicked.connect(self._preview_psf)
        blind_lay.addWidget(self._btn_preview_psf)

        self._psf_stack.addWidget(blind_page)

        # Page 1: Gaussian
        gauss_page = QWidget()
        gauss_lay  = QVBoxLayout(gauss_page)
        gauss_lay.setContentsMargins(0, 0, 0, 0)
        gform = QFormLayout()
        gform.setSpacing(4)

        self._fwhm_spin = QDoubleSpinBox()
        self._fwhm_spin.setRange(0.5, 30.0)
        self._fwhm_spin.setValue(3.0)
        self._fwhm_spin.setSingleStep(0.5)
        self._fwhm_spin.setSuffix(" px")
        gform.addRow("FWHM:", self._fwhm_spin)

        self._ell_spin = QDoubleSpinBox()
        self._ell_spin.setRange(0.0, 0.8)
        self._ell_spin.setValue(0.0)
        self._ell_spin.setSingleStep(0.05)
        gform.addRow("Ellipticity:", self._ell_spin)

        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(0.0, 180.0)
        self._angle_spin.setValue(0.0)
        self._angle_spin.setSingleStep(5.0)
        self._angle_spin.setSuffix(" deg")
        gform.addRow("Angle:", self._angle_spin)

        gauss_lay.addLayout(gform)
        lbl_hint = QLabel("Values from PSF Heatmap (psf_heatmap.py)")
        lbl_hint.setObjectName("dim")
        lbl_hint.setWordWrap(True)
        gauss_lay.addWidget(lbl_hint)

        self._psf_stack.addWidget(gauss_page)

        # Page 2: Load FITS
        fits_page = QWidget()
        fits_lay  = QVBoxLayout(fits_page)
        fits_lay.setContentsMargins(0, 0, 0, 0)

        frow = QHBoxLayout()
        self._psf_fits_edit = QLineEdit()
        self._psf_fits_edit.setPlaceholderText("Path to PSF FITS...")
        btn_browse_psf = QPushButton("Browse")
        btn_browse_psf.setFixedWidth(64)
        btn_browse_psf.clicked.connect(self._browse_psf_fits)
        frow.addWidget(self._psf_fits_edit)
        frow.addWidget(btn_browse_psf)
        fits_lay.addLayout(frow)

        lbl_psf_hint = QLabel("PSF FITS must be centred and normalised to sum=1")
        lbl_psf_hint.setObjectName("dim")
        lbl_psf_hint.setWordWrap(True)
        fits_lay.addWidget(lbl_psf_hint)

        self._psf_stack.addWidget(fits_page)

        lay.addWidget(self._psf_stack)
        return grp

    # ── ALGORITHM GROUP ────────────────────────────────────────────────────────

    def _build_algo_group(self) -> QGroupBox:
        grp = QGroupBox("Algorithm")
        lay = QVBoxLayout(grp)

        self._algo_combo = QComboBox()
        self._algo_combo.addItems([
            "Wiener  (fast, balanced)",
            "Richardson-Lucy  (iterative, flux-preserving)",
            "Tikhonov  (smooth, minimal ringing)",
        ])
        lay.addWidget(self._algo_combo)

        self._algo_hint = QLabel("")
        self._algo_hint.setObjectName("dim")
        self._algo_hint.setWordWrap(True)
        lay.addWidget(self._algo_hint)

        self._algo_params_stack = QStackedWidget()

        # Page 0: Wiener
        w_page = QWidget()
        w_lay  = QVBoxLayout(w_page)
        w_lay.setContentsMargins(0, 4, 0, 0)

        reg_row = QHBoxLayout()
        lbl_reg = QLabel("Regularisation:")
        reg_row.addWidget(lbl_reg)
        self._w_slider = QSlider(Qt.Orientation.Horizontal)
        self._w_slider.setRange(0, 100)
        self._w_slider.setValue(50)
        reg_row.addWidget(self._w_slider, 1)
        self._w_val_lbl = QLabel("0.010")
        self._w_val_lbl.setFixedWidth(42)
        reg_row.addWidget(self._w_val_lbl)
        w_lay.addLayout(reg_row)

        w_hint = QLabel("Lower = sharper · Higher = smoother")
        w_hint.setObjectName("dim")
        w_lay.addWidget(w_hint)
        self._algo_params_stack.addWidget(w_page)

        # Page 1: RL
        rl_page = QWidget()
        rl_lay  = QVBoxLayout(rl_page)
        rl_lay.setContentsMargins(0, 4, 0, 0)
        rl_form = QFormLayout()
        rl_form.setSpacing(4)

        self._rl_iter_spin = QSpinBox()
        self._rl_iter_spin.setRange(5, 200)
        self._rl_iter_spin.setValue(30)
        rl_form.addRow("Iterations:", self._rl_iter_spin)

        self._rl_accel_spin = QDoubleSpinBox()
        self._rl_accel_spin.setRange(1.0, 2.0)
        self._rl_accel_spin.setValue(1.0)
        self._rl_accel_spin.setSingleStep(0.1)
        rl_form.addRow("Acceleration:", self._rl_accel_spin)

        self._rl_conv_spin = QDoubleSpinBox()
        self._rl_conv_spin.setRange(1e-5, 1e-3)
        self._rl_conv_spin.setValue(1e-4)
        self._rl_conv_spin.setSingleStep(1e-5)
        self._rl_conv_spin.setDecimals(5)
        rl_form.addRow("Convergence:", self._rl_conv_spin)

        rl_lay.addLayout(rl_form)
        rl_hint = QLabel("More iterations = sharper · risk ringing at >50")
        rl_hint.setObjectName("dim")
        rl_lay.addWidget(rl_hint)
        self._algo_params_stack.addWidget(rl_page)

        # Page 2: Tikhonov (reuse same slider-style layout)
        t_page = QWidget()
        t_lay  = QVBoxLayout(t_page)
        t_lay.setContentsMargins(0, 4, 0, 0)

        t_row = QHBoxLayout()
        lbl_t = QLabel("Regularisation λ:")
        t_row.addWidget(lbl_t)
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setRange(0, 100)
        self._t_slider.setValue(50)
        t_row.addWidget(self._t_slider, 1)
        self._t_val_lbl = QLabel("0.010")
        self._t_val_lbl.setFixedWidth(42)
        t_row.addWidget(self._t_val_lbl)
        t_lay.addLayout(t_row)

        t_hint = QLabel("Lower = sharper · Higher = smoother")
        t_hint.setObjectName("dim")
        t_lay.addWidget(t_hint)
        self._algo_params_stack.addWidget(t_page)

        lay.addWidget(self._algo_params_stack)

        # Per-channel PSF checkbox
        self._per_channel_chk = QCheckBox("Estimate PSF per channel (narrowband)")
        self._per_channel_chk.setChecked(False)
        lay.addWidget(self._per_channel_chk)

        return grp

    # ── SPATIALLY VARYING GROUP ────────────────────────────────────────────────

    def _build_sv_group(self) -> QGroupBox:
        grp = QGroupBox("Spatially varying PSF")
        lay = QVBoxLayout(grp)

        self._sv_chk = QCheckBox("Use spatially varying PSF (from psf_heatmap.py)")
        if not HAS_PSF_HEATMAP:
            self._sv_chk.setEnabled(False)
        self._sv_chk.toggled.connect(self._on_sv_toggled)
        lay.addWidget(self._sv_chk)

        if not HAS_PSF_HEATMAP:
            lbl_warn = QLabel("⚠ Requires psf_heatmap.py in same folder")
            lbl_warn.setObjectName("warn")
            lay.addWidget(lbl_warn)

        self._sv_details = QWidget()
        sv_det_lay = QVBoxLayout(self._sv_details)
        sv_det_lay.setContentsMargins(0, 4, 0, 0)

        lbl_csv = QLabel("PSF measurements CSV from psf_heatmap.py:")
        lbl_csv.setObjectName("dim")
        sv_det_lay.addWidget(lbl_csv)

        csv_row = QHBoxLayout()
        self._csv_edit = QLineEdit()
        self._csv_edit.setPlaceholderText("Path to measurements CSV...")
        btn_csv = QPushButton("Browse")
        btn_csv.setFixedWidth(64)
        btn_csv.clicked.connect(self._browse_csv)
        csv_row.addWidget(self._csv_edit)
        csv_row.addWidget(btn_csv)
        sv_det_lay.addLayout(csv_row)

        self._lbl_meas_count = QLabel("")
        self._lbl_meas_count.setObjectName("dim")
        sv_det_lay.addWidget(self._lbl_meas_count)

        zone_row = QFormLayout()
        self._zones_spin = QSpinBox()
        self._zones_spin.setRange(1, 25)
        self._zones_spin.setValue(9)
        self._zones_spin.setToolTip("1=1×1  4=2×2  9=3×3  16=4×4")
        zone_row.addRow("Grid zones (N×N):", self._zones_spin)
        sv_det_lay.addLayout(zone_row)

        lbl_sv_hint = QLabel(
            "Different PSF correction per region. Essential for wide-field "
            "images with edge coma or tilt.")
        lbl_sv_hint.setObjectName("dim")
        lbl_sv_hint.setWordWrap(True)
        sv_det_lay.addWidget(lbl_sv_hint)

        self._sv_details.setVisible(False)
        lay.addWidget(self._sv_details)
        return grp

    # ── OUTPUT GROUP ───────────────────────────────────────────────────────────

    def _build_output_group(self) -> QGroupBox:
        grp = QGroupBox("Output")
        lay = QVBoxLayout(grp)

        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output folder (blank = same as input)")
        btn_out = QPushButton("Browse")
        btn_out.setFixedWidth(64)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit)
        out_row.addWidget(btn_out)
        lay.addLayout(out_row)

        self._load_siril_chk = QCheckBox("Load result in Siril")
        self._load_siril_chk.setChecked(True)
        lay.addWidget(self._load_siril_chk)
        return grp

    # ── RIGHT PANEL ────────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(4, 8, 8, 4)
        lay.setSpacing(6)

        self._tabs = QTabWidget()

        # Tab 0: PSF
        psf_tab = QWidget()
        psf_lay = QVBoxLayout(psf_tab)
        self._psf_canvas = PSFCanvas()
        self._psf_canvas.show_placeholder()
        psf_lay.addWidget(self._psf_canvas, 1)

        self._lbl_psf_info = QLabel("")
        self._lbl_psf_info.setObjectName("dim")
        self._lbl_psf_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        psf_lay.addWidget(self._lbl_psf_info)

        self._tabs.addTab(psf_tab, "🔬  PSF")

        # Tab 1: Result
        result_tab = QWidget()
        result_lay = QVBoxLayout(result_tab)
        self._result_canvas = ResultCanvas()
        self._result_canvas.show_placeholder()
        result_lay.addWidget(self._result_canvas, 1)
        self._tabs.addTab(result_tab, "✨  Result")

        # Tab 2: Log
        log_tab = QWidget()
        log_lay = QVBoxLayout(log_tab)
        self._log_edit = QPlainTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setProperty("readOnly", True)
        self._log_edit.setFont(QFont("Courier New", 9))
        log_lay.addWidget(self._log_edit)
        self._tabs.addTab(log_tab, "📋  Log")

        lay.addWidget(self._tabs, 1)
        return panel

    # ── BOTTOM BAR ─────────────────────────────────────────────────────────────

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet(
            f"background-color: {SIRIL_BG2}; "
            f"border-top: 1px solid {SIRIL_BORDER};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 6, 8, 6)

        self._progress = QProgressBar()
        self._progress.setRange(0, 5)
        self._progress.setValue(0)
        self._progress.setFixedHeight(8)
        self._progress.setVisible(False)
        lay.addWidget(self._progress, 1)

        self._btn_run = QPushButton("▶  Deconvolve")
        self._btn_run.setObjectName("primary")
        self._btn_run.setFixedWidth(150)
        self._btn_run.setFixedHeight(32)
        self._btn_run.clicked.connect(self._run_deconv)
        lay.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setFixedWidth(90)
        self._btn_cancel.setFixedHeight(32)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self._btn_cancel)

        return bar

    # ── SIGNALS ────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._psf_mode_combo.currentIndexChanged.connect(self._on_psf_mode_changed)
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        self._w_slider.valueChanged.connect(self._on_w_slider)
        self._t_slider.valueChanged.connect(self._on_t_slider)
        self._csv_edit.editingFinished.connect(self._load_csv_auto)

    # ── SLOT HANDLERS ──────────────────────────────────────────────────────────

    def _on_psf_mode_changed(self, idx: int):
        self._psf_stack.setCurrentIndex(idx)

    def _on_algo_changed(self, idx: int):
        self._algo_params_stack.setCurrentIndex(idx)
        hints = [
            ("Fast single-pass. Best for faint extended nebulae where noise "
             "control matters most. Parameter: epsilon (0.001=sharp, 0.1=smooth)"),
            ("Iterative, photon-preserving. Best for point sources and galaxies. "
             "More iterations = sharper but risk ringing artefacts. "
             "Stop at 10-30 for nebulae, up to 100 for star fields."),
            ("Gradient-regularised Wiener. Suppresses ringing at object edges. "
             "Best for images with hard transitions. "
             "Parameter: lambda (0.001=sharp, 0.1=smooth)"),
        ]
        self._algo_hint.setText(hints[idx])

    def _on_sv_toggled(self, checked: bool):
        self._sv_details.setVisible(checked)

    def _slider_to_epsilon(self, v: int) -> float:
        """Map slider 0-100 → log scale 0.001-0.1"""
        import math
        log_min = math.log10(0.001)
        log_max = math.log10(0.1)
        return 10 ** (log_min + (log_max - log_min) * v / 100)

    def _on_w_slider(self, v: int):
        self._w_val_lbl.setText(f"{self._slider_to_epsilon(v):.3f}")

    def _on_t_slider(self, v: int):
        self._t_val_lbl.setText(f"{self._slider_to_epsilon(v):.3f}")

    # ── BROWSE ─────────────────────────────────────────────────────────────────

    def _browse_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FITS file", "",
            "FITS files (*.fit *.fits *.fts);;All files (*)")
        if path:
            self._fits_edit.setText(path)

    def _browse_psf_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PSF FITS", "",
            "FITS files (*.fit *.fits *.fts);;All files (*)")
        if path:
            self._psf_fits_edit.setText(path)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PSF measurements CSV", "",
            "CSV files (*.csv);;All files (*)")
        if path:
            self._csv_edit.setText(path)
            self._load_csv_auto()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._out_edit.setText(path)

    # ── LINEARITY CHECK ────────────────────────────────────────────────────────

    def _check_linearity(self):
        path = self._fits_edit.text().strip()
        if not path or not os.path.isfile(path):
            self._set_status("⚠ No valid FITS file selected", SIRIL_WARNING)
            return
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float32)
            if data.ndim == 3:
                if data.shape[0] == 3:
                    lum = (0.299*data[0] + 0.587*data[1] + 0.114*data[2])
                else:
                    lum = (0.299*data[:,:,0] + 0.587*data[:,:,1] + 0.114*data[:,:,2])
            else:
                lum = data
            h, w = lum.shape[-2], lum.shape[-1]
            is_linear, frac = check_linearity(lum)

            status = "✓ Linear" if is_linear else f"⚠ Possibly stretched ({frac*100:.0f}%)"
            color  = SIRIL_SUCCESS if is_linear else SIRIL_WARNING
            self._lbl_image_info.setText(
                f"{w}×{h} px  |  {status}  |  median={frac*100:.1f}% of range")
            self._lbl_stretch_warn.setVisible(not is_linear)
            self._set_status(f"Loaded {os.path.basename(path)}", color)
            self._loaded_path = path

            # Auto-fill output dir
            if not self._out_edit.text():
                self._out_edit.setText(os.path.dirname(path))

        except Exception as e:
            self._lbl_image_info.setText(f"Error: {e}")
            self._set_status(f"Error loading FITS: {e}", SIRIL_ERROR)

    # ── CSV LOAD ───────────────────────────────────────────────────────────────

    def _load_csv_auto(self):
        path = self._csv_edit.text().strip()
        if not path or not os.path.isfile(path):
            return
        try:
            self._measurements = load_psf_measurements_csv(path)
            self._lbl_meas_count.setText(
                f"Loaded {len(self._measurements)} star measurements")
            self._log(f"Loaded {len(self._measurements)} PSF measurements from CSV")
        except Exception as e:
            self._lbl_meas_count.setText(f"Error: {e}")

    # ── PSF PREVIEW ────────────────────────────────────────────────────────────

    def _preview_psf(self):
        path = self._fits_edit.text().strip()
        if not path or not os.path.isfile(path):
            self._set_status("⚠ Select a FITS file first", SIRIL_WARNING)
            return
        if self._psf_worker and self._psf_worker.isRunning():
            return

        self._btn_preview_psf.setEnabled(False)
        self._btn_preview_psf.setText("Estimating...")
        self._log("─── PSF Preview ───")

        psf_size_map = {"32 px": 32, "64 px": 64, "128 px": 128}
        psf_size = psf_size_map.get(self._psf_size_combo.currentText(), 64)
        cancel_ev = threading.Event()

        self._psf_worker = PSFPreviewWorker(
            path,
            self._n_stars_spin.value(),
            psf_size,
            self._thresh_spin.value(),
            cancel_ev
        )
        self._psf_worker.log_line.connect(self._log)
        self._psf_worker.psf_ready.connect(self._on_psf_preview_ready)
        self._psf_worker.finished.connect(self._on_psf_preview_done)
        self._psf_worker.start()

    def _on_psf_preview_ready(self, psf_arr, fwhm: float):
        self._psf_canvas.show_psf(psf_arr)
        self._lbl_psf_info.setText(
            f"Estimated FWHM ≈ {fwhm:.1f} px  |  size: {psf_arr.shape[0]}×{psf_arr.shape[1]} px")
        self._tabs.setCurrentIndex(0)

    def _on_psf_preview_done(self, ok: bool, err: str):
        self._btn_preview_psf.setEnabled(True)
        self._btn_preview_psf.setText("🔬  Preview estimated PSF")
        if not ok:
            self._set_status(f"PSF preview failed: {err}", SIRIL_ERROR)

    # ── MAIN RUN ───────────────────────────────────────────────────────────────

    def _run_deconv(self):
        path = self._fits_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No input",
                                "Please select a valid FITS file first.")
            return

        if self._worker and self._worker.isRunning():
            return

        # Build config dict
        psf_mode_map = {0: "blind", 1: "gaussian", 2: "fits"}
        algo_map     = {0: "wiener", 1: "richardson_lucy", 2: "tikhonov"}
        psf_size_map = {"32 px": 32, "64 px": 64, "128 px": 128}

        psf_mode = psf_mode_map[self._psf_mode_combo.currentIndex()]
        algorithm = algo_map[self._algo_combo.currentIndex()]

        cfg = {
            "fits_path":       path,
            "psf_mode":        psf_mode,
            "algorithm":       algorithm,
            "n_psf_stars":     self._n_stars_spin.value(),
            "psf_size":        psf_size_map.get(
                                   self._psf_size_combo.currentText(), 64),
            "threshold_sigma": self._thresh_spin.value(),
            "fwhm_px":         self._fwhm_spin.value(),
            "ellipticity":     self._ell_spin.value(),
            "angle_deg":       self._angle_spin.value(),
            "psf_fits_path":   self._psf_fits_edit.text().strip(),
            "epsilon":         self._slider_to_epsilon(self._w_slider.value()),
            "lam":             self._slider_to_epsilon(self._t_slider.value()),
            "iterations":      self._rl_iter_spin.value(),
            "acceleration":    self._rl_accel_spin.value(),
            "convergence_tol": self._rl_conv_spin.value(),
            "spatially_varying": self._sv_chk.isChecked(),
            "measurements":    self._measurements,
            "n_zones":         self._zones_spin.value(),
            "output_dir":      self._out_edit.text().strip() or
                               os.path.dirname(path),
            "load_in_siril":   self._load_siril_chk.isChecked(),
        }

        self._log("─── Starting deconvolution ───")
        self._log(f"  File:      {os.path.basename(path)}")
        self._log(f"  PSF mode:  {psf_mode}")
        self._log(f"  Algorithm: {algorithm}")
        if algorithm == "richardson_lucy":
            self._log(f"  Iterations: {cfg['iterations']}")
        else:
            eps = cfg['epsilon'] if algorithm == "wiener" else cfg['lam']
            self._log(f"  Regularisation: {eps:.4f}")
        if cfg["spatially_varying"]:
            self._log(f"  Spatially varying: {cfg['n_zones']}×{cfg['n_zones']} zones")

        self._cancel_event = threading.Event()
        self._worker = DeconvWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log)
        self._worker.psf_ready.connect(self._on_psf_ready)
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.finished.connect(self._on_finished)

        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._set_status("Running...", SIRIL_ACCENT)
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        if self._psf_worker and self._psf_worker.isRunning():
            self._psf_worker.terminate()
        self._btn_cancel.setEnabled(False)
        self._log("Cancellation requested...")

    # ── WORKER CALLBACKS ───────────────────────────────────────────────────────

    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setRange(0, total)
        self._progress.setValue(step)
        self._set_status(msg, SIRIL_ACCENT)

    def _on_psf_ready(self, psf_arr):
        if psf_arr is not None:
            self._psf_canvas.show_psf(psf_arr)
            fwhm = estimate_psf_fwhm(psf_arr)
            self._lbl_psf_info.setText(
                f"PSF: {psf_arr.shape[0]}×{psf_arr.shape[1]} px  |  "
                f"Est. FWHM ≈ {fwhm:.1f} px")

    def _on_preview_ready(self, original, deconvolved):
        self._result_canvas.show_pair(original, deconvolved)
        self._tabs.setCurrentIndex(1)

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)

        if result.get("success"):
            out = result.get("output_path", "")
            algo = result.get("algorithm", "")
            self._set_status(
                f"✓  Done  |  Algorithm: {algo}  |  Saved: {out}",
                SIRIL_SUCCESS)
            self._log(f"✓ Complete → {out}")
            self._tabs.setCurrentIndex(2)
        else:
            err = result.get("error", "Unknown error")
            self._set_status(f"✗  {err}", SIRIL_ERROR)
            self._log(f"✗ {err}")

    # ── HELPERS ────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts  = datetime.now().strftime("%H:%M:%S")
        self._log_edit.appendPlainText(f"[{ts}] {msg}")
        sb  = self._log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status_label.setText(msg)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 9pt;")


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
