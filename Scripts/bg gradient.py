"""
bg_gradient.py — AI Background Gradient Removal for Siril
PixInsight DBE equivalent: automatic star/nebulosity masking,
adaptive sample placement, polynomial / RBF surface fitting.
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import threading
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, label
from scipy.interpolate import RBFInterpolator
from astropy.io import fits as astropy_fits
from astropy.stats import sigma_clipped_stats
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSplitter, QFrame, QSizePolicy, QScrollArea, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False


# ─────────────────────────────────────────────────────────────────────────────
# Theme
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
QLabel#section {{ color: {SIRIL_SECTION}; font-weight: bold; padding-top: 6px; }}
QLabel#dim    {{ color: {SIRIL_TEXT_DIM}; font-size: 9pt; }}
QLabel#ok     {{ color: {SIRIL_SUCCESS}; font-weight: bold; }}
QLabel#err    {{ color: {SIRIL_ERROR};   font-weight: bold; }}
QLabel#warn   {{ color: {SIRIL_WARNING}; font-weight: bold; }}
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


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: exclusion mask
# ─────────────────────────────────────────────────────────────────────────────
def build_exclusion_mask(image: np.ndarray,
                         star_threshold_sigma: float = 3.0,
                         nebulosity_threshold_sigma: float = 1.5,
                         star_dilation_px: int = 15,
                         nebulosity_smooth_px: int = 50,
                         log_callback=None) -> np.ndarray:
    h, w = image.shape
    img  = image.astype(np.float64)

    _, med, std = sigma_clipped_stats(img, sigma=3.0)
    if std <= 0:
        return np.zeros((h, w), dtype=bool)

    if log_callback:
        log_callback(f"  Background: median={med:.4f}  noise={std:.4f}")

    # Level 1 — Stars
    star_thresh = med + star_threshold_sigma * std
    star_binary = (img > star_thresh)
    struct      = np.ones((star_dilation_px * 2 + 1,
                           star_dilation_px * 2 + 1), dtype=bool)
    star_mask   = binary_dilation(star_binary, structure=struct)

    if log_callback:
        star_pct = 100.0 * star_mask.sum() / (h * w)
        log_callback(f"  Star mask: {star_pct:.1f}% of image excluded")

    # Level 2 — Nebulosity
    smoothed           = gaussian_filter(img, sigma=nebulosity_smooth_px)
    _, smed, sstd      = sigma_clipped_stats(smoothed, sigma=3.0)
    neb_thresh         = smed + nebulosity_threshold_sigma * sstd
    neb_binary         = (smoothed > neb_thresh)
    struct_neb         = np.ones((21, 21), dtype=bool)
    neb_mask           = binary_dilation(neb_binary, structure=struct_neb)

    if log_callback:
        neb_pct = 100.0 * neb_mask.sum() / (h * w)
        log_callback(f"  Nebulosity mask: {neb_pct:.1f}% of image excluded")

    combined = star_mask | neb_mask

    if log_callback:
        total_pct = 100.0 * combined.sum() / (h * w)
        avail_pct = 100.0 - total_pct
        log_callback(
            f"  Combined mask: {total_pct:.1f}% excluded  "
            f"{avail_pct:.1f}% available for background sampling")
        if avail_pct < 10.0:
            log_callback(
                "  WARNING: Less than 10% of image available for "
                "background sampling. Consider loosening thresholds.")
        if avail_pct < 20.0:
            log_callback(
                "  ⚠ Extended nebulosity covers most of field. Consider:\n"
                "    1. Using manual sample placement at image edges\n"
                "    2. Reducing nebulosity threshold")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: sample placement
# ─────────────────────────────────────────────────────────────────────────────
def place_background_samples(image: np.ndarray,
                              exclusion_mask: np.ndarray,
                              grid_size: int = 20,
                              sample_box: int = 32,
                              rejection_sigma: float = 2.0,
                              min_samples: int = 16,
                              log_callback=None) -> tuple:
    h, w = image.shape
    img  = image.astype(np.float64)

    ys = np.linspace(sample_box, h - sample_box, grid_size, dtype=int)
    xs = np.linspace(sample_box, w - sample_box, grid_size, dtype=int)

    sample_x_list    = []
    sample_y_list    = []
    sample_val_list  = []
    n_rejected_mask  = 0

    for cy in ys:
        for cx in xs:
            y1, y2 = cy - sample_box // 2, cy + sample_box // 2
            x1, x2 = cx - sample_box // 2, cx + sample_box // 2
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)

            box_data = img[y1:y2, x1:x2]
            box_mask = exclusion_mask[y1:y2, x1:x2]

            mask_frac = box_mask.sum() / max(box_data.size, 1)
            if mask_frac > 0.20:
                n_rejected_mask += 1
                continue

            clean_pixels = box_data[~box_mask]
            if len(clean_pixels) < 10:
                n_rejected_mask += 1
                continue

            _, med_box, _ = sigma_clipped_stats(clean_pixels, sigma=2.5)
            sample_x_list.append(float(cx))
            sample_y_list.append(float(cy))
            sample_val_list.append(float(med_box))

    if log_callback:
        log_callback(
            f"  Initial samples: {len(sample_x_list)}  "
            f"rejected by mask: {n_rejected_mask}")

    if len(sample_x_list) < min_samples:
        if log_callback:
            log_callback(
                f"  WARNING: Only {len(sample_x_list)} samples — "
                "try lowering grid density or masking thresholds")
        return (np.array(sample_x_list),
                np.array(sample_y_list),
                np.array(sample_val_list))

    vals  = np.array(sample_val_list)
    med_v = float(np.median(vals))
    mad_v = float(np.median(np.abs(vals - med_v))) * 1.4826
    if mad_v > 0:
        keep           = np.abs(vals - med_v) <= rejection_sigma * mad_v
        n_rej_outlier  = int((~keep).sum())
        sample_x_list  = [x for x, k in zip(sample_x_list,  keep) if k]
        sample_y_list  = [y for y, k in zip(sample_y_list,  keep) if k]
        sample_val_list= [v for v, k in zip(sample_val_list, keep) if k]
        if log_callback:
            log_callback(
                f"  Outlier rejection: removed {n_rej_outlier} samples "
                f"({rejection_sigma}σ threshold)")

    if log_callback:
        log_callback(f"  Final samples: {len(sample_x_list)}")

    return (np.array(sample_x_list),
            np.array(sample_y_list),
            np.array(sample_val_list))


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: surface fitting
# ─────────────────────────────────────────────────────────────────────────────
def fit_polynomial_surface(xs, ys, vals, image_shape, degree=3):
    H, W   = image_shape
    x_norm = (xs / W) * 2.0 - 1.0
    y_norm = (ys / H) * 2.0 - 1.0

    def poly_features(x, y, deg):
        cols = []
        for d in range(deg + 1):
            for i in range(d + 1):
                j = d - i
                cols.append((x ** i) * (y ** j))
        return np.column_stack(cols)

    A      = poly_features(x_norm, y_norm, degree)
    coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)

    yy, xx   = np.mgrid[0:H, 0:W]
    xx_n     = (xx / W) * 2.0 - 1.0
    yy_n     = (yy / H) * 2.0 - 1.0
    A_full   = poly_features(xx_n.ravel(), yy_n.ravel(), degree)
    surface  = (A_full @ coeffs).reshape(H, W)
    return surface.astype(np.float32)


def fit_rbf_surface(xs, ys, vals, image_shape,
                    smoothing=0.0, kernel="thin_plate_spline"):
    H, W   = image_shape
    points = np.column_stack([xs / W, ys / H])
    rbf    = RBFInterpolator(points, vals,
                             kernel=kernel,
                             smoothing=smoothing,
                             degree=1)
    yy, xx       = np.mgrid[0:H, 0:W]
    grid_points  = np.column_stack([xx.ravel() / W, yy.ravel() / H])
    surface      = rbf(grid_points).reshape(H, W)
    return surface.astype(np.float32)


def subtract_background(image, surface,
                        mode="subtract", preserve_median=True):
    img = image.astype(np.float64)
    sf  = surface.astype(np.float64)
    if mode == "divide":
        sf_safe = np.where(np.abs(sf) > 1e-10, sf, 1.0)
        result  = img / sf_safe
    else:
        result = img - sf
        if preserve_median:
            result += float(np.median(sf))
    return np.clip(result, 0, None).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: per-channel / RGB
# ─────────────────────────────────────────────────────────────────────────────
def process_single_channel(channel, params, log_callback=None):
    mask = build_exclusion_mask(
        channel,
        star_threshold_sigma       = params.get("star_threshold",  3.0),
        nebulosity_threshold_sigma = params.get("neb_threshold",   1.5),
        star_dilation_px           = params.get("star_dilation",   15),
        nebulosity_smooth_px       = params.get("neb_smooth",      50),
        log_callback               = log_callback)

    xs, ys, vals = place_background_samples(
        channel, mask,
        grid_size       = params.get("grid_size",       20),
        sample_box      = params.get("sample_box",      32),
        rejection_sigma = params.get("rejection_sigma", 2.0),
        log_callback    = log_callback)

    if len(xs) < 4:
        if log_callback:
            log_callback("  ERROR: Too few samples. Skipping channel.")
        return {"corrected": channel.copy(),
                "surface":   np.zeros_like(channel),
                "sample_xs": xs, "sample_ys": ys, "sample_vals": vals}

    model = params.get("model", "polynomial")
    H, W  = channel.shape

    if model == "polynomial":
        surface = fit_polynomial_surface(
            xs, ys, vals, (H, W),
            degree=params.get("poly_degree", 3))
    else:
        surface = fit_rbf_surface(
            xs, ys, vals, (H, W),
            smoothing=params.get("rbf_smoothing", 0.0),
            kernel=params.get("rbf_kernel", "thin_plate_spline"))

    corrected = subtract_background(
        channel, surface,
        mode            = params.get("subtraction_mode", "subtract"),
        preserve_median = params.get("preserve_median", True))

    return {"corrected": corrected, "surface": surface,
            "sample_xs": xs, "sample_ys": ys, "sample_vals": vals}


def process_rgb_image(data, params, log_callback=None, cancel_event=None):
    if data.shape[0] == 3:
        channels = [data[c] for c in range(3)]
        layout   = "CHW"
    else:
        channels = [data[:, :, c] for c in range(3)]
        layout   = "HWC"

    corrected_channels = []
    surface_channels   = []
    all_sample_info    = {}

    for ci, (ch, name) in enumerate(zip(channels, ["R", "G", "B"])):
        if cancel_event and cancel_event.is_set():
            return None, None, None
        if log_callback:
            log_callback(f"Processing {name} channel...")
        ch_result = process_single_channel(ch, params,
                                           log_callback=log_callback)
        corrected_channels.append(ch_result["corrected"])
        surface_channels.append(ch_result["surface"])
        all_sample_info[name] = {
            "xs":   ch_result["sample_xs"],
            "ys":   ch_result["sample_ys"],
            "vals": ch_result["sample_vals"],
        }

    if layout == "CHW":
        corrected = np.stack(corrected_channels, axis=0)
        surfaces  = np.stack(surface_channels,   axis=0)
    else:
        corrected = np.stack(corrected_channels, axis=2)
        surfaces  = np.stack(surface_channels,   axis=2)

    return corrected, surfaces, all_sample_info


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────
class BGWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    samples_ready = pyqtSignal(object, object, object)
    surface_ready = pyqtSignal(object)
    result_ready  = pyqtSignal(object, object)
    finished      = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg = self.config

            # Step 1: Load
            self.progress.emit(1, 6, "Loading image...")
            with astropy_fits.open(cfg["fits_path"]) as hdul:
                data   = hdul[0].data.astype(np.float32)
                header = hdul[0].header.copy()
            self.log_line.emit(
                f"Loaded: {os.path.basename(cfg['fits_path'])}  "
                f"shape={data.shape}")

            is_rgb = (data.ndim == 3)

            if self._cancel.is_set():
                self._abort(); return

            params = {
                "star_threshold":   cfg.get("star_threshold",   3.0),
                "neb_threshold":    cfg.get("neb_threshold",    1.5),
                "star_dilation":    cfg.get("star_dilation",    15),
                "neb_smooth":       cfg.get("neb_smooth",       50),
                "grid_size":        cfg.get("grid_size",        20),
                "sample_box":       cfg.get("sample_box",       32),
                "rejection_sigma":  cfg.get("rejection_sigma",  2.0),
                "model":            cfg.get("model",            "polynomial"),
                "poly_degree":      cfg.get("poly_degree",      3),
                "rbf_smoothing":    cfg.get("rbf_smoothing",    0.0),
                "rbf_kernel":       cfg.get("rbf_kernel",       "thin_plate_spline"),
                "subtraction_mode": cfg.get("subtraction_mode", "subtract"),
                "preserve_median":  cfg.get("preserve_median",  True),
            }

            manual_xs  = cfg.get("manual_xs")
            manual_ys  = cfg.get("manual_ys")
            use_manual = (manual_xs is not None and len(manual_xs) >= 4)

            if use_manual:
                self.log_line.emit(
                    f"Using {len(manual_xs)} manually placed samples")

            # Step 2: Build mask
            self.progress.emit(2, 6, "Building star/nebulosity mask...")

            if is_rgb:
                if data.shape[0] == 3:
                    lum = (0.299 * data[0] +
                           0.587 * data[1] +
                           0.114 * data[2])
                else:
                    lum = (0.299 * data[:, :, 0] +
                           0.587 * data[:, :, 1] +
                           0.114 * data[:, :, 2])
            else:
                lum = data

            mask = build_exclusion_mask(
                lum,
                star_threshold_sigma       = params["star_threshold"],
                nebulosity_threshold_sigma = params["neb_threshold"],
                star_dilation_px           = params["star_dilation"],
                nebulosity_smooth_px       = params["neb_smooth"],
                log_callback               = self.log_line.emit)

            if self._cancel.is_set():
                self._abort(); return

            # Step 3: Place samples
            self.progress.emit(3, 6, "Placing background samples...")

            if use_manual:
                xs   = np.array(manual_xs)
                ys   = np.array(manual_ys)
                vals = np.array([
                    float(np.median(
                        lum[max(0, int(y) - 8):int(y) + 8,
                            max(0, int(x) - 8):int(x) + 8]))
                    for x, y in zip(xs, ys)])
            else:
                xs, ys, vals = place_background_samples(
                    lum, mask,
                    grid_size       = params["grid_size"],
                    sample_box      = params["sample_box"],
                    rejection_sigma = params["rejection_sigma"],
                    log_callback    = self.log_line.emit)

            self.samples_ready.emit(xs, ys, vals)

            if self._cancel.is_set():
                self._abort(); return

            # Step 4 + 5: Fit & subtract
            self.progress.emit(4, 6,
                               f"Fitting {params['model']} surface...")
            self.log_line.emit(
                f"Fitting {params['model']} background model "
                f"to {len(xs)} samples...")

            self.progress.emit(5, 6, "Subtracting background...")

            if is_rgb:
                corrected, surfaces, _ = process_rgb_image(
                    data, params,
                    log_callback = self.log_line.emit,
                    cancel_event = self._cancel)
                if corrected is None:
                    self._abort(); return
                if surfaces is not None and surfaces.ndim == 3:
                    surface_preview = (surfaces[0]
                                       if surfaces.shape[0] == 3
                                       else surfaces[:, :, 0])
                else:
                    surface_preview = np.zeros_like(lum)
            else:
                result          = process_single_channel(
                    data, params, log_callback=self.log_line.emit)
                corrected       = result["corrected"]
                surface_preview = result["surface"]
                surfaces        = result["surface"]

            self.surface_ready.emit(surface_preview)
            self.result_ready.emit(corrected, surfaces)

            if self._cancel.is_set():
                self._abort(); return

            # Step 6: Save
            self.progress.emit(6, 6, "Saving...")
            out_dir  = cfg.get("output_dir",
                                os.path.dirname(cfg["fits_path"]))
            os.makedirs(out_dir, exist_ok=True)
            base     = os.path.splitext(
                os.path.basename(cfg["fits_path"]))[0]
            out_path = os.path.join(out_dir, f"{base}_bgcorr.fit")

            hdu = astropy_fits.PrimaryHDU(
                data=corrected.astype(np.float32), header=header)
            hdu.header["HISTORY"] = \
                "Background gradient removal — bg_gradient.py"
            hdu.header["BGMODEL"] = params["model"]
            hdu.writeto(out_path, overwrite=True)
            self.log_line.emit(f"Saved: {out_path}")

            if cfg.get("save_model"):
                model_path = os.path.join(out_dir,
                                          f"{base}_bgmodel.fit")
                hdu_m = astropy_fits.PrimaryHDU(
                    data=surfaces.astype(np.float32))
                hdu_m.header["HISTORY"] = "Background model"
                hdu_m.writeto(model_path, overwrite=True)
                self.log_line.emit(f"Model saved: {model_path}")

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
                "n_samples":   len(xs),
                "model":       params["model"],
                "poly_degree": params.get("poly_degree"),
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


# Preview-only worker (mask + samples, no fitting)
class PreviewWorker(QThread):
    log_line      = pyqtSignal(str)
    mask_ready    = pyqtSignal(object, object)   # mask, lum
    samples_ready = pyqtSignal(object, object, object)
    finished      = pyqtSignal()

    def __init__(self, fits_path, params, cancel_event):
        super().__init__()
        self.fits_path    = fits_path
        self.params       = params
        self._cancel      = cancel_event

    def run(self):
        try:
            with astropy_fits.open(self.fits_path) as hdul:
                data = hdul[0].data.astype(np.float32)

            is_rgb = (data.ndim == 3)
            if is_rgb:
                lum = (0.299 * (data[0] if data.shape[0] == 3
                                else data[:, :, 0]) +
                       0.587 * (data[1] if data.shape[0] == 3
                                else data[:, :, 1]) +
                       0.114 * (data[2] if data.shape[0] == 3
                                else data[:, :, 2]))
            else:
                lum = data

            p = self.params
            mask = build_exclusion_mask(
                lum,
                star_threshold_sigma       = p.get("star_threshold",  3.0),
                nebulosity_threshold_sigma = p.get("neb_threshold",   1.5),
                star_dilation_px           = p.get("star_dilation",   15),
                nebulosity_smooth_px       = p.get("neb_smooth",      50),
                log_callback               = self.log_line.emit)

            self.mask_ready.emit(mask, lum)

            if self._cancel.is_set():
                self.finished.emit(); return

            xs, ys, vals = place_background_samples(
                lum, mask,
                grid_size       = p.get("grid_size",       20),
                sample_box      = p.get("sample_box",      32),
                rejection_sigma = p.get("rejection_sigma", 2.0),
                log_callback    = self.log_line.emit)

            self.samples_ready.emit(xs, ys, vals)

        except Exception as e:
            import traceback
            self.log_line.emit(f"Preview ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
        finally:
            self.finished.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib canvases
# ─────────────────────────────────────────────────────────────────────────────
class SampleCanvas(FigureCanvasQTAgg):
    samples_changed = pyqtSignal(list, list)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), facecolor="#1e2128")
        self.ax  = self.fig.add_subplot(111)
        self._style_ax()
        super().__init__(self.fig)
        self.setParent(parent)
        self.image_data     = None
        self.exclusion_mask = None
        self.sample_xs      = []
        self.sample_ys      = []
        self.manual_mode    = False
        self.mpl_connect("button_press_event", self._on_click)

    def _style_ax(self):
        self.ax.set_facecolor("#1e2128")
        self.ax.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for sp in self.ax.spines.values():
            sp.set_color("#3a4055")

    def show_image(self, image: np.ndarray, mask: np.ndarray = None):
        self.image_data     = image
        self.exclusion_mask = mask
        self._redraw()

    def show_samples(self, xs, ys, vals=None):
        self.sample_xs = list(xs)
        self.sample_ys = list(ys)
        self._redraw()

    def _redraw(self):
        if self.image_data is None:
            return
        self.ax.clear()
        self._style_ax()

        img  = self.image_data
        h, w = img.shape
        fac  = max(1, max(h, w) // 600)
        p_lo = float(np.percentile(img, 0.5))
        p_hi = float(np.percentile(img, 99.5))
        disp = np.clip(
            (img.astype(float) - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)
        self.ax.imshow(disp[::fac, ::fac], cmap="gray",
                       origin="lower", aspect="equal",
                       interpolation="nearest")

        if self.exclusion_mask is not None:
            mds  = self.exclusion_mask[::fac, ::fac].astype(float)
            rgba = np.zeros((*mds.shape, 4))
            rgba[:, :, 2] = mds * 0.4
            rgba[:, :, 3] = mds * 0.3
            self.ax.imshow(rgba, origin="lower",
                           aspect="equal", interpolation="nearest")

        if self.sample_xs:
            xs_sc = [x / fac for x in self.sample_xs]
            ys_sc = [y / fac for y in self.sample_ys]
            self.ax.plot(xs_sc, ys_sc, "o",
                         color="#4caf7d", markersize=5,
                         markerfacecolor="none",
                         markeredgewidth=1.5, zorder=5)

        n        = len(self.sample_xs)
        mode_str = ("Manual — click to add / right-click to remove"
                    if self.manual_mode else "Auto")
        self.ax.set_title(f"Background samples: {n}  |  {mode_str}",
                          color="#5ba3ff", fontsize=8)
        self.fig.tight_layout(pad=0.2)
        self.draw()

    def _on_click(self, event):
        if not self.manual_mode:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.image_data is None:
            return
        h, w = self.image_data.shape
        fac  = max(1, max(h, w) // 600)
        x    = event.xdata * fac
        y    = event.ydata * fac

        if event.button == 1:
            self.sample_xs.append(x)
            self.sample_ys.append(y)
        elif event.button == 3:
            if self.sample_xs:
                dists = [((xi - x) ** 2 + (yi - y) ** 2) ** 0.5
                         for xi, yi in zip(self.sample_xs,
                                           self.sample_ys)]
                idx   = int(np.argmin(dists))
                if dists[idx] < 30 * fac:
                    self.sample_xs.pop(idx)
                    self.sample_ys.pop(idx)

        self._redraw()
        self.samples_changed.emit(self.sample_xs, self.sample_ys)


class ResultCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig     = Figure(figsize=(12, 4), facecolor="#1e2128")
        self.ax_orig = self.fig.add_subplot(1, 3, 1)
        self.ax_surf = self.fig.add_subplot(1, 3, 2)
        self.ax_corr = self.fig.add_subplot(1, 3, 3)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style(self):
        for ax, title, color in [
            (self.ax_orig, "Original",         "#e8a23a"),
            (self.ax_surf, "Background model", "#7a8499"),
            (self.ax_corr, "Corrected",        "#4caf7d"),
        ]:
            ax.set_facecolor("#1e2128")
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color("#3a4055")

    def _disp(self, arr, vmin=None, vmax=None):
        if vmin is None: vmin = float(np.percentile(arr, 0.5))
        if vmax is None: vmax = float(np.percentile(arr, 99.5))
        return np.clip(
            (arr.astype(float) - vmin) / (vmax - vmin + 1e-10), 0, 1)

    def _ds(self, arr, max_px=500):
        h, w   = arr.shape
        factor = max(1, max(h, w) // max_px)
        return arr[::factor, ::factor]

    def show_result(self, original, surface, corrected):
        p_lo = float(np.percentile(original, 0.5))
        p_hi = float(np.percentile(original, 99.5))

        for ax, data, title, color, shared in [
            (self.ax_orig, original,  "Original",         "#e8a23a", True),
            (self.ax_surf, surface,   "Background model", "#7a8499", False),
            (self.ax_corr, corrected, "Corrected",        "#4caf7d", True),
        ]:
            ax.clear()
            ax.set_facecolor("#1e2128")
            disp = (self._disp(self._ds(data), p_lo, p_hi)
                    if shared else self._disp(self._ds(data)))
            ax.imshow(disp, cmap="gray", origin="lower",
                      aspect="equal", interpolation="nearest")
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color("#3a4055")

        self.fig.tight_layout(pad=0.3)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Gradient Removal — Siril")
        self.resize(1300, 820)

        # State
        self._worker          = None
        self._preview_worker  = None
        self._cancel_event    = threading.Event()
        self._loaded_data     = None   # full numpy array of loaded image
        self._lum_data        = None   # luminance for preview
        self._result_original = None
        self._result_surface  = None
        self._result_corrected= None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Title bar
        root_layout.addWidget(self._build_titlebar())

        # Body
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(6)
        root_layout.addWidget(body, 1)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        body_layout.addWidget(splitter, 1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([340, 900])

        # Bottom bar
        body_layout.addWidget(self._build_bottom_bar())

        # Status bar
        self._status = QLabel("Ready — load a FITS image to begin")
        self._status.setObjectName("dim")
        self._status.setContentsMargins(4, 2, 4, 2)
        body_layout.addWidget(self._status)

    # ─── Title bar ──────────────────────────────────────────────────────────
    def _build_titlebar(self):
        bar = QWidget()
        bar.setFixedHeight(38)
        bar.setStyleSheet(
            f"background:{SIRIL_BG3}; border-bottom:1px solid {SIRIL_BORDER};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)

        lbl = QLabel("🌌  Background Gradient Removal  —  Siril")
        lbl.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-weight:bold; font-size:11pt;")
        layout.addWidget(lbl)
        layout.addStretch()

        ver = QLabel("v1.0  |  PixInsight DBE equivalent")
        ver.setObjectName("dim")
        layout.addWidget(ver)
        return bar

    # ─── Left panel ─────────────────────────────────────────────────────────
    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)

        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_sample_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_subtraction_group())
        layout.addWidget(self._build_output_group())
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _build_input_group(self):
        grp    = QGroupBox("Input")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        row = QHBoxLayout()
        self._fits_path = QLineEdit()
        self._fits_path.setPlaceholderText("Path to FITS file…")
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(64)
        btn_browse.clicked.connect(self._browse_fits)
        row.addWidget(self._fits_path)
        row.addWidget(btn_browse)
        layout.addLayout(row)

        btn_load = QPushButton("⟳  Load && analyse")
        btn_load.setObjectName("primary")
        btn_load.clicked.connect(self._load_image)
        layout.addWidget(btn_load)

        self._img_info = QLabel("No image loaded")
        self._img_info.setObjectName("dim")
        self._img_info.setWordWrap(True)
        layout.addWidget(self._img_info)

        return grp

    def _build_sample_group(self):
        grp    = QGroupBox("Sample placement")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "Automatic (recommended)",
            "Manual (click on image)"
        ])
        self._mode_combo.currentIndexChanged.connect(
            self._on_mode_changed)
        layout.addWidget(self._mode_combo)

        # Stack: auto / manual
        self._sample_stack = QStackedWidget()
        layout.addWidget(self._sample_stack)
        self._sample_stack.addWidget(self._build_auto_sample_widget())
        self._sample_stack.addWidget(self._build_manual_sample_widget())

        return grp

    def _build_auto_sample_widget(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._star_thresh = QDoubleSpinBox()
        self._star_thresh.setRange(1.0, 8.0); self._star_thresh.setValue(3.0)
        self._star_thresh.setSingleStep(0.5); self._star_thresh.setDecimals(1)
        form.addRow("Star threshold (σ):", self._star_thresh)
        form.addRow("", self._dim("Higher = less masking"))

        self._neb_thresh = QDoubleSpinBox()
        self._neb_thresh.setRange(0.5, 5.0); self._neb_thresh.setValue(1.5)
        self._neb_thresh.setSingleStep(0.25); self._neb_thresh.setDecimals(2)
        form.addRow("Nebulosity threshold (σ):", self._neb_thresh)
        form.addRow("", self._dim("Lower = more aggressive nebulosity masking"))

        self._star_dil = QSpinBox()
        self._star_dil.setRange(5, 50); self._star_dil.setValue(15)
        form.addRow("Star dilation (px):", self._star_dil)

        self._neb_smooth = QSpinBox()
        self._neb_smooth.setRange(20, 200); self._neb_smooth.setValue(50)
        form.addRow("Nebulosity smooth (px):", self._neb_smooth)

        self._grid_size = QSpinBox()
        self._grid_size.setRange(8, 40); self._grid_size.setValue(20)
        form.addRow("Grid density:", self._grid_size)
        form.addRow("", self._dim("N×N grid of candidate sample positions"))

        self._rej_sigma = QDoubleSpinBox()
        self._rej_sigma.setRange(1.0, 4.0); self._rej_sigma.setValue(2.0)
        self._rej_sigma.setSingleStep(0.1); self._rej_sigma.setDecimals(1)
        form.addRow("Rejection sigma:", self._rej_sigma)

        layout.addLayout(form)

        btn_preview = QPushButton("🔍  Place samples (preview)")
        btn_preview.clicked.connect(self._run_preview)
        layout.addWidget(btn_preview)

        return w

    def _build_manual_sample_widget(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._dim("Left-click: add sample point"))
        layout.addWidget(self._dim("Right-click: remove nearest sample"))

        self._manual_count_lbl = QLabel("0 samples placed")
        self._manual_count_lbl.setObjectName("dim")
        layout.addWidget(self._manual_count_lbl)

        btn_clear = QPushButton("✕  Clear all samples")
        btn_clear.setObjectName("danger")
        btn_clear.clicked.connect(self._clear_manual_samples)
        layout.addWidget(btn_clear)

        self._manual_warn = QLabel(
            "⚠  Recommend at least 16 samples for stable fit")
        self._manual_warn.setObjectName("warn")
        self._manual_warn.setWordWrap(True)
        self._manual_warn.setVisible(True)
        layout.addWidget(self._manual_warn)

        return w

    def _build_model_group(self):
        grp    = QGroupBox("Surface model")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "Polynomial (fast, recommended)",
            "RBF — Thin Plate Spline (flexible)",
            "RBF — Multiquadric",
            "RBF — Gaussian (smoothest)"
        ])
        self._model_combo.currentIndexChanged.connect(
            self._on_model_changed)
        layout.addWidget(self._model_combo)

        # Stack: poly / rbf
        self._model_stack = QStackedWidget()
        layout.addWidget(self._model_stack)

        # Polynomial widget
        pw      = QWidget()
        pl      = QVBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        pform   = QFormLayout()
        pform.setSpacing(4)
        self._poly_degree = QComboBox()
        self._poly_degree.addItems([
            "1 (plane)", "2", "3 (recommended)", "4", "5"])
        self._poly_degree.setCurrentIndex(2)
        self._poly_degree.currentIndexChanged.connect(
            self._on_degree_changed)
        pform.addRow("Degree:", self._poly_degree)
        pl.addLayout(pform)
        self._degree_hint = QLabel(
            "3 = Cubic (recommended): handles most real-world gradients")
        self._degree_hint.setObjectName("dim")
        self._degree_hint.setWordWrap(True)
        pl.addWidget(self._degree_hint)
        self._model_stack.addWidget(pw)

        # RBF widget
        rw    = QWidget()
        rl    = QVBoxLayout(rw)
        rl.setContentsMargins(0, 0, 0, 0)
        rform = QFormLayout()
        rform.setSpacing(4)
        self._rbf_smooth = QDoubleSpinBox()
        self._rbf_smooth.setRange(0.0, 5.0); self._rbf_smooth.setValue(0.0)
        self._rbf_smooth.setSingleStep(0.1); self._rbf_smooth.setDecimals(2)
        rform.addRow("Smoothing:", self._rbf_smooth)
        rl.addLayout(rform)
        rl.addWidget(self._dim("0 = exact interpolation"))
        self._model_stack.addWidget(rw)
        # add two more placeholder pages so indices 2,3 also show rbf widget
        for _ in range(2):
            self._model_stack.addWidget(QWidget())
        self._model_stack.setCurrentIndex(0)

        return grp

    def _build_subtraction_group(self):
        grp    = QGroupBox("Subtraction")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        form  = QFormLayout()
        form.setSpacing(4)
        self._sub_mode = QComboBox()
        self._sub_mode.addItems([
            "Subtract (standard)",
            "Divide (multiplicative)"
        ])
        form.addRow("Mode:", self._sub_mode)
        layout.addLayout(form)

        self._preserve_median = QCheckBox("Preserve image median level")
        self._preserve_median.setChecked(True)
        self._preserve_median.setToolTip("Prevents result going negative")
        layout.addWidget(self._preserve_median)
        layout.addWidget(self._dim("Prevents result going negative"))

        return grp

    def _build_output_group(self):
        grp    = QGroupBox("Output")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        row = QHBoxLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("Same as input…")
        btn_browse_out = QPushButton("Browse")
        btn_browse_out.setFixedWidth(64)
        btn_browse_out.clicked.connect(self._browse_output)
        row.addWidget(self._out_dir)
        row.addWidget(btn_browse_out)
        layout.addLayout(row)

        self._save_model  = QCheckBox("Save background model FITS")
        self._save_model.setChecked(False)
        layout.addWidget(self._save_model)

        self._load_siril  = QCheckBox("Load result in Siril")
        self._load_siril.setChecked(True)
        layout.addWidget(self._load_siril)

        return grp

    # ─── Right panel ────────────────────────────────────────────────────────
    def _build_right_panel(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Tab 1 — Samples
        sample_tab = QWidget()
        st_layout  = QVBoxLayout(sample_tab)
        st_layout.setContentsMargins(4, 4, 4, 4)
        self._sample_canvas = SampleCanvas()
        self._sample_canvas.samples_changed.connect(
            self._on_manual_samples_changed)
        st_layout.addWidget(self._sample_canvas, 1)
        self._sample_summary = QLabel("")
        self._sample_summary.setObjectName("dim")
        st_layout.addWidget(self._sample_summary)
        self._tabs.addTab(sample_tab, "🌌  Samples")

        # Tab 2 — Result
        result_tab = QWidget()
        rt_layout  = QVBoxLayout(result_tab)
        rt_layout.setContentsMargins(4, 4, 4, 4)
        self._result_canvas = ResultCanvas()
        rt_layout.addWidget(self._result_canvas, 1)
        self._tabs.addTab(result_tab, "✨  Result")

        # Tab 3 — Log
        log_tab   = QWidget()
        lt_layout = QVBoxLayout(log_tab)
        lt_layout.setContentsMargins(4, 4, 4, 4)
        self._log_pane = QPlainTextEdit()
        self._log_pane.setReadOnly(True)
        lt_layout.addWidget(self._log_pane)
        self._tabs.addTab(log_tab, "📋  Log")

        return w

    # ─── Bottom bar ─────────────────────────────────────────────────────────
    def _build_bottom_bar(self):
        bar    = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        self._progress = QProgressBar()
        self._progress.setRange(0, 6)
        self._progress.setValue(0)
        self._progress.setFixedHeight(8)
        self._progress.setVisible(False)
        layout.addWidget(self._progress, 1)

        self._btn_run = QPushButton("▶  Remove gradient")
        self._btn_run.setObjectName("primary")
        self._btn_run.setFixedHeight(36)
        self._btn_run.setMinimumWidth(160)
        self._btn_run.clicked.connect(self._run_removal)
        layout.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setFixedHeight(36)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_worker)
        layout.addWidget(self._btn_cancel)

        return bar

    # ─── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _dim(text):
        lbl = QLabel(text)
        lbl.setObjectName("dim")
        lbl.setWordWrap(True)
        return lbl

    def _log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_pane.appendPlainText(f"[{ts}] {text}")

    def _set_busy(self, busy: bool):
        self._btn_run.setEnabled(not busy)
        self._btn_cancel.setEnabled(busy)
        self._progress.setVisible(busy)
        if not busy:
            self._progress.setValue(0)

    # ─── Event handlers ─────────────────────────────────────────────────────
    def _browse_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FITS image", "",
            "FITS files (*.fit *.fits *.fts);;All files (*.*)")
        if path:
            self._fits_path.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select output folder", "")
        if path:
            self._out_dir.setText(path)

    def _load_image(self):
        path = self._fits_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file",
                                "Please select a valid FITS file.")
            return
        try:
            with astropy_fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float32)
            self._loaded_data = data
            is_rgb = (data.ndim == 3)
            if is_rgb:
                shape_str = (f"{data.shape[1]}×{data.shape[2]}  "
                             f"RGB ({data.shape[0]} channels)")
                if data.shape[0] == 3:
                    lum = (0.299 * data[0] +
                           0.587 * data[1] +
                           0.114 * data[2])
                else:
                    lum = (0.299 * data[:, :, 0] +
                           0.587 * data[:, :, 1] +
                           0.114 * data[:, :, 2])
            else:
                shape_str = f"{data.shape[1]}×{data.shape[0]}  Mono"
                lum = data
            self._lum_data = lum
            self._img_info.setText(shape_str)
            self._sample_canvas.show_image(lum)
            self._log(f"Loaded {os.path.basename(path)}  {shape_str}")
            self._status.setText(f"Image loaded: {shape_str}")
            # Switch to samples tab
            self._tabs.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _on_mode_changed(self, idx):
        self._sample_stack.setCurrentIndex(idx)
        is_manual = (idx == 1)
        self._sample_canvas.manual_mode = is_manual
        if self._sample_canvas.image_data is not None:
            self._sample_canvas._redraw()

    def _on_model_changed(self, idx):
        self._model_stack.setCurrentIndex(0 if idx == 0 else 1)

    _DEGREE_HINTS = {
        0: "1 = Plane: corrects simple tilt gradient (LP from one direction)",
        1: "2 = Quadratic: corrects vignetting residuals and curved gradients",
        2: "3 = Cubic (recommended): handles most real-world gradient patterns",
        3: "4 = Quartic: complex gradients, needs many samples (>50 recommended)",
        4: "5 = Quintic: risk of oscillation near edges, use with caution",
    }

    def _on_degree_changed(self, idx):
        self._degree_hint.setText(self._DEGREE_HINTS.get(idx, ""))

    def _on_manual_samples_changed(self, xs, ys):
        n = len(xs)
        self._manual_count_lbl.setText(f"{n} samples placed")
        self._manual_warn.setVisible(n < 16)

    def _clear_manual_samples(self):
        self._sample_canvas.sample_xs = []
        self._sample_canvas.sample_ys = []
        self._sample_canvas._redraw()
        self._manual_count_lbl.setText("0 samples placed")
        self._manual_warn.setVisible(True)

    # ─── Preview (samples only) ──────────────────────────────────────────────
    def _run_preview(self):
        path = self._fits_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file",
                                "Please select a valid FITS file first.")
            return
        if self._preview_worker and self._preview_worker.isRunning():
            return

        self._log("Running sample placement preview…")
        self._status.setText("Running preview…")
        params = self._collect_params()
        ce     = threading.Event()

        self._preview_worker = PreviewWorker(path, params, ce)
        self._preview_worker.log_line.connect(self._log)
        self._preview_worker.mask_ready.connect(self._on_preview_mask)
        self._preview_worker.samples_ready.connect(
            self._on_preview_samples)
        self._preview_worker.finished.connect(self._on_preview_done)
        self._preview_worker.start()

    def _on_preview_mask(self, mask, lum):
        self._sample_canvas.show_image(lum, mask)
        self._tabs.setCurrentIndex(0)

    def _on_preview_samples(self, xs, ys, vals):
        self._sample_canvas.show_samples(xs, ys, vals)
        n = len(xs)
        self._sample_summary.setText(
            f"Auto samples: {n} accepted")

    def _on_preview_done(self):
        self._status.setText("Preview complete")

    # ─── Main run ────────────────────────────────────────────────────────────
    def _collect_params(self):
        return {
            "star_threshold":   self._star_thresh.value(),
            "neb_threshold":    self._neb_thresh.value(),
            "star_dilation":    self._star_dil.value(),
            "neb_smooth":       self._neb_smooth.value(),
            "grid_size":        self._grid_size.value(),
            "rejection_sigma":  self._rej_sigma.value(),
            "model":            ("polynomial"
                                 if self._model_combo.currentIndex() == 0
                                 else "rbf"),
            "poly_degree":      self._poly_degree.currentIndex() + 1,
            "rbf_smoothing":    self._rbf_smooth.value(),
            "rbf_kernel":       ["thin_plate_spline", "thin_plate_spline",
                                 "multiquadric",
                                 "gaussian"][self._model_combo.currentIndex()],
            "subtraction_mode": ("subtract"
                                 if self._sub_mode.currentIndex() == 0
                                 else "divide"),
            "preserve_median":  self._preserve_median.isChecked(),
        }

    def _run_removal(self):
        path = self._fits_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file",
                                "Please select a valid FITS file.")
            return
        if self._worker and self._worker.isRunning():
            return

        params = self._collect_params()

        # Validate sample count vs polynomial degree
        if params["model"] == "polynomial":
            d       = params["poly_degree"]
            n_req   = (d + 1) * (d + 2) // 2
            if self._mode_combo.currentIndex() == 1:
                n_s = len(self._sample_canvas.sample_xs)
                if n_s < n_req * 2:
                    QMessageBox.warning(
                        self, "Too few samples",
                        f"Degree-{d} polynomial needs at least "
                        f"{n_req * 2} samples; you have {n_s}.\n"
                        f"Reduce polynomial degree or add more samples.")
                    return

        cfg = {
            "fits_path":   path,
            "output_dir":  self._out_dir.text().strip() or None,
            "save_model":  self._save_model.isChecked(),
            "load_in_siril": self._load_siril.isChecked(),
            **params,
        }

        if self._mode_combo.currentIndex() == 1:
            cfg["manual_xs"] = list(self._sample_canvas.sample_xs)
            cfg["manual_ys"] = list(self._sample_canvas.sample_ys)

        self._cancel_event = threading.Event()
        self._result_original = None

        # Stash original for result display
        if self._lum_data is not None:
            self._result_original = self._lum_data.copy()

        self._worker = BGWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log)
        self._worker.samples_ready.connect(self._on_worker_samples)
        self._worker.surface_ready.connect(self._on_surface_ready)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

        self._set_busy(True)
        self._tabs.setCurrentIndex(2)   # jump to Log
        self._log(f"Starting background gradient removal: {path}")

    def _cancel_worker(self):
        if self._worker:
            self._worker.cancel()
        if self._preview_worker:
            self._preview_worker._cancel.set()
        self._log("Cancellation requested…")

    # ─── Worker callbacks ────────────────────────────────────────────────────
    def _on_progress(self, step, total, msg):
        self._progress.setMaximum(total)
        self._progress.setValue(step)
        self._status.setText(f"Step {step}/{total}: {msg}")

    def _on_worker_samples(self, xs, ys, vals):
        self._sample_canvas.show_samples(xs, ys, vals)
        self._sample_summary.setText(
            f"Samples used: {len(xs)}")

    def _on_surface_ready(self, surface):
        self._result_surface = surface

    def _on_result_ready(self, corrected, surfaces):
        self._result_corrected = corrected
        # For display: use luminance of corrected if RGB
        if corrected.ndim == 3:
            if corrected.shape[0] == 3:
                corr_lum = (0.299 * corrected[0] +
                            0.587 * corrected[1] +
                            0.114 * corrected[2])
            else:
                corr_lum = (0.299 * corrected[:, :, 0] +
                            0.587 * corrected[:, :, 1] +
                            0.114 * corrected[:, :, 2])
        else:
            corr_lum = corrected

        if (self._result_original is not None and
                self._result_surface is not None):
            self._result_canvas.show_result(
                self._result_original,
                self._result_surface,
                corr_lum)
            self._tabs.setCurrentIndex(1)

    def _on_worker_finished(self, info):
        self._set_busy(False)
        if info.get("success"):
            n       = info.get("n_samples", "?")
            model   = info.get("model", "?")
            degree  = info.get("poly_degree")
            out     = info.get("output_path", "")
            deg_str = f" d={degree}" if (model == "polynomial"
                                          and degree) else ""
            msg = (f"✓  Done  |  {n} samples  |  "
                   f"Model: {model}{deg_str}  |  Saved: {out}")
            self._status.setText(msg)
            self._log(f"Finished. {msg}")
        else:
            err = info.get("error", "Unknown error")
            self._status.setText(f"✗  {err}")
            self._log(f"Failed: {err}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
