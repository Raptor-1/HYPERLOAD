r"""
software_adc.py  —  HYPERLOAD Software Atmospheric Dispersion Corrector
========================================================================
Corrects atmospheric chromatic dispersion in RGB planetary and deep-sky images.

Atmospheric dispersion causes shorter wavelengths to refract more than longer
ones, separating the RGB channels vertically along the altitude axis. At 30°
altitude this can be several arcseconds — larger than the Airy disk of most
amateur telescopes.

Two correction modes:

  1. THEORETICAL (model-based)
     Uses the Peck-Reeder (1972) refractive index formula for air, which is
     accurate to 2×10⁻⁹ from 230nm to 1690nm (better than Edlén 1966).
     Inputs: altitude, temperature, pressure, pixel scale, RGB wavelengths.
     Formula: ΔR(λ) = 206265 × [n(λ) - n(λ_ref)] × tan(Z)  arcsec
     where Z = zenith angle = 90° - altitude.
     Each channel is shifted by ΔR / pixel_scale pixels along the altitude axis.

  2. EMPIRICAL (star-measurement-based)
     Detects stars in each RGB channel independently, matches them, and
     measures the actual chromatic centroid offset between channels.
     No atmospheric model needed — measures whatever dispersion is actually
     present in the image. Robust against non-standard site conditions.

Correction applies a sub-pixel shift to each channel using sinc interpolation
(scipy.ndimage.shift, order=3), preserving image resolution.

Scope:
  Software ADC corrects INTER-BAND chromatic shift (R vs G vs B centroid
  displacement). It CANNOT recover INTRA-BAND smearing (the blur within a
  single passband), which requires a hardware ADC. Software correction is most
  effective above ~35° altitude where intra-band smearing is still manageable.

Supports:
  - RGB FITS (3-channel, [3, H, W] or [H, W, 3])
  - Separate R, G, B FITS files
  - SER sequences (processes frame by frame)
  - Batch processing of entire folders

References:
  Peck & Reeder 1972, JOSA 62, 958      — refractive index formula
  Edlén 1966, Metrologia 2, 71           — original formula (reference)
  Smart 1931, Spherical Astronomy        — dispersion vs zenith angle
  Filippenko 1982, PASP 94, 715          — atmospheric dispersion in spectra
  Damian Peach, JBAA 122(4) 2012        — amateur ADC practical guide
  Birch & Downs 1993, Metrologia 30, 155 — updated Edlén equation

Place in: Siril Suites folder
Run via:  Siril → Scripts → software_adc
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("scipy")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")

import os
import sys
import glob
import struct
import threading
import traceback
from datetime import datetime

import numpy as np
from scipy.ndimage import shift as ndimage_shift, gaussian_filter
from scipy.optimize import minimize
from astropy.io import fits as astropy_fits

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QRadioButton, QButtonGroup, QSplitter, QScrollArea,
    QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QFont, QColor, QPainter, QPen

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

SIRIL_BG       = "#1e2128"
SIRIL_BG2      = "#252930"
SIRIL_BG3      = "#2d3340"
SIRIL_ACCENT   = "#4a9eff"
SIRIL_ACCENT2  = "#2d6abf"
SIRIL_TEXT     = "#dde3ee"
SIRIL_TEXT_DIM = "#7a8499"
SIRIL_BORDER   = "#3a4055"
SIRIL_SUCCESS  = "#4caf7d"
SIRIL_WARNING  = "#e8c46a"
SIRIL_ERROR    = "#cc4444"
SIRIL_NOVA     = "#ff6b35"
SIRIL_SECTION  = "#9db4d0"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: "Segoe UI", sans-serif;
    font-size: 9pt;
}}
QGroupBox {{
    border: 1px solid {SIRIL_BORDER};
    border-radius: 5px; margin-top: 8px;
    padding: 6px; font-weight: bold; color: {SIRIL_SECTION};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QTabWidget::pane {{ border: 1px solid {SIRIL_BORDER}; background: {SIRIL_BG}; }}
QTabBar::tab {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT_DIM};
    padding: 6px 14px; border: 1px solid {SIRIL_BORDER};
    border-bottom: none; border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{ background: {SIRIL_BG}; color: {SIRIL_ACCENT}; font-weight: bold; }}
QPushButton {{
    background: {SIRIL_BG3}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    padding: 5px 12px; min-height: 22px;
}}
QPushButton:hover {{ background: {SIRIL_ACCENT2}; border-color: {SIRIL_ACCENT}; }}
QPushButton[objectName="primary"] {{
    background: {SIRIL_ACCENT2}; color: white; font-weight: bold;
    border-color: {SIRIL_ACCENT};
}}
QPushButton[objectName="danger"] {{
    background: #4a1e1e; color: #e07070; border-color: #803030;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px; padding: 3px 6px;
}}
QProgressBar {{
    background: {SIRIL_BG2}; border: 1px solid {SIRIL_BORDER};
    border-radius: 4px; text-align: center;
}}
QProgressBar::chunk {{ background: {SIRIL_ACCENT}; border-radius: 3px; }}
QPlainTextEdit {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    font-family: "Courier New", monospace; font-size: 8pt;
}}
QLabel[objectName="dim"]  {{ color: {SIRIL_TEXT_DIM}; }}
QLabel[objectName="ok"]   {{ color: {SIRIL_SUCCESS}; }}
QLabel[objectName="warn"] {{ color: {SIRIL_WARNING}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QScrollArea {{ border: none; }}
"""

# Channel colours for display
CH_COLORS = {
    "R": "#e05050",
    "G": "#50c050",
    "B": "#5080e0",
}

# ─────────────────────────────────────────────────────────────────────────────
# PECK-REEDER REFRACTIVE INDEX MODEL (1972)
# ─────────────────────────────────────────────────────────────────────────────

def peck_reeder_n(wavelength_um: float) -> float:
    """
    Peck & Reeder 1972, JOSA 62, 958.
    Refractive index of dry air at STP (0°C, 101.325 kPa).
    Accurate to 2×10⁻⁹ from 0.23–1.69 μm.
    λ in micrometers.
    Returns (n - 1) × 10⁸ (refractivity in units of 10⁻⁸)
    """
    s = 1.0 / wavelength_um  # wavenumber in μm⁻¹
    # 2-term Sellmeier formula from Peck & Reeder (1972)
    N = 8060.51 + 2480990 / (132.274 - s**2) + 17455.7 / (39.32957 - s**2)
    return N  # units: 10⁻⁸


def refractive_index(wavelength_nm: float,
                     temperature_C: float = 15.0,
                     pressure_hPa: float = 1013.25,
                     humidity_pct: float = 0.0) -> float:
    """
    Full refractive index of air at given conditions.
    Birch & Downs (1993) correction applied to Peck-Reeder base.

    wavelength_nm: wavelength in nanometers
    temperature_C: ambient temperature in Celsius
    pressure_hPa:  pressure in hectopascals (mbar)
    humidity_pct:  relative humidity in percent (0–100)

    Returns n (refractive index, dimensionless).
    """
    lam_um = wavelength_nm / 1000.0
    # Base refractivity at STP (10⁻⁸ units)
    N_stp  = peck_reeder_n(lam_um)
    # Convert to n-1 at STP
    n_m1_stp = N_stp * 1e-8
    # Correct for pressure and temperature (ideal gas law)
    T_K  = temperature_C + 273.15
    T0_K = 273.15
    P0   = 1013.25
    P    = pressure_hPa
    n_m1 = n_m1_stp * (P / P0) * (T0_K / T_K)
    # Water vapour correction (Birch & Downs 1993)
    if humidity_pct > 0:
        # Saturation vapour pressure (hPa) via Buck (1981)
        e_s  = 6.1121 * np.exp((18.678 - temperature_C / 234.5) *
                                temperature_C / (257.14 + temperature_C))
        e    = (humidity_pct / 100.0) * e_s  # partial pressure of water vapour
        # Water vapour contribution (approximate, from Owens 1967)
        N_w  = (-41.8 / lam_um**2 + 0.542) * e / 100.0 * 1e-6
        n_m1 -= N_w  # water vapour reduces n
    return 1.0 + n_m1


def compute_dispersion_arcsec(altitude_deg: float,
                               wavelength_nm: float,
                               ref_wavelength_nm: float = 550.0,
                               temperature_C: float = 15.0,
                               pressure_hPa: float = 1013.25,
                               humidity_pct: float = 0.0) -> float:
    """
    Atmospheric dispersion relative to reference wavelength.
    Smart (1931): ΔR(λ) = 206265 × [n(λ) - n(λ_ref)] × tan(Z)
    where Z = zenith angle = 90° - altitude.

    Returns: dispersion in arcseconds (positive = toward zenith for blue).
    """
    Z_rad = np.radians(90.0 - altitude_deg)
    n_lam = refractive_index(wavelength_nm, temperature_C, pressure_hPa, humidity_pct)
    n_ref = refractive_index(ref_wavelength_nm, temperature_C, pressure_hPa, humidity_pct)
    return 206265.0 * (n_lam - n_ref) * np.tan(Z_rad)


def compute_rgb_shifts_px(altitude_deg: float,
                           pixel_scale_arcsec: float,
                           r_nm: float = 650.0,
                           g_nm: float = 530.0,
                           b_nm: float = 450.0,
                           temperature_C: float = 15.0,
                           pressure_hPa: float = 1013.25,
                           humidity_pct: float = 0.0) -> tuple:
    """
    Compute expected dispersion shift for R, G, B channels in pixels.
    Green channel is the reference (shift = 0).
    Returns (dr_px, dg_px, db_px) — shifts along the altitude axis.
    Negative = toward zenith (shorter wavelengths refract more toward zenith).
    """
    dr = compute_dispersion_arcsec(altitude_deg, r_nm, g_nm,
                                    temperature_C, pressure_hPa, humidity_pct)
    db = compute_dispersion_arcsec(altitude_deg, b_nm, g_nm,
                                    temperature_C, pressure_hPa, humidity_pct)
    dg = 0.0
    # Convert arcsec to pixels
    return (dr / pixel_scale_arcsec,
            dg / pixel_scale_arcsec,
            db / pixel_scale_arcsec)


def dispersion_vs_altitude(r_nm=650, g_nm=530, b_nm=450,
                            temp=15.0, pressure=1013.25,
                            pixel_scale=0.5) -> dict:
    """
    Precompute R/B dispersion vs altitude table for display.
    Returns dict of arrays for plotting.
    """
    altitudes = np.linspace(5, 90, 200)
    dr_arr = np.zeros(len(altitudes))
    db_arr = np.zeros(len(altitudes))
    for i, alt in enumerate(altitudes):
        dr, _, db = compute_rgb_shifts_px(alt, pixel_scale, r_nm, g_nm, b_nm,
                                           temp, pressure)
        dr_arr[i] = dr; db_arr[i] = db
    return {"altitudes": altitudes, "dr_px": dr_arr, "db_px": db_arr}


# ─────────────────────────────────────────────────────────────────────────────
# EMPIRICAL STAR CENTROID MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def detect_stars_centroid(image: np.ndarray, threshold_sigma: float = 5.0,
                           min_size: int = 3, max_size: int = 30) -> list:
    """
    Detect stars via thresholding + centroid fitting.
    Returns list of (x, y, flux) tuples.
    """
    from scipy.ndimage import label, center_of_mass
    bg     = np.median(image)
    noise  = 1.4826 * np.median(np.abs(image - bg))
    thresh = bg + threshold_sigma * noise
    mask   = image > thresh
    labeled, n = label(mask)
    stars  = []
    for i in range(1, n + 1):
        coords = np.where(labeled == i)
        if len(coords[0]) < min_size or len(coords[0]) > max_size**2:
            continue
        # Centroid
        flux = image[coords].sum()
        cx   = np.sum(coords[1] * image[coords]) / flux
        cy   = np.sum(coords[0] * image[coords]) / flux
        stars.append((float(cx), float(cy), float(flux)))
    return sorted(stars, key=lambda s: -s[2])  # brightest first


def match_stars(stars_a: list, stars_b: list,
                max_dist: float = 10.0) -> list:
    """
    Match star lists by nearest neighbour.
    Returns list of (a_idx, b_idx, dx, dy) matches.
    """
    matches = []
    used_b  = set()
    for i, (ax, ay, _) in enumerate(stars_a):
        best_j = -1; best_d = max_dist
        for j, (bx, by, _) in enumerate(stars_b):
            if j in used_b: continue
            d = np.sqrt((ax-bx)**2 + (ay-by)**2)
            if d < best_d:
                best_d = d; best_j = j
        if best_j >= 0:
            bx, by, _ = stars_b[best_j]
            matches.append((i, best_j, ax - bx, ay - by))
            used_b.add(best_j)
    return matches


def measure_chromatic_shift(r_channel: np.ndarray,
                              g_channel: np.ndarray,
                              b_channel: np.ndarray,
                              threshold_sigma: float = 5.0,
                              max_stars: int = 50) -> dict:
    """
    Measure actual R-G and B-G chromatic shifts from star centroids.
    Returns {"dr_y","dr_x","db_y","db_x","n_rg","n_bg","quality"}.
    G is reference (shift=0).
    """
    stars_r = detect_stars_centroid(r_channel, threshold_sigma)[:max_stars]
    stars_g = detect_stars_centroid(g_channel, threshold_sigma)[:max_stars]
    stars_b = detect_stars_centroid(b_channel, threshold_sigma)[:max_stars]

    rg_matches = match_stars(stars_r, stars_g)
    bg_matches = match_stars(stars_b, stars_g)

    def weighted_median(vals, weights):
        if not vals: return 0.0
        idx = np.argsort(vals)
        vals_s = np.array(vals)[idx]
        w_s    = np.array(weights)[idx]
        cumw   = np.cumsum(w_s)
        total  = cumw[-1]
        k      = np.searchsorted(cumw, total * 0.5)
        return float(vals_s[k])

    # R-G shift
    if rg_matches:
        rg_dx = [m[2] for m in rg_matches]
        rg_dy = [m[3] for m in rg_matches]
        # Weights = flux of fainter star
        rg_w  = [min(stars_r[m[0]][2], stars_g[m[1]][2]) for m in rg_matches]
        dr_x  = weighted_median(rg_dx, rg_w)
        dr_y  = weighted_median(rg_dy, rg_w)
    else:
        dr_x = dr_y = 0.0

    # B-G shift
    if bg_matches:
        bg_dx = [m[2] for m in bg_matches]
        bg_dy = [m[3] for m in bg_matches]
        bg_w  = [min(stars_b[m[0]][2], stars_g[m[1]][2]) for m in bg_matches]
        db_x  = weighted_median(bg_dx, bg_w)
        db_y  = weighted_median(bg_dy, bg_w)
    else:
        db_x = db_y = 0.0

    n_rg = len(rg_matches); n_bg = len(bg_matches)
    quality = "good" if (n_rg >= 5 and n_bg >= 5) else (
              "poor" if (n_rg < 3 or n_bg < 3) else "fair")

    return {"dr_y": dr_y, "dr_x": dr_x,
            "db_y": db_y, "db_x": db_x,
            "n_rg": n_rg, "n_bg": n_bg,
            "quality": quality}


# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL SHIFT APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def apply_channel_shifts(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                          dr_y: float, dr_x: float,
                          db_y: float, db_x: float,
                          interp_order: int = 3) -> tuple:
    """
    Apply sub-pixel shifts to R and B channels (G is reference).
    Uses scipy.ndimage.shift with sinc interpolation (order=3).
    Returns (r_corr, g, b_corr).
    """
    r_corr = ndimage_shift(r, (dr_y, dr_x), order=interp_order, mode="reflect")
    b_corr = ndimage_shift(b, (db_y, db_x), order=interp_order, mode="reflect")
    return r_corr, g.copy(), b_corr


# ─────────────────────────────────────────────────────────────────────────────
# FITS I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_rgb_fits(path: str) -> tuple:
    """
    Load RGB FITS. Returns (r, g, b, header) as float32 arrays.
    Handles: [3, H, W], [H, W, 3], single-channel (luminance only).
    """
    with astropy_fits.open(path) as hdul:
        data   = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    if data.ndim == 3:
        if data.shape[0] == 3:
            return data[0], data[1], data[2], header
        elif data.shape[2] == 3:
            return data[:,:,0], data[:,:,1], data[:,:,2], header
        else:
            raise ValueError(f"Unexpected shape: {data.shape}")
    elif data.ndim == 2:
        return data, data.copy(), data.copy(), header
    raise ValueError(f"Unsupported FITS shape: {data.shape}")


def load_mono_fits(path: str) -> tuple:
    with astropy_fits.open(path) as hdul:
        data   = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    if data.ndim == 3:
        data = data[0]
    return data, header


def save_rgb_fits(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                  path: str, header=None, history: str = ""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stack = np.stack([r, g, b], axis=0).astype(np.float32)
    hdu   = astropy_fits.PrimaryHDU(data=stack, header=header)
    if history:
        hdu.header["HISTORY"] = history
    hdu.header["ADCCORR"] = "HyperLoad Software ADC"
    hdu.writeto(path, overwrite=True)


# ─────────────────────────────────────────────────────────────────────────────
# SER SUPPORT (for planetary sequences)
# ─────────────────────────────────────────────────────────────────────────────

SER_HEADER_SIZE = 178

def read_ser_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(SER_HEADER_SIZE)
    w = struct.unpack_from("<i", raw, 26)[0]
    h = struct.unpack_from("<i", raw, 30)[0]
    depth = struct.unpack_from("<i", raw, 34)[0]
    n_frames = struct.unpack_from("<i", raw, 38)[0]
    color_id = struct.unpack_from("<i", raw, 18)[0]
    bytes_pp = (depth + 7) // 8
    is_color = color_id not in (0, 100)
    channels = 3 if is_color else 1
    return {
        "width": w, "height": h, "depth": depth,
        "frame_count": n_frames, "bytes_per_pixel": bytes_pp,
        "channels": channels, "is_color": is_color,
        "frame_size": w * h * bytes_pp * channels,
        "header_size": SER_HEADER_SIZE,
    }

def read_ser_frame_rgb(f, header: dict, idx: int) -> tuple:
    """Read one SER frame, return (r, g, b) float32 arrays or (None,None,None) if mono."""
    offset = header["header_size"] + idx * header["frame_size"]
    f.seek(offset)
    raw  = f.read(header["frame_size"])
    dt   = np.uint8 if header["bytes_per_pixel"] == 1 else np.uint16
    arr  = np.frombuffer(raw, dtype=dt).astype(np.float32)
    w, h = header["width"], header["height"]
    if header["channels"] == 3:
        arr = arr.reshape(h, w, 3)
        mx  = float((1 << header["depth"]) - 1)
        arr = arr / mx
        return arr[:,:,0], arr[:,:,1], arr[:,:,2]
    arr = arr.reshape(h, w)
    arr = arr / float((1 << header["depth"]) - 1)
    return arr, arr.copy(), arr.copy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ADC WORKER
# ─────────────────────────────────────────────────────────────────────────────

class ADCWorker(QThread):
    """
    ADC correction worker. Supports:
    - Single RGB FITS
    - Separate R/G/B FITS
    - FITS folder (batch)
    - SER sequence (per-frame)
    """
    progress     = pyqtSignal(int, int, str)
    log_line     = pyqtSignal(str)
    shift_measured = pyqtSignal(dict)    # measured/computed shifts
    preview_ready  = pyqtSignal(np.ndarray, np.ndarray)  # before, after
    finished     = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event,
                 parent=None):
        super().__init__(parent)
        self.cfg     = config
        self._cancel = cancel_event

    def run(self):
        try:
            self._run()
        except Exception as e:
            tb = traceback.format_exc()
            self.log_line.emit(f"CRITICAL ERROR: {e}\n{tb}")
            self.finished.emit({"success": False, "error": str(e)})

    def _run(self):
        cfg = self.cfg
        log = self.log_line.emit
        log("═══ HyperLoad Software ADC Correction ═══")
        log(f"  Mode:        {cfg['mode']}")
        log(f"  Input type:  {cfg['input_type']}")
        log(f"  Method:      {cfg['method']}")
        if cfg["method"] == "theoretical":
            log(f"  Altitude:    {cfg['altitude']:.1f}°")
            log(f"  Pixel scale: {cfg['pixel_scale']:.3f} arcsec/px")
            log(f"  Temperature: {cfg['temperature']:.1f} °C")
            log(f"  Pressure:    {cfg['pressure']:.1f} hPa")
        log("")

        mode = cfg["mode"]
        if mode == "single_rgb":
            self._process_single_rgb()
        elif mode == "separate_channels":
            self._process_separate()
        elif mode == "batch_folder":
            self._process_batch()
        elif mode == "ser_sequence":
            self._process_ser()
        else:
            self.finished.emit({"success": False, "error": f"Unknown mode: {mode}"})

    # ── Compute or measure shifts ─────────────────────────────────────────────

    def _get_shifts(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> dict:
        cfg = self.cfg
        if cfg["method"] == "theoretical":
            dr_px, _, db_px = compute_rgb_shifts_px(
                cfg["altitude"], cfg["pixel_scale"],
                cfg.get("r_nm", 650.0), cfg.get("g_nm", 530.0),
                cfg.get("b_nm", 450.0),
                cfg["temperature"], cfg["pressure"], cfg["humidity"])
            # Shifts are along altitude axis. We need to decompose into x/y
            # using the altitude direction angle (parallactic angle).
            pa_rad = np.radians(cfg.get("parallactic_angle", 0.0))
            dr_y = dr_px * np.cos(pa_rad); dr_x = dr_px * np.sin(pa_rad)
            db_y = db_px * np.cos(pa_rad); db_x = db_px * np.sin(pa_rad)
            result = {
                "dr_y": dr_y, "dr_x": dr_x,
                "db_y": db_y, "db_x": db_x,
                "dr_total": dr_px, "db_total": db_px,
                "n_rg": -1, "n_bg": -1, "quality": "theoretical",
                "method": "theoretical",
            }
        else:  # empirical
            result = measure_chromatic_shift(r, g, b,
                                              cfg.get("threshold_sigma", 5.0))
            result["method"] = "empirical"
            # Flip: we measured where stars in R/B ARE relative to G,
            # so we need to shift by the negative to align them
            result["dr_y"] = -result["dr_y"]
            result["dr_x"] = -result["dr_x"]
            result["db_y"] = -result["db_y"]
            result["db_x"] = -result["db_x"]
            result["dr_total"] = np.sqrt(result["dr_y"]**2 + result["dr_x"]**2)
            result["db_total"] = np.sqrt(result["db_y"]**2 + result["db_x"]**2)

        return result

    def _log_shifts(self, shifts: dict):
        log = self.log_line.emit
        log(f"  Method: {shifts['method']}")
        log(f"  R shift: dy={shifts['dr_y']:+.3f}  dx={shifts['dr_x']:+.3f} px  "
            f"(total {shifts['dr_total']:.3f} px)")
        log(f"  B shift: dy={shifts['db_y']:+.3f}  dx={shifts['db_x']:+.3f} px  "
            f"(total {shifts['db_total']:.3f} px)")
        if shifts["method"] == "empirical":
            log(f"  Stars matched: R-G={shifts['n_rg']}  B-G={shifts['n_bg']}  "
                f"quality={shifts['quality']}")
        self.shift_measured.emit(shifts)

    # ── Single RGB FITS ───────────────────────────────────────────────────────

    def _process_single_rgb(self):
        cfg = self.cfg; log = self.log_line.emit
        path = cfg["input_path"]
        log(f"Loading: {os.path.basename(path)}")
        self.progress.emit(5, 100, "Loading FITS…")
        try:
            r, g, b, hdr = load_rgb_fits(path)
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)}); return

        self.progress.emit(15, 100, "Computing/measuring shifts…")
        shifts = self._get_shifts(r, g, b)
        self._log_shifts(shifts)

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        self.progress.emit(40, 100, "Applying correction…")
        r_c, g_c, b_c = apply_channel_shifts(r, g, b,
                                               shifts["dr_y"], shifts["dr_x"],
                                               shifts["db_y"], shifts["db_x"],
                                               cfg.get("interp_order", 3))

        # Preview
        def _preview_rgb(rc, gc, bc):
            p_lo = np.percentile(gc, 0.5); p_hi = np.percentile(gc, 99.5)
            def _s(x): return np.clip((x - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)
            return np.stack([_s(rc), _s(gc), _s(bc)], axis=-1).astype(np.float32)
        self.preview_ready.emit(_preview_rgb(r, g, b), _preview_rgb(r_c, g_c, b_c))

        self.progress.emit(80, 100, "Saving…")
        out_path = self._make_output_path(path, cfg["output_dir"])
        history  = (f"Software ADC  alt={cfg.get('altitude','?')}°  "
                    f"dr={shifts['dr_total']:.3f}px  "
                    f"db={shifts['db_total']:.3f}px  method={shifts['method']}")
        save_rgb_fits(r_c, g_c, b_c, out_path, hdr, history)
        log(f"✓ Saved: {out_path}")
        self.progress.emit(100, 100, "Done")
        self.finished.emit({"success": True, "output_path": out_path,
                            "shifts": shifts})

    # ── Separate R/G/B FITS ───────────────────────────────────────────────────

    def _process_separate(self):
        cfg = self.cfg; log = self.log_line.emit
        r_path = cfg.get("r_path",""); g_path = cfg.get("g_path","")
        b_path = cfg.get("b_path","")
        if not all([r_path, g_path, b_path]):
            self.finished.emit({"success": False,
                                "error": "All three channel paths required"})
            return
        log("Loading separate R/G/B channels…")
        self.progress.emit(5, 100, "Loading channels…")
        try:
            r, hdr_r = load_mono_fits(r_path)
            g, _     = load_mono_fits(g_path)
            b, _     = load_mono_fits(b_path)
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)}); return

        self.progress.emit(20, 100, "Computing/measuring shifts…")
        shifts = self._get_shifts(r, g, b)
        self._log_shifts(shifts)

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        self.progress.emit(50, 100, "Applying correction…")
        r_c, g_c, b_c = apply_channel_shifts(r, g, b,
                                               shifts["dr_y"], shifts["dr_x"],
                                               shifts["db_y"], shifts["db_x"],
                                               cfg.get("interp_order", 3))

        def _preview_rgb(rc, gc, bc):
            p_lo = np.percentile(gc, 0.5); p_hi = np.percentile(gc, 99.5)
            def _s(x): return np.clip((x-p_lo)/(p_hi-p_lo+1e-10), 0, 1)
            return np.stack([_s(rc), _s(gc), _s(bc)], axis=-1).astype(np.float32)
        self.preview_ready.emit(_preview_rgb(r, g, b), _preview_rgb(r_c, g_c, b_c))

        self.progress.emit(75, 100, "Saving…")
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        history = (f"Software ADC  dr={shifts['dr_total']:.3f}px  "
                   f"db={shifts['db_total']:.3f}px")
        base    = os.path.splitext(os.path.basename(g_path))[0]
        out_path = os.path.join(out_dir, f"{base}_adc_corrected.fit")
        save_rgb_fits(r_c, g_c, b_c, out_path, hdr_r, history)
        log(f"✓ Saved RGB: {out_path}")
        self.progress.emit(100, 100, "Done")
        self.finished.emit({"success": True, "output_path": out_path,
                            "shifts": shifts})

    # ── Batch folder ──────────────────────────────────────────────────────────

    def _process_batch(self):
        cfg = self.cfg; log = self.log_line.emit
        folder = cfg["input_folder"]
        files  = sorted(
            glob.glob(os.path.join(folder, "*.fit")) +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts")))
        if not files:
            self.finished.emit({"success": False,
                                "error": "No FITS files found"}); return
        log(f"Batch: {len(files)} FITS files")
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        n_done = 0

        # Compute shifts once from first file (theoretical) or measure each
        first_shifts = None

        for i, path in enumerate(files):
            if self._cancel.is_set(): break
            self.progress.emit(i, len(files),
                               f"Processing {os.path.basename(path)}")
            try:
                r, g, b, hdr = load_rgb_fits(path)
            except Exception as e:
                log(f"  Skip {os.path.basename(path)}: {e}"); continue

            if cfg["method"] == "theoretical" and first_shifts:
                shifts = first_shifts
            else:
                shifts = self._get_shifts(r, g, b)
                if cfg["method"] == "theoretical" and not first_shifts:
                    first_shifts = shifts
                    self._log_shifts(shifts)

            r_c, g_c, b_c = apply_channel_shifts(
                r, g, b,
                shifts["dr_y"], shifts["dr_x"],
                shifts["db_y"], shifts["db_x"],
                cfg.get("interp_order", 3))

            out_path = self._make_output_path(path, out_dir)
            history  = (f"Software ADC batch  "
                        f"dr={shifts['dr_total']:.3f}px  "
                        f"db={shifts['db_total']:.3f}px")
            save_rgb_fits(r_c, g_c, b_c, out_path, hdr, history)
            n_done += 1

            if i == 0:
                def _pv(rc, gc, bc):
                    p_lo = np.percentile(gc, 0.5); p_hi = np.percentile(gc, 99.5)
                    def _s(x): return np.clip((x-p_lo)/(p_hi-p_lo+1e-10), 0, 1)
                    return np.stack([_s(rc), _s(gc), _s(bc)], axis=-1).astype(np.float32)
                self.preview_ready.emit(_pv(r, g, b), _pv(r_c, g_c, b_c))
                self.shift_measured.emit(shifts)

        log(f"✓ Batch complete: {n_done}/{len(files)} processed")
        self.progress.emit(len(files), len(files), "Done")
        self.finished.emit({"success": True, "n_processed": n_done,
                            "output_dir": out_dir})

    # ── SER sequence ──────────────────────────────────────────────────────────

    def _process_ser(self):
        cfg = self.cfg; log = self.log_line.emit
        ser_path = cfg["ser_path"]
        if not os.path.isfile(ser_path):
            self.finished.emit({"success": False,
                                "error": f"SER not found: {ser_path}"}); return
        hdr = read_ser_header(ser_path)
        if not hdr["is_color"]:
            self.finished.emit({"success": False,
                                "error": "SER is monochrome — cannot apply RGB ADC"})
            return
        N = hdr["frame_count"]
        log(f"SER: {N} frames  {hdr['width']}×{hdr['height']}  "
            f"{'RGB' if hdr['is_color'] else 'mono'}")

        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        base     = os.path.splitext(os.path.basename(ser_path))[0]
        out_path = os.path.join(out_dir, f"{base}_adc.fit")

        # Measure shifts on first frame
        with open(ser_path, "rb") as f:
            r0, g0, b0 = read_ser_frame_rgb(f, hdr, 0)
        shifts = self._get_shifts(r0, g0, b0)
        self._log_shifts(shifts)

        def _pv(rc, gc, bc):
            p_lo = np.percentile(gc, 0.5); p_hi = np.percentile(gc, 99.5)
            def _s(x): return np.clip((x-p_lo)/(p_hi-p_lo+1e-10), 0, 1)
            return np.stack([_s(rc), _s(gc), _s(bc)], axis=-1).astype(np.float32)
        self.preview_ready.emit(_pv(r0, g0, b0),
                                _pv(*apply_channel_shifts(r0, g0, b0,
                                     shifts["dr_y"], shifts["dr_x"],
                                     shifts["db_y"], shifts["db_x"])))

        # Process all frames → save as FITS stack
        frames = []
        max_frames = cfg.get("max_frames") or N
        with open(ser_path, "rb") as f:
            for i in range(min(N, max_frames)):
                if self._cancel.is_set(): break
                r, g, b = read_ser_frame_rgb(f, hdr, i)
                r_c, g_c, b_c = apply_channel_shifts(
                    r, g, b,
                    shifts["dr_y"], shifts["dr_x"],
                    shifts["db_y"], shifts["db_x"],
                    cfg.get("interp_order", 3))
                # Recombine channels
                frame_rgb = np.stack([r_c, g_c, b_c], axis=0)
                frames.append(frame_rgb.astype(np.float32))
                if i % 50 == 0:
                    self.progress.emit(i, min(N, max_frames),
                                      f"Correcting frame {i}/{min(N,max_frames)}")

        if not frames:
            self.finished.emit({"success": False, "error": "No frames processed"})
            return

        log(f"Saving {len(frames)} corrected frames → {out_path}")
        self.progress.emit(min(N, max_frames)-1, min(N, max_frames), "Saving…")
        # Save as 4-D FITS [N_frames, 3, H, W]
        stack = np.stack(frames, axis=0)
        hdu   = astropy_fits.PrimaryHDU(data=stack)
        hdu.header["HISTORY"] = (f"Software ADC  "
                                  f"dr={shifts['dr_total']:.3f}px  "
                                  f"db={shifts['db_total']:.3f}px  "
                                  f"N={len(frames)}")
        hdu.writeto(out_path, overwrite=True)
        log(f"✓ Saved: {out_path}")
        self.progress.emit(min(N, max_frames), min(N, max_frames), "Done")
        self.finished.emit({"success": True, "output_path": out_path,
                            "n_frames": len(frames), "shifts": shifts})

    def _make_output_path(self, input_path: str, out_dir: str) -> str:
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(out_dir, f"{base}_adc.fit")


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class ADCPreviewCanvas(FigureCanvasQTAgg):
    """Side-by-side before/after RGB preview + dispersion curve."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 7), facecolor=SIRIL_BG)
        self.ax_before  = self.fig.add_subplot(2, 2, 1)
        self.ax_after   = self.fig.add_subplot(2, 2, 2)
        self.ax_disp    = self.fig.add_subplot(2, 2, 3)
        self.ax_profile = self.fig.add_subplot(2, 2, 4)
        self._style_all()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style_all(self):
        for ax, title, color in [
            (self.ax_before,  "Before ADC correction",  SIRIL_ERROR),
            (self.ax_after,   "After ADC correction",   SIRIL_SUCCESS),
            (self.ax_disp,    "Dispersion vs altitude", SIRIL_ACCENT),
            (self.ax_profile, "Star colour profile",    SIRIL_SECTION),
        ]:
            ax.set_facecolor(SIRIL_BG2)
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_color(SIRIL_BORDER)

    def show_before_after(self, before: np.ndarray, after: np.ndarray):
        """before, after: [H, W, 3] float32 in [0,1]."""
        def _ds(img, max_px=600):
            h, w = img.shape[:2]; f = max(1, max(h,w)//max_px)
            return img[::f, ::f]
        for ax, img, title, color in [
            (self.ax_before, before, "Before ADC correction", SIRIL_ERROR),
            (self.ax_after,  after,  "After ADC correction",  SIRIL_SUCCESS),
        ]:
            ax.clear(); ax.set_facecolor(SIRIL_BG2)
            ax.imshow(np.clip(_ds(img), 0, 1), aspect="equal",
                      interpolation="nearest", origin="upper")
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values(): sp.set_color(SIRIL_BORDER)

        # Star profile comparison (central column)
        h, w = before.shape[:2]
        cx   = w // 2
        col_before = before[:, max(0,cx-2):cx+3, :].mean(axis=1)
        col_after  = after[:, max(0,cx-2):cx+3, :].mean(axis=1)
        ys = np.arange(h)
        self.ax_profile.clear(); self.ax_profile.set_facecolor(SIRIL_BG2)
        for chi, (ch, col_b) in enumerate(zip(["R","G","B"], [0,1,2])):
            self.ax_profile.plot(ys, col_before[:, chi],
                                 color=list(CH_COLORS.values())[chi],
                                 alpha=0.4, lw=1, ls="--")
            self.ax_profile.plot(ys, col_after[:, chi],
                                 color=list(CH_COLORS.values())[chi],
                                 lw=1.5, label=f"{ch} (corrected)")
        self.ax_profile.set_title("Star colour profile (--=before, —=after)",
                                   color=SIRIL_SECTION, fontsize=9)
        self.ax_profile.set_xlabel("Y pixel", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_profile.legend(fontsize=7, facecolor=SIRIL_BG3,
                                edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax_profile.tick_params(colors=SIRIL_TEXT_DIM, labelsize=8)
        for sp in self.ax_profile.spines.values(): sp.set_color(SIRIL_BORDER)

        self.fig.tight_layout(pad=0.4)
        self.draw()

    def show_dispersion_curve(self, pixel_scale: float,
                               r_nm: float, g_nm: float, b_nm: float,
                               temp: float, pressure: float,
                               current_alt: float = None):
        table = dispersion_vs_altitude(r_nm, g_nm, b_nm, temp, pressure,
                                        pixel_scale)
        self.ax_disp.clear(); self.ax_disp.set_facecolor(SIRIL_BG2)
        self.ax_disp.plot(table["altitudes"], np.abs(table["dr_px"]),
                          color=CH_COLORS["R"], lw=1.5, label="R-G")
        self.ax_disp.plot(table["altitudes"], np.abs(table["db_px"]),
                          color=CH_COLORS["B"], lw=1.5, label="B-G")
        if current_alt is not None:
            self.ax_disp.axvline(current_alt, color=SIRIL_NOVA, lw=1.2,
                                  ls="--", label=f"Now ({current_alt:.0f}°)")
        self.ax_disp.set_xlabel("Altitude (°)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_disp.set_ylabel("Dispersion (px)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_disp.set_title("Dispersion vs altitude", color=SIRIL_ACCENT, fontsize=9)
        self.ax_disp.legend(fontsize=7, facecolor=SIRIL_BG3,
                             edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax_disp.tick_params(colors=SIRIL_TEXT_DIM, labelsize=8)
        for sp in self.ax_disp.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.4)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Software ADC  —  HYPERLOAD  —  Siril")
        self.resize(1400, 900)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._build_ui()
        self._update_dispersion_curve()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QWidget(); header.setFixedHeight(56)
        header.setStyleSheet(
            f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon = QLabel("🌈")
        lbl_icon.setStyleSheet("font-size:20pt; background:transparent;")
        lbl_title = QLabel("Software Atmospheric Dispersion Corrector")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_NOVA}; font-size:13pt; font-weight:bold; "
            f"background:transparent;")
        lbl_sub = QLabel(
            "Peck-Reeder n(λ)  ·  Smart (1931) dispersion formula  ·  "
            "Theoretical + empirical modes  ·  Sub-pixel sinc shift")
        lbl_sub.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title)
        hl.addWidget(lbl_sub); hl.addStretch()
        root.addWidget(header)

        # ── Tabs ──────────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_tab_input(),    "📁  Input")
        self._tabs.addTab(self._build_tab_method(),   "⚙  Method")
        self._tabs.addTab(self._build_tab_preview(),  "🖼  Preview")
        self._tabs.addTab(self._build_tab_output(),   "💾  Output")
        self._tabs.addTab(self._build_tab_log(),      "📋  Log")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom = QWidget(); bottom.setFixedHeight(50)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 6, 10, 6)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(10); self._progress.setValue(0)
        bl.addWidget(self._progress, 1)
        self._btn_run = QPushButton("▶  Correct ADC")
        self._btn_run.setObjectName("primary")
        self._btn_run.setMinimumWidth(140); self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run)
        bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(34); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bl.addWidget(self._btn_cancel)
        self._status = QLabel("Ready — load input and configure method")
        self._status.setObjectName("dim")
        self._status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status, 1)
        root.addWidget(bottom)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_tab_input(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        # Input mode selector
        grp_mode = QGroupBox("Input mode")
        mg = QVBoxLayout(grp_mode)
        self._rb_single  = QRadioButton("Single RGB FITS  ([3, H, W] or [H, W, 3])")
        self._rb_separate = QRadioButton("Separate R / G / B FITS files")
        self._rb_batch   = QRadioButton("Batch folder  (all RGB FITS in folder)")
        self._rb_ser     = QRadioButton("SER sequence  (colour planetary video)")
        self._rb_single.setChecked(True)
        self._bg_mode = QButtonGroup()
        for rb in [self._rb_single, self._rb_separate, self._rb_batch, self._rb_ser]:
            self._bg_mode.addButton(rb); mg.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
        lay.addWidget(grp_mode)

        # Single RGB
        self._grp_single = QGroupBox("Single RGB FITS")
        sl = QHBoxLayout(self._grp_single)
        self._edit_single = QLineEdit(); self._edit_single.setPlaceholderText("RGB FITS path…")
        btn_s = QPushButton("Browse…"); btn_s.setFixedWidth(80)
        btn_s.clicked.connect(lambda: self._browse_fits(self._edit_single))
        sl.addWidget(self._edit_single); sl.addWidget(btn_s)
        lay.addWidget(self._grp_single)

        # Separate channels
        self._grp_sep = QGroupBox("Separate R / G / B FITS")
        sf = QFormLayout(self._grp_sep)
        self._edit_r = QLineEdit(); self._edit_r.setPlaceholderText("Red FITS…")
        self._edit_g = QLineEdit(); self._edit_g.setPlaceholderText("Green FITS…")
        self._edit_b = QLineEdit(); self._edit_b.setPlaceholderText("Blue FITS…")
        for lbl, edit in [("R:", self._edit_r), ("G:", self._edit_g), ("B:", self._edit_b)]:
            row = QHBoxLayout()
            edit_copy = edit
            row.addWidget(edit_copy, 1)
            btn = QPushButton("Browse…"); btn.setFixedWidth(80)
            btn.clicked.connect(lambda checked, e=edit_copy: self._browse_fits(e))
            row.addWidget(btn)
            sf.addRow(lbl, row)
        self._grp_sep.setVisible(False)
        lay.addWidget(self._grp_sep)

        # Batch
        self._grp_batch = QGroupBox("Batch folder")
        bf = QHBoxLayout(self._grp_batch)
        self._edit_batch_folder = QLineEdit()
        self._edit_batch_folder.setPlaceholderText("Folder with RGB FITS files…")
        btn_bf = QPushButton("Browse…"); btn_bf.setFixedWidth(80)
        btn_bf.clicked.connect(self._browse_folder_input)
        bf.addWidget(self._edit_batch_folder); bf.addWidget(btn_bf)
        self._grp_batch.setVisible(False)
        lay.addWidget(self._grp_batch)

        # SER
        self._grp_ser = QGroupBox("SER sequence")
        serf = QFormLayout(self._grp_ser)
        ser_row = QHBoxLayout()
        self._edit_ser = QLineEdit(); self._edit_ser.setPlaceholderText("SER file…")
        btn_ser = QPushButton("Browse…"); btn_ser.setFixedWidth(80)
        btn_ser.clicked.connect(self._browse_ser)
        ser_row.addWidget(self._edit_ser); ser_row.addWidget(btn_ser)
        serf.addRow("SER file:", ser_row)
        self._spin_max_frames = QSpinBox()
        self._spin_max_frames.setRange(0, 99999); self._spin_max_frames.setValue(0)
        self._spin_max_frames.setSpecialValueText("All frames")
        serf.addRow("Max frames:", self._spin_max_frames)
        self._grp_ser.setVisible(False)
        lay.addWidget(self._grp_ser)

        lay.addStretch()
        return w

    def _build_tab_method(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        # Method selector
        grp_method = QGroupBox("Correction method")
        mf = QVBoxLayout(grp_method)
        self._rb_theoretical = QRadioButton(
            "Theoretical  — compute shift from Peck-Reeder (1972) model\n"
            "  Inputs: altitude, pixel scale, temperature, pressure\n"
            "  Use when: you know the altitude at imaging time")
        self._rb_empirical   = QRadioButton(
            "Empirical  — measure actual RGB centroid offset from stars\n"
            "  No atmospheric model needed\n"
            "  Use when: many stars visible, altitude unknown, or non-standard conditions")
        self._rb_theoretical.setChecked(True)
        self._bg_meth = QButtonGroup()
        self._bg_meth.addButton(self._rb_theoretical)
        self._bg_meth.addButton(self._rb_empirical)
        for rb in [self._rb_theoretical, self._rb_empirical]:
            mf.addWidget(rb); rb.toggled.connect(self._on_method_changed)
        lay.addWidget(grp_method)

        # Theoretical params
        self._grp_theo = QGroupBox("Theoretical parameters  (Peck-Reeder 1972 / Smart 1931)")
        tf = QFormLayout(self._grp_theo)
        self._spin_alt = QDoubleSpinBox()
        self._spin_alt.setRange(5.0, 85.0); self._spin_alt.setValue(30.0)
        self._spin_alt.setSuffix(" °"); self._spin_alt.setSingleStep(1.0)
        self._spin_alt.valueChanged.connect(self._update_dispersion_curve)
        tf.addRow("Object altitude:", self._spin_alt)

        self._spin_pa = QDoubleSpinBox()
        self._spin_pa.setRange(0.0, 360.0); self._spin_pa.setValue(0.0)
        self._spin_pa.setSuffix(" °"); self._spin_pa.setSingleStep(5.0)
        tf.addRow("Parallactic angle (altitude direction):", self._spin_pa)
        lbl_pa = QLabel(
            "0° = North up (dispersion along Y axis)\n"
            "90° = East up (dispersion along X axis)\n"
            "Measure from planet elongation direction or compute from hour angle")
        lbl_pa.setObjectName("dim"); lbl_pa.setWordWrap(True); tf.addRow(lbl_pa)

        self._spin_pixel_scale = QDoubleSpinBox()
        self._spin_pixel_scale.setRange(0.01, 20.0); self._spin_pixel_scale.setValue(0.5)
        self._spin_pixel_scale.setSuffix(" arcsec/px"); self._spin_pixel_scale.setDecimals(4)
        self._spin_pixel_scale.valueChanged.connect(self._update_dispersion_curve)
        tf.addRow("Pixel scale:", self._spin_pixel_scale)

        self._spin_temp = QDoubleSpinBox()
        self._spin_temp.setRange(-30.0, 50.0); self._spin_temp.setValue(15.0)
        self._spin_temp.setSuffix(" °C"); self._spin_temp.setSingleStep(1.0)
        self._spin_temp.valueChanged.connect(self._update_dispersion_curve)
        tf.addRow("Temperature:", self._spin_temp)

        self._spin_pressure = QDoubleSpinBox()
        self._spin_pressure.setRange(600.0, 1100.0); self._spin_pressure.setValue(1013.25)
        self._spin_pressure.setSuffix(" hPa"); self._spin_pressure.setSingleStep(1.0)
        self._spin_pressure.valueChanged.connect(self._update_dispersion_curve)
        tf.addRow("Pressure:", self._spin_pressure)

        self._spin_humidity = QDoubleSpinBox()
        self._spin_humidity.setRange(0.0, 100.0); self._spin_humidity.setValue(0.0)
        self._spin_humidity.setSuffix(" %")
        tf.addRow("Humidity:", self._spin_humidity)

        # Effective wavelengths per channel
        wl_grp = QGroupBox("Effective wavelengths per channel")
        wf = QFormLayout(wl_grp)
        self._spin_r_nm = QDoubleSpinBox(); self._spin_r_nm.setRange(300,1000); self._spin_r_nm.setValue(650); self._spin_r_nm.setSuffix(" nm")
        self._spin_g_nm = QDoubleSpinBox(); self._spin_g_nm.setRange(300,1000); self._spin_g_nm.setValue(530); self._spin_g_nm.setSuffix(" nm")
        self._spin_b_nm = QDoubleSpinBox(); self._spin_b_nm.setRange(300,1000); self._spin_b_nm.setValue(450); self._spin_b_nm.setSuffix(" nm")
        wf.addRow("R effective λ:", self._spin_r_nm)
        wf.addRow("G effective λ:", self._spin_g_nm)
        wf.addRow("B effective λ:", self._spin_b_nm)
        lbl_w = QLabel(
            "Typical broadband: R=650nm G=530nm B=450nm\n"
            "Narrow R filter: ~660nm  ·  OIII: ~500nm  ·  Ha: ~656nm\n"
            "Effective λ = filter centre weighted by sensor QE")
        lbl_w.setObjectName("dim"); lbl_w.setWordWrap(True); wf.addRow(lbl_w)

        for s in [self._spin_r_nm, self._spin_g_nm, self._spin_b_nm]:
            s.valueChanged.connect(self._update_dispersion_curve)

        lay.addWidget(self._grp_theo)
        lay.addWidget(wl_grp)

        # Empirical params
        self._grp_emp = QGroupBox("Empirical parameters")
        ef = QFormLayout(self._grp_emp)
        self._spin_thresh = QDoubleSpinBox()
        self._spin_thresh.setRange(3.0, 20.0); self._spin_thresh.setValue(5.0)
        self._spin_thresh.setSuffix(" σ")
        ef.addRow("Star detection threshold:", self._spin_thresh)
        lbl_e = QLabel(
            "Stars must be clearly resolved in each channel.\n"
            "Increase threshold if too many false detections.\n"
            "Works best on bright stars, not in heavily nebulous fields.")
        lbl_e.setObjectName("dim"); lbl_e.setWordWrap(True); ef.addRow(lbl_e)
        self._grp_emp.setVisible(False)
        lay.addWidget(self._grp_emp)

        # Interpolation
        grp_interp = QGroupBox("Interpolation")
        ifl = QFormLayout(grp_interp)
        self._cmb_interp = QComboBox()
        self._cmb_interp.addItems([
            "Order 3 — sinc (recommended, sub-pixel accurate)",
            "Order 1 — bilinear (fast, slight blur)",
            "Order 5 — quintic (maximum quality, slow)",
        ])
        ifl.addRow("Interpolation:", self._cmb_interp)
        lay.addWidget(grp_interp)

        # Predicted shifts display
        self._grp_pred = QGroupBox("Predicted shifts (theoretical mode)")
        pl = QVBoxLayout(self._grp_pred)
        self._lbl_pred_r = QLabel("R: —"); self._lbl_pred_r.setStyleSheet(f"color:{CH_COLORS['R']};")
        self._lbl_pred_g = QLabel("G: 0 (reference)"); self._lbl_pred_g.setStyleSheet(f"color:{CH_COLORS['G']};")
        self._lbl_pred_b = QLabel("B: —"); self._lbl_pred_b.setStyleSheet(f"color:{CH_COLORS['B']};")
        pl.addWidget(self._lbl_pred_r); pl.addWidget(self._lbl_pred_g); pl.addWidget(self._lbl_pred_b)
        lay.addWidget(self._grp_pred)

        lay.addStretch()
        return w

    def _build_tab_preview(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self._canvas = ADCPreviewCanvas()
        lay.addWidget(self._canvas)
        return w

    def _build_tab_output(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        grp_out = QGroupBox("Output folder")
        of = QHBoxLayout(grp_out)
        self._edit_out = QLineEdit(); self._edit_out.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(80)
        btn_out.clicked.connect(lambda: self._edit_out.setText(
            QFileDialog.getExistingDirectory(self, "Output folder") or self._edit_out.text()))
        of.addWidget(self._edit_out); of.addWidget(btn_out)
        lay.addWidget(grp_out)

        grp_ref = QGroupBox("References & limitations")
        rl = QVBoxLayout(grp_ref)
        refs = [
            "Refractive index: Peck & Reeder 1972, JOSA 62, 958",
            "Dispersion model: Smart 1931, Spherical Astronomy",
            "Updated Edlén:    Birch & Downs 1993, Metrologia 30, 155",
            "Amateur guide:    Damian Peach, JBAA 122(4), 2012",
            "",
            "⚠  Software ADC corrects INTER-BAND shift (R-G, B-G centroid offset).",
            "   It CANNOT recover INTRA-BAND smearing within a single passband.",
            "   Hardware ADC is needed for altitudes below ~30° with broadband filters.",
            "   For altitudes above 40°, software correction is highly effective.",
        ]
        for r in refs:
            lbl = QLabel(r)
            lbl.setObjectName("dim" if not r.startswith("⚠") else "warn")
            rl.addWidget(lbl)
        lay.addWidget(grp_ref)
        lay.addStretch()
        return w

    def _build_tab_log(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._log.clear)
        lay.addWidget(btn_clr); lay.addWidget(self._log)
        return w

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        self._grp_single.setVisible(self._rb_single.isChecked())
        self._grp_sep.setVisible(self._rb_separate.isChecked())
        self._grp_batch.setVisible(self._rb_batch.isChecked())
        self._grp_ser.setVisible(self._rb_ser.isChecked())

    def _on_method_changed(self):
        theo = self._rb_theoretical.isChecked()
        self._grp_theo.setVisible(theo)
        self._grp_emp.setVisible(not theo)
        self._grp_pred.setVisible(theo)

    def _update_dispersion_curve(self):
        """Update predicted shifts and dispersion curve plot."""
        alt  = self._spin_alt.value()
        ps   = self._spin_pixel_scale.value()
        r_nm = self._spin_r_nm.value()
        g_nm = self._spin_g_nm.value()
        b_nm = self._spin_b_nm.value()
        temp = self._spin_temp.value()
        pres = self._spin_pressure.value()
        dr, _, db = compute_rgb_shifts_px(alt, ps, r_nm, g_nm, b_nm, temp, pres)
        self._lbl_pred_r.setText(
            f"R: {dr:+.3f} px  "
            f"({dr*ps:+.3f} arcsec)  "
            f"[{r_nm:.0f}nm vs {g_nm:.0f}nm]")
        self._lbl_pred_b.setText(
            f"B: {db:+.3f} px  "
            f"({db*ps:+.3f} arcsec)  "
            f"[{b_nm:.0f}nm vs {g_nm:.0f}nm]")
        self._canvas.show_dispersion_curve(ps, r_nm, g_nm, b_nm, temp, pres, alt)

    def _browse_fits(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FITS", "", "FITS (*.fit *.fits *.fts);;All (*)")
        if path:
            edit.setText(path)
            if not self._edit_out.text():
                self._edit_out.setText(os.path.dirname(path))

    def _browse_folder_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder")
        if d:
            self._edit_batch_folder.setText(d)
            if not self._edit_out.text():
                self._edit_out.setText(d)

    def _browse_ser(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SER file", "", "SER (*.ser);;All (*)")
        if path:
            self._edit_ser.setText(path)
            if not self._edit_out.text():
                self._edit_out.setText(os.path.dirname(path))

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _get_config(self) -> dict | None:
        out = self._edit_out.text().strip()
        if not out:
            QMessageBox.warning(self, "No output", "Set an output folder.")
            return None

        interp_map = {0: 3, 1: 1, 2: 5}
        interp_order = interp_map.get(self._cmb_interp.currentIndex(), 3)
        method = "theoretical" if self._rb_theoretical.isChecked() else "empirical"

        base = {
            "method":       method,
            "output_dir":   out,
            "interp_order": interp_order,
            "altitude":     self._spin_alt.value(),
            "pixel_scale":  self._spin_pixel_scale.value(),
            "temperature":  self._spin_temp.value(),
            "pressure":     self._spin_pressure.value(),
            "humidity":     self._spin_humidity.value(),
            "parallactic_angle": self._spin_pa.value(),
            "r_nm":         self._spin_r_nm.value(),
            "g_nm":         self._spin_g_nm.value(),
            "b_nm":         self._spin_b_nm.value(),
            "threshold_sigma": self._spin_thresh.value(),
        }

        if self._rb_single.isChecked():
            path = self._edit_single.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "No file", "Select a RGB FITS file.")
                return None
            return {**base, "mode": "single_rgb", "input_path": path}

        elif self._rb_separate.isChecked():
            r = self._edit_r.text().strip(); g = self._edit_g.text().strip()
            b = self._edit_b.text().strip()
            if not all([r, g, b]):
                QMessageBox.warning(self, "Missing channels",
                    "Select all three R/G/B FITS files.")
                return None
            return {**base, "mode": "separate_channels",
                    "r_path": r, "g_path": g, "b_path": b}

        elif self._rb_batch.isChecked():
            folder = self._edit_batch_folder.text().strip()
            if not folder or not os.path.isdir(folder):
                QMessageBox.warning(self, "No folder",
                    "Select a valid input folder.")
                return None
            return {**base, "mode": "batch_folder", "input_folder": folder}

        elif self._rb_ser.isChecked():
            path = self._edit_ser.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "No SER", "Select a SER file.")
                return None
            return {**base, "mode": "ser_sequence", "ser_path": path,
                    "max_frames": self._spin_max_frames.value() or None}

        return None

    def _run(self):
        cfg = self._get_config()
        if cfg is None: return
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._set_status("Running ADC correction…", SIRIL_ACCENT)
        self._worker = ADCWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.shift_measured.connect(self._on_shifts)
        self._worker.preview_ready.connect(self._canvas.show_before_after)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._tabs.setCurrentIndex(4)  # log

    def _cancel(self):
        self._cancel_event.set()
        self._btn_cancel.setEnabled(False)
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(msg, SIRIL_ACCENT)

    def _on_shifts(self, shifts: dict):
        dr = shifts["dr_total"]; db = shifts["db_total"]
        log_msg = (f"Applied shifts — R: {dr:.3f}px  B: {db:.3f}px  "
                   f"({shifts['method']})")
        self._set_status(log_msg, SIRIL_SUCCESS)
        self._tabs.setCurrentIndex(2)  # preview

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        if result.get("success"):
            out = result.get("output_path") or result.get("output_dir", "")
            shifts = result.get("shifts", {})
            dr = shifts.get("dr_total", 0); db = shifts.get("db_total", 0)
            self._set_status(
                f"✓ Done  R={dr:.3f}px  B={db:.3f}px  → {os.path.basename(out)}",
                SIRIL_SUCCESS)
            self._log.appendPlainText(
                f"\n✓ Complete  output: {out}")
        else:
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status.setText(msg)
        self._status.setStyleSheet(f"color:{color}; font-size:9pt;")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import traceback as _tb
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "crash_log.txt")
    def crash_handler(exc_type, exc_value, exc_tb):
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\nCRASH  {datetime.now()}\n{msg}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = crash_handler

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
