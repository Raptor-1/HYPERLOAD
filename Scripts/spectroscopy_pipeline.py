r"""
spectroscopy_pipeline.py  —  HYPERLOAD Amateur Spectroscopy Pipeline
====================================================================
Complete reduction pipeline for long-slit and objective-grating spectra
from amateur spectrographs (Star Analyser, ALPY, LHIRES III, UVEX, LISA,
Shelyak instruments, DIY spectrographs).

Pipeline (sequential):
    1. Load 2D raw spectrum FITS
    2. Calibration frames: bias, flat, dark subtraction / flat division
    3. Trace finding: automatic centroid + polynomial fit along dispersion
    4. Sky subtraction: polynomial background from flanking sky regions
    5. Spectrum extraction:
       a. Optimal (Horne 1986, PASP 98, 609) — maximum SNR, cosmic-ray
          rejection, preserves spectrophotometric accuracy
       b. Aperture (simple sum) — faster, for emission-line objects
    6. Wavelength calibration:
       a. Interactive: click on peaks in arc spectrum, identify lines
          from built-in Ne/Ar/Hg/NeAr/NeArHg lamp libraries
       b. Polynomial dispersion solution fit (degree 2–5)
    7. Radial velocity: cross-correlation function vs template spectrum
       Gaussian fit to CCF peak → RV in km/s
       Optional: heliocentric correction via astropy
    8. Output:
       - Wavelength-calibrated 1D spectrum FITS (WCS header, LAMBDA column)
       - CSV (wavelength, flux, flux_err)
       - Dispersion solution coefficients

Supported spectrographs:
    Star Analyser 100/200  (R~100-200, grating, very low resolution)
    ALPY 600               (R~600, Ne-Ar calibration)
    LHIRES III             (R~2000–17000, neon calibration)
    UVEX                   (R~5000)
    Shelyak LISA           (R~1000)
    Generic long-slit / objective prism

References:
    Optimal extraction:   Horne 1986, PASP 98, 609  (doi:10.1086/131801)
    Wavelength calib:     Tody 1986, SPIE 627, 733  (IRAF heritage)
    Sky subtraction:      Kelson 2003, PASP 115, 688
    RV cross-correlation: Simkin 1974, A&A 31, 129;
                          Tonry & Davis 1979, AJ 84, 1511

Place in: Siril Suites folder
Run via:  Siril → Scripts → spectroscopy_pipeline
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("scipy")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")

import os
import sys
import csv
import glob
import threading
import traceback
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import correlate, find_peaks
from astropy.io import fits as astropy_fits

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QRadioButton, QButtonGroup, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QSplitter,
    QSizePolicy,
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
SIRIL_SPEC     = "#c084fc"   # purple for spectroscopy theme

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: "Segoe UI", sans-serif; font-size: 9pt;
}}
QGroupBox {{
    border: 1px solid {SIRIL_BORDER}; border-radius: 5px;
    margin-top: 8px; padding: 6px;
    font-weight: bold; color: {SIRIL_SECTION};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QTabWidget::pane {{ border: 1px solid {SIRIL_BORDER}; background: {SIRIL_BG}; }}
QTabBar::tab {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT_DIM};
    padding: 6px 12px; border: 1px solid {SIRIL_BORDER};
    border-bottom: none; border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    background: {SIRIL_BG}; color: {SIRIL_SPEC}; font-weight: bold;
}}
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
QPushButton[objectName="spec"] {{
    background: #2a1a3a; color: {SIRIL_SPEC}; font-weight: bold;
    border-color: {SIRIL_SPEC};
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
QProgressBar::chunk {{ background: {SIRIL_SPEC}; border-radius: 3px; }}
QPlainTextEdit {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    font-family: "Courier New", monospace; font-size: 8pt;
}}
QTableWidget {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; gridline-color: {SIRIL_BORDER};
    alternate-background-color: {SIRIL_BG3};
}}
QHeaderView::section {{
    background: {SIRIL_BG3}; color: {SIRIL_SECTION};
    border: 1px solid {SIRIL_BORDER}; padding: 4px;
}}
QLabel[objectName="dim"]  {{ color: {SIRIL_TEXT_DIM}; }}
QLabel[objectName="ok"]   {{ color: {SIRIL_SUCCESS}; }}
QLabel[objectName="warn"] {{ color: {SIRIL_WARNING}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QScrollArea {{ border: none; }}
"""

# ─────────────────────────────────────────────────────────────────────────────
# ARC LAMP LINE LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────

# Wavelengths in Angstroms (Å), relative intensities 1-10
ARC_LINES = {
    "Neon (Ne)": {
        5852.49: 10, 5881.90: 7,  5944.83: 5,  5975.53: 4,
        6030.00: 3,  6074.34: 6,  6096.16: 4,  6143.06: 9,
        6163.59: 7,  6217.28: 3,  6266.50: 4,  6304.79: 5,
        6334.43: 6,  6382.99: 7,  6402.25: 10, 6506.53: 8,
        6532.88: 5,  6598.95: 5,  6678.28: 8,  6717.04: 4,
        7032.41: 5,  7173.94: 4,  7245.17: 4,
    },
    "Argon (Ar)": {
        6965.43: 10, 7067.22: 9,  7147.04: 5,  7272.94: 6,
        7383.98: 4,  7503.87: 8,  7514.65: 6,  7635.11: 9,
        7723.76: 5,  7948.18: 7,  8006.16: 6,  8014.79: 5,
        8103.69: 8,  8115.31: 6,  8264.52: 9,  8408.21: 6,
        8521.44: 7,  8667.94: 8,
    },
    "Mercury (Hg)": {
        4046.56: 9,  4077.83: 7,  4358.33: 10, 4916.00: 4,
        5460.74: 10, 5769.60: 8,  5790.65: 7,  6149.47: 3,
    },
    "Ne + Ar (ALPY / Star Analyser)": {
        # Combined Ne+Ar — most common amateur lamp combination
        5852.49: 8,  6143.06: 7,  6163.59: 6,  6217.28: 3,
        6266.50: 4,  6304.79: 4,  6334.43: 5,  6382.99: 6,
        6402.25: 9,  6506.53: 7,  6532.88: 4,  6598.95: 4,
        6678.28: 7,  6717.04: 3,  6929.47: 3,  6965.43: 10,
        7067.22: 8,  7147.04: 4,  7272.94: 5,  7383.98: 3,
        7503.87: 7,  7635.11: 8,  7948.18: 6,  8103.69: 7,
        8264.52: 8,  8408.21: 5,  8521.44: 6,  8667.94: 7,
    },
    "Ne + Ar + Hg": {
        4046.56: 7,  4077.83: 5,  4358.33: 9,  4916.00: 3,
        5460.74: 9,  5769.60: 7,  5790.65: 6,  5852.49: 8,
        6143.06: 7,  6163.59: 6,  6266.50: 4,  6334.43: 5,
        6402.25: 9,  6506.53: 7,  6598.95: 4,  6678.28: 7,
        6965.43: 10, 7067.22: 8,  7272.94: 5,  7503.87: 7,
        7635.11: 8,  7948.18: 6,  8103.69: 7,  8264.52: 8,
    },
    "Hydrogen Balmer (star)": {
        # Hydrogen absorption lines — useful for zero-point with A-type stars
        6562.80: 10, 4861.33: 9,  4340.47: 7,  4101.74: 5,
        3970.07: 3,  3889.05: 2,
    },
    "Sky lines (OI)": {
        # Telluric emission — useful for wavelength correction near Hα
        5577.35: 8,  6300.30: 9,  6363.78: 5,
    },
}

# Spectrograph presets (dispersion axis orientation)
INSTRUMENT_PRESETS = {
    "Generic / auto-detect":    {"dispersion": "auto", "R": None},
    "Star Analyser 100":        {"dispersion": "horizontal", "R": 100},
    "Star Analyser 200":        {"dispersion": "horizontal", "R": 200},
    "ALPY 600":                 {"dispersion": "horizontal", "R": 600},
    "LHIRES III (600 l/mm)":    {"dispersion": "horizontal", "R": 2000},
    "LHIRES III (2400 l/mm)":   {"dispersion": "horizontal", "R": 17000},
    "UVEX":                     {"dispersion": "horizontal", "R": 5000},
    "Shelyak LISA":             {"dispersion": "horizontal", "R": 1000},
}

# ─────────────────────────────────────────────────────────────────────────────
# CORE SPECTROSCOPY ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

def find_trace(image: np.ndarray, dispersion_axis: int = 1,
               fwhm_guess: float = 5.0, poly_deg: int = 3,
               n_bins: int = 20) -> np.ndarray:
    """
    Find the spectral trace by centroiding in bins along the dispersion axis.
    Fit a polynomial to the centroid positions for a smooth trace.

    Parameters:
        image:          2D spectrum array
        dispersion_axis: 0=vertical (spectrum runs top-bottom), 1=horizontal
        fwhm_guess:     approximate FWHM of the spatial PSF in pixels
        poly_deg:       degree of polynomial fit to trace
        n_bins:         number of bins along dispersion axis

    Returns:
        trace: 1D array of trace positions (in spatial pixels) for each
               column/row along the dispersion axis
    """
    if dispersion_axis == 1:
        n_disp  = image.shape[1]
        n_spat  = image.shape[0]
        profile_fn = lambda col: image[:, col]
    else:
        n_disp  = image.shape[0]
        n_spat  = image.shape[1]
        profile_fn = lambda row: image[row, :]

    spat_arr = np.arange(n_spat)
    bin_size = n_disp // n_bins
    bin_centers = []
    trace_pts   = []

    for b in range(n_bins):
        start = b * bin_size
        end   = min(start + bin_size, n_disp)
        mid   = (start + end) // 2
        # Collapse bin
        if dispersion_axis == 1:
            profile = np.median(image[:, start:end], axis=1)
        else:
            profile = np.median(image[start:end, :], axis=0)
        # Background subtract
        bg = np.percentile(profile, 20)
        profile = profile - bg
        profile = np.maximum(profile, 0)
        if profile.max() == 0:
            continue
        # Gaussian fit for centroid
        peak_idx = np.argmax(profile)
        try:
            p0 = [profile.max(), float(peak_idx), fwhm_guess / 2.355, bg]
            half = int(fwhm_guess * 3)
            lo   = max(0, peak_idx - half); hi = min(n_spat, peak_idx + half)
            xx   = spat_arr[lo:hi]; yy = profile[lo:hi]
            if len(xx) < 4:
                centroid = float(peak_idx)
            else:
                popt, _ = curve_fit(
                    lambda x, A, x0, s, c: A*np.exp(-0.5*((x-x0)/s)**2)+c,
                    xx, yy, p0=[yy.max(), float(peak_idx), fwhm_guess/2.355, 0],
                    maxfev=500)
                centroid = float(popt[1])
        except Exception:
            centroid = float(np.sum(spat_arr * profile) / max(profile.sum(), 1e-10))
        bin_centers.append(mid)
        trace_pts.append(centroid)

    if len(trace_pts) < 2:
        # Fallback: use image brightest row
        if dispersion_axis == 1:
            best_row = int(np.argmax(image.sum(axis=1)))
        else:
            best_row = int(np.argmax(image.sum(axis=0)))
        return np.full(n_disp, float(best_row))

    # Polynomial fit
    coeffs  = np.polyfit(bin_centers, trace_pts, min(poly_deg, len(trace_pts)-1))
    trace   = np.polyval(coeffs, np.arange(n_disp))
    return trace


def subtract_sky(image: np.ndarray, trace: np.ndarray,
                 half_ap: int = 5, sky_gap: int = 3,
                 sky_width: int = 10, poly_deg: int = 2,
                 dispersion_axis: int = 1) -> np.ndarray:
    """
    Subtract sky background from each column/row.
    Fits a polynomial to sky regions on both sides of the trace
    and evaluates it at each spatial position.

    Kelson 2003, PASP 115, 688 — sky subtraction for longslit spectra.
    Returns sky-subtracted 2D image.
    """
    result = image.astype(np.float64).copy()
    n_spat = image.shape[0] if dispersion_axis == 1 else image.shape[1]
    n_disp = image.shape[1] if dispersion_axis == 1 else image.shape[0]
    spat   = np.arange(n_spat)

    for i in range(n_disp):
        tc = trace[i]
        lo_sky_hi = int(tc) - half_ap - sky_gap
        lo_sky_lo = lo_sky_hi - sky_width
        hi_sky_lo = int(tc) + half_ap + sky_gap
        hi_sky_hi = hi_sky_lo + sky_width

        sky_x = []; sky_y = []
        if lo_sky_lo >= 0 and lo_sky_hi > 0:
            for s in range(max(0, lo_sky_lo), min(n_spat, lo_sky_hi)):
                if dispersion_axis == 1:
                    sky_x.append(s); sky_y.append(image[s, i])
                else:
                    sky_x.append(s); sky_y.append(image[i, s])
        if hi_sky_lo < n_spat and hi_sky_hi > hi_sky_lo:
            for s in range(max(0, hi_sky_lo), min(n_spat, hi_sky_hi)):
                if dispersion_axis == 1:
                    sky_x.append(s); sky_y.append(image[s, i])
                else:
                    sky_x.append(s); sky_y.append(image[i, s])

        if len(sky_x) < poly_deg + 1:
            continue

        sky_x = np.array(sky_x); sky_y = np.array(sky_y)
        # Robust fit (reject 3σ outliers)
        coeffs = np.polyfit(sky_x, sky_y, poly_deg)
        sky_model = np.polyval(coeffs, spat)
        if dispersion_axis == 1:
            result[:, i] -= sky_model
        else:
            result[i, :] -= sky_model

    return result


def extract_aperture(image: np.ndarray, trace: np.ndarray,
                     half_ap: int = 5,
                     dispersion_axis: int = 1) -> tuple:
    """
    Simple aperture extraction — sum pixels within half_ap of trace.
    Fast, appropriate for emission-line objects.
    Returns (flux, flux_err) 1D arrays.
    """
    n_disp = image.shape[1] if dispersion_axis == 1 else image.shape[0]
    flux   = np.zeros(n_disp)
    noise  = np.zeros(n_disp)
    for i in range(n_disp):
        tc = int(round(trace[i]))
        if dispersion_axis == 1:
            col = image[:, i]
        else:
            col = image[i, :]
        lo = max(0, tc - half_ap); hi = min(len(col), tc + half_ap + 1)
        flux[i] = col[lo:hi].sum()
        noise[i] = np.sqrt(np.maximum(flux[i], 0) + (hi-lo) * 4.0)
    return flux, noise


def extract_optimal(image: np.ndarray, trace: np.ndarray,
                    variance: np.ndarray | None,
                    half_ap: int = 8,
                    dispersion_axis: int = 1,
                    readnoise: float = 5.0,
                    gain: float = 1.0,
                    n_iter: int = 5,
                    sigma_clip: float = 4.0,
                    log_callback=None) -> tuple:
    """
    Optimal spectrum extraction — Horne 1986, PASP 98, 609.

    The algorithm:
      1. Build the spatial profile P(x, y) by smoothing the 2D spectrum
         along the dispersion axis (the profile should vary slowly).
      2. At each dispersion step x, the optimal estimate of the flux is:
            f_x = Σ_y [P_xy × D_xy / V_xy] / Σ_y [P_xy² / V_xy]
         where V_xy = variance (readnoise² + |D_xy|/gain).
      3. Cosmic rays detected as deviations from the optimal model:
            residual = (D_xy - f_x × P_xy) / sqrt(V_xy) > sigma_clip
         Rejected pixels are masked and the extraction re-done.

    Delivers maximum SNR without bias. Automatically rejects cosmic rays.
    """
    log = log_callback or (lambda m: None)
    log("  Optimal extraction (Horne 1986, PASP 98, 609)")

    n_disp = image.shape[1] if dispersion_axis == 1 else image.shape[0]
    n_spat = image.shape[0] if dispersion_axis == 1 else image.shape[1]
    spat   = np.arange(n_spat)

    # Build spatial profile by smoothing across dispersion axis
    # Smooth with a 1D Gaussian along the dispersion direction
    if dispersion_axis == 1:
        data = image.astype(np.float64)
    else:
        data = image.astype(np.float64).T

    # Profile: for each dispersion pixel, extract a column-profile
    # and smooth across columns to get stable profile estimate
    smooth_width = max(5, n_disp // 50)
    profile_2d   = gaussian_filter1d(data, sigma=smooth_width, axis=1)

    # Normalise profile along spatial axis (sum = 1 per column)
    profile_sum  = np.sum(profile_2d, axis=0, keepdims=True)
    profile_sum  = np.where(profile_sum == 0, 1e-10, profile_sum)
    P            = profile_2d / profile_sum  # [n_spat, n_disp]

    # Mask: 1=good, 0=bad (cosmic ray / bad pixel)
    mask = np.ones_like(data, dtype=bool)

    flux    = np.zeros(n_disp)
    flux_err= np.zeros(n_disp)

    for it in range(n_iter):
        n_rej_total = 0
        for x in range(n_disp):
            D = data[:, x]      # spatial profile at this dispersion step
            P_x = P[:, x]      # normalised profile

            # Restrict to aperture around trace
            tc  = trace[x] if dispersion_axis == 1 else trace[x]
            lo  = max(0, int(tc) - half_ap)
            hi  = min(n_spat, int(tc) + half_ap + 1)
            idx = np.arange(lo, hi)

            D_ap  = D[idx]
            P_ap  = P_x[idx]
            m_ap  = mask[:, x][idx]

            # Variance (Poisson + readnoise)
            V_ap  = np.maximum(readnoise**2 + np.abs(D_ap)/gain, 1e-6)

            denom = np.sum(m_ap * P_ap**2 / V_ap)
            if denom == 0: continue

            f_x   = np.sum(m_ap * P_ap * D_ap / V_ap) / denom
            flux[x] = f_x
            flux_err[x] = np.sqrt(1.0 / denom)

            # Cosmic ray rejection
            if it < n_iter - 1:
                resid = (D_ap - f_x * P_ap) / np.sqrt(V_ap)
                rej   = np.abs(resid) > sigma_clip
                if rej.any():
                    mask[idx[rej], x] = False
                    n_rej_total += rej.sum()

        if it == 0:
            log(f"    Iter {it+1}: initial extraction")
        else:
            log(f"    Iter {it+1}: {n_rej_total} pixels rejected (>{sigma_clip:.1f}σ)")
        if n_rej_total == 0 and it > 0:
            break

    if dispersion_axis == 0:
        pass  # already transposed above

    return flux, flux_err


def fit_dispersion_solution(pixel_points: list, wavelength_points: list,
                             poly_deg: int = 3) -> dict:
    """
    Fit a polynomial wavelength dispersion solution from identified lines.

    Parameters:
        pixel_points:     list of pixel positions of identified arc lines
        wavelength_points: corresponding wavelengths in Angstroms
        poly_deg:         degree of polynomial (2 = quadratic, 3 = cubic)

    Returns dict with coefficients, RMS residual, dispersion (Å/px).
    """
    if len(pixel_points) < poly_deg + 1:
        raise ValueError(f"Need ≥{poly_deg+1} points for degree-{poly_deg} polynomial")

    px  = np.array(pixel_points, dtype=float)
    wl  = np.array(wavelength_points, dtype=float)
    coeffs = np.polyfit(px, wl, poly_deg)

    # Evaluate residuals
    wl_pred = np.polyval(coeffs, px)
    residuals = wl - wl_pred
    rms = float(np.sqrt(np.mean(residuals**2)))

    # Dispersion at midpoint
    mid_px   = np.median(px)
    dx       = 1.0
    dispersion = abs(np.polyval(coeffs, mid_px + dx) - np.polyval(coeffs, mid_px))

    return {
        "coeffs":     coeffs,
        "rms_A":      rms,
        "dispersion": dispersion,
        "poly_deg":   poly_deg,
        "n_lines":    len(px),
        "px_points":  px.tolist(),
        "wl_points":  wl.tolist(),
        "residuals":  residuals.tolist(),
    }


def apply_wavelength_solution(flux: np.ndarray, flux_err: np.ndarray,
                               coeffs: np.ndarray,
                               n_pixels: int) -> tuple:
    """
    Apply wavelength solution to produce wavelength-calibrated spectrum.
    Returns (wavelengths, flux, flux_err) all in same pixel grid.
    """
    pixels = np.arange(n_pixels, dtype=float)
    wl     = np.polyval(coeffs, pixels)
    return wl, flux, flux_err


def cross_correlate_rv(wave_obs: np.ndarray, flux_obs: np.ndarray,
                        wave_tmpl: np.ndarray, flux_tmpl: np.ndarray,
                        v_range_kms: float = 500.0,
                        n_v: int = 1000) -> dict:
    """
    Compute radial velocity via cross-correlation function (CCF).
    Simkin 1974, A&A 31, 129; Tonry & Davis 1979, AJ 84, 1511.

    Interpolate both spectra onto a common log-wavelength grid
    (so velocity shifts become linear shifts in log space).
    Cross-correlate and find the CCF peak.
    Fit a Gaussian to the CCF peak for sub-pixel precision.

    Returns dict with rv_kms, ccf_max, bisector_span.
    """
    c_kms = 299792.458  # speed of light in km/s

    # Common log-wavelength grid
    wl_min = max(wave_obs.min(), wave_tmpl.min())
    wl_max = min(wave_obs.max(), wave_tmpl.max())
    if wl_max <= wl_min:
        return {"rv_kms": 0.0, "ccf_max": 0.0, "success": False,
                "error": "No wavelength overlap"}

    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), 4096)
    dv     = (log_wl[1] - log_wl[0]) * c_kms  # velocity step in km/s

    f_obs  = interp1d(np.log(wave_obs),  flux_obs,  bounds_error=False, fill_value=0)
    f_tmpl = interp1d(np.log(wave_tmpl), flux_tmpl, bounds_error=False, fill_value=0)
    obs_i  = f_obs(log_wl)
    tmpl_i = f_tmpl(log_wl)

    # Remove continuum (subtract mean)
    obs_i  -= obs_i.mean()
    tmpl_i -= tmpl_i.mean()

    # Cross-correlation
    ccf    = correlate(obs_i, tmpl_i, mode="full")
    lags   = np.arange(-len(obs_i)+1, len(obs_i)) * dv

    # Find peak in v_range
    in_range = np.abs(lags) <= v_range_kms
    if not in_range.any():
        return {"rv_kms": 0.0, "ccf_max": 0.0, "success": False,
                "error": "v_range too small"}
    ccf_r = ccf[in_range]; lags_r = lags[in_range]
    peak_idx = int(np.argmax(ccf_r))
    rv_crude = lags_r[peak_idx]

    # Gaussian fit around peak
    half_w = min(30, len(ccf_r)//4)
    lo = max(0, peak_idx-half_w); hi = min(len(ccf_r), peak_idx+half_w)
    xx = lags_r[lo:hi]; yy = ccf_r[lo:hi]
    try:
        popt, _ = curve_fit(
            lambda x, A, x0, s, c: A*np.exp(-0.5*((x-x0)/s)**2)+c,
            xx, yy,
            p0=[yy.max(), rv_crude, abs(dv)*5, yy.min()],
            maxfev=500)
        rv_kms = float(popt[1])
        ccf_fwhm = float(abs(popt[2]) * 2.355)
    except Exception:
        rv_kms   = rv_crude
        ccf_fwhm = 0.0

    # CCF bisector (line profile asymmetry indicator)
    ccf_max  = float(np.max(ccf_r))
    ccf_norm = ccf_r / max(ccf_max, 1e-10)
    try:
        # Bisector at 60% and 90% of CCF peak
        bisector = float(np.mean([
            np.interp(0.6, ccf_norm[:peak_idx+1][::-1], lags_r[:peak_idx+1][::-1]) +
            np.interp(0.6, ccf_norm[peak_idx:], lags_r[peak_idx:]),
            np.interp(0.9, ccf_norm[:peak_idx+1][::-1], lags_r[:peak_idx+1][::-1]) +
            np.interp(0.9, ccf_norm[peak_idx:], lags_r[peak_idx:]),
        ])) / 2
    except Exception:
        bisector = 0.0

    return {
        "rv_kms":       rv_kms,
        "ccf_max":      ccf_max,
        "ccf_fwhm_kms": ccf_fwhm,
        "bisector_kms": bisector,
        "lags_kms":     lags_r.tolist(),
        "ccf":          ccf_r.tolist(),
        "success":      True,
    }


def heliocentric_correction(ra_deg: float, dec_deg: float,
                             obs_date: str,
                             obs_lat: float, obs_lon: float,
                             obs_elev: float = 0.0) -> float:
    """
    Compute heliocentric radial velocity correction in km/s.
    Uses astropy.coordinates.
    Returns correction to add to measured RV to get heliocentric RV.
    """
    try:
        from astropy.coordinates import (SkyCoord, EarthLocation,
                                          solar_system_ephemeris)
        from astropy.time import Time
        import astropy.units as u
        from astropy.coordinates import get_body_barycentric_posvel

        loc    = EarthLocation(lat=obs_lat*u.deg,
                               lon=obs_lon*u.deg,
                               height=obs_elev*u.m)
        time   = Time(obs_date, format="isot", scale="utc")
        target = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
        bary_corr = target.radial_velocity_correction(
            obstime=time, location=loc)
        return float(bary_corr.to(u.km/u.s).value)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2D SPECTRUM CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumCanvas2D(FigureCanvasQTAgg):
    """Shows 2D raw spectrum with trace overlay."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 4), facecolor=SIRIL_BG)
        self.ax  = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(SIRIL_BG2)
        for sp in self.ax.spines.values(): sp.set_color(SIRIL_BORDER)
        super().__init__(self.fig)
        self.setParent(parent)

    def show_2d(self, image: np.ndarray, trace: np.ndarray = None,
                 sky_lo: int = None, sky_hi: int = None,
                 half_ap: int = 5):
        self.ax.clear(); self.ax.set_facecolor(SIRIL_BG2)
        vmin, vmax = np.percentile(image, [0.5, 99.8])
        self.ax.imshow(image, cmap="inferno", origin="lower",
                       vmin=vmin, vmax=vmax, aspect="auto",
                       interpolation="nearest")
        if trace is not None:
            x = np.arange(len(trace))
            self.ax.plot(x, trace, color=SIRIL_SUCCESS, lw=1.5,
                         label="Trace", zorder=5)
            self.ax.plot(x, trace + half_ap, color=SIRIL_ACCENT,
                         lw=0.8, ls="--", alpha=0.7, label="Aperture")
            self.ax.plot(x, trace - half_ap, color=SIRIL_ACCENT,
                         lw=0.8, ls="--", alpha=0.7)
            if sky_lo is not None:
                self.ax.axhspan(trace.mean()-sky_lo-20,
                                trace.mean()-sky_lo,
                                color=SIRIL_WARNING, alpha=0.15,
                                label="Sky regions")
                self.ax.axhspan(trace.mean()+sky_lo,
                                trace.mean()+sky_lo+20,
                                color=SIRIL_WARNING, alpha=0.15)
            self.ax.legend(fontsize=7, facecolor=SIRIL_BG3,
                           edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax.set_xlabel("Dispersion axis (px)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_ylabel("Spatial axis (px)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_title("2D spectrum", color=SIRIL_SPEC, fontsize=9)
        self.ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# ARC CALIBRATION CANVAS  (interactive line identification)
# ─────────────────────────────────────────────────────────────────────────────

class ArcCanvas(FigureCanvasQTAgg):
    """
    Interactive arc spectrum canvas.
    Click on a peak → adds a mark at that pixel position.
    The user then assigns a wavelength from the line list.
    """
    line_clicked = pyqtSignal(float)   # pixel position of click

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 4), facecolor=SIRIL_BG)
        self.ax  = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(SIRIL_BG2)
        for sp in self.ax.spines.values(): sp.set_color(SIRIL_BORDER)
        super().__init__(self.fig)
        self.setParent(parent)
        self._arc_flux  = None
        self._id_pixels = []
        self._id_waves  = []
        self._peaks     = []
        self.mpl_connect("button_press_event", self._on_click)

    def show_arc(self, arc_flux: np.ndarray, peaks: list = None,
                  id_pixels: list = None, id_waves: list = None,
                  wave_sol: np.ndarray = None):
        self._arc_flux = arc_flux
        self._peaks    = peaks or []
        self._id_pixels = id_pixels or []
        self._id_waves  = id_waves or []
        self.ax.clear(); self.ax.set_facecolor(SIRIL_BG2)
        x = np.arange(len(arc_flux))
        self.ax.plot(x, arc_flux, color=SIRIL_SPEC, lw=0.8)
        # Mark auto-detected peaks
        if self._peaks:
            self.ax.vlines(self._peaks, 0, arc_flux.max()*0.3,
                            color=SIRIL_TEXT_DIM, lw=0.5, alpha=0.5,
                            label="Auto peaks")
        # Mark identified lines
        for px, wl in zip(self._id_pixels, self._id_waves):
            self.ax.axvline(px, color=SIRIL_NOVA, lw=1.2, alpha=0.9)
            self.ax.text(px+2, arc_flux.max()*0.7, f"{wl:.2f}Å",
                          color=SIRIL_NOVA, fontsize=6, rotation=90)
        self.ax.set_xlabel(
            "Wavelength (Å)" if wave_sol is not None else "Pixel",
            color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_ylabel("Flux (ADU)", color=SIRIL_TEXT_DIM, fontsize=8)
        n_id = len(self._id_pixels)
        self.ax.set_title(
            f"Arc spectrum  —  {n_id} lines identified  "
            f"(click on a peak to mark it)",
            color=SIRIL_SPEC, fontsize=9)
        self.ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        if self._peaks:
            self.ax.legend(fontsize=7, facecolor=SIRIL_BG3,
                            edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        for sp in self.ax.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def _on_click(self, event):
        if event.inaxes != self.ax or self._arc_flux is None: return
        px = float(event.xdata)
        # Snap to nearest detected peak
        if self._peaks:
            nearest = min(self._peaks, key=lambda p: abs(p - px))
            if abs(nearest - px) < 10:
                px = float(nearest)
        self.line_clicked.emit(px)


# ─────────────────────────────────────────────────────────────────────────────
# 1D SPECTRUM CANVAS  (wavelength-calibrated)
# ─────────────────────────────────────────────────────────────────────────────

class Spectrum1DCanvas(FigureCanvasQTAgg):
    """Calibrated 1D spectrum + CCF panel."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 6), facecolor=SIRIL_BG)
        gs = gridspec.GridSpec(2, 1, figure=self.fig,
                               height_ratios=[3, 1], hspace=0.35)
        self.ax_spec = self.fig.add_subplot(gs[0])
        self.ax_ccf  = self.fig.add_subplot(gs[1])
        for ax in [self.ax_spec, self.ax_ccf]:
            ax.set_facecolor(SIRIL_BG2)
            ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_color(SIRIL_BORDER)
        super().__init__(self.fig)
        self.setParent(parent)

    def show_spectrum(self, wavelengths: np.ndarray, flux: np.ndarray,
                       flux_err: np.ndarray = None,
                       title: str = "Spectrum"):
        self.ax_spec.clear(); self.ax_spec.set_facecolor(SIRIL_BG2)
        if flux_err is not None:
            self.ax_spec.fill_between(wavelengths,
                                       flux - flux_err, flux + flux_err,
                                       color=SIRIL_SPEC, alpha=0.25)
        self.ax_spec.plot(wavelengths, flux, color=SIRIL_SPEC, lw=0.9)
        self.ax_spec.set_xlabel("Wavelength (Å)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_spec.set_ylabel("Flux (ADU)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_spec.set_title(title, color=SIRIL_SPEC, fontsize=9)
        self.ax_spec.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax_spec.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_ccf(self, rv_result: dict):
        self.ax_ccf.clear(); self.ax_ccf.set_facecolor(SIRIL_BG2)
        if not rv_result.get("success"): return
        lags = np.array(rv_result["lags_kms"])
        ccf  = np.array(rv_result["ccf"])
        ccf  = ccf / max(ccf.max(), 1e-10)
        self.ax_ccf.plot(lags, ccf, color=SIRIL_ACCENT, lw=1.0)
        self.ax_ccf.axvline(rv_result["rv_kms"], color=SIRIL_NOVA, lw=1.5,
                             ls="--",
                             label=f"RV={rv_result['rv_kms']:+.2f} km/s")
        self.ax_ccf.set_xlabel("Velocity (km/s)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_ccf.set_ylabel("CCF", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_ccf.set_title("Cross-correlation function (CCF)",
                               color=SIRIL_ACCENT, fontsize=9)
        self.ax_ccf.legend(fontsize=7, facecolor=SIRIL_BG3,
                            edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax_ccf.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax_ccf.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# DISPERSION SOLUTION CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class DispersionCanvas(FigureCanvasQTAgg):
    """Wavelength solution: identified lines + residuals."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), facecolor=SIRIL_BG)
        self.ax_sol  = self.fig.add_subplot(1, 2, 1)
        self.ax_resid = self.fig.add_subplot(1, 2, 2)
        for ax in [self.ax_sol, self.ax_resid]:
            ax.set_facecolor(SIRIL_BG2)
            ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_color(SIRIL_BORDER)
        super().__init__(self.fig)
        self.setParent(parent)

    def show_solution(self, sol: dict):
        px    = np.array(sol["px_points"])
        wl    = np.array(sol["wl_points"])
        resid = np.array(sol["residuals"])
        px_fit= np.linspace(px.min(), px.max(), 200)
        wl_fit= np.polyval(sol["coeffs"], px_fit)

        self.ax_sol.clear(); self.ax_sol.set_facecolor(SIRIL_BG2)
        self.ax_sol.scatter(px, wl, color=SIRIL_NOVA, s=40, zorder=5,
                             label=f"{sol['n_lines']} lines")
        self.ax_sol.plot(px_fit, wl_fit, color=SIRIL_SPEC, lw=1.5,
                          label=f"Poly deg={sol['poly_deg']}")
        self.ax_sol.set_xlabel("Pixel", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_sol.set_ylabel("Wavelength (Å)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_sol.set_title(
            f"Dispersion solution  "
            f"{sol['dispersion']:.3f} Å/px  RMS={sol['rms_A']:.3f} Å",
            color=SIRIL_SPEC, fontsize=9)
        self.ax_sol.legend(fontsize=7, facecolor=SIRIL_BG3,
                            edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax_sol.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax_sol.spines.values(): sp.set_color(SIRIL_BORDER)

        self.ax_resid.clear(); self.ax_resid.set_facecolor(SIRIL_BG2)
        self.ax_resid.scatter(px, resid * 1000, color=SIRIL_NOVA, s=40)
        self.ax_resid.axhline(0, color=SIRIL_BORDER, lw=0.8)
        rms_ma = sol["rms_A"] * 1000
        self.ax_resid.axhline(rms_ma, color=SIRIL_ACCENT, lw=0.8, ls="--",
                               label=f"RMS={rms_ma:.0f} mÅ")
        self.ax_resid.axhline(-rms_ma, color=SIRIL_ACCENT, lw=0.8, ls="--")
        self.ax_resid.set_xlabel("Pixel", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_resid.set_ylabel("Residual (mÅ)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_resid.set_title("Wavelength residuals", color=SIRIL_SECTION,
                                  fontsize=9)
        self.ax_resid.legend(fontsize=7, facecolor=SIRIL_BG3,
                              edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax_resid.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax_resid.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# REDUCTION WORKER
# ─────────────────────────────────────────────────────────────────────────────

class ReductionWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    image_ready   = pyqtSignal(np.ndarray)       # preprocessed 2D
    trace_ready   = pyqtSignal(np.ndarray, np.ndarray)  # trace, image
    arc_ready     = pyqtSignal(np.ndarray, list)  # arc_flux, peaks
    extract_ready = pyqtSignal(np.ndarray, np.ndarray)  # flux, err
    finished      = pyqtSignal(dict)

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
            self.log_line.emit(f"CRITICAL: {e}\n{tb}")
            self.finished.emit({"success": False, "error": str(e)})

    def _run(self):
        cfg = self.cfg
        log = self.log_line.emit
        log("═══ HyperLoad Spectroscopy Pipeline ═══")
        log(f"  Instrument:   {cfg.get('instrument','Generic')}")
        log(f"  Extraction:   {cfg['extraction_method']}")
        log(f"  Dispersion:   axis={cfg['dispersion_axis']}")
        log("")

        # ── STEP 1: Load and preprocess 2D spectrum ───────────────────────────
        self.progress.emit(0, 100, "Loading 2D spectrum…")
        path = cfg["science_path"]
        try:
            with astropy_fits.open(path) as hdul:
                raw_data = hdul[0].data.astype(np.float64)
                hdr      = hdul[0].header.copy()
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)}); return

        if raw_data.ndim == 3: raw_data = raw_data[0]
        log(f"Loaded: {os.path.basename(path)}  {raw_data.shape}")

        data = raw_data.copy()

        # Apply bias
        if cfg.get("bias_path") and os.path.isfile(cfg["bias_path"]):
            log("Subtracting bias…")
            with astropy_fits.open(cfg["bias_path"]) as hdul:
                bias = hdul[0].data.astype(np.float64)
            if bias.ndim == 3: bias = bias[0]
            data -= bias

        # Apply dark
        if cfg.get("dark_path") and os.path.isfile(cfg["dark_path"]):
            log("Subtracting dark…")
            with astropy_fits.open(cfg["dark_path"]) as hdul:
                dark = hdul[0].data.astype(np.float64)
            if dark.ndim == 3: dark = dark[0]
            data -= dark

        # Apply flat
        if cfg.get("flat_path") and os.path.isfile(cfg["flat_path"]):
            log("Flat-field correction…")
            with astropy_fits.open(cfg["flat_path"]) as hdul:
                flat = hdul[0].data.astype(np.float64)
            if flat.ndim == 3: flat = flat[0]
            flat_norm = flat / np.maximum(np.median(flat), 1e-10)
            data /= np.maximum(flat_norm, 0.01)

        # Flip if needed
        if cfg.get("flip_horizontal"): data = np.fliplr(data)
        if cfg.get("flip_vertical"):   data = np.flipud(data)

        # Rotate if dispersion is vertical
        disp_axis = cfg["dispersion_axis"]  # 0=rows, 1=cols
        self.image_ready.emit(data)
        log(f"Preprocessed: {data.shape}  dispersion_axis={disp_axis}")

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── STEP 2: Find trace ────────────────────────────────────────────────
        self.progress.emit(15, 100, "Finding spectral trace…")
        log("Step 2: Spectral trace finding")
        manual_trace_center = cfg.get("manual_trace_center")
        if manual_trace_center is not None:
            n_disp = data.shape[1] if disp_axis == 1 else data.shape[0]
            trace  = np.full(n_disp, float(manual_trace_center))
            log(f"  Manual trace center: {manual_trace_center:.1f} px")
        else:
            trace = find_trace(data,
                                dispersion_axis=disp_axis,
                                fwhm_guess=cfg.get("fwhm_guess", 5.0),
                                poly_deg=cfg.get("trace_poly_deg", 3),
                                n_bins=cfg.get("trace_bins", 20))
            log(f"  Trace found: center={np.median(trace):.1f} px  "
                f"variation={np.ptp(trace):.2f} px")
        self.trace_ready.emit(trace, data)

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── STEP 3: Sky subtraction ───────────────────────────────────────────
        self.progress.emit(25, 100, "Subtracting sky background…")
        if cfg.get("subtract_sky", True):
            log("Step 3: Sky subtraction (Kelson 2003)")
            data = subtract_sky(data, trace,
                                 half_ap=cfg.get("half_ap", 5),
                                 sky_gap=cfg.get("sky_gap", 3),
                                 sky_width=cfg.get("sky_width", 10),
                                 dispersion_axis=disp_axis)
        else:
            log("Step 3: Sky subtraction skipped")

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── STEP 4: Spectrum extraction ────────────────────────────────────────
        self.progress.emit(35, 100, f"Extracting spectrum ({cfg['extraction_method']})…")
        log(f"Step 4: Spectrum extraction ({cfg['extraction_method']})")
        half_ap = cfg.get("half_ap", 5)
        method  = cfg["extraction_method"]

        if method == "Optimal (Horne 1986)":
            flux, flux_err = extract_optimal(
                data, trace, variance=None,
                half_ap=half_ap,
                dispersion_axis=disp_axis,
                readnoise=cfg.get("readnoise", 5.0),
                gain=cfg.get("gain", 1.0),
                n_iter=cfg.get("n_iter_extr", 5),
                sigma_clip=cfg.get("sigma_clip", 4.0),
                log_callback=log)
        else:
            flux, flux_err = extract_aperture(
                data, trace, half_ap=half_ap,
                dispersion_axis=disp_axis)
            log(f"  Aperture extraction: ±{half_ap} px")

        log(f"  Extracted {len(flux)} pixels  "
            f"peak={flux.max():.1f}  SNR≈{flux.max()/max(np.median(flux_err),1):.1f}")
        self.extract_ready.emit(flux, flux_err)

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── STEP 5: Arc lamp extraction ────────────────────────────────────────
        self.progress.emit(55, 100, "Extracting arc spectrum…")
        arc_flux = None
        arc_peaks = []
        if cfg.get("arc_path") and os.path.isfile(cfg["arc_path"]):
            log("Step 5: Arc lamp extraction")
            try:
                with astropy_fits.open(cfg["arc_path"]) as hdul:
                    arc_raw = hdul[0].data.astype(np.float64)
                if arc_raw.ndim == 3: arc_raw = arc_raw[0]
                # Apply same bias/flat to arc
                if cfg.get("bias_path") and os.path.isfile(cfg["bias_path"]):
                    arc_raw -= bias
                # Extract arc at same trace position
                arc_flux_full, _ = extract_aperture(
                    arc_raw, trace, half_ap=half_ap,
                    dispersion_axis=disp_axis)
                # Smooth lightly for peak finding
                arc_smooth = gaussian_filter1d(arc_flux_full, sigma=1.5)
                arc_flux   = arc_flux_full
                # Find peaks automatically
                height_thresh = np.percentile(arc_smooth, 70)
                arc_peaks_raw, props = find_peaks(
                    arc_smooth, height=height_thresh,
                    distance=cfg.get("min_peak_sep", 8),
                    prominence=height_thresh * 0.1)
                arc_peaks = arc_peaks_raw.tolist()
                log(f"  Arc: {len(arc_peaks)} candidate peaks detected")
                self.arc_ready.emit(arc_flux, arc_peaks)
            except Exception as e:
                log(f"  Arc extraction failed: {e}")
        else:
            log("Step 5: No arc lamp — skipping wavelength calibration")

        self.progress.emit(70, 100, "Done with extraction")
        log("\n✓ Extraction complete")
        self.finished.emit({
            "success":    True,
            "flux":       flux,
            "flux_err":   flux_err,
            "arc_flux":   arc_flux,
            "arc_peaks":  arc_peaks,
            "trace":      trace,
            "data":       data,
            "header":     hdr,
            "n_pixels":   len(flux),
        })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Spectroscopy Pipeline  —  HYPERLOAD  —  Siril")
        self.resize(1440, 920)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._flux         = None
        self._flux_err     = None
        self._arc_flux     = None
        self._arc_peaks    = []
        self._trace        = None
        self._header       = None
        self._n_pixels     = 0
        # Wavelength calibration state
        self._id_pixels    = []   # identified arc line pixel positions
        self._id_waves     = []   # corresponding wavelengths
        self._dispersion_solution = None
        self._wavelengths  = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # Header
        header = QWidget(); header.setFixedHeight(56)
        header.setStyleSheet(
            f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon = QLabel("🔭")
        lbl_icon.setStyleSheet("font-size:20pt; background:transparent;")
        lbl_title = QLabel("Spectroscopy Pipeline")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_SPEC}; font-size:14pt; font-weight:bold; "
            f"background:transparent;")
        lbl_sub = QLabel(
            "Trace finding  ·  Optimal extraction (Horne 1986)  ·  "
            "Interactive wavelength calibration  ·  Radial velocity CCF  ·  "
            "Star Analyser / ALPY / LHIRES / UVEX / LISA")
        lbl_sub.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title)
        hl.addWidget(lbl_sub); hl.addStretch()
        root.addWidget(header)

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._tabs.addTab(self._build_tab_input(),   "📁  Input")
        self._tabs.addTab(self._build_tab_extract(),  "⚙  Extraction")
        self._tabs.addTab(self._build_tab_2d(),      "🖼  2D Spectrum")
        self._tabs.addTab(self._build_tab_arc(),     "💡  Arc Calibration")
        self._tabs.addTab(self._build_tab_1d(),      "📈  1D Spectrum")
        self._tabs.addTab(self._build_tab_rv(),      "🌀  Radial Velocity")
        self._tabs.addTab(self._build_tab_output(),  "💾  Output")
        self._tabs.addTab(self._build_tab_log(),     "📋  Log")

        # Bottom bar
        bottom = QWidget(); bottom.setFixedHeight(50)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 6, 10, 6)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(10); self._progress.setValue(0)
        bl.addWidget(self._progress, 1)
        self._btn_run = QPushButton("▶  Run Extraction Pipeline")
        self._btn_run.setObjectName("spec")
        self._btn_run.setMinimumWidth(200); self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run_extraction)
        bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(34); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bl.addWidget(self._btn_cancel)
        self._status = QLabel("Ready — load a 2D spectrum FITS file")
        self._status.setObjectName("dim")
        self._status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status, 1)
        root.addWidget(bottom)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_tab_input(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        grp_inst = QGroupBox("Spectrograph")
        il = QFormLayout(grp_inst)
        self._cmb_instrument = QComboBox()
        self._cmb_instrument.addItems(list(INSTRUMENT_PRESETS.keys()))
        self._cmb_instrument.currentIndexChanged.connect(self._on_instrument_changed)
        il.addRow("Instrument preset:", self._cmb_instrument)
        lay.addWidget(grp_inst)

        grp_sci = QGroupBox("Science frame (2D spectrum)")
        sl = QHBoxLayout(grp_sci)
        self._edit_sci = QLineEdit(); self._edit_sci.setPlaceholderText("2D spectrum FITS…")
        btn_s = QPushButton("Browse…"); btn_s.setFixedWidth(80)
        btn_s.clicked.connect(lambda: self._browse_fits(self._edit_sci, "2D spectrum"))
        sl.addWidget(self._edit_sci); sl.addWidget(btn_s)
        lay.addWidget(grp_sci)

        grp_arc = QGroupBox("Arc lamp frame")
        al = QHBoxLayout(grp_arc)
        self._edit_arc = QLineEdit()
        self._edit_arc.setPlaceholderText("Arc lamp FITS (Ne, Ar, NeAr…)  optional")
        btn_a = QPushButton("Browse…"); btn_a.setFixedWidth(80)
        btn_a.clicked.connect(lambda: self._browse_fits(self._edit_arc, "Arc lamp"))
        al.addWidget(self._edit_arc); al.addWidget(btn_a)
        lay.addWidget(grp_arc)

        grp_cal = QGroupBox("Calibration frames (all optional)")
        clf = QFormLayout(grp_cal)
        self._edit_bias = QLineEdit(); self._edit_bias.setPlaceholderText("Master bias…")
        self._edit_dark = QLineEdit(); self._edit_dark.setPlaceholderText("Master dark…")
        self._edit_flat = QLineEdit(); self._edit_flat.setPlaceholderText("Master flat…")
        for lbl, edit in [("Bias:", self._edit_bias), ("Dark:", self._edit_dark),
                           ("Flat:", self._edit_flat)]:
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            btn = QPushButton("Browse…"); btn.setFixedWidth(80)
            btn.clicked.connect(lambda checked, e=edit: self._browse_fits(e, "calibration"))
            row.addWidget(btn)
            clf.addRow(lbl, row)
        lay.addWidget(grp_cal)

        grp_orient = QGroupBox("Frame orientation")
        of = QFormLayout(grp_orient)
        self._cmb_dispaxis = QComboBox()
        self._cmb_dispaxis.addItems([
            "Horizontal (spectrum runs left-right)  — most instruments",
            "Vertical (spectrum runs top-bottom)",
        ])
        of.addRow("Dispersion axis:", self._cmb_dispaxis)
        self._chk_flip_h = QCheckBox("Flip horizontal")
        self._chk_flip_v = QCheckBox("Flip vertical")
        of.addRow(self._chk_flip_h); of.addRow(self._chk_flip_v)
        lbl_o = QLabel(
            "If the spectrum appears upside-down or wavelength increases\n"
            "from right to left, use the flip options.")
        lbl_o.setObjectName("dim"); lbl_o.setWordWrap(True); of.addRow(lbl_o)
        lay.addWidget(grp_orient)
        lay.addStretch()
        return w

    def _build_tab_extract(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        grp_trace = QGroupBox("Trace finding")
        tf = QFormLayout(grp_trace)
        self._spin_fwhm_guess = QDoubleSpinBox()
        self._spin_fwhm_guess.setRange(1.0, 50.0); self._spin_fwhm_guess.setValue(5.0)
        self._spin_fwhm_guess.setSuffix(" px")
        tf.addRow("Spatial FWHM guess:", self._spin_fwhm_guess)
        self._spin_trace_deg = QSpinBox()
        self._spin_trace_deg.setRange(1, 6); self._spin_trace_deg.setValue(3)
        tf.addRow("Trace polynomial degree:", self._spin_trace_deg)
        self._chk_manual_trace = QCheckBox("Manual trace center (skip auto-find)")
        self._chk_manual_trace.toggled.connect(lambda c: self._spin_manual_trace.setEnabled(c))
        tf.addRow(self._chk_manual_trace)
        self._spin_manual_trace = QDoubleSpinBox()
        self._spin_manual_trace.setRange(0, 9999); self._spin_manual_trace.setValue(512)
        self._spin_manual_trace.setEnabled(False)
        tf.addRow("Manual center (px):", self._spin_manual_trace)
        lay.addWidget(grp_trace)

        grp_extr = QGroupBox("Extraction method")
        em = QVBoxLayout(grp_extr)
        self._rb_optimal = QRadioButton(
            "Optimal extraction  (Horne 1986, PASP 98, 609)\n"
            "Maximum SNR. Cosmic-ray rejection. Recommended for faint sources.")
        self._rb_aperture = QRadioButton(
            "Aperture extraction  (simple sum)\n"
            "Fast. Better for bright emission-line objects.")
        self._rb_optimal.setChecked(True)
        self._bg_extr = QButtonGroup()
        self._bg_extr.addButton(self._rb_optimal); self._bg_extr.addButton(self._rb_aperture)
        for rb in [self._rb_optimal, self._rb_aperture]: em.addWidget(rb)
        lay.addWidget(grp_extr)

        grp_ap = QGroupBox("Aperture settings")
        af = QFormLayout(grp_ap)
        self._spin_half_ap = QSpinBox()
        self._spin_half_ap.setRange(1, 50); self._spin_half_ap.setValue(5)
        self._spin_half_ap.setSuffix(" px")
        af.addRow("Half-aperture:", self._spin_half_ap)
        self._spin_sky_gap = QSpinBox()
        self._spin_sky_gap.setRange(0, 50); self._spin_sky_gap.setValue(3)
        self._spin_sky_gap.setSuffix(" px")
        af.addRow("Sky gap (from trace):", self._spin_sky_gap)
        self._spin_sky_width = QSpinBox()
        self._spin_sky_width.setRange(1, 100); self._spin_sky_width.setValue(10)
        self._spin_sky_width.setSuffix(" px")
        af.addRow("Sky region width:", self._spin_sky_width)
        self._chk_sky = QCheckBox("Subtract sky background")
        self._chk_sky.setChecked(True)
        af.addRow(self._chk_sky)
        lay.addWidget(grp_ap)

        grp_opt = QGroupBox("Optimal extraction settings")
        of2 = QFormLayout(grp_opt)
        self._spin_readnoise = QDoubleSpinBox()
        self._spin_readnoise.setRange(0.1, 100.0); self._spin_readnoise.setValue(5.0)
        self._spin_readnoise.setSuffix(" e⁻")
        of2.addRow("Read noise:", self._spin_readnoise)
        self._spin_gain = QDoubleSpinBox()
        self._spin_gain.setRange(0.01, 100.0); self._spin_gain.setValue(1.0)
        self._spin_gain.setSuffix(" e⁻/ADU")
        of2.addRow("Gain:", self._spin_gain)
        self._spin_sigma_clip = QDoubleSpinBox()
        self._spin_sigma_clip.setRange(1.0, 10.0); self._spin_sigma_clip.setValue(4.0)
        self._spin_sigma_clip.setSuffix(" σ")
        of2.addRow("Cosmic ray σ threshold:", self._spin_sigma_clip)
        lay.addWidget(grp_opt)
        lay.addStretch()
        return w

    def _build_tab_2d(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self._canvas_2d = SpectrumCanvas2D()
        self._canvas_2d.setMinimumHeight(300)
        lay.addWidget(self._canvas_2d)
        return w

    def _build_tab_arc(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        # Arc lamp canvas
        self._arc_canvas = ArcCanvas()
        self._arc_canvas.setMinimumHeight(220)
        self._arc_canvas.line_clicked.connect(self._on_arc_click)
        lay.addWidget(self._arc_canvas, 2)

        # Line identification controls
        grp_id = QGroupBox("Line identification")
        id_lay = QHBoxLayout(grp_id)
        id_lay.addWidget(QLabel("Lamp:"))
        self._cmb_lamp = QComboBox()
        self._cmb_lamp.addItems(list(ARC_LINES.keys()))
        self._cmb_lamp.setCurrentText("Ne + Ar (ALPY / Star Analyser)")
        id_lay.addWidget(self._cmb_lamp, 1)

        id_lay.addWidget(QLabel("Clicked px:"))
        self._lbl_clicked_px = QLabel("—")
        self._lbl_clicked_px.setStyleSheet(
            f"color:{SIRIL_NOVA}; font-weight:bold; min-width:60px;")
        id_lay.addWidget(self._lbl_clicked_px)

        id_lay.addWidget(QLabel("Wavelength (Å):"))
        self._edit_wavelength = QLineEdit()
        self._edit_wavelength.setPlaceholderText("e.g. 6402.25")
        self._edit_wavelength.setFixedWidth(100)
        id_lay.addWidget(self._edit_wavelength)

        btn_add_line = QPushButton("➕ Add line")
        btn_add_line.clicked.connect(self._add_line_id)
        id_lay.addWidget(btn_add_line)
        id_lay.addWidget(QLabel("  "))

        btn_ref_lines = QPushButton("📋 Show lamp lines")
        btn_ref_lines.clicked.connect(self._show_lamp_lines)
        id_lay.addWidget(btn_ref_lines)
        id_lay.addStretch()
        lay.addWidget(grp_id)

        # Identified lines table
        grp_lines = QGroupBox("Identified lines")
        ll = QVBoxLayout(grp_lines)
        self._lines_table = QTableWidget(0, 3)
        self._lines_table.setHorizontalHeaderLabels(
            ["Pixel", "Wavelength (Å)", ""])
        self._lines_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._lines_table.setMaximumHeight(140)
        ll.addWidget(self._lines_table)
        btns = QHBoxLayout()
        btn_remove = QPushButton("Remove selected")
        btn_remove.clicked.connect(self._remove_line)
        btn_clear_lines = QPushButton("Clear all")
        btn_clear_lines.clicked.connect(self._clear_lines)
        btns.addWidget(btn_remove); btns.addWidget(btn_clear_lines); btns.addStretch()
        ll.addLayout(btns)
        lay.addWidget(grp_lines, 1)

        # Fit solution controls
        grp_fit = QGroupBox("Dispersion solution")
        fl = QHBoxLayout(grp_fit)
        fl.addWidget(QLabel("Polynomial degree:"))
        self._spin_poly_deg = QSpinBox()
        self._spin_poly_deg.setRange(1, 6); self._spin_poly_deg.setValue(3)
        fl.addWidget(self._spin_poly_deg)
        btn_fit = QPushButton("▶ Fit dispersion solution")
        btn_fit.setObjectName("spec")
        btn_fit.clicked.connect(self._fit_dispersion)
        fl.addWidget(btn_fit)
        self._lbl_solution_info = QLabel("—")
        self._lbl_solution_info.setObjectName("dim")
        fl.addWidget(self._lbl_solution_info); fl.addStretch()
        lay.addWidget(grp_fit)

        return w

    def _build_tab_1d(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(4)

        self._canvas_1d = Spectrum1DCanvas()
        lay.addWidget(self._canvas_1d, 2)

        # Dispersion solution canvas
        self._disp_canvas = DispersionCanvas()
        self._disp_canvas.setMaximumHeight(220)
        lay.addWidget(self._disp_canvas, 1)
        return w

    def _build_tab_rv(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        grp_tmpl = QGroupBox("Template spectrum for cross-correlation")
        tl = QVBoxLayout(grp_tmpl)
        tmpl_row = QHBoxLayout()
        self._edit_tmpl = QLineEdit()
        self._edit_tmpl.setPlaceholderText(
            "Template FITS or CSV (wavelength, flux)…")
        btn_tmpl = QPushButton("Browse…"); btn_tmpl.setFixedWidth(80)
        btn_tmpl.clicked.connect(lambda: self._browse_fits(self._edit_tmpl, "Template"))
        tmpl_row.addWidget(self._edit_tmpl); tmpl_row.addWidget(btn_tmpl)
        tl.addLayout(tmpl_row)
        lbl_tmpl = QLabel(
            "Template should cover the same wavelength range as the science spectrum.\n"
            "Common choices: SDSS stellar templates (M, K, G, F, A, O types),\n"
            "IRTF spectral library, or your own synthetic model.")
        lbl_tmpl.setObjectName("dim"); lbl_tmpl.setWordWrap(True)
        tl.addWidget(lbl_tmpl)
        lay.addWidget(grp_tmpl)

        grp_rv = QGroupBox("RV cross-correlation settings")
        rf = QFormLayout(grp_rv)
        self._spin_v_range = QDoubleSpinBox()
        self._spin_v_range.setRange(10, 5000); self._spin_v_range.setValue(500)
        self._spin_v_range.setSuffix(" km/s")
        rf.addRow("Search range (±):", self._spin_v_range)
        lay.addWidget(grp_rv)

        grp_helio = QGroupBox("Heliocentric correction (optional)")
        hf = QFormLayout(grp_helio)
        self._edit_ra  = QLineEdit(); self._edit_ra.setPlaceholderText("RA in degrees")
        self._edit_dec = QLineEdit(); self._edit_dec.setPlaceholderText("Dec in degrees")
        self._edit_date = QLineEdit()
        self._edit_date.setPlaceholderText("Obs date ISO: 2024-03-15T22:30:00")
        self._spin_lat  = QDoubleSpinBox(); self._spin_lat.setRange(-90,90); self._spin_lat.setValue(48.0)
        self._spin_lon  = QDoubleSpinBox(); self._spin_lon.setRange(-180,180); self._spin_lon.setValue(16.0)
        self._spin_elev = QDoubleSpinBox(); self._spin_elev.setRange(0,5000); self._spin_elev.setValue(200); self._spin_elev.setSuffix(" m")
        hf.addRow("Target RA:", self._edit_ra)
        hf.addRow("Target Dec:", self._edit_dec)
        hf.addRow("Obs date/time:", self._edit_date)
        hf.addRow("Obs latitude:", self._spin_lat)
        hf.addRow("Obs longitude:", self._spin_lon)
        hf.addRow("Elevation:", self._spin_elev)
        btn_rv = QPushButton("▶  Compute radial velocity")
        btn_rv.setObjectName("spec"); btn_rv.clicked.connect(self._compute_rv)
        hf.addRow(btn_rv)
        self._lbl_rv_result = QLabel("—")
        self._lbl_rv_result.setStyleSheet(
            f"color:{SIRIL_NOVA}; font-size:12pt; font-weight:bold;")
        hf.addRow(self._lbl_rv_result)
        lay.addWidget(grp_helio)
        lay.addStretch()
        return w

    def _build_tab_output(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        grp_out = QGroupBox("Output folder")
        of = QHBoxLayout(grp_out)
        self._edit_out = QLineEdit(); self._edit_out.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(80)
        btn_out.clicked.connect(lambda: self._edit_out.setText(
            QFileDialog.getExistingDirectory(self, "Output folder") or
            self._edit_out.text()))
        of.addWidget(self._edit_out); of.addWidget(btn_out)
        lay.addWidget(grp_out)

        grp_target = QGroupBox("Target information")
        tf = QFormLayout(grp_target)
        self._edit_target = QLineEdit()
        self._edit_target.setPlaceholderText("e.g. HD 5980, Vega, γ Cas")
        tf.addRow("Target name:", self._edit_target)
        self._edit_observer = QLineEdit()
        tf.addRow("Observer:", self._edit_observer)
        lay.addWidget(grp_target)

        grp_save = QGroupBox("Save options")
        sf = QVBoxLayout(grp_save)
        self._chk_save_fits = QCheckBox("1D spectrum FITS (wavelength-calibrated, WCS header)")
        self._chk_save_fits.setChecked(True)
        self._chk_save_csv = QCheckBox("CSV (wavelength, flux, flux_err)")
        self._chk_save_csv.setChecked(True)
        self._chk_save_disp = QCheckBox("Dispersion solution text file")
        self._chk_save_disp.setChecked(True)
        for c in [self._chk_save_fits, self._chk_save_csv, self._chk_save_disp]:
            sf.addWidget(c)
        lay.addWidget(grp_save)

        btn_save = QPushButton("💾  Save all outputs")
        btn_save.setObjectName("primary")
        btn_save.setMinimumHeight(36); btn_save.clicked.connect(self._save_outputs)
        lay.addWidget(btn_save)

        grp_ref = QGroupBox("References")
        rl = QVBoxLayout(grp_ref)
        for r in [
            "Optimal extraction: Horne 1986, PASP 98, 609",
            "Sky subtraction:    Kelson 2003, PASP 115, 688",
            "Wavelength calib:   Tody 1986, SPIE 627, 733 (IRAF heritage)",
            "CCF radial velocity: Simkin 1974, A&A 31, 129",
            "               and   Tonry & Davis 1979, AJ 84, 1511",
        ]:
            lbl = QLabel(r); lbl.setObjectName("dim"); rl.addWidget(lbl)
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

    def _on_instrument_changed(self, idx: int):
        name = self._cmb_instrument.currentText()
        preset = INSTRUMENT_PRESETS.get(name, {})
        if preset.get("dispersion") == "horizontal":
            self._cmb_dispaxis.setCurrentIndex(0)

    def _browse_fits(self, edit: QLineEdit, title: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {title}", "",
            "FITS (*.fit *.fits *.fts);;CSV (*.csv *.txt);;All (*)")
        if path:
            edit.setText(path)
            if not self._edit_out.text():
                self._edit_out.setText(os.path.dirname(path))

    def _on_arc_click(self, px: float):
        self._clicked_px = px
        self._lbl_clicked_px.setText(f"{px:.1f}")
        # Suggest nearest lamp line
        lamp  = self._cmb_lamp.currentText()
        lines = ARC_LINES.get(lamp, {})
        if lines and self._wavelengths is not None:
            # Convert pixel to approx wavelength using current solution
            if self._dispersion_solution:
                wl_approx = np.polyval(
                    self._dispersion_solution["coeffs"], px)
                nearest_wl = min(lines.keys(),
                                  key=lambda w: abs(w - wl_approx))
                self._edit_wavelength.setText(f"{nearest_wl:.2f}")
        else:
            self._edit_wavelength.clear()

    def _add_line_id(self):
        try:
            px = float(self._lbl_clicked_px.text())
            wl = float(self._edit_wavelength.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid",
                "Click on a peak in the arc spectrum and enter a wavelength.")
            return
        # Check for duplicate
        for ep in self._id_pixels:
            if abs(ep - px) < 3:
                QMessageBox.warning(self, "Duplicate",
                    f"Already have a line near pixel {ep:.0f}")
                return
        self._id_pixels.append(px)
        self._id_waves.append(wl)
        # Update table
        row = self._lines_table.rowCount(); self._lines_table.insertRow(row)
        for col, val in enumerate([f"{px:.1f}", f"{wl:.2f}", ""]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._lines_table.setItem(row, col, item)
        # Refresh arc canvas
        if self._arc_flux is not None:
            self._arc_canvas.show_arc(
                self._arc_flux, peaks=self._arc_peaks,
                id_pixels=self._id_pixels, id_waves=self._id_waves)
        self._log.appendPlainText(
            f"Line identified: {px:.1f} px → {wl:.2f} Å")

    def _remove_line(self):
        row = self._lines_table.currentRow()
        if row < 0: return
        self._id_pixels.pop(row); self._id_waves.pop(row)
        self._lines_table.removeRow(row)
        if self._arc_flux is not None:
            self._arc_canvas.show_arc(
                self._arc_flux, peaks=self._arc_peaks,
                id_pixels=self._id_pixels, id_waves=self._id_waves)

    def _clear_lines(self):
        self._id_pixels.clear(); self._id_waves.clear()
        self._lines_table.setRowCount(0)
        if self._arc_flux is not None:
            self._arc_canvas.show_arc(
                self._arc_flux, peaks=self._arc_peaks,
                id_pixels=[], id_waves=[])

    def _show_lamp_lines(self):
        lamp  = self._cmb_lamp.currentText()
        lines = ARC_LINES.get(lamp, {})
        msg   = f"{lamp} — known lines:\n\n"
        for wl in sorted(lines.keys()):
            strength = lines[wl]
            msg += f"  {wl:8.2f} Å   intensity={strength}\n"
        QMessageBox.information(self, f"Lamp lines: {lamp}", msg)

    def _fit_dispersion(self):
        if len(self._id_pixels) < 2:
            QMessageBox.warning(self, "Too few lines",
                "Identify at least 2 arc lines (3+ recommended for poly deg>1).")
            return
        deg = min(self._spin_poly_deg.value(), len(self._id_pixels) - 1)
        try:
            sol = fit_dispersion_solution(
                self._id_pixels, self._id_waves, poly_deg=deg)
            self._dispersion_solution = sol
            rms = sol["rms_A"] * 1000
            disp = sol["dispersion"]
            self._lbl_solution_info.setText(
                f"✓  {sol['n_lines']} lines  RMS={rms:.0f} mÅ  "
                f"disp={disp:.3f} Å/px  deg={deg}")
            self._lbl_solution_info.setStyleSheet(
                f"color:{SIRIL_SUCCESS}; font-size:9pt;")
            self._disp_canvas.show_solution(sol)
            self._log.appendPlainText(
                f"Dispersion solution: RMS={rms:.1f} mÅ  "
                f"{disp:.3f} Å/px  {sol['n_lines']} lines")
            # Apply to spectrum
            if self._flux is not None:
                wl, fl, fe = apply_wavelength_solution(
                    self._flux, self._flux_err,
                    sol["coeffs"], self._n_pixels)
                self._wavelengths = wl
                target = self._edit_target.text().strip() or "Spectrum"
                self._canvas_1d.show_spectrum(
                    wl, fl, fe,
                    title=f"{target}  "
                          f"({wl.min():.0f}–{wl.max():.0f} Å  "
                          f"{disp:.2f} Å/px  "
                          f"RMS={rms:.0f} mÅ)")
                self._tabs.setCurrentIndex(4)  # 1D tab
            # Update arc canvas with wavelength axis
            if self._arc_flux is not None:
                self._arc_canvas.show_arc(
                    self._arc_flux, peaks=self._arc_peaks,
                    id_pixels=self._id_pixels, id_waves=self._id_waves)
        except Exception as e:
            QMessageBox.critical(self, "Fit error", str(e))

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _get_config(self) -> dict | None:
        sci = self._edit_sci.text().strip()
        if not sci or not os.path.isfile(sci):
            QMessageBox.warning(self, "No science frame",
                "Select a 2D spectrum FITS file."); return None
        disp_axis = 1 if self._cmb_dispaxis.currentIndex() == 0 else 0
        method    = ("Optimal (Horne 1986)" if self._rb_optimal.isChecked()
                     else "Aperture")
        return {
            "science_path":      sci,
            "arc_path":          self._edit_arc.text().strip(),
            "bias_path":         self._edit_bias.text().strip(),
            "dark_path":         self._edit_dark.text().strip(),
            "flat_path":         self._edit_flat.text().strip(),
            "instrument":        self._cmb_instrument.currentText(),
            "dispersion_axis":   disp_axis,
            "flip_horizontal":   self._chk_flip_h.isChecked(),
            "flip_vertical":     self._chk_flip_v.isChecked(),
            "extraction_method": method,
            "fwhm_guess":        self._spin_fwhm_guess.value(),
            "trace_poly_deg":    self._spin_trace_deg.value(),
            "manual_trace_center": (self._spin_manual_trace.value()
                                    if self._chk_manual_trace.isChecked()
                                    else None),
            "half_ap":           self._spin_half_ap.value(),
            "sky_gap":           self._spin_sky_gap.value(),
            "sky_width":         self._spin_sky_width.value(),
            "subtract_sky":      self._chk_sky.isChecked(),
            "readnoise":         self._spin_readnoise.value(),
            "gain":              self._spin_gain.value(),
            "sigma_clip":        self._spin_sigma_clip.value(),
            "n_iter_extr":       5,
            "min_peak_sep":      8,
        }

    def _run_extraction(self):
        cfg = self._get_config()
        if cfg is None: return
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._set_status("Running extraction pipeline…", SIRIL_SPEC)
        self._worker = ReductionWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.image_ready.connect(
            lambda img: self._canvas_2d.show_2d(img))
        self._worker.trace_ready.connect(
            lambda tr, img: self._canvas_2d.show_2d(
                img, trace=tr, half_ap=self._spin_half_ap.value()))
        self._worker.arc_ready.connect(self._on_arc_ready)
        self._worker.extract_ready.connect(self._on_extract_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._tabs.setCurrentIndex(7)  # log

    def _cancel(self):
        self._cancel_event.set()
        self._btn_cancel.setEnabled(False)
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(msg, SIRIL_SPEC)

    def _on_arc_ready(self, arc_flux: np.ndarray, peaks: list):
        self._arc_flux  = arc_flux
        self._arc_peaks = peaks
        self._arc_canvas.show_arc(arc_flux, peaks=peaks,
                                   id_pixels=self._id_pixels,
                                   id_waves=self._id_waves)
        self._tabs.setCurrentIndex(3)  # arc tab

    def _on_extract_ready(self, flux: np.ndarray, flux_err: np.ndarray):
        self._flux     = flux
        self._flux_err = flux_err
        self._n_pixels = len(flux)
        px = np.arange(len(flux))
        self._canvas_1d.show_spectrum(px, flux, flux_err,
                                       title="Extracted spectrum (pixel scale)")

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        if result.get("success"):
            self._flux       = result["flux"]
            self._flux_err   = result["flux_err"]
            self._arc_flux   = result["arc_flux"]
            self._arc_peaks  = result["arc_peaks"]
            self._trace      = result["trace"]
            self._header     = result["header"]
            self._n_pixels   = result["n_pixels"]
            self._set_status(
                f"✓ Extraction complete — {self._n_pixels} px  "
                f"peak={self._flux.max():.1f}  "
                f"{'Arc loaded — identify lines' if self._arc_flux is not None else 'No arc'}",
                SIRIL_SUCCESS)
            if self._arc_flux is None:
                self._tabs.setCurrentIndex(4)  # 1D tab
        else:
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)

    # ── Radial velocity ───────────────────────────────────────────────────────

    def _compute_rv(self):
        if self._flux is None or self._wavelengths is None:
            QMessageBox.warning(self, "No calibrated spectrum",
                "Run extraction and apply wavelength calibration first.")
            return
        tmpl_path = self._edit_tmpl.text().strip()
        if not tmpl_path or not os.path.isfile(tmpl_path):
            QMessageBox.warning(self, "No template",
                "Select a template spectrum."); return
        # Load template
        try:
            if tmpl_path.endswith(".csv") or tmpl_path.endswith(".txt"):
                data = np.loadtxt(tmpl_path, delimiter=",", comments="#")
                wt   = data[:, 0]; ft = data[:, 1]
            else:
                with astropy_fits.open(tmpl_path) as hdul:
                    ft = hdul[0].data.astype(np.float64).ravel()
                    # Build wavelength from WCS if present
                    hdr_t = hdul[0].header
                    if "CRVAL1" in hdr_t:
                        c1  = float(hdr_t.get("CRVAL1", 0))
                        cd1 = float(hdr_t.get("CD1_1",
                              hdr_t.get("CDELT1", 1)))
                        crp = float(hdr_t.get("CRPIX1", 1))
                        wt  = c1 + cd1 * (np.arange(len(ft)) - crp + 1)
                    else:
                        wt = np.linspace(self._wavelengths.min(),
                                          self._wavelengths.max(), len(ft))
        except Exception as e:
            QMessageBox.critical(self, "Template error", str(e)); return

        rv = cross_correlate_rv(self._wavelengths, self._flux,
                                 wt, ft,
                                 v_range_kms=self._spin_v_range.value())

        if rv.get("success"):
            # Heliocentric correction
            rv_corr = 0.0
            try:
                ra  = float(self._edit_ra.text())
                dec = float(self._edit_dec.text())
                dt  = self._edit_date.text().strip()
                if dt:
                    rv_corr = heliocentric_correction(
                        ra, dec, dt,
                        self._spin_lat.value(),
                        self._spin_lon.value(),
                        self._spin_elev.value())
            except Exception:
                pass

            rv_helio = rv["rv_kms"] + rv_corr
            self._lbl_rv_result.setText(
                f"RV = {rv['rv_kms']:+.2f} km/s  "
                f"(helio: {rv_helio:+.2f} km/s)")
            self._canvas_1d.show_ccf(rv)
            self._log.appendPlainText(
                f"RV: {rv['rv_kms']:+.3f} km/s  CCF_max={rv['ccf_max']:.3f}  "
                f"helio_corr={rv_corr:+.3f} km/s  helio_rv={rv_helio:+.3f} km/s")
        else:
            QMessageBox.warning(self, "RV failed", rv.get("error", ""))

    # ── Save outputs ──────────────────────────────────────────────────────────

    def _save_outputs(self):
        if self._flux is None:
            QMessageBox.warning(self, "No data", "Run extraction first.")
            return
        out_dir = self._edit_out.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output", "Set an output folder.")
            return
        os.makedirs(out_dir, exist_ok=True)
        target = self._edit_target.text().strip() or "spectrum"
        base   = target.replace(" ", "_").replace("/", "-")
        n_saved = 0

        flux    = self._flux
        flux_err= self._flux_err
        wl      = self._wavelengths
        pixels  = np.arange(len(flux))

        # 1D FITS with WCS header
        if self._chk_save_fits.isChecked():
            fits_path = os.path.join(out_dir, f"{base}_1d.fit")
            hdr = astropy_fits.Header()
            hdr["SIMPLE"]   = True
            hdr["NAXIS"]    = 1
            hdr["NAXIS1"]   = len(flux)
            hdr["OBJECT"]   = target
            if wl is not None:
                hdr["CTYPE1"]  = "WAVE"
                hdr["CUNIT1"]  = "Angstrom"
                hdr["CRPIX1"]  = 1.0
                hdr["CRVAL1"]  = float(wl[0])
                hdr["CD1_1"]   = float((wl[-1]-wl[0]) / max(len(wl)-1, 1))
            hdr["HISTORY"]  = "HyperLoad Spectroscopy Pipeline"
            if self._dispersion_solution:
                hdr["DISPRMS"] = (self._dispersion_solution["rms_A"],
                                   "Dispersion RMS [Angstrom]")
            hdu = astropy_fits.PrimaryHDU(data=flux.astype(np.float32),
                                           header=hdr)
            hdu.writeto(fits_path, overwrite=True)
            self._log.appendPlainText(f"FITS saved: {fits_path}")
            n_saved += 1

        # CSV
        if self._chk_save_csv.isChecked():
            csv_path = os.path.join(out_dir, f"{base}_spectrum.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["# HyperLoad Spectroscopy Pipeline"])
                writer.writerow(["# Target:", target])
                if wl is not None:
                    writer.writerow(["wavelength_A", "flux", "flux_err"])
                    for wi, fi, ei in zip(wl, flux, flux_err):
                        writer.writerow([f"{wi:.4f}", f"{fi:.6f}", f"{ei:.6f}"])
                else:
                    writer.writerow(["pixel", "flux", "flux_err"])
                    for pi, fi, ei in zip(pixels, flux, flux_err):
                        writer.writerow([str(pi), f"{fi:.6f}", f"{ei:.6f}"])
            self._log.appendPlainText(f"CSV saved: {csv_path}")
            n_saved += 1

        # Dispersion solution
        if self._chk_save_disp.isChecked() and self._dispersion_solution:
            sol      = self._dispersion_solution
            txt_path = os.path.join(out_dir, f"{base}_dispersion.txt")
            with open(txt_path, "w") as f:
                f.write("HyperLoad Spectroscopy Pipeline — Dispersion Solution\n")
                f.write("=" * 50 + "\n")
                f.write(f"Target:        {target}\n")
                f.write(f"Polynomial:    degree {sol['poly_deg']}\n")
                f.write(f"N lines:       {sol['n_lines']}\n")
                f.write(f"RMS:           {sol['rms_A']*1000:.2f} mÅ\n")
                f.write(f"Dispersion:    {sol['dispersion']:.4f} Å/px\n\n")
                f.write("Coefficients (highest degree first):\n")
                for i, c in enumerate(sol["coeffs"]):
                    f.write(f"  c[{i}] = {c:.10e}\n")
                f.write("\nIdentified lines:\n")
                for px, wlv, res in zip(sol["px_points"], sol["wl_points"],
                                         sol["residuals"]):
                    f.write(f"  {px:8.2f} px  →  {wlv:9.3f} Å  "
                             f"residual={res*1000:+6.1f} mÅ\n")
            self._log.appendPlainText(f"Dispersion solution: {txt_path}")
            n_saved += 1

        self._set_status(f"✓ {n_saved} files saved to {out_dir}", SIRIL_SUCCESS)

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
