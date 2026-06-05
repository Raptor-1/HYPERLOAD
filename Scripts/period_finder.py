# period_finder.py — Script #32 — Siril Periodicity Analysis
# Lomb-Scargle + ACF + Wavelet cross-validated period detection
# Place in: C:\Users\Marcell\Desktop\Siril New Scripts\

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

# ── Core imports ──────────────────────────────────────────────────────────────
import os, sys, csv, glob, json, threading
from datetime import datetime
import numpy as np
from scipy.signal import argrelextrema, correlate
try:
    from scipy.signal import morlet2
except ImportError:
    from scipy.signal import argrelextrema, correlate
try:
    from scipy.signal import morlet2
except ImportError:
    try:
        from scipy.signal import morlet as morlet2
    except ImportError:
        import numpy as np
        def morlet2(M, s, w=5.0):
            x = np.linspace(-s * 2 * np.pi, s * 2 * np.pi, M)
            return np.exp(1j * w * x / s) * np.exp(-0.5 * (x / s) ** 2) / (np.sqrt(2 * np.pi) * s)
from scipy.fft import fft, ifft
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.io import fits as astropy_fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                 aperture_photometry)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSizePolicy, QScrollArea, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ── Script directory for cross-imports ───────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from transient_detector import build_light_curves
    HAS_TRANSIENT_DETECTOR = True
except ImportError:
    HAS_TRANSIENT_DETECTOR = False

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIRIL THEME
# ═══════════════════════════════════════════════════════════════════════════════

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
QLabel#period   {{
    color: {SIRIL_ACCENT};
    font-size: 20pt;
    font-weight: bold;
    padding: 4px;
    background: transparent;
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
QSplitter::handle {{ background: {SIRIL_BORDER}; width: 2px; }}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATA LOADING AND NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv_lightcurve(csv_path: str) -> tuple:
    times, fluxes, errors = [], [], []
    has_errors = False
    with open(csv_path, "r") as f:
        sample = f.read(1024)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;")
        reader  = csv.reader(f, dialect)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                t = float(row[0])
                v = float(row[1])
                times.append(t)
                fluxes.append(v)
                if len(row) >= 3:
                    errors.append(float(row[2]))
                    has_errors = True
            except (ValueError, IndexError):
                continue
    t = np.array(times,  dtype=np.float64)
    f = np.array(fluxes, dtype=np.float64)
    e = np.array(errors, dtype=np.float64) if has_errors else None
    return t, f, e


def guess_time_unit(times: np.ndarray, header_text: str = "") -> str:
    if "JD" in header_text.upper() or "BJD" in header_text.upper():
        return "Julian Days"
    if len(times) > 0:
        if times[0] > 2400000:
            return "JD"
        if times[0] < 10000:
            return "frames"
    return "time units"


def normalize_lightcurve(times, fluxes, errors=None, method="median"):
    valid = np.isfinite(fluxes) & np.isfinite(times)
    if errors is not None:
        valid &= np.isfinite(errors)
    t = times[valid]
    f = fluxes[valid]
    e = errors[valid] if errors is not None else None
    if len(f) == 0:
        return t, f, e
    if method == "median":
        med = np.median(f)
        mad = np.median(np.abs(f - med)) * 1.4826
        if mad > 0:
            f = (f - med) / mad
            if e is not None: e = e / mad
    elif method == "mean":
        mu  = np.mean(f)
        sig = np.std(f)
        if sig > 0:
            f = (f - mu) / sig
            if e is not None: e = e / sig
    elif method == "minmax":
        mn, mx = f.min(), f.max()
        if mx > mn:
            f = (f - mn) / (mx - mn)
            if e is not None: e = e / (mx - mn)
    return t, f, e


def parse_manual_data(text: str) -> tuple:
    times, fluxes, errors = [], [], []
    has_errors = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            times.append(float(parts[0]))
            fluxes.append(float(parts[1]))
            if len(parts) >= 3:
                errors.append(float(parts[2]))
                has_errors = True
        except (ValueError, IndexError):
            continue
    t = np.array(times,  dtype=np.float64)
    f = np.array(fluxes, dtype=np.float64)
    e = np.array(errors, dtype=np.float64) if has_errors else None
    return t, f, e


def compute_from_fits_sequence(fits_dir, star_x, star_y,
                                aperture_radius=8.0, log_callback=None):
    files = sorted(
        glob.glob(os.path.join(fits_dir, "*.fit"))  +
        glob.glob(os.path.join(fits_dir, "*.fits")) +
        glob.glob(os.path.join(fits_dir, "*.fts"))
    )
    if not files:
        return np.array([]), np.array([]), None

    first_data = astropy_fits.getdata(files[0]).astype(float)
    if first_data.ndim == 3:
        first_data = first_data[0]
    _, med, std = sigma_clipped_stats(first_data, sigma=3.0)
    daofind  = DAOStarFinder(fwhm=5.0, threshold=8.0*std,
                              sharplo=0.3, sharphi=0.9)
    sources  = daofind(first_data - med)

    ref_positions = []
    if sources is not None:
        for src in sources:
            rx, ry = float(src["xcentroid"]), float(src["ycentroid"])
            dist   = ((rx - star_x)**2 + (ry - star_y)**2)**0.5
            if dist > 30:
                ref_positions.append((rx, ry))
            if len(ref_positions) >= 15:
                break

    times, fluxes, errors = [], [], []
    for i, path in enumerate(files):
        if log_callback and i % 20 == 0:
            log_callback(f"  Photometry frame {i+1}/{len(files)}")
        try:
            with astropy_fits.open(path) as hdul:
                data = hdul[0].data.astype(float)
                hdr  = hdul[0].header
            if data.ndim == 3:
                data = data[0]
            date_obs = hdr.get("DATE-OBS")
            if date_obs:
                t_val = Time(date_obs, format="isot", scale="utc").jd
            else:
                t_val = float(i)

            target_aper  = CircularAperture([(star_x, star_y)], r=aperture_radius)
            target_annul = CircularAnnulus([(star_x, star_y)],
                                            r_in=aperture_radius*1.5,
                                            r_out=aperture_radius*2.5)
            t_phot = aperture_photometry(data, target_aper)
            t_bkg  = aperture_photometry(data, target_annul)
            t_flux = (float(t_phot["aperture_sum"][0]) -
                      float(t_bkg["aperture_sum"][0]) /
                      target_annul.area * target_aper.area)

            if ref_positions:
                ref_aper  = CircularAperture(ref_positions, r=aperture_radius)
                ref_phot  = aperture_photometry(data, ref_aper)
                ref_total = float(np.median(
                    [float(ref_phot["aperture_sum"][j])
                     for j in range(len(ref_positions))]))
                if ref_total > 0:
                    t_flux /= ref_total

            times.append(t_val)
            fluxes.append(t_flux)
            errors.append(abs(t_flux)**0.5 / max(abs(t_flux), 1))
        except Exception as ex:
            if log_callback:
                log_callback(f"  Frame {i} failed: {ex}")
            continue

    return (np.array(times), np.array(fluxes),
            np.array(errors) if errors else None)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LOMB-SCARGLE
# ═══════════════════════════════════════════════════════════════════════════════

def run_lomb_scargle(times, fluxes, errors=None,
                     min_period=None, max_period=None,
                     n_peaks=5, fap_levels=None):
    if fap_levels is None:
        fap_levels = [0.1, 0.01, 0.001]
    baseline  = times.max() - times.min()
    dt_median = float(np.median(np.diff(np.sort(times))))
    if min_period is None:
        min_period = max(2 * dt_median, baseline / 1000)
    if max_period is None:
        max_period = baseline / 2
    if errors is not None and len(errors) == len(fluxes):
        ls = LombScargle(times, fluxes, errors)
    else:
        ls = LombScargle(times, fluxes)
    frequency, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=10
    )
    periods = 1.0 / frequency
    fap_thresholds = {}
    for level in fap_levels:
        try:
            fap_thresholds[level] = float(ls.false_alarm_level(level))
        except Exception:
            pass
    local_max_idx = argrelextrema(power, np.greater, order=5)[0]
    if len(local_max_idx) == 0:
        local_max_idx = np.array([np.argmax(power)])
    peak_powers = power[local_max_idx]
    sorted_idx  = np.argsort(peak_powers)[::-1]
    top_idx     = local_max_idx[sorted_idx[:n_peaks]]
    top_peaks   = []
    for idx in top_idx:
        p  = float(periods[idx])
        pw = float(power[idx])
        try:
            fap = float(ls.false_alarm_probability(pw))
        except Exception:
            fap = float("nan")
        top_peaks.append({"period": p, "power": pw, "fap": fap})
    best = top_peaks[0] if top_peaks else {"period": 0, "power": 0, "fap": 1}
    return {
        "periods":        periods[::-1],
        "power":          power[::-1],
        "best_period":    best["period"],
        "best_power":     best["power"],
        "best_fap":       best["fap"],
        "top_peaks":      top_peaks,
        "fap_thresholds": fap_thresholds,
        "method":         "lomb_scargle",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ACF
# ═══════════════════════════════════════════════════════════════════════════════

def run_acf(times, fluxes, min_period=None, max_period=None, oversample=10):
    t_min, t_max = times.min(), times.max()
    baseline     = t_max - t_min
    dt_median    = float(np.median(np.diff(np.sort(times))))
    dt_uniform   = dt_median / oversample
    t_uniform    = np.arange(t_min, t_max, dt_uniform)
    f_uniform    = np.interp(t_uniform, np.sort(times),
                              fluxes[np.argsort(times)])
    f_centered   = f_uniform - np.mean(f_uniform)
    acf_full     = correlate(f_centered, f_centered, mode="full")
    n            = len(f_centered)
    acf_full     = acf_full[n-1:]
    lags         = np.arange(len(acf_full)) * dt_uniform
    if acf_full[0] > 0:
        acf_full /= acf_full[0]
    if min_period is None:
        min_period = 2 * dt_median
    if max_period is None:
        max_period = baseline / 2
    valid    = (lags >= min_period) & (lags <= max_period)
    lags_cut = lags[valid]
    acf_cut  = acf_full[valid]
    order    = max(1, len(acf_cut) // 50)
    peak_idx = argrelextrema(acf_cut, np.greater, order=order)[0]
    peak_lags = []
    for idx in peak_idx:
        if acf_cut[idx] > 0.1:
            peak_lags.append({"period": float(lags_cut[idx]),
                               "acf":    float(acf_cut[idx])})
    peak_lags.sort(key=lambda p: p["acf"], reverse=True)
    best_period = peak_lags[0]["period"] if peak_lags else 0.0
    return {
        "lags":        lags_cut,
        "acf":         acf_cut,
        "best_period": best_period,
        "peak_lags":   peak_lags,
        "method":      "acf",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WAVELET
# ═══════════════════════════════════════════════════════════════════════════════

def run_wavelet(times, fluxes, min_period=None, max_period=None,
                n_periods=100, omega0=6.0):
    baseline  = times.max() - times.min()
    dt_median = float(np.median(np.diff(np.sort(times))))
    if min_period is None:
        min_period = 4 * dt_median
    if max_period is None:
        max_period = baseline / 3
    t_min     = times.min()
    t_max     = times.max()
    dt        = dt_median
    t_uniform = np.arange(t_min, t_max, dt)
    f_uniform = np.interp(t_uniform, np.sort(times),
                           fluxes[np.argsort(times)])
    f_uniform -= np.mean(f_uniform)
    periods_grid = np.logspace(
        np.log10(min_period), np.log10(max_period), n_periods)
    n_t      = len(t_uniform)
    power_2d = np.zeros((n_periods, n_t))
    for i, period in enumerate(periods_grid):
        scale   = period * omega0 / (2 * np.pi * dt)
        wavelet = morlet2(n_t, w=omega0, s=scale)
        W       = ifft(fft(f_uniform) * np.conj(fft(wavelet)))
        power_2d[i, :] = np.abs(W) ** 2
    global_power = np.mean(power_2d, axis=1)
    best_idx     = int(np.argmax(global_power))
    best_period  = float(periods_grid[best_idx])
    coi_half     = np.sqrt(2) * periods_grid
    coi_left     = t_uniform[0]  + coi_half
    coi_right    = t_uniform[-1] - coi_half
    return {
        "times_uniform":      t_uniform,
        "periods":            periods_grid,
        "power":              power_2d,
        "global_power":       global_power,
        "best_period":        best_period,
        "cone_of_influence":  (coi_left, coi_right),
        "method":             "wavelet",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate_periods(ls_result, acf_result, wav_result, tolerance=0.05):
    ls_p  = ls_result.get("best_period", 0)  if ls_result  else 0
    acf_p = acf_result.get("best_period", 0) if acf_result else 0
    wav_p = wav_result.get("best_period", 0) if wav_result else 0

    def agree(p1, p2):
        if p1 <= 0 or p2 <= 0:
            return False
        return abs(p1 - p2) / max(p1, p2) < tolerance

    agreements = []
    if agree(ls_p, acf_p):  agreements.append(("lomb_scargle", "acf"))
    if agree(ls_p, wav_p):  agreements.append(("lomb_scargle", "wavelet"))
    if agree(acf_p, wav_p): agreements.append(("acf", "wavelet"))

    _map = {"lomb_scargle": ls_p, "acf": acf_p, "wavelet": wav_p}
    confirmed_periods = set()
    for m1, m2 in agreements:
        confirmed_periods.add(round((_map[m1] + _map[m2]) / 2, 6))

    if confirmed_periods:
        best_period = sorted(confirmed_periods)[0]
        confidence  = "confirmed"
    else:
        best_period = ls_p if ls_p > 0 else (acf_p if acf_p > 0 else wav_p)
        confidence  = "candidate" if best_period > 0 else "none"

    return {
        "best_period":     best_period,
        "confidence":      confidence,
        "agreements":      agreements,
        "all_periods":     {"lomb_scargle": ls_p, "acf": acf_p, "wavelet": wav_p},
        "confirmed_peaks": list(confirmed_periods),
        "n_methods_agree": len(agreements),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PHASE FOLD
# ═══════════════════════════════════════════════════════════════════════════════

def phase_fold(times, fluxes, period, t0=None, n_bins=50):
    if t0 is None:
        t0 = times.min()
    phases       = ((times - t0) / period) % 1.0
    sort_idx     = np.argsort(phases)
    phases_s     = phases[sort_idx]
    fluxes_s     = fluxes[sort_idx]
    bin_edges    = np.linspace(0, 1, n_bins + 1)
    bin_phases   = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_fluxes   = np.full(n_bins, np.nan)
    bin_errors   = np.full(n_bins, np.nan)
    for i in range(n_bins):
        in_bin = (phases_s >= bin_edges[i]) & (phases_s < bin_edges[i+1])
        if np.sum(in_bin) >= 2:
            bin_fluxes[i] = np.median(fluxes_s[in_bin])
            bin_errors[i] = np.std(fluxes_s[in_bin]) / np.sqrt(np.sum(in_bin))
    return {
        "phases":     phases_s,
        "fluxes":     fluxes_s,
        "bin_phases": bin_phases,
        "bin_fluxes": bin_fluxes,
        "bin_errors": bin_errors,
        "period":     period,
        "t0":         t0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 HELPERS — HARMONICS + ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

def check_harmonics(best_period, top_peaks, tolerance=0.05):
    harmonics = []
    for factor in [0.5, 2.0, 0.333, 3.0, 0.25, 4.0]:
        expected = best_period * factor
        for peak in top_peaks:
            if abs(peak["period"] - expected) / max(expected, 1e-10) < tolerance:
                harmonics.append({
                    "period": peak["period"],
                    "factor": factor,
                    "label":  f"P×{factor}",
                })
    return harmonics


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class PeriodWorker(QThread):
    progress   = pyqtSignal(int, int, str)
    log_line   = pyqtSignal(str)
    ls_ready   = pyqtSignal(dict)
    acf_ready  = pyqtSignal(dict)
    wav_ready  = pyqtSignal(dict)
    fold_ready = pyqtSignal(dict)
    consensus  = pyqtSignal(dict)
    finished   = pyqtSignal(dict)

    def __init__(self, config, times, fluxes, errors, cancel_event):
        super().__init__()
        self.config  = config
        self.times   = times
        self.fluxes  = fluxes
        self.errors  = errors
        self._cancel = cancel_event

    def run(self):
        import traceback
        try:
            cfg   = self.config
            t, f, e = self.times, self.fluxes, self.errors
            n     = len(t)
            min_p = cfg.get("min_period") or None
            max_p = cfg.get("max_period") or None
            if min_p == 0: min_p = None
            if max_p == 0: max_p = None

            self.log_line.emit(
                f"Analysis: {n} data points  |  "
                f"Baseline: {t.max()-t.min():.4f} time units")

            ls_r = acf_r = wav_r = None

            if cfg.get("run_ls", True):
                self.progress.emit(1, 5, "Lomb-Scargle periodogram...")
                self.log_line.emit("Running Lomb-Scargle periodogram...")
                ls_r = run_lomb_scargle(t, f, e, min_period=min_p,
                                        max_period=max_p,
                                        n_peaks=cfg.get("n_peaks", 10))
                self.ls_ready.emit(ls_r)
                self.log_line.emit(
                    f"  Best period: {ls_r['best_period']:.6f}  "
                    f"FAP: {ls_r['best_fap']:.2e}")
                if self._cancel.is_set(): self._abort(); return

            if cfg.get("run_acf", True):
                self.progress.emit(2, 5, "Autocorrelation function...")
                self.log_line.emit("Running ACF analysis...")
                acf_r = run_acf(t, f, min_period=min_p, max_period=max_p)
                self.acf_ready.emit(acf_r)
                self.log_line.emit(
                    f"  Best period: {acf_r['best_period']:.6f}  "
                    f"peaks: {len(acf_r['peak_lags'])}")
                if self._cancel.is_set(): self._abort(); return

            if cfg.get("run_wavelet", True):
                self.progress.emit(3, 5, "Wavelet analysis...")
                self.log_line.emit("Running Morlet wavelet transform...")
                wav_r = run_wavelet(t, f, min_period=min_p, max_period=max_p,
                                    n_periods=cfg.get("wavelet_periods", 100))
                self.wav_ready.emit(wav_r)
                self.log_line.emit(
                    f"  Best period: {wav_r['best_period']:.6f}")
                if self._cancel.is_set(): self._abort(); return

            self.progress.emit(4, 5, "Cross-validating results...")
            cv = cross_validate_periods(ls_r, acf_r, wav_r,
                                        tolerance=cfg.get("tolerance", 0.05))
            self.consensus.emit(cv)
            self.log_line.emit(
                f"Consensus: {cv['best_period']:.6f}  "
                f"confidence={cv['confidence']}  "
                f"agreements={len(cv['agreements'])}")

            if cv["best_period"] > 0:
                self.progress.emit(5, 5, "Phase folding...")
                fold = phase_fold(t, f, cv["best_period"],
                                   n_bins=cfg.get("phase_bins", 50))
                self.fold_ready.emit(fold)

            top_peaks = ls_r.get("top_peaks", []) if ls_r else []
            harmonics = check_harmonics(cv["best_period"], top_peaks)

            self.finished.emit({
                "success":     True,
                "best_period": cv["best_period"],
                "confidence":  cv["confidence"],
                "best_fap":    ls_r["best_fap"] if ls_r else float("nan"),
                "n_points":    n,
                "top_peaks":   top_peaks,
                "consensus":   cv,
                "harmonics":   harmonics,
                "ls_result":   ls_r,
                "acf_result":  acf_r,
                "wav_result":  wav_r,
            })
        except Exception as ex:
            self.log_line.emit(f"ERROR: {ex}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(ex)})

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — MATPLOTLIB CANVASES
# ═══════════════════════════════════════════════════════════════════════════════

class PeriodogramCanvas(FigureCanvasQTAgg):
    period_clicked = pyqtSignal(float)

    def __init__(self, parent=None):
        self.fig    = Figure(figsize=(10, 4), facecolor=SIRIL_BG)
        self.ax_ls  = self.fig.add_subplot(1, 2, 1)
        self.ax_acf = self.fig.add_subplot(1, 2, 2)
        self._style_all()
        super().__init__(self.fig)
        self.setParent(parent)
        self.mpl_connect("button_press_event", self._on_click)
        self._ls_result = None

    def _style_ax(self, ax, title):
        ax.set_facecolor(SIRIL_BG2)
        ax.tick_params(colors=SIRIL_TEXT, labelsize=8)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_title(title, color=SIRIL_SECTION, fontsize=9)

    def _style_all(self):
        self._style_ax(self.ax_ls,  "Lomb-Scargle Periodogram")
        self._style_ax(self.ax_acf, "Autocorrelation Function")

    def _on_click(self, event):
        if event.inaxes == self.ax_ls and event.xdata is not None:
            self.period_clicked.emit(float(event.xdata))

    def show_lomb_scargle(self, result: dict, show_aliases: bool = False):
        self._ls_result = result
        ax = self.ax_ls
        ax.clear()
        self._style_ax(ax, "Lomb-Scargle Periodogram")
        ax.plot(result["periods"], result["power"],
                color=SIRIL_ACCENT, linewidth=0.8, alpha=0.9)
        fap_colors = {0.1: SIRIL_WARNING, 0.01: SIRIL_SUCCESS,
                      0.001: SIRIL_ERROR}
        for level, thresh in result.get("fap_thresholds", {}).items():
            c = fap_colors.get(level, SIRIL_TEXT_DIM)
            ax.axhline(thresh, color=c, linewidth=0.8, linestyle="--",
                       alpha=0.7, label=f"FAP={level}")
        bp = result["best_period"]
        if bp > 0:
            ax.axvline(bp, color="#ff6b35", linewidth=1.5,
                       label=f"P={bp:.4f}")
        if show_aliases and bp > 0:
            for alias in [1.0, 0.5, 365.25, 182.63]:
                ax.axvline(alias, color=SIRIL_TEXT_DIM, linewidth=0.7,
                           linestyle=":", alpha=0.5, label=f"alias {alias}")
        ax.set_xlabel("Period", color=SIRIL_TEXT_DIM, fontsize=8)
        ax.set_ylabel("Power",  color=SIRIL_TEXT_DIM, fontsize=8)
        ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                  labelcolor=SIRIL_TEXT, fontsize=7)
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def show_acf(self, result: dict):
        ax = self.ax_acf
        ax.clear()
        self._style_ax(ax, "Autocorrelation Function")
        ax.plot(result["lags"], result["acf"],
                color=SIRIL_SUCCESS, linewidth=0.9, alpha=0.9)
        ax.axhline(0, color=SIRIL_BORDER, linewidth=0.8, linestyle="--")
        for peak in result.get("peak_lags", [])[:3]:
            ax.axvline(peak["period"], color=SIRIL_WARNING,
                       linewidth=1.2, linestyle="--", alpha=0.7)
        bp = result["best_period"]
        if bp > 0:
            ax.axvline(bp, color="#ff6b35", linewidth=1.5,
                       label=f"P={bp:.4f}")
            ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                      labelcolor=SIRIL_TEXT, fontsize=7)
        ax.set_xlabel("Lag (time units)", color=SIRIL_TEXT_DIM, fontsize=8)
        ax.set_ylabel("ACF",              color=SIRIL_TEXT_DIM, fontsize=8)
        self.fig.tight_layout(pad=0.4)
        self.draw()


class WaveletCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig     = Figure(figsize=(10, 5), facecolor=SIRIL_BG)
        self.ax_wav  = self.fig.add_subplot(1, 2, 1)
        self.ax_glob = self.fig.add_subplot(1, 2, 2)
        super().__init__(self.fig)
        self.setParent(parent)

    def show_wavelet(self, result: dict):
        self.ax_wav.clear()
        self.ax_glob.clear()
        t = result["times_uniform"]
        p = result["periods"]
        W = result["power"]

        self.ax_wav.contourf(t, p, W, levels=20, cmap="inferno")
        self.ax_wav.set_yscale("log")
        self.ax_wav.set_xlabel("Time",           color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_wav.set_ylabel("Period (log)",    color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_wav.set_title("Wavelet Power Spectrum",
                               color=SIRIL_SECTION, fontsize=9)
        self.ax_wav.tick_params(colors=SIRIL_TEXT, labelsize=7)
        self.ax_wav.set_facecolor(SIRIL_BG)

        coi_left, coi_right = result["cone_of_influence"]
        valid_left = coi_left <= t[-1]
        if np.any(valid_left):
            self.ax_wav.fill_betweenx(
                p[valid_left], t[0],
                np.clip(coi_left[valid_left], t[0], t[-1]),
                alpha=0.3, color=SIRIL_BG3, hatch="//",
                label="COI")

        self.ax_glob.plot(result["global_power"], p,
                          color=SIRIL_ACCENT, linewidth=1.2)
        self.ax_glob.set_yscale("log")
        self.ax_glob.axhline(result["best_period"], color="#ff6b35",
                              linewidth=1.5,
                              label=f"P={result['best_period']:.4f}")
        self.ax_glob.set_xlabel("Global power", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_glob.set_ylabel("Period",       color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_glob.set_title("Global Wavelet Spectrum",
                                color=SIRIL_SECTION, fontsize=9)
        self.ax_glob.set_facecolor(SIRIL_BG2)
        self.ax_glob.tick_params(colors=SIRIL_TEXT, labelsize=7)
        for sp in ["bottom", "left"]:
            self.ax_glob.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top", "right"]:
            self.ax_glob.spines[sp].set_visible(False)
        self.ax_glob.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                             labelcolor=SIRIL_TEXT, fontsize=7)
        self.fig.tight_layout(pad=0.4)
        self.draw()


class LightCurveCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig   = Figure(figsize=(10, 5), facecolor=SIRIL_BG)
        self.ax_lc = self.fig.add_subplot(2, 1, 1)
        self.ax_ph = self.fig.add_subplot(2, 1, 2)
        super().__init__(self.fig)
        self.setParent(parent)

    def _style(self, ax, title):
        ax.set_facecolor(SIRIL_BG2)
        ax.tick_params(colors=SIRIL_TEXT, labelsize=8)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_title(title, color=SIRIL_SECTION, fontsize=9)

    def show_lightcurve(self, times, fluxes, errors=None):
        self.ax_lc.clear()
        self._style(self.ax_lc, "Light Curve")
        if errors is not None:
            self.ax_lc.errorbar(times, fluxes, yerr=errors,
                                 fmt="o", color=SIRIL_ACCENT,
                                 markersize=2, linewidth=0,
                                 elinewidth=0.8, alpha=0.7)
        else:
            self.ax_lc.plot(times, fluxes, "o",
                             color=SIRIL_ACCENT, markersize=2, alpha=0.7)
        self.ax_lc.set_xlabel("Time", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_lc.set_ylabel("Flux", color=SIRIL_TEXT_DIM, fontsize=8)
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def show_phase_fold(self, fold: dict):
        self.ax_ph.clear()
        p = fold["period"]
        self._style(self.ax_ph, f"Phase-folded  (P = {p:.6f})")
        self.ax_ph.plot(fold["phases"], fold["fluxes"], "o",
                         color=SIRIL_ACCENT, markersize=2,
                         alpha=0.4, zorder=1)
        self.ax_ph.plot(fold["phases"] + 1.0, fold["fluxes"], "o",
                         color=SIRIL_ACCENT, markersize=2,
                         alpha=0.2, zorder=1)
        valid = ~np.isnan(fold["bin_fluxes"])
        if np.sum(valid) > 2:
            bp = fold["bin_phases"][valid]
            bf = fold["bin_fluxes"][valid]
            be = fold["bin_errors"][valid]
            for offset, alpha in [(0.0, 0.9), (1.0, 0.5)]:
                self.ax_ph.plot(bp + offset, bf,
                                 color="#ff6b35", linewidth=1.5,
                                 zorder=3, alpha=alpha)
                self.ax_ph.fill_between(bp + offset, bf - be, bf + be,
                                         color="#ff6b35", alpha=0.15)
        self.ax_ph.set_xlim(0, 2)
        self.ax_ph.set_xlabel("Phase", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_ph.set_ylabel("Flux",  color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_ph.axvline(1.0, color=SIRIL_BORDER,
                            linewidth=0.8, linestyle="--")
        self.fig.tight_layout(pad=0.4)
        self.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_results_csv(output_dir, result_summary, top_peaks, times, fluxes,
                     ls_result=None, acf_result=None, harmonics=None):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"period_analysis_{ts}.csv")
    cv   = result_summary.get("consensus", {})
    harmonic_periods = {h["period"] for h in (harmonics or [])}

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Period analysis results — period_finder.py"])
        w.writerow([f"# Generated: {datetime.now().isoformat()}"])
        w.writerow([f"# N_points: {result_summary.get('n_points', '?')}"])
        if len(times) > 0:
            w.writerow([f"# Baseline: {times.max()-times.min():.4f} time units"])
        w.writerow([f"# Best_period: {result_summary.get('best_period', '?')}"])
        w.writerow([f"# Confidence: {result_summary.get('confidence', '?')}"])
        fap = result_summary.get("best_fap", float("nan"))
        w.writerow([f"# FAP: {fap:.3e}" if not np.isnan(fap) else "# FAP: N/A"])
        agree_str = "+".join(
            [f"{m1}+{m2}" for m1, m2 in cv.get("agreements", [])])
        w.writerow([f"# Methods_agree: {agree_str or 'none'}"])
        w.writerow([])
        w.writerow(["Rank", "Period", "LS_Power", "FAP",
                    "ACF_Confirmation", "Wavelet_Confirmation", "Notes"])

        acf_p   = acf_result["best_period"]   if acf_result  else 0
        wav_p   = (result_summary.get("wav_result", {}) or {}).get(
            "best_period", 0)
        tol     = 0.05

        for rank, peak in enumerate(top_peaks, 1):
            p     = peak["period"]
            acf_c = "yes" if acf_p > 0 and abs(p-acf_p)/max(p,1e-10) < tol else "no"
            wav_c = "yes" if wav_p > 0 and abs(p-wav_p)/max(p,1e-10) < tol else "no"
            notes = "CONFIRMED" if (acf_c == "yes" or wav_c == "yes") else "candidate"
            if p in harmonic_periods:
                notes += " (harmonic?)"
            w.writerow([rank, f"{p:.8f}",
                        f"{peak['power']:.6f}",
                        f"{peak['fap']:.3e}" if not np.isnan(peak['fap']) else "N/A",
                        acf_c, wav_c, notes])

    return path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Periodicity Analysis — Siril")
        self.resize(1380, 820)

        self._times   = None
        self._fluxes  = None
        self._errors  = None
        self._worker  = None
        self._cancel  = threading.Event()
        self._last_result = {}
        self._ls_result   = None
        self._acf_result  = None
        self._wav_result  = None
        self._fold_result = None
        self._show_aliases = True

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(40)
        title_bar.setStyleSheet(
            f"background:{SIRIL_BG3}; border-bottom:1px solid {SIRIL_BORDER};")
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(14, 0, 14, 0)
        lbl_title = QLabel("〜  Periodicity Analysis  —  Siril")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold;")
        lbl_ver = QLabel("v1.0  |  Lomb-Scargle + ACF + Wavelet")
        lbl_ver.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        tb_lay.addWidget(lbl_title)
        tb_lay.addStretch()
        tb_lay.addWidget(lbl_ver)
        root.addWidget(title_bar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)
        root.addWidget(splitter, 1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([340, 1040])

        # Bottom bar
        bottom = QWidget()
        bottom.setFixedHeight(44)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG3}; border-top:1px solid {SIRIL_BORDER};")
        b_lay = QHBoxLayout(bottom)
        b_lay.setContentsMargins(12, 6, 12, 6)

        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 5)
        self.prog_bar.setFixedHeight(8)
        self.prog_bar.hide()
        b_lay.addWidget(self.prog_bar, 1)

        self.btn_analyze = QPushButton("▶  Analyze")
        self.btn_analyze.setObjectName("primary")
        self.btn_analyze.setFixedWidth(120)
        self.btn_analyze.clicked.connect(self._run_analysis)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setFixedWidth(100)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_analysis)

        b_lay.addWidget(self.btn_analyze)
        b_lay.addWidget(self.btn_cancel)
        root.addWidget(bottom)

        # Status bar
        self.status_lbl = QLabel("Ready  —  Load data then click Analyze")
        self.status_lbl.setStyleSheet(
            f"background:{SIRIL_BG2}; color:{SIRIL_TEXT_DIM}; "
            f"padding:3px 12px; font-size:9pt; "
            f"border-top:1px solid {SIRIL_BORDER};")
        self.status_lbl.setFixedHeight(24)
        root.addWidget(self.status_lbl)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── Data input ────────────────────────────────────────────────────────
        grp_input = QGroupBox("Data Input")
        g_lay     = QVBoxLayout(grp_input)
        g_lay.setSpacing(4)

        src_lay = QHBoxLayout()
        src_lay.addWidget(QLabel("Source:"))
        self.cmb_source = QComboBox()
        self.cmb_source.addItems([
            "CSV file (time, flux)",
            "FITS sequence (aperture photometry)",
            "Paste data manually",
        ])
        self.cmb_source.currentIndexChanged.connect(self._on_source_changed)
        src_lay.addWidget(self.cmb_source, 1)
        g_lay.addLayout(src_lay)

        self.stk_source = QStackedWidget()

        # Page 0 — CSV
        csv_page = QWidget()
        csv_lay  = QVBoxLayout(csv_page)
        csv_lay.setContentsMargins(0, 0, 0, 0)
        csv_row  = QHBoxLayout()
        self.le_csv = QLineEdit()
        self.le_csv.setPlaceholderText("Path to CSV…")
        btn_browse_csv = QPushButton("…")
        btn_browse_csv.setFixedWidth(30)
        btn_browse_csv.clicked.connect(self._browse_csv)
        csv_row.addWidget(self.le_csv, 1)
        csv_row.addWidget(btn_browse_csv)
        csv_lay.addLayout(csv_row)
        btn_load_csv = QPushButton("Load CSV")
        btn_load_csv.clicked.connect(self._load_csv)
        csv_lay.addWidget(btn_load_csv)
        self.lbl_csv_info = QLabel("")
        self.lbl_csv_info.setObjectName("dim")
        self.lbl_csv_info.setWordWrap(True)
        csv_lay.addWidget(self.lbl_csv_info)
        self.stk_source.addWidget(csv_page)

        # Page 1 — FITS
        fits_page = QWidget()
        fits_lay  = QVBoxLayout(fits_page)
        fits_lay.setContentsMargins(0, 0, 0, 0)
        fits_row  = QHBoxLayout()
        self.le_fits = QLineEdit()
        self.le_fits.setPlaceholderText("FITS folder…")
        btn_browse_fits = QPushButton("…")
        btn_browse_fits.setFixedWidth(30)
        btn_browse_fits.clicked.connect(self._browse_fits)
        fits_row.addWidget(self.le_fits, 1)
        fits_row.addWidget(btn_browse_fits)
        fits_lay.addLayout(fits_row)

        fits_lay.addWidget(QLabel("Target star position (pixels):"))
        xy_row = QHBoxLayout()
        xy_row.addWidget(QLabel("X:"))
        self.spn_star_x = QDoubleSpinBox()
        self.spn_star_x.setRange(0, 99999)
        self.spn_star_x.setDecimals(1)
        xy_row.addWidget(self.spn_star_x)
        xy_row.addWidget(QLabel("Y:"))
        self.spn_star_y = QDoubleSpinBox()
        self.spn_star_y.setRange(0, 99999)
        self.spn_star_y.setDecimals(1)
        xy_row.addWidget(self.spn_star_y)
        fits_lay.addLayout(xy_row)

        aper_row = QFormLayout()
        self.spn_aperture = QDoubleSpinBox()
        self.spn_aperture.setRange(1, 100)
        self.spn_aperture.setValue(8.0)
        self.spn_aperture.setDecimals(1)
        aper_row.addRow("Aperture (px):", self.spn_aperture)
        fits_lay.addLayout(aper_row)

        btn_phot = QPushButton("Run aperture photometry")
        btn_phot.clicked.connect(self._run_photometry)
        fits_lay.addWidget(btn_phot)
        self.stk_source.addWidget(fits_page)

        # Page 2 — Manual
        manual_page = QWidget()
        manual_lay  = QVBoxLayout(manual_page)
        manual_lay.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("One point per line: time flux [error]")
        hint.setObjectName("dim")
        manual_lay.addWidget(hint)
        self.txt_manual = QPlainTextEdit()
        self.txt_manual.setPlaceholderText("0.0 1.023\n1.0 0.998\n2.0 1.011")
        self.txt_manual.setFixedHeight(120)
        manual_lay.addWidget(self.txt_manual)
        btn_parse = QPushButton("Parse data")
        btn_parse.clicked.connect(self._parse_manual)
        manual_lay.addWidget(btn_parse)
        self.stk_source.addWidget(manual_page)

        g_lay.addWidget(self.stk_source)
        lay.addWidget(grp_input)

        # ── Normalization ─────────────────────────────────────────────────────
        grp_norm = QGroupBox("Normalization")
        n_lay    = QVBoxLayout(grp_norm)
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems([
            "Median (robust)", "Mean", "Min-Max", "None"])
        n_lay.addWidget(self.cmb_norm)
        lay.addWidget(grp_norm)

        # ── Period search range ───────────────────────────────────────────────
        grp_range = QGroupBox("Period Search Range")
        r_lay     = QFormLayout(grp_range)
        self.spn_min_period = QDoubleSpinBox()
        self.spn_min_period.setRange(0, 1e9)
        self.spn_min_period.setDecimals(6)
        self.spn_min_period.setValue(0)
        self.spn_max_period = QDoubleSpinBox()
        self.spn_max_period.setRange(0, 1e9)
        self.spn_max_period.setDecimals(6)
        self.spn_max_period.setValue(0)
        r_lay.addRow("Min period:", self.spn_min_period)
        r_lay.addRow("Max period:", self.spn_max_period)
        auto_note = QLabel("0 = auto from data baseline")
        auto_note.setObjectName("dim")
        r_lay.addRow(auto_note)
        lay.addWidget(grp_range)

        # ── Methods ───────────────────────────────────────────────────────────
        grp_meth = QGroupBox("Methods")
        m_lay    = QVBoxLayout(grp_meth)
        self.chk_ls      = QCheckBox("Lomb-Scargle")
        self.chk_acf     = QCheckBox("Autocorrelation (ACF)")
        self.chk_wavelet = QCheckBox("Wavelet (Morlet)")
        for chk in [self.chk_ls, self.chk_acf, self.chk_wavelet]:
            chk.setChecked(True)
            m_lay.addWidget(chk)

        tol_row = QFormLayout()
        self.spn_tolerance = QDoubleSpinBox()
        self.spn_tolerance.setRange(0.01, 0.2)
        self.spn_tolerance.setSingleStep(0.01)
        self.spn_tolerance.setDecimals(2)
        self.spn_tolerance.setValue(0.05)
        self.spn_tolerance.setSuffix(" frac")
        tol_row.addRow("CV tolerance:", self.spn_tolerance)
        m_lay.addLayout(tol_row)
        note2 = QLabel("Max fractional difference to call periods equal")
        note2.setObjectName("dim")
        note2.setWordWrap(True)
        m_lay.addWidget(note2)

        self.chk_aliases = QCheckBox("Show 1-day / 1-year alias lines")
        self.chk_aliases.setChecked(True)
        self.chk_aliases.stateChanged.connect(self._on_aliases_changed)
        m_lay.addWidget(self.chk_aliases)
        lay.addWidget(grp_meth)

        # ── Output ────────────────────────────────────────────────────────────
        grp_out = QGroupBox("Output")
        o_lay   = QVBoxLayout(grp_out)
        self.chk_save_csv  = QCheckBox("Save results CSV")
        self.chk_save_csv.setChecked(True)
        self.chk_save_png  = QCheckBox("Save plots as PNG")
        for chk in [self.chk_save_csv, self.chk_save_png]:
            o_lay.addWidget(chk)

        out_row = QHBoxLayout()
        self.le_out_dir = QLineEdit()
        self.le_out_dir.setPlaceholderText("Output folder (default: Desktop)")
        btn_browse_out = QPushButton("…")
        btn_browse_out.setFixedWidth(30)
        btn_browse_out.clicked.connect(self._browse_out)
        out_row.addWidget(self.le_out_dir, 1)
        out_row.addWidget(btn_browse_out)
        o_lay.addLayout(out_row)
        lay.addWidget(grp_out)

        lay.addStretch()
        scroll.setWidget(panel)
        return scroll

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self):
        right  = QWidget()
        r_lay  = QVBoxLayout(right)
        r_lay.setContentsMargins(8, 8, 8, 4)
        r_lay.setSpacing(6)

        # Consensus result box
        grp_cv = QGroupBox("Consensus Result")
        cv_lay = QVBoxLayout(grp_cv)
        cv_lay.setSpacing(4)

        badge_row = QHBoxLayout()
        lbl_p_static = QLabel("Best period:")
        lbl_p_static.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-size:10pt;")
        badge_row.addWidget(lbl_p_static)

        self.lbl_period = QLabel("—")
        self.lbl_period.setObjectName("period")
        badge_row.addWidget(self.lbl_period)

        self.lbl_fap = QLabel("FAP: —")
        self.lbl_fap.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        badge_row.addWidget(self.lbl_fap)

        self.lbl_confidence = QLabel("Confidence: —")
        self.lbl_confidence.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-weight:bold;")
        badge_row.addWidget(self.lbl_confidence)

        self.lbl_n_agree = QLabel("Methods: —/3")
        self.lbl_n_agree.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        badge_row.addWidget(self.lbl_n_agree)
        badge_row.addStretch()
        cv_lay.addLayout(badge_row)

        self.lbl_agree_detail = QLabel("")
        self.lbl_agree_detail.setObjectName("dim")
        cv_lay.addWidget(self.lbl_agree_detail)
        r_lay.addWidget(grp_cv)

        # Tab widget
        self.tabs = QTabWidget()
        r_lay.addWidget(self.tabs, 1)

        # Tab 0 — Light Curve
        lc_tab = QWidget()
        lc_lay = QVBoxLayout(lc_tab)
        lc_lay.setContentsMargins(4, 4, 4, 4)
        self.lc_canvas = LightCurveCanvas()
        lc_lay.addWidget(self.lc_canvas)
        self.tabs.addTab(lc_tab, "📈  Light Curve")

        # Tab 1 — Periodograms
        pg_tab = QWidget()
        pg_lay = QVBoxLayout(pg_tab)
        pg_lay.setContentsMargins(4, 4, 4, 4)
        self.pg_canvas = PeriodogramCanvas()
        self.pg_canvas.period_clicked.connect(self._fold_at_custom_period)
        pg_lay.addWidget(self.pg_canvas)
        click_hint = QLabel("💡 Click on the Lomb-Scargle plot to fold at any period")
        click_hint.setObjectName("dim")
        pg_lay.addWidget(click_hint)
        self.tabs.addTab(pg_tab, "🔊  Periodograms")

        # Tab 2 — Wavelet
        wv_tab = QWidget()
        wv_lay = QVBoxLayout(wv_tab)
        wv_lay.setContentsMargins(4, 4, 4, 4)
        self.wv_canvas = WaveletCanvas()
        wv_lay.addWidget(self.wv_canvas)
        self.tabs.addTab(wv_tab, "🌊  Wavelet")

        # Tab 3 — Top Periods Table
        tp_tab = QWidget()
        tp_lay = QVBoxLayout(tp_tab)
        tp_lay.setContentsMargins(4, 4, 4, 4)

        self.tbl_peaks = QTableWidget(0, 5)
        self.tbl_peaks.setHorizontalHeaderLabels(
            ["Rank", "Period", "Power", "FAP", "Status"])
        self.tbl_peaks.setAlternatingRowColors(True)
        self.tbl_peaks.horizontalHeader().setStretchLastSection(True)
        self.tbl_peaks.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_peaks.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_peaks.doubleClicked.connect(self._fold_from_table_dbl)
        tp_lay.addWidget(self.tbl_peaks, 1)

        tp_btn_row = QHBoxLayout()
        btn_fold_sel = QPushButton("Fold at selected period")
        btn_fold_sel.clicked.connect(self._fold_from_table_btn)
        btn_export   = QPushButton("Export results CSV")
        btn_export.setObjectName("export")
        btn_export.clicked.connect(self._export_csv)
        tp_btn_row.addWidget(btn_fold_sel)
        tp_btn_row.addWidget(btn_export)
        tp_lay.addLayout(tp_btn_row)
        self.tabs.addTab(tp_tab, "📋  Top Periods")

        # Tab 4 — Log
        log_tab = QWidget()
        log_lay = QVBoxLayout(log_tab)
        log_lay.setContentsMargins(4, 4, 4, 4)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        log_lay.addWidget(self.log_box)
        self.tabs.addTab(log_tab, "📋  Log")

        return right

    # ── Event handlers — source combo ─────────────────────────────────────────

    def _on_source_changed(self, idx):
        self.stk_source.setCurrentIndex(idx)

    def _on_aliases_changed(self, state):
        self._show_aliases = bool(state)
        if self._ls_result:
            self.pg_canvas.show_lomb_scargle(
                self._ls_result, show_aliases=self._show_aliases)

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV", "", "CSV files (*.csv *.txt *.dat);;All (*)")
        if path:
            self.le_csv.setText(path)

    def _browse_fits(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self.le_fits.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.le_out_dir.setText(d)

    # ── Load CSV ──────────────────────────────────────────────────────────────

    def _load_csv(self):
        path = self.le_csv.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Error", "Please select a valid CSV file.")
            return
        try:
            t, f, e = load_csv_lightcurve(path)
            if len(t) == 0:
                QMessageBox.warning(self, "Error", "No valid data rows found.")
                return
            self._set_data(t, f, e)
            baseline = t.max() - t.min()
            unit     = guess_time_unit(t, open(path).read(512))
            self.lbl_csv_info.setText(
                f"✓  {len(t)} points  |  baseline: {baseline:.4f} {unit}  "
                f"|  errors: {'yes' if e is not None else 'no'}")
            self._log(f"Loaded CSV: {len(t)} points, baseline={baseline:.4f} {unit}")
        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))

    # ── FITS photometry ───────────────────────────────────────────────────────

    def _run_photometry(self):
        fits_dir = self.le_fits.text().strip()
        if not fits_dir or not os.path.isdir(fits_dir):
            QMessageBox.warning(self, "Error", "Please select a FITS folder.")
            return
        self._log("Starting aperture photometry…")
        self.btn_analyze.setEnabled(False)

        star_x   = self.spn_star_x.value()
        star_y   = self.spn_star_y.value()
        aperture = self.spn_aperture.value()

        class PhotWorker(QThread):
            done     = pyqtSignal(object, object, object)
            log_line = pyqtSignal(str)

            def __init__(self, d, x, y, r):
                super().__init__()
                self.d, self.x, self.y, self.r = d, x, y, r

            def run(self):
                try:
                    t, f, e = compute_from_fits_sequence(
                        self.d, self.x, self.y, self.r,
                        log_callback=self.log_line.emit)
                    self.done.emit(t, f, e)
                except Exception as ex:
                    self.log_line.emit(f"Photometry error: {ex}")
                    self.done.emit(None, None, None)

        self._phot_worker = PhotWorker(fits_dir, star_x, star_y, aperture)
        self._phot_worker.log_line.connect(self._log)
        self._phot_worker.done.connect(self._on_phot_done)
        self._phot_worker.start()

    def _on_phot_done(self, t, f, e):
        self.btn_analyze.setEnabled(True)
        if t is None or len(t) == 0:
            QMessageBox.critical(self, "Photometry failed",
                                 "No data extracted from FITS sequence.")
            return
        self._set_data(t, f, e)
        self._log(f"Photometry complete: {len(t)} frames")

    # ── Manual parse ──────────────────────────────────────────────────────────

    def _parse_manual(self):
        text = self.txt_manual.toPlainText()
        t, f, e = parse_manual_data(text)
        if len(t) == 0:
            QMessageBox.warning(self, "Error", "No valid data parsed.")
            return
        self._set_data(t, f, e)
        self._log(f"Parsed manual data: {len(t)} points")

    # ── Set data and show raw light curve ─────────────────────────────────────

    def _set_data(self, t, f, e):
        norm_map = {0: "median", 1: "mean", 2: "minmax", 3: "none"}
        method   = norm_map.get(self.cmb_norm.currentIndex(), "median")
        t_n, f_n, e_n = normalize_lightcurve(t, f, e, method=method)
        self._times  = t_n
        self._fluxes = f_n
        self._errors = e_n
        self.lc_canvas.show_lightcurve(t_n, f_n, e_n)
        self.tabs.setCurrentIndex(0)
        self.status_lbl.setText(
            f"Data loaded: {len(t_n)} points  |  "
            f"baseline: {t_n.max()-t_n.min():.4f} time units  |  "
            f"norm: {method}")

    # ── Run analysis ──────────────────────────────────────────────────────────

    def _run_analysis(self):
        if self._times is None or len(self._times) < 10:
            QMessageBox.warning(self, "No data",
                                "Please load data first (≥10 points required).")
            return

        self._cancel.clear()
        self.btn_analyze.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.prog_bar.setValue(0)
        self.prog_bar.show()
        self._log("=" * 55)
        self._log("Starting periodicity analysis…")

        cfg = {
            "run_ls":        self.chk_ls.isChecked(),
            "run_acf":       self.chk_acf.isChecked(),
            "run_wavelet":   self.chk_wavelet.isChecked(),
            "min_period":    self.spn_min_period.value(),
            "max_period":    self.spn_max_period.value(),
            "tolerance":     self.spn_tolerance.value(),
            "n_peaks":       10,
            "phase_bins":    50,
            "wavelet_periods": 100,
        }

        self._worker = PeriodWorker(
            cfg, self._times, self._fluxes, self._errors, self._cancel)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log)
        self._worker.ls_ready.connect(self._on_ls_ready)
        self._worker.acf_ready.connect(self._on_acf_ready)
        self._worker.wav_ready.connect(self._on_wav_ready)
        self._worker.fold_ready.connect(self._on_fold_ready)
        self._worker.consensus.connect(self._on_consensus)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_analysis(self):
        self._cancel.set()
        self._log("Cancellation requested…")

    # ── Worker signals ────────────────────────────────────────────────────────

    def _on_progress(self, step, total, msg):
        self.prog_bar.setValue(step)
        self.status_lbl.setText(f"Step {step}/{total}  —  {msg}")

    def _on_ls_ready(self, result):
        self._ls_result = result
        self.pg_canvas.show_lomb_scargle(
            result, show_aliases=self._show_aliases)
        self.tabs.setCurrentIndex(1)

    def _on_acf_ready(self, result):
        self._acf_result = result
        self.pg_canvas.show_acf(result)

    def _on_wav_ready(self, result):
        self._wav_result = result
        self.wv_canvas.show_wavelet(result)

    def _on_fold_ready(self, fold):
        self._fold_result = fold
        self.lc_canvas.show_phase_fold(fold)

    def _on_consensus(self, cv):
        p    = cv["best_period"]
        conf = cv["confidence"]
        n_a  = cv["n_methods_agree"]

        self.lbl_period.setText(f"{p:.6f}")

        if conf == "confirmed":
            self.lbl_confidence.setText("Confidence: CONFIRMED ✓")
            self.lbl_confidence.setStyleSheet(
                f"color:{SIRIL_SUCCESS}; font-weight:bold;")
        elif conf == "candidate":
            self.lbl_confidence.setText("Confidence: CANDIDATE ?")
            self.lbl_confidence.setStyleSheet(
                f"color:{SIRIL_WARNING}; font-weight:bold;")
        else:
            self.lbl_confidence.setText("Confidence: NONE")
            self.lbl_confidence.setStyleSheet(
                f"color:{SIRIL_ERROR}; font-weight:bold;")

        self.lbl_n_agree.setText(f"Methods: {n_a}/3")
        agree_str = ", ".join(
            f"{m1} + {m2}" for m1, m2 in cv.get("agreements", []))
        self.lbl_agree_detail.setText(agree_str or "No method agreement")

    def _on_finished(self, result):
        self.btn_analyze.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.prog_bar.hide()
        self._last_result = result

        if not result.get("success"):
            err = result.get("error", "Unknown error")
            self.status_lbl.setText(f"✗  Error: {err}")
            self._log(f"Analysis failed: {err}")
            return

        cv   = result["consensus"]
        p    = result["best_period"]
        fap  = result.get("best_fap", float("nan"))
        conf = result["confidence"]
        n_a  = cv.get("n_methods_agree", 0)

        fap_str = f"{fap:.2e}" if not np.isnan(fap) else "N/A"
        self.lbl_fap.setText(f"FAP: {fap_str}")
        self.status_lbl.setText(
            f"✓  P = {p:.6f}  |  FAP = {fap_str}  |  "
            f"{conf.upper()} by {n_a} method(s)")
        self._log(f"Analysis complete. Period={p:.6f}, conf={conf}")

        # Populate top periods table
        top_peaks = result.get("top_peaks", [])
        harmonics = {h["period"] for h in result.get("harmonics", [])}
        self._populate_peaks_table(top_peaks, p, harmonics)

        # Auto-save CSV if requested
        if self.chk_save_csv.isChecked():
            out_dir = self.le_out_dir.text().strip()
            if not out_dir:
                out_dir = os.path.expanduser("~/Desktop")
            try:
                saved = save_results_csv(
                    out_dir, result, top_peaks,
                    self._times, self._fluxes,
                    ls_result=self._ls_result,
                    acf_result=self._acf_result,
                    harmonics=result.get("harmonics", []))
                self._log(f"Results saved: {saved}")
            except Exception as ex:
                self._log(f"CSV save error: {ex}")

        # Auto-save PNG if requested
        if self.chk_save_png.isChecked():
            self._save_plots_png()

    # ── Table helpers ─────────────────────────────────────────────────────────

    def _populate_peaks_table(self, top_peaks, best_period, harmonics):
        self.tbl_peaks.setRowCount(0)
        tol = 0.05
        for rank, peak in enumerate(top_peaks[:10], 1):
            p    = peak["period"]
            fap  = peak["fap"]
            is_best = abs(p - best_period) / max(best_period, 1e-10) < tol
            is_harm = p in harmonics

            status = "CONFIRMED" if is_best else (
                "harmonic?" if is_harm else "candidate")

            row = self.tbl_peaks.rowCount()
            self.tbl_peaks.insertRow(row)

            items = [
                QTableWidgetItem(str(rank)),
                QTableWidgetItem(f"{p:.8f}"),
                QTableWidgetItem(f"{peak['power']:.5f}"),
                QTableWidgetItem(f"{fap:.3e}" if not np.isnan(fap) else "N/A"),
                QTableWidgetItem(status),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if is_best:
                    item.setForeground(QColor(SIRIL_ACCENT))
                elif is_harm:
                    item.setForeground(QColor(SIRIL_WARNING))
                self.tbl_peaks.setItem(row, col, item)

        self.tbl_peaks.resizeColumnsToContents()

    def _get_selected_period(self):
        rows = self.tbl_peaks.selectedItems()
        if not rows:
            return None
        row = self.tbl_peaks.currentRow()
        item = self.tbl_peaks.item(row, 1)
        if item:
            try:
                return float(item.text())
            except ValueError:
                pass
        return None

    def _fold_from_table_btn(self):
        p = self._get_selected_period()
        if p:
            self._fold_at_custom_period(p)

    def _fold_from_table_dbl(self, idx):
        p = self._get_selected_period()
        if p:
            self._fold_at_custom_period(p)

    def _fold_at_custom_period(self, period: float):
        if self._times is None:
            return
        fold = phase_fold(self._times, self._fluxes, period)
        self.lc_canvas.show_phase_fold(fold)
        self.lbl_period.setText(f"{period:.6f}")
        self._log(f"Phase-folded at P={period:.6f} (custom)")
        self.tabs.setCurrentIndex(0)

    # ── Export / save ─────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._last_result.get("success"):
            QMessageBox.warning(self, "No results",
                                "Run an analysis first.")
            return
        out_dir = self.le_out_dir.text().strip()
        if not out_dir:
            out_dir = os.path.expanduser("~/Desktop")
        try:
            saved = save_results_csv(
                out_dir, self._last_result,
                self._last_result.get("top_peaks", []),
                self._times, self._fluxes,
                ls_result=self._ls_result,
                acf_result=self._acf_result,
                harmonics=self._last_result.get("harmonics", []))
            QMessageBox.information(self, "Saved", f"Saved to:\n{saved}")
            self._log(f"Exported: {saved}")
        except Exception as ex:
            QMessageBox.critical(self, "Save error", str(ex))

    def _save_plots_png(self):
        out_dir = self.le_out_dir.text().strip()
        if not out_dir:
            out_dir = os.path.expanduser("~/Desktop")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            for name, canvas in [("lc", self.lc_canvas),
                                  ("periodogram", self.pg_canvas),
                                  ("wavelet", self.wv_canvas)]:
                path = os.path.join(out_dir, f"period_{name}_{ts}.png")
                canvas.fig.savefig(path, dpi=150,
                                   facecolor=SIRIL_BG, bbox_inches="tight")
                self._log(f"Plot saved: {path}")
        except Exception as ex:
            self._log(f"PNG save error: {ex}")

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}]  {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
