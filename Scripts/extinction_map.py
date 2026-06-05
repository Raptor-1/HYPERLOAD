r"""
extinction_map.py  —  Script #12
Local Extinction Map for Siril
Measures atmospheric extinction coefficient using Bouguer's Law.
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

try:
    s.ensure_installed("astroquery")
    HAS_ASTROQUERY = True
except Exception:
    HAS_ASTROQUERY = False

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import csv
import glob
import json
import threading
from datetime import datetime

# ── Numeric / science ─────────────────────────────────────────────────────────
import numpy as np
from scipy.stats import linregress
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from photutils.detection import DAOStarFinder
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                 aperture_photometry)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ── PyQt6 ─────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSplitter, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ── Script directory (for shared imports) ─────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# =============================================================================
# SIRIL DARK THEME
# =============================================================================

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
QLabel#section {{ color: {SIRIL_SECTION}; font-weight: bold; padding-top: 6px; }}
QLabel#dim    {{ color: {SIRIL_TEXT_DIM}; font-size: 9pt; }}
QLabel#ok     {{ color: {SIRIL_SUCCESS}; font-weight: bold; }}
QLabel#err    {{ color: {SIRIL_ERROR};   font-weight: bold; }}
QLabel#warn   {{ color: {SIRIL_WARNING}; font-weight: bold; }}
QLabel#value  {{
    color: {SIRIL_ACCENT};
    font-size: 20pt;
    font-weight: bold;
    padding: 2px;
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

# =============================================================================
# EXTINCTION REFERENCE TABLE
# =============================================================================

EXTINCTION_REFERENCE = {
    "photometric": (0.10, 0.25, SIRIL_SUCCESS,
                    "Photometric night — flat calibration reliable"),
    "good":        (0.25, 0.40, SIRIL_ACCENT,
                    "Good night — minor corrections may be needed"),
    "variable":    (0.40, 0.60, SIRIL_WARNING,
                    "Variable transparency — check for cloud passages"),
    "poor":        (0.60, 1.00, SIRIL_ERROR,
                    "Poor night — significant cloud/haze"),
}

QUALITY_DESCRIPTIONS = {
    "photometric": "Excellent night. k < 0.25, very stable.",
    "good":        "Good night. Minor transparency variation.",
    "variable":    "Variable transparency. Check timeline for gaps.",
    "poor":        "Poor night. Significant cloud or haze present.",
}

# =============================================================================
# AIRMASS FUNCTIONS
# =============================================================================

def compute_airmass_simple(altitude_deg: float) -> float:
    alt_rad = np.radians(np.clip(altitude_deg, 1.0, 89.9))
    return float(1.0 / np.sin(alt_rad))


def compute_airmass_hardie(altitude_deg: float) -> float:
    z_rad   = np.radians(90.0 - np.clip(altitude_deg, 1.0, 89.9))
    sec_z   = 1.0 / np.cos(z_rad)
    x       = (sec_z
               - 0.0018167 * (sec_z - 1)
               - 0.002875  * (sec_z - 1)**2
               - 0.0008083 * (sec_z - 1)**3)
    return float(x)


def get_altitude_from_header(header,
                              ra_deg: float = None,
                              dec_deg: float = None) -> float | None:
    for kw in ["OBJCTALT", "ALTITUDE", "ALT"]:
        val = header.get(kw)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    try:
        date_obs = header.get("DATE-OBS", "")
        if not date_obs:
            return None

        t = Time(date_obs, format="isot", scale="utc")

        lat  = (header.get("SITELAT")  or header.get("OBSLAT")  or
                header.get("LAT-OBS")  or header.get("LATITUDE"))
        lon  = (header.get("SITELONG") or header.get("OBSLON")  or
                header.get("LONG-OBS") or header.get("LONGITUD"))
        elev = float(header.get("SITEELEV", header.get("ELEVATION", 0)))

        if lat is None or lon is None:
            return None

        location = EarthLocation(lat=float(lat)*u.deg,
                                  lon=float(lon)*u.deg,
                                  height=float(elev)*u.m)

        ra  = ra_deg  or header.get("RA")  or header.get("OBJCTRA")
        dec = dec_deg or header.get("DEC") or header.get("OBJCTDEC")

        if ra is None or dec is None:
            return None

        coord  = SkyCoord(ra=float(ra)*u.deg, dec=float(dec)*u.deg)
        altaz  = AltAz(obstime=t, location=location)
        target = coord.transform_to(altaz)
        return float(target.alt.deg)

    except Exception:
        return None

# =============================================================================
# PHOTOMETRY FUNCTIONS
# =============================================================================

def measure_instrumental_magnitudes(image: np.ndarray,
                                     star_positions: list,
                                     aperture_r: float = 8.0,
                                     annulus_r_in: float = 12.0,
                                     annulus_r_out: float = 18.0) -> list:
    if not star_positions:
        return []

    apers  = CircularAperture(star_positions, r=aperture_r)
    annuli = CircularAnnulus(star_positions,
                              r_in=annulus_r_in, r_out=annulus_r_out)

    phot = aperture_photometry(image.astype(np.float64), apers)
    bkg  = aperture_photometry(image.astype(np.float64), annuli)

    results = []
    for i in range(len(star_positions)):
        raw_flux = float(phot["aperture_sum"][i])
        bkg_mean = float(bkg["aperture_sum"][i]) / annuli.area
        net_flux = raw_flux - bkg_mean * apers.area

        if net_flux <= 0:
            results.append(None)
        else:
            results.append(-2.5 * np.log10(net_flux))

    return results


def detect_reference_stars(image: np.ndarray,
                             n_stars: int = 20,
                             threshold_sigma: float = 10.0,
                             border_margin: int = 50) -> list:
    h, w = image.shape
    _, med, std = sigma_clipped_stats(image, sigma=3.0)
    if std <= 0:
        return []

    daofind = DAOStarFinder(
        fwhm=5.0, threshold=threshold_sigma * std,
        sharplo=0.3, sharphi=0.9,
        roundlo=-0.5, roundhi=0.5
    )
    sources = daofind(image - med)
    if sources is None or len(sources) == 0:
        return []

    sources.sort("peak")
    sources.reverse()

    sat_limit = 0.9 * float(np.percentile(image, 99.99))
    positions = []

    for src in sources:
        x, y = float(src["xcentroid"]), float(src["ycentroid"])
        if (x < border_margin or x > w - border_margin or
                y < border_margin or y > h - border_margin):
            continue
        if float(src["peak"]) + med > sat_limit:
            continue
        positions.append((x, y))
        if len(positions) >= n_stars:
            break

    return positions

# =============================================================================
# EXTINCTION FITTING
# =============================================================================

def fit_bouguer_law(airmasses: np.ndarray,
                     magnitudes: np.ndarray,
                     errors: np.ndarray = None) -> dict:
    if len(airmasses) < 3:
        return {"k": 0.0, "k_err": 0.0, "r2": 0.0,
                "quality": "insufficient_data"}

    valid = np.isfinite(airmasses) & np.isfinite(magnitudes)
    X = airmasses[valid]
    m = magnitudes[valid]

    if len(X) < 3:
        return {"k": 0.0, "k_err": 0.0, "r2": 0.0,
                "quality": "insufficient_data"}

    slope, intercept, r, p_val, se = linregress(X, m)

    r2    = r ** 2
    k     = float(slope)
    k_err = float(se)

    m_fit = slope * X + intercept
    rms   = float(np.sqrt(np.mean((m - m_fit)**2)))

    if r2 > 0.90 and 0.05 < k < 0.60:
        quality = "photometric"
    elif r2 > 0.75 and 0.05 < k < 0.80:
        quality = "good"
    elif r2 > 0.50:
        quality = "variable"
    else:
        quality = "poor"

    return {
        "k":              round(k, 4),
        "k_err":          round(k_err, 4),
        "m_true_est":     round(float(intercept), 4),
        "r2":             round(r2, 4),
        "rms":            round(rms, 4),
        "is_photometric": quality == "photometric",
        "quality":        quality,
        "n_points":       int(valid.sum()),
    }


def compute_per_frame_transparency(airmasses: np.ndarray,
                                    magnitudes: np.ndarray,
                                    k: float,
                                    m_true: float) -> np.ndarray:
    expected = m_true + k * airmasses
    delta    = magnitudes - expected
    score    = 100.0 * np.exp(-delta / 0.5)
    return np.clip(score, 0, 100).astype(np.float32)


def extinction_from_single_frame(image: np.ndarray,
                                   header,
                                   n_stars: int = 30,
                                   log_callback=None) -> dict | None:
    from astropy.wcs import WCS

    positions = detect_reference_stars(image, n_stars=n_stars,
                                        threshold_sigma=8.0)
    if len(positions) < 5:
        if log_callback:
            log_callback("  Insufficient stars for single-frame extinction")
        return None

    inst_mags = measure_instrumental_magnitudes(image, positions)

    try:
        wcs    = WCS(header)
        radecs = [wcs.pixel_to_world(x, y) for x, y in positions]
    except Exception:
        if log_callback:
            log_callback("  No WCS in header — cannot compute star altitudes")
        return None

    altitudes = []
    for sky in radecs:
        alt = get_altitude_from_header(
            header,
            ra_deg=float(sky.ra.deg),
            dec_deg=float(sky.dec.deg))
        altitudes.append(alt)

    valid_X = []
    valid_m = []
    for alt, mag in zip(altitudes, inst_mags):
        if alt is None or mag is None:
            continue
        if alt < 15.0:
            continue
        valid_X.append(compute_airmass_hardie(alt))
        valid_m.append(mag)

    if len(valid_X) < 5:
        if log_callback:
            log_callback(
                f"  Only {len(valid_X)} valid stars — "
                "need wider altitude range or more stars")
        return None

    if log_callback:
        log_callback(
            f"  Single-frame extinction from {len(valid_X)} stars  "
            f"airmass range: {min(valid_X):.2f} – {max(valid_X):.2f}")

    return fit_bouguer_law(np.array(valid_X), np.array(valid_m))

# =============================================================================
# WORKER THREAD
# =============================================================================

class ExtinctionWorker(QThread):
    progress   = pyqtSignal(int, int, str)
    log_line   = pyqtSignal(str)
    frame_done = pyqtSignal(dict)
    finished   = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg  = self.config
            mode = cfg.get("mode", "time_series")

            if mode == "single_frame":
                self._run_single_frame(cfg)
            else:
                self._run_time_series(cfg)

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    # ── Time-series ───────────────────────────────────────────────────────────

    def _run_time_series(self, cfg: dict):
        files = cfg["fits_files"]
        n     = len(files)

        if n < 5:
            self.finished.emit({
                "success": False,
                "error":   "Need at least 5 frames for time-series extinction"
            })
            return

        # Step 1: Reference stars in first frame
        self.progress.emit(1, n+3, "Detecting reference stars...")
        with fits.open(files[0]) as hdul:
            first_data   = hdul[0].data.astype(np.float32)
            first_header = hdul[0].header.copy()

        if first_data.ndim == 3:
            first_data = (first_data[0] if first_data.shape[0] <= 4
                          else 0.299*first_data[0]+0.587*first_data[1]+
                               0.114*first_data[2])

        positions = detect_reference_stars(
            first_data,
            n_stars=cfg.get("n_stars", 20),
            threshold_sigma=cfg.get("threshold_sigma", 10.0)
        )
        self.log_line.emit(f"Detected {len(positions)} reference stars")

        if len(positions) < 3:
            self.finished.emit({
                "success": False,
                "error":   "Too few reference stars detected. "
                           "Lower detection threshold or use a different frame."
            })
            return

        if self._cancel.is_set():
            self._abort(); return

        # Step 2: Measure per frame
        self.log_line.emit("Measuring photometry in each frame...")
        self.log_line.emit(
            "NOTE: Frames should be registered before measuring extinction.")

        all_airmasses   = []
        all_mag_medians = []
        all_jds         = []
        all_altitudes   = []

        for fi, fits_path in enumerate(files):
            if self._cancel.is_set():
                self._abort(); return

            self.progress.emit(fi+1, n+3,
                f"Frame {fi+1}/{n}: {os.path.basename(fits_path)}")

            try:
                with fits.open(fits_path) as hdul:
                    data   = hdul[0].data.astype(np.float32)
                    header = hdul[0].header.copy()

                if data.ndim == 3:
                    data = (data[0] if data.shape[0] <= 4
                            else 0.299*data[0]+0.587*data[1]+
                                 0.114*data[2])

                target_ra  = cfg.get("target_ra")  or header.get("RA")  or header.get("OBJCTRA")
                target_dec = cfg.get("target_dec") or header.get("DEC") or header.get("OBJCTDEC")

                alt = get_altitude_from_header(
                    header,
                    ra_deg  = float(target_ra)  if target_ra  else None,
                    dec_deg = float(target_dec) if target_dec else None
                )

                if alt is None:
                    self.log_line.emit(
                        f"  Frame {fi+1}: no altitude info — skipping")
                    continue

                if alt < 10.0:
                    self.log_line.emit(
                        f"  Frame {fi+1}: altitude {alt:.1f}° too low — skipping")
                    continue

                X = compute_airmass_hardie(alt)

                mags = measure_instrumental_magnitudes(
                    data, positions,
                    aperture_r    = cfg.get("aperture_r",    8.0),
                    annulus_r_in  = cfg.get("annulus_r_in",  12.0),
                    annulus_r_out = cfg.get("annulus_r_out", 18.0)
                )

                valid_mags = [m for m in mags if m is not None]
                if not valid_mags:
                    continue

                med_mag = float(np.median(valid_mags))

                date_obs = header.get("DATE-OBS", "")
                if date_obs:
                    try:
                        jd = float(Time(date_obs,
                                        format="isot", scale="utc").jd)
                    except Exception:
                        jd = float(fi)
                else:
                    jd = float(fi)

                all_airmasses.append(X)
                all_mag_medians.append(med_mag)
                all_jds.append(jd)
                all_altitudes.append(alt)

                self.frame_done.emit({
                    "frame_idx": fi,
                    "jd":        jd,
                    "airmass":   X,
                    "altitude":  alt,
                    "mag":       med_mag,
                    "n_stars":   len(valid_mags),
                })

                self.log_line.emit(
                    f"  [{fi+1}/{n}] alt={alt:.1f}°  X={X:.3f}  "
                    f"mag={med_mag:.3f}  stars={len(valid_mags)}")

            except Exception as e:
                self.log_line.emit(f"  Frame {fi+1} error: {e}")

        if len(all_airmasses) < 5:
            self.finished.emit({
                "success": False,
                "error":   f"Only {len(all_airmasses)} usable frames. "
                           "Need at least 5 with valid altitude info."
            })
            return

        if self._cancel.is_set():
            self._abort(); return

        # Airmass range check
        X_arr = np.array(all_airmasses)
        delta_x = float(X_arr.max() - X_arr.min())
        if delta_x < 0.2:
            self.log_line.emit(
                f"WARNING: Airmass range only ΔX={delta_x:.3f} "
                "(need > 0.3 for reliable fit). "
                "Consider single-frame mode for near-zenith targets.")

        # Step 3: Fit Bouguer's Law
        self.progress.emit(n+2, n+3, "Fitting extinction...")
        m_arr = np.array(all_mag_medians)
        t_arr = np.array(all_jds)

        result = fit_bouguer_law(X_arr, m_arr)
        self.log_line.emit(
            f"Extinction k = {result['k']:.4f} ± {result['k_err']:.4f} "
            f"mag/airmass  R²={result['r2']:.3f}  "
            f"Quality: {result['quality']}")

        if result.get("k", 0) != 0:
            transparency = compute_per_frame_transparency(
                X_arr, m_arr,
                result["k"], result["m_true_est"]
            )
        else:
            transparency = np.full(len(X_arr), 50.0)

        # Step 4: Save CSV
        self.progress.emit(n+3, n+3, "Saving results...")
        out_dir  = cfg.get("output_dir", ".")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "extinction_results.csv")
        self._save_csv(
            csv_path, all_jds, all_altitudes,
            all_airmasses, all_mag_medians,
            transparency.tolist())
        self.log_line.emit(f"CSV saved: {csv_path}")

        self.finished.emit({
            "success":      True,
            "k":            result["k"],
            "k_err":        result["k_err"],
            "r2":           result["r2"],
            "quality":      result["quality"],
            "n_frames":     len(all_airmasses),
            "airmasses":    all_airmasses,
            "magnitudes":   all_mag_medians,
            "jds":          all_jds,
            "altitudes":    all_altitudes,
            "transparency": transparency.tolist(),
            "bouguer_fit":  result,
            "csv_path":     csv_path,
        })

    # ── Single-frame ──────────────────────────────────────────────────────────

    def _run_single_frame(self, cfg: dict):
        self.progress.emit(1, 3, "Loading image...")
        with fits.open(cfg["fits_path"]) as hdul:
            data   = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()

        if data.ndim == 3:
            data = (data[0] if data.shape[0] <= 4
                    else 0.299*data[0]+0.587*data[1]+0.114*data[2])

        self.progress.emit(2, 3, "Computing extinction...")
        result = extinction_from_single_frame(
            data, header,
            n_stars=cfg.get("n_stars", 30),
            log_callback=self.log_line.emit
        )

        if result is None:
            self.finished.emit({
                "success": False,
                "error":   "Could not compute extinction from single frame. "
                           "Need WCS + observer location in FITS header."
            })
            return

        self.log_line.emit(
            f"Extinction k = {result['k']:.4f} ± {result['k_err']:.4f}  "
            f"R²={result['r2']:.3f}  Quality: {result['quality']}")

        self.progress.emit(3, 3, "Done")
        self.finished.emit({
            "success": True,
            "mode":    "single_frame",
            **result,
        })

    def _save_csv(self, path, jds, alts, airmasses, mags, transparency):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "JD", "altitude_deg", "airmass",
                "inst_mag", "transparency_pct"
            ])
            for jd, alt, X, m, t in zip(
                    jds, alts, airmasses, mags, transparency):
                writer.writerow([
                    round(jd, 6), round(alt, 2),
                    round(X, 4), round(m, 4), round(t, 1)
                ])

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()

# =============================================================================
# MATPLOTLIB CANVASES
# =============================================================================

class BouguerCanvas(FigureCanvasQTAgg):
    """Bouguer plot: instrumental magnitude vs airmass."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), facecolor=SIRIL_BG)
        self.ax  = self.fig.add_subplot(111)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)
        self._xs = []
        self._ys = []

    def _style(self):
        self.ax.set_facecolor(SIRIL_BG2)
        self.ax.tick_params(colors=SIRIL_TEXT, labelsize=8)
        for spine in ["bottom", "left"]:
            self.ax.spines[spine].set_color(SIRIL_BORDER)
        for spine in ["top", "right"]:
            self.ax.spines[spine].set_visible(False)
        self.ax.set_xlabel("Airmass X", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_ylabel("Instrumental magnitude", color=SIRIL_TEXT_DIM,
                           fontsize=8)
        self.ax.set_title("Bouguer plot  (m = m₀ + k·X)",
                           color=SIRIL_SECTION, fontsize=9)

    def add_point(self, X: float, m: float):
        self._xs.append(X)
        self._ys.append(m)
        if len(self._xs) % 5 == 0:
            self._redraw_scatter()

    def show_full(self, airmasses: list, magnitudes: list,
                   fit: dict, transparency: list = None):
        self._xs = list(airmasses)
        self._ys = list(magnitudes)
        self.ax.clear()
        self._style()

        X = np.array(airmasses)
        m = np.array(magnitudes)

        if transparency and len(transparency) == len(X):
            t = np.array(transparency)
            colors = [f"#{int(255*(1-ti/100)):02x}"
                      f"{int(255*ti/100):02x}44" for ti in t]
        else:
            colors = [SIRIL_ACCENT] * len(X)

        self.ax.scatter(X, m, c=colors, s=20, zorder=3, alpha=0.8)

        if fit and fit.get("k", 0) != 0:
            X_line = np.linspace(X.min()-0.05, X.max()+0.05, 100)
            m_line = fit["m_true_est"] + fit["k"] * X_line
            self.ax.plot(X_line, m_line, color="#ff6b35",
                         linewidth=2, zorder=4,
                         label=f"k={fit['k']:.4f} mag/airmass  "
                               f"R²={fit['r2']:.3f}")
            m_upper = (fit["m_true_est"] + fit["k_err"]) + fit["k"] * X_line
            m_lower = (fit["m_true_est"] - fit["k_err"]) + fit["k"] * X_line
            self.ax.fill_between(X_line, m_lower, m_upper,
                                  alpha=0.15, color="#ff6b35")

        quality_colors = {
            "photometric": SIRIL_SUCCESS,
            "good":        SIRIL_ACCENT,
            "variable":    SIRIL_WARNING,
            "poor":        SIRIL_ERROR,
        }
        q   = fit.get("quality", "unknown") if fit else "unknown"
        clr = quality_colors.get(q, SIRIL_SECTION)

        if fit and fit.get("k", 0) != 0:
            self.ax.set_title(
                f"Bouguer plot  |  {q.upper()}  |  "
                f"k = {fit['k']:.4f} ± {fit['k_err']:.4f}",
                color=clr, fontsize=9)
        else:
            self.ax.set_title("Bouguer plot", color=SIRIL_SECTION, fontsize=9)

        self.ax.invert_yaxis()

        if len(X) > 0:
            self.ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                           labelcolor=SIRIL_TEXT, fontsize=7)
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def _redraw_scatter(self):
        self.ax.clear()
        self._style()
        if self._xs:
            self.ax.scatter(self._xs, self._ys,
                             color=SIRIL_ACCENT, s=15, alpha=0.7)
            self.ax.invert_yaxis()
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def clear_plot(self):
        self._xs = []
        self._ys = []
        self.ax.clear()
        self._style()
        self.fig.tight_layout(pad=0.4)
        self.draw()


class TransparencyCanvas(FigureCanvasQTAgg):
    """Transparency score 0-100 vs time, with altitude overlay."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 3), facecolor=SIRIL_BG)
        self.ax  = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style(self):
        self.ax.set_facecolor(SIRIL_BG2)
        self.ax.tick_params(colors=SIRIL_TEXT, labelsize=8)
        self.ax2.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for spine in ["bottom", "left"]:
            self.ax.spines[spine].set_color(SIRIL_BORDER)
        for spine in ["top", "right"]:
            self.ax.spines[spine].set_color(SIRIL_BORDER)
        self.ax.set_ylabel("Transparency (%)", color=SIRIL_SUCCESS, fontsize=8)
        self.ax2.set_ylabel("Altitude (°)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_title("Session transparency timeline",
                           color=SIRIL_SECTION, fontsize=9)

    def show_timeline(self, jds: list, transparency: list,
                       altitudes: list = None):
        self.ax.clear()
        self.ax2.clear()
        self._style()

        t  = np.array(jds)
        tr = np.array(transparency)

        t_hours = (t - t[0]) * 24.0 if len(t) > 0 else t

        self.ax.fill_between(t_hours, tr, alpha=0.4, color=SIRIL_SUCCESS)
        self.ax.plot(t_hours, tr, color=SIRIL_SUCCESS,
                      linewidth=1.5, zorder=3)
        self.ax.axhline(80, color=SIRIL_WARNING, linewidth=0.8,
                         linestyle="--", alpha=0.6, label="80% threshold")
        self.ax.set_ylim(0, 105)
        self.ax.set_xlabel("Time from session start (hours)",
                            color=SIRIL_TEXT_DIM, fontsize=8)

        if altitudes and len(altitudes) == len(t_hours):
            self.ax2.plot(t_hours, altitudes, color=SIRIL_TEXT_DIM,
                           linewidth=1.0, linestyle=":", alpha=0.6,
                           label="Target altitude")
            self.ax2.set_ylim(0, 90)

        self.ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                       labelcolor=SIRIL_TEXT, fontsize=7, loc="lower left")
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self.ax2.clear()
        self._style()
        self.fig.tight_layout(pad=0.4)
        self.draw()

# =============================================================================
# TITLE BAR WIDGET
# =============================================================================

class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)

        lbl_left = QLabel("🌫  Extinction Map  —  Siril")
        lbl_left.setStyleSheet(
            f"color: {SIRIL_ACCENT}; font-size: 13pt; font-weight: bold;")

        lbl_right = QLabel("v1.0  |  Bouguer's Law atmospheric extinction")
        lbl_right.setStyleSheet(
            f"color: {SIRIL_TEXT_DIM}; font-size: 9pt;")

        lay.addWidget(lbl_left)
        lay.addStretch()
        lay.addWidget(lbl_right)

        self.setStyleSheet(f"background: {SIRIL_BG3}; "
                           f"border-bottom: 1px solid {SIRIL_BORDER};")

# =============================================================================
# LEFT PANEL — SETTINGS
# =============================================================================

class SettingsPanel(QWidget):

    scan_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setMaximumWidth(380)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        self._lay = QVBoxLayout(inner)
        self._lay.setSpacing(8)
        self._lay.setContentsMargins(6, 6, 6, 6)

        self._build_mode()
        self._build_time_series_inputs()
        self._build_single_frame_inputs()
        self._build_coordinates()
        self._build_location()
        self._build_detection()
        self._build_output()

        self._lay.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._update_mode()

    # ── Mode ──────────────────────────────────────────────────────────────────

    def _build_mode(self):
        grp = QGroupBox("Mode")
        lay = QVBoxLayout(grp)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Time series (recommended)",
            "Single wide-field frame",
        ])
        self.mode_combo.currentIndexChanged.connect(self._update_mode)
        lay.addWidget(self.mode_combo)

        self._lay.addWidget(grp)

    # ── Time-series inputs ────────────────────────────────────────────────────

    def _build_time_series_inputs(self):
        self.grp_ts = QGroupBox("FITS Folder  (time series)")
        lay = QVBoxLayout(self.grp_ts)

        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder with calibrated frames…")
        btn = QPushButton("Browse")
        btn.setFixedWidth(70)
        btn.clicked.connect(self._browse_folder)
        row.addWidget(self.folder_edit)
        row.addWidget(btn)
        lay.addLayout(row)

        hint = QLabel("Calibrated frames, time-sorted, DATE-OBS required")
        hint.setObjectName("dim")
        lay.addWidget(hint)

        scan_btn = QPushButton("🔍  Scan folder")
        scan_btn.setObjectName("primary")
        scan_btn.clicked.connect(self.scan_requested.emit)
        lay.addWidget(scan_btn)

        self.scan_info = QLabel("")
        self.scan_info.setObjectName("dim")
        self.scan_info.setWordWrap(True)
        lay.addWidget(self.scan_info)

        self.scan_warn = QLabel("")
        self.scan_warn.setObjectName("warn")
        self.scan_warn.setWordWrap(True)
        self.scan_warn.setVisible(False)
        lay.addWidget(self.scan_warn)

        self._lay.addWidget(self.grp_ts)

    # ── Single-frame inputs ───────────────────────────────────────────────────

    def _build_single_frame_inputs(self):
        self.grp_sf = QGroupBox("FITS File  (single frame)")
        lay = QVBoxLayout(self.grp_sf)

        row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a single FITS file…")
        btn = QPushButton("Browse")
        btn.setFixedWidth(70)
        btn.clicked.connect(self._browse_file)
        row.addWidget(self.file_edit)
        row.addWidget(btn)
        lay.addLayout(row)

        hint = QLabel("Requires WCS + observer location in header")
        hint.setObjectName("dim")
        lay.addWidget(hint)

        self._lay.addWidget(self.grp_sf)

    # ── Target coordinates ────────────────────────────────────────────────────

    def _build_coordinates(self):
        grp = QGroupBox("Target coordinates  (if not in headers)")
        form = QFormLayout(grp)

        self.ra_spin = QDoubleSpinBox()
        self.ra_spin.setRange(0, 360)
        self.ra_spin.setDecimals(4)
        self.ra_spin.setSuffix(" °")

        self.dec_spin = QDoubleSpinBox()
        self.dec_spin.setRange(-90, 90)
        self.dec_spin.setDecimals(4)
        self.dec_spin.setSuffix(" °")

        form.addRow("RA:", self.ra_spin)
        form.addRow("Dec:", self.dec_spin)

        hint = QLabel("Leave at 0 to use FITS header RA/Dec")
        hint.setObjectName("dim")
        form.addRow(hint)

        self._lay.addWidget(grp)

    # ── Observer location ─────────────────────────────────────────────────────

    def _build_location(self):
        grp = QGroupBox("Observer location  (if not in headers)")
        form = QFormLayout(grp)

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90, 90)
        self.lat_spin.setDecimals(4)
        self.lat_spin.setSuffix(" °")

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180, 180)
        self.lon_spin.setDecimals(4)
        self.lon_spin.setSuffix(" °")

        self.elev_spin = QSpinBox()
        self.elev_spin.setRange(0, 5000)
        self.elev_spin.setSuffix(" m")

        form.addRow("Latitude:", self.lat_spin)
        form.addRow("Longitude:", self.lon_spin)
        form.addRow("Elevation:", self.elev_spin)

        if HAS_EQUIPMENT_MANAGER:
            btn = QPushButton("Load from equipment profile")
            btn.clicked.connect(self._load_from_profile)
            form.addRow(btn)

        self._lay.addWidget(grp)

    # ── Detection settings ────────────────────────────────────────────────────

    def _build_detection(self):
        grp = QGroupBox("Detection settings")
        form = QFormLayout(grp)

        self.n_stars_spin = QSpinBox()
        self.n_stars_spin.setRange(5, 50)
        self.n_stars_spin.setValue(20)

        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(5.0, 20.0)
        self.thresh_spin.setValue(10.0)
        self.thresh_spin.setSuffix(" σ")

        self.aperture_spin = QDoubleSpinBox()
        self.aperture_spin.setRange(3.0, 20.0)
        self.aperture_spin.setValue(8.0)
        self.aperture_spin.setSuffix(" px")

        form.addRow("Stars per frame:", self.n_stars_spin)
        form.addRow("Detection threshold:", self.thresh_spin)
        form.addRow("Aperture radius:", self.aperture_spin)

        self._lay.addWidget(grp)

    # ── Output ────────────────────────────────────────────────────────────────

    def _build_output(self):
        grp = QGroupBox("Output")
        lay = QVBoxLayout(grp)

        row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output folder…")
        btn = QPushButton("Browse")
        btn.setFixedWidth(70)
        btn.clicked.connect(self._browse_output)
        row.addWidget(self.out_edit)
        row.addWidget(btn)
        lay.addLayout(row)

        self.export_csv_chk = QCheckBox("Export CSV")
        self.export_csv_chk.setChecked(True)
        lay.addWidget(self.export_csv_chk)

        self._lay.addWidget(grp)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_mode(self):
        ts = self.mode_combo.currentIndex() == 0
        self.grp_ts.setVisible(ts)
        self.grp_sf.setVisible(not ts)

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self.folder_edit.setText(d)
            if not self.out_edit.text():
                self.out_edit.setText(d)

    def _browse_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select FITS file", "",
            "FITS files (*.fits *.fit *.fts *.FITS *.FIT *.FTS)")
        if f:
            self.file_edit.setText(f)
            if not self.out_edit.text():
                self.out_edit.setText(os.path.dirname(f))

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_edit.setText(d)

    def _load_from_profile(self):
        try:
            profiles = load_all_profiles()
            if not profiles:
                QMessageBox.information(self, "Equipment Manager",
                                        "No profiles found.")
                return
            profile = profiles[0]
            lat  = profile.get("latitude",  0.0)
            lon  = profile.get("longitude", 0.0)
            elev = profile.get("elevation", 0)
            self.lat_spin.setValue(float(lat))
            self.lon_spin.setValue(float(lon))
            self.elev_spin.setValue(int(elev))
        except Exception as e:
            QMessageBox.warning(self, "Equipment Manager",
                                f"Could not load profile: {e}")

    def update_scan_info(self, n_files: int, warnings: list):
        if n_files > 0:
            self.scan_info.setText(f"Found {n_files} FITS files")
        else:
            self.scan_info.setText("No FITS files found")

        if warnings:
            self.scan_warn.setText("  ".join(warnings))
            self.scan_warn.setVisible(True)
        else:
            self.scan_warn.setVisible(False)

    def get_config(self) -> dict:
        mode = ("time_series" if self.mode_combo.currentIndex() == 0
                else "single_frame")

        ra  = self.ra_spin.value()
        dec = self.dec_spin.value()
        lat = self.lat_spin.value()
        lon = self.lon_spin.value()

        cfg = {
            "mode":             mode,
            "n_stars":          self.n_stars_spin.value(),
            "threshold_sigma":  self.thresh_spin.value(),
            "aperture_r":       self.aperture_spin.value(),
            "annulus_r_in":     self.aperture_spin.value() * 1.5,
            "annulus_r_out":    self.aperture_spin.value() * 2.25,
            "output_dir":       self.out_edit.text() or ".",
            "export_csv":       self.export_csv_chk.isChecked(),
            "target_ra":        ra  if ra  != 0.0 else None,
            "target_dec":       dec if dec != 0.0 else None,
            "site_lat":         lat if lat != 0.0 else None,
            "site_lon":         lon if lon != 0.0 else None,
            "site_elev":        self.elev_spin.value(),
        }

        if mode == "time_series":
            folder = self.folder_edit.text()
            fits_files = sorted(
                glob.glob(os.path.join(folder, "*.fit")) +
                glob.glob(os.path.join(folder, "*.fits")) +
                glob.glob(os.path.join(folder, "*.fts")) +
                glob.glob(os.path.join(folder, "*.FIT")) +
                glob.glob(os.path.join(folder, "*.FITS")) +
                glob.glob(os.path.join(folder, "*.FTS"))
            )
            cfg["fits_files"] = fits_files
        else:
            cfg["fits_path"] = self.file_edit.text()

        return cfg

# =============================================================================
# RIGHT PANEL — RESULTS
# =============================================================================

class ResultsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(6, 6, 6, 6)

        # Result summary groupbox
        self.grp_result = QGroupBox("Result")
        grp_lay = QVBoxLayout(self.grp_result)

        # Badge row
        badge_row = QHBoxLayout()

        self.lbl_k = QLabel("—")
        self.lbl_k.setObjectName("value")
        self.lbl_k_unit = QLabel("mag/airmass")
        self.lbl_k_unit.setStyleSheet(
            f"color: {SIRIL_TEXT_DIM}; font-size: 9pt;")

        self.lbl_k_err = QLabel("")
        self.lbl_k_err.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 10pt;")

        self.lbl_r2 = QLabel("")
        self.lbl_r2.setStyleSheet(f"color: {SIRIL_TEXT_DIM}; font-size: 10pt;")

        self.lbl_quality = QLabel("")
        self.lbl_quality.setStyleSheet(
            "font-size: 11pt; font-weight: bold; padding: 4px 10px; "
            f"border-radius: 4px; background: {SIRIL_BG3};")

        badge_row.addWidget(self.lbl_k)
        badge_row.addWidget(self.lbl_k_unit)
        badge_row.addSpacing(12)
        badge_row.addWidget(self.lbl_k_err)
        badge_row.addSpacing(12)
        badge_row.addWidget(self.lbl_r2)
        badge_row.addStretch()
        badge_row.addWidget(self.lbl_quality)
        grp_lay.addLayout(badge_row)

        # Quality description
        self.lbl_quality_desc = QLabel("")
        self.lbl_quality_desc.setObjectName("dim")
        self.lbl_quality_desc.setWordWrap(True)
        grp_lay.addWidget(self.lbl_quality_desc)

        # Reference table
        ref_lbl = QLabel(
            "<b>V-band typical values:</b><br>"
            "Observatory site (high altitude, dry): &nbsp; k ≈ 0.10–0.15<br>"
            "Good suburban site: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; k ≈ 0.20–0.30<br>"
            "Humid coastal site: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; k ≈ 0.25–0.40<br>"
            "Thin cirrus: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; k ≈ 0.40–0.80<br>"
            "Thick cloud: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; k > 0.80"
        )
        ref_lbl.setStyleSheet(
            f"color: {SIRIL_TEXT_DIM}; font-size: 8pt; "
            f"background: {SIRIL_BG3}; padding: 6px; border-radius: 4px;")
        ref_lbl.setWordWrap(True)
        grp_lay.addWidget(ref_lbl)

        # Colour-dependence note
        note_lbl = QLabel(
            "Note: colour-dependent extinction correction not applied. "
            "Results most accurate for narrowband or similar-colour reference stars.")
        note_lbl.setObjectName("warn")
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet(f"color: {SIRIL_WARNING}; font-size: 8pt;")
        grp_lay.addWidget(note_lbl)

        lay.addWidget(self.grp_result)

        # Tab widget
        self.tabs = QTabWidget()

        # Bouguer plot tab
        self.bouguer_canvas = BouguerCanvas()
        self.tabs.addTab(self.bouguer_canvas, "📈  Bouguer Plot")

        # Timeline tab
        self.timeline_canvas = TransparencyCanvas()
        self.tabs.addTab(self.timeline_canvas, "🕐  Timeline")

        # Log tab
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setProperty("readOnly", True)
        font = QFont("Courier New", 9)
        self.log_edit.setFont(font)
        self.tabs.addTab(self.log_edit, "📋  Log")

        lay.addWidget(self.tabs)

    def append_log(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{ts}] {line}")
        self.log_edit.ensureCursorVisible()

    def clear_log(self):
        self.log_edit.clear()

    def add_live_point(self, data: dict):
        self.bouguer_canvas.add_point(data["airmass"], data["mag"])

    def show_results(self, result: dict):
        k     = result.get("k", 0.0)
        k_err = result.get("k_err", 0.0)
        r2    = result.get("r2", 0.0)
        q     = result.get("quality", "unknown")

        self.lbl_k.setText(f"k = {k:.4f}")
        self.lbl_k_err.setText(f"± {k_err:.4f}")
        self.lbl_r2.setText(f"R² = {r2:.3f}")

        qual_colors = {
            "photometric": SIRIL_SUCCESS,
            "good":        SIRIL_ACCENT,
            "variable":    SIRIL_WARNING,
            "poor":        SIRIL_ERROR,
        }
        color = qual_colors.get(q, SIRIL_TEXT_DIM)
        self.lbl_quality.setText(q.upper())
        self.lbl_quality.setStyleSheet(
            f"font-size: 11pt; font-weight: bold; padding: 4px 10px; "
            f"border-radius: 4px; color: {color}; "
            f"border: 1px solid {color}; background: {SIRIL_BG3};")

        desc = QUALITY_DESCRIPTIONS.get(q, "")
        self.lbl_quality_desc.setText(desc)

        # Bouguer plot
        if "airmasses" in result:
            self.bouguer_canvas.show_full(
                result["airmasses"],
                result["magnitudes"],
                result.get("bouguer_fit") or result,
                result.get("transparency"),
            )

        # Timeline
        if "jds" in result and "transparency" in result:
            self.timeline_canvas.show_timeline(
                result["jds"],
                result["transparency"],
                result.get("altitudes"),
            )

    def clear_results(self):
        self.lbl_k.setText("—")
        self.lbl_k_err.setText("")
        self.lbl_r2.setText("")
        self.lbl_quality.setText("")
        self.lbl_quality_desc.setText("")
        self.bouguer_canvas.clear_plot()
        self.timeline_canvas.clear_plot()

# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Extinction Map — Siril Script #12")
        self.resize(1100, 720)

        self._worker       = None
        self._cancel_event = threading.Event()

        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setSpacing(0)
        root_lay.setContentsMargins(0, 0, 0, 0)

        # Title bar
        root_lay.addWidget(TitleBar())

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._settings = SettingsPanel()
        self._settings.scan_requested.connect(self._scan_folder)
        splitter.addWidget(self._settings)

        self._results = ResultsPanel()
        splitter.addWidget(self._results)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 760])

        root_lay.addWidget(splitter, stretch=1)

        # Bottom bar
        bottom = QWidget()
        bottom.setStyleSheet(
            f"background: {SIRIL_BG2}; border-top: 1px solid {SIRIL_BORDER};")
        bot_lay = QHBoxLayout(bottom)
        bot_lay.setContentsMargins(8, 6, 8, 6)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(8)
        bot_lay.addWidget(self._progress, stretch=1)

        self._run_btn = QPushButton("▶  Measure extinction")
        self._run_btn.setObjectName("primary")
        self._run_btn.setFixedHeight(32)
        self._run_btn.clicked.connect(self._start)
        bot_lay.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("✕  Cancel")
        self._cancel_btn.setObjectName("danger")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        bot_lay.addWidget(self._cancel_btn)

        root_lay.addWidget(bottom)

        # Status bar
        self._status = QLabel("Ready — select a folder or file and press Measure")
        self._status.setStyleSheet(
            f"color: {SIRIL_TEXT_DIM}; padding: 3px 8px; font-size: 9pt; "
            f"background: {SIRIL_BG}; border-top: 1px solid {SIRIL_BORDER};")
        root_lay.addWidget(self._status)

    # ── Scan folder ───────────────────────────────────────────────────────────

    def _scan_folder(self):
        folder = self._settings.folder_edit.text()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Scan",
                                "Please select a valid FITS folder first.")
            return

        fits_files = sorted(
            glob.glob(os.path.join(folder, "*.fit"))  +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts"))  +
            glob.glob(os.path.join(folder, "*.FIT"))  +
            glob.glob(os.path.join(folder, "*.FITS")) +
            glob.glob(os.path.join(folder, "*.FTS"))
        )

        warnings = []
        has_date_obs = False
        has_location = False

        if fits_files:
            try:
                with fits.open(fits_files[0]) as hdul:
                    hdr = hdul[0].header
                    if hdr.get("DATE-OBS"):
                        has_date_obs = True
                    if (hdr.get("SITELAT") or hdr.get("OBSLAT") or
                            hdr.get("LAT-OBS")):
                        has_location = True
            except Exception:
                pass

        if not has_date_obs:
            warnings.append("⚠ No DATE-OBS in headers — altitude cannot be computed")
        if not has_location:
            warnings.append("⚠ No site location in headers — enter manually")

        self._settings.update_scan_info(len(fits_files), warnings)
        self._status.setText(
            f"Scanned: {len(fits_files)} FITS files found in folder")

    # ── Start / cancel ────────────────────────────────────────────────────────

    def _start(self):
        cfg = self._settings.get_config()

        # Validation
        if cfg["mode"] == "time_series":
            files = cfg.get("fits_files", [])
            if not files:
                QMessageBox.warning(self, "Error",
                    "No FITS files found in the selected folder.\n"
                    "Please choose a folder containing calibrated frames.")
                return
            if len(files) < 5:
                QMessageBox.warning(self, "Error",
                    f"Only {len(files)} FITS files found. Need at least 5.")
                return
        else:
            fp = cfg.get("fits_path", "")
            if not fp or not os.path.isfile(fp):
                QMessageBox.warning(self, "Error",
                    "Please select a valid FITS file.")
                return

        self._results.clear_results()
        self._results.clear_log()

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)

        self._cancel_event = threading.Event()
        self._worker = ExtinctionWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._results.append_log)
        self._worker.frame_done.connect(self._results.add_live_point)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

        self._status.setText("Running…")

    def _cancel(self):
        if self._worker:
            self._cancel_event.set()
        self._cancel_btn.setEnabled(False)
        self._status.setText("Cancelling…")

    # ── Worker signals ────────────────────────────────────────────────────────

    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setMaximum(total)
        self._progress.setValue(step)
        self._status.setText(msg)

    def _on_finished(self, result: dict):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)

        if not result.get("success"):
            err = result.get("error", "Unknown error")
            self._results.append_log(f"FAILED: {err}")
            self._status.setText(f"✗  {err}")
            return

        self._results.show_results(result)

        k   = result.get("k", 0.0)
        r2  = result.get("r2", 0.0)
        q   = result.get("quality", "—")
        n   = result.get("n_frames") or result.get("n_points") or "—"
        csv = result.get("csv_path", "")

        status = (
            f"✓  k = {k:.4f} mag/airmass  |  "
            f"R² = {r2:.3f}  |  "
            f"Quality: {q.upper()}  |  "
            f"N = {n} frames"
        )
        if csv:
            status += f"  |  CSV: {os.path.basename(csv)}"
        self._status.setText(status)

        self._results.append_log("=" * 60)
        self._results.append_log(f"  k = {k:.4f} ± {result.get('k_err', 0):.4f} mag/airmass")
        self._results.append_log(f"  R² = {r2:.3f}")
        self._results.append_log(f"  Quality: {q.upper()}")
        if csv:
            self._results.append_log(f"  CSV: {csv}")
        self._results.append_log("=" * 60)

        # Switch to Bouguer tab to show result immediately
        self._results.tabs.setCurrentIndex(0)

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
