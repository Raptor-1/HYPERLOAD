"""
variable_star.py  —  Script #19
Siril Variable Star Monitoring Pipeline
========================================
End-to-end: FITS sequence → differential photometry → period analysis →
variability classification → VSX/SIMBAD cross-match → AAVSO report export.

Wraps: period_finder.py (#32), transient_detector.py (#28)
"""

# ── mandatory installs ────────────────────────────────────────────────────────
import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")
try:
    s.ensure_installed("astroquery")
except Exception:
    pass

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, csv, json, glob, threading
from datetime import datetime

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from scipy.signal import argrelextrema
from astropy.io import fits as astropy_fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                aperture_photometry as phot_ap)
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QDesktopServices

# ── optional sibling scripts ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from period_finder import (
        run_lomb_scargle, run_acf, run_wavelet,
        cross_validate_periods, phase_fold,
        normalize_lightcurve, load_csv_lightcurve,
        compute_from_fits_sequence,
    )
    HAS_PERIOD_FINDER = True
except ImportError:
    HAS_PERIOD_FINDER = False

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

try:
    from astroquery.vizier import Vizier
    from astroquery.simbad import Simbad
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False

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
SIRIL_NOVA     = "#ff6b35"

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
QPushButton#aavso {{
    background-color: #1a2d1a;
    border-color: {SIRIL_SUCCESS};
    color: {SIRIL_SUCCESS};
    font-weight: bold;
    text-align: center;
}}
QPushButton#aavso:hover {{ background-color: #223a22; }}
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
QLabel#alert    {{ color: {SIRIL_NOVA};    font-weight: bold; font-size: 11pt; }}
QLabel#period   {{
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
QSplitter::handle {{ background: {SIRIL_BORDER}; width: 2px; }}
"""

# ══════════════════════════════════════════════════════════════════════════════
# VARIABILITY TYPES
# ══════════════════════════════════════════════════════════════════════════════
VARIABILITY_TYPES = {
    "EA":  {"name": "Eclipsing Algol-type",         "period_days": (0.2, 10.0),    "amplitude": (0.1, 3.0),  "description": "Sharp minima with flat maxima. Detached binary.", "shape": "flat_max_sharp_min"},
    "EB":  {"name": "Eclipsing Beta Lyrae-type",    "period_days": (0.3, 200.0),   "amplitude": (0.1, 2.0),  "description": "Continuous brightness variation. Semi-detached binary.", "shape": "sinusoidal"},
    "EW":  {"name": "Eclipsing W UMa-type",         "period_days": (0.2, 0.8),     "amplitude": (0.1, 0.9),  "description": "Short period, near-equal minima. Contact binary.", "shape": "sinusoidal_equal_minima"},
    "RRAB":{"name": "RR Lyrae ab-type",             "period_days": (0.3, 1.0),     "amplitude": (0.3, 2.0),  "description": "Fast rise, slow decline. Old metal-poor star.", "shape": "sawtooth"},
    "RRC": {"name": "RR Lyrae c-type",              "period_days": (0.2, 0.5),     "amplitude": (0.1, 0.6),  "description": "Nearly sinusoidal. First overtone pulsator.", "shape": "sinusoidal"},
    "DCEP":{"name": "Classical Cepheid",            "period_days": (1.0, 100.0),   "amplitude": (0.1, 2.0),  "description": "Sawtooth shape. Period-luminosity standard candle.", "shape": "sawtooth"},
    "DSCT":{"name": "Delta Scuti",                  "period_days": (0.01, 0.3),    "amplitude": (0.003, 0.3),"description": "Short period, multi-periodic. A-F type pulsator.", "shape": "multi_periodic"},
    "MIRA":{"name": "Mira / Long Period Variable",  "period_days": (100.0, 1000.0),"amplitude": (2.5, 10.0), "description": "Large amplitude, very red. AGB star.", "shape": "irregular"},
    "SR":  {"name": "Semi-regular",                 "period_days": (20.0, 2000.0), "amplitude": (0.1, 4.0),  "description": "Irregular period, moderate amplitude. Giant star.", "shape": "semi_regular"},
    "FLARE":{"name":"Flare Star",                   "period_days": None,           "amplitude": (0.1, 5.0),  "description": "Sudden brightening events. M-dwarf or young star.", "shape": "flare"},
    "UNK": {"name": "Unknown / Unclassified",       "period_days": None,           "amplitude": None,        "description": "Variability detected but type not determined.", "shape": "unknown"},
}


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def classify_variable(period_days, amplitude_mag, phase_curve, confidence):
    notes  = []
    scores = {}

    if period_days is None or period_days <= 0:
        return "UNK", 0.0, ["No period detected"]

    for code, vtype in VARIABILITY_TYPES.items():
        if code in ("UNK", "FLARE"):
            continue
        p_range = vtype["period_days"]
        if p_range is None:
            continue
        p_min, p_max = p_range
        if p_min <= period_days <= p_max:
            scores[code] = 0.4
        elif p_min * 0.5 <= period_days <= p_max * 2.0:
            scores[code] = 0.1
        else:
            scores[code] = 0.0

    if amplitude_mag is not None:
        for code in scores:
            a_range = VARIABILITY_TYPES[code].get("amplitude")
            if a_range:
                a_min, a_max = a_range
                if a_min <= amplitude_mag <= a_max:
                    scores[code] += 0.3
                elif amplitude_mag < a_min * 0.5 or amplitude_mag > a_max * 2:
                    scores[code] -= 0.2

    if 0.2 <= period_days <= 1.0:
        notes.append(f"Period {period_days:.3f}d consistent with RR Lyrae")
        scores["RRAB"] = scores.get("RRAB", 0) + 0.1
        scores["EW"]   = scores.get("EW",   0) + 0.1

    if period_days < 0.3:
        notes.append("Short period — Delta Scuti candidate")
        scores["DSCT"] = scores.get("DSCT", 0) + 0.2

    if period_days > 100:
        notes.append("Long period — Mira or semi-regular candidate")
        scores["MIRA"] = scores.get("MIRA", 0) + 0.1
        scores["SR"]   = scores.get("SR",   0) + 0.1

    if phase_curve and len(phase_curve.get("bin_fluxes", [])) > 10:
        bf    = np.array(phase_curve["bin_fluxes"])
        valid = ~np.isnan(bf)
        if valid.sum() > 5:
            bf_v = bf[valid]
            half = len(bf_v) // 2
            rise_slope = np.mean(np.diff(bf_v[:half]))
            fall_slope = np.mean(np.diff(bf_v[half:]))
            if rise_slope > 0 and fall_slope < 0:
                asymmetry = abs(rise_slope) / (abs(fall_slope) + 1e-10)
                if asymmetry > 2.0:
                    notes.append("Asymmetric light curve — sawtooth shape")
                    scores["RRAB"] = scores.get("RRAB", 0) + 0.2
                    scores["DCEP"] = scores.get("DCEP", 0) + 0.15

    if not scores:
        return "UNK", 0.0, notes

    best_code  = max(scores, key=scores.get)
    best_score = scores[best_code]
    if best_score < 0.2:
        return "UNK", best_score, notes + ["Low confidence — insufficient data"]
    return best_code, min(best_score, 1.0), notes


# ══════════════════════════════════════════════════════════════════════════════
# PHOTOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def _measure_star(image, x, y, aper_r, ann_in, ann_out):
    aper  = CircularAperture([(x, y)], r=aper_r)
    annul = CircularAnnulus([(x, y)], r_in=ann_in, r_out=ann_out)
    phot  = phot_ap(image, aper)
    bkg   = phot_ap(image, annul)
    bkg_pp = float(bkg["aperture_sum"][0]) / annul.area
    net   = float(phot["aperture_sum"][0]) - bkg_pp * aper.area
    noise = max(net, 1.0) ** 0.5 + bkg_pp * aper.area * 0.01
    snr   = net / max(noise, 1e-6)
    return net, snr


def aperture_photometry_target(image, target_x, target_y, comp_positions,
                                aperture_r=8.0, annulus_r_in=12.0, annulus_r_out=18.0):
    t_flux, t_snr = _measure_star(image, target_x, target_y,
                                   aperture_r, annulus_r_in, annulus_r_out)
    comp_fluxes = []
    for cx, cy in comp_positions:
        cf, _ = _measure_star(image, cx, cy, aperture_r, annulus_r_in, annulus_r_out)
        if cf > 0:
            comp_fluxes.append(cf)
    if not comp_fluxes:
        return None
    comp_flux = float(np.median(comp_fluxes))
    if comp_flux <= 0 or t_flux <= 0:
        return None
    diff_flux = t_flux / comp_flux
    diff_mag  = -2.5 * np.log10(diff_flux)
    return {"target_flux": t_flux, "comp_flux": comp_flux,
            "diff_flux": diff_flux, "diff_mag": diff_mag, "snr": t_snr}


def build_variable_star_lightcurve(fits_files, target_x, target_y,
                                    comp_positions, aperture_r=8.0,
                                    log_callback=None, cancel_event=None):
    times, diff_mags, errors, snrs = [], [], [], []
    n_failed = 0

    for i, path in enumerate(fits_files):
        if cancel_event and cancel_event.is_set():
            break
        if log_callback and i % 20 == 0:
            log_callback(f"  Photometry frame {i+1}/{len(fits_files)}")
        try:
            with astropy_fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float32)
                hdr  = hdul[0].header
            if data.ndim == 3:
                data = (data[0] if data.shape[0] <= 4
                        else 0.299*data[0]+0.587*data[1]+0.114*data[2])

            date_obs = hdr.get("DATE-OBS", "")
            if date_obs:
                try:
                    t_jd = float(Time(date_obs, format="isot", scale="utc").jd)
                except Exception:
                    t_jd = float(i)
            else:
                t_jd = float(i)

            result = aperture_photometry_target(
                data, target_x, target_y, comp_positions, aperture_r)

            if result is None:
                n_failed += 1
                continue

            mag_err = 1.0857 / max(result["snr"], 0.1)
            times.append(t_jd)
            diff_mags.append(result["diff_mag"])
            errors.append(mag_err)
            snrs.append(result["snr"])
        except Exception as e:
            n_failed += 1
            if log_callback:
                log_callback(f"  Frame {i} failed: {e}")

    return {
        "times":     np.array(times),
        "diff_mags": np.array(diff_mags),
        "errors":    np.array(errors),
        "snrs":      np.array(snrs),
        "n_frames":  len(fits_files),
        "n_failed":  n_failed,
    }


def auto_detect_comp_stars(image, target_x, target_y, n_max=8):
    """Use DAOStarFinder to suggest comparison star candidates."""
    try:
        _, median, std = sigma_clipped_stats(image, sigma=3.0)
        finder = DAOStarFinder(fwhm=4.0, threshold=10.*std)
        sources = finder(image - median)
        if sources is None or len(sources) == 0:
            return []
        candidates = []
        for row in sources:
            x, y = float(row["xcentroid"]), float(row["ycentroid"])
            dist = ((x - target_x)**2 + (y - target_y)**2)**0.5
            if dist < 50:
                continue
            # Not saturated heuristic: peak < 0.95 * max
            peak = float(row["peak"])
            if peak > 0.95 * image.max():
                continue
            candidates.append((x, y, float(row["flux"])))
        # Sort by flux descending, take top N
        candidates.sort(key=lambda r: r[2], reverse=True)
        return [(x, y) for x, y, _ in candidates[:n_max]]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CATALOG CROSS-MATCH
# ══════════════════════════════════════════════════════════════════════════════
def crossmatch_vsx(ra_deg, dec_deg, radius_arcsec=30.0):
    if not HAS_ASTROQUERY:
        return None
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coord  = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
        v      = Vizier(columns=["Name","Type","Period","Epoch","Max","Min","VarType"])
        result = v.query_region(coord, radius=radius_arcsec*u.arcsec, catalog="B/vsx/vsx")
        if result and len(result) > 0 and len(result[0]) > 0:
            row = result[0][0]
            return {
                "name":    str(row["Name"]),
                "type":    str(row.get("Type", "?")),
                "period":  float(row["Period"]) if row["Period"] else None,
                "epoch":   float(row["Epoch"])  if row["Epoch"]  else None,
                "mag_max": str(row.get("Max", "?")),
                "mag_min": str(row.get("Min", "?")),
            }
    except Exception:
        pass
    return None


def crossmatch_simbad(ra_deg, dec_deg, radius_arcsec=10.0):
    if not HAS_ASTROQUERY:
        return None
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        cs = Simbad()
        cs.add_votable_fields("otype", "distance")
        coord  = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
        result = cs.query_region(coord, radius=radius_arcsec*u.arcsec)
        if result and len(result) > 0:
            return {"name": str(result["MAIN_ID"][0]),
                    "otype": str(result["OTYPE"][0])}
    except Exception:
        pass
    return None


def get_wcs_coords(fits_header, x, y):
    try:
        from astropy.wcs import WCS
        wcs = WCS(fits_header)
        sky = wcs.pixel_to_world(x, y)
        return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# AAVSO EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def export_aavso_extended(light_curve, target_name, observer_code,
                           comp_star_mags, filter_name, output_path, notes=""):
    comp_name = list(comp_star_mags.keys())[0]  if comp_star_mags else "ENSEMBLE"
    comp_mag  = list(comp_star_mags.values())[0] if comp_star_mags else "na"

    lines = [
        "#TYPE=EXTENDED",
        f"#OBSCODE={observer_code}",
        "#SOFTWARE=Siril variable_star.py",
        "#DELIM=,",
        "#DATE=JD",
        "#OBSTYPE=CCD",
        "#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES",
    ]
    for t, m, e in zip(light_curve["times"], light_curve["diff_mags"], light_curve["errors"]):
        if not np.isfinite(m) or not np.isfinite(t):
            continue
        lines.append(f"{target_name},{t:.6f},{m:.4f},{e:.4f},"
                     f"{filter_name},NO,STD,"
                     f"{comp_name},{comp_mag},"
                     f"na,na,na,na,na,{notes}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def compute_ephemeris(period_days, epoch_jd, n_future=5):
    from astropy.time import Time
    events  = []
    now_jd  = float(Time.now().jd)
    n_cycles = int((now_jd - epoch_jd) / period_days) + 1
    for i in range(n_future):
        fjd = epoch_jd + (n_cycles + i) * period_days
        t   = Time(fjd, format="jd", scale="utc")
        events.append((round(fjd, 4), t.isot))
    return events


# ══════════════════════════════════════════════════════════════════════════════
# SESSION PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
def _to_serial(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
    return obj


def save_session(session, output_path):
    clean = {}
    for k, v in session.items():
        if isinstance(v, dict):
            clean[k] = {kk: _to_serial(vv) for kk, vv in v.items()}
        else:
            clean[k] = _to_serial(v)
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)


def load_session(path):
    with open(path, "r") as f:
        session = json.load(f)
    lc = session.get("light_curve", {})
    for key in ["times", "diff_mags", "errors", "snrs"]:
        if key in lc:
            lc[key] = np.array(lc[key])
    return session


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK PERIOD FUNCTIONS (when period_finder.py is absent)
# ══════════════════════════════════════════════════════════════════════════════
def _fallback_lomb_scargle(t, m, e, min_period=None, max_period=None):
    from astropy.timeseries import LombScargle
    ls   = LombScargle(t, m, e)
    kw   = {}
    if min_period: kw["maximum_frequency"] = 1.0 / min_period
    if max_period: kw["minimum_frequency"] = 1.0 / max_period
    freq, power = ls.autopower(**kw)
    best_period = float(1.0 / freq[np.argmax(power)])
    return {"best_period": best_period, "confidence": "candidate",
            "frequencies": freq.tolist(), "powers": power.tolist()}


def _fallback_normalize(t, m, e):
    med = np.nanmedian(m)
    return t, m - med, e


def _fallback_phase_fold(t, m, period):
    phases = ((t - t[0]) % period) / period
    idx    = np.argsort(phases)
    return {"period": period, "phases": phases[idx].tolist(),
            "fluxes": m[idx].tolist(),
            "bin_phases": [], "bin_fluxes": []}


# ══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════
class VarStarWorker(QThread):
    progress     = pyqtSignal(int, int, str)
    log_line     = pyqtSignal(str)
    lc_ready     = pyqtSignal(dict)
    period_ready = pyqtSignal(dict)
    fold_ready   = pyqtSignal(dict)
    class_ready  = pyqtSignal(str, float, list)
    finished     = pyqtSignal(dict)

    def __init__(self, config, cancel_event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg = self.config

            # ── Step 1: Build light curve ─────────────────────────────────────
            self.progress.emit(1, 6, "Building light curve...")
            lc_source = cfg.get("lc_source", "fits_sequence")

            if lc_source == "fits_sequence":
                self.log_line.emit(f"Aperture photometry on {len(cfg['fits_files'])} frames...")
                lc = build_variable_star_lightcurve(
                    fits_files=cfg["fits_files"],
                    target_x=cfg["target_x"],
                    target_y=cfg["target_y"],
                    comp_positions=cfg["comp_positions"],
                    aperture_r=cfg.get("aperture_r", 8.0),
                    log_callback=self.log_line.emit,
                    cancel_event=self._cancel,
                )

            elif lc_source == "csv":
                if HAS_PERIOD_FINDER:
                    times, fluxes, errors = load_csv_lightcurve(cfg["csv_path"])
                else:
                    rows = []
                    with open(cfg["csv_path"]) as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = line.split(",")
                            if len(parts) >= 2:
                                try:
                                    rows.append([float(p) for p in parts[:3]])
                                except ValueError:
                                    pass
                    arr    = np.array(rows)
                    times  = arr[:, 0]
                    fluxes = arr[:, 1]
                    errors = arr[:, 2] if arr.shape[1] > 2 else np.full_like(fluxes, 0.01)
                lc = {
                    "times": times, "diff_mags": fluxes,
                    "errors": errors if errors is not None else np.full_like(fluxes, 0.01),
                    "n_frames": len(times), "n_failed": 0,
                }

            else:  # transient detector dict
                lc_raw  = cfg["lc_dict"]
                star_id = cfg.get("star_id", 0)
                fluxes  = np.array(lc_raw["fluxes"][star_id])
                times   = np.arange(len(fluxes), dtype=float)
                lc = {
                    "times": times,
                    "diff_mags": -2.5 * np.log10(
                        np.clip(fluxes / max(np.median(fluxes), 1e-10), 1e-10, None)),
                    "errors":   np.full_like(fluxes, 0.02),
                    "n_frames": len(times), "n_failed": 0,
                }

            self.log_line.emit(f"Light curve: {lc['n_frames']} frames, {lc['n_failed']} failed")
            self.lc_ready.emit(lc)
            if self._cancel.is_set(): self._abort(); return

            # ── Step 2: Normalize ─────────────────────────────────────────────
            self.progress.emit(2, 6, "Normalizing light curve...")
            if HAS_PERIOD_FINDER:
                t, m, e = normalize_lightcurve(lc["times"], lc["diff_mags"], lc["errors"], method="median")
            else:
                t, m, e = _fallback_normalize(lc["times"], lc["diff_mags"], lc["errors"])
            amplitude_est = float(np.percentile(m, 95) - np.percentile(m, 5))
            self.log_line.emit(f"Amplitude estimate: {amplitude_est:.3f} mag")
            if self._cancel.is_set(): self._abort(); return

            # ── Step 3: Period analysis ───────────────────────────────────────
            self.progress.emit(3, 6, "Period analysis...")
            min_p = cfg.get("min_period") or None
            max_p = cfg.get("max_period") or None
            ls_result = acf_result = wav_result = None

            if HAS_PERIOD_FINDER:
                if cfg.get("run_ls", True):
                    ls_result = run_lomb_scargle(t, m, e, min_period=min_p, max_period=max_p)
                    self.log_line.emit(f"LS best period: {ls_result['best_period']:.6f}  FAP: {ls_result.get('best_fap', 0):.2e}")
                if cfg.get("run_acf", True):
                    acf_result = run_acf(t, m, min_period=min_p, max_period=max_p)
                if cfg.get("run_wavelet", True):
                    wav_result = run_wavelet(t, m, min_period=min_p, max_period=max_p)
                cv = cross_validate_periods(ls_result, acf_result, wav_result)
                best_period = cv["best_period"]
                self.period_ready.emit({"ls": ls_result, "acf": acf_result, "wav": wav_result, "cv": cv})
            else:
                self.log_line.emit("period_finder.py not found — using LS only")
                ls_result   = _fallback_lomb_scargle(t, m, e, min_p, max_p)
                best_period = ls_result["best_period"]
                cv          = ls_result
                self.period_ready.emit({"ls": ls_result, "acf": None, "wav": None, "cv": cv})
                self.log_line.emit(f"Period (LS only): {best_period:.6f}")

            if self._cancel.is_set(): self._abort(); return

            # ── Step 4: Phase fold ────────────────────────────────────────────
            self.progress.emit(4, 6, "Phase folding...")
            fold = {}
            if best_period > 0:
                if HAS_PERIOD_FINDER:
                    fold = phase_fold(t, m, best_period)
                else:
                    fold = _fallback_phase_fold(t, m, best_period)
                self.fold_ready.emit(fold)
                self.log_line.emit(f"Phase-folded at P={best_period:.6f}")
            if self._cancel.is_set(): self._abort(); return

            # ── Step 5: Classification ────────────────────────────────────────
            self.progress.emit(5, 6, "Classifying variability type...")
            var_type, conf, class_notes = classify_variable(
                best_period, amplitude_est, fold, cv.get("confidence", "candidate"))
            self.class_ready.emit(var_type, conf, class_notes)
            self.log_line.emit(
                f"Classification: {VARIABILITY_TYPES[var_type]['name']} (confidence: {conf:.2f})")

            # ── Step 6: Catalog + export ──────────────────────────────────────
            self.progress.emit(6, 6, "Catalog cross-match & export...")
            vsx_match = simbad_match = None
            ra = dec = None

            if cfg.get("crossmatch") and cfg.get("fits_header"):
                coords = get_wcs_coords(cfg["fits_header"], cfg["target_x"], cfg["target_y"])
                if coords:
                    ra, dec = coords
                    self.log_line.emit(f"Target coords: RA={ra:.4f}°  Dec={dec:.4f}°")
                    if HAS_ASTROQUERY:
                        vsx_match = crossmatch_vsx(ra, dec)
                        if vsx_match:
                            self.log_line.emit(f"VSX match: {vsx_match['name']} ({vsx_match['type']})")
                        simbad_match = crossmatch_simbad(ra, dec)

            out_dir     = cfg.get("output_dir", ".")
            os.makedirs(out_dir, exist_ok=True)
            target_name = cfg.get("target_name", "TARGET")

            aavso_path = None
            if cfg.get("export_aavso"):
                aavso_path = os.path.join(out_dir, f"{target_name}_aavso.csv")
                export_aavso_extended(
                    lc, target_name,
                    cfg.get("observer_code", "XXX"),
                    cfg.get("comp_mags", {}),
                    cfg.get("filter_name", "V"),
                    aavso_path)
                self.log_line.emit(f"AAVSO report: {aavso_path}")

            session_path = os.path.join(out_dir, f"{target_name}_session.json")
            save_session({
                "target_name":  target_name,
                "target_x":     cfg.get("target_x"),
                "target_y":     cfg.get("target_y"),
                "ra": ra, "dec": dec,
                "best_period":  best_period,
                "var_type":     var_type,
                "confidence":   conf,
                "amplitude":    amplitude_est,
                "vsx_match":    vsx_match,
                "simbad_match": simbad_match,
                "light_curve":  lc,
            }, session_path)
            self.log_line.emit(f"Session saved: {session_path}")

            epoch_jd  = float(t[np.argmin(m)]) if len(t) > 0 else 0.0
            ephemeris = compute_ephemeris(best_period, epoch_jd) if best_period > 0 else []

            self.finished.emit({
                "success":      True,
                "best_period":  best_period,
                "var_type":     var_type,
                "confidence":   conf,
                "amplitude":    amplitude_est,
                "vsx_match":    vsx_match,
                "ra": ra, "dec": dec,
                "aavso_path":   aavso_path,
                "session_path": session_path,
                "ephemeris":    ephemeris,
                "lc":           lc,
            })

        except Exception as exc:
            import traceback
            self.log_line.emit(f"ERROR: {exc}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(exc)})

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CANVASES
# ══════════════════════════════════════════════════════════════════════════════
class LCCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig   = Figure(figsize=(9, 5), facecolor=SIRIL_BG)
        self.ax_lc = self.fig.add_subplot(2, 1, 1)
        self.ax_ph = self.fig.add_subplot(2, 1, 2)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style_axes(self):
        for ax, xl, yl, ttl in [
            (self.ax_lc, "Time (JD)",   "Diff. magnitude", "Light curve"),
            (self.ax_ph, "Phase (0-2)", "Diff. magnitude", "Phase-folded"),
        ]:
            ax.set_facecolor(SIRIL_BG2)
            ax.tick_params(colors=SIRIL_TEXT, labelsize=8)
            ax.set_xlabel(xl,  color=SIRIL_TEXT_DIM, fontsize=8)
            ax.set_ylabel(yl,  color=SIRIL_TEXT_DIM, fontsize=8)
            ax.set_title(ttl,  color=SIRIL_SECTION,  fontsize=9)
            for sp in ["bottom","left"]:  ax.spines[sp].set_color(SIRIL_BORDER)
            for sp in ["top","right"]:    ax.spines[sp].set_visible(False)

    def show_lightcurve(self, lc):
        self.ax_lc.clear()
        self.ax_lc.set_facecolor(SIRIL_BG2)
        t, m, e = lc["times"], lc["diff_mags"], lc.get("errors")
        if e is not None and len(e) == len(t):
            self.ax_lc.errorbar(t, m, yerr=e, fmt="o", color=SIRIL_ACCENT,
                                 markersize=2, elinewidth=0.6, alpha=0.7)
        else:
            self.ax_lc.plot(t, m, "o", color=SIRIL_ACCENT, markersize=2, alpha=0.7)
        self.ax_lc.invert_yaxis()
        self.ax_lc.set_xlabel("Time (JD)",   color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_lc.set_ylabel("Diff. mag",   color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_lc.set_title("Light curve",  color=SIRIL_SECTION,  fontsize=9)
        for sp in ["bottom","left"]:  self.ax_lc.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top","right"]:    self.ax_lc.spines[sp].set_visible(False)
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def show_phase_fold(self, fold):
        self.ax_ph.clear()
        self.ax_ph.set_facecolor(SIRIL_BG2)
        p = fold.get("period", 0)
        phases  = np.array(fold.get("phases", []))
        fluxes  = np.array(fold.get("fluxes", []))
        bphases = np.array(fold.get("bin_phases", []))
        bfluxes = np.array(fold.get("bin_fluxes", []))

        if len(phases):
            self.ax_ph.plot(phases,   fluxes, "o", color=SIRIL_ACCENT, markersize=2, alpha=0.4, zorder=1)
            self.ax_ph.plot(phases+1, fluxes, "o", color=SIRIL_ACCENT, markersize=2, alpha=0.2, zorder=1)
        if len(bfluxes):
            valid = ~np.isnan(bfluxes)
            if valid.sum() > 2:
                for offset, alpha in [(0, 0.9), (1, 0.5)]:
                    self.ax_ph.plot(bphases[valid]+offset, bfluxes[valid],
                                     color=SIRIL_NOVA, linewidth=1.5, alpha=alpha, zorder=3)
        self.ax_ph.set_xlim(0, 2)
        self.ax_ph.axvline(1, color=SIRIL_BORDER, linewidth=0.8, linestyle="--")
        self.ax_ph.invert_yaxis()
        self.ax_ph.set_xlabel("Phase (0-2)",       color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_ph.set_ylabel("Diff. mag",          color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_ph.set_title(f"Phase fold  P={p:.6f}", color=SIRIL_SECTION, fontsize=9)
        for sp in ["bottom","left"]:  self.ax_ph.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top","right"]:    self.ax_ph.spines[sp].set_visible(False)
        self.fig.tight_layout(pad=0.4)
        self.draw()


class FieldCanvas(FigureCanvasQTAgg):
    target_changed = pyqtSignal(float, float)
    comps_changed  = pyqtSignal(list)

    def __init__(self, parent=None):
        self.fig  = Figure(figsize=(5, 5), facecolor=SIRIL_BG)
        self.ax   = self.fig.add_subplot(111)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)
        self.frame_data     = None
        self.target_pos     = None
        self.comp_positions = []
        self.mpl_connect("button_press_event", self._on_click)

    def _style(self):
        self.ax.set_facecolor(SIRIL_BG)
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_color(SIRIL_BORDER)
        self.ax.set_title("Left-click: target   Right-click: comparison stars",
                           color=SIRIL_SECTION, fontsize=8)

    def show_frame(self, frame):
        self.frame_data = frame
        self._redraw()

    def _redraw(self):
        if self.frame_data is None:
            return
        self.ax.clear()
        self._style()
        p_lo = np.percentile(self.frame_data, 0.5)
        p_hi = np.percentile(self.frame_data, 99.5)
        disp = np.clip((self.frame_data.astype(float) - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)
        self.ax.imshow(disp, cmap="gray", origin="lower", aspect="equal", interpolation="nearest")

        if self.target_pos:
            x, y = self.target_pos
            self.ax.plot(x, y, "o", color=SIRIL_ERROR, markersize=12,
                         markerfacecolor="none", markeredgewidth=2, zorder=5)
            self.ax.text(x+8, y, "T", color=SIRIL_ERROR, fontsize=8, fontweight="bold")

        for i, (cx, cy) in enumerate(self.comp_positions):
            self.ax.plot(cx, cy, "o", color=SIRIL_SUCCESS, markersize=10,
                         markerfacecolor="none", markeredgewidth=1.5, zorder=5)
            self.ax.text(cx+8, cy, f"C{i+1}", color=SIRIL_SUCCESS, fontsize=7)

        self.fig.tight_layout(pad=0.2)
        self.draw()

    def set_comp_positions(self, positions):
        self.comp_positions = list(positions)
        self._redraw()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            self.target_pos = (x, y)
            self.target_changed.emit(x, y)
        elif event.button == 3:
            remove_idx = None
            for i, (cx, cy) in enumerate(self.comp_positions):
                if ((cx-x)**2 + (cy-y)**2)**0.5 < 15:
                    remove_idx = i
                    break
            if remove_idx is not None:
                self.comp_positions.pop(remove_idx)
            else:
                self.comp_positions.append((x, y))
            self.comps_changed.emit(self.comp_positions)
        self._redraw()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⭐  Variable Star Pipeline  —  Siril")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)

        self._worker        = None
        self._cancel_event  = threading.Event()
        self._fits_files    = []
        self._first_header  = None
        self._lc_result     = None
        self._period_result = None
        self._fold_result   = None
        self._final_result  = None
        self._session_loaded = None  # path of loaded session

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 4)

        # Title bar
        title_row = QHBoxLayout()
        lbl_title = QLabel("⭐  Variable Star Pipeline  —  Siril")
        lbl_title.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold;")
        lbl_ver   = QLabel("v1.0  |  AAVSO citizen science")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        title_row.addWidget(lbl_title)
        title_row.addStretch()
        title_row.addWidget(lbl_ver)
        root.addLayout(title_row)

        sep = QFrame(); sep.setObjectName("separator"); sep.setFixedHeight(1)
        root.addWidget(sep)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(370)
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)
        self._build_left(left_layout)
        left_layout.addStretch()

        # Right panel — tabs
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self._tabs = QTabWidget()
        right_layout.addWidget(self._tabs)
        splitter.addWidget(right)
        self._build_right_tabs()

        splitter.setSizes([370, 880])

        # Bottom controls
        bot = QHBoxLayout()
        bot.setSpacing(8)
        self._progress = QProgressBar()
        self._progress.setRange(0, 6)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(8)
        bot.addWidget(self._progress, 1)

        self._btn_run = QPushButton("▶  Run pipeline")
        self._btn_run.setObjectName("primary")
        self._btn_run.setFixedWidth(160)
        self._btn_run.clicked.connect(self._run_pipeline)
        bot.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setFixedWidth(100)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_pipeline)
        bot.addWidget(self._btn_cancel)

        root.addLayout(bot)

        # Status bar
        self._status = QLabel("Ready — load a FITS sequence or CSV to begin")
        self._status.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt; padding: 2px 4px;")
        root.addWidget(self._status)

    # ── Left panel ────────────────────────────────────────────────────────────
    def _build_left(self, layout):
        # ── Data source ──
        grp_src = QGroupBox("Data source")
        vl_src  = QVBoxLayout(grp_src)

        self._combo_src = QComboBox()
        self._combo_src.addItems([
            "FITS sequence (aperture photometry)",
            "CSV light curve (from period_finder.py)",
            "Transient detector output",
        ])
        self._combo_src.currentIndexChanged.connect(self._on_source_changed)
        vl_src.addWidget(self._combo_src)

        # FITS widgets
        self._fits_widget = QWidget()
        fl = QVBoxLayout(self._fits_widget)
        fl.setContentsMargins(0,0,0,0)
        fl.setSpacing(4)

        row_folder = QHBoxLayout()
        self._edit_folder = QLineEdit(); self._edit_folder.setPlaceholderText("FITS folder…")
        btn_browse_folder = QPushButton("…"); btn_browse_folder.setFixedWidth(28)
        btn_browse_folder.clicked.connect(self._browse_folder)
        row_folder.addWidget(self._edit_folder); row_folder.addWidget(btn_browse_folder)
        fl.addLayout(row_folder)

        btn_load_seq = QPushButton("🔍  Load sequence & show field")
        btn_load_seq.clicked.connect(self._load_sequence)
        fl.addWidget(btn_load_seq)

        self._lbl_seq_info = QLabel("—")
        self._lbl_seq_info.setObjectName("dim")
        fl.addWidget(self._lbl_seq_info)

        lbl_hint = QLabel("Left-click field → target   Right-click → comp stars")
        lbl_hint.setObjectName("dim"); lbl_hint.setWordWrap(True)
        fl.addWidget(lbl_hint)

        self._lbl_target = QLabel("Target: not selected"); self._lbl_target.setObjectName("dim")
        fl.addWidget(self._lbl_target)
        self._lbl_comps  = QLabel("Comp stars: 0");         self._lbl_comps.setObjectName("dim")
        fl.addWidget(self._lbl_comps)

        form_aper = QFormLayout()
        self._spin_aper = QDoubleSpinBox(); self._spin_aper.setRange(3, 20); self._spin_aper.setValue(8.0); self._spin_aper.setSuffix(" px")
        form_aper.addRow("Aperture:", self._spin_aper)
        fl.addLayout(form_aper)

        vl_src.addWidget(self._fits_widget)

        # CSV widgets
        self._csv_widget = QWidget()
        cl = QVBoxLayout(self._csv_widget)
        cl.setContentsMargins(0,0,0,0)
        row_csv = QHBoxLayout()
        self._edit_csv = QLineEdit(); self._edit_csv.setPlaceholderText("light curve CSV…")
        btn_browse_csv = QPushButton("…"); btn_browse_csv.setFixedWidth(28)
        btn_browse_csv.clicked.connect(self._browse_csv)
        row_csv.addWidget(self._edit_csv); row_csv.addWidget(btn_browse_csv)
        cl.addLayout(row_csv)
        btn_load_csv = QPushButton("Load CSV")
        btn_load_csv.clicked.connect(self._load_csv)
        cl.addWidget(btn_load_csv)
        self._csv_widget.setVisible(False)
        vl_src.addWidget(self._csv_widget)

        layout.addWidget(grp_src)

        # ── Session resume ──
        grp_sess = QGroupBox("Session")
        sl = QHBoxLayout(grp_sess)
        btn_load_session  = QPushButton("📂  Load previous session"); btn_load_session.clicked.connect(self._load_session_dialog)
        sl.addWidget(btn_load_session)
        layout.addWidget(grp_sess)

        # ── Period search ──
        grp_period = QGroupBox("Period search")
        pl = QVBoxLayout(grp_period)
        self._chk_ls  = QCheckBox("Lomb-Scargle");  self._chk_ls.setChecked(True)
        self._chk_acf = QCheckBox("ACF");            self._chk_acf.setChecked(True)
        self._chk_wav = QCheckBox("Wavelet");        self._chk_wav.setChecked(True)
        for c in [self._chk_ls, self._chk_acf, self._chk_wav]:
            pl.addWidget(c)
        form_p = QFormLayout()
        self._spin_min_p = QDoubleSpinBox(); self._spin_min_p.setRange(0, 10000); self._spin_min_p.setValue(0); self._spin_min_p.setSpecialValueText("auto"); self._spin_min_p.setSuffix(" d")
        self._spin_max_p = QDoubleSpinBox(); self._spin_max_p.setRange(0, 10000); self._spin_max_p.setValue(0); self._spin_max_p.setSpecialValueText("auto"); self._spin_max_p.setSuffix(" d")
        form_p.addRow("Min period:", self._spin_min_p)
        form_p.addRow("Max period:", self._spin_max_p)
        pl.addLayout(form_p)
        if not HAS_PERIOD_FINDER:
            lbl_no_pf = QLabel("⚠ period_finder.py not found — LS only")
            lbl_no_pf.setObjectName("warn"); lbl_no_pf.setWordWrap(True)
            pl.addWidget(lbl_no_pf)
        layout.addWidget(grp_period)

        # ── Target info ──
        grp_target = QGroupBox("Target info")
        tl = QFormLayout(grp_target)
        self._edit_target_name  = QLineEdit(); self._edit_target_name.setPlaceholderText("e.g. V* MY Cyg")
        self._edit_obs_code     = QLineEdit(); self._edit_obs_code.setPlaceholderText("AAVSO code")
        self._combo_filter      = QComboBox(); self._combo_filter.addItems(["V","B","R","I","CV","TG","Vis"])
        tl.addRow("Target name:",   self._edit_target_name)
        tl.addRow("Observer code:", self._edit_obs_code)
        tl.addRow("Filter:",        self._combo_filter)
        layout.addWidget(grp_target)

        # ── Comparison star magnitudes ──
        grp_comp_mags = QGroupBox("Comparison star magnitudes")
        cml = QFormLayout(grp_comp_mags)
        self._spin_comp1 = QDoubleSpinBox(); self._spin_comp1.setRange(0, 25); self._spin_comp1.setValue(0); self._spin_comp1.setSpecialValueText("unknown")
        self._spin_comp2 = QDoubleSpinBox(); self._spin_comp2.setRange(0, 25); self._spin_comp2.setValue(0); self._spin_comp2.setSpecialValueText("unknown")
        cml.addRow("Comp 1 mag:", self._spin_comp1)
        cml.addRow("Comp 2 mag:", self._spin_comp2)
        lbl_comp_hint = QLabel("From AAVSO VSP chart. Leave 0 for differential only.")
        lbl_comp_hint.setObjectName("dim"); lbl_comp_hint.setWordWrap(True)
        cml.addRow(lbl_comp_hint)
        layout.addWidget(grp_comp_mags)

        # ── Catalog ──
        grp_cat = QGroupBox("Catalog cross-match")
        cat_l = QVBoxLayout(grp_cat)
        self._chk_crossmatch = QCheckBox("Cross-match VSX / SIMBAD")
        self._chk_crossmatch.setChecked(True)
        if not HAS_ASTROQUERY:
            self._chk_crossmatch.setEnabled(False)
            self._chk_crossmatch.setToolTip("astroquery not installed")
        cat_l.addWidget(self._chk_crossmatch)
        layout.addWidget(grp_cat)

        # ── Export ──
        grp_export = QGroupBox("Export")
        el = QVBoxLayout(grp_export)
        row_out = QHBoxLayout()
        self._edit_out = QLineEdit(); self._edit_out.setPlaceholderText("Output folder…")
        btn_out = QPushButton("…"); btn_out.setFixedWidth(28)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self._edit_out); row_out.addWidget(btn_out)
        el.addLayout(row_out)
        self._chk_aavso    = QCheckBox("AAVSO Extended Format report"); self._chk_aavso.setChecked(True)
        self._chk_sess_json= QCheckBox("Session JSON (for resuming)");   self._chk_sess_json.setChecked(True)
        self._chk_lc_csv   = QCheckBox("Light curve CSV");               self._chk_lc_csv.setChecked(True)
        for c in [self._chk_aavso, self._chk_sess_json, self._chk_lc_csv]:
            el.addWidget(c)
        layout.addWidget(grp_export)

    # ── Right tabs ────────────────────────────────────────────────────────────
    def _build_right_tabs(self):
        # Tab: Field
        self._field_canvas = FieldCanvas()
        self._field_canvas.target_changed.connect(self._on_target_selected)
        self._field_canvas.comps_changed.connect(self._on_comps_changed)
        self._tabs.addTab(self._field_canvas, "🌟  Field")

        # Tab: Light Curve
        self._lc_canvas = LCCanvas()
        self._tabs.addTab(self._lc_canvas, "📈  Light Curve")

        # Tab: Classification
        cls_widget = QWidget()
        cls_layout = QVBoxLayout(cls_widget)

        grp_result = QGroupBox("Result")
        rl = QVBoxLayout(grp_result)
        self._lbl_period_big = QLabel("P = —")
        self._lbl_period_big.setObjectName("period")
        self._lbl_period_big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(self._lbl_period_big)
        self._lbl_type    = QLabel("—"); self._lbl_type.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_type.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold;")
        self._lbl_conf    = QLabel("Confidence: —")
        self._lbl_amp     = QLabel("Amplitude: —")
        for lbl in [self._lbl_type, self._lbl_conf, self._lbl_amp]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rl.addWidget(lbl)
        cls_layout.addWidget(grp_result)

        grp_reasoning = QGroupBox("Reasoning")
        rrl = QVBoxLayout(grp_reasoning)
        self._txt_reasoning = QPlainTextEdit(); self._txt_reasoning.setReadOnly(True)
        self._txt_reasoning.setFixedHeight(100)
        rrl.addWidget(self._txt_reasoning)
        cls_layout.addWidget(grp_reasoning)

        grp_catalog = QGroupBox("Catalog match")
        cll = QVBoxLayout(grp_catalog)
        self._lbl_vsx    = QLabel("VSX: —")
        self._lbl_simbad = QLabel("SIMBAD: —")
        cll.addWidget(self._lbl_vsx)
        cll.addWidget(self._lbl_simbad)
        cls_layout.addWidget(grp_catalog)

        grp_ephem = QGroupBox("Ephemeris — next events")
        el2 = QVBoxLayout(grp_ephem)
        self._tbl_ephem = QTableWidget(0, 2)
        self._tbl_ephem.setHorizontalHeaderLabels(["JD", "ISO date"])
        self._tbl_ephem.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl_ephem.setAlternatingRowColors(True)
        self._tbl_ephem.setFixedHeight(140)
        el2.addWidget(self._tbl_ephem)
        cls_layout.addWidget(grp_ephem)

        cls_layout.addStretch()
        self._tabs.addTab(cls_widget, "🔭  Classification")

        # Tab: AAVSO Report
        aavso_widget = QWidget()
        al = QVBoxLayout(aavso_widget)
        self._txt_aavso = QPlainTextEdit(); self._txt_aavso.setReadOnly(True)
        al.addWidget(self._txt_aavso, 1)
        row_aavso = QHBoxLayout()
        btn_open_aavso = QPushButton("📤  Open AAVSO submission page")
        btn_open_aavso.setObjectName("aavso")
        btn_open_aavso.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://www.aavso.org/webobs")))
        btn_save_aavso = QPushButton("💾  Save AAVSO report")
        btn_save_aavso.setObjectName("export")
        btn_save_aavso.clicked.connect(self._save_aavso_dialog)
        btn_vsp = QPushButton("🗺  Open VSP chart")
        btn_vsp.setObjectName("aavso")
        btn_vsp.clicked.connect(self._open_vsp)
        row_aavso.addWidget(btn_open_aavso)
        row_aavso.addWidget(btn_save_aavso)
        row_aavso.addWidget(btn_vsp)
        al.addLayout(row_aavso)
        self._tabs.addTab(aavso_widget, "📄  AAVSO Report")

        # Tab: Log
        self._txt_log = QPlainTextEdit(); self._txt_log.setReadOnly(True)
        self._tabs.addTab(self._txt_log, "📋  Log")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _log(self, msg):
        self._txt_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _set_status(self, msg, color=None):
        c = color or SIRIL_TEXT_DIM
        self._status.setStyleSheet(f"color:{c}; font-size:9pt; padding:2px 4px;")
        self._status.setText(msg)

    def _on_source_changed(self, idx):
        self._fits_widget.setVisible(idx == 0)
        self._csv_widget.setVisible(idx == 1)

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self._edit_folder.setText(d)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", filter="CSV (*.csv *.txt);;All (*)")
        if path:
            self._edit_csv.setText(path)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self._edit_out.setText(d)

    def _load_sequence(self):
        folder = self._edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Please select a valid FITS folder.")
            return
        patterns = ["*.fit","*.fits","*.FIT","*.FITS","*.fts"]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(folder, pat)))
        files.sort()
        if not files:
            QMessageBox.warning(self, "No files", "No FITS files found in folder.")
            return
        self._fits_files = files
        self._lbl_seq_info.setText(f"{len(files)} frames found")
        self._log(f"Loaded {len(files)} FITS files from {folder}")

        # Load first frame for preview
        try:
            with astropy_fits.open(files[0]) as hdul:
                data = hdul[0].data.astype(np.float32)
                self._first_header = hdul[0].header
            if data.ndim == 3:
                data = data[0]
            self._field_canvas.show_frame(data)
            self._tabs.setCurrentIndex(0)
            self._set_status(f"Loaded {len(files)} frames. Click field to select target/comp stars.")
        except Exception as e:
            self._log(f"Preview failed: {e}")

    def _load_csv(self):
        path = self._edit_csv.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file", "Please select a valid CSV file.")
            return
        self._log(f"CSV ready: {path}")
        self._set_status(f"CSV loaded: {os.path.basename(path)}")

    def _on_target_selected(self, x, y):
        self._lbl_target.setText(f"Target: ({x:.1f}, {y:.1f}) px")
        self._log(f"Target selected at ({x:.1f}, {y:.1f})")
        # Auto-detect comps if we have a frame
        if self._field_canvas.frame_data is not None and not self._field_canvas.comp_positions:
            comps = auto_detect_comp_stars(self._field_canvas.frame_data, x, y)
            if comps:
                self._field_canvas.set_comp_positions(comps)
                self._lbl_comps.setText(f"Comp stars: {len(comps)} (auto-detected)")
                self._log(f"Auto-detected {len(comps)} comparison star candidates")

    def _on_comps_changed(self, positions):
        self._lbl_comps.setText(f"Comp stars: {len(positions)}")

    def _load_session_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load session", filter="JSON (*.json);;All (*)")
        if not path:
            return
        try:
            session = load_session(path)
            self._session_loaded = session
            self._lbl_target.setText(f"Target: {session.get('target_name','?')} "
                                     f"({session.get('target_x','?'):.1f}, {session.get('target_y','?'):.1f})")
            lc = session.get("light_curve", {})
            if lc:
                self._lc_result = lc
                self._lc_canvas.show_lightcurve(lc)
            self._log(f"Session loaded: {path}  "
                      f"({len(lc.get('times',[]))} existing data points)")
            QMessageBox.information(self, "Session loaded",
                f"Session loaded.\nNew data will be appended to existing light curve.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load session:\n{e}")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    def _run_pipeline(self):
        if self._worker and self._worker.isRunning():
            return

        src_idx = self._combo_src.currentIndex()

        # Validate inputs
        if src_idx == 0:
            if not self._fits_files:
                QMessageBox.warning(self, "No data", "Load a FITS sequence first.")
                return
            if not self._field_canvas.target_pos:
                QMessageBox.warning(self, "No target", "Click on the field image to select a target star.")
                return
            if not self._field_canvas.comp_positions:
                QMessageBox.warning(self, "No comp stars", "Right-click to add at least one comparison star.")
                return
        elif src_idx == 1:
            if not self._edit_csv.text().strip():
                QMessageBox.warning(self, "No CSV", "Select a CSV light curve file.")
                return

        out_dir = self._edit_out.text().strip() or os.path.dirname(
            self._fits_files[0] if self._fits_files else os.getcwd())

        # Build comp_mags dict
        comp_mags = {}
        if self._spin_comp1.value() > 0:
            comp_mags["COMP1"] = self._spin_comp1.value()
        if self._spin_comp2.value() > 0:
            comp_mags["COMP2"] = self._spin_comp2.value()

        # If session loaded, merge existing light curve later (handled post-run)
        cfg = {
            "lc_source":     ["fits_sequence","csv","transient"][src_idx],
            "fits_files":    self._fits_files,
            "csv_path":      self._edit_csv.text().strip(),
            "target_x":     self._field_canvas.target_pos[0] if self._field_canvas.target_pos else 0,
            "target_y":     self._field_canvas.target_pos[1] if self._field_canvas.target_pos else 0,
            "comp_positions":self._field_canvas.comp_positions,
            "aperture_r":   self._spin_aper.value(),
            "run_ls":        self._chk_ls.isChecked(),
            "run_acf":       self._chk_acf.isChecked(),
            "run_wavelet":   self._chk_wav.isChecked(),
            "min_period":   self._spin_min_p.value() if self._spin_min_p.value() > 0 else None,
            "max_period":   self._spin_max_p.value() if self._spin_max_p.value() > 0 else None,
            "target_name":   self._edit_target_name.text().strip() or "TARGET",
            "observer_code": self._edit_obs_code.text().strip() or "XXX",
            "filter_name":   self._combo_filter.currentText(),
            "comp_mags":     comp_mags,
            "crossmatch":    self._chk_crossmatch.isChecked(),
            "fits_header":   self._first_header,
            "export_aavso":  self._chk_aavso.isChecked(),
            "export_session":self._chk_sess_json.isChecked(),
            "output_dir":    out_dir,
        }

        self._cancel_event = threading.Event()
        self._worker = VarStarWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log)
        self._worker.lc_ready.connect(self._on_lc_ready)
        self._worker.period_ready.connect(self._on_period_ready)
        self._worker.fold_ready.connect(self._on_fold_ready)
        self._worker.class_ready.connect(self._on_class_ready)
        self._worker.finished.connect(self._on_finished)

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._set_status("Running pipeline…", SIRIL_ACCENT)
        self._log("=" * 60)
        self._log("Pipeline started")
        self._tabs.setCurrentIndex(4)  # log tab
        self._worker.start()

    def _cancel_pipeline(self):
        if self._worker:
            self._worker.cancel()
        self._btn_cancel.setEnabled(False)
        self._log("Cancel requested…")

    # ── Worker signals ────────────────────────────────────────────────────────
    def _on_progress(self, step, total, msg):
        self._progress.setRange(0, total)
        self._progress.setValue(step)
        self._set_status(f"Step {step}/{total}: {msg}", SIRIL_ACCENT)

    def _on_lc_ready(self, lc):
        # If session loaded, merge
        if self._session_loaded and "light_curve" in self._session_loaded:
            old_lc = self._session_loaded["light_curve"]
            merged_times    = np.concatenate([np.array(old_lc.get("times",[])),    lc["times"]])
            merged_mags     = np.concatenate([np.array(old_lc.get("diff_mags",[])),lc["diff_mags"]])
            merged_errors   = np.concatenate([np.array(old_lc.get("errors",[])),   lc["errors"]])
            merged_snrs     = np.concatenate([np.array(old_lc.get("snrs",[])),     lc.get("snrs", np.zeros_like(lc["times"]))])
            sort_idx = np.argsort(merged_times)
            lc["times"]    = merged_times[sort_idx]
            lc["diff_mags"]= merged_mags[sort_idx]
            lc["errors"]   = merged_errors[sort_idx]
            lc["snrs"]     = merged_snrs[sort_idx]
            lc["n_frames"]= int(len(lc["times"]))
            self._log(f"Merged with session — total {lc['n_frames']} points")
        self._lc_result = lc
        self._lc_canvas.show_lightcurve(lc)
        # Export LC CSV if requested
        if self._chk_lc_csv.isChecked():
            out_dir     = self._edit_out.text().strip() or "."
            target_name = self._edit_target_name.text().strip() or "TARGET"
            csv_path    = os.path.join(out_dir, f"{target_name}_lightcurve.csv")
            try:
                os.makedirs(out_dir, exist_ok=True)
                with open(csv_path, "w") as f:
                    f.write("# JD,diff_mag,error\n")
                    for t, m, e in zip(lc["times"], lc["diff_mags"], lc["errors"]):
                        f.write(f"{t:.6f},{m:.4f},{e:.4f}\n")
                self._log(f"Light curve CSV: {csv_path}")
            except Exception as ex:
                self._log(f"LC CSV write failed: {ex}")

    def _on_period_ready(self, period_dict):
        self._period_result = period_dict

    def _on_fold_ready(self, fold):
        self._fold_result = fold
        self._lc_canvas.show_phase_fold(fold)

    def _on_class_ready(self, var_type, conf, notes):
        vt = VARIABILITY_TYPES.get(var_type, {})
        self._lbl_period_big.setText(f"P = —")  # updated in _on_finished
        self._lbl_type.setText(vt.get("name", var_type))

        if conf >= 0.7:
            badge = f"HIGH confidence ({conf*100:.0f}%)"
            col   = SIRIL_SUCCESS
        elif conf >= 0.4:
            badge = f"MODERATE confidence ({conf*100:.0f}%)"
            col   = SIRIL_WARNING
        else:
            badge = f"LOW confidence ({conf*100:.0f}%) — more data needed"
            col   = SIRIL_ERROR
        self._lbl_conf.setText(badge)
        self._lbl_conf.setStyleSheet(f"color:{col}; font-weight:bold; font-size:10pt;")
        self._txt_reasoning.setPlainText(
            vt.get("description","") + "\n\n" + "\n".join(f"• {n}" for n in notes))
        self._tabs.setCurrentIndex(2)

    def _on_finished(self, result):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._final_result = result

        if not result.get("success"):
            err = result.get("error","Unknown error")
            self._set_status(f"✗  Pipeline failed: {err}", SIRIL_ERROR)
            self._log(f"FAILED: {err}")
            return

        p   = result.get("best_period", 0)
        vt  = VARIABILITY_TYPES.get(result.get("var_type","UNK"),{})
        con = result.get("confidence", 0)
        amp = result.get("amplitude", 0)

        self._lbl_period_big.setText(f"P = {p:.6f} d")
        self._lbl_amp.setText(f"Amplitude: {amp:.3f} mag")

        # Catalog match
        vsx = result.get("vsx_match")
        if vsx:
            self._lbl_vsx.setText(
                f"VSX: {vsx['name']}  type={vsx['type']}  "
                f"P={vsx.get('period','?')}  "
                f"[{vsx.get('mag_max','?')} – {vsx.get('mag_min','?')}]")
            self._lbl_vsx.setStyleSheet(f"color:{SIRIL_SUCCESS};")
        simbad = result.get("simbad_match")
        if simbad:
            self._lbl_simbad.setText(f"SIMBAD: {simbad['name']}  ({simbad['otype']})")
            self._lbl_simbad.setStyleSheet(f"color:{SIRIL_ACCENT};")

        # Ephemeris table
        ephem = result.get("ephemeris", [])
        self._tbl_ephem.setRowCount(len(ephem))
        for row, (jd, iso) in enumerate(ephem):
            self._tbl_ephem.setItem(row, 0, QTableWidgetItem(str(jd)))
            self._tbl_ephem.setItem(row, 1, QTableWidgetItem(iso))

        # AAVSO report preview
        aavso_path = result.get("aavso_path")
        if aavso_path and os.path.isfile(aavso_path):
            try:
                with open(aavso_path) as f:
                    self._txt_aavso.setPlainText(f.read())
            except Exception:
                pass

        status_msg = (f"✓  P={p:.6f} d  |  Type: {vt.get('name', result.get('var_type','?'))}  "
                      f"|  Conf: {con*100:.0f}%")
        if aavso_path:
            status_msg += "  |  AAVSO: saved"
        self._set_status(status_msg, SIRIL_SUCCESS)
        self._log("Pipeline complete.")
        self._log(status_msg)

    # ── Dialogs ───────────────────────────────────────────────────────────────
    def _save_aavso_dialog(self):
        if not self._lc_result:
            QMessageBox.warning(self, "No data", "Run the pipeline first.")
            return
        target_name = self._edit_target_name.text().strip() or "TARGET"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save AAVSO report",
            f"{target_name}_aavso.csv",
            "CSV (*.csv);;All (*)")
        if not path:
            return
        comp_mags = {}
        if self._spin_comp1.value() > 0:
            comp_mags["COMP1"] = self._spin_comp1.value()
        try:
            export_aavso_extended(
                self._lc_result, target_name,
                self._edit_obs_code.text().strip() or "XXX",
                comp_mags,
                self._combo_filter.currentText(),
                path)
            self._log(f"AAVSO report saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed:\n{e}")

    def _open_vsp(self):
        if self._final_result and self._final_result.get("ra") is not None:
            ra  = self._final_result["ra"]
            dec = self._final_result["dec"]
            url = f"https://www.aavso.org/apps/vsp/?ra={ra:.5f}&dec={dec:.5f}&fov=60"
        else:
            url = "https://www.aavso.org/apps/vsp/"
        QDesktopServices.openUrl(QUrl(url))


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
