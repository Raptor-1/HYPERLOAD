r"""
transient_detector.py  -  Siril Script #28
Multi-layer transient detection pipeline
Place in: C:\Users\Marcell\Desktop\Siril New Scripts\
Requires zogy_subtract.py in the same folder for Layer 2.
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

# ──────────────────────────────────────────────────────────────────────────────
# Standard imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import csv
import json
import glob
import threading
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import label, center_of_mass
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
    QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ──────────────────────────────────────────────────────────────────────────────
# Optional sibling-script imports
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from zogy_subtract import (
        zogy_subtract, detect_transients, align_images,
        background_match, estimate_fwhm_from_stars,
        estimate_noise, estimate_psf_gaussian,
    )
    HAS_ZOGY = True
except ImportError:
    HAS_ZOGY = False

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

try:
    from astroquery.vizier import Vizier
    from astroquery.mpc import MPC
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────────────────────
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
QListWidget {{
    background-color: {SIRIL_BG2};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
}}
QListWidget::item {{ padding: 4px 8px; }}
QListWidget::item:selected {{ background: {SIRIL_ACCENT2}; color: white; }}
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
QPushButton#alert {{
    background-color: #2d1a0a;
    border-color: {SIRIL_NOVA};
    color: {SIRIL_NOVA};
    font-weight: bold;
    text-align: center;
}}
QPushButton#alert:hover {{ background-color: #3d2510; }}
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
QLabel#alert    {{ color: {SIRIL_NOVA};    font-weight: bold; font-size: 11pt; }}
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


# ──────────────────────────────────────────────────────────────────────────────
# Detection functions
# ──────────────────────────────────────────────────────────────────────────────

def make_alert_id(frame_idx: int) -> str:
    now = datetime.now(timezone.utc)
    return f"TD_{now.strftime('%Y%m%d_%H%M%S')}_{frame_idx:04d}"


def frame_difference_detect(frame_a, frame_b,
                              threshold_sigma=6.0,
                              min_pixels=3,
                              border_margin=20):
    if frame_a.shape != frame_b.shape:
        return []
    h, w  = frame_a.shape
    diff  = frame_b.astype(float) - frame_a.astype(float)
    _, med_diff, std_diff = sigma_clipped_stats(diff, sigma=3.0)
    if std_diff <= 0:
        return []
    threshold = threshold_sigma * std_diff
    alerts    = []
    for binary, event_type in [
        ((diff > threshold).astype(int),  "positive"),
        ((diff < -threshold).astype(int), "negative"),
    ]:
        labeled, n_features = label(binary)
        for lbl in range(1, n_features + 1):
            blob = (labeled == lbl)
            area = int(np.sum(blob))
            if area < min_pixels:
                continue
            cy, cx = center_of_mass(blob)
            if (cx < border_margin or cx > w - border_margin or
                    cy < border_margin or cy > h - border_margin):
                continue
            snr   = float(np.max(np.abs(diff[blob])) / std_diff)
            delta = float(np.sum(diff[blob]))
            alerts.append({
                "x": round(cx, 2), "y": round(cy, 2),
                "snr": round(snr, 2), "delta_flux": round(delta, 4),
                "area_px": area, "type": event_type,
                "layer": 1, "method": "frame_difference",
            })
    alerts.sort(key=lambda a: a["snr"], reverse=True)
    return alerts


def zogy_layer_detect(new_image, ref_image,
                       threshold_sigma=5.0,
                       fwhm_new=None, fwhm_ref=None):
    if not HAS_ZOGY:
        return []
    new_aligned, _ = align_images(new_image, ref_image, method="phase_correlation")
    new_matched     = background_match(new_aligned, ref_image)
    result          = zogy_subtract(
        new_matched.astype(np.float64),
        ref_image.astype(np.float64),
        fwhm_new_px=fwhm_new, fwhm_ref_px=fwhm_ref,
    )
    raw = detect_transients(result["S"], result["D"],
                             threshold_sigma=threshold_sigma)
    return [{
        "x": d["x"], "y": d["y"], "snr": d["snr"],
        "delta_flux": d["flux_D"], "area_px": 1,
        "type": d["type"], "layer": 2, "method": "zogy_subtraction",
    } for d in raw]


def build_light_curves(fits_files, fwhm_px=5.0, aperture_radius=8.0,
                        n_ref_stars=20, log_callback=None, cancel_event=None):
    if not fits_files:
        return {}
    first_data = astropy_fits.getdata(fits_files[0]).astype(float)
    if first_data.ndim == 3:
        first_data = first_data[0]
    _, median, std = sigma_clipped_stats(first_data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm_px, threshold=5.0 * std,
                             sharplo=0.2, sharphi=0.9)
    sources = daofind(first_data - median)
    if sources is None or len(sources) == 0:
        return {}
    sources.sort("peak")
    sources.reverse()
    sources = sources[:n_ref_stars]
    positions = {i: (float(src["xcentroid"]), float(src["ycentroid"]))
                 for i, src in enumerate(sources)}
    fluxes    = {i: [] for i in positions}
    timestamps = []

    for frame_idx, fits_path in enumerate(fits_files):
        if cancel_event and cancel_event.is_set():
            break
        if log_callback and frame_idx % 10 == 0:
            log_callback(f"  Photometry frame {frame_idx+1}/{len(fits_files)}")
        try:
            with astropy_fits.open(fits_path) as hdul:
                data = hdul[0].data.astype(float)
                hdr  = hdul[0].header
                date = hdr.get("DATE-OBS")
            if data.ndim == 3:
                data = data[0]
            timestamps.append(date)
        except Exception:
            for i in fluxes:
                fluxes[i].append(float("nan"))
            timestamps.append(None)
            continue

        coords = [(x, y) for x, y in positions.values()]
        apers  = CircularAperture(coords, r=aperture_radius)
        annuli = CircularAnnulus(coords,
                                  r_in=aperture_radius * 1.5,
                                  r_out=aperture_radius * 2.5)
        phot_table = aperture_photometry(data, apers)
        bkg_table  = aperture_photometry(data, annuli)
        for i, star_id in enumerate(positions):
            raw_flux = float(phot_table["aperture_sum"][i])
            bkg_mean = float(bkg_table["aperture_sum"][i]) / annuli.area
            fluxes[star_id].append(raw_flux - bkg_mean * apers.area)

    return {
        "star_ids": list(positions.keys()),
        "positions": positions,
        "fluxes": fluxes,
        "timestamps": timestamps,
        "n_frames": len(fits_files),
    }


def detect_photometric_transients(light_curves, threshold_sigma=4.0,
                                   min_consecutive=2):
    alerts = []
    for star_id in light_curves.get("star_ids", []):
        flux_series = np.array(light_curves["fluxes"][star_id])
        valid = ~np.isnan(flux_series)
        if np.sum(valid) < 5:
            continue
        med = np.median(flux_series[valid])
        std = np.median(np.abs(flux_series[valid] - med)) * 1.4826
        if std <= 0 or med <= 0:
            continue
        deviating = np.abs(flux_series - med) > threshold_sigma * std
        for i in range(len(flux_series) - min_consecutive + 1):
            if np.all(deviating[i:i + min_consecutive]):
                delta_flux = flux_series[i] - med
                snr        = abs(delta_flux) / std
                delta_mag  = -2.5 * np.log10(
                    max(flux_series[i], 1e-10) / max(med, 1e-10))
                x, y = light_curves["positions"][star_id]
                alerts.append({
                    "star_id": star_id, "x": round(x, 2), "y": round(y, 2),
                    "frame_idx": i, "snr": round(snr, 2),
                    "delta_mag_est": round(float(delta_mag), 3),
                    "type": "brightening" if delta_flux > 0 else "fading",
                    "layer": 3, "method": "photometric_monitoring",
                })
                break
    alerts.sort(key=lambda a: a["snr"], reverse=True)
    return alerts


def crossmatch_alert(x_px, y_px, fits_header,
                      search_radius_arcsec=10.0, log_callback=None):
    result = {
        "ra": None, "dec": None,
        "mpc_match": None, "vsx_match": None, "tns_match": None,
        "classification": "unknown",
    }
    try:
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        wcs  = WCS(fits_header)
        sky  = wcs.pixel_to_world(x_px, y_px)
        ra   = float(sky.ra.deg)
        dec  = float(sky.dec.deg)
        result["ra"]  = round(ra, 6)
        result["dec"] = round(dec, 6)
    except Exception:
        result["classification"] = "no_wcs"
        return result

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    coord  = SkyCoord(ra=result["ra"] * u.deg, dec=result["dec"] * u.deg)
    radius = search_radius_arcsec * u.arcsec

    if HAS_ASTROQUERY:
        try:
            v = Vizier(columns=["Name", "Type", "Period", "VarType"])
            vsx_result = v.query_region(coord, radius=radius, catalog="B/vsx/vsx")
            if vsx_result and len(vsx_result) > 0 and len(vsx_result[0]) > 0:
                row = vsx_result[0][0]
                result["vsx_match"]     = {"name": str(row["Name"]),
                                            "type": str(row.get("Type", "?"))}
                result["classification"] = "known_variable"
        except Exception as e:
            if log_callback:
                log_callback(f"  VSX query failed: {e}")
        try:
            mpc_result = MPC.query_objects(
                "asteroid", ra=result["ra"], dec=result["dec"],
                radius=search_radius_arcsec / 3600.0)
            if mpc_result and len(mpc_result) > 0:
                row = mpc_result[0]
                result["mpc_match"] = {"name": str(row.get("designation", "?"))}
                if result["classification"] == "unknown":
                    result["classification"] = "known_asteroid"
        except Exception as e:
            if log_callback:
                log_callback(f"  MPC query failed (non-critical): {e}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Export functions
# ──────────────────────────────────────────────────────────────────────────────

def export_alerts_csv(alerts, output_path):
    if not alerts:
        return
    fields = ["alert_id", "timestamp_utc", "fits_file", "frame_idx",
              "x", "y", "ra", "dec", "snr", "delta_flux", "delta_mag_est",
              "area_px", "type", "layer", "method", "classification",
              "mpc_match", "vsx_match", "notes"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for a in alerts:
            row = {k: a.get(k, "") for k in fields}
            for key in ("mpc_match", "vsx_match"):
                if isinstance(row.get(key), dict):
                    row[key] = str(row[key])
            writer.writerow(row)


def export_tns_format(alerts, output_path, observer_name="", instrument=""):
    lines = [
        "TRANSIENT CANDIDATES — Siril transient_detector.py",
        f"Observer: {observer_name}", f"Instrument: {instrument}",
        f"Generated: {datetime.utcnow().isoformat()}", "=" * 60,
    ]
    for i, a in enumerate(alerts, 1):
        lines += [
            f"\nCandidate #{i}: {a['alert_id']}",
            f"  RA:     {a.get('ra', 'N/A')}°",
            f"  Dec:    {a.get('dec', 'N/A')}°",
            f"  SNR:    {a['snr']}σ",
            f"  Type:   {a['type']}",
            f"  Layer:  {a['layer']} ({a['method']})",
            f"  File:   {a.get('fits_file', 'N/A')}",
            f"  Status: {a['classification']}",
        ]
        if a.get("vsx_match"):
            lines.append(f"  VSX:    {a['vsx_match']}")
        if a.get("mpc_match"):
            lines.append(f"  MPC:    {a['mpc_match']}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def export_aavso_report(alerts, output_path,
                          observer_code="XXX", target_name="UNKNOWN"):
    lines = [
        "#TYPE=EXTENDED", "#OBSCODE=" + observer_code,
        "#SOFTWARE=Siril transient_detector.py",
        "#DELIM=,", "#DATE=JD", "#OBSTYPE=CCD",
        "#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,"
        "AMASS,GROUP,CHART,NOTES",
    ]
    for a in alerts:
        if a.get("delta_mag_est") is None:
            continue
        lines.append(
            f"{target_name},0.0,{a.get('delta_mag_est','?')},0.1,"
            f"V,NO,STD,ENSEMBLE,na,na,na,na,na,na,"
            f"Layer{a['layer']} SNR={a['snr']}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def make_cutout_preview(fits_path, x, y, size_px=64):
    data = astropy_fits.getdata(fits_path).astype(float)
    if data.ndim == 3:
        data = data[0]
    h, w = data.shape
    ix, iy = int(round(x)), int(round(y))
    half = size_px // 2
    return data[max(0, iy-half):min(h, iy+half),
                max(0, ix-half):min(w, ix+half)]


def deduplicate_alerts(alerts, radius_px=5.0):
    """Merge alerts within radius_px into single best-SNR alert."""
    if not alerts:
        return alerts
    used   = [False] * len(alerts)
    merged = []
    for i, a in enumerate(alerts):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        for j, b in enumerate(alerts):
            if used[j] or j == i:
                continue
            if abs(a["x"] - b["x"]) < radius_px and abs(a["y"] - b["y"]) < radius_px:
                group.append(b)
                used[j] = True
        best = max(group, key=lambda x: x["snr"])
        if len(group) > 1:
            best = dict(best)
            layers = sorted({g["layer"] for g in group})
            best["confirmed_by_layers"] = layers
            best["notes"] = f"Confirmed by {len(layers)} layer(s)"
        merged.append(best)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Worker thread
# ──────────────────────────────────────────────────────────────────────────────

class DetectionWorker(QThread):
    progress    = pyqtSignal(int, int, str)
    log_line    = pyqtSignal(str)
    alert_found = pyqtSignal(dict)
    lc_ready    = pyqtSignal(dict)
    finished    = pyqtSignal(dict)

    def __init__(self, config, cancel_event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg        = self.config
            all_alerts = []

            if cfg.get("layer1_enabled", True):
                self._run_layer1(cfg, all_alerts)
                if self._cancel.is_set():
                    self._abort(); return

            if cfg.get("layer2_enabled", True) and HAS_ZOGY:
                self._run_layer2(cfg, all_alerts)
                if self._cancel.is_set():
                    self._abort(); return
            elif cfg.get("layer2_enabled") and not HAS_ZOGY:
                self.log_line.emit(
                    "⚠ Layer 2 skipped: zogy_subtract.py not found")

            if cfg.get("layer3_enabled", False):
                self._run_layer3(cfg, all_alerts)
                if self._cancel.is_set():
                    self._abort(); return

            # Deduplication across layers
            before = len(all_alerts)
            all_alerts = deduplicate_alerts(all_alerts)
            if len(all_alerts) < before:
                self.log_line.emit(
                    f"Deduplication: {before} → {len(all_alerts)} alerts "
                    f"({before - len(all_alerts)} merged)")

            if cfg.get("crossmatch_enabled", False) and all_alerts:
                self._run_crossmatch(cfg, all_alerts)

            out_dir  = cfg.get("output_dir", cfg.get("fits_dir", "."))
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "transient_alerts.csv")
            export_alerts_csv(all_alerts, csv_path)
            self.log_line.emit(f"Saved {len(all_alerts)} alerts → {csv_path}")

            if cfg.get("export_tns", False):
                tns_path = os.path.join(out_dir, "tns_report.txt")
                export_tns_format(all_alerts, tns_path,
                                   cfg.get("observer_name", ""),
                                   cfg.get("instrument", ""))
                self.log_line.emit(f"TNS report → {tns_path}")

            if cfg.get("export_aavso", False):
                aavso_path = os.path.join(out_dir, "aavso_report.txt")
                export_aavso_report(all_alerts, aavso_path,
                                     cfg.get("observer_code", "XXX"),
                                     cfg.get("target_name", "UNKNOWN"))
                self.log_line.emit(f"AAVSO report → {aavso_path}")

            # Save session JSON
            session_path = os.path.join(
                out_dir,
                f"detection_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            with open(session_path, "w") as f:
                json.dump({"config": cfg, "alerts": all_alerts}, f, indent=2, default=str)
            self.log_line.emit(f"Session saved → {session_path}")

            unknown_count = sum(1 for a in all_alerts
                                if a.get("classification") == "unknown")
            self.finished.emit({
                "success": True,
                "total_alerts": len(all_alerts),
                "unknown_alerts": unknown_count,
                "csv_path": csv_path,
                "alerts": all_alerts,
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    # ── Layer runners ──────────────────────────────────────────────────────

    def _run_layer1(self, cfg, all_alerts):
        fits_files = cfg["fits_files"]
        n          = len(fits_files)
        threshold  = cfg.get("layer1_threshold", 6.0)
        self.log_line.emit(
            f"Layer 1: Frame difference — {n-1} pairs, threshold {threshold}σ")
        prev_data = None
        for i, fits_path in enumerate(fits_files):
            if self._cancel.is_set():
                return
            self.progress.emit(i, n, f"L1 frame {i+1}/{n}")
            try:
                data = astropy_fits.getdata(fits_path).astype(np.float32)
                if data.ndim == 3:
                    data = data[0]
            except Exception as e:
                self.log_line.emit(f"  Failed {fits_path}: {e}")
                prev_data = None
                continue
            if prev_data is not None:
                alerts = frame_difference_detect(
                    prev_data, data, threshold_sigma=threshold,
                    min_pixels=cfg.get("min_pixels", 3),
                    border_margin=cfg.get("border_margin", 20))
                for a in alerts:
                    a.update(fits_file=os.path.basename(fits_path),
                             frame_idx=i, alert_id=make_alert_id(i),
                             timestamp_utc=self._get_timestamp(fits_path),
                             ra=None, dec=None,
                             delta_mag_est=None, classification="unknown",
                             verified=False, notes="",
                             mpc_match=None, vsx_match=None, tns_match=None)
                    all_alerts.append(a)
                    self.alert_found.emit(a)
                    self.log_line.emit(
                        f"  L1 #{len(all_alerts)}: "
                        f"({a['x']:.1f},{a['y']:.1f}) "
                        f"SNR={a['snr']:.1f} {a['type']}")
            prev_data = data

    def _run_layer2(self, cfg, all_alerts):
        fits_files = cfg["fits_files"]
        ref_path   = cfg.get("reference_path")
        threshold  = cfg.get("layer2_threshold", 5.0)
        if not ref_path or not os.path.isfile(ref_path):
            self.log_line.emit("Layer 2: Skipped — no reference image provided")
            return
        self.log_line.emit(
            f"Layer 2: ZOGY subtraction vs reference, threshold {threshold}σ")
        ref_data = astropy_fits.getdata(ref_path).astype(np.float32)
        if ref_data.ndim == 3:
            ref_data = ref_data[0]
        n = len(fits_files)
        for i, fits_path in enumerate(fits_files):
            if self._cancel.is_set():
                return
            self.progress.emit(i, n, f"L2 ZOGY {i+1}/{n}")
            try:
                new_data = astropy_fits.getdata(fits_path).astype(np.float32)
                if new_data.ndim == 3:
                    new_data = new_data[0]
                if new_data.shape != ref_data.shape:
                    continue
            except Exception as e:
                self.log_line.emit(f"  Failed {fits_path}: {e}")
                continue
            alerts = zogy_layer_detect(
                new_data, ref_data,
                threshold_sigma=threshold,
                fwhm_new=cfg.get("fwhm_new_px"),
                fwhm_ref=cfg.get("fwhm_ref_px"))
            for a in alerts:
                a.update(fits_file=os.path.basename(fits_path),
                         frame_idx=i, alert_id=make_alert_id(i),
                         timestamp_utc=self._get_timestamp(fits_path),
                         ra=None, dec=None,
                         delta_mag_est=None, classification="unknown",
                         verified=False, notes="",
                         mpc_match=None, vsx_match=None, tns_match=None)
                all_alerts.append(a)
                self.alert_found.emit(a)
                self.log_line.emit(
                    f"  L2 #{len(all_alerts)}: "
                    f"({a['x']:.1f},{a['y']:.1f}) "
                    f"SNR={a['snr']:.1f} {a['type']}")

    def _run_layer3(self, cfg, all_alerts):
        fits_files = cfg["fits_files"]
        threshold  = cfg.get("layer3_threshold", 4.0)
        self.log_line.emit(
            f"Layer 3: Photometric monitoring — {len(fits_files)} frames, "
            f"threshold {threshold}σ")
        lc = build_light_curves(
            fits_files,
            fwhm_px=cfg.get("fwhm_px", 5.0),
            aperture_radius=cfg.get("aperture_radius", 8.0),
            n_ref_stars=cfg.get("n_ref_stars", 20),
            log_callback=self.log_line.emit,
            cancel_event=self._cancel)
        self.lc_ready.emit(lc)
        alerts = detect_photometric_transients(lc, threshold_sigma=threshold)
        for a in alerts:
            a.update(alert_id=make_alert_id(a["frame_idx"]),
                     timestamp_utc="",
                     fits_file="", delta_flux=0.0, area_px=1,
                     ra=None, dec=None, classification="unknown",
                     verified=False, notes="",
                     mpc_match=None, vsx_match=None, tns_match=None)
            all_alerts.append(a)
            self.alert_found.emit(a)
            self.log_line.emit(
                f"  L3 #{len(all_alerts)}: star {a['star_id']} "
                f"({a['x']:.1f},{a['y']:.1f}) "
                f"SNR={a['snr']:.1f} Δmag~{a.get('delta_mag_est','?')} "
                f"{a['type']}")

    def _run_crossmatch(self, cfg, all_alerts):
        fits_files = cfg["fits_files"]
        self.log_line.emit(
            f"Cross-matching {len(all_alerts)} alerts vs catalogs...")
        try:
            hdr = astropy_fits.getheader(fits_files[0])
        except Exception:
            hdr = None
        if hdr is None:
            self.log_line.emit("  No header available — skipping cross-match")
            return
        radius = cfg.get("crossmatch_radius_arcsec", 10.0)
        for a in all_alerts:
            if self._cancel.is_set():
                return
            match = crossmatch_alert(
                a["x"], a["y"], hdr,
                search_radius_arcsec=radius,
                log_callback=self.log_line.emit)
            a.update(match)
            if a["classification"] == "unknown":
                self.log_line.emit(
                    f"  ★ UNIDENTIFIED: ({a['x']:.1f},{a['y']:.1f}) "
                    f"RA={a.get('ra','?')} Dec={a.get('dec','?')}")

    def _get_timestamp(self, fits_path):
        try:
            hdr = astropy_fits.getheader(fits_path)
            return hdr.get("DATE-OBS", "")
        except Exception:
            return ""

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()


# ──────────────────────────────────────────────────────────────────────────────
# Light curve canvas
# ──────────────────────────────────────────────────────────────────────────────

class LightCurveCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 4), facecolor="#1e2128")
        self.ax  = self.fig.add_subplot(111)
        self._style_ax()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style_ax(self):
        self.ax.set_facecolor("#252930")
        self.ax.tick_params(colors="#dde3ee", labelsize=8)
        for s in ["bottom", "left"]:
            self.ax.spines[s].set_color("#3a4055")
        for s in ["top", "right"]:
            self.ax.spines[s].set_visible(False)
        self.ax.set_xlabel("Frame", color="#7a8499", fontsize=8)
        self.ax.set_ylabel("Relative flux", color="#7a8499", fontsize=8)
        self.ax.set_title("Photometric light curves", color="#5ba3ff", fontsize=9)

    def show_light_curves(self, lc, alerted_stars):
        self.ax.clear()
        self._style_ax()
        for star_id in lc.get("star_ids", []):
            fluxes = np.array(lc["fluxes"][star_id])
            valid  = ~np.isnan(fluxes)
            if np.sum(valid) < 3:
                continue
            med = np.median(fluxes[valid])
            if med <= 0:
                continue
            norm = fluxes / med
            xs   = np.arange(len(norm))
            if star_id in alerted_stars:
                self.ax.plot(xs[valid], norm[valid],
                             color="#cc4444", linewidth=1.5,
                             alpha=0.9, zorder=3, label=f"⚠ Star {star_id}")
            else:
                self.ax.plot(xs[valid], norm[valid],
                             color="#3a4055", linewidth=0.8,
                             alpha=0.5, zorder=1)
        self.ax.axhline(1.0, color="#4a9eff", linewidth=0.8,
                         linestyle="--", alpha=0.5)
        if alerted_stars:
            self.ax.legend(facecolor="#2d3240", edgecolor="#3a4055",
                           labelcolor="#dde3ee", fontsize=7, loc="upper right")
        self.fig.tight_layout(pad=0.5)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
# Cutout canvas (3-panel preview)
# ──────────────────────────────────────────────────────────────────────────────

class CutoutCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 3), facecolor="#1e2128")
        super().__init__(self.fig)
        self.setParent(parent)

    def show_cutout(self, alert, fits_files):
        self.fig.clear()
        frame_idx = alert.get("frame_idx", 0)
        x, y = alert.get("x", 0), alert.get("y", 0)

        def _load_safe(idx):
            if idx < 0 or idx >= len(fits_files):
                return None
            try:
                return make_cutout_preview(fits_files[idx], x, y)
            except Exception:
                return None

        cutout_prev = _load_safe(frame_idx - 1)
        cutout_curr = _load_safe(frame_idx)
        cutout_diff = None
        if cutout_prev is not None and cutout_curr is not None:
            if cutout_prev.shape == cutout_curr.shape:
                cutout_diff = cutout_curr - cutout_prev

        panels = [
            (cutout_prev, "Before",     "gray"),
            (cutout_curr, "Alert frame","inferno"),
            (cutout_diff, "Difference", "RdBu_r"),
        ]

        for i, (data, title, cmap) in enumerate(panels):
            ax = self.fig.add_subplot(1, 3, i + 1)
            ax.set_facecolor("#252930")
            ax.set_title(title, color="#5ba3ff", fontsize=8)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            if data is not None:
                vmin, vmax = np.percentile(data, [1, 99])
                ax.imshow(data, origin="lower", cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation="nearest")
                if i == 1:
                    half = data.shape[0] // 2
                    circ = __import__("matplotlib.patches", fromlist=["Circle"]).Circle(
                        (data.shape[1]//2, half), radius=8,
                        edgecolor="#ff6b35", facecolor="none", linewidth=1.5)
                    ax.add_patch(circ)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="#7a8499", transform=ax.transAxes)

        self.fig.tight_layout(pad=0.5)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transient Detection — Siril")
        self.resize(1280, 820)
        self._alerts    = []
        self._lc_data   = {}
        self._fits_files = []
        self._worker    = None
        self._cancel_ev = threading.Event()
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        # Title bar
        title_bar = QWidget()
        tbl = QHBoxLayout(title_bar)
        tbl.setContentsMargins(0, 0, 0, 0)
        lbl_title = QLabel("⚡  Transient Detection  —  Siril")
        lbl_title.setStyleSheet(f"color:{SIRIL_NOVA}; font-size:13pt; font-weight:bold;")
        lbl_version = QLabel("v1.0  |  3-layer detection pipeline")
        lbl_version.setObjectName("dim")
        tbl.addWidget(lbl_title)
        tbl.addStretch()
        tbl.addWidget(lbl_version)
        root.addWidget(title_bar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizes([400, 880])
        root.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())

        # Bottom controls
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(8)
        bl.addWidget(self._progress, stretch=1)

        self._btn_run = QPushButton("▶  Run detection")
        self._btn_run.setObjectName("primary")
        self._btn_run.setFixedHeight(34)
        self._btn_run.clicked.connect(self._run)
        bl.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setFixedHeight(34)
        self._btn_cancel.clicked.connect(self._cancel)
        bl.addWidget(self._btn_cancel)

        root.addWidget(bottom)

        self._status_bar = QLabel("Ready")
        self._status_bar.setObjectName("dim")
        root.addWidget(self._status_bar)

    # ── Left panel ─────────────────────────────────────────────────────────

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(420)
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(0, 0, 4, 0)
        vl.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner  = QWidget()
        il     = QVBoxLayout(inner)
        il.setContentsMargins(4, 4, 4, 4)
        il.setSpacing(8)
        scroll.setWidget(inner)
        vl.addWidget(scroll)

        # Input sequence
        grp_input = QGroupBox("Input sequence")
        gl = QVBoxLayout(grp_input)

        fl_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("FITS folder path…")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_folder)
        fl_row.addWidget(self._folder_edit)
        fl_row.addWidget(btn_browse)
        gl.addLayout(fl_row)

        scan_row = QHBoxLayout()
        btn_scan = QPushButton("🔍  Scan")
        btn_scan.clicked.connect(self._scan_folder)
        self._lbl_file_count = QLabel("No files loaded")
        self._lbl_file_count.setObjectName("dim")
        scan_row.addWidget(btn_scan)
        scan_row.addWidget(self._lbl_file_count, stretch=1)
        gl.addLayout(scan_row)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Filename", "DATE-OBS header"])
        sort_row.addWidget(self._sort_combo, stretch=1)
        gl.addLayout(sort_row)
        il.addWidget(grp_input)

        # Detection layers
        grp_layers = QGroupBox("Detection layers")
        ll = QVBoxLayout(grp_layers)

        # Layer 1
        self._chk_l1 = QCheckBox("Layer 1 — Frame difference")
        self._chk_l1.setChecked(True)
        ll.addWidget(self._chk_l1)
        l1_params = QHBoxLayout()
        l1_params.addWidget(QLabel("Threshold:"))
        self._sl_l1 = QSlider(Qt.Orientation.Horizontal)
        self._sl_l1.setRange(20, 150)
        self._sl_l1.setValue(60)
        self._lbl_l1 = QLabel("6.0 σ")
        self._sl_l1.valueChanged.connect(
            lambda v: self._lbl_l1.setText(f"{v/10:.1f} σ"))
        l1_params.addWidget(self._sl_l1)
        l1_params.addWidget(self._lbl_l1)
        l1_params.addWidget(QLabel("Min px:"))
        self._sp_minpx = QSpinBox()
        self._sp_minpx.setRange(1, 20)
        self._sp_minpx.setValue(3)
        self._sp_minpx.setFixedWidth(52)
        l1_params.addWidget(self._sp_minpx)
        ll.addLayout(l1_params)

        sep1 = QFrame(); sep1.setObjectName("separator"); ll.addWidget(sep1)

        # Layer 2
        self._chk_l2 = QCheckBox("Layer 2 — ZOGY subtraction")
        self._chk_l2.setChecked(HAS_ZOGY)
        self._chk_l2.setEnabled(HAS_ZOGY)
        ll.addWidget(self._chk_l2)
        if not HAS_ZOGY:
            lbl_no_zogy = QLabel("⚠ Requires zogy_subtract.py in same folder")
            lbl_no_zogy.setObjectName("warn")
            ll.addWidget(lbl_no_zogy)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference:"))
        self._ref_edit = QLineEdit()
        self._ref_edit.setPlaceholderText("Reference FITS image…")
        btn_ref = QPushButton("…")
        btn_ref.setFixedWidth(32)
        btn_ref.clicked.connect(self._browse_reference)
        ref_row.addWidget(self._ref_edit)
        ref_row.addWidget(btn_ref)
        ll.addLayout(ref_row)

        l2_params = QHBoxLayout()
        l2_params.addWidget(QLabel("Threshold:"))
        self._sl_l2 = QSlider(Qt.Orientation.Horizontal)
        self._sl_l2.setRange(20, 150)
        self._sl_l2.setValue(50)
        self._lbl_l2 = QLabel("5.0 σ")
        self._sl_l2.valueChanged.connect(
            lambda v: self._lbl_l2.setText(f"{v/10:.1f} σ"))
        l2_params.addWidget(self._sl_l2)
        l2_params.addWidget(self._lbl_l2)
        ll.addLayout(l2_params)

        sep2 = QFrame(); sep2.setObjectName("separator"); ll.addWidget(sep2)

        # Layer 3
        self._chk_l3 = QCheckBox("Layer 3 — Photometric monitoring")
        self._chk_l3.setChecked(False)
        ll.addWidget(self._chk_l3)
        lbl_l3_hint = QLabel("Slower — builds light curves for all stars")
        lbl_l3_hint.setObjectName("dim")
        ll.addWidget(lbl_l3_hint)

        l3_params = QFormLayout()
        self._sl_l3 = QSlider(Qt.Orientation.Horizontal)
        self._sl_l3.setRange(20, 100)
        self._sl_l3.setValue(40)
        self._lbl_l3 = QLabel("4.0 σ")
        self._sl_l3.valueChanged.connect(
            lambda v: self._lbl_l3.setText(f"{v/10:.1f} σ"))
        l3_thr_row = QHBoxLayout()
        l3_thr_row.addWidget(self._sl_l3)
        l3_thr_row.addWidget(self._lbl_l3)
        l3_params.addRow("Threshold:", l3_thr_row)

        self._sp_refstars = QSpinBox()
        self._sp_refstars.setRange(5, 50)
        self._sp_refstars.setValue(20)
        l3_params.addRow("Ref stars:", self._sp_refstars)

        self._dsp_aperture = QDoubleSpinBox()
        self._dsp_aperture.setRange(3.0, 20.0)
        self._dsp_aperture.setValue(8.0)
        self._dsp_aperture.setSingleStep(0.5)
        l3_params.addRow("Aperture (px):", self._dsp_aperture)
        ll.addLayout(l3_params)

        il.addWidget(grp_layers)

        # Catalog cross-match
        grp_cat = QGroupBox("Catalog cross-match")
        cl = QVBoxLayout(grp_cat)
        self._chk_crossmatch = QCheckBox("Cross-match against VSX / MPC")
        self._chk_crossmatch.setChecked(False)
        cl.addWidget(self._chk_crossmatch)
        lbl_cat_hint = QLabel("Requires internet. Identifies known variables/asteroids.")
        lbl_cat_hint.setObjectName("dim")
        cl.addWidget(lbl_cat_hint)

        cat_params = QFormLayout()
        self._dsp_radius = QDoubleSpinBox()
        self._dsp_radius.setRange(1.0, 60.0)
        self._dsp_radius.setValue(10.0)
        self._dsp_radius.setSuffix(" \"")
        cat_params.addRow("Search radius:", self._dsp_radius)
        cl.addLayout(cat_params)

        self._lbl_no_wcs = QLabel("⚠ Images need plate-solved WCS for catalog matching")
        self._lbl_no_wcs.setObjectName("warn")
        self._lbl_no_wcs.setVisible(False)
        cl.addWidget(self._lbl_no_wcs)
        self._chk_crossmatch.toggled.connect(self._lbl_no_wcs.setVisible)
        il.addWidget(grp_cat)

        # Export
        grp_exp = QGroupBox("Export")
        el = QVBoxLayout(grp_exp)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self._out_edit = QLineEdit()
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(32)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit)
        out_row.addWidget(btn_out)
        el.addLayout(out_row)

        self._chk_csv   = QCheckBox("Export CSV");          self._chk_csv.setChecked(True)
        self._chk_tns   = QCheckBox("Export TNS-format report")
        self._chk_aavso = QCheckBox("Export AAVSO report")
        el.addWidget(self._chk_csv)
        el.addWidget(self._chk_tns)
        el.addWidget(self._chk_aavso)

        obs_params = QFormLayout()
        self._obs_name = QLineEdit()
        self._obs_code = QLineEdit()
        self._inst_edit = QLineEdit()
        self._target_edit = QLineEdit()
        obs_params.addRow("Observer name:", self._obs_name)
        obs_params.addRow("AAVSO code:", self._obs_code)
        obs_params.addRow("Instrument:", self._inst_edit)
        obs_params.addRow("Target name:", self._target_edit)
        self._obs_widget = QWidget()
        self._obs_widget.setLayout(obs_params)
        self._obs_widget.setVisible(False)
        el.addWidget(self._obs_widget)

        def _update_obs_vis():
            self._obs_widget.setVisible(
                self._chk_tns.isChecked() or self._chk_aavso.isChecked())
        self._chk_tns.toggled.connect(_update_obs_vis)
        self._chk_aavso.toggled.connect(_update_obs_vis)
        il.addWidget(grp_exp)
        il.addStretch()

        return panel

    # ── Right panel ────────────────────────────────────────────────────────

    def _build_right_panel(self):
        panel = QWidget()
        vl    = QVBoxLayout(panel)
        vl.setContentsMargins(4, 0, 0, 0)
        vl.setSpacing(6)

        self._tabs = QTabWidget()
        vl.addWidget(self._tabs)

        # ── Tab: Alerts ──────────────────────────────────────────────────
        alert_tab = QWidget()
        atl = QVBoxLayout(alert_tab)

        # Summary badges
        badge_row = QHBoxLayout()
        self._badge_total  = self._make_badge("Total: 0",    SIRIL_TEXT)
        self._badge_l1     = self._make_badge("Layer 1: 0",  SIRIL_TEXT_DIM)
        self._badge_l2     = self._make_badge("Layer 2: 0",  SIRIL_ACCENT)
        self._badge_l3     = self._make_badge("Layer 3: 0",  SIRIL_SUCCESS)
        self._badge_unknwn = self._make_badge("★ Unknown: 0", SIRIL_NOVA)
        for b in [self._badge_total, self._badge_l1, self._badge_l2,
                  self._badge_l3, self._badge_unknwn]:
            badge_row.addWidget(b)
        badge_row.addStretch()
        atl.addLayout(badge_row)

        # Alert table
        self._alert_table = QTableWidget(0, 9)
        self._alert_table.setHorizontalHeaderLabels(
            ["#", "Time", "File", "X", "Y", "SNR", "Type", "Layer", "Status"])
        self._alert_table.setAlternatingRowColors(True)
        self._alert_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._alert_table.horizontalHeader().setStretchLastSection(True)
        self._alert_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._alert_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._alert_table.doubleClicked.connect(self._on_alert_dblclick)
        atl.addWidget(self._alert_table)

        # Button row
        btn_row = QHBoxLayout()
        btn_exp_csv = QPushButton("Export CSV");  btn_exp_csv.setObjectName("export")
        btn_exp_csv.clicked.connect(self._export_csv)
        btn_copy    = QPushButton("Copy unknowns"); btn_copy.setObjectName("alert")
        btn_copy.clicked.connect(self._copy_unknowns)
        btn_fp      = QPushButton("Filter false positives")
        btn_fp.clicked.connect(self._filter_fp)
        btn_verified = QPushButton("Mark verified")
        btn_verified.clicked.connect(self._mark_verified)
        for b in [btn_exp_csv, btn_copy, btn_fp, btn_verified]:
            btn_row.addWidget(b)
        btn_row.addStretch()
        atl.addLayout(btn_row)
        self._tabs.addTab(alert_tab, "⚡  Alerts  [0]")

        # ── Tab: Light curves ────────────────────────────────────────────
        lc_tab = QWidget()
        ltl = QVBoxLayout(lc_tab)
        self._lc_canvas = LightCurveCanvas()
        ltl.addWidget(self._lc_canvas)
        lbl_lc_hint = QLabel("Red lines = alerted stars")
        lbl_lc_hint.setObjectName("dim")
        ltl.addWidget(lbl_lc_hint)
        self._tabs.addTab(lc_tab, "📈  Light Curves")

        # ── Tab: Cutout preview ──────────────────────────────────────────
        cut_tab = QWidget()
        ctl = QVBoxLayout(cut_tab)
        self._cutout_canvas = CutoutCanvas()
        ctl.addWidget(self._cutout_canvas)
        self._lbl_cutout_detail = QLabel("")
        self._lbl_cutout_detail.setObjectName("dim")
        self._lbl_cutout_detail.setWordWrap(True)
        ctl.addWidget(self._lbl_cutout_detail)
        self._tabs.addTab(cut_tab, "🖼  Cutout Preview")

        # ── Tab: Log ─────────────────────────────────────────────────────
        log_tab = QWidget()
        logl = QVBoxLayout(log_tab)
        self._log_pane = QPlainTextEdit()
        self._log_pane.setReadOnly(True)
        logl.addWidget(self._log_pane)
        self._tabs.addTab(log_tab, "📋  Log")

        return panel

    def _make_badge(self, text, color):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"background:{SIRIL_BG3}; color:{color}; "
            f"border:1px solid {SIRIL_BORDER}; border-radius:4px; "
            f"padding:3px 8px; font-weight:bold;")
        return lbl

    # ── Folder / file helpers ──────────────────────────────────────────────

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self._folder_edit.setText(d)
            if not self._out_edit.text():
                self._out_edit.setText(d)

    def _browse_reference(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select reference FITS", filter="FITS (*.fit *.fits *.fts)")
        if f:
            self._ref_edit.setText(f)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self._out_edit.setText(d)

    def _scan_folder(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Please select a valid FITS folder.")
            return
        patterns = ["*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        files = list(set(files))

        if self._sort_combo.currentIndex() == 0:
            files.sort()
        else:
            def _sort_by_date(fp):
                try:
                    hdr = astropy_fits.getheader(fp)
                    return hdr.get("DATE-OBS", "")
                except Exception:
                    return ""
            files.sort(key=_sort_by_date)

        self._fits_files = files
        n = len(files)
        if n:
            sample = astropy_fits.getdata(files[0])
            size_str = f"{sample.shape[-1]}×{sample.shape[-2]}" if sample.ndim >= 2 else "?"
            self._lbl_file_count.setText(f"{n} files  |  {size_str} px")
            self._log(f"Scanned: {n} FITS files in {folder}")
        else:
            self._lbl_file_count.setText("No FITS files found")

    # ── Run / cancel ───────────────────────────────────────────────────────

    def _run(self):
        if not self._fits_files:
            QMessageBox.warning(self, "No files", "Scan a folder first.")
            return

        self._alerts    = []
        self._lc_data   = {}
        self._alert_table.setRowCount(0)
        self._update_badges()
        self._cancel_ev = threading.Event()

        cfg = {
            "fits_files":           self._fits_files,
            "fits_dir":             self._folder_edit.text().strip(),
            "output_dir":           self._out_edit.text().strip()
                                    or self._folder_edit.text().strip(),
            "layer1_enabled":       self._chk_l1.isChecked(),
            "layer1_threshold":     self._sl_l1.value() / 10,
            "min_pixels":           self._sp_minpx.value(),
            "border_margin":        20,
            "layer2_enabled":       self._chk_l2.isChecked(),
            "layer2_threshold":     self._sl_l2.value() / 10,
            "reference_path":       self._ref_edit.text().strip(),
            "layer3_enabled":       self._chk_l3.isChecked(),
            "layer3_threshold":     self._sl_l3.value() / 10,
            "n_ref_stars":          self._sp_refstars.value(),
            "aperture_radius":      self._dsp_aperture.value(),
            "fwhm_px":              5.0,
            "crossmatch_enabled":   self._chk_crossmatch.isChecked(),
            "crossmatch_radius_arcsec": self._dsp_radius.value(),
            "export_csv":           self._chk_csv.isChecked(),
            "export_tns":           self._chk_tns.isChecked(),
            "export_aavso":         self._chk_aavso.isChecked(),
            "observer_name":        self._obs_name.text().strip(),
            "observer_code":        self._obs_code.text().strip() or "XXX",
            "instrument":           self._inst_edit.text().strip(),
            "target_name":          self._target_edit.text().strip() or "UNKNOWN",
        }

        self._worker = DetectionWorker(cfg, self._cancel_ev)
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.alert_found.connect(self._on_alert_found)
        self._worker.lc_ready.connect(self._on_lc_ready)
        self._worker.finished.connect(self._on_finished)

        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._status_bar.setText("⚡ Running detection…")
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self._btn_cancel.setEnabled(False)
        self._status_bar.setText("Cancelling…")

    # ── Worker signal handlers ─────────────────────────────────────────────

    def _log(self, msg):
        self._log_pane.appendPlainText(msg)

    def _on_progress(self, current, total, msg):
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(current)
        self._status_bar.setText(f"⚡ {msg}")

    def _on_alert_found(self, alert):
        self._alerts.append(alert)
        self._add_table_row(alert)
        self._update_badges()
        n = len(self._alerts)
        self._tabs.setTabText(0, f"⚡  Alerts  [{n}]")

    def _on_lc_ready(self, lc):
        self._lc_data = lc
        alerted = {a["star_id"] for a in self._alerts if a.get("layer") == 3}
        self._lc_canvas.show_light_curves(lc, alerted)

    def _on_finished(self, result):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        if result.get("success"):
            unknown = result.get("unknown_alerts", 0)
            total   = result.get("total_alerts", 0)
            self._status_bar.setText(
                f"✓  Complete — {total} alerts, {unknown} unidentified")
            if unknown:
                self._status_bar.setStyleSheet(f"color:{SIRIL_NOVA};")
            else:
                self._status_bar.setStyleSheet(f"color:{SIRIL_SUCCESS};")
        else:
            err = result.get("error", "Unknown error")
            self._status_bar.setText(f"✗  {err}")
            self._status_bar.setStyleSheet(f"color:{SIRIL_ERROR};")

    # ── Alert table helpers ────────────────────────────────────────────────

    def _add_table_row(self, a):
        row = self._alert_table.rowCount()
        self._alert_table.insertRow(row)

        layer_colors = {1: None, 2: QColor(45, 106, 191, 30),
                        3: QColor(76, 175, 125, 30)}

        values = [
            str(row + 1),
            str(a.get("timestamp_utc", ""))[:19],
            a.get("fits_file", ""),
            f"{a.get('x', 0):.1f}",
            f"{a.get('y', 0):.1f}",
            f"{a.get('snr', 0):.2f}",
            a.get("type", ""),
            str(a.get("layer", "")),
            a.get("classification", "unknown"),
        ]

        bg   = layer_colors.get(a.get("layer"))
        nova = a.get("classification") == "unknown"

        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if bg:
                item.setBackground(bg)
            if nova:
                item.setForeground(QColor(SIRIL_NOVA))
            self._alert_table.setItem(row, col, item)

    def _update_badges(self):
        total   = len(self._alerts)
        l1 = sum(1 for a in self._alerts if a.get("layer") == 1)
        l2 = sum(1 for a in self._alerts if a.get("layer") == 2)
        l3 = sum(1 for a in self._alerts if a.get("layer") == 3)
        unk = sum(1 for a in self._alerts if a.get("classification") == "unknown")
        self._badge_total.setText(f"Total: {total}")
        self._badge_l1.setText(f"Layer 1: {l1}")
        self._badge_l2.setText(f"Layer 2: {l2}")
        self._badge_l3.setText(f"Layer 3: {l3}")
        self._badge_unknwn.setText(f"★ Unknown: {unk}")

    def _on_alert_dblclick(self, index):
        row   = index.row()
        if row >= len(self._alerts):
            return
        alert = self._alerts[row]
        self._tabs.setCurrentIndex(2)  # Cutout tab

        fits_files = self._fits_files
        self._cutout_canvas.show_cutout(alert, fits_files)
        self._lbl_cutout_detail.setText(
            f"Alert #{row+1}  |  {alert.get('alert_id','')}  |  "
            f"({alert.get('x',0):.1f}, {alert.get('y',0):.1f})  |  "
            f"SNR={alert.get('snr',0):.2f}  |  {alert.get('type','')}  |  "
            f"Layer {alert.get('layer','')}  |  "
            f"Classification: {alert.get('classification','?')}")

    # ── Alert action buttons ───────────────────────────────────────────────

    def _export_csv(self):
        if not self._alerts:
            QMessageBox.information(self, "No alerts", "No alerts to export.")
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", filter="CSV (*.csv)")
        if out:
            export_alerts_csv(self._alerts, out)
            self._log(f"Exported {len(self._alerts)} alerts → {out}")

    def _copy_unknowns(self):
        unknowns = [a for a in self._alerts
                    if a.get("classification") == "unknown"]
        if not unknowns:
            QMessageBox.information(self, "No unknowns", "No unidentified alerts.")
            return
        lines = []
        for a in unknowns:
            ra  = a.get("ra",  "N/A")
            dec = a.get("dec", "N/A")
            lines.append(
                f"{a['alert_id']}  RA={ra}  Dec={dec}  "
                f"SNR={a['snr']:.2f}  {a['type']}  {a.get('fits_file','')}")
        QApplication.clipboard().setText("\n".join(lines))
        self._log(f"Copied {len(unknowns)} unknown alerts to clipboard")

    def _filter_fp(self):
        """Remove single-pixel Layer 1 detections (likely cosmic rays)."""
        before = len(self._alerts)
        self._alerts = [
            a for a in self._alerts
            if not (a.get("layer") == 1 and a.get("area_px", 0) == 1)
        ]
        removed = before - len(self._alerts)
        self._rebuild_table()
        self._update_badges()
        self._tabs.setTabText(0, f"⚡  Alerts  [{len(self._alerts)}]")
        self._log(f"False-positive filter: removed {removed} single-pixel detections")

    def _mark_verified(self):
        for idx in self._alert_table.selectedItems():
            row = idx.row()
            if row < len(self._alerts):
                self._alerts[row]["verified"] = True
                item = self._alert_table.item(row, 8)
                if item:
                    item.setText("✓ verified")
                    item.setForeground(QColor(SIRIL_SUCCESS))

    def _rebuild_table(self):
        self._alert_table.setRowCount(0)
        for a in self._alerts:
            self._add_table_row(a)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
