"""
moving_object.py — Siril Script #17
Moving Object Detector: detects asteroids, comets, TNOs in FITS sequences.
Cross-matches against MPC. Exports MPC One-Line Astrometry format.
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

# ── Core imports ──────────────────────────────────────────────────────────────
import os, sys, csv, glob, json, threading
from datetime import datetime
import numpy as np
from scipy.stats import linregress
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

try:
    from astroquery.mpc import MPC
    from astroquery.vizier import Vizier
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False

# ── Theme ─────────────────────────────────────────────────────────────────────
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
SIRIL_ASTEROID = "#ffaa22"

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
QPushButton#asteroid {{
    background-color: #2d2000;
    border-color: {SIRIL_ASTEROID};
    color: {SIRIL_ASTEROID};
    font-weight: bold;
    text-align: center;
}}
QPushButton#asteroid:hover {{ background-color: #3d2d00; }}
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
QLabel#section   {{ color: {SIRIL_SECTION};   font-weight: bold; padding-top: 6px; }}
QLabel#dim       {{ color: {SIRIL_TEXT_DIM};  font-size: 9pt; }}
QLabel#ok        {{ color: {SIRIL_SUCCESS};   font-weight: bold; }}
QLabel#err       {{ color: {SIRIL_ERROR};     font-weight: bold; }}
QLabel#warn      {{ color: {SIRIL_WARNING};   font-weight: bold; }}
QLabel#asteroid  {{ color: {SIRIL_ASTEROID};  font-weight: bold; font-size: 11pt; }}
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
# SECTION 5 — SOURCE DETECTION PER FRAME
# ══════════════════════════════════════════════════════════════════════════════

def detect_sources_in_frame(image: np.ndarray,
                              frame_idx: int,
                              jd: float,
                              threshold_sigma: float = 5.0,
                              fwhm_guess: float = 5.0,
                              max_sources: int = 500) -> list:
    _, med, std = sigma_clipped_stats(image, sigma=3.0)
    if std <= 0:
        return []
    daofind = DAOStarFinder(
        fwhm=fwhm_guess,
        threshold=threshold_sigma * std,
        sharplo=0.2, sharphi=0.9,
        roundlo=-0.8, roundhi=0.8
    )
    sources = daofind(image - med)
    if sources is None or len(sources) == 0:
        return []
    sat_limit = 0.92 * float(np.percentile(image, 99.99))
    result = []
    sources.sort("peak")
    sources.reverse()
    for src in sources[:max_sources]:
        if float(src["peak"]) + med > sat_limit:
            continue
        result.append({
            "x":         float(src["xcentroid"]),
            "y":         float(src["ycentroid"]),
            "flux":      float(src["flux"]),
            "peak":      float(src["peak"]),
            "frame_idx": frame_idx,
            "jd":        jd,
        })
    return result


def get_jd_from_header(header) -> float:
    date_obs = (header.get("DATE-OBS", "") or header.get("DATE", ""))
    if date_obs:
        try:
            return float(Time(date_obs, format="isot", scale="utc").jd)
        except Exception:
            pass
    jd = header.get("JD", header.get("MJD-OBS", None))
    if jd is not None:
        return float(jd)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CROSS-FRAME MATCHING AND MOTION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def match_sources_across_frames(all_sources: list,
                                  n_frames: int,
                                  match_radius_px: float = 5.0,
                                  min_frames: int = None) -> list:
    if min_frames is None:
        min_frames = max(2, n_frames - 1)

    def find_nearest(source, candidates, radius):
        if not candidates:
            return None
        xs = np.array([c["x"] for c in candidates])
        ys = np.array([c["y"] for c in candidates])
        dists = np.sqrt((xs - source["x"])**2 + (ys - source["y"])**2)
        idx = int(np.argmin(dists))
        if dists[idx] <= radius:
            return idx
        return None

    tracks = []
    used_per_frame = [set() for _ in range(n_frames)]

    for i, src0 in enumerate(all_sources[0]):
        if i in used_per_frame[0]:
            continue
        track = [src0]
        last_src = src0
        used_per_frame[0].add(i)

        for frame_idx in range(1, n_frames):
            candidates = [s for j, s in enumerate(all_sources[frame_idx])
                          if j not in used_per_frame[frame_idx]]
            match_idx = find_nearest(last_src, candidates, match_radius_px)
            if match_idx is not None:
                all_cands = all_sources[frame_idx]
                used_ctr = 0
                real_idx = None
                for j, s in enumerate(all_cands):
                    if j not in used_per_frame[frame_idx]:
                        if used_ctr == match_idx:
                            real_idx = j
                            break
                        used_ctr += 1
                if real_idx is not None:
                    track.append(all_cands[real_idx])
                    used_per_frame[frame_idx].add(real_idx)
                    last_src = all_cands[real_idx]
                else:
                    break
            else:
                break

        if len(track) >= min_frames:
            tracks.append(track)

    return tracks


def fit_linear_motion(track: list) -> dict:
    frames = np.array([s["frame_idx"] for s in track], dtype=float)
    xs     = np.array([s["x"]         for s in track], dtype=float)
    ys     = np.array([s["y"]         for s in track], dtype=float)
    jds    = np.array([s["jd"]        for s in track], dtype=float)

    slope_x, inter_x, r_x, _, _ = linregress(frames, xs)
    slope_y, inter_y, r_y, _, _ = linregress(frames, ys)

    r2_x = r_x ** 2
    r2_y = r_y ** 2
    r2_mean = (r2_x + r2_y) / 2.0

    dt_days  = float(jds[-1] - jds[0]) if jds[-1] != jds[0] else 1.0
    dx_total = float(xs[-1] - xs[0])
    dy_total = float(ys[-1] - ys[0])
    vx_day   = dx_total / dt_days if dt_days > 0 else 0.0
    vy_day   = dy_total / dt_days if dt_days > 0 else 0.0
    speed_pf = float(np.sqrt(slope_x**2 + slope_y**2))
    total_px = float(np.sqrt(dx_total**2 + dy_total**2))

    return {
        "vx_px_per_frame":    round(slope_x, 4),
        "vy_px_per_frame":    round(slope_y, 4),
        "vx_px_per_day":      round(vx_day, 3),
        "vy_px_per_day":      round(vy_day, 3),
        "speed_px_per_frame": round(speed_pf, 4),
        "r2_x":               round(r2_x, 4),
        "r2_y":               round(r2_y, 4),
        "r2_mean":            round(r2_mean, 4),
        "is_linear":          r2_mean > 0.95,
        "x_start":            round(float(xs[0]), 2),
        "y_start":            round(float(ys[0]), 2),
        "x_end":              round(float(xs[-1]), 2),
        "y_end":              round(float(ys[-1]), 2),
        "total_motion_px":    round(total_px, 2),
    }


def filter_moving_objects(tracks: list,
                            min_motion_px: float = 3.0,
                            max_motion_px_per_frame: float = 200.0,
                            min_r2: float = 0.90) -> list:
    movers = []
    for track in tracks:
        if len(track) < 2:
            continue
        fit = fit_linear_motion(track)
        if fit["total_motion_px"] < min_motion_px:
            continue
        if fit["speed_px_per_frame"] > max_motion_px_per_frame:
            continue
        if fit["r2_mean"] < min_r2:
            continue
        movers.append((track, fit))
    movers.sort(key=lambda x: x[1]["total_motion_px"], reverse=True)
    return movers


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — WCS COORDINATE CONVERSION AND MPC CROSS-MATCH
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_radec(x: float, y: float, header) -> tuple:
    try:
        from astropy.wcs import WCS
        wcs = WCS(header)
        sky = wcs.pixel_to_world(x, y)
        return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        return None, None


def estimate_pixel_scale(header) -> float:
    try:
        from astropy.wcs import WCS
        import astropy.units as u
        wcs   = WCS(header)
        scale = wcs.proj_plane_pixel_scales()
        return float(scale[0].to(u.arcsec).value)
    except Exception:
        pass
    for kw in ["PIXSCALE", "SECPIX", "SCALE"]:
        val = header.get(kw)
        if val is not None:
            return float(val)
    return None


def crossmatch_mpc(ra_deg: float, dec_deg: float,
                    jd: float,
                    search_radius_arcsec: float = 30.0,
                    log_callback=None):
    if not HAS_ASTROQUERY:
        return None
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astroquery.vizier import Vizier
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
        v = Vizier(columns=["Name", "Number", "H", "Epoch", "e", "i", "Peri", "Node"])
        result = v.query_region(
            coord,
            radius=search_radius_arcsec * u.arcsec,
            catalog="B/mpc/mpcorb"
        )
        if result and len(result) > 0 and len(result[0]) > 0:
            row = result[0][0]
            return {
                "designation":     str(row.get("Name", row.get("Number", "?"))),
                "magnitude":       float(row["H"]) if "H" in row.colnames else None,
                "type":            "asteroid",
                "distance_arcsec": 0.0,
            }
    except Exception as e:
        if log_callback:
            log_callback(f"  MPC query failed: {e}")
    return None


def motion_arcsec_per_hour(motion_fit: dict,
                             pixel_scale_arcsec: float,
                             frame_interval_hours: float) -> tuple:
    vx = motion_fit["vx_px_per_frame"] * pixel_scale_arcsec
    vy = motion_fit["vy_px_per_frame"] * pixel_scale_arcsec
    if frame_interval_hours > 0:
        vx /= frame_interval_hours
        vy /= frame_interval_hours
    return round(vx, 3), round(vy, 3)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MPC ONE-LINE ASTROMETRY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def format_mpc_oneline(ra_deg, dec_deg, jd, magnitude, band,
                        observatory_code, temp_designation="     K00X00A") -> str:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    t = Time(jd, format="jd", scale="utc")
    date_str = t.strftime("%Y %m %d.")
    frac_day = (jd - int(jd)) if jd > 0 else 0.0
    date_str += f"{frac_day:.5f}"[2:]

    coord   = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    ra_str  = coord.ra.to_string(unit=u.hour, sep=" ", precision=3, pad=True)
    dec_str = coord.dec.to_string(unit=u.deg, sep=" ", precision=2,
                                   alwayssign=True, pad=True)
    mag_str  = f"{magnitude:5.1f}" if magnitude is not None else "     "
    band_str = band[0] if band else "V"

    line = (
        f"{'':5s}"
        f"{temp_designation:7s}"
        f" C "
        f"{date_str:17s}"
        f" "
        f"{ra_str:12s}"
        f"{dec_str:12s}"
        f"{'':9s}"
        f"{mag_str:5s}"
        f"{band_str:1s}"
        f"{'':6s}"
        f"{observatory_code:3s}"
    )
    return line[:80].ljust(80)


def export_mpc_report(movers, pixel_scale_arcsec, observatory_code,
                       output_path, first_header, log_callback=None):
    lines = [
        "COD " + observatory_code,
        "OBS Unknown",
        "MEA Unknown",
        "TEL Unknown CCD",
        "NET UCAC4",
        "COM Generated by Siril moving_object.py",
        "---",
    ]
    for i, (track, fit, mpc_match, ra, dec) in enumerate(movers):
        if mpc_match is not None:
            continue
        if ra is None:
            continue
        src0 = track[0]
        jd   = src0["jd"]
        line = format_mpc_oneline(
            ra_deg=ra, dec_deg=dec, jd=jd,
            magnitude=None, band="V",
            observatory_code=observatory_code,
            temp_designation=f"TMP{i+1:04d} "
        )
        lines.append(line)
    lines.append("---")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    if log_callback:
        log_callback(f"MPC report: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class MoverWorker(QThread):
    progress    = pyqtSignal(int, int, str)
    log_line    = pyqtSignal(str)
    mover_found = pyqtSignal(dict)
    finished    = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg   = self.config
            files = cfg["fits_files"]
            n     = len(files)

            if n < 3:
                self.finished.emit({"success": False,
                                    "error": "Need at least 3 frames for motion detection"})
                return

            # ── Step 1: Detect sources ────────────────────────────────────────
            self.progress.emit(1, n+4, "Detecting sources in each frame...")
            all_sources  = []
            first_header = None

            for i, path in enumerate(files):
                if self._cancel.is_set():
                    self._abort(); return
                self.progress.emit(i+1, n+4, f"Detecting sources: frame {i+1}/{n}")

                with fits.open(path) as hdul:
                    data   = hdul[0].data.astype(np.float32)
                    header = hdul[0].header.copy()
                    if i == 0:
                        first_header = header

                if data.ndim == 3:
                    data = (data[0] if data.shape[0] <= 4
                            else 0.299*data[0]+0.587*data[1]+0.114*data[2])

                jd = get_jd_from_header(header)
                if jd == 0.0:
                    jd = float(i)

                sources = detect_sources_in_frame(
                    data, i, jd,
                    threshold_sigma=cfg.get("threshold_sigma", 5.0),
                    fwhm_guess=cfg.get("fwhm_guess", 5.0),
                    max_sources=cfg.get("max_sources", 500)
                )
                all_sources.append(sources)
                self.log_line.emit(f"  Frame {i+1}: {len(sources)} sources  JD={jd:.5f}")

            if self._cancel.is_set():
                self._abort(); return

            # ── Step 2: Match across frames ───────────────────────────────────
            self.progress.emit(n+1, n+4, "Matching sources across frames...")
            self.log_line.emit("Cross-matching sources between frames...")
            tracks = match_sources_across_frames(
                all_sources, n,
                match_radius_px=cfg.get("match_radius_px", 8.0),
                min_frames=cfg.get("min_frames", max(3, n-1))
            )
            self.log_line.emit(
                f"Found {len(tracks)} candidate tracks "
                f"(spanning ≥{cfg.get('min_frames', max(3,n-1))} frames)")

            if self._cancel.is_set():
                self._abort(); return

            # ── Step 3: Filter for moving objects ─────────────────────────────
            self.progress.emit(n+2, n+4, "Filtering for moving objects...")
            movers = filter_moving_objects(
                tracks,
                min_motion_px=cfg.get("min_motion_px", 3.0),
                max_motion_px_per_frame=cfg.get("max_motion_px", 200.0),
                min_r2=cfg.get("min_r2", 0.90)
            )
            self.log_line.emit(f"Confirmed moving objects: {len(movers)}")
            for track, fit in movers:
                self.log_line.emit(
                    f"  Motion: {fit['total_motion_px']:.1f}px total  "
                    f"speed={fit['speed_px_per_frame']:.2f}px/frame  "
                    f"R²={fit['r2_mean']:.3f}  "
                    f"start=({fit['x_start']:.1f},{fit['y_start']:.1f})")

            if self._cancel.is_set():
                self._abort(); return

            # ── Step 4: WCS + MPC cross-match ─────────────────────────────────
            self.progress.emit(n+3, n+4, "Converting to RA/Dec and cross-matching MPC...")
            pixel_scale = estimate_pixel_scale(first_header) if first_header else None

            full_movers = []
            for track, fit in movers:
                src0 = track[0]
                ra, dec = pixel_to_radec(src0["x"], src0["y"], first_header) \
                          if first_header else (None, None)

                mpc_match = None
                if ra is not None and cfg.get("crossmatch_mpc", True):
                    mpc_match = crossmatch_mpc(
                        ra, dec, src0["jd"],
                        search_radius_arcsec=cfg.get("mpc_search_arcsec", 30.0),
                        log_callback=self.log_line.emit
                    )

                ra_str  = f"{ra:.4f}"  if ra  is not None else "?"
                dec_str = f"{dec:.4f}" if dec is not None else "?"
                if mpc_match:
                    status = f"KNOWN: {mpc_match['designation']}"
                else:
                    status = "UNKNOWN — candidate for reporting"

                self.log_line.emit(
                    f"  ({fit['x_start']:.0f},{fit['y_start']:.0f}) "
                    f"RA={ra_str}  Dec={dec_str}  {status}")

                full_movers.append((track, fit, mpc_match, ra, dec))
                self.mover_found.emit({
                    "track":     track,
                    "fit":       fit,
                    "mpc_match": mpc_match,
                    "ra":        ra,
                    "dec":       dec,
                    "status":    "known" if mpc_match else "unknown",
                })

            # ── Step 5: Export ─────────────────────────────────────────────────
            self.progress.emit(n+4, n+4, "Saving report...")
            out_dir = cfg.get("output_dir", ".")
            os.makedirs(out_dir, exist_ok=True)

            mpc_path = None
            if cfg.get("export_mpc") and full_movers:
                mpc_path = os.path.join(out_dir, "mpc_report.txt")
                export_mpc_report(
                    full_movers,
                    pixel_scale or 1.0,
                    cfg.get("observatory_code", "XXX"),
                    mpc_path,
                    first_header,
                    log_callback=self.log_line.emit
                )

            csv_path = os.path.join(out_dir, "moving_objects.csv")
            self._export_csv(full_movers, csv_path, pixel_scale)
            self.log_line.emit(f"CSV saved: {csv_path}")

            self.finished.emit({
                "success":     True,
                "n_movers":    len(full_movers),
                "n_known":     sum(1 for _, _, m, _, _ in full_movers if m is not None),
                "n_unknown":   sum(1 for _, _, m, _, _ in full_movers if m is None),
                "movers":      full_movers,
                "mpc_path":    mpc_path,
                "csv_path":    csv_path,
                "pixel_scale": pixel_scale,
                "fits_files":  files,
                "first_frame": all_sources,
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _export_csv(self, movers, csv_path, pixel_scale):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "status", "designation",
                "x_start", "y_start", "x_end", "y_end",
                "total_motion_px", "speed_px_per_frame",
                "vx_px_per_day", "vy_px_per_day",
                "r2_mean", "ra_deg", "dec_deg", "n_detections"
            ])
            for i, (track, fit, mpc, ra, dec) in enumerate(movers, 1):
                writer.writerow([
                    i,
                    "known" if mpc else "unknown",
                    mpc["designation"] if mpc else "",
                    fit["x_start"], fit["y_start"],
                    fit["x_end"],   fit["y_end"],
                    fit["total_motion_px"],
                    fit["speed_px_per_frame"],
                    fit["vx_px_per_day"],
                    fit["vy_px_per_day"],
                    fit["r2_mean"],
                    round(ra, 6) if ra else "",
                    round(dec, 6) if dec else "",
                    len(track),
                ])

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — TRAIL CANVAS AND BLINK CANVAS
# ══════════════════════════════════════════════════════════════════════════════

class TrailCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 6), facecolor="#1e2128")
        self.ax  = self.fig.add_subplot(111)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style(self):
        self.ax.set_facecolor("#1e2128")
        self.ax.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_color("#3a4055")
        self.ax.set_title("Detected moving objects", color="#5ba3ff", fontsize=9)

    def show_trails(self, first_frame: np.ndarray, movers: list):
        self.ax.clear()
        self._style()
        p_lo = np.percentile(first_frame, 0.5)
        p_hi = np.percentile(first_frame, 99.5)
        disp = np.clip((first_frame.astype(float)-p_lo) /
                       (p_hi-p_lo+1e-10), 0, 1)
        h, w   = disp.shape
        factor = max(1, max(h, w) // 600)
        self.ax.imshow(disp[::factor, ::factor], cmap="gray",
                       origin="lower", aspect="equal", interpolation="nearest")
        sf = 1.0 / factor
        for i, (track, fit, mpc, ra, dec) in enumerate(movers):
            x0 = fit["x_start"] * sf
            y0 = fit["y_start"] * sf
            x1 = fit["x_end"]   * sf
            y1 = fit["y_end"]   * sf
            color = "#ffcc44" if mpc else "#ff6b22"
            label = mpc["designation"] if mpc else f"Unknown #{i+1}"
            self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
            self.ax.plot(x0, y0, "o", color=color, markersize=6,
                         markerfacecolor="none", markeredgewidth=1.5, zorder=5)
            self.ax.text(x0+3*sf, y0+3*sf, label, color=color, fontsize=7,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e2128",
                                   alpha=0.6, edgecolor=color, linewidth=0.5))
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_placeholder(self):
        self.ax.clear()
        self._style()
        self.ax.text(0.5, 0.5, "Run detection to see trail map",
                     ha="center", va="center", color=SIRIL_TEXT_DIM,
                     fontsize=11, transform=self.ax.transAxes)
        self.draw()


class BlinkCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig   = Figure(figsize=(4, 4), facecolor="#1e2128")
        self.ax    = self.fig.add_subplot(111)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)
        self.frames_data  = []
        self.frame_labels = []
        self.current_idx  = 0

    def _style(self):
        self.ax.set_facecolor("#1e2128")
        self.ax.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_color("#3a4055")

    def load_mover_cutouts(self, track: list, fits_files: list, cutout_size: int = 80):
        self.frames_data  = []
        self.frame_labels = []
        for src in track:
            frame_idx = src["frame_idx"]
            if frame_idx >= len(fits_files):
                continue
            try:
                with fits.open(fits_files[frame_idx]) as hdul:
                    data = hdul[0].data.astype(np.float32)
                if data.ndim == 3:
                    data = data[0] if data.shape[0] <= 4 else \
                           0.299*data[0]+0.587*data[1]+0.114*data[2]
                h, w = data.shape
                x, y = int(round(src["x"])), int(round(src["y"]))
                half = cutout_size // 2
                y1 = max(0, y-half); y2 = min(h, y+half)
                x1 = max(0, x-half); x2 = min(w, x+half)
                cutout = data[y1:y2, x1:x2]
                self.frames_data.append(cutout)
                self.frame_labels.append(
                    f"Frame {frame_idx+1}  ({src['x']:.1f}, {src['y']:.1f})")
            except Exception:
                pass
        self.current_idx = 0
        self._show_current()

    def advance(self):
        if not self.frames_data:
            return
        self.current_idx = (self.current_idx + 1) % len(self.frames_data)
        self._show_current()

    def _show_current(self):
        if not self.frames_data:
            return
        self.ax.clear()
        self._style()
        cutout = self.frames_data[self.current_idx]
        p_lo   = np.percentile(cutout, 1)
        p_hi   = np.percentile(cutout, 99.5)
        disp   = np.clip((cutout.astype(float)-p_lo)/(p_hi-p_lo+1e-10), 0, 1)
        self.ax.imshow(disp, cmap="gray", origin="lower",
                       aspect="equal", interpolation="nearest")
        self.ax.plot(disp.shape[1]//2, disp.shape[0]//2,
                     "o", color="#ff6b22", markersize=20,
                     markerfacecolor="none", markeredgewidth=1.5, zorder=5)
        label = self.frame_labels[self.current_idx] \
                if self.current_idx < len(self.frame_labels) else ""
        self.ax.set_title(label, color="#ffaa22", fontsize=8)
        self.fig.tight_layout(pad=0.2)
        self.draw()

    def show_placeholder(self):
        self.ax.clear()
        self._style()
        self.ax.text(0.5, 0.5, "Double-click a row in Detections\nto load object",
                     ha="center", va="center", color=SIRIL_TEXT_DIM,
                     fontsize=11, transform=self.ax.transAxes)
        self.draw()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11-12 — MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("☄  Moving Object Detector — Siril")
        self.setMinimumSize(1100, 680)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._movers       = []         # list of (track,fit,mpc,ra,dec)
        self._fits_files   = []
        self._first_frame  = None
        self._has_wcs      = False

        # Blink timer
        self._blink_timer    = QTimer()
        self._blink_interval = 500
        self._blink_playing  = False
        self._blink_timer.timeout.connect(self._blink_advance)

        self._build_ui()
        self.trail_canvas.show_placeholder()
        self.blink_canvas.show_placeholder()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setStyleSheet(f"background:{SIRIL_BG3}; border-bottom:1px solid {SIRIL_BORDER};")
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(12, 6, 12, 6)
        lbl_title = QLabel("☄  Moving Object Detector  —  Siril")
        lbl_title.setStyleSheet(f"color:{SIRIL_ASTEROID}; font-size:13pt; font-weight:bold;")
        lbl_ver = QLabel("v1.0  |  Asteroid / Comet detection")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        tb_lay.addWidget(lbl_title)
        tb_lay.addStretch()
        tb_lay.addWidget(lbl_ver)
        root.addWidget(title_bar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        root.addWidget(splitter, 1)

        # LEFT PANEL
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(360)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(6)
        self._build_left_panel(left_lay)
        splitter.addWidget(left)

        # RIGHT PANEL
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(6)
        self._build_right_panel(right_lay)
        splitter.addWidget(right)
        splitter.setSizes([320, 780])

        # Bottom bar
        bot = QWidget()
        bot.setStyleSheet(f"background:{SIRIL_BG3}; border-top:1px solid {SIRIL_BORDER};")
        bot_lay = QHBoxLayout(bot)
        bot_lay.setContentsMargins(10, 6, 10, 6)
        bot_lay.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)
        bot_lay.addWidget(self.progress_bar, 1)

        self.btn_run = QPushButton("▶  Detect movers")
        self.btn_run.setObjectName("primary")
        self.btn_run.setFixedWidth(160)
        self.btn_run.clicked.connect(self._start_detection)
        bot_lay.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setFixedWidth(100)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_detection)
        bot_lay.addWidget(self.btn_cancel)

        root.addWidget(bot)

        # Status bar
        self.status_label = QLabel("Ready — load a FITS sequence folder and scan.")
        self.status_label.setStyleSheet(
            f"background:{SIRIL_BG2}; color:{SIRIL_TEXT_DIM}; "
            f"padding:3px 10px; font-size:9pt;")
        self.status_label.setFixedHeight(22)
        root.addWidget(self.status_label)

    def _build_left_panel(self, lay):
        # ── Input sequence ──────────────────────────────────────────────────
        grp_input = QGroupBox("Input sequence")
        g_lay = QVBoxLayout(grp_input)

        row_folder = QHBoxLayout()
        self.edit_folder = QLineEdit()
        self.edit_folder.setPlaceholderText("FITS sequence folder…")
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(60)
        btn_browse.clicked.connect(self._browse_folder)
        row_folder.addWidget(self.edit_folder)
        row_folder.addWidget(btn_browse)
        g_lay.addLayout(row_folder)

        lbl_hint = QLabel("Frames must be time-sorted with DATE-OBS headers")
        lbl_hint.setObjectName("dim")
        g_lay.addWidget(lbl_hint)

        btn_scan = QPushButton("🔍  Scan")
        btn_scan.clicked.connect(self._scan_folder)
        g_lay.addWidget(btn_scan)

        self.lbl_scan_result = QLabel("No folder selected.")
        self.lbl_scan_result.setObjectName("dim")
        self.lbl_scan_result.setWordWrap(True)
        g_lay.addWidget(self.lbl_scan_result)

        self.lbl_no_ts = QLabel("⚠ No timestamps — frame order used as time proxy")
        self.lbl_no_ts.setObjectName("warn")
        self.lbl_no_ts.setWordWrap(True)
        self.lbl_no_ts.setVisible(False)
        g_lay.addWidget(self.lbl_no_ts)

        lay.addWidget(grp_input)

        # ── Detection settings ──────────────────────────────────────────────
        grp_det = QGroupBox("Detection settings")
        f_lay = QFormLayout(grp_det)
        f_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(3.0, 15.0); self.spin_threshold.setValue(5.0)
        self.spin_threshold.setSingleStep(0.5); self.spin_threshold.setSuffix(" σ")
        f_lay.addRow("Detection threshold:", self.spin_threshold)

        self.spin_match_radius = QDoubleSpinBox()
        self.spin_match_radius.setRange(3.0, 30.0); self.spin_match_radius.setValue(8.0)
        self.spin_match_radius.setSuffix(" px")
        f_lay.addRow("Match radius:", self.spin_match_radius)

        self.spin_min_frames = QSpinBox()
        self.spin_min_frames.setRange(2, 99); self.spin_min_frames.setValue(3)
        f_lay.addRow("Min frames:", self.spin_min_frames)

        self.spin_min_motion = QDoubleSpinBox()
        self.spin_min_motion.setRange(1.0, 50.0); self.spin_min_motion.setValue(3.0)
        self.spin_min_motion.setSuffix(" px")
        f_lay.addRow("Min motion:", self.spin_min_motion)

        self.spin_max_speed = QDoubleSpinBox()
        self.spin_max_speed.setRange(5.0, 500.0); self.spin_max_speed.setValue(200.0)
        self.spin_max_speed.setSuffix(" px/f")
        f_lay.addRow("Max speed:", self.spin_max_speed)

        self.spin_r2 = QDoubleSpinBox()
        self.spin_r2.setRange(0.70, 1.00); self.spin_r2.setValue(0.90)
        self.spin_r2.setSingleStep(0.01)
        self.spin_r2.setDecimals(2)
        f_lay.addRow("R² threshold:", self.spin_r2)

        lay.addWidget(grp_det)

        # ── MPC cross-match ─────────────────────────────────────────────────
        grp_mpc = QGroupBox("MPC cross-match")
        m_lay = QVBoxLayout(grp_mpc)

        self.chk_mpc = QCheckBox("Cross-match MPC catalog")
        self.chk_mpc.setChecked(True)
        if not HAS_ASTROQUERY:
            self.chk_mpc.setChecked(False)
            self.chk_mpc.setEnabled(False)
        m_lay.addWidget(self.chk_mpc)

        if not HAS_ASTROQUERY:
            lbl_aq = QLabel("⚠ astroquery not available")
            lbl_aq.setObjectName("warn")
            m_lay.addWidget(lbl_aq)

        mf_lay = QFormLayout()
        self.spin_mpc_radius = QDoubleSpinBox()
        self.spin_mpc_radius.setRange(5.0, 120.0); self.spin_mpc_radius.setValue(30.0)
        self.spin_mpc_radius.setSuffix("\"")
        mf_lay.addRow("Search radius:", self.spin_mpc_radius)

        self.edit_obs_code = QLineEdit("500")
        self.edit_obs_code.setMaxLength(3)
        self.edit_obs_code.setFixedWidth(60)
        mf_lay.addRow("Observatory code:", self.edit_obs_code)

        lbl_obs_hint = QLabel("500 = geocenter if unknown")
        lbl_obs_hint.setObjectName("dim")
        mf_lay.addRow("", lbl_obs_hint)
        m_lay.addLayout(mf_lay)
        lay.addWidget(grp_mpc)

        # ── Output ──────────────────────────────────────────────────────────
        grp_out = QGroupBox("Output")
        o_lay = QVBoxLayout(grp_out)

        row_out = QHBoxLayout()
        self.edit_outdir = QLineEdit()
        self.edit_outdir.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse")
        btn_out.setFixedWidth(60)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self.edit_outdir)
        row_out.addWidget(btn_out)
        o_lay.addLayout(row_out)

        self.chk_export_mpc = QCheckBox("Export MPC report")
        self.chk_export_mpc.setChecked(True)
        o_lay.addWidget(self.chk_export_mpc)

        self.chk_export_csv = QCheckBox("Export CSV")
        self.chk_export_csv.setChecked(True)
        o_lay.addWidget(self.chk_export_csv)

        lay.addWidget(grp_out)
        lay.addStretch()

    def _build_right_panel(self, lay):
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        # ── Tab 0: Trail Map ─────────────────────────────────────────────────
        tab_trail = QWidget()
        tl = QVBoxLayout(tab_trail)
        tl.setContentsMargins(4, 4, 4, 4)
        self.trail_canvas = TrailCanvas()
        tl.addWidget(self.trail_canvas, 1)
        self.tabs.addTab(tab_trail, "☄  Trail Map")

        # ── Tab 1: Detections ────────────────────────────────────────────────
        tab_det = QWidget()
        dl = QVBoxLayout(tab_det)
        dl.setContentsMargins(6, 6, 6, 6)

        # Badge row
        badge_row = QHBoxLayout()
        self.lbl_total   = QLabel("Total: 0")
        self.lbl_known   = QLabel("Known: 0")
        self.lbl_unknown = QLabel("Unknown: 0")
        self.lbl_known.setStyleSheet(f"color:{SIRIL_ASTEROID}; font-weight:bold;")
        self.lbl_unknown.setStyleSheet(f"color:#ff6b22; font-weight:bold;")
        badge_row.addWidget(self.lbl_total)
        badge_row.addWidget(self.lbl_known)
        badge_row.addWidget(self.lbl_unknown)
        badge_row.addStretch()
        dl.addLayout(badge_row)

        # Table
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "#", "Status", "Designation", "X_start", "Y_start",
            "Motion(px)", "Speed(px/f)", "R²", "RA", "Dec"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_table_double_click)
        dl.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        btn_csv = QPushButton("💾  Export CSV")
        btn_csv.setObjectName("export")
        btn_csv.clicked.connect(self._manual_export_csv)
        btn_mpc = QPushButton("📡  Export MPC report")
        btn_mpc.setObjectName("asteroid")
        btn_mpc.clicked.connect(self._manual_export_mpc)
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_mpc)
        btn_row.addStretch()
        dl.addLayout(btn_row)

        self.tabs.addTab(tab_det, "📋  Detections")

        # ── Tab 2: Blink ─────────────────────────────────────────────────────
        tab_blink = QWidget()
        bl = QVBoxLayout(tab_blink)
        bl.setContentsMargins(4, 4, 4, 4)
        self.blink_canvas = BlinkCanvas()
        bl.addWidget(self.blink_canvas, 1)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Speed:"))
        self.slider_blink = QSlider(Qt.Orientation.Horizontal)
        self.slider_blink.setRange(100, 3000)
        self.slider_blink.setValue(500)
        self.slider_blink.valueChanged.connect(self._set_blink_speed)
        ctrl_row.addWidget(self.slider_blink, 1)
        self.lbl_blink_speed = QLabel("0.5s/frame")
        ctrl_row.addWidget(self.lbl_blink_speed)
        self.btn_blink = QPushButton("▶  Play")
        self.btn_blink.setFixedWidth(90)
        self.btn_blink.clicked.connect(self._toggle_blink)
        ctrl_row.addWidget(self.btn_blink)
        bl.addLayout(ctrl_row)

        lbl_blink_hint = QLabel("Double-click a row in Detections to load object")
        lbl_blink_hint.setObjectName("dim")
        lbl_blink_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(lbl_blink_hint)

        self.tabs.addTab(tab_blink, "🎬  Blink")

        # ── Tab 3: Log ───────────────────────────────────────────────────────
        tab_log = QWidget()
        ll = QVBoxLayout(tab_log)
        ll.setContentsMargins(4, 4, 4, 4)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setProperty("readOnly", True)
        ll.addWidget(self.log_edit)
        self.tabs.addTab(tab_log, "📋  Log")

    # ── Folder browsing ────────────────────────────────────────────────────────
    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self.edit_folder.setText(d)
            if not self.edit_outdir.text():
                self.edit_outdir.setText(d)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.edit_outdir.setText(d)

    def _scan_folder(self):
        folder = self.edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            self.lbl_scan_result.setText("Invalid folder.")
            return

        patterns = ["*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        files = sorted(set(files))

        if not files:
            self.lbl_scan_result.setText("No FITS files found.")
            return

        # Try to read timestamps
        has_ts = False
        jd_list = []
        for path in files[:min(5, len(files))]:
            try:
                with fits.open(path) as hdul:
                    h  = hdul[0].header
                    jd = get_jd_from_header(h)
                    if jd > 0:
                        jd_list.append(jd)
                        has_ts = True
            except Exception:
                pass

        self._fits_files = files
        if not self.edit_outdir.text():
            self.edit_outdir.setText(folder)

        if has_ts and len(jd_list) >= 2:
            span_min = (max(jd_list) - min(jd_list)) * 1440
            self.lbl_scan_result.setText(
                f"{len(files)} FITS files found\n"
                f"Time span ≈ {span_min:.1f} min (sample of first 5)")
        else:
            self.lbl_scan_result.setText(f"{len(files)} FITS files found")

        self.lbl_no_ts.setVisible(not has_ts)

        # Update min_frames spinbox max
        self.spin_min_frames.setMaximum(len(files))
        self.spin_min_frames.setValue(max(3, len(files) - 1))
        self._set_status(f"Scanned: {len(files)} frames in {folder}")

    # ── Detection ──────────────────────────────────────────────────────────────
    def _start_detection(self):
        if not self._fits_files:
            QMessageBox.warning(self, "No files", "Scan a folder first.")
            return
        if len(self._fits_files) < 3:
            QMessageBox.warning(self, "Too few frames",
                                "Need at least 3 frames for motion detection.")
            return

        # Clear previous results
        self.table.setRowCount(0)
        self._movers = []
        self._update_badges()
        self.log_edit.clear()
        self.trail_canvas.show_placeholder()
        self.blink_canvas.show_placeholder()

        cfg = {
            "fits_files":       self._fits_files,
            "threshold_sigma":  self.spin_threshold.value(),
            "fwhm_guess":       5.0,
            "max_sources":      500,
            "match_radius_px":  self.spin_match_radius.value(),
            "min_frames":       self.spin_min_frames.value(),
            "min_motion_px":    self.spin_min_motion.value(),
            "max_motion_px":    self.spin_max_speed.value(),
            "min_r2":           self.spin_r2.value(),
            "crossmatch_mpc":   self.chk_mpc.isChecked(),
            "mpc_search_arcsec":self.spin_mpc_radius.value(),
            "observatory_code": self.edit_obs_code.text().strip() or "500",
            "export_mpc":       self.chk_export_mpc.isChecked(),
            "export_csv":       self.chk_export_csv.isChecked(),
            "output_dir":       self.edit_outdir.text().strip() or ".",
            "blink_cutout_size":80,
        }

        self._cancel_event = threading.Event()
        self._worker = MoverWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log)
        self._worker.mover_found.connect(self._on_mover_found)
        self._worker.finished.connect(self._on_finished)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._set_status("Running detection…")

        self._worker.start()

    def _cancel_detection(self):
        if self._worker:
            self._worker.cancel()
        self.btn_cancel.setEnabled(False)
        self._set_status("Cancelling…")

    # ── Worker signals ─────────────────────────────────────────────────────────
    def _on_progress(self, val, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(val)
        self._set_status(msg)

    def _on_log(self, line):
        self.log_edit.appendPlainText(line)

    def _on_mover_found(self, data):
        track   = data["track"]
        fit     = data["fit"]
        mpc     = data["mpc_match"]
        ra      = data["ra"]
        dec     = data["dec"]
        status  = data["status"]

        self._movers.append((track, fit, mpc, ra, dec))
        self._update_badges()

        row = self.table.rowCount()
        self.table.insertRow(row)
        vals = [
            str(row + 1),
            "Known ★" if mpc else "Unknown ?",
            mpc["designation"] if mpc else "",
            f"{fit['x_start']:.1f}",
            f"{fit['y_start']:.1f}",
            f"{fit['total_motion_px']:.1f}",
            f"{fit['speed_px_per_frame']:.3f}",
            f"{fit['r2_mean']:.3f}",
            f"{ra:.4f}" if ra is not None else "—",
            f"{dec:.4f}" if dec is not None else "—",
        ]
        color = QColor("#332800") if mpc else QColor("#2a1500")
        for col, val in enumerate(vals):
            item = QTableWidgetItem(val)
            item.setBackground(color)
            self.table.setItem(row, col, item)

    def _on_finished(self, result):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        if not result["success"]:
            self._set_status(f"✗  Error: {result.get('error', 'Unknown')}")
            self.log_edit.appendPlainText(f"[FAILED] {result.get('error','')}")
            return

        n      = result["n_movers"]
        n_k    = result["n_known"]
        n_u    = result["n_unknown"]
        outdir = result.get("csv_path", "")

        self._set_status(
            f"✓  {n} movers found  |  Known: {n_k}  |  "
            f"Unknown: {n_u}  |  Output: {outdir}")

        # Draw trail map with first frame
        if self._fits_files and n > 0:
            try:
                with fits.open(self._fits_files[0]) as hdul:
                    frame = hdul[0].data.astype(np.float32)
                if frame.ndim == 3:
                    frame = frame[0] if frame.shape[0] <= 4 else \
                            0.299*frame[0]+0.587*frame[1]+0.114*frame[2]
                self.trail_canvas.show_trails(frame, self._movers)
                tab_title = f"☄  Trail Map  [{n}]"
                self.tabs.setTabText(0, tab_title)
            except Exception as e:
                self.log_edit.appendPlainText(f"Trail map error: {e}")

        self.tabs.setTabText(1, f"📋  Detections  [{n}]")
        self.log_edit.appendPlainText(
            f"\n[DONE] {n} movers  |  Known={n_k}  Unknown={n_u}")

        # Show no-WCS hint
        if not result.get("pixel_scale"):
            self.log_edit.appendPlainText(
                "\n⚠ No WCS found — MPC cross-match unavailable.\n"
                "  Tip: Run Siril plate solve (platesolve command) first\n"
                "  to add WCS to your frames for MPC identification.")

    # ── Blink controls ─────────────────────────────────────────────────────────
    def _toggle_blink(self):
        if self._blink_playing:
            self._blink_timer.stop()
            self._blink_playing = False
            self.btn_blink.setText("▶  Play")
        else:
            self._blink_timer.start(self._blink_interval)
            self._blink_playing = True
            self.btn_blink.setText("⏸  Pause")

    def _set_blink_speed(self, ms: int):
        self._blink_interval = ms
        self.lbl_blink_speed.setText(f"{ms/1000:.1f}s/frame")
        if self._blink_playing:
            self._blink_timer.setInterval(ms)

    def _blink_advance(self):
        self.blink_canvas.advance()

    def _on_table_double_click(self, index):
        row = index.row()
        if row >= len(self._movers):
            return
        track, fit, mpc, ra, dec = self._movers[row]
        self.blink_canvas.load_mover_cutouts(
            track, self._fits_files, cutout_size=80)
        self.tabs.setCurrentIndex(2)
        if not self._blink_playing:
            self._toggle_blink()

    # ── Manual exports ─────────────────────────────────────────────────────────
    def _manual_export_csv(self):
        if not self._movers:
            QMessageBox.information(self, "No data", "No movers detected yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "moving_objects.csv", "CSV (*.csv)")
        if path:
            try:
                from moving_object import MoverWorker
            except Exception:
                pass
            worker = MoverWorker.__new__(MoverWorker)
            worker._export_csv(self._movers, path, None)
            self._set_status(f"CSV saved: {path}")

    def _manual_export_mpc(self):
        if not self._movers:
            QMessageBox.information(self, "No data", "No movers detected yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save MPC report", "mpc_report.txt", "Text (*.txt)")
        if path:
            obs = self.edit_obs_code.text().strip() or "500"
            # Use first FITS header for WCS context
            first_hdr = None
            if self._fits_files:
                try:
                    with fits.open(self._fits_files[0]) as hdul:
                        first_hdr = hdul[0].header.copy()
                except Exception:
                    pass
            export_mpc_report(self._movers, 1.0, obs, path, first_hdr,
                              log_callback=self._on_log)
            self._set_status(f"MPC report saved: {path}")

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _update_badges(self):
        n_k = sum(1 for _, _, m, _, _ in self._movers if m is not None)
        n_u = sum(1 for _, _, m, _, _ in self._movers if m is None)
        n   = len(self._movers)
        self.lbl_total.setText(f"Total: {n}")
        self.lbl_known.setText(f"Known: {n_k}")
        self.lbl_unknown.setText(f"Unknown: {n_u}")

    def _set_status(self, msg: str):
        self.status_label.setText(msg)


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
