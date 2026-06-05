r"""
planet_derotation.py  —  Siril Python Tools  #13
=================================================
Correct for planetary rotation during video capture.
Free, cross-platform alternative to WinJUPOS.

Place in: C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:  Siril → Scripts menu → planet_derotation
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("astroquery")

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB / THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import struct
import glob
import json
import threading
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import (
    rotate as scipy_rotate,
    shift as ndimage_shift,
    gaussian_filter,
    center_of_mass,
)
from scipy.optimize import minimize
from astropy.io import fits as astropy_fits
from astropy.time import Time
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import sirilpy as siril_api

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QDateTimeEdit, QSizePolicy, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt6.QtGui import QFont

# ─────────────────────────────────────────────────────────────────────────────
# EQUIPMENT MANAGER (OPTIONAL)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ─────────────────────────────────────────────────────────────────────────────
# ASTROQUERY (OPTIONAL)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from astroquery.jplhorizons import Horizons
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False

# ─────────────────────────────────────────────────────────────────────────────
# THEME
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
QLabel#planet   {{
    color: {SIRIL_ACCENT};
    font-size: 24pt;
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
QDateTimeEdit {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    padding: 4px 6px;
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# PLANET DATA
# ─────────────────────────────────────────────────────────────────────────────
PLANETS = {
    "Jupiter": {
        "id":   "599",
        "emoji": "🪐",
        "systems": {
            "System I":   877.900,
            "System II":  870.270,
            "System III": 870.536,
        },
        "default_system": "System II",
        "diameter_km":    142984,
        "safe_session_min": 2,
    },
    "Saturn": {
        "id":   "699",
        "emoji": "🪐",
        "systems": {
            "System I":   844.300,
            "System III": 810.793,
        },
        "default_system": "System III",
        "diameter_km":    120536,
        "safe_session_min": 4,
    },
    "Mars": {
        "id":   "499",
        "emoji": "🔴",
        "systems": {
            "Mars": 350.892,
        },
        "default_system": "Mars",
        "diameter_km":    6779,
        "safe_session_min": 10,
    },
    "Venus": {
        "id":   "299",
        "emoji": "⚪",
        "systems": {
            "Venus": -6.520,
        },
        "default_system": "Venus",
        "diameter_km":    12104,
        "safe_session_min": 120,
    },
    "Uranus": {
        "id":   "799",
        "emoji": "🔵",
        "systems": {
            "Uranus": -501.160,
        },
        "default_system": "Uranus",
        "diameter_km":    51118,
        "safe_session_min": 3,
    },
    "Neptune": {
        "id":   "899",
        "emoji": "🔵",
        "systems": {
            "Neptune": 536.314,
        },
        "default_system": "Neptune",
        "diameter_km":    49528,
        "safe_session_min": 3,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SER_HEADER_SIZE          = 178
WINDOWS_EPOCH_OFFSET     = 116444736000000000   # 100-ns ticks from 1601→1970

# ─────────────────────────────────────────────────────────────────────────────
# TIMESTAMP UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def read_ser_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(SER_HEADER_SIZE)
    return {
        "raw_bytes":   raw,
        "color_id":    struct.unpack_from("<I", raw, 18)[0],
        "width":       struct.unpack_from("<I", raw, 26)[0],
        "height":      struct.unpack_from("<I", raw, 30)[0],
        "pixel_depth": struct.unpack_from("<I", raw, 34)[0],
        "frame_count": struct.unpack_from("<I", raw, 38)[0],
    }


def read_ser_frame(f, header: dict, frame_index: int) -> np.ndarray:
    bpp        = 1 if header["pixel_depth"] <= 8 else 2
    frame_size = header["width"] * header["height"] * bpp
    f.seek(SER_HEADER_SIZE + frame_index * frame_size)
    raw   = f.read(frame_size)
    dtype = np.uint8 if bpp == 1 else np.uint16
    return np.frombuffer(raw, dtype=dtype).reshape(
        header["height"], header["width"])


def write_ser_file(output_path: str, frames: list, header: dict):
    new_header = bytearray(header["raw_bytes"])
    struct.pack_into("<I", new_header, 38, len(frames))
    bpp = 1 if header["pixel_depth"] <= 8 else 2
    with open(output_path, "wb") as f:
        f.write(bytes(new_header))
        for frame in frames:
            dtype = np.uint8 if bpp == 1 else np.uint16
            f.write(frame.astype(dtype).tobytes())


def read_ser_timestamps(ser_path: str) -> list:
    file_size = os.path.getsize(ser_path)
    with open(ser_path, "rb") as f:
        header_bytes = f.read(SER_HEADER_SIZE)
        width        = struct.unpack_from("<I", header_bytes, 26)[0]
        height       = struct.unpack_from("<I", header_bytes, 30)[0]
        pixel_depth  = struct.unpack_from("<I", header_bytes, 34)[0]
        frame_count  = struct.unpack_from("<I", header_bytes, 38)[0]
        bpp          = 1 if pixel_depth <= 8 else 2
        frame_size   = width * height * bpp
        data_size    = SER_HEADER_SIZE + frame_count * frame_size
        trailer_size = frame_count * 8
        if file_size < data_size + trailer_size:
            return []
        f.seek(data_size)
        timestamps = []
        for _ in range(frame_count):
            raw = f.read(8)
            if len(raw) < 8:
                break
            filetime = struct.unpack("<q", raw)[0]
            unix_ts  = (filetime - WINDOWS_EPOCH_OFFSET) / 10_000_000.0
            timestamps.append(unix_ts)
    return timestamps


def read_fits_timestamps(fits_files: list) -> list:
    timestamps = []
    for path in fits_files:
        try:
            hdr  = astropy_fits.getheader(path)
            date = hdr.get("DATE-OBS") or hdr.get("DATE") or hdr.get("TIME-OBS")
            if date:
                t = Time(date, format="isot", scale="utc")
                timestamps.append(t.unix)
            else:
                timestamps.append(None)
        except Exception:
            timestamps.append(None)
    return timestamps


def timestamps_from_manual(start_unix: float,
                            frame_rate_fps: float,
                            frame_count: int) -> list:
    dt = 1.0 / frame_rate_fps if frame_rate_fps > 0 else 1.0
    return [start_unix + i * dt for i in range(frame_count)]


def session_duration_from_timestamps(timestamps: list) -> float:
    if len(timestamps) < 2:
        return 0.0
    valid = [t for t in timestamps if t is not None]
    if len(valid) < 2:
        return 0.0
    return valid[-1] - valid[0]

# ─────────────────────────────────────────────────────────────────────────────
# ROTATION ANGLE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_rotation_angles_offline(timestamps: list,
                                     rotation_rate_deg_per_day: float) -> list:
    if not timestamps:
        return []
    t0 = timestamps[0]
    return [(t - t0) / 86400.0 * rotation_rate_deg_per_day for t in timestamps]


def compute_rotation_angles_horizons(timestamps: list,
                                      planet_id: str,
                                      observer_location: str = "500") -> list:
    if not HAS_ASTROQUERY:
        raise RuntimeError("astroquery not available")
    if not timestamps:
        return []
    times      = [Time(t, format="unix", scale="utc") for t in timestamps]
    BATCH_SIZE = 200
    longitudes = []

    for i in range(0, len(times), BATCH_SIZE):
        batch    = times[i:i + BATCH_SIZE]
        duration = max(1, int(batch[-1].unix - batch[0].unix))
        epochs   = {
            "start": batch[0].isot,
            "stop":  batch[-1].isot,
            "step":  f"{duration}s",
        }
        try:
            obj = Horizons(id=planet_id, epochs=epochs,
                           location=observer_location)
            eph = obj.ephemerides(quantities="14")
            for row in eph:
                longitudes.append(float(row["PDObsLon"]))
        except Exception as e:
            raise RuntimeError(f"JPL Horizons query failed: {e}")

    if len(longitudes) < len(timestamps):
        raise RuntimeError("Horizons returned fewer values than expected")

    lon0   = longitudes[0]
    angles = []
    for lon in longitudes:
        diff = lon - lon0
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        angles.append(diff)
    return angles

# ─────────────────────────────────────────────────────────────────────────────
# PLANET CENTER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_planet_center(frame: np.ndarray, method: str = "centroid") -> tuple:
    h, w  = frame.shape
    mn    = frame.min()
    mx    = frame.max()
    norm  = (frame.astype(float) - mn) / (mx - mn + 1e-10)

    if method == "centroid":
        binary = (norm > 0.5).astype(float)
        if binary.sum() < 10:
            binary = norm
        cy, cx = center_of_mass(binary)
        return (float(cy), float(cx))

    elif method == "circle_fit":
        return _fit_circle_to_limb(norm)

    return (h / 2.0, w / 2.0)


def _fit_circle_to_limb(norm: np.ndarray) -> tuple:
    from scipy.ndimage import sobel
    h, w  = norm.shape
    gx    = sobel(norm, axis=1)
    gy    = sobel(norm, axis=0)
    edges = np.sqrt(gx**2 + gy**2)
    thr   = np.percentile(edges, 90)
    ys, xs = np.where(edges > thr)
    if len(xs) < 10:
        return (h / 2.0, w / 2.0)

    def circle_error(params):
        cy, cx, r = params
        dists = np.sqrt((ys - cy)**2 + (xs - cx)**2)
        return np.sum((dists - r)**2)

    result = minimize(circle_error, [h/2, w/2, min(h, w)/3],
                      method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 0.5})
    cy, cx, _ = result.x
    return (float(cy), float(cx))

# ─────────────────────────────────────────────────────────────────────────────
# FRAME DEROTATION
# ─────────────────────────────────────────────────────────────────────────────

def background_fill_value(frame: np.ndarray) -> float:
    corners = np.concatenate([
        frame[:20, :20].ravel(),
        frame[:20, -20:].ravel(),
        frame[-20:, :20].ravel(),
        frame[-20:, -20:].ravel(),
    ])
    return float(np.median(corners))


def derotate_frame(frame: np.ndarray,
                    angle_deg: float,
                    center: tuple,
                    fill_value: float = 0.0) -> np.ndarray:
    h, w         = frame.shape
    cy, cx       = center
    frame_cy     = h / 2.0
    frame_cx     = w / 2.0
    dy           = cy - frame_cy
    dx           = cx - frame_cx

    if abs(dy) < 5 and abs(dx) < 5:
        return scipy_rotate(frame.astype(float), angle=-angle_deg,
                            reshape=False, mode="constant",
                            cval=fill_value).astype(frame.dtype)

    shifted   = ndimage_shift(frame.astype(float), (-dy, -dx),
                               mode="constant", cval=fill_value)
    rotated   = scipy_rotate(shifted, angle=-angle_deg,
                              reshape=False, mode="constant", cval=fill_value)
    unshifted = ndimage_shift(rotated, (dy, dx),
                               mode="constant", cval=fill_value)
    return unshifted.astype(frame.dtype)


def derotate_sequence(frames: list, angles: list, centers,
                       fill_value: float = 0.0,
                       log_callback=None,
                       cancel_event=None) -> list:
    n = len(frames)
    if isinstance(centers, tuple):
        centers = [centers] * n
    derotated = []
    for i, (frame, angle, center) in enumerate(zip(frames, angles, centers)):
        if cancel_event and cancel_event.is_set():
            break
        result = derotate_frame(frame, angle, center, fill_value)
        derotated.append(result)
        if log_callback and i % 10 == 0:
            log_callback(f"  Derotated frame {i+1}/{n}  (angle: {angle:.3f}°)")
    return derotated


def estimate_blur_without_derotation(rotation_rate_deg_per_day: float,
                                      session_duration_sec: float,
                                      planet_diameter_px: float) -> float:
    total_deg = abs(rotation_rate_deg_per_day * session_duration_sec / 86400.0)
    circ_px   = planet_diameter_px * 3.14159
    return circ_px * (total_deg / 360.0)

# ─────────────────────────────────────────────────────────────────────────────
# ATMOSPHERIC PSF SIMULATOR  (Kolmogorov turbulence + annular aperture)
# ─────────────────────────────────────────────────────────────────────────────

def generate_kolmogorov_phase(size: int, pixel_scale: float, r0: float) -> np.ndarray:
    """
    Random phase screen with Kolmogorov PSD (-11/3 power law).
    size        : grid side in pixels
    pixel_scale : physical pixel size at pupil plane (metres)
    r0          : Fried parameter (metres)  — smaller = worse seeing
    Returns phase in radians.
    """
    import scipy.fft as spfft
    fx = spfft.fftfreq(size, d=pixel_scale)
    FX, FY = np.meshgrid(fx, fx)
    kappa = np.sqrt(FX**2 + FY**2)
    kappa[0, 0] = 1e-10                          # avoid DC divide-by-zero
    psd = 0.023 * (r0 ** (-5/3)) * (kappa ** (-11/3))
    noise = (np.random.normal(size=(size, size)) +
             1j * np.random.normal(size=(size, size)))
    phase_screen = np.real(spfft.ifft2(noise * np.sqrt(psd))) * size
    return phase_screen


def simulate_planetary_psf(aperture_diameter: float = 0.28,
                            central_obstruction: float = 0.34,
                            r0: float = 0.08,
                            grid_size: int = 512) -> tuple:
    """
    Simulate an atmospherically blurred PSF for a telescope with central obstruction.

    aperture_diameter   : metres  (0.28 = Celestron C11)
    central_obstruction : linear ratio of secondary  (0.34 = 34%)
    r0                  : Fried seeing parameter, metres
    grid_size           : FFT grid side (power of 2 recommended)

    Returns (psf, aperture_mask, phase_screen)
    psf is normalised so sum = 1.
    """
    import scipy.fft as spfft
    pupil_extent = aperture_diameter * 1.5
    dx = pupil_extent / grid_size
    x  = np.linspace(-pupil_extent / 2, pupil_extent / 2, grid_size)
    X, Y   = np.meshgrid(x, x)
    radius = np.sqrt(X**2 + Y**2)

    outer = radius <= (aperture_diameter / 2)
    inner = radius >  (aperture_diameter * central_obstruction / 2)
    aperture_mask = (outer & inner).astype(float)

    phase_error = generate_kolmogorov_phase(grid_size, dx, r0)
    pupil_field = aperture_mask * np.exp(1j * phase_error)

    image_amplitude = spfft.fftshift(spfft.fft2(pupil_field))
    psf = np.abs(image_amplitude) ** 2
    psf /= psf.sum()
    return psf, aperture_mask, phase_error


def crop_psf(psf: np.ndarray, target_size: int) -> np.ndarray:
    """
    Crop a large PSF array to target_size × target_size centred on the peak.
    Used to match PSF kernel size to the image being deconvolved.
    """
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    h2 = target_size // 2
    r0 = max(0, cy - h2);  r1 = r0 + target_size
    c0 = max(0, cx - h2);  c1 = c0 + target_size
    # clamp to array bounds
    r1 = min(r1, psf.shape[0]);  r0 = r1 - target_size
    c1 = min(c1, psf.shape[1]);  c0 = c1 - target_size
    out = psf[r0:r1, c0:c1].copy()
    s   = out.sum()
    return out / s if s > 0 else out


# ─────────────────────────────────────────────────────────────────────────────
# DECONVOLUTION ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────
# All functions accept:
#   image : 2-D float ndarray  (normalised 0-1 recommended)
#   psf   : 2-D float ndarray  (must sum to 1)
# and return a 2-D float ndarray of the same shape as image.
# ─────────────────────────────────────────────────────────────────────────────

# ── helpers ──────────────────────────────────────────────────────────────────

def _fft2(x):  return np.fft.rfft2(x)
def _ifft2(X, s): return np.fft.irfft2(X, s=s)

def _psf_otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """Zero-padded OTF (rfft2) of PSF centred at corner."""
    pad = np.zeros(shape, dtype=np.float64)
    h, w = psf.shape
    pad[:h, :w] = psf
    # roll so that PSF centre is at corner (circular convolution)
    pad = np.roll(pad, -h // 2, axis=0)
    pad = np.roll(pad, -w // 2, axis=1)
    return np.fft.rfft2(pad)

def _tv_grad(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Isotropic Total Variation gradient (divergence of normalised gradient)."""
    ux = np.diff(u, axis=1, append=u[:, :1])
    uy = np.diff(u, axis=0, append=u[:1, :])
    norm = np.sqrt(ux**2 + uy**2 + eps**2)
    px   = ux / norm
    py   = uy / norm
    # divergence
    dpx  = px - np.roll(px, 1, axis=1)
    dpy  = py - np.roll(py, 1, axis=0)
    return dpx + dpy

def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft thresholding (shrinkage)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


# ── 1. Wiener deconvolution ──────────────────────────────────────────────────

def deconv_wiener(image: np.ndarray, psf: np.ndarray,
                  snr: float = 30.0) -> np.ndarray:
    """
    Classical Wiener deconvolution in the frequency domain.

    snr : signal-to-noise ratio estimate.  Higher = less regularisation.
         Typical range 10–1000.  Start at 30 for well-exposed planetary.

    Reference: Wiener (1949), "Extrapolation, Interpolation, and
    Smoothing of Stationary Time Series."
    """
    img  = image.astype(np.float64)
    H    = _psf_otf(psf, img.shape)
    G    = _fft2(img)
    K    = 1.0 / (snr ** 2)           # noise-to-signal power ratio
    W    = np.conj(H) / (np.abs(H)**2 + K)
    return np.clip(_ifft2(W * G, img.shape), 0, None)


# ── 2. Richardson–Lucy with Total Variation regularisation ───────────────────

def deconv_rl_tv(image: np.ndarray, psf: np.ndarray,
                 iterations: int = 30,
                 tv_lambda: float = 5e-4,
                 eps: float = 1e-7,
                 callback=None) -> np.ndarray:
    """
    Richardson–Lucy EM algorithm with isotropic TV regularisation.

    Maximises the Poisson log-likelihood with a TV penalty:
        f* = argmax  Σ [d·log(H*f) - H*f]  −  λ·TV(f)

    The TV term is incorporated as a multiplicative correction following
    the approach of Dey et al. (2006) "3D Microscopy Deconvolution using
    Richardson-Lucy Algorithm with Total Variation Regularization".

    tv_lambda : regularisation weight.  0 = pure RL.  1e-3 typical.
    """
    img  = image.astype(np.float64)
    psf_ = psf.astype(np.float64)
    psf_flip = psf_[::-1, ::-1]          # PSF for back-projection

    H    = _psf_otf(psf_,      img.shape)
    Ht   = _psf_otf(psf_flip,  img.shape)

    u    = np.full_like(img, img.mean() + eps)

    for i in range(iterations):
        Hu   = np.real(_ifft2(H  * _fft2(u),   img.shape)).clip(eps)
        ratio = img / Hu
        corr = np.real(_ifft2(Ht * _fft2(ratio), img.shape))

        # TV multiplicative regulariser
        tv_pen = 1.0 - tv_lambda * _tv_grad(u)
        tv_pen = np.clip(tv_pen, 0.1, 10.0)

        u = u * corr * tv_pen
        u = np.clip(u, 0, None)
        if callback:
            callback(i + 1, iterations)
    return u


# ── 3. Blind Richardson–Lucy (alternating PSF & image estimation) ────────────

def deconv_blind_rl(image: np.ndarray, psf_init: np.ndarray,
                    iterations: int = 20,
                    inner_iter: int = 5,
                    psf_tv_lambda: float = 1e-3,
                    img_tv_lambda: float = 2e-4,
                    eps: float = 1e-7,
                    callback=None) -> tuple:
    """
    Blind deconvolution via alternating RL updates for both image and PSF.

    Alternates between:
      (a) Update image u  with current PSF estimate k
      (b) Update PSF k    with current image estimate u
    Both steps use TV regularisation to enforce smoothness/sparsity.

    Returns (restored_image, estimated_psf).

    Reference: Fish et al. (1995) "Blind deconvolution by means of the
    Richardson-Lucy algorithm."  JOSA A 12(1):58-65.
    """
    img  = image.astype(np.float64)
    u    = img.copy().clip(eps)
    k    = psf_init.astype(np.float64).clip(0)
    k   /= k.sum() + eps

    for outer in range(iterations):
        # (a) image update with current PSF
        K  = _psf_otf(k,        img.shape)
        Kt = _psf_otf(k[::-1, ::-1], img.shape)
        for _ in range(inner_iter):
            Ku   = np.real(_ifft2(K  * _fft2(u),   img.shape)).clip(eps)
            corr = np.real(_ifft2(Kt * _fft2(img / Ku), img.shape))
            tv_p = (1.0 - img_tv_lambda * _tv_grad(u)).clip(0.1, 10.0)
            u    = (u * corr * tv_p).clip(0)

        # (b) PSF update with current image estimate
        U  = _psf_otf(u,        img.shape)
        Ut = _psf_otf(u[::-1, ::-1], img.shape)
        for _ in range(inner_iter):
            Uk   = np.real(_ifft2(U  * _fft2(k),   img.shape)).clip(eps)
            corr_k = np.real(_ifft2(Ut * _fft2(img / Uk), img.shape))
            tv_pk  = (1.0 - psf_tv_lambda * _tv_grad(k)).clip(0.1, 10.0)
            k    = (k * corr_k * tv_pk).clip(0)
            s    = k.sum()
            k   /= s if s > eps else 1.0

        if callback:
            callback(outer + 1, iterations)

    return u, k


# ── 4. Plug-and-Play ADMM (PnP-ADMM) ────────────────────────────────────────

def _bm3d_like_denoiser(u: np.ndarray, sigma: float) -> np.ndarray:
    """
    Lightweight substitute for BM3D: anisotropic diffusion via Gaussian
    scale-space + edge-preserving soft-threshold in the wavelet domain.
    Replace with actual bm3d.bm3d(u, sigma) if the bm3d package is installed.
    """
    from scipy.ndimage import gaussian_filter as gf
    try:
        import bm3d
        return bm3d.bm3d(u.clip(0, 1), sigma_psd=sigma,
                          stage_arg=bm3d.BM3DStages.ALL_STAGES)
    except ImportError:
        pass
    # Fallback: bilateral-like multi-scale Gaussian blend
    smooth  = gf(u, sigma=max(0.5, sigma * 10))
    detail  = u - smooth
    detail  = _soft_threshold(detail, lam=sigma * 0.5)
    return (smooth + detail).clip(0)


def deconv_pnp_admm(image: np.ndarray, psf: np.ndarray,
                    iterations: int = 30,
                    rho: float = 0.01,
                    denoiser_sigma: float = 0.02,
                    eps: float = 1e-8,
                    callback=None) -> np.ndarray:
    """
    Plug-and-Play ADMM deconvolution.

    Solves:  min_f  ½‖Hf - d‖²  +  g(f)
    where g is implicitly defined by a denoiser (BM3D or fallback).

    ADMM splitting:
        f-update : closed-form in frequency domain
        v-update : apply denoiser to (f + u)
        u-update : dual variable update

    rho            : ADMM penalty parameter (~0.005–0.05)
    denoiser_sigma : noise level passed to denoiser (~0.01–0.05)

    Reference: Chan et al. (2017) "Plug-and-Play ADMM for Image Restoration."
    IEEE Trans. Computational Imaging 3(1):84-98.
    """
    img  = image.astype(np.float64)
    H    = _psf_otf(psf, img.shape)
    Ht   = np.conj(H)
    HtH  = Ht * H                         # |H|^2 in frequency domain
    Htd  = _fft2(np.real(_ifft2(Ht * _fft2(img), img.shape)))  # H^T d

    denom = HtH + rho                     # element-wise, no matrix inversion

    f = img.copy()
    v = img.copy()
    u = np.zeros_like(img)               # scaled dual variable

    for i in range(iterations):
        # f-update (frequency-domain least squares)
        rhs = Htd + rho * _fft2(v - u)
        f   = np.real(_ifft2(rhs / denom, img.shape)).clip(0)

        # v-update (denoiser proximal step)
        v = _bm3d_like_denoiser(f + u, sigma=denoiser_sigma)

        # u-update (dual ascent)
        u = u + f - v

        if callback:
            callback(i + 1, iterations)

    return f.clip(0)


# ── 5. Poisson-Markov MAP estimation ────────────────────────────────────────

def deconv_poisson_map(image: np.ndarray, psf: np.ndarray,
                       iterations: int = 50,
                       beta: float = 1e-3,
                       alpha: float = 1.5,
                       eps: float = 1e-7,
                       callback=None) -> np.ndarray:
    """
    Poisson-Markov MAP deconvolution via gradient ascent.

    Maximises:
        log p(f|d) ∝  Σ[d·log(Hf) - Hf]  -  β·Σ|∇f|^α

    The Markov prior uses a Gibbs potential |∇f|^α which interpolates
    between TV (α=1) and Tikhonov (α=2).

    Step size is chosen adaptively via Barzilai-Borwein.

    Reference: Molina et al. (2003) "Bayesian super-resolution with
    non-stationary Markov random field image models."
    """
    img  = image.astype(np.float64).clip(eps)
    psf_ = psf.astype(np.float64)
    H    = _psf_otf(psf_,         img.shape)
    Ht   = _psf_otf(psf_[::-1,::-1], img.shape)

    f    = img.copy()
    step = 1e-3

    grad_prev = None
    f_prev    = None

    for i in range(iterations):
        # Poisson data gradient:  H^T (1 - d / Hf)
        Hf       = np.real(_ifft2(H  * _fft2(f),   img.shape)).clip(eps)
        data_grad = np.real(_ifft2(Ht * _fft2(1.0 - img / Hf), img.shape))

        # Markov prior gradient:  β·α·|∇f|^(α-2)·Δf  (TV generalisation)
        gx   = np.diff(f, axis=1, append=f[:, :1])
        gy   = np.diff(f, axis=0, append=f[:1, :])
        norm = (gx**2 + gy**2 + eps) ** ((alpha - 2) / 2)
        dpx  = norm * gx;  dpy = norm * gy
        prior_grad = beta * alpha * (
            dpx - np.roll(dpx, 1, axis=1) +
            dpy - np.roll(dpy, 1, axis=0))

        grad = data_grad + prior_grad

        # Barzilai-Borwein step size
        if grad_prev is not None and f_prev is not None:
            df   = (f - f_prev).ravel()
            dg   = (grad - grad_prev).ravel()
            denom_ = np.dot(dg, dg)
            if denom_ > 1e-20:
                step = np.clip(np.dot(df, dg) / denom_, 1e-6, 1.0)

        f_prev    = f.copy()
        grad_prev = grad.copy()

        f = (f - step * grad).clip(0)
        if callback:
            callback(i + 1, iterations)

    return f


# ── 6. Hyper-Laplacian prior (non-convex, half-quadratic splitting) ──────────

def deconv_hyper_laplacian(image: np.ndarray, psf: np.ndarray,
                            iterations: int = 30,
                            lam: float = 2e-3,
                            alpha: float = 0.5,
                            beta_hq: float = 1.0,
                            beta_max: float = 256.0,
                            eps: float = 1e-8,
                            callback=None) -> np.ndarray:
    """
    Non-convex deconvolution with a hyper-Laplacian (sparse gradient) prior.

    Minimises:  ½‖Hf - d‖²  +  λ·Σ|∇f|^α,   0 < α < 1

    Solved via half-quadratic splitting (Krishnan & Fergus 2009):
      Introduce auxiliary variable w ≈ ∇f, add penalty β‖∇f - w‖².
      β is progressively increased (continuation) until convergence.

      f-subproblem : closed-form Wiener-like solve in freq. domain.
      w-subproblem : element-wise generalised soft-threshold.

    alpha  : sparsity exponent (0.5 typical, 0.8 less sparse)
    lam    : prior weight
    beta_hq: initial continuation penalty (doubled each outer iter)
    beta_max: stop increasing beta when reached

    Reference: Krishnan & Fergus (2009) "Fast Image Deconvolution using
    Hyper-Laplacian Priors." NeurIPS.
    """
    import scipy.fft as spfft

    img   = image.astype(np.float64)
    H_otf = _psf_otf(psf, img.shape)
    Ht    = np.conj(H_otf)
    HtH   = Ht * H_otf
    Htd   = np.real(_ifft2(Ht * _fft2(img), img.shape))

    # Precompute derivative OTFs  (finite differences → freq. domain)
    dx_kernel = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float64)
    dy_kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float64)
    Dx  = _psf_otf(dx_kernel, img.shape)
    Dy  = _psf_otf(dy_kernel, img.shape)
    DtD = np.conj(Dx)*Dx + np.conj(Dy)*Dy

    f    = img.copy()
    beta = beta_hq

    def _w_threshold(z: np.ndarray, threshold: float) -> np.ndarray:
        """
        Generalised shrinkage for |w|^α: closed-form for α=0.5 (Krishnan 2009),
        numerical for other values.
        """
        if abs(alpha - 0.5) < 1e-3:
            # Exact closed-form shrinkage for α = 1/2
            t  = (3.0 / 2.0) * threshold ** (2.0 / 3.0)
            w  = np.zeros_like(z)
            m  = np.abs(z) > t
            w[m] = (2.0/3.0 * np.abs(z[m]) * (
                1 + np.cos(2*np.pi/3 - 2/3 * np.arccos(
                    threshold / 4.0 * (3 / np.abs(z[m])) ** (3/2)
                )))).real
            w *= np.sign(z)
            return w
        else:
            # Iterative Newton solve for general α
            t   = lam / (beta + eps)
            w   = np.abs(z).copy()
            for _ in range(5):
                fw  = w + t * alpha * (w ** (alpha - 1) + eps) - np.abs(z)
                dfw = 1.0 + t * alpha * (alpha - 1) * (w ** (alpha - 2) + eps)
                w   = (w - fw / (dfw + eps)).clip(0)
            return w * np.sign(z)

    for i in range(iterations):
        # f-subproblem: (H^T H + β·D^T D) f = H^T d + β·D^T w
        wx  = np.diff(f, axis=1, append=f[:, :1])
        wy  = np.diff(f, axis=0, append=f[:1, :])
        threshold_w = lam / (beta + eps)

        wx = _w_threshold(wx, threshold_w)
        wy = _w_threshold(wy, threshold_w)

        # D^T w  (adjoint finite-difference = -divergence)
        Dtw = (wx - np.roll(wx, 1, axis=1) +
               wy - np.roll(wy, 1, axis=0))

        rhs = _fft2(Htd + beta * Dtw)
        f   = np.real(_ifft2(rhs / (HtH + beta * DtD + eps), img.shape)).clip(0)

        if beta < beta_max:
            beta = min(beta * 2.0, beta_max)

        if callback:
            callback(i + 1, iterations)

    return f


# ─────────────────────────────────────────────────────────────────────────────
# DECONVOLUTION WORKER
# ─────────────────────────────────────────────────────────────────────────────

class DeconvWorker(QThread):
    progress   = pyqtSignal(int, int)       # step, total
    log_line   = pyqtSignal(str)
    result_ready = pyqtSignal(object, object)  # original, restored
    finished   = pyqtSignal(dict)

    def __init__(self, cfg: dict, cancel_event: threading.Event):
        super().__init__()
        self.cfg     = cfg
        self._cancel = cancel_event

    def run(self):
        try:
            cfg = self.cfg
            self.log_line.emit(f"── Deconvolution: {cfg['algorithm']} ──")

            # Load image
            img_path = cfg["image_path"]
            if img_path.lower().endswith((".fit", ".fits")):
                data = astropy_fits.getdata(img_path).astype(np.float64)
                if data.ndim == 3:
                    data = data[0] if data.shape[0] <= 4 else data[:, :, 0]
            else:
                from PIL import Image as PILImage
                pil = PILImage.open(img_path).convert("L")
                data = np.array(pil, dtype=np.float64)

            # Normalise to 0–1
            mn, mx = data.min(), data.max()
            img = (data - mn) / (mx - mn + 1e-10)
            self.log_line.emit(f"Image: {img.shape[1]}×{img.shape[0]}  "
                               f"({data.dtype}  range {mn:.0f}–{mx:.0f})")

            # Build or load PSF
            psf = self._get_psf(cfg, img.shape)
            self.log_line.emit(
                f"PSF: {psf.shape[0]}×{psf.shape[1]}  sum={psf.sum():.4f}")

            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"}); return

            def _cb(step, total):
                self.progress.emit(step, total)
                if step % 5 == 0:
                    self.log_line.emit(f"  iteration {step}/{total}")

            algo = cfg["algorithm"]
            self.progress.emit(0, cfg.get("iterations", 30))

            if algo == "wiener":
                restored = deconv_wiener(img, psf, snr=cfg.get("snr", 30.0))

            elif algo == "rl_tv":
                restored = deconv_rl_tv(
                    img, psf,
                    iterations=cfg.get("iterations", 30),
                    tv_lambda=cfg.get("tv_lambda", 5e-4),
                    callback=_cb)

            elif algo == "blind_rl":
                restored, est_psf = deconv_blind_rl(
                    img, psf,
                    iterations=cfg.get("iterations", 20),
                    inner_iter=cfg.get("inner_iter", 5),
                    psf_tv_lambda=cfg.get("psf_tv_lambda", 1e-3),
                    img_tv_lambda=cfg.get("img_tv_lambda", 2e-4),
                    callback=_cb)
                self.log_line.emit("Blind RL: estimated PSF saved alongside output")
                cfg["_est_psf"] = est_psf

            elif algo == "pnp_admm":
                restored = deconv_pnp_admm(
                    img, psf,
                    iterations=cfg.get("iterations", 30),
                    rho=cfg.get("rho", 0.01),
                    denoiser_sigma=cfg.get("denoiser_sigma", 0.02),
                    callback=_cb)

            elif algo == "poisson_map":
                restored = deconv_poisson_map(
                    img, psf,
                    iterations=cfg.get("iterations", 50),
                    beta=cfg.get("beta", 1e-3),
                    alpha=cfg.get("alpha_pm", 1.5),
                    callback=_cb)

            elif algo == "hyper_laplacian":
                restored = deconv_hyper_laplacian(
                    img, psf,
                    iterations=cfg.get("iterations", 30),
                    lam=cfg.get("hl_lambda", 2e-3),
                    alpha=cfg.get("hl_alpha", 0.5),
                    callback=_cb)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"}); return

            # Re-scale back and save
            restored = np.clip(restored, 0, 1)
            out_path = self._save_result(restored, img, data, cfg)

            self.result_ready.emit(img, restored)
            self.log_line.emit(f"Saved → {out_path}")
            self.finished.emit({"success": True, "output_path": out_path})

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _get_psf(self, cfg: dict, img_shape: tuple) -> np.ndarray:
        psf_mode = cfg.get("psf_mode", "simulate")
        if psf_mode == "simulate":
            self.log_line.emit(
                f"Simulating PSF: D={cfg['aperture']:.3f}m  "
                f"obs={cfg['obstruction']:.2f}  r0={cfg['r0']:.3f}m")
            psf_full, _, _ = simulate_planetary_psf(
                aperture_diameter=cfg["aperture"],
                central_obstruction=cfg["obstruction"],
                r0=cfg["r0"],
                grid_size=cfg.get("psf_grid", 512),
            )
            psf_size = cfg.get("psf_crop", 64)
            return crop_psf(psf_full, psf_size)

        elif psf_mode == "file":
            psf_path = cfg["psf_path"]
            if psf_path.lower().endswith((".fit", ".fits")):
                psf = astropy_fits.getdata(psf_path).astype(np.float64)
            else:
                from PIL import Image as PILImage
                psf = np.array(PILImage.open(psf_path).convert("L"), dtype=np.float64)
            psf = psf.squeeze()
            if psf.ndim != 2:
                raise ValueError("PSF file must be a 2-D image")
            s = psf.sum()
            return psf / s if s > 0 else psf

        else:  # gaussian fallback
            sigma = cfg.get("gauss_sigma", 2.0)
            sz    = max(7, int(sigma * 6) | 1)   # odd size
            ax    = np.arange(sz) - sz // 2
            xx, yy = np.meshgrid(ax, ax)
            g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            return g / g.sum()

    def _save_result(self, restored: np.ndarray, img_norm: np.ndarray,
                     img_orig: np.ndarray, cfg: dict) -> str:
        src    = cfg["image_path"]
        stem   = os.path.splitext(os.path.basename(src))[0]
        algo   = cfg["algorithm"]
        out_dir = cfg.get("deconv_out_dir") or os.path.dirname(src) or "."
        os.makedirs(out_dir, exist_ok=True)

        out_mn = img_orig.min()
        out_mx = img_orig.max()
        out_arr = (restored * (out_mx - out_mn) + out_mn)

        out_path = os.path.join(out_dir, f"{stem}_{algo}_deconv.fits")
        hdu = astropy_fits.PrimaryHDU(data=out_arr.astype(np.float32))
        hdu.header["HISTORY"] = f"Deconvolved: {algo}  (planet_derotation.py)"
        for k, v in cfg.items():
            if isinstance(v, (int, float, str, bool)) and not k.startswith("_"):
                key = k[:8].upper()
                try:
                    hdu.header[key] = v
                except Exception:
                    pass
        hdu.writeto(out_path, overwrite=True)

        # Also save estimated PSF if blind RL
        est_psf = cfg.get("_est_psf")
        if est_psf is not None:
            psf_path = os.path.join(out_dir, f"{stem}_blind_psf.fits")
            astropy_fits.writeto(psf_path, est_psf.astype(np.float32), overwrite=True)

        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class RotationPreviewCanvas(FigureCanvasQTAgg):
    """Three-panel frame preview + rotation angle timeline."""

    def __init__(self, parent=None):
        self.fig      = Figure(figsize=(10, 7), facecolor=SIRIL_BG)
        self.ax_first = self.fig.add_subplot(2, 3, 1)
        self.ax_last  = self.fig.add_subplot(2, 3, 2)
        self.ax_derot = self.fig.add_subplot(2, 3, 3)
        self.ax_angles = self.fig.add_subplot(2, 1, 2)
        self._style_all()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _style_all(self):
        labels = [
            (self.ax_first,  "First frame (t=0)",          SIRIL_TEXT_DIM),
            (self.ax_last,   "Last frame (raw)",            SIRIL_WARNING),
            (self.ax_derot,  "Last frame (derotated)",      SIRIL_SUCCESS),
        ]
        for ax, title, color in labels:
            ax.set_facecolor(SIRIL_BG)
            ax.set_title(title, color=color, fontsize=8)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color(SIRIL_BORDER)

        self.ax_angles.set_facecolor(SIRIL_BG2)
        self.ax_angles.tick_params(colors=SIRIL_TEXT, labelsize=8)
        for sp in ["bottom", "left"]:
            self.ax_angles.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top", "right"]:
            self.ax_angles.spines[sp].set_visible(False)
        self.ax_angles.set_xlabel("Frame number", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_angles.set_ylabel("Rotation angle (°)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_angles.set_title("Rotation angle per frame (relative to t=0)",
                                  color=SIRIL_SECTION, fontsize=9)

    def _disp(self, frame: np.ndarray) -> np.ndarray:
        p_lo = np.percentile(frame, 0.5)
        p_hi = np.percentile(frame, 99.5)
        return np.clip((frame.astype(float) - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)

    def _ds(self, arr: np.ndarray, max_px: int = 300) -> np.ndarray:
        h, w   = arr.shape
        factor = max(1, max(h, w) // max_px)
        return arr[::factor, ::factor]

    def show_frames(self, first: np.ndarray, last_raw: np.ndarray,
                    last_derot: np.ndarray):
        for ax, frame in zip(
                [self.ax_first, self.ax_last, self.ax_derot],
                [first, last_raw, last_derot]):
            ax.clear()
            ax.imshow(self._ds(self._disp(frame)), cmap="gray",
                      origin="lower", aspect="equal", interpolation="nearest")
        self._style_all()
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def show_angles(self, angles: list, timestamps: list = None):
        self.ax_angles.clear()
        self._style_all()
        xs = list(range(len(angles)))
        self.ax_angles.plot(xs, angles, color=SIRIL_ACCENT,
                            linewidth=1.5, label="Rotation angle")
        self.ax_angles.axhline(0, color=SIRIL_BORDER, linewidth=0.8, linestyle="--")
        self.ax_angles.fill_between(xs, angles, alpha=0.15, color=SIRIL_ACCENT)
        if angles:
            self.ax_angles.set_xlim(0, max(len(angles) - 1, 1))
            ma = max(abs(min(angles)), abs(max(angles)))
            self.ax_angles.set_ylim(-ma * 1.2 - 0.01, ma * 1.2 + 0.01)
        self.ax_angles.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                               labelcolor=SIRIL_TEXT, fontsize=8)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def clear_all(self):
        for ax in [self.ax_first, self.ax_last, self.ax_derot, self.ax_angles]:
            ax.clear()
        self._style_all()
        self.draw()

# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW-ONLY WORKER  (angles + first/last preview, no full output)
# ─────────────────────────────────────────────────────────────────────────────

class PreviewWorker(QThread):
    log_line      = pyqtSignal(str)
    angles_ready  = pyqtSignal(list, list)
    preview_ready = pyqtSignal(object, object, object)
    finished      = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg = self.config
            self.log_line.emit("── Preview: loading frames ──")

            frames, timestamps = _load_frames(cfg, self._cancel, self.log_line.emit)
            if not frames:
                self.finished.emit({"success": False, "error": "No frames loaded"})
                return

            n = len(frames)
            self.log_line.emit(f"Loaded {n} frames for preview")

            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"}); return

            angles = _compute_angles(timestamps, cfg, self.log_line.emit)
            self.angles_ready.emit(angles, timestamps)

            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"}); return

            # Derotate ONLY the last frame for preview
            center_method = cfg.get("center_method", "centroid")
            if cfg.get("manual_center"):
                center = cfg["manual_center"]
            else:
                center = find_planet_center(frames[0], method=center_method)
                self.log_line.emit(
                    f"Planet center: ({center[1]:.1f}, {center[0]:.1f})")

            last_derot = derotate_frame(
                frames[-1], angles[-1], center,
                fill_value=_fill_value(frames[-1], cfg))

            self.preview_ready.emit(frames[0], frames[-1], last_derot)
            self.finished.emit({"success": True, "angles": angles,
                                 "timestamps": timestamps})
        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# FULL DEROTATION WORKER
# ─────────────────────────────────────────────────────────────────────────────

class DerotationWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    angles_ready  = pyqtSignal(list, list)
    preview_ready = pyqtSignal(object, object, object)
    finished      = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg = self.config
            self.progress.emit(1, 7, "Reading input frames…")

            frames, timestamps = _load_frames(cfg, self._cancel, self.log_line.emit)
            n = len(frames)
            self.log_line.emit(f"Loaded {n} frames")
            if n == 0:
                self.finished.emit({"success": False, "error": "No frames found"})
                return
            if self._cancel.is_set():
                self._abort(); return

            self.progress.emit(2, 7, "Computing rotation angles…")
            angles = _compute_angles(timestamps, cfg, self.log_line.emit)
            self.log_line.emit(
                f"Rotation range: {min(angles):.3f}° → {max(angles):.3f}°  "
                f"(span: {max(angles)-min(angles):.3f}°)")
            self.angles_ready.emit(angles, timestamps)
            if self._cancel.is_set():
                self._abort(); return

            self.progress.emit(3, 7, "Finding planet center…")
            center_method = cfg.get("center_method", "centroid")
            if cfg.get("manual_center"):
                center = cfg["manual_center"]
                self.log_line.emit(
                    f"Manual center: ({center[1]:.1f}, {center[0]:.1f})")
            else:
                center = find_planet_center(frames[0], method=center_method)
                self.log_line.emit(
                    f"Planet center: ({center[1]:.1f}, {center[0]:.1f})")

            dur     = session_duration_from_timestamps(timestamps)
            planet  = PLANETS[cfg["planet"]]
            rate    = planet["systems"][cfg.get("system", planet["default_system"])]
            h, w    = frames[0].shape
            diam_px = min(h, w) * 0.5
            blur_px = estimate_blur_without_derotation(rate, dur, diam_px)
            self.log_line.emit(
                f"Without derotation: ~{blur_px:.1f}px blur over "
                f"{dur/60:.1f} min session")
            if self._cancel.is_set():
                self._abort(); return

            self.progress.emit(4, 7, "Derotating frames…")
            fv = _fill_value(frames[0], cfg)
            derotated = derotate_sequence(
                frames, angles, center,
                fill_value=fv,
                log_callback=self.log_line.emit,
                cancel_event=self._cancel,
            )
            if self._cancel.is_set():
                self._abort(); return

            if derotated:
                self.preview_ready.emit(frames[0], frames[-1], derotated[-1])

            self.progress.emit(5, 7, "Saving output…")
            out_path = _save_output(derotated, cfg, self.log_line.emit)
            self.log_line.emit(f"Saved → {out_path}")
            if self._cancel.is_set():
                self._abort(); return

            # Optionally load in Siril
            if cfg.get("load_in_siril") and out_path:
                try:
                    out_dir = out_path if os.path.isdir(out_path) \
                              else os.path.dirname(out_path)
                    siril_api.cmd("cd", out_dir)
                    self.log_line.emit("Loaded output directory in Siril")
                except Exception as e:
                    self.log_line.emit(f"Siril load skipped: {e}")

            self.progress.emit(7, 7, "Complete")
            self.finished.emit({
                "success":        True,
                "output_path":    out_path,
                "frame_count":    n,
                "total_rotation": round(max(angles) - min(angles), 3),
                "blur_prevented": round(blur_px, 1),
                "session_min":    round(dur / 60, 1),
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})


# ─────────────────────────────────────────────────────────────────────────────
# SHARED WORKER HELPERS  (module-level so both workers can call them)
# ─────────────────────────────────────────────────────────────────────────────

def _load_frames(cfg: dict, cancel_event, log):
    input_type = cfg.get("input_type", "ser")
    frames, timestamps = [], []

    if input_type == "ser":
        ser_path   = cfg["input_path"]
        header     = read_ser_header(ser_path)
        n          = header["frame_count"]
        timestamps = read_ser_timestamps(ser_path)
        with open(ser_path, "rb") as f:
            for i in range(n):
                if cancel_event and cancel_event.is_set():
                    break
                frames.append(read_ser_frame(f, header, i).astype(np.float32))
        if not timestamps:
            log("No SER timestamps — using manual time settings")
            timestamps = timestamps_from_manual(
                cfg.get("manual_start_unix", 0.0),
                cfg.get("frame_rate_fps", 25.0),
                n,
            )

    elif input_type == "fits":
        fits_dir   = cfg["input_path"]
        fits_files = sorted(
            glob.glob(os.path.join(fits_dir, "*.fit")) +
            glob.glob(os.path.join(fits_dir, "*.fits"))
        )
        timestamps = read_fits_timestamps(fits_files)
        for path in fits_files:
            if cancel_event and cancel_event.is_set():
                break
            d = astropy_fits.getdata(path).astype(np.float32)
            if d.ndim == 3:
                d = d[0] if d.shape[0] <= 4 else d[:, :, 0]
            frames.append(d)
        timestamps = _fill_timestamps(timestamps, cfg.get("frame_rate_fps", 1.0), log)

    return frames, timestamps


def _fill_timestamps(timestamps: list, fps: float, log) -> list:
    result = list(timestamps)
    dt     = 1.0 / fps if fps > 0 else 1.0
    first_valid = next((t for t in result if t is not None), None)
    if first_valid is None:
        log("No timestamps found — using frame index × dt")
        return [i * dt for i in range(len(result))]
    idx0 = result.index(first_valid)
    for i in range(idx0):
        result[i] = first_valid - (idx0 - i) * dt
    last = result[0]
    for i in range(len(result)):
        if result[i] is None:
            result[i] = last + dt
        last = result[i]
    return result


def _compute_angles(timestamps: list, cfg: dict, log) -> list:
    planet = PLANETS[cfg["planet"]]
    system = cfg.get("system", planet["default_system"])
    rate   = planet["systems"][system]
    if cfg.get("use_horizons") and HAS_ASTROQUERY:
        log("Querying JPL Horizons for precise angles…")
        try:
            angles = compute_rotation_angles_horizons(
                timestamps,
                planet_id=planet["id"],
                observer_location=cfg.get("observer_location", "500"),
            )
            log("JPL Horizons query successful")
            return angles
        except Exception as e:
            log(f"Horizons failed: {e} — falling back to offline mode")
    log(f"Offline mode: {system} = {rate}°/day")
    return compute_rotation_angles_offline(timestamps, rate)


def _fill_value(frame: np.ndarray, cfg: dict) -> float:
    if cfg.get("fill_mode") == "background":
        return background_fill_value(frame)
    return 0.0


def _save_output(derotated: list, cfg: dict, log) -> str:
    input_path = cfg.get("input_path", "")
    out_dir    = cfg.get("output_dir") or (
        os.path.dirname(input_path) if not os.path.isdir(input_path) else input_path
    )
    os.makedirs(out_dir, exist_ok=True)
    planet   = cfg["planet"].replace(" ", "_")
    out_type = cfg.get("output_type", "ser")

    if out_type == "ser" and cfg.get("input_type") == "ser":
        orig_header = read_ser_header(input_path)
        px_depth    = orig_header["pixel_depth"]
        dtype       = np.uint8 if px_depth <= 8 else np.uint16
        out_path    = os.path.join(out_dir, f"{planet}_derotated.ser")
        frames_out  = []
        for fr in derotated:
            arr = np.clip(fr, 0, 255 if px_depth <= 8 else 65535)
            frames_out.append(arr.astype(dtype))
        write_ser_file(out_path, frames_out, orig_header)
    else:
        out_path = os.path.join(out_dir, f"{planet}_derotated_frames")
        os.makedirs(out_path, exist_ok=True)
        for i, frame in enumerate(derotated):
            hdu = astropy_fits.PrimaryHDU(data=frame.astype(np.float32))
            hdu.header["HISTORY"] = "Derotated by planet_derotation.py"
            astropy_fits.writeto(
                os.path.join(out_path, f"derot_{i:05d}.fit"),
                frame.astype(np.float32),
                hdu.header,
                overwrite=True,
            )
        out_path = out_path + os.sep

    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planet Derotation — Siril")
        self.setMinimumSize(900, 700)
        self.resize(960, 780)

        self._cancel_event   = threading.Event()
        self._worker         = None
        self._preview_worker = None
        self._angles         = []
        self._timestamps     = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # ── Title bar ──────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_lbl = QLabel("🪐  Planet Derotation  —  Siril")
        title_lbl.setStyleSheet(
            f"font-size:14pt; font-weight:bold; color:{SIRIL_ACCENT};")
        ver_lbl = QLabel("v1.0  |  WinJUPOS alternative")
        ver_lbl.setObjectName("dim")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        title_row.addWidget(ver_lbl)
        root.addLayout(title_row)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # ── Tabs ───────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self._build_tab_planet()
        self._build_tab_settings()
        self._build_tab_preview()
        self._build_tab_run()
        self._build_tab_deconv()
        self._build_tab_log()

        # ── Bottom bar ─────────────────────────────────────────────────────
        bot = QHBoxLayout()
        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setObjectName("dim")
        self._cancel_btn = QPushButton("✕  Cancel")
        self._cancel_btn.setObjectName("danger")
        self._cancel_btn.setFixedWidth(110)
        self._cancel_btn.clicked.connect(self._do_cancel)
        bot.addWidget(self._status_lbl, 1)
        bot.addWidget(self._cancel_btn)
        root.addLayout(bot)

        # Default planet
        self._select_planet("Jupiter")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1  —  Planet & Input
    # ────────────────────────────────────────────────────────────────────────

    def _build_tab_planet(self):
        tab = QWidget()
        sv  = QScrollArea()
        sv.setWidgetResizable(True)
        sv.setWidget(tab)
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        self.tabs.addTab(sv, "🪐  Planet & Input")

        # Planet selection ───────────────────────────────────────────────────
        gb_planet = QGroupBox("Planet selection")
        layout.addWidget(gb_planet)
        pv = QVBoxLayout(gb_planet)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._planet_btns: dict[str, QPushButton] = {}
        for name, data in PLANETS.items():
            btn = QPushButton(f"{data['emoji']}  {name}")
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda checked, n=name: self._select_planet(n))
            self._planet_btns[name] = btn
            btn_row.addWidget(btn)
        pv.addLayout(btn_row)

        sys_row = QHBoxLayout()
        sys_row.addWidget(QLabel("Rotation system:"))
        self._system_combo = QComboBox()
        self._system_combo.setMinimumWidth(200)
        self._system_combo.currentTextChanged.connect(self._on_system_changed)
        sys_row.addWidget(self._system_combo)
        sys_row.addStretch()
        pv.addLayout(sys_row)

        self._rate_lbl = QLabel()
        self._rate_lbl.setObjectName("ok")
        pv.addWidget(self._rate_lbl)

        self._safe_lbl = QLabel()
        self._safe_lbl.setObjectName("dim")
        pv.addWidget(self._safe_lbl)

        # Input ──────────────────────────────────────────────────────────────
        gb_input = QGroupBox("Input")
        layout.addWidget(gb_input)
        iv = QVBoxLayout(gb_input)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Input type:"))
        self._input_type = QComboBox()
        self._input_type.addItems(["SER file", "FITS sequence folder"])
        self._input_type.currentIndexChanged.connect(self._on_input_type_changed)
        type_row.addWidget(self._input_type)
        type_row.addStretch()
        iv.addLayout(type_row)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to SER file or FITS folder…")
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setFixedWidth(90)
        self._browse_btn.clicked.connect(self._browse_input)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(self._browse_btn)
        iv.addLayout(path_row)

        self._analyze_btn = QPushButton("🔍  Load & analyze")
        self._analyze_btn.setObjectName("primary")
        self._analyze_btn.clicked.connect(self._do_analyze)
        iv.addWidget(self._analyze_btn)

        self._input_info = QLabel("No file loaded.")
        self._input_info.setObjectName("dim")
        self._input_info.setWordWrap(True)
        iv.addWidget(self._input_info)

        # Timestamps ─────────────────────────────────────────────────────────
        gb_ts = QGroupBox("Timestamps")
        layout.addWidget(gb_ts)
        tv = QVBoxLayout(gb_ts)

        self._ts_status = QLabel("—")
        tv.addWidget(self._ts_status)

        self._manual_ts_widget = QWidget()
        mv = QFormLayout(self._manual_ts_widget)
        self._start_dt = QDateTimeEdit(QDateTime.currentDateTime())
        self._start_dt.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self._start_dt.setCalendarPopup(True)
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.1, 1000.0)
        self._fps_spin.setValue(25.0)
        self._fps_spin.setSuffix(" fps")
        self._tz_spin = QSpinBox()
        self._tz_spin.setRange(-12, 14)
        self._tz_spin.setValue(0)
        self._tz_spin.setSuffix(" h from UTC")
        mv.addRow("Session start (local):", self._start_dt)
        mv.addRow("Frame rate:", self._fps_spin)
        mv.addRow("Timezone offset:", self._tz_spin)
        self._manual_ts_widget.setVisible(False)
        tv.addWidget(self._manual_ts_widget)

        self._ts_extra = QLabel()
        self._ts_extra.setObjectName("dim")
        tv.addWidget(self._ts_extra)

        # Center detection ───────────────────────────────────────────────────
        gb_center = QGroupBox("Planet center detection")
        layout.addWidget(gb_center)
        cv = QVBoxLayout(gb_center)

        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("Method:"))
        self._center_combo = QComboBox()
        self._center_combo.addItems([
            "Automatic centroid",
            "Circle fit to limb",
            "Manual",
        ])
        self._center_combo.currentIndexChanged.connect(self._on_center_method_changed)
        center_row.addWidget(self._center_combo)
        center_row.addStretch()
        cv.addLayout(center_row)

        self._center_advice = QLabel()
        self._center_advice.setObjectName("dim")
        cv.addWidget(self._center_advice)

        self._manual_center_widget = QWidget()
        mcv = QFormLayout(self._manual_center_widget)
        self._center_x = QSpinBox()
        self._center_x.setRange(0, 99999)
        self._center_y = QSpinBox()
        self._center_y.setRange(0, 99999)
        mcv.addRow("Center X (px):", self._center_x)
        mcv.addRow("Center Y (px):", self._center_y)
        self._manual_center_widget.setVisible(False)
        cv.addWidget(self._manual_center_widget)

        layout.addStretch()

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2  —  Settings
    # ────────────────────────────────────────────────────────────────────────

    def _build_tab_settings(self):
        tab = QWidget()
        sv  = QScrollArea()
        sv.setWidgetResizable(True)
        sv.setWidget(tab)
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        self.tabs.addTab(sv, "⚙️  Settings")

        # Angle computation ──────────────────────────────────────────────────
        gb_ang = QGroupBox("Angle computation")
        layout.addWidget(gb_ang)
        av = QVBoxLayout(gb_ang)

        self._horizons_chk = QCheckBox("Use JPL Horizons  (accurate, needs internet)")
        if not HAS_ASTROQUERY:
            self._horizons_chk.setEnabled(False)
            self._horizons_chk.setText(
                "⚠ astroquery not available — offline mode only")
        self._horizons_chk.toggled.connect(self._on_horizons_toggled)
        av.addWidget(self._horizons_chk)

        self._horizons_widget = QWidget()
        hv = QFormLayout(self._horizons_widget)
        self._observer_edit = QLineEdit("500")
        self._observer_edit.setPlaceholderText("500=geocenter, 568=Mauna Kea, 950=La Palma")
        hv.addRow("Observer MPC code:", self._observer_edit)
        self._horizons_widget.setVisible(False)
        av.addWidget(self._horizons_widget)

        # Derotation ─────────────────────────────────────────────────────────
        gb_derot = QGroupBox("Derotation")
        layout.addWidget(gb_derot)
        dv = QFormLayout(gb_derot)

        self._fill_combo = QComboBox()
        self._fill_combo.addItems(["0 (black)", "Background estimate"])
        dv.addRow("Fill value:", self._fill_combo)

        fill_hint = QLabel("Value used for pixels outside frame after rotation")
        fill_hint.setObjectName("dim")
        dv.addRow("", fill_hint)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(10, 2000)
        self._batch_spin.setValue(200)
        self._batch_spin.setSuffix(" frames")
        dv.addRow("Batch size:", self._batch_spin)

        self._color_combo = QComboBox()
        self._color_combo.addItems(["Derotate raw Bayer", "Debayer then derotate"])
        dv.addRow("Color SER handling:", self._color_combo)

        # Output ─────────────────────────────────────────────────────────────
        gb_out = QGroupBox("Output")
        layout.addWidget(gb_out)
        ov = QVBoxLayout(gb_out)

        out_type_row = QHBoxLayout()
        out_type_row.addWidget(QLabel("Output type:"))
        self._out_type_combo = QComboBox()
        self._out_type_combo.addItems(["SER file (same as input)", "FITS sequence"])
        out_type_row.addWidget(self._out_type_combo)
        out_type_row.addStretch()
        ov.addLayout(out_type_row)

        out_dir_row = QHBoxLayout()
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("Default: same as input")
        self._out_browse_btn = QPushButton("Browse…")
        self._out_browse_btn.setFixedWidth(90)
        self._out_browse_btn.clicked.connect(self._browse_output)
        out_dir_row.addWidget(QLabel("Output folder:"))
        out_dir_row.addWidget(self._out_dir_edit, 1)
        out_dir_row.addWidget(self._out_browse_btn)
        ov.addLayout(out_dir_row)

        self._load_siril_chk = QCheckBox(
            "Load derotated sequence in Siril after export")
        ov.addWidget(self._load_siril_chk)

        self._stack_siril_chk = QCheckBox(
            "Stack in Siril after derotation")
        ov.addWidget(self._stack_siril_chk)

        layout.addStretch()

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3  —  Rotation Preview
    # ────────────────────────────────────────────────────────────────────────

    def _build_tab_preview(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        self.tabs.addTab(tab, "📊  Rotation Preview")

        self._canvas = RotationPreviewCanvas()
        layout.addWidget(self._canvas, 1)

        # Stats badges ───────────────────────────────────────────────────────
        gb_stats = QGroupBox("Session statistics")
        layout.addWidget(gb_stats)
        stats_row = QHBoxLayout(gb_stats)
        self._badge_rotation = self._make_badge("Total rotation", "—")
        self._badge_session  = self._make_badge("Session",        "—")
        self._badge_blur     = self._make_badge("Blur prevented", "—")
        self._badge_frames   = self._make_badge("Frames",         "—")
        for badge in [self._badge_rotation, self._badge_session,
                      self._badge_blur, self._badge_frames]:
            stats_row.addWidget(badge)

        self._preview_btn = QPushButton(
            "▶  Compute angles & show preview")
        self._preview_btn.setObjectName("primary")
        self._preview_btn.clicked.connect(self._do_preview)
        layout.addWidget(self._preview_btn)

    def _make_badge(self, label: str, value: str) -> QWidget:
        w  = QWidget()
        w.setStyleSheet(
            f"background:{SIRIL_BG3}; border:1px solid {SIRIL_BORDER};"
            f"border-radius:5px; padding:4px 10px;")
        bv = QVBoxLayout(w)
        bv.setContentsMargins(6, 4, 6, 4)
        lbl = QLabel(label)
        lbl.setObjectName("dim")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val = QLabel(value)
        val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:12pt; font-weight:bold;")
        bv.addWidget(lbl)
        bv.addWidget(val)
        w._val_lbl = val
        return w

    def _set_badge(self, badge: QWidget, text: str):
        badge._val_lbl.setText(text)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4  —  Run
    # ────────────────────────────────────────────────────────────────────────

    def _build_tab_run(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        self.tabs.addTab(tab, "▶  Run")

        self._run_btn = QPushButton("▶  Derotate all frames")
        self._run_btn.setObjectName("primary")
        self._run_btn.setMinimumHeight(44)
        self._run_btn.clicked.connect(self._do_run)
        layout.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 7)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        self._step_lbl = QLabel("Idle.")
        self._step_lbl.setObjectName("dim")
        layout.addWidget(self._step_lbl)

        # Result box ─────────────────────────────────────────────────────────
        self._result_box = QGroupBox("Result")
        self._result_box.setVisible(False)
        rv = QVBoxLayout(self._result_box)

        self._result_ok    = QLabel()
        self._result_path  = QLabel()
        self._result_path.setWordWrap(True)
        self._result_rot   = QLabel()
        self._result_blur  = QLabel()
        self._result_ok.setObjectName("ok")
        self._result_path.setObjectName("dim")

        rv.addWidget(self._result_ok)
        rv.addWidget(self._result_path)
        rv.addWidget(self._result_rot)
        rv.addWidget(self._result_blur)

        open_btn = QPushButton("📂  Open output folder")
        open_btn.clicked.connect(self._open_output_folder)
        rv.addWidget(open_btn)

        self._load_btn = QPushButton("Load in Siril")
        self._load_btn.setObjectName("export")
        self._load_btn.clicked.connect(self._load_in_siril)
        rv.addWidget(self._load_btn)

        layout.addWidget(self._result_box)
        layout.addStretch()

    # ────────────────────────────────────────────────────────────────────────
    # TAB 5  —  Deconvolution
    # ────────────────────────────────────────────────────────────────────────

    def _build_tab_deconv(self):
        tab = QWidget()
        sv  = QScrollArea()
        sv.setWidgetResizable(True)
        sv.setWidget(tab)
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        self.tabs.addTab(sv, "🔬  Deconvolution")

        self._deconv_worker  = None
        self._deconv_cancel  = threading.Event()

        # ── Image input ────────────────────────────────────────────────────
        gb_img = QGroupBox("Input image")
        layout.addWidget(gb_img)
        iv = QVBoxLayout(gb_img)

        img_row = QHBoxLayout()
        self._deconv_img_edit = QLineEdit()
        self._deconv_img_edit.setPlaceholderText("FITS or PNG/TIF image to deconvolve…")
        img_browse = QPushButton("Browse…")
        img_browse.setFixedWidth(90)
        img_browse.clicked.connect(self._browse_deconv_img)
        img_row.addWidget(self._deconv_img_edit, 1)
        img_row.addWidget(img_browse)
        iv.addLayout(img_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self._deconv_out_edit = QLineEdit()
        self._deconv_out_edit.setPlaceholderText("Default: same as input")
        out_browse = QPushButton("Browse…")
        out_browse.setFixedWidth(90)
        out_browse.clicked.connect(self._browse_deconv_out)
        out_row.addWidget(self._deconv_out_edit, 1)
        out_row.addWidget(out_browse)
        iv.addLayout(out_row)

        # ── PSF source ─────────────────────────────────────────────────────
        gb_psf = QGroupBox("PSF (Point Spread Function)")
        layout.addWidget(gb_psf)
        pv = QVBoxLayout(gb_psf)

        psf_mode_row = QHBoxLayout()
        psf_mode_row.addWidget(QLabel("PSF source:"))
        self._psf_mode_combo = QComboBox()
        self._psf_mode_combo.addItems([
            "Simulate (Kolmogorov atmosphere)",
            "Load from file",
            "Gaussian (simple estimate)",
        ])
        self._psf_mode_combo.currentIndexChanged.connect(self._on_psf_mode_changed)
        psf_mode_row.addWidget(self._psf_mode_combo)
        psf_mode_row.addStretch()
        pv.addLayout(psf_mode_row)

        # Simulate panel
        self._psf_sim_widget = QWidget()
        sim_form = QFormLayout(self._psf_sim_widget)
        self._ap_spin  = QDoubleSpinBox(); self._ap_spin.setRange(0.05, 10.0)
        self._ap_spin.setValue(0.28); self._ap_spin.setSuffix(" m")
        self._obs_spin = QDoubleSpinBox(); self._obs_spin.setRange(0.0, 0.95)
        self._obs_spin.setValue(0.34); self._obs_spin.setSingleStep(0.01)
        self._r0_spin  = QDoubleSpinBox(); self._r0_spin.setRange(0.01, 1.0)
        self._r0_spin.setValue(0.08); self._r0_spin.setSuffix(" m")
        self._r0_spin.setSingleStep(0.01)
        self._psf_grid_spin = QSpinBox(); self._psf_grid_spin.setRange(128, 2048)
        self._psf_grid_spin.setValue(512); self._psf_grid_spin.setSingleStep(128)
        self._psf_crop_spin = QSpinBox(); self._psf_crop_spin.setRange(8, 256)
        self._psf_crop_spin.setValue(64)
        sim_form.addRow("Aperture diameter:", self._ap_spin)
        sim_form.addRow("Central obstruction:", self._obs_spin)
        sim_form.addRow("r₀ (Fried param):", self._r0_spin)
        r0_hint = QLabel("0.05=bad  0.10=average  0.20=excellent seeing")
        r0_hint.setObjectName("dim"); sim_form.addRow("", r0_hint)
        sim_form.addRow("FFT grid size:", self._psf_grid_spin)
        sim_form.addRow("PSF crop size (px):", self._psf_crop_spin)
        pv.addWidget(self._psf_sim_widget)

        # File panel
        self._psf_file_widget = QWidget()
        pf_row = QHBoxLayout(self._psf_file_widget)
        self._psf_file_edit = QLineEdit()
        self._psf_file_edit.setPlaceholderText("Path to PSF FITS / image…")
        psf_file_browse = QPushButton("Browse…")
        psf_file_browse.setFixedWidth(90)
        psf_file_browse.clicked.connect(self._browse_psf_file)
        pf_row.addWidget(self._psf_file_edit, 1)
        pf_row.addWidget(psf_file_browse)
        self._psf_file_widget.setVisible(False)
        pv.addWidget(self._psf_file_widget)

        # Gaussian panel
        self._psf_gauss_widget = QWidget()
        gf = QFormLayout(self._psf_gauss_widget)
        self._gauss_sigma = QDoubleSpinBox()
        self._gauss_sigma.setRange(0.3, 20.0); self._gauss_sigma.setValue(2.0)
        self._gauss_sigma.setSuffix(" px")
        gf.addRow("Gaussian σ:", self._gauss_sigma)
        self._psf_gauss_widget.setVisible(False)
        pv.addWidget(self._psf_gauss_widget)

        # ── Algorithm selector ─────────────────────────────────────────────
        gb_algo = QGroupBox("Algorithm")
        layout.addWidget(gb_algo)
        av = QVBoxLayout(gb_algo)

        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Method:"))
        self._algo_combo = QComboBox()
        self._algo_combo.addItems([
            "Wiener deconvolution",
            "Richardson–Lucy + Total Variation",
            "Blind Richardson–Lucy",
            "Plug-and-Play ADMM (PnP-ADMM)",
            "Poisson-Markov MAP",
            "Hyper-Laplacian (non-convex)",
        ])
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        algo_row.addWidget(self._algo_combo)
        algo_row.addStretch()
        av.addLayout(algo_row)

        self._algo_desc = QLabel()
        self._algo_desc.setObjectName("dim")
        self._algo_desc.setWordWrap(True)
        av.addWidget(self._algo_desc)

        # ── Per-algorithm parameter panels ─────────────────────────────────
        self._param_panels: dict[str, QWidget] = {}

        # Wiener
        pw = self._make_param_panel()
        pf_w = QFormLayout(pw)
        self._snr_spin = QDoubleSpinBox(); self._snr_spin.setRange(1, 10000)
        self._snr_spin.setValue(30); self._snr_spin.setSuffix("  (SNR estimate)")
        pf_w.addRow("SNR:", self._snr_spin)
        pf_w.addRow("", self._dim_label("Higher = sharper but more noise amplification"))
        self._param_panels["wiener"] = pw; av.addWidget(pw)

        # RL-TV
        prl = self._make_param_panel()
        pf_rl = QFormLayout(prl)
        self._rl_iter  = self._iter_spin(30)
        self._tv_lam   = QDoubleSpinBox(); self._tv_lam.setRange(0, 1)
        self._tv_lam.setValue(5e-4); self._tv_lam.setDecimals(6); self._tv_lam.setSingleStep(1e-4)
        pf_rl.addRow("Iterations:", self._rl_iter)
        pf_rl.addRow("TV λ:", self._tv_lam)
        pf_rl.addRow("", self._dim_label("λ=0 is pure RL. 1e-4–1e-3 typical."))
        self._param_panels["rl_tv"] = prl; av.addWidget(prl)

        # Blind RL
        pbl = self._make_param_panel()
        pf_bl = QFormLayout(pbl)
        self._blind_iter  = self._iter_spin(20)
        self._blind_inner = QSpinBox(); self._blind_inner.setRange(1, 20)
        self._blind_inner.setValue(5)
        self._blind_psf_lam = QDoubleSpinBox(); self._blind_psf_lam.setRange(0, 1)
        self._blind_psf_lam.setValue(1e-3); self._blind_psf_lam.setDecimals(5)
        self._blind_img_lam = QDoubleSpinBox(); self._blind_img_lam.setRange(0, 1)
        self._blind_img_lam.setValue(2e-4); self._blind_img_lam.setDecimals(5)
        pf_bl.addRow("Outer iterations:", self._blind_iter)
        pf_bl.addRow("Inner iterations:", self._blind_inner)
        pf_bl.addRow("PSF TV λ:", self._blind_psf_lam)
        pf_bl.addRow("Image TV λ:", self._blind_img_lam)
        pf_bl.addRow("", self._dim_label(
            "PSF is simultaneously estimated. Output includes estimated PSF FITS."))
        self._param_panels["blind_rl"] = pbl; av.addWidget(pbl)

        # PnP-ADMM
        padmm = self._make_param_panel()
        pf_admm = QFormLayout(padmm)
        self._admm_iter   = self._iter_spin(30)
        self._admm_rho    = QDoubleSpinBox(); self._admm_rho.setRange(1e-4, 10)
        self._admm_rho.setValue(0.01); self._admm_rho.setDecimals(4)
        self._admm_sigma  = QDoubleSpinBox(); self._admm_sigma.setRange(0.001, 0.5)
        self._admm_sigma.setValue(0.02); self._admm_sigma.setDecimals(4)
        pf_admm.addRow("Iterations:", self._admm_iter)
        pf_admm.addRow("ADMM ρ:", self._admm_rho)
        pf_admm.addRow("Denoiser σ:", self._admm_sigma)
        pf_admm.addRow("", self._dim_label(
            "Uses BM3D denoiser if installed, else Gaussian scale-space fallback."))
        self._param_panels["pnp_admm"] = padmm; av.addWidget(padmm)

        # Poisson MAP
        ppm = self._make_param_panel()
        pf_pm = QFormLayout(ppm)
        self._pm_iter  = self._iter_spin(50)
        self._pm_beta  = QDoubleSpinBox(); self._pm_beta.setRange(1e-6, 10)
        self._pm_beta.setValue(1e-3); self._pm_beta.setDecimals(6)
        self._pm_alpha = QDoubleSpinBox(); self._pm_alpha.setRange(1.0, 2.0)
        self._pm_alpha.setValue(1.5); self._pm_alpha.setSingleStep(0.1)
        pf_pm.addRow("Iterations:", self._pm_iter)
        pf_pm.addRow("Prior β:", self._pm_beta)
        pf_pm.addRow("Gradient exponent α:", self._pm_alpha)
        pf_pm.addRow("", self._dim_label(
            "α=1 → TV prior. α=2 → Tikhonov. 1.5 balances both."))
        self._param_panels["poisson_map"] = ppm; av.addWidget(ppm)

        # Hyper-Laplacian
        phl = self._make_param_panel()
        pf_hl = QFormLayout(phl)
        self._hl_iter   = self._iter_spin(30)
        self._hl_lam    = QDoubleSpinBox(); self._hl_lam.setRange(1e-6, 1)
        self._hl_lam.setValue(2e-3); self._hl_lam.setDecimals(6)
        self._hl_alpha  = QDoubleSpinBox(); self._hl_alpha.setRange(0.1, 0.99)
        self._hl_alpha.setValue(0.5); self._hl_alpha.setSingleStep(0.05)
        pf_hl.addRow("Iterations:", self._hl_iter)
        pf_hl.addRow("λ (prior weight):", self._hl_lam)
        pf_hl.addRow("α (sparsity, 0<α<1):", self._hl_alpha)
        pf_hl.addRow("", self._dim_label(
            "α=0.5 (Krishnan & Fergus 2009): best for natural images with sparse gradients."))
        self._param_panels["hyper_laplacian"] = phl; av.addWidget(phl)

        # ── Progress + run ──────────────────────────────────────────────────
        self._deconv_run_btn = QPushButton("▶  Run deconvolution")
        self._deconv_run_btn.setObjectName("primary")
        self._deconv_run_btn.setMinimumHeight(40)
        self._deconv_run_btn.clicked.connect(self._do_deconv)
        layout.addWidget(self._deconv_run_btn)

        self._deconv_progress = QProgressBar()
        self._deconv_progress.setTextVisible(False)
        layout.addWidget(self._deconv_progress)

        self._deconv_status = QLabel("Ready.")
        self._deconv_status.setObjectName("dim")
        layout.addWidget(self._deconv_status)

        layout.addStretch()

        # Initialise visible panels
        self._on_algo_changed(0)
        self._update_algo_desc(0)

    # helpers used only by _build_tab_deconv
    def _make_param_panel(self) -> QWidget:
        w = QWidget(); w.setVisible(False); return w

    def _iter_spin(self, default: int) -> QSpinBox:
        s = QSpinBox(); s.setRange(1, 500); s.setValue(default); return s

    def _dim_label(self, text: str) -> QLabel:
        l = QLabel(text); l.setObjectName("dim"); l.setWordWrap(True); return l

    # ── PSF mode switching ─────────────────────────────────────────────────
    def _on_psf_mode_changed(self, idx: int):
        self._psf_sim_widget.setVisible(idx == 0)
        self._psf_file_widget.setVisible(idx == 1)
        self._psf_gauss_widget.setVisible(idx == 2)

    # ── Algorithm switching ────────────────────────────────────────────────
    _ALGO_KEYS = ["wiener", "rl_tv", "blind_rl",
                  "pnp_admm", "poisson_map", "hyper_laplacian"]

    _ALGO_DESCS = [
        "Classical Wiener filter. Fastest. Best starting point for well-exposed frames. "
        "No iterative step — single frequency-domain division.",

        "Richardson–Lucy EM with isotropic Total Variation regularisation. "
        "Preserves edges, suppresses ringing. Good for most planetary imaging. "
        "(Dey et al. 2006)",

        "Alternating RL updates for image and PSF simultaneously. "
        "Use when the true PSF is uncertain. Outputs estimated PSF FITS alongside result. "
        "(Fish et al. 1995)",

        "Plug-and-Play ADMM. Replaces proximal step with a denoiser (BM3D if installed). "
        "Most flexible — denoiser implicitly defines the prior. "
        "(Chan et al. 2017)",

        "Poisson-Markov MAP via gradient ascent with Barzilai-Borwein step. "
        "Correct noise model for photon-counting data (low-light planetary). "
        "(Molina et al. 2003)",

        "Non-convex deconvolution with hyper-Laplacian gradient prior (α<1). "
        "Sharpest results on well-exposed targets. Half-quadratic splitting with "
        "continuation. (Krishnan & Fergus 2009, NeurIPS)",
    ]

    def _on_algo_changed(self, idx: int):
        for key, panel in self._param_panels.items():
            panel.setVisible(key == self._ALGO_KEYS[idx])
        self._update_algo_desc(idx)

    def _update_algo_desc(self, idx: int):
        if hasattr(self, "_algo_desc"):
            self._algo_desc.setText(self._ALGO_DESCS[idx])

    # ── File browsers ──────────────────────────────────────────────────────
    def _browse_deconv_img(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "FITS / Images (*.fit *.fits *.png *.tif *.tiff *.jpg);;All (*)")
        if path:
            self._deconv_img_edit.setText(path)

    def _browse_deconv_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._deconv_out_edit.setText(path)

    def _browse_psf_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PSF file", "",
            "FITS / Images (*.fit *.fits *.png *.tif *.tiff);;All (*)")
        if path:
            self._psf_file_edit.setText(path)

    # ── Build deconv config ────────────────────────────────────────────────
    def _build_deconv_config(self) -> dict | None:
        img_path = self._deconv_img_edit.text().strip()
        if not img_path or not os.path.exists(img_path):
            QMessageBox.warning(self, "No image", "Please select an input image.")
            return None

        psf_mode_map = {0: "simulate", 1: "file", 2: "gaussian"}
        psf_mode     = psf_mode_map[self._psf_mode_combo.currentIndex()]

        if psf_mode == "file":
            psf_path = self._psf_file_edit.text().strip()
            if not psf_path or not os.path.exists(psf_path):
                QMessageBox.warning(self, "No PSF", "Please select a PSF file.")
                return None
        else:
            psf_path = ""

        algo = self._ALGO_KEYS[self._algo_combo.currentIndex()]

        return {
            "image_path":      img_path,
            "deconv_out_dir":  self._deconv_out_edit.text().strip() or None,
            "psf_mode":        psf_mode,
            "psf_path":        psf_path,
            "aperture":        self._ap_spin.value(),
            "obstruction":     self._obs_spin.value(),
            "r0":              self._r0_spin.value(),
            "psf_grid":        self._psf_grid_spin.value(),
            "psf_crop":        self._psf_crop_spin.value(),
            "gauss_sigma":     self._gauss_sigma.value(),
            "algorithm":       algo,
            # Wiener
            "snr":             self._snr_spin.value(),
            # RL-TV
            "iterations":      self._get_iter_for_algo(algo),
            "tv_lambda":       self._tv_lam.value(),
            # Blind RL
            "inner_iter":      self._blind_inner.value(),
            "psf_tv_lambda":   self._blind_psf_lam.value(),
            "img_tv_lambda":   self._blind_img_lam.value(),
            # ADMM
            "rho":             self._admm_rho.value(),
            "denoiser_sigma":  self._admm_sigma.value(),
            # Poisson MAP
            "beta":            self._pm_beta.value(),
            "alpha_pm":        self._pm_alpha.value(),
            # Hyper-Laplacian
            "hl_lambda":       self._hl_lam.value(),
            "hl_alpha":        self._hl_alpha.value(),
        }

    def _get_iter_for_algo(self, algo: str) -> int:
        return {
            "wiener":           1,
            "rl_tv":            self._rl_iter.value(),
            "blind_rl":         self._blind_iter.value(),
            "pnp_admm":         self._admm_iter.value(),
            "poisson_map":      self._pm_iter.value(),
            "hyper_laplacian":  self._hl_iter.value(),
        }.get(algo, 30)

    # ── Run deconvolution ──────────────────────────────────────────────────
    def _do_deconv(self):
        if self._deconv_worker and self._deconv_worker.isRunning():
            QMessageBox.information(self, "Running", "Deconvolution already in progress.")
            return
        cfg = self._build_deconv_config()
        if cfg is None:
            return

        self._deconv_cancel.clear()
        self._deconv_progress.setValue(0)
        self._deconv_status.setText("Running…")
        self._deconv_run_btn.setEnabled(False)
        self._log_line(f"── Deconvolution start: {cfg['algorithm']} ──")

        self._deconv_worker = DeconvWorker(cfg, self._deconv_cancel)
        self._deconv_worker.progress.connect(
            lambda s, t: self._deconv_progress.setMaximum(t) or
                         self._deconv_progress.setValue(s))
        self._deconv_worker.log_line.connect(self._log_line)
        self._deconv_worker.result_ready.connect(self._on_deconv_preview)
        self._deconv_worker.finished.connect(self._on_deconv_finished)
        self._deconv_worker.start()

    def _on_deconv_preview(self, original: np.ndarray, restored: np.ndarray):
        """Show before/after in the Rotation Preview canvas (reuse)."""
        dummy = np.zeros_like(original)
        self._canvas.show_frames(original, dummy, restored)
        self.tabs.setCurrentIndex(2)   # jump to preview tab

    def _on_deconv_finished(self, result: dict):
        self._deconv_run_btn.setEnabled(True)
        if result.get("success"):
            out = result.get("output_path", "")
            self._deconv_status.setText(f"✓  Saved: {out}")
            self._log_line(f"✓ Deconvolution complete → {out}")
            self._set_status("Deconvolution done.")
        else:
            err = result.get("error", "Unknown error")
            self._deconv_status.setText(f"✗  {err}")
            self._log_line(f"✗ Deconvolution failed: {err}")
            self._set_status(f"Deconvolution failed: {err}")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 6  —  Log
    # ────────────────────────────────────────────────────────────────────────

    def _build_tab_log(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        self.tabs.addTab(tab, "📋  Log")

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        layout.addWidget(self._log, 1)

        save_btn = QPushButton("Save log…")
        save_btn.setFixedWidth(120)
        save_btn.clicked.connect(self._save_log)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(save_btn)
        layout.addLayout(row)

    # ────────────────────────────────────────────────────────────────────────
    # PLANET SELECTION
    # ────────────────────────────────────────────────────────────────────────

    def _select_planet(self, name: str):
        self._planet = name
        data = PLANETS[name]

        # Button highlights
        for n, btn in self._planet_btns.items():
            if n == name:
                btn.setStyleSheet(
                    f"background:{SIRIL_ACCENT2}; border:2px solid {SIRIL_ACCENT};"
                    f"color:white; font-weight:bold; border-radius:4px; padding:6px 12px;")
            else:
                btn.setStyleSheet("")

        # System combo
        self._system_combo.blockSignals(True)
        self._system_combo.clear()
        for sys_name in data["systems"]:
            self._system_combo.addItem(sys_name)
        idx = list(data["systems"]).index(data["default_system"])
        self._system_combo.setCurrentIndex(idx)
        self._system_combo.blockSignals(False)

        self._update_rate_labels()

    def _on_system_changed(self, _):
        self._update_rate_labels()

    def _update_rate_labels(self):
        data    = PLANETS[self._planet]
        system  = self._system_combo.currentText()
        if not system:
            return
        rate = data["systems"].get(system, 0.0)
        self._rate_lbl.setText(
            f"Rotation rate: {rate:+.3f}°/day  ({system})")
        safe = data["safe_session_min"]
        self._safe_lbl.setText(
            f"Safe session without derotation: ~{safe} minute{'s' if safe != 1 else ''}")

    # ────────────────────────────────────────────────────────────────────────
    # INPUT HANDLING
    # ────────────────────────────────────────────────────────────────────────

    def _on_input_type_changed(self):
        is_ser = self._input_type.currentIndex() == 0
        self._path_edit.setPlaceholderText(
            "Path to .ser file…" if is_ser else "Path to FITS folder…")

    def _browse_input(self):
        if self._input_type.currentIndex() == 0:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select SER file", "", "SER files (*.ser);;All files (*)")
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Select FITS sequence folder")
        if path:
            self._path_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._out_dir_edit.setText(path)

    def _do_analyze(self):
        path = self._path_edit.text().strip()
        if not path or not os.path.exists(path):
            self._input_info.setText("⚠ File or folder not found.")
            self._input_info.setObjectName("warn")
            return

        try:
            is_ser = self._input_type.currentIndex() == 0
            if is_ser:
                hdr = read_ser_header(path)
                n   = hdr["frame_count"]
                w   = hdr["width"]
                h   = hdr["height"]
                bpp = hdr["pixel_depth"]
                ts  = read_ser_timestamps(path)
                self._input_info.setText(
                    f"✓  {n} frames  ·  {w}×{h}  ·  {bpp}-bit"
                    f"  ·  Timestamps: {'from SER trailer' if ts else 'none (manual required)'}")
                self._input_info.setStyleSheet(f"color:{SIRIL_SUCCESS};")
                self._update_ts_panel(ts, "SER trailer" if ts else None)
            else:
                fits_files = sorted(
                    glob.glob(os.path.join(path, "*.fit")) +
                    glob.glob(os.path.join(path, "*.fits"))
                )
                n  = len(fits_files)
                ts = read_fits_timestamps(fits_files) if fits_files else []
                has_ts = any(t is not None for t in ts)
                self._input_info.setText(
                    f"✓  {n} FITS files found  ·  "
                    f"Timestamps: {'FITS DATE-OBS' if has_ts else 'none (manual required)'}")
                self._input_info.setStyleSheet(f"color:{SIRIL_SUCCESS};")
                self._update_ts_panel(ts if has_ts else [], "FITS DATE-OBS" if has_ts else None)

            self._log_line(f"Analyzed: {path}")
            self._set_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self._input_info.setText(f"Error: {e}")
            self._input_info.setStyleSheet(f"color:{SIRIL_ERROR};")

    def _update_ts_panel(self, timestamps: list, source: str | None):
        if source and timestamps:
            self._ts_status.setText(f"✓  Timestamps from {source}")
            self._ts_status.setObjectName("ok")
            self._manual_ts_widget.setVisible(False)
            dur = session_duration_from_timestamps(timestamps)
            n   = len(timestamps)
            fps = n / dur if dur > 0 else 0.0
            self._ts_extra.setText(
                f"Session duration: {dur/60:.1f} min  ·  ~{fps:.1f} fps")
        else:
            self._ts_status.setText("⚠  No timestamps found — manual entry required")
            self._ts_status.setObjectName("warn")
            self._manual_ts_widget.setVisible(True)
            self._ts_extra.setText("")

        # Re-apply style
        self._ts_status.style().unpolish(self._ts_status)
        self._ts_status.style().polish(self._ts_status)

    def _on_center_method_changed(self, idx: int):
        is_manual = idx == 2
        self._manual_center_widget.setVisible(is_manual)

    def _on_horizons_toggled(self, checked: bool):
        self._horizons_widget.setVisible(checked)

    # ────────────────────────────────────────────────────────────────────────
    # BUILD CONFIG DICT
    # ────────────────────────────────────────────────────────────────────────

    def _build_config(self) -> dict | None:
        path = self._path_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "No input",
                                "Please select and analyze an input file first.")
            return None

        is_ser     = self._input_type.currentIndex() == 0
        input_type = "ser" if is_ser else "fits"

        # Manual timestamp params
        qdt        = self._start_dt.dateTime()
        start_unix = qdt.toSecsSinceEpoch() - self._tz_spin.value() * 3600

        # Center
        center_map    = {0: "centroid", 1: "circle_fit", 2: "manual"}
        center_method = center_map[self._center_combo.currentIndex()]
        manual_center = None
        if center_method == "manual":
            manual_center = (float(self._center_y.value()),
                             float(self._center_x.value()))

        # Output type
        out_type = "ser" if self._out_type_combo.currentIndex() == 0 else "fits"

        # Fill mode
        fill_mode = "background" if self._fill_combo.currentIndex() == 1 else "black"

        planet = self._planet
        data   = PLANETS[planet]
        system = self._system_combo.currentText() or data["default_system"]

        return {
            "planet":            planet,
            "system":            system,
            "input_path":        path,
            "input_type":        input_type,
            "manual_start_unix": start_unix,
            "frame_rate_fps":    self._fps_spin.value(),
            "center_method":     center_method,
            "manual_center":     manual_center,
            "use_horizons":      self._horizons_chk.isChecked() and HAS_ASTROQUERY,
            "observer_location": self._observer_edit.text().strip() or "500",
            "output_type":       out_type,
            "output_dir":        self._out_dir_edit.text().strip() or None,
            "fill_mode":         fill_mode,
            "load_in_siril":     self._load_siril_chk.isChecked(),
            "stack_in_siril":    self._stack_siril_chk.isChecked(),
            "batch_size":        self._batch_spin.value(),
        }

    # ────────────────────────────────────────────────────────────────────────
    # PREVIEW RUN
    # ────────────────────────────────────────────────────────────────────────

    def _do_preview(self):
        if self._preview_worker and self._preview_worker.isRunning():
            return
        cfg = self._build_config()
        if cfg is None:
            return
        self._cancel_event.clear()
        self._log_line("── Starting preview ──")
        self._canvas.clear_all()
        self._preview_btn.setEnabled(False)
        self._set_status("Computing preview…")

        self._preview_worker = PreviewWorker(cfg, self._cancel_event)
        self._preview_worker.log_line.connect(self._log_line)
        self._preview_worker.angles_ready.connect(self._on_angles_ready)
        self._preview_worker.preview_ready.connect(self._on_preview_ready)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.start()

    def _on_angles_ready(self, angles: list, timestamps: list):
        self._angles     = angles
        self._timestamps = timestamps
        self._canvas.show_angles(angles, timestamps)

        dur = session_duration_from_timestamps(timestamps)
        if angles:
            span = max(angles) - min(angles)
            data = PLANETS[self._planet]
            rate = data["systems"].get(
                self._system_combo.currentText(), 1.0)
            h_dummy, w_dummy = 500, 500
            blur = estimate_blur_without_derotation(rate, dur, min(h_dummy, w_dummy) * 0.5)
            self._set_badge(self._badge_rotation, f"{span:.2f}°")
            self._set_badge(self._badge_session,  f"{dur/60:.1f} min")
            self._set_badge(self._badge_blur,      f"~{blur:.1f} px")
            self._set_badge(self._badge_frames,    str(len(angles)))

    def _on_preview_ready(self, first, last_raw, last_derot):
        self._canvas.show_frames(first, last_raw, last_derot)

    def _on_preview_finished(self, result: dict):
        self._preview_btn.setEnabled(True)
        if result.get("success"):
            self._set_status("Preview complete.")
            self.tabs.setCurrentIndex(2)   # jump to preview tab
        else:
            self._set_status(f"Preview failed: {result.get('error','')}")

    # ────────────────────────────────────────────────────────────────────────
    # FULL DEROTATION RUN
    # ────────────────────────────────────────────────────────────────────────

    def _do_run(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Running",
                                    "Derotation already in progress.")
            return
        cfg = self._build_config()
        if cfg is None:
            return

        self._cancel_event.clear()
        self._result_box.setVisible(False)
        self._progress.setValue(0)
        self._step_lbl.setText("Starting…")
        self._run_btn.setEnabled(False)
        self._set_status("Derotating frames…")
        self._log_line("── Starting full derotation ──")

        self._worker = DerotationWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log_line)
        self._worker.angles_ready.connect(self._on_angles_ready)
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.start()

    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setMaximum(total)
        self._progress.setValue(step)
        self._step_lbl.setText(msg)
        self._set_status(msg)

    def _on_run_finished(self, result: dict):
        self._run_btn.setEnabled(True)
        if result.get("success"):
            out_path  = result.get("output_path", "")
            n         = result.get("frame_count", 0)
            total_rot = result.get("total_rotation", 0.0)
            blur      = result.get("blur_prevented", 0.0)
            sess      = result.get("session_min", 0.0)

            self._last_output_path = out_path
            self._result_ok.setText(f"✓  Derotated {n} frames successfully")
            self._result_path.setText(f"Output: {out_path}")
            self._result_rot.setText(f"Total rotation corrected: {total_rot:.3f}°")
            self._result_blur.setText(
                f"Estimated blur prevented: ~{blur:.1f} px at equator")
            self._result_box.setVisible(True)
            self._set_status(f"Done — {n} frames derotated, {total_rot:.2f}° corrected")
            self._log_line(f"✓ Complete. {n} frames · {total_rot:.2f}° · {sess:.1f} min session")
            self.tabs.setCurrentIndex(3)  # jump to Run tab to show result
        else:
            err = result.get("error", "Unknown error")
            self._set_status(f"Failed: {err}")
            self._log_line(f"✗ Failed: {err}")

    def _do_cancel(self):
        self._cancel_event.set()
        self._set_status("Cancelling…")
        self._log_line("Cancel requested.")

    def _open_output_folder(self):
        path = getattr(self, "_last_output_path", None)
        if not path:
            return
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        try:
            import subprocess, platform
            if platform.system() == "Windows":
                os.startfile(folder)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception:
            pass

    def _load_in_siril(self):
        path = getattr(self, "_last_output_path", None)
        if not path:
            return
        try:
            folder = path if os.path.isdir(path) else os.path.dirname(path)
            siril_api.cmd("cd", folder)
            self._log_line(f"Siril: cd {folder}")
        except Exception as e:
            self._log_line(f"Siril load error: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # LOG
    # ────────────────────────────────────────────────────────────────────────

    def _log_line(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{ts}]  {text}")
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save log", "planet_derotation_log.txt",
            "Text files (*.txt);;All files (*)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log.toPlainText())

    def _set_status(self, text: str):
        self._status_lbl.setText(text)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
