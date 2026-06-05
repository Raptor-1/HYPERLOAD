# zogy_subtract.py - Script #29
# ZOGY Proper Image Subtraction for Siril
# Zackay, Ofek, Gal-Yam (2016), ApJ 830, 27 - arXiv:1601.02655
#
# Place in: C:/Users/Marcell/Desktop/Siril New Scripts/
# Run via:  Siril -> Scripts menu -> zogy_subtract

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")
s.ensure_installed("astroquery")

import os
import sys
import csv
import threading
from datetime import datetime

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import shift as ndimage_shift
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QFrame, QScrollArea, QMenu
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ─────────────────────────────────────────────────────────────────────────────
# SIRIL THEME
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
    font-size: 9pt;
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
    margin-top: 6px;
    padding: 4px;
    font-weight: bold;
    color: {SIRIL_SECTION};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
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
    padding: 4px 8px;
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
QPushButton#warning_btn {{
    background-color: {SIRIL_BG3};
    border-color: {SIRIL_WARNING};
    color: {SIRIL_WARNING};
    text-align: center;
}}
QPushButton#warning_btn:hover {{ background-color: #2d2010; }}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    padding: 2px 4px;
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
    background: {SIRIL_BG2}; width: 14px; border-radius: 7px; margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {SIRIL_BORDER}; border-radius: 7px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {SIRIL_ACCENT2}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
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
QLabel#badge_val {{
    color: {SIRIL_ACCENT};
    font-size: 16pt;
    font-weight: bold;
    padding: 2px;
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


# ─────────────────────────────────────────────────────────────────────────────
# CORE ALGORITHMS  (headless — no PyQt6 dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_fwhm_from_stars(image: np.ndarray, n_stars: int = 20) -> float:
    """Estimate median FWHM from bright isolated stars. Returns FWHM in pixels."""
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    if std <= 0:
        return 3.0

    daofind = DAOStarFinder(
        fwhm=5.0, threshold=8.0 * std,
        sharpness_range=(0.3, 0.9),
        roundness_range=(-0.5, 0.5)
    )
    sources = daofind(image - median)
    if sources is None or len(sources) == 0:
        return 3.0

    sources.sort("peak")
    sources.reverse()
    sources = sources[:n_stars]

    fwhms = []
    h, w  = image.shape

    def gaussian_1d(x, a, mu, sigma):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    for src in sources:
        x0, y0 = int(src["xcentroid"]), int(src["ycentroid"])
        box = 16
        if (y0 - box < 0 or y0 + box >= h or
                x0 - box < 0 or x0 + box >= w):
            continue
        cutout  = image[y0-box:y0+box, x0-box:x0+box].astype(float)
        profile = cutout[box, :]
        xs      = np.arange(len(profile), dtype=float)
        try:
            popt, _ = curve_fit(gaussian_1d, xs, profile,
                                p0=[profile.max(), box, 3.0],
                                maxfev=300)
            fwhm = 2.355 * abs(popt[2])
            if 1.0 < fwhm < box:
                fwhms.append(fwhm)
        except Exception:
            continue

    return float(np.median(fwhms)) if fwhms else 3.0


def estimate_noise(image: np.ndarray) -> float:
    """Estimate background noise standard deviation using sigma-clipped stats."""
    _, _, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    return float(std)


def estimate_psf_gaussian(image: np.ndarray, fwhm_px: float = None) -> np.ndarray:
    """
    Estimate PSF as a 2D Gaussian kernel embedded in a full-size array.
    PSF center placed at (0,0) for Fourier convention.
    """
    h, w = image.shape
    if fwhm_px is None:
        fwhm_px = estimate_fwhm_from_stars(image)

    sigma = fwhm_px / 2.355
    size  = int(6 * sigma) | 1
    half  = size // 2

    y, x   = np.ogrid[-half:half+1, -half:half+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    psf_full = np.zeros((h, w))
    psf_full[:size, :size] = kernel
    psf_full = np.roll(np.roll(psf_full, -half, axis=0), -half, axis=1)
    return psf_full


def zogy_subtract(new_image: np.ndarray,
                   ref_image: np.ndarray,
                   psf_new: np.ndarray = None,
                   psf_ref: np.ndarray = None,
                   sigma_new: float = None,
                   sigma_ref: float = None,
                   flux_new: float = 1.0,
                   flux_ref: float = 1.0,
                   fwhm_new_px: float = None,
                   fwhm_ref_px: float = None) -> dict:
    """
    ZOGY proper image subtraction (Zackay, Ofek, Gal-Yam 2016).

    Returns dict with keys:
        D, S, P_D, sigma_D, fwhm_new, fwhm_ref, sigma_new, sigma_ref
    """
    assert new_image.shape == ref_image.shape, \
        "New and reference images must be the same size"

    N = new_image.astype(np.float64)
    R = ref_image.astype(np.float64)

    if sigma_new is None:
        sigma_new = estimate_noise(N)
    if sigma_ref is None:
        sigma_ref = estimate_noise(R)

    fwhm_new_est = fwhm_new_px
    fwhm_ref_est = fwhm_ref_px

    if psf_new is None:
        if fwhm_new_est is None:
            fwhm_new_est = estimate_fwhm_from_stars(N)
        psf_new = estimate_psf_gaussian(N, fwhm_new_est)
    else:
        fwhm_new_est = fwhm_new_est or 3.0

    if psf_ref is None:
        if fwhm_ref_est is None:
            fwhm_ref_est = estimate_fwhm_from_stars(R)
        psf_ref = estimate_psf_gaussian(R, fwhm_ref_est)
    else:
        fwhm_ref_est = fwhm_ref_est or 3.0

    N_hat   = fft2(N)
    R_hat   = fft2(R)
    P_n_hat = fft2(psf_new)
    P_r_hat = fft2(psf_ref)

    sigma_n2 = sigma_new ** 2
    sigma_r2 = sigma_ref ** 2
    F_n      = flux_new
    F_r      = flux_ref

    P_n_conj = np.conj(P_n_hat)
    P_r_conj = np.conj(P_r_hat)

    denom_fourier = np.sqrt(
        sigma_n2 * F_r**2 * np.abs(P_r_hat)**2 +
        sigma_r2 * F_n**2 * np.abs(P_n_hat)**2 +
        1e-20
    )

    D_hat = (F_r * P_r_conj * N_hat - F_n * P_n_conj * R_hat) / denom_fourier
    D     = np.real(ifft2(D_hat))

    P_D_hat      = (F_r * F_n * P_r_conj * P_n_conj) / denom_fourier
    P_D_hat_norm = P_D_hat / (np.sum(np.abs(P_D_hat)) + 1e-20)

    denom_sq = sigma_n2 * F_r**2 + sigma_r2 * F_n**2
    F_D      = np.sqrt(denom_sq)

    S_hat = F_D * D_hat * np.conj(P_D_hat_norm)
    S     = np.real(ifft2(S_hat))

    _, _, s_std = sigma_clipped_stats(S, sigma=3.0)
    if s_std > 0:
        S = S / s_std

    P_D_real = np.real(ifft2(P_D_hat_norm))
    sigma_D  = F_D * np.sqrt(np.sum(P_D_real**2))

    return {
        "D":         D,
        "S":         S,
        "P_D":       P_D_real,
        "sigma_D":   float(sigma_D),
        "fwhm_new":  float(fwhm_new_est or 3.0),
        "fwhm_ref":  float(fwhm_ref_est or 3.0),
        "sigma_new": float(sigma_new),
        "sigma_ref": float(sigma_ref),
    }


def _deblend_detections(detections: list, min_sep: float) -> list:
    """Remove detections within min_sep pixels of a brighter detection."""
    if not detections:
        return []
    kept = []
    for d in sorted(detections, key=lambda x: x["snr"], reverse=True):
        too_close = any(
            ((d["x"] - k["x"])**2 + (d["y"] - k["y"])**2)**0.5 < min_sep
            for k in kept
        )
        if not too_close:
            kept.append(d)
    return kept


def detect_transients(S: np.ndarray,
                       D: np.ndarray,
                       threshold_sigma: float = 5.0,
                       min_separation_px: float = 5.0,
                       border_margin: int = 20) -> list:
    """
    Detect transient candidates in the significance image.
    Returns list of dicts: {x, y, snr, flux_D, type}
    """
    h, w = S.shape
    detections = []

    dao_kwargs = dict(fwhm=3.0, threshold=threshold_sigma,
                      sharpness_range=(0.0, 2.0),
                      roundness_range=(-1.5, 1.5))

    for sign, label in [(+1, "positive"), (-1, "negative")]:
        src_map = DAOStarFinder(**dao_kwargs)(sign * S)
        if src_map is None:
            continue
        for src in src_map:
            x, y = float(src["xcentroid"]), float(src["ycentroid"])
            if (x < border_margin or x > w - border_margin or
                    y < border_margin or y > h - border_margin):
                continue
            snr  = float(src["peak"])
            ix, iy = int(round(x)), int(round(y))
            flux_d = float(D[iy, ix]) if 0 <= iy < h and 0 <= ix < w else 0.0
            detections.append({
                "x":      round(x, 2),
                "y":      round(y, 2),
                "snr":    round(snr, 2),
                "flux_D": round(flux_d, 4),
                "type":   label,
            })

    detections = _deblend_detections(detections, min_separation_px)
    detections.sort(key=lambda d: d["snr"], reverse=True)
    return detections


def align_images(new_image: np.ndarray,
                  ref_image: np.ndarray,
                  method: str = "phase_correlation") -> tuple:
    """
    Align new_image to ref_image.
    Returns (aligned_new, shift_applied) where shift_applied = (dy, dx).
    """
    if method == "none":
        return new_image, (0.0, 0.0)

    R_hat = fft2(ref_image.astype(float))
    N_hat = fft2(new_image.astype(float))
    cross = R_hat * np.conj(N_hat)
    cross_norm = cross / (np.abs(cross) + 1e-20)
    corr  = np.real(ifft2(cross_norm))
    corr  = fftshift(corr)

    h, w  = corr.shape
    peak  = np.unravel_index(np.argmax(corr), corr.shape)
    dy    = float(peak[0] - h // 2)
    dx    = float(peak[1] - w // 2)

    aligned = ndimage_shift(new_image.astype(float),
                             shift=(dy, dx),
                             mode="constant", cval=0.0)
    return aligned.astype(new_image.dtype), (dy, dx)


def background_match(new_image: np.ndarray,
                      ref_image: np.ndarray) -> np.ndarray:
    """Scale and offset new_image to match reference background level."""
    _, n_med, n_std = sigma_clipped_stats(new_image, sigma=3.0)
    _, r_med, r_std = sigma_clipped_stats(ref_image, sigma=3.0)

    if n_std <= 0 or r_std <= 0:
        return new_image

    scale   = r_std / n_std
    offset  = r_med - n_med * scale
    matched = new_image.astype(float) * scale + offset
    return matched.astype(new_image.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# SKY SURVEY REFERENCE FETCHER
# ─────────────────────────────────────────────────────────────────────────────

SURVEY_OPTIONS = [
    "simg.de  (Hamburg Schmidt — deep red, best for supernovae)",
    "DSS2 Red  (STScI — wide coverage)",
    "DSS2 Blue (STScI)",
    "DSS1      (STScI — oldest, very wide coverage)",
    "2MASS J   (near-infrared)",
    "2MASS H   (near-infrared)",
    "2MASS K   (near-infrared)",
    "SDSS r    (optical, northern sky)",
    "SDSS g    (optical, northern sky)",
    "SDSS i    (optical, northern sky)",
]

# Map display name → SkyView survey identifier
_SKYVIEW_MAP = {
    "DSS2 Red":  "DSS2 Red",
    "DSS2 Blue": "DSS2 Blue",
    "DSS1":      "DSS1",
    "2MASS J":   "2MASS-J",
    "2MASS H":   "2MASS-H",
    "2MASS K":   "2MASS-K",
    "SDSS r":    "SDSSr",
    "SDSS g":    "SDSSg",
    "SDSS i":    "SDSSi",
}


def _get_wcs_center(fits_path: str):
    """
    Extract RA/Dec centre from FITS WCS header.
    Returns (ra_deg, dec_deg, width_deg, height_deg) or None.
    """
    from astropy.wcs import WCS
    from astropy.wcs.utils import pixel_to_skycoord
    with fits.open(fits_path) as hdul:
        hdr  = hdul[0].header
        data = hdul[0].data
        if data.ndim == 3:
            data = data[0]
        h, w = data.shape[-2], data.shape[-1]

    wcs = WCS(hdr, naxis=2)
    if not wcs.has_celestial:
        return None

    centre     = pixel_to_skycoord(w / 2, h / 2, wcs)
    corner_tl  = pixel_to_skycoord(0, 0,     wcs)
    corner_br  = pixel_to_skycoord(w, h,     wcs)

    width_deg  = abs(float(corner_tl.ra.deg  - corner_br.ra.deg))
    height_deg = abs(float(corner_tl.dec.deg - corner_br.dec.deg))

    # clamp to sensible range for surveys
    width_deg  = max(0.05, min(width_deg,  5.0))
    height_deg = max(0.05, min(height_deg, 5.0))

    return (float(centre.ra.deg), float(centre.dec.deg),
            width_deg, height_deg, w, h)


def fetch_simg_de(ra_deg: float, dec_deg: float,
                   width_deg: float, height_deg: float,
                   width_px: int, height_px: int,
                   out_path: str,
                   progress_cb=None) -> str:
    """
    Download a reference image from simg.de (Hamburg Schmidt Observatory).
    simg.de serves FITS via a simple HTTP API.

    Returns path to saved FITS file.
    """
    import urllib.request

    if progress_cb:
        progress_cb("Querying simg.de (Hamburg Schmidt)…")

    # simg.de FITS endpoint (SkyView-compatible gateway)
    # Falls back to direct SkyView DSS2Red if simg.de is unavailable
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ra_str  = coord.ra.to_string(unit=u.hour, sep=":", precision=2)
    dec_str = coord.dec.to_string(sep=":", precision=1, alwayssign=True)

    # simg.de HTTP API
    # GET http://www.simg.de/cgi-bin/simdss.cgi?object=HH:MM:SS+DD:MM:SS&scale=X&size=Y
    # Returns FITS image
    size_arcmin = max(width_deg, height_deg) * 60.0
    url = (
        f"http://www.simg.de/cgi-bin/simdss.cgi"
        f"?object={urllib.request.quote(ra_str + dec_str)}"
        f"&scale=1"
        f"&size={size_arcmin:.1f}"
    )

    try:
        if progress_cb:
            progress_cb(f"Downloading from simg.de  ({size_arcmin:.1f} arcmin)…")
        req = urllib.request.Request(url, headers={"User-Agent": "SirilZOGY/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        # Verify it's a FITS file
        if raw[:6] == b"SIMPLE":
            with open(out_path, "wb") as f:
                f.write(raw)
            if progress_cb:
                progress_cb(f"simg.de reference saved: {out_path}")
            return out_path
        else:
            raise ValueError("simg.de did not return a FITS file")
    except Exception as e:
        if progress_cb:
            progress_cb(f"simg.de unavailable ({e}), falling back to SkyView DSS2 Red…")
        return None


def fetch_skyview_survey(ra_deg: float, dec_deg: float,
                          width_deg: float, height_deg: float,
                          width_px: int, height_px: int,
                          survey: str,
                          out_path: str,
                          progress_cb=None) -> str:
    """
    Download a reference image from NASA SkyView.
    Returns path to saved FITS file.
    """
    from astroquery.skyview import SkyView
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    fov   = max(width_deg, height_deg) * u.deg
    npix  = max(width_px, height_px)
    npix  = min(npix, 2048)  # SkyView cap

    if progress_cb:
        progress_cb(f"Querying SkyView: {survey}  "
                    f"({ra_deg:.4f}, {dec_deg:.4f})  "
                    f"FOV={fov.value:.3f}°  {npix}px…")

    paths = SkyView.get_images(
        position=coord,
        survey=[survey],
        radius=fov / 2,
        pixels=str(npix),
        show_progress=False,
    )
    if not paths:
        raise RuntimeError(f"SkyView returned no data for survey '{survey}'")

    hdul = paths[0]
    hdul.writeto(out_path, overwrite=True)
    if progress_cb:
        progress_cb(f"SkyView reference saved: {out_path}")
    return out_path


def fetch_reference_from_survey(fits_path: str,
                                  survey_display: str,
                                  out_dir: str,
                                  progress_cb=None) -> str:
    """
    High-level: given a science FITS image with WCS, download a matching
    reference image from the chosen sky survey.

    Returns path to downloaded reference FITS.
    Raises RuntimeError on failure.
    """
    wcs_info = _get_wcs_center(fits_path)
    if wcs_info is None:
        raise RuntimeError(
            "No WCS found in the science image. "
            "Plate-solve it in Siril first (Image > Plate Solve), "
            "then retry the sky survey download.")

    ra, dec, w_deg, h_deg, w_px, h_px = wcs_info
    os.makedirs(out_dir, exist_ok=True)

    survey_key = survey_display.split("(")[0].strip()
    safe_name  = survey_key.replace(" ", "_").replace("/", "_")
    out_path   = os.path.join(out_dir, f"reference_{safe_name}.fit")

    if survey_key.startswith("simg.de"):
        result = fetch_simg_de(ra, dec, w_deg, h_deg, w_px, h_px,
                                out_path, progress_cb)
        if result:
            return result
        # simg.de failed → fall back to DSS2 Red
        survey_key = "DSS2 Red"
        out_path   = os.path.join(out_dir, "reference_DSS2_Red_fallback.fit")

    sv_survey = _SKYVIEW_MAP.get(survey_key, "DSS2 Red")
    return fetch_skyview_survey(ra, dec, w_deg, h_deg, w_px, h_px,
                                 sv_survey, out_path, progress_cb)


# ─────────────────────────────────────────────────────────────────────────────
# SURVEY DOWNLOAD WORKER
# ─────────────────────────────────────────────────────────────────────────────

class SurveyDownloadWorker(QThread):
    log_line  = pyqtSignal(str)
    finished  = pyqtSignal(dict)   # {success, path, error}

    def __init__(self, fits_path: str, survey: str, out_dir: str):
        super().__init__()
        self.fits_path = fits_path
        self.survey    = survey
        self.out_dir   = out_dir

    def run(self):
        try:
            path = fetch_reference_from_survey(
                self.fits_path, self.survey, self.out_dir,
                progress_cb=lambda msg: self.log_line.emit(msg)
            )
            self.finished.emit({"success": True, "path": path})
        except Exception as e:
            import traceback
            self.log_line.emit(f"Survey download ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────

class ZOGYWorker(QThread):
    progress     = pyqtSignal(int, int, str)
    log_line     = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    finished     = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_fits(self, path: str):
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            hdr  = hdul[0].header.copy()
        if data.ndim == 3:
            data = (0.299*data[0] + 0.587*data[1] + 0.114*data[2]) \
                   if data.shape[0] == 3 else data[0]
        return data, hdr

    def _save_fits(self, data, path, history=""):
        hdu = fits.PrimaryHDU(data=data.astype(np.float32))
        if history:
            hdu.header["HISTORY"] = history
        hdu.writeto(path, overwrite=True)
        self.log_line.emit(f"Saved: {path}")
        return path

    def _save_detections_csv(self, detections, out_dir):
        path = os.path.join(out_dir, "transient_candidates.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["rank", "x", "y", "snr", "flux_D", "type"])
            writer.writeheader()
            for i, d in enumerate(detections, 1):
                writer.writerow({"rank": i, **d})
        self.log_line.emit(f"Saved {len(detections)} candidates to: {path}")
        return path

    def _abort(self, msg="Cancelled by user"):
        self.finished.emit({"success": False, "error": msg})

    # ── main run ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            cfg = self.config

            # Step 1 — load
            self.progress.emit(1, 8, "Loading images...")
            new_data, new_hdr = self._load_fits(cfg["new_path"])
            ref_data, ref_hdr = self._load_fits(cfg["ref_path"])
            self.log_line.emit(f"New image : {new_data.shape}  {cfg['new_path']}")
            self.log_line.emit(f"Ref image : {ref_data.shape}  {cfg['ref_path']}")

            if new_data.shape != ref_data.shape:
                return self._abort(
                    f"Image sizes don't match: {new_data.shape} vs "
                    f"{ref_data.shape}. Plate-solve + resample both images "
                    f"in Siril first.")

            if self._cancel.is_set():
                return self._abort()

            # Step 2 — background match
            self.progress.emit(2, 8, "Matching backgrounds...")
            if cfg.get("match_background", True):
                new_data = background_match(new_data, ref_data)
                self.log_line.emit("Background levels matched")

            # Step 3 — align
            self.progress.emit(3, 8, "Aligning images...")
            align_method = cfg.get("align_method", "phase_correlation")
            if align_method != "none":
                new_data, shift = align_images(new_data, ref_data, align_method)
                self.log_line.emit(
                    f"Alignment shift: dy={shift[0]:.2f}px  dx={shift[1]:.2f}px")
            else:
                self.log_line.emit("Alignment skipped (assumed pre-aligned)")

            if self._cancel.is_set():
                return self._abort()

            # Step 4 — estimate PSFs
            self.progress.emit(4, 8, "Estimating PSFs...")
            fwhm_new = cfg.get("fwhm_new_px") or None
            fwhm_ref = cfg.get("fwhm_ref_px") or None
            if fwhm_new is None:
                fwhm_new = estimate_fwhm_from_stars(new_data)
                self.log_line.emit(f"New image FWHM (auto): {fwhm_new:.2f} px")
            else:
                self.log_line.emit(f"New image FWHM (manual): {fwhm_new:.2f} px")
            if fwhm_ref is None:
                fwhm_ref = estimate_fwhm_from_stars(ref_data)
                self.log_line.emit(f"Ref image FWHM (auto): {fwhm_ref:.2f} px")
            else:
                self.log_line.emit(f"Ref image FWHM (manual): {fwhm_ref:.2f} px")

            if self._cancel.is_set():
                return self._abort()

            # Step 5 — ZOGY
            self.progress.emit(5, 8, "Running ZOGY subtraction...")
            self.log_line.emit("─" * 60)
            self.log_line.emit("ZOGY proper image subtraction  (arXiv:1601.02655)")
            result = zogy_subtract(
                new_image=new_data.astype(np.float64),
                ref_image=ref_data.astype(np.float64),
                fwhm_new_px=fwhm_new,
                fwhm_ref_px=fwhm_ref,
                sigma_new=cfg.get("sigma_new"),
                sigma_ref=cfg.get("sigma_ref"),
            )
            self.log_line.emit(
                f"σ_D={result['sigma_D']:.4f}  "
                f"FWHM_new={result['fwhm_new']:.2f}px  "
                f"FWHM_ref={result['fwhm_ref']:.2f}px  "
                f"σ_new={result['sigma_new']:.4f}  "
                f"σ_ref={result['sigma_ref']:.4f}")
            self.log_line.emit("─" * 60)

            if self._cancel.is_set():
                return self._abort()

            # Step 6 — detect
            self.progress.emit(6, 8, "Detecting transients...")
            threshold  = cfg.get("detection_threshold", 5.0)
            detections = detect_transients(
                result["S"], result["D"],
                threshold_sigma=threshold,
                min_separation_px=cfg.get("min_separation_px", 5.0),
                border_margin=cfg.get("border_margin", 20)
            )
            n_pos = sum(1 for d in detections if d["type"] == "positive")
            n_neg = len(detections) - n_pos
            self.log_line.emit(
                f"Detected {len(detections)} candidates above {threshold}σ  "
                f"(▲ positive={n_pos}  ▼ negative={n_neg})")
            for i, d in enumerate(detections[:10]):
                arrow = "▲" if d["type"] == "positive" else "▼"
                self.log_line.emit(
                    f"  {arrow} #{i+1}: ({d['x']:.1f}, {d['y']:.1f})  "
                    f"SNR={d['snr']:.1f}  flux_D={d['flux_D']:.4f}")

            if self._cancel.is_set():
                return self._abort()

            # Step 7 — save
            self.progress.emit(7, 8, "Saving outputs...")
            out_dir = cfg.get("output_dir") or os.path.dirname(cfg["new_path"])
            os.makedirs(out_dir, exist_ok=True)

            d_path = csv_path = s_path = ""
            if cfg.get("save_D", True):
                d_path = self._save_fits(
                    result["D"],
                    os.path.join(out_dir, "D_difference.fit"),
                    "ZOGY difference image — zogy_subtract.py")
            if cfg.get("save_S", True):
                s_path = self._save_fits(
                    result["S"],
                    os.path.join(out_dir, "S_significance.fit"),
                    "ZOGY significance (S/N) image — zogy_subtract.py")
            if cfg.get("save_csv", True):
                csv_path = self._save_detections_csv(detections, out_dir)

            # Load D in Siril?
            if cfg.get("load_D_in_siril", False) and d_path:
                try:
                    iface = s.SirilInterface()
                    iface.connect()
                    iface.cmd("load", d_path)
                    iface.disconnect()
                    self.log_line.emit("Loaded D image in Siril")
                except Exception as e:
                    self.log_line.emit(f"Could not load D in Siril: {e}")

            # Step 8 — done
            self.progress.emit(8, 8, "Complete")

            full_result = {
                **result,
                "detections": detections,
                "new_data":   new_data,
                "ref_data":   ref_data,
                "d_path":     d_path,
                "s_path":     s_path,
                "csv_path":   csv_path,
            }
            self.result_ready.emit(full_result)
            self.finished.emit({
                "success":      True,
                "n_detections": len(detections),
                "n_positive":   n_pos,
                "n_negative":   n_neg,
                "d_path":       d_path,
                "s_path":       s_path,
                "csv_path":     csv_path,
                "fwhm_new":     result["fwhm_new"],
                "fwhm_ref":     result["fwhm_ref"],
                "sigma_new":    result["sigma_new"],
                "sigma_ref":    result["sigma_ref"],
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def cancel(self):
        self._cancel.set()


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class ZOGYPreviewCanvas(FigureCanvasQTAgg):
    """4-panel preview: New | Ref | Difference D | Significance S"""

    def __init__(self, parent=None):
        self.fig    = Figure(figsize=(10, 8), facecolor=SIRIL_BG)
        self.ax_new = self.fig.add_subplot(2, 2, 1)
        self.ax_ref = self.fig.add_subplot(2, 2, 2)
        self.ax_D   = self.fig.add_subplot(2, 2, 3)
        self.ax_S   = self.fig.add_subplot(2, 2, 4)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _style_axes(self):
        configs = [
            (self.ax_new, "New image (tonight)",    SIRIL_TEXT_DIM),
            (self.ax_ref, "Reference image (deep)", SIRIL_TEXT_DIM),
            (self.ax_D,   "Difference D",           SIRIL_WARNING),
            (self.ax_S,   "Significance S (σ)",     SIRIL_ACCENT),
        ]
        for ax, title, color in configs:
            ax.set_facecolor(SIRIL_BG)
            ax.set_title(title, color=color, fontsize=8)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color(SIRIL_BORDER)

    @staticmethod
    def _disp(arr, lo=0.5, hi=99.5):
        p_lo = np.percentile(arr, lo)
        p_hi = np.percentile(arr, hi)
        return np.clip((arr.astype(float) - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)

    @staticmethod
    def _ds(arr, max_px=600):
        h, w = arr.shape
        f = max(1, max(h, w) // max_px)
        return arr[::f, ::f], f

    def show_inputs(self, new_data, ref_data):
        for ax, data, title in [
            (self.ax_new, new_data, "New image (tonight)"),
            (self.ax_ref, ref_data, "Reference image (deep)"),
        ]:
            ax.clear()
            thumb, _ = self._ds(self._disp(data))
            ax.imshow(thumb, cmap="gray", origin="lower",
                      aspect="equal", interpolation="nearest")
            ax.set_title(title, color=SIRIL_TEXT_DIM, fontsize=8)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_results(self, D, S, detections, threshold):
        # Difference — diverging colormap centred on 0
        self.ax_D.clear()
        d_scale = np.percentile(np.abs(D), 99) or 1.0
        d_thumb, _ = self._ds(D)
        self.ax_D.imshow(d_thumb, cmap="RdBu_r", origin="lower",
                          aspect="equal", vmin=-d_scale, vmax=d_scale,
                          interpolation="nearest")
        self.ax_D.set_title("Difference D", color=SIRIL_WARNING, fontsize=8)

        # Significance — grayscale + detection markers
        self.ax_S.clear()
        s_thumb, sf = self._ds(np.clip(S, -10, 10))
        self.ax_S.imshow(s_thumb, cmap="gray", origin="lower",
                          aspect="equal",
                          vmin=-threshold, vmax=threshold * 2,
                          interpolation="nearest")

        for d in detections:
            px = d["x"] / sf
            py = d["y"] / sf
            color = SIRIL_SUCCESS if d["type"] == "positive" else SIRIL_ERROR
            self.ax_S.plot(px, py, "o", color=color,
                            markersize=8, markerfacecolor="none",
                            markeredgewidth=1.5)

        n_pos = sum(1 for d in detections if d["type"] == "positive")
        n_neg = len(detections) - n_pos
        legend_els = [
            Line2D([0],[0], marker="o", color=SIRIL_SUCCESS,
                   label=f"▲ Positive ({n_pos})",
                   markerfacecolor="none", markersize=6, linestyle="None"),
            Line2D([0],[0], marker="o", color=SIRIL_ERROR,
                   label=f"▼ Negative ({n_neg})",
                   markerfacecolor="none", markersize=6, linestyle="None"),
        ]
        if detections:
            self.ax_S.legend(handles=legend_els,
                              facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                              labelcolor=SIRIL_TEXT, fontsize=7,
                              loc="upper right")

        self.ax_S.set_title(
            f"Significance S — {len(detections)} candidates > {threshold}σ",
            color=SIRIL_ACCENT, fontsize=8)

        for ax in [self.ax_D, self.ax_S]:
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_color(SIRIL_BORDER)

        self.fig.tight_layout(pad=0.3)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION TABLE WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class DetectionTable(QTableWidget):
    row_double_clicked = pyqtSignal(dict)  # emits detection dict

    COLS = ["Rank", "X (px)", "Y (px)", "SNR (σ)", "Flux_D", "Type", "Notes"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSortingEnabled(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.doubleClicked.connect(self._on_double_click)
        self._detections: list = []

    def load_detections(self, detections: list):
        self._detections = detections
        self.setRowCount(0)
        self.setSortingEnabled(False)
        for i, d in enumerate(detections):
            self.insertRow(i)
            is_pos = d["type"] == "positive"
            color  = QColor(SIRIL_SUCCESS) if is_pos else QColor(SIRIL_ERROR)
            type_text = "▲ New/Brighter" if is_pos else "▼ Gone/Fainter"

            items = [
                str(i + 1),
                f"{d['x']:.1f}",
                f"{d['y']:.1f}",
                f"{d['snr']:.2f}",
                f"{d['flux_D']:.4f}",
                type_text,
                "",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 5:
                    item.setForeground(color)
                self.setItem(i, col, item)

        self.setSortingEnabled(True)

    def _on_double_click(self, index):
        row = index.row()
        if 0 <= row < len(self._detections):
            self.row_double_clicked.emit(self._detections[row])

    def _context_menu(self, pos):
        row = self.rowAt(pos.y())
        if row < 0 or row >= len(self._detections):
            return
        d   = self._detections[row]
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background: {SIRIL_BG3}; color: {SIRIL_TEXT};
                     border: 1px solid {SIRIL_BORDER}; }}
            QMenu::item:selected {{ background: {SIRIL_ACCENT2}; }}
        """)
        copy_act = menu.addAction("Copy coordinates")
        siril_act = menu.addAction("Open position in Siril")
        action = menu.exec(self.viewport().mapToGlobal(pos))
        if action == copy_act:
            QApplication.clipboard().setText(f"{d['x']:.2f}, {d['y']:.2f}")
        elif action == siril_act:
            try:
                iface = s.SirilInterface()
                iface.connect()
                iface.cmd("setstars", f"{int(d['x'])},{int(d['y'])}")
                iface.disconnect()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZOGY Image Subtraction — Siril")
        self.resize(1340, 820)
        self._worker: ZOGYWorker | None = None
        self._survey_worker: SurveyDownloadWorker | None = None
        self._cancel_event = threading.Event()
        self._last_result: dict | None = None
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(4)

        # Title bar
        title_bar = QHBoxLayout()
        lbl_title = QLabel("🔭  ZOGY Image Subtraction  —  Siril")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold;")
        lbl_ver = QLabel("v1.0  |  Zackay-Ofek-Gal-Yam 2016")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        title_bar.addWidget(lbl_title)
        title_bar.addStretch()
        title_bar.addWidget(lbl_ver)
        root.addLayout(title_bar)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFixedHeight(1)
        root.addWidget(sep)

        # Main body — fixed left panel + expanding right, NO splitter
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        # ── Left panel (fixed width, no draggable bar) ───────────────────────
        left = QWidget()
        left.setFixedWidth(320)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(4)
        body.addWidget(left)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_w = QWidget()
        scroll_layout = QVBoxLayout(scroll_w)
        scroll_layout.setSpacing(6)
        # Right margin = 18px so the scrollbar floats outside the content, never overlapping text
        scroll_layout.setContentsMargins(0, 2, 18, 4)
        scroll.setWidget(scroll_w)
        lv.addWidget(scroll, 1)

        # ── Input images ────────────────────────────────────────────────────
        grp_input = QGroupBox("Input images")
        gi = QVBoxLayout(grp_input)
        gi.setSpacing(4)

        gi.addWidget(QLabel("New image (tonight):"))
        row_new = QHBoxLayout()
        self.edit_new = QLineEdit()
        self.edit_new.setPlaceholderText("Tonight's stacked image…")
        btn_new = QPushButton("Browse")
        btn_new.setFixedWidth(56)
        btn_new.clicked.connect(lambda: self._browse(self.edit_new))
        row_new.addWidget(self.edit_new); row_new.addWidget(btn_new)
        gi.addLayout(row_new)

        gi.addWidget(QLabel("Reference image:"))
        row_ref = QHBoxLayout()
        self.edit_ref = QLineEdit()
        self.edit_ref.setPlaceholderText("Manual FITS or use survey below…")
        btn_ref = QPushButton("Browse")
        btn_ref.setFixedWidth(56)
        btn_ref.clicked.connect(lambda: self._browse(self.edit_ref))
        row_ref.addWidget(self.edit_ref); row_ref.addWidget(btn_ref)
        gi.addLayout(row_ref)

        btn_load_preview = QPushButton("Load & preview inputs")
        btn_load_preview.clicked.connect(self._load_and_preview)
        gi.addWidget(btn_load_preview)

        self.lbl_sizes = QLabel("")
        self.lbl_sizes.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        self.lbl_sizes.setWordWrap(True)
        gi.addWidget(self.lbl_sizes)
        self.lbl_size_warn = QLabel("")
        self.lbl_size_warn.setObjectName("warn")
        self.lbl_size_warn.setWordWrap(True)
        self.lbl_size_warn.hide()
        gi.addWidget(self.lbl_size_warn)
        scroll_layout.addWidget(grp_input)

        # ── Sky survey reference ─────────────────────────────────────────────
        grp_survey = QGroupBox("Sky survey reference")
        sv = QVBoxLayout(grp_survey)
        sv.setSpacing(4)

        lbl_sv = QLabel("New image must be plate-solved.\nDownloads reference matching your FOV.")
        lbl_sv.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        lbl_sv.setWordWrap(True)
        sv.addWidget(lbl_sv)

        self.combo_survey = QComboBox()
        self.combo_survey.addItems(SURVEY_OPTIONS)
        sv.addWidget(self.combo_survey)

        self.btn_download_ref = QPushButton("Download reference from survey")
        self.btn_download_ref.setObjectName("warning_btn")
        self.btn_download_ref.clicked.connect(self._download_reference)
        sv.addWidget(self.btn_download_ref)

        self.lbl_survey_status = QLabel("")
        self.lbl_survey_status.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        self.lbl_survey_status.setWordWrap(True)
        sv.addWidget(self.lbl_survey_status)
        scroll_layout.addWidget(grp_survey)

        # ── Pre-processing ───────────────────────────────────────────────────
        grp_pre = QGroupBox("Pre-processing")
        gp = QVBoxLayout(grp_pre)
        gp.setSpacing(4)

        gp.addWidget(QLabel("Alignment:"))
        self.combo_align = QComboBox()
        self.combo_align.addItems([
            "Phase correlation (sub-pixel)",
            "Cross-correlation (integer px)",
            "None (pre-aligned)",
        ])
        gp.addWidget(self.combo_align)

        self.chk_bg_match = QCheckBox("Match backgrounds")
        self.chk_bg_match.setChecked(True)
        gp.addWidget(self.chk_bg_match)
        scroll_layout.addWidget(grp_pre)

        # ── PSF settings ─────────────────────────────────────────────────────
        grp_psf = QGroupBox("PSF settings")
        gpsf = QVBoxLayout(grp_psf)
        gpsf.setSpacing(4)

        row_fwhm = QHBoxLayout()
        lbl_fn = QLabel("New FWHM (px):")
        lbl_fn.setFixedWidth(110)
        self.spin_fwhm_new = QDoubleSpinBox()
        self.spin_fwhm_new.setRange(0.0, 20.0)
        self.spin_fwhm_new.setValue(0.0)
        self.spin_fwhm_new.setSingleStep(0.5)
        self.spin_fwhm_new.setDecimals(1)
        row_fwhm.addWidget(lbl_fn); row_fwhm.addWidget(self.spin_fwhm_new)
        gpsf.addLayout(row_fwhm)

        row_fwhm2 = QHBoxLayout()
        lbl_fr = QLabel("Ref FWHM (px):")
        lbl_fr.setFixedWidth(110)
        self.spin_fwhm_ref = QDoubleSpinBox()
        self.spin_fwhm_ref.setRange(0.0, 20.0)
        self.spin_fwhm_ref.setValue(0.0)
        self.spin_fwhm_ref.setSingleStep(0.5)
        self.spin_fwhm_ref.setDecimals(1)
        row_fwhm2.addWidget(lbl_fr); row_fwhm2.addWidget(self.spin_fwhm_ref)
        gpsf.addLayout(row_fwhm2)

        lbl_auto = QLabel("0 = auto-estimate from stars")
        lbl_auto.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        gpsf.addWidget(lbl_auto)

        gpsf.addWidget(QLabel("PSF type:"))
        self.combo_psf_type = QComboBox()
        self.combo_psf_type.addItems(["Gaussian (fast)", "Empirical (future)"])
        gpsf.addWidget(self.combo_psf_type)
        scroll_layout.addWidget(grp_psf)

        # ── Detection ────────────────────────────────────────────────────────
        grp_det = QGroupBox("Detection")
        gd = QVBoxLayout(grp_det)
        gd.setSpacing(4)

        gd.addWidget(QLabel("Detection threshold:"))
        row_thresh = QHBoxLayout()
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(30, 100)
        self.slider_thresh.setValue(50)
        self.slider_thresh.valueChanged.connect(self._on_thresh_change)
        self.lbl_thresh_val = QLabel("5.0 σ")
        self.lbl_thresh_val.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-weight:bold;")
        self.lbl_thresh_val.setFixedWidth(42)
        row_thresh.addWidget(self.slider_thresh)
        row_thresh.addWidget(self.lbl_thresh_val)
        gd.addLayout(row_thresh)
        lbl_thresh_hint = QLabel("5σ recommended — lower = more false positives")
        lbl_thresh_hint.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        lbl_thresh_hint.setWordWrap(True)
        gd.addWidget(lbl_thresh_hint)

        row_sep = QHBoxLayout()
        lbl_sep = QLabel("Min separation (px):")
        lbl_sep.setFixedWidth(140)
        self.spin_min_sep = QSpinBox()
        self.spin_min_sep.setRange(1, 30)
        self.spin_min_sep.setValue(5)
        row_sep.addWidget(lbl_sep); row_sep.addWidget(self.spin_min_sep)
        gd.addLayout(row_sep)

        row_brd = QHBoxLayout()
        lbl_brd = QLabel("Border margin (px):")
        lbl_brd.setFixedWidth(140)
        self.spin_border = QSpinBox()
        self.spin_border.setRange(5, 100)
        self.spin_border.setValue(20)
        row_brd.addWidget(lbl_brd); row_brd.addWidget(self.spin_border)
        gd.addLayout(row_brd)
        scroll_layout.addWidget(grp_det)

        # ── Output ───────────────────────────────────────────────────────────
        grp_out = QGroupBox("Output")
        go = QVBoxLayout(grp_out)
        go.setSpacing(4)

        go.addWidget(QLabel("Output folder:"))
        row_out = QHBoxLayout()
        self.edit_outdir = QLineEdit()
        self.edit_outdir.setPlaceholderText("Same folder as new image")
        btn_outdir = QPushButton("Browse")
        btn_outdir.setFixedWidth(56)
        btn_outdir.clicked.connect(lambda: self._browse_dir(self.edit_outdir))
        row_out.addWidget(self.edit_outdir); row_out.addWidget(btn_outdir)
        go.addLayout(row_out)

        self.chk_save_D   = QCheckBox("Save D  (difference image)"); self.chk_save_D.setChecked(True)
        self.chk_save_S   = QCheckBox("Save S  (significance image)"); self.chk_save_S.setChecked(True)
        self.chk_save_csv = QCheckBox("Save candidates CSV"); self.chk_save_csv.setChecked(True)
        self.chk_load_D   = QCheckBox("Load D in Siril after run")
        go.addWidget(self.chk_save_D)
        go.addWidget(self.chk_save_S)
        go.addWidget(self.chk_save_csv)
        go.addWidget(self.chk_load_D)
        scroll_layout.addWidget(grp_out)
        scroll_layout.addStretch()

        # Run / Cancel buttons
        self.btn_run = QPushButton("▶  Run ZOGY subtraction")
        self.btn_run.setObjectName("primary")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self._run)
        lv.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setMinimumHeight(28)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.hide()
        lv.addWidget(self.btn_cancel)

        self.progress = QProgressBar()
        self.progress.setRange(0, 8)
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        self.progress.hide()
        lv.addWidget(self.progress)

        # ── Right panel (takes all remaining space) ──────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        body.addWidget(right, 1)

        self.tabs = QTabWidget()
        rv.addWidget(self.tabs)

        # Tab 1 — Preview
        tab_preview = QWidget()
        tv = QVBoxLayout(tab_preview)
        tv.setContentsMargins(2, 2, 2, 2)
        self.canvas = ZOGYPreviewCanvas()
        tv.addWidget(self.canvas)
        self.tabs.addTab(tab_preview, "🖼 Preview")

        # Tab 2 — Candidates
        tab_cands = QWidget()
        cv = QVBoxLayout(tab_cands)
        cv.setContentsMargins(6, 6, 6, 6)
        cv.setSpacing(6)

        # Badge row
        badge_row = QHBoxLayout()
        self.badge_total = self._badge("Total", "0")
        self.badge_pos   = self._badge("▲ Positive", "0", SIRIL_SUCCESS)
        self.badge_neg   = self._badge("▼ Negative", "0", SIRIL_ERROR)
        badge_row.addWidget(self.badge_total)
        badge_row.addWidget(self.badge_pos)
        badge_row.addWidget(self.badge_neg)
        badge_row.addStretch()
        cv.addLayout(badge_row)

        self.det_table = DetectionTable()
        cv.addWidget(self.det_table, 1)

        btn_row = QHBoxLayout()
        btn_export = QPushButton("Export CSV")
        btn_export.setObjectName("export")
        btn_export.clicked.connect(self._export_csv)
        btn_copy = QPushButton("Copy selected")
        btn_copy.clicked.connect(self._copy_selected)
        btn_open_folder = QPushButton("Open output folder")
        btn_open_folder.clicked.connect(self._open_folder)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_copy)
        btn_row.addWidget(btn_open_folder)
        btn_row.addStretch()
        cv.addLayout(btn_row)
        self.tabs.addTab(tab_cands, "📋 Candidates")

        # Tab 3 — Log
        tab_log = QWidget()
        logv = QVBoxLayout(tab_log)
        logv.setContentsMargins(4, 4, 4, 4)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        logv.addWidget(self.log)
        self.tabs.addTab(tab_log, "📋 Log")

        # Status bar
        self.status = QLabel(
            "Load new and reference images, configure parameters, then run.")
        self.status.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; padding:4px 8px; font-size:9pt;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.status)

        # Log startup
        self._log(f"ZOGY Image Subtraction  —  Siril Python Tools")
        self._log(f"Reference: Zackay, Ofek, Gal-Yam (2016), ApJ 830, 27")
        self._log(f"arXiv:1601.02655")
        self._log("─" * 60)
        self._log("Ready. Load your images and run ZOGY subtraction.")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _dim(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("dim")
        return lbl

    def _badge(self, label: str, value: str,
               color: str = SIRIL_ACCENT) -> QWidget:
        w  = QWidget()
        w.setStyleSheet(
            f"background:{SIRIL_BG3}; border:1px solid {SIRIL_BORDER}; "
            f"border-radius:4px; padding:4px 10px;")
        hb = QHBoxLayout(w)
        hb.setContentsMargins(6, 4, 6, 4)
        hb.setSpacing(6)
        lbl_l = QLabel(label)
        lbl_l.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        lbl_v = QLabel(value)
        lbl_v.setStyleSheet(
            f"color:{color}; font-size:14pt; font-weight:bold;")
        lbl_v.setObjectName("badge_val_" + label.replace(" ", "_"))
        hb.addWidget(lbl_l)
        hb.addWidget(lbl_v)
        w._val_label = lbl_v
        return w

    def _update_badge(self, badge, value: str):
        badge._val_label.setText(value)

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self.status.setText(msg)
        self.status.setStyleSheet(
            f"color:{color}; padding:4px 8px; font-size:9pt;")

    def _browse(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FITS image", "",
            "FITS files (*.fit *.fits *.fts);;All files (*)")
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            edit.setText(path)

    def _on_thresh_change(self, val: int):
        sigma = val / 10.0
        self.lbl_thresh_val.setText(f"{sigma:.1f} σ")

    # ── actions ──────────────────────────────────────────────────────────────

    def _download_reference(self):
        new_p = self.edit_new.text().strip()
        if not new_p or not os.path.isfile(new_p):
            QMessageBox.warning(self, "Missing file",
                "Select the new (science) image first — it needs WCS for "
                "sky survey alignment.")
            return

        out_dir = self.edit_outdir.text().strip() or os.path.dirname(new_p)
        survey  = self.combo_survey.currentText()

        self.btn_download_ref.setEnabled(False)
        self.lbl_survey_status.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-size:9pt;")
        self.lbl_survey_status.setText("Downloading…")
        self.tabs.setCurrentIndex(2)
        self._log(f"Downloading reference: {survey}")

        self._survey_worker = SurveyDownloadWorker(new_p, survey, out_dir)
        self._survey_worker.log_line.connect(self._log)
        self._survey_worker.finished.connect(self._on_survey_done)
        self._survey_worker.start()

    def _on_survey_done(self, result: dict):
        self.btn_download_ref.setEnabled(True)
        if result.get("success"):
            path = result["path"]
            self.edit_ref.setText(path)
            self.lbl_survey_status.setStyleSheet(
                f"color:{SIRIL_SUCCESS}; font-size:9pt;")
            self.lbl_survey_status.setText(f"Saved: {os.path.basename(path)}")
            self._log(f"Reference downloaded and set: {path}")
            self._set_status(f"Reference downloaded: {os.path.basename(path)}",
                             SIRIL_SUCCESS)
        else:
            err = result.get("error", "Unknown error")
            self.lbl_survey_status.setStyleSheet(
                f"color:{SIRIL_ERROR}; font-size:9pt;")
            self.lbl_survey_status.setText(f"Failed: {err}")
            self._set_status(f"Survey download failed: {err}", SIRIL_ERROR)

    def _load_and_preview(self):
        new_p = self.edit_new.text().strip()
        ref_p = self.edit_ref.text().strip()
        if not new_p or not os.path.isfile(new_p):
            QMessageBox.warning(self, "Missing file", "Please select a valid new image.")
            return
        if not ref_p or not os.path.isfile(ref_p):
            QMessageBox.warning(self, "Missing file", "Please select a valid reference image.")
            return
        try:
            with fits.open(new_p) as h:
                nd = h[0].data
                if nd.ndim == 3:
                    nd = nd[0] if nd.shape[0] != 3 else \
                         0.299*nd[0] + 0.587*nd[1] + 0.114*nd[2]
            with fits.open(ref_p) as h:
                rd = h[0].data
                if rd.ndim == 3:
                    rd = rd[0] if rd.shape[0] != 3 else \
                         0.299*rd[0] + 0.587*rd[1] + 0.114*rd[2]
            self.lbl_sizes.setText(
                f"New: {nd.shape[1]}×{nd.shape[0]}  |  "
                f"Ref: {rd.shape[1]}×{rd.shape[0]}")
            if nd.shape != rd.shape:
                self.lbl_size_warn.setText(
                    "⚠  Sizes differ — resample/align in Siril before ZOGY")
                self.lbl_size_warn.show()
            else:
                self.lbl_size_warn.hide()
            self.canvas.show_inputs(nd.astype(np.float32),
                                     rd.astype(np.float32))
            self.tabs.setCurrentIndex(0)
            self._log(f"Previewed inputs: "
                      f"new={nd.shape}  ref={rd.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _get_config(self) -> dict:
        align_map = {
            "Phase correlation (sub-pixel)":   "phase_correlation",
            "Cross-correlation (integer px)":  "cross_correlation",
            "None (pre-aligned)":              "none",
        }
        return {
            "new_path":            self.edit_new.text().strip(),
            "ref_path":            self.edit_ref.text().strip(),
            "align_method":        align_map[self.combo_align.currentText()],
            "match_background":    self.chk_bg_match.isChecked(),
            "fwhm_new_px":         self.spin_fwhm_new.value() or None,
            "fwhm_ref_px":         self.spin_fwhm_ref.value() or None,
            "detection_threshold": self.slider_thresh.value() / 10.0,
            "min_separation_px":   float(self.spin_min_sep.value()),
            "border_margin":       self.spin_border.value(),
            "output_dir":          self.edit_outdir.text().strip() or None,
            "save_D":              self.chk_save_D.isChecked(),
            "save_S":              self.chk_save_S.isChecked(),
            "save_csv":            self.chk_save_csv.isChecked(),
            "load_D_in_siril":     self.chk_load_D.isChecked(),
        }

    def _run(self):
        cfg = self._get_config()
        if not cfg["new_path"] or not os.path.isfile(cfg["new_path"]):
            QMessageBox.warning(self, "Missing file", "Please select a valid new image.")
            return
        if not cfg["ref_path"] or not os.path.isfile(cfg["ref_path"]):
            QMessageBox.warning(self, "Missing file", "Please select a valid reference image.")
            return

        self._cancel_event.clear()
        self.btn_run.setEnabled(False)
        self.btn_cancel.show()
        self.progress.setValue(0)
        self.progress.show()
        self._set_status("Running ZOGY subtraction…", SIRIL_ACCENT)
        self.tabs.setCurrentIndex(2)  # log tab
        self._log("=" * 60)
        self._log(f"Starting ZOGY run  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

        self._worker = ZOGYWorker(cfg, self._cancel_event)
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self._log("Cancellation requested…")
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _on_progress(self, step: int, total: int, msg: str):
        self.progress.setRange(0, total)
        self.progress.setValue(step)
        self._set_status(f"Step {step}/{total}: {msg}", SIRIL_ACCENT)

    def _on_result_ready(self, result: dict):
        self._last_result = result
        # Update preview
        self.canvas.show_inputs(result["new_data"], result["ref_data"])
        self.canvas.show_results(
            result["D"], result["S"],
            result["detections"],
            self.slider_thresh.value() / 10.0
        )
        # Update table
        dets = result["detections"]
        self.det_table.load_detections(dets)
        n_pos = sum(1 for d in dets if d["type"] == "positive")
        n_neg = len(dets) - n_pos
        self._update_badge(self.badge_total, str(len(dets)))
        self._update_badge(self.badge_pos,   str(n_pos))
        self._update_badge(self.badge_neg,   str(n_neg))

    def _on_finished(self, result: dict):
        self.btn_run.setEnabled(True)
        self.btn_cancel.hide()
        self.progress.hide()

        if result.get("success"):
            n   = result["n_detections"]
            msg = (f"✓  {n} candidates detected  "
                   f"(▲ {result['n_positive']}  ▼ {result['n_negative']})  |  "
                   f"D saved: {result['d_path']}")
            self._set_status(msg, SIRIL_SUCCESS)
            self._log(f"Done. {n} candidates. "
                      f"FWHM_new={result['fwhm_new']:.2f}px  "
                      f"FWHM_ref={result['fwhm_ref']:.2f}px")
            self.tabs.setCurrentIndex(0)  # jump to preview
        else:
            err = result.get("error", "Unknown error")
            self._set_status(f"✗  Error: {err}", SIRIL_ERROR)
            self._log(f"FAILED: {err}")

    # ── table actions ─────────────────────────────────────────────────────────

    def _export_csv(self):
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "transient_candidates.csv",
            "CSV files (*.csv)")
        if not path:
            return
        dets = self._last_result.get("detections", [])
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["rank", "x", "y", "snr", "flux_D", "type"])
            writer.writeheader()
            for i, d in enumerate(dets, 1):
                writer.writerow({"rank": i, **d})
        self._log(f"Exported {len(dets)} candidates to {path}")

    def _copy_selected(self):
        rows = set(idx.row() for idx in self.det_table.selectedIndexes())
        if self._last_result is None:
            return
        dets = self._last_result.get("detections", [])
        lines = []
        for r in sorted(rows):
            if r < len(dets):
                d = dets[r]
                lines.append(
                    f"{d['x']:.2f}, {d['y']:.2f}, SNR={d['snr']:.2f}, "
                    f"type={d['type']}")
        QApplication.clipboard().setText("\n".join(lines))

    def _open_folder(self):
        if self._last_result is None:
            return
        path = self._last_result.get("d_path") or self._last_result.get("csv_path")
        if path:
            folder = os.path.dirname(path)
            import subprocess
            try:
                subprocess.Popen(["explorer", folder])
            except Exception:
                pass


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
