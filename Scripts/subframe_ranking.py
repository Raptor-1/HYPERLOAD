#!/usr/bin/env python3
"""
subframe_ranking.py  —  Subframe Auto-Ranking  (Script #10)
Siril Python Tools  —  PixInsight SubframeSelector equivalent

Score and rank every FITS frame on 5 quality metrics:
  FWHM · Eccentricity · SNR · Background · Star Count
Then export only the approved frames for stacking.

Place in:  C:\\Users\\Marcell\\Desktop\\Siril New Scripts\\
Run via:   Siril → Scripts menu → subframe_ranking
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

import os, sys, csv, glob, shutil, json, threading
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits as astropy_fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QTabWidget, QComboBox,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QDialog, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

try:
    from seeing_estimator import estimate_fwhm_arcsec
    HAS_SEEING = True
except ImportError:
    HAS_SEEING = False

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
QPushButton#primary:disabled {{
    background-color: {SIRIL_BG3};
    color: {SIRIL_TEXT_DIM};
    border-color: {SIRIL_BORDER};
}}
QPushButton#danger:hover {{
    background-color: #8b2020;
    border-color: {SIRIL_ERROR};
}}
QPushButton#danger:disabled {{
    color: {SIRIL_TEXT_DIM};
    border-color: {SIRIL_BORDER};
}}
QPushButton#export {{
    background-color: {SIRIL_BG3};
    border-color: {SIRIL_SUCCESS};
    color: {SIRIL_SUCCESS};
    text-align: center;
    font-weight: bold;
}}
QPushButton#export:hover {{ background-color: #1a3d2a; }}
QPushButton#export:disabled {{
    color: {SIRIL_TEXT_DIM};
    border-color: {SIRIL_BORDER};
}}
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
QLabel#score  {{
    color: {SIRIL_ACCENT};
    font-size: 22pt;
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

# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def measure_frame_fwhm(image: np.ndarray,
                       n_stars: int = 20,
                       threshold_sigma: float = 8.0) -> tuple:
    """
    Measure median FWHM and eccentricity from stars in a frame.
    Returns (fwhm_px, eccentricity, n_measured).
    """
    h, w = image.shape
    try:
        _, med, std = sigma_clipped_stats(image, sigma=3.0)
    except Exception:
        return 0.0, 0.0, 0
    if std <= 0:
        return 0.0, 0.0, 0

    daofind = DAOStarFinder(
        fwhm=5.0, threshold=threshold_sigma * std,
        sharplo=0.2, sharphi=0.9,
        roundlo=-0.7, roundhi=0.7
    )
    try:
        sources = daofind(image - med)
    except Exception:
        return 0.0, 0.0, 0
    if sources is None or len(sources) == 0:
        return 0.0, 0.0, 0

    sources.sort("peak")
    sources.reverse()
    sat_limit = 0.9 * float(np.percentile(image, 99.99))
    fwhms, eccs = [], []
    box = 20

    def gaussian_2d(xy, amp, x0, y0, sx, sy, theta):
        x_arr, y_arr = xy
        ct, st = np.cos(theta), np.sin(theta)
        a = ct**2/(2*sx**2) + st**2/(2*sy**2)
        b = -np.sin(2*theta)/(4*sx**2) + np.sin(2*theta)/(4*sy**2)
        c = st**2/(2*sx**2) + ct**2/(2*sy**2)
        dx, dy = x_arr - x0, y_arr - y0
        return amp * np.exp(-(a*dx**2 + 2*b*dx*dy + c*dy**2))

    for src in sources[:n_stars]:
        x0 = int(round(float(src["xcentroid"])))
        y0 = int(round(float(src["ycentroid"])))
        if x0 < box or x0 >= w - box or y0 < box or y0 >= h - box:
            continue
        if float(src["peak"]) + med > sat_limit:
            continue

        cut = image[y0-box:y0+box, x0-box:x0+box].astype(np.float64)
        bg_corners = np.concatenate([
            cut[:3, :3].ravel(), cut[:3, -3:].ravel(),
            cut[-3:, :3].ravel(), cut[-3:, -3:].ravel()
        ])
        bg = float(np.median(bg_corners))
        cut -= bg
        cut = np.clip(cut, 0, None)
        if cut.max() <= 0:
            continue

        size = box * 2
        yg, xg = np.mgrid[0:size, 0:size]
        try:
            popt, _ = curve_fit(
                gaussian_2d,
                (xg.ravel(), yg.ravel()),
                cut.ravel(),
                p0=[cut.max(), box, box, 3.0, 3.0, 0.0],
                bounds=([0, 0, 0, 0.5, 0.5, -np.pi/2],
                        [cut.max()*2, size, size, box, box, np.pi/2]),
                maxfev=600
            )
        except Exception:
            continue

        _, _, _, sx, sy, _ = popt
        fwhm_x = 2.355 * abs(sx)
        fwhm_y = 2.355 * abs(sy)
        if fwhm_x < 0.8 or fwhm_y < 0.8:
            continue
        if fwhm_x > box or fwhm_y > box:
            continue

        fwhm_mean = (fwhm_x + fwhm_y) / 2.0
        fwhm_min  = min(fwhm_x, fwhm_y)
        fwhm_max  = max(fwhm_x, fwhm_y)
        ecc = float(np.sqrt(max(0.0, 1.0 - (fwhm_min / fwhm_max)**2)))
        fwhms.append(fwhm_mean)
        eccs.append(ecc)

    if not fwhms:
        return 0.0, 0.0, 0

    return float(np.median(fwhms)), float(np.median(eccs)), len(fwhms)


def measure_frame_snr(image: np.ndarray,
                      star_fwhm_px: float = 5.0,
                      n_stars: int = 20) -> float:
    """Estimate frame SNR as median star peak / background noise."""
    try:
        _, med, std = sigma_clipped_stats(image, sigma=3.0)
    except Exception:
        return 0.0
    if std <= 0 or med <= 0:
        return 0.0

    daofind = DAOStarFinder(
        fwhm=max(star_fwhm_px, 3.0),
        threshold=5.0 * std,
        sharplo=0.2, sharphi=0.9
    )
    try:
        sources = daofind(image - med)
    except Exception:
        return 0.0
    if sources is None or len(sources) == 0:
        return 0.0

    sources.sort("peak")
    sources.reverse()
    peaks = [float(src["peak"]) for src in sources[:n_stars]]
    if not peaks:
        return 0.0
    return float(np.median(peaks)) / max(std, 1e-10)


def measure_background(image: np.ndarray) -> float:
    """Sigma-clipped sky background. Lower = darker sky = better."""
    try:
        _, med, _ = sigma_clipped_stats(image, sigma=3.0)
    except Exception:
        med = float(np.median(image))
    return float(med)


def measure_star_count(image: np.ndarray,
                       threshold_sigma: float = 5.0) -> int:
    """Count detectable stars. Higher = clearer sky."""
    try:
        _, med, std = sigma_clipped_stats(image, sigma=3.0)
    except Exception:
        return 0
    if std <= 0:
        return 0
    daofind = DAOStarFinder(
        fwhm=5.0, threshold=threshold_sigma * std,
        sharplo=0.2, sharphi=0.9
    )
    try:
        sources = daofind(image - med)
    except Exception:
        return 0
    return len(sources) if sources is not None else 0


def compute_population_stats(all_metrics: list) -> dict:
    """Compute mean and std for each metric across all frames."""
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    stats = {}
    for key in keys:
        vals = [m[key] for m in all_metrics
                if m.get(key) is not None and np.isfinite(m.get(key, 0))]
        if vals:
            stats[key] = (float(np.mean(vals)), float(np.std(vals)))
        else:
            stats[key] = (0.0, 1.0)
    return stats


def score_frame(metrics: dict, population: dict, weights: dict) -> float:
    """Compute composite quality score 0-100. Higher = better frame."""
    score = 0.0
    total_weight = 0.0
    for metric, weight in weights.items():
        if weight <= 0:
            continue
        val = metrics.get(metric, 0.0)
        pop_mean, pop_std = population.get(metric, (val, 1.0))
        z = 0.0 if pop_std < 1e-10 else (val - pop_mean) / pop_std

        lower_is_better = metric in ("fwhm", "eccentricity", "background")
        component = (50.0 - z * 10.0) if lower_is_better else (50.0 + z * 10.0)
        component = float(np.clip(component, 0.0, 100.0))
        score += weight * component
        total_weight += weight

    if total_weight > 0:
        score /= total_weight
    return float(np.clip(score, 0.0, 100.0))


def find_elbow_threshold(scores: list) -> float:
    """
    Suggest rejection threshold via elbow method.
    Finds score where the quality distribution transitions from
    good cluster to poor-quality tail.
    """
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n < 3:
        return 70.0
    x = np.arange(n, dtype=float)
    y = np.array(sorted_scores)
    dx = float(n - 1)
    dy = y[-1] - y[0]
    if abs(dy) < 1e-10 and abs(dx) < 1e-10:
        return 70.0
    dist = np.abs(dy*x - dx*y + dx*y[0] - dy*0) / (np.sqrt(dy**2 + dx**2) + 1e-10)
    elbow_idx = int(np.argmax(dist))
    return float(sorted_scores[elbow_idx])


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT / EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def apply_threshold_and_export(frame_results: list,
                               threshold: float,
                               output_mode: str,
                               output_dir: str,
                               export_csv: bool = True,
                               log_callback=None) -> dict:
    """
    Apply score threshold and export approved frames.
    output_mode: "copy" | "list" | "move_rejected" | "weighted"
    """
    os.makedirs(output_dir, exist_ok=True)
    approved = [f for f in frame_results if f["score"] >= threshold]
    rejected = [f for f in frame_results if f["score"] < threshold]

    def log(msg):
        if log_callback:
            log_callback(msg)

    log(f"Threshold={threshold:.1f}  Approved={len(approved)}  Rejected={len(rejected)}")
    output_paths = []

    if output_mode == "copy":
        for frame in approved:
            dest = os.path.join(output_dir, os.path.basename(frame["path"]))
            try:
                shutil.copy2(frame["path"], dest)
                output_paths.append(dest)
            except Exception as e:
                log(f"  Copy error {frame['filename']}: {e}")
        log(f"Copied {len(output_paths)} frames → {output_dir}")

    elif output_mode == "list":
        list_path = os.path.join(output_dir, "approved_frames.txt")
        with open(list_path, "w") as f:
            for frame in approved:
                f.write(frame["path"] + "\n")
        output_paths.append(list_path)
        log(f"List saved: {list_path}")

    elif output_mode == "move_rejected":
        rej_dir = os.path.join(output_dir, "rejected")
        os.makedirs(rej_dir, exist_ok=True)
        moved = 0
        for frame in rejected:
            dest = os.path.join(rej_dir, os.path.basename(frame["path"]))
            try:
                shutil.move(frame["path"], dest)
                moved += 1
            except Exception as e:
                log(f"  Move error {frame['filename']}: {e}")
        log(f"Moved {moved} rejected frames → {rej_dir}")

    elif output_mode == "weighted":
        list_path = os.path.join(output_dir, "weighted_stack.txt")
        max_score = max((f["score"] for f in approved), default=100.0)
        with open(list_path, "w") as f:
            for frame in approved:
                weight = frame["score"] / max(max_score, 1.0)
                f.write(f"{weight:.4f} {frame['path']}\n")
        output_paths.append(list_path)
        log(f"Weighted list: {list_path}")

    if export_csv:
        csv_path = os.path.join(output_dir, "frame_ranking.csv")
        _write_ranking_csv(frame_results, threshold, csv_path)
        log(f"Ranking CSV: {csv_path}")
        output_paths.append(csv_path)

    return {
        "n_approved":   len(approved),
        "n_rejected":   len(rejected),
        "output_paths": output_paths,
        "output_dir":   output_dir,
    }


def _write_ranking_csv(frame_results: list, threshold: float, csv_path: str):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "score", "approved", "filename",
            "fwhm", "eccentricity", "snr", "background", "star_count", "path"
        ])
        for fr in frame_results:
            m = fr.get("metrics", {})
            writer.writerow([
                fr.get("rank", ""),
                round(fr.get("score", 0), 2),
                "YES" if fr.get("score", 0) >= threshold else "NO",
                fr.get("filename", ""),
                round(m.get("fwhm", 0), 3),
                round(m.get("eccentricity", 0), 3),
                round(m.get("snr", 0), 1),
                round(m.get("background", 0), 1),
                int(m.get("star_count", 0)),
                fr.get("path", ""),
            ])


# ─────────────────────────────────────────────────────────────────────────────
# WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────

class RankingWorker(QThread):
    progress    = pyqtSignal(int, int, str)
    log_line    = pyqtSignal(str)
    all_scored  = pyqtSignal(list)
    finished    = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg    = self.config
            files  = cfg["fits_files"]
            n      = len(files)
            weights = cfg.get("weights", {
                "fwhm": 0.40, "eccentricity": 0.20, "snr": 0.20,
                "background": 0.10, "star_count": 0.10,
            })
            pixel_scale = cfg.get("pixel_scale_arcsec")

            self.log_line.emit(
                f"── Ranking {n} frames on 5 quality metrics ──")
            self.log_line.emit(
                f"Weights: FWHM={weights['fwhm']*100:.0f}%  "
                f"Ecc={weights['eccentricity']*100:.0f}%  "
                f"SNR={weights['snr']*100:.0f}%  "
                f"BG={weights['background']*100:.0f}%  "
                f"Stars={weights['star_count']*100:.0f}%")
            if pixel_scale:
                self.log_line.emit(
                    f"Pixel scale: {pixel_scale:.3f} arcsec/px")

            all_metrics   = []
            frame_results = []

            for i, fits_path in enumerate(files):
                if self._cancel.is_set():
                    self._abort(); return

                fname = os.path.basename(fits_path)
                self.progress.emit(i + 1, n, f"Measuring {fname}")

                try:
                    with astropy_fits.open(fits_path) as hdul:
                        data = hdul[0].data.astype(np.float32)
                        hdr  = hdul[0].header

                    if data.ndim == 3:
                        if data.shape[0] == 3:
                            lum = 0.299*data[0] + 0.587*data[1] + 0.114*data[2]
                        else:
                            lum = data[0]
                    else:
                        lum = data

                    exptime  = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 1.0)))
                    date_obs = str(hdr.get("DATE-OBS", ""))

                    fwhm_px, ecc, n_meas = measure_frame_fwhm(lum)
                    fwhm_val = (fwhm_px * pixel_scale
                                if pixel_scale and fwhm_px > 0 else fwhm_px)
                    snr   = measure_frame_snr(lum, star_fwhm_px=fwhm_px)
                    bg    = measure_background(lum)
                    n_det = measure_star_count(lum)

                    metrics = {
                        "fwhm":         fwhm_val if fwhm_val > 0 else fwhm_px,
                        "eccentricity": ecc,
                        "snr":          snr,
                        "background":   bg,
                        "star_count":   float(n_det),
                    }
                    all_metrics.append(metrics)

                    fwhm_disp = (f"{fwhm_val:.2f}\""
                                 if pixel_scale and fwhm_val > 0
                                 else f"{fwhm_px:.2f}px")
                    if i % 5 == 0 or fwhm_px > 0:
                        self.log_line.emit(
                            f"  [{i+1}/{n}] {fname}  "
                            f"FWHM={fwhm_disp}  ecc={ecc:.3f}  "
                            f"SNR={snr:.1f}  bg={bg:.0f}  stars={n_det}")

                    frame_results.append({
                        "path":             fits_path,
                        "filename":         fname,
                        "metrics":          metrics,
                        "fwhm_px":          fwhm_px,
                        "fwhm_unit":        "arcsec" if pixel_scale else "px",
                        "n_stars_measured": n_meas,
                        "exptime":          exptime,
                        "date_obs":         date_obs,
                        "score":            0.0,
                        "rank":             i + 1,
                        "approved":         True,
                        "error":            None,
                    })

                except Exception as e:
                    self.log_line.emit(
                        f"  ERROR [{i+1}/{n}] {fname}: {e}")
                    all_metrics.append({
                        "fwhm": 0.0, "eccentricity": 0.0,
                        "snr": 0.0, "background": 0.0, "star_count": 0.0,
                    })
                    frame_results.append({
                        "path": fits_path, "filename": fname,
                        "metrics": {}, "fwhm_px": 0.0,
                        "fwhm_unit": "px", "n_stars_measured": 0,
                        "exptime": 1.0, "date_obs": "",
                        "score": 0.0, "rank": i + 1,
                        "approved": False, "error": str(e),
                    })

            if self._cancel.is_set():
                self._abort(); return

            self.log_line.emit("Computing composite scores...")
            population = compute_population_stats(all_metrics)

            for frame, metrics in zip(frame_results, all_metrics):
                frame["score"] = score_frame(metrics, population, weights)

            frame_results.sort(key=lambda f: f["score"], reverse=True)
            for rank, frame in enumerate(frame_results, 1):
                frame["rank"] = rank

            self.all_scored.emit(frame_results)
            self.log_line.emit(
                f"── Scoring complete ──  "
                f"Best: {frame_results[0]['filename']}  "
                f"score={frame_results[0]['score']:.1f}")
            self.log_line.emit(
                f"Worst: {frame_results[-1]['filename']}  "
                f"score={frame_results[-1]['score']:.1f}")

            self.finished.emit({
                "success":       True,
                "frame_results": frame_results,
                "n_frames":      n,
                "population":    population,
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"FATAL ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _abort(self):
        self.log_line.emit("Cancelled by user.")
        self.finished.emit({"success": False, "error": "Cancelled"})

    def cancel(self):
        self._cancel.set()


# ─────────────────────────────────────────────────────────────────────────────
# SCORE DISTRIBUTION CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCanvas(FigureCanvasQTAgg):
    """
    Score histogram with draggable threshold line + FWHM scatter.
    Left-click and drag the orange threshold line to set the cutoff.
    """
    threshold_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        self.fig      = Figure(figsize=(9, 4), facecolor=SIRIL_BG)
        self.ax_hist  = self.fig.add_subplot(1, 2, 1)
        self.ax_fwhm  = self.fig.add_subplot(1, 2, 2)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self._frame_results = []
        self._threshold     = 70.0
        self._elbow         = None
        self._dragging      = False
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)
        self._draw_empty()

    def _style_axes(self):
        for ax in [self.ax_hist, self.ax_fwhm]:
            ax.set_facecolor(SIRIL_BG2)
            ax.tick_params(colors=SIRIL_TEXT, labelsize=8)
            for sp in ["bottom", "left"]:
                ax.spines[sp].set_color(SIRIL_BORDER)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)

    def _draw_empty(self):
        for ax in [self.ax_hist, self.ax_fwhm]:
            ax.clear()
            ax.set_facecolor(SIRIL_BG2)
            ax.text(0.5, 0.5, "No data yet",
                    transform=ax.transAxes, ha="center", va="center",
                    color=SIRIL_TEXT_DIM, fontsize=10)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            for sp in ["bottom", "left"]:
                ax.spines[sp].set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def show_scores(self, frame_results: list, threshold: float,
                    elbow: float = None):
        self._frame_results = frame_results
        self._threshold     = threshold
        self._elbow         = elbow
        self._redraw()

    def update_threshold(self, threshold: float):
        self._threshold = threshold
        self._redraw()

    def _redraw(self):
        fr  = self._frame_results
        if not fr:
            self._draw_empty(); return

        scores = np.array([f["score"] for f in fr])
        fwhms  = np.array([f["metrics"].get("fwhm", 0) for f in fr])
        thr    = self._threshold

        # ── Histogram ──
        ax = self.ax_hist
        ax.clear()
        ax.set_facecolor(SIRIL_BG2)
        bins = np.linspace(0, 100, 26)
        approved_m = scores >= thr
        ax.hist(scores[approved_m],  bins=bins, color=SIRIL_SUCCESS,
                alpha=0.85, label="Approved")
        ax.hist(scores[~approved_m], bins=bins, color=SIRIL_ERROR,
                alpha=0.85, label="Rejected")
        ax.axvline(thr, color="#ff6b35", linewidth=2,
                   linestyle="--",
                   label=f"Threshold: {thr:.0f}",
                   picker=8)
        if self._elbow is not None:
            ax.axvline(self._elbow, color=SIRIL_WARNING, linewidth=1.2,
                       linestyle=":", alpha=0.8,
                       label=f"Auto (elbow): {self._elbow:.0f}")
        n_app = int(approved_m.sum())
        ax.set_xlabel("Score", color=SIRIL_TEXT_DIM, fontsize=8)
        ax.set_ylabel("Frames", color=SIRIL_TEXT_DIM, fontsize=8)
        ax.set_title(f"Score Distribution  ({n_app}/{len(scores)} approved)",
                     color=SIRIL_SECTION, fontsize=9)
        ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                  labelcolor=SIRIL_TEXT, fontsize=7)
        ax.set_xlim(0, 100)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=8)

        # ── FWHM scatter ──
        ax2 = self.ax_fwhm
        ax2.clear()
        ax2.set_facecolor(SIRIL_BG2)
        if fwhms.max() > 0:
            clrs = [SIRIL_SUCCESS if s >= thr else SIRIL_ERROR
                    for s in scores]
            ax2.scatter(scores, fwhms, c=clrs, s=14, alpha=0.75, zorder=3)
            ax2.axvline(thr, color="#ff6b35", linewidth=1.5,
                        linestyle="--", alpha=0.8)
            fwhm_unit = fr[0].get("fwhm_unit", "px")
            ax2.set_xlabel("Score", color=SIRIL_TEXT_DIM, fontsize=8)
            ax2.set_ylabel(f"FWHM ({fwhm_unit})",
                           color=SIRIL_TEXT_DIM, fontsize=8)
            ax2.set_title("FWHM vs Score", color=SIRIL_SECTION, fontsize=9)
        else:
            ax2.text(0.5, 0.5, "FWHM not available",
                     transform=ax2.transAxes, ha="center",
                     color=SIRIL_TEXT_DIM, fontsize=9)
        for sp in ["bottom", "left"]:
            ax2.spines[sp].set_color(SIRIL_BORDER)
        for sp in ["top", "right"]:
            ax2.spines[sp].set_visible(False)
        ax2.tick_params(colors=SIRIL_TEXT_DIM, labelsize=8)

        self.fig.tight_layout(pad=0.5)
        self.draw()

    def _on_press(self, event):
        if event.inaxes is self.ax_hist and event.button == 1:
            self._dragging = True
            v = float(np.clip(event.xdata, 0, 100))
            self._threshold = v
            self.threshold_changed.emit(v)

    def _on_motion(self, event):
        if self._dragging and event.inaxes is self.ax_hist and event.xdata is not None:
            v = float(np.clip(event.xdata, 0, 100))
            self._threshold = v
            self.threshold_changed.emit(v)

    def _on_release(self, _event):
        self._dragging = False


# ─────────────────────────────────────────────────────────────────────────────
# THUMBNAIL DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class ThumbnailDialog(QDialog):
    """
    Non-modal dialog showing a FITS frame thumbnail with metrics overlay.
    User can click through multiple frames — stays open.
    """
    def __init__(self, frame_result: dict, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        fname = frame_result.get("filename", "Frame")
        self.setWindowTitle(f"Frame Preview — {fname}")
        self.setMinimumSize(540, 480)
        self._build(frame_result)

    def _build(self, fr: dict):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Canvas
        fig  = Figure(figsize=(5, 4), facecolor=SIRIL_BG)
        ax   = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(SIRIL_BG)
        canvas = FigureCanvasQTAgg(fig)

        try:
            with astropy_fits.open(fr["path"]) as hdul:
                data = hdul[0].data.astype(np.float32)
            if data.ndim == 3:
                lum = data[0] if data.shape[0] in (1, 3) else data[0]
                if data.shape[0] == 3:
                    lum = 0.299*data[0] + 0.587*data[1] + 0.114*data[2]
            else:
                lum = data

            # Downsample to ≤400px
            h, w = lum.shape
            scale = min(400 / max(h, w), 1.0)
            if scale < 1.0:
                from scipy.ndimage import zoom
                lum = zoom(lum, scale, order=1)

            # Auto-stretch: linear 0.1% – 99.9%
            lo, hi = np.percentile(lum, [0.1, 99.9])
            if hi > lo:
                lum = np.clip((lum - lo) / (hi - lo), 0, 1)

            ax.imshow(lum, cmap="gray", origin="lower", aspect="equal",
                      interpolation="nearest")

            m     = fr.get("metrics", {})
            score = fr.get("score", 0)
            unit  = fr.get("fwhm_unit", "px")
            txt   = (f"Score: {score:.1f}   Rank: #{fr.get('rank','-')}\n"
                     f"FWHM: {m.get('fwhm',0):.2f} {unit}   "
                     f"Ecc: {m.get('eccentricity',0):.3f}\n"
                     f"SNR: {m.get('snr',0):.1f}   "
                     f"Stars: {int(m.get('star_count',0))}")
            ax.set_title(txt, color=SIRIL_TEXT, fontsize=8, pad=4)

        except Exception as e:
            ax.text(0.5, 0.5, f"Cannot load frame:\n{e}",
                    transform=ax.transAxes, ha="center", va="center",
                    color=SIRIL_ERROR, fontsize=9)

        ax.axis("off")
        fig.tight_layout(pad=0.3)
        layout.addWidget(canvas)

        # Filename label
        lbl = QLabel(fr.get("filename", ""))
        lbl.setObjectName("dim")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        btn = QPushButton("Close")
        btn.clicked.connect(self.close)
        btn.setFixedWidth(80)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(btn)
        layout.addLayout(row)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

METRIC_KEYS    = ["fwhm", "eccentricity", "snr", "background", "star_count"]
METRIC_LABELS  = ["FWHM", "Eccentricity", "SNR", "Background", "Star Count"]
DEFAULT_WEIGHTS = [40, 20, 20, 10, 10]   # integer percentages, must sum to 100

TABLE_COLS = ["Rank", "Score", "✓/✗", "Filename",
              "FWHM", "Ecc", "SNR", "BG", "Stars"]

OUTPUT_MODES = [
    "Copy approved frames to folder",
    "Write approved list (text file)",
    "Move rejected to subfolder",
    "Write weighted stack list",
]
OUTPUT_MODE_KEYS = ["copy", "list", "move_rejected", "weighted"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏆  Subframe Auto-Ranking  —  Siril")
        self.setMinimumSize(1200, 720)

        self._frame_results   = []
        self._cancel_event    = threading.Event()
        self._worker          = None
        self._weights_updating = False
        self._threshold_updating = False
        self._thumbnail_dialogs  = []
        self._threshold          = 70
        self._elbow_threshold    = None

        self._build_ui()
        self._set_running_state(False)
        self._set_ranked_state(False)

    # ── UI CONSTRUCTION ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 6, 8, 4)
        root.setSpacing(5)

        root.addWidget(self._make_title_bar())

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._make_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 870])
        root.addWidget(splitter, 1)

        root.addWidget(self._make_bottom_row())

        self._status = QLabel(
            "Ready — select a folder and scan for FITS files")
        self._status.setObjectName("dim")
        self.statusBar().addPermanentWidget(self._status, 1)

    def _make_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(36)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(4, 0, 4, 0)

        left = QLabel("🏆  Subframe Auto-Ranking  —  Siril")
        left.setStyleSheet(
            f"font-size: 14pt; font-weight: bold; color: {SIRIL_ACCENT};")
        hl.addWidget(left)
        hl.addStretch()

        right = QLabel("v1.0  |  PixInsight SubframeSelector equivalent")
        right.setObjectName("dim")
        hl.addWidget(right)
        return bar

    # ── LEFT PANEL ───────────────────────────────────────────────────────────

    def _make_left_panel(self) -> QWidget:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.setFixedWidth(340)

        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(8)

        vl.addWidget(self._make_group_input())
        vl.addWidget(self._make_group_pixscale())
        vl.addWidget(self._make_group_weights())
        vl.addWidget(self._make_group_threshold())
        vl.addWidget(self._make_group_output())
        vl.addStretch()

        outer.setWidget(container)
        return outer

    def _make_group_input(self) -> QGroupBox:
        grp = QGroupBox("Input Sequence")
        vl  = QVBoxLayout(grp)
        vl.setSpacing(5)

        # Folder row
        hl = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("FITS folder…")
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(70)
        btn_browse.clicked.connect(self._browse_input)
        hl.addWidget(self._folder_edit)
        hl.addWidget(btn_browse)
        vl.addLayout(hl)

        # FITS extension filter
        ext_row = QHBoxLayout()
        ext_row.addWidget(QLabel("Extension:"))
        self._ext_combo = QComboBox()
        self._ext_combo.addItems([".fit", ".fits", ".fts", "All FITS"])
        ext_row.addWidget(self._ext_combo)
        ext_row.addStretch()
        vl.addLayout(ext_row)

        self._btn_scan = QPushButton("🔍  Scan for FITS")
        self._btn_scan.setObjectName("primary")
        self._btn_scan.clicked.connect(self._scan_folder)
        vl.addWidget(self._btn_scan)

        self._lbl_file_info = QLabel("No files loaded")
        self._lbl_file_info.setObjectName("dim")
        vl.addWidget(self._lbl_file_info)

        return grp

    def _make_group_pixscale(self) -> QGroupBox:
        grp = QGroupBox("Pixel Scale (optional)")
        vl  = QVBoxLayout(grp)
        vl.setSpacing(5)

        self._pixscale_mode = QComboBox()
        modes = ["Manual entry", "Skip (use pixels)"]
        if HAS_EQUIPMENT_MANAGER:
            modes.insert(1, "From equipment profile")
        self._pixscale_mode.addItems(modes)
        self._pixscale_mode.currentIndexChanged.connect(
            self._on_pixscale_mode)
        vl.addWidget(self._pixscale_mode)

        self._pixscale_spin = QDoubleSpinBox()
        self._pixscale_spin.setRange(0.01, 20.0)
        self._pixscale_spin.setValue(1.0)
        self._pixscale_spin.setSuffix(" arcsec/px")
        self._pixscale_spin.setDecimals(3)
        vl.addWidget(self._pixscale_spin)

        # Profile selector (shown only if equipment manager available)
        self._profile_combo = QComboBox()
        self._profile_combo.setVisible(False)
        if HAS_EQUIPMENT_MANAGER:
            try:
                profiles = load_all_profiles()
                for p in profiles:
                    self._profile_combo.addItem(p.get("name", "?"), p)
            except Exception:
                pass
            self._profile_combo.currentIndexChanged.connect(
                self._load_profile_scale)
        vl.addWidget(self._profile_combo)

        hint = QLabel("Used to display FWHM in arcsec instead of pixels")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        vl.addWidget(hint)

        return grp

    def _make_group_weights(self) -> QGroupBox:
        grp = QGroupBox("Metric Weights")
        vl  = QVBoxLayout(grp)
        vl.setSpacing(4)

        hint = QLabel("Sliders auto-normalise to 100%")
        hint.setObjectName("dim")
        vl.addWidget(hint)

        self._weight_sliders = []
        self._weight_labels  = []

        for i, (key, label, default) in enumerate(
                zip(METRIC_KEYS, METRIC_LABELS, DEFAULT_WEIGHTS)):
            row = QHBoxLayout()
            lbl_name = QLabel(label)
            lbl_name.setFixedWidth(90)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(default)
            lbl_pct = QLabel(f"{default}%")
            lbl_pct.setFixedWidth(36)
            lbl_pct.setAlignment(Qt.AlignmentFlag.AlignRight |
                                  Qt.AlignmentFlag.AlignVCenter)
            lbl_pct.setStyleSheet(f"color: {SIRIL_ACCENT};")

            sl.valueChanged.connect(
                lambda val, idx=i: self._on_weight_changed(idx))

            row.addWidget(lbl_name)
            row.addWidget(sl)
            row.addWidget(lbl_pct)
            vl.addLayout(row)

            self._weight_sliders.append(sl)
            self._weight_labels.append(lbl_pct)

        row_btns = QHBoxLayout()
        btn_reset = QPushButton("Reset to defaults")
        btn_reset.clicked.connect(self._reset_weights)
        row_btns.addWidget(btn_reset)

        self._btn_recompute = QPushButton("Recompute scores")
        self._btn_recompute.clicked.connect(self._recompute_scores)
        self._btn_recompute.setEnabled(False)
        row_btns.addWidget(self._btn_recompute)
        vl.addLayout(row_btns)

        return grp

    def _make_group_threshold(self) -> QGroupBox:
        grp = QGroupBox("Rejection Threshold")
        vl  = QVBoxLayout(grp)
        vl.setSpacing(5)

        vl.addWidget(QLabel("Reject frames scoring below:"))

        thr_row = QHBoxLayout()
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 100)
        self._threshold_slider.setValue(70)
        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(0, 100)
        self._threshold_spin.setValue(70)
        self._threshold_spin.setFixedWidth(55)
        thr_row.addWidget(self._threshold_slider)
        thr_row.addWidget(self._threshold_spin)
        vl.addLayout(thr_row)

        self._threshold_slider.valueChanged.connect(
            lambda v: self._on_threshold_changed(v))
        self._threshold_spin.valueChanged.connect(
            lambda v: self._on_threshold_changed(v))

        self._lbl_approval = QLabel("No frames ranked yet")
        self._lbl_approval.setObjectName("warn")
        vl.addWidget(self._lbl_approval)

        self._btn_use_elbow = QPushButton("Use Auto (elbow) threshold")
        self._btn_use_elbow.clicked.connect(self._use_elbow_threshold)
        self._btn_use_elbow.setEnabled(False)
        vl.addWidget(self._btn_use_elbow)

        hint = QLabel("Drag the orange line in the chart to adjust")
        hint.setObjectName("dim")
        vl.addWidget(hint)

        return grp

    def _make_group_output(self) -> QGroupBox:
        grp = QGroupBox("Output")
        vl  = QVBoxLayout(grp)
        vl.setSpacing(5)

        vl.addWidget(QLabel("Output mode:"))
        self._output_mode_combo = QComboBox()
        self._output_mode_combo.addItems(OUTPUT_MODES)
        vl.addWidget(self._output_mode_combo)

        vl.addWidget(QLabel("Output folder:"))
        out_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        self._output_folder_edit.setPlaceholderText(
            "Default: [input]_ranked/")
        btn_out = QPushButton("Browse")
        btn_out.setFixedWidth(70)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(self._output_folder_edit)
        out_row.addWidget(btn_out)
        vl.addLayout(out_row)

        self._chk_csv = QCheckBox("Export ranking CSV")
        self._chk_csv.setChecked(True)
        vl.addWidget(self._chk_csv)

        self._chk_stack = QCheckBox(
            "Stack approved frames in Siril after export")
        self._chk_stack.setChecked(False)
        vl.addWidget(self._chk_stack)

        return grp

    # ── RIGHT PANEL ──────────────────────────────────────────────────────────

    def _make_right_panel(self) -> QWidget:
        outer = QWidget()
        vl    = QVBoxLayout(outer)
        vl.setContentsMargins(2, 0, 2, 0)
        vl.setSpacing(4)

        self._tabs = QTabWidget()

        # Tab 1 — Distribution
        dist_tab = QWidget()
        dt_layout = QVBoxLayout(dist_tab)
        dt_layout.setContentsMargins(4, 4, 4, 4)
        dt_layout.setSpacing(6)

        self._score_canvas = ScoreCanvas()
        self._score_canvas.threshold_changed.connect(
            self._on_threshold_changed)
        dt_layout.addWidget(self._score_canvas, 1)
        dt_layout.addWidget(self._make_stats_group())
        self._tabs.addTab(dist_tab, "📊  Distribution")

        # Tab 2 — Rankings
        rank_tab = QWidget()
        rt_layout = QVBoxLayout(rank_tab)
        rt_layout.setContentsMargins(4, 4, 4, 4)
        rt_layout.setSpacing(4)

        self._table = QTableWidget()
        self._table.setColumnCount(len(TABLE_COLS))
        self._table.setHorizontalHeaderLabels(TABLE_COLS)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self._table.horizontalHeader().setSectionsClickable(True)
        rt_layout.addWidget(self._table, 1)

        tbl_btns = QHBoxLayout()
        self._btn_export_csv = QPushButton("💾  Export CSV")
        self._btn_export_csv.setObjectName("export")
        self._btn_export_csv.clicked.connect(self._export_csv_only)

        self._btn_copy_approved = QPushButton("📋  Copy approved paths")
        self._btn_copy_approved.clicked.connect(self._copy_approved_paths)

        self._btn_open_output = QPushButton("📂  Open output folder")
        self._btn_open_output.clicked.connect(self._open_output_folder)

        for btn in [self._btn_export_csv,
                    self._btn_copy_approved,
                    self._btn_open_output]:
            tbl_btns.addWidget(btn)
        tbl_btns.addStretch()
        rt_layout.addLayout(tbl_btns)
        self._tabs.addTab(rank_tab, "🏆  Rankings")

        # Tab 3 — Log
        log_tab = QWidget()
        ll = QVBoxLayout(log_tab)
        ll.setContentsMargins(4, 4, 4, 4)
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Courier New", 9))
        ll.addWidget(self._log_view)

        log_btns = QHBoxLayout()
        btn_clear = QPushButton("Clear log")
        btn_clear.clicked.connect(self._log_view.clear)
        log_btns.addWidget(btn_clear)
        log_btns.addStretch()
        ll.addLayout(log_btns)
        self._tabs.addTab(log_tab, "📋  Log")

        vl.addWidget(self._tabs)
        return outer

    def _make_stats_group(self) -> QGroupBox:
        grp = QGroupBox("Summary Statistics")
        hl  = QHBoxLayout(grp)
        hl.setSpacing(20)

        self._stat_labels = {}
        stats_defs = [
            ("fwhm_med",  "Median FWHM:"),
            ("ecc_med",   "Median Ecc:"),
            ("snr_med",   "Median SNR:"),
            ("best",      "Best score:"),
            ("worst",     "Worst score:"),
        ]
        for key, caption in stats_defs:
            col = QVBoxLayout()
            cap = QLabel(caption)
            cap.setObjectName("dim")
            val = QLabel("—")
            val.setStyleSheet(
                f"color: {SIRIL_ACCENT}; font-weight: bold; font-size: 11pt;")
            col.addWidget(cap)
            col.addWidget(val)
            hl.addLayout(col)
            self._stat_labels[key] = val

        hl.addStretch()
        return grp

    def _make_bottom_row(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(0, 2, 0, 2)
        hl.setSpacing(6)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(8)
        self._progress.setVisible(False)
        hl.addWidget(self._progress, 1)

        self._btn_rank = QPushButton("▶  Rank Frames")
        self._btn_rank.setObjectName("primary")
        self._btn_rank.setFixedHeight(30)
        self._btn_rank.clicked.connect(self._start_ranking)
        hl.addWidget(self._btn_rank)

        self._btn_apply = QPushButton("✓  Apply threshold & export")
        self._btn_apply.setObjectName("export")
        self._btn_apply.setFixedHeight(30)
        self._btn_apply.clicked.connect(self._apply_and_export)
        hl.addWidget(self._btn_apply)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setFixedHeight(30)
        self._btn_cancel.clicked.connect(self._cancel)
        hl.addWidget(self._btn_cancel)

        return bar

    # ── STATE HELPERS ─────────────────────────────────────────────────────────

    def _set_running_state(self, running: bool):
        self._btn_rank.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._btn_scan.setEnabled(not running)
        self._progress.setVisible(running)

    def _set_ranked_state(self, ranked: bool):
        self._btn_apply.setEnabled(ranked)
        self._btn_export_csv.setEnabled(ranked)
        self._btn_copy_approved.setEnabled(ranked)
        self._btn_open_output.setEnabled(ranked)
        self._btn_recompute.setEnabled(ranked)
        self._btn_use_elbow.setEnabled(ranked)

    # ── PIXEL SCALE ──────────────────────────────────────────────────────────

    def _on_pixscale_mode(self, idx):
        mode = self._pixscale_mode.currentText()
        manual = "Manual" in mode
        profile_mode = "profile" in mode.lower()
        self._pixscale_spin.setVisible(manual)
        self._profile_combo.setVisible(profile_mode)

    def _load_profile_scale(self):
        if not HAS_EQUIPMENT_MANAGER:
            return
        data = self._profile_combo.currentData()
        if data and "pixel_scale" in data:
            self._pixscale_spin.setValue(float(data["pixel_scale"]))

    def _get_pixel_scale(self):
        mode = self._pixscale_mode.currentText()
        if "Skip" in mode:
            return None
        return self._pixscale_spin.value()

    # ── SCAN ─────────────────────────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select FITS folder", "")
        if folder:
            self._folder_edit.setText(folder)
            default_out = folder.rstrip("/\\") + "_ranked"
            self._output_folder_edit.setText(default_out)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", "")
        if folder:
            self._output_folder_edit.setText(folder)

    def _scan_folder(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder",
                                "Please enter a valid FITS folder path.")
            return

        ext_text = self._ext_combo.currentText()
        if ext_text == "All FITS":
            patterns = ["*.fit", "*.fits", "*.fts",
                        "*.FIT", "*.FITS", "*.FTS"]
        else:
            patterns = [f"*{ext_text}", f"*{ext_text.upper()}"]

        files = []
        for pat in patterns:
            files += glob.glob(os.path.join(folder, pat))
        files = sorted(set(files))

        if not files:
            self._lbl_file_info.setText("No FITS files found")
            self._lbl_file_info.setObjectName("err")
            return

        self._fits_files = files
        # Try to get image size from first file
        try:
            with astropy_fits.open(files[0]) as hdul:
                d = hdul[0].data
                shape_str = f"{d.shape[-1]}×{d.shape[-2]}"
        except Exception:
            shape_str = "?"

        msg = f"{len(files)} FITS files  |  {shape_str} px"
        self._lbl_file_info.setText(msg)
        self._lbl_file_info.setObjectName("ok")
        self._lbl_file_info.setStyleSheet(f"color: {SIRIL_SUCCESS};")
        self._log(f"Scanned: {len(files)} files in {folder}")

        if not self._output_folder_edit.text().strip():
            self._output_folder_edit.setText(
                folder.rstrip("/\\") + "_ranked")

    # ── WEIGHT SLIDERS ───────────────────────────────────────────────────────

    def _on_weight_changed(self, changed_idx: int):
        if self._weights_updating:
            return
        self._weights_updating = True
        try:
            vals = [sl.value() for sl in self._weight_sliders]
            new_val = vals[changed_idx]
            others  = [i for i in range(len(vals)) if i != changed_idx]
            sum_others_old = sum(vals[i] for i in others)
            sum_others_new = 100 - new_val

            if sum_others_old > 0:
                for i in others:
                    vals[i] = int(round(vals[i] * sum_others_new
                                        / sum_others_old))
            elif sum_others_new > 0 and others:
                per = sum_others_new // len(others)
                for i in others:
                    vals[i] = per
            else:
                for i in others:
                    vals[i] = 0

            # Fix rounding to ensure exact sum of 100
            total = sum(vals)
            diff  = 100 - total
            if diff != 0 and others:
                # Add to the largest other slider
                biggest = max(others, key=lambda i: vals[i])
                vals[biggest] += diff

            for i, (sl, lbl) in enumerate(
                    zip(self._weight_sliders, self._weight_labels)):
                sl.setValue(vals[i])
                lbl.setText(f"{vals[i]}%")
        finally:
            self._weights_updating = False

    def _reset_weights(self):
        self._weights_updating = True
        for sl, lbl, default in zip(
                self._weight_sliders, self._weight_labels, DEFAULT_WEIGHTS):
            sl.setValue(default)
            lbl.setText(f"{default}%")
        self._weights_updating = False

    def _get_weights_dict(self) -> dict:
        vals = [sl.value() for sl in self._weight_sliders]
        total = sum(vals)
        if total <= 0:
            total = 1
        return {k: v / total for k, v in zip(METRIC_KEYS, vals)}

    # ── THRESHOLD ─────────────────────────────────────────────────────────────

    def _on_threshold_changed(self, value):
        if self._threshold_updating:
            return
        self._threshold_updating = True
        try:
            v = int(round(float(value)))
            v = max(0, min(100, v))
            self._threshold = v
            self._threshold_slider.setValue(v)
            self._threshold_spin.setValue(v)
            self._update_approval_label()
            if self._frame_results:
                self._score_canvas.update_threshold(v)
                self._update_table_colors()
        finally:
            self._threshold_updating = False

    def _update_approval_label(self):
        if not self._frame_results:
            self._lbl_approval.setText("No frames ranked yet")
            return
        n_total = len(self._frame_results)
        n_app   = sum(1 for f in self._frame_results
                      if f["score"] >= self._threshold)
        pct     = n_app / max(n_total, 1) * 100
        self._lbl_approval.setText(
            f"Approving {n_app}/{n_total} frames  ({pct:.0f}%)")
        self._lbl_approval.setStyleSheet(
            f"color: {SIRIL_SUCCESS};" if pct >= 50
            else f"color: {SIRIL_WARNING};")

    def _use_elbow_threshold(self):
        if self._elbow_threshold is not None:
            self._on_threshold_changed(self._elbow_threshold)

    # ── RANKING WORKER ────────────────────────────────────────────────────────

    def _start_ranking(self):
        if not hasattr(self, "_fits_files") or not self._fits_files:
            QMessageBox.warning(self, "No files",
                                "Please scan a folder first.")
            return

        self._cancel_event.clear()
        self._log("=" * 60)
        self._log(f"Starting ranking run  —  "
                  f"{datetime.now().strftime('%H:%M:%S')}")

        config = {
            "fits_files":         self._fits_files,
            "pixel_scale_arcsec": self._get_pixel_scale(),
            "weights":            self._get_weights_dict(),
        }
        self._progress.setRange(0, len(self._fits_files))
        self._progress.setValue(0)
        self._set_running_state(True)
        self._set_ranked_state(False)
        self._frame_results = []
        self._tabs.setCurrentIndex(2)   # Switch to Log while running

        self._worker = RankingWorker(config, self._cancel_event)
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.all_scored.connect(self._on_all_scored)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._cancel_event.set()
            self._worker.cancel()
            self._log("Cancelling…")

    def _on_progress(self, done: int, total: int, msg: str):
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status.setText(f"  {msg}  [{done}/{total}]")

    def _on_all_scored(self, frame_results: list):
        """Called when all frames have been scored (before finished signal)."""
        self._frame_results = frame_results
        self._populate_table(frame_results)
        self._update_approval_label()

        scores = [f["score"] for f in frame_results]
        self._elbow_threshold = find_elbow_threshold(scores)
        self._score_canvas.show_scores(
            frame_results, self._threshold,
            elbow=self._elbow_threshold)
        self._update_stats(frame_results)

    def _on_finished(self, result: dict):
        self._set_running_state(False)
        if result.get("success"):
            self._set_ranked_state(True)
            n = result["n_frames"]
            thr = self._threshold
            n_app = sum(1 for f in self._frame_results
                        if f["score"] >= thr)
            self._status.setText(
                f"✓  {n} frames ranked  |  "
                f"Threshold: {thr}  |  "
                f"Approving {n_app}/{n}")
            self._log(
                f"── Done  |  Elbow auto-threshold: "
                f"{self._elbow_threshold:.1f} ──")
            self._tabs.setCurrentIndex(0)   # Switch to Distribution
        else:
            err = result.get("error", "Unknown error")
            self._status.setText(f"✗  Failed: {err}")

    # ── RECOMPUTE ─────────────────────────────────────────────────────────────

    def _recompute_scores(self):
        """Recompute scores with current weights — no I/O."""
        if not self._frame_results:
            return
        weights    = self._get_weights_dict()
        all_mets   = [f["metrics"] for f in self._frame_results]
        population = compute_population_stats(all_mets)
        for frame in self._frame_results:
            frame["score"] = score_frame(
                frame["metrics"], population, weights)
        self._frame_results.sort(
            key=lambda f: f["score"], reverse=True)
        for rank, frame in enumerate(self._frame_results, 1):
            frame["rank"] = rank

        scores = [f["score"] for f in self._frame_results]
        self._elbow_threshold = find_elbow_threshold(scores)
        self._score_canvas.show_scores(
            self._frame_results, self._threshold,
            elbow=self._elbow_threshold)
        self._populate_table(self._frame_results)
        self._update_stats(self._frame_results)
        self._update_approval_label()
        self._log("Scores recomputed with new weights.")

    # ── TABLE ─────────────────────────────────────────────────────────────────

    def _populate_table(self, frame_results: list):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(frame_results))

        col_widths = [46, 58, 36, 240, 70, 58, 70, 70, 54]
        for c, w in enumerate(col_widths):
            self._table.setColumnWidth(c, w)

        for row, fr in enumerate(frame_results):
            m     = fr.get("metrics", {})
            score = fr.get("score", 0.0)
            unit  = fr.get("fwhm_unit", "px")

            vals = [
                fr.get("rank", row+1),
                f"{score:.1f}",
                "✓" if score >= self._threshold else "✗",
                fr.get("filename", ""),
                (f"{m.get('fwhm',0):.2f}{unit[0]}"
                 if m.get("fwhm", 0) > 0 else "—"),
                f"{m.get('eccentricity',0):.3f}",
                f"{m.get('snr',0):.1f}",
                f"{m.get('background',0):.0f}",
                str(int(m.get("star_count", 0))),
            ]

            for col, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                    if col != 3
                    else Qt.AlignmentFlag.AlignLeft |
                         Qt.AlignmentFlag.AlignVCenter)
                if col == 0:    # store path in UserRole
                    item.setData(Qt.ItemDataRole.UserRole, fr["path"])
                self._table.setItem(row, col, item)

            self._colour_row(row, score)

        self._table.setSortingEnabled(True)

    def _colour_row(self, row: int, score: float):
        if score < self._threshold:
            bg = QColor("#2a1e1e")
            fg = QColor(SIRIL_TEXT_DIM)
        else:
            bg = QColor(SIRIL_BG2)
            fg = QColor(SIRIL_TEXT)
        for col in range(self._table.columnCount()):
            item = self._table.item(row, col)
            if item:
                item.setBackground(bg)
                item.setForeground(fg)

    def _update_table_colors(self):
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 1)   # score column
            if item:
                try:
                    score = float(item.text())
                except ValueError:
                    score = 0.0
                self._colour_row(row, score)
            # Update ✓/✗ column
            check_item = self._table.item(row, 2)
            if check_item and item:
                try:
                    s = float(item.text())
                except ValueError:
                    s = 0.0
                check_item.setText("✓" if s >= self._threshold else "✗")

    # ── STATS ─────────────────────────────────────────────────────────────────

    def _update_stats(self, frame_results: list):
        if not frame_results:
            return
        mets  = [f["metrics"] for f in frame_results]
        fwhms = [m.get("fwhm", 0) for m in mets if m.get("fwhm", 0) > 0]
        eccs  = [m.get("eccentricity", 0) for m in mets]
        snrs  = [m.get("snr", 0) for m in mets if m.get("snr", 0) > 0]
        unit  = frame_results[0].get("fwhm_unit", "px")

        med_fwhm = np.median(fwhms) if fwhms else 0.0
        med_ecc  = np.median(eccs)  if eccs  else 0.0
        med_snr  = np.median(snrs)  if snrs  else 0.0
        best     = frame_results[0]
        worst    = frame_results[-1]

        self._stat_labels["fwhm_med"].setText(
            f"{med_fwhm:.2f} {unit}")
        self._stat_labels["ecc_med"].setText(
            f"{med_ecc:.3f}")
        self._stat_labels["snr_med"].setText(
            f"{med_snr:.1f}")
        self._stat_labels["best"].setText(
            f"{best['score']:.1f}  ({best['filename'][:20]}…)"
            if len(best["filename"]) > 20
            else f"{best['score']:.1f}  ({best['filename']})")
        self._stat_labels["worst"].setText(
            f"{worst['score']:.1f}  ({worst['filename'][:20]}…)"
            if len(worst["filename"]) > 20
            else f"{worst['score']:.1f}  ({worst['filename']})")

    # ── THUMBNAILS ────────────────────────────────────────────────────────────

    def _on_row_double_clicked(self, row: int, _col: int):
        item = self._table.item(row, 0)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        # Find matching frame result
        fr = next((f for f in self._frame_results
                   if f["path"] == path), None)
        if not fr:
            return
        dlg = ThumbnailDialog(fr, parent=None)
        dlg.setStyleSheet(SIRIL_STYLESHEET)
        dlg.show()
        self._thumbnail_dialogs.append(dlg)
        # Remove closed dialogs from list
        self._thumbnail_dialogs = [
            d for d in self._thumbnail_dialogs if d.isVisible()]

    # ── EXPORT ────────────────────────────────────────────────────────────────

    def _get_output_dir(self) -> str:
        out = self._output_folder_edit.text().strip()
        if not out:
            folder = self._folder_edit.text().strip()
            out = (folder.rstrip("/\\") + "_ranked") if folder else "ranked"
        return out

    def _apply_and_export(self):
        if not self._frame_results:
            QMessageBox.warning(self, "No data", "No frames have been ranked yet.")
            return

        mode_idx  = self._output_mode_combo.currentIndex()
        mode_key  = OUTPUT_MODE_KEYS[mode_idx]
        output_dir = self._get_output_dir()

        self._log(f"Exporting — mode: {OUTPUT_MODES[mode_idx]}")
        self._log(f"Output: {output_dir}")

        result = apply_threshold_and_export(
            self._frame_results,
            float(self._threshold),
            mode_key,
            output_dir,
            export_csv=self._chk_csv.isChecked(),
            log_callback=self._log,
        )

        n_app = result["n_approved"]
        n_rej = result["n_rejected"]
        self._status.setText(
            f"✓  {n_app} frames exported  "
            f"({n_rej} rejected)  →  {output_dir}")
        self._log(f"Export complete: {n_app} approved, {n_rej} rejected.")

        if self._chk_stack.isChecked():
            self._trigger_siril_stack(output_dir, mode_key)

    def _trigger_siril_stack(self, output_dir: str, mode_key: str):
        """Optionally trigger Siril stacking after export."""
        if mode_key not in ("copy",):
            self._log(
                "Siril auto-stack only available with 'Copy approved' mode.")
            return
        try:
            siril = s.Siril()
            self._log("Triggering Siril stacking on approved frames…")
            siril.cmd("cd", output_dir)
            siril.cmd("convert", "light", "-out=ranked_lights_")
            siril.cmd("register", "ranked_lights_")
            siril.cmd("stack", "ranked_lights_", "rej", "w",
                      "3.0", "3.0", "-output=master_ranked")
            self._log("Siril stacking commands sent.")
        except Exception as e:
            self._log(f"Siril stack error: {e}")

    def _export_csv_only(self):
        if not self._frame_results:
            return
        out_dir = self._get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "frame_ranking.csv")
        _write_ranking_csv(self._frame_results,
                           float(self._threshold), csv_path)
        self._log(f"CSV exported: {csv_path}")
        self._status.setText(f"✓  CSV saved: {csv_path}")

    def _copy_approved_paths(self):
        if not self._frame_results:
            return
        approved = [f["path"] for f in self._frame_results
                    if f["score"] >= self._threshold]
        text = "\n".join(approved)
        QApplication.clipboard().setText(text)
        self._log(f"Copied {len(approved)} approved paths to clipboard.")

    def _open_output_folder(self):
        import subprocess
        out_dir = self._get_output_dir()
        if os.path.isdir(out_dir):
            if sys.platform == "win32":
                subprocess.Popen(["explorer", os.path.normpath(out_dir)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", out_dir])
            else:
                subprocess.Popen(["xdg-open", out_dir])
        else:
            QMessageBox.information(
                self, "Folder",
                f"Output folder does not exist yet:\n{out_dir}")

    # ── LOG ───────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts  = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_view.appendPlainText(line)
        # Auto-scroll
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())


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
