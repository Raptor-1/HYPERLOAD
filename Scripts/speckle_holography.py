"""
speckle_holography.py — Script #25
Speckle Holography Reconstruction for Siril
Uses 100% of frames; PSF estimated per-frame and Wiener-deconvolved.
Reference: Schödel et al. 2013, A&A 558, A33
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")
s.ensure_installed("photutils")

import os
import sys
import glob
import struct
import threading
import json
from datetime import datetime

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import center_of_mass
from astropy.io import fits as astropy_fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.patches

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QFrame, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ─── Theme ────────────────────────────────────────────────────────────────────
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

# ─── Epsilon guide ─────────────────────────────────────────────────────────────
EPSILON_GUIDE = {
    (0.001, 0.003): ("Very aggressive",
                     "Maximum sharpness. Use only with excellent seeing "
                     "and many frames (>500). May amplify noise."),
    (0.003, 0.02):  ("Recommended",
                     "Good balance of sharpness and noise. "
                     "Works well for most planetary imaging sessions."),
    (0.02,  0.05):  ("Conservative",
                     "Softer result, less noise. Good for short sessions "
                     "or poor seeing. Similar to lucky imaging quality."),
    (0.05,  0.1):   ("Minimal",
                     "Very smooth. Mainly useful to check reconstruction "
                     "is working without noise amplification."),
}

# ─── SER constants ─────────────────────────────────────────────────────────────
SER_HEADER_SIZE      = 178
WINDOWS_EPOCH_OFFSET = 116444736000000000


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def extract_psf_from_frame(frame, star_pos, box_size=64, normalize=True):
    h, w   = frame.shape
    cy, cx = int(round(star_pos[0])), int(round(star_pos[1]))
    half   = box_size // 2
    if (cy - half < 0 or cy + half >= h or cx - half < 0 or cx + half >= w):
        return None
    cutout = frame[cy-half:cy+half, cx-half:cx+half].astype(np.float64).copy()
    corners = np.concatenate([
        cutout[:4, :4].ravel(), cutout[:4, -4:].ravel(),
        cutout[-4:, :4].ravel(), cutout[-4:, -4:].ravel(),
    ])
    bg = np.median(corners)
    cutout -= bg
    cutout  = np.clip(cutout, 0, None)
    if cutout.max() <= 0:
        return None
    if cutout.max() > 0.9 * (2**16 - 1):
        return None
    if normalize:
        total = cutout.sum()
        if total <= 0:
            return None
        cutout /= total
    psf_shifted = np.roll(np.roll(cutout, -half, axis=0), -half, axis=1)
    return psf_shifted


def extract_psf_multi_star(frame, star_positions, box_size=64):
    psfs = []
    for pos in star_positions:
        psf = extract_psf_from_frame(frame, pos, box_size=box_size)
        if psf is not None:
            psfs.append(psf)
    if not psfs:
        return None
    avg_psf = np.mean(psfs, axis=0)
    total   = avg_psf.sum()
    if total <= 0:
        return None
    return avg_psf / total


def embed_psf_in_frame(psf, frame_shape):
    h, w   = frame_shape
    ph, pw = psf.shape
    full   = np.zeros((h, w), dtype=np.float64)
    full[:ph, :pw] = psf
    return full


def wiener_deconvolve_frame(frame, psf_full, epsilon=0.01):
    F      = fft2(frame.astype(np.float64))
    H      = fft2(psf_full)
    H_abs2 = np.abs(H) ** 2
    eps_scaled = epsilon * H_abs2.max()
    W          = np.conj(H) / (H_abs2 + eps_scaled + 1e-30)
    return F * W


def speckle_holography_reconstruct(frames, star_positions_per_frame,
                                   psf_box_size=64, epsilon=0.01,
                                   log_callback=None, cancel_event=None):
    if not frames:
        return None
    h, w        = frames[0].shape
    accumulator = np.zeros((h, w), dtype=np.complex128)
    n_used      = 0
    n_skipped   = 0
    for i, (frame, star_positions) in enumerate(zip(frames, star_positions_per_frame)):
        if cancel_event and cancel_event.is_set():
            break
        if not star_positions:
            n_skipped += 1
            continue
        if len(star_positions) == 1:
            psf = extract_psf_from_frame(frame, star_positions[0], box_size=psf_box_size)
        else:
            psf = extract_psf_multi_star(frame, star_positions, box_size=psf_box_size)
        if psf is None:
            n_skipped += 1
            if log_callback and n_skipped <= 5:
                log_callback(f"  Frame {i}: PSF extraction failed, skipping")
            continue
        psf_full     = embed_psf_in_frame(psf, (h, w))
        contribution = wiener_deconvolve_frame(frame, psf_full, epsilon)
        accumulator += contribution
        n_used      += 1
        if log_callback and i % 50 == 0:
            log_callback(f"  Processed {i+1}/{len(frames)} frames "
                         f"(used: {n_used}, skipped: {n_skipped})")
    if n_used == 0:
        if log_callback:
            log_callback("ERROR: All frames failed PSF extraction")
        return None
    if log_callback:
        log_callback(f"Reconstruction complete: {n_used} frames used, "
                     f"{n_skipped} skipped "
                     f"({100*n_skipped/(n_used+n_skipped):.1f}%)")
    result = np.real(ifft2(accumulator))
    result = np.clip(result, 0, None)
    if result.max() > 0:
        result /= result.max()
    return result.astype(np.float32)


def detect_reference_stars(frame, n_stars=5, min_separation_px=None,
                            psf_box_size=64, threshold_sigma=8.0,
                            border_margin=None):
    h, w = frame.shape
    if border_margin is None:
        border_margin = psf_box_size // 2 + 5
    if min_separation_px is None:
        min_separation_px = psf_box_size * 1.5
    _, median, std = sigma_clipped_stats(frame, sigma=3.0)
    if std <= 0:
        return []
    daofind = DAOStarFinder(fwhm=5.0, threshold=threshold_sigma * std,
                            sharplo=0.3, sharphi=0.9,
                            roundlo=-0.5, roundhi=0.5)
    sources = daofind(frame - median)
    if sources is None or len(sources) == 0:
        return []
    sources.sort("peak")
    sources.reverse()
    positions = []
    sat_limit = 0.9 * (2**16 - 1)
    for src in sources:
        x, y = float(src["xcentroid"]), float(src["ycentroid"])
        if (x < border_margin or x > w - border_margin or
                y < border_margin or y > h - border_margin):
            continue
        if float(src["peak"]) > sat_limit:
            continue
        too_close = False
        for py, px in positions:
            dist = ((x - px)**2 + (y - py)**2)**0.5
            if dist < min_separation_px:
                too_close = True
                break
        if too_close:
            continue
        positions.append((y, x))
        if len(positions) >= n_stars:
            break
    return positions


def track_stars_across_sequence(frames, first_frame_positions,
                                 search_radius=20, log_callback=None):
    n_frames      = len(frames)
    all_positions = [first_frame_positions]
    for i, frame in enumerate(frames[1:], 1):
        if log_callback and i % 100 == 0:
            log_callback(f"  Tracking stars: frame {i}/{n_frames}")
        frame_positions = []
        prev_positions  = all_positions[-1]
        h, w = frame.shape
        for py, px in prev_positions:
            y1 = max(0, int(py) - search_radius)
            y2 = min(h, int(py) + search_radius)
            x1 = max(0, int(px) - search_radius)
            x2 = min(w, int(px) + search_radius)
            region = frame[y1:y2, x1:x2].astype(float)
            if region.max() <= 0:
                frame_positions.append((py, px))
                continue
            threshold = np.percentile(region, 80)
            binary    = (region > threshold).astype(float)
            if binary.sum() < 1:
                frame_positions.append((py, px))
                continue
            cy, cx = center_of_mass(region * binary)
            frame_positions.append((cy + y1, cx + x1))
        all_positions.append(frame_positions)
    return all_positions


def estimate_strehl(frame, star_pos, theoretical_airy_peak=None):
    h, w   = frame.shape
    cy, cx = int(round(star_pos[0])), int(round(star_pos[1]))
    box    = 16
    if (cy-box < 0 or cy+box >= h or cx-box < 0 or cx+box >= w):
        return 0.0
    cutout = frame[cy-box:cy+box, cx-box:cx+box].astype(float)
    bg     = np.median([cutout[:3, :3], cutout[:3, -3:],
                        cutout[-3:, :3], cutout[-3:, -3:]])
    cutout -= bg
    peak   = float(cutout.max())
    total  = float(cutout.sum())
    if total <= 0 or peak <= 0:
        return 0.0
    if theoretical_airy_peak is None:
        return peak / total
    return min(1.0, peak / theoretical_airy_peak)


def compute_mean_stack(frames):
    if not frames:
        return np.zeros((1, 1), dtype=np.float32)
    stack  = np.stack([f.astype(np.float64) for f in frames], axis=0)
    result = np.mean(stack, axis=0)
    mn, mx = result.min(), result.max()
    if mx > mn:
        result = (result - mn) / (mx - mn)
    return result.astype(np.float32)


# ─── SER / FITS loaders ────────────────────────────────────────────────────────

def read_ser_header(path):
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


def read_ser_frame(f, header, frame_index):
    bpp        = 1 if header["pixel_depth"] <= 8 else 2
    frame_size = header["width"] * header["height"] * bpp
    f.seek(SER_HEADER_SIZE + frame_index * frame_size)
    raw   = f.read(frame_size)
    dtype = np.uint8 if bpp == 1 else np.uint16
    return np.frombuffer(raw, dtype=dtype).reshape(header["height"], header["width"])


def load_ser_as_frames(ser_path, max_frames=None, log_callback=None):
    header = read_ser_header(ser_path)
    n = header["frame_count"]
    if max_frames:
        n = min(n, max_frames)
    frames = []
    with open(ser_path, "rb") as f:
        for i in range(n):
            frame = read_ser_frame(f, header, i).astype(np.float32)
            frames.append(frame)
            if log_callback and i % 100 == 0:
                log_callback(f"  Loading SER frame {i+1}/{n}")
    return frames


def load_fits_sequence(fits_dir, max_frames=None, log_callback=None):
    files = sorted(
        glob.glob(os.path.join(fits_dir, "*.fit")) +
        glob.glob(os.path.join(fits_dir, "*.fits")) +
        glob.glob(os.path.join(fits_dir, "*.fts"))
    )
    if max_frames:
        files = files[:max_frames]
    frames = []
    for i, path in enumerate(files):
        if log_callback and i % 50 == 0:
            log_callback(f"  Loading FITS {i+1}/{len(files)}: "
                         f"{os.path.basename(path)}")
        try:
            data = astropy_fits.getdata(path).astype(np.float32)
            if data.ndim == 3:
                data = data[0]
            frames.append(data)
        except Exception as e:
            if log_callback:
                log_callback(f"  WARNING: Could not load {path}: {e}")
    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class SpeckleWorker(QThread):
    progress     = pyqtSignal(int, int, str)
    log_line     = pyqtSignal(str)
    psf_preview  = pyqtSignal(object)
    strehl_data  = pyqtSignal(list)
    mean_ready   = pyqtSignal(object)
    result_ready = pyqtSignal(object)
    finished     = pyqtSignal(dict)

    def __init__(self, config, cancel_event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg = self.config

            # 1 — Load frames
            self.progress.emit(1, 7, "Loading frames...")
            frames = self._load_frames(cfg)
            n = len(frames)
            if n == 0:
                self.finished.emit({"success": False,
                                    "error": "No frames could be loaded"})
                return
            self.log_line.emit(f"Loaded {n} frames: "
                               f"{frames[0].shape[1]}×{frames[0].shape[0]} px")
            if self._cancel.is_set():
                self._abort(); return

            # 2 — Mean stack (for comparison preview)
            self.log_line.emit("Computing mean stack (for comparison)...")
            mean_stack = compute_mean_stack(frames)
            self.mean_ready.emit(mean_stack)

            # 3 — Detect reference stars
            self.progress.emit(2, 7, "Detecting reference stars...")
            n_psf_stars = cfg.get("n_psf_stars", 3)
            psf_box     = cfg.get("psf_box_size", 64)
            manual_stars = cfg.get("manual_stars", [])

            if manual_stars:
                first_positions = [(y, x) for y, x in manual_stars]
                self.log_line.emit(f"Using {len(first_positions)} manually selected stars")
            else:
                first_positions = detect_reference_stars(
                    frames[0], n_stars=n_psf_stars, psf_box_size=psf_box,
                    threshold_sigma=cfg.get("star_threshold", 8.0))
                if not first_positions:
                    self.finished.emit({
                        "success": False,
                        "error": "No reference stars detected in first frame. "
                                 "Try lowering the detection threshold or "
                                 "use manual star selection."})
                    return
                self.log_line.emit(f"Found {len(first_positions)} reference stars "
                                   f"in first frame")
                for i, (y, x) in enumerate(first_positions):
                    self.log_line.emit(f"  Star {i+1}: ({x:.1f}, {y:.1f})")
            if self._cancel.is_set():
                self._abort(); return

            # 4 — Track stars
            self.progress.emit(3, 7, "Tracking reference stars...")
            if cfg.get("track_stars", True) and n > 1:
                all_positions = track_stars_across_sequence(
                    frames, first_positions,
                    search_radius=cfg.get("tracking_radius", 20),
                    log_callback=self.log_line.emit)
            else:
                all_positions = [first_positions] * n
            self.log_line.emit("Star tracking complete")
            if self._cancel.is_set():
                self._abort(); return

            # 5 — Strehl
            self.progress.emit(4, 7, "Computing frame quality (Strehl)...")
            strehl_values = []
            for i, (frame, positions) in enumerate(zip(frames, all_positions)):
                sv = estimate_strehl(frame, positions[0]) if positions else 0.0
                strehl_values.append(sv)
            self.strehl_data.emit(strehl_values)
            best_strehl   = max(strehl_values) if strehl_values else 0
            median_strehl = float(np.median(strehl_values)) if strehl_values else 0
            self.log_line.emit(f"Strehl: best={best_strehl:.3f}  "
                               f"median={median_strehl:.3f}")
            if self._cancel.is_set():
                self._abort(); return

            # 6 — Sample PSF
            self.progress.emit(5, 7, "Extracting PSFs...")
            sample_psfs = []
            for i in range(min(20, n)):
                if all_positions[i]:
                    psf = extract_psf_from_frame(
                        frames[i], all_positions[i][0],
                        box_size=psf_box, normalize=True)
                    if psf is not None:
                        sample_psfs.append(psf)
            if sample_psfs:
                avg_psf = np.mean(sample_psfs, axis=0)
                self.psf_preview.emit(avg_psf)
                self.log_line.emit(f"PSF extracted from {len(sample_psfs)} sample frames")
            if self._cancel.is_set():
                self._abort(); return

            # 7 — Reconstruct
            self.progress.emit(6, 7, "Running speckle holography...")
            self.log_line.emit(f"Reconstructing from {n} frames "
                               f"(epsilon={cfg.get('epsilon', 0.01)})...")
            result = speckle_holography_reconstruct(
                frames, all_positions,
                psf_box_size=psf_box,
                epsilon=cfg.get("epsilon", 0.01),
                log_callback=self.log_line.emit,
                cancel_event=self._cancel)
            if result is None:
                self.finished.emit({"success": False,
                                    "error": "Reconstruction failed — all frames skipped"})
                return
            if self._cancel.is_set():
                self._abort(); return

            # 8 — Save
            self.progress.emit(7, 7, "Saving output...")
            cfg["n_frames_loaded"] = n
            out_path = self._save_result(result, cfg)
            if cfg.get("save_mean_stack", False):
                mean_path = out_path.replace(".fit", "_mean_stack.fit")
                hdu = astropy_fits.PrimaryHDU(data=mean_stack)
                hdu.writeto(mean_path, overwrite=True)
                self.log_line.emit(f"Mean stack saved: {mean_path}")
            self.log_line.emit(f"Saved: {out_path}")
            self.result_ready.emit(result)

            # Load in Siril if requested
            if cfg.get("load_in_siril", False):
                try:
                    siril = s.SirilInterface()
                    siril.connect()
                    siril.cmd("load", out_path)
                    siril.disconnect()
                    self.log_line.emit("Loaded result in Siril")
                except Exception as e:
                    self.log_line.emit(f"  (Could not auto-load in Siril: {e})")

            self.finished.emit({
                "success":        True,
                "output_path":    out_path,
                "n_frames_used":  n - sum(1 for sv in strehl_values if sv == 0),
                "n_frames_total": n,
                "best_strehl":    round(best_strehl, 3),
                "median_strehl":  round(median_strehl, 3),
                "epsilon":        cfg.get("epsilon", 0.01),
            })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _load_frames(self, cfg):
        input_type = cfg.get("input_type", "fits_dir")
        max_frames = cfg.get("max_frames") or None
        if input_type == "ser":
            return load_ser_as_frames(cfg["input_path"],
                                      max_frames=max_frames,
                                      log_callback=self.log_line.emit)
        return load_fits_sequence(cfg["input_path"],
                                  max_frames=max_frames,
                                  log_callback=self.log_line.emit)

    def _save_result(self, result, cfg):
        out_dir = cfg.get("output_dir", os.path.dirname(cfg["input_path"]))
        os.makedirs(out_dir, exist_ok=True)
        name    = cfg.get("output_name", "speckle_holography_result.fit")
        path    = os.path.join(out_dir, name)
        hdu     = astropy_fits.PrimaryHDU(data=result.astype(np.float32))
        hdu.header["HISTORY"] = "Speckle holography — speckle_holography.py"
        hdu.header["NFRAMES"] = cfg.get("n_frames_loaded", 0)
        hdu.header["EPSILON"] = cfg.get("epsilon", 0.01)
        hdu.writeto(path, overwrite=True)
        return path

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()


# ══════════════════════════════════════════════════════════════════════════════
#  CANVAS WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class StarPickerCanvas(FigureCanvasQTAgg):
    stars_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), facecolor="#1e2128")
        self.ax  = self.fig.add_subplot(111)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)
        self.star_positions = []
        self.frame_data     = None
        self.mpl_connect("button_press_event", self._on_click)

    def _style(self):
        self.ax.set_facecolor("#1e2128")
        self.ax.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_color("#3a4055")

    def show_frame(self, frame, auto_stars=None):
        self.frame_data = frame
        if auto_stars is not None:
            self.star_positions = list(auto_stars)
        self._redraw()

    def _redraw(self):
        if self.frame_data is None:
            return
        self.ax.clear()
        self._style()
        p_lo = np.percentile(self.frame_data, 0.5)
        p_hi = np.percentile(self.frame_data, 99.5)
        disp = np.clip((self.frame_data.astype(float) - p_lo) /
                       (p_hi - p_lo + 1e-10), 0, 1)
        self.ax.imshow(disp, cmap="gray", origin="lower",
                       aspect="equal", interpolation="nearest")
        for i, (y, x) in enumerate(self.star_positions):
            circle = matplotlib.patches.Circle(
                (x, y), radius=15, fill=False,
                edgecolor="#4caf7d", linewidth=1.5)
            self.ax.add_patch(circle)
            self.ax.text(x + 17, y, str(i + 1),
                         color="#4caf7d", fontsize=7)
        self.ax.set_title(
            f"Reference stars: {len(self.star_positions)} selected  "
            f"(left-click=add, right-click=remove)",
            color="#5ba3ff", fontsize=8)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        cx, cy = event.xdata, event.ydata
        if event.button == 3:
            if self.star_positions:
                dists = [((cy - y)**2 + (cx - x)**2)**0.5
                         for y, x in self.star_positions]
                idx = int(np.argmin(dists))
                if dists[idx] < 30:
                    self.star_positions.pop(idx)
                    self._redraw()
                    self.stars_changed.emit(self.star_positions)
        else:
            self.star_positions.append((cy, cx))
            self._redraw()
            self.stars_changed.emit(self.star_positions)


class ResultCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig      = Figure(figsize=(9, 4), facecolor="#1e2128")
        self.ax_mean  = self.fig.add_subplot(1, 2, 1)
        self.ax_holo  = self.fig.add_subplot(1, 2, 2)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style(self):
        for ax, title, color in [
            (self.ax_mean, "Simple mean stack",         "#7a8499"),
            (self.ax_holo, "Speckle holography result", "#4a9eff"),
        ]:
            ax.set_facecolor("#1e2128")
            ax.set_title(title, color=color, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")

    def _disp(self, arr):
        p_lo = np.percentile(arr, 0.5)
        p_hi = np.percentile(arr, 99.9)
        return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-10), 0, 1)

    def _ds(self, arr, max_px=500):
        h, w = arr.shape
        f    = max(1, max(h, w) // max_px)
        return arr[::f, ::f]

    def show_mean(self, mean_stack):
        self.ax_mean.clear()
        self.ax_mean.imshow(self._ds(self._disp(mean_stack)),
                            cmap="gray", origin="lower",
                            aspect="equal", interpolation="nearest")
        self.ax_mean.set_title("Simple mean stack", color="#7a8499", fontsize=9)
        for spine in self.ax_mean.spines.values():
            spine.set_color("#3a4055")
        self.ax_mean.tick_params(left=False, bottom=False,
                                 labelleft=False, labelbottom=False)
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def show_holography(self, result):
        self.ax_holo.clear()
        self.ax_holo.imshow(self._ds(self._disp(result)),
                            cmap="gray", origin="lower",
                            aspect="equal", interpolation="nearest")
        self.ax_holo.set_title("Speckle holography result",
                               color="#4a9eff", fontsize=9)
        for spine in self.ax_holo.spines.values():
            spine.set_color("#3a4055")
        self.ax_holo.tick_params(left=False, bottom=False,
                                 labelleft=False, labelbottom=False)
        self.fig.tight_layout(pad=0.4)
        self.draw()


class PSFCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig        = Figure(figsize=(7, 3), facecolor="#1e2128")
        self.ax_psf     = self.fig.add_subplot(1, 2, 1)
        self.ax_profile = self.fig.add_subplot(1, 2, 2)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)

    def _style(self):
        for ax in [self.ax_psf, self.ax_profile]:
            ax.set_facecolor("#252930")
            ax.tick_params(colors="#dde3ee", labelsize=7)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_color("#3a4055")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    def show_psf(self, psf):
        self.ax_psf.clear()
        self.ax_psf.imshow(psf, cmap="hot", origin="lower",
                           aspect="equal", interpolation="nearest")
        self.ax_psf.set_title("Average PSF", color="#5ba3ff", fontsize=8)
        self.ax_psf.tick_params(left=False, bottom=False,
                                labelleft=False, labelbottom=False)
        # Radial profile
        self.ax_profile.clear()
        h, w   = psf.shape
        cy, cx = h // 2, w // 2
        y, x   = np.ogrid[:h, :w]
        r      = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
        r_max  = min(h, w) // 2
        radial = np.array([
            psf[r == i].mean() if np.sum(r == i) > 0 else 0
            for i in range(r_max)
        ])
        self.ax_profile.plot(radial / (radial.max() + 1e-30),
                             color="#4a9eff", linewidth=1.5)
        self.ax_profile.set_xlabel("Radius (px)", color="#7a8499", fontsize=7)
        self.ax_profile.set_ylabel("Relative intensity", color="#7a8499", fontsize=7)
        self.ax_profile.set_title("PSF radial profile", color="#5ba3ff", fontsize=8)
        self.ax_profile.set_facecolor("#252930")
        for spine in ["bottom", "left"]:
            self.ax_profile.spines[spine].set_color("#3a4055")
        for spine in ["top", "right"]:
            self.ax_profile.spines[spine].set_visible(False)
        # Strehl annotation
        peak  = float(psf.max())
        total = float(psf.sum())
        if total > 0:
            strehl_avg = peak / total
            self.ax_psf.set_xlabel(f"PSF Strehl ≈ {strehl_avg:.3f}",
                                   color="#7a8499", fontsize=7)
        self.fig.tight_layout(pad=0.4)
        self.draw()


class StrehlCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3), facecolor="#1e2128")
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def show_strehl(self, values):
        self.ax.clear()
        self.ax.set_facecolor("#252930")
        if not values:
            self.draw()
            return
        arr  = np.array(values)
        med  = float(np.median(arr))
        best = float(arr.max())
        self.ax.hist(arr, bins=30, color="#4a9eff", alpha=0.8, edgecolor="none")
        self.ax.axvline(med, color="#e8a23a", linewidth=1.5,
                        linestyle="--", label=f"Median: {med:.3f}")
        self.ax.axvline(best, color="#4caf7d", linewidth=1.5,
                        linestyle="--", label=f"Best: {best:.3f}")
        self.ax.set_xlabel("Relative Strehl ratio", color="#7a8499", fontsize=8)
        self.ax.set_ylabel("Frame count", color="#7a8499", fontsize=8)
        self.ax.set_title(
            f"Frame quality distribution — ALL {len(values)} frames used",
            color="#5ba3ff", fontsize=8)
        self.ax.legend(facecolor="#2d3240", edgecolor="#3a4055",
                       labelcolor="#dde3ee", fontsize=7)
        self.ax.tick_params(colors="#dde3ee", labelsize=7)
        for spine in ["bottom", "left"]:
            self.ax.spines[spine].set_color("#3a4055")
        for spine in ["top", "right"]:
            self.ax.spines[spine].set_visible(False)
        self.fig.tight_layout(pad=0.4)
        self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speckle Holography — Siril")
        self.resize(1280, 820)
        self._worker      = None
        self._cancel_evt  = threading.Event()
        self._frames      = []
        self._last_result = None
        self._last_mean   = None
        self._auto_stars  = []
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setSpacing(0)
        vlay.setContentsMargins(0, 0, 0, 0)

        # ── Title bar ──────────────────────────────────────────────────────
        title_bar = QWidget()
        title_bar.setFixedHeight(42)
        title_bar.setStyleSheet(
            f"background:{SIRIL_BG3}; border-bottom:1px solid {SIRIL_BORDER};")
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(14, 0, 14, 0)
        lbl_title = QLabel("✨  Speckle Holography  —  Siril")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_ACCENT}; font-weight:bold; font-size:13pt;")
        lbl_ver = QLabel("v1.0  |  100% frame usage")
        lbl_ver.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:9pt;")
        tb_lay.addWidget(lbl_title)
        tb_lay.addStretch()
        tb_lay.addWidget(lbl_ver)
        vlay.addWidget(title_bar)

        # ── Splitter ───────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        vlay.addWidget(splitter, 1)

        # LEFT panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(370)
        left_inner  = QWidget()
        left_lay    = QVBoxLayout(left_inner)
        left_lay.setSpacing(6)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)

        self._build_input_group(left_lay)
        self._build_stars_group(left_lay)
        self._build_recon_group(left_lay)
        self._build_output_group(left_lay)
        left_lay.addStretch()

        # RIGHT panel — tabs
        right_w   = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(right_w)

        self.tabs = QTabWidget()
        right_lay.addWidget(self.tabs)

        self._build_tab_preview()
        self._build_tab_quality()
        self._build_tab_psf()
        self._build_tab_result()
        self._build_tab_log()

        # ── Bottom row ─────────────────────────────────────────────────────
        bot_w   = QWidget()
        bot_w.setFixedHeight(52)
        bot_w.setStyleSheet(
            f"background:{SIRIL_BG3}; border-top:1px solid {SIRIL_BORDER};")
        bot_lay = QHBoxLayout(bot_w)
        bot_lay.setContentsMargins(12, 6, 12, 6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 7)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)

        self.btn_run = QPushButton("▶  Reconstruct")
        self.btn_run.setObjectName("primary")
        self.btn_run.setFixedWidth(160)
        self.btn_run.clicked.connect(self._start_worker)

        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setFixedWidth(100)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_worker)

        bot_lay.addWidget(self.progress_bar, 1)
        bot_lay.addWidget(self.btn_run)
        bot_lay.addWidget(self.btn_cancel)
        vlay.addWidget(bot_w)

        # ── Status bar ─────────────────────────────────────────────────────
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-size:9pt; "
            f"padding:2px 12px; "
            f"border-top:1px solid {SIRIL_BORDER};")
        self.status_lbl.setFixedHeight(22)
        vlay.addWidget(self.status_lbl)

    # ─── Input group ──────────────────────────────────────────────────────────
    def _build_input_group(self, parent_lay):
        grp = QGroupBox("Input")
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self.cmb_input_type = QComboBox()
        self.cmb_input_type.addItems(["FITS sequence folder", "SER file"])
        row1.addWidget(self.cmb_input_type, 1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.le_input = QLineEdit()
        self.le_input.setPlaceholderText("Select input path…")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.setToolTip("Browse")
        btn_browse.clicked.connect(self._browse_input)
        row2.addWidget(self.le_input, 1)
        row2.addWidget(btn_browse)
        lay.addLayout(row2)

        self.btn_analyze = QPushButton("🔍  Load & analyze")
        self.btn_analyze.clicked.connect(self._analyze_input)
        lay.addWidget(self.btn_analyze)

        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("dim")
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.lbl_info)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Max frames:"))
        self.spn_max_frames = QSpinBox()
        self.spn_max_frames.setRange(0, 99999)
        self.spn_max_frames.setValue(0)
        self.spn_max_frames.setToolTip("0 = use all frames")
        row3.addWidget(self.spn_max_frames)
        lbl_hint = QLabel("(0 = all)")
        lbl_hint.setObjectName("dim")
        row3.addWidget(lbl_hint)
        lay.addLayout(row3)

        parent_lay.addWidget(grp)

    # ─── Reference stars group ────────────────────────────────────────────────
    def _build_stars_group(self, parent_lay):
        grp = QGroupBox("Reference Stars (PSF Source)")
        lay = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.cmb_star_mode = QComboBox()
        self.cmb_star_mode.addItems(["Auto-detect", "Manual selection"])
        self.cmb_star_mode.currentIndexChanged.connect(self._star_mode_changed)
        row.addWidget(self.cmb_star_mode, 1)
        lay.addLayout(row)

        # Auto panel
        self.auto_panel = QWidget()
        ap_lay = QFormLayout(self.auto_panel)
        ap_lay.setContentsMargins(0, 0, 0, 0)

        self.spn_n_stars = QSpinBox()
        self.spn_n_stars.setRange(1, 10)
        self.spn_n_stars.setValue(3)
        ap_lay.addRow("Stars to use:", self.spn_n_stars)

        self.dsb_threshold = QDoubleSpinBox()
        self.dsb_threshold.setRange(3.0, 20.0)
        self.dsb_threshold.setValue(8.0)
        self.dsb_threshold.setSingleStep(0.5)
        ap_lay.addRow("Detection threshold (σ):", self.dsb_threshold)

        self.cmb_psf_box = QComboBox()
        self.cmb_psf_box.addItems(["32 px", "64 px", "128 px"])
        self.cmb_psf_box.setCurrentIndex(1)
        ap_lay.addRow("PSF box size:", self.cmb_psf_box)
        lay.addWidget(self.auto_panel)

        self.btn_detect = QPushButton("🔍  Detect stars in first frame")
        self.btn_detect.clicked.connect(self._detect_stars)
        lay.addWidget(self.btn_detect)

        self.lbl_detect_result = QLabel("")
        self.lbl_detect_result.setObjectName("dim")
        lay.addWidget(self.lbl_detect_result)

        # Manual panel
        self.manual_panel = QWidget()
        mp_lay = QVBoxLayout(self.manual_panel)
        mp_lay.setContentsMargins(0, 0, 0, 0)
        lbl_m = QLabel("Click stars in Preview tab (Tab 1)")
        lbl_m.setObjectName("dim")
        mp_lay.addWidget(lbl_m)
        self.lbl_manual_count = QLabel("0 stars selected")
        mp_lay.addWidget(self.lbl_manual_count)
        self.manual_panel.setVisible(False)
        lay.addWidget(self.manual_panel)

        parent_lay.addWidget(grp)

    # ─── Reconstruction group ─────────────────────────────────────────────────
    def _build_recon_group(self, parent_lay):
        grp = QGroupBox("Reconstruction Parameters")
        lay = QFormLayout(grp)

        eps_widget = QWidget()
        eps_lay    = QVBoxLayout(eps_widget)
        eps_lay.setContentsMargins(0, 0, 0, 0)

        slider_row = QHBoxLayout()
        self.sld_epsilon = QSlider(Qt.Orientation.Horizontal)
        self.sld_epsilon.setRange(1, 100)
        self.sld_epsilon.setValue(10)   # 0.010
        self.lbl_epsilon_val = QLabel("0.010")
        self.lbl_epsilon_val.setFixedWidth(44)
        slider_row.addWidget(self.sld_epsilon)
        slider_row.addWidget(self.lbl_epsilon_val)
        eps_lay.addLayout(slider_row)

        self.lbl_epsilon_guide = QLabel("")
        self.lbl_epsilon_guide.setObjectName("dim")
        self.lbl_epsilon_guide.setWordWrap(True)
        eps_lay.addWidget(self.lbl_epsilon_guide)
        self.sld_epsilon.valueChanged.connect(self._epsilon_changed)
        self._epsilon_changed(10)

        lay.addRow("Wiener epsilon:", eps_widget)

        self.chk_track = QCheckBox("Track stars between frames")
        self.chk_track.setChecked(True)
        self.chk_track.setToolTip("Handles small shifts between frames")
        lay.addRow("", self.chk_track)

        self.spn_radius = QSpinBox()
        self.spn_radius.setRange(5, 50)
        self.spn_radius.setValue(20)
        lay.addRow("Tracking radius (px):", self.spn_radius)

        parent_lay.addWidget(grp)

    # ─── Output group ─────────────────────────────────────────────────────────
    def _build_output_group(self, parent_lay):
        grp = QGroupBox("Output")
        lay = QFormLayout(grp)

        out_row = QHBoxLayout()
        self.le_out_dir = QLineEdit()
        self.le_out_dir.setPlaceholderText("Same as input")
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(32)
        btn_out.clicked.connect(self._browse_output)
        out_row.addWidget(self.le_out_dir, 1)
        out_row.addWidget(btn_out)
        lay.addRow("Output folder:", out_row)

        self.le_out_name = QLineEdit("speckle_holography_result.fit")
        lay.addRow("Filename:", self.le_out_name)

        self.chk_save_mean = QCheckBox("Also save simple mean stack")
        lay.addRow("", self.chk_save_mean)

        self.chk_load_siril = QCheckBox("Load result in Siril after processing")
        self.chk_load_siril.setChecked(True)
        lay.addRow("", self.chk_load_siril)

        parent_lay.addWidget(grp)

    # ─── Tabs ─────────────────────────────────────────────────────────────────
    def _build_tab_preview(self):
        w   = QWidget()
        lay = QVBoxLayout(w)

        self.star_canvas = StarPickerCanvas()
        self.star_canvas.stars_changed.connect(self._on_manual_stars_changed)
        lay.addWidget(self.star_canvas, 1)

        lbl = QLabel("Green circles = reference stars for PSF  |  "
                     "Left-click to add, right-click to remove in Manual mode")
        lbl.setObjectName("dim")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)

        self.tabs.addTab(w, "🖼  Preview & Stars")

    def _build_tab_quality(self):
        w   = QWidget()
        lay = QVBoxLayout(w)

        self.strehl_canvas = StrehlCanvas()
        lay.addWidget(self.strehl_canvas, 1)

        stats_grp = QGroupBox("Statistics")
        stats_lay = QHBoxLayout(stats_grp)
        self.lbl_stat_total  = self._badge("Total frames: —")
        self.lbl_stat_used   = self._badge("Frames used: —")
        self.lbl_stat_best   = self._badge("Best Strehl: —")
        self.lbl_stat_median = self._badge("Median Strehl: —")
        for lbl in [self.lbl_stat_total, self.lbl_stat_used,
                    self.lbl_stat_best, self.lbl_stat_median]:
            stats_lay.addWidget(lbl)
        lay.addWidget(stats_grp)

        lbl = QLabel("All frames used — bad frames contribute less automatically")
        lbl.setObjectName("dim")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)

        self.tabs.addTab(w, "📊  Frame Quality")

    def _build_tab_psf(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        self.psf_canvas = PSFCanvas()
        lay.addWidget(self.psf_canvas, 1)
        lbl = QLabel("Average PSF from first 20 frames")
        lbl.setObjectName("dim")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        self.tabs.addTab(w, "🔬  PSF")

    def _build_tab_result(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        self.result_canvas = ResultCanvas()
        lay.addWidget(self.result_canvas, 1)
        btn_row = QHBoxLayout()
        btn_save = QPushButton("💾  Save result")
        btn_save.setObjectName("export")
        btn_save.clicked.connect(self._save_result_dialog)
        btn_load = QPushButton("Load in Siril")
        btn_load.setObjectName("export")
        btn_load.clicked.connect(self._load_in_siril)
        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_load)
        btn_row.addStretch()
        lay.addLayout(btn_row)
        self.tabs.addTab(w, "✨  Result")

    def _build_tab_log(self):
        w   = QWidget()
        lay = QVBoxLayout(w)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setProperty("readOnly", True)
        lay.addWidget(self.log_text)
        self.tabs.addTab(w, "📋  Log")

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def _badge(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"background:{SIRIL_BG3}; color:{SIRIL_TEXT}; "
            f"border:1px solid {SIRIL_BORDER}; border-radius:3px; "
            f"padding:4px 8px; font-size:9pt;")
        return lbl

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    def _set_status(self, msg, color=None):
        c = color or SIRIL_TEXT_DIM
        self.status_lbl.setStyleSheet(
            f"color:{c}; font-size:9pt; padding:2px 12px; "
            f"border-top:1px solid {SIRIL_BORDER};")
        self.status_lbl.setText(msg)

    def _epsilon_from_slider(self, v):
        return round(v / 1000.0, 4)

    def _epsilon_changed(self, v):
        eps = v / 100.0 * 0.099 + 0.001   # map 1-100 → 0.001-0.1
        eps = round(eps, 4)
        self.lbl_epsilon_val.setText(f"{eps:.3f}")
        for (lo, hi), (label, desc) in EPSILON_GUIDE.items():
            if lo <= eps < hi:
                self.lbl_epsilon_guide.setText(f"{label}: {desc}")
                break

    def _get_epsilon(self):
        v   = self.sld_epsilon.value()
        return round(v / 100.0 * 0.099 + 0.001, 4)

    def _get_psf_box_size(self):
        return [32, 64, 128][self.cmb_psf_box.currentIndex()]

    def _star_mode_changed(self, idx):
        is_auto = (idx == 0)
        self.auto_panel.setVisible(is_auto)
        self.btn_detect.setVisible(is_auto)
        self.lbl_detect_result.setVisible(is_auto)
        self.manual_panel.setVisible(not is_auto)

    def _on_manual_stars_changed(self, positions):
        n = len(positions)
        self.lbl_manual_count.setText(f"{n} star{'s' if n != 1 else ''} selected")

    # ─── Browse ───────────────────────────────────────────────────────────────
    def _browse_input(self):
        if self.cmb_input_type.currentIndex() == 0:
            path = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select SER file", "", "SER files (*.ser);;All files (*)")
        if path:
            self.le_input.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.le_out_dir.setText(path)

    # ─── Analyze input ────────────────────────────────────────────────────────
    def _analyze_input(self):
        path = self.le_input.text().strip()
        if not path:
            QMessageBox.warning(self, "No input", "Please select an input path.")
            return
        self.lbl_info.setText("Analyzing…")
        QApplication.processEvents()
        try:
            if self.cmb_input_type.currentIndex() == 0:
                files = (glob.glob(os.path.join(path, "*.fit")) +
                         glob.glob(os.path.join(path, "*.fits")) +
                         glob.glob(os.path.join(path, "*.fts")))
                n = len(files)
                if n == 0:
                    self.lbl_info.setText("No FITS files found")
                    return
                data = astropy_fits.getdata(files[0]).astype(np.float32)
                if data.ndim == 3:
                    data = data[0]
                h, w = data.shape
                mem  = n * h * w * 4 / 1024**3
                self.lbl_info.setText(
                    f"Found {n} frames  |  {w}×{h} px  |  "
                    f"~{mem:.1f} GB RAM needed")
                self._frames = [data]   # preview frame only
                self.star_canvas.show_frame(data)
            else:
                hdr = read_ser_header(path)
                n   = hdr["frame_count"]
                w   = hdr["width"]
                h   = hdr["height"]
                mem = n * h * w * 4 / 1024**3
                self.lbl_info.setText(
                    f"SER: {n} frames  |  {w}×{h} px  |  "
                    f"~{mem:.1f} GB RAM needed")
                with open(path, "rb") as f:
                    data = read_ser_frame(f, hdr, 0).astype(np.float32)
                self._frames = [data]
                self.star_canvas.show_frame(data)
        except Exception as e:
            self.lbl_info.setText(f"Error: {e}")

    # ─── Detect stars ─────────────────────────────────────────────────────────
    def _detect_stars(self):
        if not self._frames:
            QMessageBox.warning(self, "No data", "Load & analyze first.")
            return
        self.lbl_detect_result.setText("Detecting…")
        QApplication.processEvents()
        try:
            stars = detect_reference_stars(
                self._frames[0],
                n_stars=self.spn_n_stars.value(),
                psf_box_size=self._get_psf_box_size(),
                threshold_sigma=self.dsb_threshold.value())
            self._auto_stars = stars
            self.star_canvas.show_frame(self._frames[0], auto_stars=stars)
            self.lbl_detect_result.setText(f"Found {len(stars)} stars")
        except Exception as e:
            self.lbl_detect_result.setText(f"Error: {e}")

    # ─── Worker control ───────────────────────────────────────────────────────
    def _start_worker(self):
        path = self.le_input.text().strip()
        if not path:
            QMessageBox.warning(self, "No input", "Please select an input path.")
            return

        is_ser     = (self.cmb_input_type.currentIndex() == 1)
        is_manual  = (self.cmb_star_mode.currentIndex() == 1)
        manual_stars = (self.star_canvas.star_positions
                        if is_manual else [])

        out_dir  = self.le_out_dir.text().strip() or os.path.dirname(path)
        out_name = self.le_out_name.text().strip() or "speckle_holography_result.fit"

        cfg = {
            "input_type":      "ser" if is_ser else "fits_dir",
            "input_path":      path,
            "output_dir":      out_dir,
            "output_name":     out_name,
            "max_frames":      self.spn_max_frames.value(),
            "n_psf_stars":     self.spn_n_stars.value(),
            "psf_box_size":    self._get_psf_box_size(),
            "star_threshold":  self.dsb_threshold.value(),
            "epsilon":         self._get_epsilon(),
            "track_stars":     self.chk_track.isChecked(),
            "tracking_radius": self.spn_radius.value(),
            "manual_stars":    manual_stars,
            "save_mean_stack": self.chk_save_mean.isChecked(),
            "load_in_siril":   self.chk_load_siril.isChecked(),
        }

        self._cancel_evt = threading.Event()
        self._worker     = SpeckleWorker(cfg, self._cancel_evt)
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.psf_preview.connect(self.psf_canvas.show_psf)
        self._worker.strehl_data.connect(self._on_strehl)
        self._worker.mean_ready.connect(self.result_canvas.show_mean)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_finished)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_status("Processing…", SIRIL_WARNING)
        self.tabs.setCurrentIndex(4)   # go to Log tab
        self._worker.start()

    def _cancel_worker(self):
        if self._worker and self._worker.isRunning():
            self._cancel_evt.set()
            self._worker.cancel()
            self._set_status("Cancelling…", SIRIL_WARNING)

    # ─── Worker signals ───────────────────────────────────────────────────────
    def _on_progress(self, step, total, msg):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(step)
        self._set_status(msg, SIRIL_TEXT_DIM)

    def _on_strehl(self, values):
        self.strehl_canvas.show_strehl(values)
        if values:
            arr = np.array(values)
            n   = len(values)
            used = sum(1 for v in values if v > 0)
            self.lbl_stat_total.setText(f"Total frames: {n}")
            self.lbl_stat_used.setText(f"Frames used: {used} (100%)")
            self.lbl_stat_best.setText(f"Best Strehl: {arr.max():.3f}")
            self.lbl_stat_median.setText(f"Median Strehl: {float(np.median(arr)):.3f}")

    def _on_result_ready(self, result):
        self._last_result = result
        self.result_canvas.show_holography(result)
        self.tabs.setCurrentIndex(3)   # switch to Result tab

    def _on_finished(self, info):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

        if info.get("success"):
            path  = info.get("output_path", "")
            n_u   = info.get("n_frames_used", 0)
            n_t   = info.get("n_frames_total", 0)
            bs    = info.get("best_strehl", 0)
            self._set_status(
                f"✓  Done  |  {n_u} frames used ({100*n_u//max(n_t,1)}%)  "
                f"|  Best Strehl: {bs:.3f}  |  Saved: {path}",
                SIRIL_SUCCESS)
            self._log(f"✓ Complete — {path}")
        else:
            err = info.get("error", "Unknown error")
            self._set_status(f"✗  {err}", SIRIL_ERROR)
            self._log(f"✗ Failed: {err}")

    # ─── Result buttons ───────────────────────────────────────────────────────
    def _save_result_dialog(self):
        if self._last_result is None:
            QMessageBox.information(self, "No result", "Run reconstruction first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", "speckle_holography_result.fit",
            "FITS files (*.fit *.fits);;All files (*)")
        if path:
            hdu = astropy_fits.PrimaryHDU(data=self._last_result)
            hdu.writeto(path, overwrite=True)
            self._log(f"Saved: {path}")

    def _load_in_siril(self):
        if self._last_result is None:
            QMessageBox.information(self, "No result", "Run reconstruction first.")
            return
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "speckle_holo_temp.fit")
        hdu = astropy_fits.PrimaryHDU(data=self._last_result)
        hdu.writeto(tmp, overwrite=True)
        try:
            siril = s.SirilInterface()
            siril.connect()
            siril.cmd("load", tmp)
            siril.disconnect()
            self._log("Loaded in Siril")
        except Exception as e:
            self._log(f"Could not load in Siril: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
