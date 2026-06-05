#!/usr/bin/env python3
"""
Lucky Imaging Video Preprocessor for Siril — #50 from project list
Converts raw video to SER, ranks frames by quality, exports best frames.
Cross-platform PIPP replacement. Feeds: #25 Speckle, #26 DIPLI, #14 Thresher.
"""

import os
import sys
import json
import struct
import threading
import subprocess
import glob
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("scipy")
s.ensure_installed("matplotlib")
s.ensure_installed("astropy")

import numpy as np
from scipy.ndimage import laplace, gaussian_filter
from scipy.optimize import curve_fit

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox,
    QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QTabWidget, QSizePolicy, QListWidget,
    QListWidgetItem, QSlider, QSpinBox, QDoubleSpinBox, QSplitter,
    QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ── Equipment manager (optional) ───────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from equipment_manager import load_all_profiles, get_profile, load_settings, save_settings
    HAS_EQUIPMENT_MANAGER = True
except ImportError:
    HAS_EQUIPMENT_MANAGER = False

# ── Persistent local settings (ffmpeg path, etc.) ─────────────────────────────
# Uses equipment_manager's siril_global_settings.json when available,
# otherwise falls back to lucky_preprocessor_settings.json next to this script.

_FALLBACK_SETTINGS_FILE = os.path.join(
    SCRIPT_DIR, "lucky_preprocessor_settings.json")

def _load_local_settings() -> dict:
    if HAS_EQUIPMENT_MANAGER:
        try:
            return load_settings()          # reads siril_global_settings.json
        except Exception:
            pass
    try:
        with open(_FALLBACK_SETTINGS_FILE, "r", encoding="utf-8") as _f:
            return json.load(_f)
    except Exception:
        return {}

def _save_local_settings(settings: dict):
    if HAS_EQUIPMENT_MANAGER:
        try:
            save_settings(settings)
            return
        except Exception:
            pass
    try:
        # Merge with existing so we don't clobber unrelated keys
        existing = {}
        if os.path.exists(_FALLBACK_SETTINGS_FILE):
            with open(_FALLBACK_SETTINGS_FILE, "r", encoding="utf-8") as _f:
                existing = json.load(_f)
        existing.update(settings)
        with open(_FALLBACK_SETTINGS_FILE, "w", encoding="utf-8") as _f:
            json.dump(existing, _f, indent=2)
    except Exception:
        pass


# ── Siril theme ────────────────────────────────────────────────────────────────

SIRIL_BG        = "#1e2128"
SIRIL_BG2       = "#252930"
SIRIL_BG3       = "#2d3240"
SIRIL_ACCENT    = "#4a9eff"
SIRIL_ACCENT2   = "#2d6abf"
SIRIL_TEXT      = "#dde3ee"
SIRIL_TEXT_DIM  = "#7a8499"
SIRIL_BORDER    = "#3a4055"
SIRIL_SUCCESS   = "#4caf7d"
SIRIL_WARNING   = "#e8a23a"
SIRIL_SECTION   = "#5ba3ff"
SIRIL_ERROR     = "#cc4444"

SIRIL_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 9pt;
}}
QTabWidget::pane {{
    border: 1px solid {SIRIL_BORDER};
    background: {SIRIL_BG};
}}
QTabBar::tab {{
    background: {SIRIL_BG2};
    color: {SIRIL_TEXT_DIM};
    padding: 6px 14px;
    border: 1px solid {SIRIL_BORDER};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: {SIRIL_BG3};
    color: {SIRIL_ACCENT};
    border-bottom: 2px solid {SIRIL_ACCENT};
}}
QGroupBox {{
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: bold;
    color: {SIRIL_SECTION};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SIRIL_BG2};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    color: {SIRIL_TEXT};
    padding: 3px 6px;
    min-height: 22px;
}}
QLineEdit:focus, QComboBox:focus {{
    border-color: {SIRIL_ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {SIRIL_BG2};
    color: {SIRIL_TEXT};
    selection-background-color: {SIRIL_ACCENT2};
    border: 1px solid {SIRIL_BORDER};
}}
QPushButton {{
    background: {SIRIL_BG3};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 4px;
    color: {SIRIL_TEXT};
    padding: 5px 14px;
    min-height: 24px;
}}
QPushButton:hover {{
    background: {SIRIL_ACCENT2};
    border-color: {SIRIL_ACCENT};
    color: #ffffff;
}}
QPushButton:disabled {{
    color: {SIRIL_TEXT_DIM};
    border-color: {SIRIL_BORDER};
    background: {SIRIL_BG2};
}}
QPushButton#primary {{
    background: {SIRIL_ACCENT2};
    border-color: {SIRIL_ACCENT};
    color: #ffffff;
    font-weight: bold;
}}
QPushButton#primary:hover {{
    background: {SIRIL_ACCENT};
}}
QPushButton#danger {{
    background: #5a1a1a;
    border-color: {SIRIL_ERROR};
    color: {SIRIL_ERROR};
}}
QPushButton#danger:hover {{
    background: {SIRIL_ERROR};
    color: #ffffff;
}}
QPushButton#secondary {{
    background: transparent;
    border-color: {SIRIL_BORDER};
    color: {SIRIL_ACCENT};
}}
QCheckBox {{
    color: {SIRIL_TEXT};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {SIRIL_BORDER};
    border-radius: 2px;
    background: {SIRIL_BG2};
}}
QCheckBox::indicator:checked {{
    background: {SIRIL_ACCENT2};
    border-color: {SIRIL_ACCENT};
}}
QProgressBar {{
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    background: {SIRIL_BG2};
    color: {SIRIL_TEXT};
    text-align: center;
    height: 16px;
}}
QProgressBar::chunk {{
    background: {SIRIL_ACCENT2};
    border-radius: 2px;
}}
QPlainTextEdit {{
    background: {SIRIL_BG2};
    color: #4cff7d;
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8pt;
}}
QListWidget {{
    background: {SIRIL_BG2};
    border: 1px solid {SIRIL_BORDER};
    border-radius: 3px;
    color: {SIRIL_TEXT};
}}
QListWidget::item:selected {{
    background: {SIRIL_ACCENT2};
    color: #ffffff;
}}
QListWidget::item:hover {{
    background: {SIRIL_BG3};
}}
QSlider::groove:horizontal {{
    background: {SIRIL_BG2};
    height: 4px;
    border-radius: 2px;
    border: 1px solid {SIRIL_BORDER};
}}
QSlider::handle:horizontal {{
    background: {SIRIL_ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {SIRIL_ACCENT2};
    border-radius: 2px;
}}
QScrollBar:vertical {{
    background: {SIRIL_BG2};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {SIRIL_BG3};
    border-radius: 5px;
    min-height: 20px;
}}
QLabel#warning {{
    color: {SIRIL_WARNING};
}}
QLabel#dim {{
    color: {SIRIL_TEXT_DIM};
    font-size: 8pt;
}}
"""

# ── SER format constants ───────────────────────────────────────────────────────

SER_HEADER_SIZE        = 178
SER_FILE_ID            = b"LUCAM-RECORDER"
WINDOWS_EPOCH_OFFSET   = 116444736000000000

COLOR_ID_NAMES = {
    0:   "MONO",
    8:   "BAYER_RGGB",
    9:   "BAYER_GRBG",
    10:  "BAYER_GBRG",
    11:  "BAYER_BGGR",
    100: "RGB",
    101: "BGR",
}

# ── SER I/O ────────────────────────────────────────────────────────────────────

def read_ser_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(SER_HEADER_SIZE)
    if len(raw) < SER_HEADER_SIZE:
        raise ValueError(f"File too small to be SER: {path}")
    return {
        "raw_bytes":     raw,
        "file_id":       raw[0:14],
        "lu_id":         struct.unpack_from("<I", raw, 14)[0],
        "color_id":      struct.unpack_from("<I", raw, 18)[0],
        "little_endian": struct.unpack_from("<I", raw, 22)[0],
        "width":         struct.unpack_from("<I", raw, 26)[0],
        "height":        struct.unpack_from("<I", raw, 30)[0],
        "pixel_depth":   struct.unpack_from("<I", raw, 34)[0],
        "frame_count":   struct.unpack_from("<I", raw, 38)[0],
        "observer":      raw[42:82].rstrip(b"\x00").decode("ascii", errors="replace"),
        "instrument":    raw[82:122].rstrip(b"\x00").decode("ascii", errors="replace"),
        "telescope":     raw[122:162].rstrip(b"\x00").decode("ascii", errors="replace"),
    }

def read_ser_frame(f, header: dict, frame_index: int) -> np.ndarray:
    bpp        = 1 if header["pixel_depth"] <= 8 else 2
    frame_size = header["width"] * header["height"] * bpp
    offset     = SER_HEADER_SIZE + frame_index * frame_size
    f.seek(offset)
    raw = f.read(frame_size)
    if len(raw) < frame_size:
        raise EOFError(f"Unexpected end of file at frame {frame_index}")
    dtype = np.uint8 if bpp == 1 else np.uint16
    return np.frombuffer(raw, dtype=dtype).reshape(header["height"], header["width"])

def write_ser_file(output_path: str, frames: list, header: dict):
    if not frames:
        raise ValueError("No frames to write")
    new_header = bytearray(header["raw_bytes"])
    struct.pack_into("<I", new_header, 38, len(frames))
    with open(output_path, "wb") as f:
        f.write(bytes(new_header))
        for frame in frames:
            f.write(frame.tobytes())

def unix_to_filetime(unix_ts: float) -> int:
    return int(unix_ts * 10_000_000 + WINDOWS_EPOCH_OFFSET)

def now_filetime() -> int:
    return unix_to_filetime(datetime.now(timezone.utc).timestamp())

# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def get_ffmpeg_path() -> str | None:
    import shutil
    candidates = [
        "ffmpeg", "ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
    ]
    for c in candidates:
        found = shutil.which(c)
        if found:
            return found
        if os.path.isfile(c):
            return c
    return None

def get_video_info(video_path: str, ffmpeg: str) -> dict:
    cmd    = [ffmpeg, "-i", video_path, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info   = {"width": 0, "height": 0, "fps": 0.0, "frames": 0}
    for line in result.stderr.splitlines():
        if "Stream" in line and "Video" in line:
            m = re.search(r"(\d+)x(\d+)", line)
            if m:
                info["width"]  = int(m.group(1))
                info["height"] = int(m.group(2))
            m = re.search(r"([\d.]+) fps", line)
            if m:
                info["fps"] = float(m.group(1))
        if "frame=" in line:
            m = re.search(r"frame=\s*(\d+)", line)
            if m:
                info["frames"] = int(m.group(1))
    return info

def convert_video_to_ser(video_path: str, output_ser: str,
                         ffmpeg: str, pixel_depth: int = 16,
                         cancel_event=None, log_callback=None) -> bool:
    info = get_video_info(video_path, ffmpeg)
    if not info["width"] or not info["height"]:
        if log_callback:
            log_callback(f"  ERROR: Cannot determine dimensions: {video_path}")
        return False

    w, h         = info["width"], info["height"]
    bpp          = 1 if pixel_depth <= 8 else 2
    pix_fmt      = "gray" if bpp == 1 else "gray16le"
    frame_size   = w * h * bpp
    frames_written = 0

    header = bytearray(SER_HEADER_SIZE)
    header[0:14] = b"LUCAM-RECORDER"
    struct.pack_into("<I", header, 18, 0)           # MONO
    struct.pack_into("<I", header, 22, 1)           # little-endian
    struct.pack_into("<I", header, 26, w)
    struct.pack_into("<I", header, 30, h)
    struct.pack_into("<I", header, 34, pixel_depth)
    struct.pack_into("<I", header, 38, 0)           # frame count (updated later)
    struct.pack_into("<q", header, 162, now_filetime())
    struct.pack_into("<q", header, 170, now_filetime())

    cmd = [
        ffmpeg, "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-"
    ]

    try:
        with open(output_ser, "wb") as out_f:
            out_f.write(bytes(header))

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            while True:
                if cancel_event and cancel_event.is_set():
                    proc.terminate()
                    return False
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                out_f.write(raw)
                frames_written += 1
                if log_callback and frames_written % 200 == 0:
                    log_callback(f"  → {frames_written} frames converted...")

            proc.wait()
            out_f.seek(38)
            out_f.write(struct.pack("<I", frames_written))

        if log_callback:
            log_callback(f"  ✓ Converted {frames_written} frames → {os.path.basename(output_ser)}")
        return frames_written > 0

    except Exception as e:
        if log_callback:
            log_callback(f"  ERROR converting {os.path.basename(video_path)}: {e}")
        return False

# ── Quality metrics ────────────────────────────────────────────────────────────

def laplacian_variance(frame: np.ndarray) -> float:
    return float(np.var(laplace(frame.astype(np.float32))))

def gradient_energy(frame: np.ndarray) -> float:
    f  = frame.astype(np.float32)
    gx = np.gradient(f, axis=1)
    gy = np.gradient(f, axis=0)
    return float(np.mean(gx**2 + gy**2))

def peak_pixel(frame: np.ndarray) -> float:
    return float(np.percentile(frame, 99.9))

def normalized_variance(frame: np.ndarray) -> float:
    mean = float(np.mean(frame)) + 1e-10
    return float(np.var(frame) / mean)

def fwhm_estimate(frame: np.ndarray) -> float:
    smooth      = gaussian_filter(frame.astype(float), sigma=2)
    cy, cx      = np.unravel_index(np.argmax(smooth), smooth.shape)
    size        = 32
    y1, y2      = max(0, cy - size), min(frame.shape[0], cy + size)
    x1, x2      = max(0, cx - size), min(frame.shape[1], cx + size)
    cutout      = frame[y1:y2, x1:x2].astype(float)
    if cutout.size == 0:
        return -999.0
    profile     = cutout[cutout.shape[0] // 2, :]

    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    try:
        x    = np.arange(len(profile))
        p0   = [profile.max(), len(profile) // 2, 3.0]
        popt, _ = curve_fit(gaussian, x, profile, p0=p0, maxfev=500)
        return -2.355 * abs(popt[2])   # negate: higher = sharper
    except Exception:
        return -999.0

def adaptive_elbow(scores: list) -> float:
    if len(scores) < 4:
        return min(scores)
    sorted_scores = sorted(scores, reverse=True)
    s      = np.array(sorted_scores, dtype=float)
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10)
    n      = len(s_norm)
    line   = np.linspace(s_norm[0], s_norm[-1], n)
    elbow_idx = int(np.argmax(np.abs(s_norm - line)))
    return sorted_scores[elbow_idx]

METRIC_FUNCTIONS = {
    "laplacian_variance":  laplacian_variance,
    "gradient_energy":     gradient_energy,
    "peak_pixel":          peak_pixel,
    "normalized_variance": normalized_variance,
    "fwhm_estimate":       fwhm_estimate,
}

METRIC_DESCRIPTIONS = {
    "laplacian_variance":  "Sharpness via Laplace operator. Best for planetary surface detail.",
    "gradient_energy":     "Edge strength via image gradients. Very fast — good for pre-screening.",
    "peak_pixel":          "99.9th percentile pixel. Best for faint targets barely above noise.",
    "normalized_variance": "Variance / mean. Handles variable sky backgrounds well.",
    "fwhm_estimate":       "Estimates FWHM of brightest object. Lower FWHM = sharper = better.",
}

OUTPUT_DESCRIPTIONS = {
    "stack_siril":   "Filtered .ser → Siril CLI → master.fit  (requires Siril running)",
    "filtered_ser":  "Filtered frames → new .ser file  (for AutoStakkert / RegiStax / Siril)",
    "both":          "Filtered .ser + Siril stack simultaneously",
    "fits_sequence": "Filtered frames → individual .fit files",
    "preview_only":  "Show histogram only — no files written",
}

# ── Dark calibration helper ────────────────────────────────────────────────────

def load_master_dark(dark_ser_paths: list) -> np.ndarray | None:
    """Average multiple dark SER frames to a master dark."""
    if not dark_ser_paths:
        return None
    accum, count = None, 0
    for path in dark_ser_paths:
        try:
            header = read_ser_header(path)
            with open(path, "rb") as f:
                for i in range(header["frame_count"]):
                    frame = read_ser_frame(f, header, i).astype(np.float32)
                    accum = frame if accum is None else accum + frame
                    count += 1
        except Exception:
            pass
    if count == 0 or accum is None:
        return None
    return (accum / count).astype(np.float32)

def apply_dark(frame: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Subtract dark, clip negatives, return same dtype."""
    orig_dtype = frame.dtype
    result     = frame.astype(np.float32) - dark
    result     = np.clip(result, 0, None)
    return result.astype(orig_dtype)

# ── Worker: scan ───────────────────────────────────────────────────────────────

class ScanWorker(QThread):
    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)

    VIDEO_EXTS = {".ser", ".mov", ".avi", ".mp4"}

    def __init__(self, project_dir: str):
        super().__init__()
        self.project_dir = project_dir

    def run(self):
        raw_dir = os.path.join(self.project_dir, "raw")
        if not os.path.isdir(raw_dir):
            raw_dir = self.project_dir   # fallback: scan project root
            self.log_line.emit("  ⚠ No raw/ subfolder — scanning project root.")

        files      = []
        total_size = 0
        self.log_line.emit(f"Scanning: {raw_dir}")

        for entry in sorted(Path(raw_dir).iterdir()):
            if entry.suffix.lower() in self.VIDEO_EXTS:
                size = entry.stat().st_size
                total_size += size
                info = {"path": str(entry), "ext": entry.suffix.lower(),
                        "size": size, "frames": 0, "width": 0, "height": 0}

                if entry.suffix.lower() == ".ser":
                    try:
                        h = read_ser_header(str(entry))
                        info["frames"] = h["frame_count"]
                        info["width"]  = h["width"]
                        info["height"] = h["height"]
                        info["color"]  = COLOR_ID_NAMES.get(h["color_id"], str(h["color_id"]))
                    except Exception as e:
                        self.log_line.emit(f"  WARNING: Could not read header for {entry.name}: {e}")

                files.append(info)
                self.log_line.emit(
                    f"  {'✓' if entry.suffix.lower()=='.ser' else '○'} "
                    f"{entry.name}  ({size/1e6:.1f} MB)"
                    + (f"  [{info['frames']} frames]" if info["frames"] else "")
                )

        # Dark calibration SERs
        dark_dir   = os.path.join(self.project_dir, "darks")
        dark_files = []
        if os.path.isdir(dark_dir):
            dark_files = [str(p) for p in Path(dark_dir).glob("*.ser")]
            self.log_line.emit(f"  Dark folder: {len(dark_files)} SER file(s)")

        # config.json
        config     = {}
        config_path = os.path.join(self.project_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as fp:
                    config = json.load(fp)
                self.log_line.emit(f"  ✓ config.json loaded")
            except Exception as e:
                self.log_line.emit(f"  WARNING: config.json parse error: {e}")

        self.log_line.emit(
            f"\nFound {len(files)} video file(s)  |  "
            f"Total: {total_size/1e9:.2f} GB"
        )
        self.finished.emit({
            "success": True, "files": files,
            "dark_files": dark_files, "raw_dir": raw_dir,
            "total_size": total_size, "config": config
        })

# ── Worker: convert ────────────────────────────────────────────────────────────

class ConvertWorker(QThread):
    progress = pyqtSignal(int, int)
    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, file_infos: list, ffmpeg: str, cancel_event):
        super().__init__()
        self.file_infos   = file_infos
        self.ffmpeg       = ffmpeg
        self._cancel      = cancel_event

    def run(self):
        ser_paths = []
        to_convert = [f for f in self.file_infos if f["ext"] != ".ser"]
        ser_ready  = [f for f in self.file_infos if f["ext"] == ".ser"]

        for fi in ser_ready:
            ser_paths.append(fi["path"])

        for idx, fi in enumerate(to_convert):
            if self._cancel.is_set():
                break
            self.progress.emit(idx, len(to_convert))
            base       = os.path.splitext(fi["path"])[0]
            output_ser = base + ".ser"

            if os.path.exists(output_ser):
                self.log_line.emit(f"  Skipping (already exists): {os.path.basename(output_ser)}")
                ser_paths.append(output_ser)
                continue

            self.log_line.emit(f"Converting: {os.path.basename(fi['path'])}")
            ok = convert_video_to_ser(
                fi["path"], output_ser, self.ffmpeg,
                cancel_event=self._cancel,
                log_callback=lambda m: self.log_line.emit(m)
            )
            if ok:
                ser_paths.append(output_ser)

        self.progress.emit(len(to_convert), len(to_convert))
        self.finished.emit({"success": True, "ser_paths": ser_paths})

# ── Worker: analysis ───────────────────────────────────────────────────────────

class AnalysisWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, ser_paths: list, metric: str,
                 dark: "np.ndarray | None", cancel_event):
        super().__init__()
        self.ser_paths  = ser_paths
        self.metric_fn  = METRIC_FUNCTIONS[metric]
        self.dark       = dark
        self._cancel    = cancel_event

    def run(self):
        all_scores   = []
        total_frames = 0

        for path in self.ser_paths:
            try:
                h = read_ser_header(path)
                total_frames += h["frame_count"]
            except Exception:
                pass

        done = 0
        for path in self.ser_paths:
            if self._cancel.is_set():
                break
            try:
                header = read_ser_header(path)
                self.log_line.emit(
                    f"Analyzing: {os.path.basename(path)} "
                    f"({header['frame_count']} frames)"
                )
                with open(path, "rb") as f:
                    for i in range(header["frame_count"]):
                        if self._cancel.is_set():
                            break
                        frame = read_ser_frame(f, header, i)
                        if self.dark is not None:
                            frame = apply_dark(frame, self.dark)
                        score = self.metric_fn(frame)
                        all_scores.append((path, i, score))
                        done += 1
                        if done % 20 == 0:
                            self.progress.emit(done, total_frames,
                                               os.path.basename(path))
            except Exception as e:
                self.log_line.emit(f"  ERROR {os.path.basename(path)}: {e}")

        self.progress.emit(len(all_scores), max(total_frames, 1), "done")
        self.finished.emit({
            "success": True,
            "scores":  all_scores,
            "total":   len(all_scores),
        })

# ── Worker: export ─────────────────────────────────────────────────────────────

class ExportWorker(QThread):
    progress = pyqtSignal(int, int)
    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, selected: list, output_mode: str,
                 output_dir: str, dark: "np.ndarray | None",
                 auto_stack: bool, stack_method: str,
                 cancel_event):
        super().__init__()
        self.selected    = selected
        self.output_mode = output_mode
        self.output_dir  = output_dir
        self.dark        = dark
        self.auto_stack  = auto_stack
        self.stack_method = stack_method
        self._cancel     = cancel_event

    def run(self):
        if self.output_mode == "preview_only":
            self.finished.emit({"success": True, "written": []})
            return

        by_file  = defaultdict(list)
        for path, frame_idx, score in self.selected:
            by_file[path].append((frame_idx, score))

        written  = []
        done, total = 0, len(self.selected)

        for ser_path, frame_list in by_file.items():
            if self._cancel.is_set():
                break
            frame_list.sort(key=lambda x: x[0])
            header = read_ser_header(ser_path)
            base   = os.path.splitext(os.path.basename(ser_path))[0]

            frames_data = []
            with open(ser_path, "rb") as f:
                for frame_idx, score in frame_list:
                    if self._cancel.is_set():
                        break
                    frame = read_ser_frame(f, header, frame_idx)
                    if self.dark is not None:
                        frame = apply_dark(frame, self.dark)
                    frames_data.append(frame)
                    done += 1
                    if done % 50 == 0:
                        self.progress.emit(done, total)

            if not frames_data:
                continue

            # ── SER output ──────────────────────────────────────────────────
            if self.output_mode in ("filtered_ser", "both", "stack_siril"):
                out_path = os.path.join(self.output_dir, f"{base}_filtered.ser")
                write_ser_file(out_path, frames_data, header)
                written.append(out_path)
                self.log_line.emit(
                    f"  ✓ {len(frames_data)} frames → {os.path.basename(out_path)}")

                if self.auto_stack and self.output_mode in ("stack_siril", "both"):
                    self._stack_in_siril(out_path, f"{base}_master")

            # ── FITS sequence ────────────────────────────────────────────────
            if self.output_mode in ("fits_sequence", "both"):
                try:
                    from astropy.io import fits
                    for i, frame in enumerate(frames_data):
                        out_path = os.path.join(
                            self.output_dir, f"{base}_frame{i:05d}.fit")
                        fits.writeto(out_path, frame, overwrite=True)
                    self.log_line.emit(
                        f"  ✓ {len(frames_data)} FITS frames → {self.output_dir}")
                    written.append(self.output_dir)
                except Exception as e:
                    self.log_line.emit(f"  ERROR writing FITS: {e}")

        self.progress.emit(total, total)
        self.finished.emit({"success": True, "written": written})

    def _stack_in_siril(self, ser_path: str, output_name: str):
        try:
            siril    = s.SirilInterface()
            siril.connect()
            work_dir = os.path.dirname(ser_path)
            siril.cmd("cd", work_dir)
            base = os.path.splitext(os.path.basename(ser_path))[0]
            siril.cmd("convert", base, "-out=lights_")
            siril.cmd("register", "lights_")
            rej = "w" if self.stack_method == "winsorized" else "l"
            siril.cmd("stack", "lights_", "rej", rej, "3", "3",
                      f"-output={output_name}")
            siril.disconnect()
            self.log_line.emit(f"  ✓ Siril stack complete → {output_name}.fit")
        except Exception as e:
            self.log_line.emit(f"  ERROR during Siril stack: {e}")

# ── Histogram canvas ───────────────────────────────────────────────────────────

class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3), facecolor=SIRIL_BG)
        self.ax  = self.fig.add_subplot(111)
        self._style_axes()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _style_axes(self):
        self.ax.set_facecolor(SIRIL_BG2)
        self.ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for spine in ("bottom", "left"):
            self.ax.spines[spine].set_color(SIRIL_BORDER)
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)

    def update_histogram(self, scores: list, threshold: float):
        self.ax.clear()
        self._style_axes()
        if not scores:
            self.draw()
            return

        scores_arr = np.array(scores)
        selected   = scores_arr[scores_arr >= threshold]
        rejected   = scores_arr[scores_arr < threshold]
        bins       = min(50, max(10, len(scores) // 20))

        if len(selected):
            self.ax.hist(selected, bins=bins, color=SIRIL_SUCCESS,
                         alpha=0.8, label=f"Selected ({len(selected)})")
        if len(rejected):
            self.ax.hist(rejected, bins=bins, color=SIRIL_ERROR,
                         alpha=0.6, label=f"Rejected ({len(rejected)})")

        self.ax.axvline(threshold, color=SIRIL_WARNING,
                        linestyle="--", linewidth=1.5, label="Cutoff")
        self.ax.set_xlabel("Quality score", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_ylabel("Frame count",   color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.legend(facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                       labelcolor=SIRIL_TEXT, fontsize=7)
        self.fig.tight_layout()
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self._style_axes()
        self.draw()

# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡  Lucky Imaging Preprocessor  —  Siril")
        self.setMinimumSize(880, 640)

        self._cancel_event = threading.Event()
        self._worker       = None
        _saved_ffmpeg      = _load_local_settings().get("lucky_ffmpeg_path", "")
        if _saved_ffmpeg and os.path.isfile(_saved_ffmpeg):
            self._ffmpeg = _saved_ffmpeg
        else:
            self._ffmpeg = get_ffmpeg_path()
        self._file_infos   = []
        self._ser_paths    = []
        self._all_scores   = []    # [(path, frame_idx, score)]
        self._selected     = []    # [(path, frame_idx, score)]
        self._dark         = None  # np.ndarray master dark

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # ── Title bar ──────────────────────────────────────────────────────
        title = QLabel("⚡  Lucky Imaging Preprocessor")
        title.setStyleSheet(
            f"background:{SIRIL_BG3}; color:{SIRIL_ACCENT};"
            f" font-size:12pt; font-weight:bold;"
            f" padding:8px 12px; border-radius:4px;"
            f" border:1px solid {SIRIL_BORDER};")
        root.addWidget(title)

        ffmpeg_status = "✓ ffmpeg found" if self._ffmpeg else "⚠ ffmpeg not found — .mov/.avi/.mp4 conversion unavailable"
        self._ffmpeg_lbl = QLabel(ffmpeg_status)
        self._ffmpeg_lbl.setStyleSheet(
            f"color:{'#4caf7d' if self._ffmpeg else SIRIL_WARNING}; font-size:8pt; padding:2px 4px;")
        root.addWidget(self._ffmpeg_lbl)

        # Manual ffmpeg path row (always visible so user can override)
        ffmpeg_row = QHBoxLayout()
        ffmpeg_path_lbl = QLabel("ffmpeg path:")
        ffmpeg_path_lbl.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        ffmpeg_path_lbl.setFixedWidth(80)
        self._ffmpeg_edit = QLineEdit()
        self._ffmpeg_edit.setPlaceholderText(
            r"e.g. C:\ffmpeg\bin\ffmpeg.exe  (leave blank to use system PATH)")
        self._ffmpeg_edit.setStyleSheet("font-size:8pt;")
        if self._ffmpeg:
            self._ffmpeg_edit.setText(self._ffmpeg)
        self._ffmpeg_edit.textChanged.connect(self._on_ffmpeg_path_changed)
        ffmpeg_browse_btn = QPushButton("Browse…")
        ffmpeg_browse_btn.setFixedWidth(72)
        ffmpeg_browse_btn.clicked.connect(self._browse_ffmpeg)
        ffmpeg_row.addWidget(ffmpeg_path_lbl)
        ffmpeg_row.addWidget(self._ffmpeg_edit, 3)
        ffmpeg_row.addWidget(ffmpeg_browse_btn)
        root.addLayout(ffmpeg_row)

        # ── Tabs ───────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._build_tab_input()
        self._build_tab_quality()
        self._build_tab_output()
        self._build_tab_log()

        # ── Progress ───────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        root.addWidget(self._progress)

        # ── Button row ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run Pipeline")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._run_pipeline)

        self._cancel_btn = QPushButton("✕  Cancel")
        self._cancel_btn.setObjectName("danger")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)

        self._open_btn = QPushButton("📂 Open Output Folder")
        self._open_btn.setObjectName("secondary")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_output)

        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._open_btn)
        root.addLayout(btn_row)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status = QLabel("Ready — select a project folder to begin.")
        self._status.setObjectName("dim")
        self._status.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt; padding:2px;")
        root.addWidget(self._status)

    # ── Tab 1: Input ───────────────────────────────────────────────────────

    def _build_tab_input(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        self._tabs.addTab(tab, "📁  Input")

        # Project folder
        gb_proj = QGroupBox("Project folder")
        proj_form = QHBoxLayout()
        self._proj_edit = QLineEdit()
        self._proj_edit.setPlaceholderText("e.g. /home/user/Jupiter_2025-05-22/")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_project)
        scan_btn   = QPushButton("🔍 Scan for videos")
        scan_btn.setObjectName("primary")
        scan_btn.clicked.connect(self._scan)
        proj_form.addWidget(self._proj_edit, 3)
        proj_form.addWidget(browse_btn)
        proj_form.addWidget(scan_btn)
        gb_proj.setLayout(proj_form)
        layout.addWidget(gb_proj)

        # Detected files
        gb_files = QGroupBox("Detected files")
        files_layout = QVBoxLayout()
        self._file_list = QListWidget()
        self._file_list.setMinimumHeight(120)
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        files_layout.addWidget(self._file_list)
        self._files_summary = QLabel("No files scanned yet.")
        self._files_summary.setObjectName("dim")
        self._files_summary.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        files_layout.addWidget(self._files_summary)
        gb_files.setLayout(files_layout)
        layout.addWidget(gb_files)

        # Dark calibration
        gb_dark = QGroupBox("Dark calibration (optional)")
        dark_layout = QVBoxLayout()
        self._dark_check = QCheckBox("Apply dark subtraction")
        dark_row  = QHBoxLayout()
        self._dark_edit = QLineEdit()
        self._dark_edit.setPlaceholderText("Path to dark SER (or folder of dark SERs)")
        self._dark_edit.setEnabled(False)
        dark_browse = QPushButton("Browse dark…")
        dark_browse.clicked.connect(self._browse_dark)
        dark_row.addWidget(self._dark_edit, 3)
        dark_row.addWidget(dark_browse)
        self._dark_check.toggled.connect(self._dark_edit.setEnabled)
        self._dark_check.toggled.connect(dark_browse.setEnabled)
        dark_browse.setEnabled(False)
        dark_layout.addWidget(self._dark_check)
        dark_layout.addLayout(dark_row)
        gb_dark.setLayout(dark_layout)
        layout.addWidget(gb_dark)

        # Equipment profile
        gb_eq = QGroupBox("Equipment profile (optional)")
        eq_layout = QFormLayout()
        self._profile_combo = QComboBox()
        self._profile_combo.addItem("— None —")
        self._profile_info  = QLabel("—")
        self._profile_info.setObjectName("dim")
        self._profile_info.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        if HAS_EQUIPMENT_MANAGER:
            try:
                for name in load_all_profiles():
                    self._profile_combo.addItem(name)
            except Exception:
                pass
        else:
            self._profile_combo.setEnabled(False)
            self._profile_combo.setToolTip("equipment_manager.py not found")
        self._profile_combo.currentTextChanged.connect(self._on_profile_changed)
        eq_layout.addRow("Profile:", self._profile_combo)
        eq_layout.addRow("Info:", self._profile_info)
        gb_eq.setLayout(eq_layout)
        layout.addWidget(gb_eq)

        layout.addStretch()

    # ── Tab 2: Quality ─────────────────────────────────────────────────────

    def _build_tab_quality(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        self._tabs.addTab(tab, "📊  Quality")

        # Metric picker
        gb_metric = QGroupBox("Quality metric")
        m_layout  = QFormLayout()
        self._metric_combo = QComboBox()
        for k in METRIC_FUNCTIONS:
            self._metric_combo.addItem(k)
        self._metric_desc = QLabel(METRIC_DESCRIPTIONS["laplacian_variance"])
        self._metric_desc.setWordWrap(True)
        self._metric_desc.setObjectName("dim")
        self._metric_desc.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        self._metric_combo.currentTextChanged.connect(
            lambda t: self._metric_desc.setText(METRIC_DESCRIPTIONS.get(t, "")))
        m_layout.addRow("Metric:", self._metric_combo)
        m_layout.addRow("", self._metric_desc)
        gb_metric.setLayout(m_layout)
        layout.addWidget(gb_metric)

        # Selection mode
        gb_sel = QGroupBox("Frame selection")
        sel_layout = QVBoxLayout()
        sel_row    = QHBoxLayout()
        self._sel_mode_combo = QComboBox()
        for mode in ("Fixed percent", "Fixed count", "Adaptive elbow", "Threshold"):
            self._sel_mode_combo.addItem(mode)
        sel_row.addWidget(QLabel("Mode:"))
        sel_row.addWidget(self._sel_mode_combo)
        sel_row.addStretch()
        sel_layout.addLayout(sel_row)

        slider_row = QHBoxLayout()
        self._sel_slider  = QSlider(Qt.Orientation.Horizontal)
        self._sel_slider.setRange(1, 100)
        self._sel_slider.setValue(20)
        self._sel_spinbox = QSpinBox()
        self._sel_spinbox.setRange(1, 100000)
        self._sel_spinbox.setValue(20)
        self._sel_spinbox.setMinimumWidth(70)
        self._sel_slider.valueChanged.connect(self._sel_spinbox.setValue)
        self._sel_spinbox.valueChanged.connect(self._sel_slider.setValue)
        self._sel_spinbox.valueChanged.connect(self._update_selection_preview)
        self._sel_unit_lbl = QLabel("%")
        slider_row.addWidget(self._sel_slider, 3)
        slider_row.addWidget(self._sel_spinbox)
        slider_row.addWidget(self._sel_unit_lbl)
        sel_layout.addLayout(slider_row)

        self._sel_preview = QLabel("Will keep: — / — frames")

        self._sel_mode_combo.currentTextChanged.connect(self._on_sel_mode_changed)
        self._on_sel_mode_changed("Fixed percent")   # init
        self._sel_preview.setObjectName("dim")
        self._sel_preview.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        sel_layout.addWidget(self._sel_preview)
        gb_sel.setLayout(sel_layout)
        layout.addWidget(gb_sel)

        # Analyze button + progress
        analyze_row = QHBoxLayout()
        self._analyze_btn = QPushButton("▶  Analyze frames")
        self._analyze_btn.setObjectName("primary")
        self._analyze_btn.clicked.connect(self._run_analysis)
        self._analyze_progress = QProgressBar()
        self._analyze_progress.setVisible(False)
        analyze_row.addWidget(self._analyze_btn)
        analyze_row.addWidget(self._analyze_progress, 2)
        layout.addLayout(analyze_row)

        # Histogram
        gb_hist = QGroupBox("Score distribution")
        hist_layout = QVBoxLayout()
        self._histogram = HistogramCanvas()
        hist_layout.addWidget(self._histogram)
        gb_hist.setLayout(hist_layout)
        layout.addWidget(gb_hist)

    # ── Tab 3: Output ──────────────────────────────────────────────────────

    def _build_tab_output(self):
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        self._tabs.addTab(tab, "💾  Output")

        # Output mode
        gb_mode  = QGroupBox("Output format")
        o_layout = QFormLayout()
        self._out_mode_combo = QComboBox()
        for k in OUTPUT_DESCRIPTIONS:
            self._out_mode_combo.addItem(k)
        self._out_mode_desc = QLabel(OUTPUT_DESCRIPTIONS["stack_siril"])
        self._out_mode_desc.setWordWrap(True)
        self._out_mode_desc.setObjectName("dim")
        self._out_mode_desc.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        self._out_mode_combo.setCurrentText("filtered_ser")
        self._out_mode_combo.currentTextChanged.connect(
            lambda t: self._out_mode_desc.setText(OUTPUT_DESCRIPTIONS.get(t, "")))
        self._out_mode_desc.setText(OUTPUT_DESCRIPTIONS["filtered_ser"])
        o_layout.addRow("Mode:", self._out_mode_combo)
        o_layout.addRow("", self._out_mode_desc)
        gb_mode.setLayout(o_layout)
        layout.addWidget(gb_mode)

        # Output location
        gb_out = QGroupBox("Output location")
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Defaults to project folder if blank")
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self._out_edit, 3)
        out_row.addWidget(out_browse)
        gb_out.setLayout(out_row)
        layout.addWidget(gb_out)

        # Cleanup
        gb_clean  = QGroupBox("Cleanup")
        cl_layout = QVBoxLayout()
        self._del_originals_check = QCheckBox(
            "Delete original .mov/.avi/.mp4 after successful conversion")
        self._del_unfiltered_check = QCheckBox(
            "Delete unfiltered .ser after filtering (keep only _filtered.ser)")
        warn_lbl = QLabel("⚠  Deletion cannot be undone. Only runs after verified export.")
        warn_lbl.setObjectName("warning")
        warn_lbl.setStyleSheet(f"color:{SIRIL_WARNING}; font-size:8pt;")
        cl_layout.addWidget(self._del_originals_check)
        cl_layout.addWidget(self._del_unfiltered_check)
        cl_layout.addWidget(warn_lbl)
        gb_clean.setLayout(cl_layout)
        layout.addWidget(gb_clean)

        # Siril handoff
        gb_siril  = QGroupBox("Siril handoff (stack_siril / both modes)")
        si_layout = QFormLayout()
        self._auto_stack_check = QCheckBox("Auto-stack in Siril after export")
        self._stack_method_combo = QComboBox()
        for m in ("winsorized", "linear"):
            self._stack_method_combo.addItem(m)
        si_note = QLabel("Siril must be running and connected for handoff to work.")
        si_note.setObjectName("dim")
        si_note.setStyleSheet(f"color:{SIRIL_TEXT_DIM}; font-size:8pt;")
        si_layout.addRow(self._auto_stack_check)
        si_layout.addRow("Stack rejection:", self._stack_method_combo)
        si_layout.addRow(si_note)
        gb_siril.setLayout(si_layout)
        layout.addWidget(gb_siril)

        layout.addStretch()

    # ── Tab 4: Log ─────────────────────────────────────────────────────────

    def _build_tab_log(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 8))
        clear_btn = QPushButton("Clear log")
        clear_btn.setObjectName("secondary")
        clear_btn.clicked.connect(self._log.clear)
        layout.addWidget(self._log)
        layout.addWidget(clear_btn)
        self._tabs.addTab(tab, "📋  Log")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_widget_append(f"[{ts}] {text}")

    def _log_widget_append(self, text: str):
        self._log.appendPlainText(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())

    def _set_status(self, text: str, color: str = SIRIL_TEXT_DIM):
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{color}; font-size:8pt; padding:2px;")

    def _set_busy(self, busy: bool, label: str = "Processing…"):
        self._run_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        self._analyze_btn.setEnabled(not busy)
        self._progress.setVisible(busy)
        if busy:
            self._cancel_event.clear()
            self._set_status(label, SIRIL_ACCENT)

    def _browse_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate ffmpeg executable", "",
            "Executables (*.exe);;All files (*)")
        if path:
            self._ffmpeg_edit.setText(path)

    def _on_ffmpeg_path_changed(self, text: str):
        import shutil
        path = text.strip()
        if not path:
            self._ffmpeg = get_ffmpeg_path()
        elif os.path.isfile(path):
            self._ffmpeg = path
        else:
            found = shutil.which(path)
            self._ffmpeg = found if found else None

        if self._ffmpeg:
            self._ffmpeg_lbl.setText(f"\u2713 ffmpeg: {self._ffmpeg}")
            self._ffmpeg_lbl.setStyleSheet(
                f"color:{SIRIL_SUCCESS}; font-size:8pt; padding:2px 4px;")
            # Persist so it survives restart
            _save_local_settings({"lucky_ffmpeg_path": self._ffmpeg})
        else:
            self._ffmpeg_lbl.setText(
                "\u26a0 ffmpeg not found \u2014 .mov/.avi/.mp4 conversion unavailable")
            self._ffmpeg_lbl.setStyleSheet(
                f"color:{SIRIL_WARNING}; font-size:8pt; padding:2px 4px;")

    def _browse_project(self):
        d = QFileDialog.getExistingDirectory(self, "Select project folder")
        if d:
            self._proj_edit.setText(d)

    def _browse_dark(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dark SER", "", "SER files (*.ser);;All files (*)")
        if path:
            self._dark_edit.setText(path)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self._out_edit.setText(d)

    def _on_profile_changed(self, name: str):
        if not HAS_EQUIPMENT_MANAGER or name == "— None —":
            self._profile_info.setText("—")
            return
        try:
            p = get_profile(name)
            if p:
                parts = []
                if p.get("bayer_pattern"):
                    parts.append(f"Bayer: {p['bayer_pattern']}")
                if p.get("resolution_arcsec_px"):
                    parts.append(f"Scale: {p['resolution_arcsec_px']}\"/px")
                if p.get("fov_width_arcmin"):
                    parts.append(f"FOV: {p['fov_width_arcmin']:.1f}′")
                self._profile_info.setText("  |  ".join(parts) if parts else "—")
        except Exception:
            self._profile_info.setText("—")

    def _on_sel_mode_changed(self, mode: str):
        is_elbow = (mode == "Adaptive elbow")
        self._sel_slider.setEnabled(not is_elbow)
        self._sel_spinbox.setEnabled(not is_elbow)
        if mode == "Fixed percent":
            self._sel_slider.setRange(1, 100)
            self._sel_spinbox.setRange(1, 100)
            self._sel_spinbox.setValue(20)
            self._sel_unit_lbl.setText("%")
        elif mode == "Fixed count":
            self._sel_slider.setRange(1, 5000)
            self._sel_spinbox.setRange(1, 500000)
            self._sel_spinbox.setValue(500)
            self._sel_unit_lbl.setText("frames")
        elif mode == "Adaptive elbow":
            self._sel_unit_lbl.setText("(auto)")
        elif mode == "Threshold":
            self._sel_slider.setRange(0, 10000)
            self._sel_spinbox.setRange(0, 10000000)
            self._sel_spinbox.setValue(100)
            self._sel_unit_lbl.setText("score")
        self._update_selection_preview()

    def _compute_threshold(self) -> float:
        """Return score threshold given current mode/value."""
        if not self._all_scores:
            return 0.0
        scores     = [s for _, _, s in self._all_scores]
        mode       = self._sel_mode_combo.currentText()
        val        = self._sel_spinbox.value()

        if mode == "Fixed percent":
            cutoff = max(1, int(len(scores) * val / 100))
            return sorted(scores, reverse=True)[cutoff - 1]
        elif mode == "Fixed count":
            cutoff = min(val, len(scores))
            return sorted(scores, reverse=True)[cutoff - 1]
        elif mode == "Adaptive elbow":
            return adaptive_elbow(scores)
        elif mode == "Threshold":
            return float(val)
        return 0.0

    def _update_selection_preview(self):
        if not self._all_scores:
            self._sel_preview.setText("Will keep: — / — frames  (analyze first)")
            return
        threshold = self._compute_threshold()
        selected  = [(p, i, sc) for p, i, sc in self._all_scores if sc >= threshold]
        total     = len(self._all_scores)
        pct       = 100 * len(selected) / total if total else 0
        self._sel_preview.setText(
            f"Will keep: {len(selected):,} / {total:,} frames  ({pct:.1f}%)")
        self._histogram.update_histogram(
            [sc for _, _, sc in self._all_scores], threshold)

    # ── Scan ───────────────────────────────────────────────────────────────

    def _scan(self):
        proj = self._proj_edit.text().strip()
        if not proj or not os.path.isdir(proj):
            QMessageBox.warning(self, "No folder", "Please select a valid project folder.")
            return
        self._file_list.clear()
        self._file_infos.clear()
        self._set_busy(True, "Scanning…")

        self._scan_worker = ScanWorker(proj)
        self._scan_worker.log_line.connect(self._log_widget_append)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_scan_done(self, result: dict):
        self._set_busy(False)
        if not result.get("success"):
            self._set_status("Scan failed.", SIRIL_ERROR)
            return

        self._file_infos = result["files"]
        self._dark_files = result.get("dark_files", [])

        for fi in self._file_infos:
            label = os.path.basename(fi["path"])
            if fi["frames"]:
                label += f"  [{fi['frames']} frames  {fi['width']}×{fi['height']}  {fi.get('color','')}]"
            label += f"  {fi['size']/1e6:.1f} MB"
            item = QListWidgetItem(label)
            item.setForeground(
                QColor(SIRIL_SUCCESS if fi["ext"] == ".ser" else SIRIL_WARNING))
            self._file_list.addItem(item)

        total = result["total_size"]
        self._files_summary.setText(
            f"{len(self._file_infos)} file(s)  |  {total/1e9:.2f} GB total"
            + (f"  |  {len(self._dark_files)} dark SER(s)" if self._dark_files else ""))

        # Auto-fill dark
        if self._dark_files and not self._dark_edit.text():
            self._dark_edit.setText("; ".join(self._dark_files))
            self._dark_check.setChecked(True)

        # Apply config.json overrides
        cfg = result.get("config", {})
        if cfg.get("quality_metric"):
            idx = self._metric_combo.findText(cfg["quality_metric"])
            if idx >= 0:
                self._metric_combo.setCurrentIndex(idx)
        if cfg.get("output_mode"):
            idx = self._out_mode_combo.findText(cfg["output_mode"])
            if idx >= 0:
                self._out_mode_combo.setCurrentIndex(idx)
        if cfg.get("delete_originals"):
            self._del_originals_check.setChecked(cfg["delete_originals"])
        if cfg.get("delete_unfiltered_ser"):
            self._del_unfiltered_check.setChecked(cfg["delete_unfiltered_ser"])

        self._set_status(
            f"Scan complete: {len(self._file_infos)} file(s) found.", SIRIL_SUCCESS)

    # ── Analysis ───────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._file_infos:
            QMessageBox.warning(self, "No files", "Scan a project folder first.")
            return

        ser_paths = [fi["path"] for fi in self._file_infos if fi["ext"] == ".ser"]
        non_ser   = [fi for fi in self._file_infos if fi["ext"] != ".ser"]
        if non_ser and not self._ffmpeg:
            QMessageBox.warning(
                self, "ffmpeg missing",
                "Non-SER files found but ffmpeg is not available.\n"
                "Only SER files will be analyzed.\n\n"
                "Install ffmpeg and ensure it is in your PATH to convert video files.")
        if not ser_paths:
            QMessageBox.warning(self, "No SER files",
                "No SER files to analyze. Convert videos first via Run Pipeline.")
            return

        # Load dark if requested
        dark = None
        if self._dark_check.isChecked():
            dark_path = self._dark_edit.text().strip()
            dark_paths = [p.strip() for p in dark_path.split(";") if p.strip()]
            if dark_paths:
                self._log_widget_append("Loading master dark…")
                dark = load_master_dark(dark_paths)
                if dark is None:
                    self._log_widget_append("  WARNING: Could not load dark frames.")
        self._dark = dark

        self._all_scores = []
        self._histogram.clear_plot()
        self._analyze_progress.setVisible(True)
        self._set_busy(True, "Analyzing frames…")

        self._analysis_worker = AnalysisWorker(
            ser_paths, self._metric_combo.currentText(),
            dark, self._cancel_event)
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.log_line.connect(self._log_widget_append)
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.start()

    def _on_analysis_progress(self, done: int, total: int, fname: str):
        self._analyze_progress.setMaximum(total)
        self._analyze_progress.setValue(done)
        self._set_status(f"Analyzing {fname}… {done}/{total}")

    def _on_analysis_done(self, result: dict):
        self._analyze_progress.setVisible(False)
        self._set_busy(False)
        if not result.get("success"):
            self._set_status("Analysis failed.", SIRIL_ERROR)
            return
        self._all_scores = result["scores"]
        self._log_widget_append(
            f"Analysis complete: {len(self._all_scores):,} frames scored.")
        self._update_selection_preview()
        self._tabs.setCurrentIndex(1)   # stay on Quality tab
        self._set_status(
            f"Analysis done: {len(self._all_scores):,} frames.", SIRIL_SUCCESS)

    # ── Full pipeline ──────────────────────────────────────────────────────

    def _run_pipeline(self):
        proj = self._proj_edit.text().strip()
        if not proj or not os.path.isdir(proj):
            QMessageBox.warning(self, "No folder", "Please select a valid project folder.")
            return
        if not self._file_infos:
            QMessageBox.warning(self, "Not scanned",
                "Please scan the project folder first (📁 Input tab).")
            return

        # Determine output dir
        out_dir = self._out_edit.text().strip() or proj
        os.makedirs(out_dir, exist_ok=True)

        # Build selected list
        if self._all_scores:
            threshold       = self._compute_threshold()
            self._selected  = [(p, i, sc) for p, i, sc in self._all_scores
                               if sc >= threshold]
            if not self._selected:
                QMessageBox.warning(self, "Empty selection",
                    "Current threshold selects 0 frames. Adjust selection settings.")
                return
        else:
            self._selected = []   # will be populated after analysis in pipeline

        output_mode = self._out_mode_combo.currentText()
        has_non_ser = any(fi["ext"] != ".ser" for fi in self._file_infos)

        # If no analysis done yet, run full pipeline: convert → analyze → export
        if not self._all_scores:
            self._run_full_pipeline(proj, out_dir, output_mode)
        else:
            self._run_export_only(out_dir, output_mode)

    def _run_full_pipeline(self, proj: str, out_dir: str, output_mode: str):
        """Step 1: Convert any non-SER files."""
        has_non_ser = any(fi["ext"] != ".ser" for fi in self._file_infos)
        ser_ready   = [fi for fi in self._file_infos if fi["ext"] == ".ser"]

        if not has_non_ser or not self._ffmpeg:
            # Skip straight to analysis
            ser_paths = [fi["path"] for fi in ser_ready]
            self._pipeline_ser_paths   = ser_paths
            self._pipeline_out_dir     = out_dir
            self._pipeline_output_mode = output_mode
            self._start_analysis_step()
            return

        self._set_busy(True, "Converting videos…")
        self._log_widget_append("=== STEP 1: Convert videos → SER ===")
        self._pipeline_out_dir     = out_dir
        self._pipeline_output_mode = output_mode

        self._convert_worker = ConvertWorker(
            self._file_infos, self._ffmpeg, self._cancel_event)
        self._convert_worker.progress.connect(
            lambda d, t: (self._progress.setMaximum(max(t, 1)),
                          self._progress.setValue(d)))
        self._convert_worker.log_line.connect(self._log_widget_append)
        self._convert_worker.finished.connect(self._on_convert_done)
        self._convert_worker.start()

    def _on_convert_done(self, result: dict):
        self._pipeline_ser_paths = result.get("ser_paths", [])
        self._log_widget_append(
            f"Conversion done: {len(self._pipeline_ser_paths)} SER file(s).")
        self._start_analysis_step()

    def _start_analysis_step(self):
        self._log_widget_append("=== STEP 2: Analyze frame quality ===")
        self._set_status("Analyzing frames…", SIRIL_ACCENT)

        dark = None
        if self._dark_check.isChecked():
            dark_path  = self._dark_edit.text().strip()
            dark_paths = [p.strip() for p in dark_path.split(";") if p.strip()]
            if dark_paths:
                dark = load_master_dark(dark_paths)
        self._dark = dark

        self._analysis_worker2 = AnalysisWorker(
            self._pipeline_ser_paths,
            self._metric_combo.currentText(),
            dark, self._cancel_event)
        self._analysis_worker2.progress.connect(
            lambda d, t, f: (self._progress.setMaximum(max(t, 1)),
                             self._progress.setValue(d),
                             self._set_status(f"Analyzing {f}… {d}/{t}")))
        self._analysis_worker2.log_line.connect(self._log_widget_append)
        self._analysis_worker2.finished.connect(self._on_pipeline_analysis_done)
        self._analysis_worker2.start()

    def _on_pipeline_analysis_done(self, result: dict):
        self._all_scores = result.get("scores", [])
        self._update_selection_preview()
        self._log_widget_append(
            f"Analysis done: {len(self._all_scores):,} frames scored.")

        threshold      = self._compute_threshold()
        self._selected = [(p, i, sc) for p, i, sc in self._all_scores
                         if sc >= threshold]
        self._log_widget_append(
            f"Selected: {len(self._selected):,} / {len(self._all_scores):,} frames.")

        if not self._selected:
            self._set_busy(False)
            self._set_status("No frames selected — adjust threshold.", SIRIL_WARNING)
            return
        self._run_export_only(self._pipeline_out_dir, self._pipeline_output_mode)

    def _run_export_only(self, out_dir: str, output_mode: str):
        self._log_widget_append(
            f"=== STEP 3: Export ({output_mode}) ===\n"
            f"  Exporting {len(self._selected):,} frames → {out_dir}")
        self._set_status("Exporting frames…", SIRIL_ACCENT)

        self._export_worker = ExportWorker(
            self._selected, output_mode, out_dir,
            self._dark,
            self._auto_stack_check.isChecked(),
            self._stack_method_combo.currentText(),
            self._cancel_event)
        self._export_worker.progress.connect(
            lambda d, t: (self._progress.setMaximum(max(t, 1)),
                          self._progress.setValue(d)))
        self._export_worker.log_line.connect(self._log_widget_append)
        self._export_worker.finished.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_done(self, result: dict):
        self._set_busy(False)
        if not result.get("success"):
            self._set_status("Export failed.", SIRIL_ERROR)
            return

        self._open_btn.setEnabled(True)
        self._log_widget_append("=== STEP 4: Cleanup ===")
        self._do_cleanup()
        self._set_status(
            f"✓ Done — {len(self._selected):,} frames exported.", SIRIL_SUCCESS)
        self._tabs.setCurrentIndex(3)  # show log

    def _do_cleanup(self):
        if self._del_originals_check.isChecked():
            for fi in self._file_infos:
                if fi["ext"] in (".mov", ".avi", ".mp4"):
                    try:
                        os.remove(fi["path"])
                        self._log_widget_append(
                            f"  Deleted: {os.path.basename(fi['path'])}")
                    except Exception as e:
                        self._log_widget_append(
                            f"  ERROR deleting {os.path.basename(fi['path'])}: {e}")

        if self._del_unfiltered_check.isChecked():
            used_paths = set(p for p, _, _ in self._all_scores)
            for ser_path in used_paths:
                base         = os.path.splitext(ser_path)[0]
                filtered_ser = base + "_filtered.ser"
                if os.path.exists(filtered_ser) and os.path.exists(ser_path):
                    try:
                        os.remove(ser_path)
                        self._log_widget_append(
                            f"  Deleted unfiltered: {os.path.basename(ser_path)}")
                    except Exception as e:
                        self._log_widget_append(
                            f"  ERROR deleting {os.path.basename(ser_path)}: {e}")

    def _cancel(self):
        self._cancel_event.set()
        self._log_widget_append("  ⚠ Cancel requested…")
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _open_output(self):
        out_dir = self._out_edit.text().strip() or self._proj_edit.text().strip()
        if out_dir and os.path.isdir(out_dir):
            import subprocess as sp, sys as _sys
            if _sys.platform == "win32":
                sp.Popen(["explorer", out_dir])
            elif _sys.platform == "darwin":
                sp.Popen(["open", out_dir])
            else:
                sp.Popen(["xdg-open", out_dir])

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
