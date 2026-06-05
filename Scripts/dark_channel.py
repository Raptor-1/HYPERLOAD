r"""
dark_channel.py  —  Cloud / Haze Removal for Siril
Script #6 — Dark Channel Prior (He, Sun & Tang, CVPR 2009)

Place in: C:\Users\Marcell\Desktop\Siril New Scripts\
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")
s.ensure_installed("scipy")

# ── Standard imports ─────────────────────────────────────────────────────────
import os
import sys
import glob
import threading
from datetime import datetime

import numpy as np
from scipy.ndimage import minimum_filter, gaussian_filter
from astropy.io import fits as astropy_fits
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QSlider, QSplitter, QFrame, QSizePolicy, QScrollArea
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

# ── Siril colour palette ──────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Dark channel: min over local patch of min over channels."""
    if image.ndim == 3:
        if image.shape[0] == 3:
            min_channel = np.min(image, axis=0)
        else:
            min_channel = np.min(image, axis=2)
    else:
        min_channel = image.astype(np.float64)
    dark = minimum_filter(min_channel.astype(np.float64), size=patch_size)
    return dark.astype(np.float32)


def estimate_atmospheric_light(image: np.ndarray,
                                dark_channel: np.ndarray,
                                top_fraction: float = 0.001) -> np.ndarray:
    """Estimate atmospheric light from brightest dark-channel pixels."""
    h, w = dark_channel.shape
    n_pixels = h * w
    n_top = max(1, int(n_pixels * top_fraction))
    flat_dark = dark_channel.ravel()
    top_idx = np.argpartition(flat_dark, -n_top)[-n_top:]

    if image.ndim == 2:
        flat_image = image.ravel()
        A = float(np.mean(flat_image[top_idx]))
        return np.array([A], dtype=np.float32)
    elif image.shape[0] == 3:
        A = np.zeros(3, dtype=np.float32)
        for c in range(3):
            flat_ch = image[c].ravel()
            A[c] = float(np.mean(flat_ch[top_idx]))
        return A
    else:
        A = np.zeros(3, dtype=np.float32)
        for c in range(3):
            flat_ch = image[:, :, c].ravel()
            A[c] = float(np.mean(flat_ch[top_idx]))
        return A


def estimate_transmission_map(image: np.ndarray,
                               A: np.ndarray,
                               patch_size: int = 15,
                               omega: float = 0.95) -> np.ndarray:
    """Estimate transmission t(x) = 1 - omega * dark(I/A)."""
    if image.ndim == 2:
        img_norm = image.astype(np.float64) / (float(A[0]) + 1e-10)
        img_norm = img_norm[np.newaxis, :, :]
    elif image.shape[0] == 3:
        img_norm = np.zeros_like(image, dtype=np.float64)
        for c in range(3):
            img_norm[c] = image[c].astype(np.float64) / (float(A[c]) + 1e-10)
    else:
        img_norm = np.zeros_like(image, dtype=np.float64)
        for c in range(3):
            img_norm[:, :, c] = (image[:, :, c].astype(np.float64) /
                                  (float(A[c]) + 1e-10))

    if img_norm.shape[0] <= 4:
        min_norm = np.min(img_norm, axis=0)
    else:
        min_norm = np.min(img_norm, axis=2)

    dark_norm = minimum_filter(min_norm, size=patch_size)
    t = 1.0 - omega * dark_norm
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def refine_transmission_guided_filter(transmission: np.ndarray,
                                       guide: np.ndarray,
                                       radius: int = 40,
                                       epsilon: float = 1e-3) -> np.ndarray:
    """Edge-preserving guided filter refinement of transmission map."""
    g_min, g_max = guide.min(), guide.max()
    if g_max > g_min:
        I = (guide.astype(np.float64) - g_min) / (g_max - g_min)
    else:
        I = guide.astype(np.float64)

    p = transmission.astype(np.float64)
    r = radius

    def box_filter(src, r):
        if r <= 0:
            return src
        cumsum = np.cumsum(src, axis=0)
        out = cumsum[2*r:, :] - cumsum[:-2*r, :]
        cumsum = np.cumsum(out, axis=1)
        out = cumsum[:, 2*r:] - cumsum[:, :-2*r]
        return out / ((2*r + 1) ** 2)

    mean_I  = box_filter(I, r)
    mean_p  = box_filter(p, r)
    mean_Ip = box_filter(I * p, r)
    cov_Ip  = mean_Ip - mean_I * mean_p
    mean_II = box_filter(I * I, r)
    var_I   = mean_II - mean_I * mean_I

    a_coeff = cov_Ip / (var_I + epsilon)
    b_coeff = mean_p - a_coeff * mean_I

    mean_a = box_filter(a_coeff, r)
    mean_b = box_filter(b_coeff, r)

    # Guided filter reduces size by 2r in each dimension; pad back
    orig_h, orig_w = transmission.shape
    min_h = min(orig_h, mean_a.shape[0])
    min_w = min(orig_w, mean_a.shape[1])

    I_crop = I[:min_h, :min_w]
    q = mean_a[:min_h, :min_w] * I_crop + mean_b[:min_h, :min_w]

    if q.shape != transmission.shape:
        t_out = np.zeros((orig_h, orig_w), dtype=np.float64)
        t_out[:min_h, :min_w] = q
        if min_h < orig_h:
            t_out[min_h:, :min_w] = q[-1:, :]
        if min_w < orig_w:
            t_out[:min_h, min_w:] = q[:, -1:]
        if min_h < orig_h and min_w < orig_w:
            t_out[min_h:, min_w:] = q[-1, -1]
        q = t_out

    return np.clip(q, 0.0, 1.0).astype(np.float32)


def recover_scene(image: np.ndarray,
                   A: np.ndarray,
                   transmission: np.ndarray,
                   t_min: float = 0.1) -> np.ndarray:
    """J(x) = (I(x) - A) / max(t(x), t_min) + A"""
    t_clamped = np.clip(transmission, t_min, 1.0)

    if image.ndim == 2:
        a_val = float(A[0])
        J = (image.astype(np.float64) - a_val) / t_clamped + a_val
    elif image.shape[0] == 3:
        J = np.zeros_like(image, dtype=np.float64)
        for c in range(3):
            J[c] = (image[c].astype(np.float64) - float(A[c])) / t_clamped + float(A[c])
    else:
        J = np.zeros_like(image, dtype=np.float64)
        for c in range(3):
            J[:, :, c] = ((image[:, :, c].astype(np.float64) - float(A[c])) /
                           t_clamped + float(A[c]))

    return np.clip(J, 0, None).astype(np.float32)


def dehaze_image(image: np.ndarray,
                  patch_size: int = 15,
                  omega: float = 0.95,
                  t_min: float = 0.1,
                  guided_radius: int = 40,
                  guided_epsilon: float = 1e-3,
                  top_fraction: float = 0.001,
                  refine_transmission: bool = True,
                  log_callback=None) -> tuple:
    """Full Dark Channel Prior pipeline. Returns (dehazed, t_map, A)."""
    if log_callback:
        log_callback("  Computing dark channel...")
    dark = compute_dark_channel(image, patch_size)

    if log_callback:
        log_callback("  Estimating atmospheric light...")
    A = estimate_atmospheric_light(image, dark, top_fraction)
    if log_callback:
        a_str = ", ".join(f"{v:.4f}" for v in A)
        log_callback(f"  Atmospheric light A = [{a_str}]")

    if log_callback:
        log_callback("  Estimating transmission map...")
    t_raw = estimate_transmission_map(image, A, patch_size, omega)

    if refine_transmission:
        if log_callback:
            log_callback("  Refining transmission (guided filter)...")
        if image.ndim == 2:
            guide = image.astype(np.float32)
        elif image.shape[0] == 3:
            guide = (0.299 * image[0] + 0.587 * image[1] +
                     0.114 * image[2]).astype(np.float32)
        else:
            guide = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] +
                     0.114 * image[:, :, 2]).astype(np.float32)
        t_refined = refine_transmission_guided_filter(
            t_raw, guide, guided_radius, guided_epsilon)
    else:
        t_refined = t_raw

    if log_callback:
        t_mean = float(np.mean(t_refined))
        t_min_val = float(np.min(t_refined))
        log_callback(
            f"  Transmission: mean={t_mean:.3f}  "
            f"min={t_min_val:.3f}  (1=clear, 0=fully hazed)")

    if log_callback:
        log_callback("  Recovering clean scene...")
    J = recover_scene(image, A, t_refined, t_min)

    return J, t_refined, A


def quick_haze_estimate(fits_path: str, patch_size: int = 15) -> float:
    """Fast haze fraction estimate (0.0=clear, 1.0=fully hazed)."""
    data = astropy_fits.getdata(fits_path).astype(np.float32)
    if data.ndim == 3:
        if data.shape[0] == 3:
            lum = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
        else:
            lum = data[0]
    else:
        lum = data
    dark = compute_dark_channel(lum, patch_size)
    A = estimate_atmospheric_light(lum, dark)
    t = estimate_transmission_map(lum, A, patch_size, omega=0.95)
    return float(1.0 - np.mean(t))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class DehazeWorker(QThread):
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    preview_ready = pyqtSignal(object, object, object)  # dehazed_lum, t_map, orig_lum
    finished      = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event):
        super().__init__()
        self.config  = config
        self._cancel = cancel_event

    def run(self):
        try:
            cfg  = self.config
            mode = cfg.get("mode", "single")

            params = {
                "patch_size":          cfg.get("patch_size", 15),
                "omega":               cfg.get("omega", 0.95),
                "t_min":               cfg.get("t_min", 0.1),
                "guided_radius":       cfg.get("guided_radius", 40),
                "guided_epsilon":      cfg.get("guided_epsilon", 1e-3),
                "top_fraction":        cfg.get("top_fraction", 0.001),
                "refine_transmission": cfg.get("refine_transmission", True),
            }

            if mode == "batch":
                files   = cfg["fits_files"]
                out_dir = cfg["output_dir"]
                n       = len(files)
                os.makedirs(out_dir, exist_ok=True)
                self.log_line.emit(
                    f"Batch dehazing: {n} files → {out_dir}")

                for i, fits_path in enumerate(files):
                    if self._cancel.is_set():
                        self._abort(); return

                    self.progress.emit(i + 1, n,
                                       os.path.basename(fits_path))
                    self.log_line.emit(
                        f"[{i+1}/{n}] {os.path.basename(fits_path)}")

                    try:
                        with astropy_fits.open(fits_path) as hdul:
                            data   = hdul[0].data.astype(np.float32)
                            header = hdul[0].header.copy()

                        J, t_map, A = dehaze_image(
                            data, log_callback=None, **params)

                        out_name = os.path.basename(fits_path)
                        out_path = os.path.join(out_dir, out_name)
                        hdu = astropy_fits.PrimaryHDU(
                            data=J.astype(np.float32), header=header)
                        hdu.header["HISTORY"] = \
                            "Dark channel haze removal — dark_channel.py"
                        hdu.header["DCPOMEGA"] = params["omega"]
                        hdu.writeto(out_path, overwrite=True)
                        haze_pct = int(
                            (1.0 - float(np.mean(t_map))) * 100)
                        self.log_line.emit(
                            f"  → saved  "
                            f"(estimated haze: ~{haze_pct}%)")
                    except Exception as e:
                        self.log_line.emit(
                            f"  ERROR: {os.path.basename(fits_path)}: {e}")

                self.finished.emit({
                    "success":    True,
                    "mode":       "batch",
                    "n_files":    n,
                    "output_dir": out_dir,
                })

            else:
                # ── Single image ──────────────────────────────────────────────
                fits_path = cfg["fits_path"]
                self.progress.emit(1, 4, "Loading image...")

                with astropy_fits.open(fits_path) as hdul:
                    data   = hdul[0].data.astype(np.float32)
                    header = hdul[0].header.copy()

                h = data.shape[-2] if data.ndim == 3 else data.shape[0]
                w = data.shape[-1] if data.ndim == 3 else data.shape[1]
                self.log_line.emit(
                    f"Loaded: {w}×{h}  {os.path.basename(fits_path)}")

                # Linearity check
                med = float(np.median(data))
                rng = float(np.max(data) - np.min(data)) + 1e-10
                if med / rng > 0.25:
                    self.log_line.emit(
                        "WARNING: image may not be linear (stretched). "
                        "Results may be suboptimal.")

                if self._cancel.is_set():
                    self._abort(); return

                self.progress.emit(2, 4, "Computing dark channel prior...")
                J, t_map, A = dehaze_image(
                    data,
                    log_callback=self.log_line.emit,
                    **params
                )

                if self._cancel.is_set():
                    self._abort(); return

                self.progress.emit(3, 4, "Saving result...")

                if data.ndim == 3:
                    if data.shape[0] == 3:
                        orig_lum = (0.299 * data[0] + 0.587 * data[1] +
                                     0.114 * data[2])
                        J_lum = (0.299 * J[0] + 0.587 * J[1] +
                                  0.114 * J[2])
                    else:
                        orig_lum = data[0]
                        J_lum    = J[0]
                else:
                    orig_lum = data
                    J_lum    = J

                self.preview_ready.emit(J_lum, t_map, orig_lum)

                out_dir  = cfg.get("output_dir",
                                    os.path.dirname(fits_path))
                os.makedirs(out_dir, exist_ok=True)
                base     = os.path.splitext(
                    os.path.basename(fits_path))[0]
                out_path = os.path.join(out_dir,
                                         f"{base}_dehazed.fit")

                hdu = astropy_fits.PrimaryHDU(
                    data=J.astype(np.float32), header=header)
                hdu.header["HISTORY"] = \
                    "Dark channel haze removal — dark_channel.py"
                hdu.header["DCPOMEGA"] = params["omega"]
                hdu.header["DCPTMIN"]  = params["t_min"]
                hdu.writeto(out_path, overwrite=True)
                self.log_line.emit(f"Saved: {out_path}")

                if cfg.get("load_in_siril", True):
                    try:
                        siril = s.SirilInterface()
                        siril.connect()
                        siril.cmd("load", out_path)
                        siril.disconnect()
                    except Exception as e:
                        self.log_line.emit(f"Siril load warning: {e}")

                self.progress.emit(4, 4, "Done")
                haze_estimate = int(
                    (1.0 - float(np.mean(t_map))) * 100)
                self.finished.emit({
                    "success":           True,
                    "mode":              "single",
                    "output_path":       out_path,
                    "haze_estimate":     haze_estimate,
                    "mean_transmission": float(np.mean(t_map)),
                    "atm_light":         A.tolist(),
                })

        except Exception as e:
            import traceback
            self.log_line.emit(f"ERROR: {e}")
            self.log_line.emit(traceback.format_exc())
            self.finished.emit({"success": False, "error": str(e)})

    def _abort(self):
        self.finished.emit({"success": False, "error": "Cancelled by user"})

    def cancel(self):
        self._cancel.set()


class HazeEstimateWorker(QThread):
    """Fast worker for quick haze estimate only."""
    result   = pyqtSignal(float)
    log_line = pyqtSignal(str)

    def __init__(self, fits_path: str, patch_size: int = 15):
        super().__init__()
        self.fits_path  = fits_path
        self.patch_size = patch_size

    def run(self):
        try:
            self.log_line.emit("Estimating haze level...")
            frac = quick_haze_estimate(self.fits_path, self.patch_size)
            self.result.emit(frac)
        except Exception as e:
            self.log_line.emit(f"Haze estimate error: {e}")
            self.result.emit(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — PREVIEW CANVAS
# ═══════════════════════════════════════════════════════════════════════════════

class DehazeCanvas(FigureCanvasQTAgg):
    """3-panel preview: Original | Dehazed | Transmission map."""

    def __init__(self, parent=None):
        self.fig      = Figure(figsize=(12, 4), facecolor="#1e2128")
        self.ax_orig  = self.fig.add_subplot(1, 3, 1)
        self.ax_dehaz = self.fig.add_subplot(1, 3, 2)
        self.ax_trans = self.fig.add_subplot(1, 3, 3)
        self._style_all()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(260)

    def _style_all(self):
        configs = [
            (self.ax_orig,  "Original (hazy)",                "#e8a23a"),
            (self.ax_dehaz, "Dehazed result",                  "#4caf7d"),
            (self.ax_trans, "Transmission (white=clear)",      "#5ba3ff"),
        ]
        for ax, title, color in configs:
            ax.set_facecolor("#1e2128")
            ax.set_title(title, color=color, fontsize=8)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")
        self.fig.tight_layout(pad=0.3)

    def _ds(self, arr, max_px=500):
        h, w   = arr.shape[:2]
        factor = max(1, max(h, w) // max_px)
        return arr[::factor, ::factor]

    def show_results(self, dehazed: np.ndarray,
                      transmission: np.ndarray,
                      original: np.ndarray):
        p_lo = float(np.percentile(original, 0.5))
        p_hi = float(np.percentile(original, 99.5))
        span = p_hi - p_lo + 1e-10

        for ax, data, title, color, use_shared in [
            (self.ax_orig,  original,     "Original (hazy)",           "#e8a23a", True),
            (self.ax_dehaz, dehazed,      "Dehazed result",             "#4caf7d", True),
            (self.ax_trans, transmission, "Transmission (white=clear)", "#5ba3ff", False),
        ]:
            ax.clear()
            ds = self._ds(data)
            if use_shared:
                disp = np.clip(
                    (ds.astype(float) - p_lo) / span, 0, 1)
                ax.imshow(disp, cmap="gray", origin="lower",
                          aspect="equal", interpolation="nearest")
            else:
                ax.imshow(ds, cmap="gray", origin="lower",
                          aspect="equal", vmin=0, vmax=1,
                          interpolation="nearest")
            ax.set_title(title, color=color, fontsize=8)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")

        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_placeholder(self):
        for ax, label in [
            (self.ax_orig,  "Original"),
            (self.ax_dehaz, "Dehazed"),
            (self.ax_trans, "Transmission"),
        ]:
            ax.clear()
            ax.set_facecolor("#1e2128")
            ax.text(0.5, 0.5, label, color="#3a4055",
                    ha="center", va="center",
                    fontsize=11, transform=ax.transAxes)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#3a4055")
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — HELPER WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

def _sep():
    f = QFrame()
    f.setObjectName("separator")
    f.setFrameShape(QFrame.Shape.HLine)
    return f


def _dim_label(text):
    lbl = QLabel(text)
    lbl.setObjectName("dim")
    lbl.setWordWrap(True)
    return lbl


def _browse_row(line_edit: QLineEdit,
                caption: str,
                is_dir: bool = False) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    row.addWidget(line_edit)
    btn = QPushButton("Browse…")
    btn.setFixedWidth(72)

    def _open():
        if is_dir:
            path = QFileDialog.getExistingDirectory(
                None, caption, line_edit.text() or "")
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, caption, line_edit.text() or "",
                "FITS files (*.fit *.fits *.FIT *.FITS);;All files (*)")
        if path:
            line_edit.setText(path)

    btn.clicked.connect(_open)
    row.addWidget(btn)
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E — MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("☁  Cloud / Haze Removal  —  Siril")
        self.resize(1280, 780)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._haze_worker  = None
        self._batch_files  = []

        self._build_ui()
        self._on_mode_changed(0)   # initialise single mode

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        # Title bar
        root.addWidget(self._make_title_bar())

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._make_right_panel())
        splitter.setSizes([330, 920])
        root.addWidget(splitter, 1)

        # Bottom row
        root.addWidget(self._make_bottom_row())

        # Status bar
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setObjectName("dim")
        self.statusBar().addWidget(self.status_lbl, 1)

    def _make_title_bar(self):
        w   = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(4, 2, 4, 2)

        left = QLabel("☁  Cloud / Haze Removal  —  Siril")
        left.setStyleSheet(f"color:{SIRIL_ACCENT}; font-size:13pt; font-weight:bold;")
        row.addWidget(left)
        row.addStretch()

        right = QLabel("v1.0  |  Dark Channel Prior (He et al. 2009)")
        right.setObjectName("dim")
        row.addWidget(right)
        return w

    # ── Left panel ────────────────────────────────────────────────────────────

    def _make_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(338)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(8)

        vbox.addWidget(self._make_mode_group())
        vbox.addWidget(self._make_single_group())
        vbox.addWidget(self._make_batch_group())
        vbox.addWidget(self._make_params_group())
        vbox.addWidget(self._make_output_group())
        vbox.addWidget(self._make_haze_group())
        vbox.addStretch()

        scroll.setWidget(container)
        return scroll

    def _make_mode_group(self):
        grp  = QGroupBox("Mode")
        form = QFormLayout(grp)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single image", "Batch (folder of frames)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Processing mode:", self.mode_combo)
        return grp

    def _make_single_group(self):
        self.single_grp = QGroupBox("Input file")
        vbox = QVBoxLayout(self.single_grp)

        self.fits_edit = QLineEdit()
        self.fits_edit.setPlaceholderText("path/to/image.fits")
        vbox.addLayout(_browse_row(self.fits_edit, "Select FITS file"))

        row2 = QHBoxLayout()
        self.inspect_btn = QPushButton("🔍  Estimate haze level")
        self.inspect_btn.clicked.connect(self._on_estimate_haze)
        row2.addWidget(self.inspect_btn)
        vbox.addLayout(row2)

        self.img_info_lbl = _dim_label("No file loaded.")
        vbox.addWidget(self.img_info_lbl)
        return self.single_grp

    def _make_batch_group(self):
        self.batch_grp = QGroupBox("Batch input")
        vbox = QVBoxLayout(self.batch_grp)

        self.batch_in_edit = QLineEdit()
        self.batch_in_edit.setPlaceholderText("Input folder with FITS frames")
        vbox.addLayout(_browse_row(self.batch_in_edit,
                                    "Select input folder", is_dir=True))

        scan_btn = QPushButton("🔍  Scan folder")
        scan_btn.clicked.connect(self._on_scan_folder)
        vbox.addWidget(scan_btn)

        self.batch_info_lbl = _dim_label("No folder scanned.")
        vbox.addWidget(self.batch_info_lbl)

        vbox.addWidget(QLabel("Output folder:"))
        self.batch_out_edit = QLineEdit()
        self.batch_out_edit.setPlaceholderText("Defaults to [input]_dehazed/")
        vbox.addLayout(_browse_row(self.batch_out_edit,
                                    "Select output folder", is_dir=True))
        return self.batch_grp

    def _make_params_group(self):
        grp  = QGroupBox("Algorithm parameters")
        vbox = QVBoxLayout(grp)
        form = QFormLayout()
        form.setSpacing(6)

        # Patch size
        self.patch_spin = QSpinBox()
        self.patch_spin.setRange(5, 50)
        self.patch_spin.setSingleStep(5)
        self.patch_spin.setValue(15)
        self.patch_spin.valueChanged.connect(self._update_patch_hint)
        form.addRow("Patch size (px):", self.patch_spin)
        self.patch_hint = _dim_label(
            "Recommended — balanced between local and global correction")
        form.addRow("", self.patch_hint)

        # Omega slider
        omega_row = QHBoxLayout()
        self.omega_slider = QSlider(Qt.Orientation.Horizontal)
        self.omega_slider.setRange(50, 100)
        self.omega_slider.setValue(95)
        self.omega_lbl = QLabel("0.95")
        self.omega_lbl.setFixedWidth(36)
        self.omega_slider.valueChanged.connect(self._update_omega)
        omega_row.addWidget(self.omega_slider)
        omega_row.addWidget(self.omega_lbl)
        omega_w = QWidget(); omega_w.setLayout(omega_row)
        form.addRow("Removal strength (ω):", omega_w)
        self.omega_hint = _dim_label(
            "Standard (recommended) — strong removal with small residual")
        form.addRow("", self.omega_hint)

        # t_min slider
        tmin_row = QHBoxLayout()
        self.tmin_slider = QSlider(Qt.Orientation.Horizontal)
        self.tmin_slider.setRange(5, 40)
        self.tmin_slider.setValue(10)
        self.tmin_lbl = QLabel("0.10")
        self.tmin_lbl.setFixedWidth(36)
        self.tmin_slider.valueChanged.connect(self._update_tmin)
        tmin_row.addWidget(self.tmin_slider)
        tmin_row.addWidget(self.tmin_lbl)
        tmin_w = QWidget(); tmin_w.setLayout(tmin_row)
        form.addRow("Min transmission (t_min):", tmin_w)
        self.tmin_hint = _dim_label("Recommended — good balance")
        form.addRow("", self.tmin_hint)

        vbox.addLayout(form)

        # Guided filter
        self.refine_chk = QCheckBox(
            "Refine transmission (guided filter)")
        self.refine_chk.setChecked(True)
        self.refine_chk.stateChanged.connect(self._update_refine)
        vbox.addWidget(self.refine_chk)
        vbox.addWidget(_dim_label(
            "Smoother result, prevents halo artefacts (recommended)"))

        guided_row = QFormLayout()
        self.guided_spin = QSpinBox()
        self.guided_spin.setRange(10, 100)
        self.guided_spin.setValue(40)
        guided_row.addRow("Guided filter radius:", self.guided_spin)
        self.guided_widget = QWidget()
        self.guided_widget.setLayout(guided_row)
        vbox.addWidget(self.guided_widget)

        return grp

    def _make_output_group(self):
        self.output_grp = QGroupBox("Output")
        vbox = QVBoxLayout(self.output_grp)

        vbox.addWidget(QLabel("Output folder (leave blank = same as input):"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("(same directory as input)")
        vbox.addLayout(_browse_row(self.out_edit,
                                    "Select output folder", is_dir=True))

        self.load_siril_chk = QCheckBox("Load result in Siril after saving")
        self.load_siril_chk.setChecked(True)
        vbox.addWidget(self.load_siril_chk)

        return self.output_grp

    def _make_haze_group(self):
        grp  = QGroupBox("Haze estimate")
        vbox = QVBoxLayout(grp)

        self.haze_lbl = QLabel("—")
        self.haze_lbl.setObjectName("ok")
        self.haze_lbl.setStyleSheet(f"font-size:14pt; font-weight:bold;")
        vbox.addWidget(self.haze_lbl)

        self.haze_desc = _dim_label("Run estimate or process an image first.")
        vbox.addWidget(self.haze_desc)
        vbox.addWidget(_dim_label("Mean (1 – transmission)"))

        return grp

    # ── Right panel ───────────────────────────────────────────────────────────

    def _make_right_panel(self):
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Preview tab
        preview_w = QWidget()
        p_vbox = QVBoxLayout(preview_w)
        self.canvas = DehazeCanvas()
        self.canvas.show_placeholder()
        p_vbox.addWidget(self.canvas)
        p_vbox.addWidget(_dim_label(
            "Transmission map: white = clear sky, black = heavy haze   "
            "|   Same stretch applied to Original and Dehazed for fair comparison"))
        self.tabs.addTab(preview_w, "🖼  Preview")

        # Log tab
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.tabs.addTab(self.log_edit, "📋  Log")

        vbox.addWidget(self.tabs)
        return w

    # ── Bottom row ────────────────────────────────────────────────────────────

    def _make_bottom_row(self):
        w   = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        row.addWidget(self.progress_bar, 1)

        self.run_btn = QPushButton("▶   Remove haze")
        self.run_btn.setObjectName("primary")
        self.run_btn.setFixedWidth(160)
        self.run_btn.clicked.connect(self._on_run)
        row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("✕  Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setFixedWidth(90)
        self.cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(self.cancel_btn)

        return w

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_mode_changed(self, idx):
        is_batch = (idx == 1)
        self.single_grp.setVisible(not is_batch)
        self.batch_grp.setVisible(is_batch)
        self.output_grp.setVisible(not is_batch)

    def _update_patch_hint(self, val):
        if val <= 10:
            txt = "Fine patches — corrects local thin cloud wisps, slower"
        elif val <= 15:
            txt = "Recommended — balanced between local and global correction"
        elif val <= 30:
            txt = "Coarse patches — corrects uniform haze layers, faster"
        else:
            txt = "Very coarse — only for large-scale transparency gradients"
        self.patch_hint.setText(txt)

    def _update_omega(self, val):
        v = val / 100.0
        self.omega_lbl.setText(f"{v:.2f}")
        if v <= 0.70:
            txt = "Mild correction — subtle improvement, very safe"
        elif v <= 0.90:
            txt = "Moderate — significant improvement, minimal artefacts"
        elif v < 1.00:
            txt = "Standard (recommended) — strong removal with small residual"
        else:
            txt = "Maximum — full removal, risk of halo artefacts near bright stars"
        self.omega_hint.setText(txt)

    def _update_tmin(self, val):
        v = val / 100.0
        self.tmin_lbl.setText(f"{v:.2f}")
        if v <= 0.05:
            txt = "Aggressive — strong correction even in dense haze (risk: noise)"
        elif v <= 0.12:
            txt = "Recommended — good balance"
        elif v <= 0.25:
            txt = "Conservative — safer, less correction in very hazy areas"
        else:
            txt = "Minimal — only light haze areas corrected"
        self.tmin_hint.setText(txt)

    def _update_refine(self, state):
        self.guided_widget.setEnabled(bool(state))

    def _on_scan_folder(self):
        folder = self.batch_in_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder",
                                "Please select a valid input folder first.")
            return
        patterns = ["*.fit", "*.fits", "*.FIT", "*.FITS"]
        self._batch_files = []
        for pat in patterns:
            self._batch_files += glob.glob(
                os.path.join(folder, pat))
        self._batch_files = sorted(set(self._batch_files))
        n = len(self._batch_files)
        if n == 0:
            self.batch_info_lbl.setText("No FITS files found.")
        else:
            self.batch_info_lbl.setText(f"{n} FITS files found.")
        # Auto-fill output dir
        if not self.batch_out_edit.text():
            self.batch_out_edit.setText(folder.rstrip("/\\") + "_dehazed")

    def _on_estimate_haze(self):
        fits_path = self.fits_edit.text().strip()
        if not fits_path or not os.path.isfile(fits_path):
            QMessageBox.warning(self, "No file",
                                "Please select a FITS file first.")
            return
        self.inspect_btn.setEnabled(False)
        self._haze_worker = HazeEstimateWorker(
            fits_path, self.patch_spin.value())
        self._haze_worker.result.connect(self._on_haze_estimate_done)
        self._haze_worker.log_line.connect(self._append_log)
        self._haze_worker.finished.connect(
            lambda: self.inspect_btn.setEnabled(True))
        self._haze_worker.start()

    def _on_haze_estimate_done(self, frac):
        pct = int(frac * 100)
        self._update_haze_display(pct)
        self._append_log(f"Quick haze estimate: ~{pct}%")

    def _update_haze_display(self, pct: int):
        self.haze_lbl.setText(f"Estimated haze: ~{pct}%")
        if pct < 10:
            color = SIRIL_SUCCESS
            desc  = "Light haze — good candidate for recovery"
        elif pct <= 30:
            color = SIRIL_WARNING
            desc  = "Moderate haze — recovery recommended"
        else:
            color = SIRIL_ERROR
            desc  = "Heavy haze — result may show artefacts"
        self.haze_lbl.setStyleSheet(
            f"color:{color}; font-size:14pt; font-weight:bold;")
        self.haze_desc.setText(desc)

    def _on_run(self):
        is_batch = self.mode_combo.currentIndex() == 1

        if is_batch:
            if not self._batch_files:
                QMessageBox.warning(self, "No files",
                                    "Scan a folder first.")
                return
            out_dir = self.batch_out_edit.text().strip()
            if not out_dir:
                folder  = self.batch_in_edit.text().strip()
                out_dir = folder.rstrip("/\\") + "_dehazed"
            cfg = {
                "mode":       "batch",
                "fits_files": self._batch_files,
                "output_dir": out_dir,
            }
        else:
            fits_path = self.fits_edit.text().strip()
            if not fits_path or not os.path.isfile(fits_path):
                QMessageBox.warning(self, "No file",
                                    "Please select a FITS file.")
                return
            out_dir = self.out_edit.text().strip()
            if not out_dir:
                out_dir = os.path.dirname(fits_path)
            cfg = {
                "mode":          "single",
                "fits_path":     fits_path,
                "output_dir":    out_dir,
                "load_in_siril": self.load_siril_chk.isChecked(),
            }

        cfg.update({
            "patch_size":          self.patch_spin.value(),
            "omega":               self.omega_slider.value() / 100.0,
            "t_min":               self.tmin_slider.value() / 100.0,
            "guided_radius":       self.guided_spin.value(),
            "guided_epsilon":      1e-3,
            "top_fraction":        0.001,
            "refine_transmission": self.refine_chk.isChecked(),
        })

        self._cancel_event.clear()
        self._worker = DehazeWorker(cfg, self._cancel_event)
        self._worker.log_line.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.preview_ready.connect(self._on_preview)
        self._worker.finished.connect(self._on_finished)

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.tabs.setCurrentIndex(1)  # show log while running

        self._append_log("=" * 56)
        self._append_log(
            f"[{datetime.now().strftime('%H:%M:%S')}]  "
            f"Starting {'batch' if is_batch else 'single'} dehazing…")
        self._worker.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self._append_log("Cancel requested…")

    def _on_progress(self, current, total, label):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_lbl.setText(f"[{current}/{total}]  {label}")

    def _on_preview(self, dehazed, t_map, original):
        self.canvas.show_results(dehazed, t_map, original)
        self.tabs.setCurrentIndex(0)

    def _on_finished(self, result):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        if result.get("success"):
            if result["mode"] == "single":
                pct = result.get("haze_estimate", 0)
                self._update_haze_display(pct)
                self.status_lbl.setText(
                    f"✓  Done  |  Haze: ~{pct}%  |  "
                    f"Saved: {result.get('output_path','')}")
                self._append_log(
                    f"✓ Completed.  Haze estimate: ~{pct}%")
            else:
                n       = result.get("n_files", 0)
                out_dir = result.get("output_dir", "")
                self.status_lbl.setText(
                    f"✓  {n} files processed  |  Output: {out_dir}")
                self._append_log(f"✓ Batch complete. {n} files → {out_dir}")
        else:
            err = result.get("error", "Unknown error")
            self.status_lbl.setText(f"✗  {err}")
            self._append_log(f"✗ {err}")

    def _append_log(self, text):
        self.log_edit.appendPlainText(text)
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())


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
