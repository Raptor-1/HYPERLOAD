# WinsorizedSigmaStack.py
# SPDX-License-Identifier: GPL-3.0-or-later
# Vectorized Winsorized Sigma-Clipped Stacking Engine for Siril
# Compatible with Siril 1.4+ Python scripting API (PyQt6)

"""
Vectorized Winsorized Sigma-Clipped Stacking Engine

Provides an advanced stacking workflow with full control over:
  - Winsorized sigma clipping parameters (low/high thresholds)
  - Normalization method (None, Additive, Multiplicative, Additive+Scale)
  - Stacking method (Mean, Median, Sum)
  - Output filename and sequence selection
  - Per-channel weight maps (optional)
  - Rejection map saving

Requires Siril 1.4.0+ with Python scripting enabled.
"""

import sys
import os

import sirilpy as s
from sirilpy import SirilError

s.ensure_installed("PyQt6")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QPushButton, QGroupBox, QLineEdit, QProgressBar, QTextEdit,
    QSizePolicy, QFrame, QGridLayout, QSplitter, QFileDialog,
    QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QTextCursor

VERSION = "1.0.0"

# ─── Siril dark theme palette ────────────────────────────────────────────────
SIRIL_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #2b2b2b;
    color: #d4d4d4;
    font-family: "Cantarell", "Segoe UI", sans-serif;
    font-size: 10pt;
}

QGroupBox {
    background-color: #333333;
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #e0e0e0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #7ec8e3;
    font-size: 10pt;
}

QLabel {
    color: #d4d4d4;
    background: transparent;
}

QLabel#header {
    color: #7ec8e3;
    font-size: 13pt;
    font-weight: bold;
}

QLabel#subheader {
    color: #888888;
    font-size: 9pt;
}

QLabel#statusLabel {
    color: #aaaaaa;
    font-size: 9pt;
    padding: 2px 0;
}

QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #555555;
    border-radius: 3px;
    color: #d4d4d4;
    padding: 3px 6px;
    min-height: 22px;
    selection-background-color: #3daee9;
}

QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border: 1px solid #7ec8e3;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #444444;
    border: none;
    width: 16px;
}

QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #555555;
}

QComboBox::drop-down {
    border: none;
    background-color: #444444;
    width: 22px;
}

QComboBox QAbstractItemView {
    background-color: #2b2b2b;
    border: 1px solid #555555;
    selection-background-color: #3daee9;
    color: #d4d4d4;
}

QCheckBox {
    color: #d4d4d4;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #666666;
    border-radius: 2px;
    background-color: #1e1e1e;
}

QCheckBox::indicator:checked {
    background-color: #3daee9;
    border-color: #3daee9;
}

QPushButton {
    background-color: #3c3f41;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #d4d4d4;
    padding: 5px 16px;
    min-height: 26px;
}

QPushButton:hover {
    background-color: #4a4d50;
    border-color: #7ec8e3;
}

QPushButton:pressed {
    background-color: #2a2d2f;
}

QPushButton#runButton {
    background-color: #2e6da4;
    border: 1px solid #3daee9;
    color: #ffffff;
    font-weight: bold;
    font-size: 11pt;
    min-height: 32px;
}

QPushButton#runButton:hover {
    background-color: #3a80c1;
}

QPushButton#runButton:pressed {
    background-color: #235a8a;
}

QPushButton#runButton:disabled {
    background-color: #333333;
    border-color: #444444;
    color: #666666;
}

QPushButton#cancelButton {
    background-color: #7a2020;
    border: 1px solid #a33333;
    color: #ffffff;
}

QPushButton#cancelButton:hover {
    background-color: #9a2828;
}

QPushButton#cancelButton:disabled {
    background-color: #333333;
    border-color: #444444;
    color: #666666;
}

QPushButton#importButton {
    background-color: #2e5c3a;
    border: 1px solid #4caf70;
    color: #ffffff;
    font-weight: bold;
    min-height: 26px;
}

QPushButton#importButton:hover {
    background-color: #3a7a50;
}

QPushButton#importButton:disabled {
    background-color: #333333;
    border-color: #444444;
    color: #666666;
}

QProgressBar {
    border: 1px solid #555555;
    border-radius: 3px;
    background-color: #1e1e1e;
    text-align: center;
    color: #d4d4d4;
    min-height: 18px;
}

QProgressBar::chunk {
    background-color: #2e6da4;
    border-radius: 2px;
}

QTextEdit {
    background-color: #1a1a1a;
    border: 1px solid #444444;
    border-radius: 3px;
    color: #b0b0b0;
    font-family: "Courier New", "Consolas", monospace;
    font-size: 9pt;
    padding: 4px;
}

QFrame#separator {
    color: #555555;
}

QSplitter::handle {
    background-color: #444444;
    height: 2px;
}
"""

# ─── Import worker thread ──────────────────────────────────────────────────
class ImportWorker(QObject):
    log_message = pyqtSignal(str, str)
    finished    = pyqtSignal(bool, str)   # (success, seq_basename)

    def __init__(self, siril, folder, basename, fmt, debayer):
        super().__init__()
        self.siril    = siril
        self.folder   = folder
        self.basename = basename
        self.fmt      = fmt       # "FITS Sequence (.seq)", "Single FITS (.fits)", "SER (.ser)"
        self.debayer  = debayer

    def run(self):
        try:
            siril = self.siril
            folder   = self.folder
            basename = self.basename

            self.log_message.emit("─" * 40, "info")
            self.log_message.emit(f"Importing from: {folder}", "info")
            self.log_message.emit(f"Basename: {basename}  |  Format: {self.fmt}", "info")

            # Change working directory
            siril.cmd("cd", folder)
            self.log_message.emit(f"Working dir set to {folder}", "ok")

            if self.fmt == "SER (.ser)":
                # Load SER file directly — user supplies the .ser filename as basename
                ser_path = basename if basename.endswith(".ser") else basename + ".ser"
                siril.cmd("load", ser_path)
                self.log_message.emit(f"SER file loaded: {ser_path}", "ok")
                self.finished.emit(True, basename)
                return

            if self.fmt == "Existing .seq file":
                # Already converted — just load the sequence
                siril.cmd("seqload", basename)
                self.log_message.emit(f"Sequence loaded: {basename}", "ok")
                self.finished.emit(True, basename)
                return

            # Default: convert images in folder → FITS sequence
            convert_args = ["convert", basename]
            if self.fmt == "Single FITSEQ (-fitseq)":
                convert_args.append("-fitseq")
            if self.debayer:
                convert_args.append("-debayer")

            self.log_message.emit("Running convert…", "info")
            siril.cmd(*convert_args)
            self.log_message.emit("Convert complete.", "ok")

            # Load the resulting sequence
            siril.cmd("seqload", basename)
            self.log_message.emit(f"Sequence loaded: {basename}", "ok")
            self.finished.emit(True, basename)

        except SirilError as e:
            self.log_message.emit(f"Import error: {e}", "error")
            self.finished.emit(False, "")
        except Exception as e:
            self.log_message.emit(f"Unexpected import error: {e}", "error")
            self.finished.emit(False, "")


# ─── Worker thread ─────────────────────────────────────────────────────────
class StackWorker(QObject):
    log_message   = pyqtSignal(str, str)   # (message, level)  level: info|warn|error|ok
    progress      = pyqtSignal(int)
    finished      = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, siril, params):
        super().__init__()
        self.siril  = siril
        self.params = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    # ── Core algorithm ─────────────────────────────────────────────────────
    @staticmethod
    def _winsorized_sigma_clip_stack(stack, sigma_low, sigma_high, iterations, method, weights=None):
        """
        Vectorized winsorized sigma-clipped stack.

        Parameters
        ----------
        stack       : np.ndarray  shape (N, H, W), float32
        sigma_low   : float       lower clipping threshold in σ units
        sigma_high  : float       upper clipping threshold in σ units
        iterations  : int         maximum clipping iterations
        method      : str         'mean' | 'median' | 'sum'
        weights     : np.ndarray | None  shape (N,1,1) normalised to sum=1,
                      applied during the mean combine step only

        Returns
        -------
        result      : np.ndarray  shape (H, W), float32  — combined image
        low_map     : np.ndarray  shape (H, W), uint16   — low rejection counts
        high_map    : np.ndarray  shape (H, W), uint16   — high rejection counts

        Algorithm
        ---------
        Standard sigma-clip MASKS outliers (sets them to NaN and excludes them
        from the mean). Winsorized sigma-clip instead CLAMPS outliers to the
        boundary value (median ± σ × threshold), preserving pixel count and
        suppressing noise from bright/dark artefacts without discarding data.
        The iteration converges when no pixel changes between passes.
        """
        import numpy as np

        work = stack.astype(np.float32, copy=True)   # (N, H, W)
        N    = work.shape[0]

        low_rejected  = np.zeros(work.shape[1:], dtype=np.uint16)
        high_rejected = np.zeros(work.shape[1:], dtype=np.uint16)

        for iteration in range(iterations):
            med    = np.median(work, axis=0)                     # (H, W)
            diff   = work - med[np.newaxis, :, :]                # (N, H, W)
            # MAD-based σ estimate (robust, consistent with Siril's approach)
            mad    = np.median(np.abs(diff), axis=0)             # (H, W)
            sigma  = mad * 1.4826                                 # (H, W)  MAD → σ

            lo_bound = med - sigma_low  * sigma                  # (H, W)
            hi_bound = med + sigma_high * sigma                  # (H, W)

            lo_clip = lo_bound[np.newaxis, :, :]                 # broadcast
            hi_clip = hi_bound[np.newaxis, :, :]

            prev = work.copy()

            # Winsorize: clamp to boundary, do NOT mask
            work = np.clip(work, lo_clip, hi_clip)

            # Track how many pixels were clamped this iteration
            low_rejected  += (work < prev).any(axis=0).astype(np.uint16)
            high_rejected += (prev < work).any(axis=0).astype(np.uint16)

            # Convergence: no pixel changed
            if np.array_equal(work, prev):
                break

        # Combine the winsorized stack
        if method == "median":
            result = np.median(work, axis=0)
        elif method == "sum":
            result = np.sum(work, axis=0)
        else:  # weighted or plain mean
            if weights is not None:
                # weights shape: (N,1,1) — broadcast over H×W
                result = np.sum(work * weights, axis=0)
            else:
                result = np.mean(work, axis=0)

        return result.astype(np.float32), low_rejected, high_rejected

    def run(self):
        import numpy as np

        try:
            p     = self.params
            siril = self.siril

            self.log_message.emit("═" * 52, "info")
            self.log_message.emit("  Vectorized Winsorized Sigma-Clipped Stack", "info")
            self.log_message.emit(f"  v{VERSION}", "info")
            self.log_message.emit("═" * 52, "info")
            self.progress.emit(2)

            # ── Validate sequence ─────────────────────────────────────────
            self.log_message.emit("Checking loaded sequence…", "info")
            try:
                seq_name = siril.get_seq_name()
                if not seq_name:
                    raise SirilError("No sequence loaded.")
                seq_info = siril.get_sequence_info()
            except SirilError:
                raise
            except Exception:
                raise SirilError("No sequence is loaded in Siril. Please load a sequence first.")

            n_total    = getattr(seq_info, "number", 0)
            n_included = getattr(seq_info, "selnum", n_total)
            self.log_message.emit(f"Sequence : {os.path.basename(seq_name)}", "ok")
            self.log_message.emit(f"Frames   : {n_included} included / {n_total} total", "info")

            if n_included < 2:
                raise SirilError(f"Need at least 2 included frames, found {n_included}.")

            sigma_low   = p["sigma_low"]
            sigma_high  = p["sigma_high"]
            iterations  = p["iterations"]
            method      = p["method"].lower()    # mean | median | sum
            output_name = p["output_name"].strip() or "stacked_result"

            self.log_message.emit("", "info")
            self.log_message.emit("Stack parameters:", "info")
            self.log_message.emit(f"  Algorithm  : Vectorized Winsorized Sigma Clip", "info")
            self.log_message.emit(f"  Combine    : {p['method']}", "info")
            self.log_message.emit(f"  σ low      : {sigma_low}", "info")
            self.log_message.emit(f"  σ high     : {sigma_high}", "info")
            self.log_message.emit(f"  Iterations : {iterations}", "info")
            self.log_message.emit(f"  Norm       : {p['norm']}", "info")
            self.log_message.emit(f"  Output     : {output_name}.fit", "info")
            self.log_message.emit("", "info")
            self.progress.emit(5)

            if self._cancel:
                self.finished.emit(False, "Cancelled by user.")
                return

            # ── Optional pre-normalization via Siril command ──────────────
            norm_map = {
                "None":               "no",
                "Additive":           "add",
                "Multiplicative":     "mul",
                "Additive + Scaling": "addscale",
            }
            norm_mode = norm_map.get(p["norm"], "no")
            if norm_mode != "no" and p["prenormalize"]:
                self.log_message.emit("Pre-normalizing sequence…", "info")
                try:
                    siril.cmd("seqnorm", seq_name, norm_mode)
                    self.log_message.emit("Pre-normalization complete.", "ok")
                except Exception as e:
                    self.log_message.emit(f"Pre-norm warning (continuing): {e}", "warn")
            self.progress.emit(8)

            if self._cancel:
                self.finished.emit(False, "Cancelled by user.")
                return

            # ── Load all included frames into a NumPy stack ───────────────
            self.log_message.emit(f"Loading {n_included} frames into memory…", "info")
            frames      = []
            raw_scores  = []   # per-frame quality scores for weighting
            ref_fit     = None
            loaded      = 0
            frame_index = 0   # 0-based index into the full sequence

            while loaded < n_included:
                if self._cancel:
                    self.finished.emit(False, "Cancelled during frame load.")
                    return

                # get_sequence_info gives included flags per frame
                try:
                    frame_info = siril.get_seq_frame(frame_index)
                except Exception:
                    frame_index += 1
                    if frame_index >= n_total:
                        break
                    continue

                # Skip excluded frames (included flag == False)
                included_flag = getattr(frame_info, "included", True)
                if not included_flag:
                    frame_index += 1
                    continue

                data = frame_info.data  # numpy array (H, W) mono or (H, W, C) colour
                if data is None:
                    frame_index += 1
                    continue

                # Normalise to float32 [0,1]
                if data.dtype == np.uint16:
                    data = data.astype(np.float32) / 65535.0
                elif data.dtype != np.float32:
                    data = data.astype(np.float32)

                # Colour images: shape (H,W,C) → keep as-is, stack per-channel later
                frames.append(data)
                if ref_fit is None:
                    ref_fit = frame_info   # keep for metadata

                # Collect quality metric for weighting
                metric = p.get("weight_metric", "wFWHM")
                score  = 0.0
                try:
                    ia = getattr(frame_info, "image_analysis", None)
                    if ia is None:
                        ia = getattr(frame_info, "analysis", None)
                    if ia is not None:
                        if metric == "wFWHM":
                            raw = getattr(ia, "wfwhm", 0.0) or getattr(ia, "fwhm", 0.0)
                            # wFWHM: lower is better → invert
                            score = 1.0 / raw if raw > 0 else 0.0
                        elif metric == "FWHM":
                            raw = getattr(ia, "fwhm", 0.0)
                            score = 1.0 / raw if raw > 0 else 0.0
                        elif metric == "Noise":
                            raw = getattr(ia, "bgnoise", 0.0)
                            score = 1.0 / raw if raw > 0 else 0.0
                        elif metric == "#":
                            score = float(getattr(ia, "nbstars", 0))
                except Exception:
                    score = 0.0
                raw_scores.append(score)

                loaded += 1
                frame_index += 1
                pct = 8 + int(42 * loaded / n_included)
                self.progress.emit(pct)
                if loaded % max(1, n_included // 10) == 0:
                    self.log_message.emit(f"  Loaded {loaded}/{n_included}…", "info")

            if len(frames) < 2:
                raise SirilError(f"Could only load {len(frames)} frames — need at least 2.")

            self.log_message.emit(f"All {len(frames)} frames loaded.", "ok")
            self.progress.emit(50)

            if self._cancel:
                self.finished.emit(False, "Cancelled after frame load.")
                return

            # ── Compute frame weights ─────────────────────────────────────
            weights_arr = None
            if p["use_weights"]:
                power  = p.get("weight_power", 2.0)
                metric = p.get("weight_metric", "wFWHM")
                scores = np.array(raw_scores, dtype=np.float64)
                valid  = scores > 0

                if valid.any():
                    # Raise scores to the chosen power, then normalize to sum=1
                    scores[~valid] = scores[valid].min()   # fallback: give worst score
                    scores = scores ** power
                    scores /= scores.sum()
                    weights_arr = scores.astype(np.float32)[:, np.newaxis, np.newaxis]
                    self.log_message.emit(
                        f"Frame weights ({metric}, power={power:.1f}): "
                        f"min={weights_arr.min():.4f}  max={weights_arr.max():.4f}", "info")
                else:
                    self.log_message.emit(
                        f"No quality data found for metric '{metric}' — "
                        "falling back to equal weights. Run registration first.", "warn")

            # ── Run vectorized winsorized sigma-clip ──────────────────────
            self.log_message.emit("Running vectorized winsorized sigma-clip…", "info")

            sample     = frames[0]
            is_colour  = (sample.ndim == 3)
            n_channels = sample.shape[2] if is_colour else 1

            if is_colour:
                result_channels   = []
                lo_maps, hi_maps  = [], []
                for ch in range(n_channels):
                    if self._cancel:
                        self.finished.emit(False, "Cancelled during stacking.")
                        return
                    ch_stack = np.stack([f[:, :, ch] for f in frames], axis=0)  # (N,H,W)
                    self.log_message.emit(
                        f"  Channel {ch+1}/{n_channels}: "
                        f"shape {ch_stack.shape}, dtype {ch_stack.dtype}", "info")
                    res, lo, hi = self._winsorized_sigma_clip_stack(
                        ch_stack, sigma_low, sigma_high, iterations, method, weights_arr)
                    result_channels.append(res)
                    lo_maps.append(lo)
                    hi_maps.append(hi)
                    self.progress.emit(50 + int(30 * (ch + 1) / n_channels))

                result = np.stack(result_channels, axis=2)   # (H, W, C)
                lo_rej = np.stack(lo_maps,         axis=2)
                hi_rej = np.stack(hi_maps,         axis=2)
            else:
                stack_arr = np.stack(frames, axis=0)         # (N, H, W)
                self.log_message.emit(
                    f"  Stack shape: {stack_arr.shape}, dtype: {stack_arr.dtype}", "info")
                result, lo_rej, hi_rej = self._winsorized_sigma_clip_stack(
                    stack_arr, sigma_low, sigma_high, iterations, method, weights_arr)
                self.progress.emit(80)

            total_lo = int(lo_rej.sum())
            total_hi = int(hi_rej.sum())
            self.log_message.emit(
                f"Winsorization complete — "
                f"low clamps: {total_lo:,}  high clamps: {total_hi:,}", "ok")

            # ── Optional output normalization ─────────────────────────────
            if p["output_norm"]:
                vmin, vmax = result.min(), result.max()
                if vmax > vmin:
                    result = (result - vmin) / (vmax - vmin)
                self.log_message.emit("Output normalization applied.", "info")

            self.progress.emit(85)

            if self._cancel:
                self.finished.emit(False, "Cancelled before save.")
                return

            # ── Push result back into Siril and save ──────────────────────
            self.log_message.emit("Saving result via Siril…", "info")

            # Load the first frame as the active image to inherit its header/metadata
            siril.cmd("load", f"{os.path.basename(seq_name)}_00001")

            with siril.image_lock():
                fit      = siril.get_image()
                fit.data[:] = result
                siril.set_image_pixeldata(fit.data)

            siril.cmd("save", output_name)
            self.log_message.emit(f"Saved: {output_name}.fit", "ok")

            # ── Optional rejection maps ───────────────────────────────────
            if p["save_rejection"]:
                self.log_message.emit("Saving rejection maps…", "info")
                with siril.image_lock():
                    fit      = siril.get_image()
                    fit.data[:] = lo_rej.astype(np.float32) / max(1, len(frames))
                    siril.set_image_pixeldata(fit.data)
                siril.cmd("save", f"{output_name}_rejection_low")

                with siril.image_lock():
                    fit      = siril.get_image()
                    fit.data[:] = hi_rej.astype(np.float32) / max(1, len(frames))
                    siril.set_image_pixeldata(fit.data)
                siril.cmd("save", f"{output_name}_rejection_high")
                self.log_message.emit("Rejection maps saved.", "ok")

            # Reload the finished stack as the active image
            siril.cmd("load", output_name)

            self.log_message.emit("", "info")
            self.log_message.emit(f"Stack complete → {output_name}.fit", "ok")
            self.log_message.emit(
                f"Frames used: {len(frames)}  |  "
                f"Low clamps: {total_lo:,}  |  High clamps: {total_hi:,}", "info")
            self.progress.emit(100)
            self.finished.emit(True, f"Output: {output_name}.fit")

        except SirilError as e:
            self.log_message.emit(f"Siril error: {e}", "error")
            self.finished.emit(False, str(e))
        except MemoryError:
            self.log_message.emit(
                "Out of memory — sequence may be too large. "
                "Try reducing frame count or image size.", "error")
            self.finished.emit(False, "Out of memory")
        except Exception as e:
            self.log_message.emit(f"Unexpected error: {e}", "error")
            self.finished.emit(False, str(e))


# ─── Main GUI ─────────────────────────────────────────────────────────────
class WinsorizedStackUI(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.siril  = siril
        self.worker = None
        self.thread = None
        self._build_ui()
        self._detect_sequence()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle(f"Winsorized Sigma-Clipped Stack  —  v{VERSION}")
        self.setMinimumWidth(520)
        self.setMaximumWidth(640)
        self.setStyleSheet(SIRIL_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(8)
        root_layout.setContentsMargins(12, 12, 12, 12)

        # ── Header ────────────────────────────────────────────────────────
        hdr_layout = QVBoxLayout()
        hdr_layout.setSpacing(2)
        title = QLabel("Vectorized Winsorized Sigma-Clipped Stack")
        title.setObjectName("header")
        subtitle = QLabel("Advanced rejection stacking engine for Siril")
        subtitle.setObjectName("subheader")
        hdr_layout.addWidget(title)
        hdr_layout.addWidget(subtitle)
        root_layout.addLayout(hdr_layout)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        root_layout.addWidget(sep)

        # ── Import Sequence ───────────────────────────────────────────────
        imp_grp = QGroupBox("Import Sequence")
        imp_layout = QGridLayout(imp_grp)
        imp_layout.setColumnStretch(1, 1)

        imp_layout.addWidget(QLabel("Folder:"), 0, 0)
        self.txt_imp_folder = QLineEdit()
        self.txt_imp_folder.setPlaceholderText("Directory containing images…")
        imp_layout.addWidget(self.txt_imp_folder, 0, 1)
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.setToolTip("Browse for image folder")
        btn_browse.clicked.connect(self._browse_import_folder)
        imp_layout.addWidget(btn_browse, 0, 2)

        imp_layout.addWidget(QLabel("Basename:"), 1, 0)
        self.txt_imp_basename = QLineEdit("lights")
        self.txt_imp_basename.setPlaceholderText("Sequence base name (e.g. lights)")
        imp_layout.addWidget(self.txt_imp_basename, 1, 1, 1, 2)

        imp_layout.addWidget(QLabel("Format:"), 2, 0)
        self.cmb_imp_fmt = QComboBox()
        self.cmb_imp_fmt.addItems([
            "Individual images → FITS sequence",
            "Individual images → Single FITSEQ (-fitseq)",
            "Existing .seq file",
            "SER (.ser)",
        ])
        self.cmb_imp_fmt.currentIndexChanged.connect(self._update_import_ui)
        imp_layout.addWidget(self.cmb_imp_fmt, 2, 1, 1, 2)

        self.chk_imp_debayer = QCheckBox("Debayer (CFA / OSC cameras)")
        imp_layout.addWidget(self.chk_imp_debayer, 3, 0, 1, 2)

        self.btn_import = QPushButton("⇩  Convert & Load")
        self.btn_import.setObjectName("importButton")
        self.btn_import.clicked.connect(self._on_import)
        imp_layout.addWidget(self.btn_import, 3, 2)

        root_layout.addWidget(imp_grp)

        sep2 = QFrame()
        sep2.setObjectName("separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        root_layout.addWidget(sep2)

        # ── Sequence info ─────────────────────────────────────────────────
        seq_grp = QGroupBox("Sequence")
        seq_layout = QGridLayout(seq_grp)
        seq_layout.setColumnStretch(1, 1)

        seq_layout.addWidget(QLabel("Loaded:"), 0, 0)
        self.lbl_seq = QLabel("—")
        self.lbl_seq.setObjectName("statusLabel")
        seq_layout.addWidget(self.lbl_seq, 0, 1)

        seq_layout.addWidget(QLabel("Frames:"), 1, 0)
        self.lbl_frames = QLabel("—")
        self.lbl_frames.setObjectName("statusLabel")
        seq_layout.addWidget(self.lbl_frames, 1, 1)

        refresh_btn = QPushButton("↺  Refresh")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self._detect_sequence)
        seq_layout.addWidget(refresh_btn, 0, 2, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        root_layout.addWidget(seq_grp)

        # ── Rejection parameters ──────────────────────────────────────────
        rej_grp = QGroupBox("Rejection Algorithm")
        rej_layout = QGridLayout(rej_grp)
        rej_layout.setColumnStretch(1, 1)

        algo_lbl = QLabel("Vectorized Winsorized Sigma Clip")
        algo_lbl.setStyleSheet("color: #6ec96e; font-style: italic;")
        rej_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        rej_layout.addWidget(algo_lbl, 0, 1, 1, 3)

        rej_layout.addWidget(QLabel("σ Low:"), 1, 0)
        self.spn_sigma_low = QDoubleSpinBox()
        self.spn_sigma_low.setRange(0.1, 10.0)
        self.spn_sigma_low.setSingleStep(0.1)
        self.spn_sigma_low.setValue(3.0)
        self.spn_sigma_low.setDecimals(2)
        self.spn_sigma_low.setSuffix("  σ")
        rej_layout.addWidget(self.spn_sigma_low, 1, 1)

        rej_layout.addWidget(QLabel("σ High:"), 1, 2)
        self.spn_sigma_high = QDoubleSpinBox()
        self.spn_sigma_high.setRange(0.1, 10.0)
        self.spn_sigma_high.setSingleStep(0.1)
        self.spn_sigma_high.setValue(3.0)
        self.spn_sigma_high.setDecimals(2)
        self.spn_sigma_high.setSuffix("  σ")
        rej_layout.addWidget(self.spn_sigma_high, 1, 3)

        rej_layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.spn_iterations = QSpinBox()
        self.spn_iterations.setRange(1, 20)
        self.spn_iterations.setValue(5)
        rej_layout.addWidget(self.spn_iterations, 2, 1)

        self.chk_save_rej = QCheckBox("Save rejection map")
        rej_layout.addWidget(self.chk_save_rej, 2, 2, 1, 2)
        root_layout.addWidget(rej_grp)

        # ── Stacking method ───────────────────────────────────────────────
        stack_grp = QGroupBox("Stacking Method")
        stack_layout = QGridLayout(stack_grp)
        stack_layout.setColumnStretch(1, 1)

        stack_layout.addWidget(QLabel("Method:"), 0, 0)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["Mean", "Median", "Sum"])
        stack_layout.addWidget(self.cmb_method, 0, 1, 1, 3)

        stack_layout.addWidget(QLabel("Normalization:"), 1, 0)
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems([
            "None",
            "Additive",
            "Multiplicative",
            "Additive + Scaling",
        ])
        self.cmb_norm.setCurrentIndex(0)
        stack_layout.addWidget(self.cmb_norm, 1, 1, 1, 3)

        self.chk_prenorm = QCheckBox("Pre-normalize sequence (seqnorm)")
        self.chk_prenorm.setChecked(False)
        stack_layout.addWidget(self.chk_prenorm, 2, 0, 1, 4)

        self.chk_output_norm = QCheckBox("Output normalization (scale result to [0,1])")
        self.chk_output_norm.setChecked(False)
        stack_layout.addWidget(self.chk_output_norm, 3, 0, 1, 4)

        root_layout.addWidget(stack_grp)

        # ── Frame Weighting ───────────────────────────────────────────────
        wt_grp = QGroupBox("Frame Weighting")
        wt_layout = QGridLayout(wt_grp)
        wt_layout.setColumnStretch(1, 1)

        self.chk_weights = QCheckBox("Enable frame weighting")
        self.chk_weights.setChecked(False)
        self.chk_weights.toggled.connect(self._update_weight_ui)
        wt_layout.addWidget(self.chk_weights, 0, 0, 1, 4)

        wt_layout.addWidget(QLabel("Metric:"), 1, 0)
        self.cmb_weight_metric = QComboBox()
        self.cmb_weight_metric.addItems([
            "wFWHM  (FWHM × star count — recommended)",
            "FWHM   (raw FWHM only)",
            "Noise  (background RMS)",
            "# Stars",
        ])
        self.cmb_weight_metric.setEnabled(False)
        wt_layout.addWidget(self.cmb_weight_metric, 1, 1, 1, 3)

        wt_layout.addWidget(QLabel("Power:"), 2, 0)
        self.spn_weight_power = QDoubleSpinBox()
        self.spn_weight_power.setRange(0.1, 10.0)
        self.spn_weight_power.setSingleStep(0.5)
        self.spn_weight_power.setValue(2.0)
        self.spn_weight_power.setDecimals(1)
        self.spn_weight_power.setToolTip(
            "Exponent applied to the quality score before weighting.\n"
            "Higher values give proportionally more weight to the best frames.\n"
            "1.0 = linear,  2.0 = quadratic (default),  0.5 = square-root (gentle)"
        )
        self.spn_weight_power.setEnabled(False)
        wt_layout.addWidget(self.spn_weight_power, 2, 1)

        power_hint = QLabel("(1=linear · 2=quadratic · 0.5=gentle)")
        power_hint.setObjectName("subheader")
        wt_layout.addWidget(power_hint, 2, 2, 1, 2)

        root_layout.addWidget(wt_grp)

        # ── Output ────────────────────────────────────────────────────────
        out_grp = QGroupBox("Output")
        out_layout = QGridLayout(out_grp)
        out_layout.setColumnStretch(1, 1)

        out_layout.addWidget(QLabel("Filename:"), 0, 0)
        self.txt_output = QLineEdit("stacked_result")
        self.txt_output.setPlaceholderText("Output filename (no extension)")
        out_layout.addWidget(self.txt_output, 0, 1)
        out_layout.addWidget(QLabel(".fit"), 0, 2)
        root_layout.addWidget(out_grp)

        # ── Log ───────────────────────────────────────────────────────────
        log_grp = QGroupBox("Log")
        log_layout = QVBoxLayout(log_grp)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(130)
        log_layout.addWidget(self.log_view)
        root_layout.addWidget(log_grp)

        # ── Progress bar ──────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        root_layout.addWidget(self.progress_bar)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("cancelButton")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.btn_cancel)

        btn_layout.addStretch()

        self.btn_run = QPushButton("▶  Run Stack")
        self.btn_run.setObjectName("runButton")
        self.btn_run.setMinimumWidth(130)
        self.btn_run.clicked.connect(self._on_run)
        btn_layout.addWidget(self.btn_run)

        root_layout.addLayout(btn_layout)

        self._log("Winsorized Sigma-Clipped Stack engine ready.", "ok")

    # ── Import helpers ─────────────────────────────────────────────────────
    def _browse_import_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select image folder", os.path.expanduser("~"),
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.txt_imp_folder.setText(folder)
            # Auto-fill basename from folder name if blank
            if not self.txt_imp_basename.text().strip():
                self.txt_imp_basename.setText(os.path.basename(folder).lower() or "lights")

    def _update_import_ui(self):
        fmt = self.cmb_imp_fmt.currentText()
        is_convert = "images" in fmt          # only convert modes support debayer
        is_ser     = fmt.startswith("SER")
        self.chk_imp_debayer.setEnabled(is_convert)
        # Hint the basename placeholder
        if is_ser:
            self.txt_imp_basename.setPlaceholderText("SER filename (with or without .ser)")
        elif fmt.startswith("Existing"):
            self.txt_imp_basename.setPlaceholderText("Sequence basename (without .seq)")
        else:
            self.txt_imp_basename.setPlaceholderText("Sequence base name (e.g. lights)")

    def _on_import(self):
        folder   = self.txt_imp_folder.text().strip()
        basename = self.txt_imp_basename.text().strip()
        fmt      = self.cmb_imp_fmt.currentText()
        debayer  = self.chk_imp_debayer.isChecked()

        if not folder:
            self._log("Please select an image folder first.", "warn")
            return
        if not os.path.isdir(folder):
            self._log(f"Folder not found: {folder}", "error")
            return
        if not basename:
            self._log("Please enter a sequence basename.", "warn")
            return

        self.btn_import.setEnabled(False)
        self._log(f"Starting import from {folder}…", "info")

        self._imp_thread = QThread()
        self._imp_worker = ImportWorker(self.siril, folder, basename, fmt, debayer)
        self._imp_worker.moveToThread(self._imp_thread)

        self._imp_thread.started.connect(self._imp_worker.run)
        self._imp_worker.log_message.connect(self._log)
        self._imp_worker.finished.connect(self._on_import_finished)
        self._imp_worker.finished.connect(self._imp_thread.quit)

        self._imp_thread.start()

    def _on_import_finished(self, success: bool, seq_basename: str):
        self.btn_import.setEnabled(True)
        if success:
            self._log(f"Import complete — sequence '{seq_basename}' is now active.", "ok")
            self._detect_sequence()   # refresh the Sequence status group
        else:
            self._log("Import failed. Check the log for details.", "error")

    # ── Sequence detection ─────────────────────────────────────────────────
    def _detect_sequence(self):
        try:
            seq_name = self.siril.get_seq_name()
            if seq_name:
                self.lbl_seq.setText(os.path.basename(seq_name))
                try:
                    seq_info = self.siril.get_sequence_info()
                    total    = getattr(seq_info, "number", "?")
                    included = getattr(seq_info, "selnum", "?")
                    self.lbl_frames.setText(f"{included} selected / {total} total")
                except Exception:
                    self.lbl_frames.setText("(info unavailable)")
                self._log(f"Sequence detected: {os.path.basename(seq_name)}", "ok")
            else:
                self.lbl_seq.setText("No sequence loaded")
                self.lbl_frames.setText("—")
        except Exception:
            self.lbl_seq.setText("No sequence loaded")
            self.lbl_frames.setText("—")

    # ── UI reactions ───────────────────────────────────────────────────────
    def _update_weight_ui(self, enabled):
        self.cmb_weight_metric.setEnabled(enabled)
        self.spn_weight_power.setEnabled(enabled)

    # ── Logging ────────────────────────────────────────────────────────────
    def _log(self, msg: str, level: str = "info"):
        color_map = {
            "info":  "#b0b0b0",
            "ok":    "#6ec96e",
            "warn":  "#e8c46a",
            "error": "#e06c75",
        }
        color = color_map.get(level, "#b0b0b0")
        if msg == "" or msg.startswith("═"):
            self.log_view.append(f'<span style="color:{color}">{msg}</span>')
        else:
            prefix = {"ok": "✓ ", "warn": "⚠ ", "error": "✗ ", "info": "  "}.get(level, "  ")
            self.log_view.append(f'<span style="color:{color}">{prefix}{msg}</span>')
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        # Mirror to Siril log
        try:
            self.siril.log(msg)
        except Exception:
            pass

    # ── Run / Cancel ───────────────────────────────────────────────────────
    def _on_run(self):
        params = {
            "method":        self.cmb_method.currentText(),
            "sigma_low":     self.spn_sigma_low.value(),
            "sigma_high":    self.spn_sigma_high.value(),
            "iterations":    self.spn_iterations.value(),
            "norm":          self.cmb_norm.currentText(),
            "prenormalize":  self.chk_prenorm.isChecked(),
            "output_norm":   self.chk_output_norm.isChecked(),
            "use_weights":   self.chk_weights.isChecked(),
            "weight_metric": self.cmb_weight_metric.currentText().split()[0],  # wFWHM | FWHM | Noise | #
            "weight_power":  self.spn_weight_power.value(),
            "save_rejection":self.chk_save_rej.isChecked(),
            "output_name":   self.txt_output.text().strip() or "stacked_result",
        }

        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.thread = QThread()
        self.worker = StackWorker(self.siril, params)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self._log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)

        self.thread.start()

    def _on_cancel(self):
        if self.worker:
            self.worker.cancel()
        self.btn_cancel.setEnabled(False)
        self._log("Cancellation requested…", "warn")

    def _on_finished(self, success: bool, message: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if success:
            self._log(f"Done: {message}", "ok")
            self.progress_bar.setFormat("Complete ✓")
        else:
            self._log(f"Failed: {message}", "error")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Failed ✗")


# ─── Entry point ──────────────────────────────────────────────────────────
def main():
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError as e:
        print(f"[WinsorizedSigmaStack] Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        siril.cmd("requires", "1.4.0")
    except Exception as e:
        siril.log(f"[WinsorizedSigmaStack] Siril 1.4.0+ required: {e}")
        sys.exit(1)

    app = QApplication.instance() or QApplication(sys.argv)

    window = WinsorizedStackUI(siril)
    window.show()

    exit_code = app.exec()
    siril.disconnect()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
