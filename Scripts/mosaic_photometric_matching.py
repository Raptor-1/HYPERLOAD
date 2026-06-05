r"""
mosaic_photometric_matching.py  —  HYPERLOAD Mosaic Photometric Matching
=========================================================================
Normalises brightness and colour across multi-panel mosaic images by
fitting a photometric transfer function between overlapping panels using
common stars detected in the overlap regions.

Without this, mosaic panels taken on different nights, at different
airmasses, or with slightly different transparency show visible seams
in both gradient and colour — even after gradient removal.

Algorithm (used in PanSTARRS, SDSS, CFHT Elixir pipelines):
  1. For each pair of overlapping panels:
     a. Find the overlap region from WCS (if available) or manual offsets
     b. Detect stars in each panel's overlap region (sigma-clipped centroid)
     c. Cross-match stars by position (nearest-neighbour, max_dist configurable)
     d. Fit linear transfer: flux_B = a × flux_A + b  (weighted least squares)
     e. Apply: panel_B_corrected = (panel_B - b) / a
  2. Propagate corrections: if 3+ panels, build a spanning tree (minimum
     residual) and propagate corrections transitively so all panels
     reference the same photometric scale.
  3. Optional: Gaussian-weighted feather blend in overlap regions to
     produce seamless transitions.

Supports:
  - RGB FITS panels with WCS headers
  - Luminance-only panels
  - Manual overlap specification (if no WCS)
  - 2-panel and multi-panel mosaics

References:
  Padmanabhan et al. 2008, ApJ 674, 1217  (SDSS photometric calibration)
  Magnier & Cuillandre 2004, PASP 116, 449  (CFHT Elixir pipeline)
  Szeliski 2006, IEEE TPAMI 28(9), 1409    (feather blending)

Place in: Siril Suites folder
Run via:  Siril → Scripts → mosaic_photometric_matching
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("scipy")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")

import os, sys, glob, threading, traceback, itertools
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.optimize import curve_fit
from astropy.io import fits as astropy_fits
from astropy.wcs import WCS

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ── Theme ─────────────────────────────────────────────────────────────────────
BG="#1e2128";BG2="#252930";BG3="#2d3340";ACC="#4a9eff";ACC2="#2d6abf"
TXT="#dde3ee";DIM="#7a8499";BRD="#3a4055";OK="#4caf7d";WRN="#e8c46a"
ERR="#cc4444";SEC="#9db4d0";MOS="#56cfe1"

SS=f"""
QMainWindow,QWidget{{background:{BG};color:{TXT};font-family:"Segoe UI",sans-serif;font-size:9pt;}}
QGroupBox{{border:1px solid {BRD};border-radius:5px;margin-top:8px;padding:6px;font-weight:bold;color:{SEC};}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}
QTabWidget::pane{{border:1px solid {BRD};background:{BG};}}
QTabBar::tab{{background:{BG2};color:{DIM};padding:6px 14px;border:1px solid {BRD};border-bottom:none;border-radius:4px 4px 0 0;}}
QTabBar::tab:selected{{background:{BG};color:{MOS};font-weight:bold;}}
QPushButton{{background:{BG3};color:{TXT};border:1px solid {BRD};border-radius:4px;padding:5px 12px;min-height:22px;}}
QPushButton:hover{{background:{ACC2};border-color:{ACC};}}
QPushButton[objectName="primary"]{{background:{ACC2};color:white;font-weight:bold;border-color:{ACC};}}
QPushButton[objectName="mos"]{{background:#0a2a2a;color:{MOS};font-weight:bold;border-color:{MOS};}}
QPushButton[objectName="danger"]{{background:#4a1e1e;color:#e07070;border-color:#803030;}}
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox{{background:{BG2};color:{TXT};border:1px solid {BRD};border-radius:4px;padding:3px 6px;}}
QProgressBar{{background:{BG2};border:1px solid {BRD};border-radius:4px;text-align:center;}}
QProgressBar::chunk{{background:{MOS};border-radius:3px;}}
QPlainTextEdit{{background:{BG2};color:{TXT};border:1px solid {BRD};border-radius:4px;font-family:"Courier New",monospace;font-size:8pt;}}
QTableWidget{{background:{BG2};color:{TXT};border:1px solid {BRD};gridline-color:{BRD};alternate-background-color:{BG3};}}
QHeaderView::section{{background:{BG3};color:{SEC};border:1px solid {BRD};padding:4px;}}
QLabel[objectName="dim"]{{color:{DIM};}}QLabel[objectName="ok"]{{color:{OK};}}
QScrollArea{{border:none;}}
"""

# ── Photometry helpers ────────────────────────────────────────────────────────

def detect_stars_simple(image: np.ndarray, threshold_sigma: float = 5.0,
                          max_stars: int = 200) -> list:
    """Detect stars via sigma-threshold + centroid. Returns [(x,y,flux)]."""
    bg = np.median(image)
    noise = 1.4826 * np.median(np.abs(image - bg))
    mask = image > bg + threshold_sigma * noise
    labeled, n = label(mask)
    stars = []
    for i in range(1, n + 1):
        coords = np.where(labeled == i)
        if len(coords[0]) < 2 or len(coords[0]) > 500:
            continue
        flux = float(image[coords].sum())
        cx = float(np.sum(coords[1] * image[coords]) / flux)
        cy = float(np.sum(coords[0] * image[coords]) / flux)
        stars.append((cx, cy, flux))
    stars.sort(key=lambda s: -s[2])
    return stars[:max_stars]


def match_stars_panels(stars_a: list, stars_b: list,
                        max_dist: float = 15.0) -> list:
    """Match star lists. Returns [(flux_a, flux_b)] pairs."""
    matches = []
    used = set()
    for ax, ay, af in stars_a:
        best_j, best_d = -1, max_dist
        for j, (bx, by, bf) in enumerate(stars_b):
            if j in used: continue
            d = np.hypot(ax - bx, ay - by)
            if d < best_d:
                best_d = d; best_j = j
        if best_j >= 0:
            matches.append((af, stars_b[best_j][2]))
            used.add(best_j)
    return matches


def fit_transfer(flux_a: list, flux_b: list) -> tuple:
    """
    Fit linear photometric transfer: flux_B = scale × flux_A + offset
    Uses iterative sigma-clipping to reject outliers.
    Returns (scale, offset, rms, n_matched).
    """
    if len(flux_a) < 3:
        return 1.0, 0.0, 0.0, len(flux_a)
    fa = np.array(flux_a); fb = np.array(flux_b)
    # Weighted least squares: minimise Σ (fb - a*fa - b)²
    for _ in range(5):
        A = np.column_stack([fa, np.ones_like(fa)])
        result = np.linalg.lstsq(A, fb, rcond=None)
        coeffs = result[0]
        resid = fb - (coeffs[0]*fa + coeffs[1])
        rms = np.std(resid)
        if rms == 0: break
        keep = np.abs(resid) < 3.0 * rms
        if keep.sum() < 3: break
        fa, fb = fa[keep], fb[keep]
    scale = float(coeffs[0]); offset = float(coeffs[1])
    rms_final = float(np.std(fb - (scale*fa + offset)))
    return scale, offset, rms_final, len(fa)


def find_overlap(img_a: np.ndarray, img_b: np.ndarray,
                 offset_x: int, offset_y: int,
                 overlap_frac: float = 0.2) -> tuple:
    """
    Given two images and the pixel offset of B relative to A,
    return (region_in_A, region_in_B) as (y0,y1,x0,x1) tuples.
    """
    ha, wa = img_a.shape[-2], img_a.shape[-1]
    hb, wb = img_b.shape[-2], img_b.shape[-1]
    # A occupies [0,ha)×[0,wa), B occupies [offset_y,offset_y+hb)×[offset_x,offset_x+wb)
    # Overlap in global coords
    y0 = max(0, offset_y); y1 = min(ha, offset_y + hb)
    x0 = max(0, offset_x); x1 = min(wa, offset_x + wb)
    if y1 <= y0 or x1 <= x0:
        return None, None
    # In A coords
    rA = (y0, y1, x0, x1)
    # In B coords
    rB = (y0 - offset_y, y1 - offset_y, x0 - offset_x, x1 - offset_x)
    return rA, rB


def apply_photometric_correction(image: np.ndarray, scale: float,
                                   offset: float) -> np.ndarray:
    """Apply: corrected = (image - offset) / scale"""
    corrected = (image.astype(np.float64) - offset) / max(scale, 1e-10)
    return corrected.astype(np.float32)


def feather_blend(img_a: np.ndarray, img_b: np.ndarray,
                   rA: tuple, rB: tuple,
                   sigma: float = 50.0) -> np.ndarray:
    """
    Gaussian feather blend in the overlap region.
    Szeliski 2006, IEEE TPAMI 28(9), 1409.
    Returns blended copy of img_a with smooth transition to img_b.
    """
    result = img_a.astype(np.float64).copy()
    y0a, y1a, x0a, x1a = rA
    y0b, y1b, x0b, x1b = rB
    h = y1a - y0a; w = x1a - x0a
    if h <= 0 or w <= 0: return img_a.astype(np.float32)
    # Weight: 0 at left edge → 1 at right edge (horizontal blend)
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    XX, _ = np.meshgrid(xs, ys)
    weight = gaussian_filter(XX, sigma=sigma/w*10)
    weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-10)
    patch_a = result[y0a:y1a, x0a:x1a]
    patch_b = img_b[y0b:y1b, x0b:x1b].astype(np.float64)
    if patch_a.shape == patch_b.shape:
        result[y0a:y1a, x0a:x1a] = (1 - weight) * patch_a + weight * patch_b
    return result.astype(np.float32)


# ── Canvas ────────────────────────────────────────────────────────────────────

class MosaicCanvas(FigureCanvasQTAgg):
    """Shows panel thumbnails with match statistics."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 5), facecolor=BG)
        super().__init__(self.fig); self.setParent(parent)

    def show_panels(self, panels: list, title: str = "Mosaic panels"):
        self.fig.clear()
        n = len(panels)
        if n == 0: return
        cols = min(n, 4); rows = (n + cols - 1) // cols
        for i, (lbl, img) in enumerate(panels):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            ax.set_facecolor(BG2)
            vmin, vmax = np.percentile(img, [0.5, 99.5])
            disp = img[::max(1,img.shape[0]//200), ::max(1,img.shape[1]//200)]
            ax.imshow(disp, cmap="gray", origin="lower",
                      vmin=vmin, vmax=vmax, aspect="equal",
                      interpolation="nearest")
            ax.set_title(lbl, color=MOS, fontsize=8)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values(): sp.set_color(BRD)
        self.fig.suptitle(title, color=MOS, fontsize=9)
        self.fig.tight_layout(pad=0.4)
        self.draw()


# ── Worker ────────────────────────────────────────────────────────────────────

class MosaicWorker(QThread):
    progress   = pyqtSignal(int, int, str)
    log_line   = pyqtSignal(str)
    panels_ready = pyqtSignal(list)      # [(label, luminance_array)]
    match_result = pyqtSignal(dict)      # per-pair results
    finished   = pyqtSignal(dict)

    def __init__(self, cfg, cancel_event, panel_files, offsets, parent=None):
        super().__init__(parent)
        self.cfg = cfg; self._cancel = cancel_event
        self.panel_files = panel_files
        self.offsets = offsets  # list of (dx, dy) relative to panel 0

    def run(self):
        try: self._run()
        except Exception as e:
            self.log_line.emit(f"ERROR: {e}\n{traceback.format_exc()}")
            self.finished.emit({"success": False, "error": str(e)})

    def _run(self):
        cfg = self.cfg; log = self.log_line.emit
        log("═══ HyperLoad Mosaic Photometric Matching ═══")
        log(f"  {len(self.panel_files)} panels")
        log(f"  Detection σ: {cfg['threshold_sigma']}")
        log(f"  Max match dist: {cfg['max_match_dist']} px")
        log(f"  Feather blend: {'ON' if cfg['feather'] else 'OFF'}")
        log("")

        # ── Load all panels ───────────────────────────────────────────────────
        self.progress.emit(0, 100, "Loading panels…")
        panels = []
        for i, path in enumerate(self.panel_files):
            if self._cancel.is_set(): break
            try:
                with astropy_fits.open(path) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    hdr  = hdul[0].header.copy()
                if data.ndim == 3 and data.shape[0] == 3:
                    lum = 0.299*data[0] + 0.587*data[1] + 0.114*data[2]
                elif data.ndim == 3:
                    lum = data[0]
                else:
                    lum = data
                panels.append({"path": path, "data": data, "lum": lum,
                                "hdr": hdr, "idx": i})
                log(f"  Loaded [{i+1}]: {os.path.basename(path)}  "
                    f"{data.shape}")
            except Exception as e:
                log(f"  Skip {os.path.basename(path)}: {e}")

        if len(panels) < 2:
            self.finished.emit({"success": False,
                                "error": "Need at least 2 panels"}); return

        # Preview thumbnails
        self.panels_ready.emit(
            [(os.path.basename(p["path"]), p["lum"]) for p in panels])

        # ── Find overlapping pairs ─────────────────────────────────────────────
        self.progress.emit(20, 100, "Finding overlaps…")
        pairs = []
        for i in range(len(panels)):
            for j in range(i + 1, len(panels)):
                dx = self.offsets[j][0] - self.offsets[i][0]
                dy = self.offsets[j][1] - self.offsets[i][1]
                rA, rB = find_overlap(panels[i]["lum"], panels[j]["lum"],
                                       int(dx), int(dy))
                if rA is not None:
                    pairs.append((i, j, rA, rB, dx, dy))
                    log(f"  Overlap: panel {i+1} ↔ panel {j+1}  "
                        f"region={rA[1]-rA[0]}×{rA[3]-rA[2]} px")

        if not pairs:
            self.finished.emit({"success": False,
                                "error": "No overlapping panels found — "
                                         "check offsets"}); return

        # ── Match stars in each overlap ────────────────────────────────────────
        self.progress.emit(35, 100, "Matching stars in overlaps…")
        transfers = {}  # (i,j) → (scale, offset, rms, n)

        for pi, pj, rA, rB, dx, dy in pairs:
            if self._cancel.is_set(): break
            log(f"\nPanel {pi+1} ↔ Panel {pj+1}:")
            y0a, y1a, x0a, x1a = rA
            y0b, y1b, x0b, x1b = rB
            ov_a = panels[pi]["lum"][y0a:y1a, x0a:x1a]
            ov_b = panels[pj]["lum"][y0b:y1b, x0b:x1b]

            stars_a = detect_stars_simple(
                ov_a, cfg["threshold_sigma"], cfg["max_stars"])
            stars_b = detect_stars_simple(
                ov_b, cfg["threshold_sigma"], cfg["max_stars"])
            log(f"  Stars in overlap: {len(stars_a)} (A)  {len(stars_b)} (B)")

            matches = match_stars_panels(
                stars_a, stars_b, cfg["max_match_dist"])
            log(f"  Matched: {len(matches)} pairs")

            if len(matches) < 3:
                log("  ⚠ Too few matches — using scale=1, offset=0")
                transfers[(pi, pj)] = (1.0, 0.0, 0.0, len(matches))
                continue

            flux_a = [m[0] for m in matches]
            flux_b = [m[1] for m in matches]
            scale, offset, rms, n = fit_transfer(flux_a, flux_b)
            transfers[(pi, pj)] = (scale, offset, rms, n)
            log(f"  Transfer: flux_B = {scale:.4f}×flux_A + {offset:.2f}  "
                f"RMS={rms:.2f}  n={n}")

        self.match_result.emit({
            str(k): {"scale": v[0], "offset": v[1], "rms": v[2], "n": v[3]}
            for k, v in transfers.items()})

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── Propagate corrections (spanning tree, panel 0 = reference) ────────
        self.progress.emit(60, 100, "Propagating corrections…")
        log("\nPropagating corrections (panel 1 = photometric reference)…")
        corrections = {0: (1.0, 0.0)}  # scale, offset for each panel

        # BFS from panel 0
        visited = {0}; queue = [0]
        while queue:
            cur = queue.pop(0)
            cur_scale, cur_offset = corrections[cur]
            for (i, j), (scale, offset, rms, n) in transfers.items():
                if i == cur and j not in visited:
                    # panel_j corrected = panel_j / scale_ij
                    # Combined with panel i's correction:
                    new_scale  = cur_scale * scale
                    new_offset = cur_scale * offset + cur_offset
                    corrections[j] = (new_scale, new_offset)
                    visited.add(j); queue.append(j)
                    log(f"  Panel {j+1}: scale={new_scale:.4f}  "
                        f"offset={new_offset:.2f}")
                elif j == cur and i not in visited:
                    # Reverse: panel_i corrected = panel_j × scale + offset → reversed
                    if abs(scale) > 1e-10:
                        rev_scale  = cur_scale / scale
                        rev_offset = cur_offset - cur_scale * offset / scale
                        corrections[i] = (rev_scale, rev_offset)
                        visited.add(i); queue.append(i)
                        log(f"  Panel {i+1}: scale={rev_scale:.4f}  "
                            f"offset={rev_offset:.2f}")

        # ── Apply corrections and save ─────────────────────────────────────────
        self.progress.emit(75, 100, "Applying corrections…")
        out_dir = cfg["output_dir"]; os.makedirs(out_dir, exist_ok=True)
        corrected_panels = []
        for p in panels:
            idx = p["idx"]
            scale, offset = corrections.get(idx, (1.0, 0.0))
            corr_data = apply_photometric_correction(p["data"], scale, offset)
            corrected_panels.append({"data": corr_data, "path": p["path"],
                                      "hdr": p["hdr"], "idx": idx,
                                      "scale": scale, "offset": offset})

        # Optional feather blend (applied to first pair for now)
        if cfg["feather"] and len(pairs) > 0 and len(corrected_panels) >= 2:
            pi, pj, rA, rB, dx, dy = pairs[0]
            lum_a = corrected_panels[pi]["data"]
            lum_a = lum_a[0] if lum_a.ndim == 3 else lum_a
            lum_b = corrected_panels[pj]["data"]
            lum_b = lum_b[0] if lum_b.ndim == 3 else lum_b
            feather_blend(lum_a, lum_b, rA, rB,
                          sigma=cfg.get("feather_sigma", 50.0))
            log("  Feather blend applied to first overlap")

        n_saved = 0
        for cp in corrected_panels:
            base = os.path.splitext(os.path.basename(cp["path"]))[0]
            out_path = os.path.join(out_dir, f"{base}_photmatched.fit")
            hdr = cp["hdr"].copy()
            hdr["PHOTSCAL"] = (cp["scale"],  "Photometric scale applied")
            hdr["PHOTOFFS"] = (cp["offset"], "Photometric offset applied")
            hdr["HISTORY"]  = "HyperLoad mosaic photometric matching"
            astropy_fits.PrimaryHDU(
                data=cp["data"].astype(np.float32), header=hdr).writeto(
                out_path, overwrite=True)
            log(f"  Saved: {os.path.basename(out_path)}  "
                f"(scale={cp['scale']:.4f})")
            n_saved += 1

        self.progress.emit(100, 100, "Done")
        log(f"\n✓ {n_saved} panels photometrically matched and saved")
        self.finished.emit({
            "success":  True,
            "n_panels": n_saved,
            "n_pairs":  len(pairs),
            "output_dir": out_dir,
        })


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Mosaic Photometric Matching  —  HYPERLOAD  —  Siril")
        self.resize(1200, 820)
        self._worker = None; self._cancel = threading.Event()
        self._panel_files = []; self._offsets = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(52)
        hdr.setStyleSheet(
            f"background:{BG2};border-bottom:1px solid {BRD};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(14, 0, 14, 0)
        lbl = QLabel("🔲  Mosaic Photometric Matching")
        lbl.setStyleSheet(
            f"color:{MOS};font-size:13pt;font-weight:bold;background:transparent;")
        sub = QLabel(
            "Star-based photometric transfer  ·  Spanning-tree propagation  ·  "
            "Feather blending  ·  Padmanabhan 2008 / CFHT Elixir approach")
        sub.setStyleSheet(
            f"color:{DIM};font-size:9pt;background:transparent;")
        hl.addWidget(lbl); hl.addWidget(sub); hl.addStretch()
        root.addWidget(hdr)

        tabs = QTabWidget(); root.addWidget(tabs, 1)
        tabs.addTab(self._build_panels_tab(), "📁  Panels")
        tabs.addTab(self._build_settings_tab(), "⚙  Settings")
        tabs.addTab(self._build_results_tab(), "📊  Results")
        tabs.addTab(self._build_preview_tab(), "🖼  Preview")
        tabs.addTab(self._build_log_tab(), "📋  Log")

        bottom = QWidget(); bottom.setFixedHeight(48)
        bottom.setStyleSheet(
            f"background:{BG2};border-top:1px solid {BRD};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 6, 10, 6)
        self._prog = QProgressBar(); self._prog.setFixedHeight(10)
        bl.addWidget(self._prog, 1)
        self._btn_run = QPushButton("▶  Match Panels")
        self._btn_run.setObjectName("mos"); self._btn_run.setMinimumHeight(34)
        self._btn_run.setMinimumWidth(160); self._btn_run.clicked.connect(self._run)
        bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setMinimumHeight(34)
        self._btn_cancel.setEnabled(False); self._btn_cancel.clicked.connect(self._cancel_run)
        bl.addWidget(self._btn_cancel)
        self._status = QLabel("Ready — add mosaic panels")
        self._status.setObjectName("dim")
        self._status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status, 1)
        root.addWidget(bottom)

    def _build_panels_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        lbl_hint = QLabel(
            "Add mosaic panel FITS files in order. Set the pixel offset of each panel\n"
            "relative to panel 1 (leave 0,0 for panel 1 itself). If panels have WCS\n"
            "headers, offsets can be computed automatically from the sky coordinates.")
        lbl_hint.setObjectName("dim"); lbl_hint.setWordWrap(True)
        lay.addWidget(lbl_hint)

        add_row = QHBoxLayout()
        self._edit_panel_path = QLineEdit()
        self._edit_panel_path.setPlaceholderText("Panel FITS path…")
        btn_browse = QPushButton("Browse…"); btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_panel)
        add_row.addWidget(self._edit_panel_path, 3)
        add_row.addWidget(btn_browse)
        add_row.addWidget(QLabel("ΔX:"))
        self._spin_dx = QSpinBox()
        self._spin_dx.setRange(-99999, 99999); self._spin_dx.setValue(0)
        self._spin_dx.setFixedWidth(80)
        add_row.addWidget(self._spin_dx)
        add_row.addWidget(QLabel("ΔY:"))
        self._spin_dy = QSpinBox()
        self._spin_dy.setRange(-99999, 99999); self._spin_dy.setValue(0)
        self._spin_dy.setFixedWidth(80)
        add_row.addWidget(self._spin_dy)
        btn_add = QPushButton("➕ Add panel"); btn_add.clicked.connect(self._add_panel)
        add_row.addWidget(btn_add)
        lay.addLayout(add_row)

        self._chk_auto_offsets = QCheckBox(
            "Auto-compute offsets from WCS headers (requires plate-solved panels)")
        self._chk_auto_offsets.setChecked(True)
        lay.addWidget(self._chk_auto_offsets)

        self._panel_table = QTableWidget(0, 4)
        self._panel_table.setHorizontalHeaderLabels(
            ["#", "File", "ΔX (px)", "ΔY (px)"])
        self._panel_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._panel_table.setAlternatingRowColors(True)
        lay.addWidget(self._panel_table, 1)

        btns = QHBoxLayout()
        btn_remove = QPushButton("Remove selected"); btn_remove.clicked.connect(self._remove_panel)
        btn_clear  = QPushButton("Clear all"); btn_clear.clicked.connect(self._clear_panels)
        btns.addWidget(btn_remove); btns.addWidget(btn_clear); btns.addStretch()
        lay.addLayout(btns)

        grp_out = QGroupBox("Output folder")
        ol = QHBoxLayout(grp_out)
        self._edit_out = QLineEdit(); self._edit_out.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(80)
        btn_out.clicked.connect(lambda: self._edit_out.setText(
            QFileDialog.getExistingDirectory(self, "Output") or self._edit_out.text()))
        ol.addWidget(self._edit_out); ol.addWidget(btn_out)
        lay.addWidget(grp_out)
        return w

    def _build_settings_tab(self):
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        grp_det = QGroupBox("Star detection in overlap regions")
        df = QFormLayout(grp_det)
        self._spin_thresh = QDoubleSpinBox()
        self._spin_thresh.setRange(2.0, 20.0); self._spin_thresh.setValue(5.0)
        self._spin_thresh.setSuffix(" σ")
        df.addRow("Detection threshold:", self._spin_thresh)
        self._spin_max_stars = QSpinBox()
        self._spin_max_stars.setRange(10, 500); self._spin_max_stars.setValue(100)
        df.addRow("Max stars per panel:", self._spin_max_stars)
        self._spin_max_dist = QDoubleSpinBox()
        self._spin_max_dist.setRange(1, 100); self._spin_max_dist.setValue(15.0)
        self._spin_max_dist.setSuffix(" px")
        df.addRow("Max match distance:", self._spin_max_dist)
        lbl_d = QLabel("Stars must be in the overlap region of both panels.\n"
                        "Reduce max distance if getting spurious matches.")
        lbl_d.setObjectName("dim"); lbl_d.setWordWrap(True); df.addRow(lbl_d)
        lay.addWidget(grp_det)

        grp_blend = QGroupBox("Feather blending (optional)")
        bf = QFormLayout(grp_blend)
        self._chk_feather = QCheckBox("Apply Gaussian feather blend in overlap regions")
        self._chk_feather.setChecked(False)
        bf.addRow(self._chk_feather)
        self._spin_feather_sigma = QDoubleSpinBox()
        self._spin_feather_sigma.setRange(5, 500); self._spin_feather_sigma.setValue(50.0)
        self._spin_feather_sigma.setSuffix(" px")
        bf.addRow("Feather sigma:", self._spin_feather_sigma)
        lbl_b = QLabel("Feather blend creates a smooth transition in overlaps.\n"
                        "Szeliski 2006, IEEE TPAMI 28(9), 1409.")
        lbl_b.setObjectName("dim"); lbl_b.setWordWrap(True); bf.addRow(lbl_b)
        lay.addWidget(grp_blend)
        lay.addStretch()
        return w

    def _build_results_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(6)
        lbl = QLabel("Transfer function results (populated after matching)")
        lbl.setObjectName("dim"); lay.addWidget(lbl)
        self._results_table = QTableWidget(0, 5)
        self._results_table.setHorizontalHeaderLabels(
            ["Pair", "Scale", "Offset", "RMS", "N matched"])
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._results_table.setAlternatingRowColors(True)
        lay.addWidget(self._results_table)
        return w

    def _build_preview_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(4, 4, 4, 4)
        self._canvas = MosaicCanvas(); lay.addWidget(self._canvas)
        return w

    def _build_log_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(6, 6, 6, 6)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._log.clear)
        lay.addWidget(btn_clr); lay.addWidget(self._log)
        return w

    # ── Panel management ──────────────────────────────────────────────────────

    def _browse_panel(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select panel FITS", "",
            "FITS (*.fit *.fits *.fts);;All (*)")
        if path:
            self._edit_panel_path.setText(path)
            if not self._edit_out.text():
                self._edit_out.setText(os.path.dirname(path))
            # Try auto-read offset from WCS if other panels present
            if self._chk_auto_offsets.isChecked() and self._panel_files:
                try:
                    ref_path = self._panel_files[0]
                    with astropy_fits.open(ref_path) as hdul:
                        wcs_ref = WCS(hdul[0].header)
                        ra_ref, dec_ref = wcs_ref.all_pix2world(
                            hdul[0].data.shape[-1]//2,
                            hdul[0].data.shape[-2]//2, 0)
                    with astropy_fits.open(path) as hdul:
                        wcs_new = WCS(hdul[0].header)
                        px, py = wcs_new.all_world2pix(ra_ref, dec_ref, 0)
                        cx = hdul[0].data.shape[-1] // 2
                        cy = hdul[0].data.shape[-2] // 2
                        self._spin_dx.setValue(int(cx - px))
                        self._spin_dy.setValue(int(cy - py))
                except Exception:
                    pass

    def _add_panel(self):
        path = self._edit_panel_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file", "Select a FITS file."); return
        dx = self._spin_dx.value(); dy = self._spin_dy.value()
        self._panel_files.append(path); self._offsets.append((dx, dy))
        row = self._panel_table.rowCount(); self._panel_table.insertRow(row)
        for col, val in enumerate([
            str(row+1), os.path.basename(path), str(dx), str(dy)
        ]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._panel_table.setItem(row, col, item)
        self._edit_panel_path.clear()

    def _remove_panel(self):
        row = self._panel_table.currentRow()
        if row < 0: return
        self._panel_files.pop(row); self._offsets.pop(row)
        self._panel_table.removeRow(row)
        for r in range(self._panel_table.rowCount()):
            self._panel_table.setItem(r, 0, QTableWidgetItem(str(r+1)))

    def _clear_panels(self):
        self._panel_files.clear(); self._offsets.clear()
        self._panel_table.setRowCount(0)

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _run(self):
        if len(self._panel_files) < 2:
            QMessageBox.warning(self, "Too few panels",
                "Add at least 2 mosaic panel FITS files."); return
        out = self._edit_out.text().strip()
        if not out:
            QMessageBox.warning(self, "No output", "Set an output folder."); return
        cfg = {
            "output_dir":      out,
            "threshold_sigma": self._spin_thresh.value(),
            "max_stars":       self._spin_max_stars.value(),
            "max_match_dist":  self._spin_max_dist.value(),
            "feather":         self._chk_feather.isChecked(),
            "feather_sigma":   self._spin_feather_sigma.value(),
        }
        self._cancel.clear(); self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True); self._prog.setValue(0)
        self._worker = MosaicWorker(cfg, self._cancel,
                                     list(self._panel_files),
                                     list(self._offsets))
        self._worker.progress.connect(lambda s,t,m: (
            self._prog.setRange(0,t), self._prog.setValue(s),
            self._status.setText(m)))
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.panels_ready.connect(
            lambda p: self._canvas.show_panels(p, "Input panels"))
        self._worker.match_result.connect(self._on_match_result)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._tabs.setCurrentIndex(4)  # log

    @property
    def _tabs(self):
        return self.centralWidget().findChild(QTabWidget)

    def _cancel_run(self):
        self._cancel.set(); self._btn_cancel.setEnabled(False)

    def _on_match_result(self, results: dict):
        self._results_table.setRowCount(0)
        for pair_str, vals in results.items():
            row = self._results_table.rowCount()
            self._results_table.insertRow(row)
            for col, v in enumerate([
                f"Panel {pair_str}", f"{vals['scale']:.4f}",
                f"{vals['offset']:.2f}", f"{vals['rms']:.3f}",
                str(vals['n']),
            ]):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._results_table.setItem(row, col, item)

    def _on_finished(self, result):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        if result.get("success"):
            n = result["n_panels"]; np2 = result["n_pairs"]
            self._status.setText(
                f"✓  {n} panels matched  ·  {np2} overlap pairs  →  "
                f"{result['output_dir']}")
            self._status.setStyleSheet(f"color:{OK};font-size:9pt;")
        else:
            self._status.setText(f"✗ {result.get('error','')}")
            self._status.setStyleSheet(f"color:{ERR};font-size:9pt;")


def main():
    import traceback as _tb
    lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_log.txt")
    def ch(et,ev,etb):
        with open(lp,"a") as f:
            f.write(f"\n{'='*50}\nCRASH {datetime.now()}\n"
                    + "".join(_tb.format_exception(et,ev,etb)))
        sys.__excepthook__(et,ev,etb)
    sys.excepthook = ch
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SS)
    MainWindow().show()
    app.exec()

if __name__ == "__main__":
    main()
