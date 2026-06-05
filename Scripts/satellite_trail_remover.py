r"""
satellite_trail_remover.py  —  HYPERLOAD Satellite Trail Detector & Inpainter
==============================================================================
Detects and removes satellite trails from astronomical FITS images.

With Starlink, OneWeb and other mega-constellations, satellite trails
affect a rapidly growing fraction of ground-based exposures. This tool
provides a complete pipeline:

Detection pipeline (ASTA approach, Groot et al. 2024, A&A 692, A111):
    1. Background subtract (median grid model)
    2. Gaussian blur to suppress star noise
    3. Canny edge detection (Canny 1986, IEEE TPAMI 8(6), 679)
    4. Probabilistic Hough Transform (Galamhos et al. 1999, CVPR)
       → returns candidate line segments
    5. Line clustering: merge collinear nearby segments
    6. Trail validation: verify brightness profile along each line
    7. Mask dilation: expand mask to full trail width

Inpainting — two algorithms:
    MEDIAN DIFFUSION (fast, Bertalmío et al. 2000, ACM SIGGRAPH):
        Replace masked pixels with a distance-weighted median of unmasked
        neighbours. Propagate inward from the mask boundary in layers.
        Best for: sparse star fields, thin trails, fast processing.

    CRIMINISI EXEMPLAR-BASED (Criminisi, Pérez & Toyama 2004, IEEE TIP 13(9)):
        Finds the best-matching patch from the unmasked region and copies it
        in, preserving texture and structure across the trail boundary.
        Priority: C(p) = D(p) × R(p) — data term × confidence term.
        Best for: trails crossing nebulosity, gradients, complex backgrounds.

Modes:
    PRE-STACK: process individual frames before stacking (recommended).
        Trails appear at different positions per frame — after inpainting,
        the stacker has clean data at every pixel position.
    POST-STACK: process a single stacked image (trail already baked in).

References:
    ASTA:       Groot et al. 2024, A&A 692, A111  (arXiv:2407.19461)
    Hough:      Duda & Hart 1972, CACM 15(1), 11
    PHT:        Galamhos, Matas & Kittler 1999, CVPR
    Criminisi:  Criminisi, Pérez & Toyama 2004, IEEE TIP 13(9), 1200
    Diffusion:  Bertalmío, Sapiro, Caselles & Ballester 2000, ACM SIGGRAPH
    Starlink:   McDowell 2020, ApJ 892, L36

Place in: Siril Suites folder
Run via:  Siril → Scripts → satellite_trail_remover
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("scipy")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")

import os
import sys
import glob
import threading
import traceback
from datetime import datetime

import numpy as np
from scipy.ndimage import (gaussian_filter, binary_dilation, binary_erosion,
                            label, uniform_filter)
from astropy.io import fits as astropy_fits

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QRadioButton, QButtonGroup, QSplitter, QScrollArea,
    QSlider, QFrame, QSizePolicy, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import (QFont, QColor, QPainter, QPen, QBrush,
                          QPixmap, QImage, QPolygonF)

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

SIRIL_BG       = "#1e2128"
SIRIL_BG2      = "#252930"
SIRIL_BG3      = "#2d3340"
SIRIL_ACCENT   = "#4a9eff"
SIRIL_ACCENT2  = "#2d6abf"
SIRIL_TEXT     = "#dde3ee"
SIRIL_TEXT_DIM = "#7a8499"
SIRIL_BORDER   = "#3a4055"
SIRIL_SUCCESS  = "#4caf7d"
SIRIL_WARNING  = "#e8c46a"
SIRIL_ERROR    = "#cc4444"
SIRIL_NOVA     = "#ff6b35"
SIRIL_SECTION  = "#9db4d0"
SIRIL_TRAIL    = "#ffb347"  # orange for trail highlights

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: "Segoe UI", sans-serif; font-size: 9pt;
}}
QGroupBox {{
    border: 1px solid {SIRIL_BORDER}; border-radius: 5px;
    margin-top: 8px; padding: 6px;
    font-weight: bold; color: {SIRIL_SECTION};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QTabWidget::pane {{ border: 1px solid {SIRIL_BORDER}; background: {SIRIL_BG}; }}
QTabBar::tab {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT_DIM};
    padding: 6px 14px; border: 1px solid {SIRIL_BORDER};
    border-bottom: none; border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{ background: {SIRIL_BG}; color: {SIRIL_ACCENT}; font-weight: bold; }}
QPushButton {{
    background: {SIRIL_BG3}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    padding: 5px 12px; min-height: 22px;
}}
QPushButton:hover {{ background: {SIRIL_ACCENT2}; border-color: {SIRIL_ACCENT}; }}
QPushButton[objectName="primary"] {{
    background: {SIRIL_ACCENT2}; color: white; font-weight: bold;
    border-color: {SIRIL_ACCENT};
}}
QPushButton[objectName="danger"] {{
    background: #4a1e1e; color: #e07070; border-color: #803030;
}}
QPushButton[objectName="trail"] {{
    background: #3a2a0a; color: {SIRIL_TRAIL}; font-weight: bold;
    border-color: {SIRIL_TRAIL};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px; padding: 3px 6px;
}}
QProgressBar {{
    background: {SIRIL_BG2}; border: 1px solid {SIRIL_BORDER};
    border-radius: 4px; text-align: center;
}}
QProgressBar::chunk {{ background: {SIRIL_TRAIL}; border-radius: 3px; }}
QPlainTextEdit {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    font-family: "Courier New", monospace; font-size: 8pt;
}}
QTableWidget {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; gridline-color: {SIRIL_BORDER};
    alternate-background-color: {SIRIL_BG3};
}}
QHeaderView::section {{
    background: {SIRIL_BG3}; color: {SIRIL_SECTION};
    border: 1px solid {SIRIL_BORDER}; padding: 4px;
}}
QLabel[objectName="dim"]  {{ color: {SIRIL_TEXT_DIM}; }}
QLabel[objectName="ok"]   {{ color: {SIRIL_SUCCESS}; }}
QLabel[objectName="warn"] {{ color: {SIRIL_WARNING}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QScrollArea {{ border: none; }}
"""

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_background(image: np.ndarray, grid_size: int = 32) -> np.ndarray:
    """
    Robust background estimation via median grid + bicubic interpolation.
    Stars and trails are rejected by taking the median within each grid cell.
    """
    from scipy.ndimage import zoom
    h, w  = image.shape
    n_y   = max(2, h // grid_size)
    n_x   = max(2, w // grid_size)
    grid  = np.zeros((n_y, n_x))
    for iy in range(n_y):
        for ix in range(n_x):
            y0 = iy * h // n_y; y1 = (iy + 1) * h // n_y
            x0 = ix * w // n_x; x1 = (ix + 1) * w // n_x
            cell = image[y0:y1, x0:x1]
            grid[iy, ix] = np.median(cell)
    bg = zoom(grid, (h / n_y, w / n_x), order=3)
    bg = bg[:h, :w]
    return bg


# ─────────────────────────────────────────────────────────────────────────────
# CANNY EDGE DETECTION  (pure numpy / scipy)
# ─────────────────────────────────────────────────────────────────────────────

def canny_edges(image: np.ndarray,
                sigma: float = 2.0,
                low_thresh: float = 0.1,
                high_thresh: float = 0.3) -> np.ndarray:
    """
    Canny edge detector — Canny 1986, IEEE TPAMI 8(6), 679.
    1. Gaussian blur
    2. Gradient magnitude + direction (Sobel)
    3. Non-maximum suppression
    4. Double threshold + hysteresis
    Returns binary edge map.
    """
    from scipy.ndimage import sobel
    blurred = gaussian_filter(image.astype(np.float64), sigma=sigma)

    # Gradient
    gx = sobel(blurred, axis=1)
    gy = sobel(blurred, axis=0)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)

    # Normalise magnitude
    if mag.max() > 0:
        mag_n = mag / mag.max()
    else:
        return np.zeros_like(image, dtype=bool)

    # Non-maximum suppression
    h, w   = mag_n.shape
    suppressed = np.zeros_like(mag_n)
    ang_q  = (np.round(ang / (np.pi / 4)) % 4).astype(int)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            q = ang_q[y, x]
            if q == 0:
                n1, n2 = mag_n[y, x-1], mag_n[y, x+1]
            elif q == 1:
                n1, n2 = mag_n[y-1, x+1], mag_n[y+1, x-1]
            elif q == 2:
                n1, n2 = mag_n[y-1, x], mag_n[y+1, x]
            else:
                n1, n2 = mag_n[y-1, x-1], mag_n[y+1, x+1]
            if mag_n[y, x] >= n1 and mag_n[y, x] >= n2:
                suppressed[y, x] = mag_n[y, x]

    # Double threshold
    strong = suppressed >= high_thresh
    weak   = (suppressed >= low_thresh) & ~strong

    # Hysteresis: connect weak edges to strong ones
    edges = strong.copy()
    changed = True
    while changed:
        dilated = binary_dilation(edges)
        new_edges = edges | (weak & dilated)
        changed = np.any(new_edges != edges)
        edges = new_edges

    return edges


def fast_canny(image: np.ndarray,
               sigma: float = 2.0,
               low_thresh_pct: float = 5.0,
               high_thresh_pct: float = 15.0) -> np.ndarray:
    """
    Faster Canny using scipy Sobel + vectorised NMS.
    Thresholds are percentiles of the gradient magnitude.
    """
    from scipy.ndimage import sobel
    blurred = gaussian_filter(image.astype(np.float64), sigma=sigma)
    gx = sobel(blurred, axis=1)
    gy = sobel(blurred, axis=0)
    mag = np.hypot(gx, gy)

    # Adaptive thresholds from percentile of non-zero magnitudes
    nonzero = mag[mag > 0]
    if len(nonzero) == 0:
        return np.zeros_like(image, dtype=bool)
    lo = np.percentile(nonzero, low_thresh_pct)
    hi = np.percentile(nonzero, high_thresh_pct)

    # Simple threshold without full NMS for speed
    edges = mag > hi
    # Add weak edges connected to strong ones
    weak = (mag > lo) & ~edges
    dilated = binary_dilation(edges, iterations=2)
    edges = edges | (weak & dilated)
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILISTIC HOUGH TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

def probabilistic_hough(edges: np.ndarray,
                         threshold: int = 50,
                         min_line_length: int = 100,
                         max_line_gap: int = 10,
                         n_angles: int = 180) -> list:
    """
    Probabilistic Hough Transform for line detection.
    Galamhos, Matas & Kittler 1999, CVPR.

    Returns list of ((x0,y0),(x1,y1)) line segments.
    """
    h, w      = edges.shape
    edge_pts  = list(zip(*np.where(edges)))
    if not edge_pts:
        return []

    # Shuffle for probabilistic sampling
    import random
    edge_pts  = list(edge_pts)
    random.shuffle(edge_pts)

    thetas = np.linspace(-np.pi/2, np.pi/2, n_angles, endpoint=False)
    d_max  = int(np.hypot(h, w))

    # Accumulator
    acc    = {}
    used   = np.zeros_like(edges, dtype=bool)
    lines  = []

    for (py, px) in edge_pts:
        if used[py, px]:
            continue
        # Vote
        votes = {}
        for i, theta in enumerate(thetas):
            rho = int(px * np.cos(theta) + py * np.sin(theta))
            key = (rho, i)
            votes[key] = votes.get(key, 0) + 1

        best_key = max(votes, key=lambda k: votes[k])
        if votes[best_key] < threshold:
            continue

        # Trace the line
        rho, t_idx = best_key
        theta = thetas[t_idx]
        ct, st = np.cos(theta), np.sin(theta)

        # Collect all edge points near this line
        on_line = []
        for (qy, qx) in edge_pts:
            if used[qy, qx]: continue
            d = abs(qx * ct + qy * st - rho)
            if d < 1.5:
                on_line.append((qx, qy))

        if len(on_line) < min_line_length // 2:
            continue

        # Project onto line direction
        perp_x, perp_y = -st, ct  # direction along the line
        projs = [(qx * perp_x + qy * perp_y, qx, qy) for qx, qy in on_line]
        projs.sort(key=lambda p: p[0])

        # Find contiguous segments
        segments = []; seg_start = None; seg_last = None; seg_pts = []
        for proj, qx, qy in projs:
            if seg_start is None:
                seg_start = proj; seg_last = proj; seg_pts = [(qx, qy)]
            elif proj - seg_last <= max_line_gap:
                seg_last = proj; seg_pts.append((qx, qy))
            else:
                if seg_last - seg_start >= min_line_length:
                    p0 = seg_pts[0]; p1 = seg_pts[-1]
                    segments.append(((p0[0], p0[1]), (p1[0], p1[1])))
                seg_start = proj; seg_last = proj; seg_pts = [(qx, qy)]
        if seg_start is not None and seg_last - seg_start >= min_line_length:
            p0 = seg_pts[0]; p1 = seg_pts[-1]
            segments.append(((p0[0], p0[1]), (p1[0], p1[1])))

        lines.extend(segments)
        # Mark used
        for _, qx, qy in projs:
            used[qy, qx] = True

    return lines


def cluster_lines(lines: list, angle_tol: float = 3.0,
                  dist_tol: float = 10.0) -> list:
    """
    Merge collinear nearby line segments into single trail descriptions.
    Returns list of {"angle","rho","x0","y0","x1","y1"} dicts.
    """
    if not lines:
        return []

    def line_params(x0, y0, x1, y1):
        dx = x1 - x0; dy = y1 - y0
        length = np.hypot(dx, dy)
        if length == 0: return None
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        # Perpendicular distance from origin
        rho = abs(x0 * dy - y0 * dx) / length
        return angle, rho, length

    trails = []
    used   = [False] * len(lines)

    for i, ((x0a, y0a), (x1a, y1a)) in enumerate(lines):
        if used[i]: continue
        p = line_params(x0a, y0a, x1a, y1a)
        if p is None: continue
        ang_a, rho_a, _ = p
        cluster_x = [x0a, x1a]; cluster_y = [y0a, y1a]
        used[i] = True

        for j, ((x0b, y0b), (x1b, y1b)) in enumerate(lines):
            if used[j]: continue
            q = line_params(x0b, y0b, x1b, y1b)
            if q is None: continue
            ang_b, rho_b, _ = q
            ang_diff = min(abs(ang_a - ang_b), 180 - abs(ang_a - ang_b))
            if ang_diff < angle_tol and abs(rho_a - rho_b) < dist_tol:
                cluster_x += [x0b, x1b]
                cluster_y += [y0b, y1b]
                used[j] = True

        x_min = min(cluster_x); x_max = max(cluster_x)
        y_min = min(cluster_y); y_max = max(cluster_y)
        trails.append({
            "angle": ang_a, "rho": rho_a,
            "x0": x_min, "y0": y_min, "x1": x_max, "y1": y_max,
            "length": np.hypot(x_max - x_min, y_max - y_min),
        })

    return trails


def validate_trail(image: np.ndarray, trail: dict,
                   brightness_factor: float = 1.5,
                   n_samples: int = 30) -> bool:
    """
    Validate trail by checking median brightness along the line
    vs. perpendicular offset. Trail is real if on-line pixels are
    significantly brighter than adjacent off-line pixels.
    """
    x0, y0, x1, y1 = trail["x0"], trail["y0"], trail["x1"], trail["y1"]
    h, w = image.shape

    # Sample points along the line
    xs = np.linspace(x0, x1, n_samples).astype(int)
    ys = np.linspace(y0, y1, n_samples).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[valid]; ys = ys[valid]
    if len(xs) == 0:
        return False

    on_line = np.median(image[ys, xs])

    # Sample perpendicular offset (5 pixels away)
    dx = x1 - x0; dy = y1 - y0
    length = max(np.hypot(dx, dy), 1)
    perp_x = int(-dy / length * 5); perp_y = int(dx / length * 5)
    xs_off = np.clip(xs + perp_x, 0, w-1)
    ys_off = np.clip(ys + perp_y, 0, h-1)
    off_line = np.median(image[ys_off, xs_off])

    return float(on_line) > float(off_line) * brightness_factor


# ─────────────────────────────────────────────────────────────────────────────
# MASK GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_trail_mask(image_shape: tuple, trails: list,
                     trail_width: int = 5,
                     dilation_px: int = 3) -> np.ndarray:
    """
    Build binary mask covering all detected trails.
    Mask is dilated by dilation_px to include trail wings.
    Returns bool array, True where trail pixels are.
    """
    h, w   = image_shape
    mask   = np.zeros((h, w), dtype=bool)

    for trail in trails:
        x0, y0 = int(trail["x0"]), int(trail["y0"])
        x1, y1 = int(trail["x1"]), int(trail["y1"])
        # Draw thick line by rasterising
        n_pts = max(int(np.hypot(x1-x0, y1-y0)) * 2, 2)
        xs = np.linspace(x0, x1, n_pts).astype(int)
        ys = np.linspace(y0, y1, n_pts).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        mask[ys[valid], xs[valid]] = True

    # Dilate to cover trail width
    half_w = trail_width // 2 + dilation_px
    struct = np.ones((2*half_w+1, 2*half_w+1), dtype=bool)
    mask   = binary_dilation(mask, structure=struct)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# INPAINTING — MEDIAN DIFFUSION
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_diffusion(image: np.ndarray, mask: np.ndarray,
                      n_iter: int = 50, radius: int = 5,
                      log_callback=None) -> np.ndarray:
    """
    Median diffusion inpainting.
    Bertalmío, Sapiro, Caselles & Ballester 2000, ACM SIGGRAPH.

    Iteratively replaces masked pixels with the median of their
    unmasked neighbours within a given radius. Propagates inward
    from the mask boundary, layer by layer.

    Fast and effective for thin trails in star fields.
    """
    log = log_callback or (lambda m: None)
    log("  Inpainting: Median diffusion (Bertalmío et al. 2000)")
    result = image.astype(np.float64).copy()
    remaining = mask.copy()

    for it in range(n_iter):
        if not remaining.any():
            log(f"    Converged at iteration {it+1}")
            break
        # Find boundary pixels (masked pixels with at least one unmasked neighbour)
        boundary = remaining & binary_dilation(~remaining)
        if not boundary.any():
            break
        ys, xs = np.where(boundary)
        n_filled = 0
        for y, x in zip(ys, xs):
            y0 = max(0, y - radius); y1 = min(result.shape[0], y + radius + 1)
            x0 = max(0, x - radius); x1 = min(result.shape[1], x + radius + 1)
            patch      = result[y0:y1, x0:x1]
            patch_mask = remaining[y0:y1, x0:x1]
            unmasked   = patch[~patch_mask]
            if len(unmasked) > 0:
                result[y, x]    = np.median(unmasked)
                remaining[y, x] = False
                n_filled        += 1
        if it % 10 == 0:
            pct_left = remaining.sum() / max(mask.sum(), 1) * 100
            log(f"    Iter {it+1}: {n_filled} pixels filled  "
                f"({pct_left:.1f}% remaining)")

    # Fill any remaining pixels with global background
    if remaining.any():
        result[remaining] = np.median(result[~remaining]) if (~remaining).any() else 0
    return result.astype(image.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# INPAINTING — CRIMINISI EXEMPLAR-BASED
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_criminisi(image: np.ndarray, mask: np.ndarray,
                      patch_size: int = 9,
                      log_callback=None) -> np.ndarray:
    """
    Criminisi exemplar-based inpainting.
    Criminisi, Pérez & Toyama 2004, IEEE TIP 13(9), 1200.

    Priority: C(p) = D(p) × R(p)
        D(p) = data term  = |∇I⊥(p) · n̂p| / α  (isophote direction × boundary normal)
        R(p) = confidence = Σ C(q) / |Ψp|       (known-pixel fraction in patch)

    Each iteration fills the highest-priority boundary patch with the
    best-matching patch from the source (unmasked) region.
    Best for trails crossing structured backgrounds (nebulae, gradients).
    """
    log = log_callback or (lambda m: None)
    log(f"  Inpainting: Criminisi exemplar-based (patch={patch_size}px)")
    from scipy.ndimage import convolve

    result     = image.astype(np.float64).copy()
    confidence = (~mask).astype(np.float64)  # 1 = known, 0 = unknown
    fill_mask  = mask.copy()
    h, w       = image.shape
    half       = patch_size // 2
    alpha      = 255.0  # normalization constant

    # Gradient operators
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    sobel_y = sobel_x.T

    n_total = int(fill_mask.sum())
    n_done  = 0

    max_iter = n_total * 2  # safety limit
    it = 0

    while fill_mask.any() and it < max_iter:
        it += 1
        # ── Find boundary pixels ──────────────────────────────────────────────
        interior  = binary_erosion(fill_mask)
        boundary  = fill_mask & ~interior
        bpts      = list(zip(*np.where(boundary)))
        if not bpts:
            break

        # ── Compute priority for each boundary patch ──────────────────────────
        # Gradient of result
        Gx = convolve(result, sobel_x)
        Gy = convolve(result, sobel_y)
        # Normal to fill front (gradient of mask)
        Nx = convolve(fill_mask.astype(float), sobel_x)
        Ny = convolve(fill_mask.astype(float), sobel_y)
        N_mag = np.hypot(Nx, Ny) + 1e-10

        best_p = None; best_pri = -1.0
        for (py, px) in bpts:
            # Confidence
            y0 = max(0, py-half); y1 = min(h, py+half+1)
            x0 = max(0, px-half); x1 = min(w, px+half+1)
            C_p = np.sum(confidence[y0:y1, x0:x1]) / patch_size**2

            # Data term: |∇I⊥ · n̂|
            # ∇I⊥ is the isophote direction = (-Gy, Gx)
            # n̂ is the boundary normal
            nx = Nx[py, px] / N_mag[py, px]
            ny = Ny[py, px] / N_mag[py, px]
            # Isophote direction (perpendicular to gradient)
            iso_x = -Gy[py, px]; iso_y = Gx[py, px]
            D_p = abs(iso_x * nx + iso_y * ny) / alpha

            priority = C_p * D_p
            if priority > best_pri:
                best_pri = priority; best_p = (py, px)

        if best_p is None:
            break
        py, px = best_p

        # ── Extract target patch ──────────────────────────────────────────────
        y0t = max(0, py-half); y1t = min(h, py+half+1)
        x0t = max(0, px-half); x1t = min(w, px+half+1)
        target_patch = result[y0t:y1t, x0t:x1t]
        target_mask  = fill_mask[y0t:y1t, x0t:x1t]
        ph, pw = target_patch.shape

        # ── Find best-matching source patch ──────────────────────────────────
        best_ssd = np.inf; best_src = None
        # Search over source (unmasked) region — sample for speed
        stride = max(1, patch_size // 2)
        for sy in range(half, h - half, stride):
            for sx in range(half, w - half, stride):
                if fill_mask[sy, sx]:
                    continue  # skip masked source patches
                y0s = sy - half; y1s = y0s + ph
                x0s = sx - half; x1s = x0s + pw
                if y0s < 0 or y1s > h or x0s < 0 or x1s > w:
                    continue
                src_patch = result[y0s:y1s, x0s:x1s]
                if src_patch.shape != target_patch.shape:
                    continue
                # SSD over known pixels only
                known = ~target_mask
                if not known.any():
                    continue
                ssd = np.sum((target_patch[known] - src_patch[known])**2)
                if ssd < best_ssd:
                    best_ssd = ssd; best_src = (y0s, x0s)

        if best_src is None:
            # Fallback: just use median of known pixels
            known_vals = result[y0t:y1t, x0t:x1t][~target_mask]
            fill_val = np.median(known_vals) if len(known_vals) else 0.0
            result[y0t:y1t, x0t:x1t] = np.where(
                target_mask[:ph, :pw], fill_val,
                result[y0t:y1t, x0t:x1t])
        else:
            sy0, sx0 = best_src
            src_patch = result[sy0:sy0+ph, sx0:sx0+pw]
            result[y0t:y1t, x0t:x1t] = np.where(
                target_mask, src_patch[:ph, :pw],
                result[y0t:y1t, x0t:x1t])

        # Update confidence and mask
        confidence[y0t:y1t, x0t:x1t] = np.where(
            target_mask, best_pri, confidence[y0t:y1t, x0t:x1t])
        fill_mask[y0t:y1t, x0t:x1t] = False

        n_done += int(target_mask.sum())
        if it % 20 == 0:
            pct = 100 * n_done / max(n_total, 1)
            log(f"    Iter {it}: {pct:.1f}% filled")

    # Safety: fill remaining with diffusion
    if fill_mask.any():
        log(f"    Filling {fill_mask.sum()} remaining pixels with diffusion")
        result = inpaint_diffusion(result, fill_mask, n_iter=30, radius=5,
                                    log_callback=lambda m: None)

    log(f"    Criminisi complete: {n_done} pixels filled, {it} iterations")
    return result.astype(image.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# FULL DETECTION + INPAINTING WORKER
# ─────────────────────────────────────────────────────────────────────────────

class TrailWorker(QThread):
    """
    Full satellite trail pipeline worker.
    Detection → Mask → Inpainting → Save
    Supports single FITS, batch folder, pre-stack mode.
    """
    progress      = pyqtSignal(int, int, str)
    log_line      = pyqtSignal(str)
    trails_found  = pyqtSignal(list, np.ndarray)  # trails list, detection image
    mask_ready    = pyqtSignal(np.ndarray)         # binary mask
    result_ready  = pyqtSignal(np.ndarray, np.ndarray)  # original, inpainted
    finished      = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event,
                 manual_mask: np.ndarray = None, parent=None):
        super().__init__(parent)
        self.cfg         = config
        self._cancel     = cancel_event
        self._manual_mask = manual_mask

    def run(self):
        try:
            self._run()
        except Exception as e:
            tb = traceback.format_exc()
            self.log_line.emit(f"CRITICAL ERROR: {e}\n{tb}")
            self.finished.emit({"success": False, "error": str(e)})

    def _run(self):
        cfg = self.cfg
        log = self.log_line.emit
        log("═══ HyperLoad Satellite Trail Remover ═══")
        log(f"  Mode:      {cfg['mode']}")
        log(f"  Inpainter: {cfg['inpainter']}")
        log(f"  Trail width: {cfg['trail_width']} px")
        log("")

        if cfg["mode"] in ("single", "pre_stack_single"):
            self._process_single(cfg["input_path"])
        elif cfg["mode"] in ("batch", "pre_stack_batch"):
            self._process_batch()
        else:
            self.finished.emit({"success": False,
                                "error": f"Unknown mode: {cfg['mode']}"})

    def _process_single(self, path: str):
        log = self.log_line.emit
        log(f"Loading: {os.path.basename(path)}")
        self.progress.emit(5, 100, "Loading FITS…")

        try:
            with astropy_fits.open(path) as hdul:
                data   = hdul[0].data.astype(np.float32)
                header = hdul[0].header.copy()
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)}); return

        # Handle RGB
        is_rgb = (data.ndim == 3)
        if is_rgb and data.shape[0] == 3:
            lum = 0.299*data[0] + 0.587*data[1] + 0.114*data[2]
        elif is_rgb:
            lum = data[0]
        else:
            lum = data

        self.progress.emit(10, 100, "Estimating background…")
        bg  = estimate_background(lum, grid_size=self.cfg.get("bg_grid", 64))
        sub = lum - bg
        sub = np.clip(sub, 0, None)

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── Detection ─────────────────────────────────────────────────────────
        self.progress.emit(20, 100, "Detecting edges…")
        log("Step 1: Edge detection")
        norm_sub = sub / (sub.max() + 1e-10)
        edges    = fast_canny(norm_sub,
                               sigma=self.cfg.get("canny_sigma", 2.0),
                               low_thresh_pct=self.cfg.get("low_thresh", 5.0),
                               high_thresh_pct=self.cfg.get("high_thresh", 15.0))
        log(f"  Edge pixels: {edges.sum()}")

        self.progress.emit(35, 100, "Hough transform…")
        log("Step 2: Probabilistic Hough Transform")
        lines = probabilistic_hough(
            edges,
            threshold=self.cfg.get("hough_threshold", 40),
            min_line_length=self.cfg.get("min_line_length", 80),
            max_line_gap=self.cfg.get("max_line_gap", 15))
        log(f"  Candidate line segments: {len(lines)}")

        self.progress.emit(45, 100, "Clustering lines…")
        log("Step 3: Clustering and validation")
        trails = cluster_lines(lines,
                                angle_tol=self.cfg.get("angle_tol", 3.0),
                                dist_tol=self.cfg.get("dist_tol", 10.0))
        log(f"  Clustered trails: {len(trails)}")

        # Validate brightness
        valid_trails = []
        for t in trails:
            if validate_trail(sub, t,
                              brightness_factor=self.cfg.get("brightness_factor", 1.5)):
                valid_trails.append(t)
        log(f"  Validated trails: {len(valid_trails)}")
        self.trails_found.emit(valid_trails, norm_sub)

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # Use manual mask additions if provided
        mask = build_trail_mask(
            lum.shape, valid_trails,
            trail_width=self.cfg.get("trail_width", 5),
            dilation_px=self.cfg.get("dilation_px", 3))

        if self._manual_mask is not None:
            mask = mask | self._manual_mask.astype(bool)

        n_masked = int(mask.sum())
        pct_masked = 100 * n_masked / max(lum.size, 1)
        log(f"  Masked pixels: {n_masked} ({pct_masked:.3f}% of image)")
        self.mask_ready.emit(mask)

        if not mask.any():
            log("  No trails detected — saving original unchanged")
            self._save(data, header, path, suffix="_notrails")
            self.finished.emit({"success": True, "n_trails": 0,
                                "output_path": path})
            return

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── Inpainting ────────────────────────────────────────────────────────
        self.progress.emit(60, 100, "Inpainting trails…")
        log(f"Step 4: Inpainting ({self.cfg['inpainter']})")

        def _inpaint_channel(ch: np.ndarray) -> np.ndarray:
            if self.cfg["inpainter"] == "Criminisi (exemplar)":
                return inpaint_criminisi(
                    ch, mask,
                    patch_size=self.cfg.get("patch_size", 9),
                    log_callback=log)
            else:
                return inpaint_diffusion(
                    ch, mask,
                    n_iter=self.cfg.get("diffusion_iter", 50),
                    radius=self.cfg.get("diffusion_radius", 5),
                    log_callback=log)

        if is_rgb and data.ndim == 3 and data.shape[0] == 3:
            result_channels = []
            for i, ch_name in enumerate(["R", "G", "B"]):
                if self._cancel.is_set(): break
                log(f"  Channel {ch_name}…")
                result_channels.append(_inpaint_channel(data[i]))
            if self._cancel.is_set():
                self.finished.emit({"success": False, "error": "Cancelled"})
                return
            result_data = np.stack(result_channels, axis=0)
        else:
            result_data = _inpaint_channel(lum)

        self.result_ready.emit(lum, result_data if result_data.ndim == 2 else
                               0.299*result_data[0] + 0.587*result_data[1] +
                               0.114*result_data[2])

        self.progress.emit(90, 100, "Saving…")
        out_path = self._save(result_data, header, path, suffix="_notrails")
        log(f"✓ Saved: {out_path}")

        self.progress.emit(100, 100, "Done")
        self.finished.emit({
            "success":    True,
            "n_trails":   len(valid_trails),
            "n_masked":   n_masked,
            "output_path": out_path,
        })

    def _process_batch(self):
        log = self.log_line.emit
        folder = self.cfg["input_folder"]
        files  = sorted(
            glob.glob(os.path.join(folder, "*.fit")) +
            glob.glob(os.path.join(folder, "*.fits")) +
            glob.glob(os.path.join(folder, "*.fts")))
        if not files:
            self.finished.emit({"success": False,
                                "error": "No FITS files found"}); return
        log(f"Batch: {len(files)} files")
        n_processed = 0; n_trails_total = 0

        for i, path in enumerate(files):
            if self._cancel.is_set(): break
            self.progress.emit(i, len(files),
                               f"[{i+1}/{len(files)}] {os.path.basename(path)}")
            log(f"\n── {os.path.basename(path)} ──")
            try:
                self._process_single(path)
                n_processed += 1
            except Exception as e:
                log(f"  Skip {os.path.basename(path)}: {e}")

        log(f"\n✓ Batch complete: {n_processed}/{len(files)} processed")
        self.progress.emit(len(files), len(files), "Done")
        self.finished.emit({
            "success":       True,
            "n_processed":   n_processed,
            "output_dir":    self.cfg["output_dir"],
        })

    def _save(self, data: np.ndarray, header,
              input_path: str, suffix: str = "_notrails") -> str:
        out_dir  = self.cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        base     = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(out_dir, f"{base}{suffix}.fit")
        hdu = astropy_fits.PrimaryHDU(data=data.astype(np.float32),
                                       header=header)
        hdu.header["HISTORY"] = (
            f"HyperLoad satellite trail removal  "
            f"inpainter={self.cfg['inpainter']}")
        hdu.writeto(out_path, overwrite=True)
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# TRAIL DETECTION PREVIEW CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class TrailCanvas(FigureCanvasQTAgg):
    """3-panel canvas: detection + mask overlay, before, after."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(14, 5), facecolor=SIRIL_BG)
        self.ax_detect = self.fig.add_subplot(1, 3, 1)
        self.ax_before = self.fig.add_subplot(1, 3, 2)
        self.ax_after  = self.fig.add_subplot(1, 3, 3)
        for ax, t, c in [(self.ax_detect, "Detection + mask", SIRIL_TRAIL),
                          (self.ax_before, "Original",         SIRIL_ERROR),
                          (self.ax_after,  "Inpainted",        SIRIL_SUCCESS)]:
            ax.set_facecolor(SIRIL_BG2)
            ax.set_title(t, color=c, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values(): sp.set_color(SIRIL_BORDER)
        super().__init__(self.fig)
        self.setParent(parent)

    def _ds(self, img: np.ndarray, max_px: int = 500) -> np.ndarray:
        h = img.shape[0]; f = max(1, h // max_px)
        return img[::f, ::f]

    def _disp(self, img: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(img, [0.5, 99.5])
        return np.clip((img - lo) / (hi - lo + 1e-10), 0, 1)

    def show_detection(self, detection_img: np.ndarray,
                        trails: list, mask: np.ndarray = None):
        self.ax_detect.clear()
        self.ax_detect.set_facecolor(SIRIL_BG2)
        disp = self._ds(self._disp(detection_img))
        rgb  = np.stack([disp, disp, disp], axis=-1)

        # Overlay mask in orange
        if mask is not None:
            mask_ds = self._ds(mask.astype(float))
            f = detection_img.shape[0] // disp.shape[0]
            rgb[:, :, 0] = np.clip(rgb[:, :, 0] + mask_ds * 0.7, 0, 1)
            rgb[:, :, 1] = np.clip(rgb[:, :, 1] - mask_ds * 0.4, 0, 1)
            rgb[:, :, 2] = np.clip(rgb[:, :, 2] - mask_ds * 0.4, 0, 1)

        self.ax_detect.imshow(rgb, origin="upper", interpolation="nearest")

        # Draw trail lines
        h_orig = detection_img.shape[0]
        scale  = disp.shape[0] / h_orig
        for t in trails:
            x0 = t["x0"]*scale; y0 = t["y0"]*scale
            x1 = t["x1"]*scale; y1 = t["y1"]*scale
            self.ax_detect.plot([x0, x1], [y0, y1],
                                color=SIRIL_TRAIL, lw=1.5, alpha=0.9)
        n = len(trails)
        self.ax_detect.set_title(
            f"Detection — {n} trail{'s' if n!=1 else ''} found",
            color=SIRIL_TRAIL, fontsize=9)
        self.ax_detect.tick_params(left=False, bottom=False,
                                    labelleft=False, labelbottom=False)
        for sp in self.ax_detect.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

    def show_before_after(self, before: np.ndarray, after: np.ndarray):
        for ax, img, t, c in [
            (self.ax_before, before, "Original",  SIRIL_ERROR),
            (self.ax_after,  after,  "Inpainted", SIRIL_SUCCESS),
        ]:
            ax.clear(); ax.set_facecolor(SIRIL_BG2)
            ax.imshow(self._ds(self._disp(img)), cmap="gray",
                      origin="upper", interpolation="nearest")
            ax.set_title(t, color=c, fontsize=9)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# TRAIL TABLE
# ─────────────────────────────────────────────────────────────────────────────

class TrailTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(
            ["#", "Angle (°)", "Length (px)", "Start (x,y)", "End (x,y)"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setMaximumHeight(160)

    def populate(self, trails: list):
        self.setRowCount(0)
        for i, t in enumerate(trails):
            row = self.rowCount(); self.insertRow(row)
            for col, val in enumerate([
                str(i+1),
                f"{t['angle']:.1f}",
                f"{t['length']:.0f}",
                f"({t['x0']:.0f}, {t['y0']:.0f})",
                f"({t['x1']:.0f}, {t['y1']:.0f})",
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(row, col, item)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Trail Remover  —  HYPERLOAD  —  Siril")
        self.resize(1440, 900)
        self._worker        = None
        self._cancel_event  = threading.Event()
        self._detected_trails = []
        self._current_mask    = None
        self._manual_mask     = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # Header
        header = QWidget(); header.setFixedHeight(56)
        header.setStyleSheet(
            f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon  = QLabel("🛰")
        lbl_icon.setStyleSheet("font-size:20pt; background:transparent;")
        lbl_title = QLabel("Satellite Trail Remover")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_TRAIL}; font-size:14pt; font-weight:bold; "
            f"background:transparent;")
        lbl_sub = QLabel(
            "Canny edges  ·  Probabilistic Hough Transform  ·  "
            "Criminisi exemplar inpainting  ·  Batch pre-stack support  ·  "
            "Groot et al. 2024 pipeline")
        lbl_sub.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title)
        hl.addWidget(lbl_sub); hl.addStretch()
        root.addWidget(header)

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._tabs.addTab(self._build_tab_input(),     "📁  Input")
        self._tabs.addTab(self._build_tab_detection(), "🔍  Detection")
        self._tabs.addTab(self._build_tab_inpaint(),   "🖌  Inpainting")
        self._tabs.addTab(self._build_tab_preview(),   "🖼  Preview")
        self._tabs.addTab(self._build_tab_output(),    "💾  Output")
        self._tabs.addTab(self._build_tab_log(),       "📋  Log")

        # Bottom bar
        bottom = QWidget(); bottom.setFixedHeight(50)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 6, 10, 6)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(10); self._progress.setValue(0)
        bl.addWidget(self._progress, 1)

        self._btn_detect = QPushButton("🔍  Detect only")
        self._btn_detect.clicked.connect(lambda: self._run(detect_only=True))
        bl.addWidget(self._btn_detect)

        self._btn_run = QPushButton("▶  Detect + Remove")
        self._btn_run.setObjectName("trail")
        self._btn_run.setMinimumWidth(160); self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run)
        bl.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(34); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bl.addWidget(self._btn_cancel)

        self._status = QLabel("Ready — load a FITS file")
        self._status.setObjectName("dim")
        self._status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status, 1)
        root.addWidget(bottom)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_tab_input(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        grp_mode = QGroupBox("Processing mode")
        mg = QVBoxLayout(grp_mode)
        self._rb_single  = QRadioButton("Single FITS image")
        self._rb_batch   = QRadioButton(
            "Batch folder  (process all FITS, pre-stack recommended)")
        self._rb_single.setChecked(True)
        self._bg_mode = QButtonGroup()
        self._bg_mode.addButton(self._rb_single)
        self._bg_mode.addButton(self._rb_batch)
        for rb in [self._rb_single, self._rb_batch]:
            mg.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
        lbl_m = QLabel(
            "PRE-STACK mode (batch): process individual calibrated frames.\n"
            "Trails appear at different positions per frame — after inpainting,\n"
            "the stacker has clean data at every pixel. Best approach.\n\n"
            "POST-STACK mode (single): process an already-stacked image where\n"
            "the trail has been combined in. Inpainting quality depends on how\n"
            "much clean data surrounds the trail.")
        lbl_m.setObjectName("dim"); lbl_m.setWordWrap(True); mg.addWidget(lbl_m)
        lay.addWidget(grp_mode)

        # Single file input
        self._grp_single = QGroupBox("Single FITS")
        sl = QHBoxLayout(self._grp_single)
        self._edit_single = QLineEdit()
        self._edit_single.setPlaceholderText("FITS file path…")
        btn_s = QPushButton("Browse…"); btn_s.setFixedWidth(80)
        btn_s.clicked.connect(self._browse_single)
        sl.addWidget(self._edit_single); sl.addWidget(btn_s)
        lay.addWidget(self._grp_single)

        # Batch folder input
        self._grp_batch = QGroupBox("Batch folder")
        bf = QHBoxLayout(self._grp_batch)
        self._edit_batch = QLineEdit()
        self._edit_batch.setPlaceholderText("Folder containing FITS files…")
        btn_b = QPushButton("Browse…"); btn_b.setFixedWidth(80)
        btn_b.clicked.connect(self._browse_batch)
        bf.addWidget(self._edit_batch); bf.addWidget(btn_b)
        self._grp_batch.setVisible(False)
        lay.addWidget(self._grp_batch)

        lay.addStretch()
        return w

    def _build_tab_detection(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        grp_bg = QGroupBox("Background subtraction")
        bf = QFormLayout(grp_bg)
        self._spin_bg_grid = QSpinBox()
        self._spin_bg_grid.setRange(16, 256); self._spin_bg_grid.setValue(64)
        self._spin_bg_grid.setSuffix(" px")
        bf.addRow("Grid cell size:", self._spin_bg_grid)
        lbl_bg = QLabel("Median grid model. Larger = smoother background estimation.")
        lbl_bg.setObjectName("dim"); bf.addRow(lbl_bg)
        lay.addWidget(grp_bg)

        grp_canny = QGroupBox("Canny edge detection  (Canny 1986)")
        cf = QFormLayout(grp_canny)
        self._spin_canny_sigma = QDoubleSpinBox()
        self._spin_canny_sigma.setRange(0.5, 10.0); self._spin_canny_sigma.setValue(2.0)
        self._spin_canny_sigma.setSuffix(" σ")
        cf.addRow("Gaussian blur σ:", self._spin_canny_sigma)
        self._spin_low_thresh = QDoubleSpinBox()
        self._spin_low_thresh.setRange(1.0, 40.0); self._spin_low_thresh.setValue(5.0)
        self._spin_low_thresh.setSuffix(" %ile")
        cf.addRow("Low threshold:", self._spin_low_thresh)
        self._spin_high_thresh = QDoubleSpinBox()
        self._spin_high_thresh.setRange(5.0, 80.0); self._spin_high_thresh.setValue(15.0)
        self._spin_high_thresh.setSuffix(" %ile")
        cf.addRow("High threshold:", self._spin_high_thresh)
        lbl_c = QLabel(
            "σ: larger = more blur, fewer noise edges, misses faint trails.\n"
            "Thresholds: percentiles of gradient magnitude. Raise if too many"
            " false detections.")
        lbl_c.setObjectName("dim"); lbl_c.setWordWrap(True); cf.addRow(lbl_c)
        lay.addWidget(grp_canny)

        grp_hough = QGroupBox("Probabilistic Hough Transform  (Galamhos et al. 1999)")
        hf = QFormLayout(grp_hough)
        self._spin_hough_thresh = QSpinBox()
        self._spin_hough_thresh.setRange(10, 500); self._spin_hough_thresh.setValue(40)
        hf.addRow("Vote threshold:", self._spin_hough_thresh)
        self._spin_min_length = QSpinBox()
        self._spin_min_length.setRange(10, 5000); self._spin_min_length.setValue(80)
        self._spin_min_length.setSuffix(" px")
        hf.addRow("Min line length:", self._spin_min_length)
        self._spin_max_gap = QSpinBox()
        self._spin_max_gap.setRange(1, 200); self._spin_max_gap.setValue(15)
        self._spin_max_gap.setSuffix(" px")
        hf.addRow("Max gap:", self._spin_max_gap)
        self._spin_angle_tol = QDoubleSpinBox()
        self._spin_angle_tol.setRange(0.5, 20.0); self._spin_angle_tol.setValue(3.0)
        self._spin_angle_tol.setSuffix(" °")
        hf.addRow("Cluster angle tolerance:", self._spin_angle_tol)
        self._spin_dist_tol = QDoubleSpinBox()
        self._spin_dist_tol.setRange(1.0, 100.0); self._spin_dist_tol.setValue(10.0)
        self._spin_dist_tol.setSuffix(" px")
        hf.addRow("Cluster distance tolerance:", self._spin_dist_tol)
        lbl_h = QLabel(
            "Vote threshold: min edge pixels on a line to register it.\n"
            "Min length: lines shorter than this are discarded.\n"
            "Max gap: maximum gap between collinear segments to merge.")
        lbl_h.setObjectName("dim"); lbl_h.setWordWrap(True); hf.addRow(lbl_h)
        lay.addWidget(grp_hough)

        grp_val = QGroupBox("Trail validation")
        vf = QFormLayout(grp_val)
        self._spin_brightness = QDoubleSpinBox()
        self._spin_brightness.setRange(1.1, 5.0); self._spin_brightness.setValue(1.5)
        self._spin_brightness.setSingleStep(0.1)
        vf.addRow("Brightness factor:", self._spin_brightness)
        lbl_v = QLabel(
            "Trail must be brighter than adjacent background by this factor.\n"
            "Increase if spurious detections on star spikes or diffraction rings.")
        lbl_v.setObjectName("dim"); lbl_v.setWordWrap(True); vf.addRow(lbl_v)
        lay.addWidget(grp_val)

        grp_trail = QGroupBox("Trail mask")
        tmf = QFormLayout(grp_trail)
        self._spin_trail_width = QSpinBox()
        self._spin_trail_width.setRange(1, 100); self._spin_trail_width.setValue(5)
        self._spin_trail_width.setSuffix(" px")
        tmf.addRow("Trail width:", self._spin_trail_width)
        self._spin_dilation = QSpinBox()
        self._spin_dilation.setRange(0, 50); self._spin_dilation.setValue(3)
        self._spin_dilation.setSuffix(" px")
        tmf.addRow("Dilation:", self._spin_dilation)
        lbl_t = QLabel(
            "Trail width: total mask width covering the trail.\n"
            "Dilation: extra pixels added around the mask edge to "
            "cover trail wings and PSF halo.")
        lbl_t.setObjectName("dim"); lbl_t.setWordWrap(True); tmf.addRow(lbl_t)
        lay.addWidget(grp_trail)

        # Detected trails table
        grp_trails = QGroupBox("Detected trails (populated after detection run)")
        tl = QVBoxLayout(grp_trails)
        self._trail_table = TrailTable()
        tl.addWidget(self._trail_table)
        lay.addWidget(grp_trails)
        lay.addStretch()
        return w

    def _build_tab_inpaint(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        grp_algo = QGroupBox("Inpainting algorithm")
        al = QVBoxLayout(grp_algo)
        self._rb_diffusion = QRadioButton(
            "Median Diffusion  (Bertalmío et al. 2000, ACM SIGGRAPH)\n"
            "Fast. Propagates background inward from mask edges.\n"
            "Best for: thin trails in uniform star fields, quick batch processing.")
        self._rb_criminisi = QRadioButton(
            "Criminisi Exemplar-Based  (Criminisi, Pérez & Toyama 2004, IEEE TIP)\n"
            "Finds the best-matching patch from the surrounding image\n"
            "and fills the trail, preserving texture, gradients, and structure.\n"
            "Best for: trails crossing nebulosity, gradients, complex backgrounds.\n"
            "Slower — recommended for post-stack or critical images.")
        self._rb_diffusion.setChecked(True)
        self._bg_algo = QButtonGroup()
        self._bg_algo.addButton(self._rb_diffusion)
        self._bg_algo.addButton(self._rb_criminisi)
        self._rb_diffusion.toggled.connect(self._on_algo_changed)
        for rb in [self._rb_diffusion, self._rb_criminisi]:
            al.addWidget(rb)
        lay.addWidget(grp_algo)

        # Diffusion params
        self._grp_diff = QGroupBox("Diffusion parameters")
        df = QFormLayout(self._grp_diff)
        self._spin_diff_iter = QSpinBox()
        self._spin_diff_iter.setRange(5, 500); self._spin_diff_iter.setValue(50)
        df.addRow("Iterations:", self._spin_diff_iter)
        self._spin_diff_radius = QSpinBox()
        self._spin_diff_radius.setRange(1, 30); self._spin_diff_radius.setValue(5)
        self._spin_diff_radius.setSuffix(" px")
        df.addRow("Neighbour radius:", self._spin_diff_radius)
        lay.addWidget(self._grp_diff)

        # Criminisi params
        self._grp_crim = QGroupBox("Criminisi parameters")
        cf = QFormLayout(self._grp_crim)
        self._spin_patch = QSpinBox()
        self._spin_patch.setRange(3, 31); self._spin_patch.setValue(9)
        self._spin_patch.setSuffix(" px (odd)")
        cf.addRow("Patch size:", self._spin_patch)
        lbl_p = QLabel(
            "Larger patches find better texture matches but are slower.\n"
            "Recommended: 7–15 px for typical trail widths of 5–20 px.")
        lbl_p.setObjectName("dim"); lbl_p.setWordWrap(True); cf.addRow(lbl_p)
        self._grp_crim.setVisible(False)
        lay.addWidget(self._grp_crim)
        lay.addStretch()
        return w

    def _build_tab_preview(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self._canvas = TrailCanvas()
        self._canvas.setMinimumHeight(350)
        lay.addWidget(self._canvas)
        lbl_hint = QLabel(
            "Detection: orange = detected trail mask  ·  "
            "Lines = Hough transform results  ·  "
            "Before/After: luminance channel")
        lbl_hint.setObjectName("dim")
        lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl_hint)
        return w

    def _build_tab_output(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        grp_out = QGroupBox("Output folder")
        of = QHBoxLayout(grp_out)
        self._edit_out = QLineEdit(); self._edit_out.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(80)
        btn_out.clicked.connect(lambda: self._edit_out.setText(
            QFileDialog.getExistingDirectory(self, "Output folder") or
            self._edit_out.text()))
        of.addWidget(self._edit_out); of.addWidget(btn_out)
        lay.addWidget(grp_out)

        grp_ref = QGroupBox("References")
        rl = QVBoxLayout(grp_ref)
        refs = [
            "ASTA pipeline:   Groot et al. 2024, A&A 692, A111  (arXiv:2407.19461)",
            "Hough transform: Duda & Hart 1972, CACM 15(1), 11",
            "PHT:             Galamhos, Matas & Kittler 1999, CVPR",
            "Criminisi:       Criminisi, Pérez & Toyama 2004, IEEE TIP 13(9), 1200",
            "Diffusion:       Bertalmío, Sapiro, Caselles & Ballester 2000, ACM SIGGRAPH",
            "Starlink impact: McDowell 2020, ApJ 892, L36",
        ]
        for r in refs:
            lbl = QLabel(r); lbl.setObjectName("dim"); rl.addWidget(lbl)
        lay.addWidget(grp_ref)
        lay.addStretch()
        return w

    def _build_tab_log(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._log.clear)
        lay.addWidget(btn_clr); lay.addWidget(self._log)
        return w

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        self._grp_single.setVisible(self._rb_single.isChecked())
        self._grp_batch.setVisible(self._rb_batch.isChecked())

    def _on_algo_changed(self):
        crim = self._rb_criminisi.isChecked()
        self._grp_crim.setVisible(crim)
        self._grp_diff.setVisible(not crim)

    def _browse_single(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FITS", "", "FITS (*.fit *.fits *.fts);;All (*)")
        if path:
            self._edit_single.setText(path)
            if not self._edit_out.text():
                self._edit_out.setText(os.path.dirname(path))

    def _browse_batch(self):
        d = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if d:
            self._edit_batch.setText(d)
            if not self._edit_out.text():
                self._edit_out.setText(d)

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _get_config(self, detect_only: bool = False) -> dict | None:
        out = self._edit_out.text().strip()
        if not out:
            QMessageBox.warning(self, "No output", "Set an output folder.")
            return None

        inpainter = ("Criminisi (exemplar)" if self._rb_criminisi.isChecked()
                     else "Median diffusion")
        base = {
            "inpainter":       inpainter,
            "output_dir":      out,
            "detect_only":     detect_only,
            "bg_grid":         self._spin_bg_grid.value(),
            "canny_sigma":     self._spin_canny_sigma.value(),
            "low_thresh":      self._spin_low_thresh.value(),
            "high_thresh":     self._spin_high_thresh.value(),
            "hough_threshold": self._spin_hough_thresh.value(),
            "min_line_length": self._spin_min_length.value(),
            "max_line_gap":    self._spin_max_gap.value(),
            "angle_tol":       self._spin_angle_tol.value(),
            "dist_tol":        self._spin_dist_tol.value(),
            "brightness_factor": self._spin_brightness.value(),
            "trail_width":     self._spin_trail_width.value(),
            "dilation_px":     self._spin_dilation.value(),
            "diffusion_iter":  self._spin_diff_iter.value(),
            "diffusion_radius":self._spin_diff_radius.value(),
            "patch_size":      self._spin_patch.value() | 1,  # ensure odd
        }

        if self._rb_single.isChecked():
            path = self._edit_single.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "No file",
                    "Select a valid FITS file."); return None
            return {**base, "mode": "single", "input_path": path}
        else:
            folder = self._edit_batch.text().strip()
            if not folder or not os.path.isdir(folder):
                QMessageBox.warning(self, "No folder",
                    "Select a valid FITS folder."); return None
            return {**base, "mode": "batch", "input_folder": folder}

    def _run(self, detect_only: bool = False):
        cfg = self._get_config(detect_only)
        if cfg is None: return
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._btn_detect.setEnabled(False); self._progress.setValue(0)
        self._set_status("Running…", SIRIL_TRAIL)
        self._worker = TrailWorker(cfg, self._cancel_event,
                                   self._manual_mask)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.trails_found.connect(self._on_trails_found)
        self._worker.mask_ready.connect(self._on_mask_ready)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._tabs.setCurrentIndex(5)  # log

    def _cancel(self):
        self._cancel_event.set()
        self._btn_cancel.setEnabled(False)
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(msg, SIRIL_TRAIL)

    def _on_trails_found(self, trails: list, det_img: np.ndarray):
        self._detected_trails = trails
        self._trail_table.populate(trails)
        self._canvas.show_detection(det_img, trails, self._current_mask)
        n = len(trails)
        self._log.appendPlainText(
            f"  → {n} trail{'s' if n!=1 else ''} detected")

    def _on_mask_ready(self, mask: np.ndarray):
        self._current_mask = mask
        if self._detected_trails:
            # Refresh canvas with mask
            pass  # canvas already updated in trails_found with mask

    def _on_result_ready(self, before: np.ndarray, after: np.ndarray):
        self._canvas.show_before_after(before, after)
        self._tabs.setCurrentIndex(3)  # preview

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_detect.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        if result.get("success"):
            n = result.get("n_trails", 0)
            nm = result.get("n_masked", 0)
            out = result.get("output_path") or result.get("output_dir", "")
            msg = (f"✓  {n} trail{'s' if n!=1 else ''} removed  "
                   f"({nm:,} pixels inpainted)  →  {os.path.basename(out)}")
            self._set_status(msg, SIRIL_SUCCESS)
        else:
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status.setText(msg)
        self._status.setStyleSheet(f"color:{color}; font-size:9pt;")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import traceback as _tb
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "crash_log.txt")
    def crash_handler(exc_type, exc_value, exc_tb):
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\nCRASH  {datetime.now()}\n{msg}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = crash_handler

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
