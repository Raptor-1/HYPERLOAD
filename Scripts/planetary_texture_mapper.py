r"""
HYPERLOAD — Planetary Texture Mapper
======================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Build complete cylindrical surface maps of planets from derotated stacks,
then compare maps from different dates to detect surface changes.

Designed to work with planet_derotation.py output (FITS stacks),
or any planetary image with a known central meridian longitude.

Algorithms
----------
  Limb darkening correction: quadratic law (Claret & Bloemen 2011)
      I(μ) = 1 - u₁(1−μ) - u₂(1−μ)²   μ = cos(centre-limb angle)
      Divide observed I(x,y) by I(μ) to recover surface albedo.

  Cylindrical projection (equirectangular):
      For each map pixel (lat, lon):
        - Project to planet-frame 3D coords
        - Warp to image coords accounting for oblateness
        - Sample using bilinear interpolation
      Weight by μ² (cosine² of illumination angle) so disk-centre
      pixels (cleanest seeing) dominate the composite.

  Multi-session compositing:
      Weighted mean of N sessions at different central meridian
      longitudes → full 360° coverage (typically 3 sessions covers ~85%).
      Sessions from same night and same rotation give 360° in ~10 hours.

  Surface comparison (change detection):
      D = Map₂ − Map₁  (with optional longitude drift correction)
      Smooth D with σ_s = 2 px Gaussian
      Flag |D| > τ·std(D) as significant change
      Ratio map R = Map₂/Map₁ shows fractional brightness change.

  Longitude drift correction:
      Apply integer-pixel longitude shift to Map₂ before differencing.
      Accounts for known feature drift rates (e.g. GRS: ~−0.3°/day).

Planet presets (u₁, u₂, oblateness, System III rotation rate °/day):
  Jupiter  0.35 0.20  6.5%   870.27°/d  (System III ~9h 55m 30s)
  Saturn   0.40 0.25  9.8%   808.80°/d  (System III ~10h 39m 23s)
  Mars     0.15 0.10  0.6%   350.89°/d  (sidereal)
  Custom   user-defined

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, math, traceback, json
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log,"a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nplanetary_texture_mapper.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.ndimage import (map_coordinates, gaussian_filter,
                            center_of_mass, sobel)
from scipy.optimize import minimize as sp_minimize

try:
    from astropy.io import fits as astropy_fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

try:
    from astroquery.jplhorizons import Horizons
    HAS_HORIZONS = True
except ImportError:
    HAS_HORIZONS = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSlider, QLineEdit, QFormLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon, QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Planetary Texture Mapper"

# ── Planet presets: name → (u1, u2, oblateness, rot_rate_deg_per_day, horizons_id)
PLANET_PRESETS = {
    "Jupiter": (0.35, 0.20, 0.0648, 870.27, "599"),
    "Saturn":  (0.40, 0.25, 0.0980, 808.80, "699"),
    "Mars":    (0.15, 0.10, 0.0059, 350.89, "499"),
    "Uranus":  (0.30, 0.18, 0.0229, -501.16,"799"),
    "Neptune": (0.35, 0.22, 0.0171, 536.31, "899"),
    "Custom":  (0.30, 0.20, 0.065,  870.0,  "599"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

def luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2: return data.astype(float)
    if data.ndim == 3:
        if data.shape[0] <= 4:
            return (0.299*data[0]+0.587*data[1]+0.114*data[2]).astype(float)
        return (0.299*data[:,:,0]+0.587*data[:,:,1]+0.114*data[:,:,2]).astype(float)
    return data.astype(float)


def find_planet_center_radius(image: np.ndarray) -> tuple[float, float, float]:
    """Auto-detect planet centre and equatorial radius from disk image."""
    mn, mx = float(image.min()), float(image.max())
    norm   = (image - mn) / (mx - mn + 1e-10)
    binary = (norm > 0.4).astype(float)
    if binary.sum() < 10:
        h, w = image.shape
        return w/2, h/2, min(h,w)/4

    cy, cx = center_of_mass(binary)

    # Estimate radius: fit circle to limb via Sobel edges
    gx = sobel(norm, axis=1); gy = sobel(norm, axis=0)
    edges = np.sqrt(gx**2 + gy**2)
    thr   = np.percentile(edges, 92)
    ys, xs = np.where(edges > thr)
    if len(xs) < 12:
        dists_to_centre = np.sqrt((ys-cy)**2+(xs-cx)**2)
        return float(cx), float(cy), float(np.median(dists_to_centre))

    def circle_err(p):
        cy_, cx_, r_ = p
        d = np.sqrt((ys-cy_)**2 + (xs-cx_)**2)
        return float(np.sum((d-r_)**2))

    r0 = float(np.median(np.sqrt((ys-cy)**2+(xs-cx)**2)))
    res = sp_minimize(circle_err, [cy, cx, r0], method='Nelder-Mead',
                      options={'maxiter':2000,'xatol':0.5,'fatol':1})
    cy_f, cx_f, r_f = res.x
    return float(cx_f), float(cy_f), float(abs(r_f))


def ld_model(mu: np.ndarray, u1: float, u2: float) -> np.ndarray:
    """Quadratic limb darkening law."""
    return np.maximum(1.0 - u1*(1.0-mu) - u2*(1.0-mu)**2, 0.05)


def apply_ld_correction(image: np.ndarray, cx: float, cy: float,
                         R_eq: float, u1: float, u2: float,
                         oblateness: float = 0.065) -> np.ndarray:
    """Divide out limb darkening → intrinsic surface brightness."""
    h, w = image.shape
    R_pol = R_eq * (1.0 - oblateness)
    yy, xx = np.ogrid[:h, :w]
    dx = (xx - cx) / R_eq; dy = (yy - cy) / R_pol
    r2 = dx**2 + dy**2
    mu_im = np.sqrt(np.maximum(1.0 - r2, 0.0))
    ld    = ld_model(mu_im, u1, u2)
    return np.where(r2 <= 0.97**2, image.astype(float) / ld, 0.0)


def disk_to_map(image: np.ndarray, cx: float, cy: float,
                R_eq: float, lon_center_deg: float,
                u1: float = 0.35, u2: float = 0.20,
                oblateness: float = 0.065,
                map_w: int = 720, map_h: int = 360,
                do_ld_correction: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a planetary disk image to a cylindrical equirectangular map.

    Parameters
    ----------
    image           : 2-D luminance image (already centred on planet)
    cx, cy          : planet centre in pixels
    R_eq            : equatorial radius in pixels
    lon_center_deg  : System III longitude of the central meridian (°)
    u1, u2          : quadratic limb darkening coefficients
    oblateness      : equatorial – polar radius / equatorial (e.g. 0.065 for Jupiter)
    map_w, map_h    : output map dimensions in pixels
    do_ld_correction: apply limb darkening correction before mapping

    Returns
    -------
    proj    : (map_h, map_w) weighted sum
    weights : (map_h, map_w) weight map (quality per pixel)
    """
    h, w   = image.shape
    R_pol  = R_eq * (1.0 - oblateness)

    # Limb darkening correction
    if do_ld_correction:
        img_c = apply_ld_correction(image, cx, cy, R_eq, u1, u2, oblateness)
    else:
        img_c = image.astype(float)

    # Map coordinate grids
    lat_px  = np.linspace(np.pi/2, -np.pi/2, map_h)    # +90 → -90 degrees
    lon_vis = np.linspace(-np.pi/2, np.pi/2, map_w)     # -90 → +90 relative to CM

    lon_g, lat_g = np.meshgrid(lon_vis, lat_px)

    # 3-D planet-frame unit sphere
    z_p = np.cos(lat_g) * np.cos(lon_g)   # front hemisphere  (>0 = visible)
    x_p = np.cos(lat_g) * np.sin(lon_g)   # east-west on disk
    y_p = np.sin(lat_g)                     # north-south on disk

    # Image-plane pixel coordinates (with oblateness)
    px_d = cx + R_eq  * x_p
    py_d = cy - R_pol * y_p    # flip y: image y increases downward

    # Visibility: front hemisphere, within disc boundary
    visible = (z_p > 0.01) & ((x_p**2 + y_p**2) < 0.97**2)

    # Weight = μ² where μ = z_p (cosine of illumination angle)
    w_arr = np.where(visible, z_p**2, 0.0)

    # Bilinear interpolation
    sampled = map_coordinates(
        img_c,
        np.array([py_d.ravel(), px_d.ravel()]),
        order=1, mode='constant', cval=0.0
    ).reshape(map_h, map_w)

    proj    = sampled * w_arr
    weights = w_arr

    # Shift columns by sub-observer longitude
    shift = int(round(lon_center_deg / 360.0 * map_w)) % map_w
    proj    = np.roll(proj,    shift, axis=1)
    weights = np.roll(weights, shift, axis=1)

    return proj, weights


def composite_sessions(sessions: list[tuple[np.ndarray, np.ndarray]]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted average of N (proj, weight) pairs.
    Returns (composite_map, total_weight).
    """
    total_p = np.zeros_like(sessions[0][0])
    total_w = np.zeros_like(sessions[0][1])
    for p, w in sessions:
        total_p += p; total_w += w
    composite = np.where(total_w > 0.01, total_p / total_w, np.nan)
    return composite, total_w


def compare_maps(map1: np.ndarray, map2: np.ndarray,
                 w1: np.ndarray, w2: np.ndarray,
                 lon_drift_deg: float = 0.0,
                 smooth_sigma: float = 1.5,
                 sigma_threshold: float = 2.5,
                 map_w: int = 720) -> dict:
    """
    Compare two composite maps to detect surface changes.

    Parameters
    ----------
    map1, map2       : composite maps from composite_sessions()
    w1, w2           : corresponding weight maps
    lon_drift_deg    : longitude drift of map2 relative to map1 (degrees).
                       Positive = map2 features shifted east. Applied as
                       a column roll before differencing.
    smooth_sigma     : Gaussian sigma for smoothing the difference (pixels)
    sigma_threshold  : |diff| > threshold × std(diff) → flagged as change
    map_w            : map width in pixels (for drift roll conversion)

    Returns
    -------
    dict with keys: diff, diff_smooth, ratio, changes, valid,
                    sigma, n_changed, pct_changed, max_abs_diff
    """
    # Apply longitude drift correction to map2
    if abs(lon_drift_deg) > 0.01:
        shift = int(round(lon_drift_deg / 360.0 * map_w)) % map_w
        map2  = np.roll(map2, shift, axis=1)
        w2    = np.roll(w2,   shift, axis=1)

    valid = (~np.isnan(map1)) & (~np.isnan(map2)) & (w1 > 0.01) & (w2 > 0.01)
    diff  = np.where(valid, map2 - map1, np.nan)
    ratio = np.where(valid & (np.abs(map1) > 0.02),
                     map2 / (map1 + 1e-10), np.nan)

    # Smooth to reduce shot noise
    diff_sm = gaussian_filter(np.nan_to_num(diff, 0.0), smooth_sigma)
    diff_sm = np.where(valid, diff_sm, np.nan)

    valid_vals = diff[valid]
    sigma = float(np.nanstd(valid_vals)) if len(valid_vals) > 10 else 1.0
    changes = np.where(valid, np.abs(diff_sm) > sigma_threshold * sigma, False)

    n_total  = int(valid.sum())
    n_change = int(changes.sum())

    return {
        'diff':          diff,
        'diff_smooth':   diff_sm,
        'ratio':         ratio,
        'changes':       changes,
        'valid':         valid,
        'sigma':         sigma,
        'n_changed':     n_change,
        'n_total':       n_total,
        'pct_changed':   float(n_change / n_total * 100) if n_total > 0 else 0.0,
        'max_abs_diff':  float(np.nanmax(np.abs(diff))) if valid.any() else 0.0,
        'mean_abs_diff': float(np.nanmean(np.abs(diff[valid]))) if valid.any() else 0.0,
    }


def query_horizons_cml(planet_id: str, utc_str: str,
                        obs_location: str = '500') -> float | None:
    """
    Query JPL Horizons for the System III Central Meridian Longitude.
    Returns CML in degrees, or None if unavailable.
    """
    if not HAS_HORIZONS: return None
    try:
        from astropy.time import Time
        jd = float(Time(utc_str, format='isot').jd)
        obj = Horizons(id=planet_id, location=obs_location,
                       epochs={'start': utc_str, 'stop': utc_str,
                               'step': '1m'})
        eph = obj.ephemerides(quantities='14,15')
        return float(eph['PDObsLon'][0])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Session:
    """One imaging session: a stacked planetary image + metadata."""
    def __init__(self, image: np.ndarray, label: str,
                 lon_center_deg: float, utc: str = ''):
        self.image         = image          # 2-D float luminance
        self.label         = label
        self.lon_center    = lon_center_deg
        self.utc           = utc
        self.cx = self.cy  = None
        self.R_eq          = None
        self.proj          = None           # cached projection
        self.weights       = None

    def detect_geometry(self):
        self.cx, self.cy, self.R_eq = find_planet_center_radius(self.image)

    def project(self, u1, u2, oblateness, map_w, map_h, do_ld):
        if self.cx is None: self.detect_geometry()
        self.proj, self.weights = disk_to_map(
            self.image, self.cx, self.cy, self.R_eq,
            self.lon_center, u1, u2, oblateness, map_w, map_h, do_ld)
        return self.proj, self.weights


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class MapWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, sessions_a, sessions_b, params):
        super().__init__()
        self.sessions_a = sessions_a   # list of Session (epoch A)
        self.sessions_b = sessions_b   # list of Session (epoch B) or []
        self.params     = params

    def run(self):
        try:
            p    = self.params
            u1   = p['u1']; u2 = p['u2']; obl = p['oblateness']
            mw   = p['map_w']; mh = p['map_h']
            do_ld= p['do_ld']
            n    = len(self.sessions_a) + len(self.sessions_b)
            done = 0

            # ── Project epoch A ───────────────────────────────────────────
            self.progress.emit(5, f"Projecting {len(self.sessions_a)} epoch-A sessions…")
            projs_a = []
            for i, s in enumerate(self.sessions_a):
                pr, wt = s.project(u1, u2, obl, mw, mh, do_ld)
                projs_a.append((pr, wt))
                done += 1
                self.progress.emit(5 + int(40*done/n), f"Projected A{i+1}/{len(self.sessions_a)}")

            map_a, w_a = composite_sessions(projs_a)

            # ── Project epoch B (optional) ─────────────────────────────────
            map_b = w_b = None
            comp_result = None

            if self.sessions_b:
                self.progress.emit(50, f"Projecting {len(self.sessions_b)} epoch-B sessions…")
                projs_b = []
                for i, s in enumerate(self.sessions_b):
                    pr, wt = s.project(u1, u2, obl, mw, mh, do_ld)
                    projs_b.append((pr, wt))
                    done += 1
                    self.progress.emit(50+int(35*done/n), f"Projected B{i+1}/{len(self.sessions_b)}")

                map_b, w_b = composite_sessions(projs_b)

                # ── Compare ───────────────────────────────────────────────
                self.progress.emit(88, "Comparing maps…")
                comp_result = compare_maps(
                    map_a, map_b, w_a, w_b,
                    lon_drift_deg=p['lon_drift'],
                    smooth_sigma=p['smooth_sigma'],
                    sigma_threshold=p['sigma_thresh'],
                    map_w=mw)

            cov_a = float((~np.isnan(map_a)).mean()) * 100
            self.progress.emit(100,
                f"Done!  Coverage A={cov_a:.1f}%"
                + (f"  Changed={comp_result['pct_changed']:.1f}%" if comp_result else ''))

            self.finished.emit({
                'map_a': map_a, 'w_a': w_a,
                'map_b': map_b, 'w_b': w_b,
                'comp':  comp_result,
                'cov_a': cov_a,
                'cov_b': float((~np.isnan(map_b)).mean())*100 if map_b is not None else 0,
            })

        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

SIRIL_BG  = "#0c1018"; SIRIL_BG2 = "#111828"; SIRIL_BG3 = "#1a2438"
SIRIL_ACC = "#4080e0"; SIRIL_ACC2= "#2858b0"
SIRIL_TXT = "#c8d8f0"; SIRIL_DIM = "#303850"; SIRIL_BOR = "#1e2d4a"
SIRIL_OK  = "#40b080"; SIRIL_WRN = "#e09030"; SIRIL_ERR = "#c04040"
SIRIL_SEC = "#5898e0"

DARK = f"""
QMainWindow,QWidget{{background:{SIRIL_BG};color:{SIRIL_TXT};
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}}
QTabWidget::pane{{border:1px solid {SIRIL_BOR};background:{SIRIL_BG2};}}
QTabBar::tab{{background:{SIRIL_BG3};color:{SIRIL_DIM};padding:6px 16px;
    border:1px solid {SIRIL_BOR};border-bottom:none;}}
QTabBar::tab:selected{{background:{SIRIL_BG2};color:{SIRIL_ACC};
    border-bottom:2px solid {SIRIL_ACC};}}
QGroupBox{{border:1px solid {SIRIL_BOR};border-radius:4px;margin-top:8px;
    padding-top:8px;color:{SIRIL_DIM};font-weight:bold;}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}
QPushButton{{background:{SIRIL_BG3};color:{SIRIL_TXT};border:1px solid {SIRIL_BOR};
    border-radius:4px;padding:5px 14px;}}
QPushButton:hover{{background:{SIRIL_ACC2};border-color:{SIRIL_ACC};color:white;}}
QPushButton#run{{background:{SIRIL_ACC2};border-color:{SIRIL_ACC};
    color:white;font-weight:bold;padding:7px 20px;}}
QPushButton#run:hover{{background:{SIRIL_ACC};}}
QPushButton#add{{color:{SIRIL_OK};border-color:{SIRIL_OK};}}
QPushButton#remove{{color:{SIRIL_ERR};border-color:{SIRIL_ERR};}}
QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{{background:{SIRIL_BG3};
    border:1px solid {SIRIL_BOR};border-radius:3px;padding:3px 6px;color:{SIRIL_TXT};}}
QTextEdit{{background:#080c14;border:1px solid {SIRIL_BOR};color:{SIRIL_OK};
    font-family:Consolas,monospace;font-size:10px;}}
QTableWidget{{background:#080c14;alternate-background-color:{SIRIL_BG3};
    gridline-color:{SIRIL_BOR};color:{SIRIL_TXT};}}
QTableWidget QHeaderView::section{{background:{SIRIL_BG3};color:{SIRIL_SEC};
    border:1px solid {SIRIL_BOR};padding:3px;font-weight:bold;}}
QSplitter::handle{{background:{SIRIL_BOR};}}
QSlider::groove:horizontal{{background:{SIRIL_BG3};height:6px;border-radius:3px;}}
QSlider::handle:horizontal{{background:{SIRIL_ACC};width:14px;height:14px;
    border-radius:7px;margin:-4px 0;}}
QProgressBar{{background:{SIRIL_BG3};border:1px solid {SIRIL_BOR};
    border-radius:3px;height:8px;}}
QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {SIRIL_ACC2},stop:1 {SIRIL_ACC});border-radius:2px;}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS
# ═══════════════════════════════════════════════════════════════════════════════

class MplCanvas(QWidget):
    clicked = pyqtSignal(float, float)   # lon, lat of click on map

    def __init__(self, parent=None, figsize=(8,4)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor=SIRIL_BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self._onclick)
        self._map_extent = None

    def _onclick(self, event):
        if event.inaxes and self._map_extent is not None:
            l0, l1, la0, la1 = self._map_extent
            lon = l0 + (event.xdata/(self.fig.get_axes()[0].get_xlim()[1]) if False
                        else event.xdata)
            self.clicked.emit(float(event.xdata), float(event.ydata))

    def redraw(self): self.fig.tight_layout(pad=0.3); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#080c14')
    ax.set_title(title, color=SIRIL_SEC, fontsize=9, pad=3)
    ax.set_xlabel(xl, color=SIRIL_DIM, fontsize=8)
    ax.set_ylabel(yl, color=SIRIL_DIM, fontsize=8)
    ax.tick_params(colors=SIRIL_DIM, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(SIRIL_BOR)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION TABLE WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class SessionTable(QWidget):
    sessions_changed = pyqtSignal()

    def __init__(self, label="Epoch A", parent=None):
        super().__init__(parent)
        self.sessions = []
        self.label    = label
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(3)
        hdr = QLabel(f"Sessions — {self.label}")
        hdr.setStyleSheet(f"color:{SIRIL_SEC};font-weight:bold;")
        lay.addWidget(hdr)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File","CML (°)","UTC","R (px)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setFixedHeight(140); self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table)

        btns = QHBoxLayout()
        b_add = QPushButton("＋  Add FITS/PNG…"); b_add.setObjectName("add")
        b_add.clicked.connect(self._add)
        b_rem = QPushButton("－  Remove"); b_rem.setObjectName("remove")
        b_rem.clicked.connect(self._remove)
        b_ex  = QPushButton("⚗  Synthetic example"); b_ex.clicked.connect(self._example)
        btns.addWidget(b_add); btns.addWidget(b_rem); btns.addWidget(b_ex)
        lay.addLayout(btns)

        # CML entry for selected row
        cml_row = QHBoxLayout()
        cml_row.addWidget(QLabel("CML override:"))
        self.cml_spin = QDoubleSpinBox()
        self.cml_spin.setRange(0,360); self.cml_spin.setDecimals(2)
        self.cml_spin.valueChanged.connect(self._update_cml)
        b_horiz = QPushButton("🔭 Horizons")
        b_horiz.clicked.connect(self._query_horizons)
        cml_row.addWidget(self.cml_spin); cml_row.addWidget(b_horiz)
        lay.addLayout(cml_row)

    def _add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Load sessions for {self.label}", "",
            "FITS / PNG (*.fits *.fit *.fts *.png *.tif);;All (*)")
        if not paths: return
        for p in sorted(paths):
            self._load_image(p)

    def _load_image(self, path):
        try:
            ext = Path(path).suffix.lower()
            if ext in ('.fits','.fit','.fts') and ASTROPY_OK:
                data = astropy_fits.getdata(path)
            else:
                import imageio; data = np.array(imageio.imread(path))
            img = luminance(data)
            # Ask for CML
            cml = float(self.cml_spin.value())
            s = Session(img, Path(path).name, cml)
            s.detect_geometry()
            self.sessions.append(s)
            self._refresh_table()
            self.sessions_changed.emit()
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _example(self):
        """Generate a synthetic Jupiter disk at a given CML."""
        rng = np.random.default_rng(len(self.sessions))
        N   = 256; R = 110.0; cx = cy = N//2
        yy, xx = np.ogrid[:N,:N]
        r_norm = np.sqrt((xx-cx)**2+(yy-cy)**2)/R
        mu     = np.sqrt(np.maximum(1-r_norm**2,0))
        ld     = ld_model(mu, 0.35, 0.20)
        lat    = np.degrees(np.arcsin(np.clip(-(yy-cy)/R,-1,1)))
        lon    = np.degrees(np.arctan2((xx-cx),
                    np.sqrt(np.maximum(R**2-(yy-cy)**2-(xx-cx)**2,0))))
        belt   = (1.0+0.3*np.cos(np.radians(lat*2))
                  -0.4*np.exp(-((lat-15)**2)/50)
                  -0.4*np.exp(-((lat+15)**2)/50)
                  +0.15*np.sin(np.radians(lon*3)))
        img    = np.clip(belt*ld*(r_norm<=1).astype(float)
                         +rng.normal(0,0.02,(N,N)),0,None)
        cml    = float(self.cml_spin.value()) if len(self.sessions)==0 \
                 else (self.sessions[-1].lon_center+120)%360
        s = Session(img, f"Synthetic_CML{cml:.0f}°", cml)
        s.cx = s.cy = float(N//2); s.R_eq = R
        self.sessions.append(s)
        self._refresh_table()
        self.sessions_changed.emit()

    def _remove(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.sessions):
            self.sessions.pop(row)
            self._refresh_table()
            self.sessions_changed.emit()

    def _update_cml(self, val):
        row = self.table.currentRow()
        if 0 <= row < len(self.sessions):
            self.sessions[row].lon_center = float(val)
            self._refresh_table()

    def _query_horizons(self):
        row = self.table.currentRow()
        if row < 0 or not self.sessions: return
        s = self.sessions[row]
        if not HAS_HORIZONS:
            QMessageBox.information(self,"Horizons","astroquery not installed.")
            return
        if not s.utc:
            QMessageBox.information(self,"Horizons",
                "No UTC time stored for this session.\nEnter CML manually.")
            return
        # This would need parent access to planet_id — just show a message
        QMessageBox.information(self,"Horizons",
            f"Query Horizons for {s.utc}:\n"
            f"Set planet in main controls, then use\n"
            f"query_horizons_cml(planet_id, '{s.utc}')")

    def _refresh_table(self):
        self.table.setRowCount(len(self.sessions))
        for i, s in enumerate(self.sessions):
            for j, v in enumerate([
                s.label[:30],
                f"{s.lon_center:.1f}",
                s.utc or "—",
                f"{s.R_eq:.0f}" if s.R_eq else "auto"
            ]):
                self.table.setItem(i, j, QTableWidgetItem(v))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class PlanetMapper(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 920)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._result  = None
        self._worker  = self._wthread = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._controls())
        sp.addWidget(self._tabs())
        sp.setSizes([310,1130])
        rl.addWidget(sp,1)
        rl.addWidget(self._footer())

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self):
        hdr = QWidget(); hdr.setFixedHeight(60)
        hdr.setStyleSheet(f"background:{SIRIL_BG};border-bottom:1px solid {SIRIL_BOR};")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12,0,12,0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel(); lbl.setPixmap(QPixmap(str(_logo)).scaled(
                44,44,Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Planetary Texture Mapper")
        t.setStyleSheet(f"color:{SIRIL_ACC};font-size:17px;font-weight:bold;padding:6px;")
        s = QLabel("Cylindrical projection  ·  Limb darkening correction  ·  "
                   "Multi-session compositing  ·  Surface change detection  ·  WinJUPOS replacement")
        s.setStyleSheet(f"color:{SIRIL_DIM};font-size:9pt;padding:0 8px 4px;")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    # ── Left control panel ────────────────────────────────────────────────────

    def _controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(5)

        # Planet preset
        grp = QGroupBox("Planet"); g = QFormLayout(grp)
        self.cb_planet = QComboBox()
        self.cb_planet.addItems(list(PLANET_PRESETS.keys()))
        self.cb_planet.currentTextChanged.connect(self._on_planet_changed)
        g.addRow("Preset:", self.cb_planet)
        self.sp_u1 = QDoubleSpinBox(); self.sp_u1.setRange(0,1); self.sp_u1.setDecimals(3); self.sp_u1.setSingleStep(0.01)
        self.sp_u2 = QDoubleSpinBox(); self.sp_u2.setRange(0,1); self.sp_u2.setDecimals(3); self.sp_u2.setSingleStep(0.01)
        self.sp_obl= QDoubleSpinBox(); self.sp_obl.setRange(0,0.2);self.sp_obl.setDecimals(4);self.sp_obl.setSingleStep(0.001)
        g.addRow("LD u₁:", self.sp_u1); g.addRow("LD u₂:", self.sp_u2)
        g.addRow("Oblateness:", self.sp_obl)
        self.cb_do_ld = QCheckBox("Apply LD correction"); self.cb_do_ld.setChecked(True)
        g.addRow(self.cb_do_ld)
        lay.addWidget(grp)

        # Sessions
        self.sess_a = SessionTable("Epoch A (main map)")
        self.sess_b = SessionTable("Epoch B (comparison)")
        lay.addWidget(self.sess_a); lay.addWidget(self.sess_b)

        # Map parameters
        grp2 = QGroupBox("Map Parameters"); g2 = QFormLayout(grp2)
        self.sp_mw = QSpinBox(); self.sp_mw.setRange(180,1440); self.sp_mw.setValue(720); self.sp_mw.setSingleStep(180)
        self.sp_mh = QSpinBox(); self.sp_mh.setRange(90,720);  self.sp_mh.setValue(360); self.sp_mh.setSingleStep(90)
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(['gray','inferno','hot','plasma','viridis','RdBu_r'])
        g2.addRow("Width (px):", self.sp_mw); g2.addRow("Height (px):", self.sp_mh)
        g2.addRow("Colourmap:", self.cb_cmap)
        lay.addWidget(grp2)

        # Comparison parameters
        grp3 = QGroupBox("Comparison"); g3 = QFormLayout(grp3)
        self.sp_drift  = QDoubleSpinBox(); self.sp_drift.setRange(-180,180); self.sp_drift.setDecimals(1); self.sp_drift.setSingleStep(1.0)
        self.sp_smooth = QDoubleSpinBox(); self.sp_smooth.setRange(0.5,10); self.sp_smooth.setValue(1.5); self.sp_smooth.setDecimals(1)
        self.sp_sigma  = QDoubleSpinBox(); self.sp_sigma.setRange(1.0,10.0);self.sp_sigma.setValue(2.5);self.sp_sigma.setDecimals(1)
        g3.addRow("Lon drift (°):", self.sp_drift)
        g3.addRow("Smooth σ (px):", self.sp_smooth)
        g3.addRow("Change σ:", self.sp_sigma)
        lbl_note = QLabel("Drift: positive = map B shifted east\nbefore differencing")
        lbl_note.setStyleSheet(f"color:{SIRIL_DIM};font-size:9px;")
        g3.addRow(lbl_note)
        lay.addWidget(grp3)

        # Run
        self.btn_run = QPushButton("▶  Build Map")
        self.btn_run.setObjectName("run"); self.btn_run.setFixedHeight(38)
        self.btn_run.clicked.connect(self._run)
        self.btn_exp = QPushButton("💾  Export maps…")
        self.btn_exp.setEnabled(False); self.btn_exp.clicked.connect(self._export)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_exp)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(f"color:{SIRIL_OK};font-size:10px;")
        self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)

        self._on_planet_changed("Jupiter")
        scroll.setWidget(inner); return scroll

    def _on_planet_changed(self, name):
        u1,u2,obl,_,_ = PLANET_PRESETS.get(name, (0.30,0.20,0.065,870,"599"))
        self.sp_u1.setValue(u1); self.sp_u2.setValue(u2); self.sp_obl.setValue(obl)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    def _tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_disk(),   "🪐  Disk Preview")
        self.tabs.addTab(self._tab_map_a(),  "🗺  Epoch A Map")
        self.tabs.addTab(self._tab_compare(),"🔍  Surface Comparison")
        self.tabs.addTab(self._tab_profile(),"📈  Belt Profiles")
        self.tabs.addTab(self._tab_log(),    "📋  Log")
        return self.tabs

    def _tab_disk(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.disk_canvas = MplCanvas(figsize=(9,5)); lay.addWidget(self.disk_canvas)
        ctrl = QHBoxLayout()
        b_prev = QPushButton("◀  Preview epoch A session")
        b_prev.clicked.connect(self._preview_disk)
        ctrl.addWidget(b_prev); ctrl.addStretch()
        lay.addLayout(ctrl)
        return w

    def _tab_map_a(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.mapa_canvas = MplCanvas(figsize=(9,5)); lay.addWidget(self.mapa_canvas)
        return w

    def _tab_compare(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.comp_canvas = MplCanvas(figsize=(9,7)); lay.addWidget(self.comp_canvas)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Live drift:"))
        self.drift_live = QSlider(Qt.Orientation.Horizontal)
        self.drift_live.setRange(-180,180); self.drift_live.setValue(0)
        self.drift_live.setFixedWidth(200)
        self.drift_lbl = QLabel("0.0°")
        self.drift_lbl.setFixedWidth(40)
        self.drift_live.valueChanged.connect(self._on_drift_slider)
        ctrl.addWidget(self.drift_live); ctrl.addWidget(self.drift_lbl)
        self.cb_show_changes = QCheckBox("Show change overlay")
        self.cb_show_changes.setChecked(True)
        self.cb_show_changes.stateChanged.connect(self._redraw_compare)
        ctrl.addWidget(self.cb_show_changes); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.comp_stats = QLabel("")
        self.comp_stats.setStyleSheet(f"color:{SIRIL_WRN};font-size:10px;")
        lay.addWidget(self.comp_stats)
        return w

    def _tab_profile(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.prof_canvas = MplCanvas(figsize=(9,5)); lay.addWidget(self.prof_canvas)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Latitude band (°):"))
        self.sp_lat1 = QDoubleSpinBox(); self.sp_lat1.setRange(-90,90); self.sp_lat1.setValue(-30)
        self.sp_lat2 = QDoubleSpinBox(); self.sp_lat2.setRange(-90,90); self.sp_lat2.setValue( 30)
        b_prof = QPushButton("▶  Plot profile"); b_prof.clicked.connect(self._draw_profile)
        ctrl.addWidget(self.sp_lat1); ctrl.addWidget(QLabel("to")); ctrl.addWidget(self.sp_lat2)
        ctrl.addWidget(b_prof); ctrl.addStretch()
        lay.addLayout(ctrl)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True); lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear); lay.addWidget(btn)
        return w

    def _footer(self):
        foot = QWidget(); foot.setFixedHeight(22)
        foot.setStyleSheet(f"background:{SIRIL_BG};border-top:1px solid {SIRIL_BOR};")
        lay = QHBoxLayout(foot); lay.setContentsMargins(6,0,6,0)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet(f"color:{SIRIL_DIM};font-size:10px;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(200,12); self.pbar.setVisible(False)
        lay.addWidget(self.status_lbl,1); lay.addWidget(self.pbar)
        return foot

    # ── Preview ───────────────────────────────────────────────────────────────

    def _preview_disk(self):
        slist = self.sess_a.sessions
        if not slist: self._log("No epoch-A sessions. Add sessions first."); return
        s = slist[0]; s.detect_geometry()
        fig = self.disk_canvas.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)
        axes = fig.subplots(1,2)
        u1,u2,obl = self.sp_u1.value(),self.sp_u2.value(),self.sp_obl.value()
        corr = apply_ld_correction(s.image, s.cx, s.cy, s.R_eq, u1, u2, obl)
        lo,hi = np.nanpercentile(s.image[s.image>0],[1,99.5])
        _ax(axes[0],f"Original  CML={s.lon_center:.1f}°")
        axes[0].imshow(s.image, origin='lower', cmap='gray', vmin=lo, vmax=hi,
                       aspect='equal', interpolation='bicubic')
        axes[0].plot(s.cx, s.cy, '+', color='red', ms=14, mew=2)
        import matplotlib.patches as mpp
        axes[0].add_patch(mpp.Circle((s.cx,s.cy),s.R_eq,ec='yellow',fc='none',lw=1))
        _ax(axes[1],"LD-corrected (intrinsic surface brightness)")
        lo2,hi2 = np.nanpercentile(corr[corr>0],[1,99.5])
        axes[1].imshow(corr, origin='lower', cmap='gray', vmin=lo2, vmax=hi2,
                       aspect='equal', interpolation='bicubic')
        self.disk_canvas.redraw()
        self._log(f"Preview: {s.label}  cx={s.cx:.1f} cy={s.cy:.1f} R={s.R_eq:.1f}px")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        sa = self.sess_a.sessions; sb = self.sess_b.sessions
        if not sa:
            QMessageBox.warning(self,"No sessions","Add at least one epoch-A session."); return

        params = dict(
            u1=self.sp_u1.value(), u2=self.sp_u2.value(),
            oblateness=self.sp_obl.value(), do_ld=self.cb_do_ld.isChecked(),
            map_w=self.sp_mw.value(), map_h=self.sp_mh.value(),
            lon_drift=self.sp_drift.value(),
            smooth_sigma=self.sp_smooth.value(),
            sigma_thresh=self.sp_sigma.value(),
        )

        self.btn_run.setEnabled(False); self.btn_exp.setEnabled(False)
        self._wthread = QThread(self)
        self._worker  = MapWorker(sa, sb, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status_lbl.setText(msg)
        self.pbar.setVisible(True); self.pbar.setValue(pct)
        if pct >= 100: QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}"); self.btn_run.setEnabled(True)

    def _on_finished(self, result):
        self._result = result; self.btn_run.setEnabled(True)
        self.btn_exp.setEnabled(True)
        r = result
        self.summary_lbl.setText(
            f"Epoch A coverage: {r['cov_a']:.1f}%"
            + (f"\nEpoch B coverage: {r['cov_b']:.1f}%\n"
               f"Changed: {r['comp']['pct_changed']:.1f}%"
               if r['comp'] else ''))
        self._log(f"\n{'='*48}")
        self._log(f"✓  Map built   coverage_A={r['cov_a']:.1f}%")
        if r['comp']:
            c = r['comp']
            self._log(f"   Comparison:  σ={c['sigma']:.4f}")
            self._log(f"   Changed px:  {c['n_changed']} ({c['pct_changed']:.1f}%)")
            self._log(f"   Max |diff|:  {c['max_abs_diff']:.4f}")
        self._log(f"{'='*48}\n")

        self._draw_map_a()
        if r['comp']: self._redraw_compare()
        if r['comp']: self._draw_profile()
        self.tabs.setCurrentIndex(1)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_map_a(self):
        r = self._result; ma = r['map_a']
        if ma is None: return
        fig = self.mapa_canvas.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)
        ax = fig.add_subplot(111)
        _ax(ax, "Epoch A — composite cylindrical map",
            "System III Longitude (°)", "Latitude (°)")
        mw = ma.shape[1]; mh = ma.shape[0]
        lo,hi = np.nanpercentile(ma[~np.isnan(ma)],[1,99.5]) if (~np.isnan(ma)).any() else (0,1)
        im = ax.imshow(ma, origin='upper', cmap=self.cb_cmap.currentText(),
                       vmin=lo, vmax=hi, aspect='auto',
                       extent=[0,360,-90,90], interpolation='bicubic')
        ax.set_xticks(range(0,361,60)); ax.set_yticks(range(-90,91,30))
        ax.axhline(0,color=SIRIL_BOR,lw=0.5,ls='--')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                     label="Relative brightness").ax.tick_params(
                         colors=SIRIL_DIM, labelsize=7)
        # Mark session CMLs
        for s in self.sess_a.sessions:
            ax.axvline(s.lon_center, color=SIRIL_WRN, lw=0.8, alpha=0.5)
            ax.text(s.lon_center+1, 88, f"{s.lon_center:.0f}°",
                    color=SIRIL_WRN, fontsize=6, va='top')
        n_sess = len(self.sess_a.sessions)
        ax.text(0.02,0.02,f"{n_sess} session(s)  coverage={r['cov_a']:.1f}%",
                transform=ax.transAxes, color=SIRIL_DIM, fontsize=8)
        self.mapa_canvas.redraw()

    def _redraw_compare(self):
        r = self._result
        if not r or not r['comp']: return

        drift = float(self.drift_live.value())
        self.drift_lbl.setText(f"{drift:.0f}°")

        # Recompute comparison with live drift
        mw = self.sp_mw.value()
        c = compare_maps(
            r['map_a'], r['map_b'],
            r['w_a'],   r['w_b'],
            lon_drift_deg=drift,
            smooth_sigma=self.sp_smooth.value(),
            sigma_threshold=self.sp_sigma.value(),
            map_w=mw)

        cmap_main = self.cb_cmap.currentText()

        fig = self.comp_canvas.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)
        axes = fig.subplots(2,2)

        # ── Panel 1: Epoch A ──────────────────────────────────────────────
        ma = r['map_a']
        lo,hi = np.nanpercentile(ma[~np.isnan(ma)],[1,99.5]) if (~np.isnan(ma)).any() else (0,1)
        _ax(axes[0,0],f"Epoch A  (coverage={r['cov_a']:.1f}%)",
            "Longitude (°)","Latitude (°)")
        axes[0,0].imshow(ma,origin='upper',cmap=cmap_main,vmin=lo,vmax=hi,
                         aspect='auto',extent=[0,360,-90,90],interpolation='bicubic')
        axes[0,0].set_xticks(range(0,361,90)); axes[0,0].set_yticks([-60,-30,0,30,60])

        # ── Panel 2: Epoch B (drift-corrected) ────────────────────────────
        mb = r['map_b']
        if mb is not None and abs(drift) > 0.1:
            shift = int(round(drift/360*mw))%mw
            mb_show = np.roll(mb, shift, axis=1)
        else:
            mb_show = mb
        _ax(axes[0,1],f"Epoch B  (coverage={r['cov_b']:.1f}%)  drift={drift:.0f}°",
            "Longitude (°)","")
        axes[0,1].imshow(mb_show,origin='upper',cmap=cmap_main,vmin=lo,vmax=hi,
                         aspect='auto',extent=[0,360,-90,90],interpolation='bicubic')
        axes[0,1].set_xticks(range(0,361,90)); axes[0,1].set_yticks([])

        # ── Panel 3: Difference map ───────────────────────────────────────
        diff_sm = c['diff_smooth']
        vmax_d = max(np.nanstd(diff_sm[c['valid']])*3, 1e-5) if c['valid'].any() else 0.1
        _ax(axes[1,0],"Difference (B − A)  [red=brighter, blue=darker]",
            "Longitude (°)","Latitude (°)")
        im_d = axes[1,0].imshow(diff_sm,origin='upper',cmap='RdBu_r',
                                 vmin=-vmax_d,vmax=vmax_d,
                                 aspect='auto',extent=[0,360,-90,90],
                                 interpolation='bicubic')
        axes[1,0].set_xticks(range(0,361,90)); axes[1,0].set_yticks([-60,-30,0,30,60])
        fig.colorbar(im_d,ax=axes[1,0],fraction=0.02,pad=0.01).ax.tick_params(
            colors=SIRIL_DIM,labelsize=7)

        # Change overlay
        if self.cb_show_changes.isChecked() and c['n_changed'] > 0:
            chg = c['changes'].astype(float)
            chg[~c['valid']] = np.nan
            axes[1,0].imshow(chg, origin='upper', cmap='autumn_r',
                             alpha=0.55, aspect='auto', extent=[0,360,-90,90],
                             vmin=0, vmax=1, interpolation='nearest')
            axes[1,0].text(0.02,0.05,f"⚠ {c['pct_changed']:.1f}% changed",
                           transform=axes[1,0].transAxes,
                           color=SIRIL_WRN,fontsize=9,fontweight='bold')

        # ── Panel 4: Ratio map ────────────────────────────────────────────
        ratio = c['ratio']
        _ax(axes[1,1],"Ratio B/A  (green=unchanged, red=brighter, blue=darker)",
            "Longitude (°)","")
        # Custom ratio colormap centred at 1.0
        axes[1,1].imshow(ratio,origin='upper',cmap='RdYlGn',
                         vmin=0.7,vmax=1.3,aspect='auto',
                         extent=[0,360,-90,90],interpolation='bicubic')
        axes[1,1].set_xticks(range(0,361,90)); axes[1,1].set_yticks([])

        self.comp_canvas.redraw()

        self.comp_stats.setText(
            f"σ_diff={c['sigma']:.4f}  |  "
            f"Changed: {c['n_changed']} px = {c['pct_changed']:.1f}%  |  "
            f"Max |diff|: {c['max_abs_diff']:.4f}  |  "
            f"Drift correction: {drift:.0f}°")

    def _on_drift_slider(self, val):
        self.drift_lbl.setText(f"{val:.0f}°")
        if self._result and self._result['comp']:
            self._redraw_compare()

    def _draw_profile(self):
        r = self._result
        if not r or r['map_a'] is None: return
        ma = r['map_a']; mh, mw = ma.shape
        lat1 = self.sp_lat1.value(); lat2 = self.sp_lat2.value()
        # Map lat to row indices
        r1 = int((90-max(lat1,lat2))/180 * mh)
        r2 = int((90-min(lat1,lat2))/180 * mh)
        r1 = max(0,min(r1,mh-1)); r2 = max(r1+1,min(r2,mh))
        lons = np.linspace(0,360,mw)

        fig = self.prof_canvas.fig; fig.clear()
        fig.patch.set_facecolor(SIRIL_BG)
        ax = fig.add_subplot(111)
        _ax(ax,f"Longitude profile  lat={lat1:.0f}° to {lat2:.0f}°",
            "System III Longitude (°)","Mean brightness")

        strip_a = np.nanmean(ma[r1:r2,:],axis=0)
        ax.plot(lons, strip_a, color=SIRIL_ACC, lw=1.5, label='Epoch A', alpha=0.9)

        if r['map_b'] is not None:
            mb = r['map_b']
            drift = float(self.drift_live.value())
            if abs(drift) > 0.1:
                shift = int(round(drift/360*mw))%mw
                mb = np.roll(mb, shift, axis=1)
            strip_b = np.nanmean(mb[r1:r2,:], axis=0)
            ax.plot(lons, strip_b, color=SIRIL_WRN, lw=1.5,
                    label='Epoch B', alpha=0.9)
            diff_strip = strip_b - strip_a
            ax2 = ax.twinx()
            ax2.plot(lons, diff_strip, color=SIRIL_ERR, lw=0.8, ls='--',
                     alpha=0.6, label='B−A')
            ax2.axhline(0, color=SIRIL_BOR, lw=0.5)
            ax2.set_ylabel('Diff B−A', color=SIRIL_DIM, fontsize=8)
            ax2.tick_params(colors=SIRIL_DIM, labelsize=7)

        ax.set_xticks(range(0,361,60))
        ax.legend(fontsize=7, facecolor=SIRIL_BG3, edgecolor=SIRIL_BOR,
                  labelcolor=SIRIL_TXT, loc='upper right')
        self.prof_canvas.redraw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        r = self._result
        if not r or r['map_a'] is None: return
        path, _ = QFileDialog.getSaveFileName(
            self,"Export maps","planetary_map_A.fits",
            "FITS (*.fits);;All (*)")
        if not path: return
        try:
            out_path = Path(path)
            if ASTROPY_OK:
                from astropy.io import fits as fts
                hdr = fts.Header()
                hdr['CRVAL1'] = 0.0;   hdr['CDELT1'] = 360/r['map_a'].shape[1]
                hdr['CRVAL2'] = 90.0;  hdr['CDELT2'] = -180/r['map_a'].shape[0]
                hdr['CTYPE1'] = 'LONIII'; hdr['CTYPE2'] = 'LAT'
                hdr['PLANET'] = self.cb_planet.currentText()
                fts.PrimaryHDU(r['map_a'].astype(np.float32), header=hdr).writeto(
                    str(out_path), overwrite=True)
                if r['map_b'] is not None:
                    p2 = out_path.with_name(out_path.stem+'_B'+out_path.suffix)
                    fts.PrimaryHDU(r['map_b'].astype(np.float32), header=hdr).writeto(
                        str(p2), overwrite=True)
                    if r['comp']:
                        pd = out_path.with_name(out_path.stem+'_diff'+out_path.suffix)
                        fts.PrimaryHDU(r['comp']['diff_smooth'].astype(np.float32),
                                       header=hdr).writeto(str(pd), overwrite=True)
                self._log(f"✓ Exported maps to {out_path.parent}")
            else:
                np.save(str(out_path.with_suffix('.npy')), r['map_a'])
                self._log(f"✓ Exported map as numpy array (astropy not available)")
        except Exception as e:
            QMessageBox.critical(self,"Export error",str(e))

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PlanetMapper(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
