r"""
HYPERLOAD — Sersic Surface Brightness Profiler
================================================
Galaxy morphology via Sersic profile fitting and isophote analysis.
No free GUI tool exists for amateurs — GALFIT is CLI-only on Linux.

Fits I(r) = Ie × exp(−bn × [(r/Re)^(1/n) − 1])
  Ie : surface brightness at effective radius
  Re : effective (half-light) radius in pixels → converted to arcsec
  n  : Sersic index (1=exponential disk, 4=de Vaucouleurs elliptical)
  bn : solved from n via Ciotti & Bertin 1999 approximation

PSF-convolved fitting: the model is numerically convolved with the
measured PSF before comparing to data. This recovers intrinsic Re
accurately even for compact galaxies where PSF smearing is significant.

Ellipse isophote fitting (Jedrzejewski 1987): fits ellipses to
constant-brightness contours, recovering ellipticity ε, position
angle PA, and any radial variations — indicating bars, rings, lenses.

Import API:
    from sersic_profiler import fit_sersic_1d, fit_ellipses, extract_radial_profile

References:
    Sersic 1968, Atlas de Galaxias Australes
    de Vaucouleurs 1948, Ann.Astrophys. 11, 247   (n=4 case)
    Jedrzejewski 1987, MNRAS 226, 747              (isophote fitting)
    Ciotti & Bertin 1999, A&A 352, 447             (bn approximation)
    Graham et al. 2005, AJ 130, 1535               (Sersic review)

Version: 1.0.0
Project: HYPERLOAD
"""

from __future__ import annotations

import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_position
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import gamma as gamma_fn

# ── crash logger ──────────────────────────────────────────────────────────────
def _crash(exc_type, exc_val, exc_tb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nsersic_profiler.py\n")
        traceback.print_exception(exc_type, exc_val, exc_tb, file=f)
    sys.__excepthook__(exc_type, exc_val, exc_tb)


sys.excepthook = _crash

try:
    import sirilpy as s
    s.ensure_installed("PyQt6")
    s.ensure_installed("numpy")
    s.ensure_installed("scipy")
    s.ensure_installed("matplotlib")
    s.ensure_installed("astropy")
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
VERSION = "1.0.0"
APP_TITLE = "HYPERLOAD — Sersic Surface Brightness Profiler"
ACCENT = "#c060ff"
ACCENT_DARK = "#6030a0"

DARK = """
QMainWindow,QWidget{background:#0e1118;color:#d0d8e8;
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}
QTabWidget::pane{border:1px solid #1e2840;background:#111828;}
QTabBar::tab{background:#131a28;color:#303850;padding:6px 16px;
    border:1px solid #1e2840;border-bottom:none;}
QTabBar::tab:selected{background:#111828;color:#c060ff;
    border-bottom:2px solid #c060ff;}
QGroupBox{border:1px solid #1e2840;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#303850;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#131a28;color:#c0d0e8;border:1px solid #1e2840;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#6030a0;border-color:#c060ff;color:white;}
QPushButton:disabled{color:#283040;border-color:#131a28;}
QPushButton#run{background:#6030a0;border-color:#c060ff;
    color:white;font-weight:bold;padding:7px 20px;}
QPushButton#run:hover{background:#c060ff;}
QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{background:#0e1420;
    border:1px solid #1e2840;border-radius:3px;padding:3px 6px;color:#c0d0e8;}
QCheckBox::indicator{width:14px;height:14px;border:1px solid #1e2840;
    background:#131a28;border-radius:2px;}
QCheckBox::indicator:checked{background:#c060ff;}
QTextEdit{background:#080c14;border:1px solid #1e2840;
    color:#40c080;font-family:Consolas,monospace;font-size:10px;}
QTableWidget{background:#080c14;alternate-background-color:#0c1018;
    gridline-color:#1e2840;color:#c0d0e8;}
QTableWidget QHeaderView::section{background:#0e1420;color:#404858;
    border:1px solid #1e2840;padding:3px;font-weight:bold;}
QProgressBar{background:#0e1420;border:1px solid #1e2840;
    border-radius:3px;height:8px;}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #6030a0,stop:1 #c060ff);border-radius:2px;}
QSplitter::handle{background:#1e2840;}
QLabel#title_lbl{font-size:15px;font-weight:bold;color:#c060ff;}
QLabel#sub_lbl{font-size:10px;color:#505868;}
QLabel#results_lbl{font-family:Consolas,monospace;font-size:10px;color:#c060ff;}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


def sersic_bn(n: float) -> float:
    """Ciotti & Bertin 1999 approximation for b_n."""
    return 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n**2)


def sersic_1d(r: np.ndarray, Ie: float, re: float, n: float) -> np.ndarray:
    """1D circularly-averaged Sersic profile."""
    bn = sersic_bn(n)
    return Ie * np.exp(-bn * ((np.maximum(r, 1e-4) / re) ** (1.0 / n) - 1.0))


def sersic_2d(
    shape: tuple,
    xc: float,
    yc: float,
    Ie: float,
    re: float,
    n: float,
    q: float,
    theta_deg: float,
) -> np.ndarray:
    """2D elliptical Sersic on image grid."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx = xx - xc
    dy = yy - yc
    th = np.radians(theta_deg)
    xr = dx * np.cos(th) + dy * np.sin(th)
    yr = -dx * np.sin(th) + dy * np.cos(th)
    r_ell = np.sqrt(xr**2 + (yr / max(q, 0.01)) ** 2)
    bn = sersic_bn(n)
    return Ie * np.exp(-bn * ((np.maximum(r_ell, 1e-4) / re) ** (1.0 / n) - 1.0))


def sersic_1d_convolved(
    r_mid: np.ndarray, Ie: float, re: float, n: float, psf_sigma: float
) -> np.ndarray:
    """PSF-convolved 1D Sersic profile (numerical)."""
    r_dense = np.linspace(0, r_mid[-1] * 1.5, 600)
    model_d = sersic_1d(r_dense, Ie, re, n)
    dr = r_dense[1] - r_dense[0]
    k_half = max(int(5 * psf_sigma / dr), 3)
    k_r = np.arange(-k_half, k_half + 1) * dr
    kernel = np.exp(-0.5 * (k_r / psf_sigma) ** 2)
    kernel /= kernel.sum()
    conv = fftconvolve(model_d, kernel, mode="same")
    return np.interp(r_mid, r_dense, conv)


def total_luminosity(Ie: float, re: float, n: float) -> float:
    """Total integrated flux of Sersic model."""
    bn = sersic_bn(n)
    return (
        2
        * np.pi
        * Ie
        * re**2
        * n
        * np.exp(bn)
        * bn ** (-2 * n)
        * gamma_fn(2 * n)
    )


def extract_radial_profile(
    image: np.ndarray,
    xc: float,
    yc: float,
    sky: float,
    r_max: float = None,
    bin_width: float = 2.0,
    q: float = 1.0,
    theta_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract azimuthally averaged surface brightness profile."""
    h, w = image.shape
    yy, xx = np.mgrid[:h, :w].astype(float)
    dx = xx - xc
    dy = yy - yc
    th = np.radians(theta_deg)
    xr = dx * np.cos(th) + dy * np.sin(th)
    yr = -dx * np.sin(th) + dy * np.cos(th)
    r_ell = np.sqrt(xr**2 + (yr / max(q, 0.01)) ** 2)

    if r_max is None:
        r_max = min(xc, yc, w - xc, h - yc) * 0.8

    r_bins = np.arange(0.5, r_max, bin_width)
    r_mid = (r_bins[:-1] + r_bins[1:]) / 2
    profile = np.array(
        [
            float(
                np.median(
                    (image - sky)[
                        (r_ell >= r_bins[i]) & (r_ell < r_bins[i + 1])
                    ]
                )
            )
            for i in range(len(r_bins) - 1)
        ]
    )
    return r_mid, profile


def estimate_sky(image: np.ndarray, border_frac: float = 0.1) -> float:
    """Estimate sky from sigma-clipped border median."""
    h, w = image.shape
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))
    border = np.concatenate(
        [
            image[:bh, :].ravel(),
            image[-bh:, :].ravel(),
            image[:, :bw].ravel(),
            image[:, -bw:].ravel(),
        ]
    )
    med = float(np.median(border))
    mad = float(np.median(np.abs(border - med))) * 1.4826
    clipped = border[np.abs(border - med) < 3 * mad]
    return float(np.median(clipped))


def fit_sersic_1d(
    r_mid: np.ndarray,
    profile: np.ndarray,
    psf_sigma: float = 0.0,
    n_fixed: float | None = None,
) -> dict:
    """Fit Sersic profile to radial profile data."""
    valid = (profile > 0) & np.isfinite(profile) & (r_mid > 1.0)
    r_f = r_mid[valid]
    p_f = gaussian_filter1d(np.maximum(profile[valid], 1e-3), sigma=1.5)

    def loss(params):
        if n_fixed is not None:
            log_Ie, log_re = params
            n = float(n_fixed)
        else:
            log_Ie, log_re, log_n = params
            n = float(np.clip(np.exp(log_n), 0.3, 8.0))
        Ie = float(np.exp(log_Ie))
        re = float(max(np.exp(log_re), 0.5))
        if psf_sigma > 0.3:
            model = sersic_1d_convolved(r_f, Ie, re, n, psf_sigma)
        else:
            model = sersic_1d(r_f, Ie, re, n)
        model = np.maximum(model, 1e-6)
        return float(np.sum((np.log(p_f) - np.log(model)) ** 2))

    best_loss = 1e20
    best_p0 = None
    n_tries = [0.5, 1.0, 2.0, 4.0] if n_fixed is None else [n_fixed]
    re_tries = np.percentile(r_f, [20, 40, 60])
    for n_try in n_tries:
        for re_try in re_tries:
            idx = np.argmin(np.abs(r_f - re_try))
            Ie_try = float(p_f[idx])
            if n_fixed is not None:
                p0 = [np.log(max(Ie_try, 1)), np.log(re_try)]
            else:
                p0 = [np.log(max(Ie_try, 1)), np.log(re_try), np.log(n_try)]
            try:
                l = loss(p0)
                if l < best_loss:
                    best_loss = l
                    best_p0 = p0
            except Exception:
                pass

    if best_p0 is None:
        return {"fit_ok": False, "error": "No valid starting point"}

    try:
        res = minimize(
            loss,
            best_p0,
            method="Nelder-Mead",
            options={"maxiter": 15000, "xatol": 1e-7, "fatol": 1e-7, "adaptive": True},
        )
        if n_fixed is not None:
            Ie_f = float(np.exp(res.x[0]))
            re_f = float(max(np.exp(res.x[1]), 0.5))
            n_f = float(n_fixed)
        else:
            Ie_f = float(np.exp(res.x[0]))
            re_f = float(max(np.exp(res.x[1]), 0.5))
            n_f = float(np.clip(np.exp(res.x[2]), 0.3, 8.0))

        if psf_sigma > 0.3:
            model_fit = sersic_1d_convolved(r_mid, Ie_f, re_f, n_f, psf_sigma)
        else:
            model_fit = sersic_1d(r_mid, Ie_f, re_f, n_f)
        residuals = profile - model_fit

        return {
            "Ie": Ie_f,
            "re": re_f,
            "n": n_f,
            "bn": sersic_bn(n_f),
            "total_lum": total_luminosity(Ie_f, re_f, n_f),
            "model": model_fit,
            "residuals": residuals,
            "fit_ok": True,
            "converged": res.success,
        }
    except Exception as e:
        return {"fit_ok": False, "error": str(e)}


def fit_ellipses(
    image: np.ndarray,
    xc: float,
    yc: float,
    sky: float,
    r_min: float = 5.0,
    r_max: float | None = None,
    n_ellipses: int = 15,
) -> list[dict]:
    """Simplified isophote fitting (Jedrzejewski 1987)."""
    from scipy.ndimage import map_coordinates

    h, w = image.shape
    img = image.astype(float) - sky

    if r_max is None:
        r_max = min(xc, yc, w - xc, h - yc) * 0.75

    r_values = np.linspace(r_min, r_max, n_ellipses)
    results = []

    for r0 in r_values:

        def ellipse_loss(params):
            q_try = float(np.clip(params[0], 0.1, 1.0))
            th = float(params[1])
            angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
            xe = (
                xc
                + r0 * np.cos(angles) * np.cos(th)
                - r0 / q_try * np.sin(angles) * np.sin(th)
            )
            ye = (
                yc
                + r0 * np.cos(angles) * np.sin(th)
                + r0 / q_try * np.sin(angles) * np.cos(th)
            )
            xe = np.clip(xe, 0, w - 1)
            ye = np.clip(ye, 0, h - 1)
            vals = map_coordinates(img, [ye, xe], order=1, mode="reflect")
            a1 = float(np.mean(vals * np.cos(angles)))
            b1 = float(np.mean(vals * np.sin(angles)))
            a2 = float(np.mean(vals * np.cos(2 * angles)))
            b2 = float(np.mean(vals * np.sin(2 * angles)))
            return a1**2 + b1**2 + a2**2 + b2**2

        try:
            res = minimize(
                ellipse_loss,
                [0.8, 0.0],
                method="Nelder-Mead",
                bounds=[(0.1, 1.0), (-np.pi / 2, np.pi / 2)],
                options={"maxiter": 500, "xatol": 0.01},
            )
            q_fit = float(np.clip(res.x[0], 0.1, 1.0))
            th_fit = float(res.x[1])

            angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
            xe = (
                xc
                + r0 * np.cos(angles) * np.cos(th_fit)
                - r0 / q_fit * np.sin(angles) * np.sin(th_fit)
            )
            ye = (
                yc
                + r0 * np.cos(angles) * np.sin(th_fit)
                + r0 / q_fit * np.sin(angles) * np.cos(th_fit)
            )
            xe = np.clip(xe, 0, w - 1)
            ye = np.clip(ye, 0, h - 1)
            vals = map_coordinates(img, [ye, xe], order=1, mode="reflect")
            a4 = float(np.mean(vals * np.cos(4 * angles))) / (np.mean(vals) + 1e-10)

            results.append(
                {
                    "r": r0,
                    "ellipticity": 1 - q_fit,
                    "pa_deg": float(np.degrees(th_fit)) % 180,
                    "mean_intensity": float(np.mean(vals)),
                    "a4": a4,
                }
            )
        except Exception:
            pass

    return results


def curve_of_growth(
    image: np.ndarray, xc: float, yc: float, sky: float, r_max: float = None
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative flux vs aperture radius."""
    h, w = image.shape
    yy, xx = np.mgrid[:h, :w].astype(float)
    r_arr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
    if r_max is None:
        r_max = min(xc, yc, w - xc, h - yc) * 0.8
    radii = np.arange(1, int(r_max))
    fluxes = np.array([float((image - sky)[r_arr < ri].sum()) for ri in radii])
    return radii.astype(float), fluxes


def morphology_class(n: float) -> str:
    """Classify galaxy morphology from Sersic index."""
    if n < 0.7:
        return "Extended / LSB disk"
    if n < 1.3:
        return "Pure exponential disk (Sd/Irr)"
    if n < 2.0:
        return "Late-type spiral (Sb/Sc)"
    if n < 3.0:
        return "Early-type spiral (Sa/S0)"
    if n < 5.0:
        return "Elliptical (E)"
    return "Classical elliptical / BCG"


def find_galaxy_centroid(image: np.ndarray, sky: float) -> tuple[float, float]:
    """Peak of smoothed sky-subtracted image."""
    sub = np.maximum(image.astype(float) - sky, 0)
    sm = gaussian_filter(sub, 1.5)
    pos = maximum_position(sm)
    return float(pos[1]), float(pos[0])


def cog_half_light_radius(radii: np.ndarray, fluxes: np.ndarray) -> float:
    """Radius enclosing 50% of total flux."""
    if len(fluxes) < 2:
        return float("nan")
    half = 0.5 * fluxes[-1]
    idx = np.searchsorted(fluxes, half)
    if idx <= 0:
        return float(radii[0])
    if idx >= len(fluxes):
        return float(radii[-1])
    r0, r1 = radii[idx - 1], radii[idx]
    f0, f1 = fluxes[idx - 1], fluxes[idx]
    if f1 <= f0:
        return float(r1)
    t = (half - f0) / (f1 - f0)
    return float(r0 + t * (r1 - r0))


def load_fits_2d(path: str) -> np.ndarray:
    """Load 2D science frame; RGB → luminance."""
    from astropy.io import fits as af

    with af.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if data.shape[0] in (3, 4):
            planes = data[:3]
        elif data.shape[-1] in (3, 4):
            planes = np.moveaxis(data[..., :3], -1, 0)
        else:
            return np.asarray(data[0], dtype=np.float64)
        r, g, b = planes[0], planes[1], planes[2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    raise ValueError(f"Unsupported FITS shape {data.shape}")


def make_synthetic_galaxy(seed: int = 42) -> dict:
    """Validation synthetic galaxy + metadata."""
    rng = np.random.default_rng(seed)
    n = 256
    xc = yc = 128.0
    true = {"Ie": 500.0, "re": 25.0, "n": 2.0, "q": 0.7, "theta_deg": 35.0}
    clean = sersic_2d((n, n), xc, yc, **true)
    psf_sig = 3.0 / 2.355
    conv = gaussian_filter(clean, psf_sig)
    sky_val = 150.0
    obs = conv + sky_val + rng.normal(0, 2, (n, n))
    return {
        "image": obs,
        "xc": xc,
        "yc": yc,
        "sky": sky_val,
        "psf_sigma": psf_sig,
        "true": true,
        "filename": "synthetic_galaxy.fits",
    }


def _validate_algorithms() -> None:
    rng = np.random.default_rng(42)
    n = 256
    xc = yc = 128.0
    true = {"Ie": 500.0, "re": 25.0, "n": 2.0, "q": 0.7, "theta_deg": 35.0}
    clean = sersic_2d((n, n), xc, yc, **true)
    psf_sig = 3.0 / 2.355
    conv = gaussian_filter(clean, psf_sig)
    sky_val = 150.0
    obs = conv + sky_val + rng.normal(0, 2, (n, n))

    sky_est = estimate_sky(obs, border_frac=0.1)
    if abs(sky_est - sky_val) >= 5:
        raise RuntimeError(f"Sky error: {abs(sky_est - sky_val):.1f}")

    r_mid, prof = extract_radial_profile(
        obs, xc, yc, sky=sky_est, r_max=90, bin_width=2.0
    )
    if len(r_mid) <= 10 or prof[0] <= prof[-1]:
        raise RuntimeError("Radial profile invalid")

    result = fit_sersic_1d(r_mid, prof, psf_sigma=psf_sig)
    if not result["fit_ok"]:
        raise RuntimeError(result.get("error", "fit failed"))

    re_err = abs(result["re"] - 25.0) / 25.0
    n_err = abs(result["n"] - 2.0) / 2.0
    if re_err >= 0.25:
        raise RuntimeError(f"Re error {re_err*100:.1f}%")
    if n_err >= 0.20:
        raise RuntimeError(f"n error {n_err*100:.1f}%")

    radii, fluxes = curve_of_growth(obs, xc, yc, sky=sky_est)
    if fluxes[-1] <= fluxes[0]:
        raise RuntimeError("COG monotonicity failed")

    morph = morphology_class(result["n"])
    if "spiral" not in morph.lower() and "elliptical" not in morph.lower():
        raise RuntimeError(f"Unexpected morphology: {morph}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--validate":
    _validate_algorithms()
    print("SERSIC PROFILER VALIDATED OK")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PyQt6 GUI
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse
import matplotlib.colors as mcolors

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8, 5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor="#0c1018")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def redraw(self):
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()


def _ax(ax, title="", xl="", yl="", accent=ACCENT):
    ax.set_facecolor("#080c14")
    ax.set_title(title, color=accent, fontsize=9, pad=3)
    ax.set_xlabel(xl, color="#303850", fontsize=8)
    ax.set_ylabel(yl, color="#303850", fontsize=8)
    ax.tick_params(colors="#303850", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2840")


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#080c14;border-top:1px solid #1e2840;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        self.lbl = QLabel("Ready")
        self.lbl.setStyleSheet("color:#283040;font-size:10px;")
        self.pbar = QProgressBar()
        self.pbar.setFixedSize(200, 12)
        self.pbar.setVisible(False)
        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.pbar)

    def update(self, pct, msg):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True)
            self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))


class SersicFitWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            img = p["image"]
            xc, yc = p["xc"], p["yc"]
            sky = p["sky"]
            self.progress.emit(5, "Extracting radial profile…")

            ellipses = fit_ellipses(
                img,
                xc,
                yc,
                sky,
                r_max=p["r_max"],
                n_ellipses=p["n_ellipses"],
            )
            q_use = 1.0
            pa_use = 0.0
            if ellipses:
                mid = ellipses[len(ellipses) // 2]
                q_use = max(1.0 - mid["ellipticity"], 0.1)
                pa_use = mid["pa_deg"]

            r_mid, prof = extract_radial_profile(
                img,
                xc,
                yc,
                sky=sky,
                r_max=p["r_max"],
                bin_width=p["bin_width"],
                q=q_use,
                theta_deg=pa_use,
            )
            self.progress.emit(35, "Fitting Sersic profile…")

            n_fixed = p.get("n_fixed")
            fit = fit_sersic_1d(
                r_mid,
                prof,
                psf_sigma=p["psf_sigma"],
                n_fixed=n_fixed,
            )
            if not fit.get("fit_ok"):
                self.error.emit(fit.get("error", "Sersic fit failed"))
                return

            self.progress.emit(70, "Curve of growth…")
            radii, fluxes = curve_of_growth(img, xc, yc, sky, r_max=p["r_max"])
            re_cog = cog_half_light_radius(radii, fluxes)
            morph = morphology_class(fit["n"])

            eps_med = float(np.median([e["ellipticity"] for e in ellipses])) if ellipses else 0.0
            pa_med = float(np.median([e["pa_deg"] for e in ellipses])) if ellipses else 0.0

            self.progress.emit(100, "Fit complete")
            self.finished.emit(
                {
                    "r_mid": r_mid,
                    "profile": prof,
                    "fit": fit,
                    "ellipses": ellipses,
                    "radii_cog": radii,
                    "fluxes_cog": fluxes,
                    "re_cog": re_cog,
                    "morphology": morph,
                    "eps_med": eps_med,
                    "pa_med": pa_med,
                    "q_use": q_use,
                    "pa_use": pa_use,
                }
            )
        except Exception as ex:
            self.error.emit(f"{ex}\n\n{traceback.format_exc()}")


class SersicProfilerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 920)
        self.setStyleSheet(DARK)
        logo = SCRIPT_DIR / "logo.png"
        if logo.exists():
            self.setWindowIcon(QIcon(str(logo)))

        self._image: np.ndarray | None = None
        self._filename = ""
        self._result: dict | None = None
        self._worker = None
        self._wthread = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setSpacing(0)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self._make_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._make_controls())
        splitter.addWidget(self._make_tabs())
        splitter.setSizes([280, 1120])
        rl.addWidget(splitter, 1)

        self.status = StatusFooter()
        rl.addWidget(self.status)

    def _make_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setFixedHeight(66)
        hdr.setStyleSheet("background:#080c14;border-bottom:1px solid #1e2840;")
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(12, 0, 12, 0)
        logo = SCRIPT_DIR / "logo.png"
        if logo.exists():
            lbl = QLabel()
            lbl.setPixmap(
                QPixmap(str(logo)).scaled(
                    50,
                    50,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            lay.addWidget(lbl)
        v = QVBoxLayout()
        t = QLabel("Sersic Surface Brightness Profiler")
        t.setObjectName("title_lbl")
        s = QLabel("Sersic 1968 · Jedrzejewski 1987 · PSF-convolved fitting")
        s.setObjectName("sub_lbl")
        v.addWidget(t)
        v.addWidget(s)
        lay.addLayout(v, 1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _make_controls(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        grp = QGroupBox("Image")
        gl = QVBoxLayout(grp)
        btn_load = QPushButton("📂 Load FITS…")
        btn_load.clicked.connect(self._load_fits)
        btn_syn = QPushButton("⚗ Synthetic example")
        btn_syn.clicked.connect(self._load_synthetic)
        gl.addWidget(btn_load)
        gl.addWidget(btn_syn)
        self.lbl_file = QLabel("No image loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#505868;font-size:10px;")
        gl.addWidget(self.lbl_file)
        lay.addWidget(grp)

        grp_c = QGroupBox("Galaxy Centre")
        fc = QFormLayout(grp_c)
        self.sp_xc = QDoubleSpinBox()
        self.sp_xc.setDecimals(1)
        self.sp_xc.setRange(0, 10000)
        self.sp_yc = QDoubleSpinBox()
        self.sp_yc.setDecimals(1)
        self.sp_yc.setRange(0, 10000)
        fc.addRow("X centre (px):", self.sp_xc)
        fc.addRow("Y centre (px):", self.sp_yc)
        btn_auto = QPushButton("🎯 Auto-find (centroid of brightest peak)")
        btn_auto.clicked.connect(self._auto_centre)
        fc.addRow(btn_auto)
        self.sp_sky = QDoubleSpinBox()
        self.sp_sky.setDecimals(2)
        self.sp_sky.setRange(-1e6, 1e6)
        fc.addRow("Sky background:", self.sp_sky)
        btn_sky = QPushButton("🌌 Re-estimate sky")
        btn_sky.clicked.connect(self._reestimate_sky)
        fc.addRow(btn_sky)
        lay.addWidget(grp_c)

        grp_p = QGroupBox("PSF")
        fp = QFormLayout(grp_p)
        self.sp_psf_fwhm = QDoubleSpinBox()
        self.sp_psf_fwhm.setRange(0, 20)
        self.sp_psf_fwhm.setDecimals(2)
        self.sp_psf_fwhm.setValue(0)
        fp.addRow("PSF FWHM (px):", self.sp_psf_fwhm)
        hint = QLabel("Measure from nearby stars via psf_heatmap.py output")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#505868;font-size:9px;")
        fp.addRow(hint)
        hint2 = QLabel("0 = skip PSF correction (faster, re will be PSF-broadened)")
        hint2.setWordWrap(True)
        hint2.setStyleSheet("color:#505868;font-size:9px;")
        fp.addRow(hint2)
        lay.addWidget(grp_p)

        grp_f = QGroupBox("Fitting")
        ff = QFormLayout(grp_f)
        self.sp_rmax = QDoubleSpinBox()
        self.sp_rmax.setRange(10, 5000)
        self.sp_rmax.setValue(90)
        self.sp_bin = QDoubleSpinBox()
        self.sp_bin.setRange(0.5, 5)
        self.sp_bin.setValue(2.0)
        self.sp_bin.setSingleStep(0.5)
        self.chk_fix_n = QCheckBox("Fix n (de Vaucouleurs n=4)")
        self.chk_fix_n.toggled.connect(self._toggle_fix_n)
        self.sp_n_fixed = QDoubleSpinBox()
        self.sp_n_fixed.setRange(0.3, 8)
        self.sp_n_fixed.setValue(4.0)
        self.sp_n_fixed.setEnabled(False)
        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setRange(0, 10)
        self.sp_scale.setDecimals(4)
        self.sp_scale.setSingleStep(0.01)
        self.sp_n_ell = QSpinBox()
        self.sp_n_ell.setRange(5, 30)
        self.sp_n_ell.setValue(15)
        ff.addRow("Max radius (px):", self.sp_rmax)
        ff.addRow("Bin width (px):", self.sp_bin)
        ff.addRow(self.chk_fix_n)
        ff.addRow("Fixed n value:", self.sp_n_fixed)
        ff.addRow("Pixel scale (arcsec/px):", self.sp_scale)
        ff.addRow("Isophote ellipses:", self.sp_n_ell)
        lay.addWidget(grp_f)

        self.btn_fit = QPushButton("▶ Fit Profile")
        self.btn_fit.setObjectName("run")
        self.btn_fit.setFixedHeight(38)
        self.btn_fit.clicked.connect(self._run_fit)
        lay.addWidget(self.btn_fit)

        self.btn_export = QPushButton("💾 Export results…")
        self.btn_export.clicked.connect(self._export_results)
        lay.addWidget(self.btn_export)

        self.lbl_results = QLabel("Results will appear here after fitting.")
        self.lbl_results.setObjectName("results_lbl")
        self.lbl_results.setWordWrap(True)
        lay.addWidget(self.lbl_results)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _make_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        self.canvas_img = MplCanvas(figsize=(9, 6))
        self.canvas_prof = MplCanvas(figsize=(9, 5))
        self.canvas_morph = MplCanvas(figsize=(9, 5))
        self.canvas_cog = MplCanvas(figsize=(9, 4))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        tabs.addTab(self.canvas_img, "🖼 Image")
        tabs.addTab(self.canvas_prof, "📈 Radial Profile")
        tabs.addTab(self.canvas_morph, "🪐 Morphology")
        tabs.addTab(self.canvas_cog, "📊 Curve of Growth")
        tabs.addTab(self.log, "📋 Log")
        return tabs

    def _toggle_fix_n(self, checked: bool):
        self.sp_n_fixed.setEnabled(checked)
        if checked:
            self.sp_n_fixed.setValue(4.0)

    def _log(self, msg: str):
        self.log.append(msg)
        self.status.update(-1, msg)

    def _set_image(self, img: np.ndarray, name: str):
        self._image = img
        self._filename = name
        h, w = img.shape
        self.lbl_file.setText(f"{name}\n{w}×{h} px")
        self.sp_xc.setMaximum(w - 1)
        self.sp_yc.setMaximum(h - 1)
        self.sp_xc.setValue(w / 2)
        self.sp_yc.setValue(h / 2)
        self.sp_rmax.setMaximum(min(h, w) / 2)
        sky = estimate_sky(img)
        self.sp_sky.setValue(sky)
        self._log(f"Loaded {name} shape=({h},{w}) sky≈{sky:.1f}")
        self._plot_image_preview()

    def _load_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FITS", str(SCRIPT_DIR), "FITS (*.fits *.fit *.fts)"
        )
        if not path:
            return
        try:
            img = load_fits_2d(path)
            self._set_image(img, Path(path).name)
        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))

    def _load_synthetic(self):
        syn = make_synthetic_galaxy()
        self._set_image(syn["image"], syn["filename"])
        self.sp_xc.setValue(syn["xc"])
        self.sp_yc.setValue(syn["yc"])
        self.sp_sky.setValue(syn["sky"])
        self.sp_psf_fwhm.setValue(3.0)
        self._log("Synthetic n=2 galaxy loaded (true Re=25 px)")

    def _auto_centre(self):
        if self._image is None:
            return
        xc, yc = find_galaxy_centroid(self._image, self.sp_sky.value())
        self.sp_xc.setValue(xc)
        self.sp_yc.setValue(yc)
        self._log(f"Auto centre: ({xc:.1f}, {yc:.1f})")
        self._plot_image_preview()

    def _reestimate_sky(self):
        if self._image is None:
            return
        sky = estimate_sky(self._image)
        self.sp_sky.setValue(sky)
        self._log(f"Sky re-estimated: {sky:.2f}")

    def _run_fit(self):
        if self._image is None:
            QMessageBox.warning(self, "Fit", "Load an image first.")
            return
        self.btn_fit.setEnabled(False)
        psf_fwhm = self.sp_psf_fwhm.value()
        psf_sigma = psf_fwhm / 2.355 if psf_fwhm > 0.3 else 0.0
        params = {
            "image": self._image,
            "xc": self.sp_xc.value(),
            "yc": self.sp_yc.value(),
            "sky": self.sp_sky.value(),
            "r_max": self.sp_rmax.value(),
            "bin_width": self.sp_bin.value(),
            "psf_sigma": psf_sigma,
            "n_fixed": self.sp_n_fixed.value() if self.chk_fix_n.isChecked() else None,
            "n_ellipses": self.sp_n_ell.value(),
        }
        self._worker = SersicFitWorker(params)
        self._wthread = QThread(self)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda p, m: self.status.update(p, m))
        self._worker.finished.connect(self._on_fit_done)
        self._worker.error.connect(self._on_fit_error)
        self._worker.finished.connect(self._wthread.quit)
        self._worker.error.connect(self._wthread.quit)
        self._wthread.finished.connect(self._cleanup_thread)
        self._wthread.start()

    def _cleanup_thread(self):
        self.btn_fit.setEnabled(True)
        if self._wthread:
            self._wthread.deleteLater()
            self._wthread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _on_fit_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Fit failed", msg[:500])
        self._cleanup_thread()

    def _on_fit_done(self, data: dict):
        self._result = data
        fit = data["fit"]
        scale = self.sp_scale.value()
        re_px = fit["re"]
        re_as = re_px * scale if scale > 0 else float("nan")
        morph = data["morphology"]
        eps = data["eps_med"]
        pa = data["pa_med"]
        lum = fit["total_lum"]

        lines = [
            f"Re = {re_px:.1f} px",
            f"n  = {fit['n']:.2f} ({morph})",
            f"ε  = {eps:.2f}  PA = {pa:.0f}°",
            f"L_total = {lum:.3e} [ADU]",
        ]
        if scale > 0:
            lines[0] += f" = {re_as:.2f}\""
        self.lbl_results.setText("\n".join(lines))
        self._log(
            f"Fit OK: Re={re_px:.1f}px n={fit['n']:.2f} "
            f"COG Re={data['re_cog']:.1f}px"
        )
        self._plot_all()
        self._cleanup_thread()

    def _export_results(self):
        if self._result is None:
            QMessageBox.warning(self, "Export", "Run a fit first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export folder", str(SCRIPT_DIR))
        if not folder:
            return
        out = Path(folder)
        fit = self._result["fit"]
        scale = self.sp_scale.value()
        re_px = fit["re"]
        re_as = re_px * scale if scale > 0 else float("nan")

        txt_path = out / "galaxy_sersic_result.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"File: {self._filename}\n")
            f.write(f"Centre: ({self.sp_xc.value():.1f}, {self.sp_yc.value():.1f}) px\n")
            f.write(f"Sky: {self.sp_sky.value():.1f} ADU\n")
            f.write(f"PSF FWHM: {self.sp_psf_fwhm.value():.1f} px\n")
            f.write(f"Re: {re_px:.1f} px")
            if scale > 0:
                f.write(f" / {re_as:.2f} arcsec\n")
            else:
                f.write("\n")
            f.write(f"Ie: {fit['Ie']:.1f} ADU/px²\n")
            f.write(f"n:  {fit['n']:.2f}\n")
            f.write(f"ε:  {self._result['eps_med']:.2f}\n")
            f.write(f"PA: {self._result['pa_med']:.0f}° E of N\n")
            f.write(f"Total luminosity: {fit['total_lum']:.3e} ADU\n")
            f.write(f"Morphology: {self._result['morphology']}\n")

        csv_prof = out / "galaxy_radial_profile.csv"
        r_mid = self._result["r_mid"]
        prof = self._result["profile"]
        model = fit["model"]
        resid = fit["residuals"]
        with open(csv_prof, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["r_px", "r_arcsec", "surface_brightness", "model", "residual"]
            )
            for i in range(len(r_mid)):
                r_as = r_mid[i] * scale if scale > 0 else ""
                w.writerow([r_mid[i], r_as, prof[i], model[i], resid[i]])

        csv_iso = out / "galaxy_isophotes.csv"
        with open(csv_iso, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["r_px", "ellipticity", "pa_deg", "mean_intensity", "a4"])
            for e in self._result["ellipses"]:
                w.writerow(
                    [e["r"], e["ellipticity"], e["pa_deg"], e["mean_intensity"], e["a4"]]
                )

        self._log(f"Exported to {folder}")
        QMessageBox.information(self, "Export", f"Saved 3 files to:\n{folder}")

    def _plot_image_preview(self):
        if self._image is None:
            return
        fig = self.canvas_img.fig
        fig.clear()
        ax = fig.add_subplot(111)
        sub = np.maximum(self._image - self.sp_sky.value(), 1e-3)
        ax.imshow(sub, origin="lower", cmap="gray", norm=mcolors.LogNorm())
        ax.plot(self.sp_xc.value(), self.sp_yc.value(), "+", color=ACCENT, ms=12)
        _ax(ax, "Galaxy (log)", "X px", "Y px")
        self.canvas_img.redraw()

    def _plot_all(self):
        self._plot_image_tab()
        self._plot_profile_tab()
        self._plot_morph_tab()
        self._plot_cog_tab()

    def _plot_image_tab(self):
        if self._image is None or self._result is None:
            return
        fig = self.canvas_img.fig
        fig.clear()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        sky = self.sp_sky.value()
        sub = np.maximum(self._image - sky, 1e-3)
        xc, yc = self.sp_xc.value(), self.sp_yc.value()
        fit = self._result["fit"]
        re = fit["re"]

        ax1.imshow(sub, origin="lower", cmap="gray", norm=mcolors.LogNorm())
        ax1.plot(xc, yc, "+", color=ACCENT, ms=10)
        for fac, ls in [(0.5, ":"), (1.0, "-"), (2.0, "--"), (3.0, ":")]:
            circ = Circle((xc, yc), fac * re, fill=False, ec=ACCENT, ls=ls, lw=1)
            ax1.add_patch(circ)
        _ax(ax1, "Galaxy + Re circles", "X", "Y")

        ax2.imshow(sub, origin="lower", cmap="gray", norm=mcolors.LogNorm())
        cmap_ell = plt_cmap_ellipses(len(self._result["ellipses"]))
        for i, e in enumerate(self._result["ellipses"]):
            q = max(1.0 - e["ellipticity"], 0.1)
            w = 2 * e["r"]
            h = 2 * e["r"] / q
            ell = Ellipse(
                (xc, yc),
                w,
                h,
                angle=e["pa_deg"],
                fill=False,
                ec=cmap_ell(i),
                lw=0.8,
            )
            ax2.add_patch(ell)
        _ax(ax2, "Isophote ellipses", "X", "Y")
        self.canvas_img.redraw()

    def _plot_profile_tab(self):
        data = self._result
        if data is None:
            return
        fig = self.canvas_prof.fig
        fig.clear()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        r_mid = data["r_mid"]
        prof = data["profile"]
        fit = data["fit"]
        model = fit["model"]
        scale = self.sp_scale.value()
        re = fit["re"]

        ax1.plot(r_mid, prof, "o", color="#606878", ms=3, label="Data")
        ax1.plot(r_mid, model, "-", color=ACCENT, lw=1.5, label="Sersic")
        ax1.axvline(re, color=ACCENT, ls="--", alpha=0.6)
        ax1.set_yscale("log")
        xl = "Radius (px)"
        if scale > 0:
            ax1_top = ax1.twiny()
            ax1_top.set_xlim(r_mid[0] * scale, r_mid[-1] * scale)
            ax1_top.set_xlabel("Radius (arcsec)", color="#303850", fontsize=8)
            xl += "  |  arcsec on top axis"
        _ax(ax1, "Radial profile", xl, "SB (ADU/px², log)")
        ax1.legend(fontsize=7, facecolor="#0c1018", edgecolor="#1e2840")
        ax1.text(
            0.02,
            0.05,
            f"n={fit['n']:.2f}  Re={re:.1f}px\n{data['morphology']}",
            transform=ax1.transAxes,
            color=ACCENT,
            fontsize=8,
        )

        ax2.plot(r_mid, fit["residuals"], "-", color="#80a0c0", lw=0.8)
        ax2.axhline(0, color="#303850", lw=0.5)
        _ax(ax2, "Residuals (data − model)", "Radius (px)", "Residual ADU")
        self.canvas_prof.redraw()

    def _plot_morph_tab(self):
        data = self._result
        if not data or not data["ellipses"]:
            return
        fig = self.canvas_morph.fig
        fig.clear()
        ell = data["ellipses"]
        r = [e["r"] for e in ell]
        eps = [e["ellipticity"] for e in ell]
        pa = [e["pa_deg"] for e in ell]
        a4 = [e["a4"] for e in ell]

        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.plot(r, eps, "o-", color=ACCENT, ms=3)
        _ax(ax1, "Ellipticity ε vs r", "r (px)", "ε")
        ax2.plot(r, pa, "o-", color=ACCENT, ms=3)
        _ax(ax2, "Position angle vs r", "r (px)", "PA (deg)")
        ax3.plot(r, a4, "o-", color=ACCENT, ms=3)
        ax3.axhline(0, color="#303850", lw=0.5)
        _ax(ax3, "a4 (boxy / disky)", "r (px)", "a4")
        fig.text(
            0.5,
            0.01,
            "Disky = embedded disk, boxy = triaxial / merger",
            ha="center",
            color="#505868",
            fontsize=8,
        )
        self.canvas_morph.redraw()

    def _plot_cog_tab(self):
        data = self._result
        if data is None:
            return
        fig = self.canvas_cog.fig
        fig.clear()
        ax = fig.add_subplot(111)
        radii = data["radii_cog"]
        fluxes = data["fluxes_cog"]
        fit = data["fit"]
        re_fit = fit["re"]
        re_cog = data["re_cog"]

        ax.plot(radii, fluxes, "-", color=ACCENT, lw=1.2)
        ax.axvline(re_fit, color=ACCENT, ls="--", alpha=0.5, label=f"Sersic Re={re_fit:.1f}")
        ax.axvline(re_cog, color="#80c0a0", ls=":", label=f"COG Re={re_cog:.1f}")
        _ax(ax, "Curve of growth", "Aperture radius (px)", "Cumulative flux (ADU)")
        ax.legend(fontsize=7, facecolor="#0c1018")
        ax.text(
            0.02,
            0.95,
            f"Total flux: {fluxes[-1]:.3e}\nCOG Re={re_cog:.1f} px",
            transform=ax.transAxes,
            va="top",
            color=ACCENT,
            fontsize=8,
        )
        self.canvas_cog.redraw()


def plt_cmap_ellipses(n: int):
    import matplotlib.colormaps as cmaps

    if n <= 0:
        return lambda i: ACCENT
    cmap = cmaps["hsv"].resampled(max(n, 1))

    def color(i):
        return cmap(i % n)

    return color


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SersicProfilerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
