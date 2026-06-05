r"""
HYPERLOAD — SYSREM / TFA Systematic Detrending
================================================
Removes correlated atmospheric and instrumental systematics from
ensemble light curves to reach sub-millimagnitude photometric precision.

Companion to variable_star.py — reads its CSV light curve output.

Import API:
    from sysrem_detrend import sysrem, tfa_detrend, sysrem_single

References:
    Tamuz et al. 2005, MNRAS 356, 1466  — SYSREM algorithm
    Kovacs et al. 2005, MNRAS 356, 557  — TFA (Trend Filtering Algorithm)
    Aigrain & Irwin 2004, MNRAS 350, 331 — correlated noise framework

Version: 1.0.0
Project: HYPERLOAD
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ── crash logger ──────────────────────────────────────────────────────────────
def _crash(exc_type, exc_val, exc_tb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nsysrem_detrend.py\n")
        traceback.print_exception(exc_type, exc_val, exc_tb, file=f)
    sys.__excepthook__(exc_type, exc_val, exc_tb)


sys.excepthook = _crash

try:
    import sirilpy as s
    s.ensure_installed("PyQt6")
    s.ensure_installed("numpy")
    s.ensure_installed("scipy")
    s.ensure_installed("matplotlib")
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).parent
CHILDREN_DIR = Path(r"C:\Users\Marcell\Desktop\Siril New Scripts")
if str(CHILDREN_DIR) not in sys.path:
    sys.path.insert(0, str(CHILDREN_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from period_finder import load_csv_lightcurve as _pf_load_csv
except ImportError:
    _pf_load_csv = None

VERSION = "1.0.0"
APP_TITLE = "HYPERLOAD — SYSREM / TFA Detrending"
ACCENT = "#e08040"
ACCENT_DARK = "#804020"

DARK = """
QMainWindow,QWidget{background:#0e1118;color:#d0d8e8;
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}
QTabWidget::pane{border:1px solid #1e2840;background:#111828;}
QTabBar::tab{background:#131a28;color:#303850;padding:6px 16px;
    border:1px solid #1e2840;border-bottom:none;}
QTabBar::tab:selected{background:#111828;color:#e08040;
    border-bottom:2px solid #e08040;}
QGroupBox{border:1px solid #1e2840;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#303850;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#131a28;color:#c0d0e8;border:1px solid #1e2840;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#804020;border-color:#e08040;color:white;}
QPushButton:disabled{color:#283040;border-color:#131a28;}
QPushButton#run{background:#804020;border-color:#e08040;
    color:white;font-weight:bold;padding:7px 20px;}
QPushButton#run:hover{background:#e08040;}
QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{background:#0e1420;
    border:1px solid #1e2840;border-radius:3px;padding:3px 6px;color:#c0d0e8;}
QCheckBox::indicator{width:14px;height:14px;border:1px solid #1e2840;
    background:#131a28;border-radius:2px;}
QCheckBox::indicator:checked{background:#e08040;}
QTextEdit{background:#080c14;border:1px solid #1e2840;
    color:#40c080;font-family:Consolas,monospace;font-size:10px;}
QTableWidget{background:#080c14;alternate-background-color:#0c1018;
    gridline-color:#1e2840;color:#c0d0e8;}
QTableWidget QHeaderView::section{background:#0e1420;color:#404858;
    border:1px solid #1e2840;padding:3px;font-weight:bold;}
QProgressBar{background:#0e1420;border:1px solid #1e2840;
    border-radius:3px;height:8px;}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #804020,stop:1 #e08040);border-radius:2px;}
QSplitter::handle{background:#1e2840;}
QLabel#title_lbl{font-size:15px;font-weight:bold;color:#e08040;}
QLabel#sub_lbl{font-size:10px;color:#505868;}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


def _load_csv_lightcurve(csv_path: str) -> tuple:
    """time, value, error — mirrors period_finder.load_csv_lightcurve."""
    if _pf_load_csv is not None:
        return _pf_load_csv(csv_path)
    times, fluxes, errors = [], [], []
    has_errors = False
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(1024)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                times.append(float(row[0]))
                fluxes.append(float(row[1]))
                if len(row) >= 3:
                    errors.append(float(row[2]))
                    has_errors = True
            except (ValueError, IndexError):
                continue
    t = np.array(times, dtype=np.float64)
    v = np.array(fluxes, dtype=np.float64)
    e = np.array(errors, dtype=np.float64) if has_errors else None
    return t, v, e


def load_ensemble_wide_csv(path: str) -> dict:
    """
    Wide-format CSV: column 0 = time, columns 1..N = star magnitudes,
    optional error columns named *_err or every third column if detected.
    """
    times = []
    mag_cols: dict[str, list] = {}
    err_cols: dict[str, list] = {}
    headers: list[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            parts = line.strip().split(",")
            if not parts:
                continue
            if not headers:
                headers = [p.strip() for p in parts]
                if headers[0].lower() in ("time", "jd", "bjd", "date", "t"):
                    star_names = headers[1:]
                else:
                    star_names = [f"star_{i}" for i in range(1, len(headers))]
                    headers = ["time"] + star_names
                for name in star_names:
                    if name.endswith("_err") or name.endswith("_e"):
                        continue
                    mag_cols[name] = []
                    err_cols[name] = []
                continue
            try:
                times.append(float(parts[0]))
            except ValueError:
                continue
            for j, name in enumerate(headers[1:], start=1):
                if j >= len(parts):
                    break
                if name.endswith("_err") or name.endswith("_e"):
                    continue
                try:
                    mag_cols[name].append(float(parts[j]))
                except ValueError:
                    mag_cols[name].append(np.nan)
                err_key = f"{name}_err"
                if err_key in headers:
                    ei = headers.index(err_key)
                    if ei < len(parts):
                        try:
                            err_cols[name].append(float(parts[ei]))
                        except ValueError:
                            err_cols[name].append(np.nan)

    t = np.array(times, dtype=np.float64)
    names = list(mag_cols.keys())
    mags = np.array([mag_cols[n] for n in names], dtype=np.float64)
    errs = np.array([err_cols[n] for n in names], dtype=np.float64)
    if errs.size == 0 or not np.isfinite(errs).any():
        errs = np.ones_like(mags) * 0.01
    else:
        bad = ~np.isfinite(errs) | (errs <= 0)
        errs[bad] = 0.01
    return {"times": t, "names": names, "mags": mags, "errors": errs}


def build_ensemble_from_target_and_comps(
    target_path: str,
    comp_paths: list[str],
    target_name: str = "target",
) -> dict:
    """Stack target CSV + comparison CSVs into [n_stars, n_times] ensemble."""
    t0, m0, e0 = _load_csv_lightcurve(target_path)
    names = [target_name]
    mags = [m0]
    errs = [e0 if e0 is not None else np.ones_like(m0) * 0.01]
    times = t0

    for i, cp in enumerate(comp_paths):
        tc, mc, ec = _load_csv_lightcurve(cp)
        if len(tc) != len(times) or not np.allclose(tc, times, rtol=0, atol=1e-6):
            mc_i = np.interp(times, tc, mc)
            ec_i = (np.interp(times, tc, ec) if ec is not None
                    else np.ones_like(times) * 0.01)
        else:
            mc_i, ec_i = mc, (ec if ec is not None else np.ones_like(mc) * 0.01)
        names.append(f"comp_{i+1}")
        mags.append(mc_i)
        errs.append(ec_i)

    return {
        "times": times,
        "names": names,
        "mags": np.array(mags, dtype=np.float64),
        "errors": np.array(errs, dtype=np.float64),
    }


def residuals_from_magnitudes(mags: np.ndarray) -> np.ndarray:
    """Per-star median-subtracted magnitudes (differential residuals)."""
    r = mags.astype(np.float64).copy()
    med = np.nanmedian(r, axis=1, keepdims=True)
    return r - med


def sysrem(
    residuals: np.ndarray,
    errors: np.ndarray,
    n_systematics: int = 3,
    n_epochs: int = 10,
    exclude_indices: list[int] | None = None,
    log_callback=None,
) -> tuple:
    """
    SYSREM — Tamuz, Mazeh & Zucker 2005, MNRAS 356, 1466.

    Alternating weighted least squares per systematic component k.
    Stars listed in exclude_indices do not contribute to fitting the shared
    trend vectors (transit self-subtraction prevention) but still receive
    subtraction once coefficients are found from comparison stars.
    """
    n_stars, n_times = residuals.shape
    r = residuals.copy().astype(np.float64)
    w = 1.0 / (errors.astype(np.float64) ** 2 + 1e-30)
    exclude = set(exclude_indices or [])
    fit_mask = np.array([i not in exclude for i in range(n_stars)])
    systematics = []
    log = log_callback or (lambda _m: None)

    for it in range(n_systematics):
        a = np.ones(n_times, dtype=np.float64)
        for _ep in range(n_epochs):
            denom_c = np.maximum(np.sum(w[fit_mask] * a[np.newaxis, :] ** 2, axis=1), 1e-30)
            c = np.zeros(n_stars, dtype=np.float64)
            c[fit_mask] = (
                np.sum(w[fit_mask] * r[fit_mask] * a[np.newaxis, :], axis=1) / denom_c
            )

            denom_a = np.maximum(
                np.sum(w[fit_mask] * c[fit_mask, np.newaxis] ** 2, axis=0), 1e-30
            )
            a = np.sum(w[fit_mask] * r[fit_mask] * c[fit_mask, np.newaxis], axis=0) / denom_a

        model = c[:, np.newaxis] * a[np.newaxis, :]
        r -= model
        systematics.append((c.copy(), a.copy()))

        rms_before = float(np.sqrt(np.nanmean(residuals ** 2)))
        rms_after = float(np.sqrt(np.nanmean(r ** 2)))
        log(
            f"  SYSREM k={it + 1}: RMS {rms_before:.5f} → {rms_after:.5f} "
            f"({100 * (1 - rms_after / max(rms_before, 1e-12)):.1f}% reduced)"
        )

    return r, systematics


def sysrem_single(
    times: np.ndarray,
    target_mags: np.ndarray,
    target_err: np.ndarray,
    comp_mags: list,
    comp_errs: list,
    n_systematics: int = 3,
    n_epochs: int = 10,
    log_callback=None,
) -> np.ndarray:
    """
    Detrend a single target using comparison-star systematics only.
    Target is excluded from SYSREM fitting (transit-safe).
    """
    log = log_callback or (lambda _m: None)
    if not comp_mags:
        log("  sysrem_single: no comparison stars — unchanged")
        return target_mags.copy()

    comp_mat = np.array(
        [cf - np.nanmedian(cf) for cf in comp_mags], dtype=np.float64
    )
    err_mat = np.array(comp_errs, dtype=np.float64)
    _, systematics = sysrem(
        comp_mat,
        err_mat,
        n_systematics=n_systematics,
        n_epochs=n_epochs,
        exclude_indices=None,
        log_callback=log,
    )

    corrected = target_mags.astype(np.float64) - np.nanmedian(target_mags)
    t_w = 1.0 / (target_err.astype(np.float64) ** 2 + 1e-30)

    for it, (c_comp, a_j) in enumerate(systematics):
        c_target = float(np.sum(t_w * corrected * a_j) / max(np.sum(t_w * a_j ** 2), 1e-30))
        corrected -= c_target * a_j
        log(f"  sysrem_single systematic {it + 1}: c_target={c_target:.5f}")

    return corrected + np.nanmedian(target_mags)


def tfa_detrend(
    residuals: np.ndarray,
    errors: np.ndarray,
    target_index: int = 0,
    n_neighbors: int = 5,
    log_callback=None,
) -> tuple:
    """
    Trend Filtering Algorithm — Kovacs et al. 2005, MNRAS 356, 557.

    Expresses the target light curve as a weighted linear combination of the
    nearest comparison-star trends (RMS distance in normalized magnitude space).
    """
    n_stars, _n_times = residuals.shape
    r = residuals.astype(np.float64).copy()
    errs = errors.astype(np.float64)
    log = log_callback or (lambda _m: None)

    comp_idx = [i for i in range(n_stars) if i != target_index]
    if not comp_idx:
        return r, {"neighbors": [], "coef": np.array([]), "trend": np.zeros(_n_times)}

    target_lc = r[target_index] - np.nanmedian(r[target_index])
    dists = []
    for i in comp_idx:
        comp_lc = r[i] - np.nanmedian(r[i])
        dist = float(np.sqrt(np.nanmean((target_lc - comp_lc) ** 2)))
        dists.append((dist, i))
    dists.sort(key=lambda x: x[0])
    neighbors = [idx for _, idx in dists[: min(n_neighbors, len(dists))]]

    y = r[target_index]
    x = r[neighbors].T
    w = 1.0 / (errs[target_index] ** 2 + 1e-30)
    xt_w = x.T * w
    n_nb = len(neighbors)
    coef = np.linalg.solve(
        xt_w @ x + 1e-10 * np.eye(n_nb),
        xt_w @ y,
    )
    trend = x @ coef
    offset = np.nanmedian(r[target_index])
    r[target_index] = y - trend + offset

    log(f"  TFA: neighbors={neighbors}  coef={np.round(coef, 4).tolist()}")
    return r, {"neighbors": neighbors, "coef": coef, "trend": trend}


def compute_scatter_metrics(mags: np.ndarray, errors: np.ndarray) -> dict:
    """Weighted RMS and reduced chi-squared per star."""
    r = residuals_from_magnitudes(mags)
    w = 1.0 / (errors ** 2 + 1e-30)
    wrms = []
    chi2 = []
    for i in range(mags.shape[0]):
        ri = r[i]
        wi = w[i]
        wrms.append(float(np.sqrt(np.sum(wi * ri ** 2) / max(np.sum(wi), 1e-30))))
        chi2.append(float(np.sum(wi * ri ** 2) / max(mags.shape[1] - 1, 1)))
    return {"wrms": wrms, "chi2_red": chi2}


def export_detrended_csv(
    path: str,
    times: np.ndarray,
    names: list[str],
    mags: np.ndarray,
    errors: np.ndarray,
) -> str:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# HYPERLOAD sysrem_detrend.py output\n")
        f.write("# time," + ",".join(names) + "\n")
        for t_idx, t in enumerate(times):
            row = [f"{t:.8f}"]
            for s_idx in range(len(names)):
                row.append(f"{mags[s_idx, t_idx]:.6f}")
            f.write(",".join(row) + "\n")
    return path


def _validate_algorithms() -> None:
    """Synthetic correlated-noise test — must reduce RMS after SYSREM."""
    rng = np.random.default_rng(42)
    n_stars, n_times = 12, 60
    t = np.linspace(0, 1, n_times)
    a_true = 0.02 * np.sin(2 * np.pi * 3 * t) + 0.01 * t
    r = rng.normal(0, 0.003, (n_stars, n_times))
    for i in range(n_stars):
        r[i] += (0.4 + 0.05 * i) * a_true
    err = np.ones_like(r) * 0.004

    r0_rms = float(np.sqrt(np.mean(r ** 2)))
    r_clean, _s = sysrem(r, err, n_systematics=1, n_epochs=15)
    r1_rms = float(np.sqrt(np.mean(r_clean ** 2)))
    if r1_rms >= r0_rms * 0.85:
        raise RuntimeError(f"SYSREM validation failed: RMS {r0_rms} -> {r1_rms}")

    target = r[0] + 0.5
    comps = [r[i] + 0.5 for i in range(1, 6)]
    ce = [err[i] for i in range(1, 6)]
    out = sysrem_single(t, target, err[0], comps, ce, n_systematics=1)
    if len(out) != n_times:
        raise RuntimeError("sysrem_single length mismatch")

    r2 = residuals_from_magnitudes(np.vstack([target, *comps]))
    r_tfa, _meta = tfa_detrend(r2, err[:6], target_index=0, n_neighbors=3)
    if r_tfa.shape != r2.shape:
        raise RuntimeError("TFA shape mismatch")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--validate":
    _validate_algorithms()
    print("sysrem_detrend: algorithms OK")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PyQt6 GUI
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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


class DetrendWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, ensemble: dict, params: dict):
        super().__init__()
        self.ensemble = ensemble
        self.params = params
        self._log_lines: list[str] = []

    def _log(self, msg: str):
        self._log_lines.append(msg)

    def run(self):
        try:
            self.progress.emit(5, "Preparing residuals…")
            mags = self.ensemble["mags"]
            errs = self.ensemble["errors"]
            times = self.ensemble["times"]
            names = self.ensemble["names"]

            r0 = residuals_from_magnitudes(mags)
            metrics_before = compute_scatter_metrics(mags, errs)

            method = self.params.get("method", "SYSREM")
            target_idx = int(self.params.get("target_index", 0))
            exclude = self.params.get("exclude_indices") or []
            if self.params.get("exclude_target", True) and target_idx not in exclude:
                exclude = list(set(exclude) | {target_idx})

            self.progress.emit(25, f"Running {method}…")

            if method == "TFA":
                r1 = r0.copy()
                meta = []
                for i in range(mags.shape[0]):
                    if i == target_idx:
                        r1, m = tfa_detrend(
                            r0,
                            errs,
                            target_index=i,
                            n_neighbors=self.params.get("n_neighbors", 5),
                            log_callback=self._log,
                        )
                        meta.append(m)
            elif self.params.get("single_target_mode", False):
                comp_m = [mags[i] for i in range(len(names)) if i != target_idx]
                comp_e = [errs[i] for i in range(len(names)) if i != target_idx]
                m_out = mags.copy()
                m_out[target_idx] = sysrem_single(
                    times,
                    mags[target_idx],
                    errs[target_idx],
                    comp_m,
                    comp_e,
                    n_systematics=self.params.get("n_systematics", 3),
                    n_epochs=self.params.get("n_epochs", 10),
                    log_callback=self._log,
                )
                r1 = residuals_from_magnitudes(m_out)
                systematics = []
                mags_corr = m_out
            else:
                r1, systematics = sysrem(
                    r0,
                    errs,
                    n_systematics=self.params.get("n_systematics", 3),
                    n_epochs=self.params.get("n_epochs", 10),
                    exclude_indices=exclude,
                    log_callback=self._log,
                )
                med = np.nanmedian(mags, axis=1, keepdims=True)
                mags_corr = r1 + med

            if method == "TFA":
                med = np.nanmedian(mags, axis=1, keepdims=True)
                mags_corr = r1 + med
                systematics = meta

            metrics_after = compute_scatter_metrics(mags_corr, errs)
            self.progress.emit(100, "Detrending complete")

            self.finished.emit({
                "times": times,
                "names": names,
                "mags_raw": mags,
                "mags_corr": mags_corr,
                "errors": errs,
                "residuals_raw": r0,
                "residuals_corr": r1,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "systematics": systematics,
                "log": "\n".join(self._log_lines),
                "method": method,
                "target_index": target_idx,
            })
        except Exception as ex:
            self.error.emit(f"{ex}\n\n{traceback.format_exc()}")


class SysremDetrendWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 860)
        self.setStyleSheet(DARK)
        logo = SCRIPT_DIR / "logo.png"
        if logo.exists():
            self.setWindowIcon(QIcon(str(logo)))

        self._ensemble = None
        self._result = None
        self._worker = None
        self._wthread = None
        self._comp_paths: list[str] = []

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
        splitter.setSizes([300, 980])
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
                    50, 50,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            lay.addWidget(lbl)
        v = QVBoxLayout()
        t = QLabel("SYSREM / TFA Detrending")
        t.setObjectName("title_lbl")
        s = QLabel(
            "Tamuz 2005 · Kovacs 2005 · sub-millimag ensemble photometry"
        )
        s.setObjectName("sub_lbl")
        v.addWidget(t)
        v.addWidget(s)
        lay.addLayout(v, 1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _make_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        grp = QGroupBox("Input")
        fl = QFormLayout(grp)
        self.ed_target = QLineEdit()
        btn_t = QPushButton("…")
        btn_t.setFixedWidth(28)
        btn_t.clicked.connect(self._browse_target)
        row = QHBoxLayout()
        row.addWidget(self.ed_target)
        row.addWidget(btn_t)
        fl.addRow("Target / wide CSV:", row)

        self.ed_comps = QLineEdit()
        self.ed_comps.setPlaceholderText("Optional comp CSV folder…")
        btn_c = QPushButton("…")
        btn_c.setFixedWidth(28)
        btn_c.clicked.connect(self._browse_comps)
        row2 = QHBoxLayout()
        row2.addWidget(self.ed_comps)
        row2.addWidget(btn_c)
        fl.addRow("Comp stars:", row2)

        btn_load = QPushButton("Load ensemble")
        btn_load.clicked.connect(self._load_data)
        fl.addRow(btn_load)
        self.lbl_info = QLabel("No data loaded")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#505868;font-size:10px;")
        fl.addRow(self.lbl_info)
        lay.addWidget(grp)

        grp2 = QGroupBox("Method")
        f2 = QFormLayout(grp2)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["SYSREM", "TFA", "SYSREM (single target)"])
        f2.addRow("Algorithm:", self.cmb_method)
        self.sp_nsys = QSpinBox()
        self.sp_nsys.setRange(1, 20)
        self.sp_nsys.setValue(3)
        f2.addRow("Systematics:", self.sp_nsys)
        self.sp_epochs = QSpinBox()
        self.sp_epochs.setRange(1, 50)
        self.sp_epochs.setValue(10)
        f2.addRow("ALS epochs:", self.sp_epochs)
        self.sp_neighbors = QSpinBox()
        self.sp_neighbors.setRange(1, 30)
        self.sp_neighbors.setValue(5)
        f2.addRow("TFA neighbors:", self.sp_neighbors)
        self.sp_target = QSpinBox()
        self.sp_target.setRange(0, 99)
        self.sp_target.setValue(0)
        f2.addRow("Target index:", self.sp_target)
        self.chk_exclude = QCheckBox("Exclude target from SYSREM fit")
        self.chk_exclude.setChecked(True)
        self.chk_exclude.setToolTip(
            "Prevents transit self-subtraction (Tamuz et al. 2005)."
        )
        f2.addRow(self.chk_exclude)
        lay.addWidget(grp2)

        self.btn_run = QPushButton("Run detrending")
        self.btn_run.setObjectName("run")
        self.btn_run.clicked.connect(self._run)
        lay.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export CSV…")
        self.btn_export.clicked.connect(self._export)
        lay.addWidget(self.btn_export)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _make_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        self.canvas_lc = MplCanvas(figsize=(9, 4))
        self.canvas_sys = MplCanvas(figsize=(9, 3))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        tabs.addTab(self.canvas_lc, "Light curves")
        tabs.addTab(self.canvas_sys, "Systematics")
        tabs.addTab(self.log, "Log")
        return tabs

    def _browse_target(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Light curve CSV", str(SCRIPT_DIR), "CSV (*.csv);;All (*.*)"
        )
        if path:
            self.ed_target.setText(path)

    def _browse_comps(self):
        path = QFileDialog.getExistingDirectory(self, "Comparison CSV folder")
        if path:
            self.ed_comps.setText(path)

    def _load_data(self):
        path = self.ed_target.text().strip()
        if not path:
            QMessageBox.warning(self, "Input", "Select a CSV file.")
            return
        try:
            comp_dir = self.ed_comps.text().strip()
            comp_paths = []
            if comp_dir and Path(comp_dir).is_dir():
                comp_paths = sorted(str(p) for p in Path(comp_dir).glob("*.csv"))

            if len(comp_paths) >= 1:
                self._ensemble = build_ensemble_from_target_and_comps(
                    path, comp_paths, target_name="target"
                )
            else:
                self._ensemble = load_ensemble_wide_csv(path)
                if self._ensemble["mags"].shape[0] == 1:
                    t, m, e = _load_csv_lightcurve(path)
                    e = e if e is not None else np.ones_like(m) * 0.01
                    self._ensemble = {
                        "times": t,
                        "names": ["target"],
                        "mags": m[np.newaxis, :],
                        "errors": e[np.newaxis, :],
                    }

            n_s, n_t = self._ensemble["mags"].shape
            self.sp_target.setMaximum(max(0, n_s - 1))
            self.lbl_info.setText(
                f"{n_s} stars × {n_t} epochs  |  "
                f"{', '.join(self._ensemble['names'][:4])}"
                f"{'…' if n_s > 4 else ''}"
            )
            self._plot_preview()
            self.status.update(-1, f"Loaded {n_s}×{n_t} ensemble")
        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))

    def _plot_preview(self):
        if self._ensemble is None:
            return
        times = self._ensemble["times"]
        mags = self._ensemble["mags"]
        self.canvas_lc.fig.clear()
        ax = self.canvas_lc.fig.add_subplot(111)
        _ax(ax, "Raw ensemble (median-subtracted)", "Time", "Δmag")
        for i in range(min(mags.shape[0], 12)):
            r = mags[i] - np.nanmedian(mags[i])
            ax.plot(times, r, lw=0.8, alpha=0.85, label=self._ensemble["names"][i])
        if mags.shape[0] <= 8:
            ax.legend(fontsize=6, facecolor="#0c1018", edgecolor="#1e2840")
        self.canvas_lc.redraw()

    def _run(self):
        if self._ensemble is None:
            QMessageBox.warning(self, "Data", "Load an ensemble first.")
            return
        method = self.cmb_method.currentText()
        params = {
            "method": "SYSREM" if method.startswith("SYSREM") else "TFA",
            "single_target_mode": method == "SYSREM (single target)",
            "n_systematics": self.sp_nsys.value(),
            "n_epochs": self.sp_epochs.value(),
            "n_neighbors": self.sp_neighbors.value(),
            "target_index": self.sp_target.value(),
            "exclude_target": self.chk_exclude.isChecked(),
        }
        self.btn_run.setEnabled(False)
        self._worker = DetrendWorker(self._ensemble, params)
        self._wthread = QThread(self)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _on_progress(self, pct: int, msg: str):
        self.status.update(pct, msg)

    def _on_finished(self, result: dict):
        self.btn_run.setEnabled(True)
        self._result = result
        self.log.append(result.get("log", ""))
        self._plot_result(result)
        self.status.update(100, "Done")

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.log.append(msg)
        QMessageBox.critical(self, "Error", msg)
        self.status.update(-1, "Error")

    def _plot_result(self, result: dict):
        times = result["times"]
        ti = result["target_index"]
        r0 = result["residuals_raw"]
        r1 = result["residuals_corr"]

        self.canvas_lc.fig.clear()
        ax1 = self.canvas_lc.fig.add_subplot(211)
        ax2 = self.canvas_lc.fig.add_subplot(212, sharex=ax1)
        _ax(ax1, f"Before — star {result['names'][ti]}", "Time", "Δmag")
        _ax(ax2, "After detrending", "Time", "Δmag")
        ax1.plot(times, r0[ti], color="#c06030", lw=0.9)
        ax2.plot(times, r1[ti], color="#40c080", lw=0.9)
        self.canvas_lc.redraw()

        self.canvas_sys.fig.clear()
        ax = self.canvas_sys.fig.add_subplot(111)
        _ax(ax, "Removed systematics (a vectors)", "Time", "Amplitude")
        syst = result.get("systematics") or []
        for k, item in enumerate(syst):
            if isinstance(item, tuple) and len(item) == 2:
                _c, a = item
                ax.plot(times, a, lw=1.0, label=f"k={k + 1}")
        if syst:
            ax.legend(fontsize=7, facecolor="#0c1018", edgecolor="#1e2840")
        self.canvas_sys.redraw()

        mb = result["metrics_before"]["wrms"][ti]
        ma = result["metrics_after"]["wrms"][ti]
        self.log.append(f"Target WRMS: {mb:.5f} → {ma:.5f} mag")

    def _export(self):
        if self._result is None:
            QMessageBox.warning(self, "Export", "Run detrending first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save detrended CSV", "detrended.csv", "CSV (*.csv)"
        )
        if not path:
            return
        export_detrended_csv(
            path,
            self._result["times"],
            self._result["names"],
            self._result["mags_corr"],
            self._result["errors"],
        )
        self.status.update(-1, f"Saved {path}")


def launch_sysrem_detrend(parent=None) -> SysremDetrendWindow:
    """Open companion window from another HYPERLOAD script."""
    win = SysremDetrendWindow()
    if parent is not None:
        win.setWindowFlags(win.windowFlags() | Qt.WindowType.Window)
    win.show()
    return win


def main():
    _validate_algorithms()
    app = QApplication.instance() or QApplication(sys.argv)
    win = SysremDetrendWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
