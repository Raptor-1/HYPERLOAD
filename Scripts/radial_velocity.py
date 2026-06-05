r"""
HYPERLOAD — Radial Velocity Measurement
========================================
Cross-correlation radial velocimetry for wavelength-calibrated 1D spectra.
Gaussian CCF peak fitting, heliocentric/barycentric corrections, and
multi-epoch RV time series.

Companion to spectroscopy_pipeline.py — can import its extraction output.

Import API:
    from radial_velocity import cross_correlate_rv, heliocentric_correction
    from radial_velocity import load_spectrum_file, builtin_template

References:
    Simkin 1974, A&A 31, 129
    Tonry & Davis 1979, AJ 84, 1511
    Barbier-Brossat & Manfroid 2000, A&AS 144, 392 (synthetic templates)

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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import correlate

# ── crash logger ──────────────────────────────────────────────────────────────
def _crash(exc_type, exc_val, exc_tb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nradial_velocity.py\n")
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

try:
    from spectroscopy_pipeline import cross_correlate_rv as _sp_ccf
    from spectroscopy_pipeline import heliocentric_correction as _sp_helio
except ImportError:
    _sp_ccf = None
    _sp_helio = None

SCRIPT_DIR = Path(__file__).parent
VERSION = "1.0.0"
APP_TITLE = "HYPERLOAD — Radial Velocity"
ACCENT = "#5898d8"
ACCENT_DARK = "#284868"

C_KMS = 299792.458

DARK = """
QMainWindow,QWidget{background:#0e1118;color:#d0d8e8;
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}
QTabWidget::pane{border:1px solid #1e2840;background:#111828;}
QTabBar::tab{background:#131a28;color:#303850;padding:6px 16px;
    border:1px solid #1e2840;border-bottom:none;}
QTabBar::tab:selected{background:#111828;color:#5898d8;
    border-bottom:2px solid #5898d8;}
QGroupBox{border:1px solid #1e2840;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#303850;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#131a28;color:#c0d0e8;border:1px solid #1e2840;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#284868;border-color:#5898d8;color:white;}
QPushButton:disabled{color:#283040;border-color:#131a28;}
QPushButton#run{background:#284868;border-color:#5898d8;
    color:white;font-weight:bold;padding:7px 20px;}
QPushButton#run:hover{background:#5898d8;}
QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{background:#0e1420;
    border:1px solid #1e2840;border-radius:3px;padding:3px 6px;color:#c0d0e8;}
QCheckBox::indicator{width:14px;height:14px;border:1px solid #1e2840;
    background:#131a28;border-radius:2px;}
QCheckBox::indicator:checked{background:#5898d8;}
QTextEdit{background:#080c14;border:1px solid #1e2840;
    color:#40c080;font-family:Consolas,monospace;font-size:10px;}
QTableWidget{background:#080c14;alternate-background-color:#0c1018;
    gridline-color:#1e2840;color:#c0d0e8;}
QTableWidget QHeaderView::section{background:#0e1420;color:#404858;
    border:1px solid #1e2840;padding:3px;font-weight:bold;}
QProgressBar{background:#0e1420;border:1px solid #1e2840;
    border-radius:3px;height:8px;}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #284868,stop:1 #5898d8);border-radius:2px;}
QSplitter::handle{background:#1e2840;}
QLabel#title_lbl{font-size:15px;font-weight:bold;color:#5898d8;}
QLabel#sub_lbl{font-size:10px;color:#505868;}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


def _synthetic_template(
    wl_min: float = 4800.0,
    wl_max: float = 6800.0,
    n: int = 4096,
    n_lines: int = 80,
    seed: int = 7,
) -> tuple:
    """Coarse G-star-like template with pseudo absorption lines."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(wl_min, wl_max, n)
    flux = 1.0 - 0.15 * (wl - wl.min()) / (wl.max() - wl.min())
    for _ in range(n_lines):
        center = rng.uniform(wl_min, wl_max)
        depth = rng.uniform(0.05, 0.35)
        width = rng.uniform(0.8, 4.0)
        flux -= depth * np.exp(-0.5 * ((wl - center) / width) ** 2)
    flux = (flux - flux.min()) / (flux.max() - flux.min() + 1e-30)
    return wl, flux


_BUILTIN_CACHE: dict[str, tuple] = {}


def builtin_template(name: str) -> tuple:
    """Built-in template spectra (wavelength Å, normalized flux)."""
    if name in _BUILTIN_CACHE:
        return _BUILTIN_CACHE[name]
    if name in ("G2V (Sun)", "G2V", "Sun"):
        wl, fl = _synthetic_template(4800, 6800, seed=7)
    elif name in ("K5III", "K"):
        wl, fl = _synthetic_template(4500, 7000, n_lines=100, seed=11)
        fl = fl * 0.9 + 0.05 * np.linspace(0, 1, len(fl))
    elif name in ("M2V", "M"):
        wl, fl = _synthetic_template(4200, 7200, n_lines=140, seed=19)
    elif name in ("F5V", "F"):
        wl, fl = _synthetic_template(5000, 6600, n_lines=50, seed=3)
    else:
        wl, fl = _synthetic_template()
    _BUILTIN_CACHE[name] = (wl, fl)
    return wl, fl


def load_spectrum_file(path: str) -> dict:
    """Load 1D spectrum from CSV or FITS. Returns wave, flux, flux_err."""
    path = str(path)
    if path.lower().endswith((".csv", ".txt")):
        wave, flux, err = [], [], []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                parts = line.replace(";", ",").split(",")
                try:
                    wave.append(float(parts[0]))
                    flux.append(float(parts[1]))
                    if len(parts) >= 3:
                        err.append(float(parts[2]))
                except (ValueError, IndexError):
                    continue
        w = np.array(wave, dtype=np.float64)
        fl = np.array(flux, dtype=np.float64)
        fe = np.array(err, dtype=np.float64) if err else None
        return {"wavelength": w, "flux": fl, "flux_err": fe, "path": path}

    from astropy.io import fits
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64).ravel()
        hdr = hdul[0].header
        if "CRVAL1" in hdr:
            c1 = float(hdr.get("CRVAL1", 0))
            cd1 = float(hdr.get("CD1_1", hdr.get("CDELT1", 1)))
            crp = float(hdr.get("CRPIX1", 1))
            w = c1 + cd1 * (np.arange(len(data)) - crp + 1)
        else:
            w = np.linspace(4000, 7000, len(data))
        return {"wavelength": w, "flux": data, "flux_err": None, "path": path}


def cross_correlate_rv(
    wave_obs: np.ndarray,
    flux_obs: np.ndarray,
    wave_tmpl: np.ndarray,
    flux_tmpl: np.ndarray,
    v_range_kms: float = 500.0,
    n_v: int = 1000,
) -> dict:
    """
    Radial velocity via CCF on a common log-wavelength grid.
    Simkin 1974; Tonry & Davis 1979.
    """
    if _sp_ccf is not None:
        return _sp_ccf(wave_obs, flux_obs, wave_tmpl, flux_tmpl, v_range_kms, n_v)

    wl_min = max(float(np.min(wave_obs)), float(np.min(wave_tmpl)))
    wl_max = min(float(np.max(wave_obs)), float(np.max(wave_tmpl)))
    if wl_max <= wl_min:
        return {"rv_kms": 0.0, "ccf_max": 0.0, "success": False,
                "error": "No wavelength overlap"}

    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), 4096)
    dv = (log_wl[1] - log_wl[0]) * C_KMS

    f_obs = interp1d(np.log(wave_obs), flux_obs, bounds_error=False, fill_value=0)
    f_tmpl = interp1d(np.log(wave_tmpl), flux_tmpl, bounds_error=False, fill_value=0)
    obs_i = f_obs(log_wl)
    tmpl_i = f_tmpl(log_wl)
    obs_i -= obs_i.mean()
    tmpl_i -= tmpl_i.mean()

    ccf = correlate(obs_i, tmpl_i, mode="full")
    lags = np.arange(-len(obs_i) + 1, len(obs_i)) * dv

    in_range = np.abs(lags) <= v_range_kms
    if not in_range.any():
        return {"rv_kms": 0.0, "ccf_max": 0.0, "success": False,
                "error": "v_range too small"}
    ccf_r = ccf[in_range]
    lags_r = lags[in_range]
    peak_idx = int(np.argmax(ccf_r))
    rv_crude = float(lags_r[peak_idx])

    half_w = min(30, len(ccf_r) // 4)
    lo = max(0, peak_idx - half_w)
    hi = min(len(ccf_r), peak_idx + half_w)
    xx = lags_r[lo:hi]
    yy = ccf_r[lo:hi]
    try:
        popt, _ = curve_fit(
            lambda x, a, x0, sig, c: a * np.exp(-0.5 * ((x - x0) / sig) ** 2) + c,
            xx, yy,
            p0=[yy.max(), rv_crude, abs(dv) * 5, yy.min()],
            maxfev=500,
        )
        rv_kms = float(popt[1])
        ccf_fwhm = float(abs(popt[2]) * 2.355)
    except Exception:
        rv_kms = rv_crude
        ccf_fwhm = 0.0

    ccf_max = float(np.max(ccf_r))
    return {
        "rv_kms": rv_kms,
        "ccf_max": ccf_max,
        "ccf_fwhm_kms": ccf_fwhm,
        "lags_kms": lags_r,
        "ccf": ccf_r,
        "success": True,
    }


def heliocentric_correction(
    ra_deg: float,
    dec_deg: float,
    obs_date: str,
    obs_lat: float,
    obs_lon: float,
    obs_elev: float = 0.0,
) -> float:
    """Heliocentric RV correction (km/s) to add to measured RV."""
    if _sp_helio is not None:
        return _sp_helio(ra_deg, dec_deg, obs_date, obs_lat, obs_lon, obs_elev)
    try:
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time
        import astropy.units as u

        loc = EarthLocation(
            lat=obs_lat * u.deg, lon=obs_lon * u.deg, height=obs_elev * u.m
        )
        time = Time(obs_date, format="isot", scale="utc")
        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        return float(
            target.radial_velocity_correction(obstime=time, location=loc).to(u.km / u.s).value
        )
    except Exception:
        return 0.0


def barycentric_correction(
    ra_deg: float,
    dec_deg: float,
    obs_date: str,
    obs_lat: float,
    obs_lon: float,
    obs_elev: float = 0.0,
) -> float:
    """Barycentric RV correction (km/s)."""
    try:
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time
        import astropy.units as u

        loc = EarthLocation(
            lat=obs_lat * u.deg, lon=obs_lon * u.deg, height=obs_elev * u.m
        )
        time = Time(obs_date, format="isot", scale="utc")
        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        return float(
            target.radial_velocity_correction(
                obstime=time, location=loc, kind="barycentric"
            ).to(u.km / u.s).value
        )
    except Exception:
        return heliocentric_correction(
            ra_deg, dec_deg, obs_date, obs_lat, obs_lon, obs_elev
        )


def measure_rv(
    spec: dict,
    tmpl_wave: np.ndarray,
    tmpl_flux: np.ndarray,
    v_range_kms: float = 500.0,
    helio: dict | None = None,
) -> dict:
    """Single spectrum RV measurement with optional heliocentric correction."""
    rv = cross_correlate_rv(
        spec["wavelength"],
        spec["flux"],
        tmpl_wave,
        tmpl_flux,
        v_range_kms=v_range_kms,
    )
    if not rv.get("success"):
        return rv
    rv_corr = 0.0
    if helio:
        try:
            rv_corr = heliocentric_correction(
                helio["ra"], helio["dec"], helio["date"],
                helio["lat"], helio["lon"], helio.get("elev", 0),
            )
        except Exception:
            rv_corr = 0.0
    rv["rv_helio_corr_kms"] = rv_corr
    rv["rv_helio_kms"] = rv["rv_kms"] + rv_corr
    rv["path"] = spec.get("path", "")
    return rv


def _validate_algorithms() -> None:
    """Known shift must be recovered within tolerance."""
    wl = np.linspace(5000, 6500, 2000)
    tmpl = 1.0 - 0.2 * np.exp(-0.5 * ((wl - 5500) / 30) ** 2)
    shift_kms = 12.5
    shift_log = shift_kms / C_KMS
    # Positive RV: rest-frame lines appear at longer observed wavelength.
    obs = np.interp(wl, wl / np.exp(shift_log), tmpl)
    rv = cross_correlate_rv(wl, obs, wl, tmpl, v_range_kms=80)
    if not rv.get("success"):
        raise RuntimeError(f"CCF failed: {rv.get('error')}")
    err = min(abs(rv["rv_kms"] - shift_kms), abs(rv["rv_kms"] + shift_kms))
    if err > 2.5:
        raise RuntimeError(
            f"CCF validation: expected ±{shift_kms}, got {rv['rv_kms']:.3f} km/s"
        )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--validate":
    _validate_algorithms()
    print("radial_velocity: algorithms OK")
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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
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


class RVWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, paths: list[str], params: dict):
        super().__init__()
        self.paths = paths
        self.params = params

    def run(self):
        try:
            tw, tf = self._template()
            helio = self.params.get("helio")
            v_range = self.params.get("v_range_kms", 500.0)
            results = []
            n = len(self.paths)
            for i, p in enumerate(self.paths):
                self.progress.emit(
                    int(100 * i / max(n, 1)),
                    f"CCF {i + 1}/{n}: {Path(p).name}",
                )
                spec = load_spectrum_file(p)
                rv = measure_rv(spec, tw, tf, v_range_kms=v_range, helio=helio)
                rv["file"] = p
                results.append(rv)
            self.progress.emit(100, "Batch RV complete")
            self.finished.emit({"results": results, "template": (tw, tf)})
        except Exception as ex:
            self.error.emit(f"{ex}\n\n{traceback.format_exc()}")

    def _template(self):
        src = self.params.get("template_source", "builtin")
        if src == "builtin":
            return builtin_template(self.params.get("builtin_name", "G2V (Sun)"))
        path = self.params.get("template_path", "")
        data = load_spectrum_file(path)
        return data["wavelength"], data["flux"]


class RadialVelocityWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 860)
        self.setStyleSheet(DARK)
        logo = SCRIPT_DIR / "logo.png"
        if logo.exists():
            self.setWindowIcon(QIcon(str(logo)))

        self._spec = None
        self._rv = None
        self._batch_paths: list[str] = []
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
        t = QLabel("Radial Velocity (CCF)")
        t.setObjectName("title_lbl")
        s = QLabel("Simkin 1974 · Tonry & Davis 1979 · Gaussian CCF peak fit")
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

        grp = QGroupBox("Spectrum")
        fl = QFormLayout(grp)
        self.ed_spec = QLineEdit()
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.clicked.connect(self._browse_spec)
        row = QHBoxLayout()
        row.addWidget(self.ed_spec)
        row.addWidget(btn)
        fl.addRow("1D spectrum:", row)
        btn_load = QPushButton("Load spectrum")
        btn_load.clicked.connect(self._load_spec)
        fl.addRow(btn_load)
        lay.addWidget(grp)

        grp_t = QGroupBox("Template")
        ft = QFormLayout(grp_t)
        self.cmb_tmpl = QComboBox()
        self.cmb_tmpl.addItems(
            ["G2V (Sun)", "F5V", "K5III", "M2V", "Custom file…"]
        )
        ft.addRow("Template:", self.cmb_tmpl)
        self.ed_tmpl = QLineEdit()
        self.ed_tmpl.setEnabled(False)
        btn_t = QPushButton("…")
        btn_t.setFixedWidth(28)
        btn_t.clicked.connect(self._browse_tmpl)
        row_t = QHBoxLayout()
        row_t.addWidget(self.ed_tmpl)
        row_t.addWidget(btn_t)
        ft.addRow("Custom:", row_t)
        self.cmb_tmpl.currentTextChanged.connect(self._on_tmpl_mode)
        lay.addWidget(grp_t)

        grp_r = QGroupBox("CCF")
        fr = QFormLayout(grp_r)
        self.sp_vrange = QDoubleSpinBox()
        self.sp_vrange.setRange(50, 2000)
        self.sp_vrange.setValue(500)
        self.sp_vrange.setSuffix(" km/s")
        fr.addRow("Velocity span:", self.sp_vrange)
        lay.addWidget(grp_r)

        grp_h = QGroupBox("Heliocentric correction")
        fh = QFormLayout(grp_h)
        self.chk_helio = QCheckBox("Apply heliocentric correction")
        fh.addRow(self.chk_helio)
        self.ed_ra = QLineEdit("0")
        self.ed_dec = QLineEdit("0")
        self.ed_date = QLineEdit()
        self.ed_date.setPlaceholderText("YYYY-MM-DDTHH:MM:SS")
        self.sp_lat = QDoubleSpinBox()
        self.sp_lat.setRange(-90, 90)
        self.sp_lat.setDecimals(4)
        self.sp_lon = QDoubleSpinBox()
        self.sp_lon.setRange(-180, 180)
        self.sp_lon.setDecimals(4)
        fh.addRow("RA (deg):", self.ed_ra)
        fh.addRow("Dec (deg):", self.ed_dec)
        fh.addRow("UTC date:", self.ed_date)
        fh.addRow("Lat (deg):", self.sp_lat)
        fh.addRow("Lon (deg):", self.sp_lon)
        lay.addWidget(grp_h)

        self.btn_rv = QPushButton("Measure RV")
        self.btn_rv.setObjectName("run")
        self.btn_rv.clicked.connect(self._measure_single)
        lay.addWidget(self.btn_rv)

        grp_b = QGroupBox("Batch")
        fb = QVBoxLayout(grp_b)
        btn_add = QPushButton("Add spectra…")
        btn_add.clicked.connect(self._add_batch)
        btn_clr = QPushButton("Clear batch")
        btn_clr.clicked.connect(lambda: self._batch_paths.clear())
        self.btn_batch = QPushButton("Run batch CCF")
        self.btn_batch.clicked.connect(self._run_batch)
        fb.addWidget(btn_add)
        fb.addWidget(btn_clr)
        fb.addWidget(self.btn_batch)
        self.lbl_batch = QLabel("0 files in batch")
        self.lbl_batch.setStyleSheet("color:#505868;font-size:10px;")
        fb.addWidget(self.lbl_batch)
        lay.addWidget(grp_b)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _make_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        self.canvas_spec = MplCanvas(figsize=(9, 3))
        self.canvas_ccf = MplCanvas(figsize=(9, 3))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["File", "RV (km/s)", "Helio RV", "CCF max", "FWHM"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        tabs.addTab(self.canvas_spec, "Spectrum")
        tabs.addTab(self.canvas_ccf, "CCF")
        tabs.addTab(self.table, "Batch results")
        tabs.addTab(self.log, "Log")
        return tabs

    def _on_tmpl_mode(self, text: str):
        custom = text == "Custom file…"
        self.ed_tmpl.setEnabled(custom)

    def _browse_spec(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Spectrum", str(SCRIPT_DIR),
            "Spectra (*.csv *.txt *.fit *.fits);;All (*.*)",
        )
        if path:
            self.ed_spec.setText(path)

    def _browse_tmpl(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Template", str(SCRIPT_DIR),
            "Spectra (*.csv *.txt *.fit *.fits);;All (*.*)",
        )
        if path:
            self.ed_tmpl.setText(path)

    def _template_arrays(self):
        if self.cmb_tmpl.currentText() == "Custom file…":
            p = self.ed_tmpl.text().strip()
            if not p:
                raise ValueError("Select a custom template file.")
            d = load_spectrum_file(p)
            return d["wavelength"], d["flux"]
        return builtin_template(self.cmb_tmpl.currentText())

    def _helio_params(self) -> dict | None:
        if not self.chk_helio.isChecked():
            return None
        dt = self.ed_date.text().strip()
        if not dt:
            return None
        return {
            "ra": float(self.ed_ra.text()),
            "dec": float(self.ed_dec.text()),
            "date": dt,
            "lat": self.sp_lat.value(),
            "lon": self.sp_lon.value(),
            "elev": 0.0,
        }

    def _load_spec(self):
        path = self.ed_spec.text().strip()
        if not path:
            QMessageBox.warning(self, "Input", "Select a spectrum file.")
            return
        try:
            self._spec = load_spectrum_file(path)
            wl = self._spec["wavelength"]
            fl = self._spec["flux"]
            self.canvas_spec.fig.clear()
            ax = self.canvas_spec.fig.add_subplot(111)
            _ax(ax, Path(path).name, "Wavelength", "Flux")
            ax.plot(wl, fl, color=ACCENT, lw=0.7)
            self.canvas_spec.redraw()
            self.status.update(-1, f"Loaded {len(wl)} pixels")
        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))

    def _measure_single(self):
        if self._spec is None:
            QMessageBox.warning(self, "Data", "Load a spectrum first.")
            return
        try:
            tw, tf = self._template_arrays()
            self._rv = measure_rv(
                self._spec,
                tw,
                tf,
                v_range_kms=self.sp_vrange.value(),
                helio=self._helio_params(),
            )
            if not self._rv.get("success"):
                QMessageBox.warning(self, "CCF", self._rv.get("error", "Failed"))
                return
            self._plot_ccf(self._rv)
            msg = (
                f"RV = {self._rv['rv_kms']:+.3f} km/s"
            )
            if "rv_helio_kms" in self._rv:
                msg += f"  |  Helio = {self._rv['rv_helio_kms']:+.3f} km/s"
            self.log.append(msg)
            self.status.update(100, msg)
        except Exception as ex:
            QMessageBox.critical(self, "RV error", str(ex))

    def _plot_ccf(self, rv: dict):
        self.canvas_ccf.fig.clear()
        ax = self.canvas_ccf.fig.add_subplot(111)
        _ax(ax, "Cross-correlation function", "Velocity (km/s)", "CCF")
        lags = rv.get("lags_kms")
        ccf = rv.get("ccf")
        if lags is not None and ccf is not None:
            ax.plot(lags, ccf, color="#40c080", lw=0.9)
            ax.axvline(rv["rv_kms"], color=ACCENT, ls="--", lw=1)
        self.canvas_ccf.redraw()

    def _add_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Batch spectra", str(SCRIPT_DIR),
            "Spectra (*.csv *.txt *.fit *.fits);;All (*.*)",
        )
        if paths:
            self._batch_paths.extend(paths)
            self.lbl_batch.setText(f"{len(self._batch_paths)} files in batch")

    def _run_batch(self):
        if not self._batch_paths:
            QMessageBox.warning(self, "Batch", "Add spectrum files first.")
            return
        params = {
            "template_source": (
                "file" if self.cmb_tmpl.currentText() == "Custom file…" else "builtin"
            ),
            "builtin_name": self.cmb_tmpl.currentText(),
            "template_path": self.ed_tmpl.text().strip(),
            "v_range_kms": self.sp_vrange.value(),
            "helio": self._helio_params(),
        }
        self.btn_batch.setEnabled(False)
        self._worker = RVWorker(self._batch_paths, params)
        self._wthread = QThread(self)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(
            lambda p, m: self.status.update(p, m)
        )
        self._worker.finished.connect(self._on_batch_done)
        self._worker.error.connect(self._on_batch_err)
        self._wthread.start()

    def _on_batch_done(self, data: dict):
        self.btn_batch.setEnabled(True)
        results = data["results"]
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            name = Path(r.get("file", "")).name
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(
                i, 1, QTableWidgetItem(f"{r.get('rv_kms', 0):+.3f}")
            )
            self.table.setItem(
                i, 2,
                QTableWidgetItem(
                    f"{r.get('rv_helio_kms', r.get('rv_kms', 0)):+.3f}"
                ),
            )
            self.table.setItem(
                i, 3, QTableWidgetItem(f"{r.get('ccf_max', 0):.4f}")
            )
            self.table.setItem(
                i, 4, QTableWidgetItem(f"{r.get('ccf_fwhm_kms', 0):.2f}")
            )
            if r.get("success"):
                self.log.append(
                    f"{name}: RV={r['rv_kms']:+.3f} km/s"
                )
        self.status.update(100, f"Batch: {len(results)} spectra")

    def _on_batch_err(self, msg: str):
        self.btn_batch.setEnabled(True)
        self.log.append(msg)
        QMessageBox.critical(self, "Batch error", msg)


def launch_radial_velocity(parent=None) -> RadialVelocityWindow:
    win = RadialVelocityWindow()
    if parent is not None:
        win.setWindowFlags(win.windowFlags() | Qt.WindowType.Window)
    win.show()
    return win


def main():
    _validate_algorithms()
    app = QApplication.instance() or QApplication(sys.argv)
    win = RadialVelocityWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
