r"""
HYPERLOAD — Transient Real/Bogus Vetter
========================================
Companion to transient_detector.py.
Place in: C:\Users\Marcell\Desktop\Siril Suites\

Usage (standalone):
    python transient_vetter.py
    → Load alerts CSV from transient_detector.py + FITS frames
    → Score each alert and display interactive gallery

Usage (from transient_detector.py — add after deduplicate_alerts):
    from transient_vetter import vet_alerts
    alerts = vet_alerts(alerts, fits_files, psf_fwhm=4.0)
    # Each alert now has: vetting_score (0-100), vetting_grade (REAL/MAYBE/BOGUS)

Rule-based real/bogus scoring:
  The largest source of false positives in image subtraction:
    • Single-pixel cosmic rays     → penalised by area check
    • Satellite / aircraft trails  → penalised by elongation check
    • Hot pixels / column defects  → penalised by area + roundness
    • Frame border artifacts        → penalised by position check
    • Multi-layer false positives   → penalised by layer consistency

  Scoring rules (50 = neutral):
    +25  Confirmed by 2+ detection layers
    +15  SNR ≥ 15
    +8   SNR 8–15
    +20  Area consistent with PSF (not single pixel, not satellite)
    +20  Source is round (ellipticity < 0.3)
    +10  Away from frame border
    +5   Positive flux event
    −40  Single pixel (cosmic ray signature)
    −25  Very elongated (e > 0.6, satellite trail)
    −15  Area >> PSF (likely satellite / bleed column)
    −10  Near frame border (within 20px)
    −10  Low SNR < 5

  REAL  : score ≥ 65
  MAYBE : score 40–64
  BOGUS : score < 40

References:
  Bloom et al. 2012    — real/bogus classification, PASP 124, 1175
  Mahabal et al. 2019  — ZTF real/bogus, PASP 131, 038002

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, csv, traceback, math
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log,"a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\ntransient_vetter.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.ndimage import gaussian_filter

try:
    from astropy.io import fits as astropy_fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit, QProgressBar,
    QGroupBox, QSplitter, QMessageBox, QSizePolicy, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon, QColor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Transient Real/Bogus Vetter"


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE SCORING  (importable by transient_detector.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _ellipticity(stamp: np.ndarray, cx: float, cy: float) -> float:
    """Flux-weighted ellipticity from quadrupole moments. Returns 0 (round) – 1 (line)."""
    h, w = stamp.shape
    yy, xx = np.mgrid[:h, :w].astype(float)
    yy -= cy; xx -= cx
    w_m = np.maximum(stamp - stamp.min(), 0)
    ws  = w_m.sum()
    if ws < 1e-10: return 0.0
    mxx = float((w_m * xx**2).sum() / ws)
    myy = float((w_m * yy**2).sum() / ws)
    mxy = float((w_m * xx * yy).sum() / ws)
    denom = mxx + myy
    if denom < 1e-10: return 0.0
    return float(np.sqrt((mxx - myy)**2 + 4*mxy**2) / denom)


def score_alert(alert: dict,
                image: np.ndarray | None = None,
                psf_fwhm: float = 4.0) -> dict:
    """
    Score a single transient alert 0–100 (higher = more likely real astrophysical event).

    Parameters
    ----------
    alert    : dict from transient_detector.py (must have: x, y, snr, area_px, type)
    image    : 2-D float array of the difference or science frame (for morphology)
               Pass None to skip image-based tests.
    psf_fwhm : expected PSF FWHM in pixels (for area and roundness checks)

    Returns
    -------
    alert dict with additional keys:
        vetting_score : int 0–100
        vetting_grade : 'REAL' | 'MAYBE' | 'BOGUS'
        vetting_flags : list of human-readable rule outcomes
    """
    score = 50
    flags = []

    x   = alert.get('x', 0); y = alert.get('y', 0)
    snr = float(alert.get('snr', 0))
    area= int(alert.get('area_px', 1))
    confirmed = alert.get('confirmed_by_layers', [alert.get('layer', 1)])

    # ── Layer consistency ──────────────────────────────────────────────────
    if len(confirmed) >= 2:
        score += 25
        flags.append(f'✓ Multi-layer confirmed (layers {confirmed}) +25')
    elif len(confirmed) == 1:
        flags.append(f'  Single-layer detection (layer {confirmed[0]})')

    # ── SNR ───────────────────────────────────────────────────────────────
    if snr >= 15:
        score += 15; flags.append(f'✓ High SNR={snr:.1f} +15')
    elif snr >= 8:
        score += 8;  flags.append(f'✓ Good SNR={snr:.1f} +8')
    elif snr < 5:
        score -= 10; flags.append(f'✗ Low SNR={snr:.1f} −10')
    else:
        flags.append(f'  Marginal SNR={snr:.1f}')

    # ── Area vs PSF ───────────────────────────────────────────────────────
    sigma       = psf_fwhm / 2.355
    psf_area_typ= math.pi * sigma**2        # 1-sigma circle
    psf_area_max= math.pi * (3 * sigma)**2  # 3-sigma — anything larger is suspicious

    if area <= 1:
        score -= 40
        flags.append('✗ Single-pixel spike — likely cosmic ray −40')
    elif area < psf_area_typ * 0.3:
        score -= 15
        flags.append(f'✗ Area {area}px << PSF ({psf_area_typ:.0f}px) −15')
    elif area <= psf_area_max:
        score += 20
        flags.append(f'✓ PSF-consistent area {area}px +20')
    else:
        score -= 15
        flags.append(f'✗ Area {area}px >> PSF limit ({psf_area_max:.0f}px) — satellite? −15')

    # ── Morphology from image cutout ──────────────────────────────────────
    if image is not None and image.ndim == 2:
        h, w = image.shape
        ix, iy = int(round(x)), int(round(y))
        box  = max(4, int(psf_fwhm * 4))
        x0   = max(0, ix - box); x1 = min(w, ix + box + 1)
        y0   = max(0, iy - box); y1 = min(h, iy + box + 1)

        if (x1 - x0) > 4 and (y1 - y0) > 4:
            stamp = image[y0:y1, x0:x1].astype(float)
            lx    = ix - x0; ly = iy - y0
            e     = _ellipticity(stamp, lx, ly)
            if e < 0.3:
                score += 20; flags.append(f'✓ Round source (e={e:.2f}) +20')
            elif e < 0.6:
                flags.append(f'  Slightly elongated (e={e:.2f})')
            else:
                score -= 25; flags.append(f'✗ Very elongated (e={e:.2f}) — trail? −25')

        # Border check
        border = 20
        if ix < border or ix > w-border or iy < border or iy > h-border:
            score -= 10; flags.append('✗ Near frame border −10')
        else:
            score += 10; flags.append('✓ Away from frame border +10')
    else:
        flags.append('  (No image supplied — morphology tests skipped)')

    # ── Flux sign ─────────────────────────────────────────────────────────
    if alert.get('type') == 'positive' or float(alert.get('delta_flux', 1)) > 0:
        score += 5; flags.append('✓ Positive flux event +5')
    else:
        score -= 5; flags.append('  Negative flux (brightening in reference?)')

    score = int(np.clip(score, 0, 100))
    if score >= 65:
        grade = 'REAL'
    elif score >= 40:
        grade = 'MAYBE'
    else:
        grade = 'BOGUS'

    return {**alert,
            'vetting_score': score,
            'vetting_grade': grade,
            'vetting_flags': flags}


def vet_alerts(alerts: list[dict],
               fits_files: list[str] | None = None,
               psf_fwhm: float = 4.0,
               log_cb=None) -> list[dict]:
    """
    Score all alerts. Optionally load FITS frames for morphology tests.

    Parameters
    ----------
    alerts     : list of alert dicts from transient_detector.py
    fits_files : FITS frame paths; the first frame is used for morphology
    psf_fwhm   : PSF FWHM in pixels

    Returns
    -------
    Scored alert list (same order), sorted by vetting_score descending.
    """
    image = None
    if fits_files and ASTROPY_OK:
        try:
            data = astropy_fits.getdata(fits_files[0]).astype(float)
            image = data if data.ndim == 2 else data[0]
        except Exception as e:
            if log_cb: log_cb(f"Could not load FITS for morphology: {e}")

    scored = []
    for a in alerts:
        scored.append(score_alert(a, image, psf_fwhm))
        if log_cb:
            log_cb(f"  [{scored[-1]['vetting_grade']:5s}] score={scored[-1]['vetting_score']} "
                   f"SNR={a.get('snr','?')} area={a.get('area_px','?')}px")

    scored.sort(key=lambda a: -a['vetting_score'])
    return scored


def load_alerts_csv(path: str) -> list[dict]:
    """Load alerts from transient_detector.py CSV export."""
    alerts = []
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = {}
                for k, v in row.items():
                    try: a[k] = float(v)
                    except (ValueError, TypeError): a[k] = v
                if 'area_px' in a: a['area_px'] = int(a['area_px'])
                alerts.append(a)
    except Exception as e:
        print(f"Could not load alerts: {e}")
    return alerts


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class VetWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, alerts, fits_files, fwhm):
        super().__init__()
        self.alerts = alerts; self.fits_files = fits_files; self.fwhm = fwhm

    def run(self):
        try:
            self.progress.emit(5, "Loading reference frame…")
            image = None
            if self.fits_files and ASTROPY_OK:
                try:
                    data  = astropy_fits.getdata(self.fits_files[0]).astype(float)
                    image = data if data.ndim == 2 else data[0]
                except Exception: pass

            self.progress.emit(20, f"Scoring {len(self.alerts)} alerts…")
            scored = []
            for i, a in enumerate(self.alerts):
                scored.append(score_alert(a, image, self.fwhm))
                if i % max(1, len(self.alerts)//10) == 0:
                    self.progress.emit(20 + int(70*i/len(self.alerts)),
                        f"Scored {i+1}/{len(self.alerts)}…")

            scored.sort(key=lambda a: -a['vetting_score'])
            n_real  = sum(1 for a in scored if a['vetting_grade']=='REAL')
            n_maybe = sum(1 for a in scored if a['vetting_grade']=='MAYBE')
            n_bogus = sum(1 for a in scored if a['vetting_grade']=='BOGUS')
            self.progress.emit(100,
                f"Done!  REAL={n_real}  MAYBE={n_maybe}  BOGUS={n_bogus}")
            self.finished.emit(scored)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow,QWidget{background:#0f110e;color:#c8d8c0;
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}
QTabWidget::pane{border:1px solid #1e3020;background:#111a10;}
QTabBar::tab{background:#121a10;color:#4a6040;padding:6px 16px;
    border:1px solid #1e3020;border-bottom:none;}
QTabBar::tab:selected{background:#111a10;color:#60e080;
    border-bottom:2px solid #30c050;}
QGroupBox{border:1px solid #1e3020;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#4a6040;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#121a10;color:#80c080;border:1px solid #2a5030;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#1a2818;border-color:#40a050;}
QPushButton#run_btn{background:#0e1e0e;color:#60f060;
    border:1px solid #30c050;font-weight:bold;padding:7px 20px;}
QPushButton#run_btn:hover{background:#142018;}
QSpinBox,QDoubleSpinBox,QComboBox{background:#0c1208;
    border:1px solid #1e3020;border-radius:3px;padding:3px 6px;color:#c0d8b0;}
QTextEdit{background:#080c06;border:1px solid #1e3020;color:#60d060;
    font-family:Consolas,monospace;font-size:10px;}
QTableWidget{background:#080c06;alternate-background-color:#0c1008;
    gridline-color:#1e3020;color:#c0d8b0;}
QTableWidget QHeaderView::section{background:#0c1208;color:#4a6040;
    border:1px solid #1e3020;padding:3px;font-weight:bold;}
QSplitter::handle{background:#1e3020;}
QLabel#title_lbl{font-size:17px;font-weight:bold;color:#60f060;padding:6px;}
QLabel#sub_lbl{font-size:9px;color:#2a4028;padding:0 8px 4px;}
QProgressBar{background:#0c1208;border:1px solid #2a5030;border-radius:3px;height:8px;}
QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #204030,stop:1 #40c060);border-radius:2px;}
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8,5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor='#0c1008')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
    def redraw(self): self.fig.tight_layout(pad=0.4); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#080c06')
    ax.set_title(title, color='#40c050', fontsize=9, pad=3)
    ax.set_xlabel(xl, color='#2a4028', fontsize=8)
    ax.set_ylabel(yl, color='#2a4028', fontsize=8)
    ax.tick_params(colors='#2a4028', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e3020')


class VetterApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1360, 880)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._alerts    = []
        self._scored    = []
        self._fits_files= []
        self._worker    = self._wthread = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._controls())
        sp.addWidget(self._tabs())
        sp.setSizes([260,1100])
        rl.addWidget(sp,1)
        rl.addWidget(self._footer())

    def _header(self):
        hdr = QWidget(); hdr.setFixedHeight(60)
        hdr.setStyleSheet("background:#080c06;border-bottom:1px solid #1e3020;")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12,0,12,0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                44,44,Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Transient Real/Bogus Vetter"); t.setObjectName("title_lbl")
        s = QLabel("Rule-based morphology scoring  ·  Cosmic ray / satellite rejection  "
                   "·  Multi-layer confirmation  ·  Companion to transient_detector.py")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(5)

        # Load
        grp = QGroupBox("Load Alerts"); gl = QVBoxLayout(grp)
        self.alert_lbl = QLabel("No alerts loaded")
        self.alert_lbl.setStyleSheet("color:#2a4028;font-size:10px;")
        b1 = QPushButton("📂  Load alerts CSV…"); b1.clicked.connect(self._load_csv)
        b1.setObjectName("run_btn")
        b2 = QPushButton("📋  Load synthetic example"); b2.clicked.connect(self._load_example)
        gl.addWidget(self.alert_lbl); gl.addWidget(b1); gl.addWidget(b2)
        lay.addWidget(grp)

        # FITS frames for morphology
        grp2 = QGroupBox("FITS Frames (morphology)"); gl2 = QVBoxLayout(grp2)
        self.fits_lbl = QLabel("No frames — morphology tests skipped")
        self.fits_lbl.setStyleSheet("color:#2a4028;font-size:10px;")
        self.fits_lbl.setWordWrap(True)
        b3 = QPushButton("📂  Load FITS frame(s)…"); b3.clicked.connect(self._load_fits)
        gl2.addWidget(self.fits_lbl); gl2.addWidget(b3)
        lay.addWidget(grp2)

        # Parameters
        grp3 = QGroupBox("Scoring Parameters"); g3 = QGridLayout(grp3)
        g3.addWidget(QLabel("PSF FWHM (px):"),0,0)
        self.sp_fwhm = QDoubleSpinBox(); self.sp_fwhm.setRange(1,30); self.sp_fwhm.setValue(4.0)
        g3.addWidget(self.sp_fwhm,0,1)
        g3.addWidget(QLabel("REAL threshold:"),1,0)
        self.sp_real = QSpinBox(); self.sp_real.setRange(50,95); self.sp_real.setValue(65)
        g3.addWidget(self.sp_real,1,1)
        g3.addWidget(QLabel("MAYBE threshold:"),2,0)
        self.sp_maybe = QSpinBox(); self.sp_maybe.setRange(20,64); self.sp_maybe.setValue(40)
        g3.addWidget(self.sp_maybe,2,1)
        lay.addWidget(grp3)

        self.btn_run = QPushButton("▶  Vet All Alerts")
        self.btn_run.setObjectName("run_btn"); self.btn_run.setFixedHeight(36)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_exp = QPushButton("💾  Export scored CSV…")
        self.btn_exp.setEnabled(False); self.btn_exp.clicked.connect(self._export)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_exp)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color:#40c050;font-size:10px;")
        self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_table(),   "📋  Alert Table")
        self.tabs.addTab(self._tab_gallery(),  "🔭  Gallery")
        self.tabs.addTab(self._tab_stats(),    "📊  Score Distribution")
        self.tabs.addTab(self._tab_log(),      "📝  Log")
        return self.tabs

    def _tab_table(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.table = QTableWidget(0,8)
        self.table.setHorizontalHeaderLabels(
            ["Grade","Score","SNR","Area (px)","Layer(s)","X","Y","Top flag"])
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.cellClicked.connect(self._on_row_click)
        lay.addWidget(self.table)
        return w

    def _tab_gallery(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.gallery_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.gallery_plot)
        return w

    def _tab_stats(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.stats_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.stats_plot)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    def _footer(self):
        foot = QWidget(); foot.setFixedHeight(22)
        foot.setStyleSheet("background:#080c06;border-top:1px solid #1e3020;")
        lay = QHBoxLayout(foot); lay.setContentsMargins(6,0,6,0)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color:#2a4028;font-size:10px;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(200,12); self.pbar.setVisible(False)
        lay.addWidget(self.status_lbl,1); lay.addWidget(self.pbar)
        return foot

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,"Load alerts CSV","","CSV (*.csv);;All (*)")
        if not path: return
        self._alerts = load_alerts_csv(path)
        self.alert_lbl.setText(f"{len(self._alerts)} alerts from {Path(path).name}")
        self.alert_lbl.setStyleSheet("color:#40c050;font-size:10px;")
        self.btn_run.setEnabled(True)
        self._log(f"Loaded {len(self._alerts)} alerts from {path}")

    def _load_example(self):
        """Generate synthetic alerts covering all cases."""
        import random; rng2 = random.Random(42)
        cases = [
            {'x':120,'y':130,'snr':18,'area_px':14,'type':'positive',
             'delta_flux':2000,'layer':1,'method':'zogy',
             'confirmed_by_layers':[1,2]},
            {'x':200,'y':80,'snr':22,'area_px':1,'type':'positive',
             'delta_flux':800,'layer':1,'method':'diff'},
            {'x':180,'y':200,'snr':14,'area_px':180,'type':'positive',
             'delta_flux':5000,'layer':1,'method':'diff'},
            {'x':8,'y':15,'snr':9,'area_px':10,'type':'positive',
             'delta_flux':300,'layer':1,'method':'diff'},
            {'x':300,'y':300,'snr':11,'area_px':16,'type':'positive',
             'delta_flux':1500,'layer':1,'method':'diff',
             'confirmed_by_layers':[1,2]},
            {'x':60,'y':220,'snr':7,'area_px':8,'type':'negative',
             'delta_flux':-400,'layer':1,'method':'diff'},
            {'x':400,'y':150,'snr':25,'area_px':11,'type':'positive',
             'delta_flux':3000,'layer':2,'method':'zogy'},
        ]
        self._alerts = [dict(a) for a in cases]
        self.alert_lbl.setText(f"{len(self._alerts)} synthetic alerts (mixed real/bogus)")
        self.alert_lbl.setStyleSheet("color:#40c050;font-size:10px;")
        self.btn_run.setEnabled(True)
        self._log("Loaded synthetic example: mix of real, cosmic rays, satellite, border")

    def _load_fits(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,"Load FITS frames","","FITS (*.fits *.fit *.fts);;All (*)")
        if not paths: return
        self._fits_files = paths
        self.fits_lbl.setText(f"{len(paths)} frames loaded")
        self.fits_lbl.setStyleSheet("color:#40c050;font-size:10px;")
        self._log(f"Loaded {len(paths)} FITS frames for morphology")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if not self._alerts: return
        self.btn_run.setEnabled(False); self.btn_exp.setEnabled(False)
        self._wthread = QThread(self)
        self._worker  = VetWorker(self._alerts, self._fits_files,
                                   self.sp_fwhm.value())
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()

    def _on_progress(self, pct, msg):
        self.status_lbl.setText(msg)
        self.pbar.setVisible(True); self.pbar.setValue(pct)
        if pct >= 100:
            QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}"); self.btn_run.setEnabled(True)

    def _on_finished(self, scored):
        self._scored = scored
        self.btn_run.setEnabled(True); self.btn_exp.setEnabled(True)

        n_r = sum(1 for a in scored if a['vetting_grade']=='REAL')
        n_m = sum(1 for a in scored if a['vetting_grade']=='MAYBE')
        n_b = sum(1 for a in scored if a['vetting_grade']=='BOGUS')
        self.summary_lbl.setText(f"REAL {n_r} / MAYBE {n_m} / BOGUS {n_b}")

        self._log(f"\n{'='*45}")
        self._log(f"✓  Vetting complete")
        self._log(f"   REAL:  {n_r}   MAYBE: {n_m}   BOGUS: {n_b}")
        for a in scored[:5]:
            self._log(f"   [{a['vetting_grade']:5s}] score={a['vetting_score']} "
                      f"SNR={a.get('snr','?')}  area={a.get('area_px','?')}px")
        self._log(f"{'='*45}\n")

        self._fill_table()
        self._draw_gallery()
        self._draw_stats()
        self.tabs.setCurrentIndex(0)

    # ── Plots & table ─────────────────────────────────────────────────────────

    def _fill_table(self):
        grade_colors = {'REAL':'#40c050','MAYBE':'#c0a030','BOGUS':'#c04030'}
        self.table.setRowCount(len(self._scored))
        for row, a in enumerate(self._scored):
            grade = a['vetting_grade']; score = a['vetting_score']
            layers = a.get('confirmed_by_layers', [a.get('layer','?')])
            top_flag = (a.get('vetting_flags') or ['—'])[0]
            vals = [grade, str(score), f"{a.get('snr','?'):.1f}",
                    str(a.get('area_px','?')),
                    str(layers), f"{a.get('x',0):.1f}", f"{a.get('y',0):.1f}",
                    top_flag[:60]]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if col == 0:
                    item.setForeground(QColor(grade_colors.get(grade,'white')))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def _on_row_click(self, row, col):
        """Show flags for selected alert in log."""
        if row >= len(self._scored): return
        a = self._scored[row]
        self._log(f"\nAlert at ({a.get('x','?'):.1f},{a.get('y','?'):.1f}) "
                  f"score={a['vetting_score']} [{a['vetting_grade']}]:")
        for f in a.get('vetting_flags',[]):
            self._log(f"  {f}")
        self.tabs.setCurrentIndex(3)

    def _draw_gallery(self):
        fig = self.gallery_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c1008')
        n = min(len(self._scored), 12)
        if n == 0: return
        cols = min(n, 4); rows = math.ceil(n / cols)
        axes = fig.subplots(rows, cols) if n > 1 else [[fig.add_subplot(111)]]
        if rows == 1 and n > 1: axes = [axes]
        ax_flat = [a for row in axes for a in (row if hasattr(row,'__iter__') else [row])]

        grade_colors = {'REAL':'#40c050','MAYBE':'#c0a030','BOGUS':'#c04030'}
        for i, (ax, a) in enumerate(zip(ax_flat, self._scored[:n])):
            grade = a['vetting_grade']; score = a['vetting_score']
            col = grade_colors.get(grade,'white')
            _ax(ax, f"[{grade}] s={score}  SNR={a.get('snr','?'):.0f}")
            ax.set_title(f"[{grade}] score={score}  SNR={a.get('snr','?'):.0f}",
                         color=col, fontsize=7, pad=2)
            # Show info as text since we may not have real cutouts
            ax.text(0.5, 0.5,
                    f"x={a.get('x',0):.0f} y={a.get('y',0):.0f}\n"
                    f"area={a.get('area_px','?')}px\n"
                    f"layer={a.get('layer','?')}",
                    transform=ax.transAxes, ha='center', va='center',
                    color=col, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(1.5)

        for ax in ax_flat[n:]: ax.set_visible(False)
        self.gallery_plot.redraw()

    def _draw_stats(self):
        fig = self.stats_plot.fig; fig.clear()
        fig.patch.set_facecolor('#0c1008')
        axes = fig.subplots(1,2)

        scores = [a['vetting_score'] for a in self._scored]
        _ax(axes[0], "Score distribution","Score","Count")
        n_r = sum(1 for s in scores if s >= self.sp_real.value())
        n_m = sum(1 for s in scores if self.sp_maybe.value() <= s < self.sp_real.value())
        n_b = sum(1 for s in scores if s < self.sp_maybe.value())
        axes[0].hist(scores, bins=20, range=(0,100),
                     color='#206030', edgecolor='#1e3020', alpha=0.9)
        axes[0].axvline(self.sp_real.value(),  color='#40c050', lw=1.5, ls='--', label='REAL')
        axes[0].axvline(self.sp_maybe.value(), color='#c0a030', lw=1.5, ls='--', label='MAYBE')
        axes[0].legend(fontsize=7, facecolor='#0c1008', edgecolor='#1e3020',
                       labelcolor='white')

        _ax(axes[1], "Grade breakdown","Grade","Count")
        grades = ['REAL','MAYBE','BOGUS']
        counts = [n_r, n_m, n_b]
        colors = ['#40c050','#c0a030','#c04030']
        bars = axes[1].bar(grades, counts, color=colors, edgecolor='#0c1008', alpha=0.9)
        for bar, count in zip(bars, counts):
            if count > 0:
                axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                             str(count), ha='center', va='bottom',
                             color='white', fontsize=9)

        self.stats_plot.redraw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        if not self._scored: return
        path, _ = QFileDialog.getSaveFileName(
            self,"Export scored alerts","transients_scored.csv","CSV (*.csv);;All (*)")
        if not path: return
        import csv
        keys = ['x','y','snr','area_px','type','layer','delta_flux',
                'method','vetting_score','vetting_grade']
        with open(path,'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader(); w.writerows(self._scored)
        self._log(f"✓ Exported {len(self._scored)} alerts to {path}")

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VetterApp(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
