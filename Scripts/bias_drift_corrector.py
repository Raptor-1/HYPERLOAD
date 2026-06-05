r"""
bias_drift_corrector.py  —  HYPERLOAD CMOS Bias Drift Corrector
================================================================
CMOS sensors (IMX455, IMX571, IMX294, IMX533 etc.) exhibit a slowly
drifting bias level across a session. This is distinct from the fixed
bias offset corrected by a master bias — it is a temporal drift driven
by sensor temperature changes, electronics warm-up, and power supply
variations. If uncorrected, it introduces a systematic additive error
in every calibrated light frame.

Method:
    1. Measure the bias level in each light frame from an overscan region
       or from a designated dark corner of the frame.
    2. Fit a polynomial model P(t) to the bias vs. frame-index or timestamp.
    3. Subtract P(t) − P(0) from each light frame to remove the drift
       while preserving the mean bias level.

Bias level estimation options:
    a. Overscan strip  — if your camera / spectrograph uses one
    b. Corner median   — median of a configurable corner region
                         (avoids stars, nebulosity)
    c. Pre-computed    — load a CSV of (frame_index, bias_level)
                         e.g. from a separate bias-frame sequence

References:
    Janesick 2001, Scientific Charge-Coupled Devices (SPIE Press)
    Widenhorn et al. 2002, SPIE 4669, 193
    Cloudy Nights thread on CMOS bias drift (2021)

Place in: Siril Suites folder
Run via:  Siril → Scripts → bias_drift_corrector
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("scipy")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")

import os, sys, glob, csv, threading, traceback
from datetime import datetime

import numpy as np
from scipy.ndimage import median_filter
from astropy.io import fits as astropy_fits

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QRadioButton, QButtonGroup, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# ── Theme ────────────────────────────────────────────────────────────────────
SIRIL_BG="#1e2128";SIRIL_BG2="#252930";SIRIL_BG3="#2d3340"
SIRIL_ACCENT="#4a9eff";SIRIL_ACCENT2="#2d6abf";SIRIL_TEXT="#dde3ee"
SIRIL_TEXT_DIM="#7a8499";SIRIL_BORDER="#3a4055";SIRIL_SUCCESS="#4caf7d"
SIRIL_WARNING="#e8c46a";SIRIL_ERROR="#cc4444";SIRIL_SECTION="#9db4d0"
SIRIL_DRIFT="#f4a261"  # amber for drift theme

STYLESHEET = f"""
QMainWindow,QWidget{{background:{SIRIL_BG};color:{SIRIL_TEXT};
  font-family:"Segoe UI",sans-serif;font-size:9pt;}}
QGroupBox{{border:1px solid {SIRIL_BORDER};border-radius:5px;margin-top:8px;
  padding:6px;font-weight:bold;color:{SIRIL_SECTION};}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}
QTabWidget::pane{{border:1px solid {SIRIL_BORDER};background:{SIRIL_BG};}}
QTabBar::tab{{background:{SIRIL_BG2};color:{SIRIL_TEXT_DIM};padding:6px 14px;
  border:1px solid {SIRIL_BORDER};border-bottom:none;border-radius:4px 4px 0 0;}}
QTabBar::tab:selected{{background:{SIRIL_BG};color:{SIRIL_DRIFT};font-weight:bold;}}
QPushButton{{background:{SIRIL_BG3};color:{SIRIL_TEXT};border:1px solid {SIRIL_BORDER};
  border-radius:4px;padding:5px 12px;min-height:22px;}}
QPushButton:hover{{background:{SIRIL_ACCENT2};border-color:{SIRIL_ACCENT};}}
QPushButton[objectName="primary"]{{background:{SIRIL_ACCENT2};color:white;
  font-weight:bold;border-color:{SIRIL_ACCENT};}}
QPushButton[objectName="drift"]{{background:#3a2010;color:{SIRIL_DRIFT};
  font-weight:bold;border-color:{SIRIL_DRIFT};}}
QPushButton[objectName="danger"]{{background:#4a1e1e;color:#e07070;border-color:#803030;}}
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox{{background:{SIRIL_BG2};color:{SIRIL_TEXT};
  border:1px solid {SIRIL_BORDER};border-radius:4px;padding:3px 6px;}}
QProgressBar{{background:{SIRIL_BG2};border:1px solid {SIRIL_BORDER};
  border-radius:4px;text-align:center;}}
QProgressBar::chunk{{background:{SIRIL_DRIFT};border-radius:3px;}}
QPlainTextEdit{{background:{SIRIL_BG2};color:{SIRIL_TEXT};
  border:1px solid {SIRIL_BORDER};border-radius:4px;
  font-family:"Courier New",monospace;font-size:8pt;}}
QLabel[objectName="dim"]{{color:{SIRIL_TEXT_DIM};}}
QLabel[objectName="ok"]{{color:{SIRIL_SUCCESS};}}
QLabel[objectName="warn"]{{color:{SIRIL_WARNING};}}
QCheckBox,QRadioButton{{spacing:6px;}}
QScrollArea{{border:none;}}
"""

# ── Bias level measurement ───────────────────────────────────────────────────

def measure_bias_corner(data: np.ndarray, corner: str = "top-left",
                         size: int = 64) -> float:
    h, w = (data.shape[-2], data.shape[-1]) if data.ndim == 3 else data.shape
    lum  = data[0] if data.ndim == 3 else data
    corners = {
        "top-left":     lum[:size, :size],
        "top-right":    lum[:size, w-size:],
        "bottom-left":  lum[h-size:, :size],
        "bottom-right": lum[h-size:, w-size:],
    }
    patch = corners.get(corner, lum[:size, :size])
    # Robust: use 5th percentile to avoid stars
    return float(np.percentile(patch, 5))


def measure_bias_overscan(data: np.ndarray, overscan_x0: int, overscan_x1: int,
                           overscan_y0: int, overscan_y1: int) -> float:
    lum = data[0] if data.ndim == 3 else data
    strip = lum[overscan_y0:overscan_y1, overscan_x0:overscan_x1]
    return float(np.median(strip))


def fit_drift_polynomial(indices: np.ndarray, levels: np.ndarray,
                          deg: int = 2) -> tuple:
    """Fit polynomial drift model. Returns (coeffs, model_values, rms)."""
    coeffs = np.polyfit(indices, levels, deg)
    model  = np.polyval(coeffs, indices)
    rms    = float(np.sqrt(np.mean((levels - model)**2)))
    return coeffs, model, rms

# ── Canvas ───────────────────────────────────────────────────────────────────

class DriftCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4), facecolor=SIRIL_BG)
        self.ax  = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(SIRIL_BG2)
        for sp in self.ax.spines.values(): sp.set_color(SIRIL_BORDER)
        super().__init__(self.fig)
        self.setParent(parent)

    def show_drift(self, indices, levels, model=None, corrected=None,
                    rms=None, deg=None):
        self.ax.clear(); self.ax.set_facecolor(SIRIL_BG2)
        self.ax.plot(indices, levels, ".", color=SIRIL_DRIFT, ms=4,
                     alpha=0.8, label="Measured bias")
        if model is not None:
            self.ax.plot(indices, model, color=SIRIL_ACCENT, lw=1.5,
                         label=f"Polynomial fit (deg={deg})")
        if corrected is not None:
            self.ax.plot(indices, corrected, ".", color=SIRIL_SUCCESS,
                         ms=4, alpha=0.6, label="After correction")
        title = f"Bias drift  RMS={rms:.3f} ADU" if rms is not None else "Bias drift"
        self.ax.set_title(title, color=SIRIL_DRIFT, fontsize=9)
        self.ax.set_xlabel("Frame index", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.set_ylabel("Bias level (ADU)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax.legend(fontsize=7, facecolor=SIRIL_BG3, edgecolor=SIRIL_BORDER,
                        labelcolor=SIRIL_TEXT)
        self.ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.3)
        self.draw()

# ── Worker ───────────────────────────────────────────────────────────────────

class DriftWorker(QThread):
    progress     = pyqtSignal(int, int, str)
    log_line     = pyqtSignal(str)
    drift_ready  = pyqtSignal(list, list, list, float, int)  # idx,levels,model,rms,deg
    finished     = pyqtSignal(dict)

    def __init__(self, cfg, cancel_event, parent=None):
        super().__init__(parent)
        self.cfg = cfg; self._cancel = cancel_event

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.log_line.emit(f"ERROR: {e}\n{traceback.format_exc()}")
            self.finished.emit({"success": False, "error": str(e)})

    def _run(self):
        cfg = self.cfg; log = self.log_line.emit
        log("═══ HyperLoad Bias Drift Corrector ═══")
        files = sorted(
            glob.glob(os.path.join(cfg["folder"], "*.fit")) +
            glob.glob(os.path.join(cfg["folder"], "*.fits")) +
            glob.glob(os.path.join(cfg["folder"], "*.fts")))
        if not files:
            self.finished.emit({"success": False,
                                "error": "No FITS files found"}); return
        N = len(files)
        log(f"Found {N} FITS files")

        # ── Step 1: measure bias in each frame ────────────────────────────────
        indices = []; levels = []
        for i, path in enumerate(files):
            if self._cancel.is_set(): break
            self.progress.emit(i, N, f"Measuring frame {i+1}/{N}")
            try:
                data = astropy_fits.getdata(path).astype(np.float64)
                if cfg["measure_mode"] == "corner":
                    lvl = measure_bias_corner(data, cfg["corner"],
                                               cfg["corner_size"])
                else:
                    lvl = measure_bias_overscan(
                        data, cfg["ov_x0"], cfg["ov_x1"],
                        cfg["ov_y0"], cfg["ov_y1"])
                indices.append(i); levels.append(lvl)
            except Exception as e:
                log(f"  Skip {os.path.basename(path)}: {e}")

        if len(indices) < cfg["poly_deg"] + 2:
            self.finished.emit({"success": False,
                                "error": "Too few frames for polynomial fit"})
            return

        arr_idx = np.array(indices, dtype=float)
        arr_lvl = np.array(levels, dtype=float)
        log(f"Bias range: {arr_lvl.min():.2f}–{arr_lvl.max():.2f} ADU  "
            f"drift={arr_lvl.max()-arr_lvl.min():.2f} ADU")

        # ── Step 2: fit polynomial ─────────────────────────────────────────────
        deg = cfg["poly_deg"]
        coeffs, model, rms = fit_drift_polynomial(arr_idx, arr_lvl, deg)
        log(f"Fit: degree={deg}  RMS={rms:.4f} ADU")
        corrections = model - model[0]  # correction relative to first frame
        self.drift_ready.emit(indices, levels, model.tolist(), rms, deg)

        # ── Step 3: apply corrections ──────────────────────────────────────────
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        n_done = 0
        for i, (path, corr) in enumerate(zip(files, corrections)):
            if self._cancel.is_set(): break
            self.progress.emit(N + i, 2*N,
                               f"Correcting frame {i+1}/{N}")
            try:
                with astropy_fits.open(path) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    hdr  = hdul[0].header.copy()
                data -= float(corr)
                hdr["BIASDRIFT"] = (float(corr),
                                    "Bias drift correction applied [ADU]")
                hdr["HISTORY"]   = "HyperLoad bias drift correction"
                out_path = os.path.join(out_dir, os.path.basename(path))
                astropy_fits.PrimaryHDU(data=data, header=hdr).writeto(
                    out_path, overwrite=True)
                n_done += 1
            except Exception as e:
                log(f"  Skip {os.path.basename(path)}: {e}")

        # Save drift report CSV
        report_path = os.path.join(out_dir, "bias_drift_report.csv")
        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "filename", "bias_measured",
                        "model", "correction_ADU"])
            for i, (path, lvl, mod, corr) in enumerate(
                    zip(files, levels, model, corrections)):
                w.writerow([i, os.path.basename(path),
                            f"{lvl:.4f}", f"{mod:.4f}", f"{corr:.4f}"])
        log(f"Report: {report_path}")
        log(f"\n✓ {n_done}/{N} frames corrected  "
            f"max_correction={abs(corrections).max():.3f} ADU")
        self.progress.emit(2*N, 2*N, "Done")
        self.finished.emit({"success": True, "n_done": n_done,
                            "max_correction": float(abs(corrections).max()),
                            "rms": rms, "report": report_path})

# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bias Drift Corrector  —  HYPERLOAD  —  Siril")
        self.resize(1000, 720)
        self._worker = None; self._cancel = threading.Event()
        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        header = QWidget(); header.setFixedHeight(52)
        header.setStyleSheet(
            f"background:{SIRIL_BG2};border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl = QLabel("🔩  Bias Drift Corrector")
        lbl.setStyleSheet(
            f"color:{SIRIL_DRIFT};font-size:13pt;font-weight:bold;"
            f"background:transparent;")
        sub = QLabel("Polynomial drift model  ·  Corner/overscan measurement  ·  "
                     "CMOS temporal bias compensation  ·  Janesick 2001")
        sub.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM};font-size:9pt;background:transparent;")
        hl.addWidget(lbl); hl.addWidget(sub); hl.addStretch()
        root.addWidget(header)

        tabs = QTabWidget(); root.addWidget(tabs, 1)
        tabs.addTab(self._build_settings(), "⚙  Settings")
        tabs.addTab(self._build_preview(),  "📈  Drift Preview")
        tabs.addTab(self._build_log(),      "📋  Log")

        bottom = QWidget(); bottom.setFixedHeight(48)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2};border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 6, 10, 6)
        self._prog = QProgressBar(); self._prog.setFixedHeight(10)
        bl.addWidget(self._prog, 1)
        self._btn_run = QPushButton("▶  Correct Bias Drift")
        self._btn_run.setObjectName("drift")
        self._btn_run.setMinimumHeight(34); self._btn_run.setMinimumWidth(180)
        self._btn_run.clicked.connect(self._run)
        bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setMinimumHeight(34)
        self._btn_cancel.setEnabled(False); self._btn_cancel.clicked.connect(self._cancel_run)
        bl.addWidget(self._btn_cancel)
        self._status = QLabel("Ready")
        self._status.setObjectName("dim")
        self._status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status, 1)
        root.addWidget(bottom)

    def _build_settings(self):
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        grp_in = QGroupBox("Light frames folder")
        il = QHBoxLayout(grp_in)
        self._edit_folder = QLineEdit(); self._edit_folder.setPlaceholderText("Folder with FITS light frames…")
        btn_f = QPushButton("Browse…"); btn_f.setFixedWidth(80)
        btn_f.clicked.connect(self._browse_folder)
        il.addWidget(self._edit_folder); il.addWidget(btn_f)
        lay.addWidget(grp_in)

        grp_meas = QGroupBox("Bias level measurement method")
        ml = QVBoxLayout(grp_meas)
        self._rb_corner   = QRadioButton("Corner median  — use a dark corner of each frame")
        self._rb_overscan = QRadioButton("Overscan strip  — use a physical overscan region")
        self._rb_corner.setChecked(True)
        self._bg = QButtonGroup()
        self._bg.addButton(self._rb_corner); self._bg.addButton(self._rb_overscan)
        self._rb_corner.toggled.connect(self._on_method)
        for rb in [self._rb_corner, self._rb_overscan]: ml.addWidget(rb)

        self._grp_corner = QGroupBox("Corner settings")
        cf = QFormLayout(self._grp_corner)
        self._cmb_corner = QComboBox()
        self._cmb_corner.addItems(["top-left","top-right","bottom-left","bottom-right"])
        cf.addRow("Corner:", self._cmb_corner)
        self._spin_corner_size = QSpinBox()
        self._spin_corner_size.setRange(16, 512); self._spin_corner_size.setValue(64)
        self._spin_corner_size.setSuffix(" px")
        cf.addRow("Region size:", self._spin_corner_size)
        lbl_c = QLabel("Must be a region free of stars/nebulosity in all frames.\n"
                        "Use the corner opposite to the optical axis for best results.")
        lbl_c.setObjectName("dim"); lbl_c.setWordWrap(True); cf.addRow(lbl_c)
        ml.addWidget(self._grp_corner)

        self._grp_overscan = QGroupBox("Overscan region (pixels)")
        of = QFormLayout(self._grp_overscan)
        self._spin_ov_x0 = QSpinBox(); self._spin_ov_x0.setRange(0,9999); self._spin_ov_x0.setValue(4100)
        self._spin_ov_x1 = QSpinBox(); self._spin_ov_x1.setRange(0,9999); self._spin_ov_x1.setValue(4144)
        self._spin_ov_y0 = QSpinBox(); self._spin_ov_y0.setRange(0,9999); self._spin_ov_y0.setValue(0)
        self._spin_ov_y1 = QSpinBox(); self._spin_ov_y1.setRange(0,9999); self._spin_ov_y1.setValue(2822)
        of.addRow("X start:", self._spin_ov_x0); of.addRow("X end:", self._spin_ov_x1)
        of.addRow("Y start:", self._spin_ov_y0); of.addRow("Y end:", self._spin_ov_y1)
        self._grp_overscan.setVisible(False)
        ml.addWidget(self._grp_overscan)
        lay.addWidget(grp_meas)

        grp_fit = QGroupBox("Drift model")
        ff = QFormLayout(grp_fit)
        self._spin_deg = QSpinBox(); self._spin_deg.setRange(1,6); self._spin_deg.setValue(2)
        ff.addRow("Polynomial degree:", self._spin_deg)
        lbl_d = QLabel("1=linear drift  2=quadratic (typical)  3+=complex drift\n"
                        "Higher degree fits the data better but risks overfitting.")
        lbl_d.setObjectName("dim"); lbl_d.setWordWrap(True); ff.addRow(lbl_d)
        lay.addWidget(grp_fit)

        grp_out = QGroupBox("Output folder")
        outl = QHBoxLayout(grp_out)
        self._edit_out = QLineEdit(); self._edit_out.setPlaceholderText("Output folder (corrected frames saved here)…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(80)
        btn_out.clicked.connect(lambda: self._edit_out.setText(
            QFileDialog.getExistingDirectory(self, "Output folder") or self._edit_out.text()))
        outl.addWidget(self._edit_out); outl.addWidget(btn_out)
        lay.addWidget(grp_out)
        lay.addStretch()
        return w

    def _build_preview(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(4,4,4,4)
        self._canvas = DriftCanvas(); lay.addWidget(self._canvas)
        lbl = QLabel("Drift preview populated after running correction")
        lbl.setObjectName("dim"); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        return w

    def _build_log(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(6,6,6,6)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._log.clear)
        lay.addWidget(btn_clr); lay.addWidget(self._log)
        return w

    def _on_method(self):
        self._grp_corner.setVisible(self._rb_corner.isChecked())
        self._grp_overscan.setVisible(self._rb_overscan.isChecked())

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select lights folder")
        if d:
            self._edit_folder.setText(d)
            if not self._edit_out.text():
                self._edit_out.setText(os.path.join(d, "bias_corrected"))

    def _get_cfg(self):
        folder = self._edit_folder.text().strip()
        out    = self._edit_out.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Select a FITS folder."); return None
        if not out:
            QMessageBox.warning(self, "No output", "Set an output folder."); return None
        return {
            "folder":      folder,
            "output_dir":  out,
            "measure_mode": "corner" if self._rb_corner.isChecked() else "overscan",
            "corner":      self._cmb_corner.currentText(),
            "corner_size": self._spin_corner_size.value(),
            "ov_x0": self._spin_ov_x0.value(), "ov_x1": self._spin_ov_x1.value(),
            "ov_y0": self._spin_ov_y0.value(), "ov_y1": self._spin_ov_y1.value(),
            "poly_deg": self._spin_deg.value(),
        }

    def _run(self):
        cfg = self._get_cfg()
        if cfg is None: return
        self._cancel.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._prog.setValue(0)
        self._status.setText("Running…")
        self._worker = DriftWorker(cfg, self._cancel)
        self._worker.progress.connect(lambda s,t,m: (
            self._prog.setRange(0,t), self._prog.setValue(s),
            self._status.setText(m)))
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.drift_ready.connect(self._on_drift_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_run(self):
        self._cancel.set(); self._btn_cancel.setEnabled(False)

    def _on_drift_ready(self, idx, lvl, model, rms, deg):
        self._canvas.show_drift(idx, lvl, model=model, rms=rms, deg=deg)

    def _on_finished(self, result):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        if result.get("success"):
            n = result["n_done"]; mc = result["max_correction"]; rms = result["rms"]
            msg = f"✓  {n} frames corrected  max_corr={mc:.3f} ADU  fit_RMS={rms:.3f} ADU"
            self._status.setText(msg)
            self._status.setStyleSheet(f"color:{SIRIL_SUCCESS};font-size:9pt;")
        else:
            self._status.setText(f"✗ {result.get('error','')}")
            self._status.setStyleSheet(f"color:{SIRIL_ERROR};font-size:9pt;")

def main():
    import traceback as _tb
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_log.txt")
    def crash_handler(et, ev, etb):
        with open(log_path, "a") as f:
            f.write(f"\n{'='*50}\nCRASH {datetime.now()}\n"
                    + "".join(_tb.format_exception(et, ev, etb)))
        sys.__excepthook__(et, ev, etb)
    sys.excepthook = crash_handler
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    MainWindow().show()
    app.exec()

if __name__ == "__main__":
    main()
