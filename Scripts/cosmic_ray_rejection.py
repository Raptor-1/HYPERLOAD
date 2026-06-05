r"""
cosmic_ray_rejection.py  —  HYPERLOAD Cosmic Ray Rejection (LACosmic)
======================================================================
L.A.Cosmic: Laplacian Edge Detection Cosmic Ray Rejection.
van Dokkum 2001, PASP 113, 1420  (doi:10.1086/323894)

The algorithm:
  1. Subtract sky background
  2. Laplacian of the image (fine-structure enhancer)
  3. Divide by a noise model → significance map S
  4. Pixels with S > sigclip AND not flagged as objects are CRs
  5. Grow the mask by 1 pixel (dilation)
  6. Replace CR pixels with interpolated background
  Iterate until no new CRs found or max iterations reached.

Uses astropy.ccdproc.cosmicray_lacosmic (Python port of the original IDL).
Supports: single FITS, batch folder, multi-extension FITS.

Reference: van Dokkum 2001, PASP 113, 1420
Place in: Siril Suites folder
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("astropy")
s.ensure_installed("ccdproc")
s.ensure_installed("matplotlib")

import os, sys, glob, threading, traceback
from datetime import datetime

import numpy as np
from astropy.io import fits as astropy_fits
from astropy.nddata import CCDData
import astropy.units as u

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QRadioButton,
    QButtonGroup, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ── Theme ─────────────────────────────────────────────────────────────────────
BG="#1e2128";BG2="#252930";BG3="#2d3340";ACC="#4a9eff";ACC2="#2d6abf"
TXT="#dde3ee";DIM="#7a8499";BRD="#3a4055";OK="#4caf7d";WRN="#e8c46a"
ERR="#cc4444";SEC="#9db4d0";CR="#e07070"

SS=f"""
QMainWindow,QWidget{{background:{BG};color:{TXT};font-family:"Segoe UI",sans-serif;font-size:9pt;}}
QGroupBox{{border:1px solid {BRD};border-radius:5px;margin-top:8px;padding:6px;font-weight:bold;color:{SEC};}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}
QTabWidget::pane{{border:1px solid {BRD};background:{BG};}}
QTabBar::tab{{background:{BG2};color:{DIM};padding:6px 14px;border:1px solid {BRD};border-bottom:none;border-radius:4px 4px 0 0;}}
QTabBar::tab:selected{{background:{BG};color:{CR};font-weight:bold;}}
QPushButton{{background:{BG3};color:{TXT};border:1px solid {BRD};border-radius:4px;padding:5px 12px;min-height:22px;}}
QPushButton:hover{{background:{ACC2};border-color:{ACC};}}
QPushButton[objectName="primary"]{{background:{ACC2};color:white;font-weight:bold;border-color:{ACC};}}
QPushButton[objectName="cr"]{{background:#3a1a1a;color:{CR};font-weight:bold;border-color:{CR};}}
QPushButton[objectName="danger"]{{background:#4a1e1e;color:#e07070;border-color:#803030;}}
QLineEdit,QSpinBox,QDoubleSpinBox{{background:{BG2};color:{TXT};border:1px solid {BRD};border-radius:4px;padding:3px 6px;}}
QProgressBar{{background:{BG2};border:1px solid {BRD};border-radius:4px;text-align:center;}}
QProgressBar::chunk{{background:{CR};border-radius:3px;}}
QPlainTextEdit{{background:{BG2};color:{TXT};border:1px solid {BRD};border-radius:4px;font-family:"Courier New",monospace;font-size:8pt;}}
QLabel[objectName="dim"]{{color:{DIM};}}QLabel[objectName="ok"]{{color:{OK};}}
QScrollArea{{border:none;}}
"""

# ── Canvas ────────────────────────────────────────────────────────────────────

class CRCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 4), facecolor=BG)
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor(BG2)
            for sp in ax.spines.values(): sp.set_color(BRD)
        super().__init__(self.fig); self.setParent(parent)

    def show(self, original, cleaned, cr_mask):
        def _s(d): lo,hi=np.percentile(d,[0.5,99.9]); return np.clip((d-lo)/(hi-lo+1e-10),0,1)
        def _ds(d, mx=600):
            f=max(1,max(d.shape[-2],d.shape[-1])//mx)
            return d[...,::f,::f] if d.ndim==3 else d[::f,::f]
        for ax,img,title,cmap,col in [
            (self.ax1, original, "Original",  "gray",    ACC),
            (self.ax2, cr_mask.astype(float), "CR mask", "hot", CR),
            (self.ax3, cleaned,  "Cleaned",   "gray",    OK),
        ]:
            ax.clear(); ax.set_facecolor(BG2)
            disp = _ds(img)
            if cmap == "hot":
                ax.imshow(disp, cmap=cmap, origin="lower", aspect="equal", interpolation="nearest")
            else:
                ax.imshow(_s(disp), cmap=cmap, origin="lower", aspect="equal", interpolation="nearest")
            n_cr = int(cr_mask.sum()) if cmap == "hot" else 0
            ax.set_title(f"{title}" + (f"  ({n_cr:,} px)" if cmap=="hot" else ""),
                         color=col, fontsize=9)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values(): sp.set_color(BRD)
        self.fig.tight_layout(pad=0.3); self.draw()

# ── Worker ────────────────────────────────────────────────────────────────────

class CRWorker(QThread):
    progress   = pyqtSignal(int, int, str)
    log_line   = pyqtSignal(str)
    preview    = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    finished   = pyqtSignal(dict)

    def __init__(self, cfg, cancel_event, parent=None):
        super().__init__(parent)
        self.cfg=cfg; self._cancel=cancel_event

    def run(self):
        try: self._run()
        except Exception as e:
            self.log_line.emit(f"ERROR: {e}\n{traceback.format_exc()}")
            self.finished.emit({"success":False,"error":str(e)})

    def _run(self):
        from ccdproc import cosmicray_lacosmic
        cfg=self.cfg; log=self.log_line.emit
        log("═══ HyperLoad Cosmic Ray Rejection (L.A.Cosmic) ═══")
        log(f"  sigclip={cfg['sigclip']}  sigfrac={cfg['sigfrac']}  "
            f"objlim={cfg['objlim']}  niter={cfg['niter']}")

        files = [cfg["input_path"]] if cfg["mode"]=="single" else sorted(
            glob.glob(os.path.join(cfg["folder"],"*.fit")) +
            glob.glob(os.path.join(cfg["folder"],"*.fits")) +
            glob.glob(os.path.join(cfg["folder"],"*.fts")))
        if not files:
            self.finished.emit({"success":False,"error":"No files"}); return

        os.makedirs(cfg["output_dir"], exist_ok=True)
        n_total=0; total_cr=0

        for i, path in enumerate(files):
            if self._cancel.is_set(): break
            self.progress.emit(i, len(files), f"{os.path.basename(path)}")
            try:
                with astropy_fits.open(path) as hdul:
                    raw  = hdul[0].data.astype(np.float32)
                    hdr  = hdul[0].header.copy()
                is_rgb = (raw.ndim==3 and raw.shape[0]==3)

                def _process_channel(ch):
                    ccd = CCDData(ch.astype(float), unit=u.adu)
                    cleaned_ccd, cr_mask = cosmicray_lacosmic(
                        ccd,
                        sigclip=cfg["sigclip"],
                        sigfrac=cfg["sigfrac"],
                        objlim=cfg["objlim"],
                        gain=cfg["gain"],
                        readnoise=cfg["readnoise"],
                        niter=cfg["niter"],
                        sepmed=cfg["sepmed"],
                        cleantype=cfg["cleantype"],
                        fsmode=cfg["fsmode"],
                    )
                    return np.array(cleaned_ccd.data).astype(np.float32), np.array(cr_mask)

                if is_rgb:
                    cleaned_channels=[]; masks=[]
                    for c in range(3):
                        cl, mk = _process_channel(raw[c])
                        cleaned_channels.append(cl); masks.append(mk)
                    cleaned = np.stack(cleaned_channels, axis=0)
                    cr_mask = np.any(np.stack(masks,axis=0), axis=0)
                else:
                    lum = raw[0] if raw.ndim==3 else raw
                    cl, cr_mask = _process_channel(lum)
                    cleaned = cl

                n_cr = int(cr_mask.sum())
                total_cr += n_cr
                pct = 100*n_cr/max(cr_mask.size,1)
                log(f"  {os.path.basename(path)}: {n_cr:,} CR pixels ({pct:.3f}%)")

                # Preview first frame
                if i == 0:
                    orig_disp = raw[0] if raw.ndim==3 else raw
                    cln_disp  = cleaned[0] if cleaned.ndim==3 else cleaned
                    self.preview.emit(orig_disp, cln_disp, cr_mask)

                # Save
                out_base = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(cfg["output_dir"], f"{out_base}_lacosmic.fit")
                hdr["HISTORY"] = f"L.A.Cosmic CR rejection: {n_cr} pixels"
                hdr["NCRPIX"]  = (n_cr, "LACosmic CR pixels removed")
                astropy_fits.PrimaryHDU(data=cleaned, header=hdr).writeto(out_path, overwrite=True)

                if cfg.get("save_mask"):
                    mpath = os.path.join(cfg["output_dir"], f"{out_base}_crmask.fit")
                    astropy_fits.PrimaryHDU(data=cr_mask.astype(np.uint8)).writeto(mpath, overwrite=True)

                n_total += 1
            except Exception as e:
                log(f"  SKIP {os.path.basename(path)}: {e}")

        self.progress.emit(len(files), len(files), "Done")
        log(f"\n✓ {n_total} files processed  total CR pixels: {total_cr:,}")
        self.finished.emit({"success":True,"n_processed":n_total,"total_cr":total_cr})

# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cosmic Ray Rejection (L.A.Cosmic)  —  HYPERLOAD  —  Siril")
        self.resize(1100, 780)
        self._worker=None; self._cancel=threading.Event()
        self._build_ui()

    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        hdr=QWidget(); hdr.setFixedHeight(52)
        hdr.setStyleSheet(f"background:{BG2};border-bottom:1px solid {BRD};")
        hl=QHBoxLayout(hdr); hl.setContentsMargins(14,0,14,0)
        lbl=QLabel("☄  Cosmic Ray Rejection (L.A.Cosmic)")
        lbl.setStyleSheet(f"color:{CR};font-size:13pt;font-weight:bold;background:transparent;")
        sub=QLabel("van Dokkum 2001, PASP 113, 1420  ·  Laplacian Edge Detection  ·  astropy/ccdproc")
        sub.setStyleSheet(f"color:{DIM};font-size:9pt;background:transparent;")
        hl.addWidget(lbl); hl.addWidget(sub); hl.addStretch()
        root.addWidget(hdr)

        tabs=QTabWidget(); root.addWidget(tabs,1)
        tabs.addTab(self._build_input(),   "📁  Input")
        tabs.addTab(self._build_params(),  "⚙  Parameters")
        tabs.addTab(self._build_preview(), "🖼  Preview")
        tabs.addTab(self._build_log(),     "📋  Log")

        bottom=QWidget(); bottom.setFixedHeight(48)
        bottom.setStyleSheet(f"background:{BG2};border-top:1px solid {BRD};")
        bl=QHBoxLayout(bottom); bl.setContentsMargins(10,6,10,6)
        self._prog=QProgressBar(); self._prog.setFixedHeight(10)
        bl.addWidget(self._prog,1)
        self._btn_run=QPushButton("▶  Remove Cosmic Rays")
        self._btn_run.setObjectName("cr"); self._btn_run.setMinimumHeight(34)
        self._btn_run.setMinimumWidth(180); self._btn_run.clicked.connect(self._run)
        bl.addWidget(self._btn_run)
        self._btn_cancel=QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger"); self._btn_cancel.setMinimumHeight(34)
        self._btn_cancel.setEnabled(False); self._btn_cancel.clicked.connect(self._cancel_run)
        bl.addWidget(self._btn_cancel)
        self._status=QLabel("Ready")
        self._status.setObjectName("dim")
        self._status.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status,1)
        root.addWidget(bottom)

    def _build_input(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)

        grp_mode=QGroupBox("Input mode")
        ml=QVBoxLayout(grp_mode)
        self._rb_single=QRadioButton("Single FITS file"); self._rb_single.setChecked(True)
        self._rb_batch=QRadioButton("Batch folder")
        self._bg=QButtonGroup(); self._bg.addButton(self._rb_single); self._bg.addButton(self._rb_batch)
        self._rb_single.toggled.connect(self._on_mode)
        for rb in [self._rb_single,self._rb_batch]: ml.addWidget(rb)
        lay.addWidget(grp_mode)

        self._grp_single=QGroupBox("Single FITS")
        sl=QHBoxLayout(self._grp_single)
        self._edit_single=QLineEdit(); self._edit_single.setPlaceholderText("FITS file…")
        btn_s=QPushButton("Browse…"); btn_s.setFixedWidth(80)
        btn_s.clicked.connect(lambda: self._browse(self._edit_single,"FITS"))
        sl.addWidget(self._edit_single); sl.addWidget(btn_s)
        lay.addWidget(self._grp_single)

        self._grp_batch=QGroupBox("Batch folder")
        bl2=QHBoxLayout(self._grp_batch)
        self._edit_folder=QLineEdit(); self._edit_folder.setPlaceholderText("Folder with FITS files…")
        btn_f=QPushButton("Browse…"); btn_f.setFixedWidth(80)
        btn_f.clicked.connect(self._browse_folder)
        bl2.addWidget(self._edit_folder); bl2.addWidget(btn_f)
        self._grp_batch.setVisible(False)
        lay.addWidget(self._grp_batch)

        grp_out=QGroupBox("Output")
        ol=QFormLayout(grp_out)
        out_row=QHBoxLayout()
        self._edit_out=QLineEdit(); self._edit_out.setPlaceholderText("Output folder…")
        btn_out=QPushButton("Browse…"); btn_out.setFixedWidth(80)
        btn_out.clicked.connect(lambda: self._edit_out.setText(
            QFileDialog.getExistingDirectory(self,"Output") or self._edit_out.text()))
        out_row.addWidget(self._edit_out); out_row.addWidget(btn_out)
        ol.addRow("Output folder:", out_row)
        self._chk_mask=QCheckBox("Save CR mask FITS")
        ol.addRow(self._chk_mask)
        lay.addWidget(grp_out)
        lay.addStretch()
        return w

    def _build_params(self):
        w=QWidget()
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        inner=QWidget(); lay=QVBoxLayout(inner); lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)
        scroll.setWidget(inner); ql=QVBoxLayout(w); ql.addWidget(scroll)

        grp=QGroupBox("L.A.Cosmic parameters  (van Dokkum 2001)")
        f=QFormLayout(grp)
        self._spin_sigclip=QDoubleSpinBox(); self._spin_sigclip.setRange(1,20); self._spin_sigclip.setValue(4.5); self._spin_sigclip.setSuffix(" σ")
        f.addRow("sigclip  (CR detection threshold):", self._spin_sigclip)
        self._spin_sigfrac=QDoubleSpinBox(); self._spin_sigfrac.setRange(0.01,1.0); self._spin_sigfrac.setValue(0.3); self._spin_sigfrac.setSingleStep(0.05)
        f.addRow("sigfrac  (low sensitivity threshold):", self._spin_sigfrac)
        self._spin_objlim=QDoubleSpinBox(); self._spin_objlim.setRange(0.1,10.0); self._spin_objlim.setValue(5.0)
        f.addRow("objlim   (object flux threshold):", self._spin_objlim)
        self._spin_gain=QDoubleSpinBox(); self._spin_gain.setRange(0.01,50); self._spin_gain.setValue(1.0); self._spin_gain.setSuffix(" e⁻/ADU")
        f.addRow("gain:", self._spin_gain)
        self._spin_rn=QDoubleSpinBox(); self._spin_rn.setRange(0.1,100); self._spin_rn.setValue(6.5); self._spin_rn.setSuffix(" e⁻")
        f.addRow("readnoise:", self._spin_rn)
        self._spin_niter=QSpinBox(); self._spin_niter.setRange(1,10); self._spin_niter.setValue(4)
        f.addRow("niter  (iterations):", self._spin_niter)
        self._chk_sepmed=QCheckBox("Use separable median filter (faster)"); self._chk_sepmed.setChecked(True)
        f.addRow(self._chk_sepmed)
        lbl=QLabel("sigclip: lower=more aggressive  ·  objlim: prevents star cores being masked\n"
                   "Typical: sigclip=4.5, sigfrac=0.3, objlim=5.0, gain from camera spec")
        lbl.setObjectName("dim"); lbl.setWordWrap(True); f.addRow(lbl)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _build_preview(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(4,4,4,4)
        self._canvas=CRCanvas(); lay.addWidget(self._canvas)
        lbl=QLabel("Preview of first processed frame: original | CR mask | cleaned")
        lbl.setObjectName("dim"); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        return w

    def _build_log(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(6,6,6,6)
        self._log=QPlainTextEdit(); self._log.setReadOnly(True)
        btn_clr=QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._log.clear)
        lay.addWidget(btn_clr); lay.addWidget(self._log)
        return w

    def _on_mode(self):
        self._grp_single.setVisible(self._rb_single.isChecked())
        self._grp_batch.setVisible(self._rb_batch.isChecked())

    def _browse(self, edit, title):
        p,_=QFileDialog.getOpenFileName(self,f"Select {title}","","FITS (*.fit *.fits *.fts);;All (*)")
        if p:
            edit.setText(p)
            if not self._edit_out.text(): self._edit_out.setText(os.path.dirname(p))

    def _browse_folder(self):
        d=QFileDialog.getExistingDirectory(self,"Select folder")
        if d:
            self._edit_folder.setText(d)
            if not self._edit_out.text(): self._edit_out.setText(d)

    def _get_cfg(self):
        out=self._edit_out.text().strip()
        if not out: QMessageBox.warning(self,"No output","Set output folder."); return None
        if self._rb_single.isChecked():
            p=self._edit_single.text().strip()
            if not p or not os.path.isfile(p): QMessageBox.warning(self,"No file","Select FITS."); return None
            return {"mode":"single","input_path":p,"folder":"","output_dir":out,
                    "sigclip":self._spin_sigclip.value(),"sigfrac":self._spin_sigfrac.value(),
                    "objlim":self._spin_objlim.value(),"gain":self._spin_gain.value(),
                    "readnoise":self._spin_rn.value(),"niter":self._spin_niter.value(),
                    "sepmed":self._chk_sepmed.isChecked(),"cleantype":"meanmask",
                    "fsmode":"median","save_mask":self._chk_mask.isChecked()}
        else:
            d=self._edit_folder.text().strip()
            if not d or not os.path.isdir(d): QMessageBox.warning(self,"No folder","Select folder."); return None
            return {"mode":"batch","input_path":"","folder":d,"output_dir":out,
                    "sigclip":self._spin_sigclip.value(),"sigfrac":self._spin_sigfrac.value(),
                    "objlim":self._spin_objlim.value(),"gain":self._spin_gain.value(),
                    "readnoise":self._spin_rn.value(),"niter":self._spin_niter.value(),
                    "sepmed":self._chk_sepmed.isChecked(),"cleantype":"meanmask",
                    "fsmode":"median","save_mask":self._chk_mask.isChecked()}

    def _run(self):
        cfg=self._get_cfg()
        if cfg is None: return
        self._cancel.clear(); self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._prog.setValue(0)
        self._worker=CRWorker(cfg,self._cancel)
        self._worker.progress.connect(lambda s,t,m:(self._prog.setRange(0,t),self._prog.setValue(s),self._status.setText(m)))
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.preview.connect(lambda o,c,m: self._canvas.show(o,c,m))
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_run(self):
        self._cancel.set(); self._btn_cancel.setEnabled(False)

    def _on_finished(self, result):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        if result.get("success"):
            n=result["n_processed"]; cr=result["total_cr"]
            self._status.setText(f"✓  {n} files  ·  {cr:,} CR pixels removed")
            self._status.setStyleSheet(f"color:{OK};font-size:9pt;")
        else:
            self._status.setText(f"✗ {result.get('error','')}")
            self._status.setStyleSheet(f"color:{ERR};font-size:9pt;")


def main():
    import traceback as _tb
    lp=os.path.join(os.path.dirname(os.path.abspath(__file__)),"crash_log.txt")
    def ch(et,ev,etb):
        with open(lp,"a") as f: f.write(f"\n{'='*50}\nCRASH {datetime.now()}\n"+"".join(_tb.format_exception(et,ev,etb)))
        sys.__excepthook__(et,ev,etb)
    sys.excepthook=ch
    app=QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SS)
    MainWindow().show()
    app.exec()

if __name__=="__main__":
    main()
