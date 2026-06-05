"""
multi_session_manager.py  —  Script #49  (v2 — Manifest-driven architecture)
Smart Multi-Session Pipeline Manager for Siril
Scan → Pair → Organize → Manifest → Dispatch to selectable stacker
"""

import sirilpy as s
s.ensure_installed("PyQt6")

# ── stdlib ─────────────────────────────────────────────────────────────────────
import os, sys, json, glob, shutil, threading, datetime, re

# ── PyQt6 ─────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLineEdit, QPushButton, QLabel, QTreeWidget,
    QTreeWidgetItem, QTableWidget, QTableWidgetItem, QComboBox,
    QDoubleSpinBox, QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QFormLayout, QHeaderView, QFrame, QRadioButton,
    QButtonGroup, QSplitter, QScrollArea, QTextBrowser
)
from PyQt6.QtCore  import Qt, QThread, pyqtSignal
from PyQt6.QtGui   import QColor, QFont, QTextCursor

# ── Equipment manager (optional) ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
try:
    from equipment_manager import load_all_profiles, get_siril_vars
    HAS_EM = True
except ImportError:
    HAS_EM = False

VERSION = "2.0.0"

# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════
SIRIL_BG       = "#1e2128"
SIRIL_BG2      = "#252930"
SIRIL_BG3      = "#2d3240"
SIRIL_ACCENT   = "#4a9eff"
SIRIL_ACCENT2  = "#2d6abf"
SIRIL_TEXT     = "#dde3ee"
SIRIL_TEXT_DIM = "#7a8499"
SIRIL_BORDER   = "#3a4055"
SIRIL_SUCCESS  = "#4caf7d"
SIRIL_WARNING  = "#e8a23a"
SIRIL_SECTION  = "#5ba3ff"
SIRIL_ERROR    = "#cc4444"

SIRIL_STYLESHEET = f"""
QMainWindow,QDialog,QWidget{{background:{SIRIL_BG};color:{SIRIL_TEXT};
    font-family:'Segoe UI',Arial,sans-serif;font-size:10pt;}}
QTabWidget::pane{{border:1px solid {SIRIL_BORDER};border-radius:4px;background:{SIRIL_BG2};}}
QTabBar::tab{{background:{SIRIL_BG3};color:{SIRIL_TEXT_DIM};border:1px solid {SIRIL_BORDER};
    padding:6px 16px;margin-right:2px;border-bottom:none;border-radius:4px 4px 0 0;}}
QTabBar::tab:selected{{background:{SIRIL_BG2};color:{SIRIL_ACCENT};border-bottom:2px solid {SIRIL_ACCENT};}}
QGroupBox{{background:{SIRIL_BG2};border:1px solid {SIRIL_BORDER};border-radius:5px;
    margin-top:8px;padding:8px;font-weight:bold;color:{SIRIL_SECTION};}}
QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}
QTreeWidget{{background:{SIRIL_BG2};alternate-background-color:{SIRIL_BG3};
    border:1px solid {SIRIL_BORDER};border-radius:4px;
    selection-background-color:{SIRIL_ACCENT2};}}
QTreeWidget::item{{padding:3px 6px;border:none;}}
QTreeWidget::item:selected{{background:{SIRIL_ACCENT2};}}
QHeaderView::section{{background:{SIRIL_BG3};color:{SIRIL_ACCENT};
    border:1px solid {SIRIL_BORDER};padding:5px 8px;font-weight:bold;}}
QTableWidget{{background:{SIRIL_BG2};alternate-background-color:{SIRIL_BG3};
    border:1px solid {SIRIL_BORDER};border-radius:4px;
    gridline-color:{SIRIL_BORDER};selection-background-color:{SIRIL_ACCENT2};}}
QTableWidget::item{{padding:4px 8px;border:none;}}
QPushButton{{background:{SIRIL_BG3};color:{SIRIL_TEXT};border:1px solid {SIRIL_BORDER};
    border-radius:4px;padding:6px 14px;}}
QPushButton:hover{{background:{SIRIL_ACCENT2};border-color:{SIRIL_ACCENT};color:white;}}
QPushButton:pressed{{background:{SIRIL_ACCENT};}}
QPushButton:disabled{{color:{SIRIL_TEXT_DIM};border-color:{SIRIL_BG3};}}
QPushButton#primary{{background:{SIRIL_ACCENT2};border-color:{SIRIL_ACCENT};
    color:white;font-weight:bold;}}
QPushButton#primary:hover{{background:{SIRIL_ACCENT};}}
QPushButton#danger:hover{{background:#8b2020;border-color:{SIRIL_ERROR};}}
QPushButton#warning{{background:{SIRIL_BG3};border-color:{SIRIL_WARNING};color:{SIRIL_WARNING};}}
QPushButton#warning:hover{{background:#3d2e10;}}
QLineEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox{{
    background:{SIRIL_BG3};color:{SIRIL_TEXT};border:1px solid {SIRIL_BORDER};
    border-radius:3px;padding:4px 6px;}}
QLineEdit:focus,QPlainTextEdit:focus,QSpinBox:focus,
QDoubleSpinBox:focus,QComboBox:focus{{border-color:{SIRIL_ACCENT};}}
QPlainTextEdit[readOnly="true"]{{background:#181c22;color:{SIRIL_SUCCESS};
    font-family:'Courier New',monospace;font-size:9pt;}}
QTextBrowser{{background:#181c22;color:{SIRIL_TEXT};
    font-family:'Segoe UI',Arial,sans-serif;font-size:10pt;
    border:1px solid {SIRIL_BORDER};border-radius:4px;padding:8px;}}
QComboBox::drop-down{{border:none;padding-right:4px;}}
QComboBox QAbstractItemView{{background:{SIRIL_BG3};color:{SIRIL_TEXT};
    selection-background-color:{SIRIL_ACCENT2};border:1px solid {SIRIL_BORDER};}}
QScrollBar:vertical{{background:{SIRIL_BG2};width:10px;border-radius:5px;}}
QScrollBar::handle:vertical{{background:{SIRIL_BORDER};border-radius:5px;min-height:20px;}}
QScrollBar::handle:vertical:hover{{background:{SIRIL_ACCENT2};}}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}
QProgressBar{{background:{SIRIL_BG3};border:1px solid {SIRIL_BORDER};
    border-radius:3px;text-align:center;}}
QProgressBar::chunk{{background:{SIRIL_ACCENT};border-radius:3px;}}
QLabel#section{{color:{SIRIL_SECTION};font-weight:bold;padding-top:6px;}}
QLabel#dim{{color:{SIRIL_TEXT_DIM};font-size:9pt;}}
QLabel#ok{{color:{SIRIL_SUCCESS};font-weight:bold;}}
QLabel#err{{color:{SIRIL_ERROR};font-weight:bold;}}
QLabel#warn{{color:{SIRIL_WARNING};font-weight:bold;}}
QCheckBox{{color:{SIRIL_TEXT};spacing:6px;}}
QCheckBox::indicator{{width:14px;height:14px;border:1px solid {SIRIL_BORDER};
    border-radius:2px;background:{SIRIL_BG3};}}
QCheckBox::indicator:checked{{background:{SIRIL_ACCENT};border-color:{SIRIL_ACCENT};}}
QRadioButton{{color:{SIRIL_TEXT};spacing:6px;}}
QRadioButton::indicator{{width:14px;height:14px;border:1px solid {SIRIL_BORDER};
    border-radius:7px;background:{SIRIL_BG3};}}
QRadioButton::indicator:checked{{background:{SIRIL_ACCENT};border-color:{SIRIL_ACCENT};}}
QFrame#sep{{background:{SIRIL_BORDER};max-height:1px;}}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SKIP_FOLDERS = {"darks","flats","bias","lights","output",".mspm_work","flat"}
FITS_EXTS    = ("*.fit","*.fits","*.fts","*.FIT","*.FITS","*.FTS")
MANIFEST_NAME = "mspm_manifest.json"

STACKER_OPTIONS = [
    ("siril_cli",    "Siril  (CLI — auto-drive via sirilpy)"),
    ("siril_script", "Siril  (generate .ssf script file)"),
    ("python_stub",  "Custom Python stacker  [placeholder]"),
]

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def find_fits(folder: str) -> list:
    files = []
    for ext in FITS_EXTS:
        files += glob.glob(os.path.join(folder, ext))
    return sorted(files)

def dim(text):
    l = QLabel(text); l.setObjectName("dim"); return l

def sep():
    f = QFrame(); f.setObjectName("sep")
    f.setFrameShape(QFrame.Shape.HLine); return f

def ci(text, color=None):
    """Create a non-editable QTableWidgetItem, optionally coloured."""
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if color:
        item.setForeground(QColor(color))
    return item

def ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ══════════════════════════════════════════════════════════════════════════════
#  SCANNER
# ══════════════════════════════════════════════════════════════════════════════
def _cal_source(cal_type, filter_name, night, root, nights_in_filter):
    """
    Search calibration frames with 4-level inheritance.
    Returns (file_list, source_label).
    source_label: 'own' | 'sibling:NIGHT' | 'filter' | 'project' | ''
    """
    # 1 — same night
    p = os.path.join(root, filter_name, night, cal_type)
    if os.path.isdir(p):
        f = find_fits(p)
        if f: return f, "own"
    # 2 — any sibling night
    for sib in nights_in_filter:
        if sib == night: continue
        p = os.path.join(root, filter_name, sib, cal_type)
        if os.path.isdir(p):
            f = find_fits(p)
            if f: return f, f"sibling:{sib}"
    # 3 — filter level
    p = os.path.join(root, filter_name, cal_type)
    if os.path.isdir(p):
        f = find_fits(p)
        if f: return f, "filter"
    # 4 — project level
    p = os.path.join(root, cal_type)
    if os.path.isdir(p):
        f = find_fits(p)
        if f: return f, "project"
    return [], ""

def scan_project(root: str) -> dict:
    """
    Returns a structure dict describing the entire project.
    Each night entry includes lights + resolved calibration paths + source labels.
    """
    out = {
        "project_root": root,
        "filters": [],
        "nights": {},          # filter -> [night, ...]
        "batches": {},         # filter -> night -> {lights, dark, flat, bias, *_source}
        "warnings": [],
        "errors": []
    }
    if not os.path.isdir(root):
        out["errors"].append(f"Not a directory: {root}"); return out

    try:
        top_entries = sorted(os.listdir(root))
    except PermissionError as e:
        out["errors"].append(str(e)); return out

    for entry in top_entries:
        if entry in SKIP_FOLDERS or entry.startswith("."): continue
        fp = os.path.join(root, entry)
        if not os.path.isdir(fp): continue
        # Must have at least one sub-folder containing a 'lights' folder
        try:
            subs = os.listdir(fp)
        except PermissionError:
            continue
        has_lights = any(
            os.path.isdir(os.path.join(fp, s, "lights")) for s in subs
        )
        if not has_lights: continue

        fname = entry
        out["filters"].append(fname)
        out["nights"][fname] = []
        out["batches"][fname] = {}

        for sub in sorted(subs):
            if sub in SKIP_FOLDERS or sub.startswith("."): continue
            lp = os.path.join(fp, sub, "lights")
            if not os.path.isdir(lp): continue
            night = sub
            out["nights"][fname].append(night)

        for night in out["nights"][fname]:
            lights = find_fits(os.path.join(root, fname, night, "lights"))
            nights_in_filter = out["nights"][fname]
            dark, dark_src = _cal_source("darks", fname, night, root, nights_in_filter)
            flat, flat_src = _cal_source("flats", fname, night, root, nights_in_filter)
            bias, bias_src = _cal_source("bias",  fname, night, root, nights_in_filter)

            out["batches"][fname][night] = {
                "lights": lights,
                "dark": dark, "dark_source": dark_src,
                "flat": flat, "flat_source": flat_src,
                "bias": bias, "bias_source": bias_src,
            }
            if not lights:
                out["warnings"].append(f"{fname}/{night}: no light frames found")
            if not dark and lights:
                out["warnings"].append(f"{fname}/{night}: no darks found (will calibrate without)")
            if not flat and lights:
                out["warnings"].append(f"{fname}/{night}: no flats found (will calibrate without)")
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  MANIFEST  (the contract between MSPM and any stacker)
# ══════════════════════════════════════════════════════════════════════════════
def build_manifest(structure: dict, settings: dict) -> dict:
    """
    Build the manifest dict from a scanned structure + settings.
    Does NOT write to disk — call save_manifest() for that.

    Manifest schema
    ───────────────
    {
      "version": "2",
      "created": "<iso timestamp>",
      "project_root": "...",
      "output_dir": "...",
      "stacker": "siril_cli | siril_script | python_stub",
      "stacking": {
        "method": "winsorized_sigma",
        "sigma_low": 3.0,
        "sigma_high": 3.0,
        "drizzle": 1.0,
        "debayer": false
      },
      "file_mode": "copy | move",
      "filters": ["Ha", "OIII", ...],
      "batches": [
        {
          "id": "Ha__2025-03-01",
          "filter": "Ha",
          "night": "2025-03-01",
          "order": 1,               <- global processing order
          "work_dir": "...",        <- where files will be placed
          "lights": ["abs/path", ...],
          "dark":   ["abs/path", ...],  (empty list if none)
          "flat":   ["abs/path", ...],
          "bias":   ["abs/path", ...],
          "dark_source": "own | sibling:2025-03-01 | filter | project | ",
          "flat_source": "...",
          "bias_source": "...",
          "output_name": "sub_Ha_2025-03-01",   <- expected sub-stack name
          "status": "pending | done | failed | skipped",
          "stacker_notes": ""       <- filled by stacker after processing
        },
        ...
        {
          "id": "Ha__COMBINE",
          "filter": "Ha",
          "night": "__combine__",
          "order": N,
          "work_dir": "<output_dir>",
          "lights": [],             <- stacker fills from previous sub-stacks
          "output_name": "master_Ha",
          "status": "pending",
          "is_combine": true
        }
      ]
    }
    """
    root       = structure["project_root"]
    output_dir = settings.get("output_dir") or os.path.join(root, "output")
    work_base  = os.path.join(root, ".mspm_work")

    batches = []
    order   = 1

    for flt in structure["filters"]:
        nights = sorted(structure["nights"].get(flt, []))
        for night in nights:
            b = structure["batches"][flt][night]
            safe_night = re.sub(r'[^\w\-]', '_', night)
            batch_id   = f"{flt}__{safe_night}"
            work_dir   = os.path.join(work_base, flt, safe_night)
            batches.append({
                "id":          batch_id,
                "filter":      flt,
                "night":       night,
                "order":       order,
                "work_dir":    work_dir,
                "lights":      b["lights"],
                "dark":        b["dark"],
                "flat":        b["flat"],
                "bias":        b["bias"],
                "dark_source": b["dark_source"],
                "flat_source": b["flat_source"],
                "bias_source": b["bias_source"],
                "output_name": f"sub_{flt}_{safe_night}",
                "status":      "pending",
                "stacker_notes": "",
                "is_combine":  False,
            })
            order += 1

        # Combine batch for this filter
        batches.append({
            "id":          f"{flt}__COMBINE",
            "filter":      flt,
            "night":       "__combine__",
            "order":       order,
            "work_dir":    output_dir,
            "lights":      [],   # stacker fills from completed sub-stacks
            "dark":        [], "flat": [], "bias": [],
            "dark_source": "", "flat_source": "", "bias_source": "",
            "output_name": f"master_{flt}",
            "status":      "pending",
            "stacker_notes": "",
            "is_combine":  True,
        })
        order += 1

    return {
        "version":      "2",
        "created":      ts(),
        "project_root": root,
        "output_dir":   output_dir,
        "stacker":      settings.get("stacker", "siril_cli"),
        "stacking": {
            "method":    settings.get("stacking_method", "winsorized_sigma"),
            "sigma_low": settings.get("sigma_low", 3.0),
            "sigma_high":settings.get("sigma_high", 3.0),
            "drizzle":   settings.get("drizzle", 1.0),
            "debayer":   settings.get("debayer", False),
        },
        "file_mode": settings.get("file_mode", "copy"),
        "filters":   structure["filters"],
        "batches":   batches,
    }

def save_manifest(manifest: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def manifest_path(project_root: str) -> str:
    return os.path.join(project_root, MANIFEST_NAME)

def manifest_summary(manifest: dict) -> str:
    batches  = manifest.get("batches", [])
    pending  = sum(1 for b in batches if b["status"] == "pending")
    done     = sum(1 for b in batches if b["status"] == "done")
    failed   = sum(1 for b in batches if b["status"] == "failed")
    skipped  = sum(1 for b in batches if b["status"] == "skipped")
    total    = len(batches)
    return (f"{total} batch(es) total  ·  {pending} pending  ·  "
            f"{done} done  ·  {failed} failed  ·  {skipped} skipped")

# ══════════════════════════════════════════════════════════════════════════════
#  ORGANIZE WORKER  (Stage 2 — file distribution)
# ══════════════════════════════════════════════════════════════════════════════
class OrganizeWorker(QThread):
    log_line  = pyqtSignal(str)
    progress  = pyqtSignal(int, int, str)
    finished  = pyqtSignal(dict)   # {"success": bool, "error": str}

    def __init__(self, manifest: dict, cancel_event: threading.Event):
        super().__init__()
        self._manifest = manifest
        self._cancel   = cancel_event

    def run(self):
        try:
            self._organize()
            self.finished.emit({"success": True})
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})

    def _cp_or_mv(self, src, dst_dir):
        mode = self._manifest.get("file_mode", "copy")
        dst  = os.path.join(dst_dir, os.path.basename(src))
        if mode == "move":
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    def _organize(self):
        batches   = [b for b in self._manifest["batches"] if not b.get("is_combine")]
        mode_verb = "Moving" if self._manifest.get("file_mode") == "move" else "Copying"

        for idx, batch in enumerate(batches):
            if self._cancel.is_set():
                self.log_line.emit("[CANCELLED]"); return

            if batch["status"] in ("done", "skipped"):
                self.log_line.emit(f"  [SKIP already organized] {batch['id']}")
                self.progress.emit(idx + 1, len(batches), batch["id"])
                continue

            self.progress.emit(idx, len(batches), batch["id"])
            wd = batch["work_dir"]
            self.log_line.emit(f"\n── {batch['id']}")
            self.log_line.emit(f"   Work dir: {wd}")

            # Create sub-dirs
            for sub in ("lights", "darks", "flats", "bias"):
                os.makedirs(os.path.join(wd, sub), exist_ok=True)

            # Distribute files
            for flist, subdir, label in [
                (batch["lights"], "lights", "light"),
                (batch["dark"],   "darks",  "dark"),
                (batch["flat"],   "flats",  "flat"),
                (batch["bias"],   "bias",   "bias"),
            ]:
                if not flist:
                    continue
                dst = os.path.join(wd, subdir)
                self.log_line.emit(f"   {mode_verb} {len(flist)} {label} frame(s)…")
                for src in flist:
                    if self._cancel.is_set():
                        return
                    try:
                        self._cp_or_mv(src, dst)
                    except Exception as e:
                        self.log_line.emit(f"   [WARN] {os.path.basename(src)}: {e}")

            self.log_line.emit(f"   ✓ organized")

        self.progress.emit(len(batches), len(batches), "Done")
        self.log_line.emit(f"\nOrganize stage complete  {ts()}")

# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCH WORKER  (Stage 3 — hand to stacker)
# ══════════════════════════════════════════════════════════════════════════════
class DispatchWorker(QThread):
    log_line  = pyqtSignal(str)
    progress  = pyqtSignal(int, int, str)
    finished  = pyqtSignal(dict)

    def __init__(self, manifest: dict, manifest_file: str,
                 cancel_event: threading.Event):
        super().__init__()
        self._manifest      = manifest
        self._manifest_file = manifest_file
        self._cancel        = cancel_event
        self._siril         = None

    def run(self):
        stacker = self._manifest.get("stacker", "siril_cli")
        try:
            if stacker == "siril_cli":
                self._dispatch_siril_cli()
            elif stacker == "siril_script":
                self._dispatch_siril_script()
            elif stacker == "python_stub":
                self._dispatch_python_stub()
            else:
                raise ValueError(f"Unknown stacker: {stacker}")
            self.finished.emit({"success": True})
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})
        finally:
            if self._siril:
                try: self._siril.disconnect()
                except Exception: pass

    # ── helpers ────────────────────────────────────────────────────────────
    def _log(self, t): self.log_line.emit(t)

    def _mark(self, batch, status, notes=""):
        batch["status"]        = status
        batch["stacker_notes"] = notes
        save_manifest(self._manifest, self._manifest_file)

    def _rej_args(self) -> list:
        st = self._manifest["stacking"]
        m  = st.get("method", "winsorized_sigma")
        sl = str(st.get("sigma_low",  3.0))
        sh = str(st.get("sigma_high", 3.0))
        return {"winsorized_sigma":["rej","w",sl,sh],
                "linear_sigma":    ["rej","l",sl,sh],
                "percentile":      ["rej","p",sl,sh],
                "none":            []}.get(m, ["rej","w",sl,sh])

    def _siril_cmd(self, *args):
        cmd = " ".join(str(a) for a in args)
        self._log(f"siril> {cmd}")
        if self._siril:
            self._siril.cmd(*args)

    # ── Siril CLI ──────────────────────────────────────────────────────────
    def _dispatch_siril_cli(self):
        self._siril = s.SirilInterface()
        self._siril.connect()
        self._log("Connected to Siril.")

        batches = self._manifest["batches"]
        pending = [b for b in batches if b["status"] == "pending"]
        total   = len(pending)

        for idx, batch in enumerate(pending):
            if self._cancel.is_set():
                self._log("[CANCELLED]"); return
            self.progress.emit(idx, total, batch["id"])

            if batch.get("is_combine"):
                self._combine_siril(batch, batches)
            else:
                self._process_night_siril(batch)

        self.progress.emit(total, total, "Done")

    def _process_night_siril(self, batch):
        self._log(f"\n── {batch['id']}  ({batch['filter']} / {batch['night']})")
        wd = batch["work_dir"]
        st = self._manifest["stacking"]

        try:
            self._siril_cmd("cd", wd)
            self._siril_cmd("convert", "lights", "-out=pp_")

            if batch["dark"]:
                self._siril_cmd("stack","darks","rej","w","3","3",
                                "-nonorm","-output=master_dark")
            if batch["flat"]:
                self._siril_cmd("stack","flats","rej","w","3","3",
                                "-nonorm","-output=master_flat")
            if batch["bias"]:
                self._siril_cmd("stack","bias","rej","w","3","3",
                                "-nonorm","-output=master_bias")

            cal = []
            if batch["dark"]: cal += ["-dark=master_dark"]
            if batch["flat"]: cal += ["-flat=master_flat"]
            if batch["bias"]: cal += ["-bias=master_bias"]
            if st.get("debayer"): cal += ["-debayer"]
            self._siril_cmd("calibrate","pp_", *cal)
            self._siril_cmd("register","pp_")

            rej = self._rej_args()
            self._siril_cmd("stack","pp_",*rej,
                            f"-output={batch['output_name']}")

            self._mark(batch, "done")
            self._log(f"   ✓ sub-stack: {batch['output_name']}.fit")
        except Exception as e:
            self._mark(batch, "failed", str(e))
            self._log(f"   ✗ FAILED: {e}")

    def _combine_siril(self, batch, all_batches):
        flt = batch["filter"]
        self._log(f"\n── COMBINE  {flt}")
        # Collect done sub-stacks for this filter
        sub_stacks = [
            b for b in all_batches
            if b["filter"] == flt
            and not b.get("is_combine")
            and b["status"] == "done"
        ]
        if not sub_stacks:
            self._mark(batch, "skipped", "no sub-stacks available")
            self._log("   [SKIP] no completed sub-stacks"); return

        out_dir = batch["work_dir"]
        os.makedirs(out_dir, exist_ok=True)

        # Copy sub-stacks to output dir
        for sb in sub_stacks:
            src = os.path.join(sb["work_dir"], sb["output_name"] + ".fit")
            if os.path.isfile(src):
                shutil.copy2(src, out_dir)
                self._log(f"   + {sb['output_name']}.fit")
            else:
                self._log(f"   [WARN] not found: {src}")

        try:
            self._siril_cmd("cd", out_dir)
            rej = self._rej_args()
            self._siril_cmd("stack", f"sub_{flt}_", *rej,
                            f"-output={batch['output_name']}")
            self._mark(batch, "done")
            self._log(f"   ✓ master: {batch['output_name']}.fit")
        except Exception as e:
            self._mark(batch, "failed", str(e))
            self._log(f"   ✗ FAILED: {e}")

    # ── Siril script file ──────────────────────────────────────────────────
    def _dispatch_siril_script(self):
        out_dir = self._manifest["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        script_path = os.path.join(out_dir, "mspm_pipeline.ssf")

        lines = [
            "# MSPM-generated Siril script",
            f"# Created: {ts()}",
            f"# Project: {self._manifest['project_root']}",
            "",
        ]
        st  = self._manifest["stacking"]
        rej = " ".join(self._rej_args())

        batches = [b for b in self._manifest["batches"]
                   if b["status"] == "pending"]
        total   = len(batches)

        for idx, batch in enumerate(batches):
            if self._cancel.is_set():
                self._log("[CANCELLED]"); return
            self.progress.emit(idx, total, batch["id"])

            if batch.get("is_combine"):
                flt    = batch["filter"]
                lines += [
                    f"",
                    f"# ── COMBINE {flt} ──",
                    f"cd {batch['work_dir']}",
                    f"stack sub_{flt}_ {rej} -output={batch['output_name']}",
                ]
                self._mark(batch, "done", "script generated")
            else:
                wd  = batch["work_dir"]
                cal = []
                if batch["dark"]: cal.append("-dark=master_dark")
                if batch["flat"]: cal.append("-flat=master_flat")
                if batch["bias"]: cal.append("-bias=master_bias")
                if st.get("debayer"): cal.append("-debayer")

                lines += [
                    f"",
                    f"# ── {batch['id']} ──",
                    f"cd {wd}",
                    f"convert lights -out=pp_",
                ]
                if batch["dark"]:
                    lines.append("stack darks rej w 3 3 -nonorm -output=master_dark")
                if batch["flat"]:
                    lines.append("stack flats rej w 3 3 -nonorm -output=master_flat")
                if batch["bias"]:
                    lines.append("stack bias rej w 3 3 -nonorm -output=master_bias")
                lines.append(f"calibrate pp_ {' '.join(cal)}")
                lines.append(f"register pp_")
                lines.append(f"stack pp_ {rej} -output={batch['output_name']}")
                self._mark(batch, "done", "script generated")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self._log(f"Script written to:\n  {script_path}")
        self._log("Open it in Siril: Script menu → Load script")
        self.progress.emit(total, total, "Script saved")

    # ── Python stub ────────────────────────────────────────────────────────
    def _dispatch_python_stub(self):
        self._log("Python stacker placeholder.")
        self._log("To implement a custom stacker:")
        self._log("  1. Read the manifest JSON from the project root.")
        self._log("  2. Iterate batches where status == 'pending'.")
        self._log("  3. Process each batch using your algorithm.")
        self._log("  4. Set batch['status'] = 'done' and save manifest.")
        self._log(f"\nManifest: {self._manifest_file}")
        for b in self._manifest["batches"]:
            self._log(f"  [{b['status']:8s}] {b['id']}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PROJECT  (scan + tree)
# ══════════════════════════════════════════════════════════════════════════════
class ProjectTab(QWidget):
    scanned = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._structure = {}
        self._build()

    def _build(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(8,8,8,8); vl.setSpacing(8)

        # Root row
        grp = QGroupBox("Project root")
        hl  = QHBoxLayout(grp)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select project root folder…")
        hl.addWidget(self.path_edit)
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._browse)
        hl.addWidget(b_browse)
        self.b_scan = QPushButton("🔍  Scan")
        self.b_scan.setObjectName("primary")
        self.b_scan.clicked.connect(self._scan)
        hl.addWidget(self.b_scan)
        vl.addWidget(grp)

        # Tree
        grp2 = QGroupBox("Detected structure")
        vl2  = QVBoxLayout(grp2)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(
            ["Name","Frames","Darks","Flats","Bias","Status"])
        self.tree.setAlternatingRowColors(True)
        h = self.tree.header()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1,6):
            h.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        vl2.addWidget(self.tree)
        self.totals = dim("No project scanned yet.")
        vl2.addWidget(self.totals)
        vl.addWidget(grp2)

        # Cal table
        grp3 = QGroupBox("Calibration pairing preview")
        vl3  = QVBoxLayout(grp3)
        vl3.addWidget(dim(
            "🟢 own files   🔵 inherited   🟠 missing"))
        self.cal = QTableWidget(0, 5)
        self.cal.setHorizontalHeaderLabels(
            ["Night","Lights","Darks","Flats","Bias"])
        self.cal.setAlternatingRowColors(True)
        self.cal.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ch = self.cal.horizontalHeader()
        ch.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        ch.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(2,5): ch.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        vl3.addWidget(self.cal)
        vl.addWidget(grp3)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select project root", self.path_edit.text() or os.path.expanduser("~"))
        if d: self.path_edit.setText(d)

    def get_root(self): return self.path_edit.text().strip()
    def get_structure(self): return self._structure

    def _scan(self):
        root = self.get_root()
        if not os.path.isdir(root):
            QMessageBox.warning(self,"No folder","Select a valid project root first."); return
        self.b_scan.setEnabled(False); self.b_scan.setText("Scanning…")
        QApplication.processEvents()
        self._structure = scan_project(root)
        self._fill_tree(self._structure)
        self._fill_cal(self._structure)
        self.scanned.emit(self._structure)
        self.b_scan.setEnabled(True); self.b_scan.setText("🔍  Scan")

    def _fill_tree(self, st):
        self.tree.clear()
        total_f = total_n = 0
        for flt in st["filters"]:
            fi = QTreeWidgetItem(self.tree)
            fi.setText(0, f"📂  {flt}/")
            fo = fi.font(0); fo.setBold(True); fi.setFont(0, fo)
            fi.setForeground(0, QColor(SIRIL_SECTION))
            nights = st["nights"].get(flt, [])
            total_n += len(nights); ff = 0
            for night in sorted(nights):
                b  = st["batches"][flt][night]
                ni = QTreeWidgetItem(fi)
                ni.setText(0, f"  {night}")
                ni.setText(1, str(len(b["lights"]))); ff += len(b["lights"])

                def ct(files, src):
                    if not files: return "⚠ none"
                    return f"✓ {len(files)}" if src == "own" else f"↑ {len(files)}"
                def cc(files, src):
                    if not files: return SIRIL_WARNING
                    return SIRIL_SUCCESS if src == "own" else SIRIL_ACCENT

                for col, key_f, key_s in [(2,"dark","dark_source"),
                                           (3,"flat","flat_source"),
                                           (4,"bias","bias_source")]:
                    ni.setText(col, ct(b[key_f], b[key_s]))
                    ni.setForeground(col, QColor(cc(b[key_f], b[key_s])))

                warns = [w for w in st["warnings"] if w.startswith(f"{flt}/{night}")]
                if warns:
                    ni.setText(5,"⚠"); ni.setForeground(5,QColor(SIRIL_WARNING))
                    ni.setToolTip(5, "\n".join(warns))
                else:
                    ni.setText(5,"OK"); ni.setForeground(5,QColor(SIRIL_SUCCESS))
                total_f += len(b["lights"])
            fi.setText(1, f"{ff}"); fi.setExpanded(True)

        est = total_f * 30 / 1024
        self.totals.setText(
            f"{len(st['filters'])} filter(s)  ·  {total_n} night(s)  "
            f"·  {total_f} frames  ·  ~{est:.1f} GB est."
            + (f"  ·  ⚠ {len(st['warnings'])} warning(s)"
               if st["warnings"] else ""))

    def _fill_cal(self, st):
        self.cal.setRowCount(0); row = 0
        for flt in st["filters"]:
            for night in sorted(st["nights"].get(flt,[])):
                b = st["batches"][flt][night]
                self.cal.insertRow(row)
                self.cal.setItem(row, 0, ci(f"{flt}/{night}"))
                self.cal.setItem(row, 1, ci(str(len(b["lights"]))))
                def si(files, src):
                    if not files: return ci("⚠ missing", SIRIL_WARNING)
                    label = f"✓ own ({len(files)})" if src=="own" else f"↑ {src} ({len(files)})"
                    return ci(label, SIRIL_SUCCESS if src=="own" else SIRIL_ACCENT)
                self.cal.setItem(row,2,si(b["dark"],b["dark_source"]))
                self.cal.setItem(row,3,si(b["flat"],b["flat_source"]))
                self.cal.setItem(row,4,si(b["bias"],b["bias_source"]) if b["bias"] else ci("—",SIRIL_TEXT_DIM))
                row += 1

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ORGANIZE  (file distribution + manifest)
# ══════════════════════════════════════════════════════════════════════════════
class OrganizeTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        vl = QVBoxLayout(self); vl.setContentsMargins(8,8,8,8); vl.setSpacing(8)

        # File mode
        grp_mode = QGroupBox("File handling")
        hl_mode  = QHBoxLayout(grp_mode)
        self.rb_copy = QRadioButton("Copy files  (safe — originals untouched)")
        self.rb_move = QRadioButton("Move files  (saves disk space)")
        self.rb_copy.setChecked(True)
        self._mode_grp = QButtonGroup()
        self._mode_grp.addButton(self.rb_copy, 0)
        self._mode_grp.addButton(self.rb_move, 1)
        hl_mode.addWidget(self.rb_copy)
        hl_mode.addWidget(self.rb_move)
        hl_mode.addStretch()
        vl.addWidget(grp_mode)

        # Output dir
        grp_out = QGroupBox("Output / work directories")
        fl      = QFormLayout(grp_out); fl.setSpacing(8)
        out_hl  = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Default: <project_root>/output/")
        out_hl.addWidget(self.out_edit)
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self._browse_out)
        out_hl.addWidget(b_out)
        fl.addRow("Output folder:", out_hl)
        fl.addRow("", dim("Work folders go into <project_root>/.mspm_work/"))
        vl.addWidget(grp_out)

        # Stacking settings
        grp_st = QGroupBox("Stacking settings  (written into manifest)")
        fl2    = QFormLayout(grp_st); fl2.setSpacing(8)

        self.rej_combo = QComboBox()
        self.rej_combo.addItems(["winsorized_sigma","linear_sigma","percentile","none"])
        fl2.addRow("Rejection method:", self.rej_combo)

        self.sig_lo = QDoubleSpinBox(); self.sig_lo.setRange(0.5,10); self.sig_lo.setValue(3.0)
        fl2.addRow("Sigma low:", self.sig_lo)

        self.sig_hi = QDoubleSpinBox(); self.sig_hi.setRange(0.5,10); self.sig_hi.setValue(3.0)
        fl2.addRow("Sigma high:", self.sig_hi)

        self.drizzle = QComboBox()
        self.drizzle.addItems(["1.0","1.5","2.0"])
        fl2.addRow("Drizzle factor:", self.drizzle)

        self.debayer_chk = QCheckBox("OSC / color camera  (adds -debayer to calibrate)")
        fl2.addRow("", self.debayer_chk)

        # Equipment profile
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("— None —")
        if HAS_EM:
            try:
                for n in load_all_profiles(): self.profile_combo.addItem(n)
            except Exception: pass
        else:
            self.profile_combo.setEnabled(False)
        fl2.addRow("Equipment profile:", self.profile_combo)
        self.profile_info = dim("No profile selected.")
        fl2.addRow("", self.profile_info)
        self.profile_combo.currentTextChanged.connect(self._on_profile)
        vl.addWidget(grp_st)

        # Manifest preview
        grp_mf = QGroupBox("Manifest preview")
        vl_mf  = QVBoxLayout(grp_mf)
        self.manifest_view = QPlainTextEdit()
        self.manifest_view.setReadOnly(True)
        self.manifest_view.setMinimumHeight(130)
        self.manifest_view.setPlaceholderText(
            "Manifest will appear here after Build is clicked…")
        vl_mf.addWidget(self.manifest_view)

        btn_hl = QHBoxLayout()
        self.b_build = QPushButton("📋  Build manifest")
        self.b_build.setObjectName("warning")
        btn_hl.addWidget(self.b_build)
        self.b_organize = QPushButton("📁  Organize files  ▶")
        self.b_organize.setObjectName("primary")
        self.b_organize.setEnabled(False)
        btn_hl.addWidget(self.b_organize)
        vl_mf.addLayout(btn_hl)
        vl.addWidget(grp_mf)

        self.b_build.clicked.connect(self._build_manifest_preview)
        self.b_organize.clicked.connect(self._request_organize)

        self._structure   = {}
        self._manifest    = {}

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self,"Select output folder")
        if d: self.out_edit.setText(d)

    def _on_profile(self, name):
        if not HAS_EM or name == "— None —":
            self.profile_info.setText("No profile selected."); return
        try:
            v = get_siril_vars(name)
            parts = []
            for k, label in [("gain","Gain"),("drizzle","Drizzle"),
                              ("bayer","Bayer"),("stack_rejection","Rejection")]:
                if v.get(k): parts.append(f"{label}: {v[k]}")
            self.profile_info.setText("  ·  ".join(parts) or "Profile loaded.")
        except Exception as e:
            self.profile_info.setText(f"Error: {e}")

    def set_structure(self, structure: dict):
        self._structure = structure
        if structure.get("project_root") and not self.out_edit.text().strip():
            self.out_edit.setText(
                os.path.join(structure["project_root"], "output"))

    def get_settings(self) -> dict:
        s = {
            "stacking_method": self.rej_combo.currentText(),
            "sigma_low":       self.sig_lo.value(),
            "sigma_high":      self.sig_hi.value(),
            "drizzle":         float(self.drizzle.currentText()),
            "debayer":         self.debayer_chk.isChecked(),
            "output_dir":      self.out_edit.text().strip(),
            "file_mode":       "move" if self.rb_move.isChecked() else "copy",
            "stacker":         "siril_cli",  # set by Dispatch tab
        }
        # Profile overrides
        pname = self.profile_combo.currentText()
        if HAS_EM and pname != "— None —":
            try:
                v = get_siril_vars(pname)
                if v.get("drizzle"):         s["drizzle"]         = float(v["drizzle"])
                if v.get("stack_rejection"): s["stacking_method"] = v["stack_rejection"]
                if v.get("stack_sigma_low"): s["sigma_low"]       = float(v["stack_sigma_low"])
                if v.get("stack_sigma_high"):s["sigma_high"]      = float(v["stack_sigma_high"])
                if v.get("bayer") and v["bayer"] not in (None,"","mono"):
                    s["debayer"] = True
            except Exception: pass
        return s

    def get_manifest(self): return self._manifest

    def _build_manifest_preview(self):
        if not self._structure.get("filters"):
            QMessageBox.warning(self,"No project","Scan a project first (📁 Project tab)."); return
        s = self.get_settings()
        self._manifest = build_manifest(self._structure, s)
        # Pretty summary
        lines = [f"Manifest v{self._manifest['version']}  —  {self._manifest['created']}",
                 f"Project:  {self._manifest['project_root']}",
                 f"Output:   {self._manifest['output_dir']}",
                 f"Mode:     {self._manifest['file_mode']}",
                 f"Stacking: {self._manifest['stacking']['method']}  "
                 f"σ={self._manifest['stacking']['sigma_low']}/"
                 f"{self._manifest['stacking']['sigma_high']}",
                 ""]
        for b in self._manifest["batches"]:
            tag  = "COMBINE" if b.get("is_combine") else f"{len(b['lights'])} lights"
            dstr = f"  dark:{b['dark_source'] or '—'}  flat:{b['flat_source'] or '—'}"
            lines.append(f"  [{b['order']:>2}] {b['id']:<30} {tag}{'' if b.get('is_combine') else dstr}")
        self.manifest_view.setPlainText("\n".join(lines))
        self.b_organize.setEnabled(True)

    def _request_organize(self):
        # Called by button; wired to main window via signal from outside
        pass   # MainWindow connects b_organize.clicked directly

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DISPATCH  (send to stacker)
# ══════════════════════════════════════════════════════════════════════════════
class DispatchTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        vl = QVBoxLayout(self); vl.setContentsMargins(8,8,8,8); vl.setSpacing(8)

        # Stacker selector
        grp_stk = QGroupBox("Stacker")
        fl      = QFormLayout(grp_stk)
        self.stacker_combo = QComboBox()
        for key, label in STACKER_OPTIONS:
            self.stacker_combo.addItem(label, key)
        fl.addRow("Target stacker:", self.stacker_combo)
        fl.addRow("", dim("Select what receives the manifest and drives processing."))
        self.stacker_combo.currentIndexChanged.connect(self._on_stacker_changed)
        vl.addWidget(grp_stk)

        # Manifest status
        grp_mf = QGroupBox("Manifest status")
        vl_mf  = QVBoxLayout(grp_mf); vl_mf.setSpacing(6)
        self.mf_path_lbl = dim("No manifest loaded.")
        vl_mf.addWidget(self.mf_path_lbl)
        self.mf_status_lbl = QLabel("")
        vl_mf.addWidget(self.mf_status_lbl)

        hl_mf = QHBoxLayout()
        self.b_load_mf = QPushButton("📂  Load existing manifest…")
        self.b_load_mf.clicked.connect(self._load_manifest)
        hl_mf.addWidget(self.b_load_mf)
        hl_mf.addStretch()
        vl_mf.addLayout(hl_mf)
        vl.addWidget(grp_mf)

        # Batch table
        grp_bt = QGroupBox("Batch queue")
        vl_bt  = QVBoxLayout(grp_bt)
        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(
            ["Order","Batch ID","Lights","Stacker","Status"])
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        bh = self.batch_table.horizontalHeader()
        bh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        bh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for i in range(2,5): bh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        vl_bt.addWidget(self.batch_table)
        vl.addWidget(grp_bt)

        # Progress
        grp_pg = QGroupBox("Progress")
        vl_pg  = QVBoxLayout(grp_pg)
        self.prog_bar = QProgressBar()
        self.prog_bar.setTextVisible(True)
        vl_pg.addWidget(self.prog_bar)
        self.op_lbl = QLabel("Idle.")
        vl_pg.addWidget(self.op_lbl)
        vl.addWidget(grp_pg)

        # Buttons
        hl_btn = QHBoxLayout()
        self.b_dispatch = QPushButton("▶  Dispatch to stacker")
        self.b_dispatch.setObjectName("primary")
        self.b_dispatch.setEnabled(False)
        hl_btn.addWidget(self.b_dispatch)
        self.b_resume = QPushButton("↩  Resume (skip done batches)")
        self.b_resume.setEnabled(False)
        hl_btn.addWidget(self.b_resume)
        vl.addLayout(hl_btn)

        self._manifest      = {}
        self._manifest_file = ""

    def _on_stacker_changed(self, _):
        key = self.stacker_combo.currentData()
        if key == "python_stub":
            self.b_dispatch.setText("▶  Generate stub info")
        elif key == "siril_script":
            self.b_dispatch.setText("▶  Generate .ssf script")
        else:
            self.b_dispatch.setText("▶  Dispatch to Siril")

    def _load_manifest(self):
        path, _ = QFileDialog.getOpenFileName(
            self,"Load manifest","",
            f"Manifest ({MANIFEST_NAME});;JSON (*.json);;All (*)")
        if path:
            try:
                m = load_manifest(path)
                self.set_manifest(m, path)
            except Exception as e:
                QMessageBox.warning(self,"Load failed",str(e))

    def set_manifest(self, manifest: dict, mf_path: str):
        self._manifest      = manifest
        self._manifest_file = mf_path
        self.mf_path_lbl.setText(f"  {mf_path}")
        self.mf_status_lbl.setText(manifest_summary(manifest))
        self._fill_batch_table(manifest)
        has_pending = any(b["status"]=="pending" for b in manifest.get("batches",[]))
        has_done    = any(b["status"]=="done"    for b in manifest.get("batches",[]))
        self.b_dispatch.setEnabled(has_pending)
        self.b_resume.setEnabled(has_done and has_pending)
        # Sync stacker combo to manifest
        key = manifest.get("stacker","siril_cli")
        for i in range(self.stacker_combo.count()):
            if self.stacker_combo.itemData(i) == key:
                self.stacker_combo.setCurrentIndex(i); break

    def get_stacker_key(self) -> str:
        return self.stacker_combo.currentData()

    def get_manifest(self): return self._manifest
    def get_manifest_file(self): return self._manifest_file

    def _fill_batch_table(self, manifest: dict):
        self.batch_table.setRowCount(0)
        batches = manifest.get("batches",[])
        stacker = manifest.get("stacker","siril_cli")
        for row, b in enumerate(sorted(batches, key=lambda x: x["order"])):
            self.batch_table.insertRow(row)
            self.batch_table.setItem(row,0, ci(str(b["order"])))
            self.batch_table.setItem(row,1, ci(b["id"]))
            lcount = "COMBINE" if b.get("is_combine") else str(len(b["lights"]))
            self.batch_table.setItem(row,2, ci(lcount))
            self.batch_table.setItem(row,3, ci(stacker))
            status = b["status"]
            sc = {
                "pending": SIRIL_TEXT_DIM,
                "done":    SIRIL_SUCCESS,
                "failed":  SIRIL_ERROR,
                "skipped": SIRIL_WARNING,
            }.get(status, SIRIL_TEXT)
            self.batch_table.setItem(row,4, ci(status, sc))

    def refresh_table(self):
        if self._manifest: self._fill_batch_table(self._manifest)

    def update_progress(self, step, total, desc):
        self.prog_bar.setMaximum(max(total,1))
        self.prog_bar.setValue(step)
        self.op_lbl.setText(desc)

    def set_dispatching(self, running: bool):
        self.b_dispatch.setEnabled(not running)
        self.b_resume.setEnabled(not running)
        self.b_load_mf.setEnabled(not running)
        self.stacker_combo.setEnabled(not running)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — HELP  (inline folder convention docs)
# ══════════════════════════════════════════════════════════════════════════════
HELP_HTML = """
<style>
body  { color:#dde3ee; font-family:'Segoe UI',Arial,sans-serif; font-size:10pt; }
h2    { color:#5ba3ff; border-bottom:1px solid #3a4055; padding-bottom:4px; }
h3    { color:#4a9eff; margin-top:14px; }
code  { background:#2d3240; color:#4caf7d; padding:1px 5px;
        border-radius:3px; font-family:'Courier New',monospace; font-size:9pt; }
pre   { background:#181c22; color:#4caf7d; padding:10px; border-radius:5px;
        font-family:'Courier New',monospace; font-size:9pt; line-height:1.5; }
.warn { color:#e8a23a; font-weight:bold; }
.ok   { color:#4caf7d; }
.dim  { color:#7a8499; }
table { border-collapse:collapse; width:100%; margin:8px 0; }
th    { background:#2d3240; color:#4a9eff; padding:5px 10px;
        border:1px solid #3a4055; text-align:left; }
td    { padding:4px 10px; border:1px solid #3a4055; }
tr:nth-child(even) td { background:#252930; }
</style>

<h2>📁  Folder Convention</h2>
<p>MSPM reads <b>only folder names</b> — no FITS headers required.
Select your project root and the scanner does the rest automatically.</p>

<h3>Expected Structure</h3>
<pre>
/MyProject/                      ← project root (you select this)
│
├── Ha/                          ← filter name  (any string)
│   ├── 2025-03-01/              ← night folder (any string, sorted A→Z)
│   │   ├── lights/              ← MUST be named exactly "lights"
│   │   ├── darks/               ← optional, named exactly "darks"
│   │   └── flats/               ← optional, named exactly "flats"
│   ├── 2025-03-15/
│   │   └── lights/              ← no darks here → will inherit
│   └── bias/                    ← filter-level shared bias
│
├── OIII/
│   ├── 2025-03-01/
│   │   └── lights/
│   └── bias/
│
├── darks/                       ← project-level fallback darks
├── bias/                        ← project-level fallback bias
└── mspm_manifest.json           ← written by MSPM after Build
</pre>

<h3>Detection Rules</h3>
<table>
<tr><th>What</th><th>How detected</th></tr>
<tr><td><b>Filters</b></td><td>Any subfolder of project root that contains at least one
subfolder which itself contains a <code>lights/</code> folder.</td></tr>
<tr><td><b>Nights</b></td><td>Any subfolder of a filter folder that contains a
<code>lights/</code> folder.</td></tr>
<tr><td><b>Light frames</b></td><td>All <code>.fit .fits .fts</code> files inside
a <code>lights/</code> folder.</td></tr>
<tr><td><b>Darks / Flats / Bias</b></td><td>All FITS inside <code>darks/</code>,
<code>flats/</code>, <code>bias/</code> at the appropriate level.</td></tr>
</table>

<p class="dim">Folder names that are always ignored at root level:
<code>darks</code> <code>flats</code> <code>bias</code>
<code>lights</code> <code>output</code> <code>.mspm_work</code></p>

<h2>🔗  Calibration Inheritance</h2>
<p>For each night's lights, MSPM searches <b>four levels</b> in order,
stopping at the first match:</p>
<table>
<tr><th>Level</th><th>Example path</th><th>Label shown</th></tr>
<tr><td>1 — Same night</td><td><code>Ha/2025-03-01/darks/</code></td>
    <td class="ok">✓ own</td></tr>
<tr><td>2 — Sibling night</td><td><code>Ha/2025-03-08/darks/</code></td>
    <td style="color:#4a9eff">↑ sibling:2025-03-08</td></tr>
<tr><td>3 — Filter level</td><td><code>Ha/darks/</code></td>
    <td style="color:#4a9eff">↑ filter</td></tr>
<tr><td>4 — Project level</td><td><code>darks/</code></td>
    <td style="color:#4a9eff">↑ project</td></tr>
</table>
<p>Each calibration type (darks / flats / bias) is resolved <b>independently</b>.
A night can use its own darks but inherit flats from the filter level — that's normal.</p>
<p class="warn">⚠  Missing darks or flats trigger a warning but do not abort.
Missing bias is info-only (many workflows skip bias entirely).</p>

<h2>📋  The Manifest  (<code>mspm_manifest.json</code>)</h2>
<p>The manifest is the <b>contract</b> between MSPM and any stacker.
It is written to the project root after you click <b>Build manifest</b>.</p>
<ul>
  <li>Each <b>batch</b> = one night for one filter + its resolved calibration paths.</li>
  <li>Batches have a <b>processing order</b> (per-night sub-stacks first,
      then a final combine batch per filter).</li>
  <li>Each batch has a <b>status</b>: <code>pending → done / failed / skipped</code>.</li>
  <li>If a run is interrupted, MSPM resumes from the first <code>pending</code> batch.</li>
</ul>

<h2>🔀  Three-Stage Workflow</h2>
<table>
<tr><th>Stage</th><th>Tab</th><th>What happens</th></tr>
<tr><td><b>1 — Scan</b></td><td>📁 Project</td>
    <td>Reads folder names, pairs calibration via inheritance, shows tree + pairing table.</td></tr>
<tr><td><b>2 — Organize</b></td><td>📁 Organize</td>
    <td>Copies or moves files into structured work dirs. Writes <code>mspm_manifest.json</code>.</td></tr>
<tr><td><b>3 — Dispatch</b></td><td>▶ Dispatch</td>
    <td>Reads the manifest, sends batches to the selected stacker in order.
        Marks each batch done/failed. Can resume interrupted runs.</td></tr>
</table>

<h2>🔧  Stacker Options</h2>
<table>
<tr><th>Option</th><th>What it does</th></tr>
<tr><td><b>Siril CLI</b></td>
    <td>Connects to Siril via sirilpy and drives it batch by batch in real time.
        Best for automated unattended runs.</td></tr>
<tr><td><b>Siril script (.ssf)</b></td>
    <td>Generates a <code>mspm_pipeline.ssf</code> script file you can load in
        Siril's Script menu. Good for review-before-run workflows.</td></tr>
<tr><td><b>Custom Python stacker</b></td>
    <td>Placeholder for future algorithms. MSPM prints the manifest path and
        batch list so your own script can pick it up.</td></tr>
</table>

<h2>💡  Tips</h2>
<ul>
  <li>Night folder names can be anything — dates, session names, camera IDs.
      They are sorted alphabetically, so <code>YYYY-MM-DD</code> format is recommended.</li>
  <li>Filter names can be anything: <code>Ha</code>, <code>Hα</code>,
      <code>L</code>, <code>Red</code>, <code>CH4</code>.</li>
  <li>You can have multiple projects open. Each has its own manifest.</li>
  <li>Use <b>Copy</b> mode until you've verified the first run. Switch to
      <b>Move</b> once you trust the pipeline.</li>
  <li>The <code>.mspm_work/</code> folder is the only thing MSPM creates inside
      your project. Delete it to fully reset.</li>
</ul>
"""

class HelpTab(QWidget):
    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self); vl.setContentsMargins(8,8,8,8)
        tb = QTextBrowser()
        tb.setHtml(HELP_HTML)
        tb.setOpenExternalLinks(False)
        vl.addWidget(tb)

# ══════════════════════════════════════════════════════════════════════════════
#  LOG WIDGET  (shared, docked at bottom of log tab)
# ══════════════════════════════════════════════════════════════════════════════
class LogTab(QWidget):
    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self); vl.setContentsMargins(8,8,8,8); vl.setSpacing(6)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(QFont("Courier New",9))
        vl.addWidget(self.log)
        hl = QHBoxLayout(); hl.addStretch()
        b_save = QPushButton("💾  Save log…"); b_save.clicked.connect(self._save)
        hl.addWidget(b_save)
        b_clr = QPushButton("🗑  Clear"); b_clr.clicked.connect(self.log.clear)
        hl.addWidget(b_clr)
        vl.addLayout(hl)

    def append(self, t):
        self.log.appendPlainText(t)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _save(self):
        p,_ = QFileDialog.getSaveFileName(self,"Save log","",
                                           "Text (*.txt);;All (*)")
        if p:
            try:
                open(p,"w",encoding="utf-8").write(self.log.toPlainText())
                QMessageBox.information(self,"Saved",p)
            except Exception as e:
                QMessageBox.warning(self,"Error",str(e))

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._worker       = None
        self._cancel_event = threading.Event()
        self.setWindowTitle("Multi-Session Pipeline Manager  —  Siril")
        self.resize(1060, 770)
        self._build()
        self._wire()

    def _build(self):
        c  = QWidget(); self.setCentralWidget(c)
        vl = QVBoxLayout(c); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)

        # Title bar
        tb = QWidget()
        tb.setStyleSheet(f"background:{SIRIL_BG3};border-bottom:1px solid {SIRIL_BORDER};")
        tbl = QHBoxLayout(tb); tbl.setContentsMargins(12,6,12,6)
        ltt = QLabel("🔭  Multi-Session Pipeline Manager")
        ltt.setStyleSheet(f"color:{SIRIL_ACCENT};font-size:13pt;font-weight:bold;")
        tbl.addWidget(ltt); tbl.addStretch()
        self.lbl_proj = QLabel(""); self.lbl_proj.setObjectName("dim")
        tbl.addWidget(self.lbl_proj)
        tbl.addWidget(dim(f"v{VERSION}"))
        vl.addWidget(tb)

        # Tabs
        self.tabs = QTabWidget()
        self.t_project  = ProjectTab()
        self.t_organize = OrganizeTab()
        self.t_dispatch = DispatchTab()
        self.t_help     = HelpTab()
        self.t_log      = LogTab()
        self.tabs.addTab(self.t_project,  "📁  Project")
        self.tabs.addTab(self.t_organize, "🗂  Organize")
        self.tabs.addTab(self.t_dispatch, "▶  Dispatch")
        self.tabs.addTab(self.t_help,     "❓  Help")
        self.tabs.addTab(self.t_log,      "📋  Log")
        vl.addWidget(self.tabs, 1)

        # Bottom bar
        bb = QWidget()
        bb.setStyleSheet(f"background:{SIRIL_BG3};border-top:1px solid {SIRIL_BORDER};")
        bbl = QHBoxLayout(bb); bbl.setContentsMargins(10,6,10,6)
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setObjectName("dim")
        bbl.addWidget(self.status_lbl, 1)
        self.cancel_btn = QPushButton("✕  Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        bbl.addWidget(self.cancel_btn)
        vl.addWidget(bb)

    def _wire(self):
        self.t_project.scanned.connect(self._on_scanned)
        self.t_organize.b_organize.clicked.connect(self._run_organize)
        self.t_dispatch.b_dispatch.clicked.connect(lambda: self._run_dispatch(resume=False))
        self.t_dispatch.b_resume.clicked.connect(lambda: self._run_dispatch(resume=True))

    # ── scan complete ──────────────────────────────────────────────────────
    def _on_scanned(self, structure: dict):
        self.t_organize.set_structure(structure)
        self.lbl_proj.setText(f"  {structure['project_root']}")
        n = sum(len(v) for v in structure["nights"].values())
        self.status_lbl.setText(
            f"Scanned: {len(structure['filters'])} filter(s), {n} night(s)")
        for w in structure["warnings"]:
            self.t_log.append(f"[WARN] {w}")

        # Auto-load manifest if one exists
        mp = manifest_path(structure["project_root"])
        if os.path.isfile(mp):
            try:
                m = load_manifest(mp)
                self.t_dispatch.set_manifest(m, mp)
                self.t_log.append(f"[INFO] Existing manifest loaded: {mp}")
                self.t_log.append(f"       {manifest_summary(m)}")
            except Exception as e:
                self.t_log.append(f"[WARN] Could not load manifest: {e}")

    # ── organize stage ─────────────────────────────────────────────────────
    def _run_organize(self):
        st = self.t_project.get_structure()
        if not st.get("filters"):
            QMessageBox.warning(self,"No project","Scan a project first."); return

        settings          = self.t_organize.get_settings()
        settings["stacker"] = self.t_dispatch.get_stacker_key()

        manifest           = build_manifest(st, settings)
        manifest["stacker"]= self.t_dispatch.get_stacker_key()

        # If existing manifest with done batches, carry over status
        mp = manifest_path(st["project_root"])
        if os.path.isfile(mp):
            try:
                old = load_manifest(mp)
                old_map = {b["id"]: b["status"] for b in old.get("batches",[])}
                for b in manifest["batches"]:
                    if old_map.get(b["id"]) in ("done","skipped"):
                        b["status"] = old_map[b["id"]]
            except Exception:
                pass

        save_manifest(manifest, mp)
        self.t_dispatch.set_manifest(manifest, mp)
        self.t_log.append(f"[MANIFEST] Written to {mp}")

        mode = settings["file_mode"]
        reply = QMessageBox.question(
            self, "Organize files",
            f"{'Move' if mode=='move' else 'Copy'} files into work directories?\n\n"
            f"Work dir: {st['project_root']}/.mspm_work/\n"
            f"{'⚠ MOVE is irreversible from the source location.' if mode=='move' else ''}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes: return

        self._cancel_event.clear()
        self.cancel_btn.setEnabled(True)
        self.status_lbl.setText("Organizing files…")
        self.t_log.append(f"\n[ORGANIZE] {ts()}")

        self._worker = OrganizeWorker(manifest, self._cancel_event)
        self._worker.log_line.connect(self.t_log.append)
        self._worker.progress.connect(self._on_org_progress)
        self._worker.finished.connect(self._on_org_done)
        self._worker.start()

    def _on_org_progress(self, step, total, desc):
        self.status_lbl.setText(f"Organizing… {step}/{total}  {desc}")

    def _on_org_done(self, result):
        self.cancel_btn.setEnabled(False)
        if result["success"]:
            self.status_lbl.setText("Organize complete. Ready to dispatch.")
            self.t_log.append("[ORGANIZE] Done.")
            self.tabs.setCurrentWidget(self.t_dispatch)
        else:
            self.status_lbl.setText(f"Organize failed: {result.get('error','')}")
            QMessageBox.critical(self,"Organize failed",result.get("error",""))

    # ── dispatch stage ─────────────────────────────────────────────────────
    def _run_dispatch(self, resume: bool):
        manifest      = self.t_dispatch.get_manifest()
        manifest_file = self.t_dispatch.get_manifest_file()

        if not manifest or not manifest.get("batches"):
            QMessageBox.warning(self,"No manifest",
                "Build and organize first, or load an existing manifest."); return

        # Set stacker key from combo
        stacker = self.t_dispatch.get_stacker_key()
        manifest["stacker"] = stacker
        save_manifest(manifest, manifest_file)

        if not resume:
            # Reset all pending (re-run from scratch pending only)
            pass  # pending batches will run; done ones are skipped by worker

        self._cancel_event.clear()
        self.cancel_btn.setEnabled(True)
        self.t_dispatch.set_dispatching(True)
        self.status_lbl.setText(f"Dispatching to {stacker}…")
        self.t_log.append(f"\n[DISPATCH] {ts()}  stacker={stacker}")

        self._worker = DispatchWorker(manifest, manifest_file, self._cancel_event)
        self._worker.log_line.connect(self.t_log.append)
        self._worker.progress.connect(self._on_disp_progress)
        self._worker.finished.connect(self._on_disp_done)
        self._worker.start()

    def _on_disp_progress(self, step, total, desc):
        self.t_dispatch.update_progress(step, total, desc)
        self.status_lbl.setText(f"Dispatching… {step}/{total}  {desc}")
        self.t_dispatch.refresh_table()

    def _on_disp_done(self, result):
        self.cancel_btn.setEnabled(False)
        self.t_dispatch.set_dispatching(False)
        self.t_dispatch.refresh_table()

        manifest = self.t_dispatch.get_manifest()
        summary  = manifest_summary(manifest)

        if result["success"]:
            self.status_lbl.setText(f"Dispatch complete.  {summary}")
            self.t_log.append(f"[DISPATCH] Done.  {summary}")
            QMessageBox.information(self,"Done",
                f"Dispatch finished.\n\n{summary}")
        else:
            err = result.get("error","")
            self.status_lbl.setText(f"Dispatch failed: {err}")
            self.t_log.append(f"[ERROR] {err}")
            QMessageBox.critical(self,"Dispatch failed",
                f"{err}\n\nCheck Log tab. {summary}")

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._cancel_event.set()
            self.cancel_btn.setEnabled(False)
            self.status_lbl.setText("Cancelling… finishing current step")
            self.t_log.append("[CANCEL] requested")

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(SIRIL_STYLESHEET)
    w = MainWindow()
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
