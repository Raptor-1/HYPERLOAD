"""
HYPERLOAD — Main Entry Point
=============================
This is the ONLY file Siril needs to know about.

Setup:
  1. Download HYPERLOAD.zip and extract anywhere
  2. In Siril: Preferences → Scripts → add the HYPERLOAD/ folder
  3. HYPERLOAD appears in Siril's script menu
  4. Click it — this file runs — the full launcher opens

Folder structure (auto-discovered, zero hardcoded paths):
  HYPERLOAD/
  ├── hyperload.py          ← this file
  ├── hyperload_config.json ← auto-created on first run
  ├── assets/
  │   └── logo.png
  ├── Suites/               ← merged multi-tab suite scripts
  │   ├── image_processing_suite.py
  │   ├── stacking_suite.py
  │   └── ...
  └── Scripts/              ← standalone scripts + equipment_manager
      ├── equipment_manager.py
      ├── wavelet_stretch.py
      └── ...

Version: 2.0.0
"""

import sys
import os
import json
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime

# ── Root resolution — works on ANY machine, ANY path ─────────────────────────
ROOT        = Path(__file__).parent.resolve()
SUITES_DIR  = ROOT / "Suites"
SCRIPTS_DIR = ROOT / "Scripts"
ASSETS_DIR  = ROOT / "assets"
CONFIG_FILE = ROOT / "hyperload_config.json"
LOG_FILE    = ROOT / "hyperload_log.txt"
LOGO        = ASSETS_DIR / "logo.png"

# ── Crash handler ─────────────────────────────────────────────────────────────
def _crash(et, ev, eb):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nhyperload.py CRASH\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)

sys.excepthook = _crash

# ── Python version guard ──────────────────────────────────────────────────────
if sys.version_info < (3, 10):
    print(f"HYPERLOAD requires Python 3.10+. You have {sys.version}.")
    print("Update Python and try again.")
    sys.exit(1)

# ── Inject Suites/ and Scripts/ into the module search path ──────────────────
# This allows any suite to do: from equipment_manager import ...
# without knowing the absolute path — it just works.
for d in [str(SUITES_DIR), str(SCRIPTS_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

# ── Ensure PyQt6 is available ─────────────────────────────────────────────────
try:
    import sirilpy as s
    s.ensure_installed("PyQt6")
    s.ensure_installed("astropy")
    s.ensure_installed("scipy")
    s.ensure_installed("matplotlib")
except ImportError:
    pass  # Running outside Siril — deps assumed present

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtGui import QIcon
except ImportError:
    print("PyQt6 not found. Install with: pip install PyQt6")
    sys.exit(1)

# ── Default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "version":      "2.0.0",
    "working_dir":  "",            # filled with user's home on first run
    "import_dir":   "",
    "export_dir":   "",
    "temp_dir":     "",
    "loaded_suites": [],           # empty = all loaded
    "open_windows":  {},
    "thread_count":  os.cpu_count() or 4,
    "use_memmap":    True,         # astropy memmap FITS — huge RAM saving
    "result_cache":  True,         # hash-based result reuse
    "lazy_import":   True,         # import scripts only on Run
}

def load_config() -> dict:
    """Load config, create defaults if missing."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            # Merge in any new keys from defaults
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass  # Corrupted — recreate

    cfg = DEFAULT_CONFIG.copy()
    home = Path.home()
    cfg["working_dir"] = str(ROOT)
    cfg["import_dir"]  = str(home / "Pictures")
    cfg["export_dir"]  = str(home / "Pictures" / "Processed")
    cfg["temp_dir"]    = str(home / ".hyperload_temp")
    save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[HYPERLOAD] Could not save config: {e}")


def discover_suites() -> list[dict]:
    """
    Scan Suites/ folder for .py files.
    Returns list of dicts: {id, name, path, loaded}
    Reads the module docstring for display name — no import needed.
    """
    suites = []
    if not SUITES_DIR.exists():
        return suites
    for pyfile in sorted(SUITES_DIR.glob("*.py")):
        if pyfile.name.startswith("_"):
            continue
        # Extract suite name from filename (no import = zero startup cost)
        sid  = pyfile.stem
        name = sid.replace("_", " ").replace("suite", "Suite").title()
        suites.append({
            "id":   sid,
            "name": name,
            "path": pyfile,
        })
    return suites


def discover_scripts() -> list[dict]:
    """
    Scan Scripts/ folder for .py files.
    Returns list of dicts: {id, name, path}
    """
    scripts = []
    if not SCRIPTS_DIR.exists():
        return scripts
    for pyfile in sorted(SCRIPTS_DIR.glob("*.py")):
        if pyfile.name.startswith("_"):
            continue
        sid  = pyfile.stem
        name = sid.replace("_", " ").title()
        scripts.append({
            "id":   sid,
            "name": name,
            "path": pyfile,
        })
    return scripts


def lazy_import(path: Path):
    """
    Import a script module on demand (when user clicks Run).
    NOT called at startup — only when the user actually wants to run something.
    This keeps startup time and memory usage minimal.
    """
    spec   = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_structure() -> list[str]:
    """Check folder structure and return list of warnings."""
    warnings = []
    if not SUITES_DIR.exists():
        warnings.append(f"Suites/ folder not found at {SUITES_DIR}\n"
                        f"Create it and add suite scripts inside.")
    if not SCRIPTS_DIR.exists():
        warnings.append(f"Scripts/ folder not found at {SCRIPTS_DIR}\n"
                        f"Create it and add standalone scripts inside.")
    if not ASSETS_DIR.exists():
        warnings.append("assets/ folder not found — logo and icons will be missing.")
    n_suites  = len(list(SUITES_DIR.glob("*.py"))) if SUITES_DIR.exists() else 0
    n_scripts = len(list(SCRIPTS_DIR.glob("*.py"))) if SCRIPTS_DIR.exists() else 0
    if n_suites == 0 and n_scripts == 0:
        warnings.append("No .py files found in Suites/ or Scripts/.\n"
                        "Check that HYPERLOAD.zip was extracted correctly.")
    return warnings


# ── Main launcher ─────────────────────────────────────────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    if LOGO.exists():
        app.setWindowIcon(QIcon(str(LOGO)))

    # Validate structure — show error and exit if critical files missing
    warnings = validate_structure()
    if warnings:
        msg = QMessageBox()
        msg.setWindowTitle("HYPERLOAD — Setup Warning")
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText("HYPERLOAD found some issues with the folder structure:")
        msg.setDetailedText("\n\n".join(warnings))
        msg.setInformativeText("You can continue, but some features may not work.")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if msg.exec() == QMessageBox.StandardButton.Cancel:
            sys.exit(0)

    # Load config (creates defaults on first run)
    config = load_config()

    # Discover available suites and scripts (no imports — just file names)
    suites  = discover_suites()
    scripts = discover_scripts()

    log_entry = (f"\n{datetime.now()} | HYPERLOAD started | "
                 f"{len(suites)} suites | {len(scripts)} scripts | "
                 f"root={ROOT}")
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

    # Import and launch the main window
    # The launcher UI itself lives in Scripts/hyperload_launcher.py
    # (kept separate so hyperload.py stays minimal and readable)
    launcher_path = SCRIPTS_DIR / "hyperload_launcher.py"
    if launcher_path.exists():
        launcher_mod = lazy_import(launcher_path)
        launcher_mod.run(
            app=app,
            root=ROOT,
            suites=suites,
            scripts=scripts,
            config=config,
            save_config_fn=save_config,
            lazy_import_fn=lazy_import,
            logo_path=LOGO,
        )
    else:
        # Fallback: minimal launcher built into this file
        _run_minimal_launcher(app, config, suites, scripts)


def _run_minimal_launcher(app, config, suites, scripts):
    """
    Minimal fallback launcher if hyperload_launcher.py is missing.
    Shows discovered suites and scripts as a simple list.
    """
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QLabel, QListWidget
    )
    win = QMainWindow()
    win.setWindowTitle("HYPERLOAD")
    win.resize(600, 400)
    body = QWidget(); lay = QVBoxLayout(body)
    lay.addWidget(QLabel(f"Root: {ROOT}"))
    lay.addWidget(QLabel(f"{len(suites)} suites found in Suites/"))
    lw = QListWidget()
    for s in suites: lw.addItem(s["name"])
    lay.addWidget(lw)
    lay.addWidget(QLabel(f"{len(scripts)} scripts found in Scripts/"))
    win.setCentralWidget(body)
    win.show()
    sys.exit(app.exec())


# ── Siril entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
