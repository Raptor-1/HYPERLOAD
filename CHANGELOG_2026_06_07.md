# HYPERLOAD Update — June 7, 2026

## What's new in this update

---

### Launcher (hyperload_launcher.py)

**Fixed — scripts no longer freeze on launch**
The most significant fix in this update. Previously clicking Run on any script caused the launcher to freeze completely. Root cause: every script creates its own `QApplication` instance, and launching them inside the existing launcher process caused Qt to deadlock when a second `QApplication` was created. Fixed by launching each script as a fully independent OS process via `subprocess.Popen`. Each script now opens in its own window with its own Python interpreter — isolated, stable, and crash-safe.

**Fixed — stylesheet parse errors on every panel open**
Every time a script panel was opened, Qt logged `Could not parse stylesheet of object ScriptPanel`. Caused by `QFrame#objectName()` returning an empty string at stylesheet parse time, generating invalid CSS selector `QFrame#,`. Fixed by applying styles directly to the widget instance without a class/ID selector.

**Added — crash detection with error display**
If a launched script exits with an error within 3 seconds of launch, the launcher now reads the stderr output from the process and shows a popup with the full Python traceback. Previously errors were silent — you only saw a cryptic exit code in the Siril log.

**Fixed — file dock**
Removed the hardcoded example files that appeared in the dock on every launch. Dock now starts empty. Added `＋ Add files…` button to load real files from disk and `✕ Clear` button to empty the dock.

**UI — font sizes increased**
All font sizes increased by 4px across the board following user feedback that text was too small. Base UI font: 11px → 15px. All other sizes scale proportionally.

**UI — sidebar widened**
Sidebar width increased from 228px to 310px to prevent text clipping after the font size increase.

**UI — suite row and script row heights increased**
Suite rows: 30px → 38px. Script rows: 24px → 30px. Rows now fit the larger font without clipping.

**UI — chevron indicators improved**
Replaced the rotation-based chevron animation (small `▶` that rotated 90°) with explicit `▶`/`▼` glyphs at 15px font. Closed and open states now look visually consistent.

---

### ab_compare.py

**Fixed — script failed to open (AttributeError)**
`_build_top_bar()` tried to connect a button to `self._canvas.reset_zoom` before `self._canvas` was constructed. Fixed with a lambda to defer the lookup until the button is actually clicked.

**Fixed — script failed to open (NameError: QShortcut)**
`QShortcut` moved from `QtWidgets` to `QtGui` in PyQt6. Added to the correct import line.

---

### blind_deconv.py

**Fixed — script failed to open (SyntaxError)**
The file's opening `r"""` docstring was never closed before the code began. Python's parser closed it at the `f"""` of the stylesheet definition, leaving the stylesheet content as raw Python code. Fixed by properly closing the docstring after the title lines.

**Fixed — deconvolution failed (KeyError: 'x')**
DAOStarFinder column names changed across photutils versions:
- Pre-0.7: `x`, `y`
- 0.7–1.x: `xcentroid`, `ycentroid`
- 2.x (current Siril venv): `x_centroid`, `y_centroid`

The script used the old `x`/`y` names. Replaced with a runtime probe that detects whichever naming convention is present, covering all five known variants. Logs the detected column names for diagnostic purposes.

**Fixed — output FITS save failed (ValueError: non-ASCII)**
The FITS HISTORY header contained an em-dash character `—` (Unicode U+2014). FITS headers require strict 7-bit ASCII. Replaced with a plain hyphen. Full sweep of all non-ASCII characters (em-dashes, curly quotes, ellipsis, degree signs) replaced with ASCII equivalents throughout the file.

---

### pattern_noise_corrector.py

**Fixed — green colour scheme replaced with Siril palette**
The script was using a custom green-tinted colour palette (`#0e1210` background, `#50d060` accent, `#c8e0c0` text) inconsistent with the rest of HYPERLOAD. Replaced with the standard Siril dark grey/blue palette used across all other scripts (`#1e2128` background, `#4a9eff` accent, `#dde3ee` text).

---

### local_norm.py

**Fixed — left panel too narrow, text clipped with scrollbar**
Fixed panel width from 380px to 520px to match the increased font size (proportional to the 11px → 15px font increase across the suite).

---

### hyperload.py (entry point)

No functional changes. The fallback minimal launcher (list view) now correctly triggers only when `hyperload_launcher.py` is genuinely missing from `Scripts/`, not as a default.

---

### License

HYPERLOAD is now formally licensed under **GPL-3.0** following community feedback. PyQt6 is GPL-3.0 licensed and requires projects using it to also be GPL-3.0. The previous custom license (free personal use, commercial use requires permission) was incompatible with this requirement. GPL-3.0 is consistent with Siril itself, which is also GPL-3.0.

---

### Known issues / in progress

- Scripts using DAOStarFinder (`psf_heatmap`, `seeing_estimator`, `transient_detector`, `subframe_ranking`) will need the same `x_centroid` column fix — to be addressed in the next update as each script is tested
- Scripts with fixed left panel widths may need widening at the new font size — being fixed on a per-script basis as reported
- `blind_deconv` produces poor results on raw uncalibrated single frames (expected behaviour — requires stacked calibrated input)

---

*Testing environment: Windows 11, Siril venv, Python 3.14, photutils 2.x, PyQt6*
