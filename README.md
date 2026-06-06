# HYPERLOAD
### Science-Grade Astrophotography Script Suite for Siril

HYPERLOAD is a collection of 67+ Python scripts that extend [Siril](https://siril.org) — the free, open-source astrophotography application — with algorithms from cutting-edge research papers and professional observatory pipelines. The goal is to match or exceed the capabilities of commercial software like PixInsight, entirely within a free and open ecosystem.

Every script has a full **PyQt6 graphical interface**. No command line required. All processing runs in background threads so the UI stays responsive.

---

## What's included

| Suite | Scripts | What it does |
|---|---|---|
| **Image Processing** | 8 | Wavelet stretch, dark channel haze removal, mask builder, narrowband palettes, local normalisation, blind deconvolution, pattern noise correction |
| **Stacking** | 4 | Winsorized sigma-clipping stack, subframe ranking, drizzle resampling, multi-session manager |
| **Lucky & Restoration** | 8 | Lucky imaging, bispectrum speckle, speckle holography, The Thresher, DIPLI (deep-learning), ImageMM, phase diversity, ADI-KLIP |
| **Solar System** | 6 | Comet dual-stack pipeline, coma analyser, moving object / MPC tracklets, planet derotation, texture mapper, orbit determination |
| **Variable Star** | 6 | Differential photometry, Broeg optimal weighting, extinction map, Lomb-Scargle period finder, transit MCMC, occultation timing |
| **Transient Pipeline** | 4 | ZOGY subtraction, SFFT subtraction, transient detector, real/bogus vetter |
| **Session QA** | 5 | Seeing estimator, PSF heatmap, astrometric distortion mapper, proper motion finder, A/B comparator |
| **Spectroscopy** | 3 | 1D spectral extraction, CCD fringe corrector, radial velocity via CCF |

Plus **15 standalone scripts**: bias drift corrector, cosmic ray rejection, dark temperature corrector, PSF photometry, mosaic photometric matching, planetary zone stacker, satellite trail remover, software ADC, background gradient, exoplanet transit suite, SYSREM detrending, Sersic profiler, spatially varying PSF deconvolution, and more.

For full descriptions of every script and algorithm see **[HYPERLOAD_Description.pdf](HYPERLOAD_Description.pdf)**.

---

## How it works

```
HYPERLOAD/
├── hyperload.py          ← The only file Siril needs to know about
├── HYPERLOAD_Description.pdf
├── assets/
│   └── logo.png
├── Suites/               ← 8 merged multi-tab suite scripts
│   ├── image_processing_suite.py
│   ├── stacking_suite.py
│   ├── lucky_suite.py
│   ├── solar_system_suite.py
│   ├── variable_star_suite.py
│   ├── transient_pipeline.py
│   ├── session_qa_suite.py
│   └── spectroscopy_suite.py
└── Scripts/              ← All standalone scripts + launcher UI
    ├── hyperload_launcher.py
    ├── equipment_manager.py
    └── ... (60+ scripts)
```

When Siril executes `hyperload.py`, it:
1. Discovers all scripts in `Suites/` and `Scripts/` automatically
2. Launches the full PyQt6 launcher window
3. Lets you open any script as a floating panel from the tree sidebar

Scripts run as **separate OS processes** — each gets its own window, its own Python interpreter, and cannot crash the launcher.

---

## Installation

### Requirements

- [Siril](https://siril.org) (any recent version)
- Python 3.10 or newer
- The following Python packages:

```
pip install PyQt6 astropy scipy numpy matplotlib photutils astroquery
```

GPU-accelerated scripts (DIPLI, some stacking) additionally benefit from:
```
pip install torch cupy-cuda12x
```
*(optional — scripts fall back to CPU automatically if not installed)*

---

### Setup — 3 steps

**1. Download**

Click the green **Code** button above → **Download ZIP**

Extract the ZIP anywhere on your computer. The folder can live on your Desktop, Documents, an external drive — anywhere.

**2. Add to Siril**

Open Siril. Go to:

> **Edit → Preferences → Scripts → Script directories → Add**

Select the `HYPERLOAD/` folder (the one containing `hyperload.py`). Click OK and restart Siril.

**3. Run**

In Siril's top menu, go to **Scripts** → click **hyperload**.

The launcher window opens. From there, click any suite or script to open its panel.

---

## The Launcher

The HYPERLOAD launcher is a PixInsight-style workspace with:

- **Suite tree** on the left — expand any suite to see its scripts, click a script to open its panel
- **Floating panels** — each script opens as a movable, resizable, collapsible window in the workspace
- **File dock** at the bottom — drag FITS, SER or CSV files onto script panels to queue them for processing
- **Live system stats** — CPU, RAM and disk usage shown in the top bar
- **Path management** — set working, import and export directories once; all scripts inherit them

Panels can be:
- **Moved** by dragging the title bar
- **Resized** from any edge or corner
- **Collapsed** to just the title bar by double-clicking it

---

## Scientific references

Key algorithms implemented in HYPERLOAD:

| Script | Algorithm | Reference |
|---|---|---|
| DIPLI | Deep Image Prior + SGLD | Singh et al. 2025, arXiv:2503.15984 |
| Thresher Stack | Online blind deconvolution | Hitchcock et al. 2022, MNRAS 511, 5372 |
| ZOGY Subtraction | Optimal image subtraction | Zackay, Ofek & Gal-Yam 2016, ApJ 830, 27 |
| SFFT Subtraction | Fourier-space tiled subtraction | Hu et al. 2022, arXiv:2109.09334 |
| Transit MCMC | Transit light curve fitting | Mandel & Agol 2002, ApJ 580, L171 |
| Broeg Photometry | Optimal comparison star weighting | Broeg et al. 2005, AN 326, 134 |
| Period Finder | Lomb-Scargle periodogram | Scargle 1982, ApJ 263, 835 |
| Spectral Extractor | Optimal 1D extraction | Horne 1986, PASP 98, 609 |
| Cosmic Ray Rejection | L.A.Cosmic | van Dokkum 2001, PASP 113, 1420 |
| Bispectrum Speckle | Phase-preserving reconstruction | Lohmann et al. 1983, Applied Optics |
| Sersic Profiler | Galaxy surface brightness fitting | Ciotti & Bertin 1999, A&A 352, 447 |
| ADI-KLIP | Exoplanet companion detection | Soummer et al. 2012, ApJ 755, L28 |

---

## Contributing

Found a bug or want to suggest an improvement? Open an **Issue** on this repository.

If you want to contribute a script, the conventions are:
- PyQt6 GUI with `QObject` + `QThread` worker pattern (no `input()` calls)
- Entry point: `def main()` at module level
- No hardcoded paths — use `Path(__file__).parent` for all file references
- Siril dark theme stylesheet (see any existing script for the palette)

---

## License

HYPERLOAD is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

You are free to use, modify and distribute this software under the terms of the GPL-3.0.
Any modified version you distribute must also be released under GPL-3.0.

See the [LICENSE](LICENSE) file for the full license text.

> HYPERLOAD uses PyQt6 which is GPL-3.0 licensed. This project is therefore
> GPL-3.0 in full compliance with PyQt6 licensing requirements.
> Siril itself is also GPL-3.0 — HYPERLOAD is consistent with the
> entire ecosystem it extends.

---

*Built for the amateur astronomy community. If HYPERLOAD helps your imaging, consider starring the repository.*
