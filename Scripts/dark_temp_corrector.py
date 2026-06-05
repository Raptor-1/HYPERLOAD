r"""
dark_temp_corrector.py  —  HYPERLOAD Dark Current Temperature Corrector
========================================================================
Scales a master dark frame to a different sensor temperature using the
Arrhenius thermal activation model for semiconductor dark current generation.

The problem: master dark frames must be taken at the same sensor temperature
as the light frames. If you cool to -20°C for lights but your darks were
taken at -15°C, the dark current is different — subtraction leaves residuals.
This is common when:
  - Using uncooled or loosely regulated cameras
  - Darks taken in different seasons (ambient affects set-point accuracy)
  - Reusing an old dark library at a different temperature
  - Matching darks across a long imaging session where temperature drifted

Physics:
    Dark current in silicon follows the Arrhenius thermal activation law:

        D(T) = D0 × exp(−Ea / k × T)

    where Ea is the activation energy (~0.65 eV for silicon depletion current,
    ~1.12 eV for diffusion current at higher temperatures) and k is Boltzmann's
    constant (8.617×10⁻⁵ eV/K).

    The scaling factor between dark temperature Td and light temperature Tl is:

        scale = D(Tl) / D(Td) = exp(−Ea/k × (1/Tl − 1/Td))

    Applied as:
        scaled_dark = bias + scale × (master_dark − bias)

    The bias is subtracted before scaling because the dark current component
    is what changes with temperature — the bias offset does not.
    Amp glow (fixed additive pattern) is NOT temperature-dependent and is
    excluded from scaling via the bias subtraction step.

Three operating modes:
    AUTO:   Reads CCD-TEMP or TEMP-CCD FITS keywords from master dark and
            light frames. Temperature delta computed automatically.
    MANUAL: User specifies dark temperature and target light temperature.
    MULTI:  User provides darks at 3+ temperatures. Fits the Arrhenius
            equation directly to measure the sensor's actual Ea and D0.
            Most accurate — removes dependence on the literature Ea value.

References:
    Arrhenius equation:   Arrhenius 1889, Z. Phys. Chem. 4, 226
    CCD dark current:     Widenhorn et al. 2002, SPIE 4669, 193
    Doubling rule:        Howell 2006, Handbook of CCD Astronomy (Cambridge)
    CMOS compensation:    Xu et al. 2023, Sensors 23(22), 9118
    Antarctic telescope:  Hu et al. 2014, arXiv:1407.8279

Place in: Siril Suites folder
Run via:  Siril → Scripts → dark_temp_corrector
"""

import sirilpy as s
s.ensure_installed("PyQt6")
s.ensure_installed("numpy")
s.ensure_installed("scipy")
s.ensure_installed("astropy")
s.ensure_installed("matplotlib")

import os
import sys
import glob
import threading
import traceback
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from astropy.io import fits as astropy_fits

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QComboBox,
    QRadioButton, QButtonGroup, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

SIRIL_BG       = "#1e2128"
SIRIL_BG2      = "#252930"
SIRIL_BG3      = "#2d3340"
SIRIL_ACCENT   = "#4a9eff"
SIRIL_ACCENT2  = "#2d6abf"
SIRIL_TEXT     = "#dde3ee"
SIRIL_TEXT_DIM = "#7a8499"
SIRIL_BORDER   = "#3a4055"
SIRIL_SUCCESS  = "#4caf7d"
SIRIL_WARNING  = "#e8c46a"
SIRIL_ERROR    = "#cc4444"
SIRIL_NOVA     = "#ff6b35"
SIRIL_SECTION  = "#9db4d0"
SIRIL_COLD     = "#60cfff"   # blue for cold/temperature theme

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {SIRIL_BG};
    color: {SIRIL_TEXT};
    font-family: "Segoe UI", sans-serif; font-size: 9pt;
}}
QGroupBox {{
    border: 1px solid {SIRIL_BORDER}; border-radius: 5px;
    margin-top: 8px; padding: 6px;
    font-weight: bold; color: {SIRIL_SECTION};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QTabWidget::pane {{ border: 1px solid {SIRIL_BORDER}; background: {SIRIL_BG}; }}
QTabBar::tab {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT_DIM};
    padding: 6px 14px; border: 1px solid {SIRIL_BORDER};
    border-bottom: none; border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    background: {SIRIL_BG}; color: {SIRIL_COLD}; font-weight: bold;
}}
QPushButton {{
    background: {SIRIL_BG3}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    padding: 5px 12px; min-height: 22px;
}}
QPushButton:hover {{ background: {SIRIL_ACCENT2}; border-color: {SIRIL_ACCENT}; }}
QPushButton[objectName="primary"] {{
    background: {SIRIL_ACCENT2}; color: white; font-weight: bold;
    border-color: {SIRIL_ACCENT};
}}
QPushButton[objectName="cold"] {{
    background: #0a2a3a; color: {SIRIL_COLD}; font-weight: bold;
    border-color: {SIRIL_COLD};
}}
QPushButton[objectName="danger"] {{
    background: #4a1e1e; color: #e07070; border-color: #803030;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px; padding: 3px 6px;
}}
QProgressBar {{
    background: {SIRIL_BG2}; border: 1px solid {SIRIL_BORDER};
    border-radius: 4px; text-align: center;
}}
QProgressBar::chunk {{ background: {SIRIL_COLD}; border-radius: 3px; }}
QPlainTextEdit {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; border-radius: 4px;
    font-family: "Courier New", monospace; font-size: 8pt;
}}
QTableWidget {{
    background: {SIRIL_BG2}; color: {SIRIL_TEXT};
    border: 1px solid {SIRIL_BORDER}; gridline-color: {SIRIL_BORDER};
    alternate-background-color: {SIRIL_BG3};
}}
QHeaderView::section {{
    background: {SIRIL_BG3}; color: {SIRIL_SECTION};
    border: 1px solid {SIRIL_BORDER}; padding: 4px;
}}
QLabel[objectName="dim"]  {{ color: {SIRIL_TEXT_DIM}; }}
QLabel[objectName="ok"]   {{ color: {SIRIL_SUCCESS}; }}
QLabel[objectName="warn"] {{ color: {SIRIL_WARNING}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QScrollArea {{ border: none; }}
"""

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS — ARRHENIUS MODEL
# ─────────────────────────────────────────────────────────────────────────────

# Boltzmann constant in eV/K
k_B = 8.617333262e-5

# Silicon activation energies (eV)
# Depletion current dominates at cooled amateur CCD/CMOS temperatures
Ea_DEPLETION  = 0.65   # most commonly used for amateur imaging
Ea_DIFFUSION  = 1.12   # diffusion current at higher T (near room temp)
Ea_CMOS_TYPICAL = 0.63  # empirically measured typical CMOS value


def arrhenius_scale(T_dark_C: float, T_light_C: float,
                    Ea_eV: float = Ea_DEPLETION) -> float:
    """
    Compute the dark current scaling factor from dark temperature to
    light temperature using the Arrhenius thermal activation law.

    scale = D(T_light) / D(T_dark) = exp(−Ea/k × (1/T_light − 1/T_dark))

    Parameters:
        T_dark_C:  dark frame sensor temperature in Celsius
        T_light_C: light frame sensor temperature in Celsius
        Ea_eV:     activation energy in electron-volts

    Returns:
        scale factor (>1 if light is warmer than dark, <1 if cooler)
    """
    T_dark_K  = T_dark_C  + 273.15
    T_light_K = T_light_C + 273.15
    return float(np.exp(-Ea_eV / k_B * (1.0/T_light_K - 1.0/T_dark_K)))


def doubling_temperature(Ea_eV: float = Ea_DEPLETION) -> float:
    """
    Compute the temperature interval at which dark current doubles.
    From d/dT [exp(-Ea/kT)] = 2 at reference T:
    ΔT ≈ kT²×ln(2)/Ea  at T=253K (−20°C typical imaging temperature)
    """
    T_ref = 253.15  # -20°C
    return k_B * T_ref**2 * np.log(2) / Ea_eV


def fit_arrhenius(temperatures_C: list, dark_medians: list) -> dict:
    """
    Fit Ea and D0 from measured dark medians at multiple temperatures.
    Uses the linearised Arrhenius: ln(D) = ln(D0) − Ea/(k×T)
    Equivalent to a straight-line fit on the Arrhenius plot.

    Returns dict with Ea_eV, D0, r2, doubling_T, residuals.
    """
    T_K   = np.array([t + 273.15 for t in temperatures_C])
    D     = np.array(dark_medians, dtype=float)
    inv_T = 1.0 / T_K
    ln_D  = np.log(np.maximum(D, 1e-10))

    # Linear fit: ln(D) = a×(1/T) + b → Ea = −a×k, D0 = exp(b)
    coeffs  = np.polyfit(inv_T, ln_D, 1)
    a, b    = coeffs
    Ea_fit  = float(-a * k_B)
    D0_fit  = float(np.exp(b))

    # R²
    ln_D_pred = a * inv_T + b
    ss_res    = np.sum((ln_D - ln_D_pred)**2)
    ss_tot    = np.sum((ln_D - np.mean(ln_D))**2)
    r2        = float(1 - ss_res / max(ss_tot, 1e-20))

    # Doubling temperature from fitted Ea
    dbl_T = doubling_temperature(Ea_fit)

    return {
        "Ea_eV":      Ea_fit,
        "D0":         D0_fit,
        "r2":         r2,
        "doubling_T": dbl_T,
        "a":          a,
        "b":          b,
        "inv_T":      inv_T,
        "ln_D":       ln_D,
    }


def read_fits_temperature(path: str) -> float | None:
    """
    Read sensor temperature from FITS header.
    Tries common keyword names in priority order.
    Returns temperature in Celsius, or None if not found.
    """
    keywords = [
        "CCD-TEMP", "CCDTEMP", "TEMP-CCD", "CCD_TEMP",
        "SENSORCCD", "SET-TEMP", "SETTEMP", "FOCAL-T",
        "TEMP",
    ]
    try:
        with astropy_fits.open(path) as hdul:
            hdr = hdul[0].header
            for kw in keywords:
                if kw in hdr:
                    val = hdr[kw]
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
    except Exception:
        pass
    return None


def scale_dark(master_dark: np.ndarray,
               master_bias: np.ndarray | None,
               scale_factor: float) -> np.ndarray:
    """
    Apply temperature scaling to a master dark frame.

    scaled_dark = bias + scale × (dark − bias)

    If no bias frame: assume bias is the minimum signal in the dark.
    The bias subtraction is critical — the DC offset does not scale
    with temperature, only the dark current component does.
    """
    dark = master_dark.astype(np.float64)

    if master_bias is not None:
        bias = master_bias.astype(np.float64)
    else:
        # Estimate bias as robust minimum (5th percentile)
        bias = np.percentile(dark, 5.0)

    dark_current = dark - bias
    dark_current = np.maximum(dark_current, 0)  # can't have negative dark current
    scaled       = bias + scale_factor * dark_current
    return scaled.astype(master_dark.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# ARRHENIUS CANVAS
# ─────────────────────────────────────────────────────────────────────────────

class ArrheniusCanvas(FigureCanvasQTAgg):
    """
    3-panel canvas:
    1. Arrhenius plot (1/T vs ln(D)) — linear in Arrhenius coordinates
    2. Scale factor vs temperature delta
    3. Dark frame before/after comparison (histogram)
    """
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(14, 4), facecolor=SIRIL_BG)
        self.ax_arr   = self.fig.add_subplot(1, 3, 1)
        self.ax_scale = self.fig.add_subplot(1, 3, 2)
        self.ax_hist  = self.fig.add_subplot(1, 3, 3)
        for ax in [self.ax_arr, self.ax_scale, self.ax_hist]:
            ax.set_facecolor(SIRIL_BG2)
            ax.tick_params(colors=SIRIL_TEXT_DIM, labelsize=8)
            for sp in ax.spines.values(): sp.set_color(SIRIL_BORDER)
        super().__init__(self.fig)
        self.setParent(parent)
        self._draw_scale_curve()

    def _draw_scale_curve(self, Ea: float = Ea_DEPLETION,
                           T_dark_C: float = -15.0):
        """Draw scale factor vs temperature delta."""
        self.ax_scale.clear(); self.ax_scale.set_facecolor(SIRIL_BG2)
        dT = np.linspace(-20, 20, 200)
        scales = [arrhenius_scale(T_dark_C, T_dark_C + d, Ea) for d in dT]
        self.ax_scale.plot(dT, scales, color=SIRIL_COLD, lw=1.5)
        self.ax_scale.axhline(1.0, color=SIRIL_BORDER, lw=0.8, ls="--")
        self.ax_scale.axvline(0.0, color=SIRIL_BORDER, lw=0.8, ls="--")
        # Mark doubling
        dbl = doubling_temperature(Ea)
        self.ax_scale.axvline(dbl, color=SIRIL_NOVA, lw=0.8, ls=":",
                               label=f"Doubling ({dbl:.1f}°C)")
        self.ax_scale.axvline(-dbl, color=SIRIL_NOVA, lw=0.8, ls=":")
        self.ax_scale.set_xlabel("ΔT (T_light − T_dark, °C)",
                                  color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_scale.set_ylabel("Scale factor", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_scale.set_title(
            f"Scale factor vs ΔT  (Ea={Ea:.3f} eV, T_dark={T_dark_C}°C)",
            color=SIRIL_SECTION, fontsize=9)
        self.ax_scale.legend(fontsize=7, facecolor=SIRIL_BG3,
                              edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax_scale.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax_scale.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def show_arrhenius_fit(self, fit: dict, temps_C: list):
        """Show Arrhenius plot with fitted line for multi-temp mode."""
        self.ax_arr.clear(); self.ax_arr.set_facecolor(SIRIL_BG2)
        inv_T = fit["inv_T"]; ln_D = fit["ln_D"]
        a = fit["a"]; b = fit["b"]
        inv_T_fit = np.linspace(inv_T.min(), inv_T.max(), 100)
        ln_D_fit  = a * inv_T_fit + b

        # Convert inverse T to temperature in Celsius for labels
        temp_labels = [f"{tc:.0f}°C" for tc in temps_C]
        self.ax_arr.scatter(inv_T * 1000, ln_D, color=SIRIL_COLD, s=60,
                             zorder=5, label="Measured")
        for xi, yi, lbl in zip(inv_T * 1000, ln_D, temp_labels):
            self.ax_arr.annotate(lbl, (xi, yi), textcoords="offset points",
                                  xytext=(4, 4), fontsize=6,
                                  color=SIRIL_TEXT_DIM)
        self.ax_arr.plot(inv_T_fit * 1000, ln_D_fit,
                          color=SIRIL_NOVA, lw=1.5, ls="--",
                          label=f"Fit  Ea={fit['Ea_eV']:.3f} eV  R²={fit['r2']:.4f}")
        self.ax_arr.set_xlabel("1000/T (K⁻¹)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_arr.set_ylabel("ln(dark current)", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_arr.set_title(
            f"Arrhenius plot  Ea={fit['Ea_eV']:.4f} eV  "
            f"doubling={fit['doubling_T']:.1f}°C",
            color=SIRIL_COLD, fontsize=9)
        self.ax_arr.legend(fontsize=7, facecolor=SIRIL_BG3,
                            edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax_arr.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax_arr.spines.values(): sp.set_color(SIRIL_BORDER)
        # Update scale curve with fitted Ea
        self._draw_scale_curve(fit["Ea_eV"], temps_C[0])
        self.draw()

    def show_histogram(self, original: np.ndarray, scaled: np.ndarray,
                        scale_factor: float, T_dark: float, T_light: float):
        """Show before/after dark pixel distribution."""
        self.ax_hist.clear(); self.ax_hist.set_facecolor(SIRIL_BG2)
        flat_orig   = original.ravel()
        flat_scaled = scaled.ravel()
        lo = np.percentile(flat_orig, 1); hi = np.percentile(flat_orig, 99)
        bins = np.linspace(lo, hi, 80)
        self.ax_hist.hist(flat_orig, bins=bins, color=SIRIL_SECTION,
                           alpha=0.7, label=f"Original ({T_dark:.1f}°C)")
        self.ax_hist.hist(flat_scaled, bins=bins, color=SIRIL_COLD,
                           alpha=0.7, label=f"Scaled ({T_light:.1f}°C)")
        self.ax_hist.set_xlabel("Pixel value (ADU)", color=SIRIL_TEXT_DIM,
                                  fontsize=8)
        self.ax_hist.set_ylabel("Count", color=SIRIL_TEXT_DIM, fontsize=8)
        self.ax_hist.set_title(
            f"Dark histogram  scale={scale_factor:.4f}×",
            color=SIRIL_SUCCESS, fontsize=9)
        self.ax_hist.legend(fontsize=7, facecolor=SIRIL_BG3,
                             edgecolor=SIRIL_BORDER, labelcolor=SIRIL_TEXT)
        self.ax_hist.tick_params(colors=SIRIL_TEXT_DIM, labelsize=7)
        for sp in self.ax_hist.spines.values(): sp.set_color(SIRIL_BORDER)
        self.fig.tight_layout(pad=0.4)
        self.draw()

    def update_scale_preview(self, Ea: float, T_dark: float):
        self._draw_scale_curve(Ea, T_dark)


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION WORKER
# ─────────────────────────────────────────────────────────────────────────────

class DarkCorrectionWorker(QThread):
    progress       = pyqtSignal(int, int, str)
    log_line       = pyqtSignal(str)
    arrhenius_fit  = pyqtSignal(dict, list)      # fit result, temps list
    result_ready   = pyqtSignal(np.ndarray, np.ndarray, float, float, float)
    finished       = pyqtSignal(dict)

    def __init__(self, config: dict, cancel_event: threading.Event,
                 parent=None):
        super().__init__(parent)
        self.cfg     = config
        self._cancel = cancel_event

    def run(self):
        try:
            self._run()
        except Exception as e:
            tb = traceback.format_exc()
            self.log_line.emit(f"CRITICAL ERROR: {e}\n{tb}")
            self.finished.emit({"success": False, "error": str(e)})

    def _run(self):
        cfg = self.cfg
        log = self.log_line.emit
        log("═══ HyperLoad Dark Current Temperature Corrector ═══")
        log(f"  Mode:          {cfg['mode']}")
        log(f"  Activation Ea: {cfg['Ea_eV']:.4f} eV")
        log(f"  Doubling ΔT:   {doubling_temperature(cfg['Ea_eV']):.2f} °C")
        log("")

        mode = cfg["mode"]

        # ── MULTI-TEMPERATURE MODE: fit Ea from data ──────────────────────────
        if mode == "multi_temp":
            self._run_multi_temp()
            return

        # ── AUTO or MANUAL: single master dark → scale to target temperature ──
        self.progress.emit(5, 100, "Loading master dark…")
        dark_path = cfg["dark_path"]
        bias_path = cfg.get("bias_path", "")

        try:
            with astropy_fits.open(dark_path) as hdul:
                dark_data  = hdul[0].data.astype(np.float64)
                dark_hdr   = hdul[0].header.copy()
        except Exception as e:
            self.finished.emit({"success": False,
                                "error": f"Cannot load dark: {e}"}); return

        if dark_data.ndim == 3:
            dark_data = dark_data[0]

        bias_data = None
        if bias_path and os.path.isfile(bias_path):
            try:
                with astropy_fits.open(bias_path) as hdul:
                    bias_data = hdul[0].data.astype(np.float64)
                if bias_data.ndim == 3:
                    bias_data = bias_data[0]
                log(f"Bias loaded: {os.path.basename(bias_path)}")
            except Exception as e:
                log(f"Bias load failed ({e}) — estimating from dark")

        # ── Get temperatures ──────────────────────────────────────────────────
        self.progress.emit(15, 100, "Reading temperatures…")
        if mode == "auto":
            T_dark = read_fits_temperature(dark_path)
            if T_dark is None:
                self.finished.emit({"success": False,
                                    "error": "CCD-TEMP not found in dark FITS header. "
                                             "Use Manual mode to enter temperatures."})
                return
            log(f"Dark temperature from FITS: {T_dark:.2f} °C")

            # Try to read from light frames
            light_temps = []
            for lp in cfg.get("light_paths", []):
                t = read_fits_temperature(lp)
                if t is not None:
                    light_temps.append(t)
            if light_temps:
                T_light = np.median(light_temps)
                log(f"Light temperature from {len(light_temps)} frames: "
                    f"{T_light:.2f} °C (median)")
            else:
                self.finished.emit({"success": False,
                                    "error": "CCD-TEMP not found in light FITS headers. "
                                             "Use Manual mode."})
                return
        else:  # manual
            T_dark  = cfg["T_dark_C"]
            T_light = cfg["T_light_C"]
            log(f"Dark temperature (manual):  {T_dark:.2f} °C")
            log(f"Light temperature (manual): {T_light:.2f} °C")

        if abs(T_dark - T_light) < 0.01:
            log("⚠  Temperatures are identical — no correction needed")
            self.finished.emit({"success": True, "n_scaled": 0,
                                "scale_factor": 1.0,
                                "T_dark": T_dark, "T_light": T_light})
            return

        # ── Compute scale factor ──────────────────────────────────────────────
        self.progress.emit(25, 100, "Computing scale factor…")
        Ea      = cfg["Ea_eV"]
        scale   = arrhenius_scale(T_dark, T_light, Ea)
        dbl_T   = doubling_temperature(Ea)

        log(f"Scale factor: {scale:.6f}×")
        log(f"  ({'+' if T_light > T_dark else ''}{T_light-T_dark:.2f}°C  "
            f"Ea={Ea:.4f} eV  doubling={dbl_T:.2f}°C)")

        if scale > 3.0 or scale < 0.1:
            log(f"⚠  Scale factor {scale:.3f} is extreme — "
                f"verify temperatures (ΔT={T_light-T_dark:+.1f}°C)")

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── Apply correction ──────────────────────────────────────────────────
        self.progress.emit(40, 100, "Applying Arrhenius correction…")
        scaled_dark = scale_dark(dark_data, bias_data, scale)
        log(f"  Original dark median: {np.median(dark_data):.2f} ADU")
        log(f"  Scaled dark median:   {np.median(scaled_dark):.2f} ADU")
        log(f"  Bias estimate:        "
            f"{np.median(bias_data):.2f} ADU" if bias_data is not None else
            f"{np.percentile(dark_data, 5):.2f} ADU (auto)")

        self.result_ready.emit(dark_data, scaled_dark, scale, T_dark, T_light)

        if self._cancel.is_set():
            self.finished.emit({"success": False, "error": "Cancelled"}); return

        # ── Save scaled dark ──────────────────────────────────────────────────
        self.progress.emit(70, 100, "Saving scaled dark…")
        out_dir  = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        base     = os.path.splitext(os.path.basename(dark_path))[0]
        suffix   = f"_T{T_light:+.0f}C"
        out_path = os.path.join(out_dir, f"{base}{suffix}_scaled.fit")

        with astropy_fits.open(dark_path) as hdul:
            new_hdr = hdul[0].header.copy()
        new_hdr["CCD-TEMP"]  = (T_light, "Target temperature [C]")
        new_hdr["DARKTMPO"]  = (T_dark,  "Original dark temperature [C]")
        new_hdr["DARKSCAL"]  = (scale,   "Arrhenius scale factor applied")
        new_hdr["DARKEA"]    = (Ea,      "Activation energy used [eV]")
        new_hdr["HISTORY"]   = (f"HyperLoad dark T correction: "
                                 f"{T_dark:.2f}C → {T_light:.2f}C  "
                                 f"scale={scale:.6f}  Ea={Ea:.4f}eV")

        hdu = astropy_fits.PrimaryHDU(data=scaled_dark.astype(np.float32),
                                       header=new_hdr)
        hdu.writeto(out_path, overwrite=True)
        log(f"✓ Saved: {out_path}")

        # ── Optionally apply to light frames ──────────────────────────────────
        if cfg.get("apply_to_lights") and cfg.get("light_paths"):
            n_lights = len(cfg["light_paths"])
            self.progress.emit(75, 100, "Applying to light frames…")
            log(f"\nApplying scaled dark to {n_lights} light frames…")
            n_done = 0
            for i, lp in enumerate(cfg["light_paths"]):
                if self._cancel.is_set(): break
                self.progress.emit(75 + int(20*i/n_lights), 100,
                                    f"Calibrating light {i+1}/{n_lights}")
                try:
                    with astropy_fits.open(lp) as hdul:
                        light = hdul[0].data.astype(np.float32)
                        lhdr  = hdul[0].header.copy()
                    if light.ndim == 3:
                        for ch in range(light.shape[0]):
                            light[ch] -= scaled_dark.astype(np.float32)
                    else:
                        light -= scaled_dark.astype(np.float32)
                    lhdr["HISTORY"] = (f"Dark subtracted: {os.path.basename(out_path)}")
                    lb  = os.path.splitext(os.path.basename(lp))[0]
                    lout = os.path.join(out_dir, f"{lb}_darkcal.fit")
                    astropy_fits.PrimaryHDU(data=light, header=lhdr).writeto(
                        lout, overwrite=True)
                    n_done += 1
                except Exception as e:
                    log(f"  Skip {os.path.basename(lp)}: {e}")
            log(f"  Calibrated {n_done}/{n_lights} light frames → {out_dir}")

        self.progress.emit(100, 100, "Done")
        log(f"\n✓ Temperature correction complete")
        log(f"  ΔT = {T_light-T_dark:+.2f}°C  scale = {scale:.6f}×")
        self.finished.emit({
            "success":      True,
            "scale_factor": scale,
            "T_dark":       T_dark,
            "T_light":      T_light,
            "Ea_eV":        Ea,
            "output_path":  out_path,
        })

    def _run_multi_temp(self):
        """Fit Arrhenius equation from darks at multiple temperatures."""
        cfg = self.cfg
        log = self.log_line.emit
        log("Multi-temperature Arrhenius fit mode")
        entries = cfg["multi_entries"]  # list of {"path": str, "temp_C": float}

        if len(entries) < 3:
            self.finished.emit({"success": False,
                                "error": "Need at least 3 temperature points"})
            return

        temps_C   = []
        medians   = []
        bias_est  = None

        for i, entry in enumerate(entries):
            self.progress.emit(i, len(entries),
                               f"Reading {os.path.basename(entry['path'])}…")
            try:
                with astropy_fits.open(entry["path"]) as hdul:
                    data = hdul[0].data.astype(np.float64)
                    hdr  = hdul[0].header
                if data.ndim == 3: data = data[0]
                T = entry.get("temp_C") or read_fits_temperature(entry["path"])
                if T is None:
                    log(f"  Skip {os.path.basename(entry['path'])}: no temperature")
                    continue
                # Estimate bias from coldest frame as proxy
                if bias_est is None:
                    bias_est = np.percentile(data, 5)
                dark_current_med = np.median(data) - bias_est
                if dark_current_med <= 0:
                    dark_current_med = max(np.median(data), 0.01)
                temps_C.append(float(T))
                medians.append(float(dark_current_med))
                log(f"  {T:.1f}°C → median dark current: {dark_current_med:.2f} ADU")
            except Exception as e:
                log(f"  Skip {os.path.basename(entry['path'])}: {e}")

        if len(temps_C) < 3:
            self.finished.emit({"success": False,
                                "error": "Fewer than 3 valid data points"})
            return

        # Fit Arrhenius
        self.progress.emit(80, 100, "Fitting Arrhenius equation…")
        fit = fit_arrhenius(temps_C, medians)
        log(f"\nArrhenius fit result:")
        log(f"  Activation energy Ea = {fit['Ea_eV']:.4f} eV")
        log(f"  Pre-exponential D0   = {fit['D0']:.4f}")
        log(f"  R²                   = {fit['r2']:.6f}")
        log(f"  Doubling ΔT          = {fit['doubling_T']:.2f} °C")
        log(f"  (Literature: Ea ≈ 0.65 eV for depletion, 1.12 eV for diffusion)")

        self.arrhenius_fit.emit(fit, temps_C)

        # Save fitted parameters to text file
        out_dir = cfg.get("output_dir", os.path.dirname(entries[0]["path"]))
        os.makedirs(out_dir, exist_ok=True)
        fit_path = os.path.join(out_dir, "arrhenius_fit.txt")
        with open(fit_path, "w") as f:
            f.write("HyperLoad Arrhenius Fit Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Activation energy Ea: {fit['Ea_eV']:.6f} eV\n")
            f.write(f"Pre-exponential D0:   {fit['D0']:.6f} ADU/s\n")
            f.write(f"R²:                   {fit['r2']:.6f}\n")
            f.write(f"Doubling temperature: {fit['doubling_T']:.2f} °C\n\n")
            f.write("Data points:\n")
            for T, D in zip(temps_C, medians):
                f.write(f"  {T:+6.1f}°C  dark={D:.4f} ADU\n")
        log(f"\n✓ Fit results saved: {fit_path}")
        log(f"  Use Ea={fit['Ea_eV']:.4f} eV in Manual or Auto mode "
            f"for best accuracy with your specific sensor")

        self.progress.emit(100, 100, "Done")
        self.finished.emit({
            "success":    True,
            "fit":        fit,
            "temps_C":    temps_C,
            "fit_path":   fit_path,
        })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Dark Current Temperature Corrector  —  HYPERLOAD  —  Siril")
        self.resize(1300, 850)
        self._worker       = None
        self._cancel_event = threading.Event()
        self._multi_entries = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # Header
        header = QWidget(); header.setFixedHeight(56)
        header.setStyleSheet(
            f"background:{SIRIL_BG2}; border-bottom:1px solid {SIRIL_BORDER};")
        hl = QHBoxLayout(header); hl.setContentsMargins(14, 0, 14, 0)
        lbl_icon = QLabel("🌡")
        lbl_icon.setStyleSheet("font-size:20pt; background:transparent;")
        lbl_title = QLabel("Dark Current Temperature Corrector")
        lbl_title.setStyleSheet(
            f"color:{SIRIL_COLD}; font-size:13pt; font-weight:bold; "
            f"background:transparent;")
        lbl_sub = QLabel(
            "Arrhenius thermal model  ·  Auto/Manual/Multi-temp modes  ·  "
            "Ea fit from data  ·  Bias-aware scaling  ·  Doubling rule")
        lbl_sub.setStyleSheet(
            f"color:{SIRIL_TEXT_DIM}; font-size:9pt; background:transparent;")
        hl.addWidget(lbl_icon); hl.addWidget(lbl_title)
        hl.addWidget(lbl_sub); hl.addStretch()
        root.addWidget(header)

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._tabs.addTab(self._build_tab_mode(),    "⚙  Mode & Temps")
        self._tabs.addTab(self._build_tab_files(),   "📁  Files")
        self._tabs.addTab(self._build_tab_physics(), "🔬  Physics")
        self._tabs.addTab(self._build_tab_multi(),   "📊  Multi-Temp Fit")
        self._tabs.addTab(self._build_tab_preview(), "📈  Preview")
        self._tabs.addTab(self._build_tab_log(),     "📋  Log")

        # Bottom bar
        bottom = QWidget(); bottom.setFixedHeight(50)
        bottom.setStyleSheet(
            f"background:{SIRIL_BG2}; border-top:1px solid {SIRIL_BORDER};")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(10, 6, 10, 6)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(10); self._progress.setValue(0)
        bl.addWidget(self._progress, 1)
        self._btn_run = QPushButton("▶  Apply Temperature Correction")
        self._btn_run.setObjectName("cold")
        self._btn_run.setMinimumWidth(220); self._btn_run.setMinimumHeight(34)
        self._btn_run.clicked.connect(self._run)
        bl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton("✕  Cancel")
        self._btn_cancel.setObjectName("danger")
        self._btn_cancel.setMinimumHeight(34); self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        bl.addWidget(self._btn_cancel)
        self._status = QLabel("Ready — select a mode and load your master dark")
        self._status.setObjectName("dim")
        self._status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self._status, 1)
        root.addWidget(bottom)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_tab_mode(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        grp_mode = QGroupBox("Operating mode")
        ml = QVBoxLayout(grp_mode)
        self._rb_auto   = QRadioButton(
            "Auto  — read CCD-TEMP from FITS headers of dark and light frames\n"
            "         Requires temperature in FITS header (modern cooled cameras write this)")
        self._rb_manual = QRadioButton(
            "Manual  — enter dark and light temperatures directly\n"
            "           Use when FITS headers don't contain temperature")
        self._rb_multi  = QRadioButton(
            "Multi-Temperature Fit  — provide darks at 3+ temperatures\n"
            "                         Fits Ea from your actual sensor data (most accurate)\n"
            "                         Use the fitted Ea in Auto/Manual mode afterwards")
        self._rb_manual.setChecked(True)
        self._bg_mode = QButtonGroup()
        for rb in [self._rb_auto, self._rb_manual, self._rb_multi]:
            self._bg_mode.addButton(rb); ml.addWidget(rb)
            rb.toggled.connect(self._on_mode_changed)
        lay.addWidget(grp_mode)

        # Manual temps
        self._grp_manual = QGroupBox("Manual temperatures")
        mf = QFormLayout(self._grp_manual)
        self._spin_T_dark = QDoubleSpinBox()
        self._spin_T_dark.setRange(-60, 50); self._spin_T_dark.setValue(-15.0)
        self._spin_T_dark.setSuffix(" °C"); self._spin_T_dark.setSingleStep(0.5)
        self._spin_T_dark.valueChanged.connect(self._update_scale_preview)
        mf.addRow("Dark frame temperature:", self._spin_T_dark)
        self._spin_T_light = QDoubleSpinBox()
        self._spin_T_light.setRange(-60, 50); self._spin_T_light.setValue(-20.0)
        self._spin_T_light.setSuffix(" °C"); self._spin_T_light.setSingleStep(0.5)
        self._spin_T_light.valueChanged.connect(self._update_scale_preview)
        mf.addRow("Light frame temperature:", self._spin_T_light)
        lay.addWidget(self._grp_manual)

        # Live scale display
        grp_live = QGroupBox("Live scale factor")
        lf = QVBoxLayout(grp_live)
        self._lbl_scale = QLabel("Scale factor: —")
        self._lbl_scale.setStyleSheet(
            f"color:{SIRIL_COLD}; font-size:14pt; font-weight:bold; padding:4px;")
        self._lbl_dbl = QLabel("")
        self._lbl_dbl.setObjectName("dim")
        self._lbl_warn = QLabel("")
        self._lbl_warn.setObjectName("warn")
        for l in [self._lbl_scale, self._lbl_dbl, self._lbl_warn]:
            lf.addWidget(l)
        lay.addWidget(grp_live)
        self._update_scale_preview()
        lay.addStretch()
        return w

    def _build_tab_files(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        grp_dark = QGroupBox("Master dark frame")
        df = QHBoxLayout(grp_dark)
        self._edit_dark = QLineEdit()
        self._edit_dark.setPlaceholderText("Master dark FITS…")
        btn_d = QPushButton("Browse…"); btn_d.setFixedWidth(80)
        btn_d.clicked.connect(lambda: self._browse_fits(self._edit_dark,
                                                         "Master dark"))
        df.addWidget(self._edit_dark); df.addWidget(btn_d)
        lay.addWidget(grp_dark)

        grp_bias = QGroupBox("Master bias frame  (strongly recommended)")
        bf = QVBoxLayout(grp_bias)
        bias_row = QHBoxLayout()
        self._edit_bias = QLineEdit()
        self._edit_bias.setPlaceholderText(
            "Master bias FITS (optional — estimated from dark if absent)…")
        btn_b = QPushButton("Browse…"); btn_b.setFixedWidth(80)
        btn_b.clicked.connect(lambda: self._browse_fits(self._edit_bias,
                                                         "Master bias"))
        bias_row.addWidget(self._edit_bias); bias_row.addWidget(btn_b)
        bf.addLayout(bias_row)
        lbl_bias = QLabel(
            "⚠  Without a bias frame, the bias level is estimated as the 5th "
            "percentile of the dark. This is approximate — providing the actual "
            "master bias gives a more accurate correction, especially at low "
            "dark temperatures where the bias dominates over dark current.")
        lbl_bias.setObjectName("warn"); lbl_bias.setWordWrap(True)
        bf.addWidget(lbl_bias)
        lay.addWidget(grp_bias)

        grp_lights = QGroupBox("Light frames  (optional — auto-mode reads temperatures from these)")
        ll = QVBoxLayout(grp_lights)
        lights_row = QHBoxLayout()
        self._edit_lights_folder = QLineEdit()
        self._edit_lights_folder.setPlaceholderText(
            "Folder of light FITS (for auto temp reading and/or calibration)…")
        btn_l = QPushButton("Browse…"); btn_l.setFixedWidth(80)
        btn_l.clicked.connect(self._browse_lights_folder)
        lights_row.addWidget(self._edit_lights_folder)
        lights_row.addWidget(btn_l)
        ll.addLayout(lights_row)
        self._lbl_lights_count = QLabel("No folder selected")
        self._lbl_lights_count.setObjectName("dim"); ll.addWidget(self._lbl_lights_count)
        self._chk_apply_lights = QCheckBox(
            "Apply calibrated dark to all light frames in this folder")
        ll.addWidget(self._chk_apply_lights)
        lay.addWidget(grp_lights)

        grp_out = QGroupBox("Output folder")
        of = QHBoxLayout(grp_out)
        self._edit_out = QLineEdit(); self._edit_out.setPlaceholderText("Output folder…")
        btn_out = QPushButton("Browse…"); btn_out.setFixedWidth(80)
        btn_out.clicked.connect(lambda: self._edit_out.setText(
            QFileDialog.getExistingDirectory(self, "Output folder") or
            self._edit_out.text()))
        of.addWidget(self._edit_out); of.addWidget(btn_out)
        lay.addWidget(grp_out)
        lay.addStretch()
        return w

    def _build_tab_physics(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)
        scroll.setWidget(inner); ql = QVBoxLayout(w); ql.addWidget(scroll)

        grp_ea = QGroupBox("Activation energy (Ea)")
        ef = QFormLayout(grp_ea)
        self._cmb_ea_preset = QComboBox()
        self._cmb_ea_preset.addItems([
            f"Silicon depletion current  (Ea=0.65 eV — most amateur CCDs/CMOS)",
            f"Silicon diffusion current  (Ea=1.12 eV — above 0°C, less common)",
            f"Typical CMOS (ASI, QHY)   (Ea=0.63 eV — measured mean)",
            "Custom",
        ])
        self._cmb_ea_preset.currentIndexChanged.connect(self._on_ea_preset)
        ef.addRow("Preset:", self._cmb_ea_preset)
        self._spin_ea = QDoubleSpinBox()
        self._spin_ea.setRange(0.1, 2.0); self._spin_ea.setValue(0.65)
        self._spin_ea.setDecimals(4); self._spin_ea.setSingleStep(0.01)
        self._spin_ea.setSuffix(" eV")
        self._spin_ea.valueChanged.connect(self._update_scale_preview)
        ef.addRow("Ea (electron-volts):", self._spin_ea)
        lbl_ea = QLabel(
            "Ea = activation energy of the dominant dark current generation mechanism.\n\n"
            "0.65 eV (depletion-region current): correct for cooled silicon sensors\n"
            "  at typical imaging temperatures (−40°C to −5°C). Most common.\n\n"
            "1.12 eV (diffusion current): dominates near room temperature. Rarely\n"
            "  relevant for amateur imaging.\n\n"
            "For best accuracy, use the Multi-Temperature Fit mode to measure Ea\n"
            "directly from your sensor's behaviour.")
        lbl_ea.setObjectName("dim"); lbl_ea.setWordWrap(True); ef.addRow(lbl_ea)
        lay.addWidget(grp_ea)

        grp_ref = QGroupBox("Physics background")
        rl = QVBoxLayout(grp_ref)
        for line in [
            "Arrhenius equation:  D(T) = D0 × exp(−Ea / k×T)",
            "Scale factor:        scale = exp(−Ea/k × (1/T_light − 1/T_dark))",
            "Correction applied:  dark_scaled = bias + scale × (dark − bias)",
            "",
            "k (Boltzmann) = 8.617×10⁻⁵ eV/K",
            "T must be in Kelvin (= Celsius + 273.15)",
            "",
            "Doubling rule: dark current doubles every ~5-10°C at typical",
            "imaging temperatures (depends on Ea and operating temperature).",
            "",
            "References:",
            "  Arrhenius 1889, Z. Phys. Chem. 4, 226",
            "  Widenhorn et al. 2002, SPIE 4669, 193",
            "  Howell 2006, Handbook of CCD Astronomy (Cambridge)",
            "  Xu et al. 2023, Sensors 23(22), 9118",
        ]:
            lbl = QLabel(line)
            lbl.setObjectName("dim" if line else "dim")
            rl.addWidget(lbl)
        lay.addWidget(grp_ref)
        lay.addStretch()
        return w

    def _build_tab_multi(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        grp_entry = QGroupBox("Add dark frame at known temperature")
        el = QVBoxLayout(grp_entry)
        row = QHBoxLayout()
        self._edit_multi_path = QLineEdit()
        self._edit_multi_path.setPlaceholderText("Dark FITS path…")
        btn_mp = QPushButton("Browse…"); btn_mp.setFixedWidth(80)
        btn_mp.clicked.connect(self._browse_multi)
        row.addWidget(self._edit_multi_path); row.addWidget(btn_mp)
        el.addLayout(row)
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temperature:"))
        self._spin_multi_temp = QDoubleSpinBox()
        self._spin_multi_temp.setRange(-60, 50); self._spin_multi_temp.setValue(-10.0)
        self._spin_multi_temp.setSuffix(" °C")
        self._chk_auto_temp = QCheckBox("Read from FITS header")
        self._chk_auto_temp.setChecked(True)
        temp_row.addWidget(self._spin_multi_temp)
        temp_row.addWidget(self._chk_auto_temp); temp_row.addStretch()
        el.addLayout(temp_row)
        btn_add = QPushButton("➕  Add to list")
        btn_add.clicked.connect(self._add_multi_entry)
        el.addWidget(btn_add)
        lay.addWidget(grp_entry)

        grp_list = QGroupBox("Dark frames list  (need ≥3 for a valid fit)")
        gl = QVBoxLayout(grp_list)
        self._multi_table = QTableWidget(0, 3)
        self._multi_table.setHorizontalHeaderLabels(["#", "File", "Temperature"])
        self._multi_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._multi_table.setAlternatingRowColors(True)
        self._multi_table.setMaximumHeight(160)
        gl.addWidget(self._multi_table)
        btn_clear = QPushButton("🗑  Clear list"); btn_clear.setFixedWidth(100)
        btn_clear.clicked.connect(self._clear_multi)
        gl.addWidget(btn_clear)
        lay.addWidget(grp_list)

        grp_out_multi = QGroupBox("Output folder")
        of = QHBoxLayout(grp_out_multi)
        self._edit_out_multi = QLineEdit()
        self._edit_out_multi.setPlaceholderText("Output folder for fit results…")
        btn_om = QPushButton("Browse…"); btn_om.setFixedWidth(80)
        btn_om.clicked.connect(lambda: self._edit_out_multi.setText(
            QFileDialog.getExistingDirectory(self, "Output folder") or
            self._edit_out_multi.text()))
        of.addWidget(self._edit_out_multi); of.addWidget(btn_om)
        lay.addWidget(grp_out_multi)
        lay.addStretch()
        return w

    def _build_tab_preview(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        self._canvas = ArrheniusCanvas()
        lay.addWidget(self._canvas)
        lbl = QLabel(
            "Left: Arrhenius plot (populated after multi-temp fit)  ·  "
            "Centre: scale factor vs ΔT (updates live)  ·  "
            "Right: dark histogram before/after (populated after correction)")
        lbl.setObjectName("dim"); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        return w

    def _build_tab_log(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        btn_clr = QPushButton("Clear"); btn_clr.setFixedWidth(90)
        btn_clr.clicked.connect(self._log.clear)
        lay.addWidget(btn_clr); lay.addWidget(self._log)
        return w

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        self._grp_manual.setVisible(self._rb_manual.isChecked())

    def _on_ea_preset(self, idx: int):
        presets = [0.65, 1.12, 0.63]
        if idx < len(presets):
            self._spin_ea.setValue(presets[idx])

    def _update_scale_preview(self):
        Ea     = self._spin_ea.value()
        T_dark = self._spin_T_dark.value()
        T_light= self._spin_T_light.value()
        scale  = arrhenius_scale(T_dark, T_light, Ea)
        dbl_T  = doubling_temperature(Ea)
        self._lbl_scale.setText(
            f"Scale factor: {scale:.6f}×  "
            f"(ΔT = {T_light-T_dark:+.1f}°C)")
        self._lbl_dbl.setText(
            f"Doubling temperature: {dbl_T:.2f}°C  "
            f"(Ea={Ea:.4f} eV  T_dark={T_dark:.1f}°C)")
        if scale > 2.0:
            self._lbl_warn.setText(
                "⚠ Large correction — verify temperatures are correct")
        elif scale < 0.3:
            self._lbl_warn.setText(
                "⚠ Light is much colder than dark — dark will be scaled down significantly")
        else:
            self._lbl_warn.setText("")
        self._canvas.update_scale_preview(Ea, T_dark)

    def _browse_fits(self, edit: QLineEdit, title: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {title}", "", "FITS (*.fit *.fits *.fts);;All (*)")
        if path:
            edit.setText(path)
            if not self._edit_out.text():
                self._edit_out.setText(os.path.dirname(path))
            # Auto-read temperature
            T = read_fits_temperature(path)
            if T is not None and title == "Master dark":
                self._spin_T_dark.setValue(T)
                self._log.appendPlainText(
                    f"[Auto] Dark temperature from FITS: {T:.2f} °C")

    def _browse_lights_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select lights folder")
        if not d: return
        self._edit_lights_folder.setText(d)
        files = sorted(
            glob.glob(os.path.join(d, "*.fit")) +
            glob.glob(os.path.join(d, "*.fits")) +
            glob.glob(os.path.join(d, "*.fts")))
        self._lbl_lights_count.setText(f"{len(files)} FITS files found")
        # Try to read temperature from first light frame
        if files:
            T = read_fits_temperature(files[0])
            if T is not None:
                self._spin_T_light.setValue(T)
                self._log.appendPlainText(
                    f"[Auto] Light temperature from FITS: {T:.2f} °C")
        if not self._edit_out.text():
            self._edit_out.setText(d)

    def _browse_multi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dark FITS", "", "FITS (*.fit *.fits *.fts);;All (*)")
        if path:
            self._edit_multi_path.setText(path)
            if self._chk_auto_temp.isChecked():
                T = read_fits_temperature(path)
                if T is not None:
                    self._spin_multi_temp.setValue(T)

    def _add_multi_entry(self):
        path = self._edit_multi_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "No file", "Select a FITS file."); return
        T = (read_fits_temperature(path) if self._chk_auto_temp.isChecked()
             else self._spin_multi_temp.value())
        if T is None:
            T = self._spin_multi_temp.value()
        self._multi_entries.append({"path": path, "temp_C": T})
        row = self._multi_table.rowCount(); self._multi_table.insertRow(row)
        for col, val in enumerate([
            str(row+1), os.path.basename(path), f"{T:.1f} °C"
        ]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._multi_table.setItem(row, col, item)
        if not self._edit_out_multi.text():
            self._edit_out_multi.setText(os.path.dirname(path))
        self._edit_multi_path.clear()

    def _clear_multi(self):
        self._multi_entries.clear()
        self._multi_table.setRowCount(0)

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _get_config(self) -> dict | None:
        if self._rb_multi.isChecked():
            if len(self._multi_entries) < 3:
                QMessageBox.warning(self, "Too few entries",
                    "Add at least 3 dark frames at different temperatures.")
                return None
            out = self._edit_out_multi.text().strip()
            if not out:
                QMessageBox.warning(self, "No output",
                    "Set an output folder."); return None
            return {
                "mode":         "multi_temp",
                "multi_entries": list(self._multi_entries),
                "output_dir":   out,
                "Ea_eV":        self._spin_ea.value(),
            }

        dark = self._edit_dark.text().strip()
        if not dark or not os.path.isfile(dark):
            QMessageBox.warning(self, "No dark",
                "Select a master dark FITS file."); return None
        out = self._edit_out.text().strip()
        if not out:
            QMessageBox.warning(self, "No output",
                "Set an output folder."); return None

        lights_folder = self._edit_lights_folder.text().strip()
        light_paths   = []
        if lights_folder and os.path.isdir(lights_folder):
            light_paths = sorted(
                glob.glob(os.path.join(lights_folder, "*.fit")) +
                glob.glob(os.path.join(lights_folder, "*.fits")) +
                glob.glob(os.path.join(lights_folder, "*.fts")))

        mode = "auto" if self._rb_auto.isChecked() else "manual"

        return {
            "mode":           mode,
            "dark_path":      dark,
            "bias_path":      self._edit_bias.text().strip(),
            "light_paths":    light_paths,
            "apply_to_lights": self._chk_apply_lights.isChecked(),
            "output_dir":     out,
            "Ea_eV":          self._spin_ea.value(),
            "T_dark_C":       self._spin_T_dark.value(),
            "T_light_C":      self._spin_T_light.value(),
        }

    def _run(self):
        cfg = self._get_config()
        if cfg is None: return
        self._cancel_event.clear()
        self._btn_run.setEnabled(False); self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)
        self._set_status("Running…", SIRIL_COLD)
        self._worker = DarkCorrectionWorker(cfg, self._cancel_event)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._log.appendPlainText)
        self._worker.arrhenius_fit.connect(
            lambda fit, temps: self._canvas.show_arrhenius_fit(fit, temps))
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self._tabs.setCurrentIndex(5)  # log

    def _cancel(self):
        self._cancel_event.set()
        self._btn_cancel.setEnabled(False)
        self._set_status("Cancelling…", SIRIL_WARNING)

    def _on_progress(self, step: int, total: int, msg: str):
        self._progress.setRange(0, total); self._progress.setValue(step)
        self._set_status(msg, SIRIL_COLD)

    def _on_result_ready(self, original: np.ndarray, scaled: np.ndarray,
                          scale: float, T_dark: float, T_light: float):
        self._canvas.show_histogram(original, scaled, scale, T_dark, T_light)
        self._tabs.setCurrentIndex(4)  # preview

    def _on_finished(self, result: dict):
        self._btn_run.setEnabled(True); self._btn_cancel.setEnabled(False)
        if result.get("success"):
            if result.get("fit"):
                fit = result["fit"]
                self._set_status(
                    f"✓ Fit complete — Ea={fit['Ea_eV']:.4f} eV  "
                    f"R²={fit['r2']:.4f}  "
                    f"doubling={fit['doubling_T']:.1f}°C",
                    SIRIL_SUCCESS)
                # Update the Ea spinner with fitted value
                self._spin_ea.setValue(fit["Ea_eV"])
            else:
                scale = result["scale_factor"]
                Td    = result["T_dark"]
                Tl    = result["T_light"]
                out   = result.get("output_path", "")
                self._set_status(
                    f"✓  {Td:.1f}°C → {Tl:.1f}°C  "
                    f"scale={scale:.6f}×  → {os.path.basename(out)}",
                    SIRIL_SUCCESS)
        else:
            self._set_status(f"✗ {result.get('error','')}", SIRIL_ERROR)

    def _set_status(self, msg: str, color: str = SIRIL_TEXT_DIM):
        self._status.setText(msg)
        self._status.setStyleSheet(f"color:{color}; font-size:9pt;")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import traceback as _tb
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "crash_log.txt")
    def crash_handler(exc_type, exc_value, exc_tb):
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\nCRASH  {datetime.now()}\n{msg}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = crash_handler

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
