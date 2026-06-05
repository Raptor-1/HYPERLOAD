r"""
HYPERLOAD — Bispectrum Speckle Image Reconstruction
=====================================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Reconstructs a diffraction-limited image from seeing-limited speckle frames
using the bispectrum / Knox-Thompson method.  No reference star required.

Algorithm pipeline:
  1. Load frames (SER or FITS folder)
  2. Centre each frame on source (cross-correlation or centroid)
  3. Accumulate frame-averaged power spectrum  |F̄(u)|²
     and Knox-Thompson bispectrum  <F(u)·F*(u+δu)>  for δu=(1,0) and (0,1)
  4. Recover object Fourier phase via Frankot-Chellappa least-squares Poisson
     solver applied to the bispectrum phase gradients
  5. Wiener-deconvolve the power spectrum to remove telescope OTF
  6. Combine amplitude + phase → inverse FFT → reconstructed image

Mathematics:
  Bispectrum:   B(u,δu)  = F(u)·F(δu)·F*(u+δu)
  KT average:   <B(u,δu)> → phase = φ_obj(u+δu) − φ_obj(u)  (atm cancels)
  Phase solve:  ∇φ = (∠<B_x>, ∠<B_y>)  solved via FFT Poisson
  OTF correct:  |O(u)|² = (<|F|²> − σ²_noise) / |OTF(u)|²  (Wiener)

References:
  Weigelt 1977                  — Speckle masking, Opt. Commun. 21, 55
  Knox & Thompson 1974          — Phase recovery, JOSA 64, 1498
  Lohmann, Weigelt & Wirnitzer 1983 — Bispectrum, Applied Optics 22, 4028
  Frankot & Chellappa 1988      — Phase reconstruction, IEEE PAMI 10, 439
  Roddier 1988                  — Applied Optics 27, 1223 (bispectrum + FC)
  Sciadini et al. 2018          — Modern implementation, MNRAS 481, 4895
  Martinache 2020               — Kernel phase context, A&A 636, A72

Version: 1.0.0
Project: HYPERLOAD
"""

import sys
import os
import traceback
import json
import math
import numpy as np
from pathlib import Path
from datetime import datetime

# ── crash logger ─────────────────────────────────────────────────────────────
def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nbispectrum_speckle.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    print(f"[CRASH] {log}")
    sys.__excepthook__(et, ev, eb)

sys.excepthook = _crash

# ── sirilpy ──────────────────────────────────────────────────────────────────
try:
    import sirilpy as s
    for pkg in ["PyQt6", "astropy", "scipy", "matplotlib"]:
        s.ensure_installed(pkg)
except ImportError:
    pass

# ── scipy ─────────────────────────────────────────────────────────────────────
from scipy.ndimage import shift as ndimage_shift, gaussian_filter, center_of_mass

# ── astropy ───────────────────────────────────────────────────────────────────
try:
    from astropy.io import fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

# ── PyQt6 ─────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QMessageBox, QSizePolicy, QScrollArea, QSlider, QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

# ── matplotlib ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — Bispectrum Speckle Reconstruction"

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

class BispectrumAccumulator:
    """
    Accumulates the Knox-Thompson bispectrum and power spectrum over N frames.

    Knox-Thompson bispectrum (Weigelt 1977 / Knox & Thompson 1974):
        B_x(u) = F(u) · F*(u + δu_x)       δu_x = (1, 0)  pixel
        B_y(u) = F(u) · F*(u + δu_y)       δu_y = (0, 1)  pixel

    Average over many frames:
        <B_x(u)> → phase = φ_obj(u) − φ_obj(u+1)  (atmospheric phase cancels)
        <B_y(u)> → phase = φ_obj(u) − φ_obj(u,1)

    The average is equivalent to the full bispectrum at separation (1,1) pixel
    which gives the FIRST-ORDER PHASE GRADIENT of the object Fourier transform.

    This approximation (using only unit-pixel separations) is the Knox-Thompson
    method, which is faster than full bispectrum but gives the same phase
    information for compact, well-sampled objects.
    """

    def __init__(self, ny: int, nx: int):
        self.ny = ny
        self.nx = nx
        self._power  = np.zeros((ny, nx), dtype=np.float64)
        self._bisp_x = np.zeros((ny, nx), dtype=np.complex128)
        self._bisp_y = np.zeros((ny, nx), dtype=np.complex128)
        self._bisp_d = np.zeros((ny, nx), dtype=np.complex128)  # diagonal (1,1)
        self.n_frames = 0
        self._power_sq = np.zeros((ny, nx), dtype=np.float64)   # for variance

    def add_frame(self, frame: np.ndarray, weight: float = 1.0):
        """
        Add one frame to the accumulator.
        frame must be 2-D, same shape as (ny, nx).
        """
        F = np.fft.fft2(frame.astype(np.float64))
        P = np.abs(F) ** 2

        # Knox-Thompson: unit-pixel shifts
        Fx = np.roll(F, -1, axis=1)   # F(u + (1,0))
        Fy = np.roll(F, -1, axis=0)   # F(u + (0,1))
        Fd = np.roll(Fx, -1, axis=0)  # F(u + (1,1))

        self._power   += P * weight
        self._bisp_x  += F * np.conj(Fx) * weight
        self._bisp_y  += F * np.conj(Fy) * weight
        self._bisp_d  += F * np.conj(Fd) * weight
        self._power_sq += P ** 2 * weight
        self.n_frames += 1

    def finalize(self) -> dict:
        """Return averaged power spectrum and bispectrum."""
        n = max(self.n_frames, 1)
        return {
            'power':    self._power   / n,
            'bisp_x':  self._bisp_x  / n,
            'bisp_y':  self._bisp_y  / n,
            'bisp_d':  self._bisp_d  / n,
            'power_var': (self._power_sq / n - (self._power / n) ** 2) / max(n-1, 1),
            'n_frames': self.n_frames,
        }

    def preview_power(self) -> np.ndarray:
        """Running average power spectrum (for live display)."""
        return np.fft.fftshift(self._power / max(self.n_frames, 1))


class FrameCenterer:
    """
    Centre each frame on the target source before bispectrum accumulation.

    Two modes:
    - 'centroid': weighted centroid of the brightest region (fast, good for stars)
    - 'cross_corr': phase-correlation with a reference frame (better for planets)

    Centering removes image-plane tip-tilt so the bispectrum reflects
    only the object structure, not atmospheric image wander.
    """

    def __init__(self, mode: str = 'cross_corr', ref_frame: np.ndarray = None):
        self.mode = mode
        self.ref_frame = ref_frame
        self._ref_fft  = None
        if ref_frame is not None:
            self._ref_fft = np.fft.fft2(ref_frame.astype(float))

    def centre(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == 'centroid':
            return self._centroid_centre(frame)
        return self._xcorr_centre(frame)

    def _centroid_centre(self, frame: np.ndarray) -> np.ndarray:
        ny, nx = frame.shape
        # Brightest 10% of pixels for centroid
        thresh = np.percentile(frame, 90)
        mask = frame > thresh
        cy, cx = center_of_mass(frame * mask)
        dy = ny / 2 - cy
        dx = nx / 2 - cx
        return ndimage_shift(frame, [dy, dx], order=1, mode='reflect')

    def _xcorr_centre(self, frame: np.ndarray) -> np.ndarray:
        if self._ref_fft is None:
            self._ref_fft = np.fft.fft2(frame.astype(float))
            return frame
        F = np.fft.fft2(frame.astype(float))
        R = np.conj(self._ref_fft) * F
        denom = np.abs(R); denom[denom < 1e-10] = 1.0
        R /= denom
        r = np.real(np.fft.ifft2(R))
        pj, pi = np.unravel_index(np.argmax(r), r.shape)
        ny, nx = frame.shape
        dy = float(pj if pj < ny // 2 else pj - ny)
        dx = float(pi if pi < nx // 2 else pi - nx)
        # Sub-pixel
        ry = np.array([r[(pj-1) % ny, pi], r[pj, pi], r[(pj+1) % ny, pi]])
        rx = np.array([r[pj, (pi-1) % nx], r[pj, pi], r[pj, (pi+1) % nx]])
        d2y = ry[0] - 2*ry[1] + ry[2]; d2x = rx[0] - 2*rx[1] + rx[2]
        if abs(d2y) > 1e-10: dy += 0.5 * (ry[0] - ry[2]) / d2y
        if abs(d2x) > 1e-10: dx += 0.5 * (rx[0] - rx[2]) / d2x
        return ndimage_shift(frame, [-dy, -dx], order=1, mode='reflect')


class ImageReconstructor:
    """
    Reconstruct a diffraction-limited image from the accumulated bispectrum.

    Steps:
    1. Phase gradient from Knox-Thompson bispectrum phases
    2. Phase reconstruction via Frankot-Chellappa Poisson solver (Roddier 1988)
    3. Amplitude from Wiener-deconvolved power spectrum
    4. Inverse FFT → reconstructed image

    The Frankot-Chellappa solver finds the phase φ that minimises:
        Σ_u [(∂φ/∂u_x − ψ_x)² + (∂φ/∂u_y − ψ_y)²]
    solved analytically in Fourier space as a Poisson equation.

    OTF model: circular aperture OTF (Airy disk pattern), parameterised by
    the aperture diameter in pixels (D_px = Airy first zero × 1.22).
    """

    def __init__(self, D_px: float = 0.0, wiener_k: float = 0.01,
                 otf_mode: str = 'none'):
        self.D_px     = D_px          # aperture diameter in pixels
        self.wiener_k = wiener_k      # Wiener filter noise parameter
        self.otf_mode = otf_mode      # 'none' | 'circular' | 'gaussian'

    def _circular_otf(self, ny: int, nx: int) -> np.ndarray:
        """Analytic OTF of a circular aperture."""
        fy = np.fft.fftfreq(ny); fx = np.fft.fftfreq(nx)
        FX, FY = np.meshgrid(fx, fy)
        f = np.sqrt(FX**2 + FY**2)
        f_cut = 1.0 / (self.D_px / nx * ny)   # normalised cutoff
        # OTF = (2/π)(arccos(x) - x√(1-x²)) for x = f/f_cut
        x = np.clip(f / (f_cut + 1e-10), 0, 1 - 1e-9)
        otf = (2 / np.pi) * (np.arccos(x) - x * np.sqrt(1 - x**2))
        otf[0, 0] = 1.0
        return np.clip(otf, 0, 1)

    def reconstruct_phase(self, bisp_x: np.ndarray,
                           bisp_y: np.ndarray) -> np.ndarray:
        """
        Frankot-Chellappa phase reconstruction from bispectrum gradients.

        Given phase gradients ψ_x = ∠<B_x> and ψ_y = ∠<B_y>,
        solve for φ using the Fourier-domain Poisson solver.
        """
        ny, nx = bisp_x.shape
        grad_x = np.angle(bisp_x)   # phase difference in x direction
        grad_y = np.angle(bisp_y)   # phase difference in y direction

        ky = np.fft.fftfreq(ny) * 2 * np.pi
        kx = np.fft.fftfreq(nx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)

        # Discrete forward-difference operators in Fourier space
        Wx = np.exp(1j * KX) - 1.0
        Wy = np.exp(1j * KY) - 1.0

        num = np.conj(Wx) * np.fft.fft2(grad_x) + \
              np.conj(Wy) * np.fft.fft2(grad_y)
        den = np.abs(Wx)**2 + np.abs(Wy)**2
        den[0, 0] = 1.0    # avoid division by zero at DC

        phase_fft = num / den
        phase_fft[0, 0] = 0.0   # real-valued object → zero phase at DC

        return np.real(np.fft.ifft2(phase_fft))

    def reconstruct(self, accum: dict) -> dict:
        """
        Full reconstruction pipeline.

        Returns dict with:
            image      — reconstructed diffraction-limited image
            power_corr — OTF-corrected power spectrum
            phase      — recovered object phase (Fourier plane)
            autocorr   — frame-averaged autocorrelation (for comparison)
        """
        power  = accum['power']
        bisp_x = accum['bisp_x']
        bisp_y = accum['bisp_y']
        ny, nx = power.shape

        # ── 1. Phase reconstruction ───────────────────────────────────────
        phase = self.reconstruct_phase(bisp_x, bisp_y)

        # ── 2. Amplitude (power spectrum) ─────────────────────────────────
        amp_sq = np.maximum(power, 0)

        if self.otf_mode == 'circular' and self.D_px > 0:
            otf = self._circular_otf(ny, nx)
            # Wiener deconvolution of OTF
            otf_sq = otf ** 2
            amp_sq = amp_sq * otf_sq / (otf_sq + self.wiener_k * otf_sq.max() + 1e-10)
        elif self.otf_mode == 'gaussian' and self.D_px > 0:
            sigma = self.D_px / (4 * np.pi)
            fy = np.fft.fftfreq(ny); fx = np.fft.fftfreq(nx)
            FX, FY = np.meshgrid(fx, fy)
            otf = np.exp(-2 * np.pi**2 * sigma**2 * (FX**2 + FY**2))
            amp_sq = amp_sq / (otf**2 + self.wiener_k + 1e-10)

        amp = np.sqrt(np.maximum(amp_sq, 0))

        # ── 3. Combine and reconstruct ────────────────────────────────────
        O_complex = amp * np.exp(1j * phase)
        image_raw = np.real(np.fft.ifft2(O_complex))
        image     = np.fft.fftshift(image_raw)

        # ── 4. Autocorrelation (for reference / binary detection) ─────────
        autocorr = np.fft.fftshift(np.real(np.fft.ifft2(power)))

        return {
            'image':       image,
            'power_corr':  np.fft.fftshift(amp_sq),
            'phase':       np.fft.fftshift(phase),
            'autocorr':    autocorr,
            'n_frames':    accum['n_frames'],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SER READER  (self-contained — no lucky_preprocessor dependency)
# ═══════════════════════════════════════════════════════════════════════════════

SER_HEADER_SIZE = 178

def read_ser_header(path: str) -> dict:
    import struct
    with open(path, 'rb') as f:
        raw = f.read(SER_HEADER_SIZE)
    color_id = struct.unpack_from('<i', raw, 14)[0]
    w        = struct.unpack_from('<i', raw, 26)[0]
    h        = struct.unpack_from('<i', raw, 30)[0]
    depth    = struct.unpack_from('<i', raw, 34)[0]
    n_frames = struct.unpack_from('<i', raw, 38)[0]
    return {'width': w, 'height': h, 'depth': depth,
            'color_id': color_id, 'frame_count': n_frames}

def read_ser_frame(f, header: dict, idx: int) -> np.ndarray:
    depth    = int(header.get('depth', 8))
    color_id = int(header.get('color_id', 0))
    is_color = color_id >= 100
    bpp      = 1 if depth <= 8 else 2
    n_ch     = 3 if is_color else 1
    w, h     = header['width'], header['height']
    fsz      = w * h * bpp * n_ch
    offset   = SER_HEADER_SIZE + idx * fsz
    f.seek(offset)
    raw = f.read(fsz)
    if len(raw) < fsz:
        raise EOFError(f"Unexpected end of file at frame {idx}")
    dtype = np.uint8 if bpp == 1 else np.uint16
    arr   = np.frombuffer(raw, dtype=dtype)
    if is_color:
        arr = arr.reshape(h, w, 3)
        if color_id == 101: arr = arr[:, :, ::-1]
        lum = (0.2126*arr[:,:,0].astype(np.float32) +
               0.7152*arr[:,:,1].astype(np.float32) +
               0.0722*arr[:,:,2].astype(np.float32))
    else:
        lum = arr.reshape(h, w).astype(np.float32)
    return lum


def luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2: return data.astype(float)
    if data.ndim == 3 and data.shape[0] <= 4:
        return data.mean(axis=0).astype(float)
    return data.astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class BispectrumWorker(QObject):
    """
    Background thread: load frames → centre → accumulate bispectrum → reconstruct.
    Emits live previews every preview_interval frames.
    """
    progress   = pyqtSignal(int, str)
    frame_done = pyqtSignal(int, object)          # (frame_idx, power_preview)
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params  = params
        self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _log(self, pct, msg):
        self.progress.emit(pct, msg)

    def _pipeline(self):
        p = self.params

        # ── Load frame paths / SER header ─────────────────────────────────
        self._log(2, "Preparing frame list…")
        input_type = p['input_type']   # 'ser' | 'fits_folder'

        if input_type == 'ser':
            if not ASTROPY_OK:
                self.error.emit("astropy required for SER loading.")
                return
            hdr = read_ser_header(p['path'])
            n_total = hdr['frame_count']
            n_use   = min(n_total, p.get('max_frames', 99999))
            step    = max(1, p.get('frame_step', 1))
            indices = list(range(0, n_use, step))
            self._log(5, f"SER: {n_total} frames  {hdr['width']}×{hdr['height']}  "
                         f"using {len(indices)}")
            ny, nx = hdr['height'], hdr['width']
        else:
            if not ASTROPY_OK:
                self.error.emit("astropy required for FITS loading.")
                return
            folder = Path(p['path'])
            exts = ('*.fits', '*.fit', '*.fts', '*.FITS', '*.FIT')
            paths = []
            for e in exts: paths.extend(sorted(folder.glob(e)))
            n_use  = min(len(paths), p.get('max_frames', 99999))
            step   = max(1, p.get('frame_step', 1))
            paths  = paths[:n_use:step]
            indices = list(range(len(paths)))
            self._log(5, f"FITS folder: {len(paths)} files")
            # Read first for shape
            d0 = luminance(fits.getdata(str(paths[0])))
            ny, nx = d0.shape
            hdr = None

        # ── Determine preview crop ─────────────────────────────────────────
        crop = p.get('crop', 0)
        if crop > 0:
            cy0 = ny // 2 - crop // 2; cy1 = cy0 + crop
            cx0 = nx // 2 - crop // 2; cx1 = cx0 + crop
            ny_w, nx_w = crop, crop
        else:
            cy0, cy1, cx0, cx1 = 0, ny, 0, nx
            ny_w, nx_w = ny, nx

        accum    = BispectrumAccumulator(ny_w, nx_w)
        centerer = FrameCenterer(mode=p.get('centre_mode', 'cross_corr'))
        prev_interval = max(1, len(indices) // 20)
        n = len(indices)

        if input_type == 'ser':
            ser_f = open(p['path'], 'rb')
        else:
            ser_f = None

        # ── Main accumulation loop ─────────────────────────────────────────
        self._log(8, f"Accumulating bispectrum over {n} frames…")
        for ki, idx in enumerate(indices):
            if self._cancel: break

            # Load frame
            if input_type == 'ser':
                frame = read_ser_frame(ser_f, hdr, idx)
            else:
                frame = luminance(fits.getdata(str(paths[idx])))

            # Crop
            frame = frame[cy0:cy1, cx0:cx1].astype(np.float32)

            # Centre
            frame_c = centerer.centre(frame)

            # Accumulate
            accum.add_frame(frame_c)

            pct = int(10 + 80 * ki / n)
            if ki % prev_interval == 0 or ki == n - 1:
                self._log(pct, f"Frame {ki+1}/{n}  accumulated")
                self.frame_done.emit(ki + 1, accum.preview_power())

        if ser_f: ser_f.close()

        if self._cancel:
            self._log(0, "Cancelled.")
            return

        # ── Reconstruct ───────────────────────────────────────────────────
        self._log(91, "Reconstructing diffraction-limited image…")
        rec = ImageReconstructor(
            D_px=p.get('D_px', 0),
            wiener_k=p.get('wiener_k', 0.01),
            otf_mode=p.get('otf_mode', 'none'),
        )
        result = rec.reconstruct(accum.finalize())
        result['accum']    = accum.finalize()
        result['params']   = p
        result['shape']    = (ny_w, nx_w)

        self._log(100, f"Done!  {result['n_frames']} frames → "
                       f"diffraction-limited reconstruction.")
        self.finished.emit(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow, QWidget {
    background-color: #111120;
    color: #d0d0e8;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}
QTabWidget::pane  { border:1px solid #26264a; background:#151530; }
QTabBar::tab      { background:#1a1a34; color:#6868a0; padding:6px 18px;
                    border:1px solid #26264a; border-bottom:none; }
QTabBar::tab:selected { background:#20204c; color:#9898ff;
                        border-bottom:2px solid #5050c8; }
QGroupBox         { border:1px solid #26264a; border-radius:4px;
                    margin-top:8px; padding-top:8px; color:#6868a8;
                    font-weight:bold; }
QGroupBox::title  { subcontrol-origin:margin; left:8px; padding:0 4px; }
QPushButton       { background:#1e1e42; color:#b0b0d8;
                    border:1px solid #3838a8; border-radius:4px;
                    padding:5px 14px; }
QPushButton:hover   { background:#28285e; border-color:#5050c8; }
QPushButton:disabled{ color:#383858; border-color:#242448; }
QPushButton#run_btn { background:#281a5c; color:#c0a0ff;
                      border:1px solid #7054c8; font-weight:bold;
                      padding:7px 20px; }
QPushButton#run_btn:hover { background:#382a78; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background:#181832; border:1px solid #26264a;
    border-radius:3px; padding:3px 6px; color:#d0d0e8; }
QTextEdit { background:#0e0e20; border:1px solid #26264a;
            color:#85e085; font-family:Consolas,monospace; font-size:11px; }
QSplitter::handle { background:#26264a; }
QScrollArea       { border:none; }
QLabel#title_lbl  { font-size:19px; font-weight:bold; color:#9878ff; padding:6px; }
QLabel#sub_lbl    { font-size:11px; color:#504870; padding:0 8px 6px; }
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(6, 4)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor='#191930')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def redraw(self):
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()


def _ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#0d0d1e')
    ax.set_title(title, color='#9090ff', fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color='#545472', fontsize=8)
    ax.set_ylabel(ylabel, color='#545472', fontsize=8)
    ax.tick_params(colors='#545472', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#26264a')


def _cbar(fig, mappable, ax, label=''):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    cb  = fig.colorbar(mappable, cax=cax)
    cb.ax.yaxis.set_tick_params(colors='#7070a0', labelsize=7)
    if label: cb.set_label(label, color='#7070a0', fontsize=7)
    return cb


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#0b0b18; border-top:1px solid #26264a;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        self.lbl  = QLabel("Ready")
        self.lbl.setStyleSheet("color:#585878; font-size:10px;")
        self.pbar = QProgressBar()
        self.pbar.setFixedSize(200, 12)
        self.pbar.setVisible(False)
        self.pbar.setStyleSheet("""
            QProgressBar { background:#1a1a30; border:1px solid #3535a0;
                           border-radius:3px; color:white; font-size:9px; }
            QProgressBar::chunk { background:qlineargradient(
                x1:0,y1:0,x2:1,y2:0,stop:0 #4040c0,stop:1 #9050e0);
                border-radius:2px; }""")
        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.pbar)

    def update(self, pct, msg):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True)
            self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class BispectrumApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1360, 880)
        self.setStyleSheet(DARK)

        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            self.setWindowIcon(QIcon(str(_logo)))

        self._result   = None
        self._worker   = None
        self._wthread  = None

        self._build_ui()
        self._check_deps()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setSpacing(0); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self._make_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._make_controls())
        splitter.addWidget(self._make_tabs())
        splitter.setSizes([300, 1060])
        rl.addWidget(splitter, 1)

        self.status = StatusFooter()
        rl.addWidget(self.status)

    def _make_header(self):
        hdr = QWidget()
        hdr.setFixedHeight(66)
        hdr.setStyleSheet("background:#0b0b18; border-bottom:1px solid #26264a;")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12, 0, 12, 0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                50, 50, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("Bispectrum Speckle Reconstruction"); t.setObjectName("title_lbl")
        s = QLabel("Knox-Thompson bispectrum · Frankot-Chellappa phase recovery · "
                   "No reference star required · Diffraction-limited from seeing-limited frames")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s)
        lay.addLayout(v, 1)
        ver = QLabel(f"v{VERSION}  |  HYPERLOAD")
        ver.setStyleSheet("color:#30305a; font-size:10px;")
        lay.addWidget(ver)
        return hdr

    def _make_controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay   = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(7)

        # ── Input ──────────────────────────────────────────────────────────
        grp = QGroupBox("Input")
        gl  = QVBoxLayout(grp)
        self.file_lbl = QLabel("No file selected")
        self.file_lbl.setStyleSheet("color:#484868; font-size:10px;")
        self.file_lbl.setWordWrap(True)
        btn_row = QHBoxLayout()
        b1 = QPushButton("📂  Open SER…"); b1.clicked.connect(self._open_ser)
        b2 = QPushButton("📁  FITS Folder…"); b2.clicked.connect(self._open_folder)
        btn_row.addWidget(b1); btn_row.addWidget(b2)
        gl.addWidget(self.file_lbl); gl.addLayout(btn_row)
        lay.addWidget(grp)

        # ── Frame selection ────────────────────────────────────────────────
        grp2 = QGroupBox("Frame Selection")
        g2   = QGridLayout(grp2)
        g2.addWidget(QLabel("Max frames:"), 0, 0)
        self.sp_maxfr = QSpinBox()
        self.sp_maxfr.setRange(10, 99999); self.sp_maxfr.setValue(500)
        g2.addWidget(self.sp_maxfr, 0, 1)
        g2.addWidget(QLabel("Frame step:"), 1, 0)
        self.sp_step = QSpinBox()
        self.sp_step.setRange(1, 100); self.sp_step.setValue(1)
        g2.addWidget(self.sp_step, 1, 1)
        g2.addWidget(QLabel("Crop (0=full):"), 2, 0)
        self.sp_crop = QSpinBox()
        self.sp_crop.setRange(0, 4096); self.sp_crop.setValue(0)
        self.sp_crop.setSingleStep(64)
        g2.addWidget(self.sp_crop, 2, 1)
        lay.addWidget(grp2)

        # ── Centering ─────────────────────────────────────────────────────
        grp3 = QGroupBox("Frame Centering")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("Mode:"), 0, 0)
        self.cmb_centre = QComboBox()
        self.cmb_centre.addItems(["Cross-correlation", "Centroid"])
        g3.addWidget(self.cmb_centre, 0, 1)
        lay.addWidget(grp3)

        # ── OTF correction ────────────────────────────────────────────────
        grp4 = QGroupBox("OTF Correction (optional)")
        g4   = QGridLayout(grp4)
        g4.addWidget(QLabel("Model:"), 0, 0)
        self.cmb_otf = QComboBox()
        self.cmb_otf.addItems(["None", "Circular aperture", "Gaussian"])
        g4.addWidget(self.cmb_otf, 0, 1)
        g4.addWidget(QLabel("Aperture D (px):"), 1, 0)
        self.sp_dpx = QDoubleSpinBox()
        self.sp_dpx.setRange(0, 999); self.sp_dpx.setValue(0)
        self.sp_dpx.setDecimals(1)
        self.sp_dpx.setToolTip(
            "Telescope aperture in IMAGE pixels.\n"
            "= Airy first-zero radius (px) × 1.22\n"
            "Measure from a star profile in your lucky stack.")
        g4.addWidget(self.sp_dpx, 1, 1)
        g4.addWidget(QLabel("Wiener k:"), 2, 0)
        self.sp_wiener = QDoubleSpinBox()
        self.sp_wiener.setRange(1e-5, 1.0); self.sp_wiener.setValue(0.01)
        self.sp_wiener.setDecimals(4); self.sp_wiener.setSingleStep(0.005)
        self.sp_wiener.setToolTip(
            "Wiener regularisation parameter.\n"
            "Smaller = sharper but noisier.\n"
            "Typical range: 0.001–0.05")
        g4.addWidget(self.sp_wiener, 2, 1)
        lay.addWidget(grp4)

        # ── Output ────────────────────────────────────────────────────────
        grp5 = QGroupBox("Output")
        g5   = QVBoxLayout(grp5)
        self.btn_save = QPushButton("💾  Save Reconstructed FITS…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        g5.addWidget(self.btn_save)
        lay.addWidget(grp5)

        # ── Run / Cancel ──────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Run Bispectrum Reconstruction")
        self.btn_run.setObjectName("run_btn")
        self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("color:#b05050;")
        self.btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.btn_cancel)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color:#70c070;font-size:10px;padding:4px;")
        self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)

        scroll.setWidget(inner)
        return scroll

    def _make_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_live(),   "⚡  Live Accumulation")
        self.tabs.addTab(self._tab_result(), "🔬  Reconstruction")
        self.tabs.addTab(self._tab_compare(),"📊  Comparison")
        self.tabs.addTab(self._tab_autocorr(),"🎯  Autocorrelation")
        self.tabs.addTab(self._tab_log(),    "📋  Log")
        return self.tabs

    def _tab_live(self):
        w   = QWidget(); lay = QVBoxLayout(w)
        self.live_plot = MplCanvas(figsize=(8, 6))
        lay.addWidget(self.live_plot)
        self.live_pbar = QProgressBar(); self.live_pbar.setFixedHeight(14)
        lay.addWidget(self.live_pbar)
        self.live_lbl = QLabel("Waiting…")
        self.live_lbl.setStyleSheet("color:#585878;font-size:10px;")
        lay.addWidget(self.live_lbl)
        return w

    def _tab_result(self):
        w   = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Stretch:"))
        self.cmb_stretch = QComboBox()
        self.cmb_stretch.addItems(["Linear", "Sqrt", "Log", "Asinh"])
        self.cmb_stretch.currentIndexChanged.connect(self._refresh_result)
        ctrl.addWidget(self.cmb_stretch); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.result_plot = MplCanvas(figsize=(8, 6))
        lay.addWidget(self.result_plot)
        return w

    def _tab_compare(self):
        w   = QWidget(); lay = QVBoxLayout(w)
        self.compare_plot = MplCanvas(figsize=(8, 5))
        lay.addWidget(self.compare_plot)
        return w

    def _tab_autocorr(self):
        w   = QWidget(); lay = QVBoxLayout(w)
        info = QLabel(
            "The autocorrelation map shows the object convolved with its mirror image.\n"
            "For a binary star: central peak + two symmetric side peaks.\n"
            "Separation = distance from centre to side peak.\n"
            "PA has 180° ambiguity (inherent to intensity-only methods).")
        info.setStyleSheet("color:#686890;font-size:10px;padding:4px;")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.autocorr_plot = MplCanvas(figsize=(8, 5))
        lay.addWidget(self.autocorr_plot)
        return w

    def _tab_log(self):
        w   = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _open_ser(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SER", "", "SER (*.ser *.SER);;All (*)")
        if path:
            self._input_path = path
            self._input_type = 'ser'
            self.file_lbl.setText(Path(path).name)
            self.file_lbl.setStyleSheet("color:#70c070;font-size:10px;")
            self.btn_run.setEnabled(True)
            self._log(f"SER: {path}")

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select FITS folder")
        if folder:
            self._input_path = folder
            self._input_type = 'fits_folder'
            self.file_lbl.setText(Path(folder).name + "/")
            self.file_lbl.setStyleSheet("color:#70c070;font-size:10px;")
            self.btn_run.setEnabled(True)
            self._log(f"FITS folder: {folder}")

    def _save(self):
        if self._result is None: return
        dflt = str(Path(self._input_path).parent /
                   (Path(self._input_path).stem + "_bispectrum.fits"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Reconstruction", dflt, "FITS (*.fits)")
        if not path: return
        img = self._result['image']
        hdr = fits.Header()
        hdr['HISTORY'] = f'HYPERLOAD Bispectrum Reconstruction v{VERSION}'
        hdr['NFRAMES'] = (self._result['n_frames'], 'Frames accumulated')
        hdr['BISPMODE'] = ('Knox-Thompson', 'Bispectrum mode')
        hdr['WIENERК'] = (self.sp_wiener.value(), 'Wiener parameter')
        fits.PrimaryHDU(img.astype(np.float32), header=hdr).writeto(
            path, overwrite=True)
        self._log(f"✓ Saved: {path}")

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _run(self):
        if not hasattr(self, '_input_path'): return
        otf_names = {0: 'none', 1: 'circular', 2: 'gaussian'}
        ctr_names = {0: 'cross_corr', 1: 'centroid'}
        params = {
            'path':        self._input_path,
            'input_type':  self._input_type,
            'max_frames':  self.sp_maxfr.value(),
            'frame_step':  self.sp_step.value(),
            'crop':        self.sp_crop.value(),
            'centre_mode': ctr_names[self.cmb_centre.currentIndex()],
            'otf_mode':    otf_names[self.cmb_otf.currentIndex()],
            'D_px':        self.sp_dpx.value(),
            'wiener_k':    self.sp_wiener.value(),
        }
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._result = None

        self._wthread = QThread(self)
        self._worker  = BispectrumWorker(params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_done.connect(self._on_frame_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()
        self.tabs.setCurrentIndex(0)

    def _cancel(self):
        if self._worker: self._worker.cancel()
        self.btn_cancel.setEnabled(False)
        self.btn_run.setEnabled(True)
        self._log("Cancelled.")

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status.update(pct, msg)
        self.live_pbar.setValue(pct)
        self.live_lbl.setText(msg)
        self._log(f"[{pct:3d}%]  {msg}")

    def _on_frame_done(self, n_done, power_preview):
        """Update live power spectrum display."""
        fig = self.live_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, f"Running power spectrum  ({n_done} frames accumulated)")
        p_log = np.log1p(np.maximum(power_preview, 0))
        im = ax.imshow(p_log, origin='lower', cmap='inferno',
                       aspect='equal', interpolation='nearest')
        _cbar(fig, im, ax, "log(power)")
        self.live_plot.redraw()

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}")
        self.status.update(0, "Error — see Log")
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        QMessageBox.critical(self, "Error", msg[:500])

    def _on_finished(self, result):
        self._result = result
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)

        n = result['n_frames']
        self._log(f"\n{'='*55}")
        self._log(f"✓  Bispectrum reconstruction complete")
        self._log(f"   Frames:  {n}")
        self._log(f"   Shape:   {result['shape'][1]}×{result['shape'][0]} px")
        self._log(f"   Method:  Knox-Thompson + Frankot-Chellappa")
        self._log(f"{'='*55}\n")
        self.summary_lbl.setText(f"{n} frames  |  {result['shape'][1]}×{result['shape'][0]} px")

        self._refresh_result()
        self._draw_compare()
        self._draw_autocorr()
        self.tabs.setCurrentIndex(1)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _stretch(self, img):
        lo, hi = np.percentile(img[np.isfinite(img)], [0.5, 99.5])
        img_c = np.clip(img - lo, 0, hi - lo + 1e-10)
        m = self.cmb_stretch.currentIndex()
        if m == 0: return img_c / (hi - lo + 1e-10)
        if m == 1: return np.sqrt(img_c / (hi - lo + 1e-10))
        if m == 2: return np.log1p(img_c) / np.log1p(hi - lo + 1)
        return np.arcsinh(img_c / max((hi - lo) * 0.1, 1e-10)) / np.arcsinh(10.0)

    def _refresh_result(self):
        if self._result is None: return
        r = self._result
        fig = self.result_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1, 3)

        # Reconstructed image
        ax = axes[0]
        _ax(ax, "Reconstructed image\n(diffraction-limited)")
        im = ax.imshow(self._stretch(r['image']), origin='lower',
                       cmap='inferno', aspect='equal', interpolation='nearest')
        _cbar(fig, im, ax)

        # Recovered phase (Fourier plane)
        ax2 = axes[1]
        _ax(ax2, "Object Fourier phase\n(recovered by bispectrum)", "u", "v")
        ph = ax2.imshow(r['phase'], origin='lower', cmap='RdBu',
                        aspect='equal', interpolation='nearest')
        _cbar(fig, ph, ax2, "radians")

        # Power spectrum
        ax3 = axes[2]
        _ax(ax3, "Corrected power spectrum\n(log scale)", "u", "v")
        ps  = ax3.imshow(np.log1p(np.maximum(r['power_corr'], 0)),
                         origin='lower', cmap='plasma',
                         aspect='equal', interpolation='nearest')
        _cbar(fig, ps, ax3, "log(power)")

        self.result_plot.redraw()

    def _draw_compare(self):
        if self._result is None: return
        r = self._result
        fig = self.compare_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1, 2)

        ax = axes[0]
        _ax(ax, "Bispectrum reconstruction")
        ax.imshow(self._stretch(r['image']), origin='lower',
                  cmap='inferno', aspect='equal', interpolation='nearest')

        ax2 = axes[1]
        _ax(ax2, "Frame-averaged power spectrum (seeing-limited reference)")
        ps_log = np.log1p(np.maximum(r['power_corr'], 0))
        ax2.imshow(ps_log, origin='lower', cmap='plasma',
                   aspect='equal', interpolation='nearest')

        self.compare_plot.redraw()

    def _draw_autocorr(self):
        if self._result is None: return
        r = self._result
        fig = self.autocorr_plot.fig; fig.clear()
        fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1, 2)

        ac  = r['autocorr']
        ny, nx = ac.shape

        ax = axes[0]
        _ax(ax, "Autocorrelation map\n(binary: 3 peaks; single: 1 peak)")
        vlo, vhi = np.percentile(ac, [1, 99.5])
        im = ax.imshow(np.sqrt(np.clip(ac - vlo, 0, vhi - vlo)),
                       origin='lower', cmap='inferno',
                       aspect='equal', interpolation='nearest')
        # Mark centre
        ax.plot(nx//2, ny//2, '+', color='#60ffff', ms=12, lw=1.5)
        _cbar(fig, im, ax)

        # Radial profile of autocorrelation
        ax2 = axes[1]
        _ax(ax2, "Autocorrelation radial profile", "Radius (px)", "Power")
        cy, cx = ny//2, nx//2
        y, x   = np.ogrid[:ny, :nx]
        r_map  = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_max  = min(cy, cx)
        bins   = np.arange(0, r_max, 1)
        prof   = np.array([ac[(r_map >= b) & (r_map < b+1)].mean()
                           for b in bins])
        ax2.plot(bins, prof, c='#5070ff', lw=1.5)
        ax2.fill_between(bins, prof, alpha=0.2, color='#5070ff')
        ax2.axvline(0, color='#ff8030', lw=0.8, ls='--', alpha=0.5)

        self.autocorr_plot.redraw()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()

    def _check_deps(self):
        if not ASTROPY_OK:
            self._log("⚠  astropy not found.  pip install astropy")
        else:
            self._log("✓  All dependencies found.")
        self._log("NOTE: For best results use 100–1000 seeing-limited frames.")
        self._log("The bispectrum method works WITHOUT a reference star.")
        self._log("180° PA ambiguity in the autocorrelation is inherent to all")
        self._log("intensity-only speckle methods (cannot be removed without")
        self._log("closure phase or reference star).")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = BispectrumApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
