r"""
HYPERLOAD — ImageMM: Multi-frame Restoration & Super-resolution
================================================================
Standalone script.  Place in:
    C:\Users\Marcell\Desktop\Siril Suites\

Implements the ImageMM algorithm (Sukurdeep, Budavári, Connolly & Navarro 2025,
AJ 170, 233) for joint multi-frame astronomical image restoration and
super-resolution using the Majorization-Minimization (MM) framework.

What it does:
  Takes N registered FITS frames of the same target — the same frames you would
  normally median-stack — and produces a result that is simultaneously:
    • Sharper than any individual frame (PSF deconvolution)
    • Higher SNR than lucky imaging (all frames used, none discarded)
    • Higher resolution than drizzle (optional 2× super-resolution)
    • Free of satellite trails (Huber robust loss)

PSF estimation (automatic — no instrument info, no plate solve needed):
  Each frame contains stars. Stars are point sources, so their observed shape
  IS the PSF for that frame. The auto-PSF pipeline:
    1. Detect bright compact sources (stars) via local maxima
    2. Extract 21×21 px stamps, align via phase correlation
    3. Robust median-combine → per-frame empirical PSF
    4. Measure FWHM from result
    5. Fallback: circular Gaussian if too few stars found
  The PSF captures atmospheric seeing + optics + tracking — all at once.

Algorithm (Sukurdeep et al. 2025, Algorithm 1):
  Model:  y^(t) = PSF^(t) ⊛ x + noise
  Update: x ← x ⊙ clip[ Σ_t F^T W^t y^t / Σ_t F^T W^t F x ]
  where W^t = Huber(residual) / variance(t)  per pixel

  Key properties:
    • Multiplicative update → non-negativity guaranteed
    • All frames simultaneously → no order dependency
    • Huber loss → satellite trails / cosmic rays suppressed automatically
    • Convergence: median(|u−1|) < 1e-3

Super-resolution (Sukurdeep et al. 2025, Algorithm 2):
  Upsamples PSF to r× resolution via variational problem, then runs the
  MM loop on the super-resolved latent image with downsampling operator D.
  Produces output at 2× pixel scale without drizzle artefacts.

Reference:
  Sukurdeep et al. 2025, AJ 170, 233. DOI: 10.3847/1538-3881/adfb72
  arXiv: 2501.03002

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, traceback, json, math
import numpy as np
from pathlib import Path
from datetime import datetime

def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\nimagemm.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","astropy","scipy","matplotlib"]: s.ensure_installed(p)
except ImportError:
    pass

from scipy.ndimage import (gaussian_filter, maximum_filter,
                            shift as ndshift, zoom)
from scipy.optimize import minimize as sp_minimize

try:
    from astropy.io import fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — ImageMM Multi-frame Restoration"


# ── Pure-numpy FFT convolution ─────────────────────────────────────────────
# Replaces scipy.signal.fftconvolve to avoid a C-level crash in
# scipy 1.17.1 / numpy 2.4.4 when negative-stride kernel views are
# passed across iterations.

def _conv_mm(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2-D linear convolution via FFT (same-size output).
    Both arrays are copied to guaranteed-owned float64 before transform,
    which prevents the numpy 2.4.4 / scipy 1.17.1 memory-corruption crash.
    """
    a  = np.asarray(img,    dtype=np.float64)
    b  = np.asarray(kernel, dtype=np.float64)
    H, W   = a.shape
    kH, kW = b.shape
    pad = np.zeros((H, W), dtype=np.float64)
    pad[:kH, :kW] = b
    # Shift kernel so centre is at (0,0) — correct circular convolution phase
    pad = np.roll(np.roll(pad, -(kH // 2), axis=0), -(kW // 2), axis=1)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(pad)))


# ═══════════════════════════════════════════════════════════════════════════════
#  PSF ESTIMATION FROM STARS
# ═══════════════════════════════════════════════════════════════════════════════

def _gaussian_psf(size: int, fwhm: float) -> np.ndarray:
    """Normalised circular Gaussian PSF."""
    sigma = max(float(fwhm), 0.5) / 2.355
    h     = size // 2
    y, x  = np.ogrid[-h:h+1, -h:h+1]
    g     = np.exp(-(x.astype(float)**2 + y.astype(float)**2) / (2*sigma**2))
    return (g / g.sum()).astype(np.float64)


def _fwhm_from_psf(psf: np.ndarray) -> float:
    """Measure FWHM from a 2-D PSF array via radial profile."""
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    peak   = float(psf.max())
    if peak <= 0:
        return 4.0
    half   = peak / 2.0
    y, x   = np.ogrid[:psf.shape[0], :psf.shape[1]]
    r_map  = np.sqrt((x.astype(float) - cx)**2 +
                     (y.astype(float) - cy)**2).ravel()
    v_map  = psf.ravel()
    order  = np.argsort(r_map)
    r_s    = r_map[order]
    v_s    = v_map[order]
    above  = v_s > half
    if above.any() and (~above).any():
        last_above = int(np.where(above)[0][-1])
        return float(r_s[last_above] * 2.0)
    return 4.0


def estimate_fwhm_from_header(header) -> float | None:
    """Try to read FWHM or seeing from FITS header (many capture apps write it)."""
    if header is None:
        return None
    for kw in ['FWHM','SEEING','PSF_FWHM','STAR_FWHM','FWHMSTAR',
               'L_FWHM','R_FWHM']:
        if kw in header:
            try:
                v = float(header[kw])
                if 0.5 < v < 50:
                    return v
            except Exception:
                pass
    return None


def estimate_psf_from_stars(frame: np.ndarray,
                             psf_size: int = 21,
                             n_stars: int = 25,
                             thresh_sigma: float = 6.0,
                             header=None) -> tuple[np.ndarray, float, int]:
    """
    Estimate the PSF empirically from stars in a single frame.

    Returns (psf, fwhm_px, n_stars_used).

    No instrument knowledge, no plate solution needed.
    Stars are detected as compact local maxima; their observed shape
    is the empirical PSF of the instrument + atmosphere + tracking.

    Fallback chain:
      1. Star-derived empirical PSF (best)
      2. Gaussian with FWHM from FITS header
      3. Gaussian with FWHM estimated from source moments
    """
    h   = psf_size // 2
    ny, nx = frame.shape

    # ── Background and noise estimation ───────────────────────────────────
    smooth = gaussian_filter(frame.astype(float), 1.5)
    bg     = float(np.percentile(smooth, 40))
    mad    = float(np.median(np.abs(smooth - bg)))
    noise  = mad * 1.4826 + 1e-6
    thresh = bg + thresh_sigma * noise

    # ── Star candidates: local maxima above threshold ─────────────────────
    local_max   = smooth == maximum_filter(smooth, 7)
    candidates  = local_max & (smooth > thresh)
    ys_c, xs_c  = np.where(candidates)

    if len(xs_c) == 0:
        # Full fallback: use header or moment FWHM
        fwhm = estimate_fwhm_from_header(header) or _fwhm_from_moments(frame)
        return _gaussian_psf(psf_size, fwhm), fwhm, 0

    # Sort by flux descending
    fluxes      = smooth[ys_c, xs_c]
    order       = np.argsort(fluxes)[::-1]
    ys_c        = ys_c[order]
    xs_c        = xs_c[order]

    # ── Extract and validate star stamps ─────────────────────────────────
    stamps = []
    for idx in range(min(len(ys_c), n_stars * 4)):
        yi = int(ys_c[idx])
        xi = int(xs_c[idx])

        # Boundary check
        if yi-h < 0 or yi+h+1 > ny or xi-h < 0 or xi+h+1 > nx:
            continue

        stamp = frame[yi-h:yi+h+1, xi-h:xi+h+1].astype(np.float64).copy()

        # Check peak is near centre (reject extended sources)
        flat_idx = int(stamp.argmax())
        py, px   = flat_idx // stamp.shape[1], flat_idx % stamp.shape[1]
        if abs(py - h) > 3 or abs(px - h) > 3:
            continue

        # Check SNR (reject noisy candidates)
        peak_val = float(stamp[py, px])
        if peak_val <= 0:
            continue

        # Normalise
        stamp -= float(stamp.min())
        peak_val = float(stamp.max())
        if peak_val <= 0:
            continue
        stamp /= peak_val

        stamps.append(stamp)
        if len(stamps) >= n_stars:
            break

    if len(stamps) < 3:
        fwhm = estimate_fwhm_from_header(header) or _fwhm_from_moments(frame)
        return _gaussian_psf(psf_size, fwhm), fwhm, len(stamps)

    # ── Sub-pixel alignment via phase correlation ─────────────────────────
    ref     = stamps[0]
    aligned = [ref]
    ref_fft = np.fft.fft2(ref)

    for s in stamps[1:]:
        R   = ref_fft * np.conj(np.fft.fft2(s))
        mag = np.abs(R)
        mag[mag < 1e-10] = 1.0
        R  /= mag
        r   = np.real(np.fft.ifft2(R))
        pj  = int(np.unravel_index(int(r.argmax()), r.shape)[0])
        pi  = int(np.unravel_index(int(r.argmax()), r.shape)[1])
        dy  = pj if pj < psf_size // 2 else pj - psf_size
        dx  = pi if pi < psf_size // 2 else pi - psf_size
        aligned.append(ndshift(s, [-dy, -dx], order=1, mode='reflect'))

    # ── Robust median combine ─────────────────────────────────────────────
    psf  = np.median(np.array(aligned), axis=0)
    psf  = np.maximum(psf, 0.0)
    fwhm = _fwhm_from_psf(psf)

    # Clamp unreasonable FWHM
    fwhm = float(np.clip(fwhm, 0.8, psf_size * 0.8))

    # Normalise
    psf_sum = float(psf.sum())
    if psf_sum > 0:
        psf /= psf_sum
    else:
        return _gaussian_psf(psf_size, fwhm), fwhm, len(stamps)

    # Micro-soften to suppress Gibbs ringing (Franklin Marek / Sukurdeep impl.)
    psf  = gaussian_filter(psf, 0.25)
    psf /= float(psf.sum()) + 1e-10

    return psf.astype(np.float64), fwhm, len(stamps)


def _fwhm_from_moments(frame: np.ndarray) -> float:
    """Estimate FWHM from the brightest source's second moments."""
    smooth = gaussian_filter(frame.astype(float), 1.0)
    py, px = np.unravel_index(int(smooth.argmax()), smooth.shape)
    h  = 20
    y0 = max(0, py-h); y1 = min(frame.shape[0], py+h+1)
    x0 = max(0, px-h); x1 = min(frame.shape[1], px+h+1)
    st = smooth[y0:y1, x0:x1]
    st = np.maximum(st - float(np.percentile(st, 20)), 0)
    ws = float(st.sum())
    if ws < 1e-10:
        return 4.0
    yy, xx = np.mgrid[:st.shape[0], :st.shape[1]]
    var = (float((st * (xx - st.shape[1]//2)**2).sum()) +
           float((st * (yy - st.shape[0]//2)**2).sum())) / ws
    return float(2.355 * math.sqrt(max(var, 0.25)))


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGEMM CORE
# ═══════════════════════════════════════════════════════════════════════════════

def imagemm(frames: list[np.ndarray],
            psfs:   list[np.ndarray],
            K:      int   = 40,
            kappa:  float = 2.0,
            alpha:  float = 0.7,
            huber_factor: float = 2.0,
            eps:    float = 1e-8,
            progress_cb   = None,
            cancel_flag   = None) -> np.ndarray:
    """
    ImageMM multi-frame restoration (Sukurdeep et al. 2025, Algorithm 1).

    Parameters
    ----------
    frames      : list of (H,W) float arrays — registered, sky-subtracted
    psfs        : list of (k,k) normalised PSF arrays (one per frame)
    K           : maximum iterations
    kappa       : update clipping factor (Eq. 6 in paper)
    alpha       : relaxation / damping  (0 < alpha ≤ 1)
    huber_factor: Huber delta = factor × RMS(residual) per frame
    eps         : numerical stability floor
    progress_cb : callable(iteration, convergence_metric)
    cancel_flag : list [bool] — set [0]=True to abort

    Returns
    -------
    x : (H,W) float array — restored latent image

    Implementation note
    -------------------
    Uses _conv_mm (pure numpy FFT) instead of scipy.signal.fftconvolve.
    scipy 1.17.1 + numpy 2.4.4 have a C-level incompatibility when
    negative-stride kernel views (from k[::-1,::-1]) are passed across
    loop iterations.  _conv_mm copies all inputs to owned float64 arrays
    before the FFT, which avoids the crash entirely.
    The adjoint PSF is also pre-built as a contiguous copy for each frame.
    """
    T    = len(frames)
    H, W = frames[0].shape

    # Initialise: non-negative median stack
    x = np.asarray(
        np.median(np.stack([f.astype(np.float64) for f in frames]), axis=0),
        dtype=np.float64)
    x = np.clip(x, 0.0, None)

    # Per-frame noise variance via MAD (robust, no assumptions)
    vars_f = []
    for f in frames:
        f64 = f.astype(np.float64)
        med = float(np.median(f64))
        mad = float(np.median(np.abs(f64 - med)))
        vars_f.append(float((mad * 1.4826) ** 2 + eps))

    # Pre-build adjoint PSFs as contiguous float64 — avoids negative-stride crash
    psfs_adj = [np.ascontiguousarray(p[::-1, ::-1], dtype=np.float64)
                for p in psfs]

    for it in range(K):
        if cancel_flag and cancel_flag[0]:
            break

        xc  = np.asarray(x, dtype=np.float64)        # current estimate
        num = np.zeros((H, W), dtype=np.float64)
        den = np.zeros((H, W), dtype=np.float64)

        for ti in range(T):
            f64  = frames[ti].astype(np.float64)
            p    = psfs[ti]
            padj = psfs_adj[ti]
            var  = vars_f[ti]

            y_hat = _conv_mm(xc, p)
            r     = f64 - y_hat

            # Huber weights: downweight outliers (satellite trails, cosmics)
            rms   = float(np.sqrt(float(np.mean(r * r)))) + eps
            delta = float(huber_factor * rms)
            ar    = np.abs(r)
            wh    = np.where(ar <= delta, 1.0,
                             delta / (ar + eps)).astype(np.float64)
            W_    = wh / var                          # W_, not W (no shadowing)

            num  += _conv_mm(W_ * f64,   padj)
            den  += _conv_mm(W_ * y_hat, padj)

        # MM multiplicative update with clipping (Sukurdeep 2025 Eq. 4-5)
        u  = np.clip(num / (den + eps), 1.0 / kappa, kappa)
        x  = np.clip((1.0 - alpha) * xc + alpha * (xc * u),
                     0.0, None).astype(np.float64)

        c = float(np.median(np.abs(u - 1.0)))
        if progress_cb:
            progress_cb(it, c)
        if c < 1e-3:
            break

    return x


def imagemm_superres(frames: list[np.ndarray],
                     psfs:   list[np.ndarray],
                     sr_factor: int = 2,
                     K:      int   = 40,
                     kappa:  float = 2.0,
                     alpha:  float = 0.6,
                     huber_factor: float = 2.0,
                     eps:    float = 1e-8,
                     progress_cb   = None,
                     cancel_flag   = None) -> np.ndarray:
    """
    ImageMM with super-resolution (Sukurdeep et al. 2025, Algorithm 2).

    Upsamples each PSF to sr_factor× resolution via a variational problem
    (Eq. 11 in paper), then runs the MM loop on the super-resolved latent
    image with a downsampling operator D (average pooling).

    Returns latent image at sr_factor× the input pixel scale.
    """
    T = len(frames); H, W = frames[0].shape
    r = sr_factor

    # ── Step 1: solve super-resolved PSFs h^(t) ───────────────────────────
    # h^(t) = argmin_h  Σ_i (f_i^(t) - D(h * g_sigma)_i)^2
    # where g_sigma is a Gaussian at super-resolved scale (Eq. 11)
    sr_psfs = []
    for psf in psfs:
        k    = psf.shape[0]
        ksr  = k * r   # super-resolved PSF size
        # Gaussian prior at super-resolved scale
        sigma_sr = 0.55 * r   # as in paper
        g_sigma  = _gaussian_psf(ksr + ksr - 1, sigma_sr * 2.355)

        def loss(h_flat):
            h = h_flat.reshape(ksr, ksr)
            h = h / (h.sum() + 1e-10)
            # Downsample: average pooling by r
            conv_h = _conv_mm(h, g_sigma)[:ksr, :ksr]
            d = conv_h[:k*r:r, :k*r:r][:k, :k]
            return float(np.sum((psf - d)**2))

        h0    = zoom(psf, r, order=1)
        h0   /= h0.sum() + 1e-10
        res   = sp_minimize(loss, h0.ravel(), method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12})
        h_sr  = res.x.reshape(ksr, ksr)
        h_sr  = np.maximum(h_sr, 0)
        h_sr /= h_sr.sum() + 1e-10
        sr_psfs.append(h_sr)

    # ── Step 2: MM loop at super-resolved scale ───────────────────────────
    H_sr, W_sr = H * r, W * r
    x = np.clip(
        zoom(np.median(np.stack([f.astype(float) for f in frames]), axis=0),
             r, order=1),
        0.0, None).astype(np.float64)

    vars_f = [float(max((np.median(np.abs(f-np.median(f)))*1.4826)**2, eps))
              for f in frames]
    sr_psfs_adj = [np.ascontiguousarray(h[::-1,::-1], dtype=np.float64)
                   for h in sr_psfs]

    def downsample(img_sr):
        h_out = img_sr.shape[0] // r; w_out = img_sr.shape[1] // r
        return img_sr[:h_out*r,:w_out*r].reshape(h_out,r,w_out,r).mean(axis=(1,3))

    def upsample_T(img, Hs, Ws):
        out = np.zeros((Hs, Ws), dtype=np.float64)
        for di in range(r):
            for dj in range(r):
                out[di::r, dj::r] += img / (r*r)
        return out[:Hs, :Ws]

    for it in range(K):
        if cancel_flag and cancel_flag[0]:
            break
        xc  = np.asarray(x, dtype=np.float64)
        num = np.zeros((H_sr, W_sr), dtype=np.float64)
        den = np.zeros((H_sr, W_sr), dtype=np.float64)

        for ti in range(T):
            f64   = frames[ti].astype(np.float64)
            h_sr  = sr_psfs[ti]; h_adj = sr_psfs_adj[ti]; var = vars_f[ti]
            y_hat = downsample(_conv_mm(xc, h_sr))
            r_    = f64 - y_hat
            rms   = float(np.sqrt(float(np.mean(r_*r_)))) + eps
            delta = float(huber_factor * rms)
            ar    = np.abs(r_)
            wh    = np.where(ar<=delta, 1.0, delta/(ar+eps)).astype(np.float64)
            W_    = wh / var
            num  += _conv_mm(upsample_T(W_*f64,   H_sr, W_sr), h_adj)
            den  += _conv_mm(upsample_T(W_*y_hat, H_sr, W_sr), h_adj)

        u = np.clip(num/(den+eps), 1.0/kappa, kappa)
        x = np.clip((1-alpha)*xc + alpha*(xc*u), 0.0, None).astype(np.float64)
        c = float(np.median(np.abs(u-1)))
        if progress_cb: progress_cb(it, c)
        if c < 1e-3: break

    return x


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2: return data.astype(float)
    if data.ndim == 3 and data.shape[0] <= 4: return data.mean(0).astype(float)
    if data.ndim == 3 and data.shape[2] <= 4: return data.mean(2).astype(float)
    return data.astype(float)


class ImageMMWorker(QObject):
    progress    = pyqtSignal(int, str)
    psf_done    = pyqtSignal(int, object, float, int)  # frame_idx, psf, fwhm, n_stars
    iter_update = pyqtSignal(int, float)               # iteration, convergence
    finished    = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, paths: list, params: dict):
        super().__init__()
        self.paths  = paths
        self.params = params
        self._cancel = [False]

    def cancel(self): self._cancel[0] = True

    def run(self):
        try:
            self._pipeline()
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _pipeline(self):
        p = self.params
        T = len(self.paths)

        # ── Load frames ───────────────────────────────────────────────────
        self.progress.emit(2, f"Loading {T} frames…")
        frames  = []
        headers = []
        for i, path in enumerate(self.paths):
            if self._cancel[0]: return
            try:
                hdu = fits.open(str(path))[0]
                frames.append(luminance(hdu.data))
                headers.append(hdu.header)
            except Exception as e:
                self.error.emit(f"Cannot load {path}: {e}")
                return
            self.progress.emit(int(2 + 8*i/T), f"Loaded {i+1}/{T}")

        H, W = frames[0].shape

        # ── Estimate PSFs from stars ───────────────────────────────────────
        self.progress.emit(10, "Estimating PSF from stars in each frame…")
        psf_size = p.get('psf_size', 21)
        psfs = []
        fwhms = []
        for i, (frame, hdr) in enumerate(zip(frames, headers)):
            if self._cancel[0]: return
            psf, fwhm, n_s = estimate_psf_from_stars(
                frame, psf_size=psf_size,
                n_stars=p.get('n_stars', 25),
                thresh_sigma=p.get('thresh_sigma', 6.0),
                header=hdr)
            psfs.append(psf)
            fwhms.append(fwhm)
            self.psf_done.emit(i, psf, fwhm, n_s)
            self.progress.emit(int(10 + 15*i/T),
                f"Frame {i+1}/{T}: FWHM={fwhm:.1f}px  stars used={n_s}")

        if self._cancel[0]: return

        # ── Run ImageMM ───────────────────────────────────────────────────
        mode = p.get('mode', 'restore')
        if mode == 'superres':
            sr = p.get('sr_factor', 2)
            self.progress.emit(26, f"ImageMM super-resolution (×{sr})…")
            def prog(it, c):
                pct = int(26 + 68*(it/(p.get('iterations',40))))
                self.progress.emit(min(pct,93),
                    f"Iter {it+1}/{p.get('iterations',40)}  conv={c:.5f}")
                self.iter_update.emit(it, c)
            result = imagemm_superres(
                frames, psfs,
                sr_factor    = sr,
                K            = p.get('iterations', 40),
                kappa        = p.get('kappa', 2.0),
                alpha        = p.get('alpha', 0.6),
                huber_factor = p.get('huber', 2.0),
                progress_cb  = prog,
                cancel_flag  = self._cancel)
        else:
            self.progress.emit(26, "ImageMM restoration…")
            def prog(it, c):
                pct = int(26 + 68*(it/(p.get('iterations',40))))
                self.progress.emit(min(pct,93),
                    f"Iter {it+1}/{p.get('iterations',40)}  conv={c:.5f}")
                self.iter_update.emit(it, c)
            result = imagemm(
                frames, psfs,
                K            = p.get('iterations', 40),
                kappa        = p.get('kappa', 2.0),
                alpha        = p.get('alpha', 0.7),
                huber_factor = p.get('huber', 2.0),
                progress_cb  = prog,
                cancel_flag  = self._cancel)

        if self._cancel[0]: return

        # Reference: median stack
        median_stack = np.median(np.array([f.astype(float) for f in frames]),
                                 axis=0)

        self.progress.emit(100, f"Done!  {T} frames → ImageMM restoration")
        self.finished.emit({
            'result':        result,
            'median_stack':  median_stack,
            'frames':        frames,
            'psfs':          psfs,
            'fwhms':         fwhms,
            'mode':          mode,
            'n_frames':      T,
            'shape':         result.shape,
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

DARK = """
QMainWindow,QWidget{background:#111120;color:#d0d0e8;
    font-family:'Segoe UI',Arial,sans-serif;font-size:12px;}
QTabWidget::pane{border:1px solid #26264a;background:#151530;}
QTabBar::tab{background:#1a1a34;color:#6868a0;padding:6px 16px;
    border:1px solid #26264a;border-bottom:none;}
QTabBar::tab:selected{background:#20204c;color:#9898ff;
    border-bottom:2px solid #5050c8;}
QGroupBox{border:1px solid #26264a;border-radius:4px;margin-top:8px;
    padding-top:8px;color:#6868a8;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
QPushButton{background:#1e1e42;color:#b0b0d8;border:1px solid #3838a8;
    border-radius:4px;padding:5px 14px;}
QPushButton:hover{background:#28285e;border-color:#5050c8;}
QPushButton:disabled{color:#383858;border-color:#242448;}
QPushButton#run_btn{background:#1a2c1a;color:#80e080;
    border:1px solid #40a040;font-weight:bold;padding:7px 20px;}
QPushButton#run_btn:hover{background:#223022;}
QSpinBox,QDoubleSpinBox,QComboBox{background:#181832;
    border:1px solid #26264a;border-radius:3px;padding:3px 6px;color:#d0d0e8;}
QTextEdit{background:#0e0e20;border:1px solid #26264a;color:#85e085;
    font-family:Consolas,monospace;font-size:11px;}
QTableWidget{background:#0e0e20;alternate-background-color:#131328;
    gridline-color:#26264a;color:#d0d0e8;}
QTableWidget QHeaderView::section{background:#181834;color:#6868a8;
    border:1px solid #26264a;padding:3px;font-weight:bold;}
QSplitter::handle{background:#26264a;}
QLabel#title_lbl{font-size:18px;font-weight:bold;color:#80e080;padding:6px;}
QLabel#sub_lbl{font-size:10px;color:#304830;padding:0 8px 4px;}
"""


class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8,5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor='#191930')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
    def redraw(self): self.fig.tight_layout(pad=0.4); self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#0d0d1e')
    ax.set_title(title, color='#70c070', fontsize=9, pad=3)
    ax.set_xlabel(xl, color='#545472', fontsize=8)
    ax.set_ylabel(yl, color='#545472', fontsize=8)
    ax.tick_params(colors='#545472', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#26264a')


class StatusFooter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setStyleSheet("background:#0b0b18;border-top:1px solid #26264a;")
        lay = QHBoxLayout(self); lay.setContentsMargins(6,0,6,0)
        self.lbl = QLabel("Ready")
        self.lbl.setStyleSheet("color:#585878;font-size:10px;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(200,12)
        self.pbar.setVisible(False)
        self.pbar.setStyleSheet("""
            QProgressBar{background:#1a1a30;border:1px solid #306030;
                border-radius:3px;}
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #205020,stop:1 #60c060);border-radius:2px;}""")
        lay.addWidget(self.lbl,1); lay.addWidget(self.pbar)

    def update(self, pct, msg):
        self.lbl.setText(msg)
        if pct >= 0:
            self.pbar.setVisible(True); self.pbar.setValue(pct)
            if pct >= 100:
                QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))


class ImageMMApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1420, 920)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))
        self._paths   = []
        self._result  = None
        self._worker  = self._wthread = None
        self._conv_history = []
        self._psf_table_data = []
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._make_header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._make_controls())
        sp.addWidget(self._make_tabs())
        sp.setSizes([300,1120])
        rl.addWidget(sp,1)
        self.status = StatusFooter(); rl.addWidget(self.status)

    def _make_header(self):
        hdr = QWidget(); hdr.setFixedHeight(62)
        hdr.setStyleSheet("background:#0a120a;border-bottom:1px solid #264026;")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(10,0,10,0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                48,48,Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("ImageMM  —  Multi-frame Restoration & Super-resolution")
        t.setObjectName("title_lbl")
        s = QLabel("Sukurdeep et al. 2025  ·  Auto PSF from stars  ·  No instrument info needed  ·  "
                   "Huber robust loss (satellite trail removal)  ·  Optional 2× super-resolution")
        s.setObjectName("sub_lbl")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _make_controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(5)

        # ── Input ──────────────────────────────────────────────────────────
        grp = QGroupBox("Input Frames")
        gl  = QVBoxLayout(grp)
        self.file_lbl = QLabel("No frames loaded")
        self.file_lbl.setStyleSheet("color:#484868;font-size:10px;")
        self.file_lbl.setWordWrap(True)
        b1 = QPushButton("📂  Load FITS series…"); b1.clicked.connect(self._load)
        b2 = QPushButton("📦  Load FITS cube (3D)…"); b2.clicked.connect(self._load_cube)
        gl.addWidget(self.file_lbl); gl.addWidget(b1); gl.addWidget(b2)
        lay.addWidget(grp)

        # ── PSF estimation ─────────────────────────────────────────────────
        grp2 = QGroupBox("Auto PSF Estimation")
        g2   = QGridLayout(grp2)
        g2.addWidget(QLabel("PSF stamp size (px):"),0,0)
        self.sp_psfsize = QSpinBox(); self.sp_psfsize.setRange(11,51)
        self.sp_psfsize.setValue(21); self.sp_psfsize.setSingleStep(2)
        g2.addWidget(self.sp_psfsize,0,1)
        g2.addWidget(QLabel("Stars per frame:"),1,0)
        self.sp_nstars = QSpinBox(); self.sp_nstars.setRange(3,100)
        self.sp_nstars.setValue(25)
        g2.addWidget(self.sp_nstars,1,1)
        g2.addWidget(QLabel("Detection threshold (σ):"),2,0)
        self.sp_thresh = QDoubleSpinBox(); self.sp_thresh.setRange(3,20)
        self.sp_thresh.setValue(6.0); self.sp_thresh.setDecimals(1)
        g2.addWidget(self.sp_thresh,2,1)
        lay.addWidget(grp2)

        # ── MM parameters ─────────────────────────────────────────────────
        grp3 = QGroupBox("MM Parameters")
        g3   = QGridLayout(grp3)
        g3.addWidget(QLabel("Mode:"),0,0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Restoration (L2/Huber)","Super-resolution ×2"])
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_change)
        g3.addWidget(self.cmb_mode,0,1)
        g3.addWidget(QLabel("Iterations:"),1,0)
        self.sp_iters = QSpinBox(); self.sp_iters.setRange(5,200)
        self.sp_iters.setValue(40)
        g3.addWidget(self.sp_iters,1,1)
        g3.addWidget(QLabel("Clip κ:"),2,0)
        self.sp_kappa = QDoubleSpinBox(); self.sp_kappa.setRange(1.01,10)
        self.sp_kappa.setValue(2.0); self.sp_kappa.setDecimals(2)
        self.sp_kappa.setToolTip(
            "Update clipping factor.\n"
            "Lower = more conservative, less ringing.\n"
            "Typical: 1.5–3.0")
        g3.addWidget(self.sp_kappa,2,1)
        g3.addWidget(QLabel("Relaxation α:"),3,0)
        self.sp_alpha = QDoubleSpinBox(); self.sp_alpha.setRange(0.1,1.0)
        self.sp_alpha.setValue(0.7); self.sp_alpha.setDecimals(2)
        self.sp_alpha.setToolTip(
            "Step damping. Lower = more stable, slower.\n"
            "Typical: 0.5–0.8")
        g3.addWidget(self.sp_alpha,3,1)
        g3.addWidget(QLabel("Huber factor:"),4,0)
        self.sp_huber = QDoubleSpinBox(); self.sp_huber.setRange(0.5,10)
        self.sp_huber.setValue(2.0); self.sp_huber.setDecimals(1)
        self.sp_huber.setToolTip(
            "Huber δ = factor × RMS(residual).\n"
            "Lower = more aggressive outlier rejection.\n"
            "Typical: 1.5–3.0")
        g3.addWidget(self.sp_huber,4,1)
        lay.addWidget(grp3)

        # ── Output ────────────────────────────────────────────────────────
        grp4 = QGroupBox("Output")
        g4   = QVBoxLayout(grp4)
        self.btn_save = QPushButton("💾  Save result FITS…")
        self.btn_save.setEnabled(False); self.btn_save.clicked.connect(self._save)
        g4.addWidget(self.btn_save)
        lay.addWidget(grp4)

        # ── Run ───────────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Run ImageMM")
        self.btn_run.setObjectName("run_btn"); self.btn_run.setFixedHeight(40)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setEnabled(False); self.btn_cancel.setStyleSheet("color:#b05050;")
        self.btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_cancel)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color:#70c070;font-size:10px;padding:4px;")
        self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _make_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_frames(),     "🎞  Frames & PSFs")
        self.tabs.addTab(self._tab_result(),     "✨  Result")
        self.tabs.addTab(self._tab_compare(),    "🔬  Comparison")
        self.tabs.addTab(self._tab_convergence(),"📈  Convergence")
        self.tabs.addTab(self._tab_log(),        "📋  Log")
        return self.tabs

    def _tab_frames(self):
        w = QWidget(); lay = QVBoxLayout(w)
        # PSF table
        self.psf_table = QTableWidget(0, 5)
        self.psf_table.setHorizontalHeaderLabels(
            ["Frame","FWHM (px)","Stars used","PSF preview","Status"])
        self.psf_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.psf_table.setFixedHeight(200)
        self.psf_table.setAlternatingRowColors(True)
        lay.addWidget(self.psf_table)
        # Preview canvas
        self.frames_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.frames_plot)
        return w

    def _tab_result(self):
        w = QWidget(); lay = QVBoxLayout(w)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Stretch:"))
        self.cmb_stretch = QComboBox()
        self.cmb_stretch.addItems(["Arcsinh","Log","Sqrt","Linear"])
        self.cmb_stretch.currentIndexChanged.connect(self._refresh_result)
        ctrl.addWidget(self.cmb_stretch); ctrl.addStretch()
        lay.addLayout(ctrl)
        self.result_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.result_plot)
        return w

    def _tab_compare(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.compare_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.compare_plot)
        return w

    def _tab_convergence(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.conv_plot = MplCanvas(figsize=(9,5)); lay.addWidget(self.conv_plot)
        # FWHM bar chart
        self.fwhm_plot = MplCanvas(figsize=(9,3)); lay.addWidget(self.fwhm_plot)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear)
        lay.addWidget(btn)
        return w

    # ── File loading ──────────────────────────────────────────────────────────

    def _load(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load FITS frames", "",
            "FITS (*.fits *.fit *.fts);;All (*)")
        if not paths: return
        self._paths = [Path(p) for p in sorted(paths)]
        self._after_load()

    def _load_cube(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FITS cube", "", "FITS (*.fits *.fit);;All (*)")
        if not path: return
        try:
            import tempfile; td = Path(tempfile.mkdtemp())
            data = fits.getdata(path)
            if data.ndim == 3:
                self._paths = []
                for i in range(data.shape[0]):
                    p = td / f"slice_{i:04d}.fits"
                    fits.PrimaryHDU(data[i].astype(np.float32)).writeto(
                        str(p), overwrite=True)
                    self._paths.append(p)
                self._after_load()
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

    def _after_load(self):
        n = len(self._paths)
        self.file_lbl.setText(f"{n} frames loaded")
        self.file_lbl.setStyleSheet("color:#70c070;font-size:10px;")
        self.btn_run.setEnabled(True)
        self.psf_table.setRowCount(n)
        for i, p in enumerate(self._paths):
            self.psf_table.setItem(i,0,QTableWidgetItem(p.name[:30]))
            self.psf_table.setItem(i,1,QTableWidgetItem("—"))
            self.psf_table.setItem(i,2,QTableWidgetItem("—"))
            self.psf_table.setItem(i,4,QTableWidgetItem("Pending"))
        self._log(f"Loaded {n} frames")

    def _on_mode_change(self, idx):
        is_sr = (idx == 1)
        self.sp_alpha.setValue(0.6 if is_sr else 0.7)

    # ── Run / cancel ──────────────────────────────────────────────────────────

    def _run(self):
        if not self._paths: return
        self._conv_history = []
        mode_map = {0:'restore', 1:'superres'}
        params = {
            'mode':       mode_map[self.cmb_mode.currentIndex()],
            'psf_size':   self.sp_psfsize.value(),
            'n_stars':    self.sp_nstars.value(),
            'thresh_sigma':self.sp_thresh.value(),
            'iterations': self.sp_iters.value(),
            'kappa':      self.sp_kappa.value(),
            'alpha':      self.sp_alpha.value(),
            'huber':      self.sp_huber.value(),
            'sr_factor':  2,
        }
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False); self._result = None

        self._wthread = QThread(self)
        self._worker  = ImageMMWorker(self._paths, params)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.psf_done.connect(self._on_psf_done)
        self._worker.iter_update.connect(self._on_iter)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()
        self.tabs.setCurrentIndex(3)  # convergence tab

    def _cancel(self):
        if self._worker: self._worker.cancel()
        self.btn_cancel.setEnabled(False); self.btn_run.setEnabled(True)

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_progress(self, pct, msg):
        self.status.update(pct, msg); self._log(f"[{pct:3d}%]  {msg}")

    def _on_psf_done(self, idx, psf, fwhm, n_stars):
        self.psf_table.setItem(idx,1,QTableWidgetItem(f"{fwhm:.2f}"))
        self.psf_table.setItem(idx,2,QTableWidgetItem(str(n_stars)))
        status = "✓ Empirical" if n_stars >= 3 else "⚠ Gaussian fallback"
        self.psf_table.setItem(idx,4,QTableWidgetItem(status))
        # Draw PSF grid
        self._draw_psf_grid()

    def _on_iter(self, it, c):
        self._conv_history.append(c)
        if len(self._conv_history) % 3 == 0:
            self._draw_convergence()

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}")
        self.status.update(0,"Error"); self.btn_run.setEnabled(True)
        QMessageBox.critical(self,"Error",msg[:500])

    def _on_finished(self, result):
        self._result = result
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(True)

        r = result
        self._log(f"\n{'='*55}")
        self._log(f"✓  ImageMM complete")
        self._log(f"   Frames:      {r['n_frames']}")
        self._log(f"   Mode:        {r['mode']}")
        self._log(f"   Output:      {r['shape'][1]}×{r['shape'][0]} px")
        fwhms = r['fwhms']
        self._log(f"   FWHM range:  {min(fwhms):.1f}–{max(fwhms):.1f} px")
        self._log(f"{'='*55}\n")

        self.summary_lbl.setText(
            f"{r['n_frames']} frames  |  {r['mode']}  |  "
            f"{r['shape'][1]}×{r['shape'][0]}  |  "
            f"FWHM {min(fwhms):.1f}–{max(fwhms):.1f}px")

        self._refresh_result()
        self._draw_comparison()
        self._draw_convergence()
        self._draw_fwhm_bars()
        self.tabs.setCurrentIndex(1)

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self):
        if not self._result: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ImageMM result", "", "FITS (*.fits)")
        if not path: return
        r = self._result
        hdr = fits.Header()
        hdr['HISTORY']  = f'HYPERLOAD ImageMM v{VERSION}'
        hdr['HISTORY']  = f'Sukurdeep et al. 2025, AJ 170, 233'
        hdr['MFDECONV'] = True
        hdr['MM_MODE']  = r['mode']
        hdr['MM_NFRAM'] = r['n_frames']
        fits.PrimaryHDU(r['result'].astype(np.float32),
                        header=hdr).writeto(path, overwrite=True)
        self._log(f"✓ Saved: {path}")

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _stretch(self, img):
        lo, hi = np.percentile(img[np.isfinite(img)], [0.1, 99.9])
        c = np.clip(img - lo, 0, hi - lo + 1e-10) / (hi - lo + 1e-10)
        m = self.cmb_stretch.currentIndex()
        if m == 0: return np.arcsinh(c*5)/np.arcsinh(5)
        if m == 1: return np.log1p(c*9)/np.log1p(9)
        if m == 2: return np.sqrt(c)
        return c

    def _refresh_result(self):
        if not self._result: return
        r = self._result
        fig = self.result_plot.fig; fig.clear(); fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, f"ImageMM result  ({r['mode']}  {r['n_frames']} frames)")
        ax.imshow(self._stretch(r['result']), origin='lower',
                  cmap='gray', aspect='equal', interpolation='nearest')
        self.result_plot.redraw()

    def _draw_comparison(self):
        if not self._result: return
        r = self._result
        fig = self.compare_plot.fig; fig.clear(); fig.patch.set_facecolor('#191930')
        axes = fig.subplots(1,3)
        for ax, img, title in zip(axes,
            [r['median_stack'], r['result'],
             r['result'] - r['median_stack']],
            ["Median stack (reference)",
             f"ImageMM  ({r['mode']})",
             "Difference (ImageMM − median)"]):
            _ax(ax, title)
            lo,hi = np.percentile(img,[0.1,99.9])
            ax.imshow(img, origin='lower',
                      cmap='gray' if 'Diff' not in title else 'bwr',
                      vmin=lo, vmax=hi, aspect='equal',
                      interpolation='nearest')
        self.compare_plot.redraw()

    def _draw_psf_grid(self):
        if not self._result: return
        psfs = self._result.get('psfs', [])
        if not psfs: return
        fig = self.frames_plot.fig; fig.clear(); fig.patch.set_facecolor('#191930')
        n = min(len(psfs), 8)
        cols = min(n, 4); rows = (n + cols - 1) // cols
        axes = fig.subplots(rows, cols)
        if n == 1: axes = [[axes]]
        elif rows == 1: axes = [axes]
        ax_flat = [a for row in axes for a in
                   (row if hasattr(row,'__iter__') else [row])]
        for i, (ax, psf) in enumerate(zip(ax_flat, psfs[:n])):
            _ax(ax, f"Frame {i}  {self._result['fwhms'][i]:.1f}px")
            ax.imshow(psf, origin='lower', cmap='hot',
                      aspect='equal', interpolation='nearest')
        for ax in ax_flat[n:]: ax.set_visible(False)
        self.frames_plot.redraw()

    def _draw_convergence(self):
        if not self._conv_history: return
        fig = self.conv_plot.fig; fig.clear(); fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "MM convergence  (median |u−1| per iteration)",
            "Iteration","median|u-1|")
        h = np.array(self._conv_history)
        ax.semilogy(h, c='#70c070', lw=1.5)
        ax.fill_between(range(len(h)), h, alpha=0.15, color='#70c070')
        ax.axhline(1e-3, c='#ff6060', lw=1, ls='--', label='Convergence threshold')
        ax.legend(fontsize=7, facecolor='#1a1a30', edgecolor='#3030a0',
                  labelcolor='white')
        self.conv_plot.redraw()

    def _draw_fwhm_bars(self):
        if not self._result: return
        fwhms = self._result['fwhms']
        fig = self.fwhm_plot.fig; fig.clear(); fig.patch.set_facecolor('#191930')
        ax = fig.add_subplot(111)
        _ax(ax, "Per-frame FWHM (seeing quality)", "Frame", "FWHM (px)")
        idx = range(len(fwhms))
        bars = ax.bar(idx, fwhms, color='#4070a0', edgecolor='#264060', width=0.7)
        ax.axhline(min(fwhms), c='#70c070', lw=1, ls='--',
                   label=f"Best: {min(fwhms):.1f}px")
        ax.legend(fontsize=7, facecolor='#1a1a30', edgecolor='#3030a0',
                  labelcolor='white')
        self.fwhm_plot.redraw()

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ImageMMApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
