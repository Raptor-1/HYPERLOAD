r"""
HYPERLOAD — DIPLI  (Deep Image Prior Lucky Imaging)
=====================================================
Blind astronomical image restoration from 7–13 frames.
Place in: C:\Users\Marcell\Desktop\Siril Suites\

Script #26 in the original HYPERLOAD project plan.
Mentioned as downstream target in lucky_preprocessor.py (#50) and
thresher_stack.py (#14). Part of lucky_suite.py (B6).

Reference
---------
Singh et al. 2025, "DIPLI: Deep Image Prior Lucky Imaging for Blind
Astronomical Image Restoration", arXiv:2503.15984v3

Algorithm (Algorithm 2 in paper)
---------------------------------
1. Pivot selection: x_pivot = argmax_k Laplacian_energy(x_k)
2. Optical flow: ω_k = TVNet(x_pivot, x_k) for all k  [TV-L1 here]
3. Confidence maps: c_k(p) = exp(-α ‖∇ω_k(p)‖_F)   (flow smoothness proxy)
4. Fixed latent:    z ~ N(0,I),  reused throughout
5. SGLD loop (N iterations, mini-batch K_b=min(K,4)):
     z_n ~ N(0,σ²_z I)
     θ_n = θ_{n-1} - λ ∇L_w(X_{batch}, G_θ(z+z_n); Ω) + ξ_n
     ξ_n ~ N(0,σ²_ξ I)   ← SGLD noise = Bayesian regularisation
6. MC average after warm-up n₀:
     y* = (1/(N-n₀)) Σ_{n>n₀} G_θn(z+z_n)

Loss (Eq. 12): L_w = Σ_k ‖ c_k ⊙ (d∘h∘ω_k(y*) − x_k) ‖²₂
  d = avg-pool downsample by scale
  h = optional Gaussian PSF blur
  ω_k = differentiable bilinear warp from optical flow

Network: 4-stage U-Net, InstanceNorm, skip connections, Dropout.
  Output: super-resolved HR image (scale × LQ resolution) via sigmoid.

Key results from paper (synthetic benchmarks):
  LPIPS best in 12/12 scenes  (perceptual fidelity)
  DISTS best in 10/12 scenes
  Only 7–13 frames needed  (vs thousands for Lucky Imaging)

Pipeline connections
--------------------
  lucky_preprocessor.py  →  DIPLI  (ranked FITS frames)
  planet_derotation.py   →  DIPLI  (derotated planetary frames)
  bispectrum_speckle.py  ←→ DIPLI  (both process lucky frames; compare results)
  lucky_suite.py         ←  DIPLI  (component of B6 suite)

Version: 1.0.0
Project: HYPERLOAD
"""

import sys, os, traceback, math, glob
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Crash handler ──────────────────────────────────────────────────────────────
def _crash(et, ev, eb):
    log = Path(__file__).parent / "crash_log.txt"
    with open(log, "a") as f:
        f.write(f"\n{'='*60}\n{datetime.now()}\ndipli.py\n")
        traceback.print_exception(et, ev, eb, file=f)
    sys.__excepthook__(et, ev, eb)
sys.excepthook = _crash

try:
    import sirilpy as s
    for p in ["PyQt6","torch","scipy","matplotlib","scikit-image","astropy"]:
        s.ensure_installed(p)
except ImportError:
    pass

# ── Science / imaging imports ──────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import laplace, gaussian_filter, map_coordinates

try:
    from skimage.registration import optical_flow_tvl1
    HAS_TVL1 = True
except ImportError:
    HAS_TVL1 = False

try:
    from astropy.io import fits as astropy_fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_OK = False
    DEVICE = None

# ── Qt imports ────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QSplitter, QMessageBox, QSizePolicy,
    QScrollArea, QFormLayout, QSlider, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QPixmap, QIcon, QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

VERSION   = "1.0.0"
APP_TITLE = "HYPERLOAD — DIPLI  (Deep Image Prior Lucky Imaging)"


# ═══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORK  (exact architecture from Singh et al. 2025)
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_OK:
    class _Block(nn.Module):
        """Two-conv block: Conv → InstanceNorm → ReLU → Dropout → Conv → IN → ReLU"""
        def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class DIPLIUNet(nn.Module):
        """
        4-stage U-Net with symmetric skip connections.
        Architecture per Singh et al. 2025 DIPLI paper.

        Parameters
        ----------
        n_ch  : input/output channels (1=mono, 3=RGB)
        scale : super-resolution factor {2,4,8}
        feat  : base feature channels (paper: 128; CPU-friendly: 32 or 64)
        """
        def __init__(self, n_ch: int = 1, scale: int = 4, feat: int = 32):
            super().__init__()
            f = feat
            # Encoder
            self.e1   = _Block(n_ch, f)
            self.e2   = _Block(f,    f*2)
            self.e3   = _Block(f*2,  f*4)
            self.e4   = _Block(f*4,  f*4)
            self.bot  = _Block(f*4,  f*4)
            self.pool = nn.AvgPool2d(2)
            # Decoder upsampling + skip-cat blocks
            self.u4 = self._up(f*4, f*4); self.d4 = _Block(f*8, f*4)
            self.u3 = self._up(f*4, f*4); self.d3 = _Block(f*8, f*2)
            self.u2 = self._up(f*2, f*2); self.d2 = _Block(f*4, f)
            self.u1 = self._up(f,   f);   self.d1 = _Block(f*2, f)
            # Output head: upsample by scale + sigmoid
            self.head = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
                nn.Conv2d(f, max(f//2, 8), 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(max(f//2, 8), n_ch, 3, padding=1), nn.Sigmoid(),
            )

        @staticmethod
        def _up(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 1))

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            b  = self.bot(self.pool(e4))
            d4 = self.d4(torch.cat([self.u4(b),  e4], 1))
            d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.head(d1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ALGORITHM  (pure-function layer, importable without Qt)
# ═══════════════════════════════════════════════════════════════════════════════

def luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2: return data.astype(float)
    if data.ndim == 3 and data.shape[0] <= 4:
        return (0.299*data[0]+0.587*data[1]+0.114*data[2]).astype(float)
    if data.ndim == 3:
        return (0.299*data[:,:,0]+0.587*data[:,:,1]+0.114*data[:,:,2]).astype(float)
    return data.astype(float)


def laplacian_energy(img: np.ndarray) -> float:
    """Quality metric q(x_k): variance of Laplacian. Used for pivot selection."""
    return float(np.var(laplace(img.astype(float))))


def select_pivot(frames: list[np.ndarray]) -> int:
    """Return index of the sharpest frame (highest Laplacian energy)."""
    return int(np.argmax([laplacian_energy(f) for f in frames]))


def estimate_optical_flow(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """
    Estimate dense optical flow from ref → mov.
    Uses TV-L1 (skimage) if available, else phase-correlation fallback.
    Returns (2, H, W): [flow_y, flow_x] in pixels.
    """
    ref_n = (ref - ref.min()) / (np.ptp(ref) + 1e-10)
    mov_n = (mov - mov.min()) / (np.ptp(mov) + 1e-10)
    if HAS_TVL1:
        v, u = optical_flow_tvl1(ref_n, mov_n, num_warp=3, num_iter=10,
                                  attachment=15.0)
        return np.stack([v, u], axis=0)
    # Fallback: global phase shift
    H, W = ref.shape
    cc   = np.real(np.fft.ifft2(np.fft.fft2(ref_n)*np.conj(np.fft.fft2(mov_n))))
    idx  = np.unravel_index(cc.argmax(), cc.shape)
    dy   = idx[0] if idx[0] < H//2 else idx[0]-H
    dx   = idx[1] if idx[1] < W//2 else idx[1]-W
    return np.array([np.full((H,W), float(dy)), np.full((H,W), float(dx))])


def confidence_map(flow: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Flow confidence map c_k(p) = exp(-α ‖∇ω_k(p)‖_F)   (Eq. 11 in paper).
    Regions with rapidly varying flow (likely estimation errors) get low weight.
    """
    fy, fx = flow[0], flow[1]
    dfy_dy, dfy_dx = np.gradient(fy)
    dfx_dy, dfx_dx = np.gradient(fx)
    frob = np.sqrt(dfy_dy**2 + dfy_dx**2 + dfx_dy**2 + dfx_dx**2)
    return np.exp(-alpha * frob).astype(np.float32)


def _warp_downsample(img_hr: "torch.Tensor",
                     flow_lq: "torch.Tensor",
                     scale: int,
                     psf_sigma: float = 0.0) -> "torch.Tensor":
    """
    Differentiable forward degradation model: warp → optional PSF → downsample.

    img_hr   : (B, C, H*scale, W*scale)  — current HR prediction from U-Net
    flow_lq  : (B, 2, H, W)              — TV-L1 flow in LQ pixel coords
    scale    : downsampling factor
    psf_sigma: Gaussian PSF sigma (0 = skip)
    """
    B, C, Hhr, Whr = img_hr.shape
    # Upsample LQ flow to HR resolution and scale to HR pixel coords
    flow_hr = F.interpolate(flow_lq * scale, size=(Hhr, Whr),
                             mode='bilinear', align_corners=False)
    # Build sampling grid normalised to [-1, 1]
    base_y = torch.linspace(-1, 1, Hhr, device=img_hr.device)
    base_x = torch.linspace(-1, 1, Whr, device=img_hr.device)
    gy, gx = torch.meshgrid(base_y, base_x, indexing='ij')
    grid   = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B,-1,-1,-1)
    dy     = flow_hr[:,0:1] * (2.0 / Hhr)
    dx     = flow_hr[:,1:2] * (2.0 / Whr)
    grid   = grid + torch.stack([dx.squeeze(1), dy.squeeze(1)], dim=-1)
    warped = F.grid_sample(img_hr, grid, mode='bilinear',
                           align_corners=True, padding_mode='reflection')
    # Optional PSF convolution (Gaussian approximation)
    if psf_sigma > 0.5:
        k = max(3, int(psf_sigma*4+1)|1)
        coords = torch.arange(k, dtype=torch.float32, device=img_hr.device) - k//2
        g = torch.exp(-0.5*(coords/psf_sigma)**2)
        g /= g.sum()
        kx = g.view(1,1,1,-1).expand(C,1,-1,-1)
        ky = g.view(1,1,-1,1).expand(C,1,-1,-1)
        pad = k//2
        warped = F.conv2d(warped, kx, padding=(0,pad), groups=C)
        warped = F.conv2d(warped, ky, padding=(pad,0), groups=C)
    # Average-pool to LQ resolution
    return F.avg_pool2d(warped, scale)


def dipli(frames: list[np.ndarray],
          scale: int = 4,
          K: int = 11,
          N: int = 2000,
          n0_frac: float = 0.9,
          lr: float = 0.0025,
          sigma_xi: float = 0.0025,
          sigma_z: float = 0.01,
          alpha_conf: float = 0.10,
          psf_sigma: float = 0.0,
          feat: int = 32,
          n_ch: int = 1,
          progress_cb=None,
          cancel_flag: list = None) -> dict:
    """
    DIPLI image restoration (Singh et al. 2025).

    Parameters
    ----------
    frames     : list of (H,W) float arrays — LQ frames (7–13 optimal)
    scale      : super-resolution factor {2,4,8}
    K          : number of frames to use (paper: 11)
    N          : total SGLD iterations (paper: 6500; CPU default: 2000)
    n0_frac    : fraction of N used as warm-up (paper: 6000/6500 ≈ 0.92)
    lr         : SGLD learning rate (paper: 0.0025)
    sigma_xi   : SGLD noise std (paper: 0.0025, equal to lr)
    sigma_z    : latent perturbation std (paper: 0.01)
    alpha_conf : confidence map decay (paper ablation: 0.1)
    psf_sigma  : optional Gaussian PSF sigma in pixels (0 = skip)
    feat       : U-Net base features (paper: 128; CPU: 32)
    n_ch       : channels (1=mono, 3=RGB)
    progress_cb: callable(iter, total, loss, preview_np)
    cancel_flag: [False] — set [0]=True to abort

    Returns
    -------
    dict: y_star, pivot_idx, energies, flows (numpy arrays)
    """
    if not TORCH_OK:
        return {'error': 'PyTorch not installed. Run: pip install torch'}

    # Select K best frames
    energies = [laplacian_energy(f) for f in frames]
    order    = np.argsort(energies)[::-1]
    frames_k = [frames[i] for i in order[:K]]

    pivot_idx_k = 0  # first in sorted order = sharpest
    pivot       = frames_k[pivot_idx_k]
    H, W        = pivot.shape[-2], pivot.shape[-1]

    # Optical flows and confidence maps
    if progress_cb: progress_cb(0, N, 0.0, None)
    flows_np = [estimate_optical_flow(pivot, f) for f in frames_k]
    confs_np = [confidence_map(fl, alpha_conf) for fl in flows_np]

    # Torch tensors (LQ)
    frames_t = [torch.from_numpy(f.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                for f in frames_k]
    flows_t  = [torch.from_numpy(fl.astype(np.float32)).unsqueeze(0).to(DEVICE)
                for fl in flows_np]
    confs_t  = [torch.from_numpy(c.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                for c in confs_np]

    # Network and fixed latent code
    net = DIPLIUNet(n_ch=n_ch, scale=scale, feat=feat).to(DEVICE).train()
    z   = torch.randn(1, n_ch, H, W, device=DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    n0  = int(N * n0_frac)
    Kb  = min(K, 4)  # mini-batch size per step (paper default)

    # MC accumulator
    mc_sum   = torch.zeros(1, n_ch, H*scale, W*scale, device=DEVICE)
    mc_count = 0
    loss_history = []

    rng = np.random.default_rng()

    for n in range(1, N+1):
        if cancel_flag and cancel_flag[0]: break

        opt.zero_grad()

        # Latent perturbation
        z_n  = torch.randn_like(z) * sigma_z
        y_hr = net(z + z_n)  # (1, n_ch, H*scale, W*scale)

        # Mini-batch of frames
        batch_idx = rng.choice(K, Kb, replace=False)
        loss = torch.zeros(1, device=DEVICE)
        for i in batch_idx:
            y_lq = _warp_downsample(y_hr, flows_t[i], scale, psf_sigma)
            r    = confs_t[i] * (y_lq - frames_t[i])
            loss = loss + (r**2).mean()
        loss = loss / Kb
        loss.backward()

        # SGLD: inject Gaussian noise into gradients before step
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p) * sigma_xi)
        opt.step()

        # MC averaging after warm-up
        if n > n0:
            with torch.no_grad():
                mc_sum  += net(z + torch.randn_like(z)*sigma_z)
                mc_count += 1

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if progress_cb and (n % 20 == 0 or n == N):
            preview = None
            if mc_count > 0:
                with torch.no_grad():
                    prev = (mc_sum / mc_count).squeeze().cpu().numpy()
                    preview = np.clip(prev, 0, 1)
            progress_cb(n, N, loss_val, preview)

    # Final MC average
    if mc_count > 0:
        y_star = (mc_sum / mc_count).squeeze(0).cpu().numpy()
    else:
        with torch.no_grad():
            y_star = net(z).squeeze(0).cpu().numpy()
    y_star = np.clip(y_star, 0, 1)
    if y_star.ndim == 3 and y_star.shape[0] == 1:
        y_star = y_star[0]

    return {
        'y_star':       y_star,
        'pivot':        pivot,
        'pivot_global': int(order[pivot_idx_k]),
        'energies':     energies,
        'frames_used':  [int(i) for i in order[:K]],
        'mc_count':     mc_count,
        'loss_history': loss_history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class DIPLIWorker(QObject):
    progress = pyqtSignal(int, int, float, object)   # iter, total, loss, preview
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, frames, params):
        super().__init__()
        self.frames = frames; self.params = params
        self._cancel = [False]

    def cancel(self): self._cancel[0] = True

    def run(self):
        try:
            p = self.params
            result = dipli(
                frames     = self.frames,
                scale      = p['scale'],
                K          = p['K'],
                N          = p['N'],
                n0_frac    = p['n0_frac'],
                lr         = p['lr'],
                sigma_xi   = p['sigma_xi'],
                sigma_z    = p['sigma_z'],
                alpha_conf = p['alpha_conf'],
                psf_sigma  = p['psf_sigma'],
                feat       = p['feat'],
                progress_cb= lambda it,tot,l,prev: self.progress.emit(it,tot,l,prev),
                cancel_flag= self._cancel,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

BG  = "#0a0e18"; BG2 = "#101420"; BG3 = "#161c2c"; BOR = "#1a2240"
ACC = "#5b8aff"; AC2 = "#2d4fb0"; TXT = "#c8d4f0"; DIM = "#2a3554"
OK  = "#48b080"; WRN = "#d89030"; ERR = "#c04040"; SEC = "#6090e0"

DARK = f"""
QMainWindow,QWidget{{background:{BG};color:{TXT};
    font-family:'Segoe UI',Arial,sans-serif;font-size:11px;}}
QTabWidget::pane{{border:1px solid {BOR};background:{BG2};}}
QTabBar::tab{{background:{BG3};color:{DIM};padding:6px 16px;
    border:1px solid {BOR};border-bottom:none;}}
QTabBar::tab:selected{{background:{BG2};color:{ACC};
    border-bottom:2px solid {ACC};}}
QGroupBox{{border:1px solid {BOR};border-radius:4px;margin-top:8px;
    padding-top:8px;color:{DIM};font-weight:bold;}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}
QPushButton{{background:{BG3};color:{TXT};border:1px solid {BOR};
    border-radius:4px;padding:5px 14px;}}
QPushButton:hover{{background:{AC2};border-color:{ACC};color:white;}}
QPushButton#run{{background:{AC2};border-color:{ACC};
    color:white;font-weight:bold;padding:7px 20px;}}
QPushButton#run:hover{{background:{ACC};}}
QPushButton:disabled{{color:{DIM};border-color:{BG3};}}
QSpinBox,QDoubleSpinBox,QComboBox{{background:{BG3};
    border:1px solid {BOR};border-radius:3px;padding:3px 6px;color:{TXT};}}
QTextEdit{{background:#060810;border:1px solid {BOR};color:{OK};
    font-family:Consolas,monospace;font-size:10px;}}
QProgressBar{{background:{BG3};border:1px solid {BOR};
    border-radius:3px;height:8px;}}
QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {AC2},stop:1 {ACC});border-radius:2px;}}
QSplitter::handle{{background:{BOR};}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS
# ═══════════════════════════════════════════════════════════════════════════════

class MplCanvas(QWidget):
    def __init__(self, parent=None, figsize=(8, 4)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, facecolor=BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def redraw(self):
        self.fig.tight_layout(pad=0.3)
        self.canvas.draw_idle()


def _ax(ax, title='', xl='', yl=''):
    ax.set_facecolor('#060810')
    ax.set_title(title, color=SEC, fontsize=9, pad=3)
    ax.set_xlabel(xl, color=DIM, fontsize=8)
    ax.set_ylabel(yl, color=DIM, fontsize=8)
    ax.tick_params(colors=DIM, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(BOR)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class DIPLIApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1440, 920)
        self.setStyleSheet(DARK)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists(): self.setWindowIcon(QIcon(str(_logo)))

        self._frames  = []
        self._result  = None
        self._worker  = self._wthread = None
        self._loss_history = []
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)
        rl.addWidget(self._header())
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._controls())
        sp.addWidget(self._tabs())
        sp.setSizes([280, 1160])
        rl.addWidget(sp, 1)
        rl.addWidget(self._footer())

    def _header(self):
        hdr = QWidget(); hdr.setFixedHeight(60)
        hdr.setStyleSheet(f"background:{BG};border-bottom:1px solid {BOR};")
        lay = QHBoxLayout(hdr); lay.setContentsMargins(12,0,12,0)
        _logo = Path(__file__).parent / "logo.png"
        if _logo.exists():
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(_logo)).scaled(
                44,44,Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lay.addWidget(lbl)
        v = QVBoxLayout(); v.setSpacing(0)
        t = QLabel("DIPLI — Deep Image Prior Lucky Imaging")
        t.setStyleSheet(f"color:{ACC};font-size:17px;font-weight:bold;padding:6px;")
        torch_str = f"PyTorch {torch.__version__}  |  {DEVICE}" if TORCH_OK else "⚠  PyTorch not found"
        flow_str  = "TV-L1 flow" if HAS_TVL1 else "phase-corr fallback"
        s = QLabel(f"Blind multi-frame restoration  ·  U-Net + SGLD  ·  {flow_str}  ·  "
                   f"{torch_str}  ·  Singh et al. 2025")
        s.setStyleSheet(f"color:{DIM};font-size:9pt;padding:0 8px 4px;")
        v.addWidget(t); v.addWidget(s); lay.addLayout(v,1)
        lay.addWidget(QLabel(f"v{VERSION}  |  HYPERLOAD"))
        return hdr

    def _controls(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner); lay.setContentsMargins(8,8,8,8); lay.setSpacing(5)

        # Load
        grp = QGroupBox("Load Frames"); gl = QVBoxLayout(grp)
        self.frame_lbl = QLabel("No frames loaded")
        self.frame_lbl.setStyleSheet(f"color:{DIM};font-size:10px;"); self.frame_lbl.setWordWrap(True)
        b1 = QPushButton("📂  Load FITS folder (lucky_preprocessor output)…")
        b1.setObjectName("run"); b1.clicked.connect(self._load_folder)
        b2 = QPushButton("📋  Load FITS files…"); b2.clicked.connect(self._load_files)
        b3 = QPushButton("⚗  Synthetic example"); b3.clicked.connect(self._load_example)
        gl.addWidget(self.frame_lbl); gl.addWidget(b1); gl.addWidget(b2); gl.addWidget(b3)
        lay.addWidget(grp)

        # DIPLI parameters
        grp2 = QGroupBox("DIPLI Parameters  (Singh et al. 2025)"); g2 = QFormLayout(grp2)
        self.cb_scale = QComboBox(); self.cb_scale.addItems(["2× (fast)","4× (paper default)","8× (large)"])
        self.cb_scale.setCurrentIndex(1)
        self.sp_K    = QSpinBox(); self.sp_K.setRange(3,50); self.sp_K.setValue(11)
        self.sp_N    = QSpinBox(); self.sp_N.setRange(100,20000); self.sp_N.setValue(2000); self.sp_N.setSingleStep(500)
        self.sp_n0   = QDoubleSpinBox(); self.sp_n0.setRange(0.5,0.99); self.sp_n0.setValue(0.90); self.sp_n0.setDecimals(2)
        self.sp_lr   = QDoubleSpinBox(); self.sp_lr.setRange(1e-4,0.01); self.sp_lr.setValue(0.0025); self.sp_lr.setDecimals(4)
        self.sp_xi   = QDoubleSpinBox(); self.sp_xi.setRange(0,0.05); self.sp_xi.setValue(0.0025); self.sp_xi.setDecimals(4)
        self.sp_sz   = QDoubleSpinBox(); self.sp_sz.setRange(0,0.1); self.sp_sz.setValue(0.01); self.sp_sz.setDecimals(3)
        self.sp_alpha= QDoubleSpinBox(); self.sp_alpha.setRange(0.01,2.0); self.sp_alpha.setValue(0.10); self.sp_alpha.setDecimals(2)
        self.sp_psf  = QDoubleSpinBox(); self.sp_psf.setRange(0,10); self.sp_psf.setValue(0.0); self.sp_psf.setDecimals(1)
        self.cb_feat = QComboBox(); self.cb_feat.addItems(["32 (CPU, fast)","64 (GPU / patience)","128 (paper, GPU)"])
        g2.addRow("Scale factor:", self.cb_scale)
        g2.addRow("K frames (7–13):", self.sp_K)
        g2.addRow("N iterations:", self.sp_N)
        g2.addRow("Warm-up fraction:", self.sp_n0)
        g2.addRow("Learning rate λ:", self.sp_lr)
        g2.addRow("SGLD noise σ_ξ:", self.sp_xi)
        g2.addRow("Latent noise σ_z:", self.sp_sz)
        g2.addRow("Confidence α:", self.sp_alpha)
        g2.addRow("PSF σ (px, 0=skip):", self.sp_psf)
        g2.addRow("Feature channels:", self.cb_feat)
        note = QLabel("Paper defaults: K=11, N=6500, σ_ξ=0.0025.\nCPU defaults shown — reduce N for speed.")
        note.setStyleSheet(f"color:{DIM};font-size:9px;"); note.setWordWrap(True)
        g2.addRow(note); lay.addWidget(grp2)

        # Run
        self.btn_run = QPushButton("▶  Restore")
        self.btn_run.setObjectName("run"); self.btn_run.setFixedHeight(38)
        self.btn_run.setEnabled(False); self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("✖  Cancel")
        self.btn_cancel.setEnabled(False); self.btn_cancel.clicked.connect(self._cancel)
        self.btn_export = QPushButton("💾  Export FITS…")
        self.btn_export.setEnabled(False); self.btn_export.clicked.connect(self._export)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_cancel); lay.addWidget(self.btn_export)
        lay.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(f"color:{OK};font-size:10px;"); self.summary_lbl.setWordWrap(True)
        lay.addWidget(self.summary_lbl)
        scroll.setWidget(inner); return scroll

    def _tabs(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_frames(),   "🌠  Frames")
        self.tabs.addTab(self._tab_training(), "📈  Training")
        self.tabs.addTab(self._tab_result(),   "🔬  Result")
        self.tabs.addTab(self._tab_log(),      "📋  Log")
        return self.tabs

    def _tab_frames(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.frames_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.frames_plot)
        return w

    def _tab_training(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.train_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.train_plot)
        ctrl = QHBoxLayout()
        self.mc_lbl = QLabel("MC samples: 0"); self.mc_lbl.setStyleSheet(f"color:{WRN};")
        ctrl.addWidget(self.mc_lbl); ctrl.addStretch()
        lay.addLayout(ctrl)
        return w

    def _tab_result(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.result_plot = MplCanvas(figsize=(9,6)); lay.addWidget(self.result_plot)
        return w

    def _tab_log(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True); lay.addWidget(self.log)
        btn = QPushButton("Clear"); btn.clicked.connect(self.log.clear); lay.addWidget(btn)
        return w

    def _footer(self):
        foot = QWidget(); foot.setFixedHeight(22)
        foot.setStyleSheet(f"background:{BG};border-top:1px solid {BOR};")
        lay = QHBoxLayout(foot); lay.setContentsMargins(6,0,6,0)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet(f"color:{DIM};font-size:10px;")
        self.pbar = QProgressBar(); self.pbar.setFixedSize(220,12); self.pbar.setVisible(False)
        lay.addWidget(self.status_lbl,1); lay.addWidget(self.pbar)
        return foot

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select folder with ranked FITS frames")
        if not path: return
        fits_files = sorted(glob.glob(os.path.join(path,"*.fits")) +
                            glob.glob(os.path.join(path,"*.fit")) +
                            glob.glob(os.path.join(path,"*.fts")))
        if not fits_files:
            QMessageBox.warning(self,"No FITS","No FITS files found in folder."); return
        self._load_fits_list(fits_files)

    def _load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select FITS frames", "",
            "FITS (*.fits *.fit *.fts);;All (*)")
        if paths: self._load_fits_list(sorted(paths))

    def _load_fits_list(self, paths):
        frames = []
        for p in paths:
            try:
                data = astropy_fits.getdata(p)
                frames.append(luminance(data))
            except Exception: pass
        if not frames:
            QMessageBox.critical(self,"Load error","Could not read any frames."); return
        self._set_frames(frames, f"{len(frames)} frames from FITS")

    def _load_example(self):
        """Synthetic: Jupiter-like disk, 11 frames with atmospheric distortion."""
        rng2 = np.random.default_rng(42); N=128; R=55.0; cx=cy=64.0
        yy,xx=np.ogrid[:N,:N]
        r_norm=np.sqrt((xx-cx)**2+(yy-cy)**2)/R
        mu=np.sqrt(np.maximum(1-r_norm**2,0))
        ld=(1-0.35*(1-mu)-0.20*(1-mu)**2)
        lat=np.degrees(np.arcsin(np.clip(-(yy-cy)/R,-1,1)))
        lon=np.degrees(np.arctan2((xx-cx),np.sqrt(np.maximum(R**2-(yy-cy)**2-(xx-cx)**2,0))))
        belt=(1+0.3*np.cos(np.radians(lat*2))-0.35*np.exp(-((lat-15)**2)/50)
              -0.35*np.exp(-((lat+15)**2)/50)+0.15*np.sin(np.radians(lon*3)))
        gt=np.clip(belt*ld*(r_norm<=1).astype(float),0,None)
        # 11 degraded frames: atmospheric shift + noise + blur
        from scipy.ndimage import shift, gaussian_filter
        frames=[]
        for i in range(11):
            dy=rng2.uniform(-2,2); dx=rng2.uniform(-2,2)
            blurred=gaussian_filter(gt,sigma=rng2.uniform(0.8,1.5))
            shifted=shift(blurred,[dy,dx],mode='reflect')
            frames.append(np.clip(shifted+rng2.normal(0,0.04,(N,N)),0,None).astype(np.float32))
        self._set_frames(frames, "11 synthetic Jupiter frames (example)")

    def _set_frames(self, frames, label):
        self._frames = frames
        self.frame_lbl.setText(f"{label}\n{frames[0].shape[1]}×{frames[0].shape[0]}px each")
        self.frame_lbl.setStyleSheet(f"color:{OK};font-size:10px;")
        self.btn_run.setEnabled(TORCH_OK)
        self._draw_frames()
        self._log(f"Loaded {len(frames)} frames: {label}")
        if not TORCH_OK:
            self._log("⚠  PyTorch not installed. Install with: pip install torch")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _params(self):
        scale_map = {0:2, 1:4, 2:8}
        feat_map  = {0:32, 1:64, 2:128}
        return dict(
            scale      = scale_map[self.cb_scale.currentIndex()],
            K          = min(self.sp_K.value(), len(self._frames)),
            N          = self.sp_N.value(),
            n0_frac    = self.sp_n0.value(),
            lr         = self.sp_lr.value(),
            sigma_xi   = self.sp_xi.value(),
            sigma_z    = self.sp_sz.value(),
            alpha_conf = self.sp_alpha.value(),
            psf_sigma  = self.sp_psf.value(),
            feat       = feat_map[self.cb_feat.currentIndex()],
        )

    def _run(self):
        if not self._frames or not TORCH_OK: return
        p = self._params()
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True)
        self.btn_export.setEnabled(False)
        self._loss_history = []
        self._log(f"\n{'='*50}")
        self._log(f"Starting DIPLI: scale={p['scale']}× K={p['K']} N={p['N']} "
                  f"feat={p['feat']} device={DEVICE}")

        self._wthread = QThread(self)
        self._worker  = DIPLIWorker(self._frames, p)
        self._worker.moveToThread(self._wthread)
        self._wthread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._wthread.start()
        self.tabs.setCurrentIndex(1)

    def _cancel(self):
        if self._worker: self._worker.cancel()
        self.btn_cancel.setEnabled(False)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_progress(self, it, total, loss, preview):
        self._loss_history.append(loss)
        pct = int(100*it/total)
        self.pbar.setVisible(True); self.pbar.setValue(pct)
        n0  = int(total * self.sp_n0.value())
        mc  = max(0, it - n0)
        self.mc_lbl.setText(f"MC samples: {mc}")
        self.status_lbl.setText(f"Iter {it}/{total}  loss={loss:.5f}  MC={mc}")
        self._draw_training(preview)
        if it >= total:
            QTimer.singleShot(3000, lambda: self.pbar.setVisible(False))

    def _on_error(self, msg):
        self._log(f"\n❌  {msg}"); self.btn_run.setEnabled(True)
        QMessageBox.critical(self,"Error",msg[:500])

    def _on_finished(self, result):
        self._result = result; self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False); self.btn_export.setEnabled(True)
        mc = result['mc_count']; pivot = result['pivot_global']
        self.summary_lbl.setText(
            f"Done!  MC samples: {mc}\n"
            f"Pivot frame: #{pivot}  |  Used: {result['frames_used'][:5]}…")
        self._log(f"✓  Done  MC samples={mc}  pivot=#{pivot}")
        self._log(f"   Output: {result['y_star'].shape}")
        self._draw_result(); self.tabs.setCurrentIndex(2)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_frames(self):
        frames = self._frames[:12]
        n = len(frames); cols = min(n,6); rows = math.ceil(n/cols)
        energies = [laplacian_energy(f) for f in frames]
        best_idx = int(np.argmax(energies))

        fig = self.frames_plot.fig; fig.clear()
        fig.patch.set_facecolor(BG)
        axes = fig.subplots(rows, cols) if n>1 else [[fig.add_subplot(111)]]
        if rows==1 and n>1: axes=[axes]
        ax_flat=[a for row in axes for a in (row if hasattr(row,'__iter__') else [row])]
        for i,(ax,f) in enumerate(zip(ax_flat,frames)):
            lo,hi=np.percentile(f,[1,99])
            ax.imshow(f,origin='lower',cmap='gray',vmin=lo,vmax=hi,
                      aspect='equal',interpolation='bicubic')
            color=OK if i==best_idx else TXT
            ax.set_title(f"#{i}  e={energies[i]:.3f}{'  ★PIVOT' if i==best_idx else ''}",
                         color=color,fontsize=7,pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(OK if i==best_idx else BOR)
                sp.set_linewidth(2 if i==best_idx else 0.5)
        for ax in ax_flat[n:]: ax.set_visible(False)
        self.frames_plot.redraw()

    def _draw_training(self, preview):
        fig = self.train_plot.fig; fig.clear()
        fig.patch.set_facecolor(BG)
        n_mc_start = int(self.sp_N.value() * self.sp_n0.value())

        if preview is not None:
            axes = fig.subplots(1,2)
        else:
            axes = [fig.add_subplot(111), None]

        # Loss curve
        ax = axes[0]; _ax(ax,"Loss (SGLD convergence)","Iteration","Loss")
        if self._loss_history:
            iters = range(1, len(self._loss_history)+1)
            ax.semilogy(list(iters), self._loss_history, color=ACC, lw=1, alpha=0.8)
            if n_mc_start < len(self._loss_history):
                ax.axvline(n_mc_start, color=WRN, lw=1, ls='--', label='MC start')
                ax.legend(fontsize=7, facecolor=BG3, edgecolor=BOR, labelcolor=TXT)
        ax.set_xlim(0, max(len(self._loss_history),1))

        # Live MC preview
        if preview is not None and axes[1] is not None:
            _ax(axes[1], "MC estimate (live preview)")
            lo,hi = np.percentile(preview,[0.5,99.5])
            axes[1].imshow(preview, origin='lower', cmap='gray',
                           vmin=lo, vmax=hi, aspect='equal', interpolation='bicubic')
            axes[1].set_xticks([]); axes[1].set_yticks([])

        self.train_plot.redraw()

    def _draw_result(self):
        r = self._result
        if not r or r.get('y_star') is None: return
        pivot = r['pivot']; y_star = r['y_star']

        fig = self.result_plot.fig; fig.clear()
        fig.patch.set_facecolor(BG)
        axes = fig.subplots(1,2)

        _ax(axes[0],"Pivot frame (best LQ input)")
        lo,hi=np.percentile(pivot,[1,99.5])
        axes[0].imshow(pivot,origin='lower',cmap='gray',vmin=lo,vmax=hi,
                       aspect='equal',interpolation='bicubic')
        axes[0].text(0.02,0.02,f"{pivot.shape[1]}×{pivot.shape[0]}px",
                     transform=axes[0].transAxes,color=DIM,fontsize=8)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        _ax(axes[1],f"DIPLI restored  ({y_star.shape[1]}×{y_star.shape[0]}px, "
                    f"MC samples={r['mc_count']})")
        lo2,hi2=np.percentile(y_star,[0.5,99.5])
        axes[1].imshow(y_star,origin='lower',cmap='gray',vmin=lo2,vmax=hi2,
                       aspect='equal',interpolation='bicubic')
        axes[1].set_xticks([]); axes[1].set_yticks([])

        self.result_plot.redraw()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        r = self._result
        if not r or r.get('y_star') is None: return
        path, _ = QFileDialog.getSaveFileName(
            self,"Export DIPLI result","dipli_result.fits","FITS (*.fits);;All (*)")
        if not path: return
        try:
            hdr = astropy_fits.Header()
            hdr['COMMENT'] = 'DIPLI restoration (Singh et al. 2025)'
            hdr['DIPLI_MC']= r['mc_count']
            hdr['PIVOT']   = r['pivot_global']
            astropy_fits.PrimaryHDU(r['y_star'].astype(np.float32), header=hdr).writeto(
                path, overwrite=True)
            self._log(f"✓  Exported: {path}")
        except Exception as e:
            QMessageBox.critical(self,"Export error",str(e))

    def _log(self, msg):
        self.log.append(msg); self.log.ensureCursorVisible()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DIPLIApp(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
