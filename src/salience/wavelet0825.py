import torch
import math
import torch.nn.functional as F

PI = math.pi

def odd_square_wave(x, lam):
    """T_o(x)  - odd square wave, vectorized for torch tensors."""
    # x is in pixel coordinates (float)
    k = torch.floor(x / lam)
    # region (k*lam, lam/2 + k*lam)
    cond1 = (x > k * lam) & (x < lam / 2 + k * lam)
    # region (-lam/2 + k*lam, k*lam)
    cond2 = (x > -lam / 2 + k * lam) & (x < k * lam)
    out = torch.where(cond1, torch.ones_like(x),
          torch.where(cond2, -torch.ones_like(x),
          torch.zeros_like(x)))
    return out

def even_square_wave(x, lam):
    """T_e(x) = T_o(x + lam/4)"""
    return odd_square_wave(x + lam / 4.0, lam)

def gaussian_window(x, y, sigma, gamma):
    """ψ(x,y) = exp(-(x² + γ² y²) / (2σ²))"""
    return torch.exp(-(x**2 + (gamma*y)**2) / (2.0 * sigma**2))

def rotate_grid(H, W, theta, device=None):
    """Return meshgrid (x',y') after rotation θ (rad)."""
    if device is None:
        device = theta.device if isinstance(theta, torch.Tensor) else "cpu"
    theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device) - (W-1)/2.0
    ys = torch.arange(H, dtype=torch.float32, device=device) - (H-1)/2.0
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    # rotation
    x_rot =  xx * torch.cos(theta) - yy * torch.sin(theta)
    y_rot =  xx * torch.sin(theta) + yy * torch.cos(theta)
    return x_rot, y_rot

def square_wavelet(H, W, lam, sigma, gamma, theta, even=False, device=None):
    """Return 2‑D kernel w_o or w_e."""
    x_rot, y_rot = rotate_grid(H, W, theta, device=device)
    if even:
        sq = even_square_wave(x_rot, lam)
    else:
        sq = odd_square_wave(x_rot, lam)
    win = gaussian_window(x_rot, y_rot, sigma, gamma)
    return sq * win   # shape (H,W)

