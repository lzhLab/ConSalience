import math

import torch
import torch.nn.functional as F
from .wavelet import square_wavelet

def salience_2d(image, delta_theta=15, delta_sigma=1.0, K=5, gamma=1.0):
    """
    image: Tensor (1, H, W)  – already mean‑centered (g')
    Returns: salience map s (H,W)
    """
    device = image.device
    _, H, W = image.shape
    s = torch.zeros((H, W), device=device)

    # convert degrees → rad
    thetas = torch.arange(0, 180, delta_theta, device=device) * math.pi / 180.0
    sigmas = torch.arange(1, K+1, device=device, dtype=torch.float32) * delta_sigma

    for k, sigma in enumerate(sigmas, start=1):
        for m, theta in enumerate(thetas):
            kernel_size = 2 * int(math.ceil(3 * float(sigma))) + 1
            wo = square_wavelet(kernel_size, kernel_size, lam=4*sigma, sigma=sigma,
                               gamma=gamma, theta=theta, even=False, device=device)
            we = square_wavelet(kernel_size, kernel_size, lam=4*sigma, sigma=sigma,
                               gamma=gamma, theta=theta, even=True, device=device)

            # shape (1,1,H,W) for conv2d
            wo = wo.unsqueeze(0).unsqueeze(0)
            we = we.unsqueeze(0).unsqueeze(0)

            # convolution (valid padding → same size via padding)
            pad = wo.shape[-1] // 2
            s += F.conv2d(image.unsqueeze(0), wo, padding=pad).squeeze()
            s += F.conv2d(image.unsqueeze(0), we, padding=pad).squeeze()
    return s
