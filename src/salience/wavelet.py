"""Square-wavelet kernels used by the ConSalience salience generator.

This module is a drop-in replacement for the original ``wavelet.py``.  The
public ``square_wavelet`` signature is kept compatible with
``src.salience.multi_scale``.

Important implementation details:
    * The square wave contains both ``+1`` and ``-1`` half-periods.
    * Values on discontinuities ``x = k * lambda / 2`` are set to zero.
    * The finite Gaussian-windowed kernel is made zero-mean and L1-normalized
      to avoid scale-dependent DC leakage and response-amplitude drift.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


Number = Union[float, int, torch.Tensor]


def _scalar_tensor(
    value: Number,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a scalar or scalar tensor to the requested device and dtype."""
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.numel() != 1:
        raise ValueError(f"Expected a scalar, got shape {tuple(tensor.shape)}")
    return tensor.reshape(())


def _check_positive(name: str, value: torch.Tensor) -> None:
    """Validate a positive scalar parameter with a useful error message."""
    if not bool(torch.isfinite(value).item()) or not bool((value > 0).item()):
        raise ValueError(f"{name} must be a finite positive scalar, got {value}")


def odd_square_wave(x: torch.Tensor, lam: Number) -> torch.Tensor:
    """Evaluate the odd square wave from Eq. (1).

    The periodic definition is:

        +1,  k*lambda       < x < k*lambda + lambda/2
        -1,  k*lambda+...  < x < (k+1)*lambda
         0,  x = k*lambda/2

    ``torch.remainder`` is used instead of ``floor(x / lambda)`` interval
    logic.  This is important for negative half-periods and for the second
    half of every period.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)

    lam_t = _scalar_tensor(lam, device=x.device, dtype=x.dtype)
    _check_positive("lam", lam_t)

    phase = torch.remainder(x, lam_t)
    half_lam = lam_t / 2.0

    # First half-period is positive, second half-period is negative.
    wave = torch.where(
        phase < half_lam,
        torch.ones_like(x),
        -torch.ones_like(x),
    )

    # Match the piecewise definition at all half-period discontinuities.
    atol = max(1e-6, 1e-5 * float(lam_t.detach().cpu()))
    at_zero = torch.isclose(
        phase,
        torch.zeros_like(phase),
        atol=atol,
        rtol=0.0,
    )
    at_half = torch.isclose(
        phase,
        half_lam.expand_as(phase),
        atol=atol,
        rtol=0.0,
    )

    return torch.where(at_zero | at_half, torch.zeros_like(wave), wave)


def even_square_wave(x: torch.Tensor, lam: Number) -> torch.Tensor:
    """Evaluate the even square wave using the phase shift from Eq. (3)."""
    lam_t = _scalar_tensor(lam, device=x.device, dtype=x.dtype)
    return odd_square_wave(x + lam_t / 4.0, lam_t)


def gaussian_window(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Number,
    gamma: Number,
) -> torch.Tensor:
    """Evaluate the anisotropic Gaussian window from Eq. (4)."""
    sigma_t = _scalar_tensor(sigma, device=x.device, dtype=x.dtype)
    gamma_t = _scalar_tensor(gamma, device=x.device, dtype=x.dtype)
    _check_positive("sigma", sigma_t)
    _check_positive("gamma", gamma_t)

    return torch.exp(
        -(x.square() + (gamma_t * y).square()) / (2.0 * sigma_t.square())
    )


def rotate_grid(
    height: int,
    width: int,
    theta: Number,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create centered coordinates rotated by ``theta`` radians.

    Returns ``x'`` and ``y'`` with shape ``(height, width)``.  The coordinate
    convention matches the paper:

        x' = x cos(theta) - y sin(theta)
        y' = x sin(theta) + y cos(theta)
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}x{width}")

    if device is None:
        if isinstance(theta, torch.Tensor):
            device = theta.device
        else:
            device = torch.device("cpu")

    theta_t = _scalar_tensor(theta, device=device, dtype=dtype)

    x = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2.0
    y = torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2.0
    xx, yy = torch.meshgrid(x, y, indexing="xy")

    cos_theta = torch.cos(theta_t)
    sin_theta = torch.sin(theta_t)

    x_rot = xx * cos_theta - yy * sin_theta
    y_rot = xx * sin_theta + yy * cos_theta
    return x_rot, y_rot


def _normalize_kernel(kernel: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Remove DC leakage and stabilize response magnitude across scales."""
    kernel = kernel - kernel.mean()
    l1 = kernel.abs().sum()
    return torch.where(
        l1 > eps,
        kernel / (l1 + eps),
        torch.zeros_like(kernel),
    )


def square_wavelet(
    height: int,
    width: int,
    lam: Number,
    sigma: Number,
    gamma: Number,
    theta: Number,
    even: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Construct one Gaussian-windowed square-wavelet kernel.

    Parameters are compatible with the original implementation.  ``theta``
    is in radians, while ``lam``, ``sigma`` and ``gamma`` are expressed in
    pixel-coordinate units.

    Returns:
        Tensor of shape ``(height, width)`` on ``device``.
    """
    if device is None:
        for value in (lam, sigma, gamma, theta):
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        else:
            device = torch.device("cpu")

    # Use the input parameter dtype when possible; otherwise float32 is the
    # correct default for convolution kernels used by salience_2d.
    dtype = torch.float32
    for value in (lam, sigma, gamma, theta):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            dtype = value.dtype
            break

    x_rot, y_rot = rotate_grid(
        height,
        width,
        theta,
        device=device,
        dtype=dtype,
    )

    if even:
        square = even_square_wave(x_rot, lam)
    else:
        square = odd_square_wave(x_rot, lam)

    window = gaussian_window(x_rot, y_rot, sigma, gamma)
    kernel = square * window
    return _normalize_kernel(kernel)


__all__ = [
    "odd_square_wave",
    "even_square_wave",
    "gaussian_window",
    "rotate_grid",
    "square_wavelet",
]
