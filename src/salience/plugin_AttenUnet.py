# src/salience/plugin.py
import torch
import numpy as np
from .multi_scale import salience_2d


class SalienceGenerator:
    """
    Generate 3-channel salience-enhanced CT volume.

    Input:
        volume: (H, W, D), clipped and normalized CT.

    Output:
        G_hat: (3, H, W, D)
    """

    def __init__(self, delta_theta=15, delta_sigma=1.0, K=5, alpha=0.5):
        self.delta_theta = delta_theta
        self.delta_sigma = delta_sigma
        self.K = K
        self.alpha = alpha

    def _calc_gamma(self, volume):
        return 1.0

    def _normalize_salience(self, sal):
        """
        Normalize one salience volume to [0, 1].
        """
        sal = torch.nan_to_num(sal.float(), nan=0.0, posinf=0.0, neginf=0.0)

        sal_min = sal.min()
        sal_max = sal.max()

        if (sal_max - sal_min) < 1e-6:
            return torch.zeros_like(sal)

        sal = (sal - sal_min) / (sal_max - sal_min + 1e-6)
        return sal

    def _salience_per_view(self, volume, view):
        """
        view: 'cor' / 'sag' / 'tra'
        """
        if view == "cor":
            slices = volume.permute(0, 2, 1)
        elif view == "sag":
            slices = volume.permute(1, 2, 0)
        elif view == "tra":
            slices = volume
        else:
            raise ValueError(f"Unsupported view: {view}")

        H, W, D = slices.shape
        saliences = []
        gamma = self._calc_gamma(volume)

        for idx in range(D):
            img = slices[:, :, idx].float()
            img = img - img.mean()

            sal = salience_2d(
                img.unsqueeze(0),
                delta_theta=self.delta_theta,
                delta_sigma=self.delta_sigma,
                K=self.K,
                gamma=gamma,
            )
            saliences.append(sal)

        sal = torch.stack(saliences, dim=2)

        if view == "cor":
            sal = sal.permute(0, 2, 1)
        elif view == "sag":
            sal = sal.permute(2, 0, 1)

        return self._normalize_salience(sal)

    def __call__(self, volume):
        """
        volume: torch Tensor (H, W, D), clipped and normalized CT.
        Returns: enhanced volume (3, H, W, D).
        """
        volume = torch.nan_to_num(volume.float(), nan=0.0, posinf=1.0, neginf=0.0)
        volume = torch.clamp(volume, 0.0, 1.0)

        S_c = self._salience_per_view(volume, "cor")
        S_s = self._salience_per_view(volume, "sag")
        S_t = self._salience_per_view(volume, "tra")

        Gc_hat = volume + self.alpha * volume * S_c
        Gs_hat = volume + self.alpha * volume * S_s
        Gt_hat = volume + self.alpha * volume * S_t

        G_hat = torch.stack([Gc_hat, Gs_hat, Gt_hat], dim=0)
        G_hat = torch.clamp(G_hat, 0.0, 1.0)

        return G_hat
