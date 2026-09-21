import torch
import numpy as np
from .multi_scale import salience_2d

class SalienceGenerator:
    """
    负责：
        1) 根据整幅 3-D CT (H,W,D) 计算三视图的 2-D salience
        2) 通过残差门 (Eq.10-11) 生成增强体积  Ĝ ∈ R^{3×H×W×D}
    """
    def __init__(self, delta_theta=15, delta_sigma=1.0, K=5):
        self.delta_theta = delta_theta
        self.delta_sigma = delta_sigma
        self.K = K

    def _calc_gamma(self, volume):
        """
        根据论文 Sec II.B.1：在 vessel mask区域计算 Hessian 的平均长宽比 r̄。
        这里我们用一个简易近似：在已知的 GT (或粗分割) 上求 eigenvalue ratio。
        若没有 GT，则使用经验值 γ=1.0。
        """
        # volume: (H,W,D) torch Tensor
        # 这里返回 1.0（默认），实际实验中可改为基于预估 mask 的计算
        return 1.0

    def _salience_per_view(self, volume, view):
        """
        view: 'cor' / 'sag' / 'tra'
        """
        if view == 'cor':   # H×D  (slice along width)
            slices = volume.permute(0, 2, 1)   # (H, D, W) -> treat (H,W) as plane
        elif view == 'sag': # W×D
            slices = volume.permute(1, 2, 0)   # (W, D, H)
        else:               # 'tra' transverse
            slices = volume  # (H, W, D)

        H, W, D = slices.shape
        saliences = []
        gamma = self._calc_gamma(volume)

        for idx in range(D):
            img = slices[:, :, idx]               # (H,W)
            img = img.float()
            img = img - img.mean()                # g'
            s = salience_2d(img.unsqueeze(0), 
                            delta_theta=self.delta_theta,
                            delta_sigma=self.delta_sigma,
                            K=self.K,
                            gamma=gamma)
            saliences.append(s)

        # stack back to original orientation
        sal = torch.stack(saliences, dim=2)  # (H,W,D)
        if view == 'cor':
            sal = sal.permute(0, 2, 1)      # (H,W,D) → (H,W,D) correct orientation
        elif view == 'sag':
            sal = sal.permute(2, 0, 1)      # (W,H,D) → (H,W,D)
        return sal

    def __call__(self, volume):
        """
        volume: torch Tensor (H,W,D)  - raw CT (HU) already clipped & normalized
        Returns: enhanced volume ~\  (3, H, W, D)
        """
        # 1) compute 3 salience maps
        S_c = self._salience_per_view(volume, 'cor')
        S_s = self._salience_per_view(volume, 'sag')
        S_t = self._salience_per_view(volume, 'tra')

        # 2) residual~@~Qgate (Eq.10)
        Gc = volume
        Gs = volume
        Gt = volume
        Gc_hat = Gc + Gc * S_c
        Gs_hat = Gs + Gs * S_s
        Gt_hat = Gt + Gt * S_t

        # 3) channel~@~Qwise concat (Eq.11)
        G_hat = torch.stack([Gc_hat, Gs_hat, Gt_hat], dim=0)  # (3, H, W, D)
        return G_hat

