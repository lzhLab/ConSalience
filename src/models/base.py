# src/models/base.py
import torch.nn as nn

class BaseSegModel(nn.Module):
    """
    所有分割网络的父类，提供统一的 forward 接口:
        forward(volume, use_salience=False, salience_gen=None)
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone   # e.g. UNet3D, AttnUNet, ...

    def forward(self, volume, use_salience=False, salience_gen=None):
        """
        volume: (H,W,D) raw CT (float)
        use_salience: bool
        salience_gen: SalienceGenerator instance (or None)
        """
        if use_salience:
            assert salience_gen is not None, "SalienceGenerator required"
            volume = salience_gen(volume)   # (3,H,W,D)
        else:
            # baseline: expand channel dim to 1 (or 3 for compatibility)
            volume = volume.unsqueeze(0)    # (1,H,W,D)

        # 将 (C,H,W,D) 送入网络
        # 注意：不同网络接受的维度不同，统一在子类里实现
        out = self.backbone(volume)        # (1,H,W,D) logits
        return out

