# src/models/build_model.py
from src.models.unet3d import UNet3D


def build_model(cfg):
    """
    Build segmentation model from config.

    Expected cfg example:
        model:
          name: UNet3D
          in_channels: 1
          out_channels: 1
          base_channels: 16
    """
    model_name = cfg.get("name", "UNet3D")

    if model_name == "UNet3D":
        return UNet3D(
            in_channels=cfg.get("in_channels", 1),
            out_channels=cfg.get("out_channels", 1),
            base_channels=cfg.get("base_channels", 16),
        )

    raise ValueError(f"Unsupported model name: {model_name}")
