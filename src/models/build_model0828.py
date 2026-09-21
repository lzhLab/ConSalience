# src/models/build_model.py
from src.models.unet3d import UNet3D
from src.models.AttnUNet3D import AttnUNet3D


def build_model(cfg: dict):
    model_name = cfg.get("name", "UNet3D")

    print("=== Build Model ===")
    print(f"model.name: {model_name}")
    print(f"in_channels: {cfg.get('in_channels', 1)}")
    print(f"out_channels: {cfg.get('out_channels', 1)}")
    print(f"base_channels: {cfg.get('base_channels', 32)}")

    if model_name == "UNet3D":
        return UNet3D(
            in_channels=cfg.get("in_channels", 1),
            out_channels=cfg.get("out_channels", 1),
            base_channels=cfg.get("base_channels", 32),
        )

    if model_name in ("AttnUNet3D", "AttnUNet"):
        return AttnUNet3D(
            in_channels=cfg.get("in_channels", 1),
            out_channels=cfg.get("out_channels", 1),
            base_channels=cfg.get("base_channels", 32),
        )

    raise ValueError(f"Unsupported model name: {model_name}")
