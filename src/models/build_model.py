from src.models.unet3d import UNet3D
from src.models.AttnUNet3D import AttnUNet3D
from src.models.Hvmunet import HVMUNet3D
from src.models.DSwinUNet import DSwinUNet
from src.models.MambaVesselNet import MambaVesselNet

def build_model(cfg: dict):
    model_name = cfg.get("name", "UNet3D")
    default_base_channels = 8 if model_name in (
        "Hvmunet",
        "HVMUNet",
        "H_vmunet",
        "H-vmunet",
    ) else 32

    print("=== Build Model ===")
    print(f"model.name: {model_name}")
    print(f"in_channels: {cfg.get('in_channels', 1)}")
    print(f"out_channels: {cfg.get('out_channels', 1)}")
    print(f"base_channels: {cfg.get('base_channels', default_base_channels)}")

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

    if model_name in ("Hvmunet", "HVMUNet", "H_vmunet", "H-vmunet"):
        return HVMUNet3D(
            in_channels=cfg.get("in_channels", 1),
            out_channels=cfg.get("out_channels", 1),
            base_channels=cfg.get("base_channels", 8),
            c_list=cfg.get("c_list"),
            depths=cfg.get("depths"),
            order=cfg.get("order"),
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
            layer_scale_init=cfg.get("layer_scale_init", 1e-6),
            bridge=cfg.get("bridge", True),
        )
    if model_name in ("DSwinUNet", "DynamicSwinUNet", "DSwinUNet3D"):
        return DSwinUNet(
            in_channels=cfg.get("in_channels", 1),
            out_channels=cfg.get("out_channels", 1),
            base_channels=cfg.get("base_channels", 8),
            c_list=cfg.get("c_list"),
            depths=cfg.get("depths"),
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
        )

    if model_name in (
        "MambaVesselNet",
        "MambaVesselNet3D",
        "MambaVesselNetPP",
        "mvnNet",
        "MVNNet",
    ):
        return MambaVesselNet(
            in_channels=cfg.get("in_channels", 1),
            out_channels=cfg.get("out_channels", 1),
            base_channels=cfg.get("base_channels", 8),
            feature_dims=cfg.get("feature_dims"),
            c_list=cfg.get("c_list"),
            depths=cfg.get("depths"),
            bottleneck_depth=cfg.get("bottleneck_depth", 4),
            decoder_depth=cfg.get("decoder_depth", 4),
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
            use_salience=cfg.get("use_salience"),
            bridge=cfg.get("bridge", True),
        )

    raise ValueError(f"Unsupported model name: {model_name}")
