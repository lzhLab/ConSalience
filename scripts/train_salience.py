# scripts/train_salience.py
import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.experiment import Experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train segmentation model with salience enabled."
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="Path to yaml config, e.g. src/config/unet3d.yaml",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help=(
            "Override config by key=value pairs, "
            "e.g. data.dataset=3Dircadb1 train.epochs=100 train.save_dir=checkpoints/unet3d_salience"
        ),
    )
    return parser.parse_args()


def parse_value(value: str):
    lower = value.lower()

    if lower in ("true", "false"):
        return lower == "true"

    if lower in ("none", "null"):
        return None

    try:
        if "." not in value:
            return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def set_by_dotted_key(cfg: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    current = cfg

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}

        if not isinstance(current[key], dict):
            raise TypeError(
                f"Cannot override '{dotted_key}': '{key}' is not a dict in config."
            )

        current = current[key]

    current[keys[-1]] = value


def get_by_dotted_key(cfg: dict, dotted_key: str, default=None):
    keys = dotted_key.split(".")
    current = cfg

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def apply_overrides(cfg: dict, overrides: list) -> dict:
    cfg = copy.deepcopy(cfg)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected format: key=value")

        key, value = item.split("=", 1)
        key = key.strip()
        value = parse_value(value.strip())

        if not key:
            raise ValueError(f"Invalid override '{item}': empty key.")

        set_by_dotted_key(cfg, key, value)

    return cfg


def load_config(cfg_path: str) -> dict:
    path = Path(cfg_path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty config file: {cfg_path}")

    return cfg


def force_salience_config(cfg: dict) -> dict:
    """
    Force salience training options.

    Salience generator produces 3-channel enhanced input:
        (B, 3, D, H, W)

    Therefore model.in_channels should be 3.
    """
    cfg = copy.deepcopy(cfg)

    if "data" not in cfg:
        cfg["data"] = {}

    if "model" not in cfg:
        cfg["model"] = {}

    set_by_dotted_key(cfg, "data.use_salience", True)
    set_by_dotted_key(cfg, "model.in_channels", 3)

    salience_cfg = get_by_dotted_key(cfg, "salience", default=None)
    if salience_cfg is None:
        cfg["salience"] = {
            "delta_theta": 15,
            "delta_sigma": 1.0,
            "K": 5,
            "gamma": 1.0,
            "cache": False,
        }

    return cfg


def main():
    args = parse_args()

    cfg = load_config(args.cfg)
    cfg = apply_overrides(cfg, args.override)
    cfg = force_salience_config(cfg)

    print("=== Salience Training Config ===")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    exp = Experiment(cfg)
    exp.run()


if __name__ == "__main__":
    main()

