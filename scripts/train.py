# scripts/train.py
import argparse
import copy
from pathlib import Path

import torch
import yaml

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.experiment import Experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model.")
    parser.add_argument(
        "--cfg",
        required=True,
        help="Path to yaml config, e.g. src/config/unet.yaml",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help=(
            "Override config by key=value pairs, "
            "e.g. data.use_salience=False train.batch_size=2 data.dataset=3Dircadb1"
        ),
    )
    return parser.parse_args()


def parse_value(value: str):
    """Cast command line string to bool, int, float, or keep it as string."""
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
    """
    Set nested config value by dotted key.

    Example:
        set_by_dotted_key(cfg, "data.use_salience", False)
    """
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


def apply_overrides(cfg: dict, overrides: list) -> dict:
    cfg = copy.deepcopy(cfg)

    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected format: key=value"
            )

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


def main():
    args = parse_args()

    cfg = load_config(args.cfg)
    cfg = apply_overrides(cfg, args.override)

    print("=== Training Config ===")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # Optional but useful for CUDA speed when input size is fixed.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    exp = Experiment(cfg)
    exp.run()


if __name__ == "__main__":
    main()
