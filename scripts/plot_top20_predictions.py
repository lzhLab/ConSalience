#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Please install PyYAML: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a dictionary: {path}")

    return data


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return root / path


def checkpoint_config(checkpoint: Any) -> Optional[dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        return None

    for key in ("cfg", "config", "resolved_config"):
        value = checkpoint.get(key)

        if isinstance(value, dict):
            return value

    return None


def get_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a dictionary")

    state = None

    for key in ("model_state_dict", "model_state", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break

    if state is None:
        state = checkpoint

    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    if not cleaned:
        raise RuntimeError("No tensor state dict was found in checkpoint")

    return cleaned


def infer_input_channels(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> int:
    for attribute in ("in_channels", "input_channels", "in_chans"):
        value = getattr(model, attribute, None)

        if isinstance(value, int):
            return value

    for key in (
        "stem.0.weight",
        "stem.weight",
        "inc.double_conv.0.weight",
        "encoder1.conv1.weight",
        "conv1.weight",
        "input_proj.weight",
    ):
        weight = state_dict.get(key)

        if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
            return int(weight.shape[1])

    for key, weight in state_dict.items():
        if not isinstance(weight, torch.Tensor):
            continue

        if weight.ndim >= 4 and (
            "conv" in key.lower()
            or "stem" in key.lower()
            or "patch" in key.lower()
            or "input" in key.lower()
        ):
            return int(weight.shape[1])

    raise RuntimeError(
        "Could not determine model input channels. "
        "Please ensure the model exposes `in_channels`."
    )


def build_model(
    cfg: dict[str, Any],
    checkpoint: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    try:
        from src.models.build_model import build_model as model_factory
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import src.models.build_model. "
            "Run the command from the project root."
        ) from exc

    model_cfg = deepcopy(cfg.get("model", {}))

    saved_cfg = checkpoint_config(checkpoint)

    if saved_cfg is not None and isinstance(saved_cfg.get("model"), dict):
        model_cfg = deepcopy(saved_cfg["model"])

    model = model_factory(model_cfg).to(device)

    state_dict = get_state_dict(checkpoint)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print(f"checkpoint missing keys: {len(missing)}")

    if unexpected:
        print(f"checkpoint unexpected keys: {len(unexpected)}")

    model.eval()

    input_channels = infer_input_channels(model, state_dict)

    return model, input_channels


def add_npz_suffix(value: str) -> str:
    lower = value.lower()

    if lower.endswith((".npz", ".npy", ".nii", ".nii.gz")):
        return value

    return value + ".npz"


def dataset_name_from_item(value: str) -> str:
    value = value.replace("\\", "/")
    return value.split("/")[0]


def load_manifest(
    split_path: Path,
    split: str,
    dataset: Optional[str],
) -> list[dict[str, Any]]:
    if not split_path.exists():
        raise FileNotFoundError(f"Manifest not found: {split_path}")

    with split_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("splits.json must contain a dictionary")

    if split not in data:
        raise KeyError(
            f"Split {split!r} not found. "
            f"Available splits: {list(data.keys())}"
        )

    items = data[split]

    if not isinstance(items, list):
        raise ValueError(f"Manifest split {split!r} must be a list")

    records: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        if isinstance(item, str):
            item = add_npz_suffix(item)

            if dataset is not None:
                current_dataset = dataset_name_from_item(item)

                if current_dataset.lower() != dataset.lower():
                    continue

            records.append(
                {
                    "id": Path(item).stem,
                    "img": item,
                    "mask": item,
                }
            )
            continue

        if isinstance(item, dict):
            record = dict(item)

            image_value = (
                record.get("img")
                or record.get("image")
                or record.get("image_path")
                or record.get("image_file")
                or record.get("ct")
            )

            if image_value is None:
                raise KeyError(
                    f"Manifest record {index} has no image field: {record}"
                )

            if dataset is not None:
                current_dataset = dataset_name_from_item(str(image_value))

                if current_dataset.lower() != dataset.lower():
                    continue

            records.append(record)
            continue

        raise ValueError(
            f"Invalid manifest record at index {index}: {item!r}"
        )

    if not records:
        suffix = f" for dataset {dataset!r}" if dataset else ""
        raise RuntimeError(
            f"No records found in split {split!r}{suffix}"
        )

    return records


def load_npz_array(
    path: Path,
    key: str,
    fallback_keys: Sequence[str],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.name.lower().endswith(".npz"):
        if path.name.lower().endswith(".npy"):
            return np.asarray(np.load(path, allow_pickle=False))

        if path.name.lower().endswith((".nii", ".nii.gz")):
            try:
                import nibabel as nib
            except ImportError as exc:
                raise RuntimeError(
                    "Please install nibabel: pip install nibabel"
                ) from exc

            return np.asarray(nib.load(str(path)).get_fdata())

        raise ValueError(f"Unsupported file format: {path}")

    with np.load(path, allow_pickle=False) as data:
        candidates = [key, *fallback_keys, *list(data.keys())]

        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate and candidate in data
            ),
            None,
        )

        if selected is None:
            raise KeyError(
                f"Could not find key {key!r} in {path}. "
                f"Available keys: {list(data.keys())}"
            )

        return np.asarray(data[selected])


def get_record_paths(
    record: dict[str, Any],
    data_root: Path,
) -> tuple[Path, Path, str]:
    image_value = (
        record.get("img")
        or record.get("image")
        or record.get("image_path")
        or record.get("image_file")
        or record.get("ct")
    )

    label_value = (
        record.get("mask")
        or record.get("label")
        or record.get("seg")
        or record.get("label_path")
        or record.get("label_file")
    )

    if image_value is None:
        raise KeyError(f"Record has no image path: {record}")

    image_value = add_npz_suffix(str(image_value))
    image_path = resolve_path(image_value, data_root)

    if label_value is None:
        label_path = image_path
    else:
        label_value = add_npz_suffix(str(label_value))
        label_path = resolve_path(label_value, data_root)

    case_id = str(
        record.get("id")
        or record.get("case_id")
        or record.get("name")
        or Path(image_value).stem
    )

    return image_path, label_path, case_id


def ct_to_hwd(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    array = np.squeeze(array)

    if array.ndim != 3:
        raise ValueError(
            f"CT must be 3-D after squeeze, got shape {array.shape}"
        )

    return array.astype(np.float32, copy=False)


def label_to_dhw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    array = np.squeeze(array)

    if array.ndim != 3:
        raise ValueError(
            f"Mask must be 3-D after squeeze, got shape {array.shape}"
        )

    # Dataset preprocessing uses:
    # ct = torch.from_numpy(ct).permute(2, 0, 1)
    #
    # Therefore the NPZ layout is H,W,D and model layout is D,H,W.
    return array.transpose(2, 0, 1).astype(
        np.float32,
        copy=False,
    )


def build_salience_generator() -> Any:
    try:
        from src.salience.plugin_normalized import SalienceGenerator

        print("Using src.salience.plugin_normalized.SalienceGenerator")
        return SalienceGenerator()
    except ImportError:
        pass

    try:
        from src.salience.plugin import SalienceGenerator

        print("Using src.salience.plugin.SalienceGenerator")
        return SalienceGenerator()
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import SalienceGenerator. Expected either:\n"
            "  src/salience/plugin_normalized.py\n"
            "  src/salience/plugin.py"
        ) from exc


def normalize_ct_for_salience(
    ct_hwd: torch.Tensor,
) -> torch.Tensor:
    ct_hwd = torch.nan_to_num(
        ct_hwd.float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    value_min = ct_hwd.min()
    value_max = ct_hwd.max()

    if float(value_min) < 0.0 or float(value_max) > 1.0:
        ct_hwd = (ct_hwd - value_min) / (
            value_max - value_min + 1e-6
        )

    return torch.clamp(ct_hwd, 0.0, 1.0)


@torch.no_grad()
def build_model_input(
    ct_hwd_np: np.ndarray,
    expected_channels: int,
    salience_generator: Any,
    device: torch.device,
) -> torch.Tensor:
    """Return model input with shape [C,D,H,W]."""
    ct_hwd = torch.from_numpy(ct_hwd_np).float()
    ct_hwd = normalize_ct_for_salience(ct_hwd)

    if expected_channels == 1:
        # H,W,D -> C,D,H,W
        return ct_hwd.permute(2, 0, 1).unsqueeze(0).to(device)

    if expected_channels != 3:
        raise RuntimeError(
            f"Unsupported model input channel count: {expected_channels}. "
            "This script supports 1-channel baseline and "
            "3-channel Salience models."
        )

    if salience_generator is None:
        salience_generator = build_salience_generator()

    # SalienceGenerator expects H,W,D and returns 3,H,W,D.
    salience_input = ct_hwd.cpu()
    enhanced_hwd = salience_generator(salience_input)

    if not isinstance(enhanced_hwd, torch.Tensor):
        enhanced_hwd = torch.as_tensor(enhanced_hwd)

    if enhanced_hwd.ndim != 4:
        raise RuntimeError(
            "SalienceGenerator must return [3,H,W,D], "
            f"got {tuple(enhanced_hwd.shape)}"
        )

    if enhanced_hwd.shape[0] != 3:
        raise RuntimeError(
            "SalienceGenerator must return three channels, "
            f"got {enhanced_hwd.shape[0]}"
        )

    # [3,H,W,D] -> [3,D,H,W]
    model_input = enhanced_hwd.permute(0, 3, 1, 2).contiguous()

    return model_input.float().to(device)


def pad_to_shape(
    tensor: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    _, _, depth, height, width = tensor.shape

    pad_d = target_shape[0] - depth
    pad_h = target_shape[1] - height
    pad_w = target_shape[2] - width

    if pad_d < 0 or pad_h < 0 or pad_w < 0:
        raise ValueError("Target padding shape is smaller than input shape")

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return tensor

    return F.pad(
        tensor,
        (0, pad_w, 0, pad_h, 0, pad_d),
        mode="replicate",
    )


def starts_for(size: int, roi: int, step: int) -> list[int]:
    if size <= roi:
        return [0]

    starts = list(range(0, size - roi + 1, step))
    last = size - roi

    if starts[-1] != last:
        starts.append(last)

    return starts


def model_output_to_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Model returned an empty tuple/list")
        output = output[0]

    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)

    if output.ndim != 5:
        raise RuntimeError(
            f"Expected model output [B,C,D,H,W], got {tuple(output.shape)}"
        )

    return output


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: Sequence[int],
    overlap: float,
) -> np.ndarray:
    if not 0.0 <= overlap < 1.0:
        raise ValueError("--overlap must be in [0, 1)")

    roi = tuple(int(value) for value in roi_size)

    if len(roi) != 3 or any(value <= 0 for value in roi):
        raise ValueError(f"Invalid ROI size: {roi}")

    if image.ndim != 4:
        raise ValueError(
            f"Image must have shape [C,D,H,W], got {tuple(image.shape)}"
        )

    channels, depth, height, width = image.shape

    original_shape = (depth, height, width)

    padded_shape = tuple(
        max(size, window)
        for size, window in zip(original_shape, roi)
    )

    image = pad_to_shape(
        image.unsqueeze(0),
        padded_shape,
    )

    steps = tuple(
        max(1, int(window * (1.0 - overlap)))
        for window in roi
    )

    starts = [
        starts_for(size, window, step)
        for size, window, step in zip(padded_shape, roi, steps)
    ]

    logits_sum: Optional[torch.Tensor] = None

    weight = torch.zeros(
        (1, 1, *padded_shape),
        device=image.device,
        dtype=torch.float32,
    )

    for start_d in starts[0]:
        for start_h in starts[1]:
            for start_w in starts[2]:
                patch = image[
                    :,
                    :,
                    start_d : start_d + roi[0],
                    start_h : start_h + roi[1],
                    start_w : start_w + roi[2],
                ]

                output = model_output_to_tensor(model(patch))

                if output.shape[0] != 1:
                    raise RuntimeError(
                        f"Expected batch size 1, got {output.shape[0]}"
                    )

                if output.shape[1] < 1:
                    raise RuntimeError("Model has no output channel")

                if tuple(output.shape[-3:]) != roi:
                    raise RuntimeError(
                        f"Model output shape {tuple(output.shape[-3:])} "
                        f"does not match ROI {roi}"
                    )

                if logits_sum is None:
                    logits_sum = torch.zeros(
                        (1, output.shape[1], *padded_shape),
                        device=output.device,
                        dtype=torch.float32,
                    )

                logits_sum[
                    :,
                    :,
                    start_d : start_d + roi[0],
                    start_h : start_h + roi[1],
                    start_w : start_w + roi[2],
                ] += output.float()

                weight[
                    :,
                    :,
                    start_d : start_d + roi[0],
                    start_h : start_h + roi[1],
                    start_w : start_w + roi[2],
                ] += 1.0

    if logits_sum is None:
        raise RuntimeError("No sliding-window patch was processed")

    logits = logits_sum / weight.clamp_min(1.0)

    probability = torch.sigmoid(
        logits[0, 0, :depth, :height, :width]
    )

    return probability.cpu().numpy()


def dice_score(
    prediction: np.ndarray,
    label: np.ndarray,
) -> float:
    prediction = prediction.astype(bool, copy=False)
    label = label.astype(bool, copy=False)

    intersection = np.logical_and(
        prediction,
        label,
    ).sum(dtype=np.float64)

    denominator = (
        prediction.sum(dtype=np.float64)
        + label.sum(dtype=np.float64)
    )

    if denominator == 0:
        return 1.0

    return float(2.0 * intersection / denominator)


def representative_slice(
    prediction: np.ndarray,
    label: np.ndarray,
    mode: str,
) -> tuple[int, float]:
    if mode == "center":
        index = prediction.shape[0] // 2
        return index, dice_score(prediction[index], label[index])

    if mode == "largest":
        areas = label.reshape(label.shape[0], -1).sum(axis=1)
        index = int(np.argmax(areas))
        return index, dice_score(prediction[index], label[index])

    scores = np.asarray(
        [
            dice_score(prediction[index], label[index])
            for index in range(prediction.shape[0])
        ]
    )

    nonempty = label.reshape(label.shape[0], -1).sum(axis=1) > 0

    if nonempty.any():
        scores = np.where(nonempty, scores, -1.0)

    index = int(np.argmax(scores))

    return index, float(scores[index])


def overlay_rgb(
    prediction: np.ndarray,
    label: np.ndarray,
) -> np.ndarray:
    prediction = prediction.astype(bool, copy=False)
    label = label.astype(bool, copy=False)

    union = np.logical_or(prediction, label)
    intersection = np.logical_and(prediction, label)

    rgb = np.zeros(
        (*prediction.shape, 3),
        dtype=np.float32,
    )

    rgb[union] = (0.02, 0.12, 0.42)
    rgb[intersection] = (1.0, 0.42, 0.02)

    return rgb


def safe_name(value: str) -> str:
    value = re.sub(
        r"[^0-9A-Za-z._-]+",
        "_",
        value,
    ).strip("._")

    return value or "case"


def save_case_png(
    path: Path,
    prediction: np.ndarray,
    label: np.ndarray,
    case_id: str,
    volume_dice: float,
    slice_index: int,
    slice_dice: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(
        figsize=(5, 5),
        facecolor="black",
    )

    axis.imshow(
        overlay_rgb(
            prediction[slice_index],
            label[slice_index],
        ),
        interpolation="nearest",
    )

    axis.set_axis_off()
    axis.set_facecolor("black")

    axis.set_title(
        f"{case_id}\n"
        f"volume DSC={volume_dice:.4f} | "
        f"slice={slice_index} DSC={slice_dice:.4f}",
        color="white",
        fontsize=9,
    )

    figure.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.86,
    )

    figure.savefig(
        path,
        dpi=180,
        facecolor="black",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(figure)


def save_grid(
    path: Path,
    entries: list[dict[str, Any]],
    columns: int = 4,
) -> None:
    if not entries:
        return

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    rows = math.ceil(len(entries) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3.2, rows * 3.3),
        squeeze=False,
        facecolor="black",
    )

    for axis in axes.flat:
        axis.set_axis_off()
        axis.set_facecolor("black")

    for axis, entry in zip(axes.flat, entries):
        axis.imshow(
            entry["rgb"],
            interpolation="nearest",
        )

        axis.set_title(
            f"#{entry['rank']} {entry['case_id']}\n"
            f"DSC={entry['dice']:.4f}, z={entry['slice_index']}",
            color="white",
            fontsize=8,
        )

    figure.subplots_adjust(
        wspace=0.04,
        hspace=0.18,
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
    )

    figure.savefig(
        path,
        dpi=180,
        facecolor="black",
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(figure)


def get_roi_size(
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> tuple[int, int, int]:
    infer_cfg = cfg.get("infer") or {}
    data_cfg = cfg.get("data") or {}

    roi_value = (
        args.roi_size
        or infer_cfg.get("roi_size")
        or infer_cfg.get("patch_size")
        or data_cfg.get("roi_size")
        or data_cfg.get("patch_size")
        or (96, 96, 96)
    )

    roi = tuple(int(value) for value in roi_value)

    if len(roi) != 3 or any(value <= 0 for value in roi):
        raise ValueError(f"Invalid ROI size: {roi}")

    return roi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top-K 3-D segmentation predictions"
    )

    parser.add_argument(
        "--cfg",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/preprocessed/splits.json"),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/preprocessed"),
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter: LiVS, MSD, or 3Dircadb1",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--roi-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("D", "H", "W"),
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--slice-mode",
        choices=("best", "largest", "center"),
        default="best",
    )

    parser.add_argument(
        "--image-key",
        default="ct",
    )

    parser.add_argument(
        "--label-key",
        default="mask",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = resolve_path(args.cfg, PROJECT_ROOT)
    ckpt_path = resolve_path(args.ckpt, PROJECT_ROOT)
    splits_path = resolve_path(args.splits, PROJECT_ROOT)
    data_root = resolve_path(args.data_root, PROJECT_ROOT)

    if args.out_dir is None:
        output_dir = ckpt_path.parent / "top20_plots"
    else:
        output_dir = resolve_path(args.out_dir, PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(cfg_path)

    if args.device is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device

    device = torch.device(device_name)

    checkpoint = torch.load(
        str(ckpt_path),
        map_location=device,
    )

    model, expected_channels = build_model(
        cfg,
        checkpoint,
        device,
    )

    records = load_manifest(
        split_path=splits_path,
        split=args.split,
        dataset=args.dataset,
    )

    roi_size = get_roi_size(args, cfg)

    salience_generator = None

    if expected_channels == 3:
        salience_generator = build_salience_generator()

    print(f"device={device}")
    print(f"checkpoint={ckpt_path}")
    print(f"manifest={splits_path}")
    print(f"data_root={data_root}")
    print(f"dataset={args.dataset or 'all'}")
    print(f"split={args.split}")
    print(f"cases={len(records)}")
    print(f"model input channels={expected_channels}")
    print(f"roi_size={roi_size}")
    print(f"overlap={args.overlap}")
    print(f"threshold={args.threshold}")

    results: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        image_path, label_path, case_id = get_record_paths(
            record,
            data_root,
        )

        ct = load_npz_array(
            image_path,
            key=args.image_key,
            fallback_keys=("ct", "img", "image", "data"),
        )

        mask = load_npz_array(
            label_path,
            key=args.label_key,
            fallback_keys=("mask", "label", "seg", "target"),
        )

        ct_hwd = ct_to_hwd(ct)
        label_dhw = label_to_dhw(mask)
        label_volume = label_dhw > 0.5

        image = build_model_input(
            ct_hwd_np=ct_hwd,
            expected_channels=expected_channels,
            salience_generator=salience_generator,
            device=device,
        )

        if tuple(image.shape[-3:]) != tuple(label_volume.shape):
            raise ValueError(
                f"Spatial mismatch for {case_id}: "
                f"model input={tuple(image.shape[-3:])}, "
                f"label={tuple(label_volume.shape)}"
            )

        probability = predict_volume(
            model=model,
            image=image,
            roi_size=roi_size,
            overlap=args.overlap,
        )

        prediction = probability >= args.threshold

        volume_dice = dice_score(
            prediction,
            label_volume,
        )

        slice_index, slice_dice = representative_slice(
            prediction,
            label_volume,
            args.slice_mode,
        )

        results.append(
            {
                "case_id": case_id,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "dice": volume_dice,
                "slice_index": slice_index,
                "slice_dice": slice_dice,
                "prediction": prediction,
                "label": label_volume,
            }
        )

        print(
            f"[{index}/{len(records)}] "
            f"{case_id}: DSC={volume_dice:.4f}, "
            f"slice={slice_index}"
        )

    results.sort(
        key=lambda item: (
            -item["dice"],
            item["case_id"],
        )
    )

    selected = results[: max(0, args.top_k)]
    grid_entries: list[dict[str, Any]] = []

    for rank, entry in enumerate(selected, start=1):
        entry["rank"] = rank

        entry["rgb"] = overlay_rgb(
            entry["prediction"][entry["slice_index"]],
            entry["label"][entry["slice_index"]],
        )

        output_path = output_dir / (
            f"rank_{rank:02d}_"
            f"{safe_name(entry['case_id'])}.png"
        )

        save_case_png(
            path=output_path,
            prediction=entry["prediction"],
            label=entry["label"],
            case_id=entry["case_id"],
            volume_dice=entry["dice"],
            slice_index=entry["slice_index"],
            slice_dice=entry["slice_dice"],
        )

        grid_entries.append(entry)

    grid_path = output_dir / "top20_grid.png"

    save_grid(
        path=grid_path,
        entries=grid_entries,
    )

    csv_path = output_dir / "top20_ranking.csv"

    fieldnames = (
        "rank",
        "case_id",
        "dice",
        "slice_index",
        "slice_dice",
        "image_path",
        "label_path",
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for entry in selected:
            writer.writerow(
                {
                    key: entry[key]
                    for key in fieldnames
                }
            )

    print(f"saved {len(selected)} case images to {output_dir}")
    print(f"grid: {grid_path}")
    print(f"ranking: {csv_path}")


if __name__ == "__main__":
    main()
