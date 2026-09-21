# scripts/stack_png_to_volume.py
import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


CLIP_MIN = 0
CLIP_MAX = 400
DEFAULT_CROP_SIZE = (256, 256)
DEFAULT_RATIOS = (0.7, 0.15, 0.15)


def clip_and_norm(volume: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    volume = np.clip(volume, clip_min, clip_max).astype(np.float32)
    vmin = float(volume.min())
    vmax = float(volume.max())

    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)

    return (volume - vmin) / (vmax - vmin)


def center_crop_hwd(volume: np.ndarray, crop_hw: Tuple[int, int]) -> np.ndarray:
    crop_h, crop_w = crop_hw
    h, w = volume.shape[:2]

    if h < crop_h or w < crop_w:
        raise ValueError(
            f"Input size {(h, w)} is smaller than crop size {crop_hw}. "
            "Please use a smaller --crop_size."
        )

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    if volume.ndim == 2:
        return volume[top:top + crop_h, left:left + crop_w]

    if volume.ndim == 3:
        return volume[top:top + crop_h, left:left + crop_w, :]

    raise ValueError(f"Expected 2D or 3D array, got shape {volume.shape}")


def read_png_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Failed to read PNG: {path}")

    if image.ndim == 3:
        image = image[..., 0]

    return image


def parse_3dircadb1_name(path: Path) -> Tuple[str, int]:
    """
    Example:
        3Dircadb1.3_image_130.png -> ("3Dircadb1.3", 130)
    """
    match = re.match(r"^(3Dircadb1\.\d+)_image_(\d+)$", path.stem)
    if match is None:
        raise ValueError(
            f"Invalid 3Dircadb1 filename: {path.name}. "
            "Expected format like 3Dircadb1.3_image_130.png"
        )

    case_id = match.group(1)
    slice_id = int(match.group(2))
    return case_id, slice_id


def parse_msd_name(path: Path) -> Tuple[str, int]:
    """
    Example:
        hepaticvessel_265_27.png -> ("hepaticvessel_265", 27)
    """
    match = re.match(r"^(hepaticvessel_\d+)_(\d+)$", path.stem)
    if match is None:
        raise ValueError(
            f"Invalid MSD filename: {path.name}. "
            "Expected format like hepaticvessel_265_27.png"
        )

    case_id = match.group(1)
    slice_id = int(match.group(2))
    return case_id, slice_id


def group_png_files(
    image_dir: Path,
    mask_dir: Path,
    parser,
    dataset_name: str,
) -> Dict[str, List[Tuple[int, Path, Path]]]:
    image_paths = sorted(image_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))

    if not image_paths:
        raise RuntimeError(f"No PNG images found in {image_dir}")
    if not mask_paths:
        raise RuntimeError(f"No PNG masks found in {mask_dir}")

    mask_map = {p.name: p for p in mask_paths}
    grouped = defaultdict(list)

    missing_masks = []

    for image_path in image_paths:
        mask_path = mask_map.get(image_path.name)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue

        case_id, slice_id = parser(image_path)
        grouped[case_id].append((slice_id, image_path, mask_path))

    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        raise RuntimeError(
            f"[{dataset_name}] Missing masks for {len(missing_masks)} images. "
            f"Examples: {preview}"
        )

    return dict(grouped)


def stack_one_case(
    items: List[Tuple[int, Path, Path]],
    crop_hw: Tuple[int, int],
    clip_min: float,
    clip_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    items = sorted(items, key=lambda x: x[0])

    image_slices = []
    mask_slices = []

    expected_shape = None

    for slice_id, image_path, mask_path in items:
        image = read_png_gray(image_path).astype(np.float32)
        mask = read_png_gray(mask_path)

        if expected_shape is None:
            expected_shape = image.shape

        if image.shape != expected_shape:
            raise ValueError(
                f"Inconsistent image shape in case. "
                f"Expected {expected_shape}, got {image.shape}: {image_path}"
            )

        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape does not match image shape at slice {slice_id}. "
                f"Image {image.shape}, mask {mask.shape}: {mask_path}"
            )

        image = center_crop_hwd(image, crop_hw)
        mask = center_crop_hwd(mask, crop_hw)

        image_slices.append(image)
        mask_slices.append((mask > 0).astype(np.uint8))

    ct = np.stack(image_slices, axis=-1)
    mask = np.stack(mask_slices, axis=-1)

    ct = clip_and_norm(ct, clip_min, clip_max).astype(np.float32)
    mask = mask.astype(np.uint8)

    return ct, mask


def save_case(out_dir: Path, case_id: str, ct: np.ndarray, mask: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.npz"
    np.savez_compressed(out_path, ct=ct, mask=mask)


def process_png_dataset(
    raw_root: Path,
    out_root: Path,
    dataset_name: str,
    parser,
    crop_hw: Tuple[int, int],
    clip_min: float,
    clip_max: float,
    clean_output: bool,
) -> List[str]:
    image_dir = raw_root / dataset_name / "train"
    mask_dir = raw_root / dataset_name / "trainmask"
    out_dir = out_root / dataset_name

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    if clean_output and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_png_files(
        image_dir=image_dir,
        mask_dir=mask_dir,
        parser=parser,
        dataset_name=dataset_name,
    )

    case_ids = []

    for case_id, items in tqdm(
        sorted(grouped.items()),
        desc=f"Stack {dataset_name}",
        unit="case",
    ):
        ct, mask = stack_one_case(
            items=items,
            crop_hw=crop_hw,
            clip_min=clip_min,
            clip_max=clip_max,
        )

        save_case(out_dir, case_id, ct, mask)
        case_ids.append(case_id)

    print(f"[{dataset_name}] Saved {len(case_ids)} cases to {out_dir}")
    return case_ids


def collect_existing_livs_cases(out_root: Path) -> List[str]:
    livs_dir = out_root / "LiVS"
    if not livs_dir.is_dir():
        return []

    return sorted([p.stem for p in livs_dir.glob("*.npz")])


def split_cases(
    case_keys: List[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[str]]:
    train_ratio, val_ratio, test_ratio = ratios
    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    rng = random.Random(seed)
    case_keys = list(case_keys)
    rng.shuffle(case_keys)

    n = len(case_keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": case_keys[:n_train],
        "val": case_keys[n_train:n_train + n_val],
        "test": case_keys[n_train + n_val:],
    }


def write_splits(
    out_root: Path,
    dircadb_cases: List[str],
    msd_cases: List[str],
    include_livs: bool,
    ratios: Tuple[float, float, float],
    seed: int,
):
    all_case_keys = []

    all_case_keys.extend([f"3Dircadb1/{case_id}" for case_id in dircadb_cases])
    all_case_keys.extend([f"MSD/{case_id}" for case_id in msd_cases])

    if include_livs:
        livs_cases = collect_existing_livs_cases(out_root)
        all_case_keys.extend([f"LiVS/{case_id}" for case_id in livs_cases])
        print(f"[LiVS] Found {len(livs_cases)} existing cases in {out_root / 'LiVS'}")

    if not all_case_keys:
        raise RuntimeError("No cases available for split generation.")

    splits = split_cases(all_case_keys, ratios=ratios, seed=seed)

    split_path = out_root / "splits.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"[Split] Saved splits to {split_path}")
    print(
        f"[Split] train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stack 3Dircadb1/MSD PNG slices into 3D NPZ volumes."
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="data/raw",
        help="Raw data root directory.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/preprocessed",
        help="Preprocessed output root directory.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="3Dircadb1,MSD",
        help="Comma-separated datasets to process: 3Dircadb1,MSD.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs=2,
        default=DEFAULT_CROP_SIZE,
        metavar=("H", "W"),
        help="Center crop size, default: 256 256.",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=CLIP_MIN,
        help="Intensity clipping lower bound.",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=CLIP_MAX,
        help="Intensity clipping upper bound.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for train/val/test split.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=DEFAULT_RATIOS,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test ratios, default: 0.7 0.15 0.15.",
    )
    parser.add_argument(
        "--no_livs",
        action="store_true",
        help="Do not include existing LiVS cases when regenerating splits.json.",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Do not clean existing output folders for selected PNG datasets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
    unsupported = selected - {"3Dircadb1", "MSD"}
    if unsupported:
        raise ValueError(f"Unsupported datasets: {sorted(unsupported)}")

    crop_hw = (args.crop_size[0], args.crop_size[1])
    clean_output = not args.no_clean

    dircadb_cases = []
    msd_cases = []

    if "3Dircadb1" in selected:
        dircadb_cases = process_png_dataset(
            raw_root=raw_root,
            out_root=out_root,
            dataset_name="3Dircadb1",
            parser=parse_3dircadb1_name,
            crop_hw=crop_hw,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            clean_output=clean_output,
        )

    if "MSD" in selected:
        msd_cases = process_png_dataset(
            raw_root=raw_root,
            out_root=out_root,
            dataset_name="MSD",
            parser=parse_msd_name,
            crop_hw=crop_hw,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            clean_output=clean_output,
        )

    write_splits(
        out_root=out_root,
        dircadb_cases=dircadb_cases,
        msd_cases=msd_cases,
        include_livs=not args.no_livs,
        ratios=tuple(args.ratios),
        seed=args.seed,
    )

    print("Done.")
    print(f"Output root: {out_root}")
    print("NPZ format: ct shape=(H, W, D), mask shape=(H, W, D)")


if __name__ == "__main__":
    main()

