#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/preprocess.py

Unified preprocessing for the 3Dircadb1, MSD, and LiVS datasets.
- PNG series (3Dircadb1, MSD) → read as uint16/uint8 → normalize → center crop → .npz
- NIfTI series (LiVS)        → read as float32      → same       → center crop → .npz

Typical usage:
    python scripts/preprocess.py \
        --raw_root   data/raw \
        --out_root   data/preprocessed \
        --crop_size  256 256 \
        --seed       2024

Use `--datasets 3Dircadb1,MSD,LiVS` to specify which subsets to process (default: all).
"""

import os
import argparse
import json
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import cv2               # for PNG reading (fast)
import nibabel as nib
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------
# ---------------------- Parameters & Constants -----------------
# --------------------------------------------------------------
CLIP_MIN = 0          # lower HU bound
CLIP_MAX = 400        # upper HU bound
CROP_SIZE_DEFAULT = (256, 256)   # (height, width)
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SUPPORTED_DATASETS = {"3Dircadb1", "MSD", "LiVS"}

# --------------------------------------------------------------
# ---------------------- Utility Functions ---------------------
# --------------------------------------------------------------

def clip_and_norm(volume: np.ndarray,
                 clip_min: int = CLIP_MIN,
                 clip_max: int = CLIP_MAX) -> np.ndarray:
    """
    Clip to [clip_min, clip_max] (HU) and linearly scale to [0,1].
    Returns float32 array.
    """
    vol = np.clip(volume, clip_min, clip_max)
    vol = vol.astype(np.float32)
    # Prevent division by zero (extreme case: all zeros or all 400)
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(vol, dtype=np.float32)
    vol = (vol - vmin) / (vmax - vmin)
    return vol


def center_crop(volume: np.ndarray,
                target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Crop the (H,W) plane of a 3‑D volume to target_hw = (h, w)
    by removing equal margins on both sides.
    volume shape: (H, W, D)  or  (H, W) for 2‑D images.
    """
    h_target, w_target = target_hw
    H, W = volume.shape[:2]
    if H < h_target or W < w_target:
        raise ValueError(f"Volume size ({H},{W}) smaller than target crop {target_hw}")

    top  = (H - h_target) // 2
    left = (W - w_target) // 2
    if volume.ndim == 3:
        return volume[top:top + h_target, left:left + w_target, :]
    else:   # 2‑D (only occurs in rare abnormal cases)
        return volume[top:top + h_target, left:left + w_target]


def save_npz(out_dir: Path, case_id: str,
             ct: np.ndarray, mask: np.ndarray):
    """
    Save as compressed npz; keys must be `ct` and `mask`
    (the Dataset class relies on these names).
    """
    out_path = out_dir / f"{case_id}.npz"
    np.savez_compressed(out_path, ct=ct, mask=mask)


# --------------------------------------------------------------
# ---------------------- Dataset‑Specific Processing ------------
# --------------------------------------------------------------

def _process_png_folder(img_dir: Path, mask_dir: Path,
                       out_dir: Path, crop_hw: Tuple[int, int],
                       dataset_name: str) -> List[str]:
    """
    Read PNGs → build 3‑D volume → preprocess → save .npz.
    Returns the list of case IDs (without suffix) processed.
    """
    case_ids = []

    # Collect all image files (assuming consistent naming)
    img_files = sorted([p for p in img_dir.glob("*.png")])
    mask_files = sorted([p for p in mask_dir.glob("*.png")])

    # Simple filename matching check
    if len(img_files) != len(mask_files):
        warnings.warn(
            f"[{dataset_name}] Number of images ({len(img_files)}) does not match masks ({len(mask_files)}), "
            "will attempt to match by filename prefix."
        )

    # Build prefix → path mappings; prefix = full filename without extension
    img_dict = {p.stem: p for p in img_files}
    mask_dict = {p.stem: p for p in mask_files}
    common_keys = sorted(set(img_dict.keys()) & set(mask_dict.keys()))
    if not common_keys:
        raise RuntimeError(f"[{dataset_name}] No matching image-mask pairs found. Please check file naming.")

    for key in tqdm(common_keys, desc=f"Preprocess {dataset_name}", unit="case"):
        img_path = img_dict[key]
        mask_path = mask_dict[key]

        # Read PNG (OpenCV reads as BGR; convert to grayscale)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)   # shape (H,W) or (H,W,3)
        if img is None:
            raise IOError(f"Failed to read image: {img_path}")
        if img.ndim == 3:   # some data may be 3‑channel PNG; take the first channel
            img = img[..., 0]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise IOError(f"Failed to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]

        # PNG data is already 0‑255 (or 0‑65535); convert to float directly
        img = img.astype(np.float32)
        mask = (mask > 0).astype(np.uint8)   # binarize

        # ① Clip & Normalize
        img = clip_and_norm(img, CLIP_MIN, CLIP_MAX)

        # ② PNGs are single slices; stack them into 3‑D (z direction)
        #    For multi‑slice cases, filenames usually contain slice indices.
        #    For simplicity, we assume each PNG is already a full 3‑D volume
        #    (e.g., 3Dircadb1.3_image_130.png). So we treat the single slice
        #    as a volume with depth 1.
        #    To keep the code compatible, we add a singleton z‑dimension.
        img = img[..., np.newaxis]      # (H,W,1)
        mask = mask[..., np.newaxis]    # (H,W,1)

        # ③ Center crop (H,W) → (crop_h, crop_w)
        img = center_crop(img, crop_hw)     # (crop_h, crop_w, 1)
        mask = center_crop(mask, crop_hw)

        # ④ Remove the singleton dimension to obtain (H,W,D)
        img = np.squeeze(img, axis=2)       # (H,W)
        mask = np.squeeze(mask, axis=2)

        # ⑤ Save
        case_id = key                     # use the prefix directly as ID
        save_npz(out_dir, case_id, img, mask)
        case_ids.append(case_id)

    return case_ids


def process_3Dircadb1(raw_root: Path, out_root: Path,
                     crop_hw: Tuple[int, int]) -> List[str]:
    """
    3Dircadb1 directory structure:
        raw_root/3Dircadb1/train/
        raw_root/3Dircadb1/trainmask/
    """
    img_dir = raw_root / "3Dircadb1" / "train"
    mask_dir = raw_root / "3Dircadb1" / "trainmask"
    out_dir = out_root / "3Dircadb1"
    out_dir.mkdir(parents=True, exist_ok=True)

    return _process_png_folder(img_dir, mask_dir, out_dir, crop_hw, "3Dircadb1")


def process_MSD(raw_root: Path, out_root: Path,
                crop_hw: Tuple[int, int]) -> List[str]:
    """
    MSD (Medical Segmentation Decathlon) hepaticvessel subset structure:
        raw_root/MSD/train/
        raw_root/MSD/trainmask/
    """
    img_dir = raw_root / "MSD" / "train"
    mask_dir = raw_root / "MSD" / "trainmask"
    out_dir = out_root / "MSD"
    out_dir.mkdir(parents=True, exist_ok=True)

    return _process_png_folder(img_dir, mask_dir, out_dir, crop_hw, "MSD")


def process_LiVS(raw_root: Path, out_root: Path,
                 crop_hw: Tuple[int, int]) -> List[str]:
    """
    LiVS (Liver Vessel Segmentation) structure:
        raw_root/LiVS/my_train_nii_file/
            patient_t1.nii.gz   -> raw CT
            patient_seg.nii.gz   -> corresponding mask
    Assumes each patient has a pair of files with the same prefix (patient).
    """
    nii_dir = raw_root / "LiVS" / "my_train_nii_file"
    out_dir = out_root / "LiVS"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all *_t1.nii.gz files as raw CT
    ct_paths = sorted(nii_dir.glob("*_t1.nii.gz"))
    case_ids = []

    for ct_path in tqdm(ct_paths, desc="Preprocess LiVS", unit="case"):
        case_id = ct_path.stem.replace("_t1.nii", "")   # patient
        seg_path = nii_dir / f"{case_id}_seg.nii.gz"
        if not seg_path.is_file():
            warnings.warn(f"Mask not found: {seg_path}, skipping this case.")
            continue

        # Read NIfTI
        ct_nii = nib.load(str(ct_path))
        seg_nii = nib.load(str(seg_path))

        ct = ct_nii.get_fdata(dtype=np.float32)   # shape (H,W,D)
        mask = seg_nii.get_fdata()               # shape (H,W,D)

        # ① Clip & Normalize
        ct = clip_and_norm(ct, CLIP_MIN, CLIP_MAX)

        # ② Binarize mask (sometimes masks contain labels 0/1/2 etc.)
        mask = (mask > 0).astype(np.uint8)

        # ③ Center crop (H,W) → (crop_h, crop_w)
        ct = center_crop(ct, crop_hw)      # (crop_h, crop_w, D)
        mask = center_crop(mask, crop_hw)

        # ④ Save
        save_npz(out_dir, case_id, ct, mask)
        case_ids.append(case_id)

    return case_ids


# --------------------------------------------------------------
# ---------------------- Splitting & Main Function -------------
# --------------------------------------------------------------

def split_cases(all_cases: List[str],
                ratios: dict = RATIOS,
                seed: int = 42) -> dict:
    """
    Randomly split into train / val / test, returning {"train": [...], "val": [...], "test": [...]}
    """
    random.seed(seed)
    shuffled = all_cases.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(ratios["train"] * n)
    n_val   = int(ratios["val"]   * n)

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess 3Dircadb1 / MSD / LiVS datasets into .npz files "
                    "and generate train/val/test splits."
    )
    parser.add_argument("--raw_root", type=str, default="data/raw",
                        help="Root directory that must contain 3Dircadb1, MSD, and LiVS subfolders")
    parser.add_argument("--out_root", type=str, default="data/preprocessed",
                        help="Root directory for saving preprocessed files")
    parser.add_argument("--crop_size", nargs=2, type=int, default=CROP_SIZE_DEFAULT,
                        help="Center crop size (height width), default 256 256")
    parser.add_argument("--datasets", type=str, default="3Dircadb1,MSD,LiVS",
                        help="Comma-separated dataset names to process; supports 3Dircadb1,MSD,LiVS")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed for train/val/test split")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    crop_hw = tuple(args.crop_size)   # (h, w)
    selected = set([d.strip() for d in args.datasets.split(",")])

    # ----------------------------------------------------------
    # 1) Process each dataset and collect all case IDs
    # ----------------------------------------------------------
    all_cases = []          # for global splitting (across datasets)
    dataset_case_map = {}   # record case list per dataset for later verification

    if "3Dircadb1" in selected:
        cases = process_3Dircadb1(raw_root, out_root, crop_hw)
        dataset_case_map["3Dircadb1"] = cases
        all_cases.extend([f"3Dircadb1/{c}" for c in cases])

    if "MSD" in selected:
        cases = process_MSD(raw_root, out_root, crop_hw)
        dataset_case_map["MSD"] = cases
        all_cases.extend([f"MSD/{c}" for c in cases])

    if "LiVS" in selected:
        cases = process_LiVS(raw_root, out_root, crop_hw)
        dataset_case_map["LiVS"] = cases
        all_cases.extend([f"LiVS/{c}" for c in cases])

    if not all_cases:
        raise RuntimeError("No cases detected to process. Please check `--datasets` or the raw file structure.")

    # ----------------------------------------------------------
    # 2) Perform global split (across datasets) and write splits.json
    # ----------------------------------------------------------
    splits = split_cases(all_cases, RATIOS, seed=args.seed)

    split_path = out_root / "splits.json"
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\n=== Preprocessing complete ===")
    print(f"Output directory: {out_root}")
    print(f"Total cases: {len(all_cases)}")
    for k, v in splits.items():
        print(f"  {k:5s}: {len(v)} cases")
    print(f"Split information saved to: {split_path}")

    # ----------------------------------------------------------
    # 3) (Optional) Print per‑dataset statistics for inspection
    # ----------------------------------------------------------
    print("\n--- Per‑dataset case counts (for reference) ---")
    for ds, lst in dataset_case_map.items():
        print(f"{ds:10s}: {len(lst)} cases")
    print("\nIf you wish to train only on a single dataset, modify the `load_dataset` "
          "function in `src/utils/io.py` to use the corresponding subfolder or adapt the split file accordingly.")

if __name__ == "__main__":
    main()
