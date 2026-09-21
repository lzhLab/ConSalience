# src/utils/io.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SUPPORTED_DATASETS = {"3Dircadb1", "MSD", "LiVS"}


def read_splits(split_path: str) -> Dict[str, List[str]]:
    split_path = Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    for key in ("train", "val", "test"):
        if key not in splits:
            raise KeyError(f"Missing key '{key}' in split file: {split_path}")

    return splits


def ensure_hwd(array: np.ndarray) -> np.ndarray:
    """
    Ensure array shape is (H, W, D).
    """
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected array shape (H, W, D), got {array.shape}")
    return array


class VolDataset(Dataset):
    """
    Load preprocessed .npz files.

    Returns:
        img:  FloatTensor, shape (1, D, H, W)
        mask: FloatTensor, shape (1, D, H, W)
    """

    def __init__(
        self,
        case_list: List[str],
        root_dir: str = "data/preprocessed",
    ):
        self.case_list = case_list
        self.root_dir = Path(root_dir)

        if len(self.case_list) == 0:
            raise RuntimeError("Empty case list. Please check dataset name and splits.json.")

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx: int):
        case_key = self.case_list[idx]

        if "/" not in case_key:
            raise ValueError(
                f"Invalid case key '{case_key}'. "
                "Expected format like 'MSD/hepaticvessel_286'."
            )

        dataset_name, case_id = case_key.split("/", 1)
        npz_path = self.root_dir / dataset_name / f"{case_id}.npz"

        if not npz_path.is_file():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path)
        ct = ensure_hwd(data["ct"]).astype(np.float32)
        mask = ensure_hwd(data["mask"]).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        # npz: (H, W, D) -> Conv3d: (C, D, H, W)
        ct = torch.from_numpy(ct).permute(2, 0, 1).unsqueeze(0).contiguous()
        mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).contiguous()

        return {
            "img": ct,
            "mask": mask,
            "case_id": case_key,
        }


def pad_to_shape_4d(tensor: torch.Tensor, target_shape: Tuple[int, int, int, int]):
    """
    Pad tensor from (C, D, H, W) to target_shape=(C, D, H, W).
    Padding order for F.pad on 4D tensor is: (W_left, W_right, H_left, H_right, D_left, D_right).
    """
    _, d, h, w = tensor.shape
    _, target_d, target_h, target_w = target_shape

    pad_d = target_d - d
    pad_h = target_h - h
    pad_w = target_w - w

    if pad_d < 0 or pad_h < 0 or pad_w < 0:
        raise ValueError(f"Cannot pad tensor shape {tensor.shape} to smaller shape {target_shape}")

    return F.pad(
        tensor,
        (
            0, pad_w,
            0, pad_h,
            0, pad_d,
        ),
        mode="constant",
        value=0,
    )


def pad_collate_fn(batch: List[Dict]):
    """
    Collate volumes with different D/H/W by padding to the max size in current batch.

    Input sample:
        img/mask: (1, D, H, W)

    Output batch:
        img/mask: (B, 1, max_D, max_H, max_W)
    """
    max_d = max(item["img"].shape[1] for item in batch)
    max_h = max(item["img"].shape[2] for item in batch)
    max_w = max(item["img"].shape[3] for item in batch)

    target_shape = (1, max_d, max_h, max_w)

    imgs = torch.stack([pad_to_shape_4d(item["img"], target_shape) for item in batch], dim=0)
    masks = torch.stack([pad_to_shape_4d(item["mask"], target_shape) for item in batch], dim=0)
    case_ids = [item["case_id"] for item in batch]

    return {
        "img": imgs,
        "mask": masks,
        "case_id": case_ids,
    }


def filter_cases_by_dataset(splits: Dict[str, List[str]], dataset_name: str):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: {sorted(SUPPORTED_DATASETS)}"
        )

    prefix = f"{dataset_name}/"

    train_cases = [x for x in splits["train"] if x.startswith(prefix)]
    val_cases = [x for x in splits["val"] if x.startswith(prefix)]
    test_cases = [x for x in splits["test"] if x.startswith(prefix)]

    return train_cases, val_cases, test_cases


def load_dataset(
    dataset_name: str,
    split_path: str = "data/preprocessed/splits.json",
    root_dir: str = "data/preprocessed",
) -> Tuple[Dataset, Dataset, Dataset]:
    splits = read_splits(split_path)
    train_cases, val_cases, test_cases = filter_cases_by_dataset(splits, dataset_name)

    train_set = VolDataset(train_cases, root_dir=root_dir)
    val_set = VolDataset(val_cases, root_dir=root_dir)
    test_set = VolDataset(test_cases, root_dir=root_dir)

    print(
        f"[Dataset] {dataset_name}: "
        f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )

    return train_set, val_set, test_set


if __name__ == "__main__":
    train_set, val_set, test_set = load_dataset(
        dataset_name="3Dircadb1",
        split_path="data/preprocessed/splits.json",
        root_dir="data/preprocessed",
    )

    sample = train_set[0]
    print("case_id:", sample["case_id"])
    print("img shape:", sample["img"].shape)
    print("mask shape:", sample["mask"].shape)

    batch = pad_collate_fn([train_set[0], train_set[1]])
    print("batch img shape:", batch["img"].shape)
    print("batch mask shape:", batch["mask"].shape)

