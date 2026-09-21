# scripts/regenerate_splits_by_dataset.py
import argparse
import json
import random
from pathlib import Path


SUPPORTED_DATASETS = ["3Dircadb1", "MSD", "LiVS"]


def collect_cases(root_dir: Path, dataset: str):
    dataset_dir = root_dir / dataset
    if not dataset_dir.is_dir():
        print(f"[Warning] Dataset directory not found: {dataset_dir}")
        return []

    cases = sorted([p.stem for p in dataset_dir.glob("*.npz")])
    return [f"{dataset}/{case}" for case in cases]


def split_one_dataset(cases, train_ratio, val_ratio, seed):
    cases = list(cases)
    rng = random.Random(seed)
    rng.shuffle(cases)

    n = len(cases)

    if n == 0:
        return [], [], []

    if n == 1:
        return cases, [], []

    if n == 2:
        return cases[:1], [], cases[1:]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Ensure every dataset has at least one test case when possible.
    n_train = max(1, n_train)
    n_val = max(1, n_val) if n >= 4 else 0

    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]

    return train_cases, val_cases, test_cases


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate stratified train/val/test splits for each dataset."
    )
    parser.add_argument(
        "--root_dir",
        default="data/preprocessed",
        help="Preprocessed root directory.",
    )
    parser.add_argument(
        "--out",
        default="data/preprocessed/splits.json",
        help="Output splits.json path.",
    )
    parser.add_argument(
        "--datasets",
        default="3Dircadb1,MSD,LiVS",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    print("=== Stratified Split By Dataset ===")

    for dataset in datasets:
        cases = collect_cases(root_dir, dataset)
        train_cases, val_cases, test_cases = split_one_dataset(
            cases,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        splits["train"].extend(train_cases)
        splits["val"].extend(val_cases)
        splits["test"].extend(test_cases)

        print(
            f"{dataset:10s}: total={len(cases)}, "
            f"train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSaved splits to: {out_path}")


if __name__ == "__main__":
    main()

