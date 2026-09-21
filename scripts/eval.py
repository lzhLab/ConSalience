# scripts/eval.py
# -*- coding: utf-8 -*-
import argparse
import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models.build_model import build_model
from src.salience.build_salience import build_salience_generator
from src.utils.io import load_dataset
from src.utils.metrics import compute_all_metrics


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3D segmentation model.")
    parser.add_argument("--cfg", required=True, help="Path to yaml config.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint, e.g. best.pth.")
    parser.add_argument(
        "--use_salience",
        type=str2bool,
        default=False,
        help="Whether to use salience input. Example: --use_salience true",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name, e.g. 3Dircadb1, MSD, LiVS.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Prediction threshold. If not set, use cfg['val']['threshold'] or 0.5.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resampling iterations for 95% CI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for bootstrap CI.",
    )
    parser.add_argument(
        "--save_csv",
        default=None,
        help="Optional path to save per-case metrics CSV.",
    )
    parser.add_argument(
        "--baseline_csv",
        default=None,
        help=(
            "Optional baseline CSV for p-value calculation. "
            "Format: case_id,DSC,clDice,IoU,Sensitivity,Specificity,HD95,ASSD"
        ),
    )
    return parser.parse_args()


def apply_salience_to_batch(imgs, salience_gen):
    """
    Convert original batch to salience-enhanced batch.

    Input:
        imgs: (B, 1, D, H, W)

    SalienceGenerator:
        input:  (H, W, D)
        output: (3, H, W, D)

    Return:
        (B, 3, D, H, W)
    """
    if salience_gen is None:
        return imgs

    enhanced_list = []

    with torch.no_grad():
        for b in range(imgs.size(0)):
            volume_dhw = imgs[b, 0]  # (D, H, W)
            volume_hwd = volume_dhw.permute(1, 2, 0).contiguous()  # (H, W, D)

            enhanced_hwd = salience_gen(volume_hwd)

            if not isinstance(enhanced_hwd, torch.Tensor):
                enhanced_hwd = torch.as_tensor(enhanced_hwd)

            enhanced_hwd = enhanced_hwd.to(
                device=imgs.device,
                dtype=imgs.dtype,
                non_blocking=True,
            )

            if enhanced_hwd.ndim != 4 or enhanced_hwd.size(0) != 3:
                raise ValueError(
                    "SalienceGenerator must return shape (3, H, W, D). "
                    f"Got {enhanced_hwd.shape}"
                )

            enhanced_dhw = enhanced_hwd.permute(0, 3, 1, 2).contiguous()
            enhanced_list.append(enhanced_dhw)

    return torch.stack(enhanced_list, dim=0).contiguous()


def load_checkpoint(model, ckpt_path, device, checkpoint=None):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if checkpoint is None:
        checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint val_dice: {checkpoint.get('val_dice', 'unknown')}")
        return checkpoint

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint: {ckpt_path}")
        return checkpoint

    model.load_state_dict(checkpoint)
    print(f"Loaded raw state_dict: {ckpt_path}")
    return checkpoint


# ---------------- clDice 计算（soft skeleton approximation）----------------

def _skeletonize_3d(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """
    Approximate soft 3D skeletonization used by clDice.

    x: (B, 1, D, H, W), values in [0, 1]
    return: same shape
    """
    x = x.float().clamp(0.0, 1.0)
    skeleton = torch.zeros_like(x)

    for _ in range(iterations):
        # 3D morphological erosion using min-pooling.
        eroded = -torch.nn.functional.max_pool3d(
            -x,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Opening: dilation after erosion.
        opened = torch.nn.functional.max_pool3d(
            eroded,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        delta = torch.relu(x - opened)
        skeleton = torch.maximum(skeleton, delta)
        x = eroded

    return skeleton


def _cldice_from_batch(preds: torch.Tensor, targets: torch.Tensor,
                       smooth: float = 1e-6, iterations: int = 10) -> float:
    """
    Compute soft clDice for binary 3D segmentation, per batch.

    preds, targets: (B, 1, D, H, W) with {0,1} or [0,1].

    Returns:
        scalar float (batch mean clDice).
    """
    preds = preds.float().clamp(0.0, 1.0)
    targets = targets.float().clamp(0.0, 1.0)

    skel_pred = _skeletonize_3d(preds, iterations=iterations)
    skel_target = _skeletonize_3d(targets, iterations=iterations)

    # Topology precision: predicted skeleton inside target mask.
    tprec = (
        (skel_pred * targets).sum(dim=(1, 2, 3, 4)) + smooth
    ) / (
        skel_pred.sum(dim=(1, 2, 3, 4)) + smooth
    )

    # Topology sensitivity: target skeleton inside predicted mask.
    tsens = (
        (skel_target * preds).sum(dim=(1, 2, 3, 4)) + smooth
    ) / (
        skel_target.sum(dim=(1, 2, 3, 4)) + smooth
    )

    score = 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return float(score.mean().item())


# ---------------- Bootstrap CI ----------------

def bootstrap_mean_ci(values, n_bootstrap=1000, ci=95, seed=2024):
    """
    Return mean and two-sided bootstrap confidence interval half-width.

    Returns:
        mean, half_width, lower, upper
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(values.mean())

    if values.size == 1 or n_bootstrap <= 0:
        return mean, 0.0, mean, mean

    rng = np.random.default_rng(seed)
    n = values.size
    boot_means = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    half_width = float((upper - lower) / 2.0)

    return mean, half_width, lower, upper


def summarize_metrics_with_ci(metrics_list, n_bootstrap=1000, seed=2024):
    if len(metrics_list) == 0:
        raise ValueError("metrics_list is empty.")

    metric_names = list(metrics_list[0].keys())
    summary = {}

    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list]
        mean, half_width, lower, upper = bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            ci=95,
            seed=seed,
        )
        summary[metric_name] = {
            "mean": mean,
            "ci95": half_width,
            "lower": lower,
            "upper": upper,
            "n": int(np.isfinite(np.asarray(values, dtype=np.float64)).sum()),
            "values": np.asarray(values, dtype=np.float64),
        }

    return summary


def print_metric_summary(summary):
    """
    固定输出顺序 & Mean ± 95% CI：
      DSC, clDice, IoU, Sensitivity, Specificity, HD95, ASSD
    """
    print("=== Test Results: mean +/- 95% CI ===")

    ordered_keys = [
        "DSC",
        "clDice",
        "IoU",
        "Sensitivity",
        "Specificity",
        "HD95",
        "ASSD",
    ]

    for key in ordered_keys:
        if key not in summary:
            continue
        item = summary[key]
        mean = item["mean"]
        ci = item["ci95"]
        lower = item["lower"]
        upper = item["upper"]
        n = item["n"]

        print(
            f"{key:12s}: "
            f"{mean:.4f} +/- {ci:.4f} "
            f"[{lower:.4f}, {upper:.4f}] "
            f"(n={n})"
        )


def save_metrics_csv(metrics_list, case_ids, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = list(metrics_list[0].keys())

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("case_id," + ",".join(metric_names) + "\n")

        for case_id, metrics in zip(case_ids, metrics_list):
            values = [str(metrics[name]) for name in metric_names]
            f.write(str(case_id) + "," + ",".join(values) + "\n")

    print(f"Saved per-case metrics CSV: {save_path}")


def get_case_id_from_batch(batch):
    if "case_id" not in batch:
        return "unknown"

    case_id = batch["case_id"]

    if isinstance(case_id, (list, tuple)):
        return str(case_id[0])

    return str(case_id)


def apply_checkpoint_cfg(cfg, checkpoint):
    """
    Use model/salience config saved in checkpoint when available.
    """
    if not isinstance(checkpoint, dict):
        return cfg, {}

    ckpt_cfg = checkpoint.get("cfg", {})
    if not isinstance(ckpt_cfg, dict):
        return cfg, {}

    if "model" in ckpt_cfg:
        cfg["model"] = dict(ckpt_cfg["model"])
        print("Using model config from checkpoint.")

    if "salience" in ckpt_cfg:
        cfg["salience"] = dict(ckpt_cfg["salience"])
        print("Using salience config from checkpoint.")

    return cfg, ckpt_cfg


def print_attnunet_debug_info(model, cfg):
    model_name = cfg.get("model", {}).get("name", "")

    if model_name not in ("AttnUNet3D", "AttnUNet"):
        return

    import inspect

    print("=== AttnUNet3D Debug ===")
    print("Model class file:", inspect.getfile(model.__class__))

    if hasattr(model, "up1") and hasattr(model.up1, "up"):
        print("up1.up.weight shape:", tuple(model.up1.up.weight.shape))
    if hasattr(model, "up2") and hasattr(model.up2, "up"):
        print("up2.up.weight shape:", tuple(model.up2.up.weight.shape))
    if hasattr(model, "up3") and hasattr(model.up3, "up"):
        print("up3.up.weight shape:", tuple(model.up3.up.weight.shape))


# ---------------- p-value 计算 ----------------

def _paired_t_test(x, y):
    """
    配对 t 检验：x、y 为同一批 case 上的两个方法结果。
    返回 t 值和双侧 p 值（用正态近似）。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 2:
        return np.nan, np.nan

    d = x - y
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    if sd_d == 0:
        return np.inf, 0.0
    t_stat = mean_d / (sd_d / math.sqrt(n))

    # 正态近似计算双侧 p 值
    from math import erf, sqrt
    z = abs(t_stat)
    p = 2 * (1.0 - 0.5 * (1 + erf(z / sqrt(2))))
    return t_stat, p


def _load_baseline_csv(path):
    """
    读取 baseline csv:
        case_id,DSC,clDice,IoU,Sensitivity,Specificity,HD95,ASSD
    返回: dict(case_id -> {metric: value})
    """
    baseline = {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        metric_names = header[1:]
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            cid = parts[0]
            values = parts[1:]
            record = {}
            for name, v in zip(metric_names, values):
                try:
                    record[name] = float(v)
                except ValueError:
                    record[name] = np.nan
            baseline[cid] = record
    return baseline


def _compute_p_values(summary, case_ids, metrics_list, baseline_csv):
    """
    基于 baseline CSV 计算 p-value。
    使用配对 t 检验（按 case_id 对齐）。
    """
    baseline = _load_baseline_csv(baseline_csv)

    # 将当前结果按 case_id 组织
    current_metrics_by_case = {}
    for cid, metrics in zip(case_ids, metrics_list):
        current_metrics_by_case[cid] = metrics

    metric_keys = [
        "DSC",
        "clDice",
        "IoU",
        "Sensitivity",
        "Specificity",
        "HD95",
        "ASSD",
    ]

    print("\n=== P-value vs baseline (paired t-test where possible) ===")
    for key in metric_keys:
        cur_vals = []
        base_vals = []
        for cid in case_ids:
            if cid in baseline and key in baseline[cid] and key in current_metrics_by_case[cid]:
                cur = current_metrics_by_case[cid][key]
                base = baseline[cid][key]
                if np.isfinite(cur) and np.isfinite(base):
                    cur_vals.append(cur)
                    base_vals.append(base)
        if len(cur_vals) < 2:
            print(f"{key:12s}: not enough matched cases for p-value (n={len(cur_vals)})")
            continue

        t_stat, p = _paired_t_test(cur_vals, base_vals)
        diff_mean = float(np.mean(cur_vals) - np.mean(base_vals))
        print(
            f"{key:12s}: "
            f"mean_diff={diff_mean:+.4f}, "
            f"t={t_stat:.4f}, "
            f"p={p:.4e}, "
            f"n={len(cur_vals)}"
        )


def main():
    args = parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg, ckpt_cfg = apply_checkpoint_cfg(cfg, checkpoint)

    dataset_name = (
        args.dataset
        or ckpt_cfg.get("data", {}).get("dataset")
        or cfg["data"]["dataset"]
    )

    threshold = args.threshold
    if threshold is None:
        threshold = cfg.get("val", {}).get("threshold", 0.5)

    cfg["data"]["use_salience"] = args.use_salience

    if args.use_salience:
        cfg["model"]["in_channels"] = 3
    else:
        cfg["model"]["in_channels"] = 1

    print("=== Eval Config ===")
    print(f"Dataset: {dataset_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Use salience: {args.use_salience}")
    print(f"Threshold: {threshold}")
    print(f"Bootstrap: {args.bootstrap}")
    print(f"Device: {device}")
    print(f"Model name: {cfg['model'].get('name')}")
    print(f"Model in_channels: {cfg['model'].get('in_channels')}")
    print(f"Model base_channels: {cfg['model'].get('base_channels')}")

    _, _, test_set = load_dataset(
        dataset_name=dataset_name,
        split_path=cfg["data"]["split_path"],
        root_dir=cfg["data"].get("root_dir", "data/preprocessed"),
    )

    print(f"Test cases: {len(test_set)}")
    if hasattr(test_set, "case_list"):
        print(f"First test cases: {test_set.case_list[:5]}")

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("train", {}).get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg["model"]).to(device)
    print_attnunet_debug_info(model, cfg)

    load_checkpoint(model, ckpt_path, device, checkpoint=checkpoint)
    model.eval()

    salience_gen = None
    if args.use_salience:
        salience_gen, plugin_name, kwargs = build_salience_generator(cfg)

        print("=== Eval Salience ===")
        print(f"Plugin: {plugin_name}")
        print(f"delta_theta: {kwargs.get('delta_theta')}")
        print(f"delta_sigma: {kwargs.get('delta_sigma')}")
        print(f"K: {kwargs.get('K')}")
        if "alpha" in kwargs:
            print(f"alpha: {kwargs.get('alpha')}")

    all_metrics = []
    case_ids = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", unit="case")

        for batch in pbar:
            imgs = batch["img"].to(device, non_blocking=True).float()
            masks = batch["mask"].to(device, non_blocking=True).float()

            imgs = apply_salience_to_batch(imgs, salience_gen)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # 基础指标（假设 compute_all_metrics 已实现 DSC, IoU, Sen/Spe, HD95, ASSD）
            metrics = compute_all_metrics(preds, masks, threshold=0.5)

            # 名称对齐：若内部用 Sen/Spe，则映射为 Sensitivity/Specificity
            if "Sen" in metrics and "Sensitivity" not in metrics:
                metrics["Sensitivity"] = metrics["Sen"]
            if "Spe" in metrics and "Specificity" not in metrics:
                metrics["Specificity"] = metrics["Spe"]

            # 计算 clDice（基于 preds / masks）
            cld = _cldice_from_batch(preds, masks, iterations=10)
            metrics["clDice"] = cld

            all_metrics.append(metrics)

            case_id = get_case_id_from_batch(batch)
            case_ids.append(case_id)

            if "DSC" in metrics:
                pbar.set_postfix(DSC=f"{metrics['DSC']:.4f}")

    summary = summarize_metrics_with_ci(
        all_metrics,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    print_metric_summary(summary)

    if args.save_csv is not None:
        save_metrics_csv(all_metrics, case_ids, args.save_csv)

    if args.baseline_csv is not None:
        _compute_p_values(summary, case_ids, all_metrics, args.baseline_csv)


if __name__ == "__main__":
    main()
