# src/utils/metrics.py
import math
from typing import Dict, List

import numpy as np
import torch


EPS = 1e-7


def _to_numpy_binary(x, threshold: float = 0.5) -> np.ndarray:
    """
    Convert torch.Tensor or np.ndarray to binary numpy array.

    Accepts shapes like:
        (B, 1, D, H, W)
        (1, D, H, W)
        (D, H, W)
        (H, W, D)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()

    x = np.asarray(x)

    if x.dtype != np.bool_:
        x = x > threshold

    return x.astype(np.bool_)


def confusion_counts(pred, target, threshold: float = 0.5):
    pred = _to_numpy_binary(pred, threshold)
    target = _to_numpy_binary(target, threshold)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    tp = np.logical_and(pred, target).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(target)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), target).sum(dtype=np.float64)
    tn = np.logical_and(np.logical_not(pred), np.logical_not(target)).sum(dtype=np.float64)

    return tp, fp, fn, tn


def dice_score(pred, target, threshold: float = 0.5) -> float:
    tp, fp, fn, _ = confusion_counts(pred, target, threshold)
    return float((2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS))


def iou_score(pred, target, threshold: float = 0.5) -> float:
    tp, fp, fn, _ = confusion_counts(pred, target, threshold)
    return float((tp + EPS) / (tp + fp + fn + EPS))


def precision_score(pred, target, threshold: float = 0.5) -> float:
    tp, fp, _, _ = confusion_counts(pred, target, threshold)
    return float((tp + EPS) / (tp + fp + EPS))


def sensitivity_score(pred, target, threshold: float = 0.5) -> float:
    tp, _, fn, _ = confusion_counts(pred, target, threshold)
    return float((tp + EPS) / (tp + fn + EPS))


def recall_score(pred, target, threshold: float = 0.5) -> float:
    return sensitivity_score(pred, target, threshold)


def specificity_score(pred, target, threshold: float = 0.5) -> float:
    _, fp, _, tn = confusion_counts(pred, target, threshold)
    return float((tn + EPS) / (tn + fp + EPS))


def accuracy_score(pred, target, threshold: float = 0.5) -> float:
    tp, fp, fn, tn = confusion_counts(pred, target, threshold)
    return float((tp + tn + EPS) / (tp + fp + fn + tn + EPS))


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing=None) -> np.ndarray:
    """
    Compute distances from surface voxels of mask_a to surface voxels of mask_b.

    Requires scipy.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError("scipy is required for HD95 calculation.") from exc

    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if spacing is None:
        spacing = (1.0,) * mask_a.ndim

    if not mask_a.any() or not mask_b.any():
        return np.array([np.inf], dtype=np.float32)

    structure = ndimage.generate_binary_structure(mask_a.ndim, 1)

    eroded_a = ndimage.binary_erosion(mask_a, structure=structure, border_value=0)
    eroded_b = ndimage.binary_erosion(mask_b, structure=structure, border_value=0)

    surface_a = np.logical_xor(mask_a, eroded_a)
    surface_b = np.logical_xor(mask_b, eroded_b)

    distance_map_b = ndimage.distance_transform_edt(~surface_b, sampling=spacing)
    distances = distance_map_b[surface_a]

    return distances.astype(np.float32)


def hd95_score(pred, target, threshold: float = 0.5, spacing=None) -> float:
    """
    95th percentile Hausdorff distance.

    If one mask is empty and the other is not, returns inf.
    If both masks are empty, returns 0.0.
    If scipy is unavailable, returns nan.
    """
    pred = _to_numpy_binary(pred, threshold)
    target = _to_numpy_binary(target, threshold)

    pred = np.squeeze(pred)
    target = np.squeeze(target)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    if not pred.any() and not target.any():
        return 0.0

    if pred.any() != target.any():
        return float("inf")

    try:
        d_pred_to_target = _surface_distances(pred, target, spacing=spacing)
        d_target_to_pred = _surface_distances(target, pred, spacing=spacing)
    except ImportError:
        return float("nan")

    distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(distances, 95))


def assd_score(pred, target, threshold: float = 0.5, spacing=None) -> float:
    """
    Average symmetric surface distance.

    If scipy is unavailable, returns nan.
    """
    pred = _to_numpy_binary(pred, threshold)
    target = _to_numpy_binary(target, threshold)

    pred = np.squeeze(pred)
    target = np.squeeze(target)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    if not pred.any() and not target.any():
        return 0.0

    if pred.any() != target.any():
        return float("inf")

    try:
        d_pred_to_target = _surface_distances(pred, target, spacing=spacing)
        d_target_to_pred = _surface_distances(target, pred, spacing=spacing)
    except ImportError:
        return float("nan")

    return float((d_pred_to_target.mean() + d_target_to_pred.mean()) / 2.0)


def compute_all_metrics(
    pred,
    target,
    threshold: float = 0.5,
    spacing=None,
) -> Dict[str, float]:
    """
    Compute common binary segmentation metrics.

    Parameters
    ----------
    pred:
        Binary prediction or probability/logit after external thresholding.
        If pred is probability, threshold=0.5 is used.
    target:
        Binary ground truth.
    threshold:
        Threshold for binarization.
    spacing:
        Optional voxel spacing for HD95/ASSD.

    Returns
    -------
    dict:
        {
            "DSC": ...,
            "IoU": ...,
            "Precision": ...,
            "Sensitivity": ...,
            "Recall": ...,
            "Specificity": ...,
            "Accuracy": ...,
            "HD95": ...,
            "ASSD": ...
        }
    """
    return {
        "DSC": dice_score(pred, target, threshold),
        "IoU": iou_score(pred, target, threshold),
        "Precision": precision_score(pred, target, threshold),
        "Sensitivity": sensitivity_score(pred, target, threshold),
        "Recall": recall_score(pred, target, threshold),
        "Specificity": specificity_score(pred, target, threshold),
        "Accuracy": accuracy_score(pred, target, threshold),
        "HD95": hd95_score(pred, target, threshold, spacing=spacing),
        "ASSD": assd_score(pred, target, threshold, spacing=spacing),
    }


def compute_average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average a list of metric dictionaries.

    Non-finite values such as inf/nan are ignored.
    If all values for one metric are non-finite, returns nan.
    """
    if len(metrics_list) == 0:
        raise ValueError("metrics_list is empty.")

    keys = metrics_list[0].keys()
    averaged = {}

    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=np.float64)
        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            averaged[key] = float("nan")
        else:
            averaged[key] = float(finite_values.mean())

    return averaged


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dict for logging.
    """
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float) and math.isfinite(value):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
