import numpy as np
import torch
import math
from medpy import metric
from metrics import clDice
from scipy.stats import norm

def bootstrap_ci(metric_values, confidence=0.95):
    """Calculate the mean and confidence interval"""
    mean = np.mean(metric_values)
    std = np.std(metric_values, ddof=1)
    n = len(metric_values)
    z = norm.ppf(1 - (1 - confidence) / 2)
    ci_half = z * std / np.sqrt(n)
    return mean, ci_half

def test_metrics_with_ci(pred, gt, num_samples=100):
    gt[gt > 0] = 1
    pred = np.where(pred > 0.5, 1, 0)
    
    if pred.sum() == 0 or gt.sum() == 0:
        return {k: (0.0, 0.0, 0.0) for k in ['dice', 'cl_dice', 'acc', 'sens', 'spec', 'hd95', 'assd']}

    # Obtain the position of the positive samples 
    coords = np.argwhere(gt == 1)
    n_voxels = len(coords)
    if n_voxels == 0:
        return {k: (0.0, 0.0, 0.0) for k in ['dice', 'cl_dice', 'acc', 'sens', 'spec', 'hd95', 'assd']}
    
    # bootstrap samples and calculates metrics
    dice_list, cldice_list, acc_list, sens_list, spec_list, hd95_list, assd_list = [], [], [], [], [], [], []

    for _ in range(num_samples):
        indices = np.random.choice(n_voxels, n_voxels, replace=True)
        sample_coords = coords[indices]

        gt_sample = np.zeros_like(gt)
        pred_sample = np.zeros_like(pred)

        for x, y, z in sample_coords:
            gt_sample[x, y, z] = gt[x, y, z]
            pred_sample[x, y, z] = pred[x, y, z]

        tp = pred_sample * gt_sample
        tn = (1 - pred_sample) * (1 - gt_sample)
        fp = pred_sample - tp
        fn = (1 - pred_sample) - tn
        TP, TN, FP, FN = tp.sum(), tn.sum(), fp.sum(), fn.sum()

        if TP + FP + FN == 0:
            continue  # Skip the invalid denominator

        dice = 2 * TP / (2 * TP + FP + FN)
        cl_d = clDice(pred_sample, gt_sample)
        acc = (TP + TN) / (TP + TN + FP + FN)
        sens = TP / (TP + FN) if (TP + FN) != 0 else 0
        spec = TN / (TN + TP) if (TN + TP) != 0 else 0

        try:
            hd = metric.binary.hd95(pred_sample, gt_sample)
            assd = metric.binary.assd(pred_sample, gt_sample)
        except:
            hd, assd = 0.0, 0.0

        dice_list.append(dice * 100)
        cldice_list.append(cl_d * 100)
        acc_list.append(acc * 100)
        sens_list.append(sens * 100)
        spec_list.append(spec * 100)
        hd95_list.append(hd)
        assd_list.append(assd)

    return {
        "dice": bootstrap_ci(dice_list),
        "cl_dice": bootstrap_ci(cldice_list),
        "acc": bootstrap_ci(acc_list),
        "sens": bootstrap_ci(sens_list),
        "spec": bootstrap_ci(spec_list),
        "hd95": bootstrap_ci(hd95_list),
        "assd": bootstrap_ci(assd_list)
    }


def test_metrics(pred, gt):
    gt[gt > 0] = 1
    pred = np.where(pred > 0.05, 1, 0)
    if pred.sum() > 0 and gt.sum() > 0:
        tp = pred * gt
        tn = (1 - pred) * (1 - gt)
        fp = pred - tp
        fn = (1 - pred) - tn
        TP, TN, FP, FN = tp.sum(), tn.sum(), fp.sum(), fn.sum()
       
        dice = 2*TP / (2*TP + FP + FN)
        cl_dice = clDice(pred, gt)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + TP)
        
        hd95 = metric.binary.hd95(pred, gt)
        assd = metric.binary.assd(pred, gt)
        
        return 100*dice, 100*cl_dice, 100*accuracy, 100*sensitivity, 100*specificity, hd95, assd
    else:
        return .0, .0, .0, .0, .0, .0, .0


def eval_metrics(pred, gt):
    gt[gt > 0] = 1
    pred = np.where(pred > 0.05, 1, 0)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        cl_dice = clDice(pred, gt)
        return 100 * dice, 100 * cl_dice
    else:
        return .0, .0


def test_single_volume(model, image_vol, label_vol, mode="test", img_size=(96, 96, 96)):
    image_vol = image_vol.squeeze().detach().cpu().numpy()  # (H, W, D)
    label_vol = label_vol.squeeze().detach().cpu().numpy()
    pred_vol = np.zeros_like(label_vol)
    cnt = np.zeros_like(label_vol)
    H, W, D = image_vol.shape
    h, w, d = img_size

    num_h = math.ceil(H / h)
    num_w = math.ceil(W / w)
    num_d = math.ceil(D / d)
    h_step = 0 if num_h == 1 else  (H - h) // (num_h - 1)
    w_step = 0 if num_w == 1 else  (W - w) // (num_w - 1)
    d_step = 0 if num_d == 1 else (D - d) // (num_d - 1)

    model.eval()
    for i in range(0, num_h):
        for j in range(0, num_w):
            for k in range(0, num_d):
                inputs = torch.from_numpy(
                    image_vol[i*h_step:i*h_step+h, j*w_step:j*w_step+w, k*d_step:k*d_step+d]
                ).float().cuda()
                inputs = inputs.view(1, 1, *img_size)
                with torch.no_grad():
                    outputs = torch.sigmoid(model(inputs))
                    outputs = outputs.squeeze().detach().cpu().numpy()
                    pred_vol[i*h_step:i*h_step+h, j*w_step:j*w_step+w, k*d_step:k*d_step+d] += outputs
                    cnt[i*h_step:i*h_step+h, j*w_step:j*w_step+w, k*d_step:k*d_step+d] += 1
    model.train()
    cnt[cnt == 0] = 1
    pred_vol = pred_vol / cnt
    if mode == "test":
        metrics = test_metrics_with_ci(pred_vol, label_vol)
    else:
        metrics = eval_metrics(pred_vol, label_vol)
    return metrics, pred_vol
