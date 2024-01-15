import math
import torch
import numpy as np 
from medpy import metric

from .metrics import clDice


def test_metrics(pred, gt):
    gt[gt > 0] = 1
    pred = np.where(pred > 0.5, 1, 0)
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
    pred = np.where(pred > 0.5, 1, 0)
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
        metrics = test_metrics(pred_vol, label_vol)
    else:
        metrics = eval_metrics(pred_vol, label_vol)
    
    return metrics, pred_vol