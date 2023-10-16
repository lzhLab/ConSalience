import torch
import numpy as np
import scipy.ndimage as nd
from medpy import metric
from skimage import measure
from skimage.morphology import skeletonize_3d


def skel_coverage(skel, vessel, smooth=1e-5):
    intersection = (skel * vessel).sum()
    return (intersection + smooth) / (skel.sum() + smooth)


def cl_dice(pred, gt):
    pred_skel = skeletonize_3d(pred)
    gt_skel = skeletonize_3d(gt)

    pred_SC = skel_coverage(pred_skel, gt)
    gt_SC = skel_coverage(gt_skel, pred)

    return 2 * pred_SC * gt_SC / (pred_SC + gt_SC)


def test(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        cldice = cl_dice(pred, gt)
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        sen = metric.binary.sensitivity(pred, gt)
        spe = metric.binary.specificity(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        assd = metric.binary.assd(pred, gt)
        return dice, cldice, jc, sen, spe, hd95, assd
    else:
        return 0, 0, 0, 0, 0, 0, 0


def evaluate(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        clDice = cl_dice(pred, gt)
        return dice, clDice
    else:
        return .0, .0


def inference(input_vol, label_vol, net, mode='test'):
    # (1, 1, H, W, D) -> (H, W, D)
    label_vol, input_vol = label_vol.squeeze(), input_vol.squeeze(0)
    label_vol = label_vol.cpu().detach().numpy()
    pred_vol = np.zeros_like(label_vol)

    # The LiVS dataset has sparse annotation. The label of an image without any annotation is assigned as zero.
    lst = []
    for idx in range(label_vol.shape[-1]):
        if not np.all(label_vol[..., idx] == 0):
            lst.append(idx)
    
    for idx in lst:
        input_slice = input_vol[..., idx]  # (C, H, W)
        input_slice = input_slice.unsqueeze(0).float().cuda()
        with torch.no_grad():
            pred_slice = torch.sigmoid(net(input_slice).squeeze()) > 0.5
            pred_slice = pred_slice.float().cpu().detach().numpy()
            pred_vol[..., idx] = pred_slice
    
    if mode == 'eval':
        return evaluate(pred_vol[..., lst], label_vol[..., lst])
    else:
        return test(pred_vol[..., lst], label_vol[..., lst])


def post_processing(prediction):
    # tutorial: https://www.bookstack.cn/read/scipy-lecture-notes_cn/spilt.5.ca8111fbd818492f.md

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
    prediction = nd.binary_fill_holes(prediction)
    label_cc, num_cc = measure.label(prediction, return_num=True)
    total_cc = np.sum(prediction)
    # https://scikit-image.org/docs/dev/api/skimage.measure.html
    measure.regionprops(label_cc)
    for cc in range(1, num_cc + 1):
        single_cc = label_cc == cc
        single_vol = np.sum(single_cc)
        # remove some some volume
        if single_vol / total_cc < 0.2:
            prediction[single_cc] = 0

    return prediction
