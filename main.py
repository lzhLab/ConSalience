import os
import time
import random
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from networks.unet import UNet
from trainer import trainer_ircad


parser = argparse.ArgumentParser(description="Train the net on images and target masks")
parser.add_argument("--dataset", type=str, default="3D-IRCADb", help="dataset name")
parser.add_argument("--img_size", type=tuple, default=(256, 256), help="image size")
parser.add_argument("--backbone", type=str, default="unet", help="backbone network name")
parser.add_argument("--in_channels", type=int, default=3, help="input channel of network")
parser.add_argument("--out_channels", type=int, default=1, help="output channel of network")
parser.add_argument("--experiment", "-exp", type=str, default=time.ctime().replace(" ", "_"), help="experiment name")
parser.add_argument("--max_epochs", "-epo", type=int, default=100, help="maximum epoch number to train")
parser.add_argument("--base_lr", "-lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--batch_size", "-bs", type=int, default=4, help="batch size per GPU")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "checkpoint/{}/{}/".format(args.dataset, args.backbone)
    snapshot_path = snapshot_path + "epo" + str(args.max_epochs)
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = snapshot_path + "_lr" + str(args.base_lr) + "/"
    snapshot_path = (snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path)
    snapshot_path = snapshot_path + args.experiment

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    model_config = {
        "unet": UNet(args.in_channels, args.out_channels),
    }
    model = model_config[args.backbone].cuda()

    trainer_ircad(args, model, snapshot_path)
