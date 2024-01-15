import os
import time
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.unet3d_with_salience import UNet_3D
from trainer import liver_vessel_seg


parser = argparse.ArgumentParser(description="Train the net on images and target masks")
parser.add_argument("--dataset", type=str, default="3Dircadb1")
parser.add_argument("--img_size", type=tuple, default=(96,96,96))
parser.add_argument("--method", type=str, default="unet_3d")
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--out_channels", type=int, default=1)
parser.add_argument("--max_epochs", "-epo", type=int, default=1000)
parser.add_argument("--batch_size", "-bs", type=int, default=1)
parser.add_argument("--base_lr", "-lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument(
    "--experiment", "-exp", type=str, default=time.ctime().replace(" ", "_"), help="suffix of snapshotPath",
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
args = parser.parse_args()


if __name__ == '__main__':
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

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    snapshot_path = "checkpoint/{}/{}/{}/".format(
        args.dataset, 
        str(args.img_size[0])+'x'+str(args.img_size[1])+'x'+str(args.img_size[2]), 
        args.method
    )
    snapshot_path = snapshot_path + "epo" + str(args.max_epochs)
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = snapshot_path + "_lr" + str(args.base_lr) + "/"
    snapshot_path = (
        snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path
    )
    snapshot_path = snapshot_path + args.experiment
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    model_config = {
        "unet_3d": UNet_3D(args.in_channels, args.out_channels).cuda(), 
    }
    model = model_config[args.method]

    liver_vessel_seg(args, model, snapshot_path)
