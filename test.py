import os
import time
import random
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms

from utils import inference
from loguru import logger
from rich.progress import track
from networks.unet import UNet
from torch.utils.data import DataLoader
from dataloaders.dataset import VolumeDataSet, ToTensor


parser = argparse.ArgumentParser(description="Train the net on images and target masks")
parser.add_argument("--dataset", type=str, default="3D-IRCADb", help="dataset name")
parser.add_argument("--img_size", type=tuple, default=(256, 256), help="image size")
parser.add_argument("--backbone", type=str, default="unet", help="backbone network name")
parser.add_argument("--in_channels", type=int, default=3, help="input channel of network")
parser.add_argument("--out_channels", type=int, default=1, help="output channel of network")
parser.add_argument("--model_path", type=str, default="./checkpoint/3D-IRCADb/unet/epo100_bs4_lr0.01/ours/best_model.pth", help="The .pth file path of saved model")
args = parser.parse_args()


if __name__ == '__main__':
    logger.add(
        (args.model_path).replace('best_model.pth', 'test_result.txt'),
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
        level="INFO",
    )

    model_config = {
        "unet": UNet(args.in_channels, args.out_channels),
    }
    model = model_config[args.backbone]
    model.load_state_dict(
        torch.load(args.model_path)
    )
    model = model.cuda()

    # Create dataset
    dataset_name = args.dataset
    root_dir = f"data/{dataset_name}/"
    list_dir = f"list/{dataset_name}/"

    db_test = VolumeDataSet(root_dir, list_dir, "test", transforms.Compose([ToTensor()]), img_size=args.img_size)
    test_loader = DataLoader(db_test, batch_size=1)

    model.eval()
    metrics = .0
    for batch in track(test_loader, description="Testing"):
        images, labels = batch["image"], batch["label"]
        saliences = batch["salience"]
        # inputs = images
        inputs = (saliences + 1) * images
        metrics += np.array(inference(inputs, labels, model, 'eval'))
    metrics = metrics / len(test_loader)
    logger.info(
        """Metric percase:
        Dice:       {:.2f}
        clDice:     {:.2f}
        IoU:        {:.2f}
        Sen:        {:.2f}
        Spe:        {:.2f}
        HD95:       {:.2f}
        ASSD:       {:.2f}
    """.format(
            metrics[0] * 100,
            metrics[1] * 100,
            metrics[2] * 100,
            metrics[3] * 100,
            metrics[4] * 100,
            metrics[5],
            metrics[6],
        )
    )