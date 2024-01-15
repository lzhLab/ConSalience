import os
import cv2
import random
import torch
import torch.nn as nn
import numpy as np 
import pandas as pd

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from loguru import logger
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.data import DataLoader
from .utils import test_single_volume


def liver_vessel_seg(args, model, snapshot_path):
    from datasets.liver_vessel import LiverVesselVolume, RandomGenerator, ToTensor, RandomCrop3D
    logger.add(
        snapshot_path + "/log.txt",
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
        level="INFO",
    )

    # Hyperparameters
    bs = args.batch_size
    max_epochs = args.max_epochs
    base_lr = args.base_lr
    dataset = args.dataset
    img_size = args.img_size


    # Create database
    root_dir = f"./data/{dataset}/data_h5"
    list_dir = f"./list/{dataset}"
    db_train = LiverVesselVolume(root_dir, list_dir, split="train",
        transform=transforms.Compose([RandomCrop3D(img_size), RandomGenerator(), ToTensor()])
    )
    db_val = LiverVesselVolume(root_dir, list_dir, split="val", transform=transforms.Compose([ToTensor()]))
    db_test = LiverVesselVolume(root_dir, list_dir, split="test", transform=transforms.Compose([ToTensor()]))

    # Deterministic training
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # Create dataloader
    train_loader = DataLoader(
        db_train, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, generator=generator
    )
    val_loader = DataLoader(db_val, batch_size=1)
    test_loader = DataLoader(db_test, batch_size=1)
    max_iterations = max_epochs * len(train_loader)

    # Logging informations
    logger.info(f'''Starting training:
        Iterations:      {max_iterations}
        Epochs:          {max_epochs}
        Batch size:      {bs}
        Learning rate:   {base_lr}
        Dataset:         {dataset}
        Image size:      {img_size}
        Training size:   {len(db_train)}
        Validation size: {len(db_val)}
        Testing size:    {len(db_test)}
        Checkpoints:     {snapshot_path}
    ''')

    # Model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    criterion = nn.BCEWithLogitsLoss()

    progress_bar = Progress(
        TextColumn("training..."), 
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )
    task = progress_bar.add_task(description="training", total=max_iterations)
    progress_bar.start()

    #################################### training #######################################
    iter_num = 0
    best_performance = 0
    writer = SummaryWriter(snapshot_path + f"/log")
    for epoch_num in range(1, max_epochs + 1):
        epoch_loss = 0
        for train_batch in train_loader:
            images = train_batch['image'].to('cuda')
            labels = train_batch['label'].to('cuda')

            outputs= model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(task, advance=1)

            iter_num += 1
            epoch_loss += loss.item()
            print("loss: {:.4f}".format(loss.item()))

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            writer.add_scalar('LearningRate', optimizer.param_groups[0]["lr"], iter_num)
            writer.add_scalar('Loss/batch', loss, iter_num)

            #################################### Validating #####################################
            interval = max_iterations // 100
            if iter_num % interval == 0:
                progress_bar.stop()
                writer.add_image(
                    tag=f"image_train/iter_{iter_num}", 
                    img_tensor=torch.stack([images[0, :, :, :, 0], labels[0, :, :, :, 0], torch.sigmoid(outputs[0, :, :, :, 0])], dim=0),  
                    dataformats='NCHW', 
                )
                print("validating...")
                metrics = 0.0
                for val_batch in val_loader:
                    image_vol = val_batch["image"].cuda()
                    label_vol = val_batch["label"].cuda()
                    metric, prob = test_single_volume(model, image_vol, label_vol, "eval", img_size)
                    metrics += np.array(metric)
                metrics = np.array(metrics) / len(val_loader)
                logger.info("[Iter {}] Dice:{:.2f} clDice:{:.2f}".format(iter_num, *metrics))
                writer.add_scalar("Metric/Dice", metrics[0], iter_num)
                writer.add_scalar("Metric/clDice", metrics[1], iter_num)
                if metrics[0] > best_performance:
                    save_model_path = snapshot_path + "/best_model.pth"
                    torch.save(model.state_dict(), save_model_path)
                    logger.info("[iter {}] Dice increased: {:.2f} --> {:.2f}\n".format(iter_num, best_performance, metrics[0]))
                    best_performance = metrics[0]
                progress_bar.start()
            writer.add_scalar("best_performance", best_performance, iter_num)
        epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/epoch', epoch_loss, epoch_num)
    progress_bar.stop()
    print("="*25 + "Training Finished!" + "="*25)

    ########################################### Testing ################################################
    print("testing...")
    model.load_state_dict(torch.load(save_model_path))
    metrics = 0.0
    for test_batch in test_loader:
        image_vol = test_batch['image'].cuda()
        label_vol = test_batch['label'].cuda()
        metric, _ = test_single_volume(model, image_vol, label_vol, "test", img_size)
        metrics += np.array(metric)
    metrics = np.array(metrics) / len(test_loader)
    logger.info("="*25 + "Testing Result" + "="*25)
    logger.info(
        "Dice:{:.2f} clDice:{:.2f} Acc:{:.2f} Sen:{:.2f} Spe:{:.2f} HD95:{:.2f} ASSD:{:.2f}".format(
            *metrics
        )
    )
    writer.close()
    return


