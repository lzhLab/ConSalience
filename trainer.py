import os
import cv2
import random
import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import csv

from scipy import stats

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
from torch.utils.data import DataLoader
from utils import test_single_volume


def liver_vessel_seg(args, model, snapshot_path):
    import sys
    sys.path.append('./datasets')
    from liver_vessel import LiverVesselVolume, RandomGenerator, ToTensor, RandomCrop3D
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
        TextColumn("鈥?"),
        TimeElapsedColumn(),
        TextColumn("鈥?"),
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
    print("Model loaded!")

    #The index structure of each sample is: {'dice': (mean, ci), 'acc': (mean, ci), ...} 
    all_metrics_list = []
    for test_batch in test_loader:
        image_vol = test_batch['image'].cuda()
        label_vol = test_batch['label'].cuda()
        metric, _ = test_single_volume(model, image_vol, label_vol, "test", img_size)
        all_metrics_list.append(metric)
 
    ## Convert to a DataFrame, where each row is a sample and each column is a metric
    headers = all_metrics_list[0].keys()
    filename = snapshot_path + "/Per_sample_metrics.csv"
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_metrics_list)
        print(f"Per-sample metrics saved to {filename}")
    except IOError as e:
        print(f"Failed to save per-sample metrics to {filename}")


    # Aggregation index
    final_metrics = {}
    metric_names = list(all_metrics_list[0].keys()) # such as: dice, acc...
    for metric in metric_names:
        means = [sample[metric][0] for sample in all_metrics_list]
        cis = [sample[metric][1] for sample in all_metrics_list]
        print("means----",means,"cis-------",cis)
        mean_avg = np.mean(means)
        ci_avg = np.mean(cis)
        final_metrics[metric] = (mean_avg, ci_avg)

    logger.info("=" * 25 + " Final Metrics (95% CI)" + "=" * 25)
    for k, v in final_metrics.items():
            logger.info(f"{k}: {v[0]:.2f} 卤 {v[1]:.2f}")
#    writer.close()

    # Save as CSV file
    csv_path = snapshot_path + "/final_test_metrics.csv"
    df = pd.DataFrame([
        {'Metric': k, 'Mean': v[0], '95% CI': v[1]}
        for k, v in final_metrics.items()
    ])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return
