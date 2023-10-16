import os
import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from rich.progress import track
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import inference
from dataloaders.dataset import SliceDataSet, VolumeDataSet, RandomRotFlip, ToTensor


def trainer_ircad(args, model, snapshot_path):
    writer = SummaryWriter(snapshot_path + "/log")
    logger.add(
        snapshot_path + "/log.txt",
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
        level="INFO",
    )
    logger.info(str(args))

    # Hyperparameters
    img_size = args.img_size
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs

    # Create dataset
    dataset_name = args.dataset
    root_dir = f"data/{dataset_name}/"
    list_dir = f"list/{dataset_name}/"
    db_train = SliceDataSet(root_dir, list_dir, transform=transforms.Compose([RandomRotFlip(), ToTensor()]), img_size=img_size)
    db_val = VolumeDataSet(root_dir, list_dir, "val", transforms.Compose([ToTensor()]), img_size=img_size)

    # Deterministic training
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Create dataloader
    train_loader = DataLoader(
        db_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    max_iterations = max_epochs * len(train_loader)
    val_loader = DataLoader(db_val, batch_size=1)

    # Choose optimizer and loss function
    optimizer = optim.SGD(model.parameters(), base_lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    iter_num = 0
    best_perfermance = .0
    for epoch_num in range(1, max_epochs + 1):
        model.train()
        for batch in track(train_loader, description="Epoch {}/{}".format(epoch_num, max_epochs)):
            images, labels = batch["image"], batch["label"]
            images, labels = images.cuda(), labels.cuda()

            saliences = batch["salience"].cuda()
            inputs = (saliences + 1) * images
            # inputs = images

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            writer.add_scalar("loss", loss.item(), iter_num)
            writer.add_scalar("learning_rate", lr_, iter_num)
        imgs = torch.stack([images[0], torch.sigmoid(outputs)[0], labels[0]])
        writer.add_image(f"images", imgs, epoch_num, dataformats="NCHW")

        # Validation
        if epoch_num >= 1:
            model.eval()
            metrics = .0
            for batch in track(val_loader, description="Evaluating"):
                images, labels = batch["image"], batch["label"]
                saliences = batch["salience"]
                # inputs = images
                inputs = (saliences + 1) * images
                metrics += np.array(inference(inputs, labels, model, 'eval'))
            metrics = metrics / len(val_loader)
            logger.info(
                "[Epoch{:0>4d}] Dice:{:.4f} clDice:{:.4f}".format(epoch_num, *metrics)
            )          

            # Save best model
            if metrics[0] > best_perfermance:
                logger.info("Dice increased! ({:.4f} -> {:.4f})\n".format(best_perfermance, metrics[0]))
                best_perfermance = metrics[0]
                save_model_path = os.path.join(snapshot_path, "best_model.pth")
                torch.save(model.state_dict(), save_model_path)

    logger.info("Training finished\n\n")
    return "Training finished!"