# src/trainer/base_trainer.py
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    logits:  (B, 1, D, H, W)
    targets: (B, 1, D, H, W)
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


@torch.no_grad()
def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice.mean().item()


class BaseTrainer:
    """
    Trainer for 3D U-Net baseline and salience training.

    Baseline input:
        imgs: (B, 1, D, H, W)

    Salience input:
        imgs: (B, 1, D, H, W)
        converted to: (B, 3, D, H, W)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        cfg: dict,
        salience_gen=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.salience_gen = salience_gen
        self.use_salience = salience_gen is not None

        train_cfg = cfg.get("train", {})
        loss_cfg = cfg.get("loss", {})
        val_cfg = cfg.get("val", {})

        self.epochs = train_cfg.get("epochs", 100)
        self.lr = train_cfg.get("lr", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.save_dir = Path(train_cfg.get("save_dir", "checkpoints/unet3d"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.threshold = val_cfg.get("threshold", 0.5)

        self.bce_weight = loss_cfg.get("bce_weight", 0.5)
        self.dice_weight = loss_cfg.get("dice_weight", 0.5)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = self._build_scheduler(train_cfg)

        self.best_dice = -1.0
        self.start_epoch = 1

        print("=== Trainer Built ===")
        print(f"Epochs: {self.epochs}")
        print(f"LR: {self.lr}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"BCE weight: {self.bce_weight}")
        print(f"Dice weight: {self.dice_weight}")
        print(f"Use salience: {self.use_salience}")
        print(f"Save dir: {self.save_dir}")

    def _build_scheduler(self, train_cfg: dict):
        scheduler_cfg = train_cfg.get("scheduler", None)

        if scheduler_cfg is None:
            return None

        scheduler_type = scheduler_cfg.get("type", "none")

        if scheduler_type == "none":
            return None

        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=scheduler_cfg.get("eta_min", 1e-6),
            )

        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_cfg.get("step_size", 30),
                gamma=scheduler_cfg.get("gamma", 0.1),
            )

        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _apply_salience_to_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert original batch to salience-enhanced batch.

        Input:
            imgs: (B, 1, D, H, W)

        SalienceGenerator expected input:
            volume: (H, W, D)

        SalienceGenerator expected output:
            enhanced: (3, H, W, D)

        Return:
            enhanced batch: (B, 3, D, H, W)
        """
        if not self.use_salience:
            return imgs

        if imgs.ndim != 5:
            raise ValueError(f"Expected imgs shape (B, 1, D, H, W), got {imgs.shape}")

        if imgs.size(1) != 1:
            raise ValueError(
                f"Salience input must have one original image channel, got {imgs.size(1)}"
            )

        enhanced_list = []

        # Salience is a deterministic preprocessing plugin; no gradient is needed.
        with torch.no_grad():
            for b in range(imgs.size(0)):
                volume_dhw = imgs[b, 0]  # (D, H, W)
                volume_hwd = volume_dhw.permute(1, 2, 0).contiguous()  # (H, W, D)

                enhanced_hwd = self.salience_gen(volume_hwd)

                if isinstance(enhanced_hwd, tuple):
                    enhanced_hwd = enhanced_hwd[0]

                if not isinstance(enhanced_hwd, torch.Tensor):
                    enhanced_hwd = torch.as_tensor(enhanced_hwd)

                enhanced_hwd = enhanced_hwd.to(
                    device=imgs.device,
                    dtype=imgs.dtype,
                    non_blocking=True,
                )

                if enhanced_hwd.ndim != 4:
                    raise ValueError(
                        "SalienceGenerator must return shape (3, H, W, D). "
                        f"Got {enhanced_hwd.shape}"
                    )

                if enhanced_hwd.size(0) != 3:
                    raise ValueError(
                        "SalienceGenerator must return 3 channels. "
                        f"Got shape {enhanced_hwd.shape}"
                    )

                # (3, H, W, D) -> (3, D, H, W)
                enhanced_dhw = enhanced_hwd.permute(0, 3, 1, 2).contiguous()
                enhanced_list.append(enhanced_dhw)

        return torch.stack(enhanced_list, dim=0).contiguous()

    def compute_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = self.bce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss, bce, dice

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()

        total_loss = 0.0
        total_dice = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d} [Train]",
            leave=False,
        )

        for batch in pbar:
            imgs = batch["img"].to(self.device, non_blocking=True).float()
            masks = batch["mask"].to(self.device, non_blocking=True).float()

            imgs = self._apply_salience_to_batch(imgs)
            logits = self.model(imgs)

            if logits.shape != masks.shape:
                raise RuntimeError(
                    f"Logits shape {logits.shape} does not match mask shape {masks.shape}."
                )

            loss, bce, dice_loss = self.compute_loss(logits, masks)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            batch_dice = dice_score_from_logits(
                logits=logits,
                targets=masks,
                threshold=self.threshold,
            )

            total_loss += loss.item()
            total_dice += batch_dice

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                bce=f"{bce.item():.4f}",
                dice_loss=f"{dice_loss.item():.4f}",
                dice=f"{batch_dice:.4f}",
            )

        num_batches = len(self.train_loader)
        return total_loss / num_batches, total_dice / num_batches

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()

        total_loss = 0.0
        total_dice = 0.0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:03d} [Val]",
            leave=False,
        )

        for batch in pbar:
            imgs = batch["img"].to(self.device, non_blocking=True).float()
            masks = batch["mask"].to(self.device, non_blocking=True).float()

            imgs = self._apply_salience_to_batch(imgs)
            logits = self.model(imgs)

            if logits.shape != masks.shape:
                raise RuntimeError(
                    f"Logits shape {logits.shape} does not match mask shape {masks.shape}."
                )

            loss, bce, dice_loss = self.compute_loss(logits, masks)

            batch_dice = dice_score_from_logits(
                logits=logits,
                targets=masks,
                threshold=self.threshold,
            )

            total_loss += loss.item()
            total_dice += batch_dice

            pbar.set_postfix(
                val_loss=f"{loss.item():.4f}",
                val_dice=f"{batch_dice:.4f}",
            )

        num_batches = len(self.val_loader)
        return total_loss / num_batches, total_dice / num_batches

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        train_dice: float,
        val_loss: float,
        val_dice: float,
        is_best: bool,
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "best_dice": self.best_dice,
            "use_salience": self.use_salience,
            "cfg": self.cfg,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        last_path = self.save_dir / "last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.save_dir / "best.pth"
            torch.save(checkpoint, best_path)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss, train_dice = self.train_one_epoch(epoch)
            val_loss, val_dice = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice

            self.save_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                train_dice=train_dice,
                val_loss=val_loss,
                val_dice=val_dice,
                is_best=is_best,
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f} | "
                f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f} | "
                f"best_dice={self.best_dice:.4f}"
            )

