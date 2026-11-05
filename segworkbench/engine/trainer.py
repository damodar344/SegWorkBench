"""Training engine for UNet segmentation."""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

from segworkbench.configs.model_config import CheckpointManager, ModelConfig
from segworkbench.models.unet import UNet

logger = logging.getLogger(__name__)


def _auto_device(device: Optional[torch.device] = None) -> torch.device:
    """Pick the best available accelerator."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def _limit_cpu_threads() -> None:
    """Cap PyTorch CPU threads so the GUI stays responsive."""
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
        logger.debug("Limited torch CPU threads to %d", torch.get_num_threads())


def _dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute Dice coefficient from raw logits and target mask."""
    pred = torch.sigmoid(pred_logits) > 0.5
    target = target > 0.5
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred & target).float().sum()
    return float((2.0 * intersection + eps) / (pred.float().sum() + target.float().sum() + eps))


def _iou_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute IoU from raw logits and target mask."""
    pred = torch.sigmoid(pred_logits) > 0.5
    target = target > 0.5
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return float((intersection + eps) / (union + eps))


class SegDataset(Dataset):
    """Dataset that pairs images with masks by matching basenames."""

    _VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, image_dir: str, mask_dir: str, size: int = 256, augment: bool = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment
        self.files = self._match_files()

    def _match_files(self):
        imgs = {
            os.path.splitext(f)[0].lower(): f
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(self._VALID_EXTS)
        }
        msks = [
            f
            for f in os.listdir(self.mask_dir)
            if f.lower().endswith(self._VALID_EXTS)
        ]
        pairs = []
        for m in msks:
            b = os.path.splitext(m)[0].lower()
            if b in imgs:
                pairs.append((imgs[b], m))
        return pairs

    @staticmethod
    def _random_affine(img, msk):
        """Apply identical random flip / rotation to image and mask."""
        if torch.rand(1) < 0.5:
            img = T.functional.hflip(img)
            msk = T.functional.hflip(msk)
        if torch.rand(1) < 0.5:
            img = T.functional.vflip(img)
            msk = T.functional.vflip(msk)
        angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())
        img = T.functional.rotate(img, angle)
        msk = T.functional.rotate(msk, angle, interpolation=T.InterpolationMode.NEAREST)
        return img, msk

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        imgf, mskf = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, imgf)).convert("RGB")
        msk = Image.open(os.path.join(self.mask_dir, mskf)).convert("L")

        img = T.Resize((self.size, self.size))(img)
        msk = T.Resize((self.size, self.size), interpolation=T.InterpolationMode.NEAREST)(msk)

        if self.augment:
            img, msk = self._random_affine(img, msk)

        img = T.ToTensor()(img)
        img = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        msk = T.ToTensor()(msk)
        return img, msk


class Trainer:
    """UNet training runner with metrics, LR scheduling, and early stopping."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = _auto_device(device)
        if self.device.type == "cpu":
            _limit_cpu_threads()

    def train(
        self,
        image_dir: str,
        mask_dir: str,
        save_dir: str,
        epochs: int = 5,
        batch_size: int = 2,
        lr: float = 1e-4,
        image_size: int = 256,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        config: Optional[ModelConfig] = None,
        stop_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        on_epoch_end: Optional[
            Callable[[int, int, float, float, float, float], None]
        ] = None,
    ) -> dict:
        """Run training and return metrics / saved model path.

        Args:
            image_dir: Folder with training images.
            mask_dir: Folder with corresponding masks.
            save_dir: Folder to write the best model checkpoint.
            epochs: Number of training epochs.
            batch_size: Batch size for DataLoader.
            lr: Learning rate for AdamW.
            image_size: Spatial square size to resize images to.
            val_ratio: Fraction of data used for validation.
            num_workers: Workers for DataLoader (0 for main-thread loading).
            config: Optional ``ModelConfig`` that overrides the above
                hyperparameters when provided.
            stop_event: When set, training stops early.
            pause_event: When set, training pauses between batches.
            on_epoch_end: Callback(epoch, total_epochs, train_loss, val_loss, dice, iou).

        Returns:
            Dictionary with keys:
                - best_model_path: path to saved .pth
                - best_val_loss: best validation loss achieved
                - best_dice: best validation Dice score
                - best_iou: best validation IoU score
                - train_losses: list of average train losses per epoch
                - val_losses: list of average validation losses per epoch
                - dice_scores: list of validation Dice scores per epoch
                - iou_scores: list of validation IoU scores per epoch
        """
        # Apply ModelConfig overrides if given
        patience = 5
        min_delta = 1e-3
        if config is not None:
            epochs = config.epochs
            batch_size = config.batch_size
            lr = config.learning_rate
            image_size = config.image_size[0]
            patience = config.patience
            min_delta = config.min_delta

        if stop_event is None:
            stop_event = threading.Event()
        if pause_event is None:
            pause_event = threading.Event()

        train_dataset = SegDataset(image_dir, mask_dir, size=image_size, augment=True)
        if len(train_dataset) == 0:
            raise ValueError(
                "No matched image-mask pairs found in the given directories."
            )

        # Train/val split - create a separate non-augmented dataset for validation
        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size
        if val_size == 0:
            val_size = 1
            train_size = len(train_dataset) - 1
        if train_size <= 0:
            raise ValueError(
                "Not enough data for train/val split (need >1 sample)."
            )

        train_indices, val_indices = random_split(
            range(len(train_dataset)), [train_size, val_size]
        )

        # Re-use same file list but disable augmentation for validation
        val_dataset = SegDataset(image_dir, mask_dir, size=image_size, augment=False)
        val_dataset.files = train_dataset.files  # ensure identical pairing

        # Build subset samplers manually so train gets augmentation and val doesn't
        train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        model = UNet(n_channels=3, n_classes=1).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(1, patience // 2)
        )

        ckpt_mgr = CheckpointManager(save_dir, keep_last_n=3)
        best_val_loss = float("inf")
        best_model_state = None
        train_losses = []
        val_losses = []
        dice_scores = []
        iou_scores = []

        patience_counter = 0

        logger.info(
            "Starting training: epochs=%d, batch=%d, lr=%.2e, image_size=%d, workers=%d, patience=%d, device=%s",
            epochs, batch_size, lr, image_size, num_workers, patience, self.device,
        )

        for epoch in range(1, epochs + 1):
            if stop_event.is_set():
                logger.info("Training stopped by user at epoch %d", epoch)
                break

            logger.debug("Epoch %d/%d - training", epoch, epochs)

            # --- Training ---
            model.train()
            ep_loss = 0.0
            nb = 0

            for imgs, msks in train_loader:
                while pause_event.is_set() and not stop_event.is_set():
                    time.sleep(0.1)
                if stop_event.is_set():
                    break

                imgs = imgs.to(self.device)
                msks = msks.to(self.device)

                opt.zero_grad()
                out = model(imgs)
                loss = criterion(out, msks)
                loss.backward()
                opt.step()

                ep_loss += float(loss.item())
                nb += 1

            avg_train_loss = ep_loss / nb if nb > 0 else 0.0
            train_losses.append(avg_train_loss)

            logger.debug("Epoch %d/%d - validation", epoch, epochs)

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            val_nb = 0
            dice_sum = 0.0
            iou_sum = 0.0
            with torch.no_grad():
                for imgs, msks in val_loader:
                    imgs = imgs.to(self.device)
                    msks = msks.to(self.device)
                    out = model(imgs)
                    loss = criterion(out, msks)
                    val_loss += float(loss.item())
                    val_nb += 1
                    dice_sum += _dice_score(out, msks)
                    iou_sum += _iou_score(out, msks)

            avg_val_loss = val_loss / val_nb if val_nb > 0 else 0.0
            avg_dice = dice_sum / val_nb if val_nb > 0 else 0.0
            avg_iou = iou_sum / val_nb if val_nb > 0 else 0.0
            val_losses.append(avg_val_loss)
            dice_scores.append(avg_dice)
            iou_scores.append(avg_iou)

            scheduler.step(avg_val_loss)
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else float(lr)

            logger.info(
                "Epoch %d/%d | Train: %.4f | Val: %.4f | Dice: %.4f | IoU: %.4f | LR: %.2e",
                epoch, epochs, avg_train_loss, avg_val_loss, avg_dice, avg_iou, current_lr,
            )

            improved = avg_val_loss < best_val_loss - min_delta
            if improved:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                if config is not None:
                    ckpt_mgr.save(
                        model=model,
                        optimizer=opt,
                        epoch=epoch,
                        best_metric=best_val_loss,
                        config=config,
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

            if on_epoch_end is not None:
                on_epoch_end(
                    epoch, epochs, avg_train_loss, avg_val_loss, avg_dice, avg_iou
                )

        # Save best model
        os.makedirs(save_dir, exist_ok=True)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"{ts}_unet_best_val.pth")
        torch.save(model.state_dict(), out_path)
        logger.info("Training finished. Best model saved to %s", out_path)

        return {
            "best_model_path": out_path,
            "best_val_loss": best_val_loss,
            "best_dice": max(dice_scores) if dice_scores else 0.0,
            "best_iou": max(iou_scores) if iou_scores else 0.0,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "dice_scores": dice_scores,
            "iou_scores": iou_scores,
        }
