"""Training loop for SAM3 CellMap fine-tuning.

Handles:
- Video-mode training (z-slices as frames with memory propagation)
- Mixed precision (AMP) with gradient scaling
- Gradient accumulation for effective larger batches
- Per-class Dice validation
- Checkpoint saving/loading (LoRA params + head only)
- WandB / TensorBoard logging
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ZConsistencyLoss(nn.Module):
    """Penalize prediction discontinuities across adjacent z-slices.

    Only penalizes where labels are actually consistent (don't penalize
    at real organelle boundaries between slices).

    L_z = mean(|p_z - p_{z+1}|^2 * (1 - |y_z - y_{z+1}|))
    """

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: [T, C, H, W] logits across z-slices.
            labels: [T, C, H, W] binary labels across z-slices.
        """
        if predictions.shape[0] < 2:
            return torch.tensor(0.0, device=predictions.device)

        probs = torch.sigmoid(predictions)

        # Prediction difference between adjacent slices
        pred_diff = (probs[1:] - probs[:-1]).pow(2)

        # Label difference — weight: only penalize where labels are consistent
        label_diff = (labels[1:] - labels[:-1]).abs()
        weight = 1.0 - label_diff

        return (pred_diff * weight).mean()


class CellMapTrainer:
    """Training loop for SAM3 CellMap fine-tuning.

    Per training step (Mode A):
    1. Sample video batch from CellMapVideoDataset
    2. For each z-slice: forward through SAM3 + CellMap head
    3. Compute per-slice masked BCE+Dice + hierarchical loss
    4. Add z-consistency loss across slice predictions
    5. Backward + optimizer step (with gradient accumulation)

    Args:
        model: SAM3CellMapModel.
        loss_fn: CellMapLoss from losses module.
        optimizer: torch optimizer (only LoRA + head params).
        scheduler: LR scheduler.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        config: Training configuration dict.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        config: Dict = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

        self.device = next(model.parameters()).device
        self.use_amp = self.config.get("use_amp", True)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.accumulation_steps = self.config.get("accumulation_steps", 8)
        loss_cfg = self.config.get("loss", {})
        self.z_consistency_weight = loss_cfg.get("z_consistency_weight", 0.1)
        self.z_consistency_loss = ZConsistencyLoss()

        self.global_step = 0
        self.epoch = 0
        self.best_val_dice = 0.0

        # Checkpoint directory
        self.checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {}
        n_batches = 0
        max_epochs = self.config.get("max_epochs", 1)
        progress = self.epoch / max(max_epochs, 1)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            loss, log_dict = self._train_step(batch, progress)

            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate metrics
            for k, v in log_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v if isinstance(v, float) else v.item()
            n_batches += 1

        # Average metrics
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        if self.scheduler is not None:
            self.scheduler.step()

        self.epoch += 1
        return epoch_losses

    def _train_step(
        self, batch: Dict[str, torch.Tensor], progress: float
    ) -> tuple:
        """Process one video batch (z-stack as frames).

        Args:
            batch: Dict from CellMapVideoDataset.__getitem__
            progress: Training progress 0->1 for dynamic loss weighting.

        Returns:
            (total_loss, log_dict)
        """
        # DataLoader adds batch dim: [B, T, ...] -> squeeze to [T, ...]
        # since we use batch_size=1 and process frames individually
        images = batch["images"].squeeze(0).to(self.device)          # [T, 3, H, W]
        labels = batch["labels"].squeeze(0).to(self.device)           # [T, C, Hm, Wm]
        annotated_mask = batch["annotated_mask"].squeeze(0).to(self.device)  # [C]
        spatial_masks = batch["spatial_masks"].squeeze(0).to(self.device)    # [T, 1, Hm, Wm]

        T = images.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        all_logits = []
        log_dict = {}

        with autocast(device_type="cuda", enabled=self.use_amp):
            # Process each z-slice through the model
            for t in range(T):
                frame = images[t].unsqueeze(0)  # [1, 3, H, W]
                outputs = self.model(frame, target_size=labels.shape[-2:])

                # Per-slice loss (unsqueeze to add dummy depth dim for 5D loss)
                slice_labels = labels[t].unsqueeze(0).unsqueeze(2)  # [1, C, 1, Hm, Wm]
                slice_mask = annotated_mask.unsqueeze(0)  # [1, C]
                slice_spatial = spatial_masks[t].unsqueeze(0).unsqueeze(2)  # [1, 1, 1, Hm, Wm]

                # Also unsqueeze the logits to 5D
                outputs_5d = {}
                for k, v in outputs.items():
                    outputs_5d[k] = v.unsqueeze(2)  # [1, C, 1, H, W]

                slice_loss, slice_log = self.loss_fn(
                    outputs_5d, slice_labels, slice_mask,
                    progress=progress, spatial_mask=slice_spatial,
                )
                total_loss = total_loss + slice_loss / T

                all_logits.append(outputs["fine"].squeeze(0))  # [C, H, W]

            # Z-consistency loss
            if T > 1 and self.z_consistency_weight > 0:
                stacked_logits = torch.stack(all_logits, dim=0)  # [T, C, H, W]
                stacked_labels = labels  # [T, C, Hm, Wm]

                # Resize logits to match label size if different
                if stacked_logits.shape[-2:] != stacked_labels.shape[-2:]:
                    stacked_logits = F.interpolate(
                        stacked_logits,
                        size=stacked_labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                z_loss = self.z_consistency_loss(stacked_logits, stacked_labels)
                total_loss = total_loss + self.z_consistency_weight * z_loss
                log_dict["z_consistency_loss"] = z_loss.detach().item()

        log_dict["total_loss"] = total_loss.detach().item()
        return total_loss, log_dict

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute per-class Dice scores."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        n_classes = self.model.cellmap_head.n_fine

        # Accumulate per-class dice components
        dice_inter = torch.zeros(n_classes, device=self.device)
        dice_union = torch.zeros(n_classes, device=self.device)
        dice_count = torch.zeros(n_classes, device=self.device)

        for batch in self.val_loader:
            images = batch["images"].squeeze(0).to(self.device)
            labels = batch["labels"].squeeze(0).to(self.device)
            annotated_mask = batch["annotated_mask"].squeeze(0).to(self.device)
            spatial_masks = batch["spatial_masks"].squeeze(0).to(self.device)

            T = images.shape[0]

            for t in range(T):
                frame = images[t].unsqueeze(0)
                outputs = self.model(frame, target_size=labels.shape[-2:])
                probs = torch.sigmoid(outputs["fine"]).squeeze(0)  # [C, H, W]

                pred_binary = (probs > 0.5).float()
                gt = labels[t]  # [C, Hm, Wm]
                sp = spatial_masks[t]  # [1, Hm, Wm]

                for c in range(n_classes):
                    if not annotated_mask[c]:
                        continue
                    p = pred_binary[c] * sp.squeeze(0)
                    g = gt[c] * sp.squeeze(0)
                    inter = (p * g).sum()
                    union = p.sum() + g.sum()
                    if union > 0:
                        dice_inter[c] += inter
                        dice_union[c] += union
                        dice_count[c] += 1

        # Compute per-class dice
        metrics = {}
        valid_dice = []
        for c in range(n_classes):
            if dice_count[c] > 0:
                dice = (2 * dice_inter[c] / (dice_union[c] + 1e-8)).item()
                metrics[f"val_dice/{c}"] = dice
                valid_dice.append(dice)

        if valid_dice:
            metrics["val_dice/mean"] = sum(valid_dice) / len(valid_dice)

        return metrics

    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save checkpoint with LoRA params + head weights only."""
        from sam3m.model.lora import lora_state_dict

        if path is None:
            path = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch{self.epoch}.pt"
            )

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_dice": self.best_val_dice,
            "lora_state_dict": lora_state_dict(self.model),
            "cellmap_head_state_dict": self.model.cellmap_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(state, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_dice = state.get("best_val_dice", 0.0)

        # Load LoRA params
        lora_sd = state["lora_state_dict"]
        model_sd = self.model.state_dict()
        model_sd.update(lora_sd)
        self.model.load_state_dict(model_sd, strict=False)

        # Load head
        self.model.cellmap_head.load_state_dict(state["cellmap_head_state_dict"])

        # Load optimizer
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
