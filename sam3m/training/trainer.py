"""Training loop for SAM3 CellMap fine-tuning.

Handles:
- Z-stack training (z-slices as frames)
- Mixed precision (AMP) with gradient scaling
- Gradient accumulation for effective larger batches
- Per-class Dice validation
- Checkpoint saving/loading (LoRA params + head only)
- TensorBoard logging
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Classes to visualize (common, visually distinct organelles)
_VIS_CLASSES = ["mito", "er", "nuc", "golgi", "ld", "lyso", "ves", "endo"]
# Fixed colors per visualized class (RGB 0-1)
_VIS_COLORS = [
    (1.0, 0.0, 0.0),    # mito — red
    (0.0, 1.0, 0.0),    # er — green
    (0.0, 0.4, 1.0),    # nuc — blue
    (1.0, 1.0, 0.0),    # golgi — yellow
    (1.0, 0.5, 0.0),    # ld — orange
    (0.8, 0.0, 1.0),    # lyso — purple
    (0.0, 1.0, 1.0),    # ves — cyan
    (1.0, 0.4, 0.7),    # endo — pink
]


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
    1. Sample z-stack batch from CellMapZStackDataset
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
        tb_writer=None,
        log_every_n_steps: int = 10,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.tb_writer = tb_writer
        self.log_every_n_steps = log_every_n_steps
        self.log_images_every_n_steps = self.config.get("log_images_every_n_steps", 50)
        self.n_image_samples = self.config.get("n_image_samples", 4)

        self.device = next(model.parameters()).device
        # Unwrap DDP for attribute access (checkpointing, validation)
        self._base_model = model.module if hasattr(model, "module") else model
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

        # Resolve visualization class indices
        from sam3m.data.dataset import EVALUATED_CLASSES
        self._vis_indices = []
        self._vis_colors = []
        for cls_name, color in zip(_VIS_CLASSES, _VIS_COLORS):
            if cls_name in EVALUATED_CLASSES:
                self._vis_indices.append(EVALUATED_CLASSES.index(cls_name))
                self._vis_colors.append(color)

    @staticmethod
    def _draw_text_on_tensor(img: torch.Tensor, text: str) -> torch.Tensor:
        """Draw text label on a [3, H, W] image tensor (top-left corner)."""
        H, W = img.shape[-2:]
        pil = Image.fromarray(
            (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        )
        draw = ImageDraw.Draw(pil)
        font_size = max(12, H // 20)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        # Draw with black outline for readability
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((4 + dx, 4 + dy), text, fill=(0, 0, 0), font=font)
        draw.text((4, 4), text, fill=(255, 255, 255), font=font)
        return torch.from_numpy(
            np.array(pil, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)

    @torch.no_grad()
    def _log_sample_grid(
        self,
        tag: str,
        batches: list,
        step: int,
    ):
        """Log a multi-row grid of [raw | GT | prediction] to TensorBoard.

        Each row is one sample (middle z-slice of a batch). The grid has
        3 columns (raw, GT overlay, prediction overlay) and N rows.

        Args:
            tag: TensorBoard tag prefix ("train" or "val").
            batches: List of batch dicts from the DataLoader.
            step: TensorBoard step.
        """
        if self.tb_writer is None or not self._vis_indices or not batches:
            return

        all_panels = []  # flat list: [raw0, gt0, pred0, raw1, gt1, pred1, ...]

        for batch in batches:
            # Use first item in the batch for visualization
            images = batch["images"][:1].squeeze(0).to(self.device)   # [T, 3, H, W]
            labels = batch["labels"][:1].squeeze(0).to(self.device)    # [T, C, Hm, Wm]
            sf = batch.get("scale_factor")
            if sf is not None:
                sf = sf[:1].to(self.device)

            mid = images.shape[0] // 2
            with autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(
                    images[mid].unsqueeze(0), target_size=labels.shape[-2:],
                    scale_factor=sf,
                )

            image = images[mid]          # [3, H, W]
            lbl = labels[mid]            # [C, Hm, Wm]
            logits = out["fine"].squeeze(0)  # [C, Hm, Wm]

            H, W = lbl.shape[-2:]

            # Raw grayscale → [3, H, W]
            raw = F.interpolate(
                image.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0).cpu().clamp(0, 1)

            gt_overlay = raw.clone()
            pred_overlay = raw.clone()
            probs = torch.sigmoid(logits).cpu()
            labels_cpu = lbl.cpu().float()

            alpha = 0.5
            for idx, (r, g, b) in zip(self._vis_indices, self._vis_colors):
                color = torch.tensor([r, g, b], dtype=torch.float32).view(3, 1)

                gt_mask = labels_cpu[idx] > 0.5
                if gt_mask.any():
                    gt_overlay[:, gt_mask] = (
                        (1 - alpha) * gt_overlay[:, gt_mask]
                        + alpha * color.expand_as(gt_overlay[:, gt_mask])
                    )

                pred_mask = probs[idx] > 0.5
                if pred_mask.any():
                    pred_overlay[:, pred_mask] = (
                        (1 - alpha) * pred_overlay[:, pred_mask]
                        + alpha * color.expand_as(pred_overlay[:, pred_mask])
                    )

            # Draw crop name on the raw panel
            crop_name = batch.get("crop_name")
            if crop_name is not None:
                label_text = crop_name[0] if isinstance(crop_name, (list, tuple)) else crop_name
                raw = self._draw_text_on_tensor(raw, str(label_text))

            all_panels.extend([raw, gt_overlay, pred_overlay])

        # N rows × 3 columns
        grid = vutils.make_grid(all_panels, nrow=3, padding=4)
        self.tb_writer.add_image(f"{tag}/samples", grid, step)

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {}
        n_batches = 0
        max_epochs = self.config.get("max_epochs", 1)
        progress = self.epoch / max(max_epochs, 1)

        self.optimizer.zero_grad()
        saved_batches = []
        n_total = len(self.train_loader)
        # Evenly spaced indices to collect image samples from
        if n_total > 0 and self.n_image_samples > 0:
            save_indices = set(
                int(i * n_total / self.n_image_samples)
                for i in range(self.n_image_samples)
            )
        else:
            save_indices = set()

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx in save_indices and len(saved_batches) < self.n_image_samples:
                saved_batches.append(batch)
            loss, log_dict = self._train_step(batch, progress)

            # Scale loss for gradient accumulation
            loss = loss / self.accumulation_steps

            # Skip DDP gradient sync on intermediate accumulation steps
            is_accumulation_step = (batch_idx + 1) % self.accumulation_steps != 0
            sync_context = (
                self.model.no_sync()
                if is_accumulation_step and isinstance(self.model, DDP)
                else contextlib.nullcontext()
            )
            with sync_context:
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

                # Per-step TensorBoard logging
                if (
                    self.tb_writer is not None
                    and self.global_step % self.log_every_n_steps == 0
                ):
                    for k, v in log_dict.items():
                        self.tb_writer.add_scalar(f"train/{k}", v, self.global_step)
                    self.tb_writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                    )

                # Mid-epoch image logging (single sample)
                if (
                    self.tb_writer is not None
                    and self.global_step % self.log_images_every_n_steps == 0
                ):
                    self.model.eval()
                    self._log_sample_grid("train_step", [batch], self.global_step)
                    self.model.train()

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

        # Epoch-level TensorBoard logging
        if self.tb_writer is not None:
            for k, v in epoch_losses.items():
                self.tb_writer.add_scalar(f"train_epoch/{k}", v, self.epoch)
            self.tb_writer.add_scalar(
                "train_epoch/lr", self.optimizer.param_groups[0]["lr"], self.epoch
            )

            # Log training sample grid (multiple crops spread across the epoch)
            if saved_batches:
                self.model.eval()
                self._log_sample_grid("train", saved_batches, self.epoch)
                self.model.train()

        self.epoch += 1
        return epoch_losses

    def _train_step(
        self, batch: Dict[str, torch.Tensor], progress: float
    ) -> tuple:
        """Process one z-stack batch.

        Args:
            batch: Dict from CellMapZStackDataset.__getitem__
            progress: Training progress 0->1 for dynamic loss weighting.

        Returns:
            (total_loss, log_dict)
        """
        images = batch["images"].to(self.device)              # [B, T, 3, H, W]
        labels = batch["labels"].to(self.device)               # [B, T, C, Hm, Wm]
        annotated_mask = batch["annotated_mask"].to(self.device)  # [B, C]
        spatial_masks = batch["spatial_masks"].to(self.device)    # [B, T, 1, Hm, Wm]
        scale_factors = batch.get("scale_factor")
        if scale_factors is not None:
            scale_factors = scale_factors.to(self.device)  # [B]

        B, T = images.shape[:2]
        total_loss = torch.tensor(0.0, device=self.device)
        log_dict = {}

        with autocast(device_type="cuda", enabled=self.use_amp):
            for b in range(B):
                sf = scale_factors[b] if scale_factors is not None else None

                # Batch all T frames through backbone at once (T as batch dim)
                all_frames = images[b]  # [T, 3, H, W]
                outputs = self.model(
                    all_frames, target_size=labels.shape[-2:],
                    scale_factor=sf,
                )
                # outputs["fine"]: [T, C, Hm, Wm]

                # Reshape to 5D for loss: T becomes depth dim → [1, C, T, Hm, Wm]
                outputs_5d = {}
                for k, v in outputs.items():
                    outputs_5d[k] = v.permute(1, 0, 2, 3).unsqueeze(0)

                labels_5d = labels[b].permute(1, 0, 2, 3).unsqueeze(0)        # [1, C, T, Hm, Wm]
                spatial_5d = spatial_masks[b].permute(1, 0, 2, 3).unsqueeze(0)  # [1, 1, T, Hm, Wm]
                mask_1 = annotated_mask[b].unsqueeze(0)                         # [1, C]

                batch_loss, batch_log = self.loss_fn(
                    outputs_5d, labels_5d, mask_1,
                    progress=progress, spatial_mask=spatial_5d,
                )
                total_loss = total_loss + batch_loss / B

                # Z-consistency loss
                if T > 1 and self.z_consistency_weight > 0:
                    all_logits = outputs["fine"]  # [T, C, Hm, Wm]
                    stacked_labels = labels[b]     # [T, C, Hm, Wm]

                    if all_logits.shape[-2:] != stacked_labels.shape[-2:]:
                        all_logits = F.interpolate(
                            all_logits,
                            size=stacked_labels.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    z_loss = self.z_consistency_loss(all_logits, stacked_labels)
                    total_loss = total_loss + self.z_consistency_weight * z_loss / B
                    log_dict["z_consistency_loss"] = (
                        log_dict.get("z_consistency_loss", 0.0)
                        + z_loss.detach().item() / B
                    )

        log_dict["total_loss"] = total_loss.detach().item()
        return total_loss, log_dict

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute per-class Dice scores."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        n_classes = self._base_model.cellmap_head.n_fine

        # Accumulate per-class dice components
        dice_inter = torch.zeros(n_classes, device=self.device)
        dice_union = torch.zeros(n_classes, device=self.device)
        dice_count = torch.zeros(n_classes, device=self.device)

        val_image_batches = []
        n_val_total = len(self.val_loader)
        if n_val_total > 0 and self.n_image_samples > 0:
            val_save_indices = set(
                int(i * n_val_total / self.n_image_samples)
                for i in range(self.n_image_samples)
            )
        else:
            val_save_indices = set()

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx in val_save_indices and len(val_image_batches) < self.n_image_samples:
                val_image_batches.append(batch)

            images = batch["images"].to(self.device)              # [B, T, 3, H, W]
            labels = batch["labels"].to(self.device)               # [B, T, C, Hm, Wm]
            annotated_mask = batch["annotated_mask"].to(self.device)  # [B, C]
            spatial_masks = batch["spatial_masks"].to(self.device)    # [B, T, 1, Hm, Wm]
            scale_factors = batch.get("scale_factor")
            if scale_factors is not None:
                scale_factors = scale_factors.to(self.device)

            B, T = images.shape[:2]

            for b in range(B):
                sf = scale_factors[b] if scale_factors is not None else None

                # Batch all T frames through backbone at once
                all_frames = images[b]  # [T, 3, H, W]
                with autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(
                        all_frames, target_size=labels.shape[-2:],
                        scale_factor=sf,
                    )
                all_probs = torch.sigmoid(outputs["fine"])  # [T, C, Hm, Wm]

                for t in range(T):
                    pred_binary = (all_probs[t] > 0.5).float()
                    gt = labels[b, t]  # [C, Hm, Wm]
                    sp = spatial_masks[b, t]  # [1, Hm, Wm]

                    for c in range(n_classes):
                        if not annotated_mask[b, c]:
                            continue
                        p = pred_binary[c] * sp.squeeze(0)
                        g = gt[c] * sp.squeeze(0)
                        inter = (p * g).sum()
                        union = p.sum() + g.sum()
                        if union > 0:
                            dice_inter[c] += inter
                            dice_union[c] += union
                            dice_count[c] += 1

        # Log val sample grid
        if val_image_batches:
            self._log_sample_grid("val", val_image_batches, self.epoch)

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

        # TensorBoard validation logging
        if self.tb_writer is not None and metrics:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f"val/{k}", v, self.epoch)

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
            "lora_state_dict": lora_state_dict(self._base_model),
            "cellmap_head_state_dict": self._base_model.cellmap_head.state_dict(),
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
        model_sd = self._base_model.state_dict()
        model_sd.update(lora_sd)
        self._base_model.load_state_dict(model_sd, strict=False)

        # Load head
        self._base_model.cellmap_head.load_state_dict(state["cellmap_head_state_dict"])

        # Load optimizer
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
