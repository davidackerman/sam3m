#!/usr/bin/env python3
"""Main training entry point for SAM3 CellMap fine-tuning.

Usage:
    pixi run train
    python scripts/train.py
    python scripts/train.py --config configs/train_video.yaml
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train SAM3 for CellMap")
    parser.add_argument(
        "--config", default="configs/train_video.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded config from {args.config}")

    # ---- Data ----
    from sam3m.data.dataset import CellMapDataset3D
    from sam3m.data.video_dataset import CellMapVideoDataset
    from sam3m.data.transforms import get_train_transforms
    from sam3m.data.sampler import ClassBalancedSampler
    from sam3m.training.split import split_dataset

    data_cfg = cfg["data"]
    base_dataset = CellMapDataset3D(
        data_root=data_cfg["data_root"],
        norms_csv=data_cfg.get("norms_csv"),
        target_resolution=data_cfg["target_resolution"],
        patch_size=tuple(data_cfg["patch_size"]),
        samples_per_epoch=data_cfg["samples_per_epoch"],
        skip_datasets=data_cfg.get("skip_datasets", []),
    )
    logger.info(f"\n{base_dataset.summary()}")

    # Train/val split
    train_base, val_base = split_dataset(
        base_dataset,
        val_fraction=cfg["training"].get("val_fraction", 0.1),
    )

    # Wrap in video dataset
    train_dataset = CellMapVideoDataset(
        train_base,
        num_frames=data_cfg.get("num_frames", 16),
        frame_stride=data_cfg.get("frame_stride", 8),
        image_size=data_cfg.get("image_size", 1008),
        mask_size=data_cfg.get("mask_size", 256),
    )
    val_dataset = CellMapVideoDataset(
        val_base,
        num_frames=data_cfg.get("num_frames", 16),
        frame_stride=data_cfg.get("frame_stride", 8),
        image_size=data_cfg.get("image_size", 1008),
        mask_size=data_cfg.get("mask_size", 256),
    )

    train_cfg = cfg["training"]

    # Sampler
    sampler = None
    if train_cfg.get("balanced_sampling", True):
        sampler = ClassBalancedSampler(
            train_base, samples_per_epoch=data_cfg["samples_per_epoch"]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 1),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 8),
        prefetch_factor=train_cfg.get("prefetch_factor", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=True,
    )

    # ---- Model ----
    from sam3m.model.sam3_cellmap import build_cellmap_model

    model_cfg = cfg["model"]
    lora_cfg = model_cfg.get("lora", {})
    head_cfg = model_cfg.get("heads", {})

    model = build_cellmap_model(
        sam3_checkpoint=model_cfg.get("sam3_checkpoint"),
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 8.0),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_targets=lora_cfg.get("targets", ["attn.qkv", "attn.proj"]),
        n_fine=head_cfg.get("n_fine", 48),
        n_medium=head_cfg.get("n_medium", 17),
        n_coarse=head_cfg.get("n_coarse", 7),
        use_auxiliary=head_cfg.get("use_auxiliary", True),
    )

    # ---- Loss ----
    from sam3m.data.class_mapping import fine_to_medium_matrix, fine_to_coarse_matrix
    from sam3m.data.dataset import EVALUATED_INSTANCE_CLASSES, EVALUATED_CLASSES
    from sam3m.losses import CellMapLoss

    instance_indices = [
        EVALUATED_CLASSES.index(c)
        for c in EVALUATED_INSTANCE_CLASSES
        if c in EVALUATED_CLASSES
    ]

    loss_cfg = train_cfg.get("loss", {})
    loss_fn = CellMapLoss(
        fine_to_medium=fine_to_medium_matrix(),
        fine_to_coarse=fine_to_coarse_matrix(),
        instance_indices=instance_indices,
        boundary_weight=loss_cfg.get("boundary_weight", 5.0),
        boundary_loss_weight=loss_cfg.get("boundary_loss_weight", 0.1),
        dynamic_weights=loss_cfg.get("dynamic_weights", True),
    ).to(model.device if hasattr(model, 'device') else "cuda")

    # ---- Optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.get("lr", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_cfg.get("T_0", 50),
        T_mult=train_cfg.get("T_mult", 2),
        eta_min=train_cfg.get("eta_min", 1e-6),
    )

    # ---- Trainer ----
    from sam3m.training.trainer import CellMapTrainer

    trainer = CellMapTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---- Training loop ----
    max_epochs = train_cfg.get("max_epochs", 200)
    val_every = train_cfg.get("val_every_n_epochs", 5)
    save_every = train_cfg.get("save_every_n_epochs", 10)

    logger.info(f"Starting training for {max_epochs} epochs")

    for epoch in range(trainer.epoch, max_epochs):
        train_metrics = trainer.train_epoch()
        logger.info(
            f"Epoch {epoch}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        )

        # Validate
        if (epoch + 1) % val_every == 0:
            val_metrics = trainer.validate()
            mean_dice = val_metrics.get("val_dice/mean", 0.0)
            logger.info(f"Validation: mean_dice={mean_dice:.4f}")

            if mean_dice > trainer.best_val_dice:
                trainer.best_val_dice = mean_dice
                trainer.save_checkpoint(is_best=True)

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint()

    # Final checkpoint
    trainer.save_checkpoint()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
