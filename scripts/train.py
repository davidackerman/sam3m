#!/usr/bin/env python3
"""Main training entry point for SAM3 CellMap fine-tuning.

Single-GPU:
    pixi run train
    python scripts/train.py
    python scripts/train.py --config configs/train.yaml

Multi-GPU (DDP):
    torchrun --nproc_per_node=8 scripts/train.py
    torchrun --nproc_per_node=8 scripts/train.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from datetime import datetime

import torch
import torch.backends.cudnn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    """Initialize DDP from torchrun environment variables.

    Returns:
        (rank, world_size, local_rank).  Falls back to (0, 1, 0) when not
        launched with torchrun (single-GPU mode).
    """
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def setup_run_dir(cfg: dict, run_name: str | None, config_path: str) -> str:
    """Create a timestamped run directory and snapshot the config into it.

    Directory layout::

        <run_dir_base>/<run_name>/
        ├── config.yaml      # full frozen config
        ├── train.log        # file log
        └── checkpoints/     # saved during training

    Returns the absolute path to the run directory.
    """
    base = cfg.get("logging", {}).get("run_dir", "runs")
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Checkpoints subdirectory
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Override checkpoint_dir so trainer saves inside the run dir
    cfg.setdefault("training", {})["checkpoint_dir"] = ckpt_dir

    # Snapshot full config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Copy original config file for reference
    shutil.copy2(config_path, os.path.join(run_dir, "original_config.yaml"))

    # Add file handler to root logger
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train SAM3 for CellMap")
    parser.add_argument(
        "--config", default="configs/train.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--run-name", default=None,
        help="Name for this run (default: YYYY-MM-DD_HH-MM-SS)",
    )
    args = parser.parse_args()

    # ---- DDP setup ----
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)

    # Suppress verbose logging on non-main ranks
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Run directory (main rank only — others just need checkpoint_dir)
    if is_main:
        run_dir = setup_run_dir(cfg, args.run_name, args.config)
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Loaded config from {args.config}")
        if world_size > 1:
            logger.info(f"DDP: {world_size} GPUs")
    else:
        run_dir = None

    # ---- Data ----
    from sam3m.data.dataset import CellMapDataset3D
    from sam3m.data.zstack_dataset import CellMapZStackDataset
    from sam3m.data.transforms import get_train_transforms
    from sam3m.data.sampler import ClassBalancedSampler
    from sam3m.training.split import split_dataset

    data_cfg = cfg["data"]
    scale_factors = data_cfg.get("scale_factors", [1])
    base_dataset = CellMapDataset3D(
        data_root=data_cfg["data_root"],
        norms_csv=data_cfg.get("norms_csv"),
        target_resolution=data_cfg["target_resolution"],
        patch_size=tuple(data_cfg["patch_size"]),
        samples_per_epoch=data_cfg["samples_per_epoch"],
        skip_datasets=data_cfg.get("skip_datasets", []),
        include_datasets=data_cfg.get("include_datasets"),
        challenge_split=data_cfg.get("challenge_split"),
        scale_factors=scale_factors,
    )
    if is_main:
        logger.info(f"\n{base_dataset.summary()}")

    # Train/val split
    train_base, val_base = split_dataset(
        base_dataset,
        val_fraction=cfg["training"].get("val_fraction", 0.1),
    )

    # Wrap in z-stack dataset
    train_transforms = get_train_transforms()
    train_dataset = CellMapZStackDataset(
        train_base,
        num_frames=data_cfg.get("num_frames", 16),
        frame_stride=data_cfg.get("frame_stride", 8),
        image_size=data_cfg.get("image_size", 1008),
        mask_size=data_cfg.get("mask_size", 256),
        transforms=train_transforms,
    )
    val_dataset = CellMapZStackDataset(
        val_base,
        num_frames=data_cfg.get("num_frames", 16),
        frame_stride=data_cfg.get("frame_stride", 8),
        image_size=data_cfg.get("image_size", 1008),
        mask_size=data_cfg.get("mask_size", 256),
    )

    train_cfg = cfg["training"]

    # Auto-detect available CPUs, divided across DDP ranks
    available_cpus = len(os.sched_getaffinity(0)) // world_size
    num_workers = train_cfg.get("num_workers", max(available_cpus, 1))
    prefetch_factor = train_cfg.get("prefetch_factor", max(num_workers, 2))
    if is_main:
        logger.info(f"DataLoader: num_workers={num_workers}, prefetch_factor={prefetch_factor}")

    # Sampler (DDP-aware: shards across ranks)
    sampler = None
    if train_cfg.get("balanced_sampling", True):
        sampler = ClassBalancedSampler(
            train_base,
            samples_per_epoch=data_cfg["samples_per_epoch"],
            rank=rank,
            world_size=world_size,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 1),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    # Validation only on main rank (infrequent, simpler than distributed val)
    val_loader = None
    if is_main:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    # ---- Model ----
    from sam3m.model.sam3_cellmap import build_cellmap_model

    model_cfg = cfg["model"]
    lora_cfg = model_cfg.get("lora", {})
    head_cfg = model_cfg.get("heads", {})

    use_scale_conditioning = len(scale_factors) > 1
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
        use_scale_conditioning=use_scale_conditioning,
    )

    # Wrap in DDP (after model is on device, before optimizer creation)
    if world_size > 1:
        dist.barrier()  # ensure all ranks finished model loading
        model = DDP(model, device_ids=[local_rank])

    # ---- Loss ----
    from sam3m.data.class_mapping import fine_to_medium_matrix, fine_to_coarse_matrix
    from sam3m.data.dataset import EVALUATED_INSTANCE_CLASSES, EVALUATED_CLASSES
    from sam3m.losses import CellMapLoss

    instance_indices = [
        EVALUATED_CLASSES.index(c)
        for c in EVALUATED_INSTANCE_CLASSES
        if c in EVALUATED_CLASSES
    ]

    device = f"cuda:{local_rank}"
    loss_cfg = train_cfg.get("loss", {})
    loss_fn = CellMapLoss(
        fine_to_medium=fine_to_medium_matrix(),
        fine_to_coarse=fine_to_coarse_matrix(),
        instance_indices=instance_indices,
        boundary_weight=loss_cfg.get("boundary_weight", 5.0),
        boundary_loss_weight=loss_cfg.get("boundary_loss_weight", 0.1),
        dynamic_weights=loss_cfg.get("dynamic_weights", True),
    ).to(device)

    # ---- Optimizer (per-group LRs: head learns from scratch, LoRA adapts) ----
    base_model = model.module if hasattr(model, "module") else model
    head_params = list(base_model.cellmap_head.parameters())
    head_ids = {id(p) for p in head_params}
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    lr = train_cfg.get("lr", 2e-4)
    head_lr = train_cfg.get("head_lr", lr * 5)  # 5x base LR for randomly-init head
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )
    if is_main:
        logger.info(f"Optimizer: LoRA lr={lr}, head lr={head_lr}")

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_cfg.get("T_0", 50),
        T_mult=train_cfg.get("T_mult", 2),
        eta_min=train_cfg.get("eta_min", 1e-6),
    )

    # ---- Performance tuning ----
    torch.backends.cudnn.benchmark = True

    # ---- Device info ----
    if is_main and torch.cuda.is_available():
        dev = torch.cuda.current_device()
        logger.info(
            f"CUDA device {dev}: {torch.cuda.get_device_name(dev)}, "
            f"memory: {torch.cuda.get_device_properties(dev).total_memory / 1e9:.1f} GB"
        )
        logger.info(
            f"GPU memory allocated: {torch.cuda.memory_allocated(dev) / 1e9:.2f} GB, "
            f"reserved: {torch.cuda.memory_reserved(dev) / 1e9:.2f} GB"
        )

    # ---- TensorBoard (main rank only) ----
    tb_writer = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter

        log_cfg = cfg.get("logging", {})
        tb_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
        logger.info(f"TensorBoard logs: {tb_writer.log_dir}")

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
        tb_writer=tb_writer,
        log_every_n_steps=cfg.get("logging", {}).get("log_every_n_steps", 10),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---- Training loop ----
    max_epochs = train_cfg.get("max_epochs", 200)
    val_every = train_cfg.get("val_every_n_epochs", 5)
    save_every = train_cfg.get("save_every_n_epochs", 10)

    if is_main:
        logger.info(f"Starting training for {max_epochs} epochs")
        for handler in logging.getLogger().handlers:
            handler.flush()

    for epoch in range(trainer.epoch, max_epochs):
        # Update sampler epoch for deterministic DDP sharding
        if sampler is not None:
            sampler.set_epoch(epoch)

        train_metrics = trainer.train_epoch()

        if is_main:
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
    if is_main:
        trainer.save_checkpoint()
        logger.info("Training complete!")
        if tb_writer is not None:
            tb_writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training failed with exception")
        cleanup_ddp()
        raise
