# SAM3M

Fine-tuning [SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) for 3D EM organelle segmentation on the [CellMap Segmentation Challenge](https://cellmapchallenge.janelia.org/).

## Overview

SAM3M adapts SAM3's 848M-parameter vision-language model to segment 48 organelle classes in FIB-SEM electron microscopy volumes. Z-stacks are processed as sequences of 2D slices, with adjacent z-slices fed as RGB channels to give the ViT 3D context.

Only ~0.25% of parameters are trained: LoRA adapters on the vision encoder plus a fully-trained segmentation head.

## Setup

```bash
pixi install
```

This installs Python 3.12, SAM3, PyTorch, and all dependencies.

## Project Structure

```
sam3m/
├── sam3m/
│   ├── data/           # CellMapDataset3D, z-stack wrapper, class hierarchy, augmentations
│   ├── model/          # LoRA adapters, CellMap segmentation head, SAM3 assembly
│   ├── losses/         # Masked BCE+Dice, hierarchical, boundary-aware, z-consistency
│   ├── training/       # Trainer loop, train/val split
│   └── inference/      # Sliding window prediction, watershed postprocessing
├── configs/
│   ├── train.yaml                 # Training configuration
│   ├── norms.csv                  # Per-dataset normalization parameters
│   ├── challenge_train_crops.csv  # CellMap challenge training manifest (22 datasets, ~250 crops)
│   └── challenge_test_crops.csv   # CellMap challenge test manifest (6 datasets, 16 crops)
├── scripts/
│   ├── train.py          # Training entry point
│   └── smoke_test.py     # End-to-end pipeline verification
├── runs/                 # Auto-created: timestamped run dirs (config, logs, checkpoints, TensorBoard)
└── knowledge/            # Architecture docs and known gotchas
```

## Architecture

```
Input: [B, 3, 1008, 1008] (adjacent z-slices [z-1, z, z+1] as RGB)
  → SAM3 ViT backbone + LoRA (frozen weights + rank-16 adapters)
  → FPN neck → multi-scale features (288², 144², 72²)
  → PixelDecoder → [B, 256, 288, 288]
  → CellMapSegmentationHead (Conv3x3 + Conv1x1 → 48 classes)
  → [B, 48, H, W] per-class logits (sigmoid, multi-label)
```

| Component | Params | Training |
|-----------|--------|----------|
| ViT vision encoder | ~630M | LoRA only (~2M) |
| Text encoder | ~150M | Frozen |
| Pixel decoder (FPN) | ~15M | Frozen |
| CellMapSegmentationHead | ~1.5M | Fully trained |
| **Total trainable** | | **~3.5M (0.4%)** |

## Data

Uses the CellMap challenge data at `/nrs/cellmap/data/` (48 organelle classes across 22 datasets). The data pipeline:

1. **CellMapDataset3D** extracts 128³ patches from zarr volumes, resampled to 8nm isotropic
2. **CellMapZStackDataset** converts patches to 48-frame z-stacks; each frame uses adjacent z-slices [z-1, z, z+1] as RGB channels for 3D context
3. **ClassBalancedSampler** ensures rare classes get sufficient training signal (DDP-aware)
4. **Multi-scale training** (optional) — randomly varies the effective resolution per sample, with FiLM conditioning on the segmentation head

### Dataset filtering

By default, training uses only the official CellMap challenge training crops (`challenge_split: train` in config). This is controlled in `configs/train.yaml`:

```yaml
# Only challenge training crops (default)
challenge_split: train

# All available data (includes internal non-challenge datasets)
challenge_split: null

# Manual allowlist (independent of challenge manifests)
include_datasets:
  - jrc_hela-2
  - jrc_hela-3
```

Crop manifests are sourced from [janelia-cellmap/cellmap-segmentation-challenge](https://github.com/janelia-cellmap/cellmap-segmentation-challenge). Discovery results are cached per config, so the first run walks the filesystem but subsequent starts are instant.

### Multi-scale training

By default, patches are extracted at 8nm resolution (`scale_factors: [1]`). Enabling multi-scale training reads proportionally larger world volumes at the same 128³ voxel output, giving the model different effective resolutions:

```yaml
# configs/train.yaml
scale_factors: [1, 2, 4]  # 8nm, 16nm, 32nm effective resolution
```

The scale factor is **explicitly passed to the model** (not predicted) via FiLM conditioning on the segmentation head. This lets the model learn scale-aware features — e.g., a large blob at 32nm is more likely a nucleus than a mitochondrion. At inference time, the scale factor is also passed explicitly based on the resolution being predicted at.

When `scale_factors: [1]` (default), the FiLM layer is not created and there is zero overhead.

## Training

```bash
# Single GPU
pixi run train
pixi run train --config configs/train.yaml --run-name my-experiment

# Multi-GPU (DDP)
pixi run train-ddp
torchrun --nproc_per_node=8 scripts/train.py

# Resume from checkpoint
pixi run train --resume runs/2026-03-03_14-30-00/checkpoints/checkpoint_epoch50.pt
```

Each run creates a timestamped directory under `runs/`:

```
runs/2026-03-03_14-30-00/
├── config.yaml           # frozen config snapshot (use to reproduce)
├── original_config.yaml  # copy of the original YAML
├── train.log             # full training log
├── tensorboard/          # loss curves, LR, per-class Dice
└── checkpoints/
    ├── checkpoint_epoch10.pt
    └── best.pt
```

View training curves: `tensorboard --logdir runs/`

Key settings (see `configs/train.yaml`):
- Batch size 1 per GPU, gradient accumulation 4
- AdamW with per-group LRs: LoRA lr=2e-4, head lr=1e-3
- Mixed precision (AMP), cosine annealing with warm restarts
- Multi-GPU via DDP (torchrun), up to 8 GPUs

## Verification

```bash
pixi run python scripts/smoke_test.py
```

Verifies the full pipeline: SAM3 backbone → pixel decoder → CellMap head, with gradient flow check (only LoRA + head get gradients).

## Class Hierarchy

48 fine classes → 17 medium groups → 7 coarse super-classes. The hierarchical loss trains all three levels with dynamic weighting that shifts from coarse to fine as training progresses. 10 classes are evaluated with instance segmentation (watershed postprocessing).

## Key Design Decisions

- **Multi-label sigmoid** (not softmax) — organelles overlap spatially
- **Sparse annotation masking** — loss zeroed for unannotated classes per crop
- **No 4x upsample in head** — SAM3's pixel decoder already outputs at 288×288 (high enough resolution)
- **Lightweight checkpoints** — only LoRA + head weights saved (not the full 842M model)

See `knowledge/` for detailed architecture docs and known gotchas.
