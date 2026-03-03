# SAM3M

Fine-tuning [SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) for 3D EM organelle segmentation on the [CellMap Segmentation Challenge](https://cellmapchallenge.janelia.org/).

## Overview

SAM3M adapts SAM3's 848M-parameter vision-language model to segment 48 organelle classes in FIB-SEM electron microscopy volumes. Z-stacks are treated as pseudo-video, leveraging SAM3's memory-based video predictor for 3D consistency.

**Two output modes:**
- **Mode A** (primary): Dense 48-class segmentation head on SAM3's pixel decoder — one forward pass produces all class predictions
- **Mode B** (future): SAM3's native text/point prompting fine-tuned for organelles — say "mito" to segment mitochondria, click to select instances

Only ~1.5% of parameters are trained via LoRA adapters on the vision encoder plus a new segmentation head.

## Setup

```bash
pixi install
```

This installs Python 3.12, SAM3, PyTorch, and all dependencies.

## Project Structure

```
sam3m/
├── sam3m/
│   ├── data/           # CellMapDataset3D, video wrapper, class hierarchy, augmentations
│   ├── model/          # LoRA adapters, CellMap segmentation head, SAM3 assembly
│   ├── losses/         # Masked BCE+Dice, hierarchical, boundary-aware, z-consistency
│   ├── training/       # Trainer loop, train/val split
│   └── inference/      # Sliding window prediction, watershed postprocessing
├── configs/
│   ├── train_video.yaml  # Training configuration
│   └── norms.csv         # Per-dataset normalization parameters
├── scripts/
│   ├── train.py          # Training entry point
│   └── smoke_test.py     # End-to-end pipeline verification
└── knowledge/            # Architecture docs and known gotchas
```

## Architecture

```
Input: [B, 3, 1008, 1008] (grayscale EM repeated to RGB)
  → SAM3 ViT backbone + LoRA (frozen weights + rank-8 adapters)
  → FPN neck → multi-scale features (288², 144², 72²)
  → PixelDecoder → [B, 256, 288, 288]
  → CellMapSegmentationHead → [B, 48, H, W] per-class logits
```

| Component | Params | Training |
|-----------|--------|----------|
| ViT vision encoder | ~630M | LoRA only (1.57M) |
| Text encoder | ~150M | Frozen |
| Pixel decoder (FPN) | ~15M | Frozen |
| CellMapSegmentationHead | ~0.5M | Fully trained |
| **Total trainable** | | **~2.1M (0.2%)** |

## Data

Uses the CellMap challenge data at `/nrs/cellmap/data/` (~289 annotated 3D FIB-SEM volumes, 48 organelle classes across ~22 datasets). The data pipeline:

1. **CellMapDataset3D** extracts 128³ patches from zarr volumes, resampled to 8nm isotropic
2. **CellMapVideoDataset** converts patches to 16-frame pseudo-videos (z-slices at stride 8, resized to 1008×1008)
3. **ClassBalancedSampler** ensures rare classes get sufficient training signal

## Training

```bash
pixi run train
# or with custom config:
pixi run python scripts/train.py --config configs/train_video.yaml
```

Key settings (see `configs/train_video.yaml`):
- Batch size 1 per GPU, gradient accumulation 8
- AdamW, lr=2e-4, cosine annealing with warm restarts
- Mixed precision (AMP)
- Multi-GPU via DDP (up to 8 GPUs)

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
