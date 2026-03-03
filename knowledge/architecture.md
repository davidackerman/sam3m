# SAM3M Architecture

## Overview

SAM3M fine-tunes Meta's SAM3 (Segment Anything Model 3, ~848M params) for the CellMap Segmentation Challenge — 3D FIB-SEM organelle segmentation with 48 classes across ~289 annotated volumes.

## Two Output Modes

### Mode A (Primary) — Dense 48-Class Head
- `CellMapSegmentationHead` attached to SAM3's pixel decoder output
- Architecture: Conv1x1(256→256)+GN+GELU → Conv1x1(256→128)+GN+GELU → Conv1x1(128→48) → 4x bilinear upsample
- Produces multi-label (sigmoid) predictions for all 48 classes in one pass
- Optional auxiliary heads for medium (17) and coarse (7) levels for hierarchical loss
- This is what drives CellMap challenge evaluation

### Mode B (Secondary, Future) — SAM3 Native Prompting
- Text prompt: "mito" → segments all mitochondria
- Point prompt: click → segments instance at location
- Requires LoRA on text encoder / DETR decoder (not yet implemented)
- Preserves SAM3's interactive segmentation capability

## LoRA Strategy

Applied to vision encoder attention layers only (Mode A):
- Targets: `attn.qkv` (1024→3072) and `attn.proj` (1024→1024), 32 transformer blocks
- Rank=8, alpha=8.0, dropout=0.05
- ~12.6M trainable LoRA params + ~0.5M head params = ~13.1M total (1.5% of model)
- B initialized to zeros so LoRA starts as identity (no initial perturbation)

## Video Mode (3D Consistency)

Z-stacks treated as pseudo-video for SAM3's memory-based predictor:
- 128-depth patches → 16 z-slices at stride 8
- Each slice: 128×128 → 1008×1008 (SAM3 ViT input size)
- Grayscale → RGB by repeating 3 channels
- SAM3's memory mechanism propagates features across z-slices
- Bidirectional inference (forward + backward) for better z-consistency

## Data Flow

```
CellMapDataset3D (zarr 3D patches)
    ↓ [1,128,128,128] raw + [48,128,128,128] labels
CellMapVideoDataset (z-slice extraction)
    ↓ [T=16, 3, 1008, 1008] images + [T, 48, 256, 256] labels
EMSliceTransforms (augmentation)
    ↓ spatially-consistent across all T frames
SAM3 ViT encoder + LoRA
    ↓ multi-scale features
FPN pixel decoder (frozen)
    ↓ [B, 256, H/4, W/4]
CellMapSegmentationHead
    ↓ [B, 48, H, W] logits
Sigmoid → probabilities
```

## Loss Stack (from OrganelleNet)

- `MaskedBCELoss` + `MaskedDiceLoss` → combined as 0.5/0.5
- `HierarchicalLoss`: fine(48) + medium(17) + coarse(7) with dynamic weighting (progress 0→1)
- `BoundaryAwareLoss`: 5× weight on instance-class boundaries
- `ZConsistencyLoss`: penalizes prediction discontinuities across adjacent z-slices where labels are consistent
- All losses masked by `annotated_mask [B, 48]` and `spatial_mask [B, 1, D, H, W]` for sparse annotations

## Key Design Decisions

1. **Multi-label sigmoid, NOT softmax** — organelles overlap (e.g., mito_mem contains mito_lum)
2. **Sparse annotation masking** — not all crops annotate all 48 classes; loss zeroed for unannotated
3. **5D→2D bridging** — OrganelleNet losses expect [B,C,D,H,W]; we unsqueeze depth dim for per-slice
4. **Labels resized with nearest-neighbor** — prevents interpolation artifacts in binary masks
5. **Gaussian-weighted blending** for overlapping patches during inference
6. **Lightweight checkpoints** — save only LoRA params + head weights (not full 848M model)
