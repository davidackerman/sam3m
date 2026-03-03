# Data Pipeline

## Source Data

- Location: `/nrs/cellmap/data/` (zarr format)
- ~289 annotated 3D FIB-SEM volumes across ~22 datasets
- Per-dataset normalization parameters in `configs/norms.csv`
- Anisotropic resolutions resampled to target_resolution=8nm isotropic

## Class Hierarchy

- **48 fine classes**: mito, mito_mem, mito_lum, er, er_mem, er_lum, nucpl, nuc, etc.
- **17 medium groups**: e.g., "mito" group = {mito, mito_mem, mito_lum, mito_ribo}
- **7 coarse super-classes**: e.g., "organelles" = {mito group, er group, ...}
- Mapping matrices: `fine_to_medium_matrix()`, `fine_to_coarse_matrix()` in `class_mapping.py`

## Instance-Evaluated Classes (10)

The CellMap challenge evaluates instance segmentation for:
mito, nuc, ves, endo, lyso, ld, perox, np, mt, cell

These require post-processing (watershed) after semantic prediction.

## CellMapDataset3D

Copied from OrganelleNet (read-only source at `/groups/cellmap/cellmap/zouinkhim/OrganelleNet/`).

Returns per sample:
- `raw`: [1, D, H, W] float32, normalized to [0,1]
- `labels`: [48, D, H, W] float32 binary masks
- `annotated_mask`: [48] bool — which classes are annotated in this crop
- `spatial_mask`: [1, D, H, W] float32 — which voxels have valid annotations
- `crop_name`: str — dataset/crop identifier

Key behaviors:
- Caches dataset discovery (zarr paths, resolutions, crop lists)
- Handles per-dataset normalization and inversion
- Resamples to target resolution using scipy zoom
- `skip_datasets` config to exclude broken zarrs

## CellMapVideoDataset

Wraps CellMapDataset3D for SAM3 video mode:
- Selects `num_frames=16` z-slices at `frame_stride=8` from 128-depth patches
- Resizes raw: 128×128 → 1008×1008 (bilinear) for SAM3's ViT
- Converts grayscale → RGB (repeat 3×)
- Resizes labels: 128×128 → 256×256 (nearest-neighbor) for mask output
- Centers frame selection in the volume; reduces stride/count for shallow volumes

## ClassBalancedSampler

From OrganelleNet: tracks how often each class has been seen, picks the least-seen class, then randomly samples a crop containing that class. Ensures rare classes (like nuclear pores) get sufficient training signal.

## Augmentations (EMSliceTransforms)

Applied consistently across all z-slices in a video:
- Spatial: horizontal flip, vertical flip, 90° rotation (same seed per sample)
- Intensity: brightness, contrast, Gaussian noise
- No elastic deformation (too expensive per-slice, breaks z-consistency)
- Re-binarize labels after potential interpolation

**Note**: Transforms are defined but not yet integrated into CellMapVideoDataset.__getitem__.
This should be added when training begins.
