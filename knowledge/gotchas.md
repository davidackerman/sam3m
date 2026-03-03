# Known Gotchas and TODOs

## SAM3 Internal API (CRITICAL)

`sam3_cellmap.py` has **placeholder code** for extracting pixel decoder features from SAM3:
```python
backbone_features = self.sam3.image_encoder(images)
if hasattr(self.sam3, 'pixel_decoder'):
    pixel_features = self.sam3.pixel_decoder(backbone_features)
```
This MUST be adapted once SAM3 is installed and its internal module structure can be inspected.
The actual API may differ — SAM3 may use different attribute names, may return features differently,
or may require specific preprocessing. Run:
```python
from sam3.model_builder import build_sam3_image_model
model = build_sam3_image_model()
print([n for n, _ in model.named_children()])  # top-level modules
```

## Transforms Not Integrated

`EMSliceTransforms` is implemented in `transforms.py` but NOT called in
`CellMapVideoDataset.__getitem__`. Need to instantiate and apply transforms
after extracting z-slices. Pass `transform=get_train_transforms(...)` to the dataset.

## Loss Dimension Bridging

OrganelleNet losses expect 5D tensors `[B, C, D, H, W]`. For per-slice processing:
- Unsqueeze depth: `slice_labels.unsqueeze(2)` → `[1, C, 1, Hm, Wm]`
- This is done in `trainer.py:_train_step()` lines 192-194
- If you change the loss interface, update the unsqueeze logic

## CellMapLoss Signature

The `CellMapLoss.forward()` method expects:
```python
loss_fn(outputs_5d, labels_5d, annotated_mask, progress=float, spatial_mask=spatial_5d)
```
Where `outputs_5d` is a dict with keys "fine" (and optionally "medium", "coarse").
The `progress` parameter (0→1) controls dynamic loss weighting (hierarchical loss
shifts from coarse→fine as training progresses).

## Deprecated torch.cuda.amp Import

`trainer.py` uses `from torch.cuda.amp import GradScaler, autocast` which is deprecated
in PyTorch 2.7+. Should migrate to:
```python
from torch.amp import GradScaler, autocast
```
And use `autocast(device_type="cuda")` instead of `autocast()`.

## 128→1008 Upsampling

8× upsampling from 128×128 EM patches to 1008×1008 may create artifacts.
If quality is poor, consider:
1. Extracting larger patches at higher resolution (e.g., 504×504 → 1008×1008 = 2× upsample)
2. Using adaptive padding instead of stretching

## Memory Concerns

16 frames × 1008×1008 × SAM3 ViT = ~30GB activations per GPU.
If OOM on A100 80GB:
1. Enable activation checkpointing (`torch.utils.checkpoint`)
2. Reduce to 8 frames
3. Use gradient accumulation with smaller effective batch

## Missing Files

These are referenced in pixi.toml tasks but don't exist yet:
- `scripts/evaluate.py` — evaluation script
- `scripts/prepare_data.py` — data cache preparation
- `tests/` — test suite
- `sam3m/data/prompts.py` — organelle→text prompt mapping (Mode B)

## annotated_mask Must Be Bool

The `CellMapLoss` uses bitwise `&` on the annotated_mask (line 134 of losses.py: `valid = mask & has_positive`).
The mask MUST be a `torch.bool` tensor, not float. The CellMapDataset3D returns it as numpy bool,
which PyTorch's DataLoader correctly converts. If constructing test data manually, use `torch.ones(...).bool()`.

## Sparse Annotation Edge Case

Some datasets have NO annotations for certain classes. The `annotated_mask` handles this,
but verify that `CellMapLoss` correctly zeros out loss when `annotated_mask[c] = False`.
The boundary loss also needs the mask — check that boundary targets aren't computed for
unannotated classes.
