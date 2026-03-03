#!/usr/bin/env python3
"""End-to-end smoke test: SAM3 + LoRA + CellMap head."""

import torch
from sam3.model_builder import build_sam3_image_model
from sam3m.model.cellmap_head import CellMapSegmentationHead
from sam3m.model.lora import apply_lora, count_parameters
from sam3m.model.sam3_cellmap import SAM3CellMapModel

# Build SAM3 on CPU without checkpoint
print("Building SAM3 (no checkpoint)...")
sam3 = build_sam3_image_model(
    device="cpu", load_from_HF=False, eval_mode=False, enable_segmentation=True
)

# Freeze
for p in sam3.parameters():
    p.requires_grad_(False)

# Apply LoRA
replaced = apply_lora(sam3, target_modules=["attn.qkv", "attn.proj"], rank=8, alpha=8.0)
print(f"LoRA applied to {len(replaced)} modules, {sum(replaced.values()):,} params added")
counts = count_parameters(sam3)
total = counts["total"]
trainable = counts["trainable"]
print(f"SAM3: {total:,} total, {trainable:,} trainable")

# Forward through backbone + pixel decoder
print("Running forward pass...")
x = torch.randn(1, 3, 1008, 1008)
with torch.no_grad():
    backbone_out = sam3.backbone.forward_image(x)
    fpn = backbone_out["backbone_fpn"]
    pixel_features = sam3.segmentation_head.pixel_decoder(fpn)
print(f"pixel_features: {pixel_features.shape}")

# CellMap head
head = CellMapSegmentationHead(
    in_channels=256, hidden_channels=128, n_fine=48, upsample_factor=1
)
out = head(pixel_features, target_size=(128, 128))
print(f"fine: {out['fine'].shape}, medium: {out['medium'].shape}, coarse: {out['coarse'].shape}")

# End-to-end via SAM3CellMapModel
print("Testing SAM3CellMapModel end-to-end...")
model = SAM3CellMapModel(sam3, head)
out2 = model(x, target_size=(128, 128))
print(f"E2E fine: {out2['fine'].shape}")

# Verify gradients flow to LoRA and head only
print("Checking gradient flow...")
loss = out2["fine"].sum()
loss.backward()
lora_grads = sum(1 for n, p in model.named_parameters() if p.grad is not None and "lora" in n)
head_grads = sum(1 for n, p in model.named_parameters() if p.grad is not None and "cellmap_head" in n)
frozen_grads = sum(
    1 for n, p in model.named_parameters()
    if p.grad is not None and "lora" not in n and "cellmap_head" not in n
)
print(f"Gradients: LoRA={lora_grads}, head={head_grads}, frozen={frozen_grads}")

assert frozen_grads == 0, f"Frozen params should have no gradients, got {frozen_grads}"
assert lora_grads > 0, "LoRA params should have gradients"
assert head_grads > 0, "Head params should have gradients"

print()
print("All checks passed!")
