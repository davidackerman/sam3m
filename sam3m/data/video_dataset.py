"""Video dataset wrapper for SAM3 training.

Wraps CellMapDataset3D to produce z-stack sequences as pseudo-video
for SAM3's video predictor. Each 3D patch becomes a sequence of 2D
z-slices treated as video frames.

Key design decisions:
- num_frames=16 with frame_stride=8 covers a full 128-voxel depth patch
- image_size=1008 matches SAM3's ViT input expectation (14px patches -> 72x72 grid)
- Grayscale EM images are repeated to 3 channels for RGB-pretrained ViT
- Labels are resized with nearest-neighbor to avoid interpolation artifacts
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dataset import CellMapDataset3D

logger = logging.getLogger(__name__)


class CellMapVideoDataset(Dataset):
    """Wraps CellMapDataset3D to produce z-slice sequences for SAM3.

    Each sample is a "video" of D z-slices from a 3D EM patch, formatted
    for SAM3's video predictor with memory propagation across frames.

    Args:
        base_dataset: CellMapDataset3D instance for 3D patch extraction.
        num_frames: Number of z-slices per video sequence.
        frame_stride: Stride between selected z-slices (1=consecutive).
        image_size: SAM3 input resolution (default 1008 for SAM3's ViT).
        mask_size: Output mask resolution for labels (default 256).
    """

    def __init__(
        self,
        base_dataset: CellMapDataset3D,
        num_frames: int = 16,
        frame_stride: int = 8,
        image_size: int = 1008,
        mask_size: int = 256,
    ):
        self.base_dataset = base_dataset
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.mask_size = mask_size

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Extract a 3D patch and convert to a video of z-slices.

        Returns:
            dict with:
                images: [T, 3, H, W] float32 — z-slices as RGB frames
                labels: [T, C, Hm, Wm] float32 — per-slice, per-class masks
                annotated_mask: [C] bool — which classes are annotated
                spatial_masks: [T, 1, Hm, Wm] float32 — per-slice spatial validity
                crop_name: str — dataset/crop identifier
        """
        # Get 3D patch from base dataset
        result = self.base_dataset[idx]
        raw = result[0]              # [1, D, H, W]
        labels = result[1]           # [C, D, H, W]
        annotated_mask = result[2]   # [C]
        spatial_mask = result[3]     # [1, D, H, W]
        crop_name = result[4]        # str

        D = raw.shape[1]

        # Select z-slice indices
        z_indices = self._select_z_indices(D)
        T = len(z_indices)

        # Extract and resize z-slices
        images = torch.zeros(T, 3, self.image_size, self.image_size)
        slice_labels = torch.zeros(T, labels.shape[0], self.mask_size, self.mask_size)
        slice_spatial = torch.zeros(T, 1, self.mask_size, self.mask_size)

        for t, z in enumerate(z_indices):
            # Raw slice [1, H, W] -> resize to [1, image_size, image_size]
            raw_slice = raw[:, z, :, :]  # [1, H, W]
            raw_resized = F.interpolate(
                raw_slice.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)  # [1, image_size, image_size]

            # Grayscale -> RGB: repeat across 3 channels
            images[t] = raw_resized.expand(3, -1, -1)

            # Labels [C, H, W] -> resize to [C, mask_size, mask_size]
            label_slice = labels[:, z, :, :]  # [C, H, W]
            slice_labels[t] = F.interpolate(
                label_slice.unsqueeze(0), size=(self.mask_size, self.mask_size),
                mode="nearest",
            ).squeeze(0)

            # Spatial mask [1, H, W] -> resize
            sp_slice = spatial_mask[:, z, :, :]  # [1, H, W]
            slice_spatial[t] = F.interpolate(
                sp_slice.unsqueeze(0), size=(self.mask_size, self.mask_size),
                mode="nearest",
            ).squeeze(0)

        return {
            "images": images,                 # [T, 3, H, W]
            "labels": slice_labels,           # [T, C, Hm, Wm]
            "annotated_mask": annotated_mask,  # [C]
            "spatial_masks": slice_spatial,    # [T, 1, Hm, Wm]
            "crop_name": crop_name,
        }

    def _select_z_indices(self, depth: int) -> List[int]:
        """Select z-slice indices from a volume of given depth.

        Uses frame_stride to space slices evenly. If the volume is
        shorter than num_frames * frame_stride, reduces stride or
        number of frames to fit.
        """
        max_frames = min(self.num_frames, depth)
        stride = self.frame_stride

        # Reduce stride if volume is too shallow
        while max_frames * stride > depth and stride > 1:
            stride -= 1

        # Reduce frame count if still too many
        actual_frames = min(max_frames, depth // max(stride, 1))
        actual_frames = max(actual_frames, 1)

        # Center the sequence in the volume
        total_span = (actual_frames - 1) * stride
        start = (depth - total_span) // 2

        return [start + i * stride for i in range(actual_frames)]
