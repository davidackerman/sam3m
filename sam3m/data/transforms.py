"""2D augmentation transforms for EM z-slices.

Adapted from OrganelleNet's 3D MONAI transforms for per-slice application.
All slices in a z-stack must share the same spatial transform to maintain
z-consistency, so spatial transforms are seeded per-sample.

Key differences from 3D transforms:
- No elastic deformation (too expensive per-slice, breaks z-consistency)
- 90-degree rotations in XY plane only (no XZ/YZ since we're 2D)
- Intensity transforms applied identically across all slices in a sequence
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torchvision.transforms.functional as TF


class EMSliceTransforms:
    """2D augmentations for EM z-slices in a video sequence.

    Designed to apply consistent spatial transforms across all z-slices
    and consistent intensity transforms to maintain temporal coherence.

    Args:
        spatial_prob: Probability for each spatial transform.
        intensity_prob: Probability for each intensity transform.
        noise_std: Std dev for Gaussian noise.
        brightness_range: (min, max) for brightness adjustment factor.
        contrast_range: (min, max) for contrast adjustment factor.
    """

    def __init__(
        self,
        spatial_prob: float = 0.5,
        intensity_prob: float = 0.3,
        noise_std: float = 0.03,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
    ):
        self.spatial_prob = spatial_prob
        self.intensity_prob = intensity_prob
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        spatial_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentations to a video sequence.

        All spatial transforms use the same random seed across frames
        for z-consistency.

        Args:
            images: [T, 3, H, W] float32 — z-slice frames
            labels: [T, C, Hm, Wm] float32 — per-slice masks
            spatial_masks: [T, 1, Hm, Wm] float32 — spatial validity masks

        Returns:
            Augmented (images, labels, spatial_masks).
        """
        # Sample random state for this video (consistent across slices)
        rng = torch.Generator()
        rng.manual_seed(torch.randint(0, 2**31, (1,)).item())

        # --- Spatial transforms (applied to all frames consistently) ---

        # Random horizontal flip
        if torch.rand(1, generator=rng).item() < self.spatial_prob:
            images = images.flip(-1)
            labels = labels.flip(-1)
            if spatial_masks is not None:
                spatial_masks = spatial_masks.flip(-1)

        # Random vertical flip
        if torch.rand(1, generator=rng).item() < self.spatial_prob:
            images = images.flip(-2)
            labels = labels.flip(-2)
            if spatial_masks is not None:
                spatial_masks = spatial_masks.flip(-2)

        # Random 90-degree rotation (0, 90, 180, 270)
        if torch.rand(1, generator=rng).item() < self.spatial_prob:
            k = torch.randint(1, 4, (1,), generator=rng).item()
            images = torch.rot90(images, k, dims=(-2, -1))
            labels = torch.rot90(labels, k, dims=(-2, -1))
            if spatial_masks is not None:
                spatial_masks = torch.rot90(spatial_masks, k, dims=(-2, -1))

        # --- Intensity transforms (applied identically to all frames) ---

        # Random brightness
        if torch.rand(1, generator=rng).item() < self.intensity_prob:
            lo, hi = self.brightness_range
            factor = lo + (hi - lo) * torch.rand(1, generator=rng).item()
            images = images * factor

        # Random contrast
        if torch.rand(1, generator=rng).item() < self.intensity_prob:
            lo, hi = self.contrast_range
            factor = lo + (hi - lo) * torch.rand(1, generator=rng).item()
            mean = images.mean(dim=(-2, -1), keepdim=True)
            images = (images - mean) * factor + mean

        # Random Gaussian noise
        if torch.rand(1, generator=rng).item() < self.intensity_prob:
            noise = torch.randn_like(images) * self.noise_std
            images = images + noise

        # Clamp to [0, 1]
        images = images.clamp(0.0, 1.0)

        # Re-binarize labels after any potential interpolation
        labels = (labels > 0.5).float()

        return images, labels, spatial_masks


def get_train_transforms(**kwargs) -> EMSliceTransforms:
    """Get training augmentation pipeline."""
    return EMSliceTransforms(**kwargs)


def get_val_transforms():
    """Get validation transforms (no augmentation)."""
    return None
