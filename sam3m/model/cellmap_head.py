"""Dense multi-label segmentation head for CellMap's 48 organelle classes.

Taps SAM3's pixel decoder (FPN) output and produces per-pixel multi-label
predictions via sigmoid activation. This head is fully trained (not LoRA)
since it is entirely new — SAM3 has no pre-existing 48-class dense head.

Architecture:
    pixel_features [B, 256, H/4, W/4] (from SAM3's PixelDecoder)
    -> Conv1x1(256, 256) + GroupNorm + GELU
    -> Conv1x1(256, 128) + GroupNorm + GELU
    -> Conv1x1(128, n_fine) (48 classes)
    -> Upsample 4x to full resolution (bilinear)
    -> sigmoid (during inference)

Optional auxiliary heads for medium (17) and coarse (7) level predictions,
tapping features at different depths for hierarchical loss.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellMapSegmentationHead(nn.Module):
    """48-class dense segmentation head for CellMap challenge.

    Args:
        in_channels: Input channels from SAM3's pixel decoder (default 256).
        hidden_channels: Hidden layer width (default 128).
        n_fine: Number of fine-level classes (default 48).
        n_medium: Number of medium-level classes for auxiliary head (default 17).
        n_coarse: Number of coarse-level classes for auxiliary head (default 7).
        use_auxiliary: Whether to create medium/coarse auxiliary heads.
        upsample_factor: Factor to upsample output to full resolution.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        n_fine: int = 48,
        n_medium: int = 17,
        n_coarse: int = 7,
        use_auxiliary: bool = True,
        upsample_factor: int = 4,
    ):
        super().__init__()
        self.n_fine = n_fine
        self.n_medium = n_medium
        self.n_coarse = n_coarse
        self.use_auxiliary = use_auxiliary
        self.upsample_factor = upsample_factor

        # Main fine-level head
        self.fine_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GroupNorm(32, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, n_fine, 1),
        )

        # Auxiliary heads for hierarchical loss (medium + coarse)
        if use_auxiliary:
            self.medium_head = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1),
                nn.GroupNorm(16, hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, n_medium, 1),
            )
            self.coarse_head = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels // 2, 1),
                nn.GroupNorm(8, hidden_channels // 2),
                nn.GELU(),
                nn.Conv2d(hidden_channels // 2, n_coarse, 1),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        pixel_features: torch.Tensor,
        target_size: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass producing multi-label predictions.

        Args:
            pixel_features: [B, 256, H/4, W/4] from SAM3's pixel decoder.
            target_size: Optional (H, W) to upsample predictions to.
                If None, uses upsample_factor.

        Returns:
            dict with:
                "fine": [B, 48, H, W] logits (pre-sigmoid)
                "medium": [B, 17, H, W] logits (if use_auxiliary)
                "coarse": [B, 7, H, W] logits (if use_auxiliary)
        """
        fine_logits = self.fine_head(pixel_features)

        # Upsample to target resolution
        if target_size is not None:
            fine_logits = F.interpolate(
                fine_logits, size=target_size, mode="bilinear", align_corners=False,
            )
        elif self.upsample_factor > 1:
            fine_logits = F.interpolate(
                fine_logits, scale_factor=self.upsample_factor,
                mode="bilinear", align_corners=False,
            )

        outputs = {"fine": fine_logits}

        if self.use_auxiliary:
            medium_logits = self.medium_head(pixel_features)
            coarse_logits = self.coarse_head(pixel_features)

            if target_size is not None:
                medium_logits = F.interpolate(
                    medium_logits, size=target_size, mode="bilinear", align_corners=False,
                )
                coarse_logits = F.interpolate(
                    coarse_logits, size=target_size, mode="bilinear", align_corners=False,
                )
            elif self.upsample_factor > 1:
                medium_logits = F.interpolate(
                    medium_logits, scale_factor=self.upsample_factor,
                    mode="bilinear", align_corners=False,
                )
                coarse_logits = F.interpolate(
                    coarse_logits, scale_factor=self.upsample_factor,
                    mode="bilinear", align_corners=False,
                )

            outputs["medium"] = medium_logits
            outputs["coarse"] = coarse_logits

        return outputs
