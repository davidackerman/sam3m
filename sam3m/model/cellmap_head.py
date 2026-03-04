"""Dense multi-label segmentation head for CellMap's 48 organelle classes.

Taps SAM3's pixel decoder (FPN) output and produces per-pixel multi-label
predictions via sigmoid activation. This head is fully trained (not LoRA)
since it is entirely new — SAM3 has no pre-existing 48-class dense head.

Architecture:
    pixel_features [B, 256, H/4, W/4] (from SAM3's PixelDecoder)
    -> Conv3x3(256, 256) + GroupNorm + GELU   (local spatial context)
    -> Conv1x1(256, 256) + GroupNorm + GELU
    -> Conv1x1(256, n_fine) (48 classes)
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


class ScaleConditioner(nn.Module):
    """FiLM (Feature-wise Linear Modulation) for scale conditioning.

    Takes an explicit scale factor as input and produces per-channel
    gamma/beta to modulate pixel features. The model does not predict
    the scale — it receives it as a known input.
    """

    def __init__(self, n_channels: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * n_channels),
        )
        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].bias.data[:n_channels] = 1.0  # gamma = 1

    def forward(
        self, features: torch.Tensor, scale_factor: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            features: [B, C, H, W] pixel features.
            scale_factor: [B] or scalar — known scale factor.

        Returns:
            Modulated features [B, C, H, W].
        """
        # Ensure scale_factor is [B, 1]
        if scale_factor.dim() == 0:
            scale_factor = scale_factor.unsqueeze(0)
        scale_factor = scale_factor.view(-1, 1).to(features.device)

        params = self.mlp(scale_factor)  # [B, 2*C]
        gamma, beta = params.chunk(2, dim=-1)  # each [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * features + beta


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
        use_scale_conditioning: Whether to add FiLM scale conditioning.
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
        use_scale_conditioning: bool = False,
    ):
        super().__init__()
        self.n_fine = n_fine
        self.n_medium = n_medium
        self.n_coarse = n_coarse
        self.use_auxiliary = use_auxiliary
        self.upsample_factor = upsample_factor
        self.use_scale_conditioning = use_scale_conditioning

        if use_scale_conditioning:
            self.scale_conditioner = ScaleConditioner(in_channels)

        # Main fine-level head
        self.fine_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),  # 3x3 for spatial context
            nn.GroupNorm(32, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GroupNorm(32, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, n_fine, 1),
        )

        # Auxiliary heads for hierarchical loss (medium + coarse)
        if use_auxiliary:
            self.medium_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.GroupNorm(32, in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, n_medium, 1),
            )
            self.coarse_head = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1),
                nn.GroupNorm(16, hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, n_coarse, 1),
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
        scale_factor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass producing multi-label predictions.

        Args:
            pixel_features: [B, 256, H/4, W/4] from SAM3's pixel decoder.
            target_size: Optional (H, W) to upsample predictions to.
                If None, uses upsample_factor.
            scale_factor: Optional [B] tensor — known scale factor for
                FiLM conditioning. Only used if use_scale_conditioning=True.

        Returns:
            dict with:
                "fine": [B, 48, H, W] logits (pre-sigmoid)
                "medium": [B, 17, H, W] logits (if use_auxiliary)
                "coarse": [B, 7, H, W] logits (if use_auxiliary)
        """
        # Apply FiLM scale conditioning before heads
        if self.use_scale_conditioning and scale_factor is not None:
            pixel_features = self.scale_conditioner(pixel_features, scale_factor)

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
