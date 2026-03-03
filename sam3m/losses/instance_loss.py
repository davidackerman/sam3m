"""Loss functions for Phase 2 instance segmentation fine-tuning.

Centroid-offset architecture with morphology-grouped post-processing.
Combines six components:
1. Semantic loss (BCE+Dice on all 48 classes) — prevents decoder drift
2. Center focal loss (focal loss on center heatmap) — sparse center detection
3. Offset loss (smooth L1 on offset vectors, foreground only) — centroid offsets
4. Boundary loss (BCE+Dice on boundary map) — inter-instance boundaries
5. Deep supervision loss (BCE+Dice on intermediate decoder outputs)
6. Coarse/medium auxiliary losses (BCE+Dice on aggregated labels)

Usage:
    loss_fn = InstanceSegmentationLoss(instance_indices=inst_idx,
                                       fine_to_medium=f2m, fine_to_coarse=f2c)
    total, log = loss_fn(outputs, labels, mask, center_targets, offset_targets,
                         boundary_targets, instance_foreground,
                         spatial_mask=spatial_mask)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import MaskedMultiLabelLoss, _combine_masks


class CenterFocalLoss(nn.Module):
    """CenterNet-style focal loss for sparse center heatmap.

    Centers are extremely sparse (99%+ of voxels are NOT centers).
    Standard BCE would learn to predict 0 everywhere. Focal loss
    down-weights easy negatives and focuses on hard positives.

    Operates on raw logits for AMP safety — uses logsigmoid instead of
    log(sigmoid(x)) to avoid numerical issues in float16.

    Args:
        alpha: Focusing parameter for hard examples (default 2.0).
        beta: Penalty reduction near centers (default 4.0).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, 1, D, H, W] center logits (raw, before sigmoid).
            target: [B, 1, D, H, W] GT center heatmap (Gaussian peaks).
            spatial_mask: [B, 1, D, H, W] float — 1 inside GT crop.
        """
        pred = torch.sigmoid(logits)

        # Determine valid computation region
        if spatial_mask is not None:
            valid = spatial_mask > 0.5
        else:
            valid = torch.ones_like(pred, dtype=torch.bool)

        pos_mask = (target >= 0.99) & valid
        neg_mask = (target < 0.99) & valid

        # log(sigmoid(x)) = logsigmoid(x) — numerically stable under AMP
        log_p = F.logsigmoid(logits)
        # log(1 - sigmoid(x)) = logsigmoid(-x)
        log_1mp = F.logsigmoid(-logits)

        pos_loss = -((1 - pred) ** self.alpha) * log_p
        pos_total = pos_loss[pos_mask].sum()

        neg_loss = (
            -((1 - target) ** self.beta)
            * (pred ** self.alpha)
            * log_1mp
        )
        neg_total = neg_loss[neg_mask].sum()

        # Normalize by total valid voxels (not just n_pos).
        # The focal weighting (pred^alpha on negatives) already handles
        # class imbalance. Dividing by n_pos would amplify the sum of
        # millions of negative voxels by 1/few_centers ≈ 100K×.
        n_valid = valid.sum().clamp(min=1)
        return (pos_total + neg_total) / n_valid


class OffsetLoss(nn.Module):
    """Smooth L1 on offset vectors, foreground voxels only.

    Background offsets are meaningless — only compute loss where
    instances actually exist. Operates on raw logits (no tanh needed
    for smooth L1 — the loss naturally pushes values toward targets).
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        foreground: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, D, H, W] offset logits (raw, before tanh).
            target: [B, 3, D, H, W] GT offsets normalized to [-1, 1].
            foreground: [B, 1, D, H, W] union of all instance foreground.
            spatial_mask: [B, 1, D, H, W] float — 1 inside GT crop.
        """
        mask = foreground.expand_as(pred)
        if spatial_mask is not None:
            mask = mask * spatial_mask.expand_as(pred)

        diff = F.smooth_l1_loss(pred * mask, target * mask, reduction="sum")
        return diff / mask.sum().clamp(min=1)


class BoundaryLoss(nn.Module):
    """BCE + Dice for thin boundary prediction.

    Dice handles extreme class imbalance (boundaries are 2-4 voxels thin
    in a volume of millions). BCE provides gradient stability.

    Operates on raw logits for AMP safety — uses binary_cross_entropy_with_logits.
    """

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        foreground: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, 1, D, H, W] boundary logits (raw, before sigmoid).
            target: [B, 1, D, H, W] GT boundary map.
            foreground: [B, 1, D, H, W] union of all instance foreground.
            spatial_mask: [B, 1, D, H, W] float — 1 inside GT crop.
        """
        mask = foreground
        if spatial_mask is not None:
            mask = mask * spatial_mask

        # BCE on logits — AMP-safe, no need for manual clamping
        t = target * mask
        # Mask logits so out-of-region voxels contribute zero loss
        masked_logits = logits * mask
        n_valid = mask.sum().clamp(min=1)
        bce = F.binary_cross_entropy_with_logits(
            masked_logits, t, reduction="sum"
        ) / n_valid

        # Dice needs probabilities
        p = torch.sigmoid(logits) * mask
        inter = (p * t).sum()
        dice = 1.0 - (2 * inter + 1) / (p.sum() + t.sum() + 1)

        return 0.5 * bce + 0.5 * dice


class InstanceSegmentationLoss(nn.Module):
    """Combined loss for Phase 2 instance segmentation fine-tuning.

    total = w_semantic * semantic + w_center * center_focal
          + w_offset * offset_smooth_l1 + w_boundary * boundary_bce_dice
          + w_deep * deep_supervision + w_coarse * coarse + w_medium * medium

    Args:
        instance_indices: Fine-class indices (into the 48-class output) for
            the 10 instance classes. Used to build foreground masks.
        fine_to_medium: [n_medium, n_fine] binary mapping matrix.
        fine_to_coarse: [n_coarse, n_fine] binary mapping matrix.
        w_semantic: Weight for semantic loss (default 1.0).
        w_center: Weight for center focal loss (default 0.2).
        w_offset: Weight for offset loss (default 0.2).
        w_boundary: Weight for boundary loss (default 0.1).
        w_deep: Weight for deep supervision loss (default 0.2).
        w_coarse: Weight for coarse auxiliary loss (default 0.1).
        w_medium: Weight for medium auxiliary loss (default 0.15).
    """

    def __init__(
        self,
        instance_indices: List[int],
        fine_to_medium: torch.Tensor,
        fine_to_coarse: torch.Tensor,
        w_semantic: float = 1.0,
        w_center: float = 0.2,
        w_offset: float = 0.2,
        w_boundary: float = 0.1,
        w_deep: float = 0.2,
        w_coarse: float = 0.1,
        w_medium: float = 0.15,
    ):
        super().__init__()
        self.instance_indices = instance_indices
        self.register_buffer("f2m", fine_to_medium)   # [n_medium, n_fine]
        self.register_buffer("f2c", fine_to_coarse)   # [n_coarse, n_fine]
        self.w_semantic = w_semantic
        self.w_center = w_center
        self.w_offset = w_offset
        self.w_boundary = w_boundary
        self.w_deep = w_deep
        self.w_coarse = w_coarse
        self.w_medium = w_medium

        # Semantic: BCE+Dice on all 48 classes
        self.semantic_loss = MaskedMultiLabelLoss(bce_weight=0.5, dice_weight=0.5)
        # Instance heads
        self.center_loss = CenterFocalLoss(alpha=2.0, beta=4.0)
        self.offset_loss = OffsetLoss()
        self.boundary_loss = BoundaryLoss()
        # Auxiliary: BCE+Dice for coarse/medium/deep
        self.aux_loss = MaskedMultiLabelLoss(bce_weight=0.5, dice_weight=0.5)

    def _aggregate_labels(
        self,
        fine_labels: torch.Tensor,
        fine_mask: torch.Tensor,
        mapping: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate fine labels/mask to coarser level using mapping matrix."""
        B = fine_labels.shape[0]
        spatial = fine_labels.shape[2:]
        fine_flat = fine_labels.view(B, fine_labels.shape[1], -1)
        group_flat = torch.einsum("gf,bfn->bgn", mapping, fine_flat)
        group_labels = group_flat.clamp(0, 1).view(B, mapping.shape[0], *spatial)
        group_mask = torch.einsum("gf,bf->bg", mapping, fine_mask.float()) > 0
        return group_labels, group_mask

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
        center_targets: torch.Tensor,
        offset_targets: torch.Tensor,
        boundary_targets: torch.Tensor,
        instance_foreground: torch.Tensor,
        progress: float = 0.0,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: dict with "fine" [B,48,D,H,W], "instance" [B,4,D,H,W],
                     "boundary" [B,1,D,H,W], and optionally "deep_fine",
                     "coarse" [B,7,D,H,W], "medium" [B,17,D,H,W].
            labels: [B, 48, D, H, W] binary semantic labels.
            mask: [B, 48] annotation mask for semantic classes.
            center_targets: [B, 1, D, H, W] GT center heatmap.
            offset_targets: [B, 3, D, H, W] GT offset vectors.
            boundary_targets: [B, 1, D, H, W] GT boundary map.
            instance_foreground: [B, 1, D, H, W] union of instance foreground.
            progress: training progress 0→1 (reserved).
            spatial_mask: [B, 1, D, H, W] float — 1 inside GT crop.

        Returns:
            total_loss: scalar
            log_dict: per-component losses for logging
        """
        log = {}

        # 1. Semantic loss (all 48 classes)
        sem_loss, sem_comp = self.semantic_loss(
            outputs["fine"], labels, mask, spatial_mask
        )
        log["semantic_loss"] = sem_loss.detach()
        log["semantic_bce"] = sem_comp["bce"]
        log["semantic_dice"] = sem_comp["dice"]

        # 2. Center focal loss (class-agnostic heatmap) — operates on logits
        inst_logits = outputs["instance"]     # [B, 4, D, H, W] raw logits
        center_logits = inst_logits[:, :1]    # [B, 1, D, H, W]
        offset_logits = inst_logits[:, 1:4]   # [B, 3, D, H, W]

        ctr_loss = self.center_loss(center_logits, center_targets, spatial_mask)
        log["center_loss"] = ctr_loss.detach()

        # 3. Offset loss (smooth L1 on raw logits, foreground only)
        off_loss = self.offset_loss(
            offset_logits, offset_targets, instance_foreground, spatial_mask
        )
        log["offset_loss"] = off_loss.detach()

        # 4. Boundary loss (BCE + Dice)
        bnd_loss = self.boundary_loss(
            outputs["boundary"], boundary_targets, instance_foreground, spatial_mask
        )
        log["boundary_loss"] = bnd_loss.detach()

        total = (
            self.w_semantic * sem_loss
            + self.w_center * ctr_loss
            + self.w_offset * off_loss
            + self.w_boundary * bnd_loss
        )

        # 5. Deep supervision (intermediate decoder outputs → fine labels)
        if "deep_fine" in outputs:
            deep_loss = torch.tensor(0.0, device=labels.device)
            for i, deep_logits in enumerate(outputs["deep_fine"]):
                dl, _ = self.aux_loss(deep_logits, labels, mask, spatial_mask)
                deep_loss = deep_loss + dl
            deep_loss = deep_loss / max(len(outputs["deep_fine"]), 1)
            total = total + self.w_deep * deep_loss
            log["deep_loss"] = deep_loss.detach()

        # 6. Coarse auxiliary loss (7-class aggregated)
        if "coarse" in outputs:
            coarse_labels, coarse_mask = self._aggregate_labels(
                labels, mask, self.f2c
            )
            coarse_loss, _ = self.aux_loss(
                outputs["coarse"], coarse_labels, coarse_mask, spatial_mask
            )
            total = total + self.w_coarse * coarse_loss
            log["coarse_loss"] = coarse_loss.detach()

        # 7. Medium auxiliary loss (17-class aggregated)
        if "medium" in outputs:
            medium_labels, medium_mask = self._aggregate_labels(
                labels, mask, self.f2m
            )
            medium_loss, _ = self.aux_loss(
                outputs["medium"], medium_labels, medium_mask, spatial_mask
            )
            total = total + self.w_medium * medium_loss
            log["medium_loss"] = medium_loss.detach()

        log["total_loss"] = total.detach()

        return total, log
