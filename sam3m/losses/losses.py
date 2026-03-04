"""Loss functions for multi-label organelle segmentation.

Three loss components:
1. MaskedMultiLabelLoss: BCE + Dice with class masking for unannotated classes
2. HierarchicalLoss: fine + medium + coarse with dynamic weighting
3. BoundaryAwareLoss: boundary-weighted loss for instance segmentation classes

Combined via CellMapLoss which handles all outputs from HierarchicalResUNet.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. Masked multi-label loss (BCE + Dice)
# ---------------------------------------------------------------------------

def _combine_masks(
    class_mask: torch.Tensor,
    spatial_mask: Optional[torch.Tensor],
    target_shape: torch.Size,
) -> torch.Tensor:
    """Combine per-class mask [B,C] with spatial mask [B,1,D,H,W] into [B,C,D,H,W].

    Args:
        class_mask: [B, C] bool — which classes are annotated
        spatial_mask: [B, 1, D, H, W] float or None — which voxels are inside GT crop
        target_shape: shape of logits [B, C, D, H, W]

    Returns:
        [B, C, D, H, W] float — combined mask (1 where both class and voxel are valid)
    """
    # class_mask [B,C] → [B,C,1,1,1]
    cm = class_mask.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    if spatial_mask is not None:
        # spatial_mask [B,1,D,H,W] broadcasts across C
        return cm * spatial_mask
    else:
        return cm.expand(target_shape)


class MaskedBCELoss(nn.Module):
    """Binary cross-entropy with per-class + spatial masking.

    Only computes loss for annotated classes inside the GT crop region.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, D, H, W] raw model output (pre-sigmoid)
            targets: [B, C, D, H, W] binary labels
            mask:    [B, C] bool — which classes are annotated
            spatial_mask: [B, 1, D, H, W] float — 1 inside GT crop (optional)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        combined = _combine_masks(mask, spatial_mask, logits.shape)
        bce = bce * combined

        if self.class_weights is not None:
            w = self.class_weights.reshape(1, -1, 1, 1, 1)
            bce = bce * w

        n_annotated_voxels = combined.sum()
        if n_annotated_voxels > 0:
            return bce.sum() / n_annotated_voxels
        return bce.sum() * 0.0


class MaskedDiceLoss(nn.Module):
    """Soft Dice loss with per-class + spatial masking.

    Only counts voxels inside the GT crop for annotated classes.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, D, H, W]
            targets: [B, C, D, H, W]
            mask:    [B, C] bool — which classes are annotated
            spatial_mask: [B, 1, D, H, W] float — 1 inside GT crop (optional)
        """
        probs = torch.sigmoid(logits)
        B, C = logits.shape[:2]

        combined = _combine_masks(mask, spatial_mask, logits.shape)

        probs_masked = probs * combined
        targets_masked = targets * combined

        probs_flat = probs_masked.reshape(B, C, -1)
        targets_flat = targets_masked.reshape(B, C, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Valid = annotated class with positive target voxels
        has_positive = targets_flat.sum(dim=2) > 0
        valid = mask & has_positive

        dice_loss = 1.0 - dice
        dice_loss = dice_loss * valid.float()

        n_valid = valid.float().sum()
        if n_valid > 0:
            return dice_loss.sum() / n_valid
        return dice_loss.sum() * 0.0


class MaskedMultiLabelLoss(nn.Module):
    """Combined BCE + Dice with class masking.

    Args:
        bce_weight: Weight for BCE component.
        dice_weight: Weight for Dice component.
        class_weights: Optional per-class weights [C] for BCE.
        smooth: Smoothing for Dice denominator.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = MaskedBCELoss(class_weights=class_weights)
        self.dice_loss = MaskedDiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits:  [B, C, D, H, W]
            targets: [B, C, D, H, W]
            mask:    [B, C] bool
            spatial_mask: [B, 1, D, H, W] float (optional)

        Returns:
            total_loss: scalar
            components: dict with 'bce' and 'dice' for logging
        """
        bce = self.bce_loss(logits, targets, mask, spatial_mask)
        dice = self.dice_loss(logits, targets, mask, spatial_mask)
        total = self.bce_weight * bce + self.dice_weight * dice
        return total, {"bce": bce.detach(), "dice": dice.detach()}


# ---------------------------------------------------------------------------
# 2. Hierarchical loss
# ---------------------------------------------------------------------------

class HierarchicalLoss(nn.Module):
    """Combines fine, medium, and coarse losses using mapping matrices.

    Aggregates fine-level labels to medium/coarse using binary mapping matrices
    from class_mapping.py. Supports dynamic weighting that shifts toward fine
    as training progresses.

    Args:
        fine_to_medium: [n_medium, n_fine] binary mapping matrix.
        fine_to_coarse: [n_coarse, n_fine] binary mapping matrix.
        w_fine: Initial weight for fine loss.
        w_medium: Initial weight for medium loss.
        w_coarse: Initial weight for coarse loss.
        dynamic_weights: If True, shift weights toward fine over training.
    """

    def __init__(
        self,
        fine_to_medium: torch.Tensor,
        fine_to_coarse: torch.Tensor,
        w_fine: float = 0.6,
        w_medium: float = 0.25,
        w_coarse: float = 0.15,
        dynamic_weights: bool = True,
    ):
        super().__init__()
        self.register_buffer("f2m", fine_to_medium)  # [n_medium, n_fine]
        self.register_buffer("f2c", fine_to_coarse)  # [n_coarse, n_fine]
        self.w_fine_init = w_fine
        self.w_medium_init = w_medium
        self.w_coarse_init = w_coarse
        self.dynamic_weights = dynamic_weights

        self.base_loss = MaskedMultiLabelLoss()

    def _get_weights(self, progress: float) -> Tuple[float, float, float]:
        """Get loss weights for current training progress (0→1).

        Shifts from balanced toward fine-dominant:
          progress=0: fine=0.3, med=0.35, coarse=0.35
          progress=1: fine=0.7, med=0.2, coarse=0.1
        """
        if not self.dynamic_weights:
            return self.w_fine_init, self.w_medium_init, self.w_coarse_init

        w_fine = 0.3 + 0.4 * progress
        w_medium = 0.35 - 0.15 * progress
        w_coarse = 0.35 - 0.25 * progress
        return w_fine, w_medium, w_coarse

    def _aggregate_labels(
        self,
        fine_labels: torch.Tensor,
        fine_mask: torch.Tensor,
        mapping: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate fine labels/mask to coarser level.

        Args:
            fine_labels: [B, n_fine, D, H, W]
            fine_mask: [B, n_fine] bool
            mapping: [n_group, n_fine]

        Returns:
            group_labels: [B, n_group, D, H, W] — union of member labels
            group_mask: [B, n_group] — True if ANY member class is annotated
        """
        B = fine_labels.shape[0]
        spatial = fine_labels.shape[2:]

        fine_flat = fine_labels.reshape(B, fine_labels.shape[1], -1)
        group_flat = torch.einsum("gf,bfn->bgn", mapping, fine_flat)
        group_labels = group_flat.clamp(0, 1).reshape(B, mapping.shape[0], *spatial)

        group_mask = torch.einsum("gf,bf->bg", mapping, fine_mask.float()) > 0

        return group_labels, group_mask

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        fine_labels: torch.Tensor,
        fine_mask: torch.Tensor,
        progress: float = 0.0,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: dict from HierarchicalResUNet with 'fine', 'medium', 'coarse'
            fine_labels: [B, n_fine, D, H, W]
            fine_mask: [B, n_fine] bool
            progress: training progress 0→1 for dynamic weighting
            spatial_mask: [B, 1, D, H, W] float — 1 inside GT crop (optional)

        Returns:
            total_loss: scalar
            log_dict: per-component losses for logging
        """
        w_fine, w_medium, w_coarse = self._get_weights(progress)
        log = {}

        # Fine loss
        fine_loss, fine_components = self.base_loss(
            outputs["fine"], fine_labels, fine_mask, spatial_mask
        )
        log["fine_loss"] = fine_loss.detach()
        log["fine_bce"] = fine_components["bce"]
        log["fine_dice"] = fine_components["dice"]

        total = w_fine * fine_loss

        # Medium loss (if available — training only)
        if "medium" in outputs:
            medium_labels, medium_mask = self._aggregate_labels(
                fine_labels, fine_mask, self.f2m
            )
            medium_loss, med_comp = self.base_loss(
                outputs["medium"], medium_labels, medium_mask, spatial_mask
            )
            total = total + w_medium * medium_loss
            log["medium_loss"] = medium_loss.detach()

        # Coarse loss (if available — training only)
        if "coarse" in outputs:
            coarse_labels, coarse_mask = self._aggregate_labels(
                fine_labels, fine_mask, self.f2c
            )
            coarse_loss, coarse_comp = self.base_loss(
                outputs["coarse"], coarse_labels, coarse_mask, spatial_mask
            )
            total = total + w_coarse * coarse_loss
            log["coarse_loss"] = coarse_loss.detach()

        # Deep supervision on fine (if available)
        if "deep_fine" in outputs:
            deep_loss = torch.tensor(0.0, device=fine_labels.device)
            for i, deep_logits in enumerate(outputs["deep_fine"]):
                dl, _ = self.base_loss(deep_logits, fine_labels, fine_mask, spatial_mask)
                deep_loss = deep_loss + dl
            deep_loss = deep_loss / len(outputs["deep_fine"])
            total = total + 0.2 * deep_loss
            log["deep_loss"] = deep_loss.detach()

        log["total_loss"] = total.detach()
        log["w_fine"] = torch.tensor(w_fine)
        log["w_medium"] = torch.tensor(w_medium)
        log["w_coarse"] = torch.tensor(w_coarse)

        return total, log


# ---------------------------------------------------------------------------
# 3. Boundary-aware loss for instance segmentation classes
# ---------------------------------------------------------------------------

def compute_boundary_targets(
    labels: torch.Tensor,
    instance_indices: List[int],
    erosion_iterations: int = 2,
) -> torch.Tensor:
    """Compute boundary masks for instance classes by erosion.

    For each instance class, erode the binary mask and compute
    boundary = original - eroded.

    Args:
        labels: [B, C, D, H, W] binary labels
        instance_indices: list of class indices that are instance-evaluated
        erosion_iterations: number of erosion iterations

    Returns:
        boundaries: [B, n_instance, D, H, W] binary boundary masks
    """
    B = labels.shape[0]
    spatial = labels.shape[2:]
    n_inst = len(instance_indices)

    boundaries = torch.zeros(B, n_inst, *spatial, device=labels.device)

    for b in range(B):
        for j, cls_idx in enumerate(instance_indices):
            mask_np = labels[b, cls_idx].cpu().numpy()
            if mask_np.sum() == 0:
                continue
            eroded = ndimage.binary_erosion(
                mask_np, iterations=erosion_iterations
            ).astype(mask_np.dtype)
            boundary = mask_np - eroded
            boundaries[b, j] = torch.from_numpy(boundary).to(labels.device)

    return boundaries


class BoundaryAwareLoss(nn.Module):
    """Boundary-weighted loss for instance segmentation classes.

    Computes boundary masks via erosion, then applies extra weight to
    boundary voxels in the BCE loss for those classes.

    Args:
        instance_indices: Fine-class indices for instance-evaluated classes.
        boundary_weight: Extra weight multiplier for boundary voxels.
        erosion_iterations: Number of erosion steps to compute boundaries.
    """

    def __init__(
        self,
        instance_indices: List[int],
        boundary_weight: float = 5.0,
        erosion_iterations: int = 2,
    ):
        super().__init__()
        self.instance_indices = instance_indices
        self.boundary_weight = boundary_weight
        self.erosion_iterations = erosion_iterations

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, D, H, W] — full fine logits
            targets: [B, C, D, H, W] — full fine labels
            mask:    [B, C] bool
            spatial_mask: [B, 1, D, H, W] float (optional)

        Returns:
            boundary_loss: scalar (extra boundary-weighted BCE for instance classes)
        """
        if not self.instance_indices:
            return torch.tensor(0.0, device=logits.device)

        idx = self.instance_indices
        inst_logits = logits[:, idx]
        inst_targets = targets[:, idx]
        inst_mask = mask[:, idx]  # [B, n_inst]

        boundaries = compute_boundary_targets(
            targets, idx, self.erosion_iterations
        )

        bce = F.binary_cross_entropy_with_logits(
            inst_logits, inst_targets, reduction="none"
        )

        weight_map = torch.ones_like(bce)
        weight_map = weight_map + (self.boundary_weight - 1.0) * boundaries
        bce = bce * weight_map

        # Combine per-class + spatial mask
        combined = _combine_masks(inst_mask, spatial_mask, inst_logits.shape)
        bce = bce * combined

        n_voxels = combined.sum()
        if n_voxels > 0:
            return bce.sum() / n_voxels
        return bce.sum() * 0.0


# ---------------------------------------------------------------------------
# 4. Combined training loss
# ---------------------------------------------------------------------------

class CellMapLoss(nn.Module):
    """Full training loss for HierarchicalResUNet.

    Combines:
    - Hierarchical loss (fine + medium + coarse + deep supervision)
    - Boundary-aware loss for instance segmentation classes
    - Optional class weights

    Args:
        fine_to_medium: [n_medium, n_fine] mapping matrix.
        fine_to_coarse: [n_coarse, n_fine] mapping matrix.
        instance_indices: Fine-class indices for instance-evaluated classes.
        boundary_weight: Extra weight for boundary voxels.
        boundary_loss_weight: Weight of boundary loss in total.
        dynamic_weights: Enable dynamic hierarchy weighting.
        class_weights: Optional per-class weights for BCE.
    """

    def __init__(
        self,
        fine_to_medium: torch.Tensor,
        fine_to_coarse: torch.Tensor,
        instance_indices: Optional[List[int]] = None,
        boundary_weight: float = 5.0,
        boundary_loss_weight: float = 0.1,
        dynamic_weights: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.hierarchical_loss = HierarchicalLoss(
            fine_to_medium=fine_to_medium,
            fine_to_coarse=fine_to_coarse,
            dynamic_weights=dynamic_weights,
        )

        # Override base_loss with class weights if provided
        if class_weights is not None:
            self.hierarchical_loss.base_loss = MaskedMultiLabelLoss(
                class_weights=class_weights
            )

        self.boundary_loss_weight = boundary_loss_weight
        if instance_indices:
            self.boundary_loss = BoundaryAwareLoss(
                instance_indices=instance_indices,
                boundary_weight=boundary_weight,
            )
        else:
            self.boundary_loss = None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
        progress: float = 0.0,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: dict from HierarchicalResUNet
            labels: [B, n_fine, D, H, W] fine binary labels
            mask: [B, n_fine] bool annotation mask
            progress: training progress 0→1
            spatial_mask: [B, 1, D, H, W] float — 1 inside GT crop (optional)

        Returns:
            total_loss: scalar
            log_dict: all component losses for logging
        """
        total, log = self.hierarchical_loss(
            outputs, labels, mask, progress, spatial_mask
        )

        if self.boundary_loss is not None:
            b_loss = self.boundary_loss(
                outputs["fine"], labels, mask, spatial_mask
            )
            total = total + self.boundary_loss_weight * b_loss
            log["boundary_loss"] = b_loss.detach()

        log["total_loss"] = total.detach()
        return total, log
