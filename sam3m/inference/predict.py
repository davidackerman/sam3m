"""Inference pipeline: sliding window + video propagation.

Processes full 3D EM volumes using overlapping sliding windows.
Each window is processed as a pseudo-video (z-slices as frames)
with optional bidirectional propagation for z-consistency.

Output: [48, Z, Y, X] probability maps for the full volume.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def gaussian_blend_weights(shape: Tuple[int, ...], sigma_frac: float = 0.125) -> np.ndarray:
    """Create Gaussian blending weights for a 3D patch.

    Args:
        shape: (D, H, W) patch dimensions.
        sigma_frac: Sigma as fraction of each dimension.

    Returns:
        weights: [D, H, W] Gaussian weights, peak=1 at center.
    """
    weights = np.ones(shape, dtype=np.float32)
    for axis, size in enumerate(shape):
        sigma = size * sigma_frac
        center = size / 2.0
        coords = np.arange(size, dtype=np.float32)
        gaussian_1d = np.exp(-((coords - center) ** 2) / (2 * sigma ** 2))
        # Reshape for broadcasting
        reshape = [1, 1, 1]
        reshape[axis] = size
        weights *= gaussian_1d.reshape(reshape)
    return weights


@torch.no_grad()
def predict_volume(
    model,
    raw_volume: np.ndarray,
    norm_fn=None,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    stride: Tuple[int, int, int] = (64, 64, 64),
    num_frames: int = 16,
    frame_stride: int = 8,
    image_size: int = 1008,
    n_classes: int = 48,
    device: str = "cuda",
    bidirectional: bool = True,
) -> np.ndarray:
    """Predict on a full 3D volume using sliding window + video mode.

    Args:
        model: Trained SAM3CellMapModel.
        raw_volume: [Z, Y, X] uint8 or float32 EM volume.
        norm_fn: Optional function to normalize raw values to [0, 1].
        patch_size: 3D patch size for sliding window.
        stride: Stride between patches (< patch_size for overlap).
        num_frames: Z-slices per video.
        frame_stride: Stride between z-slices in video.
        image_size: SAM3 input resolution.
        n_classes: Number of output classes.
        device: CUDA device.
        bidirectional: If True, run forward + backward passes and average.

    Returns:
        predictions: [n_classes, Z, Y, X] float32 probability maps.
    """
    model.eval()
    Z, Y, X = raw_volume.shape

    # Normalize
    if norm_fn is not None:
        raw_volume = norm_fn(raw_volume)
    elif raw_volume.dtype == np.uint8:
        raw_volume = raw_volume.astype(np.float32) / 255.0

    # Output accumulation
    pred_sum = np.zeros((n_classes, Z, Y, X), dtype=np.float32)
    weight_sum = np.zeros((1, Z, Y, X), dtype=np.float32)
    blend = gaussian_blend_weights(patch_size)

    D, H, W = patch_size
    sz, sy, sx = stride

    # Generate patch positions
    z_starts = list(range(0, max(Z - D + 1, 1), sz))
    y_starts = list(range(0, max(Y - H + 1, 1), sy))
    x_starts = list(range(0, max(X - W + 1, 1), sx))

    # Ensure last patch is included
    if z_starts[-1] + D < Z:
        z_starts.append(Z - D)
    if y_starts[-1] + H < Y:
        y_starts.append(Y - H)
    if x_starts[-1] + W < X:
        x_starts.append(X - W)

    n_patches = len(z_starts) * len(y_starts) * len(x_starts)
    logger.info(f"Processing {n_patches} patches from volume {raw_volume.shape}")

    patch_idx = 0
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = raw_volume[z0:z0+D, y0:y0+H, x0:x0+W]
                patch_pred = _predict_patch(
                    model, patch, num_frames, frame_stride,
                    image_size, n_classes, device, bidirectional,
                )
                # Blend into output
                pred_sum[:, z0:z0+D, y0:y0+H, x0:x0+W] += patch_pred * blend[np.newaxis]
                weight_sum[:, z0:z0+D, y0:y0+H, x0:x0+W] += blend[np.newaxis]

                patch_idx += 1
                if patch_idx % 10 == 0:
                    logger.info(f"  Patch {patch_idx}/{n_patches}")

    # Normalize by weights
    pred_sum /= np.maximum(weight_sum, 1e-8)
    return pred_sum


def _predict_patch(
    model,
    patch: np.ndarray,
    num_frames: int,
    frame_stride: int,
    image_size: int,
    n_classes: int,
    device: str,
    bidirectional: bool,
) -> np.ndarray:
    """Predict on a single 3D patch using video mode.

    Args:
        patch: [D, H, W] float32 normalized EM patch.

    Returns:
        predictions: [n_classes, D, H, W] float32 probabilities.
    """
    D, H, W = patch.shape

    # Select z-slices
    z_indices = _select_z_indices(D, num_frames, frame_stride)
    all_preds = np.zeros((n_classes, D, H, W), dtype=np.float32)
    all_counts = np.zeros((1, D, H, W), dtype=np.float32)

    def run_direction(z_list):
        for z in z_list:
            # Extract slice, resize, make RGB
            raw_slice = torch.from_numpy(patch[z:z+1]).float()  # [1, H, W]
            raw_resized = F.interpolate(
                raw_slice.unsqueeze(0), size=(image_size, image_size),
                mode="bilinear", align_corners=False,
            )  # [1, 1, image_size, image_size]
            rgb = raw_resized.expand(-1, 3, -1, -1).to(device)  # [1, 3, H, W]

            outputs = model(rgb, target_size=(H, W))
            probs = torch.sigmoid(outputs["fine"]).cpu().numpy()[0]  # [C, H, W]

            all_preds[:, z] += probs
            all_counts[:, z] += 1.0

    # Forward pass
    run_direction(z_indices)

    # Backward pass (if bidirectional)
    if bidirectional:
        run_direction(reversed(z_indices))

    # Average
    all_preds /= np.maximum(all_counts, 1.0)

    # Interpolate for z-slices not directly predicted
    # (Simple: use nearest predicted slice)
    for z in range(D):
        if all_counts[0, z, 0, 0] == 0:
            # Find nearest predicted z
            predicted_zs = np.where(all_counts[0, :, 0, 0] > 0)[0]
            if len(predicted_zs) > 0:
                nearest = predicted_zs[np.argmin(np.abs(predicted_zs - z))]
                all_preds[:, z] = all_preds[:, nearest]

    return all_preds


def _select_z_indices(depth, num_frames, frame_stride):
    """Select z-slice indices (same logic as video_dataset)."""
    max_frames = min(num_frames, depth)
    stride = frame_stride
    while max_frames * stride > depth and stride > 1:
        stride -= 1
    actual_frames = min(max_frames, depth // max(stride, 1))
    actual_frames = max(actual_frames, 1)
    total_span = (actual_frames - 1) * stride
    start = (depth - total_span) // 2
    return [start + i * stride for i in range(actual_frames)]
