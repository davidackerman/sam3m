"""Instance segmentation post-processing via watershed.

Converts dense multi-label probability maps to instance segmentations
for the 11 instance-evaluated classes in the CellMap challenge.

Pipeline per instance class:
1. Threshold probability -> binary foreground
2. Distance transform -> find local maxima as seeds
3. Watershed segmentation using seeds
4. Filter small instances
5. Output: integer instance label volume
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from sam3m.data.dataset import EVALUATED_INSTANCE_CLASSES

logger = logging.getLogger(__name__)

# Per-class parameters tuned for EM organelle morphology
DEFAULT_PARAMS = {
    "mito":  {"min_seed_size": 50,  "min_instance_size": 50,  "min_peak_dist": 10},
    "nuc":   {"min_seed_size": 200, "min_instance_size": 200, "min_peak_dist": 20},
    "ves":   {"min_seed_size": 10,  "min_instance_size": 5,   "min_peak_dist": 3},
    "endo":  {"min_seed_size": 20,  "min_instance_size": 15,  "min_peak_dist": 5},
    "lyso":  {"min_seed_size": 15,  "min_instance_size": 10,  "min_peak_dist": 5},
    "ld":    {"min_seed_size": 15,  "min_instance_size": 10,  "min_peak_dist": 5},
    "perox": {"min_seed_size": 10,  "min_instance_size": 5,   "min_peak_dist": 3},
    "np":    {"min_seed_size": 5,   "min_instance_size": 3,   "min_peak_dist": 2},
    "mt":    {"min_seed_size": 5,   "min_instance_size": 3,   "min_peak_dist": 2},
    "cell":  {"min_seed_size": 500, "min_instance_size": 500, "min_peak_dist": 30},
}


def instances_from_semantic(
    prob_map: np.ndarray,
    class_name: str,
    threshold: float = 0.5,
    params: Optional[Dict] = None,
) -> np.ndarray:
    """Convert a semantic probability map to instance segmentation.

    Args:
        prob_map: [Z, Y, X] float32 probability map for one class.
        class_name: Name of the class (for parameter lookup).
        threshold: Probability threshold for foreground.
        params: Override parameters (min_seed_size, min_instance_size, min_peak_dist).

    Returns:
        instances: [Z, Y, X] int32 instance labels (0 = background).
    """
    p = params or DEFAULT_PARAMS.get(class_name, {})
    min_seed_size = p.get("min_seed_size", 20)
    min_instance_size = p.get("min_instance_size", 10)
    min_peak_dist = p.get("min_peak_dist", 5)

    # 1. Threshold
    foreground = prob_map > threshold
    if foreground.sum() == 0:
        return np.zeros_like(prob_map, dtype=np.int32)

    # 2. Distance transform
    distance = ndimage.distance_transform_edt(foreground)

    # 3. Find seeds (local maxima in distance transform)
    try:
        coords = peak_local_max(
            distance,
            min_distance=min_peak_dist,
            labels=foreground.astype(int),
        )
    except Exception:
        # Fallback: connected components
        labeled, n_features = ndimage.label(foreground)
        return _filter_small(labeled, min_instance_size)

    if len(coords) == 0:
        # No peaks found - use connected components
        labeled, n_features = ndimage.label(foreground)
        return _filter_small(labeled, min_instance_size)

    # 4. Create seed mask
    seed_mask = np.zeros_like(foreground, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, start=1):
        seed_mask[z, y, x] = i

    # Expand seeds slightly and filter small ones
    seed_labeled, n_seeds = ndimage.label(
        ndimage.binary_dilation(seed_mask > 0, iterations=2)
    )

    # 5. Watershed
    instances = watershed(
        -distance,
        markers=seed_mask,
        mask=foreground,
    )

    # 6. Filter small instances
    instances = _filter_small(instances, min_instance_size)

    return instances.astype(np.int32)


def _filter_small(labeled: np.ndarray, min_size: int) -> np.ndarray:
    """Remove instances smaller than min_size voxels."""
    if min_size <= 0:
        return labeled

    output = labeled.copy()
    for inst_id in np.unique(labeled):
        if inst_id == 0:
            continue
        if (labeled == inst_id).sum() < min_size:
            output[labeled == inst_id] = 0

    # Re-label to make IDs contiguous
    unique_ids = np.unique(output[output > 0])
    remap = np.zeros(output.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        remap[old_id] = new_id
    return remap[output]


def postprocess_all_classes(
    prob_maps: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Run instance segmentation on all instance-evaluated classes.

    Args:
        prob_maps: [n_classes, Z, Y, X] probability maps.
        class_names: List of class names matching prob_maps channels.
        threshold: Probability threshold.

    Returns:
        Dict mapping class_name -> [Z, Y, X] int32 instance labels.
    """
    results = {}

    for cls_name in EVALUATED_INSTANCE_CLASSES:
        if cls_name not in class_names:
            continue
        cls_idx = class_names.index(cls_name)

        logger.info(f"Post-processing instances for {cls_name}...")
        instances = instances_from_semantic(
            prob_maps[cls_idx],
            class_name=cls_name,
            threshold=threshold,
        )
        n_instances = len(np.unique(instances)) - 1  # exclude background
        logger.info(f"  {cls_name}: {n_instances} instances")
        results[cls_name] = instances

    return results
