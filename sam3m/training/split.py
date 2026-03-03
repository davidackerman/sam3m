"""Train/validation split utilities for CellMapDataset3D.

Splits crops (not voxels) with stratification by dataset to ensure
each dataset contributes to both train and val sets. Preserves rare
class coverage in the training split.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Tuple

import numpy as np

from sam3m.data.dataset import CellMapDataset3D

logger = logging.getLogger(__name__)


def split_dataset(
    dataset: CellMapDataset3D,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[CellMapDataset3D, CellMapDataset3D]:
    """Split dataset into train and val by holding out entire crops.

    Stratifies by dataset so each dataset contributes roughly val_fraction
    of its crops to validation. Datasets with very few crops (<=2) keep
    all crops in training to avoid losing rare class coverage.

    Args:
        dataset: The full CellMapDataset3D (with all crops discovered).
        val_fraction: Fraction of crops to hold out per dataset.
        seed: Random seed.

    Returns:
        train_ds, val_ds: Two CellMapDataset3D instances sharing the same
            config but with disjoint crop lists.
    """
    rng = np.random.default_rng(seed)

    # Group crop indices by dataset
    dataset_crops = defaultdict(list)
    for i, crop in enumerate(dataset.crops):
        dataset_crops[crop.dataset_name].append(i)

    train_indices = []
    val_indices = []

    for ds_name, indices in sorted(dataset_crops.items()):
        n = len(indices)
        if n <= 2:
            # Too few crops — keep all in train
            train_indices.extend(indices)
            continue

        n_val = max(1, int(n * val_fraction))
        shuffled = rng.permutation(indices).tolist()
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    # Build new datasets by copying and replacing crop lists
    train_ds = copy.copy(dataset)
    train_ds.crops = [dataset.crops[i] for i in train_indices]

    val_ds = copy.copy(dataset)
    val_ds.crops = [dataset.crops[i] for i in val_indices]
    val_ds.transforms = None  # no augmentation on validation

    # Log split stats
    train_datasets = set(c.dataset_name for c in train_ds.crops)
    val_datasets = set(c.dataset_name for c in val_ds.crops)

    logger.info(
        f"Split: {len(train_ds.crops)} train crops ({len(train_datasets)} datasets), "
        f"{len(val_ds.crops)} val crops ({len(val_datasets)} datasets)"
    )

    return train_ds, val_ds
