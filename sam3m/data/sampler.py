"""Class-balanced sampling for CellMapDataset3D.

Problem: Some classes appear in 200+ crops, others in <10.
Uniform crop sampling means rare classes barely appear in training.

Solution: At each step, pick the least-seen class so far, then sample
a crop that contains it. This ensures all 48 classes get roughly equal
representation over the course of an epoch.

Supports DDP: all ranks generate the same balanced sequence (via
deterministic per-epoch seeding), then each rank takes its shard.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """Sampler that balances class representation across batches.

    Algorithm:
        1. Build a crop-class matrix from the dataset (which crops annotate which classes)
        2. Maintain a running count of how many times each class has been seen
        3. At each step: pick the class with the lowest count, sample a crop
           that annotates it, return that crop index
        4. After returning a crop, increment counts for ALL classes that crop annotates

    This guarantees rare classes (e.g., perox with ~15 crops) get sampled
    as often as common classes (e.g., mito with 200+ crops).

    DDP support:
        All ranks generate the identical balanced index sequence (same seed +
        epoch), then each rank yields every ``world_size``-th element starting
        at ``rank``. Call :meth:`set_epoch` each epoch so the sequence varies.

    Args:
        dataset: A CellMapDataset3D instance (must have .crops and .get_crop_class_matrix()).
        samples_per_epoch: Number of samples per epoch (total across all ranks).
        seed: Random seed for reproducibility.
        rank: DDP rank (0 for single-GPU).
        world_size: DDP world size (1 for single-GPU).
    """

    def __init__(
        self,
        dataset,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        # [n_crops x n_classes] boolean matrix
        self.crop_class_matrix = dataset.get_crop_class_matrix()
        self.n_crops, self.n_classes = self.crop_class_matrix.shape

        # For each class, precompute which crop indices annotate it
        self.class_to_crops = {}
        for c in range(self.n_classes):
            crop_indices = np.where(self.crop_class_matrix[:, c])[0]
            if len(crop_indices) > 0:
                self.class_to_crops[c] = crop_indices

        # Classes that actually have at least one crop
        self.active_classes = sorted(self.class_to_crops.keys())

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling (required for DDP)."""
        self.epoch = epoch

    def __iter__(self):
        # Fresh RNG seeded per epoch — all ranks produce the same sequence
        rng = np.random.default_rng(self.seed + self.epoch)
        class_counts = np.zeros(self.n_classes, dtype=np.float64)
        all_indices = []

        for _ in range(self.samples_per_epoch):
            # Pick the least-seen active class (break ties randomly)
            active_counts = np.array([class_counts[c] for c in self.active_classes])
            min_count = active_counts.min()
            tied = [self.active_classes[i]
                    for i, v in enumerate(active_counts) if v == min_count]
            target_class = rng.choice(tied)

            # Sample a random crop that annotates this class
            crop_candidates = self.class_to_crops[target_class]
            crop_idx = rng.choice(crop_candidates)

            # Increment counts for all classes this crop annotates
            annotated = np.where(self.crop_class_matrix[crop_idx])[0]
            class_counts[annotated] += 1

            all_indices.append(crop_idx)

        # Pad to make evenly divisible across ranks (same as DistributedSampler)
        per_rank = math.ceil(len(all_indices) / self.world_size)
        total_size = per_rank * self.world_size
        while len(all_indices) < total_size:
            all_indices.append(all_indices[len(all_indices) % self.samples_per_epoch])

        # Each rank takes every world_size-th element
        yield from all_indices[self.rank::self.world_size]

    def __len__(self):
        return math.ceil(self.samples_per_epoch / self.world_size)
