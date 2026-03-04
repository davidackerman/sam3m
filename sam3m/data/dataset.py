import csv
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from scipy.ndimage import (
    binary_dilation,
    center_of_mass,
    distance_transform_edt,
    zoom as ndimage_zoom,
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 48 evaluated classes (from tested_classes.csv)
# ---------------------------------------------------------------------------
EVALUATED_CLASSES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "lyso_mem", "lyso_lum",
    "ld_mem", "ld_lum", "er_mem", "er_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in",
    "nuc", "golgi", "ves", "endo", "lyso", "ld", "eres",
    "perox_mem", "perox_lum", "perox",
    "mito", "er", "ne", "np", "chrom", "mt",
    "cell", "er_mem_all", "ne_mem_all",
]
N_CLASSES = len(EVALUATED_CLASSES)

INSTANCE_CLASSES = [
    "nuc", "vim", "ves", "endo", "lyso", "ld",
    "perox", "mito", "np", "mt", "cell",
]

# Instance classes that are in EVALUATED_CLASSES (vim excluded)
_EVALUATED_SET = set(EVALUATED_CLASSES)
EVALUATED_INSTANCE_CLASSES = [c for c in INSTANCE_CLASSES if c in _EVALUATED_SET]
N_INSTANCE_CLASSES = len(EVALUATED_INSTANCE_CLASSES)  # 10
INSTANCE_CLASS_INDEX = {name: i for i, name in enumerate(EVALUATED_INSTANCE_CLASSES)}


# ---------------------------------------------------------------------------
# Challenge manifest helpers
# ---------------------------------------------------------------------------

def load_challenge_manifest(csv_path: str) -> set:
    """Load a challenge crop manifest CSV and return (dataset, crop_id) pairs.

    The manifest has columns: crop_name,dataset,class_label,...
    Multiple rows per crop (one per class). Returns unique (dataset, crop_id)
    pairs where crop_id is the numeric string (e.g., "234").
    """
    pairs = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.add((row["dataset"].strip(), row["crop_name"].strip()))
    return pairs


_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

CHALLENGE_MANIFESTS = {
    "train": os.path.join(_CONFIGS_DIR, "challenge_train_crops.csv"),
    "test": os.path.join(_CONFIGS_DIR, "challenge_test_crops.csv"),
}


# ---------------------------------------------------------------------------
# Zarr metadata helpers
# ---------------------------------------------------------------------------

def get_scale_info(zarr_grp_path: str):
    """Read multiscale metadata from a zarr group.

    Returns:
        offsets: dict mapping scale path -> [z, y, x] translation (world nm)
        resolutions: dict mapping scale path -> [z, y, x] voxel size (nm)
        shapes: dict mapping scale path -> volume shape (voxels)
    """
    grp = zarr.open(zarr_grp_path, mode="r")
    attrs = grp.attrs
    offsets, resolutions, shapes = {}, {}, {}
    for scale in attrs["multiscales"][0]["datasets"]:
        path = scale["path"]
        resolutions[path] = scale["coordinateTransformations"][0]["scale"]
        offsets[path] = scale["coordinateTransformations"][1]["translation"]
        shapes[path] = grp[path].shape
    return offsets, resolutions, shapes


def get_raw_path(em_base: str) -> Optional[str]:
    """Select fibsem-uint8 if available, else fibsem-uint16."""
    uint8_path = os.path.join(em_base, "fibsem-uint8")
    uint16_path = os.path.join(em_base, "fibsem-uint16")
    if os.path.isdir(uint8_path):
        return uint8_path
    if os.path.isdir(uint16_path):
        return uint16_path
    return None


def find_scale_for_resolution(
    zarr_grp_path: str,
    target_res: float,
    max_ratio: float = 2.0,
) -> Optional[Tuple[str, List[float], List[float], Tuple[int, ...]]]:
    """Find the scale level closest to target_res (matching on Y axis).

    Returns (scale_path, resolution, offset, shape) or None.
    """
    offsets, resolutions, shapes = get_scale_info(zarr_grp_path)
    candidates = []
    for name, res in resolutions.items():
        y_res = res[1]
        ratio = max(y_res / target_res, target_res / y_res)
        if ratio <= max_ratio:
            candidates.append((abs(y_res - target_res), name, res, offsets[name], shapes[name]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, best_path, best_res, best_off, best_shape = candidates[0]
    return best_path, best_res, best_off, best_shape


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@dataclass
class NormParams:
    min_val: float
    max_val: float
    inverted: bool


def load_norms(csv_path: str) -> Dict[str, NormParams]:
    """Load per-dataset normalization parameters from CSV."""
    norms = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["dataset"].strip()
            norms[name] = NormParams(
                min_val=float(row["min"]),
                max_val=float(row["max"]),
                inverted=row["inverted"].strip() == "True",
            )
    return norms


# ---------------------------------------------------------------------------
# Crop metadata
# ---------------------------------------------------------------------------

@dataclass
class ClassInfo:
    """Metadata for one class label within a crop."""
    zarr_path: str        # path to the class zarr group (e.g., .../crop1/mito)
    scale_path: str       # e.g., "s1"
    resolution: List[float]   # [z, y, x] in nm
    offset_world: List[float]  # [z, y, x] in nm
    shape: Tuple[int, ...]     # voxels at this scale


@dataclass
class CropInfo:
    """All pre-computed metadata for one annotated crop."""
    dataset_name: str
    crop_id: str
    # Raw EM
    raw_zarr_path: str
    raw_scale_path: str
    raw_resolution: List[float]
    raw_offset_world: List[float]
    raw_shape: Tuple[int, ...]
    # Normalization
    norm_params: NormParams
    # Per-class label info (only for classes present in this crop)
    class_info: Dict[str, ClassInfo] = field(default_factory=dict)
    # Which of the target classes are annotated in this crop
    annotated_classes: set = field(default_factory=set)
    # Crop bounding box in world coordinates [z, y, x]
    crop_origin_world: Optional[List[float]] = None
    crop_extent_world: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Instance target helpers
# ---------------------------------------------------------------------------


def _compute_boundary_map(
    instance_mask: np.ndarray, dilation_width: int = 2
) -> np.ndarray:
    """Inter-instance boundary map.

    1.0 where adjacent voxels have different non-zero instance IDs.
    Dilated by ``dilation_width`` voxels so boundaries are 2-4 voxels wide
    (thin boundaries are harder to learn).
    """
    boundary = np.zeros(instance_mask.shape, dtype=np.float32)
    for axis in range(instance_mask.ndim):
        slc_a = [slice(None)] * instance_mask.ndim
        slc_b = [slice(None)] * instance_mask.ndim
        slc_a[axis] = slice(None, -1)
        slc_b[axis] = slice(1, None)

        a = instance_mask[tuple(slc_a)]
        b = instance_mask[tuple(slc_b)]

        is_bnd = (a != b) & (a > 0) & (b > 0)
        boundary[tuple(slc_a)] = np.maximum(boundary[tuple(slc_a)], is_bnd)
        boundary[tuple(slc_b)] = np.maximum(boundary[tuple(slc_b)], is_bnd)

    if dilation_width > 1:
        struct = np.ones([3] * instance_mask.ndim)
        boundary = binary_dilation(
            boundary > 0, struct, iterations=dilation_width - 1
        ).astype(np.float32)

    return boundary


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CellMapDataset3D(Dataset):
    """3D multi-label dataset for the CellMap Segmentation Challenge.

    Discovers all annotated crops from zarr stores, and samples random 3D
    patches returning raw EM + binary labels for all target classes + an
    annotation mask indicating which classes were annotated in the crop.
    """

    def __init__(
        self,
        data_root: str = "/nrs/cellmap/data",
        norms_csv: str = None,
        target_classes: List[str] = None,
        target_resolution: float = 8.0,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        samples_per_epoch: int = 1000,
        min_crop_voxels: int = 32,
        max_resolution_ratio: float = 2.0,
        cache_dir: str = None,
        seed: int = 42,
        transforms=None,
        skip_datasets: List[str] = None,
        include_datasets: List[str] = None,
        challenge_split: str = None,
        instance_mode: bool = False,
        scale_factors: List[float] = None,
    ):
        self.data_root = data_root
        self.target_classes = target_classes or EVALUATED_CLASSES
        self.n_classes = len(self.target_classes)
        self.target_resolution = target_resolution
        self.patch_size = np.array(patch_size)
        self.patch_world = self.patch_size * target_resolution
        self.scale_factors = scale_factors or [1.0]
        self.samples_per_epoch = samples_per_epoch
        self.min_crop_voxels = min_crop_voxels
        self.max_resolution_ratio = max_resolution_ratio
        self.rng = np.random.default_rng(seed)
        self.transforms = transforms
        self.skip_datasets = set(skip_datasets or [])
        self.include_datasets = set(include_datasets) if include_datasets else None
        self.challenge_split = challenge_split
        self.instance_mode = instance_mode

        # Load challenge manifest for crop-level filtering
        self._challenge_crops = None
        if challenge_split is not None:
            if challenge_split not in CHALLENGE_MANIFESTS:
                raise ValueError(
                    f"challenge_split must be one of {list(CHALLENGE_MANIFESTS)}, "
                    f"got '{challenge_split}'"
                )
            manifest_path = CHALLENGE_MANIFESTS[challenge_split]
            self._challenge_crops = load_challenge_manifest(manifest_path)
            # Also restrict to only the datasets in the manifest (speeds up discovery)
            manifest_datasets = {ds for ds, _ in self._challenge_crops}
            if self.include_datasets is not None:
                self.include_datasets &= manifest_datasets
            else:
                self.include_datasets = manifest_datasets
            logger.info(
                f"Challenge split '{challenge_split}': {len(self._challenge_crops)} "
                f"crops across {len(manifest_datasets)} datasets"
            )

        # Instance class mapping (for instance_mode)
        if instance_mode:
            self.instance_class_to_idx = {
                c: i for i, c in enumerate(EVALUATED_INSTANCE_CLASSES)
            }

        # Build class name -> index lookup
        self.class_to_idx = {c: i for i, c in enumerate(self.target_classes)}

        # Load normalization
        if norms_csv is None:
            norms_csv = os.path.join(
                os.path.dirname(__file__), "..", "..", "configs", "norms.csv"
            )
        self.norms = load_norms(norms_csv)

        # Cache directory for discovered crops
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "configs"
            )
        self.cache_dir = cache_dir

        # Discover crops (or load from cache)
        self.crops: List[CropInfo] = []
        self._discover_crops_cached()

        logger.info(
            f"CellMapDataset3D: {len(self.crops)} crops, "
            f"{self.n_classes} classes, "
            f"target_resolution={target_resolution}nm, "
            f"patch_size={tuple(self.patch_size)}"
        )
        if len(self.crops) == 0:
            raise RuntimeError(
                f"No valid crops found in {data_root} at resolution {target_resolution}nm"
            )

    # ------------------------------------------------------------------
    # Discovery with caching
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Build a unique cache key from discovery parameters."""
        key = json.dumps({
            "data_root": self.data_root,
            "target_classes": self.target_classes,
            "target_resolution": self.target_resolution,
            "min_crop_voxels": self.min_crop_voxels,
            "max_resolution_ratio": self.max_resolution_ratio,
            "skip_datasets": sorted(self.skip_datasets),
            "include_datasets": sorted(self.include_datasets) if self.include_datasets else None,
            "challenge_split": self.challenge_split,
        }, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _cache_path(self) -> str:
        return os.path.join(
            self.cache_dir, f"crops_cache_{self._cache_key()}.pkl"
        )

    def _discover_crops_cached(self):
        """Load crops from cache if available, else discover and save."""
        path = self._cache_path()
        if os.path.exists(path):
            logger.info(f"Loading crop cache from {path}")
            with open(path, "rb") as f:
                self.crops = pickle.load(f)
            # Re-apply current norms (in case norms.csv changed)
            for crop in self.crops:
                crop.norm_params = self.norms.get(
                    crop.dataset_name,
                    NormParams(min_val=0.0, max_val=255.0, inverted=False),
                )
            return

        logger.info("No cache found, running full discovery (this takes ~8 min)...")
        self._discover_crops()

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.crops, f)
        logger.info(f"Saved crop cache to {path}")

    def _discover_crops(self):
        """Walk data_root and build CropInfo for every valid crop."""
        if not os.path.isdir(self.data_root):
            logger.warning(f"data_root {self.data_root} does not exist")
            return

        datasets = sorted(os.listdir(self.data_root))
        for dataset_name in datasets:
            if self.include_datasets is not None and dataset_name not in self.include_datasets:
                continue
            if dataset_name in self.skip_datasets:
                logger.info(f"Skipping dataset: {dataset_name}")
                continue

            dataset_dir = os.path.join(self.data_root, dataset_name)
            if not os.path.isdir(dataset_dir):
                continue

            # Check for groundtruth labels
            gt_base = os.path.join(
                dataset_dir, f"{dataset_name}.zarr", "recon-1", "labels", "groundtruth"
            )
            if not os.path.isdir(gt_base):
                continue

            # Find raw EM path
            em_base = os.path.join(
                dataset_dir, f"{dataset_name}.zarr", "recon-1", "em"
            )
            raw_path = get_raw_path(em_base)
            if raw_path is None:
                continue

            # Check raw has target resolution
            raw_scale_info = find_scale_for_resolution(
                raw_path, self.target_resolution, max_ratio=1.01
            )
            if raw_scale_info is None:
                continue
            raw_scale_path, raw_res, raw_off, raw_shape = raw_scale_info

            # Get normalization
            norm = self.norms.get(
                dataset_name,
                NormParams(min_val=0.0, max_val=255.0, inverted=False),
            )

            # Process each crop
            for crop_name in sorted(os.listdir(gt_base)):
                if not crop_name.startswith("crop"):
                    continue
                crop_dir = os.path.join(gt_base, crop_name)
                if not os.path.isdir(crop_dir):
                    continue

                # Challenge crop-level filtering: manifest uses numeric IDs
                # (e.g., "234"), filesystem uses "crop234"
                if self._challenge_crops is not None:
                    crop_number = crop_name.removeprefix("crop")
                    if (dataset_name, crop_number) not in self._challenge_crops:
                        continue

                crop_info = self._build_crop_info(
                    dataset_name, crop_name, crop_dir,
                    raw_path, raw_scale_path, raw_res, raw_off, raw_shape,
                    norm,
                )
                if crop_info is not None:
                    self.crops.append(crop_info)

    def _build_crop_info(
        self,
        dataset_name: str,
        crop_name: str,
        crop_dir: str,
        raw_zarr_path: str,
        raw_scale_path: str,
        raw_res: List[float],
        raw_off: List[float],
        raw_shape: Tuple[int, ...],
        norm: NormParams,
    ) -> Optional[CropInfo]:
        """Build CropInfo for a single crop by inspecting its class directories."""
        crop = CropInfo(
            dataset_name=dataset_name,
            crop_id=crop_name,
            raw_zarr_path=raw_zarr_path,
            raw_scale_path=raw_scale_path,
            raw_resolution=raw_res,
            raw_offset_world=raw_off,
            raw_shape=raw_shape,
            norm_params=norm,
        )

        # Check which target classes exist in this crop
        crop_subdirs = set(os.listdir(crop_dir))
        ref_class_info = None

        for cls_name in self.target_classes:
            if cls_name not in crop_subdirs:
                continue
            cls_path = os.path.join(crop_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue

            info = find_scale_for_resolution(
                cls_path, self.target_resolution, self.max_resolution_ratio
            )
            if info is None:
                continue

            scale_path, res, off, shape = info
            ci = ClassInfo(
                zarr_path=cls_path,
                scale_path=scale_path,
                resolution=res,
                offset_world=off,
                shape=shape,
            )
            crop.class_info[cls_name] = ci
            crop.annotated_classes.add(cls_name)
            if ref_class_info is None:
                ref_class_info = ci

        if ref_class_info is None:
            return None

        # Compute crop bounding box in world coordinates from the reference class
        ref_off = np.array(ref_class_info.offset_world)
        ref_res = np.array(ref_class_info.resolution)
        ref_shape = np.array(ref_class_info.shape)
        crop.crop_origin_world = ref_off.tolist()
        crop.crop_extent_world = (ref_shape * ref_res).tolist()

        # Skip crops that are too small (check in target-resolution voxels)
        crop_voxels = np.array(crop.crop_extent_world) / self.target_resolution
        if np.any(crop_voxels < self.min_crop_voxels):
            return None

        return crop

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _normalize_raw(self, raw: np.ndarray, norm: NormParams) -> np.ndarray:
        """Normalize raw EM to [0, 1] using per-dataset params."""
        raw = raw.astype(np.float32)
        denom = norm.max_val - norm.min_val
        if denom == 0:
            denom = 1.0
        raw = (raw - norm.min_val) / denom
        np.clip(raw, 0.0, 1.0, out=raw)
        if norm.inverted:
            raw = 1.0 - raw
        return raw

    def _extract_patch(self, crop: CropInfo, scale_factor: float = 1.0):
        """Extract a random 3D patch from a crop, resampled to isotropic target_resolution.

        All reads are done in world coordinates, then resampled to
        patch_size voxels at target_resolution. This correctly handles
        anisotropic raw/label resolutions.

        When scale_factor > 1, a proportionally larger world volume is read
        and downsampled to the same patch_size, giving the model a wider
        field of view at coarser effective resolution.

        Returns:
            raw: np.ndarray [D, H, W] float32 in [0, 1]
            labels: np.ndarray [n_classes, D, H, W] float32 binary
            annotated_mask: np.ndarray [n_classes] bool — which classes are annotated
            spatial_mask: np.ndarray [D, H, W] float32 — 1 inside GT crop, 0 outside
        """
        D, H, W = self.patch_size
        # Scale the world extent: scale_factor=2 reads 2x larger volume
        patch_world = self.patch_world * scale_factor

        crop_origin = np.array(crop.crop_origin_world)
        crop_extent = np.array(crop.crop_extent_world)
        crop_end = crop_origin + crop_extent
        crop_center = crop_origin + crop_extent / 2

        # Whether the crop is smaller than the patch in any dimension
        small_crop = np.any(crop_extent < patch_world)

        if small_crop:
            # Center the patch on the crop so raw EM context surrounds the GT
            sample_origin = crop_center - patch_world / 2
        else:
            # Random origin within valid range
            max_origin = crop_end - patch_world
            sample_origin = np.array([
                self.rng.uniform(crop_origin[i], max_origin[i])
                for i in range(3)
            ])

        # --- Read raw (always full patch, never padded black) ---
        raw_res = np.array(crop.raw_resolution)
        raw_off = np.array(crop.raw_offset_world)
        raw_shape = np.array(crop.raw_shape)

        # How many raw voxels cover the target world extent per axis
        raw_read_vox = np.ceil(patch_world / raw_res).astype(int)

        raw_start_vox = np.round((sample_origin - raw_off) / raw_res).astype(int)
        # Clamp so the full read window stays within the raw volume
        raw_start_vox = np.clip(raw_start_vox, 0, raw_shape - raw_read_vox)
        raw_start_vox = np.maximum(raw_start_vox, 0)
        raw_end_vox = np.minimum(raw_start_vox + raw_read_vox, raw_shape)

        # Update sample_origin to match what we actually read (after clamping)
        sample_origin = raw_off + raw_start_vox * raw_res

        raw_arr = zarr.open(
            os.path.join(crop.raw_zarr_path, crop.raw_scale_path), mode="r"
        )
        raw_patch = np.array(
            raw_arr[
                raw_start_vox[0]:raw_end_vox[0],
                raw_start_vox[1]:raw_end_vox[1],
                raw_start_vox[2]:raw_end_vox[2],
            ]
        )
        raw_patch = self._normalize_raw(raw_patch, crop.norm_params)

        # Resample raw to isotropic patch_size if needed (anisotropic resolution)
        if raw_patch.shape != (D, H, W):
            zoom_factors = (
                D / raw_patch.shape[0],
                H / raw_patch.shape[1],
                W / raw_patch.shape[2],
            )
            raw_patch = ndimage_zoom(raw_patch, zoom_factors, order=1)
        raw_patch = raw_patch[:D, :H, :W]

        # --- Build spatial annotation mask from crop bounding box ---
        # Compute which output voxels fall inside the annotated crop region.
        # The crop covers [crop_origin, crop_end) in world coords; we need
        # to map that to output-patch voxel coordinates [0..D, 0..H, 0..W].
        # Effective voxel size = patch_world / patch_size (accounts for scale_factor)
        effective_res = patch_world / self.patch_size
        crop_start_in_patch = np.maximum(
            np.round((crop_origin - sample_origin) / effective_res).astype(int), 0
        )
        crop_end_in_patch = np.minimum(
            np.round((crop_end - sample_origin) / effective_res).astype(int),
            self.patch_size,
        )

        # 3-D spatial mask: 1 inside crop, 0 outside
        spatial_mask = np.zeros((D, H, W), dtype=np.float32)
        spatial_mask[
            crop_start_in_patch[0]:crop_end_in_patch[0],
            crop_start_in_patch[1]:crop_end_in_patch[1],
            crop_start_in_patch[2]:crop_end_in_patch[2],
        ] = 1.0

        # --- Read labels (padded with zeros outside crop boundary) ---
        labels = np.zeros((self.n_classes, D, H, W), dtype=np.float32)
        annotated_mask = np.zeros(self.n_classes, dtype=bool)

        # Instance mode: also collect raw integer instance IDs
        if self.instance_mode:
            instance_ids = np.zeros((N_INSTANCE_CLASSES, D, H, W), dtype=np.int32)
            instance_annotated = np.zeros(N_INSTANCE_CLASSES, dtype=bool)

        for cls_name, cls_idx in self.class_to_idx.items():
            if cls_name not in crop.annotated_classes:
                continue

            annotated_mask[cls_idx] = True

            ci = crop.class_info[cls_name]
            cls_res = np.array(ci.resolution)
            cls_off = np.array(ci.offset_world)

            # How many class voxels cover the target world extent
            cls_patch_vox = np.ceil(patch_world / cls_res).astype(int)
            cls_patch_vox = np.maximum(cls_patch_vox, 1)

            cls_start_vox = np.round((sample_origin - cls_off) / cls_res).astype(int)
            cls_start_vox = np.maximum(cls_start_vox, 0)

            # Clip to label array bounds
            cls_shape = np.array(ci.shape)
            cls_end_vox = np.minimum(cls_start_vox + cls_patch_vox, cls_shape)
            cls_read_size = cls_end_vox - cls_start_vox

            if np.any(cls_read_size <= 0):
                continue

            try:
                label_arr = zarr.open(
                    os.path.join(ci.zarr_path, ci.scale_path), mode="r"
                )
                label_patch = np.array(
                    label_arr[
                        cls_start_vox[0]:cls_end_vox[0],
                        cls_start_vox[1]:cls_end_vox[1],
                        cls_start_vox[2]:cls_end_vox[2],
                    ]
                )
            except Exception as e:
                logger.warning(
                    f"Failed to read label {cls_name} for "
                    f"{crop.dataset_name}/{crop.crop_id}: {e}"
                )
                continue

            # Binarize: >0 and !=255 means present
            label_binary = ((label_patch > 0) & (label_patch != 255)).astype(np.float32)

            # Instance mode: preserve raw integer IDs for instance classes
            # (before binarization, zeroing out the 255 exclusion marker)
            is_instance_cls = (
                self.instance_mode and cls_name in INSTANCE_CLASS_INDEX
            )
            if is_instance_cls:
                label_int = label_patch.copy()
                label_int[label_patch == 255] = 0  # exclude marker → background

            # Place label into full patch-sized canvas, then resample
            # For small crops, label only covers part of the patch
            if label_binary.shape != tuple(cls_patch_vox):
                label_origin_in_patch = np.round(
                    (cls_off + cls_start_vox * cls_res - sample_origin) / cls_res
                ).astype(int)
                label_origin_in_patch = np.maximum(label_origin_in_patch, 0)

                canvas = np.zeros(tuple(cls_patch_vox), dtype=np.float32)
                slices = tuple(
                    slice(label_origin_in_patch[i],
                          min(label_origin_in_patch[i] + label_binary.shape[i], cls_patch_vox[i]))
                    for i in range(3)
                )
                trimmed = tuple(
                    slice(0, slices[i].stop - slices[i].start)
                    for i in range(3)
                )
                canvas[slices] = label_binary[trimmed]
                label_binary = canvas

                # Same placement for integer instance IDs
                if is_instance_cls:
                    int_canvas = np.zeros(tuple(cls_patch_vox), dtype=np.int32)
                    int_canvas[slices] = label_int[trimmed]
                    label_int = int_canvas

            # Resample to patch_size
            if label_binary.shape != (D, H, W):
                zoom_factors = (
                    D / label_binary.shape[0],
                    H / label_binary.shape[1],
                    W / label_binary.shape[2],
                )
                label_binary = ndimage_zoom(label_binary, zoom_factors, order=0)
                if is_instance_cls:
                    label_int = ndimage_zoom(label_int, zoom_factors, order=0)

            labels[cls_idx] = label_binary[:D, :H, :W]

            # Store instance IDs
            if is_instance_cls:
                inst_idx = INSTANCE_CLASS_INDEX[cls_name]
                instance_ids[inst_idx] = label_int[:D, :H, :W]
                instance_annotated[inst_idx] = True

        if self.instance_mode:
            return raw_patch, labels, annotated_mask, spatial_mask, scale_factor, instance_ids, instance_annotated
        return raw_patch, labels, annotated_mask, spatial_mask, scale_factor

    # ------------------------------------------------------------------
    # Instance target generation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_instance_targets(
        instance_ids: np.ndarray,
        center_sigma: float = 3.0,
    ) -> tuple:
        """Compute class-agnostic instance targets from integer instance IDs.

        Generates center heatmap (Gaussian at each instance centroid),
        offset vectors (normalized direction to centroid), boundary map
        (where different instances touch), and foreground mask.

        Args:
            instance_ids: [N_INSTANCE, D, H, W] int32 — raw instance IDs.
            center_sigma: Gaussian sigma for center heatmap blobs.

        Returns:
            center_heatmap: [D, H, W] float32 — Gaussian peaks at centroids.
            offsets: [3, D, H, W] float32 — normalized vectors to centroid.
            boundary: [D, H, W] float32 — inter-instance boundary map.
            foreground: [D, H, W] float32 — union of all instance foreground.
        """
        N = instance_ids.shape[0]
        D, H, W = instance_ids.shape[1:]

        center_heatmap = np.zeros((D, H, W), dtype=np.float32)
        offsets = np.zeros((3, D, H, W), dtype=np.float32)

        # Build combined instance map (unique global IDs) for boundary detection
        combined_ids = np.zeros((D, H, W), dtype=np.int32)
        next_global_id = 1

        coords = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)

        for c in range(N):
            ids = instance_ids[c]
            for inst_id in np.unique(ids):
                if inst_id == 0:
                    continue
                mask = (ids == inst_id)
                com = np.array(center_of_mass(mask), dtype=np.float32)

                # Offset vectors: normalized to [-1, 1] per instance
                for axis in range(3):
                    raw_offset = com[axis] - coords[axis]
                    axis_coords = coords[axis][mask]
                    half_extent = max(
                        (axis_coords.max() - axis_coords.min()) / 2.0, 1.0
                    )
                    offsets[axis][mask] = raw_offset[mask] / half_extent

                # Center heatmap: Gaussian at center of mass
                radius = int(4 * center_sigma)
                z0 = int(round(com[0]))
                y0 = int(round(com[1]))
                x0 = int(round(com[2]))

                z_lo = max(0, z0 - radius)
                z_hi = min(D, z0 + radius + 1)
                y_lo = max(0, y0 - radius)
                y_hi = min(H, y0 + radius + 1)
                x_lo = max(0, x0 - radius)
                x_hi = min(W, x0 + radius + 1)

                zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
                dist_sq = (
                    (zz - com[0]) ** 2
                    + (yy - com[1]) ** 2
                    + (xx - com[2]) ** 2
                )
                gaussian = np.exp(-dist_sq / (2.0 * center_sigma ** 2))

                center_heatmap[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] = np.maximum(
                    center_heatmap[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi],
                    gaussian,
                )

                # Assign global ID for boundary computation
                combined_ids[mask] = next_global_id
                next_global_id += 1

        # Boundary: inter-instance boundaries from combined instance map
        boundary = _compute_boundary_map(combined_ids)

        # Foreground: union of all instance voxels
        foreground = (combined_ids > 0).astype(np.float32)

        return center_heatmap, offsets, boundary, foreground

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        """Sample a random 3D patch.

        If idx < len(self.crops), it is used as a crop index (for use with
        ClassBalancedSampler). Otherwise a random crop is chosen.

        Augmentations (if set via `self.transforms`) are applied here inside
        the DataLoader worker process, allowing parallel augmentation across
        workers and keeping the GPU fed continuously.

        Returns:
            raw: torch.Tensor [1, D, H, W] float32
            labels: torch.Tensor [n_classes, D, H, W] float32
            annotated_mask: torch.Tensor [n_classes] bool — which classes annotated
            spatial_mask: torch.Tensor [1, D, H, W] float32 — 1 inside GT crop
            crop_name: str, e.g. "jrc_hela-2/crop1"
            scale_factor: float — effective scale (1.0 = native resolution)

        When instance_mode=True, additionally returns:
            center_targets: torch.Tensor [1, D, H, W] float32 — center heatmap
            offset_targets: torch.Tensor [3, D, H, W] float32 — offset vectors
            boundary_targets: torch.Tensor [1, D, H, W] float32 — boundary map
            foreground: torch.Tensor [1, D, H, W] float32 — union foreground mask
        """
        if idx < len(self.crops):
            crop = self.crops[idx]
        else:
            crop = self.crops[self.rng.integers(len(self.crops))]

        # Pick a random scale factor for multi-scale training
        scale_factor = float(self.rng.choice(self.scale_factors))

        if self.instance_mode:
            raw, labels, annotated_mask, spatial_mask, scale_factor, instance_ids, instance_annotated = (
                self._extract_patch(crop, scale_factor=scale_factor)
            )
        else:
            raw, labels, annotated_mask, spatial_mask, scale_factor = (
                self._extract_patch(crop, scale_factor=scale_factor)
            )

        raw = torch.from_numpy(raw[np.newaxis])  # [1, D, H, W]
        labels = torch.from_numpy(labels)
        annotated_mask = torch.from_numpy(annotated_mask)
        spatial_mask = torch.from_numpy(spatial_mask[np.newaxis])  # [1, D, H, W]

        # Apply augmentations inside worker
        if self.transforms is not None:
            if self.instance_mode:
                # Concatenate binary labels + integer instance IDs for joint
                # spatial augmentation. First 48 channels are binary labels,
                # last 10 channels are integer instance IDs.
                instance_ids_t = torch.from_numpy(instance_ids.astype(np.float32))
                combined = torch.cat([labels, instance_ids_t], dim=0)
                raw, combined = self.transforms(
                    raw, combined, n_binary_channels=self.n_classes
                )
                labels = combined[:self.n_classes]
                instance_ids = combined[self.n_classes:].round().long().numpy()
            else:
                raw, labels = self.transforms(raw, labels)

        crop_name = f"{crop.dataset_name}/{crop.crop_id}"

        if self.instance_mode:
            # Compute class-agnostic instance targets from (augmented) instance IDs
            center_heatmap, offsets, boundary, foreground = (
                self._compute_instance_targets(instance_ids)
            )
            center_t = torch.from_numpy(center_heatmap[np.newaxis])   # [1, D, H, W]
            offsets_t = torch.from_numpy(offsets)                      # [3, D, H, W]
            boundary_t = torch.from_numpy(boundary[np.newaxis])        # [1, D, H, W]
            foreground_t = torch.from_numpy(foreground[np.newaxis])    # [1, D, H, W]
            return (
                raw, labels, annotated_mask, spatial_mask, crop_name,
                scale_factor,
                center_t, offsets_t, boundary_t, foreground_t,
            )

        return raw, labels, annotated_mask, spatial_mask, crop_name, scale_factor

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_crop_class_matrix(self) -> np.ndarray:
        """Build a [n_crops x n_classes] boolean matrix of class presence."""
        matrix = np.zeros((len(self.crops), self.n_classes), dtype=bool)
        for i, crop in enumerate(self.crops):
            for cls_name in crop.annotated_classes:
                if cls_name in self.class_to_idx:
                    matrix[i, self.class_to_idx[cls_name]] = True
        return matrix

    def summary(self) -> str:
        """Return a summary string of the dataset."""
        n_datasets = len(set(c.dataset_name for c in self.crops))
        class_matrix = self.get_crop_class_matrix()
        class_counts = class_matrix.sum(axis=0)
        lines = [
            f"CellMapDataset3D Summary",
            f"  Datasets: {n_datasets}",
            f"  Crops: {len(self.crops)}",
            f"  Target classes: {self.n_classes}",
            f"  Resolution: {self.target_resolution}nm",
            f"  Patch size: {tuple(self.patch_size)}",
            f"",
            f"  Class presence (crops):",
        ]
        for i, cls in enumerate(self.target_classes):
            lines.append(f"    {cls:20s}: {int(class_counts[i]):4d} crops")
        return "\n".join(lines)
