"""SAM3 + LoRA + CellMap head assembly.

Builds the full model for CellMap fine-tuning:
1. Load pre-trained SAM3 (image or video mode)
2. Freeze all parameters
3. Apply LoRA to vision encoder attention layers
4. Attach CellMapSegmentationHead (fully trainable)
5. SAM3's native DETR + mask head stays intact for Mode B (prompting)

Mode A (primary): pixel_features -> CellMapSegmentationHead -> [B, 48, H, W]
Mode B (future):  text/point prompts -> SAM3 DETR detector -> instance masks

SAM3 architecture (from inspection):
  Sam3Image
    ├── backbone: SAM3VLBackbone
    │   ├── vision_backbone: Sam3DualViTDetNeck (ViT + FPN neck)
    │   └── language_backbone: VETextEncoder
    ├── transformer: TransformerWrapper (DETR decoder)
    ├── geometry_encoder: SequenceGeometryEncoder
    ├── segmentation_head: UniversalSegmentationHead
    │   ├── pixel_decoder: PixelDecoder (FPN upsampling)
    │   ├── mask_predictor: MaskPredictor
    │   ├── semantic_seg_head: Conv2d
    │   └── instance_seg_head: Conv2d
    └── dot_prod_scoring: DotProductScoring

Feature extraction path (Mode A):
  [B, 3, 1008, 1008] input
  → backbone.forward_image() → backbone_fpn: 3 levels of [B, 256, H, W]
    level 0: [B, 256, 288, 288]  (stride ~3.5)
    level 1: [B, 256, 144, 144]  (stride ~7)
    level 2: [B, 256, 72, 72]    (stride 14)
  → segmentation_head.pixel_decoder(backbone_fpn) → [B, 256, 288, 288]
  → CellMapSegmentationHead → [B, 48, H, W]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cellmap_head import CellMapSegmentationHead
from .lora import apply_lora, freeze_except_lora, count_parameters

logger = logging.getLogger(__name__)


class SAM3CellMapModel(nn.Module):
    """SAM3 with LoRA and CellMap segmentation head.

    Wraps SAM3's image model and adds a dense 48-class head on top of
    the pixel decoder features. LoRA adapts the vision encoder to EM
    imagery while keeping 98.5% of parameters frozen.

    Args:
        sam3_model: Pre-built SAM3 image model (Sam3Image instance).
        cellmap_head: CellMapSegmentationHead instance.
    """

    def __init__(
        self,
        sam3_model: nn.Module,
        cellmap_head: CellMapSegmentationHead,
    ):
        super().__init__()
        self.sam3 = sam3_model
        self.cellmap_head = cellmap_head

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract pixel decoder features from SAM3's backbone.

        Path: images → ViT backbone → FPN neck → multi-scale features
              → PixelDecoder → high-res pixel embeddings

        Args:
            images: [B, 3, 1008, 1008] RGB images.

        Returns:
            pixel_features: [B, 256, 288, 288] pixel decoder output.
        """
        # backbone.forward_image returns dict with:
        #   "backbone_fpn": list of 3 feature maps [B, 256, 288/144/72, ...]
        #   "vision_features": [B, 256, 72, 72] (lowest-res)
        #   "vision_pos_enc": list of 3 positional encodings
        backbone_out = self.sam3.backbone.forward_image(images)
        backbone_fpn = backbone_out["backbone_fpn"]

        # PixelDecoder: FPN upsampling to highest resolution
        pixel_features = self.sam3.segmentation_head.pixel_decoder(backbone_fpn)

        return pixel_features

    def forward(
        self,
        images: torch.Tensor,
        target_size: Optional[tuple] = None,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for Mode A (dense 48-class prediction).

        Args:
            images: [B, 3, 1008, 1008] RGB images (grayscale repeated 3x).
            target_size: Optional (H, W) for output resolution.
            scale_factor: Optional [B] tensor — known scale for FiLM conditioning.

        Returns:
            dict with "fine" [B, 48, H, W], optionally "medium", "coarse".
        """
        pixel_features = self.forward_features(images)
        return self.cellmap_head(
            pixel_features, target_size=target_size, scale_factor=scale_factor
        )


def build_cellmap_model(
    sam3_checkpoint: Optional[str] = None,
    lora_rank: int = 8,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.05,
    lora_targets: Optional[List[str]] = None,
    n_fine: int = 48,
    n_medium: int = 17,
    n_coarse: int = 7,
    use_auxiliary: bool = True,
    use_scale_conditioning: bool = False,
    device: str = "cuda",
) -> SAM3CellMapModel:
    """Build SAM3 model configured for CellMap segmentation.

    Args:
        sam3_checkpoint: Path to checkpoint file. If None and load_from_HF
            is True, downloads from HuggingFace.
        lora_rank: LoRA adapter rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.
        lora_targets: Module name patterns for LoRA. Defaults to
            vision encoder attention projections.
        n_fine: Number of fine-level classes.
        n_medium: Number of medium-level classes.
        n_coarse: Number of coarse-level classes.
        use_auxiliary: Whether to create auxiliary heads.
        device: Device to load model on.

    Returns:
        SAM3CellMapModel with LoRA applied and cellmap head attached.
    """
    if lora_targets is None:
        lora_targets = ["attn.qkv", "attn.proj"]

    # 1. Load pre-trained SAM3
    logger.info("Loading SAM3 model...")
    try:
        from sam3.model_builder import build_sam3_image_model
    except ImportError:
        raise ImportError(
            "SAM3 not installed. Install with: "
            "pip install -e '.[train]' from the sam3 repo"
        )

    sam3_model = build_sam3_image_model(
        device=device,
        eval_mode=False,  # need training mode for LoRA gradients
        checkpoint_path=sam3_checkpoint,
        load_from_HF=(sam3_checkpoint is None),
        enable_segmentation=True,
    )

    # 2. Freeze all parameters
    for param in sam3_model.parameters():
        param.requires_grad_(False)
    logger.info("Froze all SAM3 parameters")

    # 3. Apply LoRA to vision encoder
    replaced = apply_lora(
        sam3_model,
        target_modules=lora_targets,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    logger.info(
        f"Applied LoRA (rank={lora_rank}) to {len(replaced)} modules, "
        f"adding {sum(replaced.values()):,} parameters"
    )

    # 4. Create CellMap segmentation head
    # Pixel decoder output is [B, 256, 288, 288] for 1008x1008 input
    cellmap_head = CellMapSegmentationHead(
        in_channels=256,  # SAM3 pixel decoder output dim
        hidden_channels=128,
        n_fine=n_fine,
        n_medium=n_medium,
        n_coarse=n_coarse,
        use_auxiliary=use_auxiliary,
        upsample_factor=1,  # pixel decoder already at high res (288x288)
        use_scale_conditioning=use_scale_conditioning,
    ).to(device)

    # 5. Assemble
    model = SAM3CellMapModel(sam3_model, cellmap_head)

    # Log parameter counts
    counts = count_parameters(model)
    logger.info(
        f"Model: {counts['total']:,} total, "
        f"{counts['trainable']:,} trainable ({100*counts['trainable']/counts['total']:.1f}%), "
        f"{counts['frozen']:,} frozen"
    )

    return model
