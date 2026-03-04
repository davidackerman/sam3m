"""LoRA (Low-Rank Adaptation) for SAM3 fine-tuning.

Applies trainable low-rank matrices to frozen linear layers, enabling
efficient domain adaptation with ~1.5% trainable parameters.

Design:
- LoRALinear wraps a frozen nn.Linear: output = frozen(x) + B(A(x)) * scale
- A initialized with Kaiming, B with zeros -> starts as identity
- apply_lora() walks a model and replaces matching modules
- lora_state_dict() extracts only LoRA params for lightweight checkpointing
- merge_lora() folds LoRA weights into base weights for inference speed
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Adds trainable low-rank matrices A, B to a frozen linear layer:
        output = frozen_linear(x) + B(A(dropout(x))) * (alpha / rank)

    Args:
        original_linear: The nn.Linear to wrap (will be frozen).
        rank: LoRA rank (4-16 typical).
        alpha: Scaling factor (typically = rank).
        dropout: Dropout on the LoRA path.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        device = original_linear.weight.device

        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device)
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features


def apply_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.05,
) -> Dict[str, int]:
    """Apply LoRA adapters to matching linear layers in a model.

    Walks the module tree and replaces nn.Linear modules whose full
    name contains any of the target_module patterns.

    Args:
        model: The model to modify (in-place).
        target_modules: List of substrings to match against module names.
            E.g., ["attn.qkv", "attn.proj"] matches all attention projections.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout rate on LoRA path.

    Returns:
        Dict mapping replaced module name -> number of LoRA params added.
    """
    replaced = {}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Check if this module name matches any target pattern
        if not any(pattern in name for pattern in target_modules):
            continue

        # Navigate to parent module to replace the child
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]
        original = getattr(parent, child_name)

        lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)

        n_params = (
            lora_layer.lora_A.weight.numel()
            + lora_layer.lora_B.weight.numel()
        )
        replaced[name] = n_params

    return replaced


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from a model.

    Returns a state dict containing only the lora_A and lora_B weights,
    suitable for lightweight checkpointing.
    """
    lora_params = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params[name] = param.data
    return lora_params


def merge_lora(model: nn.Module) -> None:
    """Merge LoRA weights into base linear layers for inference speed.

    After merging, the model has no LoRA overhead — the adapted weights
    are baked into the original nn.Linear layers. This is irreversible.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        # Compute merged weight: W' = W + B @ A * scaling
        with torch.no_grad():
            merged_weight = (
                module.original.weight.data
                + module.scaling
                * module.lora_B.weight.data @ module.lora_A.weight.data
            )
            module.original.weight.data = merged_weight

        # Replace LoRALinear with the merged original Linear
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.original)


def freeze_except_lora(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapters."""
    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
