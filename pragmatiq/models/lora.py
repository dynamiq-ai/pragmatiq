"""LoRA adapters.

``inject_lora(model, rank, alpha, targets)`` swaps targeted ``nn.Linear`` layers
for :class:`LoRALinear`, freezing the base weight and training only the low-rank
update. ``merge_lora(model)`` folds the update back into the base weights and
restores plain ``nn.Linear`` layers for export.

Targets are matched by attribute name fragments (default: the attention ``qkv``
and ``out`` projections and the feed-forward ``net`` linears).
"""

from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_TARGETS: tuple[str, ...] = ("qkv", "out", "net")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` update ``B @ A``."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        # Match the base layer's device/dtype so injecting LoRA into a model that
        # already lives on CUDA does not leave the adapters on CPU.
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features, device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dt))
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))  # B stays zero → identity at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.drop(x) @ self.lora_a.t() @ self.lora_b.t()
        return self.base(x) + self.scaling * update

    def merged_linear(self) -> nn.Linear:
        """Return a plain ``nn.Linear`` with the LoRA update folded in."""
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.scaling * (self.lora_b @ self.lora_a))
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def _iter_named_linears(model: nn.Module):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                yield module, child_name, child, f"{name}.{child_name}" if name else child_name


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    dropout: float = 0.0,
) -> int:
    """Replace targeted ``nn.Linear`` layers in-place with :class:`LoRALinear`.

    Returns the number of layers adapted. A layer is targeted if any string in
    ``targets`` appears in its qualified name.
    """
    n = 0
    for parent, child_name, child, qual in list(_iter_named_linears(model)):
        if any(t in qual for t in targets):
            setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
            n += 1
    return n


def merge_lora(model: nn.Module) -> int:
    """Fold every :class:`LoRALinear` back into a plain ``nn.Linear`` in-place.

    Returns the number of layers merged.
    """
    n = 0
    for _name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged_linear())
                n += 1
    return n


def lora_parameters(model: nn.Module):
    """Yield only the trainable LoRA parameters (for the optimizer)."""
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze everything except LoRA A/B parameters."""
    lora_ids = {id(p) for p in lora_parameters(model)}
    for p in model.parameters():
        p.requires_grad_(id(p) in lora_ids)
