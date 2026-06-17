"""Hands-off training sizing from the data + device (Phase 5).

A user pointing pragmatiq at 1M–26M tokenized records should not have to hand-tune the
batch and schedule. :func:`autoconfigure` reads the shard index (user count, token
distribution) and the target device, then picks:

- ``token_budget`` — the per-forward token cap, sized to the device's memory and the
  model width so a single micro-batch fits without OOM;
- ``grad_accum_steps`` — micro-batches per optimizer step, set so the *effective* batch
  (``token_budget × grad_accum × world_size``) reaches a stable target regardless of how
  small each device's slice must be;
- ``max_steps`` / ``warmup_steps`` — derived from how many optimizer steps one (or
  ``epochs``) passes over the data take at that effective batch.

The numbers are deliberately conservative GUESSes with headroom; every choice is returned
in ``rationale`` so a user can see and override it. The caps themselves stay at the paper
defaults (set on the tokenizer), so this module only sizes the optimization loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Per-forward tokens a single GiB of accelerator memory comfortably holds, by model
# size (activations scale with width × depth). Conservative so the auto budget leaves
# headroom for the optimizer state and fragmentation; the OOM-retry path is the backstop.
_TOKENS_PER_GIB: dict[str, int] = {"nano": 4096, "small": 1400, "medium": 480, "large": 200}
_CPU_TOKEN_BUDGET = 4096  # correctness-over-speed default when no accelerator is present
_TOKEN_BUDGET_BOUNDS = (2048, 131_072)
# A stable optimizer-step batch for MLM at this scale (paper trained at ~16–32 H100s of
# packed tokens); reached via accumulation when one device cannot hold it.
_DEFAULT_EFFECTIVE_TOKENS = 262_144


@dataclass
class AutoTrainPlan:
    """A sized training plan plus the reasoning behind each number."""

    token_budget: int
    grad_accum_steps: int
    max_steps: int
    warmup_steps: int
    effective_tokens: int
    rationale: dict[str, Any] = field(default_factory=dict)

    def as_overrides(self) -> dict[str, int]:
        """The subset that overrides ``TrainConfig`` fields."""
        return {"token_budget": self.token_budget, "grad_accum_steps": self.grad_accum_steps,
                "max_steps": self.max_steps, "warmup_steps": self.warmup_steps}


def _device_memory_gib(device: str) -> float | None:
    """Total memory of the target accelerator in GiB, or None on CPU."""
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return torch.cuda.get_device_properties(index).total_memory / (1024**3)
    except Exception:
        return None


def token_budget_for(device: str, model_size: str, mem_gib: float | None = None) -> int:
    """Pick a per-forward token budget for ``device`` and ``model_size``.

    On CPU returns a small correctness-first budget; on CUDA scales the per-GiB capacity
    by usable memory (≈80% of total, leaving room for optimizer state) and clamps to a
    sane range.
    """
    if not device.startswith("cuda"):
        return _CPU_TOKEN_BUDGET
    gib = mem_gib if mem_gib is not None else _device_memory_gib(device)
    if not gib:
        return _TOKEN_BUDGET_BOUNDS[0]  # CUDA requested but unreadable → conservative floor
    per_gib = _TOKENS_PER_GIB.get(model_size, _TOKENS_PER_GIB["small"])
    raw = int(per_gib * gib * 0.8)
    lo, hi = _TOKEN_BUDGET_BOUNDS
    return max(lo, min(hi, (raw // 256) * 256))  # round to a multiple of 256


def autoconfigure(
    shard_dir: str | Path,
    *,
    device: str = "cpu",
    world_size: int = 1,
    model_size: str = "small",
    epochs: float = 1.0,
    target_effective_tokens: int = _DEFAULT_EFFECTIVE_TOKENS,
) -> AutoTrainPlan:
    """Size a training plan for the shards at ``shard_dir`` (see module docstring)."""
    from ..data.sharding import UserIndex

    idx = UserIndex(Path(shard_dir))
    try:
        n_users = len(idx)
        total_tokens = int(idx.n_tokens.sum() + idx.n_prof_tokens.sum())
    finally:
        idx.close()
    if n_users == 0:
        raise ValueError(f"no users indexed under {shard_dir}; nothing to size a run for")

    world_size = max(1, world_size)
    token_budget = token_budget_for(device, model_size)
    # Accumulate enough micro-batches (across the whole world) to reach the target batch.
    per_step_tokens = token_budget * world_size
    grad_accum_steps = max(1, math.ceil(target_effective_tokens / per_step_tokens))
    effective_tokens = token_budget * grad_accum_steps * world_size
    steps_per_epoch = max(1, total_tokens // effective_tokens)
    max_steps = max(1, round(epochs * steps_per_epoch))
    warmup_steps = max(1, round(0.05 * max_steps))  # 5% warmup is a robust default

    rationale = {
        "n_users": n_users, "total_tokens": total_tokens, "device": device,
        "world_size": world_size, "model_size": model_size,
        "mem_gib": round(_device_memory_gib(device) or 0.0, 1),
        "steps_per_epoch": steps_per_epoch, "epochs": epochs,
        "note": (f"effective batch ≈ {effective_tokens:,} tokens "
                 f"({token_budget:,} budget × {grad_accum_steps} accum × {world_size} ranks); "
                 f"{max_steps:,} steps ≈ {epochs} epoch(s) over {total_tokens:,} tokens"),
    }
    return AutoTrainPlan(token_budget=token_budget, grad_accum_steps=grad_accum_steps,
                         max_steps=max_steps, warmup_steps=warmup_steps,
                         effective_tokens=effective_tokens, rationale=rationale)
