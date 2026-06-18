"""Optimizers: Muon for 2-D hidden weights, AdamW for everything else.

Muon (MomentUm Orthogonalized by Newton-schulz) is vendored from Keller Jordan's
reference implementation (https://github.com/KellerJordan/Muon, MIT) with light
edits; it orthogonalizes each 2-D weight's momentum via a quintic Newton–Schulz
iteration before the step. 2-D hidden weights use Muon
while embeddings, norms and biases use AdamW.

``build_optimizers`` returns ``(muon, adamw)`` plus a combined cosine schedule
with warmup over both.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton–Schulz orthogonalization (Keller Jordan, MIT).

    Returns an approximately orthogonal matrix with the same shape as ``G``.
    Runs in bf16 for speed on GPU, except under deterministic mode (or on CPU)
    where it runs in fp32 — so a deterministic fp32 run's weight update is genuinely
    fp32 end to end, honoring the bit-exact guarantee.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    fp32 = (not G.is_cuda) or torch.are_deterministic_algorithms_enabled()
    work_dtype = torch.float32 if fp32 else torch.bfloat16
    X = G.to(work_dtype)
    if G.size(0) > G.size(1):
        X = X.T
    norm = X.norm() + 1e-7
    X = X / norm
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2-D hidden weights (vendored, Keller Jordan, MIT).

    Args:
        params: the 2-D weights to optimize.
        lr: learning rate (a GUESS default of 3e-3 per SPEC, set via config).
        momentum: heavy-ball momentum coefficient.
        nesterov: use Nesterov-style momentum.
        ns_steps: Newton–Schulz iteration count.
        weight_decay: decoupled weight decay.
    """

    def __init__(self, params: Iterable[nn.Parameter], lr: float = 3e-3, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # noqa: D401
        """Perform a single optimization step."""
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError("Muon only supports 2-D parameters; route others to AdamW")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                ortho = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                # Keller-Jordan rescale: keeps a tall matrix's update spectral
                # norm ~1 (only m>=n matters); does NOT make per-element RMS
                # shape-invariant (that is Moonshot's 0.2*sqrt(max(rows,cols))).
                scale = max(1.0, g.size(0) / g.size(1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)
        return loss


def split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition params into (muon_2d_hidden, adamw_rest).

    2-D hidden weights (Linear/attention/MLP matrices) go to Muon; embeddings,
    LayerNorm weights/biases, and any 1-D tensor go to AdamW. LoRA factor
    matrices (``lora_a``/``lora_b``) are 2-D but deliberately low-rank, so they
    are routed to AdamW too: Newton-Schulz orthogonalization would destroy the
    rank-r structure, so LoRA adapters stay on AdamW during continued pretraining.
    """
    muon, adamw = [], []
    embed_ids: set[int] = set()
    lora_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            embed_ids.add(id(module.weight))
        if type(module).__name__ == "LoRALinear":  # avoid a models<-training import cycle
            for p in module.parameters(recurse=False):
                lora_ids.add(id(p))
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and id(p) not in embed_ids and id(p) not in lora_ids:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def build_optimizers(
    model: nn.Module,
    lr_muon: float = 3e-3,  # GUESS (SPEC)
    lr_adamw: float = 3e-4,  # GUESS (SPEC)
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    momentum: float = 0.95,
) -> tuple[list[torch.optim.Optimizer], list[str]]:
    """Build ``[Muon, AdamW]`` for a model; returns optimizers + their names."""
    muon_params, adamw_params = split_parameters(model)
    opts: list[torch.optim.Optimizer] = []
    names: list[str] = []
    if muon_params:
        opts.append(Muon(muon_params, lr=lr_muon, momentum=momentum, weight_decay=weight_decay))
        names.append("muon")
    if adamw_params:
        # No weight decay on 1-D params (LayerNorm gains/biases): the Muon and
        # Moonshot reference setups exclude them, and decaying normalization
        # gains degrades training. The embedding table (2-D) keeps weight decay.
        decay = [p for p in adamw_params if p.ndim >= 2]
        nodecay = [p for p in adamw_params if p.ndim < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        opts.append(torch.optim.AdamW(groups, lr=lr_adamw, betas=betas))
        names.append("adamw")
    return opts, names


def cosine_warmup_factor(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> float:
    """LR multiplier: linear warmup then cosine decay to ``min_ratio`` of peak.

    ``step`` is the 0-based step the factor is applied to *before* its optimizer
    update. Warmup is 1-indexed (``(step+1)/warmup``) so step 0 gets a small
    positive LR rather than 0 or the full base LR.
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


class WarmupCosine:
    """Applies :func:`cosine_warmup_factor` to a list of optimizers' base LRs."""

    def __init__(self, optimizers: list[torch.optim.Optimizer], warmup_steps: int,
                 total_steps: int, min_ratio: float = 0.1) -> None:
        self.optimizers = optimizers
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_ratio = min_ratio
        self.base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
        self._step = 0

    def apply(self, step: int) -> float:
        """Set every optimizer's LR to ``base * factor(step)`` and return the factor.

        ``step`` is the global training step. Driving the schedule off the
        trainer's step counter (called *before* the optimizer update) makes the
        LR a pure function of ``step``: the first update is warmed (not full
        base LR), skipped/NaN steps cannot desync the curve, and resume is exact
        because ``step`` is checkpointed.
        """
        self._step = step
        factor = cosine_warmup_factor(step, self.warmup_steps, self.total_steps, self.min_ratio)
        for opt, bases in zip(self.optimizers, self.base_lrs):
            for g, base in zip(opt.param_groups, bases):
                g["lr"] = base * factor
        return factor

    def step(self) -> None:
        """Advance one internal step (kept for standalone use; trainer uses ``apply``)."""
        self.apply(self._step)
        self._step += 1

    @property
    def last_factor(self) -> float:
        return cosine_warmup_factor(self._step, self.warmup_steps, self.total_steps, self.min_ratio)

    def state_dict(self) -> dict[str, Any]:
        return {"step": self._step, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_ratio": self.min_ratio}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step = state["step"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.min_ratio = state["min_ratio"]
