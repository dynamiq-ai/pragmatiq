"""Device and precision resolution for pragmatiq.

:func:`resolve_device` is the single place where ``"auto"`` becomes a concrete
device and :func:`inference_context` the single place that decides how a forward
pass runs at inference time. Importable without touching torch until called.

pragmatiq is GPU-first and CPU-complete: every path runs on CPU in fp32, and on
CUDA the same code runs under bf16 autocast, which is what routes attention
through the flash varlen kernel. ``PRAGMATIQ_DEVICE`` pins the device that
``"auto"`` resolves to; ``PRAGMATIQ_INFERENCE_PRECISION`` pins the inference
precision that ``"auto"`` resolves to (``bf16`` or ``fp32``).
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

PRECISIONS = ("auto", "bf16", "fp32")


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to a concrete device; pass other values through.

    ``"auto"`` honours ``PRAGMATIQ_DEVICE`` when set (``cpu``, ``cuda``,
    ``cuda:1``), otherwise picks CUDA when available and CPU otherwise. An
    explicit ``"cpu"`` / ``"cuda"`` is returned as given.
    """
    if device == "auto":
        pinned = os.environ.get("PRAGMATIQ_DEVICE", "").strip()
        if pinned:
            return pinned
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_precision(precision: str, device: str) -> str:
    """Resolve an inference precision to ``"bf16"`` or ``"fp32"`` for ``device``.

    ``"auto"`` reads ``PRAGMATIQ_INFERENCE_PRECISION`` when set, else picks
    ``bf16`` on CUDA and ``fp32`` elsewhere. ``bf16`` on a CPU device is
    downgraded to ``fp32`` (CPU inference stays fp32 so results are byte-stable).
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "auto":
        pinned = os.environ.get("PRAGMATIQ_INFERENCE_PRECISION", "").strip().lower()
        precision = pinned if pinned in ("bf16", "fp32") else "auto"
    is_cuda = str(device).startswith("cuda")
    if precision == "auto":
        return "bf16" if is_cuda else "fp32"
    if precision == "bf16" and not is_cuda:
        return "fp32"
    return precision


def inference_context(device: str | Any, precision: str = "auto") -> contextlib.AbstractContextManager:
    """Context for an inference forward: ``inference_mode`` plus bf16 autocast on CUDA.

    Use it around every model call that does not need gradients. On CUDA with
    ``bf16`` the attention runs on the flash varlen kernel (bf16 is required
    for it) and activations halve; on CPU the context is fp32 and the numerics
    are unchanged from a plain ``no_grad`` forward.
    """
    import torch

    stack = contextlib.ExitStack()
    stack.enter_context(torch.inference_mode())
    if resolve_precision(precision, str(device)) == "bf16":
        stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
    return stack
