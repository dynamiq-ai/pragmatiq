"""Device resolution helpers for pragmatiq.

:func:`resolve_device` is the single place where ``"auto"`` becomes a concrete
device.  Importable without touching torch or any optional dependency.
"""

from __future__ import annotations


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to CUDA when available, else CPU; pass other values through.

    CPU is always a correct target (global rule 5); ``"auto"`` uses the GPU
    purely as an acceleration when one is available.  An explicit ``"cpu"``
    or ``"cuda"`` is honored as given.

    Args:
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        The resolved device string.
    """
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


