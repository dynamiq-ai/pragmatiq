"""Environment and device helpers for pragmatiq.

Provides device resolution (:func:`resolve_device`) and opt-in feature
flags that read environment variables (:func:`telemetry_enabled`,
:func:`offline_mode`).  All functions are pure (no side effects) and
importable without touching torch or any optional dependency.
"""

from __future__ import annotations

import os


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


def telemetry_enabled() -> bool:
    """Return ``True`` only if the ``PRAGMATIQ_TELEMETRY`` env var is set truthy.

    Default is ``False`` (opt-in).  W7 will wire the actual telemetry calls;
    this function is the canonical gate check so all callers agree on the
    env-var name and truthy test.
    """
    val = os.environ.get("PRAGMATIQ_TELEMETRY", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def offline_mode() -> bool:
    """Return ``True`` if the ``PRAGMATIQ_OFFLINE`` env var is set truthy.

    When offline mode is active, components that would fetch remote resources
    should skip those requests or raise a clear error.
    """
    val = os.environ.get("PRAGMATIQ_OFFLINE", "").strip().lower()
    return val in ("1", "true", "yes", "on")
