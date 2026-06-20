"""Configuration loading utilities for pragmatiq.

Provides :func:`load_yaml`, a thin wrapper around OmegaConf that loads a
YAML config file into a plain ``dict`` with resolved interpolations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file via OmegaConf and return a plain ``dict``.

    Resolves OmegaConf variable interpolations before returning so callers
    receive concrete values.  The top-level mapping constraint is enforced
    here so config files that accidentally wrap their content in a list or
    scalar surface a clear error immediately.

    Args:
        path: filesystem path to the YAML config file.

    Returns:
        A plain ``dict[str, Any]`` with string keys.

    Raises:
        ValueError: if the top-level YAML value is not a mapping.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    out = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(out, dict):
        raise ValueError(f"config {path} must contain a top-level mapping, got {type(out).__name__}")
    return {str(k): v for k, v in out.items()}
