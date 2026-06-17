"""Synthetic banking-world generator (Phase 1).

Public surface: ``WorldConfig`` + ``generate`` (also exposed via
``pragmatiq.api.synthesize``).
"""

from .config import WorldConfig
from .generate import generate
from .world import World

__all__ = ["World", "WorldConfig", "generate"]
