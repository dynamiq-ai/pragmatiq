"""GPU validation harness for pragmatiq, split by concern.

``scripts/validate_gpu.py`` is the thin entry point the pod runs; the leg
bodies, sweeps, serving measurement, flash check and report writer live here so
each piece is testable on a CPU. ``SCRIPT`` is the path the leg subprocesses
re-invoke (Lightning Fabric re-launches it per rank); ``REPO_ROOT`` the repo.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_gpu.py"
