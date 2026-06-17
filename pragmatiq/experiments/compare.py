"""Run comparison (Phase 5: ``pragmatiq runs list|compare``).

Pure functions over the ``runs/`` directory so the CLI stays a thin wrapper
(global rule 1). ``list_runs`` lives in :mod:`pragmatiq.experiments.run`;
:func:`compare_runs` aligns several runs' last metrics side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .run import list_runs


def compare_runs(names: list[str], runs_root: str | Path = "runs") -> list[dict[str, Any]]:
    """Return each named run's summary (last step/loss/metrics), in input order.

    Missing runs are returned as ``{"name": name, "missing": True}`` rather than
    omitted, so callers can see which requested runs were not found.
    """
    by_name = {r["name"]: r for r in list_runs(runs_root)}
    return [by_name.get(n, {"name": n, "missing": True}) for n in names]
