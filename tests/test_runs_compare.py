"""compare_runs alignment semantics (runs list|compare)."""

from __future__ import annotations

import pragmatiq.experiments.compare as compare_mod
from pragmatiq.experiments.compare import compare_runs


def test_compare_runs_preserves_input_order_and_flags_missing(monkeypatch) -> None:
    summaries = [
        {"name": "alpha", "step": 100, "loss": 1.5},
        {"name": "beta", "step": 200, "loss": 0.9},
    ]
    monkeypatch.setattr(compare_mod, "list_runs", lambda root: summaries)

    out = compare_runs(["beta", "alpha", "ghost"], runs_root="ignored")

    assert [r["name"] for r in out] == ["beta", "alpha", "ghost"]  # input order, not disk order
    assert out[0]["loss"] == 0.9 and out[1]["loss"] == 1.5  # aligned to the right run by name
    assert out[2] == {"name": "ghost", "missing": True}  # unknown run surfaced, not dropped
