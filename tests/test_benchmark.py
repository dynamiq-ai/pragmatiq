"""Serving-benchmark reporting helpers (Phase 7)."""

from __future__ import annotations

import math

from pragmatiq.inference.benchmark import (
    _cost_estimate,
    perf_analyzer_command,
    write_results,
)


def test_perf_analyzer_command_sweeps_concurrency_range() -> None:
    cmd = perf_analyzer_command(concurrencies=(2, 8, 32))
    assert "perf_analyzer -m pragmatiq_embedder" in cmd
    assert "--concurrency-range 2:32" in cmd  # min:max of the sweep
    assert "--input-data records.json" in cmd


def test_cost_estimate_cpu_vs_cuda_and_nan_guard() -> None:
    cpu = _cost_estimate(100.0, "cpu")
    cuda = _cost_estimate(100.0, "cuda")
    assert cuda > cpu > 0  # GPU hourly rate is higher, so $/1M users is higher
    assert math.isnan(_cost_estimate(0.0, "cpu"))  # no throughput → undefined cost


def test_write_results_renders_table_and_attribution(tmp_path) -> None:
    out = write_results(
        {"device": "cpu", "users_per_sec": 12.3, "tokens_per_sec": 45678,
         "usd_per_million_users": 2.26},
        tmp_path / "RESULTS.md",
    )
    text = out.read_text()
    assert "| users/sec | 12.3 |" in text
    assert "not affiliated with or endorsed by Revolut" in text  # required attribution
    assert "perf_analyzer -m pragmatiq_embedder" in text
    assert "None" not in text  # every stat key was rendered
