"""Serving + batch-embed benchmarks.

``benchmark_batch_embed`` measures local batch-embedding throughput and a cost
table; ``write_results`` renders ``deploy/benchmarks/RESULTS.md``. The Triton
``perf_analyzer`` wrapper (latency percentiles vs concurrency) is emitted as a
runnable command when a live Triton endpoint is configured — it needs the GPU
serving stack, so it is not executed in CI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from ..data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
from ..models.pragmatiq import PragmaModel


@torch.no_grad()
def benchmark_batch_embed(
    model: PragmaModel, shard_dir: str | Path, device: str = "cpu",
    token_budget: int = 16_384, max_users: int | None = None,
) -> dict[str, Any]:
    """Measure batch-embedding throughput (users/sec, tokens/sec)."""
    model = model.to(device).eval()
    ds = ShardDataset(shard_dir)
    sampler = DynamicBatchSampler(ds.index, token_budget=token_budget, shuffle=False)
    sampler.set_epoch(0)
    loader = ShardDataLoader(ds, sampler)
    n_users = n_tokens = 0
    # CUDA kernels launch asynchronously, so the wall clock must bracket a
    # synchronize on each side — otherwise the loop returns before the GPU has
    # finished and the measured throughput is overstated.
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for batch in loader:
        batch = batch.to(device)
        model.embed_users(batch)
        n_users += batch.n_users
        n_tokens += batch.n_tokens
        if max_users is not None and n_users >= max_users:
            break
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-6)
    ds.close()
    return {
        "device": device, "n_users": n_users, "n_tokens": n_tokens,
        "elapsed_sec": round(elapsed, 3),
        "users_per_sec": round(n_users / elapsed, 1),
        "tokens_per_sec": round(n_tokens / elapsed, 1),
        "usd_per_million_users": _cost_estimate(n_users / elapsed, device),
    }


def _cost_estimate(users_per_sec: float, device: str) -> float:
    # Rough on-demand cloud rates (USD/hr): A100 ~$1.10, generic CPU box ~$0.10.
    rate = 1.10 if device.startswith("cuda") else 0.10
    if users_per_sec <= 0:
        return float("nan")
    return round(rate / 3600.0 / users_per_sec * 1_000_000, 2)


def perf_analyzer_command(model_name: str = "pragmatiq_embedder", url: str = "localhost:8001",
                          concurrencies: tuple[int, ...] = (1, 4, 16, 64)) -> str:
    """The perf_analyzer command to sweep latency (p50/p95/p99) vs concurrency.

    Measured latency percentiles require a *live* Triton endpoint, which is not
    available in CI or this library context, so ``pragmatiq benchmark`` emits
    this runnable command (for ``deploy/docker-compose up``) alongside the
    batch-embed throughput + cost table it can measure locally. Run the command
    against a serving endpoint to fill in the p50/p95/p99 vs concurrency rows.

    ``records_json`` is a STRING tensor perf_analyzer cannot auto-generate, so the
    command references ``--input-data records.json``; create that perf_analyzer
    input file with a sample batch, e.g.
    ``{"data":[{"records_json":["[{\\"user_id\\":\\"u0\\",\\"events\\":[]}]"]}]}``.
    """
    rng = f"{min(concurrencies)}:{max(concurrencies)}"
    return (f"perf_analyzer -m {model_name} -u {url} -i grpc "
            f"--concurrency-range {rng} --percentile=99 --measurement-interval 5000 "
            f"--input-data records.json --shape records_json:1")


def write_results(stats: dict[str, Any], out_path: str | Path = "deploy/benchmarks/RESULTS.md") -> Path:
    """Render a benchmark results markdown file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# pragmatiq serving benchmarks", "",
        "> pragmatiq is an independent implementation inspired by the PRAGMA paper "
        "(arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.", "",
        "## Batch embedding throughput", "",
        "| metric | value |", "|---|---|",
        f"| device | {stats.get('device')} |",
        f"| users/sec | {stats.get('users_per_sec')} |",
        f"| tokens/sec | {stats.get('tokens_per_sec')} |",
        f"| USD / 1M users | {stats.get('usd_per_million_users')} |", "",
        "## Triton latency vs concurrency", "",
        "Run against a live Triton endpoint:", "",
        "```", perf_analyzer_command(), "```", "",
        "perf_analyzer reports p50/p95/p99 latency and throughput per concurrency level.",
    ]
    out.write_text("\n".join(lines))
    return out
