"""Multi-GPU (DDP) × gradient-accumulation correctness test (audit finding).

A real 2-rank gloo (CPU DDP) run that pins down the audit's CRITICAL bug: under
``world_size > 1`` AND ``grad_accum_steps > 1`` the per-micro-batch gradient
all-reduce + per-rank rescale desynchronizes (and can deadlock) the model replicas
when ranks see a different number of contributing micro-batches.

The fix defers the all-reduce to once per window (``no_backward_sync``), issues a
zero-graph backward for empty micro-batches so every rank calls ``fabric.backward``
the same number of times, and rescales by a GLOBAL (all-reduced) contributing count.
This test is the spec for that fix:

(a) the 2-rank run COMPLETES (the deadlock detector — a hang trips the timeout);
(b) BOTH ranks end with byte-identical parameters (replicas stayed in sync); and
(c) the 2-rank parameters match a single-process ``grad_accum_steps>1`` reference
    over the SAME total data within a tight fp32 tolerance — i.e. the global rescale
    is correct, not just internally consistent.

The window deliberately includes one micro-batch that selects NOTHING (the empty
micro-batch), so the variable-``contributing`` path that previously desynced is
exercised. The runner is driven via ``subprocess`` (like ``tests/_ddp_finetune_runner``)
because Fabric's CPU DDP launcher re-executes the target module per rank, which does
not compose with the in-process pytest runner.

pragmatiq is an independent implementation inspired by the PRAGMA paper
(arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pragmatiq import api

# A hang (the failure this test exists to catch) trips the timeout instead of blocking CI.
_RUN_TIMEOUT_S = 600
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gradaccum_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Tiny tokenized dataset shared (read-only) by the ref and ddp runs."""
    work = tmp_path_factory.mktemp("ddpacc")
    api.synthesize(
        {"n_users": 80, "months": 14, "n_merchants": 400, "mule_ring_count": 1,
         "seed": 7, "eval_month_credit": 2, "eval_month_short": 8},
        out=work / "raw", n_workers=0, write_report=False,
    )
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 2500, "n_buckets": 16, "categorical_threshold": 150})
    return {"tok": work / "tok"}


def _run(fixture: dict[str, Path], out_dir: Path, mode: str, devices: int) -> list[dict]:
    """Invoke the grad-accum runner in a subprocess; collect each rank's result file.

    Each rank writes ``result_{mode}_rank{r}.json`` (stdout is not reliably piped back
    for every gloo rank), so the per-rank files are the source of truth.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "tests._ddp_grad_accum_runner",
         str(fixture["tok"]), str(out_dir), mode, str(devices)],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=_RUN_TIMEOUT_S,
    )
    assert proc.returncode == 0, (
        f"mode={mode} devices={devices} runner exited {proc.returncode} "
        "(a hang trips the timeout before this).\n"
        f"STDOUT tail:\n{proc.stdout[-2000:]}\nSTDERR tail:\n{proc.stderr[-4000:]}"
    )
    results = []
    for r in range(devices):
        path = out_dir / f"result_{mode}_rank{r}.json"
        assert path.exists(), (
            f"mode={mode} rank {r} produced no result file (did it hang or crash?).\n"
            f"STDOUT tail:\n{proc.stdout[-2000:]}\nSTDERR tail:\n{proc.stderr[-4000:]}"
        )
        results.append(json.loads(path.read_text()))
    return results


def _max_abs_diff(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b), f"param count mismatch: {len(a)} vs {len(b)}"
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def test_ddp_grad_accum_completes_in_sync_and_matches_single_process(
    gradaccum_fixture: dict[str, Path], tmp_path: Path,
) -> None:
    # (a) Both runs COMPLETE — if the 2-rank run deadlocked the subprocess would hit the
    # timeout and _run would raise, never reaching the assertions below.
    ref_results = _run(gradaccum_fixture, tmp_path, mode="ref", devices=1)
    ddp_results = _run(gradaccum_fixture, tmp_path, mode="ddp", devices=2)

    assert len(ref_results) == 1, "single-process ref should produce exactly one result"
    assert len(ddp_results) == 2, (
        f"expected a result from each of 2 ranks, got {len(ddp_results)} "
        "(a rank crashed or the all-reduce hung)"
    )

    # The variable-`contributing` desync path is genuinely exercised: exactly one rank
    # saw an empty micro-batch (so the per-rank `contributing` counts DIFFER — the case
    # that previously desynced replicas / deadlocked the reduce).
    empties = [r["n_empty"] for r in ddp_results]
    assert sum(empties) == 1 and max(empties) == 1, (
        f"expected exactly one rank to see one empty micro-batch, got n_empty={empties} "
        "(the desync case is not being exercised)"
    )

    # (b) Replicas stayed in SYNC: both gloo ranks end with byte-identical parameters.
    # This is the load-bearing anti-desync check — per-rank rescale would break it.
    r0, r1 = ddp_results[0]["params"], ddp_results[1]["params"]
    cross_rank = _max_abs_diff(r0, r1)
    assert cross_rank == 0.0, (
        f"2-rank replicas DESYNCHRONIZED: max |Δ| between ranks = {cross_rank:.3e} "
        "(expected exactly 0 — both ranks must apply the same global rescale)"
    )

    # (c) The 2-rank step matches the single-process grad-accum reference over the same
    # total data: the global (all-reduced) rescale reproduces Σgrad / global_contributing.
    ref = ref_results[0]["params"]
    vs_single = _max_abs_diff(r0, ref)
    assert vs_single < 1e-5, (
        f"2-rank grad-accum step diverged from single-process reference (max |Δ|={vs_single:.3e}); "
        "the DDP rescale normalization is wrong"
    )
