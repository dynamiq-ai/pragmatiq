"""Multi-GPU (DDP) LoRA fine-tuning smoke test (GA2).

A real 2-rank gloo (CPU DDP) run of :func:`pragmatiq.api.finetune`. Its job is to
catch the #1 distributed risk: divergent early-stopping. If any rank computed a
*local* validation AUC, the ranks would make different stop decisions, one would
break the epoch loop while the other kept iterating, and the next epoch's gradient
all-reduce (or the val all_gather) would deadlock. So the primary assertion is
simply *it completes* — plus that the 2-rank run produces a finite AUC and the two
ranks agree (identical RESULT), proving the AUC is gathered globally.

Fabric's CPU DDP launcher spawns worker processes that re-execute the target
module, which does not compose with the in-process pytest runner — so the fine-tune
is driven through a standalone script (``tests/_ddp_finetune_runner.py``) invoked
via ``subprocess``. The single-process (``devices=1``) reference run goes through
the SAME script for an apples-to-apples comparison.

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

# Generous wall-clock ceiling: nano model, 90 users, 5 epochs, 2 ranks. A hang
# (the failure this test exists to catch) trips the timeout instead of blocking CI.
_RUN_TIMEOUT_S = 600
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ddp_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Tiny tokenized dataset + a 5-step nano pretrained run + a 2-class label table.

    Built once (single process); both the ``devices=1`` and ``devices=2`` runs load
    it read-only. ``ltv_positive`` is used because it carries plenty of BOTH classes
    (~72 pos / ~15 neg over the cohort), so every rank's val slice — and the gathered
    global val set — has both classes and the ROC-AUC is finite.
    """
    work = tmp_path_factory.mktemp("ftddp")
    api.synthesize(
        {"n_users": 90, "months": 14, "n_merchants": 400, "mule_ring_count": 1,
         "seed": 4, "eval_month_credit": 2, "eval_month_short": 8},
        out=work / "raw", n_workers=0, write_report=False,
    )
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 2500, "n_buckets": 16, "categorical_threshold": 150})
    res = api.pretrain(work / "tok", "p", model_size="nano", runs_root=work / "runs",
                       max_steps=5, token_budget=4096, warmup_steps=0, log_every=100,
                       checkpoint_every_min=1000.0)
    return {
        "tok": work / "tok",
        "run": Path(res["run_dir"]),
        "label": work / "raw" / "labels" / "ltv_positive.parquet",
    }


def _run(fixture: dict[str, Path], devices: int) -> dict:
    """Invoke the fine-tune runner in a subprocess; parse the RESULT json.

    Under ``devices=2`` Fabric spawns two gloo workers that each print an identical
    RESULT line (the AUC is global); we take the first.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "tests._ddp_finetune_runner",
         str(fixture["tok"]), str(fixture["run"]), str(fixture["label"]), str(devices)],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=_RUN_TIMEOUT_S,
    )
    results = [json.loads(line[len("RESULT "):]) for line in proc.stdout.splitlines()
               if line.startswith("RESULT ")]
    assert results, (
        f"devices={devices} run emitted no RESULT line (did it hang or crash?).\n"
        f"returncode={proc.returncode}\nSTDOUT tail:\n{proc.stdout[-2000:]}\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}"
    )
    assert proc.returncode == 0, f"devices={devices} runner exited {proc.returncode}\n{proc.stderr[-3000:]}"
    return results[0] if devices == 1 else _assert_ranks_agree(results)


def _assert_ranks_agree(results: list[dict]) -> dict:
    """Both gloo ranks must report the SAME result — proof the AUC (and the
    early-stop decision it drives) is global, not per-rank-local."""
    assert len(results) >= 2, f"expected a RESULT from each of 2 ranks, got {len(results)}"
    first = results[0]
    for other in results[1:]:
        assert other["epochs_run"] == first["epochs_run"], "ranks ran a different number of epochs"
        assert abs(other["best_val_auc"] - first["best_val_auc"]) < 1e-9, \
            "ranks disagree on best_val_auc -> early-stop is NOT driven by a global AUC (would hang on real DDP)"
        assert other["val_auc_history"] == pytest.approx(first["val_auc_history"]), \
            "ranks disagree on the val-AUC history -> AUC was computed per-rank-locally"
    return first


def test_ddp_finetune_completes_and_matches_single_process(ddp_fixture: dict[str, Path]) -> None:
    """The 2-rank gloo fine-tune must COMPLETE (no early-stop deadlock) and return a
    finite, sane val AUC; the contract return keys are unchanged.

    We do NOT require the 2-rank AUC to equal the single-process AUC exactly: DDP
    averages gradients and offsets the seed per rank, so the optimization trajectory
    (and thus the fitted model) differs even though the val set is identical. We
    assert both are finite and in a sane band, and — the load-bearing check — that
    the two ranks agree on the AUC (so the stop decision is global).
    """
    single = _run(ddp_fixture, devices=1)
    ddp = _run(ddp_fixture, devices=2)

    # Both runs are well-formed and used the injected LoRA adapters.
    assert single.keys() == {"best_val_auc", "epochs_run", "n_adapted", "val_auc_history"}
    assert ddp.keys() == {"best_val_auc", "epochs_run", "n_adapted", "val_auc_history"}
    assert single["n_adapted"] > 0 and ddp["n_adapted"] == single["n_adapted"]

    # The 2-rank run produced a finite, in-range global AUC (not the -1.0 sentinel,
    # not NaN) — i.e. the cross-rank val gather succeeded and scored both classes.
    for val in (single["best_val_auc"], ddp["best_val_auc"]):
        assert val == val, "best_val_auc is NaN"  # noqa: PLR0124  (NaN check)
        assert 0.0 <= val <= 1.0, f"val AUC out of [0,1]: {val}"

    # Same val set on both paths, so the AUC should be in the same neighborhood even
    # though the trajectory differs — a wide tolerance catches gross divergence (a
    # broken gather / wrong-label scoring) without flaking on the tiny-model noise.
    assert abs(ddp["best_val_auc"] - single["best_val_auc"]) < 0.25, (
        f"2-rank AUC {ddp['best_val_auc']:.3f} far from single-process {single['best_val_auc']:.3f}; "
        "the distributed val gather may be scoring the wrong set"
    )
