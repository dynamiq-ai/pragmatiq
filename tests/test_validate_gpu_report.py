"""Verdict correctness of the GPU-validation harness report.

Regression guards for Bugbot PR #12 findings 3460855084 / 3460874154 /
3460874162: the REPORT.md verdict used to ignore finetune failures/timeouts
and flash-check failures, and every DDP rank wrote the leg result JSON.

The harness lives in scripts/ (not the package), so it is loaded by path;
its module-level imports are stdlib-only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_gpu.py"


@pytest.fixture(scope="module")
def vg():
    spec = importlib.util.spec_from_file_location("validate_gpu_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(vg_mod, tmp_path: Path, *, training=(), finetune=(), flash=None) -> str:
    path = vg_mod._write_report(
        tmp_path,
        start_ts="2026-01-01T00:00:00",
        end_ts="2026-01-01T01:00:00",
        dry_run=False,
        model_size="nano",
        n_users=10,
        steps=2,
        training_results=list(training),
        finetune_results=list(finetune),
        serving_results=[],
        triton_results=[],
        triton_skip_reason="skipped (test)",
        util_records=[],
        flash_check_result=flash,
    )
    return Path(path).read_text()


GOOD_TRAIN = {"devices": 1, "tokens_per_sec": 100.0, "efficiency": 1.0, "returncode": 0,
              "wall_time_s": 10.0, "mlm_loss": 3.0}
GOOD_FT = {"devices": 1, "wall_time_s": 5.0, "returncode": 0, "best_val_auc": 0.71,
           "epochs_run": 3, "val_auc_history": [0.6, 0.7, 0.71]}
FLASH_PASS = {"skipped": False, "skip_reason": "", "flash_available": True, "passed": True,
              "max_abs_diff": 1e-3, "mean_abs_diff": 1e-4, "tol": 1e-2}


class TestVerdict:
    def test_all_green(self, vg, tmp_path) -> None:
        text = _write(vg, tmp_path, training=[GOOD_TRAIN], finetune=[GOOD_FT], flash=FLASH_PASS)
        assert "All legs completed successfully" in text

    def test_finetune_nonzero_exit_fails_verdict(self, vg, tmp_path) -> None:
        bad = {**GOOD_FT, "returncode": 1, "best_val_auc": float("nan")}
        text = _write(vg, tmp_path, training=[GOOD_TRAIN], finetune=[bad], flash=FLASH_PASS)
        assert "All legs completed successfully" not in text
        assert "finetune" in text.split("## Verdict")[1]

    def test_finetune_timeout_fails_verdict(self, vg, tmp_path) -> None:
        bad = {**GOOD_FT, "status": "timeout"}
        text = _write(vg, tmp_path, training=[GOOD_TRAIN], finetune=[bad], flash=FLASH_PASS)
        assert "All legs completed successfully" not in text

    def test_flash_check_failure_fails_verdict(self, vg, tmp_path) -> None:
        flash_fail = {**FLASH_PASS, "passed": False, "max_abs_diff": 0.5}
        text = _write(vg, tmp_path, training=[GOOD_TRAIN], finetune=[GOOD_FT], flash=flash_fail)
        assert "All legs completed successfully" not in text
        assert "flash-attn" in text.split("## Verdict")[1]

    def test_flash_skipped_does_not_fail_verdict(self, vg, tmp_path) -> None:
        skipped = {"skipped": True, "skip_reason": "skipped (no CUDA)"}
        text = _write(vg, tmp_path, training=[GOOD_TRAIN], finetune=[GOOD_FT], flash=skipped)
        assert "All legs completed successfully" in text


class TestFinetuneLegFailed:
    def test_success_is_not_failed(self, vg) -> None:
        assert not vg._finetune_leg_failed(GOOD_FT)

    @pytest.mark.parametrize("patch", [
        {"returncode": 1},
        {"status": "timeout"},
        {"best_val_auc": float("nan")},
        {"best_val_auc": -1},
    ])
    def test_failure_modes(self, vg, patch) -> None:
        assert vg._finetune_leg_failed({**GOOD_FT, **patch})


class TestRankZero:
    def test_default_is_rank_zero(self, vg, monkeypatch) -> None:
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert vg._is_rank_zero()

    def test_nonzero_rank_is_not(self, vg, monkeypatch) -> None:
        monkeypatch.setenv("RANK", "1")
        assert not vg._is_rank_zero()

    def test_local_rank_fallback(self, vg, monkeypatch) -> None:
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert not vg._is_rank_zero()
