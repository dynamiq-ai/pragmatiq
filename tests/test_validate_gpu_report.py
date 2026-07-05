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


class TestNvidiaSampler:
    """Bugbot 3523826988: '-l 1' put nvidia-smi in loop mode, so every sample
    timed out under subprocess.run(timeout=5) and no GPU row was ever recorded."""

    def test_sample_cmd_has_no_loop_flag(self, vg, tmp_path) -> None:
        sampler = vg._NvidiaSampler(tmp_path / "gpu.csv")
        sampler._cmd = "/usr/bin/nvidia-smi"
        cmd = sampler._sample_cmd()
        assert "-l" not in cmd
        assert not any(arg.startswith("--loop") for arg in cmd)

    def test_loop_records_snapshot_rows(self, vg, tmp_path, monkeypatch) -> None:
        sampler = vg._NvidiaSampler(tmp_path / "gpu.csv", interval_s=0.01)
        sampler._cmd = "/usr/bin/nvidia-smi"
        calls: list[list[str]] = []

        class _Result:
            stdout = "0, 87, 40000, 81920, 350\n1, 93, 41000, 81920, 360\n"

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if len(calls) >= 2:
                sampler._stop.set()
            return _Result()

        monkeypatch.setattr(vg.subprocess, "run", fake_run)
        sampler._loop()

        assert all("-l" not in c for c in calls)
        summary = sampler._summarize()
        assert summary["n_gpus"] == 2
        assert summary["peak_util_pct"] == 93.0
        assert summary["n_samples"] == 4

    def test_consecutive_failures_warn_once(self, vg, tmp_path, monkeypatch, capsys) -> None:
        sampler = vg._NvidiaSampler(tmp_path / "gpu.csv", interval_s=0.01)
        sampler._cmd = "/usr/bin/nvidia-smi"
        n_calls = 0

        def fake_run(cmd, **kwargs):
            nonlocal n_calls
            n_calls += 1
            if n_calls >= 5:
                sampler._stop.set()
            raise vg.subprocess.TimeoutExpired(cmd, 5)

        monkeypatch.setattr(vg.subprocess, "run", fake_run)
        sampler._loop()

        out = capsys.readouterr().out
        assert out.count("[nvidia-smi]") == 1
        assert sampler._summarize() == {}


BASE_LEG = {"devices": 1, "tokens_per_sec": 100.0, "returncode": 0,
            "model_size": "large", "token_budget": 1024}


class TestLegEfficiency:
    """F23: OOM fallback must not poison scaling efficiency — legs whose
    (model_size, token_budget) differ from the d=1 baseline get NaN."""

    def test_identical_config_scales(self, vg) -> None:
        leg = {"devices": 4, "tokens_per_sec": 360.0,
               "model_size": "large", "token_budget": 1024}
        assert vg._leg_efficiency(leg, dict(BASE_LEG)) == pytest.approx(0.9)
        assert "baseline_incomparable" not in leg

    def test_baseline_leg_is_its_own_reference(self, vg) -> None:
        leg = dict(BASE_LEG)
        assert vg._leg_efficiency(leg, leg) == pytest.approx(1.0)

    def test_token_budget_fallback_marks_incomparable(self, vg) -> None:
        leg = {"devices": 4, "tokens_per_sec": 360.0,
               "model_size": "large", "token_budget": 512}
        eff = vg._leg_efficiency(leg, dict(BASE_LEG))
        assert eff != eff  # NaN
        assert leg["baseline_incomparable"] is True

    def test_model_size_fallback_marks_incomparable(self, vg) -> None:
        baseline = {**BASE_LEG, "model_size": "medium"}  # d=1 fell back to medium
        leg = {"devices": 8, "tokens_per_sec": 700.0,
               "model_size": "large", "token_budget": 1024}
        eff = vg._leg_efficiency(leg, baseline)
        assert eff != eff
        assert leg["baseline_incomparable"] is True

    def test_no_baseline_is_nan(self, vg) -> None:
        leg = {"devices": 2, "tokens_per_sec": 150.0,
               "model_size": "large", "token_budget": 1024}
        eff = vg._leg_efficiency(leg, None)
        assert eff != eff
        assert "baseline_incomparable" not in leg


class TestReportIncomparableBaseline:
    def test_headline_and_table_note_incomparable(self, vg, tmp_path) -> None:
        d1 = {**GOOD_TRAIN, "model_size": "medium", "token_budget": 512,
              "oom_fallback": True, "token_budget_used": 512}
        d8 = {"devices": 8, "tokens_per_sec": 700.0, "efficiency": float("nan"),
              "returncode": 0, "elapsed_s": 30.0, "gpu_mem_gb": 40.0,
              "model_size": "large", "token_budget": 1024,
              "baseline_incomparable": True}
        text = _write(vg, tmp_path, training=[d1, d8], finetune=[GOOD_FT], flash=FLASH_PASS)

        headline = text.split("## Training Scaling Sweep")[0]
        assert "baseline incomparable (OOM fallback)" in headline

        sweep = text.split("## Training Scaling Sweep")[1].split("## Fine-tuning")[0]
        assert "baseline incomparable (OOM fallback)" in sweep

    def test_comparable_legs_have_no_incomparable_note(self, vg, tmp_path) -> None:
        text = _write(vg, tmp_path, training=[GOOD_TRAIN], finetune=[GOOD_FT], flash=FLASH_PASS)
        assert "baseline incomparable" not in text


class TestCountFailedLegs:
    """Bugbot 3523826317/3523826986: exit code must match the REPORT.md verdict —
    a finetune leg exiting 0 with a NaN/-1 AUC is a failure."""

    def test_all_green(self, vg) -> None:
        assert vg._count_failed_legs([GOOD_TRAIN], [GOOD_FT], FLASH_PASS) == 0

    def test_finetune_zero_exit_nan_auc_counts(self, vg) -> None:
        bad = {**GOOD_FT, "best_val_auc": float("nan")}
        assert vg._count_failed_legs([GOOD_TRAIN], [bad], FLASH_PASS) == 1

    def test_finetune_zero_exit_sentinel_auc_counts(self, vg) -> None:
        bad = {**GOOD_FT, "best_val_auc": -1}
        assert vg._count_failed_legs([GOOD_TRAIN], [bad], FLASH_PASS) == 1

    def test_finetune_timeout_counts(self, vg) -> None:
        bad = {**GOOD_FT, "status": "timeout"}
        assert vg._count_failed_legs([GOOD_TRAIN], [bad], FLASH_PASS) == 1

    def test_training_timeout_and_rc_count(self, vg) -> None:
        t1 = {**GOOD_TRAIN, "status": "timeout", "returncode": -1}
        t2 = {**GOOD_TRAIN, "returncode": 2}
        assert vg._count_failed_legs([t1, t2], [GOOD_FT], FLASH_PASS) == 2

    def test_flash_fail_counts(self, vg) -> None:
        flash_fail = {**FLASH_PASS, "passed": False}
        assert vg._count_failed_legs([GOOD_TRAIN], [GOOD_FT], flash_fail) == 1

    def test_flash_skipped_or_absent_does_not_count(self, vg) -> None:
        skipped = {"skipped": True, "skip_reason": "no CUDA"}
        assert vg._count_failed_legs([GOOD_TRAIN], [GOOD_FT], skipped) == 0
        assert vg._count_failed_legs([GOOD_TRAIN], [GOOD_FT], None) == 0


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
