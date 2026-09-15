"""Pretrain scaling sweep and fine-tune sweep (one subprocess per leg; metrics parsed from the run)."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

from . import SCRIPT
from .legs import _run_leg_with_timeout
from .monitoring import _monitor_workload


def _parse_metrics_jsonl(metrics_path: Path) -> list[dict[str, Any]]:
    """Return parsed rows from a metrics.jsonl file."""
    rows: list[dict[str, Any]] = []
    if not metrics_path.exists():
        return rows
    for line in metrics_path.read_text().strip().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _steady_state_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Steady-state tokens_per_sec and peak gpu_mem_gb from the last half of logged steps.

    Uses the per-log-window rate (``tokens_per_sec_window``) when the trainer
    logs it — the cumulative rate carries warmup and cold-cache stalls for the
    rest of a short run — and falls back to the cumulative figure otherwise.
    """
    if not rows:
        return {"tokens_per_sec": float("nan"), "gpu_mem_gb": float("nan")}
    # Keep only the last half (skip warmup); with ≤2 rows use all rows
    half = len(rows) // 2 if len(rows) > 2 else 0
    tail = rows[half:]
    tps_vals = [r.get("tokens_per_sec_window", r.get("tokens_per_sec")) for r in tail
                if "tokens_per_sec" in r or "tokens_per_sec_window" in r]
    mem_vals = [r["gpu_mem_gb"] for r in tail if "gpu_mem_gb" in r]
    return {
        "tokens_per_sec": median(tps_vals) if tps_vals else float("nan"),
        "gpu_mem_gb": max(mem_vals, default=float("nan")),
    }


def _run_pretrain_leg(
    *,
    devices: int,
    run_name: str,
    shard_dir: Path,
    runs_root: Path,
    model_size: str,
    steps: int,
    token_budget: int,
    out_dir: Path,
    util_records: list[dict[str, Any]],
    leg_timeout_sec: float,
    nccl_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one pretrain leg via subprocess, sample GPU utilisation, parse metrics."""
    print(f"[pretrain] leg devices={devices} run={run_name}", flush=True)
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--_leg-pretrain",
        "--devices", str(devices),
        "--run-name", run_name,
        "--shard-dir", str(shard_dir),
        "--runs-root", str(runs_root),
        "--model-size", model_size,
        "--steps", str(steps),
        "--token-budget", str(token_budget),
    ]
    with _monitor_workload(f"pretrain_d{devices}", out_dir, util_records):
        t0 = time.time()
        returncode, timed_out = _run_leg_with_timeout(
            cmd, leg_timeout_sec, f"pretrain_d{devices}",
            log_dir=out_dir / "leg_logs", extra_env=nccl_env,
        )
        elapsed = time.time() - t0

    run_dir = runs_root / run_name
    metrics_path = run_dir / "metrics.jsonl"
    rows = _parse_metrics_jsonl(metrics_path)
    ss = _steady_state_metrics(rows)

    result: dict[str, Any] = {
        "devices": devices,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "returncode": returncode,
        "elapsed_s": round(elapsed, 1),
        "tokens_per_sec": ss["tokens_per_sec"],
        "gpu_mem_gb": ss["gpu_mem_gb"],
        # Record the config this leg ACTUALLY ran with — OOM fallbacks change
        # model_size/token_budget mid-sweep, and scaling efficiency is only
        # meaningful between legs with an identical (model_size, token_budget).
        "model_size": model_size,
        "token_budget": token_budget,
    }
    if timed_out:
        result["status"] = "timeout"
    return result


def _leg_efficiency(result: dict[str, Any], baseline: dict[str, Any] | None) -> float:
    """Scaling efficiency of one sweep leg against the d=1 baseline leg.

    tokens/sec ratios are only meaningful when both legs ran the identical
    (model_size, token_budget): an OOM fallback halves the per-batch work or
    shrinks the model, so an efficiency computed against that baseline would
    be silently wrong.  Incomparable legs are marked ``baseline_incomparable``
    (surfaced in REPORT.md) and get ``NaN``.
    """
    if baseline is None:
        return float("nan")
    if (result.get("model_size"), result.get("token_budget")) != (
        baseline.get("model_size"), baseline.get("token_budget")
    ):
        result["baseline_incomparable"] = True
        return float("nan")
    d = result.get("devices", 0)
    tps = result.get("tokens_per_sec", float("nan"))
    tps_1 = baseline.get("tokens_per_sec", float("nan"))
    if tps == tps and tps_1 == tps_1 and tps_1 > 0 and d > 0:
        return tps / (d * tps_1)
    return float("nan")


def _training_sweep(
    *,
    devices_sweep: list[int],
    shard_dir: Path,
    runs_root: Path,
    model_size: str,
    steps: int,
    token_budget: int,
    out_dir: Path,
    util_records: list[dict[str, Any]],
    leg_timeout_sec: float,
    run_start: float,
    max_runtime_sec: float,
    nccl_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run the training scaling sweep legs and return per-leg results."""
    results: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None  # the d=1 leg (efficiency reference)

    for d in devices_sweep:
        # Overall harness wall-clock cap: skip remaining legs if exceeded.
        if max_runtime_sec > 0 and (time.time() - run_start) >= max_runtime_sec:
            print(
                f"[pretrain] max-runtime-min exceeded before d={d}; "
                "skipping remaining legs",
                flush=True,
            )
            break

        run_name = f"sweep_d{d}"
        # Try with the current token_budget; on OOM retry at half budget
        for attempt, tb in enumerate([token_budget, token_budget // 2]):
            result = _run_pretrain_leg(
                devices=d,
                run_name=run_name,
                shard_dir=shard_dir,
                runs_root=runs_root,
                model_size=model_size,
                steps=steps,
                token_budget=tb,
                out_dir=out_dir,
                util_records=util_records,
                leg_timeout_sec=leg_timeout_sec,
                nccl_env=nccl_env,
            )
            if result.get("status") == "timeout":
                # Timed-out leg: record it and continue to next d (no retry).
                print(
                    f"[pretrain] d={d} TIMED OUT after {leg_timeout_sec / 60:.1f} min; "
                    "continuing to next leg",
                    flush=True,
                )
                break
            if result["returncode"] == 0:
                if attempt > 0:
                    result["oom_fallback"] = True
                    result["token_budget_used"] = tb
                break
            # Only retry once; if 1-GPU large still fails, fall back to medium
            if d == 1 and attempt == 0 and model_size == "large":
                print("[pretrain] OOM on d=1 large; retrying at medium model_size", flush=True)
                result = _run_pretrain_leg(
                    devices=d,
                    run_name=run_name,
                    shard_dir=shard_dir,
                    runs_root=runs_root,
                    model_size="medium",
                    steps=steps,
                    token_budget=tb,
                    out_dir=out_dir,
                    util_records=util_records,
                    leg_timeout_sec=leg_timeout_sec,
                    nccl_env=nccl_env,
                )
                result["oom_fallback_model_size"] = "medium"
                break

        if d == 1:
            baseline = result

        # Scaling efficiency vs the d=1 baseline; NaN + 'baseline_incomparable'
        # when an OOM fallback made the leg configs differ.
        tps = result.get("tokens_per_sec", float("nan"))
        eff = _leg_efficiency(result, baseline)
        result["efficiency"] = eff
        if result.get("baseline_incomparable"):
            print(
                f"[pretrain] d={d} efficiency=N/A — baseline incomparable "
                "(OOM fallback changed model_size/token_budget)",
                flush=True,
            )
        results.append(result)
        rc = result.get("returncode", 0)
        status = result.get("status", "")
        if status == "timeout":
            print(f"[pretrain] d={d} TIMEOUT", flush=True)
        elif rc != 0:
            print(f"[pretrain] d={d} FAILED (rc={rc})", flush=True)
        elif tps == tps and eff == eff:
            print(f"[pretrain] d={d} tps={tps:,.0f} eff={eff:.1%}", flush=True)
        else:
            print(f"[pretrain] d={d} tps=N/A (no metrics logged yet, rc=0)", flush=True)

    # Save CSV
    csv_path = out_dir / "training_sweep.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["devices", "run_name", "tokens_per_sec",
                                            "gpu_mem_gb", "efficiency", "elapsed_s",
                                            "model_size", "token_budget",
                                            "oom_fallback", "oom_fallback_model_size",
                                            "token_budget_used", "baseline_incomparable"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})  # type: ignore[arg-type]

    return results


def _run_finetune_leg(
    *,
    devices: int,
    finetune_token_budget: int,
    finetune_max_users: int,
    shard_dir: Path,
    run_dir: Path,
    label_path: Path,
    steps: int,
    out_dir: Path,
    util_records: list[dict[str, Any]],
    leg_timeout_sec: float,
    nccl_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one finetune leg via subprocess; return timing + AUC."""
    result_json = out_dir / f"finetune_d{devices}_result.json"
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--_leg-finetune",
        "--devices", str(devices),
        "--shard-dir", str(shard_dir),
        "--run-dir", str(run_dir),
        "--label-path", str(label_path),
        "--steps", str(steps),
        "--finetune-token-budget", str(finetune_token_budget),
        "--finetune-max-users", str(finetune_max_users),
        "--result-json", str(result_json),
    ]
    # Fine-tuning backprops through the whole frozen backbone; the caching
    # allocator's reserved pool ratchets across varlen batch shapes and can
    # exhaust the card even when per-batch allocation fits (measured: reserve
    # 38->61 GiB over 15 batches without this, flat 19 GiB with it).
    leg_env = {**(nccl_env or {}), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    with _monitor_workload(f"finetune_d{devices}", out_dir, util_records):
        t0 = time.time()
        returncode, timed_out = _run_leg_with_timeout(
            cmd, leg_timeout_sec, f"finetune_d{devices}",
            log_dir=out_dir / "leg_logs", extra_env=leg_env,
        )
        elapsed = time.time() - t0

    ft_result: dict[str, Any] = {}
    if result_json.exists():
        try:
            ft_result = json.loads(result_json.read_text())
        except json.JSONDecodeError:
            pass

    result: dict[str, Any] = {
        "devices": devices,
        "wall_time_s": round(elapsed, 1),
        "returncode": returncode,
        "best_val_auc": ft_result.get("best_val_auc", float("nan")),
        "epochs_run": ft_result.get("epochs_run", 0),
        "val_auc_history": ft_result.get("val_auc_history", []),
        "epoch_stats": ft_result.get("epoch_stats", []),
        "token_budget": ft_result.get("token_budget"),
    }
    if timed_out:
        result["status"] = "timeout"
    return result


def _finetune_sweep(
    *,
    finetune_devices: list[int],
    finetune_token_budget: int,
    finetune_max_users: int,
    shard_dir: Path,
    pretrained_run_dir: Path,
    label_path: Path,
    finetune_steps: int,
    out_dir: Path,
    util_records: list[dict[str, Any]],
    leg_timeout_sec: float,
    run_start: float,
    max_runtime_sec: float,
    nccl_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run fine-tuning legs and return per-leg results."""
    results: list[dict[str, Any]] = []
    for d in finetune_devices:
        # Overall harness wall-clock cap: skip remaining legs if exceeded.
        if max_runtime_sec > 0 and (time.time() - run_start) >= max_runtime_sec:
            print(
                f"[finetune] max-runtime-min exceeded before d={d}; "
                "skipping remaining legs",
                flush=True,
            )
            break

        print(f"[finetune] devices={d}", flush=True)
        result = _run_finetune_leg(
            devices=d,
            finetune_token_budget=finetune_token_budget,
            finetune_max_users=finetune_max_users,
            shard_dir=shard_dir,
            run_dir=pretrained_run_dir,
            label_path=label_path,
            steps=finetune_steps,
            out_dir=out_dir,
            util_records=util_records,
            leg_timeout_sec=leg_timeout_sec,
            nccl_env=nccl_env,
        )
        if result.get("status") == "timeout":
            print(f"[finetune] d={d} TIMEOUT", flush=True)
        else:
            print(
                f"[finetune] d={d} best_val_auc={result['best_val_auc']:.4f} "
                f"wall={result['wall_time_s']:.0f}s" if result["best_val_auc"] == result["best_val_auc"]
                else f"[finetune] d={d} failed (rc={result['returncode']})",
                flush=True,
            )
        results.append(result)

    # Save CSV
    csv_path = out_dir / "finetune_results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["devices", "wall_time_s", "best_val_auc", "epochs_run"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})  # type: ignore[arg-type]

    return results
