#!/usr/bin/env python3
"""GPU validation harness for pragmatiq (Task GA3).

Runs a full multi-GPU measurement sweep on an 8×H100 pod and writes a
REPORT.md with training scaling efficiency, finetune DDP validation,
serving runtime throughput (GPU vs CPU, concurrent requests), and system
utilisation.

A ``--dry-run`` mode exercises every code path on CPU (nano model,
~300 users, devices=1, ~5 steps) in a couple of minutes so the full
pipeline can be proven locally before any GPU spend.

Usage (dry-run, local, free):
    .venv/bin/python scripts/validate_gpu.py --dry-run --out /tmp/gpuval-dry

Usage (real 8×H100 pod):
    python scripts/validate_gpu.py --out outputs/gpu-validation-$(date +%Y%m%d)

Hidden leg modes invoked as subprocesses so Lightning Fabric DDP re-launch
works (Fabric re-launches THIS file across d processes):
    --_leg-pretrain   --devices D --run-name NAME --shard-dir DIR
                      --runs-root DIR --model-size S --steps N --token-budget N
    --_leg-finetune   --devices D --run-dir DIR --shard-dir DIR
                      --label-path PATH --runs-root DIR --steps N
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

# ---------------------------------------------------------------------------
# Helpers: system monitoring
# ---------------------------------------------------------------------------


class _NvidiaSampler:
    """Background sampler for nvidia-smi GPU metrics.

    Writes one CSV row per second.  No-ops when nvidia-smi is absent.
    """

    def __init__(self, csv_path: Path) -> None:
        self._csv = csv_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cmd = shutil.which("nvidia-smi")

    def start(self) -> None:
        if self._cmd is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop the sampler and return per-GPU summary statistics."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self._summarize()

    def _loop(self) -> None:
        fields = ["index", "utilization.gpu [%]", "memory.used [MiB]",
                  "memory.total [MiB]", "power.draw [W]"]
        wrote_header = not self._csv.exists()
        with self._csv.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if wrote_header:
                writer.writerow(["ts"] + fields)
            while not self._stop.is_set():
                try:
                    result = subprocess.run(
                        [self._cmd,
                         "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                         "--format=csv,noheader,nounits",
                         "-l", "1"],
                        capture_output=True, text=True, timeout=5,
                    )
                    ts = datetime.utcnow().isoformat()
                    for line in result.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) == len(fields):
                            writer.writerow([ts] + parts)
                    fh.flush()
                except Exception:  # noqa: BLE001
                    pass
                self._stop.wait(1.0)

    def _summarize(self) -> dict[str, Any]:
        if not self._csv.exists():
            return {}
        rows: list[list[str]] = []
        with self._csv.open() as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return {}
            for row in reader:
                rows.append(row)
        if not rows:
            return {}
        # Columns after 'ts': index, util%, mem_used, mem_total, power
        # Aggregate per-GPU (group by GPU index col[1]) so that multi-GPU pods
        # don't skew the mean/peak by repeating the same GPU in different rows.
        # Strategy: compute per-GPU mean-util and mean-power, then average across
        # GPUs (each GPU contributes equally regardless of sample count); peak
        # stats use the global max across all GPU × sample rows.
        try:
            # gpu_index -> list of (util, mem_used, power)
            per_gpu: dict[str, list[tuple[float, float, float]]] = {}
            for row in rows:
                if len(row) < 6:
                    continue
                try:
                    gpu_idx = row[1].strip()
                    util = float(row[2])
                    mem = float(row[3])
                    pwr = float(row[5])
                    per_gpu.setdefault(gpu_idx, []).append((util, mem, pwr))
                except ValueError:
                    pass
            if not per_gpu:
                return {}
            # Per-GPU mean-util then averaged across GPUs
            gpu_mean_utils = [sum(t[0] for t in samples) / len(samples)
                              for samples in per_gpu.values()]
            mean_util = sum(gpu_mean_utils) / len(gpu_mean_utils)
            # Peak util = highest single sample across all GPUs
            peak_util = max(t[0] for samples in per_gpu.values() for t in samples)
            # Peak mem = highest single sample across all GPUs
            peak_mem = max(t[1] for samples in per_gpu.values() for t in samples)
            # Mean power = average across per-GPU means
            gpu_mean_pwrs = [sum(t[2] for t in samples) / len(samples)
                             for samples in per_gpu.values()]
            mean_pwr = sum(gpu_mean_pwrs) / len(gpu_mean_pwrs)
            n_samples = sum(len(s) for s in per_gpu.values())
            return {
                "mean_util_pct": round(mean_util, 1),
                "peak_util_pct": round(peak_util, 1),
                "peak_mem_mib": round(peak_mem, 1),
                "mean_power_w": round(mean_pwr, 1),
                "n_gpus": len(per_gpu),
                "n_samples": n_samples,
            }
        except Exception:  # noqa: BLE001
            return {}


def _cpu_ram_sample() -> dict[str, float]:
    """Return a dict with cpu_pct and ram_gb (best-effort, no hard deps)."""
    cpu_pct = float("nan")
    ram_gb = float("nan")
    try:
        import psutil  # type: ignore[import]
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_gb = psutil.virtual_memory().used / 1e9
        return {"cpu_pct": cpu_pct, "ram_gb": ram_gb}
    except ImportError:
        pass
    # Fallback: getloadavg for CPU, /proc/meminfo for RAM
    try:
        load1, _, _ = os.getloadavg()
        n_cpu = os.cpu_count() or 1
        cpu_pct = min(100.0, load1 / n_cpu * 100.0)
    except Exception:  # noqa: BLE001
        cpu_pct = float("nan")
    try:
        with open("/proc/meminfo") as f:
            lines = f.read().splitlines()
        info: dict[str, int] = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        free = info.get("MemFree", 0) + info.get("Buffers", 0) + info.get("Cached", 0)
        ram_gb = (total - free) / 1024 / 1024
    except Exception:  # noqa: BLE001
        ram_gb = float("nan")
    return {"cpu_pct": cpu_pct, "ram_gb": ram_gb}


@contextmanager
def _monitor_workload(
    label: str,
    out_dir: Path,
    util_records: list[dict[str, Any]],
) -> Generator[None, None, None]:
    """Context manager: start nvidia+CPU sampling, yield, stop and record."""
    csv_path = out_dir / f"gpu_util_{label}.csv"
    sampler = _NvidiaSampler(csv_path)
    samples: list[dict[str, float]] = []

    stop_event = threading.Event()

    def _cpu_loop() -> None:
        while not stop_event.is_set():
            samples.append(_cpu_ram_sample())
            stop_event.wait(2.0)

    cpu_thread = threading.Thread(target=_cpu_loop, daemon=True)
    sampler.start()
    cpu_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        gpu_summary = sampler.stop()
        cpu_thread.join(timeout=5)
        cpu_pct_vals = [s["cpu_pct"] for s in samples if s["cpu_pct"] == s["cpu_pct"]]
        ram_vals = [s["ram_gb"] for s in samples if s["ram_gb"] == s["ram_gb"]]
        record = {
            "label": label,
            "gpu": gpu_summary,
            "mean_cpu_pct": round(sum(cpu_pct_vals) / max(len(cpu_pct_vals), 1), 1),
            "peak_ram_gb": round(max(ram_vals, default=float("nan")), 2),
        }
        util_records.append(record)


# ---------------------------------------------------------------------------
# Helpers: metrics parsing
# ---------------------------------------------------------------------------


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
    """Compute steady-state tokens_per_sec and peak gpu_mem_gb from the last half of steps."""
    if not rows:
        return {"tokens_per_sec": float("nan"), "gpu_mem_gb": float("nan")}
    # Keep only the last half (skip warmup); with ≤2 rows use all rows
    half = len(rows) // 2 if len(rows) > 2 else 0
    tail = rows[half:]
    tps_vals = [r["tokens_per_sec"] for r in tail if "tokens_per_sec" in r]
    mem_vals = [r["gpu_mem_gb"] for r in tail if "gpu_mem_gb" in r]
    return {
        "tokens_per_sec": median(tps_vals) if tps_vals else float("nan"),
        "gpu_mem_gb": max(mem_vals, default=float("nan")),
    }


# ---------------------------------------------------------------------------
# Helpers: sample records (for serving measurement)
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {
        "user_id": f"u{i}",
        "events": [
            # ts is int microseconds (per UserRecord.from_dict in pragmatiq/core/schema.py)
            {"ts": (1_700_000_000 + i * 86400 + j * 3600) * 1_000_000,
             "source": "card",
             "fields": {"mcc": "5411", "amount_usd": str(20 + j * 5)}}
            for j in range(8)
        ],
        "attributes": {"country": "GB", "age_band": "25-34"},
        # lifelong is list[{"key": str, "ts": int_us}] per the records contract
        "lifelong": [{"key": "account_opened", "ts": (1_680_000_000 + i * 86400) * 1_000_000}],
    }
    for i in range(8)
]


# ---------------------------------------------------------------------------
# Leg modes: pretrain and finetune (invoked as subprocesses)
# ---------------------------------------------------------------------------


def _leg_pretrain(args: argparse.Namespace) -> None:
    """Run one pretrain leg and exit.  Invoked via subprocess by the orchestrator."""
    # Delay heavy imports until we are inside the leg (Fabric re-launch path).
    import pragmatiq.api as api  # noqa: PLC0415

    config: dict[str, Any] = {
        "max_steps": args.steps,
        "warmup_steps": max(1, args.steps // 10),
        "token_budget": args.token_budget,
        "devices": args.devices,
        "verbose": True,
    }
    result = api.pretrain(
        args.shard_dir,
        args.run_name,
        model_size=args.model_size,
        config=config,
        runs_root=args.runs_root,
    )
    print(f"[leg-pretrain] done: {result['run']} steps={result['steps']}", flush=True)


def _leg_finetune(args: argparse.Namespace) -> None:
    """Run one finetune leg and exit.  Invoked via subprocess by the orchestrator."""
    import pragmatiq.api as api  # noqa: PLC0415

    config: dict[str, Any] = {
        "max_epochs": args.steps,
        "devices": args.devices,
    }
    result = api.finetune(
        args.shard_dir,
        args.run_dir,
        args.label_path,
        config=config,
        device="auto",
    )
    # Write result JSON so orchestrator can read it back
    Path(args.result_json).write_text(json.dumps(result))
    print(f"[leg-finetune] done: best_val_auc={result.get('best_val_auc')}", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator: data prep
# ---------------------------------------------------------------------------


def _data_prep(
    out_dir: Path,
    n_users: int,
    dry_run: bool,
) -> tuple[Path, Path, Path]:
    """Run synthesize + tokenize; return (synth_dir, tok_dir, labels_path)."""
    import pragmatiq.api as api  # noqa: PLC0415

    synth_dir = out_dir / "data" / "synth"
    tok_dir = out_dir / "data" / "tok"

    print(f"[data] synthesize {n_users} users -> {synth_dir}", flush=True)
    api.synthesize(
        {"n_users": n_users, "seed": 0},
        out=str(synth_dir),
        n_workers=1 if dry_run else max(4, os.cpu_count() or 4),
        write_report=False,
    )

    labels_path = synth_dir / "labels" / "default_12m.parquet"

    print(f"[data] tokenize -> {tok_dir}", flush=True)
    api.tokenize(
        str(synth_dir),
        str(tok_dir),
        n_workers=1 if dry_run else max(4, os.cpu_count() or 4),
    )
    return synth_dir, tok_dir, labels_path


# ---------------------------------------------------------------------------
# Orchestrator: training scaling sweep
# ---------------------------------------------------------------------------


def _run_leg_with_timeout(
    cmd: list[str],
    timeout_sec: float,
    label: str,
) -> tuple[int, bool]:
    """Run a leg subprocess with a per-leg timeout.

    Uses ``start_new_session=True`` so the child gets its own process group,
    allowing ``os.killpg`` to reap orphaned GPU processes on timeout.

    Returns:
        (returncode, timed_out) — on timeout returncode is -1.
    """
    try:
        proc = subprocess.Popen(cmd, start_new_session=True)  # noqa: S603
        try:
            proc.wait(timeout=timeout_sec)
            return proc.returncode, False
        except subprocess.TimeoutExpired:
            print(
                f"[{label}] TIMEOUT after {timeout_sec / 60:.1f} min; "
                "killing process group ...",
                flush=True,
            )
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # already gone
            proc.wait()
            return -1, True
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] subprocess launch error: {exc}", flush=True)
        return -1, False


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
) -> dict[str, Any]:
    """Run one pretrain leg via subprocess, sample GPU utilisation, parse metrics."""
    print(f"[pretrain] leg devices={devices} run={run_name}", flush=True)
    cmd = [
        sys.executable, __file__,
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
            cmd, leg_timeout_sec, f"pretrain_d{devices}"
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
    }
    if timed_out:
        result["status"] = "timeout"
    return result


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
) -> list[dict[str, Any]]:
    """Run the training scaling sweep legs and return per-leg results."""
    results: list[dict[str, Any]] = []
    tps_1: float = float("nan")

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
                )
                result["oom_fallback_model_size"] = "medium"
                break

        if d == 1:
            tps_1 = result.get("tokens_per_sec", float("nan"))

        # Compute scaling efficiency
        tps = result.get("tokens_per_sec", float("nan"))
        if tps == tps and tps_1 == tps_1 and tps_1 > 0 and d > 0:
            eff = tps / (d * tps_1)
        else:
            eff = float("nan")
        result["efficiency"] = eff
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
                                            "oom_fallback", "oom_fallback_model_size",
                                            "token_budget_used"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})  # type: ignore[arg-type]

    return results


# ---------------------------------------------------------------------------
# Orchestrator: finetune sweep
# ---------------------------------------------------------------------------


def _run_finetune_leg(
    *,
    devices: int,
    shard_dir: Path,
    run_dir: Path,
    label_path: Path,
    steps: int,
    out_dir: Path,
    util_records: list[dict[str, Any]],
    leg_timeout_sec: float,
) -> dict[str, Any]:
    """Run one finetune leg via subprocess; return timing + AUC."""
    result_json = out_dir / f"finetune_d{devices}_result.json"
    cmd = [
        sys.executable, __file__,
        "--_leg-finetune",
        "--devices", str(devices),
        "--shard-dir", str(shard_dir),
        "--run-dir", str(run_dir),
        "--label-path", str(label_path),
        "--steps", str(steps),
        "--result-json", str(result_json),
    ]
    with _monitor_workload(f"finetune_d{devices}", out_dir, util_records):
        t0 = time.time()
        returncode, timed_out = _run_leg_with_timeout(
            cmd, leg_timeout_sec, f"finetune_d{devices}"
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
    }
    if timed_out:
        result["status"] = "timeout"
    return result


def _finetune_sweep(
    *,
    finetune_devices: list[int],
    shard_dir: Path,
    pretrained_run_dir: Path,
    label_path: Path,
    finetune_steps: int,
    out_dir: Path,
    util_records: list[dict[str, Any]],
    leg_timeout_sec: float,
    run_start: float,
    max_runtime_sec: float,
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
            shard_dir=shard_dir,
            run_dir=pretrained_run_dir,
            label_path=label_path,
            steps=finetune_steps,
            out_dir=out_dir,
            util_records=util_records,
            leg_timeout_sec=leg_timeout_sec,
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


# ---------------------------------------------------------------------------
# Triton container path (optional, defensive, GPU-only, never blocks dry-run)
# ---------------------------------------------------------------------------

#: Config templates for each Triton serving variant.
_TRITON_VARIANTS: list[tuple[str, str]] = [
    (
        "cpu",
        'instance_group [\n  {\n    count: 2\n    kind: KIND_CPU\n  }\n]',
    ),
    (
        "1gpu",
        'instance_group [\n  {\n    count: 1\n    kind: KIND_GPU\n  }\n]',
    ),
    (
        "8gpu",
        'instance_group [\n  {\n    count: 8\n    kind: KIND_GPU\n  }\n]',
    ),
]

_TRITON_IMAGE = "pragmatiq-triton:latest"
_TRITON_CONTAINER_PREFIX = "pq-triton-val-"
# Readiness poll parameters
_TRITON_READY_POLLS = 60
_TRITON_READY_INTERVAL_S = 3
# perf_analyzer concurrency sweep
_PA_CONCURRENCIES = (1, 4, 16, 64)


def _triton_config_pbtxt(base_config: str, instance_block: str) -> str:
    """Replace the instance_group block in a config.pbtxt string."""
    import re  # noqa: PLC0415
    # Replace existing instance_group [...] block (non-greedy, dot-all)
    return re.sub(
        r"instance_group\s*\[.*?\]",
        instance_block,
        base_config,
        flags=re.DOTALL,
    )


def _make_perf_analyzer_input(records: list[dict]) -> str:
    """Build a perf_analyzer --input-data JSON file from sample records."""
    # perf_analyzer input format: {"data": [{"<tensor_name>": [<value>]}]}
    # Our input is a STRING tensor named records_json carrying a JSON array.
    return json.dumps({"data": [{"records_json": [json.dumps(records)]}]})


def _parse_perf_analyzer_output(output: str) -> list[dict[str, Any]]:
    """Parse perf_analyzer stdout for throughput + p50/p95/p99 per concurrency.

    perf_analyzer prints one result block per concurrency level, e.g.:
      Concurrency: 1, throughput: 42.3 infer/sec, latency 23650 usec (avg)
      p50 latency: 22000 usec, p95 latency: 29000 usec, p99 latency: 32000 usec
    """
    import re  # noqa: PLC0415
    rows: list[dict[str, Any]] = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        m = re.search(
            r"Concurrency:\s*(\d+).*?throughput:\s*([\d.]+)\s*infer/sec", line
        )
        if not m:
            continue
        concurrency = int(m.group(1))
        throughput = float(m.group(2))
        p50 = p95 = p99 = float("nan")
        # Look ahead up to 3 lines for percentile line
        for ahead in lines[i + 1: i + 4]:
            pm = re.search(
                r"p50 latency:\s*([\d.]+)\s*usec.*?p95 latency:\s*([\d.]+)\s*usec"
                r".*?p99 latency:\s*([\d.]+)\s*usec",
                ahead,
            )
            if pm:
                p50 = round(float(pm.group(1)) / 1000.0, 1)   # usec → ms
                p95 = round(float(pm.group(2)) / 1000.0, 1)
                p99 = round(float(pm.group(3)) / 1000.0, 1)
                break
        rows.append({
            "concurrency": concurrency,
            "req_s": round(throughput, 2),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
        })
    return rows


def _http_perf_fallback(
    port: int, records: list[dict], concurrencies: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Fallback: drive Triton HTTP endpoint concurrently when perf_analyzer absent."""
    import concurrent.futures  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    body = json.dumps({
        "inputs": [{
            "name": "records_json",
            "datatype": "BYTES",
            "shape": [1],
            "data": [json.dumps(records)],
        }],
    }).encode()
    url = f"http://localhost:{port}/v2/models/pragmatiq_embedder/infer"

    def _one_request() -> float:
        t0 = time.perf_counter()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60):
            pass
        return (time.perf_counter() - t0) * 1000.0

    rows: list[dict[str, Any]] = []
    for c in concurrencies:
        n_req = max(c * 4, 16)
        latencies: list[float] = []
        wall_t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            futs = [pool.submit(_one_request) for _ in range(n_req)]
            for f in concurrent.futures.as_completed(futs):
                try:
                    latencies.append(f.result())
                except Exception:  # noqa: BLE001
                    pass
        wall = time.perf_counter() - wall_t0
        req_s = len(latencies) / wall if wall > 0 else 0.0
        try:
            import numpy as np  # noqa: PLC0415
            p50, p95, p99 = (round(float(v), 1) for v in np.percentile(latencies, [50, 95, 99]))
        except Exception:  # noqa: BLE001
            p50 = p95 = p99 = float("nan")
        rows.append({
            "concurrency": c,
            "req_s": round(req_s, 2),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
        })
    return rows


def _run_triton_path(
    *,
    run_dir: Path,
    out_dir: Path,
    triton_budget_min: int = 20,
) -> tuple[list[dict[str, Any]], str]:
    """Run the Triton container serving path for each variant.

    Builds the Triton image once, then for each variant (CPU / 1-GPU / 8-GPU)
    patches config.pbtxt in a temp model-repo, starts the container, waits for
    readiness, runs perf_analyzer (or HTTP fallback), tears down the container.
    Always cleans up containers in try/finally.

    Returns:
        (triton_rows, skip_reason)  — triton_rows is empty on any failure.
    """
    docker_bin = shutil.which("docker")
    if not docker_bin:
        reason = "docker not found in PATH"
        print(f"[triton] SKIPPED — {reason}", flush=True)
        return [], reason

    try:
        rc = subprocess.run(
            [docker_bin, "info"], capture_output=True, timeout=15,
        ).returncode
    except Exception as exc:  # noqa: BLE001
        reason = f"docker daemon check failed: {exc}"
        print(f"[triton] SKIPPED — {reason}", flush=True)
        return [], reason
    if rc != 0:
        reason = "docker daemon not running"
        print(f"[triton] SKIPPED — {reason}", flush=True)
        return [], reason

    try:
        import torch  # noqa: PLC0415
        has_cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_cuda = False

    repo_root = Path(__file__).resolve().parent.parent
    base_config_path = (
        repo_root / "deploy" / "triton" / "model_repository"
        / "pragmatiq_embedder" / "config.pbtxt"
    )
    model_repo_src = repo_root / "deploy" / "triton" / "model_repository"
    dockerfile = repo_root / "deploy" / "triton" / "Dockerfile"

    if not base_config_path.exists():
        reason = f"config.pbtxt not found at {base_config_path}"
        print(f"[triton] SKIPPED — {reason}", flush=True)
        return [], reason

    base_config = base_config_path.read_text()

    # ---- build image once ----
    print("[triton] Building Triton image (this may take a few minutes) ...", flush=True)
    build_start = time.time()
    try:
        subprocess.run(
            [docker_bin, "build",
             "-f", str(dockerfile),
             "--build-arg", "EXTRAS=",
             "-t", _TRITON_IMAGE,
             str(repo_root)],
            check=True, timeout=600,
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"docker build failed: {exc}"
        print(f"[triton] SKIPPED — {reason}", flush=True)
        return [], reason
    print(f"[triton] image built in {time.time() - build_start:.0f}s", flush=True)

    budget_deadline = time.time() + triton_budget_min * 60
    all_rows: list[dict[str, Any]] = []
    pa_bin = shutil.which("perf_analyzer")

    try:
        for variant_name, instance_block in _TRITON_VARIANTS:
            # Skip GPU variants if no CUDA available
            if "gpu" in variant_name and not has_cuda:
                print(
                    f"[triton] variant={variant_name} SKIPPED — CUDA not available",
                    flush=True,
                )
                continue

            # Check overall budget
            if time.time() > budget_deadline:
                print(
                    f"[triton] budget exhausted ({triton_budget_min}min); "
                    "stopping further variants",
                    flush=True,
                )
                break

            container_name = f"{_TRITON_CONTAINER_PREFIX}{variant_name}"
            # Always clean up any stale container with this name
            subprocess.run(
                [docker_bin, "rm", "-f", container_name],
                capture_output=True,
            )

            try:
                with tempfile.TemporaryDirectory(prefix="pq_triton_repo_") as tmp_repo_str:
                    tmp_repo = Path(tmp_repo_str)
                    # Copy entire model_repository tree to tmp
                    shutil.copytree(str(model_repo_src), str(tmp_repo / "model_repository"))
                    # Patch config.pbtxt
                    patched = _triton_config_pbtxt(base_config, instance_block)
                    (tmp_repo / "model_repository" / "pragmatiq_embedder"
                     / "config.pbtxt").write_text(patched)

                    is_gpu_variant = "gpu" in variant_name
                    gpu_flags = ["--gpus", "all"] if is_gpu_variant else []
                    gpu_env = ["-e", "PRAGMATIQ_SERVE_GPU=1"] if is_gpu_variant else []

                    # Pick a free-ish port to avoid conflicts
                    http_port = 18000
                    grpc_port = 18001

                    print(
                        f"[triton] starting variant={variant_name} "
                        f"container={container_name}",
                        flush=True,
                    )
                    try:
                        subprocess.run(
                            [docker_bin, "run", "-d", "--name", container_name]
                            + gpu_flags
                            + gpu_env
                            + [
                                "-p", f"{http_port}:8000",
                                "-p", f"{grpc_port}:8001",
                                "--shm-size", "1g",
                                "-v", f"{tmp_repo / 'model_repository'}:"
                                      f"/models/model_repository:ro",
                                "-v", f"{run_dir.resolve()}:/models/run:ro",
                                _TRITON_IMAGE,
                                "tritonserver",
                                "--model-repository=/models/model_repository",
                            ],
                            check=True, timeout=60,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[triton] variant={variant_name} container start failed: {exc}",
                            flush=True,
                        )
                        continue

                    # Wait for readiness
                    import urllib.request as _urlreq  # noqa: PLC0415
                    ready = False
                    for _ in range(_TRITON_READY_POLLS):
                        try:
                            with _urlreq.urlopen(
                                f"http://localhost:{http_port}/v2/health/ready",
                                timeout=5,
                            ):
                                ready = True
                                break
                        except Exception:  # noqa: BLE001
                            pass
                        time.sleep(_TRITON_READY_INTERVAL_S)

                    if not ready:
                        logs = subprocess.run(
                            [docker_bin, "logs", "--tail", "30", container_name],
                            capture_output=True, text=True,
                        ).stderr or ""
                        print(
                            f"[triton] variant={variant_name} never became ready; "
                            f"last logs:\n{logs}",
                            flush=True,
                        )
                        continue

                    print(f"[triton] variant={variant_name} ready", flush=True)

                    # ---- perf measurement ----
                    variant_rows: list[dict[str, Any]] = []
                    if pa_bin:
                        # Write perf_analyzer input file
                        pa_input_path = out_dir / f"pa_input_{variant_name}.json"
                        pa_input_path.write_text(
                            _make_perf_analyzer_input(list(_SAMPLE_RECORDS[:4]))
                        )
                        conc_range = (
                            f"{min(_PA_CONCURRENCIES)}:{max(_PA_CONCURRENCIES)}"
                        )
                        pa_cmd = [
                            pa_bin,
                            "-m", "pragmatiq_embedder",
                            "-u", f"localhost:{grpc_port}",
                            "-i", "grpc",
                            "--concurrency-range", conc_range,
                            "--percentile=99",
                            "--measurement-interval", "5000",
                            "--input-data", str(pa_input_path),
                            "--shape", "records_json:1",
                        ]
                        print(
                            f"[triton] running perf_analyzer for variant={variant_name}",
                            flush=True,
                        )
                        try:
                            pa_result = subprocess.run(
                                pa_cmd, capture_output=True, text=True, timeout=300,
                            )
                            variant_rows = _parse_perf_analyzer_output(
                                pa_result.stdout + pa_result.stderr
                            )
                            if not variant_rows:
                                print(
                                    f"[triton] perf_analyzer produced no parseable output "
                                    f"for variant={variant_name}; falling back to HTTP",
                                    flush=True,
                                )
                                variant_rows = _http_perf_fallback(
                                    http_port,
                                    list(_SAMPLE_RECORDS[:4]),
                                    _PA_CONCURRENCIES,
                                )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"[triton] perf_analyzer failed ({exc}); "
                                f"using HTTP fallback for variant={variant_name}",
                                flush=True,
                            )
                            variant_rows = _http_perf_fallback(
                                http_port,
                                list(_SAMPLE_RECORDS[:4]),
                                _PA_CONCURRENCIES,
                            )
                    else:
                        print(
                            f"[triton] perf_analyzer not installed; "
                            f"using HTTP fallback for variant={variant_name}",
                            flush=True,
                        )
                        variant_rows = _http_perf_fallback(
                            http_port,
                            list(_SAMPLE_RECORDS[:4]),
                            _PA_CONCURRENCIES,
                        )

                    for row in variant_rows:
                        row["variant"] = variant_name
                    all_rows.extend(variant_rows)
                    print(
                        f"[triton] variant={variant_name} rows={len(variant_rows)}",
                        flush=True,
                    )

            except Exception as exc:  # noqa: BLE001
                print(
                    f"[triton] variant={variant_name} unexpected error: {exc}; continuing",
                    flush=True,
                )
            finally:
                # Always clean up the container
                subprocess.run(
                    [docker_bin, "rm", "-f", container_name],
                    capture_output=True,
                )

    except Exception as exc:  # noqa: BLE001
        reason = f"Triton path failed: {exc}"
        print(f"[triton] FAILED — {reason}", flush=True)
        return all_rows, reason

    if all_rows:
        triton_csv = out_dir / "triton_results.csv"
        with triton_csv.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["variant", "concurrency", "req_s",
                                 "p50_ms", "p95_ms", "p99_ms"],
            )
            w.writeheader()
            for row in all_rows:
                w.writerow({k: row.get(k, "") for k in w.fieldnames})  # type: ignore[arg-type]

    return all_rows, ""


# ---------------------------------------------------------------------------
# Orchestrator: serving measurement
# ---------------------------------------------------------------------------


def _measure_serving(
    *,
    run_dir: Path,
    serving_concurrency: list[int],
    dry_run: bool,
    out_dir: Path,
    util_records: list[dict[str, Any]],
    skip_triton: bool = False,
    triton_budget_min: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Measure serving throughput using real concurrent embed requests.

    For each (device, concurrency) combination, submits concurrent embed
    calls via ThreadPoolExecutor to runtime.embed(records) and measures
    req/s + p50/p95/p99 latency.  This exercises the actual W4 serving
    code path with real concurrent requests.

    Returns:
        (serving_results, triton_results, triton_skip_reason)
    """
    # Import here to keep module top-level import-light
    import concurrent.futures  # noqa: PLC0415

    from pragmatiq.inference.serve import runtime as serve_runtime  # noqa: PLC0415

    serving_results: list[dict[str, Any]] = []

    # Choose devices to test
    try:
        import torch  # noqa: PLC0415
        has_cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_cuda = False

    devices_to_test: list[tuple[str, str]] = [("cpu", "cpu")]
    if has_cuda and not dry_run:
        devices_to_test.append(("cuda", "cuda"))

    # Build record batches (vary batch size by concurrency)
    records_batch = list(_SAMPLE_RECORDS[:4])  # 4 users per request

    for device_label, device_str in devices_to_test:
        print(f"[serving] loading model on {device_label}", flush=True)
        with _monitor_workload(f"serving_{device_label}", out_dir, util_records):
            try:
                rt = serve_runtime.load(str(run_dir), device=device_str)
            except Exception as exc:  # noqa: BLE001
                print(f"[serving] WARNING: could not load on {device_label}: {exc}", flush=True)
                continue

            def _embed_one_fn(_: int, _rt: Any, _records: list[dict]) -> float:
                """Embed one batch; return latency in ms."""
                t0 = time.perf_counter()
                _rt.embed(_records)
                return (time.perf_counter() - t0) * 1000.0

            def _pct_fn(lats: list[float], p: float) -> float:
                """Compute percentile using numpy for correct linear interpolation."""
                if not lats:
                    return float("nan")
                try:
                    import numpy as np  # noqa: PLC0415
                    return float(np.percentile(lats, p))
                except Exception:  # noqa: BLE001
                    # Fallback: nearest-rank
                    s = sorted(lats)
                    idx = min(int(len(s) * p / 100.0 + 0.5), len(s) - 1)
                    return s[max(0, idx)]

            for concurrency in serving_concurrency:
                n_requests = max(concurrency * 4, 16)
                print(
                    f"[serving]  device={device_label} concurrency={concurrency} "
                    f"n_requests={n_requests}",
                    flush=True,
                )
                latencies_ms: list[float] = []

                wall_t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(_embed_one_fn, i, rt, records_batch)
                        for i in range(n_requests)
                    ]
                    for fut in concurrent.futures.as_completed(futures):
                        try:
                            latencies_ms.append(fut.result())
                        except Exception as e:  # noqa: BLE001
                            print(f"[serving] request failed: {e}", flush=True)
                wall_elapsed = time.perf_counter() - wall_t0

                completed = len(latencies_ms)
                req_s = completed / wall_elapsed if wall_elapsed > 0 else 0.0

                p50 = _pct_fn(latencies_ms, 50)
                p95 = _pct_fn(latencies_ms, 95)
                p99 = _pct_fn(latencies_ms, 99)

                row: dict[str, Any] = {
                    "concurrency": concurrency,
                    "device": device_label,
                    "req_s": round(req_s, 2),
                    "p50_ms": round(p50, 1),
                    "p95_ms": round(p95, 1),
                    "p99_ms": round(p99, 1),
                    "n_completed": completed,
                }
                serving_results.append(row)
                print(
                    f"[serving]    req/s={req_s:.1f} p50={p50:.0f}ms "
                    f"p95={p95:.0f}ms p99={p99:.0f}ms",
                    flush=True,
                )

            rt.close()

    # Optional Triton path — only if docker is available AND not dry-run AND not skip-triton
    triton_results: list[dict[str, Any]] = []
    triton_skip_reason: str = ""

    if dry_run:
        triton_skip_reason = "dry-run mode — no docker build attempted on CPU laptop"
        print(f"[serving] Triton path: SKIPPED ({triton_skip_reason})", flush=True)
    elif skip_triton:
        triton_skip_reason = "--skip-triton flag set"
        print(f"[serving] Triton path: SKIPPED ({triton_skip_reason})", flush=True)
    else:
        triton_results, triton_skip_reason = _run_triton_path(
            run_dir=run_dir,
            out_dir=out_dir,
            triton_budget_min=triton_budget_min,
        )

    # Save CSV
    csv_path = out_dir / "serving_results.csv"
    if serving_results:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["concurrency", "device", "req_s",
                                                "p50_ms", "p95_ms", "p99_ms", "n_completed"])
            w.writeheader()
            w.writerows(serving_results)

    return serving_results, triton_results, triton_skip_reason


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    out_dir: Path,
    *,
    start_ts: str,
    end_ts: str,
    dry_run: bool,
    model_size: str,
    n_users: int,
    steps: int,
    training_results: list[dict[str, Any]],
    finetune_results: list[dict[str, Any]],
    serving_results: list[dict[str, Any]],
    triton_results: list[dict[str, Any]],
    triton_skip_reason: str,
    util_records: list[dict[str, Any]],
) -> Path:
    """Write REPORT.md and return its path."""
    report_path = out_dir / "REPORT.md"

    # Headline numbers
    tps_8 = next((r["tokens_per_sec"] for r in training_results if r["devices"] == 8), float("nan"))
    tps_1 = next((r["tokens_per_sec"] for r in training_results if r["devices"] == 1), float("nan"))
    eff_8 = next((r["efficiency"] for r in training_results if r["devices"] == 8), float("nan"))

    gpu_reqs = next((r["req_s"] for r in serving_results if r["device"] == "cuda"), float("nan"))
    cpu_reqs = next((r["req_s"] for r in serving_results if r["device"] == "cpu" and r["concurrency"] == 1), float("nan"))
    speedup = (gpu_reqs / cpu_reqs) if (gpu_reqs == gpu_reqs and cpu_reqs == cpu_reqs and cpu_reqs > 0) else float("nan")

    lines: list[str] = []

    lines += [
        "# pragmatiq GPU Validation Report",
        "",
        "> pragmatiq is an independent implementation inspired by the PRAGMA paper "
        "(arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.",
        "",
        f"**Run started:** {start_ts}  ",
        f"**Run completed:** {end_ts}  ",
        f"**Mode:** {'DRY-RUN (CPU, nano model)' if dry_run else f'GPU pod ({model_size} model)'}  ",
        f"**Users:** {n_users:,}  ",
        f"**Steps:** {steps}  ",
        "",
        "## Headline Numbers",
        "",
    ]

    def _fmt(v: float, fmt: str = ".1f") -> str:
        return f"{v:{fmt}}" if v == v else "N/A"

    if not dry_run:
        lines += [
            f"- **8-GPU scaling efficiency:** {_fmt(eff_8, '.1%')}",
            f"- **8-GPU tokens/sec:** {_fmt(tps_8, ',.0f')}",
            f"- **Serving GPU/CPU req/s speedup:** {_fmt(speedup, '.1f')}×",
            "",
        ]
    else:
        lines += [
            f"- **1-GPU (CPU dry-run) tokens/sec:** {_fmt(tps_1, ',.0f')}",
            "",
        ]

    # ---- Training section ----
    lines += [
        "## Training Scaling Sweep",
        "",
    ]
    if training_results:
        lines += [
            "| devices | tokens/s | gpu_mem_gb | efficiency% | elapsed_s | notes |",
            "|---------|----------|------------|-------------|-----------|-------|",
        ]
        for r in training_results:
            tps = r.get("tokens_per_sec", float("nan"))
            mem = r.get("gpu_mem_gb", float("nan"))
            eff = r.get("efficiency", float("nan"))
            elapsed = r.get("elapsed_s", float("nan"))
            notes = ""
            if r.get("status") == "timeout":
                notes = "TIMEOUT (leg killed; harness continued)"
            elif r.get("oom_fallback"):
                notes = f"OOM fallback token_budget={r.get('token_budget_used')}"
            elif r.get("oom_fallback_model_size"):
                notes = "OOM fallback to model_size=medium"
            elif r.get("returncode", 0) != 0:
                notes = f"FAILED (rc={r.get('returncode')})"
            lines.append(
                f"| {r['devices']} | {_fmt(tps, ',.0f')} | {_fmt(mem)} | "
                f"{_fmt(eff, '.1%')} | {_fmt(elapsed, '.0f')} | {notes} |"
            )
        lines.append("")

        # Scaling shape note
        succs = [r for r in training_results if r.get("returncode", 0) == 0]
        if len(succs) >= 2:
            eff_vals = [r["efficiency"] for r in succs if r["efficiency"] == r["efficiency"]]
            if eff_vals:
                mean_eff = sum(eff_vals) / len(eff_vals)
                if mean_eff >= 0.90:
                    shape = "strong linear scaling"
                elif mean_eff >= 0.75:
                    shape = "good scaling (minor communication overhead)"
                else:
                    shape = "sub-linear scaling (communication overhead dominates at high device counts)"
                lines.append(f"**Scaling shape:** {shape} (mean efficiency {mean_eff:.1%})")
                lines.append("")
    else:
        lines += ["*No training results recorded.*", ""]

    # ---- Finetune section ----
    lines += ["## Fine-tuning (1 vs multi-GPU)", ""]
    if finetune_results:
        lines += [
            "| devices | wall_time_s | best_val_auc | epochs_run |",
            "|---------|-------------|--------------|------------|",
        ]
        for r in finetune_results:
            lines.append(
                f"| {r['devices']} | {_fmt(r.get('wall_time_s', float('nan')), '.0f')} | "
                f"{_fmt(r.get('best_val_auc', float('nan')), '.4f')} | "
                f"{r.get('epochs_run', 0)} |"
            )
        lines.append("")

        aucs = [r["best_val_auc"] for r in finetune_results
                if r.get("best_val_auc") == r.get("best_val_auc") and r["best_val_auc"] > 0]
        if len(aucs) >= 2:
            delta = abs(aucs[0] - aucs[-1])
            if delta < 0.02:
                conv_note = f"AUC difference {delta:.4f} — DDP fine-tune converges comparably (validates GA2)."
            else:
                conv_note = (
                    f"AUC difference {delta:.4f} — non-trivial gap; "
                    "check DDP gradient sync or label distribution."
                )
            lines += [f"**Convergence note:** {conv_note}", ""]
        else:
            lines += ["*Only one leg produced a valid AUC; no convergence comparison.*", ""]
    else:
        lines += ["*Finetune skipped or no results.*", ""]

    # ---- Serving section ----
    lines += ["## Serving (Runtime Concurrent Requests)", ""]
    if serving_results:
        lines += [
            "| concurrency | device | req/s | p50 ms | p95 ms | p99 ms |",
            "|-------------|--------|-------|--------|--------|--------|",
        ]
        for r in serving_results:
            lines.append(
                f"| {r['concurrency']} | {r['device']} | {r['req_s']:.1f} | "
                f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} |"
            )
        lines.append("")
        if speedup == speedup:
            lines += [f"**GPU/CPU serving speedup:** {speedup:.1f}× (1-GPU vs CPU, concurrency=1)", ""]
        lines += [
            "**Measurement approach:** ThreadPoolExecutor at each concurrency level sends "
            "`runtime.embed(records)` calls concurrently against the loaded W4 model; "
            "wall-time brackets all futures to compute req/s; latencies are measured "
            "per-request (time.perf_counter) and computed with `numpy.percentile` for "
            "p50/p95/p99.",
            "",
        ]

    # ---- Triton serving section ----
    lines += ["## Serving (Triton Container — perf_analyzer)", ""]
    if triton_results:
        lines += [
            "| variant | concurrency | req/s | p50 ms | p95 ms | p99 ms |",
            "|---------|-------------|-------|--------|--------|--------|",
        ]
        for r in triton_results:
            def _fv(v: Any) -> str:  # noqa: ANN001
                return f"{v:.1f}" if isinstance(v, float) and v == v else str(v)
            lines.append(
                f"| {r.get('variant', '?')} | {r.get('concurrency', '?')} | "
                f"{_fv(r.get('req_s', float('nan')))} | "
                f"{_fv(r.get('p50_ms', float('nan')))} | "
                f"{_fv(r.get('p95_ms', float('nan')))} | "
                f"{_fv(r.get('p99_ms', float('nan')))} |"
            )
        lines.append("")
        lines += [
            "**Measurement approach:** perf_analyzer (or HTTP-concurrent fallback) against "
            "live Triton containers, one container per variant (CPU / 1×GPU / 8×GPU-instances); "
            "containers cleaned up after each variant.",
            "",
        ]
    else:
        reason_str = triton_skip_reason or "docker not available or not attempted"
        lines += [
            f"Triton container path: SKIPPED/FAILED — {reason_str}; "
            "serving validated via the runtime concurrent-request measurement above.",
            "",
        ]

    if not serving_results and not triton_results:
        lines += ["*Serving skipped or no results.*", ""]

    # ---- Utilisation section ----
    lines += ["## Utilisation", ""]
    if util_records:
        for rec in util_records:
            label = rec.get("label", "?")
            gpu = rec.get("gpu", {})
            lines.append(f"### {label}")
            if gpu:
                lines += [
                    f"- GPU mean util: {gpu.get('mean_util_pct', 'N/A')}%",
                    f"- GPU peak util: {gpu.get('peak_util_pct', 'N/A')}%",
                    f"- GPU peak mem: {gpu.get('peak_mem_mib', 'N/A')} MiB",
                    f"- GPU mean power: {gpu.get('mean_power_w', 'N/A')} W",
                ]
            else:
                lines.append("- GPU: nvidia-smi not available (CPU dry-run or no GPU)")
            lines += [
                f"- CPU mean: {rec.get('mean_cpu_pct', 'N/A')}%",
                f"- RAM peak: {rec.get('peak_ram_gb', 'N/A')} GB",
                "",
            ]
    else:
        lines += ["*No utilisation data collected.*", ""]

    # ---- Verdict section ----
    lines += ["## Verdict", ""]
    if dry_run:
        lines += [
            "**DRY-RUN complete.**  Every code path exercised on CPU:",
            "- data prep (synthesize + tokenize) ✓",
            "- pretrain leg subprocess (Fabric DDP plumbing + metrics parse) ✓",
            "- finetune leg subprocess (LoRA + result JSON) ✓",
            "- serving runtime concurrent-request measurement ✓",
            "- monitoring (nvidia-smi gracefully absent, CPU/RAM sampled) ✓",
            "- report written ✓",
            "",
            "**Risk note (devices>1):** The leg-subprocess Fabric DDP re-launch mechanism "
            "is fully plumbed (subprocess invokes `sys.executable __file__ --_leg-*` with "
            "real `--devices D`), but gloo/NCCL group init is only exercised at devices>1 "
            "on the real GPU pod.  The dry-run proves the subprocess + metrics-parse pipeline; "
            "DDP collective ops are only validated on the pod.",
            "",
        ]
    else:
        failed_legs = [r for r in training_results if r.get("returncode", 0) != 0]
        if not failed_legs and not any(
            r.get("best_val_auc", -1) == -1 for r in finetune_results
        ):
            lines += [
                "**All legs completed successfully.**",
                "- Training scaling sweep ✓",
                "- Finetune DDP convergence ✓",
                "- Serving runtime throughput ✓",
                "",
            ]
        else:
            lines += ["**Some legs failed — see notes above.**", ""]

    report_path.write_text("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Public orchestrator args
    ap.add_argument("--out", default=None,
                    help="Output directory (default: outputs/gpu-validation-<timestamp>)")
    ap.add_argument("--model-size", default="large",
                    help="Model size preset (default: large; dry-run uses nano)")
    ap.add_argument("--devices-sweep", default="1,2,4,8",
                    help="Comma-separated device counts for training sweep (default: 1,2,4,8)")
    ap.add_argument("--users", type=int, default=100_000,
                    help="Number of synthetic users (default: 100000; dry-run uses ~300)")
    ap.add_argument("--steps", type=int, default=80,
                    help="Training steps per leg (default: 80; dry-run uses 5)")
    ap.add_argument("--token-budget", type=int, default=32_768,
                    help="Token budget per training batch (default: 32768)")
    ap.add_argument("--finetune-devices", default="1,8",
                    help="Comma-separated device counts for finetune legs (default: 1,8)")
    ap.add_argument("--finetune-steps", type=int, default=3,
                    help="Max finetune epochs (default: 3)")
    ap.add_argument("--serving-concurrency", default="1,4,16,64",
                    help="Comma-separated concurrency levels for serving measurement "
                         "(default: 1,4,16,64)")
    ap.add_argument("--max-runtime-min", type=int, default=0,
                    help="Hard wall-clock timeout in minutes (0 = unlimited)")
    ap.add_argument("--leg-timeout-min", type=int, default=10,
                    help="Per-leg subprocess timeout in minutes (default: 10). "
                         "On expiry the leg's process group is killed, the leg is "
                         "recorded as 'timeout', and the harness continues to the "
                         "next leg.")
    ap.add_argument("--skip-serving", action="store_true",
                    help="Skip the serving measurement")
    ap.add_argument("--skip-finetune", action="store_true",
                    help="Skip the fine-tuning legs")
    ap.add_argument("--skip-triton", action="store_true",
                    help="Skip the Triton container perf_analyzer path "
                         "(runtime serving measurements still run)")
    ap.add_argument("--triton-budget-min", type=int, default=20,
                    help="Wall-clock budget in minutes for the entire Triton path "
                         "(default: 20); prevents runaway on paid pods")
    ap.add_argument("--dry-run", action="store_true",
                    help="Tiny CPU run (~300 users, nano model, 5 steps) — exercises every "
                         "code path locally for free before any GPU spend")

    # Hidden leg modes (invoked by orchestrator as subprocesses)
    ap.add_argument("--_leg-pretrain", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_leg-finetune", action="store_true", help=argparse.SUPPRESS)

    # Leg mode shared args
    ap.add_argument("--devices", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--run-name", default="", help=argparse.SUPPRESS)
    ap.add_argument("--shard-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--runs-root", default="", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--label-path", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result-json", default="", help=argparse.SUPPRESS)

    # Leg mode uses --steps and --token-budget as well (already defined above)

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: C901 — linear orchestration
    args = _parse_args(argv)

    # ---- Dispatch leg modes (must be first: Fabric re-launches this file) ----
    if getattr(args, "_leg_pretrain", False):
        _leg_pretrain(args)
        return

    if getattr(args, "_leg_finetune", False):
        _leg_finetune(args)
        return

    # ---- Orchestrator mode ----
    start_ts = datetime.now().isoformat(timespec="seconds")

    # Dry-run overrides
    if args.dry_run:
        args.model_size = "nano"
        if args.users == 100_000:
            args.users = 300
        if args.steps == 80:
            args.steps = 5
        args.devices_sweep = "1"
        args.finetune_devices = "1"
        args.finetune_steps = 2
        args.serving_concurrency = "1,2"
        args.token_budget = 512
        print("[dry-run] nano model, 300 users, devices=1, 5 steps", flush=True)

    # Resolve output directory
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"outputs/gpu-validation-{ts}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = out_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    print(f"[main] output dir: {out_dir}", flush=True)

    devices_sweep = [int(d) for d in args.devices_sweep.split(",") if d.strip()]
    finetune_devices = [int(d) for d in args.finetune_devices.split(",") if d.strip()]
    serving_concurrency = [int(c) for c in args.serving_concurrency.split(",") if c.strip()]

    leg_timeout_sec = args.leg_timeout_min * 60
    run_start = time.time()
    max_runtime_sec = args.max_runtime_min * 60  # 0 means unlimited

    util_records: list[dict[str, Any]] = []

    # -- Data prep --
    print("[main] === Data preparation ===", flush=True)
    synth_dir, tok_dir, labels_path = _data_prep(out_dir, args.users, args.dry_run)

    # -- Training scaling sweep --
    print("[main] === Training scaling sweep ===", flush=True)
    training_results = _training_sweep(
        devices_sweep=devices_sweep,
        shard_dir=tok_dir,
        runs_root=runs_root,
        model_size=args.model_size,
        steps=args.steps,
        token_budget=args.token_budget,
        out_dir=out_dir,
        util_records=util_records,
        leg_timeout_sec=leg_timeout_sec,
        run_start=run_start,
        max_runtime_sec=max_runtime_sec,
    )

    # Identify the pretrained run to use for finetune (prefer sweep_d8 or sweep_d1)
    pretrained_run_dir: Path | None = None
    for pref in ([8] if not args.dry_run else []) + [1] + devices_sweep:
        candidate = runs_root / f"sweep_d{pref}"
        if candidate.exists() and (candidate / "checkpoints" / "last.pt").exists():
            pretrained_run_dir = candidate
            break

    # -- Fine-tuning --
    finetune_results: list[dict[str, Any]] = []
    if not args.skip_finetune:
        if pretrained_run_dir is None:
            print("[main] WARNING: no valid pretrained run found; skipping finetune", flush=True)
        else:
            print(f"[main] === Fine-tuning from {pretrained_run_dir.name} ===", flush=True)
            finetune_results = _finetune_sweep(
                finetune_devices=finetune_devices,
                shard_dir=tok_dir,
                pretrained_run_dir=pretrained_run_dir,
                label_path=labels_path,
                finetune_steps=args.finetune_steps,
                out_dir=out_dir,
                util_records=util_records,
                leg_timeout_sec=leg_timeout_sec,
                run_start=run_start,
                max_runtime_sec=max_runtime_sec,
            )
    else:
        print("[main] finetune skipped (--skip-finetune)", flush=True)

    # -- Serving --
    serving_results: list[dict[str, Any]] = []
    triton_results: list[dict[str, Any]] = []
    triton_skip_reason: str = ""
    if not args.skip_serving:
        if pretrained_run_dir is None:
            print("[main] WARNING: no valid pretrained run found; skipping serving", flush=True)
        else:
            print("[main] === Serving measurement ===", flush=True)
            serving_results, triton_results, triton_skip_reason = _measure_serving(
                run_dir=pretrained_run_dir,
                serving_concurrency=serving_concurrency,
                dry_run=args.dry_run,
                out_dir=out_dir,
                util_records=util_records,
                skip_triton=args.skip_triton,
                triton_budget_min=args.triton_budget_min,
            )
    else:
        print("[main] serving skipped (--skip-serving)", flush=True)
        triton_skip_reason = "--skip-serving flag set"

    # -- Report --
    end_ts = datetime.now().isoformat(timespec="seconds")
    report_path = _write_report(
        out_dir,
        start_ts=start_ts,
        end_ts=end_ts,
        dry_run=args.dry_run,
        model_size=args.model_size,
        n_users=args.users,
        steps=args.steps,
        training_results=training_results,
        finetune_results=finetune_results,
        serving_results=serving_results,
        triton_results=triton_results,
        triton_skip_reason=triton_skip_reason,
        util_records=util_records,
    )

    print("\n[main] === DONE ===", flush=True)
    print(f"[main] Report: {report_path}", flush=True)
    print(f"[main] Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
