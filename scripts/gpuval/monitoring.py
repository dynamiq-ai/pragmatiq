"""GPU/CPU utilisation sampling around each validation leg (nvidia-smi + psutil/proc)."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


class _NvidiaSampler:
    """Background sampler for nvidia-smi GPU metrics.

    Writes one CSV row per second.  No-ops when nvidia-smi is absent.
    """

    #: Consecutive sampling failures before a single warning is printed.
    _FAIL_LOG_AFTER = 3

    def __init__(self, csv_path: Path, interval_s: float = 1.0) -> None:
        self._csv = csv_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cmd = shutil.which("nvidia-smi")
        self._interval_s = interval_s

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

    def _sample_cmd(self) -> list[str]:
        """nvidia-smi invocation for a single snapshot.

        Deliberately no ``-l``/``--loop`` flag: loop mode never exits, so under
        ``subprocess.run(..., timeout=5)`` every sample would raise
        TimeoutExpired and nothing would ever be recorded.  The sampler thread
        itself provides the cadence via ``interval_s``.
        """
        assert self._cmd is not None
        return [
            self._cmd,
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]

    def _loop(self) -> None:
        fields = ["index", "utilization.gpu [%]", "memory.used [MiB]",
                  "memory.total [MiB]", "power.draw [W]"]
        wrote_header = not self._csv.exists()
        failures = 0
        warned = False
        with self._csv.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if wrote_header:
                writer.writerow(["ts"] + fields)
            while not self._stop.is_set():
                try:
                    result = subprocess.run(
                        self._sample_cmd(),
                        capture_output=True, text=True, timeout=5,
                    )
                    ts = datetime.utcnow().isoformat()
                    for line in result.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) == len(fields):
                            writer.writerow([ts] + parts)
                    fh.flush()
                    failures = 0
                except Exception as exc:  # noqa: BLE001 — sampling must never kill the run
                    failures += 1
                    if failures >= self._FAIL_LOG_AFTER and not warned:
                        warned = True
                        print(
                            f"[nvidia-smi] {failures} consecutive sampling failures "
                            f"(last: {exc!r}); GPU utilisation will be missing "
                            "from the report",
                            flush=True,
                        )
                self._stop.wait(self._interval_s)

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
