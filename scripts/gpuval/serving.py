"""Serving throughput: concurrent runtime.embed requests per device, plus the optional Triton container path."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from . import REPO_ROOT
from .monitoring import _monitor_workload

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


_TRITON_READY_POLLS = 60


_TRITON_READY_INTERVAL_S = 3


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

    repo_root = REPO_ROOT
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
                    # Serving is GPU-first; the CPU variant pins the backend explicitly.
                    gpu_env = [] if is_gpu_variant else ["-e", "PRAGMATIQ_SERVE_CPU=1"]

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

            # Warm up before measuring: the first requests on a device pay for
            # kernel/allocator initialisation and would otherwise dominate p99 and
            # the concurrency-1 throughput.
            for _ in range(4):
                try:
                    _embed_one_fn(-1, rt, records_batch)
                except Exception as e:  # noqa: BLE001
                    print(f"[serving] warmup request failed: {e}", flush=True)
            for concurrency in serving_concurrency:
                n_requests = max(concurrency * 4, 64)
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
