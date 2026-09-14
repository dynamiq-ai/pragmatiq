#!/usr/bin/env python3
"""GPU validation harness for pragmatiq — the entry point a pod runs.

Runs the measurement sweep on a GPU pod and writes ``REPORT.md`` plus one
JSON evidence file (``gpu-validation-<tag>.json``: run metadata, every leg's
numbers, and the acceptance table with pass/fail). Legs:

- pretrain scaling sweep over ``--devices-sweep`` (tokens/s, peak VRAM, DDP efficiency)
- LoRA fine-tune legs (epochs, wall time, val ROC-AUC, per-epoch tok/s, GPU util)
- serving: concurrent ``runtime.embed`` requests on CPU and CUDA, optional Triton
  container path, request-cap check (chunked == whole)
- flash-attn ≡ SDPA equivalence, attention-backend report
- bf16 vs fp32: embed throughput and probe ROC-AUC delta
- ONNX export on the pod, quickstart wall time

The implementation lives in ``scripts/gpuval/``; this file only parses flags and
orchestrates. ``--smoke`` (alias ``--dry-run``) exercises every leg on a CPU with
a nano model in a few minutes so the harness can be proven before any GPU spend.

Usage (smoke, local, free):
    .venv/bin/python scripts/validate_gpu.py --smoke --out /tmp/gpuval-smoke

Usage (real pod):
    python scripts/validate_gpu.py --out outputs/gpu-validation-$(date +%Y%m%d) --tag a100-rc1

Hidden leg modes invoked as subprocesses so Lightning Fabric DDP re-launch
works (Fabric re-launches THIS file across d processes):
    --_leg-pretrain   --devices D --run-name NAME --shard-dir DIR
                      --runs-root DIR --model-size S --steps N --token-budget N
    --_leg-finetune   --devices D --run-dir DIR --shard-dir DIR
                      --label-path PATH --runs-root DIR --steps N
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess  # noqa: F401  (re-exported for tests that patch the leg runner's subprocess)
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpuval import extras  # noqa: E402
from gpuval.flash_check import _check_flash_vs_sdpa  # noqa: E402
from gpuval.legs import (  # noqa: E402,F401  (re-exports keep the unit tests' entry stable)
    _count_failed_legs,
    _data_prep,
    _finetune_leg_failed,
    _is_rank_zero,
    _leg_finetune,
    _leg_pretrain,
    _run_leg_with_timeout,
    _subsample_labels,
)
from gpuval.monitoring import _monitor_workload, _NvidiaSampler  # noqa: E402,F401
from gpuval.report import _write_report, write_evidence_json, write_readme_block  # noqa: E402
from gpuval.serving import _SAMPLE_RECORDS, _measure_serving  # noqa: E402
from gpuval.training import _finetune_sweep, _leg_efficiency, _training_sweep  # noqa: E402,F401


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default=None,
                    help="Output directory (default: outputs/gpu-validation-<timestamp>)")
    ap.add_argument("--tag", default=None,
                    help="Evidence tag: writes gpu-validation-<tag>.json (default: the timestamp)")
    ap.add_argument("--model-size", default="large",
                    help="Model size preset (default: large; --smoke uses nano)")
    ap.add_argument("--devices-sweep", default="1,2,4,8",
                    help="Comma-separated device counts for the training sweep (default: 1,2,4,8)")
    ap.add_argument("--users", type=int, default=100_000,
                    help="Number of synthetic users (default: 100000; --smoke uses 300)")
    ap.add_argument("--steps", type=int, default=80,
                    help="Training steps per leg (default: 80; --smoke uses 20)")
    ap.add_argument("--token-budget", type=int, default=32_768,
                    help="Token budget per training batch (default: 32768)")
    ap.add_argument("--finetune-devices", default="1,8",
                    help="Comma-separated device counts for the fine-tune legs (default: 1,8)")
    ap.add_argument("--finetune-max-users", type=int, default=2500, metavar="N",
                    help="Cap the fine-tune legs' label table via a seeded stratified "
                         "subsample (0 = use every label).")
    ap.add_argument("--finetune-token-budget", type=int, default=0, metavar="N",
                    help="Per-forward token budget for the fine-tune legs; 0 (default) lets the "
                         "fine-tuner size it from the device memory (bf16 autocast on CUDA).")
    ap.add_argument("--finetune-steps", type=int, default=3,
                    help="Max fine-tune epochs (default: 3)")
    ap.add_argument("--serving-concurrency", default="1,4,16,64",
                    help="Comma-separated concurrency levels for the serving measurement")
    ap.add_argument("--max-runtime-min", type=int, default=0,
                    help="Hard wall-clock timeout in minutes (0 = unlimited)")
    ap.add_argument("--leg-timeout-min", type=int, default=10,
                    help="Per-leg subprocess timeout in minutes (default: 10)")
    ap.add_argument("--finetune-timeout-min", type=int, default=None,
                    help="Per-leg timeout in minutes for fine-tune legs (default: 45)")
    ap.add_argument("--nccl-safe", default="on", choices=["on", "off"],
                    help="'on' (default): NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 for "
                         "DDP legs (socket transport; reliable on RunPod, a conservative lower bound "
                         "on tok/s). 'off': attempt NVLink/P2P/SHM transport.")
    ap.add_argument("--skip-serving", action="store_true", help="Skip the serving measurement")
    ap.add_argument("--skip-finetune", action="store_true", help="Skip the fine-tuning legs")
    ap.add_argument("--skip-triton", action="store_true",
                    help="Skip the Triton container perf_analyzer path")
    ap.add_argument("--skip-export", action="store_true", help="Skip the ONNX export leg")
    ap.add_argument("--skip-precision", action="store_true",
                    help="Skip the bf16-vs-fp32 embed/probe comparison")
    ap.add_argument("--skip-quickstart", action="store_true", help="Skip the quickstart timing legs")
    ap.add_argument("--full", action="store_true",
                    help="Also time the default `pragmatiq quickstart` (50k users, 400 steps)")
    ap.add_argument("--triton-budget-min", type=int, default=20,
                    help="Wall-clock budget in minutes for the entire Triton path (default: 20)")
    ap.add_argument("--smoke", "--dry-run", dest="dry_run", action="store_true",
                    help="Tiny CPU run (300 users, nano model, 20 steps, 4 serving requests) — "
                         "exercises every leg locally for free before any GPU spend")
    ap.add_argument("--skip-flash-check", action="store_true",
                    help="Skip the flash-attn ≡ SDPA numeric equivalence check")
    ap.add_argument("--render-json", default=None, metavar="JSON",
                    help="Render an existing validation JSON into --write-readme and exit (no run)")
    ap.add_argument("--write-readme", default=None, metavar="README",
                    help="After the run, replace the <!-- GPU_VALIDATION_RESULTS --> block of this "
                         "README with the rendered results")

    # Hidden leg modes (invoked by the orchestrator as subprocesses)
    ap.add_argument("--_leg-pretrain", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_leg-finetune", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--devices", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--run-name", default="", help=argparse.SUPPRESS)
    ap.add_argument("--shard-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--runs-root", default="", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--label-path", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result-json", default="", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def _run_meta(args: argparse.Namespace, tag: str, start_ts: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "tag": tag, "started": start_ts, "mode": "smoke" if args.dry_run else "gpu",
        "model_size": args.model_size, "n_users": args.users, "steps": args.steps,
        "token_budget": args.token_budget, "nccl_safe": args.nccl_safe,
        "python": platform.python_version(), "host": platform.node(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        meta["gpu_count"] = torch.cuda.device_count()
        meta["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # noqa: BLE001
        meta["torch_error"] = str(exc)
    try:
        import flash_attn

        meta["flash_attn"] = flash_attn.__version__
    except Exception:  # noqa: BLE001
        meta["flash_attn"] = None
    try:
        import pragmatiq

        meta["pragmatiq"] = pragmatiq.__version__
    except Exception:  # noqa: BLE001
        pass
    return meta


def main(argv: list[str] | None = None) -> None:  # noqa: C901 — linear orchestration
    args = _parse_args(argv)
    if args.render_json:
        target = args.write_readme or "README.md"
        ok = write_readme_block(args.render_json, target)
        print(f"[main] README block {'written' if ok else 'marker missing; not written'}: {target}")
        return

    # Leg modes first: Fabric re-launches this file per rank.
    if getattr(args, "_leg_pretrain", False):
        _leg_pretrain(args)
        return
    if getattr(args, "_leg_finetune", False):
        _leg_finetune(args)
        return

    start_ts = datetime.now().isoformat(timespec="seconds")
    if args.dry_run:
        args.model_size = "nano"
        if args.users == 100_000:
            args.users = 300
        if args.steps == 80:
            args.steps = 20
        args.devices_sweep = "1"
        args.finetune_devices = "1"
        args.finetune_steps = 2
        args.serving_concurrency = "1,2"
        args.token_budget = 512
        print("[smoke] nano model, 300 users, devices=1, 20 steps", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        args.out = f"outputs/gpu-validation-{ts}"
    tag = args.tag or ts
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = out_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    print(f"[main] output dir: {out_dir} (tag {tag})", flush=True)

    devices_sweep = [int(d) for d in args.devices_sweep.split(",") if d.strip()]
    finetune_devices = [int(d) for d in args.finetune_devices.split(",") if d.strip()]
    serving_concurrency = [int(c) for c in args.serving_concurrency.split(",") if c.strip()]
    leg_timeout_sec = args.leg_timeout_min * 60
    finetune_timeout_min = (args.finetune_timeout_min if args.finetune_timeout_min is not None
                            else 45)
    run_start = time.time()
    max_runtime_sec = args.max_runtime_min * 60

    # NCCL env for DDP legs: socket transport by default (reliable on RunPod
    # containers with a tiny /dev/shm); tok/s is then a conservative lower bound.
    nccl_env: dict[str, str] = {"NCCL_DEBUG": "WARN"}
    if args.nccl_safe == "on":
        nccl_env.update({"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_IB_DISABLE": "1"})
    print(f"[nccl] --nccl-safe={args.nccl_safe}: {nccl_env}", flush=True)

    util_records: list[dict[str, Any]] = []
    legs: dict[str, Any] = {}

    print("[main] === Data preparation ===", flush=True)
    synth_dir, tok_dir, labels_path = _data_prep(out_dir, args.users, args.dry_run)

    print("[main] === Training scaling sweep ===", flush=True)
    training_results = _training_sweep(
        devices_sweep=devices_sweep, shard_dir=tok_dir, runs_root=runs_root,
        model_size=args.model_size, steps=args.steps, token_budget=args.token_budget,
        out_dir=out_dir, util_records=util_records, leg_timeout_sec=leg_timeout_sec,
        run_start=run_start, max_runtime_sec=max_runtime_sec, nccl_env=nccl_env,
    )
    legs["training"] = training_results

    pretrained_run_dir: Path | None = None
    for pref in ([8] if not args.dry_run else []) + [1] + devices_sweep:
        candidate = runs_root / f"sweep_d{pref}"
        if (candidate / "checkpoints" / "last.pt").exists():
            pretrained_run_dir = candidate
            break

    finetune_results: list[dict[str, Any]] = []
    if args.skip_finetune:
        print("[main] finetune skipped (--skip-finetune)", flush=True)
    elif pretrained_run_dir is None:
        print("[main] WARNING: no valid pretrained run found; skipping finetune", flush=True)
    else:
        print(f"[main] === Fine-tuning from {pretrained_run_dir.name} ===", flush=True)
        finetune_results = _finetune_sweep(
            finetune_token_budget=args.finetune_token_budget,
            finetune_max_users=args.finetune_max_users, finetune_devices=finetune_devices,
            shard_dir=tok_dir, pretrained_run_dir=pretrained_run_dir, label_path=labels_path,
            finetune_steps=args.finetune_steps, out_dir=out_dir, util_records=util_records,
            leg_timeout_sec=finetune_timeout_min * 60, run_start=run_start,
            max_runtime_sec=max_runtime_sec, nccl_env=nccl_env,
        )
    legs["finetune"] = finetune_results

    serving_results: list[dict[str, Any]] = []
    triton_results: list[dict[str, Any]] = []
    triton_skip_reason = ""
    if args.skip_serving:
        print("[main] serving skipped (--skip-serving)", flush=True)
        triton_skip_reason = "--skip-serving flag set"
    elif pretrained_run_dir is None:
        print("[main] WARNING: no valid pretrained run found; skipping serving", flush=True)
    else:
        print("[main] === Serving measurement ===", flush=True)
        serving_results, triton_results, triton_skip_reason = _measure_serving(
            run_dir=pretrained_run_dir, serving_concurrency=serving_concurrency,
            dry_run=args.dry_run, out_dir=out_dir, util_records=util_records,
            skip_triton=args.skip_triton, triton_budget_min=args.triton_budget_min,
        )
        print("[main] === Serving request caps ===", flush=True)
        legs["serve_caps"] = extras.serving_request_caps(pretrained_run_dir, list(_SAMPLE_RECORDS[:4]))
    legs["serving"] = serving_results
    legs["triton"] = {"rows": triton_results, "skip_reason": triton_skip_reason}

    flash_check_result: dict[str, Any] | None
    if args.skip_flash_check:
        flash_check_result = {"skipped": True, "skip_reason": "skipped (--skip-flash-check)"}
    else:
        print("[main] === Flash-attn ≡ SDPA equivalence check ===", flush=True)
        flash_check_result = _check_flash_vs_sdpa()  # skips itself without CUDA / flash
    legs["flash_check"] = flash_check_result
    legs["attention"] = extras.attention_backend_report()

    if pretrained_run_dir is not None and not args.skip_precision:
        print("[main] === bf16 vs fp32 ===", flush=True)
        legs["precision"] = extras.precision_agreement(
            pretrained_run_dir, tok_dir, labels_path, max_users=2_000 if args.dry_run else 20_000)
    if pretrained_run_dir is not None and not args.skip_export:
        print("[main] === ONNX export ===", flush=True)
        legs["export"] = extras.export_on_device(pretrained_run_dir, tok_dir, out_dir)
    if not args.skip_quickstart:
        print("[main] === quickstart timing ===", flush=True)
        legs["quickstart_fast"] = extras.quickstart_timing(
            out_dir, n_users=300 if args.dry_run else 2000, max_steps=10 if args.dry_run else 80)
        if args.full and not args.dry_run:
            legs["quickstart_default"] = extras.quickstart_timing(out_dir, n_users=50_000, max_steps=400)

    end_ts = datetime.now().isoformat(timespec="seconds")
    report_path = _write_report(
        out_dir, start_ts=start_ts, end_ts=end_ts, dry_run=args.dry_run,
        model_size=args.model_size, n_users=args.users, steps=args.steps,
        training_results=training_results, finetune_results=finetune_results,
        serving_results=serving_results, triton_results=triton_results,
        triton_skip_reason=triton_skip_reason, util_records=util_records,
        flash_check_result=flash_check_result, nccl_safe=args.nccl_safe,
        finetune_timeout_min=finetune_timeout_min,
    )
    evidence = {"meta": {**_run_meta(args, tag, start_ts), "finished": end_ts},
                "legs": legs, "utilisation": util_records}
    json_path = write_evidence_json(out_dir, tag, evidence)
    acceptance = json.loads(json_path.read_text())["acceptance"]
    n_fail = sum(1 for r in acceptance if r["passed"] is False)
    print("\n[main] === acceptance ===", flush=True)
    for r in acceptance:
        mark = "PASS" if r["passed"] else ("skip" if r["passed"] is None else "FAIL")
        print(f"  [{mark}] {r['check']}: {r['value']} ({r['threshold']})", flush=True)
    if args.write_readme:
        ok = write_readme_block(json_path, args.write_readme)
        print(f"[main] README block {'written' if ok else 'marker missing; not written'}: {args.write_readme}")

    print("\n[main] === DONE ===", flush=True)
    print(f"[main] Report: {report_path}", flush=True)
    print(f"[main] Evidence: {json_path}", flush=True)

    # Exit-code accounting: leg failures (the report's own predicate) or a failed
    # acceptance row exit 1; a smoke run still fails on a failed leg so the CPU
    # rehearsal catches plumbing bugs.
    failed_legs = _count_failed_legs(training_results, finetune_results, flash_check_result)
    extra_fail = sum(1 for k in ("serve_caps", "attention", "precision", "export", "quickstart_fast")
                     if k in legs and not legs[k].get("passed"))
    if failed_legs or extra_fail or (n_fail and not args.dry_run):
        print(f"[main] exit=1 ({failed_legs} leg(s) failed, {extra_fail} extra leg(s) failed, "
              f"{n_fail} acceptance row(s) failed)", flush=True)
        sys.exit(1)
    print("[main] exit=0 (all legs passed)", flush=True)


if __name__ == "__main__":
    main()
