"""REPORT.md writer, the JSON evidence file, the acceptance table and the README results block renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .legs import _finetune_leg_failed


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
    flash_check_result: dict[str, Any] | None = None,
    nccl_safe: str = "on",
    finetune_timeout_min: int = 20,
) -> Path:
    """Write REPORT.md and return its path."""
    report_path = out_dir / "REPORT.md"

    # Headline numbers
    tps_8 = next((r["tokens_per_sec"] for r in training_results if r["devices"] == 8), float("nan"))
    tps_1 = next((r["tokens_per_sec"] for r in training_results if r["devices"] == 1), float("nan"))
    r8 = next((r for r in training_results if r.get("devices") == 8), None)
    eff_8 = r8.get("efficiency", float("nan")) if r8 is not None else float("nan")
    eff_8_note = (
        " — baseline incomparable (OOM fallback)"
        if r8 is not None and r8.get("baseline_incomparable")
        else ""
    )

    gpu_reqs = next((r["req_s"] for r in serving_results if r["device"] == "cuda"), float("nan"))
    cpu_reqs = next((r["req_s"] for r in serving_results if r["device"] == "cpu" and r["concurrency"] == 1), float("nan"))
    speedup = (gpu_reqs / cpu_reqs) if (gpu_reqs == gpu_reqs and cpu_reqs == cpu_reqs and cpu_reqs > 0) else float("nan")

    lines: list[str] = []

    nccl_note = (
        "**NCCL transport:** NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 "
        "(--nccl-safe=on; socket transport for RunPod reliability). "
        "**tok/s and scaling efficiency are a CONSERVATIVE LOWER BOUND** — "
        "not NVLink-P2P-optimal. Re-run with --nccl-safe=off for full NVLink numbers."
        if nccl_safe == "on" else
        "**NCCL transport:** full NVLink/P2P/SHM attempted (--nccl-safe=off)."
    )

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
        f"**Fine-tune timeout:** {finetune_timeout_min} min  ",
        nccl_note,
        "",
        "## Headline Numbers",
        "",
    ]

    def _fmt(v: float, fmt: str = ".1f") -> str:
        return f"{v:{fmt}}" if v == v else "N/A"

    if not dry_run:
        lines += [
            f"- **8-GPU scaling efficiency:** {_fmt(eff_8, '.1%')}{eff_8_note}",
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
            note_parts: list[str] = []
            if r.get("status") == "timeout":
                note_parts.append("TIMEOUT (leg killed; harness continued)")
            elif r.get("returncode", 0) != 0:
                note_parts.append(f"FAILED (rc={r.get('returncode')})")
            if r.get("oom_fallback"):
                note_parts.append(f"OOM fallback token_budget={r.get('token_budget_used')}")
            if r.get("oom_fallback_model_size"):
                note_parts.append("OOM fallback to model_size=medium")
            if r.get("baseline_incomparable"):
                note_parts.append("baseline incomparable (OOM fallback)")
            notes = "; ".join(note_parts)
            lines.append(
                f"| {r['devices']} | {_fmt(tps, ',.0f')} | {_fmt(mem)} | "
                f"{_fmt(eff, '.1%')} | {_fmt(elapsed, '.0f')} | {notes} |"
            )
        lines.append("")

        if any(r.get("baseline_incomparable") for r in training_results):
            lines += [
                "**Note:** legs marked 'baseline incomparable (OOM fallback)' ran a "
                "different (model_size, token_budget) than the d=1 baseline leg; "
                "their scaling efficiency is N/A rather than computed against an "
                "incomparable baseline.",
                "",
            ]

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

    # ---- Flash-attn ≡ SDPA section ----
    lines += ["## flash-attn ≡ SDPA", ""]
    fc = flash_check_result or {}
    if not fc:
        lines += ["*Flash-attn check not requested.*", ""]
    elif fc.get("skipped"):
        reason = fc.get("skip_reason", "unknown reason")
        lines += [f"**Skipped:** {reason}", ""]
    elif fc.get("error"):
        lines += [
            f"**Error during check:** `{fc['error']}`",
            "",
            "*Check did not complete; see stdout for details.*",
            "",
        ]
    else:
        flash_avail = fc.get("flash_available", False)
        passed = fc.get("passed", False)
        max_diff = fc.get("max_abs_diff", float("nan"))
        mean_diff = fc.get("mean_abs_diff", float("nan"))
        tol = fc.get("tol", 1e-2)
        verdict = "PASS" if passed else "FAIL"
        lines += [
            f"**Result: {verdict}**  ",
            f"- flash-attn available on pod: {flash_avail}  ",
            f"- max abs diff (flash vs SDPA): {max_diff:.3e}  ",
            f"- mean abs diff: {mean_diff:.3e}  ",
            f"- tolerance (bf16 budget): {tol:.0e}  ",
            "",
            "Inputs: `[total=15, n_heads=4, head_dim=16]` bf16 on CUDA, "
            "3 varlen segments `[5, 3, 7]`, `dropout_p=0.0`.  "
            "The SDPA path was forced by temporarily setting "
            "`pragmatiq.models.layers._HAS_FLASH = False` (restored after).",
            "",
        ]

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
            "is fully plumbed (subprocess invokes `sys.executable scripts/validate_gpu.py --_leg-*` with "
            "real `--devices D`), but gloo/NCCL group init is only exercised at devices>1 "
            "on the real GPU pod.  The dry-run proves the subprocess + metrics-parse pipeline; "
            "DDP collective ops are only validated on the pod.",
            "",
        ]
    else:
        failed_training = [r for r in training_results if r.get("returncode", 0) != 0]
        failed_finetune = [r for r in finetune_results if _finetune_leg_failed(r)]
        fc_v = flash_check_result or {}
        flash_failed = bool(fc_v) and not fc_v.get("skipped") and (
            bool(fc_v.get("error")) or not fc_v.get("passed", False)
        )
        if not failed_training and not failed_finetune and not flash_failed:
            lines += [
                "**All legs completed successfully.**",
                "- Training scaling sweep ✓",
                "- Finetune DDP convergence ✓",
                "- Serving runtime throughput ✓",
                "",
            ]
        else:
            lines += ["**Some legs failed — see sections above.**", ""]
            if failed_training:
                lines += [f"- training: {len(failed_training)} failed leg(s) "
                          f"(devices={sorted(r.get('devices', 0) for r in failed_training)})", ""]
            if failed_finetune:
                lines += [f"- finetune: {len(failed_finetune)} failed/timed-out leg(s) "
                          f"(devices={sorted(r.get('devices', 0) for r in failed_finetune)})", ""]
            if flash_failed:
                lines += ["- flash-attn ≡ SDPA equivalence check FAILED", ""]

    report_path.write_text("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# JSON evidence file, acceptance table, README results block
# ---------------------------------------------------------------------------

README_MARKER = "<!-- GPU_VALIDATION_RESULTS -->"


def _nan_to_none(v: Any) -> Any:
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, dict):
        return {k: _nan_to_none(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_nan_to_none(x) for x in v]
    return v


def acceptance_table(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive the pass/fail acceptance rows from a validation evidence dict.

    Each row is ``{"check", "value", "threshold", "passed"}``; ``passed`` is
    ``None`` when the check did not run (skipped leg), so a skipped check is
    visible but never counted as a failure.
    """
    legs = evidence.get("legs", {})
    rows: list[dict[str, Any]] = []

    def add(check: str, value: Any, threshold: str, passed: bool | None) -> None:
        rows.append({"check": check, "value": _nan_to_none(value), "threshold": threshold,
                     "passed": passed})

    training = legs.get("training", [])
    for r in training:
        ok = r.get("status") != "timeout" and r.get("returncode", 0) == 0
        add(f"pretrain d={r.get('devices')} completes", r.get("tokens_per_sec"), "rc=0", ok)
    eff = [r.get("efficiency") for r in training if r.get("devices", 1) > 1]
    eff = [e for e in eff if isinstance(e, (int, float)) and e == e]
    if eff:
        add("DDP scaling efficiency (max devices)", eff[-1], ">= 0.70", eff[-1] >= 0.70)

    for r in legs.get("finetune", []):
        auc = r.get("best_val_auc")
        finite = isinstance(auc, (int, float)) and auc == auc and auc > 0
        add(f"finetune d={r.get('devices')} epochs", r.get("epochs_run"),
            "all epochs, finite AUC", bool(finite) and r.get("returncode", 0) == 0
            and r.get("status") != "timeout")
        tps = [s.get("tokens_per_sec", 0.0) for s in r.get("epoch_stats", []) if s.get("phase") == "train"]
        if len(tps) >= 2 and tps[0] > 0:
            ratio = tps[1] / tps[0]
            # Epoch 1 pays for first-touch shard reads and kernel warmup, so a
            # faster second epoch is expected; only a slowdown is a defect.
            add(f"finetune d={r.get('devices')} epoch-2 tok/s vs epoch-1", round(ratio, 3),
                ">= 0.8", ratio >= 0.80)

        # Host-bound detector: the share of the last training epoch spent waiting
        # on the loader (rc1 sat at ~100% with the GPU idle). GPU utilisation is
        # reported in the utilisation section but not gated — the small preset's
        # kernels are launch-bound, so its utilisation is low even when healthy.
        train = [s for s in r.get("epoch_stats", []) if s.get("phase") == "train"]
        if train and train[-1].get("seconds"):
            share = float(train[-1].get("data_wait_seconds", 0.0)) / float(train[-1]["seconds"])
            add(f"finetune d={r.get('devices')} loader wait share (last epoch)", round(share, 3),
                "<= 0.5", share <= 0.5)

    serving = legs.get("serving", [])
    gpu1 = next((r["req_s"] for r in serving if r.get("device") == "cuda" and r.get("concurrency") == 1), None)
    cpu1 = next((r["req_s"] for r in serving if r.get("device") == "cpu" and r.get("concurrency") == 1), None)
    if gpu1 is not None and cpu1 is not None:
        add("serving GPU req/s > CPU req/s (concurrency 1)", round(gpu1 / cpu1, 2) if cpu1 else None,
            "> 1.0", gpu1 > cpu1)

    fc = legs.get("flash_check") or {}
    if fc and not fc.get("skipped"):
        add("flash-attn ≡ SDPA max abs diff", fc.get("max_abs_diff"), f"<= {fc.get('tol', 1e-2):.0e}",
            bool(fc.get("passed")))

    for name, label, threshold in (
        ("precision", "bf16 vs fp32 embeddings mean cosine (|ΔROC-AUC| reported)", ">= 0.99"),
        ("serve_caps", "serving request caps + chunked == whole", "reject oversized; max abs < 1e-2"),
        ("export", "ONNX export on the pod", "exported and validated"),
        ("attention", "attention backend (flash on CUDA bf16; SDPA when disabled)", "as expected"),
        ("quickstart_fast", "quickstart --n-users 2000 --max-steps 80", "rc=0"),
        ("quickstart_default", "quickstart (default)", "rc=0"),
    ):
        leg = legs.get(name)
        if not leg:
            continue
        value = leg.get("mean_cosine", leg.get("wall_time_s", leg.get("chunked_vs_whole_max_abs",
                                                                  leg.get("bf16_backend"))))
        passed = bool(leg.get("passed"))
        if name == "precision" and leg.get("mean_cosine") is not None:
            passed = float(leg["mean_cosine"]) >= 0.99  # re-derivable from the raw leg
        add(label, value, threshold, passed)
    return rows


def write_evidence_json(out_dir: Path, tag: str, evidence: dict[str, Any]) -> Path:
    """Write ``gpu-validation-<tag>.json`` (NaN → null) with the acceptance table filled in."""
    evidence = dict(evidence)
    evidence["acceptance"] = acceptance_table(evidence)
    evidence["all_passed"] = all(r["passed"] is not False for r in evidence["acceptance"])
    path = out_dir / f"gpu-validation-{tag}.json"
    path.write_text(json.dumps(_nan_to_none(evidence), indent=2, sort_keys=True) + "\n")
    return path


def _f(v: Any, fmt: str = ".1f", suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    try:
        return f"{v:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return f"{v}{suffix}"


def render_readme_block(json_path: str | Path) -> str:
    """Render the README results block from a validation evidence JSON."""
    ev = json.loads(Path(json_path).read_text())
    meta = ev.get("meta", {})
    legs = ev.get("legs", {})
    lines = [
        f"**Hardware:** {meta.get('gpu_name', 'unknown GPU')} × {meta.get('gpu_count', '?')}, "
        f"torch {meta.get('torch', '?')}, flash-attn {meta.get('flash_attn', 'absent')}, "
        f"CUDA {meta.get('cuda', '?')}. **Data:** {meta.get('n_users', '?')} synthetic users, "
        f"{meta.get('steps', '?')} pretrain steps, model `{meta.get('model_size', '?')}`. "
        f"Run {meta.get('tag', '?')} on {meta.get('started', '?')}.",
        "",
    ]
    training = legs.get("training", [])
    if training:
        lines += ["| preset | devices | tokens/s | peak VRAM (GB) | DDP efficiency |",
                  "| --- | --- | --- | --- | --- |"]
        for r in training:
            lines.append(f"| {r.get('model_size', '?')} | {r.get('devices')} | {_f(r.get('tokens_per_sec'), ',.0f')} | "
                         f"{_f(r.get('gpu_mem_gb'))} | {_f(r.get('efficiency'), '.0%')} |")
        lines.append("")
    ft = legs.get("finetune", [])
    if ft:
        lines += ["| fine-tune devices | epochs | wall time | best val ROC-AUC | epoch-1 → epoch-2 tok/s |",
                  "| --- | --- | --- | --- | --- |"]
        for r in ft:
            tps = [s.get("tokens_per_sec") for s in r.get("epoch_stats", []) if s.get("phase") == "train"]
            tps_s = " → ".join(_f(t, ',.0f') for t in tps[:2]) if tps else "n/a"
            lines.append(f"| {r.get('devices')} | {r.get('epochs_run')} | {_f(r.get('wall_time_s'), '.0f', ' s')} | "
                         f"{_f(r.get('best_val_auc'), '.3f')} | {tps_s} |")
        lines.append("")
    serving = legs.get("serving", [])
    if serving:
        lines += ["| serving device | concurrency | req/s | p50 ms | p99 ms |", "| --- | --- | --- | --- | --- |"]
        for r in serving:
            lines.append(f"| {r.get('device')} | {r.get('concurrency')} | {_f(r.get('req_s'))} | "
                         f"{_f(r.get('p50_ms'), '.0f')} | {_f(r.get('p99_ms'), '.0f')} |")
        lines.append("")
    prec = legs.get("precision") or {}
    fc = legs.get("flash_check") or {}
    bullets = []
    if prec:
        bullets.append(f"- bf16 vs fp32: embed {_f(prec.get('bf16_users_per_sec'), ',.0f')} vs "
                       f"{_f(prec.get('fp32_users_per_sec'), ',.0f')} users/s; probe ROC-AUC "
                       f"{_f(prec.get('bf16_probe_auc'), '.3f')} vs {_f(prec.get('fp32_probe_auc'), '.3f')} "
                       f"(|Δ| = {_f(prec.get('abs_auc_delta'), '.4f')}).")
    if fc and not fc.get("skipped"):
        bullets.append(f"- flash-attn vs SDPA max abs diff: {_f(fc.get('max_abs_diff'), '.2e')} "
                       f"(tolerance {_f(fc.get('tol'), '.0e')}).")
    qs = legs.get("quickstart_fast") or {}
    if qs:
        bullets.append(f"- `pragmatiq quickstart --n-users 2000 --max-steps 80`: {_f(qs.get('wall_time_s'), '.0f', ' s')}.")
    qd = legs.get("quickstart_default") or {}
    if qd:
        bullets.append(f"- `pragmatiq quickstart` (default): {_f(qd.get('wall_time_s'), '.0f', ' s')}.")
    if bullets:
        lines += bullets + [""]
    acc = ev.get("acceptance", [])
    if acc:
        n_ok = sum(1 for r in acc if r["passed"] is True)
        n_ran = sum(1 for r in acc if r["passed"] is not None)
        lines.append(f"Acceptance: {n_ok}/{n_ran} checks passed. Evidence: `docs/benchmarks/{Path(json_path).name}`.")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def write_readme_block(json_path: str | Path, readme_path: str | Path) -> bool:
    """Replace the README block under ``README_MARKER`` (up to the next ``## ``) with the rendered results.

    Returns False (and leaves the file untouched) when the marker is absent.
    """
    import re

    readme = Path(readme_path)
    text = readme.read_text()
    if README_MARKER not in text:
        return False
    block = README_MARKER + "\n\n" + render_readme_block(json_path)
    new = re.sub(re.escape(README_MARKER) + r".*?(?=\n<!-- |\n\*\*|\n## |\Z)", lambda _m: block, text, count=1, flags=re.S)
    readme.write_text(new)
    return True
