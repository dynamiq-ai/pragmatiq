"""Leg subprocess plumbing: the pretrain/finetune leg bodies, the timeout runner, data prep and the failure predicates the verdict uses."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _is_rank_zero() -> bool:
    """True on the coordinating DDP rank (and in any non-distributed run).

    Fabric/torchrun launchers export RANK / LOCAL_RANK into every worker; a
    plain single-process leg has neither and counts as rank 0.
    """
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _finetune_leg_failed(result: dict[str, Any]) -> bool:
    """True when a finetune leg did not produce a trustworthy AUC.

    A leg fails on a non-zero exit, a timeout, or a missing/absent AUC — the
    leg JSON is only written on success, so ``best_val_auc`` degrades to NaN
    (NaN != NaN) rather than the -1 sentinel alone.
    """
    auc = result.get("best_val_auc", float("nan"))
    return (
        result.get("returncode", 0) != 0
        or result.get("status") == "timeout"
        or auc != auc  # NaN — leg never reported
        or auc == -1
    )


def _count_failed_legs(
    training_results: list[dict[str, Any]],
    finetune_results: list[dict[str, Any]],
    flash_check_result: dict[str, Any] | None,
) -> int:
    """Number of failed legs, for exit-code accounting.

    Finetune legs go through ``_finetune_leg_failed`` — the same predicate the
    REPORT.md verdict uses — so a leg that exits 0 without producing a
    trustworthy AUC (NaN / -1 sentinel) still fails the run, and the exit code
    can never disagree with the report.  A skipped flash check is not a
    failure; only an attempted check that did not pass is.
    """
    failed = 0
    for r in training_results:
        if r.get("status") == "timeout" or r.get("returncode", 0) != 0:
            failed += 1
    for r in finetune_results:
        if _finetune_leg_failed(r):
            failed += 1
    if (
        flash_check_result is not None
        and not flash_check_result.get("skipped", False)
        and not flash_check_result.get("passed", True)
    ):
        failed += 1
    return failed


def _subsample_labels(label_path: Path, out_path: Path, max_users: int, seed: int = 0) -> int:
    """Write a seeded, label-stratified subsample of a label table.

    The fine-tune leg validates convergence, not population-scale training —
    fine-tuning every labeled user of a full-scale dataset is a multi-hour job
    by construction (~30k users x ~7k tokens x 3 epochs measured 2026-07-07).
    Proportional per-label sampling keeps class balance; returns rows written.
    """
    import numpy as np  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(label_path)
    n = table.num_rows
    if n <= max_users:
        pq.write_table(table, out_path)
        return n
    rng = np.random.default_rng(seed)
    labels = table.column("label").to_numpy()
    keep: list[np.ndarray] = []
    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        quota = max(1, int(round(len(idx) / n * max_users)))
        keep.append(rng.choice(idx, size=min(quota, len(idx)), replace=False))
    sel = np.sort(np.concatenate(keep))
    pq.write_table(pa.Table.from_batches(table.take(sel).to_batches()), out_path)
    return len(sel)


def _leg_logging() -> None:
    """Route library INFO logs (finetune heartbeats etc.) to the leg's log file.

    Legs run as subprocesses with stdout redirected to a file; without a
    handler, ``logging.info`` output vanishes (root logger prints WARNING+
    only), which made a 40-minute fine-tune leg look hung (B1, 2026-07-05).
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(message)s", force=True)


def _leg_pretrain(args: argparse.Namespace) -> None:
    """Run one pretrain leg and exit.  Invoked via subprocess by the orchestrator."""
    _leg_logging()
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
    _leg_logging()
    import pragmatiq.api as api  # noqa: PLC0415

    label_path = Path(args.label_path)
    if getattr(args, "finetune_max_users", 0):
        sub = Path(args.result_json).with_suffix(".labels.parquet")
        n = _subsample_labels(label_path, sub, int(args.finetune_max_users))
        print(f"[leg-finetune] label subsample: {n} users -> {sub}", flush=True)
        label_path = sub

    config: dict[str, Any] = {
        "max_epochs": args.steps,
        "devices": args.devices,
        # 0 = let the fine-tuner size the per-forward budget from the device
        # (bf16 autocast on CUDA); a positive value pins it.
        "token_budget": int(args.finetune_token_budget) or None,
    }
    result = api.finetune(
        args.shard_dir,
        args.run_dir,
        label_path,
        config=config,
        device="auto",
    )
    # Write result JSON so the orchestrator can read it back. Only the
    # coordinating rank writes: under DDP every Fabric rank re-executes this
    # leg body, and concurrent writes to the same path can interleave. The
    # validation AUC is gathered across ranks, so rank 0's copy is the result.
    if _is_rank_zero():
        Path(args.result_json).write_text(json.dumps(result))
    print(f"[leg-finetune] done: best_val_auc={result.get('best_val_auc')}", flush=True)


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


def _run_leg_with_timeout(
    cmd: list[str],
    timeout_sec: float,
    label: str,
    log_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, bool]:
    """Run a leg subprocess with a per-leg timeout.

    Stdout and stderr are redirected to a log FILE (not a PIPE) so the OS
    pipe buffer can never fill and deadlock the parent — a leg emitting
    megabytes of training logs will never block.  The log file path is
    ``<log_dir>/<label>.log`` when *log_dir* is given; otherwise a temporary
    file is used and discarded after the leg finishes.

    Uses ``start_new_session=True`` so the child gets its own process group,
    allowing ``os.killpg`` to reap orphaned GPU processes on timeout.

    *extra_env*, when given, is merged into a copy of ``os.environ`` and
    passed as the subprocess environment.  Use this to inject NCCL tunables
    (e.g., NCCL_DEBUG, NCCL_P2P_DISABLE) without touching the parent process
    environment — CPU / dry-run legs pass ``extra_env=None`` and are
    unaffected.

    Returns:
        (returncode, timed_out) — on timeout returncode is -1.
    """

    child_env: dict[str, str] | None = None
    if extra_env:
        child_env = {**os.environ, **extra_env}

    log_path: Path | None = None
    _tmp_fh = None
    try:
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{label}.log"
            log_fh = log_path.open("wb")
        else:
            # Use a temporary file; no name needed, the fd is enough.
            _tmp_fh = tempfile.TemporaryFile()
            log_fh = _tmp_fh

        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                start_new_session=True,
                stdout=log_fh,
                stderr=log_fh,
                env=child_env,
            )
        finally:
            # The child inherited the fd; we can close our copy now so the
            # file is flushed/released when the child exits (not when we exit).
            log_fh.close()

        try:
            proc.wait(timeout=timeout_sec)
            if log_path is not None:
                # Tail the log to the orchestrator's stdout so progress is
                # visible (keep last 50 lines to avoid flooding on dry-run).
                try:
                    lines = log_path.read_bytes().decode(errors="replace").splitlines()
                    tail = lines[-50:] if len(lines) > 50 else lines
                    for line in tail:
                        print(f"  [{label}] {line}", flush=True)
                except Exception:  # noqa: BLE001
                    pass
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
            # A crashed NCCL/CUDA leg can leave ranks in uninterruptible D-state
            # (wedged in the driver) that SIGKILL cannot reap — an unbounded
            # wait() here hung a paid 8xH100 run for 6 hours (2026-07-05). Wait
            # briefly, then abandon the corpse and let the run finish honestly.
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print(
                    f"[{label}] process group not reaped 60s after SIGKILL "
                    "(GPU driver wedge?); abandoning it and continuing",
                    flush=True,
                )
            return -1, True
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] subprocess launch error: {exc}", flush=True)
        return -1, False
