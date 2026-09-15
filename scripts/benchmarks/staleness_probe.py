#!/usr/bin/env python3
"""Event-staleness benchmark (paper §3.4.2): how much do task metrics move when the
most recent window of history is missing at scoring time?

Generates a dataset, pretrains once, then probes the user-level tasks with the
last 0 / 1h / 6h / 1d / 3d of history removed before every eval point (the
raw-count baseline gets the same cutoffs). A serving pipeline fed by a lagging
event stream is safe when the deltas stay small. With ``--write`` (or
``PRAGMATIQ_WRITE_RESULTS=1``) the table lands in the README
``<!-- STALENESS_PROBE_RESULTS -->`` marker.

Usage:
    python scripts/benchmarks/staleness_probe.py [--n-users 4000] [--model-size nano]
        [--max-steps 1200] [--seed 0] [--windows 0,1h,6h,1d,3d] [--write]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASKS: tuple[str, ...] = ("default_12m", "churn_6m")
DEFAULT_WINDOWS: tuple[str, ...] = ("0", "1h", "6h", "1d", "3d")
MARKER = "<!-- STALENESS_PROBE_RESULTS -->"


@dataclass
class StalenessRow:
    """Probe / baseline ROC-AUC and PR-AUC for one task at one staleness window."""

    task: str
    window: str
    probe_auc: float
    probe_pr_auc: float
    baseline_auc: float
    n_test: int


def _git_commit() -> str:
    pinned = os.environ.get("PRAGMATIQ_COMMIT", "").strip()
    if pinned:
        return pinned  # a git archive on a pod carries no .git; the launcher exports the sha
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_staleness_probe(shard_dir: Path, run_dir: str | Path, labels_dir: Path,
                        windows: tuple[str, ...] = DEFAULT_WINDOWS,
                        tasks: tuple[str, ...] = TASKS, seed: int = 0,
                        device: str = "auto") -> list[StalenessRow]:
    """Probe each task at each staleness window with a fixed split seed."""
    from pragmatiq import api

    rows: list[StalenessRow] = []
    for task in tasks:
        lp = Path(labels_dir) / f"{task}.parquet"
        if not lp.exists():
            continue
        for w in windows:
            res = api.probe(shard_dir, run_dir, lp, device=device, seed=seed,
                            staleness_window=None if w == "0" else w)
            rows.append(StalenessRow(task=task, window=w, probe_auc=res["probe_auc"],
                                     probe_pr_auc=res["probe_pr_auc"],
                                     baseline_auc=res["baseline_auc"], n_test=res["n_test"]))
    return rows


def staleness_results_markdown(rows: list[StalenessRow], scale: dict[str, Any]) -> str:
    """Render the rows as one table per task with deltas against the fresh (0) row."""
    lines = ["| task | stale window | probe ROC-AUC | Δ vs fresh | probe PR-AUC | Δ vs fresh | baseline ROC-AUC |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    fresh = {r.task: r for r in rows if r.window == "0"}
    for r in rows:
        f = fresh.get(r.task, r)
        lines.append(f"| {r.task} | {r.window} | {r.probe_auc:.3f} | {r.probe_auc - f.probe_auc:+.3f} | "
                     f"{r.probe_pr_auc:.3f} | {r.probe_pr_auc - f.probe_pr_auc:+.3f} | {r.baseline_auc:.3f} |")
    lines.append("")
    lines.append(
        f"<sub>provenance: n_users={scale.get('n_users', '?')}, model={scale.get('model', '?')}, "
        f"steps={scale.get('steps', '?')}, seed={scale.get('seed', 0)}, commit={_git_commit()}</sub>"
    )
    return "\n".join(lines)


def write_staleness_report(rows: list[StalenessRow], scale: dict[str, Any],
                           readme_path: str | Path = "README.md") -> None:
    """Write the table into the README marker; refuse to shrink the reported scale."""
    md = staleness_results_markdown(rows, scale)
    path = Path(readme_path)
    if not path.exists():
        return
    text = path.read_text()
    if MARKER not in text:
        return
    existing = re.search(rf"{re.escape(MARKER)}.*?provenance: n_users=(\d+)", text, flags=re.S)
    if existing and int(existing.group(1)) > int(scale.get("n_users", 0)):
        print(f"existing staleness table is from a larger run (n_users={existing.group(1)}); not overwriting")
        return
    text = re.sub(re.escape(MARKER) + r".*?(?=\n<!-- |\n##+ |\Z)", MARKER + "\n\n" + md + "\n", text,
                  count=1, flags=re.S)
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-users", type=int, default=4000)
    ap.add_argument("--model-size", default="nano")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", default=",".join(DEFAULT_WINDOWS))
    ap.add_argument("--out-readme", default="README.md")
    ap.add_argument("--write", action="store_true", help="write the table into the README marker")
    args = ap.parse_args()

    from pragmatiq import api

    work = Path(tempfile.mkdtemp(prefix="staleness-"))
    api.synthesize({"n_users": args.n_users, "months": 24, "n_merchants": 3000,
                    "eval_month_credit": 12, "eval_month_short": 12, "seed": args.seed + 11},
                   out=work / "raw", n_workers=4, write_report=False)
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 28000 if args.n_users >= 20000 else 8000,
                         "n_buckets": 64, "categorical_threshold": 1000},
                 n_workers=4)
    summary = api.pretrain(work / "tok", "staleness", model_size=args.model_size,
                           config={"max_steps": args.max_steps, "token_budget": 8192,
                                   "warmup_steps": max(1, args.max_steps // 10), "log_every": 200,
                                   "checkpoint_every_min": 1000.0}, runs_root=work / "runs")

    windows = tuple(w.strip() for w in args.windows.split(",") if w.strip())
    rows = run_staleness_probe(work / "tok", summary["run_dir"], work / "raw" / "labels",
                               windows=windows, seed=args.seed)
    scale = {"n_users": args.n_users, "model": args.model_size, "steps": args.max_steps,
             "seed": args.seed}
    print(staleness_results_markdown(rows, scale))

    if args.write or os.environ.get("PRAGMATIQ_WRITE_RESULTS") == "1":
        write_staleness_report(rows, scale, readme_path=args.out_readme)
        print(f"wrote staleness results to {args.out_readme}")


if __name__ == "__main__":
    main()
