#!/usr/bin/env python3
"""Multi-task probe benchmark: does the embedding beat a raw-count baseline across
the user-level downstream tasks (credit / churn / LTV), not just credit?

Generates a dataset, pretrains once, probes each task vs a logistic-on-raw-counts
baseline (eval-point truncated), and prints a provenance-stamped table. With
``--write`` (or ``PRAGMATIQ_WRITE_RESULTS=1``) it writes the table into the
README ``<!-- MULTITASK_PROBE_RESULTS -->`` marker.

Usage:
    python scripts/benchmarks/multitask_probe.py [--n-users 4000] [--model-size nano]
        [--max-steps 1200] [--seed 0] [--write]
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

USER_LEVEL_TASKS: tuple[str, ...] = ("default_12m", "churn_6m", "ltv_positive")
MARKER = "<!-- MULTITASK_PROBE_RESULTS -->"


@dataclass
class MultiTaskRow:
    """One task's probe vs raw-count-baseline ROC-AUC and PR-AUC."""

    task: str
    probe_auc: float
    baseline_auc: float
    probe_pr_auc: float
    baseline_pr_auc: float
    prevalence: float
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


def run_multitask_probe(shard_dir: Path, run_dir: str | Path, labels_dir: Path,
                        tasks: tuple[str, ...] = USER_LEVEL_TASKS, seed: int = 0,
                        device: str = "auto") -> list[MultiTaskRow]:
    """Probe each user-level task's label table against the raw-count baseline."""
    from pragmatiq import api

    rows: list[MultiTaskRow] = []
    for task in tasks:
        lp = Path(labels_dir) / f"{task}.parquet"
        if not lp.exists():
            continue
        res = api.probe(shard_dir, run_dir, lp, device=device, seed=seed)
        rows.append(MultiTaskRow(task=task, probe_auc=res["probe_auc"],
                                 baseline_auc=res["baseline_auc"],
                                 probe_pr_auc=res["probe_pr_auc"],
                                 baseline_pr_auc=res["baseline_pr_auc"],
                                 prevalence=res["prevalence"], n_test=res["n_test"]))
    return rows


def multitask_results_markdown(rows: list[MultiTaskRow], scale: dict[str, Any]) -> str:
    """Render probe rows as a provenance-stamped markdown table."""
    lines = ["| task | probe ROC-AUC | baseline ROC-AUC | probe PR-AUC | baseline PR-AUC | prevalence |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r.task} | {r.probe_auc:.3f} | {r.baseline_auc:.3f} | "
                     f"{r.probe_pr_auc:.3f} | {r.baseline_pr_auc:.3f} | {r.prevalence:.2f} |")
    lines.append("")
    lines.append(
        f"<sub>provenance: n_users={scale.get('n_users', '?')}, model={scale.get('model', '?')}, "
        f"steps={scale.get('steps', '?')}, seed={scale.get('seed', 0)}, commit={_git_commit()}</sub>"
    )
    return "\n".join(lines)


def write_multitask_report(rows: list[MultiTaskRow], scale: dict[str, Any],
                           readme_path: str | Path = "README.md") -> None:
    """Write the table into the README marker; refuse to shrink the reported scale."""
    md = multitask_results_markdown(rows, scale)
    path = Path(readme_path)
    if not path.exists():
        return
    text = path.read_text()
    if MARKER not in text:
        return
    existing = re.search(rf"{re.escape(MARKER)}.*?provenance: n_users=(\d+)", text, flags=re.S)
    if existing and int(existing.group(1)) > int(scale.get("n_users", 0)):
        print(f"existing multi-task table is from a larger run (n_users={existing.group(1)}); not overwriting")
        return
    text = re.sub(re.escape(MARKER) + r".*?(?=\n<!-- |\n\*\*|\n## |\Z)", MARKER + "\n\n" + md + "\n", text,
                  count=1, flags=re.S)
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-users", type=int, default=4000)
    ap.add_argument("--model-size", default="nano")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-readme", default="README.md")
    ap.add_argument("--write", action="store_true", help="write the table into the README marker")
    args = ap.parse_args()

    from pragmatiq import api

    work = Path(tempfile.mkdtemp(prefix="multitask-"))
    api.synthesize({"n_users": args.n_users, "months": 24, "n_merchants": 3000,
                    "eval_month_credit": 12, "eval_month_short": 12, "seed": args.seed + 11},
                   out=work / "raw", n_workers=4, write_report=False)
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 28000 if args.n_users >= 20000 else 8000,
                         "n_buckets": 64, "categorical_threshold": 1000},
                 n_workers=4)
    summary = api.pretrain(work / "tok", "multitask", model_size=args.model_size,
                           config={"max_steps": args.max_steps, "token_budget": 8192,
                                   "warmup_steps": max(1, args.max_steps // 10), "log_every": 200,
                                   "checkpoint_every_min": 1000.0}, runs_root=work / "runs")

    rows = run_multitask_probe(work / "tok", summary["run_dir"], work / "raw" / "labels",
                               seed=args.seed)
    scale = {"n_users": args.n_users, "model": args.model_size, "steps": args.max_steps,
             "seed": args.seed}
    print(multitask_results_markdown(rows, scale))

    if args.write or os.environ.get("PRAGMATIQ_WRITE_RESULTS") == "1":
        write_multitask_report(rows, scale, readme_path=args.out_readme)
        print(f"wrote multi-task results to {args.out_readme}")


if __name__ == "__main__":
    main()
