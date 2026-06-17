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
import tempfile
from pathlib import Path


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
    from pragmatiq.inference.multitask import (
        multitask_results_markdown,
        run_multitask_probe,
        write_multitask_report,
    )

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
