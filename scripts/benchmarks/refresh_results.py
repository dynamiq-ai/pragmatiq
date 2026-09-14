#!/usr/bin/env python3
"""Regenerate every README / notebook result table on one GPU pod.

Runs, at the documented scale and on the current generator:

1. the multi-task probe benchmark (README ``<!-- MULTITASK_PROBE_RESULTS -->``),
2. the event-staleness benchmark (README ``<!-- STALENESS_PROBE_RESULTS -->``),
3. the full-scale AML ablation (gate 6 with ``PRAGMATIQ_GATE_FULL=1``; writes the
   README ``<!-- AML_ABLATION_RESULTS -->`` block and notebook 04 on a pass),

then copies the rewritten README and notebook into ``--out`` so the RunPod
launcher's artifact pull brings them back. Every table carries its provenance
stamp (scale, seed, commit).

Usage (on the pod, via scripts/runpod_launch.py --remote-script):
    python scripts/benchmarks/refresh_results.py [--n-users 50000] [--model-size small]
        [--max-steps 2000] [--seed 0] [--skip-aml] [--out outputs/gpu-validation-results]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print(f"[refresh] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    print(f"[refresh] rc={rc} after {time.time() - t0:.0f}s", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-users", type=int, default=50_000)
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-aml", action="store_true", help="skip the full-scale gate 6")
    ap.add_argument("--skip-multitask", action="store_true", help="skip the multi-task probe benchmark")
    ap.add_argument("--skip-staleness", action="store_true", help="skip the staleness benchmark")
    ap.add_argument("--out", default="outputs/gpu-validation-results")
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    scale = ["--n-users", str(args.n_users), "--model-size", args.model_size,
             "--max-steps", str(args.max_steps), "--seed", str(args.seed)]
    failures: list[str] = []
    if not args.skip_multitask and _run([py, "scripts/benchmarks/multitask_probe.py", *scale, "--write"]):
        failures.append("multitask_probe")
    if not args.skip_staleness and _run([py, "scripts/benchmarks/staleness_probe.py", *scale, "--write"]):
        failures.append("staleness_probe")
    if not args.skip_aml:
        env = dict(os.environ, PRAGMATIQ_GATE_FULL="1", PRAGMATIQ_WRITE_RESULTS="1")
        if _run(["bash", "scripts/gates/gate_6.sh"], env=env):
            failures.append("gate_6_full")

    for rel in ("README.md", "notebooks/04_aml_gnn.ipynb"):
        src = REPO / rel
        if src.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (out / "refresh_status.txt").write_text(
        "OK\n" if not failures else "FAILED: " + ", ".join(failures) + "\n")
    print(f"[refresh] done; failures={failures or 'none'}; artifacts in {out}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
