"""Full GPU validation: the measurement sweep, then full-scale acceptance gates.

Intended as ``runpod_launch.py --remote-script scripts/gpu_full_validation.py``
so one pod runs everything the release evidence needs:

1. ``scripts/validate_gpu.py`` (all CLI args pass straight through) — the
   scaling/finetune/serving/flash sweep that writes ``REPORT.md``;
2. ``gate_5.sh`` and ``gate_6.sh`` at full scale (``PRAGMATIQ_GATE_FULL=1``) —
   probe quality and the AML ablation, the two gates whose full-scale runs are
   too long for CI.

Exits non-zero on the first failing step so the launcher's exit code is
trustworthy. Thread pools are bounded like the smoke pipeline so the
sequential CPU stages don't stall on oversubscription.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"[full-validation] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print(f"[full-validation] FAILED rc={rc}: {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}",
              flush=True)
        sys.exit(rc)


def _tee_to_out_dir() -> None:
    """Duplicate this orchestrator's stdout/stderr into <out>/orchestrator.log.

    The launcher streams our output over SSH, and that stream has been observed
    to drop mid-run — the pod-side copy is what makes a post-mortem possible
    when it does. Best-effort: parses --out from the pass-through args.
    """
    if "--out" not in sys.argv:
        return
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = subprocess.Popen(  # noqa: S603,S607 — tee is the whole point
        ["tee", "-a", str(out_dir / "orchestrator.log")],
        stdin=subprocess.PIPE,
    )
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def main() -> None:
    _tee_to_out_dir()
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    _run([sys.executable, "-u", "scripts/validate_gpu.py", *sys.argv[1:]], env)
    _run(["bash", "scripts/gates/gate_5.sh"],
         {**env, "PRAGMATIQ_GATE_FULL": "1", "PRAGMATIQ_GATE_SKIP_UNIT": "1"})
    _run(["bash", "scripts/gates/gate_6.sh"], {**env, "PRAGMATIQ_GATE_FULL": "1"})
    print("[full-validation] ALL STEPS PASSED", flush=True)


if __name__ == "__main__":
    main()
