"""Full GPU validation: the measurement sweep, then full-scale acceptance gates.

Intended as ``runpod_launch.py --remote-script scripts/gpu_full_validation.py``
so one pod runs everything the release evidence needs:

1. ``scripts/validate_gpu.py`` (all CLI args pass straight through) — the
   scaling/finetune/serving/flash sweep that writes ``REPORT.md``;
2. ``gate_5.sh`` and ``gate_6.sh`` at full scale (``PRAGMATIQ_GATE_FULL=1``) —
   probe quality and the AML ablation, the two gates whose full-scale runs are
   too long for CI.

Output discipline (hard-won): the launcher's SSH stream has repeatedly stalled
mid-run, and anything blocking on that stream freezes the whole orchestration
(observed 2026-07-05/06). All child output therefore goes ONLY to
``<out>/orchestrator.log`` on the pod; the SSH stream carries just stage
banners and a one-line heartbeat per minute — small enough to sit in kernel
buffers for a day even if the stream is dead. Exit codes stay authoritative.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_LOG_FH = None  # orchestrator.log handle; None means plain stdout (local use)
_STAGE = {"name": "starting", "t0": time.time()}


def _say(msg: str) -> None:
    """One line to the SSH stream AND the pod-side log."""
    line = f"[full-validation] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write((line + "\n").encode())
        _LOG_FH.flush()


def _heartbeat() -> None:
    while True:
        time.sleep(60)
        mins = (time.time() - _STAGE["t0"]) / 60.0
        print(f"[full-validation] hb: {_STAGE['name']} running {mins:.0f}m", flush=True)


def _run(cmd: list[str], env: dict[str, str], stage: str) -> None:
    _STAGE["name"], _STAGE["t0"] = stage, time.time()
    _say(f"$ {' '.join(cmd)}")
    if _LOG_FH is not None:
        rc = subprocess.call(cmd, env=env, stdout=_LOG_FH, stderr=_LOG_FH)
    else:
        rc = subprocess.call(cmd, env=env)
    if rc != 0:
        _say(f"FAILED rc={rc}: {stage}")
        sys.exit(rc)
    _say(f"OK: {stage}")


def main() -> None:
    global _LOG_FH
    if "--out" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FH = (out_dir / "orchestrator.log").open("ab")
    threading.Thread(target=_heartbeat, daemon=True).start()

    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    _run([sys.executable, "-u", "scripts/validate_gpu.py", *sys.argv[1:]], env,
         "measurement sweep")
    _run(["bash", "scripts/gates/gate_5.sh"],
         {**env, "PRAGMATIQ_GATE_FULL": "1", "PRAGMATIQ_GATE_SKIP_UNIT": "1"},
         "gate_5 (full scale)")
    _run(["bash", "scripts/gates/gate_6.sh"], {**env, "PRAGMATIQ_GATE_FULL": "1"},
         "gate_6 (full scale)")
    _say("ALL STEPS PASSED")


if __name__ == "__main__":
    main()
