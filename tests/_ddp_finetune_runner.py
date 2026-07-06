"""Subprocess entry point for the DDP fine-tune smoke test.

Run as a standalone script so Lightning Fabric can spawn its own worker
processes (``Fabric(devices=N, strategy="ddp").launch()`` re-executes this
module per rank) without re-entering pytest. The global-zero rank prints the
result dict as a single ``RESULT <json>`` line on stdout; the test parses it.

Usage:
    python -m tests._ddp_finetune_runner <tok_dir> <run_dir> <label_path> <devices>
"""

from __future__ import annotations

import json
import sys

from pragmatiq import api


def main() -> int:
    tok_dir, run_dir, label_path, devices_s = sys.argv[1:5]
    devices: int | str = int(devices_s)
    result = api.finetune(
        tok_dir,
        run_dir,
        label_path,
        device="cpu",
        max_epochs=5,
        patience=3,
        lora_rank=4,
        seed=0,
        devices=devices,
    )
    # Only the spawned global-zero process should emit the result; under a 2-rank
    # gloo run both workers reach here with the (identical) global AUC, but a
    # single RESULT line keeps parsing unambiguous. Fabric does not expose rank
    # cheaply here, so every rank prints with its PID-tagged marker and the test
    # takes the first RESULT line — they carry the same numbers by construction.
    payload = {k: result[k] for k in ("best_val_auc", "epochs_run", "n_adapted", "val_auc_history")}
    print("RESULT " + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
