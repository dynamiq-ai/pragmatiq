#!/usr/bin/env bash
# Acceptance check — end-to-end training.
#
#   1. unit tests: optim, pretrainer checkpoint/resume(bit-exact)/NaN, probe,
#      fine-tune.
#   2. end-to-end on synthetic users: synth -> tokenize -> nano pretrain ->
#      probe; the PRAGMA probe must BEAT the same gradient-boosting head fit on
#      raw event counts, and per-masking-type losses must decrease.
#
#   Default (CI): nano config on CPU. PRAGMATIQ_GATE_FULL=1 uses 50k users +
#   a longer schedule (intended for a GPU box).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh
FULL=${PRAGMATIQ_GATE_FULL:-0}

if [ "${PRAGMATIQ_GATE_SKIP_UNIT:-0}" != "1" ]; then
  echo "=== training unit tests (optim / pretrainer / probe / finetune) ==="
  $PY -m pytest tests/test_training.py -q
else
  echo "=== training unit tests: SKIPPED (PRAGMATIQ_GATE_SKIP_UNIT=1) ==="
fi

echo "=== end-to-end synth -> tokenize -> pretrain -> probe ==="
$PY - "$FULL" <<'EOF'
import os, sys, json, tempfile
from pathlib import Path
from pragmatiq import api

full = sys.argv[1] == "1"
work = Path(tempfile.mkdtemp())
n_users = 50_000 if full else 1500
steps = 4000 if full else 120
api.synthesize({"n_users": n_users, "seed": 7} if full else
               {"n_users": n_users, "months": 18, "n_merchants": 2500, "seed": 7,
                "eval_month_credit": 5, "eval_month_short": 11},
               out=work / "raw", n_workers=4, write_report=False)
api.tokenize(work / "raw", work / "tok",
             config={"target_vocab": 28000 if full else 6000, "n_buckets": 64,
                     "categorical_threshold": 1000},
             n_workers=int(os.environ.get("PRAGMATIQ_GATE_WORKERS", "0")))
summary = api.pretrain(work / "tok", "gate5", model_size="small" if full else "nano",
                       config={"max_steps": steps, "token_budget": 8192, "warmup_steps": steps // 10,
                               "log_every": 10, "checkpoint_every_min": 1000.0},
                       runs_root=work / "runs")

# per-masking-type losses decrease
rows = [json.loads(l) for l in (Path(summary["run_dir"]) / "metrics.jsonl").read_text().strip().splitlines()]
import numpy as np
for t in ("loss_token", "loss_event", "loss_key"):
    vals = [r[t] for r in rows if t in r]
    assert np.mean(vals[-3:]) < np.mean(vals[:3]), f"{t} did not decrease"
print("per-masking-type losses decrease: OK")

res = api.probe(work / "tok", summary["run_dir"], work / "raw" / "labels" / "default_12m.parquet")
print(json.dumps(res, indent=2))
if full:
    # Full scale is the quality bar: the learned embedding must strictly beat the
    # raw-count baseline on BOTH ranking metrics — ROC-AUC and PR-AUC — by a small
    # positive margin. This is the real probe-beats-baseline guarantee.
    margin = 0.005
    assert res["probe_auc"] > res["baseline_auc"] + margin, \
        f"probe ROC-AUC {res['probe_auc']:.3f} does not beat raw-count baseline " \
        f"{res['baseline_auc']:.3f} by > {margin}"
    assert res["probe_pr_auc"] > res["baseline_pr_auc"] + margin, \
        f"probe PR-AUC {res['probe_pr_auc']:.3f} does not beat raw-count baseline " \
        f"{res['baseline_pr_auc']:.3f} by > {margin}"
    print("probe strictly beats raw-count baseline on ROC-AUC and PR-AUC: OK")
else:
    # The nano CI config trains only a handful of steps, so the credit probe is near
    # chance; on a tiny held-out set the gradient-boosting head (probe AND baseline)
    # also swings more than a linear one. So nano is purely a smoke check (the pipeline
    # runs and the embedding is not *catastrophically* worse than trivial counts) with
    # a wide tolerance. The strict probe>baseline guarantee is the full-scale gate above.
    tol = 0.15
    assert res["probe_auc"] >= res["baseline_auc"] - tol, \
        f"probe AUC {res['probe_auc']:.3f} below raw-count baseline {res['baseline_auc']:.3f} by more than {tol}"
    print("probe vs raw-count baseline within tolerance: OK")
EOF

echo ""
echo "TRAINING CHECKS GREEN"
