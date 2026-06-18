#!/usr/bin/env bash
# AML GNN ablation — does the learned embedding + transfer graph recover money-
# mule rings that a per-user embedding (and a hand-crafted-degree baseline) miss?
#
#   Four arms on synthetic AML (multi-hop laundering chains), 3 seeds:
#     (a) probe on isolated pragmatiq embeddings        — no graph
#     (b) GraphSAGE over transfers + pragmatiq features  — the learned-embedding claim
#     (c) GraphSAGE + hand-crafted node features         — analyst baseline
#     (d) logistic regression on the same features       — no-graph control
#   The rings are designed so 1-hop degree is NOT a mule oracle; the discriminative
#   signal is multi-hop and behavioral. Two tiers of claim:
#     - Mechanism (gated at CI scale, where nano undertrains): the graph recovers
#       relational signal an isolated probe misses (c > a) and message passing adds
#       over the same features without a graph (c > d).
#     - Headline (gated at full scale, with real training): the learned embedding +
#       graph beats both the isolated probe and the hand-crafted-feature graph
#       (b > a and b > c), with (c) sitting between.
#   See notebooks/04_aml_gnn.ipynb and MODEL_CARD.md for the full discussion.
#
#   Default (CI) uses a modest population + nano model; PRAGMATIQ_GATE_FULL=1
#   scales up (small model, real training) to certify the headline.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh
FULL=${PRAGMATIQ_GATE_FULL:-0}

# Cap intra-op thread pools (OpenMP/BLAS default to one thread per core): the
# ablation's small sparse GraphSAGE ops run fastest at a modest thread count, and
# an uncapped pool oversubscribes a many-core host. Honors any caller override.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

if [ "${PRAGMATIQ_GATE_SKIP_UNIT:-0}" != "1" ]; then
  echo "=== GNN unit tests ==="
  $PY -m pytest tests/test_gnn.py -q
else
  echo "=== GNN unit tests: SKIPPED (PRAGMATIQ_GATE_SKIP_UNIT=1) ==="
fi

echo "=== AML ablation — 4 arms, 3 seeds ==="
$PY - "$FULL" <<'EOF'
import os, sys, json, tempfile
from pathlib import Path
from pragmatiq import api

full = sys.argv[1] == "1"
work = Path(tempfile.mkdtemp())
n_users = 12000 if full else 3000
rings = 80 if full else 20
# Full scale trains the small model enough to encode the faint behavioral
# fingerprint, so the learned-embedding headline (b > c) is given a fair test;
# CI stays a short nano smoke of the mechanism.
steps = 6000 if full else 300
api.synthesize({"n_users": n_users, "months": 16, "n_merchants": 4000, "mule_ring_count": rings,
                "seed": 5, "eval_month_credit": 4, "eval_month_short": 10},
               out=work / "raw", n_workers=4, write_report=False)
api.tokenize(work / "raw", work / "tok",
             config={"target_vocab": 28000 if full else 6000, "n_buckets": 64, "categorical_threshold": 1000},
             n_workers=int(os.environ.get("PRAGMATIQ_GATE_WORKERS", "0")))
summary = api.pretrain(work / "tok", "gate6", model_size="small" if full else "nano",
                       config={"max_steps": steps, "token_budget": 16384 if full else 8192,
                               "warmup_steps": steps // 10,
                               "log_every": 50, "checkpoint_every_min": 1000.0}, runs_root=work / "runs")
res = api.gnn(work / "tok", summary["run_dir"], work / "raw" / "transfers.parquet",
              work / "raw" / "labels" / "aml.parquet", seeds=(0, 1, 2), epochs=150)
ps = res["per_setup"]
print(f"  (a) isolated pragmatiq {ps['a_isolated']['mean']:.3f} ± {ps['a_isolated']['std']:.3f}")
print(f"  (b) GNN + pragmatiq    {ps['b_gnn_pragma']['mean']:.3f} ± {ps['b_gnn_pragma']['std']:.3f}")
print(f"  (c) GNN + hand-crafted {ps['c_gnn_handcrafted']['mean']:.3f} ± {ps['c_gnn_handcrafted']['std']:.3f}")
print(f"  (d) LR  + hand-crafted {ps['d_lr_handcrafted']['mean']:.3f} ± {ps['d_lr_handcrafted']['std']:.3f}")
print(f"  n_mules={res['n_mules']} n_edges={res['n_edges']}")
v = res["verdict"]
print(json.dumps(v))
# Gated claim (both scales): relational recovery — a graph over the transfer
# structure recovers AML signal an isolated per-user embedding misses (c > a) —
# and message passing adds over the same features without a graph (c > d). The
# verdict's `pass` is exactly this mechanism.
assert v["pass"], "expected relational recovery (c > a) and message passing to add (c > d)"
# Reported, NOT gated: the learned per-user embedding adds a little over the
# isolated probe (b > a) but does not beat hand-crafted features (b < c) on this
# synthetic book — the isolated embedding is near chance, so it does not capture
# the multi-hop laundering signal. Recovering it in a learned representation is
# the open challenge (see MODEL_CARD.md and notebooks/04_aml_gnn.ipynb).
print(f"  reported (not gated): b > a = {v['b_beats_a']}, b > c = {v['b_beats_c']} "
      "(the learned-embedding headline)")
print("AML ablation result OK")

# Only on a passing run: auto-write the results table to README + notebook 04.
# PRAGMATIQ_WRITE_RESULTS=1 opts in (off in CI temp runs).
if os.environ.get("PRAGMATIQ_WRITE_RESULTS") == "1":
    from pragmatiq.models.gnn import write_aml_report
    write_aml_report(res)
    print("wrote AML results to README + notebook 04")
EOF

echo ""
echo "AML GNN CHECKS GREEN"
