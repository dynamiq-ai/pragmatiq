#!/usr/bin/env bash
# Gate 1 — synthetic data generator acceptance (Phase 1).
#
#   1. unit tests for the synthetic modules
#   2. determinism: same seed -> identical file hashes (also across n_workers)
#   3. throughput: 100k users < 10 min on 8 cores
#        - PRAGMATIQ_GATE_FULL=1 : actually generate 100k users
#        - default (CI)          : generate 4k users, extrapolate linearly
#   4. credit GBDT baseline AUC in the realistic band (leakage guard < 0.95)
#
# Runnable outside Claude Code: bash scripts/gates/gate_1.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/gates/_env.sh
FULL=${PRAGMATIQ_GATE_FULL:-0}
WORK=$(mktemp -d -t gate1.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

echo "=== gate 1.1: unit tests (synthetic modules) ==="
$PY -m pytest tests/test_synthetic_world.py tests/test_synthetic_generate.py \
    tests/test_synthetic_labels.py -q

echo "=== gate 1.2: determinism (covered by tests, re-asserted here) ==="
$PY - "$WORK" <<'EOF'
import hashlib, sys
from pathlib import Path
from pragmatiq.data.synthetic import WorldConfig, generate
work = Path(sys.argv[1])
kw = dict(n_users=300, months=16, n_merchants=1500, mule_ring_count=1, seed=77,
          eval_month_credit=4, eval_month_short=9)
generate(WorldConfig(**kw), work/"a", n_workers=0, write_report=False)
generate(WorldConfig(**kw), work/"b", n_workers=2, write_report=False)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
for rel in ("events.parquet", "profiles.parquet", "transfers.parquet",
            "labels/default_12m.parquet"):
    assert sha(work/"a"/rel) == sha(work/"b"/rel), f"non-deterministic: {rel}"
print("determinism OK (seed-stable, worker-count invariant)")
EOF

echo "=== gate 1.3: throughput (100k users < 10 min on 8 cores) ==="
$PY - <<EOF
import multiprocessing, time
from pragmatiq.data.synthetic import WorldConfig, generate
import tempfile
full = "$FULL" == "1"
cores = multiprocessing.cpu_count()
n = 100_000 if full else 4_000
workers = min(cores, 8)
t0 = time.time()
m = generate(WorldConfig(n_users=n, seed=0), tempfile.mkdtemp(), n_workers=workers,
             write_report=full)
dt = time.time() - t0
ups = n / dt
# scale measured throughput to 8 cores (linear in workers; phase B dominates)
ups8 = ups * (8 / workers)
proj = 100_000 / ups8
print(f"n={n} workers={workers} elapsed={dt:.1f}s -> {ups:.0f} users/s "
      f"(8-core projection: {proj:.0f}s for 100k)")
assert proj < 600, f"projected 100k time {proj:.0f}s exceeds 10 min budget"
print("throughput OK")
EOF

echo "=== gate 1.4: credit GBDT baseline AUC in [0.72, 0.88], hard fail > 0.95 ==="
DATA="$WORK/auc_data"
if [ "$FULL" = "1" ]; then N_AUC=20000; else N_AUC=6000; fi
$PY - "$DATA" "$N_AUC" <<'EOF'
import sys
from pragmatiq.data.synthetic import WorldConfig, generate
generate(WorldConfig(n_users=int(sys.argv[2]), seed=3), sys.argv[1],
         n_workers=4, write_report=False)
EOF
$PY tests/baselines/credit_gbdt.py --data "$DATA" --out-json "$WORK/auc.json"
$PY - "$WORK/auc.json" <<'EOF'
import json, sys
res = json.load(open(sys.argv[1]))
auc = res["auc"]
assert auc < 0.95, f"AUC {auc} > 0.95: leakage / unrealistic separability"
assert 0.72 <= auc <= 0.88, f"AUC {auc} outside [0.72, 0.88] band (CI band; SPEC asks ~0.75-0.85)"
prev = res["prevalence"]
assert 0.012 <= prev <= 0.05, f"default prevalence {prev} far from configured 0.03"
print(f"baseline OK: AUC={auc} prevalence={prev}")
EOF

echo ""
echo "GATE 1 GREEN"
