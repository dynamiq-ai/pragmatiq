#!/usr/bin/env bash
# Acceptance check — sharding, dataset, varlen collation.
#
#   1. unit tests: shard round-trip, dynamic batching, resumability
#   2. padding-equivalence: packed (block-diagonal) attention == padded
#      per-event attention (atol 1e-4, fp32) — the critical property
#   3. end-to-end smoke: synth -> tokenize -> shard -> dataloader yields
#      padding-free PackedBatches covering every user exactly once
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== sharding / dataset / padding-equivalence tests ==="
$PY -m pytest tests/test_sharding_dataset.py -q

echo "=== synth -> tokenize -> shard -> dataloader smoke ==="
$PY - <<'EOF'
import tempfile
from pathlib import Path
from pragmatiq import api
from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset

work = Path(tempfile.mkdtemp())
api.synthesize({"n_users": 200, "months": 16, "n_merchants": 1200, "mule_ring_count": 1,
                "seed": 5, "eval_month_credit": 4, "eval_month_short": 9},
               out=work / "raw", n_workers=0, write_report=False)
m = api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 5000, "n_buckets": 32, "categorical_threshold": 200})
assert m["n_users"] == 200, m
ds = ShardDataset(work / "tok")
sampler = DynamicBatchSampler(ds.index, token_budget=8192, seed=0)
sampler.set_epoch(0)
loader = ShardDataLoader(ds, sampler)
seen = 0
for batch in loader:
    assert batch.cu_seqlens_event[-1] == batch.n_tokens       # no padding
    assert batch.cu_seqlens_history[-1] == batch.n_events
    seen += batch.n_users
assert seen == 200, seen
ds.close()
print(f"smoke OK: vocab={m['vocab_size']} users={m['n_users']} "
      f"events={m['total_events']} tokens={m['total_tokens']}")
EOF

echo ""
echo "SHARDING CHECKS GREEN"
