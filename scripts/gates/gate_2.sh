#!/usr/bin/env bash
# Gate 2 — key-value-time tokenizer (Phase 2).
#
#   1. round-trip property tests (categorical exact, numeric bucket contains
#      value, BPE pieces, calendar features, positions)
#   2. unseen key/value -> [UNK] + warning, never KeyError
#   3. save/load with content-hash verification (tamper rejected)
#   4. vocab size in the expected range on synthetic data
#
# Runnable outside Claude Code: bash scripts/gates/gate_2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== gate 2.1-2.3: tokenizer property / unknown / save-load tests ==="
$PY -m pytest tests/test_tokenizer.py -q

echo "=== gate 2.4: time encoding + vocab range on synthetic data ==="
$PY - <<'EOF'
import math, tempfile
from pathlib import Path
from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig, time_encode

# SPEC: time encoding is exactly 8*ln(1+dt/8)
for dt in (0.0, 8.0, 3600.0, 86400.0):
    assert abs(float(time_encode(dt)) - 8.0 * math.log1p(dt / 8.0)) < 1e-9, dt

work = Path(tempfile.mkdtemp())
generate(WorldConfig(n_users=600, months=18, n_merchants=4000, seed=2,
                     eval_month_credit=5, eval_month_short=11),
         work, n_workers=0, write_report=False)
tok = PragmaTokenizer(TokenizerConfig(target_vocab=28000)).fit(work)
print(f"keys={len(tok.key_vocab)} vocab={tok.vocab_size} "
      f"numeric={sum(v=='numeric' for v in tok.field_kind.values())} "
      f"bpe={'yes' if tok.bpe is not None else 'no'}")
assert 15 <= len(tok.key_vocab) <= 80, f"key count {len(tok.key_vocab)} out of ~60 range"
assert tok.vocab_size <= 28000 + 64, f"vocab {tok.vocab_size} exceeds target"
assert tok.field_kind["amount"] == "numeric"
assert tok.field_kind["mcc"] == "categorical"
print("tokenizer gate OK")
EOF

echo ""
echo "GATE 2 GREEN"
