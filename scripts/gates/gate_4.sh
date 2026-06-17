#!/usr/bin/env bash
# Gate 4 — model (Phase 4).
#
#   1. shape tests across all three sizes (forward produces token/event/user reprs)
#   2. param counts within 10% of 10M / 100M / 1B at the canonical ~28k vocab
#   3. masking unit tests (.15/.10/.10 union, 10% [UNK] excluded from loss)
#   4. gradcheck on TimeRoPE
#   5. per-user equivalence: a user's embedding is identical alone or batched
#      (varlen attention has no cross-user contamination) + LoRA inject/merge
#
# Runnable outside Claude Code: bash scripts/gates/gate_4.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== gate 4.1-4.5: model shape / param / masking / rope / equivalence tests ==="
$PY -m pytest tests/test_models.py -q

echo "=== gate 4 summary: param counts at vocab=28000 ==="
$PY - <<'EOF'
import torch
from pragmatiq.models import ModelConfig, PragmaModel, MLMHead
for size, target in (("small", 10e6), ("medium", 100e6), ("large", 1e9)):
    cfg = ModelConfig.preset(size, vocab_size=28000)
    with torch.device("meta"):
        m = PragmaModel(cfg)
        h = MLMHead(cfg.dim)
    n = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in h.parameters())
    pct = 100 * (n - target) / target
    flag = "OK" if abs(pct) <= 10 else "FAIL"
    print(f"  {size:6s} {n/1e6:8.1f}M  ({pct:+.1f}% vs {target/1e6:.0f}M)  [{flag}]")
    assert abs(pct) <= 10, f"{size} out of 10% band"
print("param counts OK")
EOF

echo ""
echo "GATE 4 GREEN"
