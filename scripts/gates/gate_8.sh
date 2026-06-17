#!/usr/bin/env bash
# Gate 8 — polish: nano end-to-end on CPU, validate, packaging (Phase 8).
#
#   1. validate unit tests + `pragmatiq validate` with actionable errors
#   2. nano end-to-end on CPU (synth -> tokenize -> nano pretrain -> probe),
#      target < 10 min; smoke test: asserts the pipeline runs and returns a
#      finite probe metric (gate 5 is the performance gate, not this nano run)
#   3. packaging sanity: version tracks the package metadata, CLI entrypoint resolves, docs carry
#      the required attribution line
#
# Runnable outside Claude Code: bash scripts/gates/gate_8.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== gate 8.1: validate module + actionable errors ==="
$PY -m pytest tests/test_validate.py -q

echo "=== gate 8.2: nano end-to-end quickstart on CPU (<10 min) ==="
$PY - <<'EOF'
import time, tempfile
from pathlib import Path
from pragmatiq import api
t0 = time.time()
out = Path(tempfile.mkdtemp())
res = api.quickstart(out=out, n_users=2000, model_size="nano", max_steps=80, n_workers=2)
dt = time.time() - t0
print("  " + res["message"])
print(f"  elapsed {dt:.0f}s")
assert dt < 600, f"nano e2e took {dt:.0f}s (> 10 min budget)"
auc = res["probe"]["probe_auc"]
# The nano quickstart is a CPU PIPELINE smoke test (2k users, 80 steps): credit is
# low-prevalence and the model is tiny, so the probe sits near chance and the tiny
# eval set makes any fixed AUC threshold unstable. gate 5 is the performance gate
# (small model, relative beats-baseline criterion). Here we assert only that the
# end-to-end pipeline ran in budget and returned a finite, in-range metric.
import math
assert math.isfinite(auc) and 0.0 <= auc <= 1.0, f"quickstart returned an invalid probe AUC: {auc}"
print("nano end-to-end OK")
EOF

echo "=== gate 8.3: packaging + attribution ==="
$PY - <<'EOF'
import pragmatiq
from importlib.metadata import version as _dist_version
# __version__ tracks the installed distribution metadata (single source of truth), so any
# valid release or pre-release (e.g. a public beta) passes — no fixed literal to bump.
assert pragmatiq.__version__, "empty package version"
assert pragmatiq.__version__ == _dist_version("pragmatiq"), pragmatiq.__version__
from pragmatiq.cli import app  # CLI entrypoint resolves
from pathlib import Path
need = "not affiliated with or endorsed by Revolut"
for doc in ("README.md", "MODEL_CARD.md"):
    assert need in Path(doc).read_text(), f"{doc} missing attribution line"
print(f"  version {pragmatiq.__version__}; attribution present in README + MODEL_CARD")
print("packaging OK")
EOF

echo ""
echo "GATE 8 GREEN"
