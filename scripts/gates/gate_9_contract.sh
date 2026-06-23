#!/usr/bin/env bash
# Acceptance check — public-API stability contract harness.
#
#   Runs tests/contract/ to verify that the pinned public surface
#   (api.py function signatures, CLI command paths + param names,
#   PragmaModel.from_pretrained/embed_records signatures) is intact.
#
#   This gate is ADDITIVE-ONLY: it does not modify production code.
#   It is the safety net that must pass before any restructure task.
#
#   PRAGMATIQ_GATE_FULL=1 has no effect here — the contract suite is
#   always deterministic and fast (no network, no heavyweight training).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== public-API contract tests ==="
$PY -m pytest tests/contract -q

echo ""
echo "CONTRACT CHECKS GREEN"
