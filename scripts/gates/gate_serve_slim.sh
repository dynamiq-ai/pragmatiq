#!/usr/bin/env bash
# Acceptance check — slim-serve install boundary.
#
#   Proves that the inference/embedding path (PragmaModel.embed_records) works
#   with lightning, torch_geometric, transformers, and lightgbm BLOCKED, and
#   that none of those heavy packages are imported as a side-effect.
#
#   The test uses a subprocess import-blocker technique so it runs correctly
#   inside any .venv that has all deps installed (the blocker makes them
#   unimportable in the child process regardless).
#
#   Stronger check (not run here, performed in CI serve-slim job):
#     python -m venv /tmp/pragmatiq-serve
#     /tmp/pragmatiq-serve/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
#     /tmp/pragmatiq-serve/bin/pip install -e ".[serve]" pytest
#     ! /tmp/pragmatiq-serve/bin/python -c "import lightning"   # must fail
#     /tmp/pragmatiq-serve/bin/python -m pytest tests/boundaries/test_serve_import_safe.py -q
set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/gates/_env.sh

echo "=== slim-serve boundary (import-blocker subprocess) ==="
"$PY" -m pytest tests/boundaries/test_serve_import_safe.py -q

echo ""
echo "SERVE-SLIM CHECKS GREEN"
