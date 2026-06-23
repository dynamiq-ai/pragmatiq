#!/usr/bin/env bash
# Acceptance check — object-store staging at the api boundary.
#
#   1. storage unit tests (put_dir + staging context manager)
#   2. pipeline equivalence: local vs memory:// staging (synthesize→tokenize→pretrain→embed)
#   3. negative test: remote embed with nonexistent run raises
set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/gates/_env.sh

echo "=== storage unit tests ==="
"$PY" -m pytest tests/test_storage.py -q

echo "=== staging unit tests + pipeline equivalence ==="
"$PY" -m pytest tests/test_storage_pipeline.py -q -m "not slow" \
    --tb=short

echo "=== pipeline equivalence (slow, end-to-end) ==="
"$PY" -m pytest tests/test_storage_pipeline.py::test_local_vs_remote_pipeline_equivalence \
    tests/test_storage_pipeline.py::test_embed_missing_remote_run_raises \
    -q --tb=short

echo ""
echo "STORAGE STAGING CHECKS GREEN"
