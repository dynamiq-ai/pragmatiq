#!/usr/bin/env bash
# Acceptance check — cloud adapter integrations (W6a + W6b).
#
# Runs all three integration test files (SageMaker, Databricks, Azure+Nebius
# stubs) to verify:
#   1. All adapters satisfy the CloudAdapter protocol shape.
#   2. Offline-testable methods (manifest, package) work without cloud SDKs.
#   3. Live-op guards (push/register/deploy_live) raise the correct errors.
#   4. Stub adapters generate real offline artifacts (Helm chart / job specs).
#
# NO live cloud calls are made — all tests run without credentials or network.
set -euo pipefail
cd "$(dirname "$0")/../.."

source scripts/gates/_env.sh

echo "=== integration adapter tests (offline — no cloud calls) ==="
"$PY" -m pytest \
    tests/test_integrations_sagemaker.py \
    tests/test_integrations_databricks.py \
    tests/test_integrations_stubs.py \
    -q

echo ""
echo "INTEGRATIONS CHECKS GREEN"
