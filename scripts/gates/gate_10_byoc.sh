#!/usr/bin/env bash
# Acceptance check — BYOC security / compliance (W7).
# Offline-provable parts only: no network required.
#
#   1. Static audit: scripts/supply_chain/no_phone_home.py
#   2. Dynamic proof: tests/boundaries/test_no_phone_home.py
#   3. SBOM + offline scripts exist and are syntactically valid
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== static no-phone-home audit ==="
$PY scripts/supply_chain/no_phone_home.py

echo ""
echo "=== dynamic no-phone-home proof (sockets blocked, offline) ==="
$PY -m pytest tests/boundaries/test_no_phone_home.py -q

echo ""
echo "=== SBOM + offline scripts exist and are syntactically valid ==="
# Check files exist
test -f scripts/supply_chain/gen_sbom.sh
test -f deploy/offline/build_offline_bundle.sh
test -f deploy/offline/README.md
test -f sbom/README.md

# Syntax check
bash -n scripts/supply_chain/gen_sbom.sh
bash -n deploy/offline/build_offline_bundle.sh

echo ""
echo "BYOC SECURITY CHECKS GREEN"
