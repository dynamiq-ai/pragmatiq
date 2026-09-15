#!/usr/bin/env bash
# Acceptance check — BYOC security / compliance (W7).
# Offline-provable parts only: no network required.
#
#   1. Static audit: scripts/supply_chain/no_phone_home.py
#   2. Dynamic proof: tests/boundaries/test_no_phone_home.py
#   3. The SBOM generator exists and is syntactically valid
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== static no-phone-home audit ==="
$PY scripts/supply_chain/no_phone_home.py

echo ""
echo "=== dynamic no-phone-home proof (sockets blocked, offline) ==="
$PY -m pytest tests/boundaries/test_no_phone_home.py -q

echo ""
echo "=== SBOM generator exists and is syntactically valid ==="
test -f scripts/supply_chain/gen_sbom.sh
bash -n scripts/supply_chain/gen_sbom.sh

echo ""
echo "BYOC SECURITY CHECKS GREEN"
