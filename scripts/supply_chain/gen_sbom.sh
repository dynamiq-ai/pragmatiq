#!/usr/bin/env bash
# Generate a CycloneDX SBOM for the current pragmatiq environment.
#
# Usage:
#   bash scripts/supply_chain/gen_sbom.sh
#
# Output:
#   sbom/pragmatiq-<version>.cdx.json
#
# In CI the cyclonedx-bom package is installed as part of the supply-chain job.
# For local runs it is installed on demand (into the active environment).
set -euo pipefail

cd "$(dirname "$0")/../.."

# ── Resolve Python interpreter ───────────────────────────────────────────────
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x ".venv/bin/python" ]; then
        PY=".venv/bin/python"
    else
        PY="python3"
    fi
fi
echo "Using Python: $PY ($($PY --version 2>&1))"

# ── Ensure cyclonedx-py is available ────────────────────────────────────────
if ! "$PY" -c "import cyclonedx_py" 2>/dev/null; then
    echo "cyclonedx-bom not found — installing..."
    "$PY" -m pip install --quiet cyclonedx-bom
fi

# ── Resolve pragmatiq version ────────────────────────────────────────────────
VERSION=$("$PY" -c "import pragmatiq; print(pragmatiq.__version__)")
echo "pragmatiq version: $VERSION"

# ── Create output directory ──────────────────────────────────────────────────
mkdir -p sbom

OUTFILE="sbom/pragmatiq-${VERSION}.cdx.json"

# ── Generate SBOM ────────────────────────────────────────────────────────────
echo "Generating SBOM → $OUTFILE"
"$PY" -m cyclonedx_py environment \
    --output-format JSON \
    --output-file "$OUTFILE"

echo ""
echo "SBOM generated: $OUTFILE"
echo "Entries: $("$PY" -c "import json; d=json.load(open('$OUTFILE')); print(len(d.get('components', [])))" 2>/dev/null || echo '(count unavailable)')"
