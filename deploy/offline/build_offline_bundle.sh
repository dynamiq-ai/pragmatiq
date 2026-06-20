#!/usr/bin/env bash
# Build an offline wheel bundle for air-gapped pragmatiq installs.
#
# Run this on a CONNECTED host.  The downloaded wheels are placed in
# ./offline_bundle/ and can be transferred to an air-gapped environment.
#
# Usage:
#   bash deploy/offline/build_offline_bundle.sh [--extras serve|full|both]
#
# Options:
#   --extras serve   Download pragmatiq[serve] wheels only (default)
#   --extras full    Download pragmatiq[full] wheels only
#   --extras both    Download both pragmatiq[serve] and pragmatiq[full]
#
# On the air-gapped host:
#   pip install --no-index --find-links=./offline_bundle/ 'pragmatiq[serve]'
set -euo pipefail

cd "$(dirname "$0")/../.."

EXTRAS="serve"
BUNDLE_DIR="./offline_bundle"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --extras)
            EXTRAS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

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

# ── Check network connectivity ───────────────────────────────────────────────
echo "Checking network connectivity..."
if command -v curl &>/dev/null; then
    if ! curl --silent --max-time 10 --head https://pypi.org > /dev/null 2>&1; then
        echo "ERROR: Cannot reach https://pypi.org — network required for bundle creation." >&2
        echo "       Run this script on a connected host, then transfer offline_bundle/ to the air-gapped host." >&2
        exit 1
    fi
elif command -v wget &>/dev/null; then
    if ! wget --quiet --timeout=10 --spider https://pypi.org 2>/dev/null; then
        echo "ERROR: Cannot reach https://pypi.org — network required for bundle creation." >&2
        echo "       Run this script on a connected host, then transfer offline_bundle/ to the air-gapped host." >&2
        exit 1
    fi
else
    echo "WARNING: Neither curl nor wget found — cannot verify network. Proceeding anyway..." >&2
fi
echo "Network OK."

# ── Resolve pragmatiq version ────────────────────────────────────────────────
if "$PY" -c "import pragmatiq" 2>/dev/null; then
    VERSION=$("$PY" -c "import pragmatiq; print(pragmatiq.__version__)")
    echo "pragmatiq version: $VERSION"
else
    VERSION="(not installed locally)"
    echo "Note: pragmatiq not installed in current env — downloading latest from PyPI."
fi

# ── Create bundle directory ──────────────────────────────────────────────────
mkdir -p "$BUNDLE_DIR"
echo "Bundle directory: $BUNDLE_DIR"

# ── Download wheels ──────────────────────────────────────────────────────────
download_extra() {
    local extra="$1"
    echo ""
    echo "=== Downloading pragmatiq[$extra] wheels ==="
    "$PY" -m pip download \
        "pragmatiq[$extra]" \
        --dest "$BUNDLE_DIR" \
        --prefer-binary
    echo "Done: pragmatiq[$extra]"
}

case "$EXTRAS" in
    serve)
        download_extra "serve"
        ;;
    full)
        download_extra "full"
        ;;
    both)
        download_extra "serve"
        download_extra "full"
        ;;
    *)
        echo "ERROR: Unknown --extras value: $EXTRAS (expected serve, full, or both)" >&2
        exit 1
        ;;
esac

# ── Summary ──────────────────────────────────────────────────────────────────
WHEEL_COUNT=$(find "$BUNDLE_DIR" -name "*.whl" -o -name "*.tar.gz" | wc -l | tr -d ' ')
echo ""
echo "============================================================"
echo "Offline bundle ready: $BUNDLE_DIR"
echo "  Packages downloaded: $WHEEL_COUNT"
echo ""
echo "Air-gapped install instructions:"
echo "  1. Transfer the '$BUNDLE_DIR/' directory to the air-gapped host."
echo "  2. On the air-gapped host, run:"
echo ""
if [[ "$EXTRAS" == "both" ]]; then
    echo "       pip install --no-index --find-links=./offline_bundle/ 'pragmatiq[serve]'"
    echo "     or:"
    echo "       pip install --no-index --find-links=./offline_bundle/ 'pragmatiq[full]'"
else
    echo "       pip install --no-index --find-links=./offline_bundle/ 'pragmatiq[$EXTRAS]'"
fi
echo ""
echo "  See deploy/offline/README.md for full instructions."
echo "============================================================"
