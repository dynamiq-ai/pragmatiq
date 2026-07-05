#!/usr/bin/env bash
# Build an offline wheel bundle for air-gapped pragmatiq installs.
#
# Run this on a CONNECTED host.  The downloaded wheels are placed in
# ./offline_bundle/ and can be transferred to an air-gapped environment.
#
# Usage:
#   bash deploy/offline/build_offline_bundle.sh [--extras serve|full|both]
#       [--platform TAG]... [--python-version X.Y] [--abi TAG] [--implementation NAME]
#
# Options:
#   --extras serve          Download pragmatiq[serve] wheels only (default)
#   --extras full           Download pragmatiq[full] wheels only
#   --extras both           Download both pragmatiq[serve] and pragmatiq[full]
#   --platform TAG          Target platform tag of the AIR-GAPPED host (e.g.
#                           manylinux2014_x86_64). Repeatable for compatible tags.
#   --python-version X.Y    Target Python version of the air-gapped host (e.g. 3.11)
#   --abi TAG               Target ABI tag (e.g. cp311)
#   --implementation NAME   Target implementation (e.g. cp); rarely needed
#
# Without target options, wheels are resolved for THIS build host's
# platform/Python — the bundle only installs on a matching air-gapped host.
# With any target option, pip requires wheels for everything
# (--only-binary=:all:), so sdist-only dependencies will fail the download.
#
# On the air-gapped host:
#   pip install --no-index --find-links=./offline_bundle/ 'pragmatiq[serve]'
set -euo pipefail

cd "$(dirname "$0")/../.."

EXTRAS="serve"
BUNDLE_DIR="./offline_bundle"
PLATFORMS=()
PY_VERSION=""
ABI=""
IMPLEMENTATION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --extras)
            EXTRAS="$2"
            shift 2
            ;;
        --platform)
            PLATFORMS+=("$2")
            shift 2
            ;;
        --python-version)
            PY_VERSION="$2"
            shift 2
            ;;
        --abi)
            ABI="$2"
            shift 2
            ;;
        --implementation)
            IMPLEMENTATION="$2"
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

# ── Resolve the target platform ──────────────────────────────────────────────
# pip only allows --platform/--python-version/--abi/--implementation together
# with --only-binary (or --no-deps), so target flags force an all-wheels
# download; without them, pip resolves for the build host and the two hosts
# must match.
TARGET_ARGS=()
for p in ${PLATFORMS+"${PLATFORMS[@]}"}; do
    TARGET_ARGS+=(--platform "$p")
done
[ -n "$PY_VERSION" ]     && TARGET_ARGS+=(--python-version "$PY_VERSION")
[ -n "$ABI" ]            && TARGET_ARGS+=(--abi "$ABI")
[ -n "$IMPLEMENTATION" ] && TARGET_ARGS+=(--implementation "$IMPLEMENTATION")

echo ""
echo "============================================================"
if [ ${#TARGET_ARGS[@]} -gt 0 ]; then
    TARGET_ARGS+=(--only-binary=:all:)
    echo "TARGET PLATFORM: cross-platform download for the air-gapped host:"
    [ ${#PLATFORMS[@]} -gt 0 ] && echo "  platform tag(s): ${PLATFORMS[*]}"
    [ -n "$PY_VERSION" ]       && echo "  python version : $PY_VERSION"
    [ -n "$ABI" ]              && echo "  abi            : $ABI"
    [ -n "$IMPLEMENTATION" ]   && echo "  implementation : $IMPLEMENTATION"
    echo "  (wheels only: --only-binary=:all: — sdist-only deps will fail here)"
else
    echo "TARGET PLATFORM ASSUMPTION: no --platform/--python-version given —"
    echo "  wheels are resolved for THIS build host:"
    echo "    OS/arch : $(uname -s)/$(uname -m)"
    echo "    Python  : $($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    echo "  The bundle will only install on an air-gapped host with a matching"
    echo "  platform and Python. If it differs, pass --platform / --python-version"
    echo "  / --abi (e.g. --platform manylinux2014_x86_64 --python-version 3.11 --abi cp311)."
fi
echo "============================================================"

# ── Download wheels ──────────────────────────────────────────────────────────
download_extra() {
    local extra="$1"
    echo ""
    echo "=== Downloading pragmatiq[$extra] wheels ==="
    "$PY" -m pip download \
        "pragmatiq[$extra]" \
        --dest "$BUNDLE_DIR" \
        --prefer-binary \
        ${TARGET_ARGS+"${TARGET_ARGS[@]}"}
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
