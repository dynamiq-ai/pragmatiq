#!/usr/bin/env bash
# Turnkey serving deploy + smoke check (Phase 7).
#
# Builds the Triton image (with pragmatiq installed), boots tritonserver with a trained
# run mounted, waits for readiness, then sends a real embedding request through the HTTP
# API and verifies the [n_users, dim] response. Works for the default model and the
# PRAGMA+Nemotron variant. Designed for a GPU box (RunPod) but boots CPU-only too.
#
# Usage (--run is the trained run directory, the one that contains checkpoints/):
#   bash scripts/deploy_serving.sh                                            # default model
#   bash scripts/deploy_serving.sh --run runs/nemo --variant nemotron
#   bash scripts/deploy_serving.sh --keep                                     # leave it running
#
# The default matches `pragmatiq quickstart` (out=runs/quickstart), which lays out a
# self-contained workspace and writes the run under <out>/runs/<name>.
#
# Requires docker; uses the host GPU automatically when nvidia-smi is present.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

RUN_DIR="runs/quickstart/runs/quickstart"
VARIANT="default"
PORT=8000
METRICS_PORT=8002
NAME="pragmatiq-triton"
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN_DIR="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required for serving deploy." >&2; exit 1
fi
if [ ! -d "$RUN_DIR/checkpoints" ]; then
  echo "ERROR: '$RUN_DIR' is not a trained run (no checkpoints/). Train one first, e.g." >&2
  echo "       python -m pragmatiq.cli quickstart   # writes runs/quickstart/runs/quickstart" >&2
  exit 1
fi
RUN_ABS="$(cd "$RUN_DIR" && pwd)"

EXTRAS=""
TAG="latest"
if [ "$VARIANT" = "nemotron" ]; then EXTRAS="nemotron"; TAG="nemotron"; fi
IMAGE="pragmatiq-triton:${TAG}"

GPU_FLAG=""
if command -v nvidia-smi >/dev/null 2>&1; then GPU_FLAG="--gpus all"; echo "GPU detected → serving on CUDA"; fi

echo "=== building $IMAGE (EXTRAS='${EXTRAS}') ==="
docker build -f deploy/triton/Dockerfile --build-arg "EXTRAS=${EXTRAS}" -t "$IMAGE" .

cleanup() { [ "$KEEP" = "0" ] && docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "=== starting tritonserver ($NAME) ==="
# shellcheck disable=SC2086
docker run -d --rm --name "$NAME" $GPU_FLAG \
  -p "${PORT}:8000" -p "${METRICS_PORT}:8002" --shm-size 1g \
  -v "${REPO_ROOT}/deploy/triton/model_repository:/models/model_repository:ro" \
  -v "${RUN_ABS}:/models/run:ro" \
  "$IMAGE" \
  tritonserver --model-repository=/models/model_repository --metrics-port=8002 >/dev/null

echo "=== waiting for readiness (http://localhost:${PORT}/v2/health/ready) ==="
ready=0
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:${PORT}/v2/health/ready" >/dev/null 2>&1; then ready=1; break; fi
  sleep 3
done
if [ "$ready" != "1" ]; then
  echo "ERROR: Triton did not become ready; recent logs:" >&2
  docker logs --tail 50 "$NAME" >&2 || true
  exit 1
fi
echo "ready."

echo "=== embedding request smoke ==="
PYBIN="${PYTHON:-}"
[ -z "$PYBIN" ] && { [ -x .venv/bin/python ] && PYBIN=.venv/bin/python || PYBIN=python3; }
"$PYBIN" - "$PORT" <<'PY'
import json, sys, urllib.request

port = sys.argv[1]
records = [
    {"user_id": "svc_1", "events": [
        {"ts": 1_700_000_000_000_000, "source": "transaction",
         "fields": {"amount": "42.10", "mcc": "5411", "merchant": "TESCO 1"}}],
     "attributes": {"country": "GB"}, "lifelong": []},
    {"user_id": "svc_2", "events": [
        {"ts": 1_700_000_000_000_000, "source": "app",
         "fields": {"screen": "home", "action": "view"}}],
     "attributes": {}, "lifelong": []},
]
body = {"inputs": [{"name": "records_json", "datatype": "BYTES", "shape": [1],
                    "data": [json.dumps(records)]}]}
url = f"http://localhost:{port}/v2/models/pragmatiq_embedder/infer"
req = urllib.request.Request(url, data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as resp:
    out = json.loads(resp.read())
o = out["outputs"][0]
shape = o["shape"]
assert shape[0] == 2, f"expected 2 users, got shape {shape}"
print(f"OK: embeddings shape={shape}; first 4 dims of user 0: {o['data'][:4]}")
PY

echo ""
echo "SERVING SMOKE GREEN (variant=${VARIANT})"
[ "$KEEP" = "1" ] && echo "container '${NAME}' left running on port ${PORT} (curl http://localhost:${PORT}/v2/health/ready)"
