#!/usr/bin/env bash
# Acceptance check — inference, serving, demo.
#
#   The serving requirement is "docker-compose up serves an embedding request end
#   to end; demo runs against a bundled tiny checkpoint." Docker isn't available in
#   CI, so this check validates the exact code paths the containers run:
#     1. inference unit tests (embedder, attribution, benchmark, ONNX)
#     2. the Triton python-backend request path (PragmaModel.embed_records on
#        plain dicts) end to end against a freshly trained nano checkpoint
#     3. deploy manifests are well-formed (compose services, Triton config)
#     4. demo/app.py imports/compiles
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gates/_env.sh

echo "=== inference unit tests ==="
$PY -m pytest tests/test_inference.py -q

echo "=== serving request path (Triton backend's embed_records) ==="
$PY - <<'EOF'
import json, tempfile
from pathlib import Path
import numpy as np
from pragmatiq import api
from pragmatiq.models.pragmatiq import PragmaModel

work = Path(tempfile.mkdtemp())
api.synthesize({"n_users": 300, "months": 14, "n_merchants": 800, "seed": 1,
                "eval_month_credit": 2, "eval_month_short": 8}, out=work/"raw",
               n_workers=2, write_report=False)
api.tokenize(work/"raw", work/"tok", config={"target_vocab": 4000, "n_buckets": 32})
summary = api.pretrain(work/"tok", "gate7", model_size="nano",
                       config={"max_steps": 40, "token_budget": 4096, "warmup_steps": 5},
                       runs_root=work/"runs")
# Exactly what deploy/triton/.../model.py:execute() does with a request payload.
model = PragmaModel.from_pretrained(summary["run_dir"])
records = [
    {"user_id": "svc_1", "events": [
        {"ts": 1_700_000_000_000_000, "source": "transaction",
         "fields": {"amount": "42.10", "mcc": "5411", "merchant": "TESCO 1", "txn_type": "card_payment"}},
        {"ts": 1_700_003_600_000_000, "source": "app",
         "fields": {"screen": "home", "action": "view"}},
    ], "attributes": {"country": "GB", "age_band": "30-39"}, "lifelong": []},
    {"user_id": "svc_2", "events": [
        {"ts": 1_700_000_000_000_000, "source": "transaction",
         "fields": {"amount": "9.99", "mcc": "5814", "merchant": "MCDONALDS", "txn_type": "card_payment"}},
    ], "attributes": {"country": "IE"}, "lifelong": []},
]
emb = model.embed_records(records).astype(np.float32)
assert emb.shape[0] == 2, emb.shape
assert np.isfinite(emb).all()
# unseen keys/values must NOT raise (global rule 4)
weird = [{"user_id": "svc_3", "events": [
    {"ts": 1_700_000_000_000_000, "source": "transaction",
     "fields": {"amount": "1.00", "totally_new_key": "xyz"}}], "attributes": {}, "lifelong": []}]
emb2 = model.embed_records(weird)
assert emb2.shape == (1, emb.shape[1])
print(f"  embed_records OK: {emb.shape}; unseen-key request handled -> {emb2.shape}")
print("serving request path OK")
EOF

echo "=== deploy manifests well-formed ==="
$PY - <<'EOF'
from pathlib import Path
import yaml
compose = yaml.safe_load(Path("deploy/docker-compose.yaml").read_text())
services = set(compose.get("services", {}))
need = {"triton", "demo"}
assert need <= services, f"compose missing services: {need - services}"
cfg = Path("deploy/triton/model_repository/pragmatiq_embedder/config.pbtxt").read_text()
# python backend (faithful varlen path), 2-instance group, in-model batching
assert 'backend: "python"' in cfg, "Triton config must use the python backend"
assert "instance_group" in cfg and "count: 2" in cfg, "expected a 2-instance group"
assert "max_batch_size: 0" in cfg, "ragged in-model batching requires max_batch_size: 0"
print(f"  compose services: {sorted(services)}")
print("deploy manifests OK")
EOF

echo "=== demo compiles ==="
$PY -m py_compile demo/app.py && echo "  demo/app.py compiles OK"

echo ""
echo "INFERENCE CHECKS GREEN"
