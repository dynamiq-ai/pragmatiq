# pragmatiq cloud integrations

pragmatiq ships thin cloud adapter classes in `integrations/` that package a
trained run directory into cloud-native deployable artifacts (SageMaker and
Databricks; earlier stub adapters were removed in 1.1.0 — the generic Triton
image covers every other platform, see below). This document describes the
status of each adapter and provides runbooks for operators.

> **Attribution:** pragmatiq is an independent implementation inspired by the
> PRAGMA paper (arXiv 2604.08649) and is not affiliated with or endorsed by
> Revolut.

---

## Status table

| Adapter    | Status            | What `package()` / `manifest()` produce               | Live ops                              |
|------------|-------------------|-------------------------------------------------------|---------------------------------------|
| SageMaker  | **Real**          | `model.tar.gz` (BYOC Triton layout)                   | `push()` uploads to S3; `healthcheck()` hits endpoint |
| Databricks | **Real**          | MLflow pyfunc artifact directory                      | `register()` logs to Unity Catalog; `healthcheck()` hits serving endpoint |

---

## Shared serving contract

All adapters speak the same wire format defined in
`pragmatiq.inference.serve.contract`:

- **Container port:** `8000`
- **Health path:** `/v2/health/ready`
- **Infer path:** `/v2/models/pragmatiq_embedder/infer`
- **Wire encoding:** `encode_request(records)` → UTF-8 JSON bytes
  (`json.dumps(records).encode("utf-8")`); `decode_request(raw)` → `list[dict]`

The contract is tested in `tests/contract/` and is independent of the cloud
adapter.  Every adapter's `healthcheck()` builds its request via
`encode_request` so the format is consistent across adapters.

---

## Adapter reference

### SageMaker (`integrations.sagemaker.SageMakerAdapter`)

**Status: Real** — offline packaging + live deploy both implemented.

**What is implemented:**
- `manifest()` — returns the SageMaker `CreateModel` + `CreateEndpointConfig`
  parameters as a plain dict.
- `package(run_dir, dest, image)` — builds `model.tar.gz` in the BYOC Triton
  layout (`run_dir/` inside the archive; `PRAGMATIQ_RUN=/opt/ml/model/run_dir`).
- `push(artifact_path, role_arn, s3_bucket, ...)` — uploads the tarball to S3
  (requires `boto3`).
- `healthcheck(endpoint)` — invokes the SageMaker endpoint with a contract
  payload (requires `boto3`). The manifest's container env carries
  `PRAGMATIQ_RUN`; serving picks the instance's GPU automatically, set
  `PRAGMATIQ_SERVE_CPU=1` for a CPU instance type.

**Runbook:**
```bash
# 1. Build the artifact
python -c "
from integrations.sagemaker import SageMakerAdapter
a = SageMakerAdapter(image='123456789012.dkr.ecr.us-east-1.amazonaws.com/pragmatiq:latest')
a.package('runs/my-run', dest='/tmp/model.tar.gz', image=a._image)
"

# 2. Push to S3
python -c "
from integrations.sagemaker import SageMakerAdapter
a = SageMakerAdapter(image='...')
uri = a.push('/tmp/model.tar.gz', role_arn='arn:aws:iam::...', s3_bucket='my-bucket')
print('S3 URI:', uri)
"

# 3. Create the SageMaker Model + Endpoint via AWS CLI or boto3
#    (see manifest() output for the exact parameters)
```

---

### Databricks (`integrations.databricks.DatabricksAdapter`)

**Status: Real** — offline packaging + live register both implemented.

**What is implemented:**
- `manifest()` — returns the Unity Catalog model URI, pyfunc entry point, and
  MLflow signature.
- `package(run_dir, dest, image)` — writes an MLflow pyfunc artifact directory
  (`MLmodel` + `run_dir/` + `requirements.txt`).
- `register(artifact_path, ...)` — registers the pyfunc in Unity Catalog via
  MLflow (requires `mlflow[databricks]`).
- `healthcheck(endpoint)` — POSTs a contract payload to the Databricks Model
  Serving HTTPS endpoint (requires `requests`).

**Runbook:**
```bash
# 1. Package
python -c "
from integrations.databricks import DatabricksAdapter
a = DatabricksAdapter(catalog='main', schema='pragmatiq', model_name='embedder')
a.package('runs/my-run', dest='/tmp/pyfunc_artifact', image='unused')
"

# 2. Register in Unity Catalog (requires mlflow + Databricks workspace config)
python -c "
from integrations.databricks import DatabricksAdapter
a = DatabricksAdapter(catalog='main', schema='pragmatiq', model_name='embedder')
version_uri = a.register('/tmp/pyfunc_artifact')
print('Registered:', version_uri)
"

# 3. Create a Model Serving endpoint via the Databricks console or REST API.
```

---

### Other platforms (AKS, GKE, bare Kubernetes, any GPU cloud)

No adapter code is needed for platforms without a managed model-serving
product: the serving image built by `scripts/deploy_serving.sh` exposes the
shared serving contract above, so deploy it like any Triton container —
mount or stage the run directory at the path `PRAGMATIQ_RUN` points to, expose
port 8000, and use `/v2/health/ready` as the readiness probe. The image is
GPU-first (the python backend serves in bf16 on the node's GPU); for CPU-only
nodes mount `deploy/triton/config.cpu.pbtxt` over the model's `config.pbtxt`
and set `PRAGMATIQ_SERVE_CPU=1`, exactly as `deploy/docker-compose.cpu.yaml`
does. `PRAGMATIQ_SERVE_MAX_RECORDS` / `PRAGMATIQ_SERVE_TOKEN_BUDGET` cap one
request (defaults 1024 users / 16384 tokens per forward).

---

## Extending with new adapters

To add a new cloud adapter, subclass or implement the `CloudAdapter` protocol
defined in `integrations/_base.py` with the required methods and attributes.
For new cloud adapters:

1. Create `integrations/<provider>/` with `_adapter.py` + `__init__.py`.
2. Implement `name`, `manifest()`, `package()`, `healthcheck()` following the
   `CloudAdapter` Protocol in `integrations/_base.py`.
3. Add offline tests in `tests/test_integrations_<provider>.py`.
4. Add the test file to `gate_integrations.sh`.
5. Update this document with the status row and runbook. Raise
   `MissingExtraError` (from `pragmatiq.core.errors`) for optional SDKs.
