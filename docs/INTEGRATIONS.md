# pragmatiq cloud integrations

pragmatiq ships thin cloud adapter classes in `integrations/` that package a
trained run directory into cloud-native deployable artifacts.  This document
describes the status of each adapter and provides runbooks for operators.

> **Attribution:** pragmatiq is an independent implementation inspired by the
> PRAGMA paper (arXiv 2604.08649) and is not affiliated with or endorsed by
> Revolut.

---

## Status table

| Adapter    | Status            | What `package()` / `manifest()` produce               | Live ops                              |
|------------|-------------------|-------------------------------------------------------|---------------------------------------|
| SageMaker  | **Real**          | `model.tar.gz` (BYOC Triton layout)                   | `push()` uploads to S3; `healthcheck()` hits endpoint |
| Databricks | **Real**          | MLflow pyfunc artifact directory                      | `register()` logs to Unity Catalog; `healthcheck()` hits serving endpoint |
| Azure      | **Stub + runbook**| Helm chart skeleton (`Chart.yaml` + `values.yaml` + `templates/`) | `deploy_live()` raises `NotImplementedError` — see runbook below |
| Nebius     | **Stub + runbook**| Job-spec YAML files (`serving_spec.yaml` + `batch_embed_job.yaml`) | `deploy_live()` raises `NotImplementedError` — see runbook below |

---

## Shared serving contract

All adapters speak the same wire format defined in
`pragmatiq.inference.serve.contract`:

- **Container port:** `8000`
- **Health path:** `/v2/health/ready`
- **Infer path:** `/v2/models/pragmatiq_embedder/infer`
- **Wire encoding:** `encode_request(records)` → raw `msgpack` bytes;
  `decode_request(raw)` → `list[dict]`

The contract is tested in `tests/contract/` and is independent of the cloud
adapter.  Every adapter's `healthcheck()` builds its request via
`encode_request` so the format is consistent across all four adapters.

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
  payload (requires `boto3`).

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

### Azure (`integrations.azure.AzureAdapter`)

**Status: Stub + runbook** — offline Helm chart generation is real; live AKS
deploy requires manual operator steps.

**What is implemented (offline, no cloud SDK):**
- `manifest()` — returns a declarative AKS / Helm deploy spec (namespace,
  image, replica count, contract port, storage PVC config).
- `package(run_dir, dest, image)` — writes a ready-to-use Helm chart skeleton:
  - `Chart.yaml` — Helm metadata.
  - `values.yaml` — image, port (8000), health path, PVC name, env vars.
  - `templates/deployment.yaml` — Kubernetes Deployment + Service using
    `{{ .Values.* }}` references throughout.

**What is NOT implemented (raises `NotImplementedError`):**
- `deploy_live()` — live AKS deploy is documented below, not automated.

**Runbook — Azure AKS deploy:**

```bash
# Prerequisites
# - Azure CLI: az login
# - kubectl configured for your AKS cluster: az aks get-credentials ...
# - Helm 3.x installed

IMAGE="myacr.azurecr.io/pragmatiq:latest"
RUN_DIR="runs/my-run"
DEST="/tmp/pragmatiq-helm"

# 1. Build the Helm chart skeleton
python -c "
from integrations.azure import AzureAdapter
a = AzureAdapter(image='$IMAGE')
a.package('$RUN_DIR', dest='$DEST', image='$IMAGE')
print('Helm chart written to:', '$DEST')
"

# 2. Push the Triton image to Azure Container Registry (ACR)
az acr login --name myacr
docker tag pragmatiq:latest myacr.azurecr.io/pragmatiq:latest
docker push myacr.azurecr.io/pragmatiq:latest

# 3. Stage the run directory to Azure Blob Storage
az storage blob upload-batch \
    --source "$RUN_DIR" \
    --destination "pragmatiq-runs/run_dir" \
    --account-name mystorageaccount

# 4. Create a PersistentVolumeClaim backed by Azure Blob CSI Driver
#    (see https://learn.microsoft.com/en-us/azure/aks/azure-blob-csi-driver)
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pragmatiq-run-pvc
  namespace: pragmatiq
spec:
  accessModes: [ReadWriteMany]
  storageClassName: azureblob-nfs-premium
  resources:
    requests:
      storage: 10Gi
EOF

# 5. Deploy with Helm
helm install pragmatiq-embedder "$DEST" \
    --namespace pragmatiq \
    --create-namespace \
    --set image.repository=myacr.azurecr.io/pragmatiq \
    --set image.tag=latest

# 6. Verify
kubectl rollout status deployment/pragmatiq-embedder -n pragmatiq
kubectl port-forward svc/pragmatiq-embedder 8000:8000 -n pragmatiq &
curl http://localhost:8000/v2/health/ready
```

---

### Nebius (`integrations.nebius.NebiusAdapter`)

**Status: Stub + runbook** — offline YAML spec generation is real; live Nebius
provisioning requires manual operator steps.

**What is implemented (offline, no cloud SDK):**
- `manifest()` — returns a declarative Nebius deploy spec covering both the
  Token Factory serving mode and the Soperator (Slurm-on-Kubernetes) batch
  embed mode.
- `package(run_dir, dest, image)` — writes two ready-to-submit YAML specs:
  - `serving_spec.yaml` — Nebius AI Token Factory model-serving spec with
    image, GPU config, S3 mount, and contract port.
  - `batch_embed_job.yaml` — Soperator `SlurmJob` spec for batch embedding,
    referencing the pragmatiq CLI `embed` command and Nebius Object Storage.

**What is NOT implemented (raises `NotImplementedError`):**
- `deploy_live()` — live Nebius provisioning is documented below, not automated.

**Runbook — Nebius Token Factory (serving):**

```bash
IMAGE="cr.eu-north1.nebius.cloud/pragmatiq:latest"
RUN_DIR="runs/my-run"
DEST="/tmp/nebius-specs"

# 1. Generate the job specs
python -c "
from integrations.nebius import NebiusAdapter
a = NebiusAdapter(
    image='$IMAGE',
    s3_bucket='my-pragmatiq-bucket',
)
a.package('$RUN_DIR', dest='$DEST', image='$IMAGE')
print('Specs written to:', '$DEST')
"

# 2. Push the image to Nebius Container Registry
docker tag pragmatiq:latest cr.eu-north1.nebius.cloud/pragmatiq:latest
docker push cr.eu-north1.nebius.cloud/pragmatiq:latest

# 3. Upload run directory to Nebius Object Storage (S3-compatible)
aws s3 sync "$RUN_DIR" s3://my-pragmatiq-bucket/run_dir \
    --endpoint-url https://storage.eu-north1.nebius.cloud:443

# 4. Submit to Token Factory
nebius ai token-factory model create --spec "$DEST/serving_spec.yaml"

# 5. Verify via the contract health path
curl https://<endpoint>.inference.eu-north1.nebius.cloud/v2/health/ready
```

**Runbook — Nebius Soperator (batch embed):**

```bash
# (after completing steps 1-3 above)

# 4. Submit the batch embed job
kubectl apply -f "$DEST/batch_embed_job.yaml"

# 5. Monitor job status
kubectl get slurmjobs -n pragmatiq
kubectl logs -l job-name=pragmatiq-embedder-embed -n pragmatiq
```

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
5. Update this document with the status row and runbook.
