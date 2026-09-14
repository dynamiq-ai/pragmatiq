"""Nebius adapter implementation (STUB + job-spec skeleton, offline-testable).

Nebius AI Cloud offers two deployment modes relevant to pragmatiq:

1. **Token Factory** (model-serving): a managed inference endpoint backed by
   the Nebius AI Token Factory service, exposing an OpenAI-compatible API.
   The manifest describes the serving spec; live provisioning is via the
   Nebius AI console or CLI.

2. **Soperator / batch embed** (Slurm-on-Kubernetes): a ``sbatch``-style job
   that runs the pragmatiq ``embed`` command on a Nebius GPU pod, storing
   results to Nebius Object Storage (S3-compatible endpoint).

This adapter generates YAML specs for BOTH modes so operators can choose the
pattern that suits their workload.

Live-deploy status: DOCUMENTED STUB
------------------------------------
The offline artifact generation (``manifest()``, ``package()``) is real and
produces valid YAML job specs that operators can submit directly.  The LIVE
Nebius provisioning (creating a Token Factory endpoint, submitting a Soperator
job) is documented in ``docs/INTEGRATIONS.md`` but NOT implemented here —
``deploy_live()`` raises ``NotImplementedError`` with a pointer to the runbook.

Job-spec layout
---------------
``package()`` writes the following to *dest*::

    dest/
    ├── serving_spec.yaml      (Token Factory serving spec)
    └── batch_embed_job.yaml   (Soperator batch embed job)

Contract wire format
--------------------
The serving spec and healthcheck use the contract port / path from
``pragmatiq.inference.serve.contract`` (port 8000, ``/v2/health/ready``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from integrations._base import Artifact

# ---------------------------------------------------------------------------
# Serving contract constants (mirrored to avoid a runtime import in offline
# path; values are stable and match pragmatiq.inference.serve.contract)
# ---------------------------------------------------------------------------

_CONTRACT_PORT = 8000
_CONTRACT_HEALTH_PATH = "/v2/health/ready"
_CONTRACT_INFER_PATH = "/v2/models/pragmatiq_embedder/infer"

# Nebius Object Storage S3-compatible endpoint (eu-north1 region default)
_DEFAULT_S3_ENDPOINT = "https://storage.eu-north1.nebius.cloud:443"
_DEFAULT_REGION = "eu-north1"

# Container mount points shared by the serving spec, the batch job spec and
# ``manifest()``.  The batch job passes _SHARD_DIR_MOUNT as the positional
# ``shard_dir`` of ``pragmatiq embed`` and _RUN_DIR_MOUNT as ``--run``; each MUST
# have a matching ``storage:`` entry in the job spec or the CLI cannot open it.
_RUN_DIR_MOUNT = "/opt/pragmatiq/run_dir"
_SHARD_DIR_MOUNT = "/opt/pragmatiq/shard_dir"

# ---------------------------------------------------------------------------
# YAML spec templates
# ---------------------------------------------------------------------------

_SERVING_SPEC_TMPL = """\
# Nebius AI Token Factory — model-serving spec for pragmatiq embedder.
# Submit via the Nebius console or CLI:
#   nebius ai token-factory model create --spec serving_spec.yaml
# See docs/INTEGRATIONS.md for the full runbook.

apiVersion: ai.nebius.com/v1alpha1
kind: TokenFactoryModel
metadata:
  name: {release_name}
  namespace: {namespace}
spec:
  image: "{image}"
  containerPort: {contract_port}
  healthPath: "{health_path}"
  inferPath: "{infer_path}"
  replicas: {replicas}
  resources:
    gpuType: "{gpu_type}"
    gpuCount: {gpu_count}
    memoryGiB: {memory_gib}
  env:
    PRAGMATIQ_RUN: "{run_dir_mount}"
    PRAGMATIQ_SERVE_GPU: "1"
  storage:
    type: s3
    bucket: "{s3_bucket}"
    prefix: "{s3_prefix}"
    endpoint: "{s3_endpoint}"
    region: "{region}"
    mountPath: "{run_dir_mount}"
  livenessProbe:
    httpGet:
      path: "{health_path}"
      port: {contract_port}
    initialDelaySeconds: 30
    periodSeconds: 10
"""

_BATCH_JOB_TMPL = """\
# Nebius Soperator (Slurm-on-Kubernetes) — batch embed job spec.
# Submit via: sbatch batch_embed_job.yaml
# Or via Soperator API:
#   kubectl apply -f batch_embed_job.yaml
# See docs/INTEGRATIONS.md for the full runbook.
#
# CLI usage: pragmatiq embed <shard_dir> --run <run_dir> --out <output.parquet>
# Both directories are Object Storage mounts (see `storage:` below): the trained
# run dir from s3://{s3_bucket}/{s3_prefix} and the tokenized shards from
# s3://{s3_bucket}/{s3_shard_prefix}. The parquet is written back to Object
# Storage by the CLI itself, so the image needs `pragmatiq[s3]` and the AWS_*
# env below must point at the Nebius endpoint.

apiVersion: soperator.nebius.com/v1alpha1
kind: SlurmJob
metadata:
  name: {release_name}-embed
  namespace: {namespace}
spec:
  image: "{image}"
  command:
    - python
    - -m
    - pragmatiq.cli
    - embed
    - "{shard_dir_mount}"
    - --run
    - "{run_dir_mount}"
    - --out
    - "{s3_output_path}"
  resources:
    gpuType: "{gpu_type}"
    gpuCount: {gpu_count}
    memoryGiB: {memory_gib}
  env:
    PRAGMATIQ_RUN: "{run_dir_mount}"
    AWS_ACCESS_KEY_ID: "<NEBIUS_ACCESS_KEY>"
    AWS_SECRET_ACCESS_KEY: "<NEBIUS_SECRET_KEY>"
    AWS_DEFAULT_REGION: "{region}"
    AWS_ENDPOINT_URL_S3: "{s3_endpoint}"
  storage:
    - type: s3
      bucket: "{s3_bucket}"
      prefix: "{s3_prefix}"
      endpoint: "{s3_endpoint}"
      mountPath: "{run_dir_mount}"
    - type: s3
      bucket: "{s3_bucket}"
      prefix: "{s3_shard_prefix}"
      endpoint: "{s3_endpoint}"
      mountPath: "{shard_dir_mount}"
  restartPolicy: Never
"""


class NebiusAdapter:
    """Stub adapter for deploying pragmatiq on Nebius AI Cloud.

    Supports two modes:

    * **Token Factory** (serving): managed inference endpoint.
    * **Soperator batch** (embed): Slurm-on-Kubernetes batch job.

    ``manifest()`` and ``package()`` are fully offline.  ``deploy_live()``
    raises ``NotImplementedError`` — live Nebius provisioning is documented in
    ``docs/INTEGRATIONS.md``.  ``healthcheck()`` builds the request offline
    and requires ``requests`` for the live leg.

    Args:
        image: The container image URI for the Triton / pragmatiq serving image.
        s3_bucket: Nebius Object Storage bucket holding the run directory, the
                   tokenized shards, and the batch embed output.
        s3_prefix: Key prefix (under ``s3_bucket``) of the trained run
                   directory; both specs mount it at ``/opt/pragmatiq/run_dir``.
                   Defaults to ``"pragmatiq/run_dir"``.
        s3_shard_prefix: Key prefix (under ``s3_bucket``) of the tokenized
                   shards; the batch embed job mounts it at
                   ``/opt/pragmatiq/shard_dir``, the positional ``shard_dir``
                   argument of ``pragmatiq embed``.  Defaults to
                   ``"pragmatiq/shards"``.
        s3_endpoint: Nebius S3-compatible endpoint URL.
        region: Nebius region.  Defaults to ``"eu-north1"``.
        namespace: Kubernetes / Soperator namespace.
        replicas: Replica count for the Token Factory spec.
        gpu_type: GPU type for serving and batch specs.
        gpu_count: Number of GPUs per replica / job.
        memory_gib: Memory in GiB per replica / job.
        release_name: Name used in spec metadata.
    """

    #: Short adapter identifier used in logging and artifact kind strings.
    name: str = "nebius"

    def __init__(
        self,
        image: str,
        *,
        s3_bucket: str = "pragmatiq-runs",
        s3_prefix: str = "pragmatiq/run_dir",
        s3_shard_prefix: str = "pragmatiq/shards",
        s3_endpoint: str = _DEFAULT_S3_ENDPOINT,
        region: str = _DEFAULT_REGION,
        namespace: str = "pragmatiq",
        replicas: int = 1,
        gpu_type: str = "H100",
        gpu_count: int = 1,
        memory_gib: int = 32,
        release_name: str = "pragmatiq-embedder",
    ) -> None:
        self._image = image
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix
        self._s3_shard_prefix = s3_shard_prefix
        self._s3_endpoint = s3_endpoint
        self._region = region
        self._namespace = namespace
        self._replicas = replicas
        self._gpu_type = gpu_type
        self._gpu_count = gpu_count
        self._memory_gib = memory_gib
        self._release_name = release_name

    def _s3_output_path(self) -> str:
        """Object Storage key the batch embed job writes its parquet to.

        ``pragmatiq embed --out`` treats the path as a *file* (it is staged out
        with ``is_dir=False``), so the key must name the parquet itself rather
        than a ``.../`` prefix.
        """
        return f"s3://{self._s3_bucket}/embeddings/{self._release_name}.parquet"

    # ------------------------------------------------------------------
    # OFFLINE: manifest()
    # ------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Return a declarative Nebius deploy spec as a plain ``dict``.

        Describes both the Token Factory serving spec and the Soperator batch
        embed job spec.  Fully offline — no Nebius SDK required.

        Returns:
            A dict with:

            * ``"adapter"`` — adapter name ``"nebius"``.
            * ``"token_factory"`` — Token Factory serving configuration.
            * ``"soperator_batch"`` — Soperator batch embed job configuration.
            * ``"storage"`` — Nebius Object Storage configuration: bucket, the
              run-dir and shard prefixes, and their container mount paths.
            * ``"live_ops_status"`` — honest stub declaration.
        """
        return {
            "adapter": self.name,
            "token_factory": {
                "release_name": self._release_name,
                "namespace": self._namespace,
                "image": self._image,
                "replicas": self._replicas,
                "resources": {
                    "gpu_type": self._gpu_type,
                    "gpu_count": self._gpu_count,
                    "memory_gib": self._memory_gib,
                },
                "container": {
                    "port": _CONTRACT_PORT,
                    "health_path": _CONTRACT_HEALTH_PATH,
                    "infer_path": _CONTRACT_INFER_PATH,
                },
            },
            "soperator_batch": {
                "job_name": f"{self._release_name}-embed",
                "namespace": self._namespace,
                "image": self._image,
                "command": [
                    "python", "-m", "pragmatiq.cli", "embed",
                    _SHARD_DIR_MOUNT,
                    "--run", _RUN_DIR_MOUNT,
                    "--out", self._s3_output_path(),
                ],
                "resources": {
                    "gpu_type": self._gpu_type,
                    "gpu_count": self._gpu_count,
                    "memory_gib": self._memory_gib,
                },
            },
            "storage": {
                "s3_bucket": self._s3_bucket,
                "s3_prefix": self._s3_prefix,
                "s3_shard_prefix": self._s3_shard_prefix,
                "s3_endpoint": self._s3_endpoint,
                "region": self._region,
                "run_dir_mount": _RUN_DIR_MOUNT,
                "shard_dir_mount": _SHARD_DIR_MOUNT,
                "note": (
                    "Nebius Object Storage is S3-compatible; use standard AWS SDK "
                    "with the Nebius endpoint and access keys."
                ),
            },
            "live_ops_status": (
                "STUB — live Nebius provisioning is documented, not implemented; "
                "see docs/INTEGRATIONS.md"
            ),
        }

    # ------------------------------------------------------------------
    # OFFLINE: package()
    # ------------------------------------------------------------------

    def package(
        self,
        run_dir: str | Path,
        *,
        dest: str,
        image: str,
    ) -> Artifact:
        """Write Nebius job-spec YAML files to *dest* and return an Artifact.

        Produces::

            dest/
            ├── serving_spec.yaml      (Token Factory serving spec)
            └── batch_embed_job.yaml   (Soperator batch embed job)

        Both specs reference the Triton image URI and the serving contract's
        container port / health path.  The batch job mounts two Object Storage
        prefixes: ``s3_prefix`` (run dir) at ``/opt/pragmatiq/run_dir`` and
        ``s3_shard_prefix`` (tokenized shards) at ``/opt/pragmatiq/shard_dir``,
        the positional argument of ``pragmatiq embed``.  Fully offline — stdlib
        file writes only.

        Args:
            run_dir: Path to the trained run directory (used in artifact details;
                     the spec references it via the S3 mount path).
            dest: Local directory path where the spec files will be written.
            image: Container image URI; used in both YAML specs.

        Returns:
            An :class:`~integrations._base.Artifact` with
            ``kind="nebius-job-spec"`` and ``path_or_uri=dest``.
        """
        run_dir = Path(run_dir)
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        common_kw = dict(
            release_name=self._release_name,
            namespace=self._namespace,
            image=image,
            contract_port=_CONTRACT_PORT,
            health_path=_CONTRACT_HEALTH_PATH,
            infer_path=_CONTRACT_INFER_PATH,
            replicas=self._replicas,
            gpu_type=self._gpu_type,
            gpu_count=self._gpu_count,
            memory_gib=self._memory_gib,
            s3_bucket=self._s3_bucket,
            s3_prefix=self._s3_prefix,
            s3_shard_prefix=self._s3_shard_prefix,
            s3_endpoint=self._s3_endpoint,
            s3_output_path=self._s3_output_path(),
            region=self._region,
            run_dir_mount=_RUN_DIR_MOUNT,
            shard_dir_mount=_SHARD_DIR_MOUNT,
        )

        # 1. Token Factory serving spec
        (dest_path / "serving_spec.yaml").write_text(
            _SERVING_SPEC_TMPL.format(**common_kw),
            encoding="utf-8",
        )

        # 2. Soperator batch embed job
        (dest_path / "batch_embed_job.yaml").write_text(
            _BATCH_JOB_TMPL.format(**common_kw),
            encoding="utf-8",
        )

        return Artifact(
            kind="nebius-job-spec",
            path_or_uri=str(dest_path),
            details={
                "run_dir": str(run_dir),
                "image": image,
                "s3_bucket": self._s3_bucket,
                "s3_prefix": self._s3_prefix,
                "s3_shard_prefix": self._s3_shard_prefix,
                "s3_endpoint": self._s3_endpoint,
                "run_dir_mount": _RUN_DIR_MOUNT,
                "shard_dir_mount": _SHARD_DIR_MOUNT,
                "s3_output_path": self._s3_output_path(),
                "files": ["serving_spec.yaml", "batch_embed_job.yaml"],
            },
        )

    # ------------------------------------------------------------------
    # DOCUMENTED STUB: deploy_live()
    # ------------------------------------------------------------------

    def deploy_live(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Live Nebius provisioning is documented, not implemented.

        The YAML specs produced by :meth:`package` are real and ready to
        submit.  Completing the deployment requires:

        1. Uploading the run directory (and, for batch embed, the tokenized
           shards) to Nebius Object Storage under ``s3_prefix`` /
           ``s3_shard_prefix``.
        2. Creating a Token Factory endpoint or submitting a Soperator job.

        See ``docs/INTEGRATIONS.md`` for the full step-by-step runbook.

        Raises:
            NotImplementedError: Always — live Nebius provisioning is not automated.
        """
        raise NotImplementedError(
            "Nebius live deploy is documented, not implemented; "
            "see docs/INTEGRATIONS.md"
        )

    # ------------------------------------------------------------------
    # LIVE (lazy SDK): healthcheck()
    # ------------------------------------------------------------------

    def healthcheck(self, endpoint: str) -> bool:
        """Hit a Nebius Token Factory endpoint with a KServe v2 envelope.

        Triton's HTTP ``/infer`` endpoint accepts the v2 JSON envelope, not the
        raw records payload, so the request is built offline via
        ``pragmatiq.inference.serve.contract.encode_v2_request`` — the same
        form every Triton-based adapter uses.  The live HTTP call requires
        ``requests``.

        Args:
            endpoint: The full HTTPS URL of the Token Factory endpoint
                      (e.g. ``"https://<endpoint>.inference.eu-north1.nebius.cloud"``).

        Returns:
            ``True`` if the endpoint returned a 2-D embedding matrix with one
            row per healthcheck record.

        Raises:
            MissingExtraError: If the ``requests`` package is not installed.
        """
        from integrations._base import _require

        _require("requests", "requests")
        import requests  # noqa: PLC0415 — intentionally lazy

        from pragmatiq.inference.serve.contract import decode_v2_response, encode_v2_request

        records = [{"user_id": "healthcheck", "events": [], "attributes": {}, "lifelong": []}]
        payload = encode_v2_request(records)

        infer_url = endpoint.rstrip("/") + _CONTRACT_INFER_PATH
        response = requests.post(
            infer_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        emb = decode_v2_response(response.content)
        return emb.ndim == 2 and emb.shape[0] == len(records)
