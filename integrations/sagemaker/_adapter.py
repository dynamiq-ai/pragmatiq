"""SageMaker adapter implementation.

AWS SageMaker can host the NVIDIA Triton Inference Server container natively
using the `SageMaker multi-model server / Triton` flow.  The adapter packages
the run directory into the layout that SageMaker's Triton container expects
when loading from ``model_data`` (an S3 URI pointing to a ``model.tar.gz``).

model.tar.gz layout (SageMaker Triton contract)
-----------------------------------------------
SageMaker's Triton hosting expects the following structure inside the archive
(reference: AWS Triton on SageMaker documentation):

    model.tar.gz
    └── model_repository/
        └── pragmatiq_embedder/
            ├── config.pbtxt           (optional — Triton model config)
            ├── checkpoints/
            │   └── last.pt
            └── tokenizer/
                └── ...

The ``pragmatiq_embedder`` name matches the model name used in
``deploy/triton/Dockerfile`` and ``deploy/triton/model_repository/``.

For a BYOC (Bring Your Own Container) Triton hosting pattern we place the run
artifacts under a sub-directory that the Triton ``PRAGMATIQ_RUN`` env var
points to, allowing the Triton model.py to call ``runtime.load(PRAGMATIQ_RUN)``
exactly as it does in Docker.  This is the simpler and more maintainable
approach: it decouples the packaging layout from the Triton config.pbtxt path.

Final layout chosen (BYOC pattern):

    model.tar.gz
    └── run_dir/
        ├── checkpoints/
        │   └── last.pt
        └── tokenizer/
            └── ...

``PRAGMATIQ_RUN`` in the container environment is set to ``/opt/ml/model/run_dir``
(SageMaker mounts the extracted tar.gz at ``/opt/ml/model``).
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from integrations._base import Artifact, _require

# SageMaker Triton default instance type — cost-effective GPU for inference.
_DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"

# SageMaker mounts model.tar.gz at this path inside the container.
_SM_MODEL_DIR = "/opt/ml/model"

# Sub-directory name inside the tar.gz that holds the run artifacts.
_RUN_SUBDIR = "run_dir"


class SageMakerAdapter:
    """Thin packaging adapter for deploying pragmatiq on AWS SageMaker (Triton).

    The adapter is OFFLINE for ``manifest()`` and ``package()``.  Live
    operations (``push()``, ``healthcheck()``) lazy-import boto3 and raise a
    clear error if it is not installed.

    Args:
        image: The Triton container image URI (e.g. from ECR).  Used in
               ``manifest()`` and stored in the artifact details.
        instance_type: SageMaker instance type for the endpoint config.
                       Defaults to ``"ml.g4dn.xlarge"``.
        model_name: Optional SageMaker model name.  Used only in the manifest.
        endpoint_name: Optional SageMaker endpoint name.  Used only in the manifest.
    """

    #: Short adapter identifier used in logging and artifact kind strings.
    name: str = "sagemaker"

    def __init__(
        self,
        image: str,
        *,
        instance_type: str = _DEFAULT_INSTANCE_TYPE,
        model_name: str = "pragmatiq-embedder",
        endpoint_name: str = "pragmatiq-embedder-endpoint",
    ) -> None:
        self._image = image
        self._instance_type = instance_type
        self._model_name = model_name
        self._endpoint_name = endpoint_name

    # ------------------------------------------------------------------
    # OFFLINE: manifest()
    # ------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Return a declarative SageMaker deploy spec as a plain ``dict``.

        The returned dict describes what callers need to call
        ``boto3.client("sagemaker").create_model(...)`` and
        ``create_endpoint_config(...)``/``create_endpoint(...)``.  It is fully
        offline — no cloud SDK required.

        Returns:
            A dict with two top-level keys:

            * ``"model"`` — SageMaker CreateModel parameters (image, env vars,
              model data placeholder).
            * ``"endpoint_config"`` — SageMaker CreateEndpointConfig parameters
              (instance type, initial instance count, variant name).
        """
        return {
            "model": {
                "model_name": self._model_name,
                "image": self._image,
                # S3 URI placeholder — filled in after push()
                "model_data_url": "<S3_URI>/model.tar.gz",
                "env": {
                    # The Triton model.py reads PRAGMATIQ_RUN at init time.
                    "PRAGMATIQ_RUN": f"{_SM_MODEL_DIR}/{_RUN_SUBDIR}",
                    # Set to '1' to enable GPU inference (CUDA must be available).
                    "PRAGMATIQ_SERVE_GPU": "1",
                    # Triton-specific: enforce single-model mode for simplicity.
                    "SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "pragmatiq_embedder",
                },
            },
            "endpoint_config": {
                "endpoint_config_name": f"{self._model_name}-config",
                "production_variants": [
                    {
                        "variant_name": "AllTraffic",
                        "model_name": self._model_name,
                        "initial_instance_count": 1,
                        "instance_type": self._instance_type,
                        "initial_variant_weight": 1.0,
                    }
                ],
                # Expose instance_type at the top level for easy manifest inspection.
                "instance_type": self._instance_type,
            },
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
        """Build the SageMaker ``model.tar.gz`` locally and return an Artifact.

        Stages the run directory into ``run_dir/`` inside the archive so that
        when SageMaker extracts to ``/opt/ml/model``, the PRAGMATIQ_RUN env var
        (set to ``/opt/ml/model/run_dir``) points to the correct location.

        This method is fully offline — it uses stdlib ``tarfile`` + ``shutil``
        and does NOT upload to S3.  Call ``push()`` to upload after packaging.

        Args:
            run_dir: Path to the trained run directory containing
                     ``checkpoints/`` and ``tokenizer/``.
            dest: Local filesystem path where the ``.tar.gz`` will be written
                  (e.g. ``"/tmp/model.tar.gz"``).
            image: Container image URI; stored in the artifact details.

        Returns:
            An :class:`~integrations._base.Artifact` with
            ``kind="sagemaker-model-tar"`` and ``path_or_uri=dest``.
        """
        run_dir = Path(run_dir)
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Stage run_dir into a temp directory as run_dir/<contents>
        with tempfile.TemporaryDirectory(prefix="pragmatiq-sm-stage-") as staging:
            stage = Path(staging) / _RUN_SUBDIR
            shutil.copytree(run_dir, stage)

            # Build the tar.gz from the staging root
            with tarfile.open(dest_path, "w:gz") as tf:
                tf.add(stage, arcname=_RUN_SUBDIR)

        return Artifact(
            kind="sagemaker-model-tar",
            path_or_uri=str(dest_path),
            details={
                "run_dir": str(run_dir),
                "image": image,
                "instance_type": self._instance_type,
                "inner_path": f"{_SM_MODEL_DIR}/{_RUN_SUBDIR}",
            },
        )

    # ------------------------------------------------------------------
    # LIVE (lazy SDK): push()
    # ------------------------------------------------------------------

    def push(
        self,
        artifact_path: str,
        *,
        role_arn: str,
        s3_bucket: str | None = None,
        s3_prefix: str = "pragmatiq/models",
        region: str = "us-east-1",
    ) -> str:
        """Upload the packaged ``model.tar.gz`` to S3 and return the S3 URI.

        LIVE operation — requires ``boto3``.  Raises :class:`MissingExtraError`
        with a clear install hint if boto3 is not installed.

        Args:
            artifact_path: Local path to the ``model.tar.gz`` produced by
                           :meth:`package`.
            role_arn: IAM role ARN with SageMaker + S3 permissions.
            s3_bucket: S3 bucket name.  Defaults to None (must be supplied).
            s3_prefix: S3 key prefix.  Defaults to ``"pragmatiq/models"``.
            region: AWS region.  Defaults to ``"us-east-1"``.

        Returns:
            The S3 URI of the uploaded archive (``s3://<bucket>/<key>``).

        Raises:
            MissingExtraError: If boto3 is not installed.
        """
        _require("boto3", "boto3")
        import os

        import boto3  # noqa: PLC0415 — intentionally lazy

        s3 = boto3.client("s3", region_name=region)
        key = f"{s3_prefix}/{os.path.basename(artifact_path)}"
        s3.upload_file(artifact_path, s3_bucket, key)
        return f"s3://{s3_bucket}/{key}"

    # ------------------------------------------------------------------
    # LIVE (lazy SDK): healthcheck()
    # ------------------------------------------------------------------

    def healthcheck(self, endpoint: str) -> bool:
        """Hit the SageMaker endpoint with a contract-compliant payload.

        LIVE operation — requires ``boto3``.  Raises :class:`MissingExtraError`
        if boto3 is not installed.  The request payload is built offline via the
        serving contract so every adapter speaks the same wire format.

        Args:
            endpoint: SageMaker endpoint name (not ARN).

        Returns:
            ``True`` if the endpoint returned a 2-D float32 response.

        Raises:
            MissingExtraError: If boto3 is not installed.
        """
        _require("boto3", "boto3")
        import boto3  # noqa: PLC0415 — intentionally lazy
        import numpy as np

        from pragmatiq.inference.serve.contract import (
            encode_request,
        )

        client = boto3.client("sagemaker-runtime")
        records = [{"user_id": "healthcheck", "events": [], "attributes": {}, "lifelong": []}]
        payload = encode_request(records)

        response = client.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/octet-stream",
            Accept="application/octet-stream",
            Body=payload,
        )
        body = response["Body"].read()
        # Expect a flat float32 array back (1 user × dim)
        arr = np.frombuffer(body, dtype=np.float32)
        return arr.ndim >= 1 and arr.size > 0
