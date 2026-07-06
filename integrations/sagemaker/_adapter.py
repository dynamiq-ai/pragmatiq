"""SageMaker adapter implementation.

AWS SageMaker can host the NVIDIA Triton Inference Server container natively
using the `SageMaker multi-model server / Triton` flow.  The adapter packages
the run directory into the layout that SageMaker's Triton container expects
when loading from ``model_data`` (an S3 URI pointing to a ``model.tar.gz``).

model.tar.gz layout (SageMaker Triton contract)
-----------------------------------------------
SageMaker extracts ``model.tar.gz`` at ``/opt/ml/model``, and the Triton
container treats that directory as its model repository — each top-level
directory named like ``<model>/<version>/`` is a loadable model
(``SAGEMAKER_TRITON_DEFAULT_MODEL_NAME`` selects which one to serve).  The
archive therefore ships BOTH the Triton python-backend model and the run
artifacts:

    model.tar.gz                    (extracted at /opt/ml/model)
    ├── pragmatiq_embedder/         (Triton model — sourced from
    │   ├── config.pbtxt             deploy/triton/model_repository/,
    │   └── 1/                       the single source of truth)
    │       └── model.py
    └── run_dir/
        ├── checkpoints/
        │   └── last.pt
        └── tokenizer/
            └── ...

The run artifacts stay OUTSIDE the model directory so the layout is
decoupled from Triton versioning rules; ``model.py`` locates them through the
``run_dir`` parameter in ``config.pbtxt``, which ``package()`` rewrites from
the repo default (``/models/run``, the docker-compose mount) to
``/opt/ml/model/run_dir``.  ``manifest()`` sets ``PRAGMATIQ_RUN`` to the same
path — model.py prefers the config parameter, and both must agree.
"""

from __future__ import annotations

import re
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

# Triton model name — must match the directory shipped in the tarball and the
# SAGEMAKER_TRITON_DEFAULT_MODEL_NAME env var in manifest().
_TRITON_MODEL_NAME = "pragmatiq_embedder"


def _repo_triton_model_dir() -> Path:
    """Locate ``deploy/triton/model_repository/pragmatiq_embedder`` in the repo.

    The integrations package is repo-only, so the Triton model sources
    (``config.pbtxt`` + ``1/model.py``) are resolved relative to this file
    instead of being duplicated here — ``deploy/triton`` stays the single
    source of truth.

    Returns:
        Path to the Triton model directory.

    Raises:
        FileNotFoundError: If the repo checkout does not contain the Triton
                           model sources (e.g. running from an installed copy).
    """
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "deploy" / "triton" / "model_repository" / _TRITON_MODEL_NAME
    if not (model_dir / "config.pbtxt").is_file() or not (model_dir / "1" / "model.py").is_file():
        raise FileNotFoundError(
            f"Triton model sources not found at {model_dir} "
            "(expected config.pbtxt and 1/model.py). SageMakerAdapter.package() "
            "requires a full pragmatiq repo checkout."
        )
    return model_dir


def _rewrite_run_dir_param(config_text: str, run_dir_path: str) -> str:
    """Point the config.pbtxt ``run_dir`` parameter at *run_dir_path*.

    The repo config ships with the docker-compose mount path (``/models/run``).
    Inside a SageMaker endpoint the archive lands at ``/opt/ml/model``, and
    model.py prefers the config parameter over the ``PRAGMATIQ_RUN`` env var —
    so an unrewritten parameter would silently override the manifest's env.
    A config with no ``run_dir`` parameter is returned unchanged (the env var
    then wins, which is equally correct).

    Args:
        config_text: The config.pbtxt content.
        run_dir_path: Absolute container path of the staged run artifacts.

    Returns:
        The config text with the ``run_dir`` string_value replaced.
    """
    return re.sub(
        r'(key:\s*"run_dir"\s*value:\s*\{\s*string_value:\s*")[^"]*(")',
        lambda m: m.group(1) + run_dir_path + m.group(2),
        config_text,
    )


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
                    # Same path the rewritten config.pbtxt run_dir parameter
                    # points at — package() keeps the two in lock-step.
                    "PRAGMATIQ_RUN": f"{_SM_MODEL_DIR}/{_RUN_SUBDIR}",
                    # Set to '1' to enable GPU inference (CUDA must be available).
                    "PRAGMATIQ_SERVE_GPU": "1",
                    # Names the Triton model directory shipped in model.tar.gz.
                    "SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": _TRITON_MODEL_NAME,
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

        Stages the full Triton layout (see module docstring): the
        ``pragmatiq_embedder/`` model directory (config.pbtxt + 1/model.py,
        copied from ``deploy/triton/model_repository/``) plus the run
        artifacts under ``run_dir/``.  The staged config.pbtxt's ``run_dir``
        parameter is rewritten to ``/opt/ml/model/run_dir`` so it matches the
        ``PRAGMATIQ_RUN`` env var from :meth:`manifest`.

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

        Raises:
            FileNotFoundError: If the repo's Triton model sources are missing.
        """
        run_dir = Path(run_dir)
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        inner_run_path = f"{_SM_MODEL_DIR}/{_RUN_SUBDIR}"

        with tempfile.TemporaryDirectory(prefix="pragmatiq-sm-stage-") as staging:
            staging_root = Path(staging)

            # 1. Run artifacts → run_dir/<contents>
            shutil.copytree(run_dir, staging_root / _RUN_SUBDIR)

            # 2. Triton model dir from the repo → pragmatiq_embedder/
            triton_stage = staging_root / _TRITON_MODEL_NAME
            shutil.copytree(_repo_triton_model_dir(), triton_stage)
            config_path = triton_stage / "config.pbtxt"
            config_path.write_text(
                _rewrite_run_dir_param(
                    config_path.read_text(encoding="utf-8"), inner_run_path
                ),
                encoding="utf-8",
            )

            # 3. Build the tar.gz from the staging root
            with tarfile.open(dest_path, "w:gz") as tf:
                tf.add(triton_stage, arcname=_TRITON_MODEL_NAME)
                tf.add(staging_root / _RUN_SUBDIR, arcname=_RUN_SUBDIR)

        return Artifact(
            kind="sagemaker-model-tar",
            path_or_uri=str(dest_path),
            details={
                "run_dir": str(run_dir),
                "image": image,
                "instance_type": self._instance_type,
                "inner_path": inner_run_path,
                "triton_model_name": _TRITON_MODEL_NAME,
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
        if not s3_bucket:
            raise ValueError(
                "s3_bucket must be supplied to push(); got None or empty string. "
                "Pass the S3 bucket name as s3_bucket='my-bucket'."
            )
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
        """Hit the SageMaker endpoint with a KServe v2 inference envelope.

        LIVE operation — requires ``boto3``.  Raises :class:`MissingExtraError`
        if boto3 is not installed.  SageMaker's Triton hosting forwards
        ``/invocations`` to the default model's v2 ``/infer`` endpoint, so the
        payload is the v2 JSON envelope built offline via the serving contract
        (``encode_v2_request``) — the same form every Triton-based adapter uses.

        Args:
            endpoint: SageMaker endpoint name (not ARN).

        Returns:
            ``True`` if the endpoint returned a 2-D embedding matrix with one
            row per healthcheck record.

        Raises:
            MissingExtraError: If boto3 is not installed.
        """
        _require("boto3", "boto3")
        import boto3  # noqa: PLC0415 — intentionally lazy

        from pragmatiq.inference.serve.contract import (
            decode_v2_response,
            encode_v2_request,
        )

        client = boto3.client("sagemaker-runtime")
        records = [{"user_id": "healthcheck", "events": [], "attributes": {}, "lifelong": []}]
        payload = encode_v2_request(records)

        response = client.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        emb = decode_v2_response(response["Body"].read())
        return emb.ndim == 2 and emb.shape[0] == len(records)
