"""Databricks adapter implementation.

Packages a pragmatiq run as an MLflow pyfunc artifact for deployment to
Databricks Model Serving (Unity Catalog).

MLflow pyfunc contract
----------------------
An MLflow pyfunc artifact is a directory with at minimum:

    artifact_dir/
    ├── MLmodel            (yaml: flavors.python_function)
    ├── run_dir/           (staged run artifacts: checkpoints/ + tokenizer/)
    └── requirements.txt   (optional but useful)

We write the ``MLmodel`` manifest manually (offline) and stage the run dir
via ``shutil.copytree``.  The actual ``mlflow.log_model`` / ``register_model``
call is delegated to the ``register()`` live method, which lazy-imports mlflow.

Reference: https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from integrations._base import Artifact, _require


class DatabricksAdapter:
    """Thin packaging adapter for deploying pragmatiq on Databricks (MLflow pyfunc).

    ``manifest()`` and ``package()`` are fully offline.  ``register()`` and
    ``healthcheck()`` are LIVE operations that lazy-import mlflow and raise
    :class:`~integrations._base.MissingExtraError` if it is not installed.

    Args:
        catalog: Unity Catalog catalog name (e.g. ``"main"``).
        schema: Unity Catalog schema name (e.g. ``"pragmatiq"``).
        model_name: Registered model name in Unity Catalog.
        serving_endpoint_name: Optional name for the Model Serving endpoint;
                               used only in the manifest.
    """

    #: Short adapter identifier used in logging and artifact kind strings.
    name: str = "databricks"

    def __init__(
        self,
        catalog: str,
        schema: str,
        model_name: str,
        *,
        serving_endpoint_name: str = "pragmatiq-embedder",
    ) -> None:
        self._catalog = catalog
        self._schema = schema
        self._model_name = model_name
        self._serving_endpoint_name = serving_endpoint_name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _model_uri(self) -> str:
        """Unity Catalog model URI: ``catalog.schema.model_name``."""
        return f"{self._catalog}.{self._schema}.{self._model_name}"

    # ------------------------------------------------------------------
    # OFFLINE: manifest()
    # ------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Return a declarative Databricks deploy spec as a plain ``dict``.

        Describes what callers need to register and serve the model on
        Databricks / Unity Catalog.  Fully offline — no MLflow required.

        Returns:
            A dict with:

            * ``"model_uri"`` — Unity Catalog model URI ``catalog.schema.model``.
            * ``"pyfunc_entry"`` — Python import path for the pyfunc loader.
            * ``"signature"`` — MLflow model signature (inputs / outputs).
            * ``"serving_endpoint_name"`` — suggested endpoint name.
        """
        return {
            "model_uri": self._model_uri,
            # The wrapper lives in the SHIPPED package so serving clusters
            # (which install only the pragmatiq wheel) can unpickle it.
            "pyfunc_entry": "pragmatiq.inference.serve.pyfunc:PragmaPyfuncWrapper",
            "signature": {
                "inputs": (
                    '[{"name": "records_json", "type": "binary"}]'
                ),
                "outputs": (
                    '[{"name": "embeddings", "type": "tensor",'
                    ' "tensor-spec": {"dtype": "float32", "shape": [-1, -1]}}]'
                ),
            },
            "serving_endpoint_name": self._serving_endpoint_name,
            "catalog": self._catalog,
            "schema": self._schema,
        }

    # ------------------------------------------------------------------
    # OFFLINE: package()
    # ------------------------------------------------------------------

    def package(
        self,
        run_dir: str | Path,
        *,
        dest: str,
        image: str,  # noqa: ARG002 — pyfunc doesn't use a container image
    ) -> Artifact:
        """Assemble the MLflow pyfunc artifact directory locally.

        Creates a directory at *dest* with the following structure::

            dest/
            ├── MLmodel          (yaml flavor manifest)
            ├── run_dir/         (staged checkpoints + tokenizer)
            └── requirements.txt

        The actual ``mlflow.log_model()`` / ``mlflow.register_model()`` call
        happens in :meth:`register` (LIVE, lazy SDK).

        Args:
            run_dir: Path to the trained run directory.
            dest: Local directory path where the pyfunc artifact is written.
            image: Unused for the pyfunc adapter (container image is for
                   Triton-based adapters).  Accepted for interface uniformity.

        Returns:
            An :class:`~integrations._base.Artifact` with
            ``kind="databricks-pyfunc"`` and ``path_or_uri=dest``.
        """
        run_dir = Path(run_dir)
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        # 1. Stage the run dir
        run_sub = dest_path / "run_dir"
        shutil.copytree(run_dir, run_sub)

        # 2. Write a minimal MLmodel manifest (offline — no mlflow required)
        mlmodel_content = (
            "artifact_path: model\n"
            "flavors:\n"
            "  python_function:\n"
            "    loader_module: mlflow.pyfunc\n"
            "    python_model: pragmatiq.inference.serve.pyfunc\n"
            "    artifacts:\n"
            "      run_dir: run_dir\n"
            f"model_uuid: pragmatiq-{self._catalog}-{self._schema}-{self._model_name}\n"
            "mlflow_version: '>=2.0.0'\n"
        )
        (dest_path / "MLmodel").write_text(mlmodel_content, encoding="utf-8")

        # 3. Write a minimal requirements.txt — pinned so the serving cluster
        # resolves the same wheel version the artifact was packaged from.
        import pragmatiq

        requirements = f"pragmatiq=={pragmatiq.__version__}\n"
        (dest_path / "requirements.txt").write_text(requirements, encoding="utf-8")

        return Artifact(
            kind="databricks-pyfunc",
            path_or_uri=str(dest_path),
            details={
                "run_dir": str(run_dir),
                "model_uri": self._model_uri,
                "catalog": self._catalog,
                "schema": self._schema,
                "model_name": self._model_name,
            },
        )

    # ------------------------------------------------------------------
    # LIVE (lazy SDK): register()
    # ------------------------------------------------------------------

    def register(
        self,
        artifact_path: str,
        *,
        version_description: str = "pragmatiq embedder",
    ) -> str:
        """Register the pyfunc artifact in Unity Catalog via MLflow.

        LIVE operation — requires ``mlflow[databricks]``.  Raises
        :class:`~integrations._base.MissingExtraError` if mlflow is absent.

        Args:
            artifact_path: Local path to the pyfunc artifact directory produced
                           by :meth:`package`.
            version_description: Human-readable version description.

        Returns:
            The full Unity Catalog model URI including version, e.g.
            ``"models:/main.pragmatiq.embedder/1"``.

        Raises:
            MissingExtraError: If mlflow is not installed.
        """
        _require("mlflow", "mlflow[databricks]")
        import mlflow  # noqa: PLC0415 — intentionally lazy

        import pragmatiq
        from pragmatiq.inference.serve.pyfunc import mlflow_pyfunc_class

        model_class = mlflow_pyfunc_class()
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model_class(),
                artifacts={"run_dir": str(Path(artifact_path) / "run_dir")},
                registered_model_name=self._model_uri,
                # The pickled wrapper class resolves from the pragmatiq wheel
                # (pragmatiq.inference.serve.pyfunc), so the serving cluster
                # must install the exact same version we logged with.
                pip_requirements=[f"pragmatiq=={pragmatiq.__version__}"],
            )
        version = getattr(info, "registered_model_version", None)
        if version is None:
            # Older mlflow returns no version on ModelInfo; ask the registry for
            # the newest version of this model instead of guessing.
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name = '{self._model_uri}'")
            if versions:
                version = max(int(mv.version) for mv in versions)
            else:
                # A just-registered model has at least one version, so an empty
                # result is registry eventual-consistency — the initial version
                # is 1.
                version = 1
        return f"models:/{self._model_uri}/{version}"

    # ------------------------------------------------------------------
    # LIVE (lazy SDK): healthcheck()
    # ------------------------------------------------------------------

    def healthcheck(self, endpoint: str) -> bool:
        """Hit a Databricks Model Serving endpoint with a pyfunc-shaped payload.

        LIVE operation — requires ``requests``.  Databricks Model Serving
        wraps pyfunc inputs in a pandas DataFrame, so the healthcheck sends
        the ``dataframe_records`` JSON form with a single ``records_json``
        column — exactly the DataFrame layout
        :class:`pragmatiq.inference.serve.pyfunc.PragmaPyfuncWrapper` decodes.

        Args:
            endpoint: The full HTTPS URL of the Databricks serving endpoint,
                      e.g. ``"https://<workspace>.azuredatabricks.net/serving-endpoints/<name>/invocations"``.

        Returns:
            ``True`` if the endpoint returned one prediction row per record.

        Raises:
            MissingExtraError: If the ``requests`` package is not installed.
        """
        _require("requests", "requests")
        import requests  # noqa: PLC0415 — intentionally lazy

        from pragmatiq.inference.serve.contract import INPUT_NAME, encode_request

        records = [{"user_id": "healthcheck", "events": [], "attributes": {}, "lifelong": []}]
        body = {
            "dataframe_records": [
                {INPUT_NAME: encode_request(records).decode("utf-8")}
            ]
        }

        response = requests.post(endpoint, json=body, timeout=30)
        response.raise_for_status()
        predictions = response.json().get("predictions")
        return isinstance(predictions, list) and len(predictions) == len(records)
