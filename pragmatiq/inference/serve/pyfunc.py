"""MLflow pyfunc wrapper for the pragmatiq embedding model.

Defines :class:`PragmaPyfuncWrapper` — the class MLflow pickles into a logged
model.  It lives in the SHIPPED ``pragmatiq`` package (not the repo-only
``integrations`` tree) because Databricks Model Serving unpickles the model on
a cluster where only the ``pragmatiq`` wheel is installed; a wrapper defined
under ``integrations`` would fail to import there.

Import discipline
-----------------
mlflow is NOT imported at module import time.  The offline ``predict`` logic
is a plain method that can be tested without any MLflow context; the
MLflow-ready subclass is created on demand by :func:`mlflow_pyfunc_class`
(used by the live ``register()`` path in ``integrations.databricks``).

Wire format
-----------
``predict(context, model_input)`` accepts the shapes MLflow scoring servers
actually deliver:

* ``list[dict]``       — plain user-record dicts (the interactive path).
* ``bytes`` / ``str``  — JSON-encoded records via ``encode_request`` (the
  serving-contract wire format).
* ``pandas.DataFrame`` — what Databricks Model Serving delivers: either a
  single ``records_json`` column of JSON payloads, or one column per record
  field (the ``dataframe_records`` form).
* ``numpy.ndarray``    — the MLflow tensor-input path (``{"inputs": ...}``);
  elements are JSON strings/bytes or record dicts.

It returns a ``numpy.ndarray`` of shape ``[n_users, dim]`` and dtype
``float32`` — consistent with ``Runtime.embed`` and ``encode_response``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from pragmatiq.inference.serve.contract import INPUT_NAME, decode_request

if TYPE_CHECKING:
    from pragmatiq.inference.serve.runtime import Runtime


def _records_from_sequence(items: list) -> list[dict]:
    """Coerce a flat sequence of dicts / JSON strings / JSON bytes to records.

    A JSON element may encode either a single record dict or a list of
    records; lists are concatenated in order so one cell can carry a whole
    batch (the ``records_json`` column form).
    """
    records: list[dict] = []
    for item in items:
        if isinstance(item, dict):
            records.append(item)
        elif isinstance(item, (bytes, bytearray, str)):
            text = item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else item
            decoded = json.loads(text)
            if isinstance(decoded, list):
                records.extend(decoded)
            elif isinstance(decoded, dict):
                records.append(decoded)
            else:
                raise ValueError(
                    "pyfunc input: JSON element must be a record dict or a list of "
                    f"records, got {type(decoded).__name__!r}"
                )
        else:
            raise ValueError(
                f"pyfunc input: cannot interpret element of type {type(item).__name__!r} "
                "as a user record"
            )
    return records


def _records_from_dataframe(df: Any) -> list[dict]:
    """Extract user records from the pandas DataFrame MLflow scoring delivers.

    Three layouts occur in practice:

    * a ``records_json`` column (the model signature's input name) whose cells
      are JSON payloads — the contract / healthcheck path;
    * a single differently-named column of JSON payloads;
    * one column per record field (``dataframe_records`` with raw dicts).
    """
    columns = list(df.columns)
    if INPUT_NAME in columns:
        return _records_from_sequence(df[INPUT_NAME].tolist())
    if len(columns) == 1:
        return _records_from_sequence(df[columns[0]].tolist())
    return list(df.to_dict(orient="records"))


class PragmaPyfuncWrapper:
    """MLflow-compatible pyfunc wrapper around a ``pragmatiq.inference.serve.Runtime``.

    The class can be used offline (no MLflow installed) by calling
    :meth:`predict` directly.  When packaging for Databricks / Unity Catalog,
    use :func:`mlflow_pyfunc_class` to get an MLflow-ready subclass.

    Args:
        runtime: An initialised :class:`~pragmatiq.inference.serve.runtime.Runtime`.
                 When ``None``, a ``run_dir`` is expected and the runtime is
                 loaded lazily on the first ``predict`` call.
        run_dir: Path to the run directory.  Ignored when *runtime* is provided.
    """

    def __init__(
        self,
        runtime: Runtime | None = None,
        run_dir: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._run_dir = run_dir

    def _get_runtime(self) -> Runtime:
        """Return the runtime, loading it lazily if only ``run_dir`` was provided."""
        if self._runtime is not None:
            return self._runtime
        if self._run_dir is None:
            raise ValueError(
                "PragmaPyfuncWrapper: supply either 'runtime' or 'run_dir'"
            )
        from pragmatiq.inference.serve.runtime import load

        self._runtime = load(self._run_dir)
        return self._runtime

    def predict(
        self,
        context: Any,  # MLflow PythonModel context (ignored offline)
        model_input: Any,
    ) -> np.ndarray:
        """Embed *model_input* and return a float32 ``[n_users, dim]`` array.

        Accepts every input shape an MLflow scoring server can deliver (see
        the module docstring).  This method is intentionally free of MLflow
        dependencies so it can be unit-tested without any MLflow context.

        Args:
            context: MLflow ``PythonModelContext`` (unused — may be ``None``).
            model_input: A ``list[dict]`` of user records, raw JSON
                         bytes/str produced by
                         ``pragmatiq.inference.serve.contract.encode_request``,
                         a pandas ``DataFrame`` (Databricks Model Serving), or
                         a numpy array of JSON payloads / record dicts.

        Returns:
            ``numpy.ndarray`` of dtype ``float32`` and shape ``[n_users, dim]``.
        """
        if isinstance(model_input, (bytes, bytearray, str)):
            # decode_request's contract is bytes | str; normalise bytearray.
            raw = bytes(model_input) if isinstance(model_input, bytearray) else model_input
            records = decode_request(raw)
        elif hasattr(model_input, "columns") and hasattr(model_input, "to_dict"):
            # pandas DataFrame, duck-typed so pandas is never imported here.
            # Must run before the generic sequence branch: list(DataFrame)
            # yields column names, never records.
            records = _records_from_dataframe(model_input)
        elif isinstance(model_input, np.ndarray):
            records = _records_from_sequence(model_input.ravel().tolist())
        else:
            records = _records_from_sequence(list(model_input))

        runtime = self._get_runtime()
        return runtime.embed(records)


def mlflow_pyfunc_class() -> type:
    """Return an MLflow-compatible subclass of ``mlflow.pyfunc.PythonModel``.

    Deferred so this module imports without mlflow installed; the live
    ``register()`` path calls it right before ``mlflow.pyfunc.log_model``.

    Raises:
        ImportError: If mlflow is not installed (install hint included).
    """
    try:
        import mlflow.pyfunc  # noqa: PLC0415 — intentionally lazy
    except ImportError as exc:
        raise ImportError(
            "mlflow is required to build the pyfunc model class.\n"
            "Install it with: pip install 'mlflow[databricks]'"
        ) from exc

    class _MLflowPragmaModel(mlflow.pyfunc.PythonModel):
        """MLflow PythonModel subclass for the pragmatiq embedding model."""

        def load_context(self, context: Any) -> None:
            """Load the runtime from the logged run_dir artifact."""
            run_dir = context.artifacts.get("run_dir")
            self._wrapper = PragmaPyfuncWrapper(run_dir=run_dir)

        def predict(self, context: Any, model_input: Any) -> np.ndarray:
            """Delegate to PragmaPyfuncWrapper.predict."""
            return self._wrapper.predict(context, model_input)

    return _MLflowPragmaModel


__all__ = ["PragmaPyfuncWrapper", "mlflow_pyfunc_class"]
