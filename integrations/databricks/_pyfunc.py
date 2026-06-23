"""MLflow pyfunc wrapper for pragmatiq embedding model.

This module defines ``PragmaPyfuncWrapper`` — an MLflow ``PythonModel``
subclass that delegates to ``pragmatiq.inference.serve.runtime.Runtime``.

Import discipline
-----------------
The class definition must NOT require mlflow at import time.  We achieve this
by inheriting from a plain ``object`` base when mlflow is absent and only
reaching for ``mlflow.pyfunc.PythonModel`` at ``mlflow_pyfunc_class()`` call
time (used by the live ``register()`` path).  The offline ``predict`` logic is
a plain method that can be tested directly without any MLflow context.

Wire format
-----------
``predict(context, model_input)`` accepts either:

* ``list[dict]`` — plain user-record dicts (the interactive path).
* ``bytes``       — JSON-encoded records via ``encode_request`` (the serving
                    contract wire format, as sent by SageMaker / Triton).

It returns a ``numpy.ndarray`` of shape ``[n_users, dim]`` and dtype
``float32`` — consistent with ``Runtime.embed`` and ``encode_response``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pragmatiq.inference.serve.runtime import Runtime


class PragmaPyfuncWrapper:
    """MLflow-compatible pyfunc wrapper around a ``pragmatiq.inference.serve.Runtime``.

    The class can be used offline (no MLflow installed) by calling
    :meth:`predict` directly.  When packaging for Databricks / Unity Catalog,
    use :meth:`mlflow_pyfunc_class` to get an MLflow-ready subclass.

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
        model_input: list[dict] | bytes,
    ) -> np.ndarray:
        """Embed *model_input* and return a float32 ``[n_users, dim]`` array.

        Accepts either plain user-record dicts or JSON-encoded bytes (the
        serving contract wire format).  This method is intentionally free of
        MLflow dependencies so it can be unit-tested without any MLflow context.

        Args:
            context: MLflow ``PythonModelContext`` (unused — may be ``None``).
            model_input: Either a ``list[dict]`` of user records, or raw
                         JSON bytes produced by
                         ``pragmatiq.inference.serve.contract.encode_request``.

        Returns:
            ``numpy.ndarray`` of dtype ``float32`` and shape ``[n_users, dim]``.
        """
        if isinstance(model_input, (bytes, bytearray)):
            from pragmatiq.inference.serve.contract import decode_request

            records: list[dict] = decode_request(model_input)
        else:
            records = list(model_input)

        runtime = self._get_runtime()
        return runtime.embed(records)


def mlflow_pyfunc_class() -> type:
    """Return an MLflow-compatible subclass of ``mlflow.pyfunc.PythonModel``.

    This is a LIVE helper used by the ``register()`` path.  It requires mlflow
    to be installed.

    Raises:
        MissingExtraError: If mlflow is not installed.
    """
    from integrations._base import _require

    _require("mlflow", "mlflow[databricks]")
    import mlflow.pyfunc  # noqa: PLC0415 — intentionally lazy

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
