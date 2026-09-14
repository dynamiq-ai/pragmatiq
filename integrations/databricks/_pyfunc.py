"""Import-compatible alias for the shipped pyfunc wrapper.

The wrapper class lives in :mod:`pragmatiq.inference.serve.pyfunc` — inside
the shipped ``pragmatiq`` wheel — because MLflow pickles the class by
reference and Databricks Model Serving must be able to re-import it on a
cluster where the repo-only ``integrations`` package does not exist.  This
module remains so existing ``integrations.databricks._pyfunc`` imports keep
working.
"""

from pragmatiq.inference.serve.pyfunc import PragmaPyfuncWrapper, mlflow_pyfunc_class

__all__ = ["PragmaPyfuncWrapper", "mlflow_pyfunc_class"]
