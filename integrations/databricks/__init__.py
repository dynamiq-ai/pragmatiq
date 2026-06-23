"""Databricks adapter for pragmatiq.

Exposes :class:`DatabricksAdapter` — a thin wrapper that packages a pragmatiq
run directory as an MLflow pyfunc artifact and produces the manifest needed to
register in Databricks Unity Catalog.

Offline discipline
------------------
* ``manifest()`` and ``package()`` work with ZERO cloud SDK installed.
* ``register()`` lazy-imports ``mlflow`` and raises ``MissingExtraError`` if absent.
* ``healthcheck()`` lazy-imports ``requests`` and raises ``MissingExtraError`` if absent.
"""

from integrations.databricks._adapter import DatabricksAdapter

__all__ = ["DatabricksAdapter"]
