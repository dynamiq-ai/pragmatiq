"""Azure adapter (stub + Helm chart skeleton) for pragmatiq.

Exposes :class:`AzureAdapter` — a stub adapter that:

* ``manifest()`` — returns a declarative Azure Marketplace / AKS deployment
  spec as a plain dict (offline).
* ``package()`` — templates a minimal Helm chart skeleton to a local directory
  (offline, real file writes).
* ``deploy_live()`` — raises ``NotImplementedError`` with a pointer to
  ``docs/INTEGRATIONS.md``; live AKS deploy is *documented, not implemented*.
* ``healthcheck()`` — builds a contract request offline; the live AKS invoke
  requires an operator to call the AKS endpoint directly (see INTEGRATIONS.md).

Offline discipline
------------------
* ``manifest()`` and ``package()`` work with ZERO cloud SDK installed.
* ``deploy_live()`` is an honest stub that raises ``NotImplementedError``.
* ``healthcheck()`` requires ``requests`` for the live leg only (lazy import).
"""

from integrations.azure._adapter import AzureAdapter

__all__ = ["AzureAdapter"]
