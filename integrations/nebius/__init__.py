"""Nebius adapter (stub + job-spec skeleton) for pragmatiq.

Exposes :class:`NebiusAdapter` — a stub adapter that:

* ``manifest()`` — returns a declarative Nebius model-serving / Soperator batch
  job spec as a plain dict (offline).
* ``package()`` — writes YAML job-spec files to a local directory (offline,
  real file writes via stdlib).
* ``deploy_live()`` — raises ``NotImplementedError`` with a pointer to
  ``docs/INTEGRATIONS.md``; live Nebius provisioning is *documented, not
  implemented*.
* ``healthcheck()`` — builds a contract request offline; the live call requires
  ``requests`` (lazy import).

Offline discipline
------------------
* ``manifest()`` and ``package()`` work with ZERO cloud SDK installed.
* ``deploy_live()`` is an honest stub that raises ``NotImplementedError``.
* ``healthcheck()`` requires ``requests`` for the live leg only (lazy import).
"""

from integrations.nebius._adapter import NebiusAdapter

__all__ = ["NebiusAdapter"]
