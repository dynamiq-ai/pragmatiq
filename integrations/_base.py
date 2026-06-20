"""Common types for cloud adapters.

Defines the ``CloudAdapter`` Protocol and supporting dataclasses that every
adapter (SageMaker, Databricks, …) implements.  No cloud SDK imports here —
this module must be importable in any environment.

Design
------
Adapters are THIN wrappers around an already-built Triton image plus a run
directory.  They contain NO model or embedding logic; all inference delegation
flows through ``pragmatiq.inference.serve``.

Offline vs live discipline
--------------------------
* **Offline** methods (``manifest``, ``package``, request-building helpers) work
  without any cloud SDK and are fully unit-testable.
* **Live** methods (ECR push, S3 upload, MLflow/UC registration) lazy-import the
  relevant SDK and raise ``MissingExtraError`` with a clear install hint if the
  SDK is absent.  They are NEVER exercised by unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Artifact — returned by every adapter's package() method
# ---------------------------------------------------------------------------


@dataclass
class Artifact:
    """A cloud-native deployable artifact produced by ``CloudAdapter.package()``.

    Attributes:
        kind: Adapter-specific artifact kind, e.g. ``"sagemaker-model-tar"``
              or ``"databricks-pyfunc"``.
        path_or_uri: Local filesystem path or cloud URI (e.g. ``s3://…``,
                     ``dbfs:/…``) pointing to the packaged artifact.
        details: Free-form metadata dict (image used, run_dir, sizes, …).
    """

    kind: str
    path_or_uri: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MissingExtraError — raised by live ops when a cloud SDK is absent
# ---------------------------------------------------------------------------


class MissingExtraError(RuntimeError):
    """Raised when a live cloud operation is attempted without the required SDK.

    Example message::

        boto3 is required for SageMaker live operations.
        Install it with: pip install boto3
    """


def _require(package: str, install_hint: str) -> None:
    """Import *package* and raise ``MissingExtraError`` with *install_hint* if absent.

    Args:
        package: Top-level package name to import (e.g. ``"boto3"``).
        install_hint: The ``pip install …`` command to show in the error.

    Raises:
        MissingExtraError: If *package* is not importable.
    """
    import importlib

    try:
        importlib.import_module(package)
    except ImportError as exc:
        raise MissingExtraError(
            f"{package!r} is required for live cloud operations.\n"
            f"Install it with: pip install {install_hint}"
        ) from exc


# ---------------------------------------------------------------------------
# CloudAdapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CloudAdapter(Protocol):
    """Protocol that every pragmatiq cloud adapter must satisfy.

    Adapters are THIN: they hold configuration (image URI, instance type, …)
    and produce cloud-native artifacts from an already-built Triton image +
    a run directory.  No embedding logic lives inside adapters.

    All concrete adapters must implement:

    * ``name`` — a short identifier used in logging and manifests.
    * ``manifest()`` — a declarative deploy spec as a plain ``dict`` (offline).
    * ``package()`` — assemble a cloud-native deployable artifact (offline).
    * ``healthcheck()`` — hit the serving contract at a live endpoint (live).
    """

    name: str

    def manifest(self) -> dict[str, Any]:
        """Return a declarative deploy spec as a plain ``dict``.

        The dict must be serialisable (JSON / YAML) and must not require any
        cloud SDK to produce.  It describes what callers need to stand up the
        deployed model (image, instance type, environment variables, …).

        Returns:
            A plain ``dict`` describing the deployment configuration.
        """
        ...

    def package(
        self,
        run_dir: str | Any,
        *,
        dest: str,
        image: str,
    ) -> Artifact:
        """Assemble a cloud-native deployable artifact and return its descriptor.

        The offline-buildable portions (staging files, building the tar.gz,
        writing pyfunc manifests) run without any cloud SDK.  Cloud upload is a
        separate ``push()`` / ``register()`` live method on each concrete adapter.

        Args:
            run_dir: Path to the trained run directory
                     (must contain ``checkpoints/`` and ``tokenizer/``).
            dest: Local path or directory where the artifact will be written.
            image: Container image URI for the Triton serving image.

        Returns:
            An :class:`Artifact` describing what was produced.
        """
        ...

    def healthcheck(self, endpoint: str) -> bool:
        """Hit the serving contract at *endpoint* and return ``True`` if healthy.

        This is a LIVE operation — it sends an HTTP / SDK request to a deployed
        endpoint and validates the response shape.  The wire format is built via
        ``pragmatiq.inference.serve.contract.encode_request`` so every adapter
        speaks the same format.

        Args:
            endpoint: The live endpoint URL or ARN to probe.

        Returns:
            ``True`` if the endpoint returned a valid embedding response.
        """
        ...


__all__ = [
    "Artifact",
    "MissingExtraError",
    "CloudAdapter",
    "_require",
]
