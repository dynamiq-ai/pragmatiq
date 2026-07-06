"""SageMaker adapter for pragmatiq.

Exposes :class:`SageMakerAdapter` — a thin wrapper that packages a pragmatiq
run directory into the SageMaker ``model.tar.gz`` format and produces the
manifest needed to create a SageMaker Model + EndpointConfig.

Offline discipline
------------------
* ``manifest()`` and ``package()`` work with ZERO cloud SDK installed.
* ``push()`` lazy-imports ``boto3`` and raises ``MissingExtraError`` if absent.
* ``healthcheck()`` lazy-imports ``boto3`` and raises ``MissingExtraError`` if absent.
"""

from integrations.sagemaker._adapter import SageMakerAdapter

__all__ = ["SageMakerAdapter"]
