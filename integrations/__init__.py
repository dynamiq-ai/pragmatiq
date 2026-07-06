"""Top-level integrations package for pragmatiq.

This package is REPO-ONLY (not shipped in the pragmatiq wheel).  It contains
thin cloud-adapter wrappers that package a pragmatiq Triton serving artifact
for deployment on various cloud platforms.

Available adapters
------------------
* :mod:`integrations.sagemaker` — AWS SageMaker (Triton Inference Server).
* :mod:`integrations.databricks` — Databricks Model Serving (MLflow pyfunc).

Common types
------------
* :class:`integrations._base.Artifact` — artifact descriptor returned by ``package()``.
* :class:`integrations._base.CloudAdapter` — Protocol that all adapters implement.
* :class:`integrations._base.MissingExtraError` — raised by live ops if SDK absent.
"""
