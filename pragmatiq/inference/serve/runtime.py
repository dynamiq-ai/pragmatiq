"""Runtime loader: device resolution + model lifecycle for serving.

This module owns the *serving-specific* logic that used to live inside the
Triton ``model.py``:

1. :func:`resolve_serve_device` — CPU-first device policy (mirrors the old
   Triton ``initialize()`` logic verbatim).
2. :class:`Runtime` / :func:`load` — stage a run directory (remote or local),
   load ``PragmaModel.from_pretrained``, and expose :meth:`Runtime.embed`.

Dependency constraints
~~~~~~~~~~~~~~~~~~~~~~
This module MUST import cleanly under the slim ``[serve]`` install:

- **Allowed**: ``torch``, ``numpy``, ``pragmatiq.models``, ``pragmatiq.storage``,
  ``pragmatiq.inference.serve.contract``.
- **Not allowed**: ``lightning``, ``torch_geometric``, ``transformers``,
  ``lightgbm``, ``pb_utils``, or any other heavy extra.

``pb_utils`` stays exclusively in ``deploy/triton/…/model.py``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pragmatiq.inference.serve.contract import encode_response

if TYPE_CHECKING:
    from pragmatiq.models.pragmatiq import PragmaModel


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_serve_device(
    instance_kind: str | None = None,
    instance_device_id: str | int | None = None,
) -> str:
    """Return the serving device string using the CPU-first policy.

    The decision tree matches the original Triton ``initialize()`` exactly:

    1. If the Triton instance kind is ``"GPU"`` **and** CUDA is available →
       ``cuda:<instance_device_id>`` (pin to the assigned GPU).
    2. Else if ``PRAGMATIQ_SERVE_GPU=1`` **and** CUDA is available →
       ``"cuda"`` (the deploy script sets this for an all-GPU pod).
    3. Else → ``"cpu"`` (global rule 5: CPU is always the safe baseline).

    Args:
        instance_kind: Triton ``model_instance_kind`` (``"GPU"`` or ``"CPU"``).
                       Pass ``None`` when not running under Triton.
        instance_device_id: Triton ``model_instance_device_id``.  Used only when
                            *instance_kind* is ``"GPU"``.  Defaults to ``"0"``.

    Returns:
        A ``torch``-compatible device string: ``"cpu"``, ``"cuda"``, or
        ``"cuda:<N>"``.
    """
    import torch  # lazy: keep top-level import-time lean

    serve_gpu = os.environ.get("PRAGMATIQ_SERVE_GPU", "") == "1"
    if instance_kind == "GPU" and torch.cuda.is_available():
        device_id = instance_device_id if instance_device_id is not None else "0"
        return f"cuda:{device_id}"
    if serve_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class Runtime:
    """A loaded pragmatiq serving runtime: holds a model and embeds records.

    Obtain an instance via :func:`load` rather than constructing directly.
    """

    def __init__(self, model: PragmaModel, device: str) -> None:
        self._model = model
        self._device = device
        self._staging_dir: str | None = None

    @property
    def model(self) -> PragmaModel:
        """The loaded :class:`~pragmatiq.models.pragmatiq.PragmaModel`."""
        return self._model

    @property
    def device(self) -> str:
        """The device string on which the model is running."""
        return self._device

    def close(self) -> None:
        """Release resources held by this Runtime.

        For remote runs, the staging temp-dir is deleted.  Safe to call more
        than once (subsequent calls are no-ops).
        """
        if self._staging_dir is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
            self._staging_dir = None

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def embed(self, records: list[dict]) -> np.ndarray:
        """Embed *records* and return a contiguous float32 ``[n_users, dim]`` array.

        Args:
            records: List of plain user-record dicts.  Each dict must carry at
                     minimum ``user_id`` and ``events``.

        Returns:
            ``np.ndarray`` of dtype ``float32`` and shape ``[n_users, dim]``.
        """
        raw = self._model.embed_records(records)
        return encode_response(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load(
    run_dir: str | Path,
    *,
    device: str | None = None,
    instance_kind: str | None = None,
    instance_device_id: str | int | None = None,
) -> Runtime:
    """Load a pragmatiq run as a serving :class:`Runtime`.

    Remote ``run_dir`` values (``s3://…``, ``gs://…``, ``az://…``) are
    materialised to a local temporary directory via
    :func:`pragmatiq.storage.materialize_dir` before calling
    ``PragmaModel.from_pretrained`` (BYOC serving).  Local paths are used as-is.

    Args:
        run_dir: Path or URL to the trained run directory (must contain
                 ``checkpoints/`` and ``tokenizer/``).
        device: Override the serving device.  When ``None``, the device is
                resolved by :func:`resolve_serve_device`.
        instance_kind: Forwarded to :func:`resolve_serve_device` (Triton kind).
        instance_device_id: Forwarded to :func:`resolve_serve_device` (GPU id).

    Returns:
        An initialised :class:`Runtime` ready to embed records.
    """
    from pragmatiq.models.pragmatiq import PragmaModel
    from pragmatiq.storage import materialize_dir
    from pragmatiq.storage.fs import is_local

    resolved_device = device if device is not None else resolve_serve_device(
        instance_kind=instance_kind,
        instance_device_id=instance_device_id,
    )

    run_str = str(run_dir)

    if is_local(run_str):
        # Local path: use directly (strip file:// prefix if present)
        local_run = run_str[len("file://"):] if run_str.startswith("file://") else run_str
    else:
        # Remote: stage to a local temp dir before loading
        tmp = tempfile.mkdtemp(prefix="pragmatiq-serve-")
        materialize_dir(run_str, tmp)
        local_run = tmp

    model = PragmaModel.from_pretrained(local_run, device=resolved_device)
    runtime = Runtime(model=model, device=resolved_device)
    if not is_local(run_str):
        runtime._staging_dir = tmp  # type: ignore[assignment]  # tmp set for remote path
    return runtime


__all__ = [
    "resolve_serve_device",
    "Runtime",
    "load",
]
