"""Serving contract: the single source of truth for request/response shapes.

The serving pipeline moves user records over the wire as JSON bytes and returns
embeddings as a flat float32 tensor.  Every adapter (Triton, REST, gRPC, cloud)
MUST use these constants and helpers so the wire format is defined exactly once.

Wire format
-----------
- Input tensor name  : ``records_json`` (BYTES / JSON)
- Input payload      : JSON-encoded ``list[dict]`` — one dict per user record.
- Output tensor name : ``embeddings``   (FP32)
- Output shape       : ``[n_users, dim]``, contiguous float32.

Dependencies: json + numpy only.  No torch, no pb_utils, no heavy packages.
This lets the module import under the slim ``[serve]`` install and inside all
cloud adapters.
"""

from __future__ import annotations

import json

import numpy as np

# ---------------------------------------------------------------------------
# Contract constants (frozen — renaming is a MAJOR contract break)
# ---------------------------------------------------------------------------

INPUT_NAME: str = "records_json"
"""Name of the input tensor carrying the JSON-encoded user records."""

OUTPUT_NAME: str = "embeddings"
"""Name of the output tensor carrying the float32 embedding matrix."""

INPUT_DTYPE: str = "BYTES"
"""Wire dtype for the input (JSON bytes, Triton / gRPC BYTES type)."""

OUTPUT_DTYPE: str = "FP32"
"""Wire dtype for the output (float32)."""

OUTPUT_SHAPE_NOTE: str = "[n_users, dim]"
"""Human-readable output shape note; actual ``dim`` comes from the model config."""

# ---------------------------------------------------------------------------
# Encode / decode helpers
# ---------------------------------------------------------------------------


def encode_request(records: list[dict]) -> bytes:
    """JSON-encode *records* to the wire bytes for the ``records_json`` input.

    Args:
        records: List of plain user-record dicts (one per user to embed).
                 Each dict must have at minimum ``user_id`` and ``events``;
                 ``attributes`` and ``lifelong`` default to empty when absent.

    Returns:
        UTF-8-encoded JSON bytes ready to be placed into the input tensor.
    """
    return json.dumps(records).encode("utf-8")


def decode_request(raw: bytes | str | np.generic) -> list[dict]:
    """Decode the ``records_json`` input payload robustly.

    Handles the three surface forms that arrive in practice:
    - ``bytes``  (numpy ``bytes_`` scalar from pb_utils / raw HTTP body).
    - ``str``    (already decoded, e.g. from a REST adapter).
    - numpy scalar (``np.bytes_`` / ``np.str_``; the Triton path produces this).

    Args:
        raw: The raw payload — bytes, str, or a numpy scalar.

    Returns:
        List of plain user-record dicts.

    Raises:
        ValueError: If *raw* cannot be decoded as a JSON list.
    """
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    elif isinstance(raw, str):
        text = raw
    else:
        # numpy scalar: np.bytes_ has .decode(), np.str_ converts via str()
        text = raw.decode("utf-8") if hasattr(raw, "decode") else str(raw)
    records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError(
            f"decode_request: expected a JSON list of dicts, got {type(records).__name__!r}"
        )
    return records


def encode_response(emb: np.ndarray) -> np.ndarray:
    """Ensure *emb* is a contiguous float32 array of shape ``[n_users, dim]``.

    This is the canonical last step before writing to the output tensor.

    Args:
        emb: Embedding array from ``PragmaModel.embed_records`` or equivalent.
             Must be 2-D; shape ``[n_users, dim]``.

    Returns:
        C-contiguous ``float32`` view / copy of *emb*.

    Raises:
        ValueError: If *emb* is not 2-D.
    """
    if emb.ndim != 2:
        raise ValueError(
            f"encode_response: expected a 2-D array [n_users, dim], got shape {emb.shape}"
        )
    out = np.ascontiguousarray(emb, dtype=np.float32)
    return out


__all__ = [
    "INPUT_NAME",
    "OUTPUT_NAME",
    "INPUT_DTYPE",
    "OUTPUT_DTYPE",
    "OUTPUT_SHAPE_NOTE",
    "encode_request",
    "decode_request",
    "encode_response",
]
