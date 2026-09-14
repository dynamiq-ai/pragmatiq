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


# ---------------------------------------------------------------------------
# KServe v2 HTTP envelope helpers (additive; used by cloud-adapter healthchecks)
# ---------------------------------------------------------------------------


def encode_v2_request(records: list[dict]) -> bytes:
    """Build the KServe v2 HTTP inference envelope carrying *records*.

    Triton's HTTP endpoint (``POST /v2/models/<name>/infer``) — and SageMaker's
    Triton hosting on ``/invocations`` — do not accept the raw JSON payload from
    :func:`encode_request`; they require the v2 envelope with the payload as a
    single BYTES element.  Send the returned body with
    ``Content-Type: application/json``.

    Args:
        records: List of plain user-record dicts (one per user to embed).

    Returns:
        UTF-8-encoded JSON bytes of the v2 envelope::

            {"inputs": [{"name": "records_json", "datatype": "BYTES",
                         "shape": [1], "data": ["<records JSON>"]}]}
    """
    envelope = {
        "inputs": [
            {
                "name": INPUT_NAME,
                "datatype": INPUT_DTYPE,
                "shape": [1],
                "data": [json.dumps(records)],
            }
        ]
    }
    return json.dumps(envelope).encode("utf-8")


def decode_v2_response(body: bytes | str) -> np.ndarray:
    """Extract the embedding matrix from a KServe v2 HTTP inference response.

    Args:
        body: The raw HTTP response body (JSON bytes or str) containing an
              ``outputs`` list, as returned by Triton's ``/infer`` endpoint.

    Returns:
        ``numpy.ndarray`` of dtype float32, reshaped to the response's declared
        shape (``[n_users, dim]`` under this contract).

    Raises:
        ValueError: If the body is not a v2 envelope or lacks the
                    ``embeddings`` output with ``shape``/``data``.
    """
    text = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body
    payload = json.loads(text)
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("decode_v2_response: response has no 'outputs' list")
    out = next((o for o in outputs if o.get("name") == OUTPUT_NAME), None)
    if out is None:
        names = [o.get("name") for o in outputs]
        raise ValueError(
            f"decode_v2_response: no {OUTPUT_NAME!r} output in response (outputs named {names})"
        )
    if "shape" not in out or "data" not in out:
        raise ValueError(
            f"decode_v2_response: {OUTPUT_NAME!r} output is missing 'shape' or 'data'"
        )
    return np.asarray(out["data"], dtype=np.float32).reshape(out["shape"])


__all__ = [
    "INPUT_NAME",
    "OUTPUT_NAME",
    "INPUT_DTYPE",
    "OUTPUT_DTYPE",
    "OUTPUT_SHAPE_NOTE",
    "encode_request",
    "decode_request",
    "encode_response",
    "encode_v2_request",
    "decode_v2_response",
]
