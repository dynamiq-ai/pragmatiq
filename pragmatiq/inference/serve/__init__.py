"""Serving HTTP API — runtime loader, contract, and convenience re-exports.

Importing this package pulls in the contract constants and helpers (json+numpy
only) plus the runtime loader (torch + PragmaModel).  Both modules are safe
under the slim ``[serve]`` install (no lightning / torch_geometric /
transformers / pb_utils).

Quick start
-----------
::

    from pragmatiq.inference.serve import load, encode_request, decode_request

    runtime = load("runs/my_run")          # local or s3:// run directory
    records = [{"user_id": "u1", "events": [...], ...}]
    emb = runtime.embed(records)           # np.ndarray [n_users, dim] float32

    # Wire helpers (used by adapters)
    raw_bytes = encode_request(records)
    records_back = decode_request(raw_bytes)
"""

from __future__ import annotations

from pragmatiq.inference.serve.contract import (
    INPUT_DTYPE,
    INPUT_NAME,
    OUTPUT_DTYPE,
    OUTPUT_NAME,
    OUTPUT_SHAPE_NOTE,
    decode_request,
    encode_request,
    encode_response,
)
from pragmatiq.inference.serve.runtime import (
    Runtime,
    load,
    resolve_serve_device,
)

__all__ = [
    # Runtime
    "Runtime",
    "load",
    "resolve_serve_device",
    # Contract constants
    "INPUT_NAME",
    "OUTPUT_NAME",
    "INPUT_DTYPE",
    "OUTPUT_DTYPE",
    "OUTPUT_SHAPE_NOTE",
    # Contract helpers
    "encode_request",
    "decode_request",
    "encode_response",
]
