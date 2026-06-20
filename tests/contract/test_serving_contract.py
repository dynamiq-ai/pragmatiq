"""Serving contract tests — in-process, no Docker, no Triton, no pb_utils.

This test suite pins the serving wire format (``records_json`` → ``embeddings
[n_users, dim]``) as a first-class contract, exercising both the contract
helpers and the full :class:`~pragmatiq.inference.serve.Runtime` path.

Structure
---------
- **Fixtures**: a tiny nano :class:`~pragmatiq.models.pragmatiq.PragmaModel`
  with an attached tokenizer (mirrors the slim-serve boundary test pattern so
  the two stay in sync).
- **Contract-constant tests**: pin ``INPUT_NAME`` / ``OUTPUT_NAME`` — renaming
  them is a MAJOR contract break caught immediately here.
- **encode/decode round-trip**: ``encode_request`` → ``decode_request`` for all
  three payload surface forms (bytes, str, numpy scalar).
- **encode_response**: shape / dtype / contiguity enforcement.
- **Runtime.embed**: end-to-end path through the serve stack with the nano model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nano_model_and_records():
    """Return ``(model, sample_records)`` using a nano PragmaModel.

    The model is built the same way the slim-serve boundary test does it:
    generate → tokenize → build ModelConfig.preset("small") → attach tokenizer.
    We use "small" here because the boundary test uses it; "nano" would also work.
    """
    from pragmatiq.data.synthetic import WorldConfig, generate
    from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig
    from pragmatiq.models import ModelConfig, PragmaModel

    tmp = Path(tempfile.mkdtemp())
    generate(
        WorldConfig(
            n_users=10,
            months=14,
            n_merchants=30,
            seed=999,  # same seed as slim-serve boundary test (known-good params)
            mule_ring_count=0,
            eval_month_credit=2,
            eval_month_short=8,
        ),
        tmp / "raw",
        n_workers=0,
        write_report=False,
    )
    tok = PragmaTokenizer(
        TokenizerConfig(
            target_vocab=512,
            n_buckets=8,
            categorical_threshold=20,
            seed=0,
        )
    ).fit(tmp / "raw")

    # nano-sized model so it is fast on CPU
    cfg = ModelConfig.preset("small", tok.vocab_size)
    model = PragmaModel(cfg).eval()
    model._tokenizer = tok  # attach tokenizer exactly as from_pretrained does

    records = [
        {
            "user_id": "contract_1",
            "events": [
                {
                    "ts": 1_700_000_000_000_000,
                    "source": "transaction",
                    "fields": {"amount": "12.50", "mcc": "5411", "merchant": "SHOP A"},
                }
            ],
            "attributes": {},
            "lifelong": [],
        },
        {
            "user_id": "contract_2",
            "events": [
                {
                    "ts": 1_700_003_600_000_000,
                    "source": "app",
                    "fields": {"screen": "home", "action": "view"},
                }
            ],
            "attributes": {},
            "lifelong": [],
        },
    ]
    return model, records


# ---------------------------------------------------------------------------
# Contract-constant tests
# ---------------------------------------------------------------------------


def test_input_name_frozen() -> None:
    """INPUT_NAME must be exactly 'records_json' — renaming is a contract break."""
    from pragmatiq.inference.serve.contract import INPUT_NAME

    assert INPUT_NAME == "records_json", (
        f"Contract break: INPUT_NAME changed to {INPUT_NAME!r}. "
        "This is a MAJOR contract break — update STABILITY.md and bump the version."
    )


def test_output_name_frozen() -> None:
    """OUTPUT_NAME must be exactly 'embeddings' — renaming is a contract break."""
    from pragmatiq.inference.serve.contract import OUTPUT_NAME

    assert OUTPUT_NAME == "embeddings", (
        f"Contract break: OUTPUT_NAME changed to {OUTPUT_NAME!r}. "
        "This is a MAJOR contract break — update STABILITY.md and bump the version."
    )


def test_contract_importable_from_package() -> None:
    """All contract symbols are accessible via ``pragmatiq.inference.serve``."""
    import pragmatiq.inference.serve as serve

    for sym in (
        "INPUT_NAME",
        "OUTPUT_NAME",
        "INPUT_DTYPE",
        "OUTPUT_DTYPE",
        "encode_request",
        "decode_request",
        "encode_response",
        "Runtime",
        "load",
        "resolve_serve_device",
    ):
        assert hasattr(serve, sym), f"pragmatiq.inference.serve is missing {sym!r}"


# ---------------------------------------------------------------------------
# encode_request / decode_request round-trip
# ---------------------------------------------------------------------------


def test_encode_request_is_bytes(nano_model_and_records) -> None:
    """encode_request returns bytes."""
    from pragmatiq.inference.serve.contract import encode_request

    _, records = nano_model_and_records
    raw = encode_request(records)
    assert isinstance(raw, bytes)


def test_decode_request_roundtrip_bytes(nano_model_and_records) -> None:
    """encode_request → decode_request(bytes) is identity on the list structure."""
    from pragmatiq.inference.serve.contract import decode_request, encode_request

    _, records = nano_model_and_records
    raw = encode_request(records)
    decoded = decode_request(raw)
    assert decoded == records


def test_decode_request_from_str(nano_model_and_records) -> None:
    """decode_request accepts a plain str (REST adapter path)."""
    import json

    from pragmatiq.inference.serve.contract import decode_request

    _, records = nano_model_and_records
    decoded = decode_request(json.dumps(records))
    assert decoded == records


def test_decode_request_from_numpy_bytes(nano_model_and_records) -> None:
    """decode_request handles a numpy bytes_ scalar (Triton pb_utils path)."""
    import json

    from pragmatiq.inference.serve.contract import decode_request

    _, records = nano_model_and_records
    np_scalar = np.bytes_(json.dumps(records).encode("utf-8"))
    decoded = decode_request(np_scalar)
    assert decoded == records


def test_decode_request_from_numpy_str(nano_model_and_records) -> None:
    """decode_request handles a numpy str_ scalar."""
    import json

    from pragmatiq.inference.serve.contract import decode_request

    _, records = nano_model_and_records
    np_scalar = np.str_(json.dumps(records))
    decoded = decode_request(np_scalar)
    assert decoded == records


def test_decode_request_rejects_non_list() -> None:
    """decode_request raises ValueError when the JSON root is not a list."""
    import json

    from pragmatiq.inference.serve.contract import decode_request

    with pytest.raises(ValueError, match="expected a JSON list"):
        decode_request(json.dumps({"not": "a list"}).encode())


# ---------------------------------------------------------------------------
# encode_response
# ---------------------------------------------------------------------------


def test_encode_response_float32_contiguous() -> None:
    """encode_response produces a C-contiguous float32 array."""
    from pragmatiq.inference.serve.contract import encode_response

    arr = np.random.randn(3, 64).astype(np.float64)
    out = encode_response(arr)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert out.shape == (3, 64)


def test_encode_response_already_float32() -> None:
    """encode_response is a no-copy fast-path for already-float32 contiguous arrays."""
    from pragmatiq.inference.serve.contract import encode_response

    arr = np.zeros((2, 32), dtype=np.float32)
    out = encode_response(arr)
    assert out.dtype == np.float32
    assert out.shape == (2, 32)


def test_encode_response_rejects_1d() -> None:
    """encode_response rejects 1-D arrays (would be ambiguous)."""
    from pragmatiq.inference.serve.contract import encode_response

    with pytest.raises(ValueError, match="2-D"):
        encode_response(np.zeros(64))


# ---------------------------------------------------------------------------
# Runtime.embed — end-to-end contract
# ---------------------------------------------------------------------------


def test_runtime_embed_shape_and_dtype(nano_model_and_records) -> None:
    """Runtime.embed returns float32 [n_users, dim]."""
    from pragmatiq.inference.serve.runtime import Runtime

    model, records = nano_model_and_records
    runtime = Runtime(model=model, device="cpu")
    emb = runtime.embed(records)

    n_users = len(records)
    assert emb.ndim == 2, f"Expected 2-D output, got shape {emb.shape}"
    assert emb.shape[0] == n_users, f"Expected {n_users} rows, got {emb.shape[0]}"
    assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
    assert np.isfinite(emb).all(), "Embedding contains non-finite values"


def test_runtime_embed_contiguous(nano_model_and_records) -> None:
    """Runtime.embed output is C-contiguous (required for Triton tensor copy)."""
    from pragmatiq.inference.serve.runtime import Runtime

    model, records = nano_model_and_records
    runtime = Runtime(model=model, device="cpu")
    emb = runtime.embed(records)
    assert emb.flags["C_CONTIGUOUS"]


def test_runtime_model_property(nano_model_and_records) -> None:
    """Runtime.model returns the underlying PragmaModel."""
    from pragmatiq.inference.serve.runtime import Runtime

    model, records = nano_model_and_records
    runtime = Runtime(model=model, device="cpu")
    assert runtime.model is model
    assert runtime.device == "cpu"
