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

import json
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
        "request_limits",
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
# KServe v2 HTTP envelope helpers (used by the cloud-adapter healthchecks)
# ---------------------------------------------------------------------------


def test_encode_v2_request_envelope_shape(nano_model_and_records) -> None:
    """encode_v2_request wraps records in the KServe v2 'inputs' envelope."""
    import json

    from pragmatiq.inference.serve.contract import (
        INPUT_DTYPE,
        INPUT_NAME,
        decode_request,
        encode_v2_request,
    )

    _, records = nano_model_and_records
    envelope = json.loads(encode_v2_request(records))
    (inp,) = envelope["inputs"]
    assert inp["name"] == INPUT_NAME
    assert inp["datatype"] == INPUT_DTYPE
    assert inp["shape"] == [1]
    # The BYTES element carries the same JSON payload the model-side decode expects.
    assert decode_request(inp["data"][0]) == records


def test_decode_v2_response_roundtrip() -> None:
    """decode_v2_response recovers the [n_users, dim] float32 matrix."""
    import json

    from pragmatiq.inference.serve.contract import OUTPUT_NAME, decode_v2_response

    emb = np.arange(8, dtype=np.float32).reshape(2, 4)
    body = json.dumps(
        {
            "outputs": [
                {
                    "name": OUTPUT_NAME,
                    "datatype": "FP32",
                    "shape": [2, 4],
                    "data": emb.ravel().tolist(),
                }
            ]
        }
    ).encode("utf-8")

    out = decode_v2_response(body)
    assert out.dtype == np.float32
    assert out.shape == (2, 4)
    np.testing.assert_array_equal(out, emb)


def test_decode_v2_response_accepts_str_body() -> None:
    """decode_v2_response also accepts an already-decoded str body."""
    import json

    from pragmatiq.inference.serve.contract import OUTPUT_NAME, decode_v2_response

    body = json.dumps(
        {"outputs": [{"name": OUTPUT_NAME, "shape": [1, 2], "data": [0.5, 1.5]}]}
    )
    out = decode_v2_response(body)
    assert out.shape == (1, 2)


def test_decode_v2_response_rejects_missing_outputs() -> None:
    """decode_v2_response raises ValueError when the body has no 'outputs' list."""
    import json

    from pragmatiq.inference.serve.contract import decode_v2_response

    with pytest.raises(ValueError, match="outputs"):
        decode_v2_response(json.dumps({"error": "model not found"}).encode())


def test_decode_v2_response_rejects_wrong_output_name() -> None:
    """decode_v2_response raises ValueError when the 'embeddings' output is absent."""
    import json

    from pragmatiq.inference.serve.contract import decode_v2_response

    body = json.dumps(
        {"outputs": [{"name": "something_else", "shape": [1, 2], "data": [0.0, 1.0]}]}
    ).encode()
    with pytest.raises(ValueError, match="embeddings"):
        decode_v2_response(body)


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


# ---------------------------------------------------------------------------
# 1.1.0: GPU-first device policy, request validation and caps
# ---------------------------------------------------------------------------


def test_serve_device_policy_is_gpu_first(monkeypatch) -> None:
    import torch

    from pragmatiq.inference.serve.runtime import resolve_serve_device

    monkeypatch.delenv("PRAGMATIQ_SERVE_CPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_serve_device() == "cuda"
    assert resolve_serve_device(instance_kind="GPU", instance_device_id=1) == "cuda:1"
    assert resolve_serve_device(instance_kind="CPU") == "cuda"  # a visible GPU is used
    monkeypatch.setenv("PRAGMATIQ_SERVE_CPU", "1")
    assert resolve_serve_device(instance_kind="GPU") == "cpu"  # the escape hatch wins
    monkeypatch.delenv("PRAGMATIQ_SERVE_CPU")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_serve_device(instance_kind="GPU") == "cpu"


def test_decode_request_validates_records() -> None:
    from pragmatiq.inference.serve.contract import decode_request

    with pytest.raises(ValueError, match="expected a dict"):
        decode_request(json.dumps([1, 2]))
    with pytest.raises(ValueError, match="user_id"):
        decode_request(json.dumps([{"events": []}]))
    with pytest.raises(ValueError, match="events"):
        decode_request(json.dumps([{"user_id": "u", "events": "nope"}]))
    assert decode_request(json.dumps([{"user_id": "u", "events": []}])) == [{"user_id": "u", "events": []}]


def test_request_limits_from_env(monkeypatch) -> None:
    from pragmatiq.inference.serve.runtime import request_limits

    monkeypatch.delenv("PRAGMATIQ_SERVE_MAX_RECORDS", raising=False)
    monkeypatch.delenv("PRAGMATIQ_SERVE_TOKEN_BUDGET", raising=False)
    assert request_limits() == (1024, 16_384)
    monkeypatch.setenv("PRAGMATIQ_SERVE_MAX_RECORDS", "2")
    monkeypatch.setenv("PRAGMATIQ_SERVE_TOKEN_BUDGET", "512")
    assert request_limits() == (2, 512)
    monkeypatch.setenv("PRAGMATIQ_SERVE_MAX_RECORDS", "0")
    with pytest.raises(ValueError, match="> 0"):
        request_limits()


def test_runtime_rejects_oversized_requests(nano_model_and_records, monkeypatch) -> None:
    from pragmatiq.inference.serve.runtime import Runtime

    model, records = nano_model_and_records
    rt = Runtime(model=model, device="cpu")
    monkeypatch.setenv("PRAGMATIQ_SERVE_MAX_RECORDS", "1")
    with pytest.raises(ValueError, match="at most 1"):
        rt.embed(records)
    monkeypatch.setenv("PRAGMATIQ_SERVE_MAX_RECORDS", "1000")
    monkeypatch.setenv("PRAGMATIQ_SERVE_TOKEN_BUDGET", "1")  # one record per forward
    split = rt.embed(records)
    monkeypatch.delenv("PRAGMATIQ_SERVE_TOKEN_BUDGET")
    whole = rt.embed(records)
    assert split.shape == whole.shape and np.allclose(split, whole, atol=1e-4)


def test_deploy_manifests_are_gpu_first_with_cpu_overlay() -> None:
    root = Path(__file__).resolve().parents[2]
    gpu = (root / "deploy/triton/model_repository/pragmatiq_embedder/config.pbtxt").read_text()
    cpu = (root / "deploy/triton/config.cpu.pbtxt").read_text()
    assert "KIND_GPU" in gpu and "KIND_CPU" in cpu
    for cfg in (gpu, cpu):  # the contract is identical in both
        assert 'name: "pragmatiq_embedder"' in cfg and 'backend: "python"' in cfg
        assert "max_batch_size: 0" in cfg and 'name: "records_json"' in cfg and 'name: "embeddings"' in cfg
    compose = (root / "deploy/docker-compose.yaml").read_text()
    compose_cpu = (root / "deploy/docker-compose.cpu.yaml").read_text()
    assert 'capabilities: ["gpu"]' in compose and "PRAGMATIQ_SERVE_CPU" not in compose
    assert "PRAGMATIQ_SERVE_CPU=1" in compose_cpu and "config.cpu.pbtxt" in compose_cpu
    assert "PRAGMATIQ_SERVE_GPU" not in (root / "scripts/deploy_serving.sh").read_text()
