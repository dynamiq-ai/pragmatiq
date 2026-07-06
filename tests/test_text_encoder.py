"""Frozen text-embedding encoders for the Nemotron variant."""

from __future__ import annotations

import torch
import torch.nn as nn

from pragmatiq.models.pragmatiq import ModelConfig, PragmaModel
from pragmatiq.models.text_encoder import HashTextEncoder, build_text_encoder
from pragmatiq.registry import get_text_encoder


def test_hash_encoder_deterministic_and_shaped() -> None:
    enc = HashTextEncoder(dim=32)
    a = enc.encode(["TESCO STORES 4421", "PAYPAL *SPOTIFY", ""])
    b = enc.encode(["TESCO STORES 4421", "PAYPAL *SPOTIFY", ""])
    assert a.shape == (3, 32)
    assert torch.equal(a, b)  # frozen + deterministic (cacheable like a real encoder)
    assert not torch.allclose(a[0], a[1])  # distinct strings → distinct vectors


def test_registry_resolves_and_builds() -> None:
    assert get_text_encoder("hash") is HashTextEncoder
    enc = build_text_encoder("hash", dim=16)
    assert enc.dim == 16
    assert enc.encode(["x"]).shape == (1, 16)


def test_empty_encode_is_well_shaped() -> None:
    assert HashTextEncoder(8).encode([]).shape == (0, 8)


def test_frozen_text_encoder_absent_from_state_dict() -> None:
    """The frozen text encoder must NOT appear in model.state_dict().

    It is stored as a plain Python attribute (not nn.Module), so PyTorch's
    parameter tracking never sees it and it is excluded from checkpoints.
    If a future refactor accidentally makes it an nn.Module submodule, this
    test will catch it before the checkpoint bloats with frozen weights.

    The trainable projection layer (text_proj) IS expected in state_dict —
    the assertion only excludes keys that belong to the frozen encoder itself.
    """
    cfg = ModelConfig(vocab_size=200, text_encoder="hash", text_encoder_dim=32)
    model = PragmaModel(cfg)

    # The frozen encoder must not be an nn.Module (that would put it in state_dict).
    assert not isinstance(model.text_encoder, nn.Module), (
        "model.text_encoder became an nn.Module — its weights will appear in "
        "state_dict() and be saved/loaded with the checkpoint (unintended)."
    )

    # No key in the checkpoint should belong to the frozen encoder.
    # The trainable text_proj / text_out layers are allowed; only the bare
    # encoder object (named "text_encoder.*") is forbidden.
    bad_keys = [k for k in model.state_dict() if k.startswith("text_encoder.")]
    assert not bad_keys, (
        f"Frozen text-encoder weights leaked into state_dict: {bad_keys}. "
        "Ensure the encoder is stored as a plain attribute, not via nn.Module."
    )
