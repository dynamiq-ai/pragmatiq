"""Frozen text-embedding encoders for the Nemotron variant."""

from __future__ import annotations

import torch

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
