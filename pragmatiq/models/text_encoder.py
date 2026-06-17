"""Frozen text-embedding encoders for the Nemotron variant (the internal spec + paper §Nemotron).

In the paper's PRAGMA+Nemotron variant, high-cardinality text field values are not
tokenised with BPE; instead a *frozen* pre-trained text model maps each value's full
string to a single vector, and the MLM objective reconstructs that continuous vector
with MSE (rather than predicting BPE token ids). A ``TextEncoder`` provides that
frozen string→vector map.

Two implementations:

- ``hash`` — a deterministic, dependency-free stand-in (a seeded hash of each string
  to a fixed vector). Not semantic; it exists so the variant is fully exercisable in
  CI and on CPU without downloading a multi-GB model.
- ``nemotron`` — the paper's frozen ``Nemotron`` text embedder via 🤗 ``transformers``
  (the ``[nemotron]`` extra). Mean-pooled last hidden state, no gradients.

Encoders are frozen and deterministic: ``encode(list[str]) -> Tensor[n, dim]``.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Protocol, runtime_checkable

import torch

from ..registry import get_text_encoder, register_text_encoder


@runtime_checkable
class TextEncoder(Protocol):
    """A frozen string→vector encoder. ``dim`` is the output width."""

    dim: int

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Map ``texts`` to a ``[len(texts), dim]`` float tensor (frozen, no grad)."""
        ...


@register_text_encoder("hash")
class HashTextEncoder:
    """Deterministic, dependency-free stand-in: a seeded hash of each string → vector.

    Same string → same vector across processes/runs (so it is reproducible and
    cacheable like a real frozen encoder), but it carries no semantic meaning. Use it
    for tests/CI and as a baseline; use ``nemotron`` for real semantic embeddings.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> torch.Tensor:
        out = torch.empty(len(texts), self.dim, dtype=torch.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha1((t or "").encode("utf-8")).hexdigest()[:15], 16)
            g = torch.Generator().manual_seed(seed)
            out[i] = torch.randn(self.dim, generator=g)
        return out


@register_text_encoder("nemotron")
class NemotronTextEncoder:
    """Frozen Nemotron text embedder (paper's variant); needs the ``[nemotron]`` extra.

    Loads a 🤗 ``transformers`` causal/encoder model once, runs it under ``no_grad`` in
    eval mode, and mean-pools the last hidden state over non-pad tokens. The model is
    never trained; its outputs are the MSE reconstruction targets for text fields.
    """

    def __init__(self, model_name: str = "nvidia/Nemotron-1B-v2",
                 device: str = "cpu", max_length: int = 64) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the Nemotron text encoder needs the 'nemotron' extra: "
                "pip install 'pragmatiq[nemotron]' (installs transformers)."
            ) from e
        self.device = device
        self.max_length = max_length
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.dim = int(self._model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.dim, dtype=torch.float32)
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_length).to(self.device)
        out = self._model(**enc).last_hidden_state  # [n, L, H]
        mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)  # [n, L, 1]
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled.float().cpu()


def build_text_encoder(name: str, **kwargs) -> TextEncoder:
    """Instantiate a registered text encoder by name (e.g. ``"hash"``/``"nemotron"``).

    Only the kwargs a given encoder's constructor accepts are forwarded, so a common
    hint like ``dim`` reaches ``hash`` (where the width is free) but is harmlessly
    dropped for ``nemotron`` (whose width is fixed by the pre-trained model). Callers
    should read the built encoder's ``.dim`` as the authoritative output width.
    """
    cls = get_text_encoder(name)
    sig = inspect.signature(cls)  # constructor signature (excludes self)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return cls(**kwargs)
    return cls(**{k: v for k, v in kwargs.items() if k in sig.parameters})
