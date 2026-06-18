"""Input embeddings.

- :class:`TokenEmbedding` — one shared table ``E`` for keys and values; a token
  vector is ``E(key) + E(value) + sinusoidal(within-field position)``.
- :class:`TimeRoPE` — rotary positional embedding whose continuous position is
  the log-seconds time feature (``8·ln(1+Δt/8)``), applied to q/k.
- :class:`CalendarEmbedding` — sin/cos of (hour, day-of-week, day-of-month)
  through a 2-layer MLP to ``d``.

The embedding table is shared with the MLM output projection (tied weights,
``MLMHead``).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_table(max_len: int, dim: int) -> torch.Tensor:
    """Standard fixed sin/cos positional table of shape ``[max_len, dim]``."""
    pe = torch.zeros(max_len, dim)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TokenEmbedding(nn.Module):
    """Shared key/value embedding + within-field sinusoidal position.

    Args:
        vocab_size: total vocabulary (keys + values + specials share one space).
        dim: model width ``d``.
        max_position: largest within-field position (BPE pieces) to encode.
        dropout: embedding dropout.
    """

    pos_table: torch.Tensor

    def __init__(self, vocab_size: int, dim: int, max_position: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.register_buffer("pos_table", sinusoidal_table(max_position, dim), persistent=False)
        self.max_position = max_position
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, key_ids: torch.Tensor, value_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Embed flat token arrays → ``[T, d]``."""
        pos = positions.clamp(max=self.max_position - 1)
        x = self.embed(key_ids) + self.embed(value_ids) + self.pos_table[pos]
        return self.drop(x)

    @property
    def weight(self) -> torch.Tensor:
        """The shared table (tied to the MLM logit projection)."""
        return self.embed.weight


class TimeRoPE(nn.Module):
    """Rotary embedding over a continuous (log-seconds) position.

    Unlike integer RoPE, the position is a real number (the time feature), so a
    token at log-seconds ``p`` rotates pair ``i`` by ``p · inv_freq[i]``. The
    ``base`` of the geometric frequency ladder is a tunable GUESS.
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"TimeRoPE needs an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def angles(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) of shape ``[L, head_dim]`` for continuous positions ``[L]``."""
        freqs = position[:, None].float() * self.inv_freq[None, :]  # [L, hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [L, hd]
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` of shape ``[H, L, head_dim]`` given cos/sin ``[L, head_dim]``."""
        cos = cos[None].to(x.dtype)
        sin = sin[None].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


class CalendarEmbedding(nn.Module):
    """sin/cos of (hour, day-of-week, day-of-month) → 2-layer MLP → ``d``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, hour: torch.Tensor, dow: torch.Tensor, dom: torch.Tensor) -> torch.Tensor:
        """Map calendar ints → ``[E, d]``."""
        two_pi = 2.0 * math.pi
        feats = torch.stack(
            [
                torch.sin(two_pi * hour.float() / 24.0), torch.cos(two_pi * hour.float() / 24.0),
                torch.sin(two_pi * dow.float() / 7.0), torch.cos(two_pi * dow.float() / 7.0),
                torch.sin(two_pi * dom.float() / 31.0), torch.cos(two_pi * dom.float() / 31.0),
            ],
            dim=-1,
        )
        return self.mlp(feats)
