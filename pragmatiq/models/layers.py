"""Transformer building blocks with varlen (no-padding) attention.

All encoders are **bidirectional, pre-norm**, with a **GELU MLP (ffn = 4d)** and
**dropout 0.1**.

Attention is varlen: tokens are flat and ``cu_seqlens`` give the segment
boundaries. On CUDA with flash-attn installed we use ``flash_attn_varlen_func``;
otherwise we fall back to a SDPA "pad-and-batch" path that gathers each segment
into a padded block and masks the padding (global rule 5). The fp32 SDPA path
matches a naive padded per-segment attention to atol 1e-4 (covered by
``test_varlen_self_attention_matches_padded``); the flash-attn path runs in
fp16/bf16 and matches the SDPA path only to bf16 precision (~1e-2), not 1e-4.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import TimeRoPE

try:  # optional GPU acceleration
    from flash_attn import flash_attn_varlen_func  # type: ignore

    _HAS_FLASH = True
except Exception:  # pragma: no cover - exercised only where flash-attn is absent
    _HAS_FLASH = False


def _segment_lengths(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)


def varlen_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rope: TimeRoPE | None = None,
    rope_pos: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Bidirectional self-attention over varlen segments.

    ``q,k,v`` are ``[T, H, head_dim]`` (flat over all segments). ``cu_seqlens``
    is int32 ``[n_seg + 1]``. Returns ``[T, H, head_dim]``. If ``rope`` is given,
    its rotation is applied to q/k using continuous positions ``rope_pos`` ``[T]``.
    """
    T, H, hd = q.shape
    lengths = _segment_lengths(cu_seqlens)
    if lengths.numel() and int(max_seqlen) < int(lengths.max()):
        raise ValueError(
            f"max_seqlen {int(max_seqlen)} < longest segment {int(lengths.max())}; "
            "pass max_seqlen >= the longest cu_seqlens segment"
        )
    if rope is not None and rope_pos is not None:
        cos, sin = rope.angles(rope_pos)  # [T, hd]
        q = rope.rotate(q.transpose(0, 1), cos, sin).transpose(0, 1)
        k = rope.rotate(k.transpose(0, 1), cos, sin).transpose(0, 1)

    # flash-attn kernels only accept fp16/bf16; fp32 CUDA inference (no
    # autocast) must take the SDPA path below.
    if _HAS_FLASH and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        # In deterministic mode use flash-attn's deterministic backward (the
        # forward is already deterministic); a no-op on the CPU/SDPA path.
        deterministic = os.environ.get("PRAGMATIQ_DETERMINISTIC") == "1"
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens.to(torch.int32), cu_seqlens.to(torch.int32),
            int(max_seqlen), int(max_seqlen), dropout_p=dropout_p, causal=False,
            deterministic=deterministic,
        )
        return out

    # SDPA fallback: gather segments into a padded [n_seg, H, max_len, hd] block.
    n_seg = lengths.numel()
    device = q.device
    # scatter index: position within segment for each flat token
    seg_of = torch.repeat_interleave(torch.arange(n_seg, device=device), lengths)
    pos_in_seg = torch.arange(T, device=device) - cu_seqlens[:-1].to(device)[seg_of]
    qb = q.new_zeros(n_seg, max_seqlen, H, hd)
    kb = torch.zeros_like(qb)
    vb = torch.zeros_like(qb)
    qb[seg_of, pos_in_seg] = q
    kb[seg_of, pos_in_seg] = k
    vb[seg_of, pos_in_seg] = v
    # [n_seg, H, max_len, hd]
    qb = qb.permute(0, 2, 1, 3)
    kb = kb.permute(0, 2, 1, 3)
    vb = vb.permute(0, 2, 1, 3)
    key_pad = torch.arange(max_seqlen, device=device)[None, :] < lengths[:, None]  # [n_seg, max_len]
    attn_mask = key_pad[:, None, None, :]  # broadcast over heads, queries
    out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=attn_mask, dropout_p=dropout_p)
    out = out.permute(0, 2, 1, 3)  # [n_seg, max_len, H, hd]
    return out[seg_of, pos_in_seg]


class VarlenAttention(nn.Module):
    """Multi-head bidirectional attention over varlen segments."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, rope_base: float = 10_000.0,
                 use_rope: bool = False) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.rope = TimeRoPE(self.head_dim, base=rope_base) if use_rope else None

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int,
                rope_pos: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``[T, d]`` flat tokens. Returns ``[T, d]``."""
        T = x.shape[0]
        qkv = self.qkv(x).view(T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        dp = self.dropout if self.training else 0.0
        out = varlen_self_attention(q, k, v, cu_seqlens, max_seqlen, self.rope, rope_pos, dp)
        return self.out(out.reshape(T, self.dim))


class FeedForward(nn.Module):
    """Pre-norm GELU MLP with ffn = 4·d."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm bidirectional transformer block (attn + GELU MLP), varlen."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4,
                 use_rope: bool = False, rope_base: float = 10_000.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VarlenAttention(dim, n_heads, dropout, rope_base, use_rope)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int,
                rope_pos: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), cu_seqlens, max_seqlen, rope_pos))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class Encoder(nn.Module):
    """A stack of ``depth`` :class:`TransformerBlock` over varlen segments."""

    def __init__(self, dim: int, depth: int, n_heads: int, dropout: float = 0.1,
                 use_rope: bool = False, rope_base: float = 10_000.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, dropout, 4, use_rope, rope_base) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int,
                rope_pos: torch.Tensor | None = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, cu_seqlens, max_seqlen, rope_pos)
        return self.norm(x)
