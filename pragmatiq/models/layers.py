"""Transformer building blocks with varlen (no-padding) attention.

All encoders are **bidirectional, pre-norm**, with a **GELU MLP (ffn = 4d)** and
**dropout 0.1**.

Attention is varlen: tokens are flat and ``cu_seqlens`` give the segment
boundaries. On CUDA under bf16/fp16 (training and, by default, inference run
under autocast) the packed stream goes straight to ``flash_attn_varlen_func``
when flash-attn is installed. Everywhere else — CPU, fp32, or
``PRAGMATIQ_DISABLE_FLASH=1`` — the SDPA fallback gathers each segment into a
padded block and masks the padding, using ``index_copy_``/``index_select`` so
the scatter has a deterministic CUDA implementation under
``torch.use_deterministic_algorithms``. The fp32 SDPA path matches a naive padded
per-segment attention to atol 1e-4 (``test_varlen_self_attention_matches_padded``);
the flash path matches it to bf16 precision (~1e-2).

Every encoder builds one :class:`VarlenLayout` per forward — segment lengths,
the RoPE angles, and (lazily) the padded-block indices — and threads it through
all of its blocks, so the per-block cost is the attention itself, not index and
mask reconstruction.
"""

from __future__ import annotations

import inspect
import os
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import TimeRoPE

try:  # optional GPU acceleration
    from flash_attn import flash_attn_varlen_func  # type: ignore

    _HAS_FLASH = True
    _FLASH_HAS_DETERMINISTIC = "deterministic" in inspect.signature(flash_attn_varlen_func).parameters
except Exception:  # pragma: no cover - exercised only where flash-attn is absent
    _HAS_FLASH = False
    _FLASH_HAS_DETERMINISTIC = False


def flash_available() -> bool:
    """True when flash-attn is importable and not disabled via ``PRAGMATIQ_DISABLE_FLASH=1``."""
    return _HAS_FLASH and os.environ.get("PRAGMATIQ_DISABLE_FLASH") != "1"


def attention_backend(device: torch.device | str, dtype: torch.dtype) -> str:
    """Name of the kernel varlen attention takes for tensors of this device/dtype.

    ``"flash"`` needs CUDA, fp16/bf16 (autocast at inference, bf16-mixed in
    training) and an importable flash-attn; everything else is ``"sdpa"``.
    """
    if flash_available() and torch.device(device).type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return "flash"
    return "sdpa"


def _segment_lengths(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)


@dataclass
class VarlenLayout:
    """Per-forward attention plan shared by every block of one encoder.

    ``seg_of``/``flat_idx``/``key_pad`` describe the padded ``[n_seg, max_seqlen]``
    block the SDPA fallback scatters into; they are built lazily on the first
    SDPA call so a flash forward never pays for them.
    """

    cu_seqlens: torch.Tensor  # int32 [n_seg + 1]
    lengths: torch.Tensor  # int64 [n_seg]
    max_seqlen: int
    n_tokens: int
    cos: torch.Tensor | None = None  # RoPE angles [T, head_dim]
    sin: torch.Tensor | None = None
    # Segments grouped into length buckets for the SDPA fallback (built lazily):
    # (token_idx, flat_idx, key_pad) per bucket — see ``padded_buckets``.
    buckets: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None

    def padded_buckets(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Length-bucketed padding plan for the SDPA fallback, computed once per forward.

        Segments are grouped by ``ceil(log2(len))`` so each bucket pads only to its
        own longest segment: one 6,500-event history no longer forces every other
        segment in the batch to a 6,500-wide score matrix (that is what made the
        padded path O(n_seg × max_len²) on heavy books). Each bucket is
        ``(token_idx, flat_idx, key_pad)``: the flat token rows that belong to the
        bucket, their destination rows in the bucket's ``[n_seg_b * L_b]`` block,
        and the ``[n_seg_b, L_b]`` key mask. A batch whose segments all fall in one
        bucket is the plain padded block.
        """
        if self.buckets is None:
            device = self.lengths.device
            n_seg = self.lengths.numel()
            seg_of = torch.repeat_interleave(torch.arange(n_seg, device=device), self.lengths)
            pos_in_seg = torch.arange(self.n_tokens, device=device) - self.cu_seqlens[:-1].long()[seg_of]
            # Bucket id: ceil(log2(len)) (lengths 1..2 → 1, 3..4 → 2, 5..8 → 3, ...).
            bucket_of_seg = torch.ceil(torch.log2(self.lengths.clamp(min=1).double())).long()
            buckets: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for b in torch.unique(bucket_of_seg).tolist():
                segs = torch.nonzero(bucket_of_seg == b).flatten()
                lens_b = self.lengths[segs]
                L_b = int(lens_b.max())
                # Dense row within the bucket for each of its segments.
                row_of_seg = torch.full((n_seg,), -1, dtype=torch.long, device=device)
                row_of_seg[segs] = torch.arange(segs.numel(), device=device)
                tok_mask = row_of_seg[seg_of] >= 0
                token_idx = torch.nonzero(tok_mask).flatten()
                flat_idx = row_of_seg[seg_of[token_idx]] * L_b + pos_in_seg[token_idx]
                key_pad = torch.arange(L_b, device=device)[None, :] < lens_b[:, None]
                buckets.append((token_idx, flat_idx, key_pad))
            self.buckets = buckets
        return self.buckets


def build_layout(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    n_tokens: int,
    rope: TimeRoPE | None = None,
    rope_pos: torch.Tensor | None = None,
    *,
    check: bool = True,
) -> VarlenLayout:
    """Build the :class:`VarlenLayout` for one encoder forward.

    ``check`` validates ``max_seqlen`` against the longest segment (one
    device→host sync); pass ``False`` when the caller already knows the exact
    value (the collator computes it on the host).
    """
    lengths = _segment_lengths(cu_seqlens)
    if check and lengths.numel():
        longest = int(lengths.max())
        if int(max_seqlen) < longest:
            raise ValueError(
                f"max_seqlen {int(max_seqlen)} < longest segment {longest}; "
                "pass max_seqlen >= the longest cu_seqlens segment"
            )
    layout = VarlenLayout(cu_seqlens=cu_seqlens.to(torch.int32), lengths=lengths,
                          max_seqlen=int(max_seqlen), n_tokens=int(n_tokens))
    if rope is not None and rope_pos is not None:
        layout.cos, layout.sin = rope.angles(rope_pos)  # [T, hd]
    return layout


def varlen_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rope: TimeRoPE | None = None,
    rope_pos: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    layout: VarlenLayout | None = None,
) -> torch.Tensor:
    """Bidirectional self-attention over varlen segments.

    ``q,k,v`` are ``[T, H, head_dim]`` (flat over all segments). ``cu_seqlens``
    is int32 ``[n_seg + 1]``. Returns ``[T, H, head_dim]``. If ``rope`` is given,
    its rotation is applied to q/k using continuous positions ``rope_pos`` ``[T]``.
    ``layout`` (built once per encoder forward by :func:`build_layout`) carries
    the RoPE angles and the SDPA scatter plan; without it they are rebuilt here.
    """
    T, H, hd = q.shape
    if T == 0:
        # A batch with no tokens at all (every user truncated to zero events by
        # an eval-point or staleness cutoff): nothing to attend over. The flash
        # kernel rejects an empty batch ("batch size must be positive").
        return q
    if layout is None:
        layout = build_layout(cu_seqlens, max_seqlen, T, rope, rope_pos)
    if rope is not None and layout.cos is not None and layout.sin is not None:
        q = rope.rotate(q.transpose(0, 1), layout.cos, layout.sin).transpose(0, 1)
        k = rope.rotate(k.transpose(0, 1), layout.cos, layout.sin).transpose(0, 1)

    if attention_backend(q.device, q.dtype) == "flash":
        # In deterministic mode use flash-attn's deterministic backward (the
        # forward is already deterministic). Wheels older than 2.4.1 lack the
        # kwarg; there the deterministic run takes the (deterministic) SDPA path.
        deterministic = os.environ.get("PRAGMATIQ_DETERMINISTIC") == "1"
        if not deterministic or _FLASH_HAS_DETERMINISTIC:
            kwargs = {"deterministic": True} if deterministic else {}
            return flash_attn_varlen_func(
                q, k, v, layout.cu_seqlens, layout.cu_seqlens,
                layout.max_seqlen, layout.max_seqlen, dropout_p=dropout_p, causal=False,
                **kwargs,
            )
        warnings.warn(
            "flash-attn < 2.4.1 has no deterministic backward; deterministic runs use "
            "the SDPA fallback (install flash-attn>=2.4.1 for the fast path)",
            stacklevel=2,
        )

    # SDPA fallback: gather each length bucket of segments into its own padded
    # [n_seg_b, H, L_b, hd] block. index_copy_/index_select (not advanced-index
    # assignment) so the scatter has a deterministic CUDA implementation; the
    # indices never overlap. Padded keys are masked, so the per-segment result is
    # independent of which bucket a segment lands in.
    buckets = layout.padded_buckets()
    out = q.new_empty(T, H, hd)
    for token_idx, flat_idx, key_pad in buckets:
        n_b, L_b = key_pad.shape
        rows = n_b * L_b
        if len(buckets) == 1:
            q_b, k_b, v_b = q, k, v
        else:
            q_b, k_b, v_b = (t.index_select(0, token_idx) for t in (q, k, v))
        qb = q.new_zeros(rows, H, hd).index_copy_(0, flat_idx, q_b).view(n_b, L_b, H, hd).permute(0, 2, 1, 3)
        kb = k.new_zeros(rows, H, hd).index_copy_(0, flat_idx, k_b).view(n_b, L_b, H, hd).permute(0, 2, 1, 3)
        vb = v.new_zeros(rows, H, hd).index_copy_(0, flat_idx, v_b).view(n_b, L_b, H, hd).permute(0, 2, 1, 3)
        attn_mask = key_pad[:, None, None, :]  # broadcast over heads, queries
        o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=attn_mask, dropout_p=dropout_p)
        o = o.permute(0, 2, 1, 3).reshape(rows, H, hd).index_select(0, flat_idx)  # [T_b, H, hd]
        if len(buckets) == 1:
            return o
        out.index_copy_(0, token_idx, o)
    return out


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
                rope_pos: torch.Tensor | None = None,
                layout: VarlenLayout | None = None) -> torch.Tensor:
        """``x``: ``[T, d]`` flat tokens. Returns ``[T, d]``."""
        T = x.shape[0]
        qkv = self.qkv(x).view(T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        dp = self.dropout if self.training else 0.0
        out = varlen_self_attention(q, k, v, cu_seqlens, max_seqlen, self.rope, rope_pos, dp, layout)
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
                rope_pos: torch.Tensor | None = None,
                layout: VarlenLayout | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), cu_seqlens, max_seqlen, rope_pos, layout))
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
                rope_pos: torch.Tensor | None = None, *, check: bool = True) -> torch.Tensor:
        """Run every block over the flat tokens ``x`` ``[T, d]`` with one shared layout."""
        first = self.blocks[0] if len(self.blocks) else None
        rope = first.attn.rope if isinstance(first, TransformerBlock) else None
        layout = build_layout(cu_seqlens, max_seqlen, x.shape[0], rope, rope_pos, check=check)
        for blk in self.blocks:
            x = blk(x, cu_seqlens, max_seqlen, rope_pos, layout)
        return self.norm(x)
