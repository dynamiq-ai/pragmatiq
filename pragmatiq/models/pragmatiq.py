"""The pragmatiq model: profile / event / history encoders.

Pipeline for one :class:`~pragmatiq.data.collate.PackedBatch`:

1. ``TokenEmbedding`` → per-token vectors ``x`` (key+value+within-field pos).
2. ``EventEncoder`` encodes each event independently (segments = events): a
   per-event ``[EVT]`` marker is prepended, blocks run with block-diagonal
   attention; token outputs are ``ẑ_e`` (for MLM) and the ``[EVT]`` outputs,
   plus ``CalendarEmbedding``, give the per-event vector ``z_e``.
3. ``ProfileStateEncoder`` runs over each user's profile tokens with a ``[USR]``
   marker and TimeRoPE on the per-item log-seconds → ``z_a`` (the ``[USR]``
   output).
4. ``HistoryEncoder`` runs over each user's sequence ``[z_a, z_e…]`` with
   TimeRoPE on log-seconds-to-last-event → ``z_h``; the ``[USR]`` slot output is
   the user embedding, the event slots are per-event history states.

The forward returns these representations; the MLM head (heads.py) consumes
``ẑ_e``, the per-event history states, and the user embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.collate import PackedBatch
from ..data.tokenizer import EVT, USR
from .embeddings import CalendarEmbedding, TokenEmbedding
from .layers import Encoder

# Checkpoint schema version, shared by the trainer (writer) and from_pretrained
# (reader) so both validate against one source of truth.
CKPT_FORMAT = 2


@dataclass
class ModelConfig:
    """pragmatiq model dimensions (size table)."""

    vocab_size: int
    dim: int = 192
    n_heads: int = 3
    depth_profile: int = 1
    depth_event: int = 5
    depth_history: int = 2
    dropout: float = 0.1
    rope_base: float = 10_000.0  # GUESS: RoPE base for the time axis
    max_position: int = 64
    # PRAGMA+Nemotron variant: name of a registered frozen text encoder (e.g.
    # "nemotron"/"hash"). None ⇒ the default BPE path with no text branch. Must match
    # the tokenizer's text_value_mode="embed" build; text_encoder_dim is that encoder's
    # output width (the MSE reconstruction target dimension).
    text_encoder: str | None = None
    text_encoder_dim: int = 64

    @classmethod
    def preset(cls, name: str, vocab_size: int, overrides: dict[str, Any] | None = None) -> ModelConfig:
        """A named size: ``nano`` (~1M, CPU/CI), ``small`` (10M), ``medium`` (100M), ``large`` (1B).

        ``nano`` is not in the paper; it exists so the gates and ``quickstart``
        run end-to-end on a CPU in minutes (the nano config targets CPU/CI).

        ``overrides`` lets callers tune any architecture field on top of the size
        table — notably ``rope_base`` and ``dropout`` (the paper-silent knobs the
        SPEC says to expose in config). Unknown keys and ``vocab_size`` are ignored.
        """
        table = {
            "nano": dict(dim=64, n_heads=2, depth_profile=1, depth_event=2, depth_history=1),
            "small": dict(dim=192, n_heads=3, depth_profile=1, depth_event=5, depth_history=2),
            "medium": dict(dim=512, n_heads=8, depth_profile=3, depth_event=16, depth_history=6),
            "large": dict(dim=1024, n_heads=16, depth_profile=9, depth_event=45, depth_history=18),
        }
        if name not in table:
            raise ValueError(f"unknown size {name!r}; choose from {sorted(table)}")
        params: dict[str, Any] = dict(table[name])
        for k, v in (overrides or {}).items():
            if k in cls.__dataclass_fields__ and k != "vocab_size":
                params[k] = v
        return cls(vocab_size=vocab_size, **params)


def assemble_segments(
    seg_lengths: torch.Tensor,
    prefix_vec: torch.Tensor,
    token_vals: torch.Tensor,
    token_pos: torch.Tensor | None = None,
    prefix_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepend one ``prefix_vec`` per segment to the segment's tokens.

    Returns ``(x, cu_seqlens, prefix_idx, token_dst, rope_pos)`` where ``x`` is
    the assembled flat sequence ``[T + S, d]``, ``prefix_idx`` ``[S]`` and
    ``token_dst`` ``[T]`` index the prefix and token rows of ``x`` respectively.
    """
    S = seg_lengths.numel()
    T = token_vals.shape[0]
    d = token_vals.shape[1]
    device = token_vals.device
    new_lengths = seg_lengths + 1
    new_cu = F.pad(new_lengths.cumsum(0), (1, 0)).to(torch.int32)  # [S+1]
    prefix_idx = new_cu[:-1].to(torch.long)
    x = token_vals.new_zeros(T + S, d)
    is_prefix = torch.zeros(T + S, dtype=torch.bool, device=device)
    is_prefix[prefix_idx] = True
    token_dst = (~is_prefix).nonzero(as_tuple=False).squeeze(1)
    # index_copy_ instead of advanced index-assignment so the scatter has a
    # deterministic CUDA implementation under torch.use_deterministic_algorithms.
    # prefix_idx/token_dst partition the rows, so the writes never overlap.
    x.index_copy_(0, prefix_idx, prefix_vec.to(x.dtype))
    x.index_copy_(0, token_dst, token_vals)
    rope_pos = None
    if token_pos is not None:
        # Positions are pure data: keep them fp32 regardless of segment dtype so
        # log-second resolution is never quantized under bf16-true precision
        # (angles() upcasts; rotate() casts cos/sin back to x.dtype).
        rope_pos = torch.zeros(T + S, dtype=torch.float32, device=device)
        rope_pos.index_copy_(0, token_dst, token_pos.float())
        if prefix_pos is not None:
            rope_pos.index_copy_(0, prefix_idx, prefix_pos.float())
    return x, new_cu, prefix_idx, token_dst, rope_pos


def _segsum(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Sum ``values`` within consecutive groups of sizes ``counts`` → ``[len(counts)]``."""
    if values.numel() == 0:
        return torch.zeros_like(counts)
    seg = torch.repeat_interleave(torch.arange(counts.numel(), device=values.device), counts)
    out = torch.zeros(counts.numel(), dtype=values.dtype, device=values.device)
    out.scatter_add_(0, seg, values)
    return out


@dataclass
class PragmaOutput:
    """Representations produced by :meth:`PragmaModel.forward`."""

    token_repr: torch.Tensor  # ẑ_e  [T, d] event-encoder token outputs (for MLM)
    event_repr: torch.Tensor  # z_e  [E, d] per-event vectors
    history_event_repr: torch.Tensor  # z_h[event]  [E, d]
    user_repr: torch.Tensor  # z_h[USR]  [n_users, d] user embeddings
    event_of_token: torch.Tensor  # [T]
    user_of_event: torch.Tensor  # [E]
    # Nemotron variant: frozen text-embedding targets and their token positions
    # (the MSE reconstruction targets); None on the BPE path.
    text_vecs: torch.Tensor | None = None  # [n_text, text_encoder_dim]
    text_token_idx: torch.Tensor | None = None  # [n_text] -> index into [T]


class PragmaModel(nn.Module):
    """Bidirectional pragmatiq encoder stack (profile → event → history)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.dim
        self.embed = TokenEmbedding(config.vocab_size, d, config.max_position, config.dropout)
        self.calendar = CalendarEmbedding(d)
        self.event_encoder = Encoder(d, config.depth_event, config.n_heads, config.dropout,
                                     use_rope=False)
        self.profile_encoder = Encoder(d, config.depth_profile, config.n_heads, config.dropout,
                                       use_rope=True, rope_base=config.rope_base)
        self.history_encoder = Encoder(d, config.depth_history, config.n_heads, config.dropout,
                                       use_rope=True, rope_base=config.rope_base)
        # PRAGMA+Nemotron variant: a frozen text encoder (not a parameter — never saved
        # or trained) plus a learned projection of its vectors into the model width.
        # text_proj is the only trainable piece of the text input path.
        self.text_encoder: Any | None = None
        self.text_proj: nn.Module | None = None
        if config.text_encoder:
            from .text_encoder import build_text_encoder

            self.text_encoder = build_text_encoder(config.text_encoder, dim=config.text_encoder_dim)
            # The frozen encoder's own width is authoritative (Nemotron's hidden size is
            # fixed by the pre-trained model); record it so the checkpoint and the MSE
            # target dimension stay consistent regardless of the configured hint.
            config.text_encoder_dim = int(self.text_encoder.dim)
            self.text_proj = nn.Linear(config.text_encoder_dim, d)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @property
    def embedding_weight(self) -> torch.Tensor:
        """Shared key/value table (tied to the MLM logits)."""
        return self.embed.weight

    # ------------------------------------------------------------------ encoders
    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        """Frozen text encoder on the batch's text strings → ``[n_text, text_dim]``."""
        assert self.text_encoder is not None
        return self.text_encoder.encode(texts).float()

    def _text_input(self, batch: PackedBatch, x_tok: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Add the projected frozen text embedding to fed text tokens.

        Returns ``(x_tok, text_vecs, text_token_idx)``: the frozen vectors are the MSE
        reconstruction targets; ``feed_text`` gates which text tokens see their value
        (masked ones are hidden, so the model must reconstruct them).
        """
        if self.text_encoder is None or self.text_proj is None or not batch.is_text.numel():
            return x_tok, None, None
        text_token_idx = batch.is_text.bool().nonzero(as_tuple=False).squeeze(1)
        if not text_token_idx.numel():
            return x_tok, None, None
        text_vecs = self._encode_text(batch.text_values).to(x_tok.device)  # [n_text, td]
        proj = self.text_proj(text_vecs).to(x_tok.dtype)  # [n_text, d]
        feed = batch.feed_text.bool()[text_token_idx]
        if bool(feed.any()):
            x_tok = x_tok.index_add(0, text_token_idx[feed], proj[feed])
        return x_tok, text_vecs, text_token_idx

    def _encode_events(
        self, batch: PackedBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x_tok = self.embed(batch.key_ids, batch.value_ids, batch.positions)  # [T, d]
        x_tok, text_vecs, text_token_idx = self._text_input(batch, x_tok)
        seg_lengths = (batch.cu_seqlens_event[1:] - batch.cu_seqlens_event[:-1]).long()
        evt_marker = self.embed.embed(torch.full((seg_lengths.numel(),), EVT, device=x_tok.device))
        x, cu, prefix_idx, token_dst, _ = assemble_segments(seg_lengths, evt_marker, x_tok)
        max_len = int((cu[1:] - cu[:-1]).max()) if cu.numel() > 1 else 1
        h = self.event_encoder(x, cu, max_len)
        z_tok = h[token_dst]  # ẑ_e  [T, d]
        evt_vec = h[prefix_idx]  # [E, d]
        z_e = evt_vec + self.calendar(batch.event_hour, batch.event_dow, batch.event_dom)
        return z_tok, z_e, text_vecs, text_token_idx

    def _encode_profile(self, batch: PackedBatch) -> torch.Tensor:
        x_prof = self.embed(batch.prof_key_ids, batch.prof_value_ids, batch.prof_positions)  # [P, d]
        item_lengths = (batch.cu_seqlens_profile_item[1:] - batch.cu_seqlens_profile_item[:-1]).long()
        items_per_user = (batch.cu_seqlens_profile[1:] - batch.cu_seqlens_profile[:-1]).long()
        tokens_per_user = _segsum(item_lengths, items_per_user)  # [n_users]
        # per-token time = its item's log-seconds
        tok_time = batch.prof_time_log[batch.item_of_prof_token] if x_prof.shape[0] else x_prof.new_zeros(0)
        usr_marker = self.embed.embed(torch.full((tokens_per_user.numel(),), USR, device=x_prof.device))
        usr_pos = x_prof.new_zeros(tokens_per_user.numel())  # [USR] anchored at log-sec 0
        x, cu, prefix_idx, _, rope_pos = assemble_segments(
            tokens_per_user, usr_marker, x_prof, tok_time, usr_pos
        )
        max_len = int((cu[1:] - cu[:-1]).max()) if cu.numel() > 1 else 1
        h = self.profile_encoder(x, cu, max_len, rope_pos)
        return h[prefix_idx]  # z_a  [n_users, d]

    def _encode_history(
        self, batch: PackedBatch, z_a: torch.Tensor, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events_per_user = (batch.cu_seqlens_history[1:] - batch.cu_seqlens_history[:-1]).long()
        evt_pos = batch.event_time_log  # log-sec to last event
        usr_pos = z_a.new_zeros(events_per_user.numel())  # [USR] at log-sec 0 (anchor)
        x, cu, prefix_idx, token_dst, rope_pos = assemble_segments(
            events_per_user, z_a, z_e, evt_pos, usr_pos
        )
        max_len = int((cu[1:] - cu[:-1]).max()) if cu.numel() > 1 else 1
        h = self.history_encoder(x, cu, max_len, rope_pos)
        return h[prefix_idx], h[token_dst]  # z_h[USR] [n_users,d], z_h[event] [E,d]

    def forward(self, batch: PackedBatch) -> PragmaOutput:
        """Encode a packed batch into token/event/user representations."""
        z_tok, z_e, text_vecs, text_token_idx = self._encode_events(batch)
        z_a = self._encode_profile(batch)
        z_h_usr, z_h_event = self._encode_history(batch, z_a, z_e)
        return PragmaOutput(
            token_repr=z_tok, event_repr=z_e, history_event_repr=z_h_event,
            user_repr=z_h_usr, event_of_token=batch.event_of_token, user_of_event=batch.user_of_event,
            text_vecs=text_vecs, text_token_idx=text_token_idx,
        )

    def embed_users(self, batch: PackedBatch) -> torch.Tensor:
        """Convenience: user embeddings ``z_h[USR]`` only ``[n_users, d]``."""
        return self.forward(batch).user_repr

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls, run: str | Path, device: str = "cpu", checkpoint: str = "last.pt"
    ) -> PragmaModel:
        """Load a trained model from a run directory (notebook entry point).

        Verifies the checkpoint's tokenizer hash against the run's copied
        tokenizer (global rule 3) and attaches it so :meth:`embed_records` works.
        ``run`` may be a run directory path or a ``runs/{name}`` path.
        """
        from ..data.tokenizer import PragmaTokenizer

        run_dir = Path(run)
        tok = PragmaTokenizer.load(run_dir / "tokenizer")
        ckpt = torch.load(run_dir / "checkpoints" / checkpoint, map_location=device, weights_only=False)
        fmt = ckpt.get("format")
        if fmt != CKPT_FORMAT:
            raise ValueError(
                f"unsupported checkpoint format {fmt!r}; this build of pragmatiq writes/reads "
                f"format {CKPT_FORMAT}. Re-train with the current version."
            )
        if ckpt.get("tokenizer_hash") != tok.content_hash:
            raise ValueError(
                "tokenizer hash mismatch: this checkpoint was trained with a different "
                "tokenizer; from_pretrained refuses to run."
            )
        config = ModelConfig(**ckpt["model_config"])
        model = cls(config)
        model.load_state_dict(ckpt["model"])
        model._tokenizer = tok  # type: ignore[assignment]
        return model.to(device).eval()

    @torch.no_grad()
    def embed_records(self, records: list[dict[str, Any]]) -> np.ndarray:
        """Embed plain-dict user records (no shard pipeline) → ``[N, d]``.

        Each dict has ``user_id`` and ``events`` (+ optional ``attributes``,
        ``lifelong``, ``as_of``); see :class:`~pragmatiq.data.schema.UserRecord`.
        Requires a model loaded via :meth:`from_pretrained` (carries a tokenizer).
        """
        from ..data.collate import VarlenCollator
        from ..data.schema import UserRecord

        tok = getattr(self, "_tokenizer", None)
        if tok is None:
            raise RuntimeError("embed_records needs a tokenizer; load via from_pretrained()")
        recs = [tok.encode(r if isinstance(r, UserRecord) else UserRecord.from_dict(r)) for r in records]
        batch = VarlenCollator()(recs)
        device = next(self.parameters()).device
        return self.embed_users(batch.to(device)).float().cpu().numpy()
