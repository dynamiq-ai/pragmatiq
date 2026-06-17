"""Varlen collation: tokenized records → a padding-free :class:`PackedBatch`.

PRAGMA encodes each event independently, then runs a history encoder over the
per-event vectors. To do this without padding we keep everything flat and carry
``cu_seqlens`` (cumulative sequence lengths, the flash-attn varlen convention):

- ``cu_seqlens_event``   token boundaries per event   (len = n_events + 1)
- ``cu_seqlens_history`` event boundaries per user     (len = n_users + 1)

The event encoder attends within each ``cu_seqlens_event`` segment; the history
encoder attends within each ``cu_seqlens_history`` segment. Profile items are
carried in their own flat arrays with ``cu_seqlens_profile``.

Within-segment attention is what lets the CPU SDPA fallback match a flash-attn
varlen forward exactly (global rule 5). The model does this by scattering each
segment into a padded block and masking the padding keys (see
``pragmatiq/models/layers.py``); :func:`block_diag_mask` is an equivalent dense
formulation used by the padding-equivalence test.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import torch

from .tokenizer import TokenizedRecord, truncate_record

_T = TypeVar("_T")


@dataclass
class PackedBatch:
    """A padding-free batch of tokenized user histories.

    Token-level arrays are concatenated across every event of every user in the
    batch; event-level arrays across every user. The ``cu_seqlens_*`` tensors
    recover the boundaries. ``event_of_token``/``user_of_event`` are convenience
    scatter indices used by the history encoder and MLM head.
    """

    # token level (sum over all events of all users)
    key_ids: torch.Tensor  # int64[T]
    value_ids: torch.Tensor  # int64[T]
    positions: torch.Tensor  # int64[T] within-field position
    cu_seqlens_event: torch.Tensor  # int32[n_events + 1]
    event_of_token: torch.Tensor  # int64[T] -> global event index
    # event level (sum over all users)
    event_ts: torch.Tensor  # int64[E] µs since epoch (raw event timestamp)
    event_time_log: torch.Tensor  # float32[E] log-seconds to last event
    event_hour: torch.Tensor  # int64[E]
    event_dow: torch.Tensor  # int64[E]
    event_dom: torch.Tensor  # int64[E]
    event_source: torch.Tensor  # int64[E]
    cu_seqlens_history: torch.Tensor  # int32[n_users + 1] event boundaries per user
    user_of_event: torch.Tensor  # int64[E] -> batch user index
    # profile level
    prof_key_ids: torch.Tensor  # int64[P]
    prof_value_ids: torch.Tensor  # int64[P]
    prof_positions: torch.Tensor  # int64[P]
    prof_time_log: torch.Tensor  # float32[n_prof_items]
    cu_seqlens_profile_item: torch.Tensor  # int32[n_prof_items + 1] tokens per profile item
    cu_seqlens_profile: torch.Tensor  # int32[n_users + 1] profile items per user
    item_of_prof_token: torch.Tensor  # int64[P] -> global profile-item index
    user_of_prof_item: torch.Tensor  # int64[n_prof_items]
    # bookkeeping
    user_ids: list[str]
    n_events_per_user: torch.Tensor  # int64[n_users]
    # Nemotron variant (text_value_mode="embed"): is_text marks event tokens whose
    # value is a frozen text embedding; text_values holds those tokens' raw strings
    # in token order (len == is_text.sum()); feed_text gates which text tokens supply
    # their embedding to the input (all of them at inference; the masker hides masked
    # ones during pretraining). All empty/false in BPE mode — the model ignores them.
    is_text: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))
    feed_text: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))
    text_values: list[str] = field(default_factory=list)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_events(self) -> int:
        return int(self.cu_seqlens_event.numel() - 1)

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.numel())

    def to(self, device: torch.device | str) -> PackedBatch:
        """Move all tensors to ``device`` (lists are left as-is)."""
        moved: dict[str, Any] = {
            f: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for f, v in self.__dict__.items()
        }
        return PackedBatch(**moved)

    def token_budget(self) -> int:
        """Total tokens (events + profile) — the dynamic-batch budget metric."""
        return self.n_tokens + int(self.prof_key_ids.numel())


def block_diag_mask(cu_seqlens: torch.Tensor, total_len: int) -> torch.Tensor:
    """Additive attention mask (0 within a segment, -inf across) from cu_seqlens.

    Used by the SDPA fallback so a packed forward matches a flash-attn varlen
    forward. ``cu_seqlens`` is int32 of shape ``[n_segments + 1]``.
    """
    seg = segment_ids(cu_seqlens, total_len)
    same = seg[:, None] == seg[None, :]
    mask = torch.zeros(total_len, total_len, dtype=torch.float32, device=cu_seqlens.device)
    mask.masked_fill_(~same, float("-inf"))
    return mask


def segment_ids(cu_seqlens: torch.Tensor, total_len: int) -> torch.Tensor:
    """Segment id per flat position (int64[total_len]) from a cu_seqlens vector.

    Uses ``repeat_interleave`` over segment lengths so the ids stay correct even
    when a segment is empty (two equal cu entries) — a boundary-mark + cumsum
    would collapse the empty segment's index. Matches the collator's event
    indexing (which counts empty events too).
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=cu_seqlens.device), lengths
    )


def run_with_oom_retry(
    fn: Callable[[int], _T],
    token_budget: int,
    min_budget: int = 256,
    logger: logging.Logger | None = None,
) -> tuple[_T, int]:
    """Call ``fn(token_budget)``; on CUDA OOM, halve the budget and retry.

    Returns ``(result, budget_used)``. Re-raises if the budget falls below
    ``min_budget`` (the OOM-retry contract). Safe on CPU:
    ``empty_cache`` is only called when CUDA is available.
    """
    log = logger or logging.getLogger(__name__)
    budget = token_budget
    while True:
        try:
            return fn(budget), budget
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_budget = budget // 2
            if new_budget < min_budget:
                log.error("CUDA OOM at token_budget=%d; below floor %d, giving up", budget, min_budget)
                raise
            log.warning("CUDA OOM at token_budget=%d; halving to %d and retrying", budget, new_budget)
            budget = new_budget


class VarlenCollator:
    """Collates a list of :class:`TokenizedRecord` into a :class:`PackedBatch`.

    No padding anywhere: arrays are concatenated and ``cu_seqlens`` carry the
    structure. The collator is pure/stateless, so it is safe across workers.
    """

    def __call__(self, records: list[TokenizedRecord]) -> PackedBatch:
        """Pack ``records`` (one user each) into a single batch."""
        if not records:
            raise ValueError("cannot collate an empty record list")

        key_ids, value_ids, positions, event_of_token = [], [], [], []
        is_text_parts: list[np.ndarray] = []
        text_values: list[str] = []
        ev_ts, ev_tlog, ev_hour, ev_dow, ev_dom, ev_src = [], [], [], [], [], []
        user_of_event = []
        evt_lens: list[int] = []  # tokens per event (for cu_seqlens_event)
        hist_lens: list[int] = []  # events per user (for cu_seqlens_history)

        p_key, p_val, p_pos, item_of_ptok = [], [], [], []
        p_tlog, prof_item_lens, prof_per_user, user_of_pitem = [], [], [], []

        global_event = 0
        global_pitem = 0
        n_events_per_user = []

        for u, rec in enumerate(records):
            n_ev = rec.n_events
            hist_lens.append(n_ev)
            n_events_per_user.append(n_ev)
            # Text markers/strings are only used when self-consistent with the tokens
            # (Nemotron variant); otherwise treat the record as text-free so is_text /
            # text_values can never fall out of alignment.
            it_full = rec.is_text
            if it_full.size != rec.key_ids.size or len(rec.text_values) != int(it_full.sum()):
                it_full = np.zeros(rec.key_ids.size, dtype=np.int8)
                rec_text_values: list[str] = []
            else:
                rec_text_values = list(rec.text_values)
            for e in range(n_ev):
                lo, hi = int(rec.event_offsets[e]), int(rec.event_offsets[e + 1])
                ntok = hi - lo
                evt_lens.append(ntok)
                key_ids.append(rec.key_ids[lo:hi])
                value_ids.append(rec.value_ids[lo:hi])
                positions.append(rec.positions[lo:hi])
                is_text_parts.append(it_full[lo:hi])
                event_of_token.append(np.full(ntok, global_event, dtype=np.int64))
                global_event += 1
            text_values.extend(rec_text_values)
            ev_ts.append(rec.event_ts)
            ev_tlog.append(rec.time_log)
            ev_hour.append(rec.hour.astype(np.int64))
            ev_dow.append(rec.dow.astype(np.int64))
            ev_dom.append(rec.dom.astype(np.int64))
            ev_src.append(rec.source_ids.astype(np.int64))
            user_of_event.append(np.full(n_ev, u, dtype=np.int64))

            n_items = len(rec.prof_offsets) - 1
            prof_per_user.append(n_items)
            for it in range(n_items):
                lo, hi = int(rec.prof_offsets[it]), int(rec.prof_offsets[it + 1])
                ntok = hi - lo
                prof_item_lens.append(ntok)
                p_key.append(rec.prof_key_ids[lo:hi])
                p_val.append(rec.prof_value_ids[lo:hi])
                p_pos.append(rec.prof_positions[lo:hi])
                item_of_ptok.append(np.full(ntok, global_pitem, dtype=np.int64))
                global_pitem += 1
            p_tlog.append(rec.prof_time_log)
            user_of_pitem.append(np.full(n_items, u, dtype=np.int64))

        def cat_long(parts: list[np.ndarray]) -> torch.Tensor:
            if not parts:
                return torch.zeros(0, dtype=torch.int64)
            return torch.from_numpy(np.concatenate(parts).astype(np.int64))

        def cat_f32(parts: list[np.ndarray]) -> torch.Tensor:
            if not parts:
                return torch.zeros(0, dtype=torch.float32)
            return torch.from_numpy(np.concatenate(parts).astype(np.float32))

        def cu(lens: list[int]) -> torch.Tensor:
            out = np.zeros(len(lens) + 1, dtype=np.int32)
            np.cumsum(np.asarray(lens, dtype=np.int32), out=out[1:])
            return torch.from_numpy(out)

        is_text = (
            torch.from_numpy(np.concatenate(is_text_parts)).bool()
            if is_text_parts else torch.zeros(0, dtype=torch.bool)
        )

        return PackedBatch(
            key_ids=cat_long(key_ids),
            value_ids=cat_long(value_ids),
            positions=cat_long(positions),
            cu_seqlens_event=cu(evt_lens),
            event_of_token=cat_long(event_of_token),
            event_ts=cat_long(ev_ts),
            event_time_log=cat_f32(ev_tlog),
            event_hour=cat_long(ev_hour),
            event_dow=cat_long(ev_dow),
            event_dom=cat_long(ev_dom),
            event_source=cat_long(ev_src),
            cu_seqlens_history=cu(hist_lens),
            user_of_event=cat_long(user_of_event),
            prof_key_ids=cat_long(p_key),
            prof_value_ids=cat_long(p_val),
            prof_positions=cat_long(p_pos),
            prof_time_log=cat_f32(p_tlog),
            cu_seqlens_profile_item=cu(prof_item_lens),
            cu_seqlens_profile=cu(prof_per_user),
            item_of_prof_token=cat_long(item_of_ptok),
            user_of_prof_item=cat_long(user_of_pitem),
            user_ids=[r.user_id for r in records],
            n_events_per_user=torch.tensor(n_events_per_user, dtype=torch.int64),
            is_text=is_text,
            feed_text=is_text.clone(),  # inference feeds every text token; the masker hides masked ones
            text_values=text_values,
        )


class TruncatingCollator(VarlenCollator):
    """Collator that truncates each record at a per-user eval cutoff first.

    ``cutoffs`` maps user_id -> cutoff µs (typically a label table's eval_ts).
    Users without a cutoff pass through untouched; users whose entire history
    falls after their cutoff collate with zero events (profile-only). This is
    the enforcement point of the no-hindcasting rule: a batch built through
    this collator can never contain an event at or past its user's eval point.
    """

    def __init__(self, cutoffs: Mapping[str, int]) -> None:
        self.cutoffs = cutoffs

    def __call__(self, records: list[TokenizedRecord]) -> PackedBatch:
        records = [
            truncate_record(r, int(self.cutoffs[r.user_id])) if r.user_id in self.cutoffs else r
            for r in records
        ]
        return super().__call__(records)
