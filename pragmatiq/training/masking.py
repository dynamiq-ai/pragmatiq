"""MLM masking strategy.

Three selection modes are unioned per batch:

- **token**  each token selected independently with ``p=0.15``;
- **event**  each event selected with ``p=0.10`` → all its tokens masked;
- **key**    per user, each key selected with ``p=0.10`` → all that user's tokens
  with that key masked ("all values of sampled keys").

Of the selected positions, **10% become ``[UNK]`` and are excluded from the loss**
(label ``-100``); the remaining 90% become ``[MASK]`` with the original value id
as the target. Non-selected positions have label ``-100``. The value token is
predicted while the key token is kept (we know the key, predict its value).

``mask_type`` records which mode selected each position (event > key > token
priority) so the trainer can log per-masking-type loss.

In the PRAGMA+Nemotron variant, text tokens (``batch.is_text``) carry a frozen text
embedding rather than a vocab id, so they are reconstructed with MSE instead of
cross-entropy: a masked text token is recorded in ``text_loss_idx`` (not the CE
``labels``), and ``feed_text`` marks the text tokens whose embedding the model may
still see (every text token except the masked/UNK-dropped ones).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..data.collate import PackedBatch
from ..data.tokenizer import MASK, UNK
from ..registry import register_masker

# mask_type codes
T_NONE, T_TOKEN, T_KEY, T_EVENT = -1, 0, 1, 2
TYPE_NAMES = {T_TOKEN: "token", T_KEY: "key", T_EVENT: "event"}


@dataclass
class MaskedBatch:
    """Masking result aligned to a batch's flat token arrays."""

    input_value_ids: torch.Tensor  # [T] value ids with [MASK]/[UNK] substitutions
    labels: torch.Tensor  # [T] original value id where predicted, else -100
    mask_type: torch.Tensor  # [T] one of T_NONE/T_TOKEN/T_KEY/T_EVENT
    selected_idx: torch.Tensor  # [M] indices of positions contributing to CE loss
    # Nemotron variant: text tokens the model may still embed (all but masked/dropped
    # ones), and the masked text tokens reconstructed via MSE. Empty in BPE mode.
    feed_text: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))
    text_loss_idx: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.int64))

    @property
    def n_selected(self) -> int:
        return int(self.selected_idx.numel())


@register_masker("pragma")
class MaskingStrategy:
    """The PRAGMA token/event/key masking scheme."""

    def __init__(self, p_token: float = 0.15, p_event: float = 0.10, p_key: float = 0.10,
                 p_unk: float = 0.10) -> None:
        self.p_token = p_token
        self.p_event = p_event
        self.p_key = p_key
        self.p_unk = p_unk  # GUESS-exposed: fraction of selected → [UNK], excluded

    def __call__(self, batch: PackedBatch, generator: torch.Generator | None = None) -> MaskedBatch:
        """Produce a :class:`MaskedBatch` for ``batch`` (deterministic given ``generator``)."""
        device = batch.key_ids.device
        T = batch.key_ids.numel()

        def rand(n: int) -> torch.Tensor:
            return torch.rand(n, generator=generator, device=device)

        mask_type = torch.full((T,), T_NONE, dtype=torch.int8, device=device)

        # token-level
        token_sel = rand(T) < self.p_token
        mask_type[token_sel] = T_TOKEN

        # key-level (per user): sample (user, key) pairs, mask all matching tokens
        user_of_token = batch.user_of_event[batch.event_of_token]  # [T]
        # encode (user, key) into a single id to sample uniquely per pair
        key_ids = batch.key_ids
        pair = user_of_token.to(torch.int64) * (int(key_ids.max().item()) + 1 if T else 1) + key_ids
        uniq, inv = torch.unique(pair, return_inverse=True)
        pair_sel = rand(uniq.numel()) < self.p_key
        key_sel = pair_sel[inv]
        mask_type[key_sel] = T_KEY  # key overrides token

        # event-level: sample events, mask all their tokens
        n_events = batch.n_events
        event_sel_e = rand(n_events) < self.p_event
        event_sel = event_sel_e[batch.event_of_token]
        mask_type[event_sel] = T_EVENT  # event overrides key/token

        selected = mask_type != T_NONE
        sel_idx = selected.nonzero(as_tuple=False).squeeze(1)

        input_value_ids = batch.value_ids.clone()
        labels = torch.full((T,), -100, dtype=torch.int64, device=device)
        # Text tokens carry a frozen embedding, not a vocab id: a masked one is an MSE
        # target, never a CE label. The model may embed every text token except those
        # masked or UNK-dropped this step.
        is_text = batch.is_text.bool() if batch.is_text.numel() == T else torch.zeros(T, dtype=torch.bool, device=device)
        text_loss_idx = torch.zeros(0, dtype=torch.int64, device=device)

        if sel_idx.numel():
            # 10% of selected → [UNK] (excluded from loss); 90% → [MASK] (predicted)
            is_unk = rand(sel_idx.numel()) < self.p_unk
            unk_idx = sel_idx[is_unk]
            mask_idx = sel_idx[~is_unk]
            input_value_ids[unk_idx] = UNK
            input_value_ids[mask_idx] = MASK  # text tokens too → hidden behind [MASK]
            mask_is_text = is_text[mask_idx]
            ce_mask_idx = mask_idx[~mask_is_text]
            text_loss_idx = mask_idx[mask_is_text]  # reconstructed via MSE, not CE
            labels[ce_mask_idx] = batch.value_ids[ce_mask_idx]
            mask_type[unk_idx] = T_NONE  # excluded positions don't count toward type loss
            mask_type[text_loss_idx] = T_NONE  # text loss is reported on its own, not per-CE-type

        feed_text = is_text & ~selected
        loss_idx = (labels != -100).nonzero(as_tuple=False).squeeze(1)
        return MaskedBatch(input_value_ids=input_value_ids, labels=labels,
                           mask_type=mask_type, selected_idx=loss_idx,
                           feed_text=feed_text, text_loss_idx=text_loss_idx)
