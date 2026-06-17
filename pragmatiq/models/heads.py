"""Task heads (Phase 4).

- :class:`MLMHead` — the pretraining head: for each (masked) token it concatenates
  ``[ẑ_e(token), z_h(its event), z_h(USR)] ∈ R^{3d}``, projects ``3d → d`` and
  produces logits via the **tied** embedding weights; trained with cross-entropy
  and label smoothing 0.1.
- :class:`ClassificationHead` — a fine-tuning head on the user embedding
  ``z_h[USR]``.

Heads are registered (``@register_head``) so configs can reference them by name.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import register_head
from .pragmatiq import PragmaOutput

LABEL_SMOOTHING = 0.1


@register_head("mlm")
class MLMHead(nn.Module):
    """Masked-language-model head: 3d concat → ``Linear(3d→d)`` → tied logits.

    For each (masked) token it concatenates ``[ẑ_e(token), z_h(its event),
    z_h(USR)] ∈ R^{3d}``, projects ``3d → d``, and reads out logits through the
    **tied** token-embedding weights. Trained with cross-entropy and label
    smoothing 0.1.

    In the PRAGMA+Nemotron variant (``text_dim > 0``) a parallel ``Linear(3d →
    text_dim)`` reconstructs masked text tokens' frozen embeddings (trained with MSE),
    sharing the same 3d context but a separate read-out from the tied vocab logits.
    """

    def __init__(self, dim: int, text_dim: int = 0) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * dim, dim)
        self.text_out = nn.Linear(3 * dim, text_dim) if text_dim else None

    @staticmethod
    def _context(out: PragmaOutput, token_idx: torch.Tensor | None) -> torch.Tensor:
        """The ``[ẑ_e, z_h(event), z_h(USR)] ∈ R^{3d}`` context per selected token."""
        tok = out.token_repr if token_idx is None else out.token_repr[token_idx]
        ev = out.event_of_token if token_idx is None else out.event_of_token[token_idx]
        z_he = out.history_event_repr[ev]
        z_hu = out.user_repr[out.user_of_event[ev]]
        return torch.cat([tok, z_he, z_hu], dim=-1)

    def forward(
        self,
        out: PragmaOutput,
        embedding_weight: torch.Tensor,
        token_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits ``[M, vocab]`` for the selected (default: all) tokens."""
        h = self.proj(self._context(out, token_idx))
        return h @ embedding_weight.t()  # tied logits

    def reconstruct_text(self, out: PragmaOutput, token_idx: torch.Tensor) -> torch.Tensor:
        """Predicted frozen text embeddings ``[M, text_dim]`` for masked text tokens."""
        if self.text_out is None:
            raise RuntimeError("MLMHead has no text head; build it with text_dim > 0")
        return self.text_out(self._context(out, token_idx))


def mlm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with label smoothing 0.1; ``targets`` use -100 to ignore.

    Returns a graph-connected zero when nothing contributes (empty batch or every
    target ignored), so an all-ignored batch yields a well-defined 0 gradient
    rather than CE's 0/0 NaN.
    """
    if targets.numel() == 0 or int((targets != -100).sum()) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets, ignore_index=-100, label_smoothing=LABEL_SMOOTHING)


def text_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss for masked text embeddings (Nemotron variant).

    Returns a graph-connected zero when no text token is masked this step, so a batch
    with no text target yields a well-defined 0 gradient rather than ``mse_loss``'s NaN
    on an empty input.
    """
    if pred.numel() == 0:
        return pred.sum() * 0.0
    # fp32 reduction so the term is stable under bf16-mixed and never trips a
    # pred/target dtype mismatch (the frozen target is fp32; pred may be bf16).
    return F.mse_loss(pred.float(), target.float())


@register_head("classification")
class ClassificationHead(nn.Module):
    """Linear (optionally MLP) head on the user embedding ``z_h[USR]``."""

    def __init__(self, dim: int, n_classes: int = 2, hidden: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_classes = n_classes
        if hidden:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes)
            )
        else:
            self.net = nn.Linear(dim, n_classes)

    def forward(self, user_repr: torch.Tensor) -> torch.Tensor:
        """Logits ``[n_users, n_classes]`` from user embeddings ``[n_users, d]``."""
        return self.net(user_repr)
