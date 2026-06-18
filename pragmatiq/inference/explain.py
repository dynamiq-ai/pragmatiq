"""Event attribution via integrated gradients.

``EventAttributor`` computes integrated gradients of a scalar prediction (e.g. a
classification-head logit, or the norm of the user embedding) with respect to
the per-event vectors ``z_e``, then aggregates to a per-event importance score
and returns the top-k events behind a prediction.

Integrated gradients are taken along the straight path from a zero baseline to
the actual per-event vectors ``z_e`` (not raw inputs). The output is a per-event
importance RANKING. IG's completeness identity — attributions sum to the score
minus its baseline — is exact only in the step→∞ limit; with a finite ``steps``
the sum is an approximation (the history encoder + score are nonlinear, so raise
``steps`` for tighter completeness at higher cost). Use it for ranking which
events drove a prediction, not as an exact additive decomposition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..data.collate import PackedBatch
from ..models.pragmatiq import PragmaModel


@dataclass
class EventAttribution:
    """Top-k events behind one user's prediction."""

    user_id: str
    event_indices: list[int]  # indices into the user's event stream (0 = oldest)
    scores: list[float]
    event_ts: list[int]  # µs timestamps of those events


class EventAttributor:
    """Integrated-gradients attribution over per-event representations."""

    def __init__(self, model: PragmaModel, steps: int = 64) -> None:
        self.model = model.eval()
        self.steps = steps

    def _forward_from_events(self, batch: PackedBatch, z_e: torch.Tensor) -> tuple:
        """Run history encoding from a (possibly perturbed) ``z_e``."""
        z_a = self.model._encode_profile(batch)
        z_h_usr, z_h_event = self.model._encode_history(batch, z_a, z_e)
        return z_h_usr, z_h_event

    def attribute(
        self,
        batch: PackedBatch,
        score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        top_k: int = 5,
    ) -> list[EventAttribution]:
        """Attribute each user's score to its events; return top-k per user.

        ``score_fn`` maps user embeddings ``[n_users, d]`` → per-user scalar
        scores ``[n_users]`` (default: the L2 norm of the user embedding).
        """
        if score_fn is None:
            def score_fn(z: torch.Tensor) -> torch.Tensor:
                return z.norm(dim=-1)

        with torch.no_grad():
            _, z_e, _, _ = self.model._encode_events(batch)  # [E, d] baseline event vectors
        baseline = torch.zeros_like(z_e)
        total_grad = torch.zeros_like(z_e)
        for s in range(1, self.steps + 1):
            alpha = s / self.steps
            z = (baseline + alpha * (z_e - baseline)).detach().requires_grad_(True)
            z_h_usr, _ = self._forward_from_events(batch, z)
            scores = score_fn(z_h_usr).sum()
            grad = torch.autograd.grad(scores, z)[0]
            total_grad = total_grad + grad
        ig = (z_e - baseline) * total_grad / self.steps  # [E, d]
        per_event = ig.sum(dim=-1)  # [E] signed importance

        out: list[EventAttribution] = []
        cu = batch.cu_seqlens_history
        for u in range(batch.n_users):
            lo, hi = int(cu[u]), int(cu[u + 1])
            ev_scores = per_event[lo:hi]
            k = min(top_k, hi - lo)
            if k <= 0:
                out.append(EventAttribution(batch.user_ids[u], [], [], []))
                continue
            top = torch.topk(ev_scores.abs(), k).indices
            out.append(EventAttribution(
                user_id=batch.user_ids[u],
                event_indices=[int(i) for i in top.tolist()],  # per-user event index
                scores=[float(ev_scores[i]) for i in top.tolist()],
                event_ts=[int(batch.event_ts[lo + int(i)]) for i in top.tolist()],  # µs timestamp
            ))
        return out
