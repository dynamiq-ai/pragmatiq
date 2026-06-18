"""Uplift (treatment-effect) evaluation on frozen user embeddings.

The ``comm_uplift`` label table stores, per (user, campaign), the treatment flag
plus *both* potential outcomes ``y0`` (had the user not been contacted) and
``y1`` (had they been). A learner only ever sees the factual outcome
``y = treated*y1 + (1-treated)*y0``; the counterfactual is held out for scoring.
Because both outcomes are recorded we can also report an *oracle* Qini — the
score of ranking by the true per-row effect ``y1 - y0`` — as a feasible ceiling.

``UpliftLearner`` fits a meta-learner on the embedding (T-learner: one model per
arm; or S-learner: one model with a treatment flag) and scores the predicted
uplift with the Qini coefficient on a user-grouped held-out split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class UpliftRows:
    """One row per (user, campaign): treatment flag and both potential outcomes."""

    user_id: list[str]
    campaign_id: np.ndarray
    ts: np.ndarray  # eval point (campaign send), µs
    treated: np.ndarray
    y0: np.ndarray
    y1: np.ndarray


def load_comm_uplift(path: str | Path) -> UpliftRows:
    """Read a ``comm_uplift`` label table into arrays."""
    table = pq.read_table(path)
    return UpliftRows(
        user_id=table.column("user_id").to_pylist(),
        campaign_id=table.column("campaign_id").to_numpy(zero_copy_only=False),
        ts=table.column("ts").cast(pa.int64()).to_numpy(),
        treated=table.column("treated").to_numpy().astype(np.int64),
        y0=table.column("y0").to_numpy().astype(np.int64),
        y1=table.column("y1").to_numpy().astype(np.int64),
    )


def cutoffs_from_uplift(rows: UpliftRows) -> dict[str, int]:
    """Per-user truncation cutoff = the user's earliest campaign send.

    Embedding is truncated at the first campaign so no campaign-window activity
    leaks into any of that user's rows (one cutoff per user keeps the single
    embed pass; a stricter per-campaign truncation would re-embed per campaign).
    """
    cut: dict[str, int] = {}
    for u, t in zip(rows.user_id, rows.ts):
        ti = int(t)
        if u not in cut or ti < cut[u]:
            cut[u] = ti
    return cut


def qini_curve(
    uplift_score: np.ndarray, treated: np.ndarray, outcome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Radcliffe Qini curve: fraction targeted vs cumulative incremental gain.

    Rows are ranked by descending predicted uplift; at each prefix the gain is
    ``Yt - Yc * Nt/Nc`` (treated positives minus control positives reweighted to
    the treated count). Returns ``(fraction_targeted, cum_gain)`` with a leading
    ``(0, 0)``.
    """
    order = np.argsort(-uplift_score, kind="stable")
    t = treated[order].astype(np.float64)
    y = outcome[order].astype(np.float64)
    n_t = np.cumsum(t)
    n_c = np.cumsum(1.0 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1.0 - t))
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    gain = y_t - y_c * ratio
    n = len(uplift_score)
    frac = np.arange(1, n + 1, dtype=np.float64) / n
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], gain])


def qini_coefficient(
    uplift_score: np.ndarray, treated: np.ndarray, outcome: np.ndarray,
) -> float:
    """Area between the Qini curve and the random-targeting diagonal.

    Positive = the ranking concentrates incremental gains earlier than random;
    ``~0`` = no better than random; negative = worse than random.
    """
    frac, gain = qini_curve(uplift_score, treated, outcome)
    area_model = float(np.sum((frac[1:] - frac[:-1]) * (gain[1:] + gain[:-1]) / 2.0))
    area_random = float(gain[-1]) / 2.0  # trapezoid under the diagonal to the same endpoint
    return area_model - area_random


@dataclass
class UpliftResult:
    """Held-out uplift evaluation: Qini of the learner, the oracle ceiling, ATE."""

    qini: float
    qini_oracle: float
    ate: float
    n_train: int
    n_test: int
    treated_frac: float


def _fit_prob(x: np.ndarray, y: np.ndarray, seed: int, max_iter: int, c: float):
    """Logistic P(y=1|x); a single-class arm degrades to its constant base rate."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        rate = float(y.mean())
        return lambda xt: np.full(xt.shape[0], rate, dtype=np.float64)
    clf = LogisticRegression(max_iter=max_iter, C=c, solver="lbfgs", random_state=seed).fit(x, y)
    return lambda xt: clf.predict_proba(xt)[:, 1]


class UpliftLearner:
    """Embedding uplift meta-learner scored by the Qini coefficient."""

    def __init__(self, seed: int = 0, learner: str = "t", max_iter: int = 1000, c: float = 1.0) -> None:
        if learner not in ("t", "s"):
            raise ValueError("learner must be 't' (two-model) or 's' (single-model)")
        self.seed = seed
        self.learner = learner
        self.max_iter = max_iter
        self.c = c

    def run(
        self, embeddings: dict[str, np.ndarray], rows: UpliftRows, test_size: float = 0.3,
    ) -> UpliftResult:
        """Fit on factual outcomes, score predicted uplift on a user-grouped split."""
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.preprocessing import StandardScaler

        keep = [i for i, u in enumerate(rows.user_id) if u in embeddings]
        if len(keep) < 10:
            raise ValueError(f"only {len(keep)} uplift rows have embeddings; need >= 10")
        keep_arr = np.asarray(keep)
        x = np.stack([embeddings[rows.user_id[i]] for i in keep])
        treated = rows.treated[keep_arr]
        y0 = rows.y0[keep_arr]
        y1 = rows.y1[keep_arr]
        y = np.where(treated == 1, y1, y0)  # factual outcome only
        groups = np.asarray([rows.user_id[i] for i in keep])

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        tr, te = next(splitter.split(x, y, groups))
        scaler = StandardScaler().fit(x[tr])
        xtr, xte = scaler.transform(x[tr]), scaler.transform(x[te])

        if self.learner == "t":
            t_mask = treated[tr] == 1
            p_t = _fit_prob(xtr[t_mask], y[tr][t_mask], self.seed, self.max_iter, self.c)
            p_c = _fit_prob(xtr[~t_mask], y[tr][~t_mask], self.seed, self.max_iter, self.c)
            pred = p_t(xte) - p_c(xte)
        else:  # s-learner: one model with the treatment flag appended
            xtr_s = np.column_stack([xtr, treated[tr].astype(np.float64)])
            fit = _fit_prob(xtr_s, y[tr], self.seed, self.max_iter, self.c)
            ones = np.column_stack([xte, np.ones(len(te))])
            zeros = np.column_stack([xte, np.zeros(len(te))])
            pred = fit(ones) - fit(zeros)

        tau_te = (y1[te] - y0[te]).astype(np.float64)  # true per-row effect (oracle ranking)
        return UpliftResult(
            qini=qini_coefficient(pred, treated[te], y[te]),
            qini_oracle=qini_coefficient(tau_te, treated[te], y[te]),
            ate=float(tau_te.mean()),
            n_train=len(tr),
            n_test=len(te),
            treated_frac=float(treated[te].mean()),
        )
