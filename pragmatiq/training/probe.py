"""Frozen-embedding probe + embedding helpers.

``embed_users`` runs a (frozen) model over a shard dataset and returns user
embeddings keyed by user_id. ``EmbeddingProbe`` fits a classifier on those
embeddings against a label table — the cheap, backbone-frozen evaluation of
embedding quality. The default classifier is gradient boosting (sklearn
``HistGradientBoostingClassifier``), which models the non-linear interactions in a
learned embedding better than a linear head; ``logistic`` stays selectable, and
``lightgbm`` is also selectable. Both ROC-AUC and PR-AUC
are reported — PR-AUC is the honest headline on the low-prevalence risk tasks. The
raw-count baseline uses the *same* classifier, so the probe-vs-baseline gap reflects
the representation, not the model class.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..data.collate import TruncatingCollator, VarlenCollator, run_with_oom_retry
from ..data.dataset import DynamicBatchSampler, ShardDataset
from ..data.tokenizer import truncate_record
from ..models.pragmatiq import PragmaModel
from ..progress import progress


@contextlib.contextmanager
def cpu_thread_cap(n: int | None = None) -> Iterator[None]:
    """Pin torch CPU intra-op threads, restoring the caller's setting on exit.

    The default of one thread per core oversubscribes the small forward passes
    used for embedding on many-core hosts, and when several stages embed at once
    it drives thread-pool contention. A small fixed count is faster there and
    fixes the float reduction order so per-batch outputs are stable; the caller's
    setting is restored so notebook and library users are unaffected elsewhere.
    """
    prev = torch.get_num_threads()
    torch.set_num_threads(n or min(8, os.cpu_count() or 8))
    try:
        yield
    finally:
        torch.set_num_threads(prev)


@torch.no_grad()
def embed_users(
    model: PragmaModel,
    dataset: ShardDataset,
    token_budget: int = 16_384,
    device: str | torch.device = "cpu",
    user_ids: list[str] | None = None,
    cutoffs: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Compute ``z_h[USR]`` embeddings for users; returns ``{user_id: vector}``.

    ``cutoffs`` (user_id -> µs) truncates each user's history at their label
    eval point before encoding, so task embeddings never see the outcome
    window (the no-hindcasting rule).
    """
    model = model.to(device).eval()
    # Restrict the forward pass to the requested users (by their position in the
    # index) so only that cohort is encoded; None embeds everyone. Users not in
    # the index are simply absent from the result.
    subset: list[int] | None = None
    if user_ids is not None:
        pos = {u: i for i, u in enumerate(dataset.index.order)}
        subset = [pos[u] for u in user_ids if u in pos]
    sampler = DynamicBatchSampler(dataset.index, token_budget=token_budget, shuffle=False,
                                  subset=subset)
    sampler.set_epoch(0)
    collator = TruncatingCollator(cutoffs) if cutoffs else VarlenCollator()
    order = dataset.index.order

    def _embed_chunk(chunk: list[str]) -> np.ndarray:
        batch = collator(dataset.get_many(chunk)).to(device)
        return model.embed_users(batch).float().cpu().numpy()

    out: dict[str, np.ndarray] = {}
    with cpu_thread_cap():
        for batch_idx in progress(sampler, total=len(sampler), desc="embed", unit="batch"):
            uids = [order[i] for i in batch_idx]

            def _embed_at(budget: int, uids: list[str] = uids) -> np.ndarray:
                # On CUDA OOM run_with_oom_retry halves `budget`; smaller budget → more
                # sub-batches of this batch's users, so a heavy history still fits.
                n_groups = max(1, (token_budget + budget - 1) // budget)
                size = max(1, (len(uids) + n_groups - 1) // n_groups)
                parts = [_embed_chunk(uids[s:s + size]) for s in range(0, len(uids), size)]
                return np.concatenate(parts, axis=0)

            z, _ = run_with_oom_retry(_embed_at, token_budget)
            for i, uid in enumerate(uids):
                out[uid] = z[i]
    return out


def _load_label_table(
    path: str | Path, label_col: str = "label",
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """Read a label table; returns (user_ids, labels, eval_ts µs or None).

    ``eval_ts`` is the per-user label eval point — downstream consumers use it
    to truncate histories so embeddings never include the outcome window.
    """
    table = pq.read_table(path)
    uids = table.column("user_id").to_pylist()
    labels = table.column(label_col).to_numpy()
    eval_us: np.ndarray | None = None
    if "eval_ts" in table.column_names:
        eval_us = table.column("eval_ts").cast(pa.int64()).to_numpy()
    return uids, labels, eval_us


def cutoffs_from_labels(uids: list[str], eval_us: np.ndarray | None) -> dict[str, int] | None:
    """Build the per-user truncation map from a label table's eval_ts column."""
    if eval_us is None:
        return None
    return {u: int(t) for u, t in zip(uids, eval_us)}


@dataclass
class ProbeResult:
    """Outcome of a probe: held-out ROC-AUC, PR-AUC, accuracy and sizes."""

    auc: float
    pr_auc: float
    accuracy: float
    n_train: int
    n_test: int
    prevalence: float


def build_probe_classifier(model: str, seed: int, max_iter: int = 1000, C: float = 1.0
                           ) -> tuple[Any, bool]:
    """Return ``(estimator, needs_scaling)`` for a probe/baseline classifier.

    ``gbdt`` (default) is sklearn's ``HistGradientBoostingClassifier``; ``logistic``
    is ``LogisticRegression`` (lbfgs); ``lightgbm`` is also selectable. Tree models
    are scale-invariant, so only the linear model asks for standardization.
    """
    if model == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed), False
    if model == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, verbosity=-1), False
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"), True
    raise ValueError(f"unknown probe_model {model!r}; choose 'gbdt', 'logistic', or 'lightgbm'")


def _fit_score(clf: Any, needs_scaling: bool, Xtr: np.ndarray, ytr: np.ndarray,
               Xte: np.ndarray, yte: np.ndarray) -> tuple[float, float, float]:
    """Fit ``clf`` (scaling the features only when asked) and return ROC-AUC, PR-AUC, accuracy."""
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if needs_scaling:
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    both = len(np.unique(yte)) > 1
    auc = float(roc_auc_score(yte, proba)) if both else float("nan")
    pr_auc = float(average_precision_score(yte, proba)) if both else float("nan")
    return auc, pr_auc, float(accuracy_score(yte, pred))


class EmbeddingProbe:
    """Frozen-embedding probe; ``model`` selects gbdt (default) / logistic / lightgbm."""

    def __init__(self, model: str = "gbdt", seed: int = 0, max_iter: int = 1000, C: float = 1.0) -> None:
        self.model = model
        self.seed = seed
        self.max_iter = max_iter
        self.C = C

    def run(
        self,
        embeddings: dict[str, np.ndarray],
        label_path: str | Path,
        test_size: float = 0.3,
    ) -> ProbeResult:
        """Fit/evaluate on the intersection of ``embeddings`` and a label table."""
        from sklearn.model_selection import train_test_split

        uids, labels, _ = _load_label_table(label_path)
        rows: list[np.ndarray] = []
        ylist: list[int] = []
        for uid, lab in zip(uids, labels):
            if uid in embeddings:
                rows.append(embeddings[uid])
                ylist.append(int(lab))
        if len(rows) < 10:
            raise ValueError(f"only {len(rows)} labeled users have embeddings; need >= 10")
        X = np.stack(rows)
        y = np.asarray(ylist)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=self.seed,
                                              stratify=y if len(np.unique(y)) > 1 else None)
        clf, needs_scaling = build_probe_classifier(self.model, self.seed, self.max_iter, self.C)
        auc, pr_auc, acc = _fit_score(clf, needs_scaling, Xtr, ytr, Xte, yte)
        return ProbeResult(auc=auc, pr_auc=pr_auc, accuracy=acc,
                           n_train=len(ytr), n_test=len(yte), prevalence=float(y.mean()))


class RawCountBaseline:
    """Hand-made raw event-count features + the probe's classifier (the baseline floor).

    Uses the same ``model`` as :class:`EmbeddingProbe` so the probe-vs-baseline gap
    isolates the representation (learned embedding vs hand-crafted counts), not the
    classifier family.
    """

    def __init__(self, seed: int = 0, model: str = "gbdt") -> None:
        self.seed = seed
        self.model = model

    def features(self, dataset: ShardDataset, user_ids: list[str],
                 cutoffs: dict[str, int] | None = None) -> np.ndarray:
        """Per-user [n_events, n_tokens, n_prof_tokens, log1p(n_events)] features.

        With ``cutoffs``, counts come from histories truncated at each user's
        eval point — the baseline must obey the same no-hindcasting rule as
        the embeddings it is compared against.
        """
        if cutoffs:
            feats = []
            for uid in progress(user_ids, total=len(user_ids),
                                desc="baseline features (truncated)", unit="user"):
                rec = dataset.get(uid)
                if uid in cutoffs:
                    rec = truncate_record(rec, cutoffs[uid])
                feats.append([rec.n_events, rec.n_tokens, int(rec.prof_key_ids.size),
                              np.log1p(rec.n_events)])
            return np.asarray(feats, dtype=np.float64)
        idx = dataset.index
        pos = {u: i for i, u in enumerate(idx.order)}
        feats = []
        for uid in user_ids:
            i = pos[uid]
            feats.append([idx.n_events[i], idx.n_tokens[i], idx.n_prof_tokens[i],
                          np.log1p(idx.n_events[i])])
        return np.asarray(feats, dtype=np.float64)

    def run(self, dataset: ShardDataset, label_path: str | Path, test_size: float = 0.3) -> ProbeResult:
        """Fit/evaluate the raw-count baseline on a label table."""
        from sklearn.model_selection import train_test_split

        uids, labels, eval_us = _load_label_table(label_path)
        cutoffs = cutoffs_from_labels(uids, eval_us)
        have = set(dataset.index.order)
        keep = [(u, int(lab)) for u, lab in zip(uids, labels) if u in have]
        users = [u for u, _ in keep]
        y = np.array([lab for _, lab in keep])
        X = self.features(dataset, users, cutoffs=cutoffs)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=self.seed,
                                              stratify=y if len(np.unique(y)) > 1 else None)
        clf, needs_scaling = build_probe_classifier(self.model, self.seed)
        auc, pr_auc, acc = _fit_score(clf, needs_scaling, Xtr, ytr, Xte, yte)
        return ProbeResult(auc=auc, pr_auc=pr_auc, accuracy=acc,
                           n_train=len(ytr), n_test=len(yte), prevalence=float(y.mean()))
