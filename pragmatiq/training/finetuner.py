"""LoRA fine-tuner (Phase 5).

Freezes the pretrained backbone, injects LoRA adapters, attaches a
:class:`ClassificationHead` on the user embedding ``z_h[USR]``, and trains with
early stopping on a held-out split. The backbone weights never move; only LoRA
A/B and the head are updated, so a downstream task is cheap to fit and ship
(``merge_lora`` folds the adapter back for export).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.collate import TruncatingCollator
from ..data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
from ..models.heads import ClassificationHead  # noqa: F401  (registers @register_head)
from ..models.lora import inject_lora, mark_only_lora_trainable
from ..models.pragmatiq import PragmaModel
from ..registry import get_head
from .probe import _load_label_table, cutoffs_from_labels

log = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Hyperparameters for LoRA fine-tuning."""

    lora_rank: int = 8
    lora_alpha: float = 8.0
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 20
    patience: int = 3  # early-stopping patience (epochs without val improvement)
    token_budget: int = 16_384
    n_classes: int = 2
    seed: int = 0
    val_fraction: float = 0.2
    # Task head, resolved from the registry by name (rule 8) so configs can
    # swap in a custom @register_head without forking the fine-tuner.
    head: str = "classification"


class LoRAFineTuner:
    """Fine-tunes a frozen backbone with LoRA + a classification head."""

    def __init__(self, model: PragmaModel, config: FineTuneConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self.model = model.to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_adapted = inject_lora(self.model, rank=config.lora_rank, alpha=config.lora_alpha)
        mark_only_lora_trainable(self.model)
        head_cls = get_head(config.head)
        self.head = head_cls(model.config.dim, n_classes=config.n_classes).to(device)

    def _trainable(self):
        yield from (p for p in self.model.parameters() if p.requires_grad)
        yield from self.head.parameters()

    def fit(self, dataset: ShardDataset, label_path: str | Path) -> dict[str, Any]:
        """Train on a label table; returns best val metrics and the fitted modules.

        Histories are truncated at each user's ``eval_ts`` (when present) so
        the fine-tune never sees the label's outcome window.
        """
        uids, labels, eval_us = _load_label_table(label_path)
        self._cutoffs = cutoffs_from_labels(uids, eval_us)
        have = {u: int(lab) for u, lab in zip(uids, labels) if u in set(dataset.index.order)}
        users = list(have)
        rng = np.random.default_rng(self.config.seed)
        rng.shuffle(users)
        n_val = max(1, int(len(users) * self.config.val_fraction))
        val_users, train_users = set(users[:n_val]), set(users[n_val:])
        label_of = have

        opt = torch.optim.AdamW(self._trainable(), lr=self.config.lr,
                                weight_decay=self.config.weight_decay)
        best_auc, best_state, bad = -1.0, None, 0
        history = []
        for _epoch in range(self.config.max_epochs):
            self._run_epoch(dataset, train_users, label_of, opt, train=True)
            val_auc = self._run_epoch(dataset, val_users, label_of, opt, train=False)
            history.append(val_auc)
            log.info("finetune epoch %d/%d  val_auc %.4f  best %.4f",
                     _epoch + 1, self.config.max_epochs, val_auc, max(best_auc, val_auc))
            if val_auc > best_auc + 1e-4:
                best_auc, bad = val_auc, 0
                best_state = {"head": self.head.state_dict(),
                              "lora": {k: v.detach().clone() for k, v in self.model.state_dict().items()
                                       if "lora" in k}}
            else:
                bad += 1
                if bad >= self.config.patience:
                    break
        if best_state is not None:
            self.head.load_state_dict(best_state["head"])
        return {"best_val_auc": best_auc, "epochs_run": len(history), "n_adapted": self.n_adapted,
                "val_auc_history": history}

    def _run_epoch(self, dataset: ShardDataset, users: set[str], label_of: dict[str, int],
                   opt: torch.optim.Optimizer, train: bool) -> float:
        from sklearn.metrics import roc_auc_score

        self.model.train(train)
        self.head.train(train)
        sampler = DynamicBatchSampler(dataset.index, token_budget=self.config.token_budget,
                                      shuffle=train, seed=self.config.seed)
        sampler.set_epoch(0)
        cutoffs = getattr(self, "_cutoffs", None)
        collator = TruncatingCollator(cutoffs) if cutoffs else None
        loader = ShardDataLoader(dataset, sampler, collator=collator)
        probs, ys = [], []
        for batch in loader:
            idx = [i for i, u in enumerate(batch.user_ids) if u in users]
            if not idx:
                continue
            batch = batch.to(self.device)
            with torch.set_grad_enabled(train):
                z = self.model.embed_users(batch)
                logits = self.head(z)
                sel = torch.tensor(idx, device=self.device)
                y = torch.tensor([label_of[batch.user_ids[i]] for i in idx], device=self.device)
                loss = torch.nn.functional.cross_entropy(logits[sel], y)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self._trainable()), 1.0)
                    opt.step()
                probs.extend(torch.softmax(logits[sel], -1)[:, 1].detach().cpu().tolist())
                ys.extend(y.cpu().tolist())
        if not train and len(set(ys)) > 1:
            return float(roc_auc_score(ys, probs))
        return float("nan")
