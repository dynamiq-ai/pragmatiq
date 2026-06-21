"""LoRA fine-tuner.

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
from .pretrainer import resolve_device_count
from .probe import _load_label_table, cutoffs_from_labels

log = logging.getLogger(__name__)


def _stratified_split(
    users: list[str], labels: dict[str, int], val_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Partition ``users`` into ``(train, val)`` sets, stratified by label.

    Splitting each class proportionally keeps both classes in the validation set
    when the data allows. An unstratified shuffle-and-slice can leave the val
    split single-class for rare labels (fraud/aml) — which makes the held-out
    ROC-AUC NaN, so ``best_val_auc`` stays at the ``-1.0`` sentinel and the
    last-epoch (not best) model is silently kept. At least one user per present
    class is held out for val whenever that class has more than one member.
    """
    rng = np.random.default_rng(seed)
    by_cls: dict[int, list[str]] = {}
    for u in users:
        by_cls.setdefault(int(labels[u]), []).append(u)
    val: set[str] = set()
    for members in by_cls.values():
        members = list(members)
        rng.shuffle(members)
        # Cap at len-1 so at least one member of each class stays in train (a
        # high val_fraction on a tiny class must not empty that class from train).
        n_val = min(len(members) - 1, max(1, int(round(len(members) * val_fraction)))) if len(members) > 1 else 0
        val.update(members[:n_val])
    return set(users) - val, val


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
    # Multi-device / multi-node DDP (Fabric), mirroring TrainConfig. devices: a
    # per-node device count or "auto" (all visible GPUs, else a single CPU
    # process); num_nodes: hosts in the job. When the resolved world size is 1
    # the fine-tuner takes the single-process path UNCHANGED (no Fabric); when it
    # is > 1 each rank trains a disjoint, equal-length slice and the validation
    # AUC is gathered across ranks so the early-stop decision is identical
    # everywhere (otherwise the ranks diverge and DDP hangs).
    devices: int | str = "auto"
    num_nodes: int = 1


class LoRAFineTuner:
    """Fine-tunes a frozen backbone with LoRA + a classification head."""

    def __init__(self, model: PragmaModel, config: FineTuneConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        # Resolve the data-parallel world size early so the single-process path
        # (world == 1) can stay byte-identical: it never touches Fabric, while
        # world > 1 routes through the DDP path. resolve_device_count is shared
        # with the pretrainer, so "auto"/numeric-string handling matches.
        world = resolve_device_count(config.devices, torch.cuda.is_available()) * max(1, config.num_nodes)
        self._ddp = world > 1
        self.fabric: Any = None
        if not self._ddp:
            # ---- single-process path: UNCHANGED (no Fabric) ----
            self.model = model.to(device)
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.n_adapted = inject_lora(self.model, rank=config.lora_rank, alpha=config.lora_alpha)
            mark_only_lora_trainable(self.model)
            head_cls = get_head(config.head)
            self.head = head_cls(model.config.dim, n_classes=config.n_classes).to(device)
            return
        # ---- DDP path (world > 1): mirror the pretrainer ----
        from .pretrainer import _make_fabric, seed_everything

        self.fabric = _make_fabric(config.devices, num_nodes=config.num_nodes)
        self.device = str(self.fabric.device)
        # Per-rank seed offset so LoRA init / any sampling draws an independent
        # stream on each rank (the global seed was already applied by the caller;
        # single-process keeps the base seed via the world==1 branch above).
        seed_everything(config.seed + int(self.fabric.global_rank))
        model = model.to(self.fabric.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.n_adapted = inject_lora(model, rank=config.lora_rank, alpha=config.lora_alpha)
        mark_only_lora_trainable(model)
        head_cls = get_head(config.head)
        head = head_cls(model.config.dim, n_classes=config.n_classes)
        # Fabric wraps the modules for DDP; the unwrapped modules are still
        # reachable via `.module` for best-state save/restore (see fit()).
        self.model = self.fabric.setup(model)
        self.head = self.fabric.setup(head)
        # The fine-tuner drives the backbone via `embed_users`, not `forward`.
        # DDP only synchronizes gradients through the registered forward path, so
        # mark `embed_users` (on the Fabric wrapper) as a forward method —
        # otherwise Fabric raises and the backbone's LoRA grads would not
        # all-reduce. (No-op-safe if the wrapper lacks the method on old Fabric.)
        mark = getattr(self.model, "mark_forward_method", None)
        if callable(mark):
            mark("embed_users")

    def _trainable(self):
        yield from (p for p in self.model.parameters() if p.requires_grad)
        yield from self.head.parameters()

    @staticmethod
    def _unwrap(module: Any) -> Any:
        """Return the underlying ``nn.Module`` behind a Fabric/DDP wrapper.

        Fabric's ``setup`` returns a ``_FabricModule`` (which itself wraps a DDP
        ``DistributedDataParallel``); their ``state_dict()`` keys carry wrapper
        prefixes. Best-state save/restore must use the plain module so the saved
        keys match the single-process layout (``module.module`` peels both the
        ``_FabricModule`` and the DDP wrapper). In the single-process path the
        attribute is absent, so the module is returned unchanged.
        """
        # `_FabricModule.module` -> DDP wrapper; DDP `.module` -> the real model.
        for _ in range(2):
            inner = getattr(module, "module", None)
            if inner is None:
                break
            module = inner
        return module

    def fit(self, dataset: ShardDataset, label_path: str | Path) -> dict[str, Any]:
        """Train on a label table; returns best val metrics and the fitted modules.

        Histories are truncated at each user's ``eval_ts`` (when present) so
        the fine-tune never sees the label's outcome window.
        """
        uids, labels, eval_us = _load_label_table(label_path)
        self._cutoffs = cutoffs_from_labels(uids, eval_us)
        have = {u: int(lab) for u, lab in zip(uids, labels) if u in set(dataset.index.order)}
        users = list(have)
        # Stratify the val split by label so rare-positive tasks keep both classes
        # held out (else val ROC-AUC is NaN and the last-epoch model is kept).
        train_users, val_users = _stratified_split(users, have, self.config.val_fraction, self.config.seed)
        label_of = have

        opt = torch.optim.AdamW(self._trainable(), lr=self.config.lr,
                                weight_decay=self.config.weight_decay)
        if self._ddp:
            opt = self.fabric.setup_optimizers(opt)
        # Best-state is saved/restored on the UNWRAPPED modules so the keys match
        # the single-process layout (and are identical on every rank: the LoRA/head
        # weights are DDP-synchronized and the AUC driving the decision is global).
        model_mod, head_mod = self._unwrap(self.model), self._unwrap(self.head)
        best_auc, best_state, bad = -1.0, None, 0
        history = []
        run_epoch = self._run_epoch_ddp if self._ddp else self._run_epoch
        for _epoch in range(self.config.max_epochs):
            run_epoch(dataset, train_users, label_of, opt, train=True)
            val_auc = run_epoch(dataset, val_users, label_of, opt, train=False)
            history.append(val_auc)
            log.info("finetune epoch %d/%d  val_auc %.4f  best %.4f",
                     _epoch + 1, self.config.max_epochs, val_auc, max(best_auc, val_auc))
            # val_auc is identical on every rank (the single-process path computes
            # it locally; the DDP path gathers val logits+labels across ranks before
            # scoring), so the early-stop decision below fires in lockstep — no rank
            # can break while another keeps iterating, which would deadlock the next
            # epoch's all-reduce.
            if val_auc > best_auc + 1e-4:
                best_auc, bad = val_auc, 0
                best_state = {"head": {k: v.detach().clone() for k, v in head_mod.state_dict().items()},
                              "lora": {k: v.detach().clone() for k, v in model_mod.state_dict().items()
                                       if "lora" in k}}
            else:
                bad += 1
                if bad >= self.config.patience:
                    break
        if best_state is not None:
            # Restore the best-validation epoch for BOTH the head and the LoRA
            # adapters, so the model a caller serves (or ``merge_lora``s) matches
            # the reported ``best_val_auc``. The saved LoRA tensors are a partial
            # state dict over the adapter keys only, hence strict=False.
            head_mod.load_state_dict(best_state["head"])
            model_mod.load_state_dict(best_state["lora"], strict=False)
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

    def _run_epoch_ddp(self, dataset: ShardDataset, users: set[str], label_of: dict[str, int],
                       opt: torch.optim.Optimizer, train: bool) -> float:
        """DDP epoch: each rank trains a disjoint, equal-length slice and the
        validation AUC is computed over the FULL val set gathered across ranks.

        Mirrors the pretrainer's data-parallel loop (``set_replica_info`` +
        ``fabric.backward``). Two correctness points keep DDP from hanging:

        * The sampler is restricted to ``users`` via ``subset`` (so every yielded
          batch has ≥1 selected user) and padded to a multiple of the world size,
          so every rank runs the SAME number of ``fabric.backward`` calls — an
          unequal count would deadlock the gradient all-reduce.
        * The validation AUC is computed from logits+labels gathered across ALL
          ranks, so ``val_auc`` (and thus the early-stop decision) is identical on
          every rank. Gathering uneven-length per-rank results uses
          ``all_gather_object`` (no padding needed); duplicates introduced by the
          sampler's replica padding are removed by de-duplicating on ``user_id``,
          so the gathered val AUC matches the single-process AUC over the same set.
        """
        import torch.distributed as dist
        from sklearn.metrics import roc_auc_score

        self.model.train(train)
        self.head.train(train)
        # Restrict batching to this split's users (by index position) so no batch
        # is empty after filtering — the per-batch backward count is then governed
        # solely by the sampler's replica padding (equal across ranks).
        pos_of = {u: i for i, u in enumerate(dataset.index.order)}
        subset = sorted(pos_of[u] for u in users if u in pos_of)
        sampler = DynamicBatchSampler(dataset.index, token_budget=self.config.token_budget,
                                      shuffle=train, seed=self.config.seed, subset=subset)
        sampler.set_replica_info(int(self.fabric.world_size), int(self.fabric.global_rank))
        sampler.set_epoch(0)
        cutoffs = getattr(self, "_cutoffs", None)
        collator = TruncatingCollator(cutoffs) if cutoffs else None
        loader = ShardDataLoader(dataset, sampler, collator=collator)
        local_probs: list[float] = []
        local_ys: list[int] = []
        local_uids: list[str] = []
        for batch in loader:
            idx = [i for i, u in enumerate(batch.user_ids) if u in users]
            if not idx:
                # With a subset over `users`, batches always carry a selected user;
                # this guard is defensive only and never skips under DDP (skipping
                # would desynchronize the per-rank backward count and deadlock).
                continue
            batch = batch.to(self.fabric.device)
            with torch.set_grad_enabled(train):
                z = self.model.embed_users(batch)
                logits = self.head(z)
                sel = torch.tensor(idx, device=self.fabric.device)
                y = torch.tensor([label_of[batch.user_ids[i]] for i in idx], device=self.fabric.device)
                loss = torch.nn.functional.cross_entropy(logits[sel], y)
                if train:
                    opt.zero_grad(set_to_none=True)
                    self.fabric.backward(loss)
                    torch.nn.utils.clip_grad_norm_(list(self._trainable()), 1.0)
                    opt.step()
                local_probs.extend(torch.softmax(logits[sel], -1)[:, 1].detach().cpu().tolist())
                local_ys.extend(y.cpu().tolist())
                local_uids.extend(batch.user_ids[i] for i in idx)
        if train:
            return float("nan")
        # Gather every rank's (user_id, prob, label) so all ranks score ONE global
        # AUC -> identical early-stop decision everywhere. all_gather_object handles
        # the uneven per-rank lengths without manual padding.
        world = int(self.fabric.world_size)
        bucket: list[Any] = [None] * world
        dist.all_gather_object(bucket, list(zip(local_uids, local_probs, local_ys)))
        # De-duplicate on user_id: the sampler's replica padding can repeat a few
        # batches, so a user may appear on >1 rank; keep one entry per user so the
        # global AUC equals the single-process AUC over the same val users.
        seen: dict[str, tuple[float, int]] = {}
        for shard in bucket:
            for uid, prob, label in shard:  # type: ignore[union-attr]
                seen.setdefault(uid, (prob, label))
        if not seen:
            return float("nan")
        probs = [p for p, _ in seen.values()]
        ys = [yy for _, yy in seen.values()]
        if len(set(ys)) > 1:
            return float(roc_auc_score(ys, probs))
        return float("nan")
