"""LoRA fine-tuner: early-stopping/restore and custom-head registration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pragmatiq import api
from pragmatiq.data.dataset import ShardDataset
from pragmatiq.data.tokenizer import PragmaTokenizer
from pragmatiq.models.lora import LoRALinear
from pragmatiq.models.pragmatiq import ModelConfig, PragmaModel
from pragmatiq.registry import register_head
from pragmatiq.training.finetuner import FineTuneConfig, LoRAFineTuner


@pytest.fixture(scope="module")
def ft_work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("ft")
    api.synthesize({"n_users": 90, "months": 14, "n_merchants": 500, "mule_ring_count": 1,
                    "seed": 4, "eval_month_credit": 2, "eval_month_short": 8},
                   out=work / "raw", n_workers=0, write_report=False)
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 3000, "n_buckets": 16, "categorical_threshold": 150})
    return work


def test_finetune_early_stops_and_restores_best(ft_work: Path, monkeypatch) -> None:
    tok = PragmaTokenizer.load(ft_work / "tok" / "tokenizer")
    model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size))
    ft = LoRAFineTuner(model, FineTuneConfig(max_epochs=20, patience=2, lora_rank=4))

    # A LoRA adapter tensor we stamp with the epoch index each training epoch, so
    # we can assert the fitted model carries the BEST epoch's adapters — not the
    # last (degraded) epoch's — after early stopping restores the best state.
    lora_param = next(p for name, p in ft.model.named_parameters() if "lora" in name)

    # Controlled validation curve: peak at epoch 2 (0.92), then two declines →
    # early-stop fires at epoch 4 and the best (0.92) state is what we keep.
    seq = iter([0.90, 0.92, 0.88, 0.86, 0.85, 0.84, 0.83])
    state = {"epoch": 0}

    def fake_epoch(dataset, users, label_of, opt, train):  # replaces the bound method
        if train:
            state["epoch"] += 1
            with torch.no_grad():
                lora_param.fill_(float(state["epoch"]))
            return 0.0
        return next(seq)

    monkeypatch.setattr(ft, "_run_epoch", fake_epoch)
    ds = ShardDataset(ft_work / "tok")
    res = ft.fit(ds, ft_work / "raw" / "labels" / "default_12m.parquet")
    ds.close()
    assert res["epochs_run"] == 4  # stopped `patience` epochs after the peak
    assert abs(res["best_val_auc"] - 0.92) < 1e-6
    assert res["n_adapted"] > 0  # LoRA adapters were injected
    # the peak val_auc was at epoch 2, so the restored adapters must be epoch 2's
    assert torch.allclose(lora_param, torch.full_like(lora_param, 2.0)), \
        "fine-tuner must restore the best-epoch LoRA adapters, not the last epoch's"


def test_finetune_restores_best_lora_adapters(ft_work: Path, monkeypatch) -> None:
    tok = PragmaTokenizer.load(ft_work / "tok" / "tokenizer")
    model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size))
    ft = LoRAFineTuner(model, FineTuneConfig(max_epochs=3, patience=2, lora_rank=4))
    seq = iter([0.80, 0.70, 0.60])
    snapshots: list[dict[str, torch.Tensor]] = []
    head_snapshots: list[dict[str, torch.Tensor]] = []

    def fake_epoch(dataset, users, label_of, opt, train):  # replaces the bound method
        if train:
            with torch.no_grad():
                for name, param in ft.model.named_parameters():
                    if "lora" in name:
                        param.add_(len(snapshots) + 1.0)
                for param in ft.head.parameters():
                    param.add_(len(head_snapshots) + 1.0)
            snapshots.append({
                k: v.detach().clone()
                for k, v in ft.model.state_dict().items()
                if "lora" in k
            })
            head_snapshots.append({k: v.detach().clone() for k, v in ft.head.state_dict().items()})
            return 0.0
        return next(seq)

    monkeypatch.setattr(ft, "_run_epoch", fake_epoch)
    ds = ShardDataset(ft_work / "tok")
    ft.fit(ds, ft_work / "raw" / "labels" / "default_12m.parquet")
    ds.close()
    current = {k: v.detach() for k, v in ft.model.state_dict().items() if "lora" in k}
    assert snapshots, "test setup did not mutate LoRA adapters"
    for name, expected in snapshots[0].items():
        assert torch.equal(current[name], expected), f"{name} was not restored to the best epoch"
    for name, expected in head_snapshots[0].items():
        assert torch.equal(ft.head.state_dict()[name], expected), f"{name} head was not restored"


def test_lora_merge_preserves_base_dtype_and_device() -> None:
    base = nn.Linear(5, 3).to(dtype=torch.bfloat16)
    layer = LoRALinear(base, rank=2, alpha=2.0)
    with torch.no_grad():
        layer.lora_b.fill_(0.25)
    merged = layer.merged_linear()
    assert merged.weight.dtype == base.weight.dtype
    assert merged.weight.device == base.weight.device
    if merged.bias is not None:
        assert merged.bias.dtype == base.weight.dtype
        assert merged.bias.device == base.weight.device


def test_custom_head_resolved_for_finetune() -> None:
    @register_head("ranking_test")
    class RankingTestHead(nn.Module):
        def __init__(self, dim: int, n_classes: int = 2) -> None:
            super().__init__()
            self.net = nn.Linear(dim, n_classes)

        def forward(self, x):  # noqa: ANN001
            return self.net(x)

    model = PragmaModel(ModelConfig.preset("nano", 1500))
    ft = LoRAFineTuner(model, FineTuneConfig(head="ranking_test", n_classes=3, lora_rank=4))
    assert isinstance(ft.head, RankingTestHead)
    assert ft.head.net.out_features == 3  # head built at the configured n_classes


def test_stratified_val_split_keeps_both_classes_for_rare_labels() -> None:
    """A rare-positive label table must yield a val split with both classes.

    An unstratified shuffle+slice can put all (or zero) positives on one side,
    leaving the held-out val single-class -> roc_auc is NaN -> best_val_auc stays
    -1.0 and the LAST-epoch (not best) model is returned. Stratifying prevents it.
    """
    from pragmatiq.training.finetuner import _stratified_split

    # 40 users, only 4 positives: a 25% unstratified val could easily miss all 4.
    labels = {f"u{i}": (1 if i < 4 else 0) for i in range(40)}
    users = list(labels)
    train, val = _stratified_split(users, labels, val_fraction=0.25, seed=0)
    assert sum(labels[u] for u in val) >= 1, "val split has no positives (not stratified)"
    assert any(labels[u] == 0 for u in val), "val split has no negatives"
    assert train.isdisjoint(val) and (train | val) == set(users), "split must partition users"


def test_stratified_split_never_empties_a_train_class() -> None:
    """A small class with a high val_fraction must still leave a member in train,
    or the backbone never sees that class during fine-tuning (n_val capped at len-1)."""
    from pragmatiq.training.finetuner import _stratified_split

    labels = {f"u{i}": (1 if i < 2 else 0) for i in range(10)}  # only 2 positives
    train, val = _stratified_split(list(labels), labels, val_fraction=0.8, seed=0)
    assert sum(labels[u] for u in train) >= 1, "train lost its only positives"
    assert sum(labels[u] for u in val) >= 1, "val lost its only positives"
    assert any(labels[u] == 0 for u in train), "train lost its negatives"
