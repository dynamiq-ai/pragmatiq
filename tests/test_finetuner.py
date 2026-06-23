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

    def fake_epoch(dataset, users, label_of, opt, train, epoch=0):  # replaces the bound method
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

    def fake_epoch(dataset, users, label_of, opt, train, epoch=0):  # replaces the bound method
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


def test_run_epoch_reshuffles_per_epoch(ft_work: Path) -> None:
    """_run_epoch must produce a different batch order each epoch (not hardcoded 0).

    DynamicBatchSampler uses (seed, epoch) as the RNG key, so set_epoch(0) every
    call yields the same shuffle, while set_epoch(N) for increasing N changes it.
    This test catches the regression of hardcoding set_epoch(0).

    Determinism guarantee: running with the same epoch index twice must yield the
    SAME order (so resume / repro still hold).
    """
    from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataset
    from pragmatiq.data.tokenizer import PragmaTokenizer

    ds = ShardDataset(ft_work / "tok")
    try:
        tok = PragmaTokenizer.load(ft_work / "tok" / "tokenizer")

        # Collect the batch-user-id sequences produced by _run_epoch for epochs 0, 1, 2
        # by instrumenting DynamicBatchSampler.set_epoch to record the epoch it receives.
        epochs_seen: list[int] = []
        real_set_epoch = DynamicBatchSampler.set_epoch

        def spy_set_epoch(self: DynamicBatchSampler, epoch: int) -> None:
            epochs_seen.append(epoch)
            real_set_epoch(self, epoch)

        # Patch the sampler's class method; ft._run_epoch creates a fresh sampler per call
        import pragmatiq.data.dataset as _ds_mod
        orig = _ds_mod.DynamicBatchSampler.set_epoch
        _ds_mod.DynamicBatchSampler.set_epoch = spy_set_epoch  # type: ignore[method-assign]
        try:
            sampler0 = DynamicBatchSampler(ds.index, token_budget=8192, shuffle=True, seed=7)
            sampler0.set_epoch(0)
            order_epoch0_run1 = [list(b) for b in sampler0]

            sampler0b = DynamicBatchSampler(ds.index, token_budget=8192, shuffle=True, seed=7)
            sampler0b.set_epoch(0)
            order_epoch0_run2 = [list(b) for b in sampler0b]

            sampler1 = DynamicBatchSampler(ds.index, token_budget=8192, shuffle=True, seed=7)
            sampler1.set_epoch(1)
            order_epoch1 = [list(b) for b in sampler1]

            sampler2 = DynamicBatchSampler(ds.index, token_budget=8192, shuffle=True, seed=7)
            sampler2.set_epoch(2)
            order_epoch2 = [list(b) for b in sampler2]
        finally:
            _ds_mod.DynamicBatchSampler.set_epoch = orig  # type: ignore[method-assign]

        # Determinism: same epoch index → same order
        assert order_epoch0_run1 == order_epoch0_run2, (
            "set_epoch(0) is not deterministic — RNG must be seeded by (seed, epoch)"
        )
        # Epoch N vs N+1 must differ (the fix: pass the actual epoch number)
        assert order_epoch0_run1 != order_epoch1, (
            "epoch 0 and epoch 1 produced identical batch orders — "
            "set_epoch is likely hardcoded to 0 (DDP reshuffle bug)"
        )
        assert order_epoch1 != order_epoch2, (
            "epoch 1 and epoch 2 produced identical batch orders — "
            "set_epoch is likely hardcoded to a constant"
        )

        # Verify _run_epoch passes the actual epoch number by running fit with a
        # monkeypatched epoch recorder.
        ft2 = LoRAFineTuner(
            PragmaModel(ModelConfig.preset("nano", tok.vocab_size)),
            FineTuneConfig(max_epochs=3, patience=10, lora_rank=4, seed=7),
        )
        epochs_passed: list[int] = []
        real_run_epoch = ft2._run_epoch

        def recording_run_epoch(dataset, users, label_of, opt, train, epoch=0):
            epochs_passed.append(epoch)
            return real_run_epoch(dataset, users, label_of, opt, train, epoch)

        import unittest.mock as _mock
        with _mock.patch.object(ft2, "_run_epoch", recording_run_epoch):
            ft2.fit(ds, ft_work / "raw" / "labels" / "default_12m.parquet")

        # fit calls _run_epoch twice per outer epoch (train=True + train=False)
        # so epochs_passed contains [0, 0, 1, 1, 2, 2, ...] or truncated by early-stop.
        train_epochs = epochs_passed[::2]  # every other call is the train pass
        assert train_epochs == list(range(len(train_epochs))), (
            f"_run_epoch must receive epoch=0, 1, 2, ... got train epochs: {train_epochs}"
        )
    finally:
        ds.close()
