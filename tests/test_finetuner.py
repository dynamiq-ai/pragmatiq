"""LoRA fine-tuner: early-stopping/restore and custom-head registration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

from pragmatiq import api
from pragmatiq.data.dataset import ShardDataset
from pragmatiq.data.tokenizer import PragmaTokenizer
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

    # Controlled validation curve: peak at epoch 2 (0.92), then two declines →
    # early-stop fires at epoch 4 and the best (0.92) state is what we keep.
    seq = iter([0.90, 0.92, 0.88, 0.86, 0.85, 0.84, 0.83])

    def fake_epoch(dataset, users, label_of, opt, train):  # replaces the bound method
        return 0.0 if train else next(seq)

    monkeypatch.setattr(ft, "_run_epoch", fake_epoch)
    ds = ShardDataset(ft_work / "tok")
    res = ft.fit(ds, ft_work / "raw" / "labels" / "default_12m.parquet")
    ds.close()
    assert res["epochs_run"] == 4  # stopped `patience` epochs after the peak
    assert abs(res["best_val_auc"] - 0.92) < 1e-6
    assert res["n_adapted"] > 0  # LoRA adapters were injected


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
