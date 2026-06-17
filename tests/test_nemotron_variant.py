"""PRAGMA+Nemotron variant: text fields embedded by a frozen encoder, masked text
tokens reconstructed with MSE alongside the cross-entropy MLM loss.

CI uses the dependency-free ``hash`` stand-in encoder so the whole path — embed-mode
tokenization, the text input projection, the MSE branch, and masking that routes text
to reconstruction — is exercised on CPU without downloading a multi-GB model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pragmatiq import api
from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
from pragmatiq.data.tokenizer import MASK, PragmaTokenizer, iter_user_records
from pragmatiq.experiments.run import Run
from pragmatiq.experiments.tracking import MetricLogger
from pragmatiq.models.pragmatiq import ModelConfig, PragmaModel
from pragmatiq.registry import register_text_encoder
from pragmatiq.training.masking import MaskingStrategy
from pragmatiq.training.pretrainer import PreTrainer, TrainConfig, seed_everything

TEXT_DIM = 64
TEXT_OVERRIDES = {"text_encoder": "hash", "text_encoder_dim": TEXT_DIM}


@register_text_encoder("_fixed7_stub")
class _Fixed7Encoder:
    """Stand-in for a Nemotron-style encoder: no ``dim`` constructor arg, fixed width."""

    def __init__(self, model_name: str = "stub") -> None:
        self.dim = 7

    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts), self.dim, dtype=torch.float32)


@pytest.fixture(scope="module")
def embed_work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("nemo")
    api.synthesize({"n_users": 250, "months": 14, "n_merchants": 700, "mule_ring_count": 1,
                    "seed": 3, "eval_month_credit": 2, "eval_month_short": 8},
                   out=work / "raw", n_workers=0, write_report=False)
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 3500, "n_buckets": 32, "categorical_threshold": 200,
                         "text_value_mode": "embed", "text_encoder": "hash",
                         "text_encoder_dim": TEXT_DIM})
    return work


def _loader(work: Path) -> tuple[ShardDataset, ShardDataLoader]:
    ds = ShardDataset(work / "tok")
    loader = ShardDataLoader(ds, DynamicBatchSampler(ds.index, token_budget=4096, seed=0))
    return ds, loader


class TestNemotronVariant:
    def test_text_fields_present_in_shards(self, embed_work: Path) -> None:
        tok = PragmaTokenizer.load(embed_work / "tok" / "tokenizer")
        assert any(v == "text" for v in tok.field_kind.values())  # variant is meaningful
        ds, loader = _loader(embed_work)
        try:
            assert any(b.is_text.any() for b in loader)  # text markers survive to the batch
        finally:
            ds.close()

    def test_masker_routes_text_to_mse(self, embed_work: Path) -> None:
        ds, loader = _loader(embed_work)
        try:
            batch = next(b for b in loader if b.is_text.any())
            masked = MaskingStrategy()(batch, torch.Generator().manual_seed(0))
            is_text = batch.is_text.bool()
            # text tokens are reconstructed via MSE, never given a cross-entropy label
            assert int((masked.labels[is_text] != -100).sum()) == 0
            assert masked.text_loss_idx.numel() > 0  # some text token was masked
            assert bool(is_text[masked.text_loss_idx].all())  # MSE targets are all text
            assert bool((masked.input_value_ids[masked.text_loss_idx] == MASK).all())  # hidden
            # fed text tokens are a subset of all text tokens, and exclude masked ones
            assert bool((masked.feed_text <= is_text).all())
            assert not bool(masked.feed_text[masked.text_loss_idx].any())  # masked → not fed
        finally:
            ds.close()

    def test_model_sizes_text_branch_from_encoder(self) -> None:
        # The encoder's own width is authoritative: a configured hint (999) is ignored,
        # and a Nemotron-style encoder with no `dim` constructor arg must build cleanly
        # (build_text_encoder drops kwargs the constructor does not accept).
        model = PragmaModel(ModelConfig.preset(
            "nano", 500, overrides={"text_encoder": "_fixed7_stub", "text_encoder_dim": 999}))
        assert model.text_encoder is not None and model.text_encoder.dim == 7
        assert model.text_proj is not None and model.text_proj.in_features == 7
        assert model.config.text_encoder_dim == 7  # recorded for a consistent checkpoint

    def test_default_model_ignores_text_state(self, embed_work: Path) -> None:
        tok = PragmaTokenizer.load(embed_work / "tok" / "tokenizer")
        ds, loader = _loader(embed_work)
        try:
            batch = next(b for b in loader if b.is_text.any())
            model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size))  # BPE path
            assert model.text_encoder is None and model.text_proj is None
            out = model(batch)  # text columns present but the default path never reads them
            assert out.text_vecs is None and out.text_token_idx is None
        finally:
            ds.close()

    def test_variant_trains_text_branch(self, embed_work: Path) -> None:
        tok = PragmaTokenizer.load(embed_work / "tok" / "tokenizer")
        run = Run.create("nemo", {}, 0, tok.content_hash, embed_work / "runs",
                         tokenizer_src=embed_work / "tok" / "tokenizer")
        cfg = TrainConfig(max_steps=30, token_budget=4096, warmup_steps=5, seed=0,
                          checkpoint_every_min=1000.0, log_every=1)
        seed_everything(cfg.seed)
        model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size, overrides=TEXT_OVERRIDES))
        assert model.text_encoder is not None and model.text_proj is not None
        trainer = PreTrainer(model, run, cfg, tok.content_hash, logger=MetricLogger(run.dir))
        assert trainer.head.text_out is not None  # MSE reconstruction head built
        ds, loader = _loader(embed_work)
        proj0 = model.text_proj.weight.detach().clone()
        head0 = trainer.head.text_out.weight.detach().clone()
        try:
            saw_text_mse = False
            steps = 0
            for batch in loader:
                metrics = trainer._train_step([batch])  # one micro-batch per step
                trainer.step += 1
                steps += 1
                if metrics is not None:
                    assert np.isfinite(metrics["loss"])
                    if "loss_text_mse" in metrics:
                        saw_text_mse = True
                        assert np.isfinite(metrics["loss_text_mse"])
                if steps >= 30:
                    break
        finally:
            ds.close()
        assert saw_text_mse, "no text token was masked across the run"
        # the text input projection and the MSE head both moved → both are in the loss
        # graph and the optimizer (a frozen/disconnected branch would not update)
        assert not torch.allclose(proj0, model.text_proj.weight)
        assert not torch.allclose(head0, trainer.head.text_out.weight)

    def test_end_to_end_hands_off_pipeline(self, embed_work: Path) -> None:
        # pretrain() auto-derives the text encoder from the embed-mode tokenizer with
        # no extra flags, then the model round-trips through save/load and embeds.
        summary = api.pretrain(
            embed_work / "tok", "e2e", model_size="nano",
            config={"max_steps": 20, "token_budget": 4096, "warmup_steps": 3, "seed": 0,
                    "checkpoint_every_min": 1000.0},
            runs_root=str(embed_work / "runs"),
        )
        assert summary["steps"] > 0
        model = PragmaModel.from_pretrained(embed_work / "runs" / "e2e")
        assert model.text_encoder is not None and model.text_proj is not None  # auto-configured
        raw = next(iter_user_records(embed_work / "raw", max_users=1))
        z = model.embed_records([raw])
        assert z.shape == (1, model.config.dim) and bool(np.isfinite(z).all())
