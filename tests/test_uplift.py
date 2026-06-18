"""Uplift evaluation: Qini metric + end-to-end api.uplift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pragmatiq import api
from pragmatiq.data.tokenizer import PragmaTokenizer
from pragmatiq.experiments.run import Run
from pragmatiq.models.pragmatiq import ModelConfig, PragmaModel
from pragmatiq.training.pretrainer import PreTrainer, TrainConfig, seed_everything
from pragmatiq.training.uplift import qini_coefficient, qini_curve


class TestQiniMetric:
    def test_ranking_beats_random(self) -> None:
        rng = np.random.default_rng(0)
        n = 600
        s = rng.random(n)  # covariate that drives the true uplift
        treated = rng.integers(0, 2, n)
        prob = 0.2 + treated * 0.6 * s  # treatment effect grows with s
        outcome = (rng.random(n) < prob).astype(int)
        good = qini_coefficient(s, treated, outcome)            # rank by true uplift proxy
        bad = qini_coefficient(rng.random(n), treated, outcome)  # random ranking
        assert good > bad
        assert good > 0.0

    def test_curve_shape(self) -> None:
        rng = np.random.default_rng(1)
        n = 50
        frac, gain = qini_curve(rng.random(n), rng.integers(0, 2, n), rng.integers(0, 2, n))
        assert len(frac) == n + 1 and len(gain) == n + 1
        assert frac[0] == 0.0 and gain[0] == 0.0
        assert frac[-1] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("uplift")
    api.synthesize({"n_users": 400, "months": 14, "n_merchants": 700, "mule_ring_count": 1,
                    "seed": 4, "eval_month_credit": 2, "eval_month_short": 8},
                   out=work / "raw", n_workers=0, write_report=False)
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 3500, "n_buckets": 32, "categorical_threshold": 200})
    return work


def _pretrain(shards: Path, name: str) -> Path:
    tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
    run = Run.create(name, {}, 0, tok.content_hash, shards / "runs",
                     tokenizer_src=shards / "tok" / "tokenizer")
    cfg = TrainConfig(max_steps=30, token_budget=4096, warmup_steps=5, seed=0,
                      checkpoint_every_min=1000.0, log_every=10)
    seed_everything(cfg.seed)
    from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
    from pragmatiq.experiments.tracking import MetricLogger
    model = PragmaModel(ModelConfig.preset("small", tok.vocab_size))
    trainer = PreTrainer(model, run, cfg, tok.content_hash, logger=MetricLogger(run.dir))
    ds = ShardDataset(shards / "tok")
    trainer.fit(ShardDataLoader(ds, DynamicBatchSampler(ds.index, token_budget=4096, seed=0)))
    ds.close()
    return run.dir


class TestUpliftApi:
    def test_uplift_runs(self, shards: Path) -> None:
        run_dir = _pretrain(shards, "up")
        res = api.uplift(shards / "tok", run_dir, shards / "raw" / "labels" / "comm_uplift.parquet")
        for k in ("qini", "qini_oracle", "ate", "n_test", "treated_frac"):
            assert k in res
        assert res["n_test"] > 0
        assert 0.0 < res["treated_frac"] < 1.0  # both arms present
        assert np.isfinite(res["qini"]) and np.isfinite(res["ate"])

    def test_uplift_deterministic(self, shards: Path) -> None:
        run_dir = _pretrain(shards, "up2")
        lp = shards / "raw" / "labels" / "comm_uplift.parquet"
        a = api.uplift(shards / "tok", run_dir, lp, seed=1)
        b = api.uplift(shards / "tok", run_dir, lp, seed=1)
        assert a["qini"] == b["qini"]
