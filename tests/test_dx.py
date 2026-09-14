"""Developer-experience surface added in 1.1.0: ``pragmatiq info`` / ``--version``,
``pretrain --show-config`` (``api.pretrain_plan``), the typed error hierarchy, and the
module layout (``pragmatiq.core.progress``, ``pragmatiq.runs``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pragmatiq import api
from pragmatiq.core.errors import ConfigError, DataContractError, PragmatiqError


def _runner():
    from pragmatiq.cli import app

    try:
        return CliRunner(mix_stderr=False), app
    except TypeError:
        return CliRunner(), app


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("dx")
    api.synthesize({"n_users": 40, "months": 14, "n_merchants": 300, "seed": 2,
                    "eval_month_credit": 2, "eval_month_short": 8},
                   out=work / "raw", write_report=False)
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 2000, "n_buckets": 16, "categorical_threshold": 100})
    return work


class TestInfo:
    def test_info_dict(self, monkeypatch) -> None:
        monkeypatch.setenv("PRAGMATIQ_DEVICE", "cpu")
        out = api.info()
        assert out["device"] == "cpu" and out["inference_precision"] == "fp32"
        assert out["env"]["PRAGMATIQ_DEVICE"] == "cpu" and "PRAGMATIQ_SERVE_CPU" in out["env"]
        assert set(out["attention_backend"]) == {"cuda/bf16", "cpu/fp32"}
        assert "lightning" in out["extras"] and out["extras"]["onnxruntime"]["extra"] == "serve"
        assert isinstance(out["cuda"]["available"], bool) and isinstance(out["cuda"]["devices"], list)
        assert "importable" in out["flash_attn"]
        json.dumps(out)  # JSON-serialisable end to end

    def test_cli_info_and_version(self) -> None:
        import pragmatiq

        runner, app = _runner()
        res = runner.invoke(app, ["info"])
        assert res.exit_code == 0, res.output
        assert json.loads(res.stdout)["pragmatiq"] == pragmatiq.__version__
        res = runner.invoke(app, ["--version"])
        assert res.exit_code == 0 and res.stdout.strip() == f"pragmatiq {pragmatiq.__version__}"


class TestPretrainPlan:
    def test_plan_matches_pretrain_resolution(self, shards: Path) -> None:
        plan = api.pretrain_plan(shards / "tok", model_size="nano",
                                 config={"max_steps": 7, "token_budget": 2048}, seed=3)
        assert plan["model_size"] == "nano" and plan["dim"] == 64
        assert plan["max_steps"] == 7 and plan["token_budget"] == 2048 and plan["seed"] == 3
        assert plan["device"] in ("cpu", "cuda") and len(plan["tokenizer_hash"]) == 64
        assert plan["resuming"] is False
        assert not (shards / "runs").exists()  # nothing was trained or created

    def test_plan_auto_sizes_from_data(self, shards: Path) -> None:
        plan = api.pretrain_plan(shards / "tok", model_size="nano", config="auto")
        assert plan["token_budget"] >= 2048 and plan["max_steps"] >= 1 and plan["warmup_steps"] >= 1

    def test_plan_rejects_bad_config(self, shards: Path) -> None:
        with pytest.raises(ConfigError, match="unknown pretrain config key"):
            api.pretrain_plan(shards / "tok", config={"nope": 1})
        with pytest.raises(ConfigError, match="model-size"):
            api.pretrain_plan(shards / "tok", model_size="giant")

    def test_cli_show_config(self, shards: Path) -> None:
        runner, app = _runner()
        res = runner.invoke(app, ["pretrain", str(shards / "tok"), "--name", "x", "--model-size",
                                  "nano", "--show-config"])
        assert res.exit_code == 0, res.output
        out = json.loads(res.stdout)
        assert out["model_size"] == "nano" and "token_budget" in out
        assert not (Path.cwd() / "runs" / "x").exists()


class TestErrors:
    def test_hierarchy(self) -> None:
        assert issubclass(ConfigError, PragmatiqError) and issubclass(ConfigError, ValueError)
        assert issubclass(DataContractError, PragmatiqError) and issubclass(DataContractError, ValueError)

    def test_missing_shard_dir(self, tmp_path: Path) -> None:
        with pytest.raises(DataContractError, match="does not exist"):
            api.embed(tmp_path / "nope", tmp_path / "run")
        (tmp_path / "empty").mkdir()
        with pytest.raises(DataContractError, match="no tokenizer/"):
            api.embed(tmp_path / "empty", tmp_path / "run")

    def test_missing_run_dir(self, shards: Path, tmp_path: Path) -> None:
        with pytest.raises(DataContractError, match="run directory .* does not exist"):
            api.embed(shards / "tok", tmp_path / "run")
        (tmp_path / "run").mkdir()
        with pytest.raises(DataContractError, match="no checkpoint"):
            api.embed(shards / "tok", tmp_path / "run")

    def test_bad_resume_is_config_error(self, shards: Path) -> None:
        with pytest.raises(ConfigError, match="resume"):
            api.pretrain(shards / "tok", "x", resume="yes")

    def test_label_table_columns(self, shards: Path, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from pragmatiq.api import _require_label_table

        pq.write_table(pa.table({"user_id": ["a"], "y": [1]}), tmp_path / "bad.parquet")
        with pytest.raises(DataContractError, match="missing column"):
            _require_label_table(tmp_path / "bad.parquet")
        with pytest.raises(DataContractError, match="does not exist"):
            _require_label_table(tmp_path / "gone.parquet")
        _require_label_table(shards / "raw" / "labels" / "default_12m.parquet")


class TestModuleLayout:
    def test_moved_modules_import(self) -> None:
        from pragmatiq.core.progress import progress
        from pragmatiq.runs.run import Run
        from pragmatiq.runs.tracking import MetricLogger

        assert callable(progress) and Run is not None and MetricLogger is not None
        assert not (Path(api.__file__).parent / "progress.py").exists()
        assert not (Path(api.__file__).parent / "experiments").exists()
