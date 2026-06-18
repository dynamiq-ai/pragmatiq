"""Hands-off training sizing: auto-config picks sane batch/schedule from data + device,
and the multi-node/grad-accum surface is wired through TrainConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from pragmatiq import api
from pragmatiq.training.autoconfig import autoconfigure, token_budget_for
from pragmatiq.training.pretrainer import TrainConfig


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("autocfg")
    api.synthesize({"n_users": 200, "months": 14, "n_merchants": 600, "mule_ring_count": 1,
                    "seed": 3, "eval_month_credit": 2, "eval_month_short": 8},
                   out=root / "raw", n_workers=0, write_report=False)
    api.tokenize(root / "raw", root / "tok",
                 config={"target_vocab": 3000, "n_buckets": 32, "categorical_threshold": 200})
    return root


class TestTokenBudget:
    def test_cpu_is_correctness_first(self) -> None:
        assert token_budget_for("cpu", "small") == 4096

    def test_cuda_scales_with_model_and_memory(self) -> None:
        wide = token_budget_for("cuda", "small", mem_gib=80)
        wider = token_budget_for("cuda", "large", mem_gib=80)
        assert wide > wider  # a wider model holds fewer tokens per GiB
        small_gpu = token_budget_for("cuda", "small", mem_gib=16)
        assert small_gpu < wide  # less memory → smaller budget
        for b in (wide, wider, small_gpu):
            assert 2048 <= b <= 131_072 and b % 256 == 0

    def test_cuda_unreadable_falls_to_floor(self) -> None:
        # No real device on CI: an unreadable CUDA target floors to the safe minimum.
        assert token_budget_for("cuda", "small", mem_gib=None) == 2048


class TestAutoconfigure:
    def test_plan_is_sane_and_valid(self, work: Path) -> None:
        plan = autoconfigure(work / "tok", device="cpu", world_size=1, model_size="small")
        assert plan.token_budget == 4096
        assert plan.grad_accum_steps >= 1
        assert plan.max_steps >= 1 and plan.warmup_steps >= 1
        assert plan.effective_tokens == plan.token_budget * plan.grad_accum_steps
        # every sized key is a real TrainConfig field, so the plan drops straight in
        assert set(plan.as_overrides()) <= set(TrainConfig.__dataclass_fields__)
        TrainConfig(**plan.as_overrides())  # constructs without error
        assert plan.rationale["n_users"] == 200 and plan.rationale["total_tokens"] > 0

    def test_more_ranks_need_no_more_accumulation(self, work: Path) -> None:
        one = autoconfigure(work / "tok", device="cpu", world_size=1)
        many = autoconfigure(work / "tok", device="cpu", world_size=8)
        assert many.grad_accum_steps <= one.grad_accum_steps  # ranks share the batch

    def test_empty_index_is_rejected(self, tmp_path: Path) -> None:
        from pragmatiq.data.sharding import ShardWriter

        empty = tmp_path / "empty"
        ShardWriter(empty, tokenizer_hash="x").close()  # writes an index with zero users
        with pytest.raises(ValueError, match="no users"):
            autoconfigure(empty, device="cpu")


class TestMultiNodeSurface:
    def test_trainconfig_carries_ddp_fields(self) -> None:
        cfg = TrainConfig(devices=4, num_nodes=2)
        assert cfg.devices == 4 and cfg.num_nodes == 2

    def test_make_fabric_accepts_num_nodes(self) -> None:
        # On a CPU box Fabric still constructs with num_nodes=1; the call must not raise
        # and the grad-accum / replica plumbing downstream reads world_size from it.
        from pragmatiq.training.pretrainer import _make_fabric

        fabric = _make_fabric(devices=1, num_nodes=1)
        assert int(getattr(fabric, "world_size", 1)) >= 1


class TestAutoPretrain:
    # "auto" must be recognized both as the literal string (Python) and as Path("auto")
    # (the CLI's --config), and route to autoconfigure rather than _load_yaml("auto").
    @pytest.mark.parametrize("config", ["auto", Path("auto")])
    def test_pretrain_auto_runs_end_to_end(self, work: Path, config) -> None:
        # config="auto" sizes the run from the data; explicit overrides (kept tiny for CI)
        # still win, proving the merge order and that "auto" routes without an unknown-key error.
        name = f"autorun-{'str' if isinstance(config, str) else 'path'}"
        summary = api.pretrain(work / "tok", name, model_size="nano", config=config,
                               runs_root=str(work / "runs"), max_steps=2, grad_accum_steps=1,
                               warmup_steps=1)
        assert summary["steps"] >= 1
