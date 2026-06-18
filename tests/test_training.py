"""Training tests: optimizers, pretrainer checkpoint/resume/NaN, probe, fine-tune."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pragmatiq import api
from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
from pragmatiq.data.tokenizer import PragmaTokenizer
from pragmatiq.experiments.run import Run, list_runs
from pragmatiq.experiments.tracking import MetricLogger
from pragmatiq.models.pragmatiq import ModelConfig, PragmaModel
from pragmatiq.training.optim import (
    Muon,
    WarmupCosine,
    cosine_warmup_factor,
    split_parameters,
    zeropower_via_newtonschulz5,
)
from pragmatiq.training.pretrainer import PreTrainer, TrainConfig, seed_everything


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    work = tmp_path_factory.mktemp("p5")
    api.synthesize({"n_users": 250, "months": 14, "n_merchants": 700, "mule_ring_count": 1,
                    "seed": 3, "eval_month_credit": 2, "eval_month_short": 8},
                   out=work / "raw", n_workers=0, write_report=False)
    api.tokenize(work / "raw", work / "tok",
                 config={"target_vocab": 3500, "n_buckets": 32, "categorical_threshold": 200})
    return work


def _nano(tok_hash: str, steps: int, run: Run, vocab: int, work: Path, **over):
    cfg = TrainConfig(max_steps=steps, token_budget=4096, warmup_steps=5, seed=0,
                      checkpoint_every_min=1000.0, log_every=1, **over)
    seed_everything(cfg.seed)
    model = PragmaModel(ModelConfig.preset("small", vocab))
    trainer = PreTrainer(model, run, cfg, tok_hash, logger=MetricLogger(run.dir))
    ds = ShardDataset(work / "tok")
    loader = ShardDataLoader(ds, DynamicBatchSampler(ds.index, token_budget=4096, seed=0))
    return trainer, loader, ds


# ---------------------------------------------------------------- optim
class TestOptim:
    def test_newton_schulz_orthogonalizes(self) -> None:
        torch.manual_seed(0)
        G = torch.randn(32, 48)
        ortho = zeropower_via_newtonschulz5(G, steps=5)
        # singular values should be pushed toward 1 (semi-orthogonal rows)
        s = torch.linalg.svdvals(ortho)
        assert s.max() < 1.3 and s.min() > 0.5

    def test_split_routes_2d_to_muon(self) -> None:
        model = PragmaModel(ModelConfig.preset("small", 2000))
        muon, adamw = split_parameters(model)
        assert all(p.ndim == 2 for p in muon)
        # embedding table (2-D) must be in AdamW, not Muon
        emb_id = id(model.embed.embed.weight)
        assert emb_id in {id(p) for p in adamw}
        assert emb_id not in {id(p) for p in muon}

    def test_split_routes_lora_to_adamw(self) -> None:
        # LoRA A/B are 2-D but deliberately low-rank: Newton-Schulz would destroy
        # them, so they must NOT land in the Muon group.
        from pragmatiq.models import inject_lora

        model = PragmaModel(ModelConfig.preset("nano", 1500))
        inject_lora(model, rank=4)
        muon, _ = split_parameters(model)
        muon_ids = {id(p) for p in muon}
        lora = [p for n, p in model.named_parameters() if "lora" in n]
        assert lora, "no LoRA params injected"
        assert all(id(p) not in muon_ids for p in lora), "LoRA factor routed to Muon"

    def test_muon_step_runs(self) -> None:
        w = torch.nn.Parameter(torch.randn(8, 8))
        opt = Muon([w], lr=1e-2)
        (w.sum()).backward()
        before = w.detach().clone()
        opt.step()
        assert not torch.allclose(before, w)

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="bf16-vs-fp32 Newton-Schulz only diverges on CUDA")
    def test_newton_schulz_fp32_under_deterministic(self, monkeypatch) -> None:
        # Under deterministic mode the orthogonalization runs in fp32, not the
        # default CUDA bf16, so the deterministic fp32 weight-update claim holds.
        # Patch the gate flag rather than enabling deterministic algorithms (which
        # would require CUBLAS_WORKSPACE_CONFIG set before cuBLAS init). fp32 work
        # orthogonalizes more tightly, so its singular values sit closer to 1.
        torch.manual_seed(0)
        g = torch.randn(96, 64, device="cuda")

        def sv_err(m: torch.Tensor) -> float:
            return float((torch.linalg.svdvals(m.double()) - 1.0).abs().max())

        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
        bf = zeropower_via_newtonschulz5(g.clone())  # bf16 work on CUDA
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
        fp = zeropower_via_newtonschulz5(g.clone())  # fp32 work under deterministic
        assert not torch.allclose(fp, bf)  # the dtype gate actually changes the computation
        assert sv_err(fp) < sv_err(bf)  # fp32 is the tighter orthogonalization

    def test_cosine_warmup_shape(self) -> None:
        # 1-indexed warmup: step 0 gets a small positive LR (not 0, not full base)
        assert cosine_warmup_factor(0, 10, 100) == pytest.approx(0.1)  # (0+1)/10
        assert cosine_warmup_factor(9, 10, 100) == pytest.approx(1.0)  # (9+1)/10 = peak
        assert cosine_warmup_factor(10, 10, 100) == pytest.approx(1.0)  # cosine start = peak
        assert cosine_warmup_factor(100, 10, 100) == pytest.approx(0.1, abs=1e-6)  # min ratio
        # warmup is strictly increasing up to the peak
        warm = [cosine_warmup_factor(s, 10, 100) for s in range(10)]
        assert warm == sorted(warm) and 0 < warm[0] < warm[-1]

    def test_scheduler_state_roundtrip(self) -> None:
        w = torch.nn.Parameter(torch.randn(4, 4))
        opt = torch.optim.AdamW([w], lr=1e-3)
        sched = WarmupCosine([opt], 5, 50)
        for _ in range(7):
            sched.step()
        s = sched.state_dict()
        sched2 = WarmupCosine([opt], 5, 50)
        sched2.load_state_dict(s)
        assert sched2.last_factor == sched.last_factor


# ---------------------------------------------------------------- pretrainer
class TestPretrainer:
    def test_loss_decreases(self, shards: Path) -> None:
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("loss", {}, 0, tok.content_hash, shards / "runs", tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 40, run, tok.vocab_size, shards)
        trainer.fit(loader)
        ds.close()
        rows = [json.loads(x) for x in run.metrics_path.read_text().strip().splitlines()]
        first = np.mean([r["loss"] for r in rows[:5]])
        last = np.mean([r["loss"] for r in rows[-5:]])
        assert last < first, f"loss did not decrease: {first:.3f} -> {last:.3f}"
        # per-masking-type losses are logged
        assert any("loss_token" in r for r in rows)
        assert any("loss_event" in r for r in rows)
        assert any("loss_key" in r for r in rows)

    def test_checkpoint_contains_required_state(self, shards: Path) -> None:
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("ckpt", {}, 0, tok.content_hash, shards / "runs", tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 5, run, tok.vocab_size, shards)
        trainer.fit(loader)
        ckpt = torch.load(run.checkpoints / "last.pt", map_location="cpu", weights_only=False)
        for key in ("model", "head", "optimizers", "scheduler", "sampler", "rng",
                    "tokenizer_hash", "model_config", "train_config"):
            assert key in ckpt, f"checkpoint missing {key}"
        assert ckpt["tokenizer_hash"] == tok.content_hash
        assert "masking_gen" in ckpt["rng"] and "torch" in ckpt["rng"] and "numpy" in ckpt["rng"]
        ds.close()

    def test_resume_bit_exact(self, shards: Path) -> None:
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        h = tok.content_hash

        def loss_map(run: Run):
            return {x["step"]: x["loss"]
                    for x in (json.loads(ln) for ln in run.metrics_path.read_text().strip().splitlines())}

        # Both runs target the SAME horizon (30 steps); the interrupted one is
        # killed at step 15 via the max_steps loop bound, then resumed.
        u = Run.create("ru", {}, 0, h, shards / "runs", tokenizer_src=shards / "tok" / "tokenizer")
        tr, ld, ds = _nano(h, 30, u, tok.vocab_size, shards)
        tr.fit(ld)
        ds.close()
        U = loss_map(u)

        i = Run.create("ri", {}, 0, h, shards / "runs", tokenizer_src=shards / "tok" / "tokenizer")
        tr1, ld1, ds1 = _nano(h, 30, i, tok.vocab_size, shards)
        tr1.fit(ld1, max_steps=15)  # simulate a kill at step 15 (horizon stays 30)
        ds1.close()
        tr2, ld2, ds2 = _nano(h, 30, i, tok.vocab_size, shards)
        tr2.fit(ld2, resume="auto")
        ds2.close()
        R = loss_map(i)
        maxd = max(abs(U[s] - R[s]) for s in range(16, 31))
        # Bit-exactness holds on CPU fp32. On CUDA, bf16 + atomic scatter
        # kernels are nondeterministic even between identical runs (observed
        # ~4e-4); genuine RNG/sampler divergence shows up as O(0.1+) diffs.
        tol = 5e-3 if torch.cuda.is_available() else 1e-4
        assert maxd < tol, f"resume not bit-exact: max loss diff {maxd:.2e}"

    def test_api_resume_rebuilds_checkpoint_size(self, shards: Path) -> None:
        # A run trained at one size must resume at that size even when the caller
        # uses the default model_size: api.pretrain reads the run's run.yaml on
        # resume and rebuilds the architecture, so the strict checkpoint load does
        # not fail on a shape mismatch.
        runs_root = shards / "resume_api_runs"
        base = {"token_budget": 4096, "warmup_steps": 2, "log_every": 1, "max_steps": 6}
        api.pretrain(shards / "tok", "rrt", model_size="nano", config=base, runs_root=runs_root)
        assert Run.open("rrt", runs_root).read_config()["model_size"] == "nano"
        # Default model_size is "small"; resume="auto" must rebuild nano (else the
        # strict load raises). The max_steps kwarg extends the stored config.
        api.pretrain(shards / "tok", "rrt", resume="auto", max_steps=9, runs_root=runs_root)
        assert Run.open("rrt", runs_root).read_config()["model_size"] == "nano"

    def test_nan_loss_dumps_and_skips(self, shards: Path, monkeypatch) -> None:
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("nanloss", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 1, run, tok.vocab_size, shards)
        import pragmatiq.training.pretrainer as P
        monkeypatch.setattr(P, "mlm_loss", lambda logits, targets: logits.sum() * float("nan"))
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.fit(loader)
        ds.close()
        assert list((run.dir / "debug").glob("nan_step*.pt")), "no debug dump on NaN loss"
        after = list(trainer.model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after)), "params changed on a skipped step"

    def test_nan_grad_dumps_and_skips(self, shards: Path) -> None:
        # finite loss but non-finite gradient must dump+skip, not crash the run
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("nangrad", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 1, run, tok.vocab_size, shards)
        p0 = next(p for p in trainer.model.parameters() if p.requires_grad)
        handle = p0.register_hook(lambda g: g * float("inf"))
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.fit(loader)  # must not raise
        handle.remove()
        ds.close()
        assert list((run.dir / "debug").glob("nan_step*.pt")), "grad-NaN not caught (would crash a run)"
        after = list(trainer.model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after))

    def test_tokenizer_mismatch_refused(self, shards: Path) -> None:
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("mm", {}, 0, tok.content_hash, shards / "runs", tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 5, run, tok.vocab_size, shards)
        trainer.fit(loader)
        # a new trainer claiming a different tokenizer hash must refuse to resume
        trainer2, loader2, ds2 = _nano("deadbeef_wrong_hash", 10, run, tok.vocab_size, shards)
        with pytest.raises(ValueError, match="tokenizer hash mismatch"):
            trainer2.load_checkpoint(run.last_checkpoint(), loader2)
        ds.close()
        ds2.close()

    def test_consecutive_skips_abort(self, shards: Path, monkeypatch) -> None:
        # A sustained non-finite streak is divergence: it must fail loud, not
        # silently burn compute to max_steps.
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("skipabort", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 100, run, tok.vocab_size, shards,
                                    max_consecutive_skips=3)
        import pragmatiq.training.pretrainer as P
        monkeypatch.setattr(P, "mlm_loss", lambda logits, targets: logits.sum() * float("nan"))
        with pytest.raises(RuntimeError, match="consecutive non-finite"):
            trainer.fit(loader)
        ds.close()

    def test_metrics_truncated_on_resume(self, shards: Path) -> None:
        # Logged-but-uncheckpointed rows from a crashed interval must not survive
        # resume (the JSONL stays monotonic in step).
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("metrunc", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        tr, ld, ds = _nano(tok.content_hash, 10, run, tok.vocab_size, shards)
        tr.fit(ld, max_steps=10)
        ds.close()
        with open(run.metrics_path, "a") as f:  # simulate uncheckpointed future rows
            f.write(json.dumps({"step": 11, "loss": 9.9}) + "\n")
            f.write(json.dumps({"step": 12, "loss": 9.9}) + "\n")
        tr2, ld2, ds2 = _nano(tok.content_hash, 14, run, tok.vocab_size, shards)
        tr2.fit(ld2, resume="auto")
        ds2.close()
        steps = [json.loads(ln)["step"] for ln in run.metrics_path.read_text().strip().splitlines()]
        assert steps == sorted(set(steps)), f"metrics not monotonic after resume: {steps}"

    def test_checkpoint_rename_guarded_to_rank_zero(self, shards: Path) -> None:
        # Under DDP only the global-zero rank renames the temp checkpoint; other
        # ranks (whose fabric.save writes no temp file) must not raise.
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("ckptrank", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 1, run, tok.vocab_size, shards)

        class _FakeFabric:
            def __init__(self, is_zero: bool, writes: bool) -> None:
                self.is_global_zero = is_zero
                self._writes = writes
                self.barriers = 0

            def save(self, path, state) -> None:
                if self._writes:
                    Path(path).write_bytes(b"ckpt")

            def barrier(self) -> None:
                self.barriers += 1

        trainer.fabric = _FakeFabric(is_zero=True, writes=True)  # rank 0 writes + renames
        p = trainer.save_checkpoint(loader, "last.pt")
        assert p.exists() and trainer.fabric.barriers == 1
        rank1 = _FakeFabric(is_zero=False, writes=False)  # non-zero rank: no temp file
        trainer.fabric = rank1
        trainer.save_checkpoint(loader, "last.pt")  # must not raise FileNotFoundError
        assert (run.checkpoints / "last.pt").exists() and rank1.barriers == 1
        ds.close()

    def test_cross_device_masking_rng_reseed(self, shards: Path) -> None:
        # A checkpoint whose masking RNG was saved on another device must resume
        # via a clean re-seed, not a raw RuntimeError from set_state.
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("xdevrng", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        tr, ld, ds = _nano(tok.content_hash, 2, run, tok.vocab_size, shards)
        tr.fit(ld)
        ds.close()
        ckpt_path = run.last_checkpoint()
        ck = torch.load(ckpt_path, weights_only=False)
        # Tag the checkpoint with the OPPOSITE of this host's masking-gen device so
        # the device-mismatch re-seed path is exercised on both CPU and GPU hosts
        # (the raw state buffer is ignored once a mismatch is detected).
        ck["rng"]["masking_gen_device"] = "cpu" if torch.cuda.is_available() else "cuda"
        ck["rng"]["masking_gen"] = torch.zeros(16, dtype=torch.uint8)
        torch.save(ck, ckpt_path)
        tr2, ld2, ds2 = _nano(tok.content_hash, 4, run, tok.vocab_size, shards)
        tr2.load_checkpoint(ckpt_path, ld2)  # must not raise (re-seeds instead)
        ds2.close()

    def test_list_runs(self, shards: Path) -> None:
        runs = list_runs(shards / "runs")
        assert len(runs) >= 1
        assert all("name" in r and "git_hash" in r for r in runs)


# ---------------------------------------------------------------- determinism
class TestDeterminism:
    """Opt-in deterministic mode toggles process-wide CUDA/cuDNN switches.

    These flags are global, so every test restores them in a ``finally`` to
    avoid poisoning sibling tests that run later in the same process.
    """

    @staticmethod
    def _restore() -> None:
        import os

        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        os.environ.pop("PRAGMATIQ_DETERMINISTIC", None)

    def test_default_leaves_deterministic_off(self) -> None:
        try:
            seed_everything(0)
            assert not torch.are_deterministic_algorithms_enabled()
        finally:
            self._restore()

    def test_deterministic_true_enables_global_switches(self) -> None:
        import os

        try:
            seed_everything(0, deterministic=True)
            assert torch.are_deterministic_algorithms_enabled()
            assert "CUBLAS_WORKSPACE_CONFIG" in os.environ
            assert os.environ.get("PRAGMATIQ_DETERMINISTIC") == "1"
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
        finally:
            self._restore()

    def test_deterministic_false_clears_global_switches(self) -> None:
        import os

        try:
            seed_everything(0, deterministic=True)
            assert torch.are_deterministic_algorithms_enabled()
            seed_everything(0, deterministic=False)  # symmetric off-switch
            assert not torch.are_deterministic_algorithms_enabled()
            assert torch.backends.cudnn.deterministic is False
            assert "PRAGMATIQ_DETERMINISTIC" not in os.environ
        finally:
            self._restore()

    def test_deterministic_forward_backward_no_runtime_error(self, shards: Path) -> None:
        # Under deterministic algorithms, a forward + backward through mlm_loss
        # must run on CPU without the "no deterministic implementation" error.
        from pragmatiq.models.heads import MLMHead, mlm_loss
        from pragmatiq.training.masking import MaskingStrategy

        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        ds = ShardDataset(shards / "tok")
        loader = ShardDataLoader(ds, DynamicBatchSampler(ds.index, token_budget=4096, seed=0))
        try:
            seed_everything(0, deterministic=True)
            model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size))
            head = MLMHead(model.config.dim)
            batch = next(iter(loader))
            masked = MaskingStrategy()(batch, torch.Generator().manual_seed(0))
            import dataclasses as _dc

            masked_batch = _dc.replace(batch, value_ids=masked.input_value_ids)
            out = model(masked_batch)
            sel = masked.selected_idx
            assert sel.numel() > 0
            logits = head(out, model.embedding_weight, sel)
            loss = mlm_loss(logits, masked.labels[sel])
            loss.backward()  # must not raise under deterministic algorithms
            assert torch.isfinite(loss)
        finally:
            self._restore()
            ds.close()


# ---------------------------------------------------------------- gradient accumulation
class _FixedMasker:
    """Deterministic masker (ignores the RNG) so gradient-accumulation equivalence is
    exact: it masks every 5th token to a [MASK] with the true value as the CE target."""

    def __call__(self, batch, generator=None):  # noqa: ANN001
        from pragmatiq.data.tokenizer import MASK
        from pragmatiq.training.masking import T_TOKEN, MaskedBatch

        T = batch.key_ids.numel()
        sel = torch.arange(0, T, 5, dtype=torch.long)
        labels = torch.full((T,), -100, dtype=torch.int64)
        labels[sel] = batch.value_ids[sel]
        ivi = batch.value_ids.clone()
        ivi[sel] = MASK
        mtype = torch.full((T,), -1, dtype=torch.int8)
        mtype[sel] = T_TOKEN
        return MaskedBatch(input_value_ids=ivi, labels=labels, mask_type=mtype, selected_idx=sel)


class TestGradAccum:
    def test_accum_equivalent_to_single_batch(self, shards: Path) -> None:
        # With a fixed masker and dropout off, accumulating N copies of a batch must
        # produce the SAME optimizer step as one batch: loss/N summed N times == one
        # gradient, and the LR schedule advances exactly once either way.
        from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset

        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        ds = ShardDataset(shards / "tok")
        loader = ShardDataLoader(ds, DynamicBatchSampler(ds.index, token_budget=4096, seed=0))
        batch = next(iter(loader))
        ds.close()

        def step_once(accum: int):
            run = Run.create(f"accum{accum}", {}, 0, tok.content_hash, shards / "runs",
                             tokenizer_src=shards / "tok" / "tokenizer")
            cfg = TrainConfig(max_steps=10, token_budget=4096, warmup_steps=0, seed=0,
                              grad_accum_steps=accum, checkpoint_every_min=1000.0)
            seed_everything(cfg.seed)
            model = PragmaModel(ModelConfig.preset("small", tok.vocab_size, overrides={"dropout": 0.0}))
            trainer = PreTrainer(model, run, cfg, tok.content_hash, masker=_FixedMasker())
            trainer._train_step([batch] * accum)
            return model

        m1 = step_once(1)
        m2 = step_once(2)
        diffs = {n: float((pa.detach() - pb.detach()).abs().max())
                 for (n, pa), (_, pb) in zip(m1.named_parameters(), m2.named_parameters())}
        worst = max(diffs.values())
        assert worst < 1e-5, f"accum step diverged from single-batch step (max |Δ|={worst:.2e})"

    def test_skipped_micro_batch_rescales_step(self, shards: Path) -> None:
        # A micro-batch that selects nothing to learn from contributes no gradient; the
        # step must average over the micro-batches that DID contribute. So a window of
        # [real, empty] must produce the same update as a single real batch.
        from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
        from pragmatiq.training.masking import MaskedBatch

        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        ds = ShardDataset(shards / "tok")
        loader = ShardDataLoader(ds, DynamicBatchSampler(ds.index, token_budget=4096, seed=0))
        batch = next(iter(loader))
        ds.close()

        class _FixedThenEmpty:
            """Mask the first micro-batch like ``_FixedMasker``; select nothing afterwards."""

            def __init__(self) -> None:
                self.base = _FixedMasker()
                self.calls = 0

            def __call__(self, b, generator=None):  # noqa: ANN001
                self.calls += 1
                if self.calls == 1:
                    return self.base(b, generator)
                t = b.key_ids.numel()
                return MaskedBatch(input_value_ids=b.value_ids.clone(),
                                   labels=torch.full((t,), -100, dtype=torch.int64),
                                   mask_type=torch.full((t,), -1, dtype=torch.int8),
                                   selected_idx=torch.zeros(0, dtype=torch.long))

        def step(name: str, masker, window):
            run = Run.create(name, {}, 0, tok.content_hash, shards / "runs",
                             tokenizer_src=shards / "tok" / "tokenizer")
            cfg = TrainConfig(max_steps=10, token_budget=4096, warmup_steps=0, seed=0,
                              grad_accum_steps=len(window), checkpoint_every_min=1000.0)
            seed_everything(cfg.seed)
            model = PragmaModel(ModelConfig.preset("small", tok.vocab_size, overrides={"dropout": 0.0}))
            trainer = PreTrainer(model, run, cfg, tok.content_hash, masker=masker)
            trainer._train_step(window)
            return model

        m_single = step("skip_single", _FixedMasker(), [batch])
        m_skip = step("skip_window", _FixedThenEmpty(), [batch, batch])
        worst = max(float((a.detach() - b.detach()).abs().max())
                    for (_, a), (_, b) in zip(m_single.named_parameters(), m_skip.named_parameters()))
        assert worst < 1e-5, f"skip-corrected step diverged from single-batch (max |Δ|={worst:.2e})"

    def test_empty_loader_raises_instead_of_hanging(self, shards: Path) -> None:
        # A loader that yields nothing for a whole epoch must fail fast with an
        # actionable error rather than spinning forever re-iterating an empty stream.
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("empty_loader", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        cfg = TrainConfig(max_steps=5, token_budget=4096, warmup_steps=0, seed=0,
                          checkpoint_every_min=1000.0)
        seed_everything(cfg.seed)
        model = PragmaModel(ModelConfig.preset("small", tok.vocab_size, overrides={"dropout": 0.0}))
        trainer = PreTrainer(model, run, cfg, tok.content_hash, masker=_FixedMasker())

        class _EmptySampler:
            def set_replica_info(self, *a) -> None: ...
            def set_epoch(self, *a) -> None: ...

        class _EmptyLoader:
            sampler = _EmptySampler()

            def __iter__(self):
                return iter(())

        with pytest.raises(RuntimeError, match="no batches for a full epoch"):
            trainer.fit(_EmptyLoader())


# ---------------------------------------------------------------- probe / baseline
class TestEmbeddingProbeModels:
    """The probe-head choice (gbdt default / logistic) on frozen embeddings, fast and
    model-free: a separable toy embedding must score well above chance and report both
    ROC-AUC and PR-AUC for every supported classifier."""

    @staticmethod
    def _toy(tmp_path: Path, n: int = 240, d: int = 16, seed: int = 0):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, d))
        y = ((X @ rng.normal(size=d)) + rng.normal(scale=0.5, size=n) > 0).astype(np.int64)
        emb = {f"u{i}": X[i].astype(np.float32) for i in range(n)}
        path = tmp_path / "labels.parquet"
        pq.write_table(pa.table({"user_id": [f"u{i}" for i in range(n)], "label": y}), path)
        return emb, path

    @pytest.mark.parametrize("model", ["gbdt", "logistic"])
    def test_reports_roc_and_pr_auc(self, tmp_path: Path, model: str) -> None:
        from pragmatiq.training.probe import EmbeddingProbe

        emb, label_path = self._toy(tmp_path)
        res = EmbeddingProbe(model=model, seed=0).run(emb, label_path)
        assert 0.0 <= res.auc <= 1.0 and 0.0 <= res.pr_auc <= 1.0
        assert res.auc > 0.6  # separable toy signal → comfortably above chance
        assert res.pr_auc > res.prevalence  # PR-AUC clears the all-positive floor

    def test_unknown_model_raises(self, tmp_path: Path) -> None:
        from pragmatiq.training.probe import EmbeddingProbe

        emb, label_path = self._toy(tmp_path)
        with pytest.raises(ValueError, match="unknown probe_model"):
            EmbeddingProbe(model="nope").run(emb, label_path)


class TestProbe:
    def test_probe_runs_and_has_baseline(self, shards: Path) -> None:
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("probe", {}, 0, tok.content_hash, shards / "runs", tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 30, run, tok.vocab_size, shards)
        trainer.fit(loader)
        ds.close()
        res = api.probe(shards / "tok", run.dir, shards / "raw" / "labels" / "default_12m.parquet")
        assert res["probe_model"] == "gbdt"  # gradient boosting is the default head
        assert 0.0 <= res["probe_auc"] <= 1.0 and 0.0 <= res["probe_pr_auc"] <= 1.0
        assert "baseline_auc" in res and "baseline_pr_auc" in res
        assert res["n_test"] > 0

    def test_embed_users_subset_matches_full(self, shards: Path) -> None:
        import numpy as np

        from pragmatiq.data.dataset import ShardDataset
        from pragmatiq.training.probe import embed_users

        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("sub", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 3, run, tok.vocab_size, shards)
        trainer.fit(loader)
        ds.close()
        model = PragmaModel.from_pretrained(run.dir)
        ds2 = ShardDataset(shards / "tok")
        full = embed_users(model, ds2)
        cohort = list(full)[::2]  # a subset spanning the index
        sub = embed_users(model, ds2, user_ids=cohort)
        ds2.close()
        assert set(sub) == set(cohort)  # only the cohort is embedded
        for u in cohort:
            # numerically equivalent to the full-dataset path; the residual is
            # float32 reduction-order noise from the different batch packing.
            assert np.allclose(sub[u], full[u], atol=1e-5, rtol=1e-4)

    def test_embed_users_oom_auto_splits(self, shards: Path, monkeypatch) -> None:
        # A batch that OOMs must be re-embedded in smaller sub-batches (the SPEC
        # OOM-retry contract), not crash — every user still gets an embedding.
        from pragmatiq.data.dataset import ShardDataset
        from pragmatiq.training.probe import embed_users

        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("oomembed", {}, 0, tok.content_hash, shards / "runs",
                         tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 3, run, tok.vocab_size, shards)
        trainer.fit(loader)
        ds.close()
        model = PragmaModel.from_pretrained(run.dir)
        clean = embed_users(model, ShardDataset(shards / "tok"))

        orig = PragmaModel.embed_users

        def flaky(self, batch, *a, **k):  # OOM on any multi-user batch, succeed on singletons
            if batch.n_users > 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return orig(self, batch, *a, **k)

        monkeypatch.setattr(PragmaModel, "embed_users", flaky)
        ds2 = ShardDataset(shards / "tok")
        recovered = embed_users(model, ds2, token_budget=16_384)
        ds2.close()
        assert set(recovered) == set(clean) and len(recovered) > 1
        for u in clean:
            assert np.allclose(recovered[u], clean[u], atol=1e-5, rtol=1e-4)

    def test_embed_records_notebook_api(self, shards: Path) -> None:
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("nb", {}, 0, tok.content_hash, shards / "runs", tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 3, run, tok.vocab_size, shards)
        trainer.fit(loader)
        ds.close()
        model = PragmaModel.from_pretrained(run.dir)
        rec = {"user_id": "u_test", "as_of": 1_700_000_000_000_000,
               "events": [(1_699_000_000_000_000, "transaction",
                           {"amount": "12.50", "currency": "GBP", "mcc": "5411",
                            "merchant": "TESCO 1", "txn_type": "card_payment", "channel": "pos"})],
               "attributes": {"country": "GB", "age_band": "20-29"},
               "lifelong": [("account_opened", 1_698_000_000_000_000)]}
        emb = model.embed_records([rec])
        assert emb.shape == (1, model.config.dim)


# ---------------------------------------------------------------- fine-tune
class TestFineTune:
    def test_lora_finetune_runs(self, shards: Path) -> None:
        tok = PragmaTokenizer.load(shards / "tok" / "tokenizer")
        run = Run.create("ft", {}, 0, tok.content_hash, shards / "runs", tokenizer_src=shards / "tok" / "tokenizer")
        trainer, loader, ds = _nano(tok.content_hash, 10, run, tok.vocab_size, shards)
        trainer.fit(loader)
        ds.close()
        res = api.finetune(shards / "tok", run.dir, shards / "raw" / "labels" / "default_12m.parquet",
                           config={"max_epochs": 2, "lora_rank": 4, "token_budget": 4096})
        assert res["n_adapted"] > 0
        assert "best_val_auc" in res
