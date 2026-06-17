"""Model tests: shapes, param counts, masking, TimeRoPE grad, equivalence."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pragmatiq.data.collate import VarlenCollator
from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.data.tokenizer import (
    MASK,
    UNK,
    PragmaTokenizer,
    TokenizerConfig,
    iter_user_records,
    time_encode,
    truncate_record,
)
from pragmatiq.models import MLMHead, ModelConfig, PragmaModel, inject_lora, merge_lora
from pragmatiq.models.embeddings import TimeRoPE
from pragmatiq.models.pragmatiq import assemble_segments
from pragmatiq.training.masking import T_EVENT, T_KEY, T_NONE, T_TOKEN, MaskingStrategy


@pytest.fixture(scope="module")
def batch_and_vocab(tmp_path_factory: pytest.TempPathFactory):
    data = tmp_path_factory.mktemp("p4_data")
    generate(
        WorldConfig(n_users=80, months=14, n_merchants=600, mule_ring_count=1, seed=4,
                    eval_month_credit=2, eval_month_short=8),
        data, n_workers=0, write_report=False,
    )
    tok = PragmaTokenizer(TokenizerConfig(target_vocab=4000, n_buckets=32,
                                          categorical_threshold=200, seed=0)).fit(data)
    recs = list(iter_user_records(data, max_users=6))
    batch = VarlenCollator()([tok.encode(r) for r in recs])
    return batch, tok.vocab_size


# ---------------------------------------------------------------- param counts
class TestParamCounts:
    @pytest.mark.parametrize("size,target", [("small", 10e6), ("medium", 100e6), ("large", 1e9)])
    def test_within_10pct(self, size: str, target: float) -> None:
        # build on the meta device so 1B params cost no memory
        cfg = ModelConfig.preset(size, vocab_size=28000)
        with torch.device("meta"):
            model = PragmaModel(cfg)
            head = MLMHead(cfg.dim)
        n = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
        assert abs(n - target) / target <= 0.10, f"{size}: {n/1e6:.1f}M vs {target/1e6:.0f}M"

    def test_size_ordering(self) -> None:
        counts = []
        for size in ("small", "medium", "large"):
            with torch.device("meta"):
                counts.append(sum(p.numel() for p in PragmaModel(ModelConfig.preset(size, 28000)).parameters()))
        assert counts[0] < counts[1] < counts[2]


# ---------------------------------------------------------------- shapes
class TestShapes:
    def test_forward_shapes(self, batch_and_vocab) -> None:
        batch, vocab = batch_and_vocab
        model = PragmaModel(ModelConfig.preset("small", vocab))
        out = model(batch)
        d = model.config.dim
        assert out.token_repr.shape == (batch.n_tokens, d)
        assert out.event_repr.shape == (batch.n_events, d)
        assert out.history_event_repr.shape == (batch.n_events, d)
        assert out.user_repr.shape == (batch.n_users, d)

    def test_mlm_head_logits(self, batch_and_vocab) -> None:
        batch, vocab = batch_and_vocab
        model = PragmaModel(ModelConfig.preset("small", vocab))
        head = MLMHead(model.config.dim)
        out = model(batch)
        idx = torch.arange(0, batch.n_tokens, 7)
        logits = head(out, model.embedding_weight, idx)
        assert logits.shape == (idx.numel(), vocab)  # tied to vocab

    def test_embed_users_helper(self, batch_and_vocab) -> None:
        batch, vocab = batch_and_vocab
        model = PragmaModel(ModelConfig.preset("small", vocab))
        z = model.embed_users(batch)
        assert z.shape == (batch.n_users, model.config.dim)

    def test_mlm_head_is_literal_spec_transform(self, batch_and_vocab) -> None:
        # The MLM head is exactly concat(3d) -> Linear(3d->d) ->
        # tied logits. No extra non-linearity or normalization.
        model = PragmaModel(ModelConfig.preset("small", batch_and_vocab[1]))
        head = MLMHead(model.config.dim)
        assert not any(
            isinstance(m, (torch.nn.LayerNorm, torch.nn.GELU, torch.nn.ReLU))
            for m in head.modules()
        )
        proj_only = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
        assert len(proj_only) == 1 and proj_only[0].in_features == 3 * model.config.dim

    def test_mlm_head_matches_manual_linear_then_tied(self, batch_and_vocab) -> None:
        batch, vocab = batch_and_vocab
        model = PragmaModel(ModelConfig.preset("small", vocab)).eval()
        head = MLMHead(model.config.dim).eval()
        out = model(batch)
        idx = torch.arange(0, batch.n_tokens, 5)
        logits = head(out, model.embedding_weight, idx)
        ev = out.event_of_token[idx]
        cat = torch.cat(
            [out.token_repr[idx], out.history_event_repr[ev], out.user_repr[out.user_of_event[ev]]],
            dim=-1,
        )
        manual = head.proj(cat) @ model.embedding_weight.t()
        assert torch.allclose(logits, manual, atol=1e-6)


# ---------------------------------------------------------------- masking
class TestMasking:
    def test_rates_and_unk_exclusion(self, batch_and_vocab) -> None:
        batch, _ = batch_and_vocab
        masker = MaskingStrategy(p_token=0.15, p_event=0.10, p_key=0.10, p_unk=0.10)
        g = torch.Generator().manual_seed(0)
        masked = masker(batch, g)
        # only [MASK]/[UNK] substitutions; key tokens untouched
        changed = masked.input_value_ids != batch.value_ids
        assert changed.any()
        # every changed position is MASK or UNK
        sub = masked.input_value_ids[changed]
        assert set(sub.unique().tolist()) <= {MASK, UNK}
        # labels: -100 everywhere except predicted (MASK) positions
        pred = masked.labels != -100
        assert torch.equal(masked.input_value_ids[pred], torch.full((int(pred.sum()),), MASK))
        # UNK positions are excluded from loss
        unk_pos = masked.input_value_ids == UNK
        assert (masked.labels[unk_pos] == -100).all()
        # predicted labels equal original values
        assert torch.equal(masked.labels[pred], batch.value_ids[pred])

    def test_union_rate_reasonable(self, batch_and_vocab) -> None:
        batch, _ = batch_and_vocab
        masker = MaskingStrategy()
        g = torch.Generator().manual_seed(1)
        masked = masker(batch, g)
        frac = (masked.mask_type != -1).float().mean().item()
        # union of .15/.10/.10 minus the 10% UNK carve-out → roughly 0.2-0.35
        assert 0.10 < frac < 0.45

    def test_mask_types_present(self, batch_and_vocab) -> None:
        batch, _ = batch_and_vocab
        masker = MaskingStrategy(p_token=0.2, p_event=0.2, p_key=0.2)
        g = torch.Generator().manual_seed(2)
        masked = masker(batch, g)
        types = set(masked.mask_type.unique().tolist())
        assert T_TOKEN in types and T_KEY in types and T_EVENT in types

    def test_deterministic(self, batch_and_vocab) -> None:
        batch, _ = batch_and_vocab
        masker = MaskingStrategy()
        a = masker(batch, torch.Generator().manual_seed(7))
        b = masker(batch, torch.Generator().manual_seed(7))
        assert torch.equal(a.input_value_ids, b.input_value_ids)
        assert torch.equal(a.labels, b.labels)


# ---------------------------------------------------------------- TimeRoPE
class TestTimeRoPE:
    def test_gradcheck(self) -> None:
        torch.manual_seed(0)
        rope = TimeRoPE(head_dim=8).double()
        x = torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True)  # [H, L, hd]
        pos = torch.tensor([0.0, 1.5, 3.0, 7.0, 12.0], dtype=torch.double)
        cos, sin = rope.angles(pos)

        def fn(inp: torch.Tensor) -> torch.Tensor:
            return rope.rotate(inp, cos, sin)

        assert torch.autograd.gradcheck(fn, (x,), atol=1e-6)

    def test_rotation_preserves_norm(self) -> None:
        rope = TimeRoPE(head_dim=16)
        x = torch.randn(3, 4, 16)
        pos = torch.tensor([0.0, 5.0, 10.0, 20.0])
        cos, sin = rope.angles(pos)
        y = rope.rotate(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)

    def test_zero_position_is_identity(self) -> None:
        rope = TimeRoPE(head_dim=16)
        x = torch.randn(2, 3, 16)
        cos, sin = rope.angles(torch.zeros(3))
        assert torch.allclose(rope.rotate(x, cos, sin), x, atol=1e-6)


# ---------------------------------------------------------------- equivalence
class TestUserEquivalence:
    """No cross-user contamination: per-user embedding == embedding in a big batch."""

    def test_user_repr_independent_of_batch(self, batch_and_vocab, tmp_path_factory) -> None:
        data = tmp_path_factory.mktemp("p4_eq")
        generate(WorldConfig(n_users=20, months=14, n_merchants=500, mule_ring_count=1, seed=8,
                             eval_month_credit=2, eval_month_short=8),
                 data, n_workers=0, write_report=False)
        tok = PragmaTokenizer(TokenizerConfig(target_vocab=3000, n_buckets=32,
                                              categorical_threshold=200, seed=0)).fit(data)
        recs = [tok.encode(r) for r in iter_user_records(data, max_users=4)]
        model = PragmaModel(ModelConfig.preset("small", tok.vocab_size)).eval()
        collate = VarlenCollator()
        with torch.no_grad():
            full = model(collate(recs)).user_repr  # [4, d]
            for i, r in enumerate(recs):
                solo = model(collate([r])).user_repr  # [1, d]
                assert torch.allclose(full[i], solo[0], atol=1e-4), f"user {i} contaminated"


# ---------------------------------------------------------------- LoRA
class TestLoRA:
    def test_inject_and_merge_roundtrip(self, batch_and_vocab) -> None:
        batch, vocab = batch_and_vocab
        model = PragmaModel(ModelConfig.preset("small", vocab)).eval()
        with torch.no_grad():
            before = model.embed_users(batch)
        n = inject_lora(model, rank=8, alpha=8)
        assert n > 0
        # LoRA init (B=0) → identical output right after injection
        with torch.no_grad():
            after = model.embed_users(batch)
        assert torch.allclose(before, after, atol=1e-5)
        # perturb LoRA, merge, and confirm merged == unmerged forward
        for p in model.parameters():
            if p.requires_grad:
                p.data.add_(torch.randn_like(p) * 0.01)
        with torch.no_grad():
            pre_merge = model.embed_users(batch)
        merge_lora(model)
        with torch.no_grad():
            post_merge = model.embed_users(batch)
        assert torch.allclose(pre_merge, post_merge, atol=1e-4)

    def test_backward_updates_only_lora(self, batch_and_vocab) -> None:
        batch, vocab = batch_and_vocab
        model = PragmaModel(ModelConfig.preset("small", vocab))
        from pragmatiq.models.lora import mark_only_lora_trainable

        inject_lora(model, rank=4)
        mark_only_lora_trainable(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert all("lora" in name for name, p in model.named_parameters() if p.requires_grad)
        assert len(trainable) > 0


# ------------------------------------------------------- model invariant guards
@pytest.fixture(scope="module")
def model_bits(tmp_path_factory: pytest.TempPathFactory):
    """A small tokenizer + a handful of encoded records (for index/gather tests)."""
    data = tmp_path_factory.mktemp("v1_model")
    generate(WorldConfig(n_users=40, months=14, n_merchants=500, mule_ring_count=1, seed=11,
                         eval_month_credit=2, eval_month_short=8),
             data, n_workers=0, write_report=False)
    tok = PragmaTokenizer(TokenizerConfig(target_vocab=3000, n_buckets=32,
                                          categorical_threshold=200, seed=0)).fit(data)
    recs = [tok.encode(r) for r in iter_user_records(data, max_users=5)]
    return tok, recs


class TestForwardDeterminism:
    """The full forward is bit-exact across calls (pins the index_copy_ assembly)."""

    def test_forward_bit_exact_across_calls(self, model_bits) -> None:
        tok, recs = model_bits
        batch = VarlenCollator()(recs)
        torch.manual_seed(0)
        model = PragmaModel(ModelConfig.preset("small", tok.vocab_size)).eval()
        with torch.no_grad():
            a = model.embed_users(batch)
            b = model.embed_users(batch)
        assert torch.equal(a, b), f"forward not bit-exact: maxdiff {(a - b).abs().max():.2e}"


class TestMLMGatherAlignment:
    """Pin the load-bearing MLM gather: event/USR row ordering must match cu_seqlens."""

    def test_index_maps_match_cu_seqlens(self, model_bits) -> None:
        tok, recs = model_bits
        batch = VarlenCollator()(recs)
        model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size)).eval()
        with torch.no_grad():
            out = model(batch)
        cu_h = batch.cu_seqlens_history
        exp_uoe = torch.repeat_interleave(torch.arange(batch.n_users), (cu_h[1:] - cu_h[:-1]).long())
        assert torch.equal(out.user_of_event, exp_uoe)
        cu_e = batch.cu_seqlens_event
        exp_eot = torch.repeat_interleave(torch.arange(batch.n_events), (cu_e[1:] - cu_e[:-1]).long())
        assert torch.equal(out.event_of_token, exp_eot)

    def test_per_user_history_and_user_repr_isolated(self, model_bits) -> None:
        tok, recs = model_bits
        model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size)).eval()
        collate = VarlenCollator()
        full_batch = collate(recs)
        cu_h = full_batch.cu_seqlens_history
        with torch.no_grad():
            full = model(full_batch)
            for i, r in enumerate(recs):
                solo = model(collate([r]))
                assert torch.allclose(full.user_repr[i], solo.user_repr[0], atol=1e-4), f"user {i}"
                lo, hi = int(cu_h[i]), int(cu_h[i + 1])
                assert torch.allclose(full.history_event_repr[lo:hi], solo.history_event_repr,
                                      atol=1e-4), f"event rows for user {i} misaligned"


class TestZeroEventBatch:
    """A batch of only profile-only (cold-start / fully-truncated) users must not crash."""

    def test_profile_only_batch_embeds_finite(self, model_bits) -> None:
        tok, recs = model_bits
        # cutoff before each user's first event → zero events kept (profile only)
        zrecs = [truncate_record(r, int(r.event_ts[0])) for r in recs if len(r.event_ts)]
        batch = VarlenCollator()(zrecs)
        assert batch.n_events == 0
        model = PragmaModel(ModelConfig.preset("nano", tok.vocab_size)).eval()
        with torch.no_grad():
            out = model(batch)
        assert out.user_repr.shape[0] == batch.n_users
        assert torch.isfinite(out.user_repr).all()
        # masking on a zero-token batch must select nothing without raising
        masked = MaskingStrategy()(batch, torch.Generator().manual_seed(0))
        assert masked.n_selected == 0


class TestMaskingUnkInvariants:
    def test_unk_excluded_and_rate(self, model_bits) -> None:
        tok, recs = model_bits
        batch = VarlenCollator()(recs)
        n_unk = n_sub = 0
        for s in range(8):
            m = MaskingStrategy(p_unk=0.10)(batch, torch.Generator().manual_seed(s))
            changed = m.input_value_ids != batch.value_ids
            unk_sub = changed & (m.input_value_ids == UNK)
            assert (m.labels[unk_sub] == -100).all()          # UNK excluded from loss
            assert (m.mask_type[unk_sub] == T_NONE).all()     # and from per-type accounting
            n_unk += int(unk_sub.sum())
            n_sub += int(changed.sum())
        frac = n_unk / max(n_sub, 1)
        assert 0.06 < frac < 0.15, f"UNK fraction of substituted = {frac:.3f}, expected ~0.10"


class TestTimeRoPERealistic:
    def test_relative_invariance_at_realistic_positions(self) -> None:
        rope = TimeRoPE(head_dim=16).double()
        q = torch.randn(2, 1, 16, dtype=torch.double)
        k = torch.randn(2, 1, 16, dtype=torch.double)
        gap = 5.0
        sims = []
        for p in (0.0, 74.0, 127.0):
            cq = rope.angles(torch.tensor([p], dtype=torch.double))
            ck = rope.angles(torch.tensor([p + gap], dtype=torch.double))
            sims.append(float((rope.rotate(q, *cq) * rope.rotate(k, *ck)).sum()))
        assert max(sims) - min(sims) < 1e-6  # <rotate(q,p),rotate(k,p+g)> depends only on g

    def test_time_axis_discriminates(self) -> None:
        # the time axis must actually move embeddings at realistic log-seconds
        rope = TimeRoPE(head_dim=32)
        x = torch.randn(1, 1, 32)
        late = float(time_encode(np.array([6 * 30 * 24 * 3600.0]))[0])  # ~6 months
        v0 = rope.rotate(x, *rope.angles(torch.tensor([0.0])))
        vl = rope.rotate(x, *rope.angles(torch.tensor([late])))
        sim = float(torch.cosine_similarity(v0.flatten(), vl.flatten(), dim=0))
        assert sim < 0.85, f"time axis nearly dead (cos-sim {sim:.3f}); check rope_base/head_dim"

    def test_rope_pos_is_fp32_under_bf16(self) -> None:
        seg = torch.tensor([2, 1])
        prefix = torch.randn(2, 8, dtype=torch.bfloat16)
        toks = torch.randn(3, 8, dtype=torch.bfloat16)
        tpos = torch.tensor([10.5, 74.3, 101.5])
        _, _, _, _, rope_pos = assemble_segments(seg, prefix, toks, token_pos=tpos,
                                                 prefix_pos=torch.zeros(2))
        assert rope_pos.dtype == torch.float32  # positions never quantized to bf16
        got = torch.sort(rope_pos[rope_pos != 0]).values
        assert torch.allclose(got, torch.tensor([10.5, 74.3, 101.5]), atol=1e-5)
