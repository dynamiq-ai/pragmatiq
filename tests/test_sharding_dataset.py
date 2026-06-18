"""Sharding tests: shard round-trip, dynamic batching, resumability, padding-equivalence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pragmatiq.data.collate import VarlenCollator, block_diag_mask, run_with_oom_retry, segment_ids
from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader, ShardDataset
from pragmatiq.data.sharding import ShardWriter, UserIndex, band_of
from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig, iter_user_records
from pragmatiq.models.embeddings import TimeRoPE
from pragmatiq.models.layers import varlen_self_attention


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, PragmaTokenizer]:
    data = tmp_path_factory.mktemp("p3_data")
    generate(
        WorldConfig(n_users=300, months=16, n_merchants=1200, mule_ring_count=1, seed=21,
                    eval_month_credit=4, eval_month_short=9),
        data, n_workers=0, write_report=False,
    )
    tok = PragmaTokenizer(TokenizerConfig(target_vocab=5000, n_buckets=32,
                                          categorical_threshold=200, seed=0)).fit(data)
    shard_dir = tmp_path_factory.mktemp("p3_shards")
    writer = ShardWriter(shard_dir, tokenizer_hash=tok.content_hash, rows_per_shard=64)
    for rec in iter_user_records(data, max_users=300):
        writer.add(tok.encode(rec),
                   profile={"attributes": rec.attributes, "lifelong": rec.lifelong, "as_of": rec.as_of})
    manifest = writer.close()
    assert manifest["n_users"] == 300
    return shard_dir, tok


class TestBands:
    def test_band_assignment(self) -> None:
        assert band_of(1) == 0
        assert band_of(8) == 0
        assert band_of(9) == 1
        assert band_of(10_000_000) == 6


class TestShardRoundTrip:
    def test_index_complete(self, shards) -> None:
        shard_dir, _ = shards
        idx = UserIndex(shard_dir)
        assert len(idx) == 300
        assert idx.n_tokens.min() > 0
        assert idx.n_prof_tokens.min() > 0
        idx.close()

    def test_profile_blob_stored(self, shards) -> None:
        shard_dir, _ = shards
        idx = UserIndex(shard_dir)
        prof = idx.profile(idx.order[3])
        assert prof is not None
        assert "attributes" in prof and "lifelong" in prof
        assert prof["attributes"].get("country")  # raw profile recoverable from LMDB
        idx.close()


class TestOOMRetry:
    def test_halves_budget_and_succeeds(self, caplog) -> None:
        import logging

        calls: list[int] = []

        def fn(budget: int) -> str:
            calls.append(budget)
            if budget > 4096:  # simulate OOM until the budget is small enough
                raise torch.cuda.OutOfMemoryError("synthetic OOM")
            return "ok"

        with caplog.at_level(logging.WARNING):
            result, used = run_with_oom_retry(fn, token_budget=16384, min_budget=256)
        assert result == "ok"
        assert used == 4096
        assert calls == [16384, 8192, 4096]
        assert any("halving" in r.message for r in caplog.records)

    def test_reraises_below_floor(self) -> None:
        def always_oom(budget: int) -> str:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")

        with pytest.raises(torch.cuda.OutOfMemoryError):
            run_with_oom_retry(always_oom, token_budget=512, min_budget=256)

    def test_record_round_trip(self, shards) -> None:
        shard_dir, tok = shards
        ds = ShardDataset(shard_dir)
        # reconstruct one user and compare to a fresh encode of the same raw record
        uid = ds.user_ids[5]
        rec = ds.get(uid)
        assert rec.user_id == uid
        assert rec.n_tokens == int(rec.key_ids.size)
        assert rec.event_offsets[-1] == rec.n_tokens
        assert rec.event_offsets[0] == 0
        assert np.all(np.diff(rec.event_offsets) >= 0)
        ds.close()

    def test_dataset_matches_encode(self, shards, tmp_path_factory) -> None:
        shard_dir, tok = shards
        # re-derive raw records and ensure tokenize->shard->read is identity
        data = tmp_path_factory.mktemp("p3_data2")
        generate(
            WorldConfig(n_users=300, months=16, n_merchants=1200, mule_ring_count=1, seed=21,
                        eval_month_credit=4, eval_month_short=9),
            data, n_workers=0, write_report=False,
        )
        ds = ShardDataset(shard_dir)
        raw = {r.user_id: r for r in iter_user_records(data, max_users=300)}
        for uid in ds.user_ids[:5]:
            stored = ds.get(uid)
            fresh = tok.encode(raw[uid])
            assert np.array_equal(stored.key_ids, fresh.key_ids)
            assert np.array_equal(stored.value_ids, fresh.value_ids)
            assert np.array_equal(stored.event_offsets, fresh.event_offsets)
            assert np.allclose(stored.time_log, fresh.time_log, atol=1e-5)
        ds.close()


class TestEmbedTextRoundTrip:
    """Nemotron variant: is_text/text_values survive tokenize → shard → reload, and
    BPE-mode shards carry no text columns (byte-identical to a no-variant build)."""

    def test_round_trip_and_bpe_has_no_text_columns(self, tmp_path_factory) -> None:
        import copy
        import dataclasses
        import glob

        import pyarrow.parquet as pq

        data = tmp_path_factory.mktemp("embed_data")
        generate(
            WorldConfig(n_users=200, months=16, n_merchants=1500, mule_ring_count=1, seed=31,
                        eval_month_credit=4, eval_month_short=9),
            data, n_workers=0, write_report=False,
        )
        bpe = PragmaTokenizer(TokenizerConfig(target_vocab=5000, n_buckets=32,
                                              categorical_threshold=200, seed=0)).fit(data)
        assert any(v == "text" for v in bpe.field_kind.values())
        emb = copy.copy(bpe)  # share the fitted vocab; switch only the text pathway
        emb.config = dataclasses.replace(bpe.config, text_value_mode="embed")

        # embed mode: write + read back, text state must match a fresh encode exactly
        embed_dir = tmp_path_factory.mktemp("embed_shards")
        writer = ShardWriter(embed_dir, tokenizer_hash=emb.content_hash, rows_per_shard=64)
        fresh = {r.user_id: emb.encode(r) for r in iter_user_records(data, max_users=200)}
        for t in fresh.values():
            writer.add(t)
        writer.close()
        ds = ShardDataset(embed_dir)
        textful = [uid for uid, t in fresh.items() if int(t.is_text.sum()) > 0]
        assert textful  # the variant is actually exercised
        for uid in textful[:5]:
            stored, want = ds.get(uid), fresh[uid]
            assert np.array_equal(stored.is_text, want.is_text)
            assert stored.text_values == want.text_values
            assert len(stored.text_values) == int(stored.is_text.sum())  # compact invariant
        ds.close()

        # BPE mode: shards must not even contain the text columns
        bpe_dir = tmp_path_factory.mktemp("bpe_shards")
        w2 = ShardWriter(bpe_dir, tokenizer_hash=bpe.content_hash, rows_per_shard=64)
        for r in iter_user_records(data, max_users=200):
            w2.add(bpe.encode(r))
        w2.close()
        for path in glob.glob(str(bpe_dir / "shards" / "*.parquet")):
            cols = pq.read_table(path).column_names
            assert "is_text" not in cols and "text_values" not in cols


class TestDynamicSampler:
    def test_token_budget_respected(self, shards) -> None:
        shard_dir, _ = shards
        idx = UserIndex(shard_dir)
        budget = 4096
        sampler = DynamicBatchSampler(idx, token_budget=budget, seed=1)
        sampler.set_epoch(0)
        seen = 0
        for batch in sampler:
            toks = int(idx.n_tokens[batch].sum()) + len(batch)
            # a batch is within budget unless it is a single oversized user
            assert toks <= budget or len(batch) == 1
            seen += len(batch)
        assert seen == len(idx)  # every user covered exactly once
        idx.close()

    def test_all_users_once(self, shards) -> None:
        shard_dir, _ = shards
        idx = UserIndex(shard_dir)
        sampler = DynamicBatchSampler(idx, token_budget=8192, seed=2)
        sampler.set_epoch(0)
        allu = [i for b in sampler for i in b]
        assert sorted(allu) == list(range(len(idx)))
        idx.close()

    def test_subset_restricts_to_cohort(self, shards) -> None:
        shard_dir, _ = shards
        idx = UserIndex(shard_dir)
        n = len(idx)
        subset = list(range(0, n, 3))  # every third user
        sampler = DynamicBatchSampler(idx, token_budget=8192, seed=2, subset=subset)
        sampler.set_epoch(0)
        seen: list[int] = []
        for batch in sampler:
            toks = int(idx.n_tokens[batch].sum()) + len(batch)
            assert toks <= 8192 or len(batch) == 1  # budget still honored
            seen.extend(batch)
        assert sorted(seen) == sorted(set(subset))  # exactly the cohort, once each

        # same seed + subset -> identical plan (rule 2)
        s2 = DynamicBatchSampler(idx, token_budget=8192, seed=2, subset=subset)
        assert s2._plan(0) == sampler._plan(0)
        idx.close()

    def test_ddp_replica_sharding(self, shards) -> None:
        import math

        shard_dir, _ = shards
        idx = UserIndex(shard_dir)
        plan = DynamicBatchSampler(idx, token_budget=4096, seed=3)._plan(0)
        n = len(plan)
        assert n >= 2  # fixture forms several batches at this budget
        nrep = 2
        ranks = [
            DynamicBatchSampler(idx, token_budget=4096, seed=3, num_replicas=nrep, rank=r)._plan(0)
            for r in range(nrep)
        ]
        # Equal step count per rank → DDP all-reduce stays in lockstep (a ragged
        # split would deadlock); each rank trains a proper slice, not the whole.
        per = math.ceil(n / nrep)
        assert all(len(rp) == per for rp in ranks)
        assert all(len(rp) < n for rp in ranks)
        # Union covers every batch of the single-replica plan (no user dropped).
        union = {tuple(b) for rp in ranks for b in rp}
        assert {tuple(b) for b in plan}.issubset(union)
        # num_replicas=1 (default / single-process / CPU) is the identity plan.
        assert DynamicBatchSampler(idx, token_budget=4096, seed=3, num_replicas=1)._plan(0) == plan
        idx.close()

    def test_replica_info_persists_through_resume(self, shards) -> None:
        shard_dir, _ = shards
        idx = UserIndex(shard_dir)
        s = DynamicBatchSampler(idx, token_budget=4096, seed=7, num_replicas=2, rank=1)
        s.set_epoch(0)
        s2 = DynamicBatchSampler(idx, token_budget=4096, seed=7)
        s2.load_state_dict(s.state_dict())
        assert (s2.num_replicas, s2.rank) == (2, 1)
        assert s2._plan(0) == s._plan(0)
        idx.close()

    def test_resume_matches_uninterrupted(self, shards) -> None:
        shard_dir, _ = shards
        idx = UserIndex(shard_dir)
        full = DynamicBatchSampler(idx, token_budget=4096, seed=7)
        full.set_epoch(0)
        all_batches = [list(b) for b in full._plan(0)]

        # consume 3 batches, snapshot, restore into a fresh sampler, continue
        s = DynamicBatchSampler(idx, token_budget=4096, seed=7)
        s.set_epoch(0)
        it = iter(s)
        consumed = [next(it) for _ in range(3)]
        state = s.state_dict()
        s2 = DynamicBatchSampler(idx, token_budget=4096, seed=7)
        s2.load_state_dict(state)
        rest = [list(b) for b in s2]
        assert consumed + rest == all_batches
        idx.close()


class TestPaddingEquivalence:
    """The packing contract: varlen (packed) attention == padded attention."""

    def _mha(self, x: torch.Tensor, qkv_w: torch.Tensor, attn_mask: torch.Tensor,
             n_heads: int) -> torch.Tensor:
        # x: [L, D]; reference multi-head self-attention with an additive mask
        L, D = x.shape
        qkv = x @ qkv_w  # [L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        hd = D // n_heads
        q = q.view(L, n_heads, hd).transpose(0, 1)  # [H, L, hd]
        k = k.view(L, n_heads, hd).transpose(0, 1)
        v = v.view(L, n_heads, hd).transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) / (hd**0.5)  # [H, L, L]
        scores = scores + attn_mask[None]
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # [H, L, hd]
        return out.transpose(0, 1).reshape(L, D)

    def test_block_diag_mask_matches_padded(self, shards) -> None:
        torch.manual_seed(0)
        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)
        batch = VarlenCollator()(ds.get_many(ds.user_ids[:4]))

        # The event encoder attends WITHIN each event only, so equivalence is a
        # local property — validate it on a bounded slice (first events up to
        # ~256 tokens) to keep the dense O(T^2) reference tiny.
        cu_full = batch.cu_seqlens_event
        n_keep = 1
        while n_keep < batch.n_events and int(cu_full[n_keep]) <= 256:
            n_keep += 1
        cu = cu_full[: n_keep + 1].clone()
        T = int(cu[-1])
        assert T >= 8 and n_keep >= 2  # meaningful slice

        D, H = 32, 4
        emb = torch.nn.Embedding(int(max(batch.key_ids.max(), batch.value_ids.max())) + 1, D)
        qkv_w = torch.randn(D, 3 * D, dtype=torch.float32) * 0.1
        x = (emb(batch.key_ids[:T]) + emb(batch.value_ids[:T])).float()  # [T, D]

        # ---- packed: one flat sequence, block-diagonal attention over events
        mask = block_diag_mask(cu, T)
        packed_out = self._mha(x, qkv_w, mask, H)

        # ---- padded: each event attended independently (the naive reference)
        for e in range(n_keep):
            lo, hi = int(cu[e]), int(cu[e + 1])
            ref = self._mha(x[lo:hi], qkv_w, torch.zeros(hi - lo, hi - lo), H)
            assert torch.allclose(packed_out[lo:hi], ref, atol=1e-4), f"event {e} mismatch"
        ds.close()

    def test_segment_ids_consistent(self, shards) -> None:
        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)
        batch = VarlenCollator()(ds.get_many(ds.user_ids[:4]))
        seg = segment_ids(batch.cu_seqlens_event, batch.n_tokens)
        # event_of_token from the collator must agree with cu_seqlens segmentation
        assert torch.equal(seg, batch.event_of_token)
        ds.close()

    def test_segment_ids_with_empty_segment(self) -> None:
        # a zero-length middle segment must not collapse later ids (cu=[0,3,3,5])
        cu = torch.tensor([0, 3, 3, 5], dtype=torch.int32)
        assert segment_ids(cu, 5).tolist() == [0, 0, 0, 2, 2]


class TestVarlenAttentionEquivalence:
    """The 'critical' property on the PRODUCTION path: varlen_self_attention
    (the real SDPA scatter/gather code) == a naive padded per-segment attention."""

    def _ref_per_segment(self, q, k, v, cu, rope=None, rope_pos=None):
        outs = []
        for i in range(cu.numel() - 1):
            lo, hi = int(cu[i]), int(cu[i + 1])
            if hi == lo:
                continue
            qs, ks, vs = q[lo:hi], k[lo:hi], v[lo:hi]  # [L, H, hd]
            if rope is not None:
                cos, sin = rope.angles(rope_pos[lo:hi])
                qs = rope.rotate(qs.transpose(0, 1), cos, sin).transpose(0, 1)
                ks = rope.rotate(ks.transpose(0, 1), cos, sin).transpose(0, 1)
            o = F.scaled_dot_product_attention(qs.transpose(0, 1), ks.transpose(0, 1),
                                               vs.transpose(0, 1))  # [H, L, hd], full bidir
            outs.append(o.transpose(0, 1))
        return torch.cat(outs, dim=0)

    @pytest.mark.parametrize("seglens", [[3, 1, 5, 2], [7, 7, 7], [1, 9, 2], [4, 0, 3]])
    def test_varlen_matches_padded(self, seglens: list[int]) -> None:
        torch.manual_seed(0)
        H, hd = 3, 8
        lens = torch.tensor(seglens)
        cu = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).to(torch.int32)
        T, max_seqlen = int(cu[-1]), int(lens.max())
        q, k, v = (torch.randn(T, H, hd) for _ in range(3))
        out = varlen_self_attention(q, k, v, cu, max_seqlen)
        ref = self._ref_per_segment(q, k, v, cu)
        assert torch.allclose(out, ref, atol=1e-4), f"no-rope mismatch for {seglens}"
        rope = TimeRoPE(head_dim=hd)
        rope_pos = torch.rand(T) * 120.0  # realistic log-seconds range
        out2 = varlen_self_attention(q, k, v, cu, max_seqlen, rope=rope, rope_pos=rope_pos)
        ref2 = self._ref_per_segment(q, k, v, cu, rope=rope, rope_pos=rope_pos)
        assert torch.allclose(out2, ref2, atol=1e-4), f"rope mismatch for {seglens}"

    def test_undersized_max_seqlen_raises(self) -> None:
        q = torch.randn(5, 2, 4)
        cu = torch.tensor([0, 5], dtype=torch.int32)
        with pytest.raises(ValueError, match="max_seqlen"):
            varlen_self_attention(q, q, q, cu, max_seqlen=3)


class TestLoader:
    def test_loader_yields_packed_batches(self, shards) -> None:
        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)
        # reuse the dataset's own index (LMDB forbids a second open per process)
        sampler = DynamicBatchSampler(ds.index, token_budget=6000, seed=3)
        sampler.set_epoch(0)
        loader = ShardDataLoader(ds, sampler)
        total_users = 0
        for batch in loader:
            assert batch.n_users >= 1
            assert batch.cu_seqlens_event[-1] == batch.n_tokens
            assert batch.cu_seqlens_history[-1] == batch.n_events
            total_users += batch.n_users
        assert total_users == len(ds)
        ds.close()
