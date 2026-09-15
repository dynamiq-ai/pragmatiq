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


def _reference_collate_arrays(records) -> dict[str, np.ndarray]:
    """The pre-1.1 per-event / per-item Python loop of VarlenCollator, kept as the oracle."""
    key_ids, value_ids, positions, event_of_token, is_text_parts = [], [], [], [], []
    evt_lens: list[int] = []
    p_key, p_val, p_pos, item_of_ptok, prof_item_lens, prof_per_user, user_of_pitem = [], [], [], [], [], [], []
    global_event = global_pitem = 0
    for u, rec in enumerate(records):
        it_full = rec.is_text
        if it_full.size != rec.key_ids.size or len(rec.text_values) != int(it_full.sum()):
            it_full = np.zeros(rec.key_ids.size, dtype=np.int8)
        for e in range(rec.n_events):
            lo, hi = int(rec.event_offsets[e]), int(rec.event_offsets[e + 1])
            evt_lens.append(hi - lo)
            key_ids.append(rec.key_ids[lo:hi])
            value_ids.append(rec.value_ids[lo:hi])
            positions.append(rec.positions[lo:hi])
            is_text_parts.append(it_full[lo:hi])
            event_of_token.append(np.full(hi - lo, global_event, dtype=np.int64))
            global_event += 1
        n_items = len(rec.prof_offsets) - 1
        prof_per_user.append(n_items)
        for it in range(n_items):
            lo, hi = int(rec.prof_offsets[it]), int(rec.prof_offsets[it + 1])
            prof_item_lens.append(hi - lo)
            p_key.append(rec.prof_key_ids[lo:hi])
            p_val.append(rec.prof_value_ids[lo:hi])
            p_pos.append(rec.prof_positions[lo:hi])
            item_of_ptok.append(np.full(hi - lo, global_pitem, dtype=np.int64))
            global_pitem += 1
        user_of_pitem.append(np.full(n_items, u, dtype=np.int64))

    def cat(parts, dtype=np.int64):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    def cu(lens):
        out = np.zeros(len(lens) + 1, dtype=np.int32)
        np.cumsum(np.asarray(lens, dtype=np.int32), out=out[1:])
        return out

    return {"key_ids": cat(key_ids), "value_ids": cat(value_ids), "positions": cat(positions),
            "event_of_token": cat(event_of_token), "cu_seqlens_event": cu(evt_lens),
            "is_text": cat(is_text_parts, np.int8).astype(bool),
            "prof_key_ids": cat(p_key), "prof_value_ids": cat(p_val), "prof_positions": cat(p_pos),
            "item_of_prof_token": cat(item_of_ptok), "cu_seqlens_profile_item": cu(prof_item_lens),
            "cu_seqlens_profile": cu(prof_per_user), "user_of_prof_item": cat(user_of_pitem)}


class TestCollatorVectorized:
    def test_matches_reference_loop(self, shards) -> None:
        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)
        for uids in (ds.user_ids[:5], ds.user_ids[40:73], ds.user_ids[290:300]):
            recs = ds.get_many(uids)
            batch = VarlenCollator()(recs)
            ref = _reference_collate_arrays(recs)
            for name, arr in ref.items():
                got = getattr(batch, name).numpy()
                assert np.array_equal(got, arr), name
        ds.close()

    def test_max_len_fields_match_cu_seqlens(self, shards) -> None:
        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)
        sampler = DynamicBatchSampler(ds.index, token_budget=2048, seed=1)
        sampler.set_epoch(0)
        for i, batch in enumerate(ShardDataLoader(ds, sampler)):
            ce = batch.cu_seqlens_event
            ch = batch.cu_seqlens_history
            assert batch.max_len_event == (int((ce[1:] - ce[:-1]).max()) + 1 if ce.numel() > 1 else 1)
            assert batch.max_len_history == int((ch[1:] - ch[:-1]).max()) + 1
            item_len = (batch.cu_seqlens_profile_item[1:] - batch.cu_seqlens_profile_item[:-1]).numpy()
            per_user = np.bincount(batch.user_of_prof_item.numpy(), weights=item_len, minlength=batch.n_users)
            assert batch.max_len_profile == int(per_user.max()) + 1
            if i >= 6:
                break
        ds.close()

    def test_malformed_offsets_rejected(self, shards) -> None:
        import dataclasses

        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)
        rec = ds.get(ds.user_ids[0])
        ds.close()
        bad = dataclasses.replace(rec, event_offsets=rec.event_offsets[:-1])
        with pytest.raises(ValueError, match="event_offsets"):
            VarlenCollator()([bad])


class TestPrefetchLoader:
    def _make(self, shard_dir, prefetch: int, pin: bool = False):
        ds = ShardDataset(shard_dir)
        sampler = DynamicBatchSampler(ds.index, token_budget=2048, seed=0)
        sampler.set_epoch(0)
        return ds, ShardDataLoader(ds, sampler, prefetch=prefetch, pin_memory=pin)

    @staticmethod
    def _same(a, b) -> bool:
        if a.user_ids != b.user_ids or a.text_values != b.text_values:
            return False
        for f, v in a.__dict__.items():
            if isinstance(v, torch.Tensor) and not torch.equal(v, getattr(b, f)):
                return False
        return True

    def test_stream_and_state_match_sync_loader(self, shards) -> None:
        shard_dir, _ = shards
        ds_a, sync = self._make(shard_dir, 0)
        ds_b, pre = self._make(shard_dir, 3)
        it_a, it_b = iter(sync), iter(pre)
        n = len(sync)
        for _ in range(n):
            a, b = next(it_a), next(it_b)
            assert self._same(a, b)
            assert sync.state_dict() == pre.state_dict()
        with pytest.raises(StopIteration):
            next(it_a)
        with pytest.raises(StopIteration):
            next(it_b)
        assert sync.state_dict() == pre.state_dict()  # epoch rolled over identically
        ds_a.close()
        ds_b.close()

    def test_resume_from_prefetch_snapshot_matches_sync(self, shards) -> None:
        shard_dir, _ = shards
        ds_c, pre = self._make(shard_dir, 2)
        it = iter(pre)
        for _ in range(3):
            next(it)
        snap = pre.state_dict()
        it.close()  # abandon mid-epoch: the producer thread must stop, not hang
        ds_d, sync2 = self._make(shard_dir, 0)
        sync2.load_state_dict(snap)
        ds_e, pre2 = self._make(shard_dir, 2)
        pre2.load_state_dict(snap)
        rest_sync = [b.user_ids for b in sync2]
        rest_pre = [b.user_ids for b in pre2]
        assert rest_sync == rest_pre
        assert len(rest_sync) == len(pre) - 3
        ds_c.close()
        ds_d.close()
        ds_e.close()

    def test_producer_error_is_raised_in_consumer(self, shards, monkeypatch) -> None:
        shard_dir, _ = shards
        ds, pre = self._make(shard_dir, 2)
        monkeypatch.setattr(ds, "get_many", lambda uids: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            next(iter(pre))
        ds.close()

    def test_pin_memory_and_non_blocking_to(self, shards) -> None:
        shard_dir, _ = shards
        ds, pre = self._make(shard_dir, 1, pin=torch.cuda.is_available())
        batch = next(iter(pre))
        moved = batch.to("cpu", non_blocking=True)
        assert self._same(batch, moved)
        ds.close()


class TestShardWriterRefactors:
    """Arrow-native shard columns and incremental profile puts leave the outputs unchanged."""

    def test_large_list_columns_equal_python_list_construction(self, shards) -> None:
        import pyarrow as pa

        from pragmatiq.data.sharding import _ARRAY_FIELDS, SHARD_SCHEMA, _large_list, _record_to_arrays

        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)
        recs = [_record_to_arrays(ds.get(u)) for u in ds.user_ids[:37]]
        ds.close()
        cols_new: dict = {"user_id": pa.array([f"u{i}" for i in range(len(recs))], type=pa.string())}
        cols_old: dict = {"user_id": [f"u{i}" for i in range(len(recs))]}
        for name, t in _ARRAY_FIELDS:
            cols_new[name] = _large_list([r[name] for r in recs], t)
            cols_old[name] = [r[name].tolist() for r in recs]
        new = pa.table(cols_new, schema=SHARD_SCHEMA)
        old = pa.table(cols_old, schema=SHARD_SCHEMA)
        assert new.equals(old)

    def test_duplicate_user_raises_at_add(self, shards, tmp_path: Path) -> None:
        shard_dir, tok = shards
        ds = ShardDataset(shard_dir)
        rec = ds.get(ds.user_ids[0])
        ds.close()
        w = ShardWriter(tmp_path / "dup", tokenizer_hash=tok.content_hash, rows_per_shard=8)
        w.add(rec)
        with pytest.raises(ValueError, match="duplicate user_id"):
            w.add(rec)

    def test_profiles_written_incrementally_and_index_complete(self, shards, tmp_path: Path) -> None:
        shard_dir, tok = shards
        src = ShardDataset(shard_dir)
        uids = src.user_ids[:20]
        recs = [src.get(u) for u in uids]
        w = ShardWriter(tmp_path / "inc", tokenizer_hash=tok.content_hash, rows_per_shard=4)
        for rec in recs:
            w.add(rec, profile={"attributes": {"country": "GB"}, "lifelong": [], "as_of": 0})
        manifest = w.close()
        assert manifest["n_users"] == 20
        idx = UserIndex(tmp_path / "inc")
        assert idx.order == uids
        assert all(idx.profile(u) == {"attributes": {"country": "GB"}, "lifelong": [], "as_of": 0} for u in uids)
        metas = idx.meta_many(uids[::3])
        assert [m.user_id for m in metas] == uids[::3]
        assert metas[1] == idx.meta(uids[3])
        idx.close()
        src.close()

    def test_get_many_uses_one_index_transaction(self, shards, monkeypatch) -> None:
        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)
        calls = {"meta_many": 0, "meta": 0}
        real_meta_many = ds.index.meta_many

        def counting_meta_many(user_ids):
            calls["meta_many"] += 1
            return real_meta_many(user_ids)

        monkeypatch.setattr(ds.index, "meta_many", counting_meta_many)
        monkeypatch.setattr(ds.index, "meta", lambda uid: calls.__setitem__("meta", calls["meta"] + 1))
        out = ds.get_many(ds.user_ids[:50])
        assert [r.user_id for r in out] == ds.user_ids[:50]
        assert calls == {"meta_many": 1, "meta": 0}  # one batched index read, no per-user lookups
        ds.close()


class TestShardCacheBudget:
    def test_byte_budget_keeps_hot_shards_and_evicts_by_size(self, shards) -> None:
        shard_dir, _ = shards
        ds = ShardDataset(shard_dir)  # default: byte budget, not a shard count
        assert ds._cache_n is None and ds._cache_bytes is not None and ds._cache_bytes > 0
        keys = sorted({(m.band, m.shard) for m in ds.index.meta_many(ds.user_ids)})
        first = ds._shard_table(*keys[0])
        assert ds._cached_bytes == int(first.nbytes)
        tiny = ShardDataset(shard_dir, cache_bytes=1)  # every load evicts the previous one
        for k in keys:
            tiny._shard_table(*k)
        assert len(tiny._cache) == 1 and tiny._cached_bytes == int(tiny._cache[keys[-1]].nbytes)
        pinned = ShardDataset(shard_dir, cache_shards=2)
        for k in keys:
            pinned._shard_table(*k)
        assert len(pinned._cache) == min(2, len(keys))
        for d in (ds, tiny, pinned):
            d.close()
