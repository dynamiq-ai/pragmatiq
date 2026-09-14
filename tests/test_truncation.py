"""Eval-point truncation (no-hindcasting rule): unit + end-to-end tests.

Labels are outcomes of a window after ``eval_ts``; embeddings, probes, and
fine-tunes must therefore never see events at or past that point. These tests
pin the enforcement layer: ``truncate_record``, ``TruncatingCollator``, and
the api.probe wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from pragmatiq import api
from pragmatiq.core.schema import UserRecord
from pragmatiq.data.collate import TruncatingCollator, VarlenCollator
from pragmatiq.data.dataset import ShardDataset
from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig, time_encode, truncate_record


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> ShardDataset:
    work = tmp_path_factory.mktemp("trunc")
    api.synthesize({"n_users": 60, "seed": 7}, out=work / "ds", write_report=False)
    api.tokenize(work / "ds", work / "tok")
    ds = ShardDataset(work / "tok")
    yield ds
    ds.close()


class TestUnsortedEventsRobustness:
    """BYO / notebook records may arrive out of time order: encode must sort so the
    time encoding never goes NaN and truncation never leaks post-cutoff events."""

    def test_encode_sorts_and_clamps_time(self, tmp_path) -> None:
        api.synthesize({"n_users": 20, "seed": 1}, out=tmp_path / "d", write_report=False)
        tok = PragmaTokenizer(TokenizerConfig(target_vocab=2000, n_buckets=16,
                                              categorical_threshold=200, seed=0)).fit(tmp_path / "d")
        day = 86_400_000_000
        ev = [  # deliberately out of order
            (5 * day, "transaction", {"amount": "10.0", "mcc": "5411"}),
            (1 * day, "transaction", {"amount": "20.0", "mcc": "5411"}),
            (3 * day, "app", {"screen": "home"}),
        ]
        enc = tok.encode(UserRecord(user_id="u", events=ev, attributes={"x": "1"}, as_of=6 * day))
        assert list(enc.event_ts) == sorted(enc.event_ts)        # encode sorted ascending
        assert np.isfinite(enc.time_log).all()                   # no NaN/-inf time positions
        assert float(enc.time_log[-1]) == 0.0                    # most-recent event → delta 0
        out = truncate_record(enc, 4 * day)                      # keep ts < 4d (the 1d & 3d events)
        assert out.n_events == 2 and (out.event_ts.size == 0 or out.event_ts.max() < 4 * day)

    def test_missing_prof_ts_raises_rather_than_leaks(self, dataset: ShardDataset) -> None:
        import dataclasses
        rec = next(dataset.get(u) for u in dataset.user_ids[:5])
        no_prof_ts = dataclasses.replace(rec, prof_ts=np.zeros(0, dtype=np.int64))
        with pytest.raises(ValueError, match="prof_ts"):
            truncate_record(no_prof_ts, 10**18)


class TestTruncateRecord:
    def test_events_strictly_before_cutoff(self, dataset: ShardDataset) -> None:
        rec = max((dataset.get(u) for u in dataset.user_ids[:20]), key=lambda r: r.n_events)
        assert rec.n_events >= 4, "need a user with history"
        cutoff = int(rec.event_ts[rec.n_events // 2])
        out = truncate_record(rec, cutoff)
        assert out.n_events < rec.n_events
        assert out.event_ts.size == 0 or out.event_ts.max() < cutoff
        # token arrays sliced consistently with the event CSR
        assert len(out.key_ids) == int(out.event_offsets[-1])
        assert len(out.event_offsets) == out.n_events + 1

    def test_time_log_rereferenced_to_new_last_event(self, dataset: ShardDataset) -> None:
        rec = max((dataset.get(u) for u in dataset.user_ids[:20]), key=lambda r: r.n_events)
        cutoff = int(rec.event_ts[rec.n_events // 2])
        out = truncate_record(rec, cutoff)
        expect = time_encode((out.event_ts[-1] - out.event_ts) / 1e6)
        np.testing.assert_allclose(out.time_log, expect.astype(np.float32), atol=1e-5)

    def test_lifelong_after_cutoff_dropped_statics_kept(self, dataset: ShardDataset) -> None:
        for uid in dataset.user_ids:
            rec = dataset.get(uid)
            lifelong = rec.prof_ts >= 0
            if lifelong.any():
                cutoff = int(rec.prof_ts[lifelong].min())  # drop every milestone
                out = truncate_record(rec, cutoff)
                assert (out.prof_ts < 0).all(), "post-cutoff lifelong milestones must drop"
                n_static = int((rec.prof_ts < 0).sum())
                assert len(out.prof_offsets) - 1 == n_static, "static attributes must survive"
                return
        pytest.skip("no user with lifelong milestones in sample")

    def test_zero_event_truncation_collates_and_embeds(self, dataset: ShardDataset) -> None:
        from pragmatiq.models.pragmatiq import ModelConfig, PragmaModel

        uids = dataset.user_ids[:3]
        recs = [dataset.get(u) for u in uids]
        cutoffs = {uids[0]: int(recs[0].event_ts[0])}  # everything after -> 0 events
        batch = TruncatingCollator(cutoffs)([r for r in recs])
        assert int(batch.n_events_per_user[0]) == 0
        model = PragmaModel(ModelConfig.preset("small", 5000))
        z = model.embed_users(batch)
        assert z.shape[0] == len(uids), "profile-only users still get an embedding"
        assert np.isfinite(z.detach().numpy()).all()

    def test_collated_batch_never_contains_post_cutoff_events(self, dataset: ShardDataset) -> None:
        uids = dataset.user_ids[:8]
        recs = [dataset.get(u) for u in uids]
        cutoffs = {r.user_id: int(r.event_ts[max(r.n_events // 2, 1) - 1]) + 1
                   for r in recs if r.n_events}
        batch = TruncatingCollator(cutoffs)(recs)
        ts = batch.event_ts.numpy()
        owner = batch.user_of_event.numpy()
        for i, uid in enumerate(batch.user_ids):
            mine = ts[owner == i]
            assert mine.size == 0 or mine.max() < cutoffs[uid]

    def test_no_cutoff_is_identity(self, dataset: ShardDataset) -> None:
        recs = [dataset.get(u) for u in dataset.user_ids[:4]]
        plain = VarlenCollator()(recs)
        wrapped = TruncatingCollator({})(recs)
        assert plain.n_tokens == wrapped.n_tokens
        assert plain.n_events == wrapped.n_events


class TestEventCapAtCollate:
    """The per-user event cap is applied when a batch is collated, after the eval-point cut."""

    @staticmethod
    def _record(n: int) -> UserRecord:
        hour = 3_600_000_000
        return UserRecord(
            user_id="heavy",
            events=[(int((i + 1) * hour), "transaction", {"amount": f"{i + 1}.50", "mcc": "5411"})
                    for i in range(n)],
            attributes={"country": "GB"}, lifelong=[("kyc_passed", hour)], as_of=int((n + 1) * hour),
        )

    def test_encode_keeps_the_full_history(self, dataset: ShardDataset) -> None:
        tok = PragmaTokenizer.load(dataset.dir / "tokenizer")
        tok.config.max_events_per_user = 10
        rec = tok.encode(self._record(40))
        assert rec.n_events == 40  # shards carry everything; the cap is a batch-time decision

    def test_cap_events_keeps_most_recent(self, dataset: ShardDataset) -> None:
        from pragmatiq.data.tokenizer import cap_events

        tok = PragmaTokenizer.load(dataset.dir / "tokenizer")
        full = tok.encode(self._record(40))
        capped = cap_events(full, 10)
        assert capped.n_events == 10
        assert np.array_equal(capped.event_ts, full.event_ts[30:])
        assert capped.event_offsets[0] == 0 and capped.event_offsets[-1] == capped.key_ids.size
        assert np.array_equal(capped.key_ids, full.key_ids[int(full.event_offsets[30]):])
        assert np.array_equal(capped.time_log, full.time_log[30:])  # still referenced to the last event
        assert np.array_equal(capped.prof_key_ids, full.prof_key_ids)  # profile untouched
        assert cap_events(full, 40) is full and cap_events(full, None) is full

    def test_truncating_collator_caps_after_the_cutoff(self, dataset: ShardDataset) -> None:
        tok = PragmaTokenizer.load(dataset.dir / "tokenizer")
        full = tok.encode(self._record(40))
        cutoff = int(full.event_ts[20])  # events 0..19 are before the eval point
        batch = TruncatingCollator({"heavy": cutoff}, max_events=10)([full])
        assert batch.n_events == 10
        assert np.array_equal(batch.event_ts.numpy(), full.event_ts[10:20])  # the 10 most recent BEFORE the cut
        plain = VarlenCollator(max_events=10)([full])
        assert np.array_equal(plain.event_ts.numpy(), full.event_ts[30:])

    def test_manifest_carries_the_cap_and_loader_applies_it(self, dataset: ShardDataset) -> None:
        assert dataset.max_events == 6500  # TokenizerConfig default, recorded at tokenize time
        from pragmatiq.data.dataset import DynamicBatchSampler, ShardDataLoader

        loader = ShardDataLoader(dataset, DynamicBatchSampler(dataset.index, token_budget=2048, seed=0))
        assert loader.collator.max_events == 6500
