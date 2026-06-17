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
from pragmatiq.data.collate import TruncatingCollator, VarlenCollator
from pragmatiq.data.dataset import ShardDataset
from pragmatiq.data.schema import UserRecord
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
