"""Bring-your-own-data robustness (real, messy inputs).

These guard the paths a bank hits when feeding its own records instead of the
synthetic generator: null field values, null/local-timezone timestamps, and
non-adjacent (unsorted) user rows.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

from pragmatiq.data.schema import UserRecord
from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.data.tokenizer import (
    UNK,
    PragmaTokenizer,
    TokenizerConfig,
    _calendar_fields,
    _iter_records_from_batches,
)

_DAY_US = 86_400_000_000
_HOUR_US = 3_600_000_000


def _epoch_us(d: dt.datetime) -> int:
    return (d - dt.datetime(1970, 1, 1, tzinfo=dt.UTC)) // dt.timedelta(microseconds=1)


@pytest.fixture(scope="module")
def tok(tmp_path_factory: pytest.TempPathFactory) -> PragmaTokenizer:
    d = tmp_path_factory.mktemp("byo")
    generate(
        WorldConfig(n_users=60, months=14, n_merchants=400, mule_ring_count=1, seed=5,
                    eval_month_credit=2, eval_month_short=6),
        d / "raw", n_workers=0, write_report=False,
    )
    return PragmaTokenizer(
        TokenizerConfig(target_vocab=2500, n_buckets=16, categorical_threshold=150)
    ).fit(d / "raw")


class TestCalendarTimezone:
    def test_utc_matches_integer_arithmetic(self) -> None:
        ts = np.array([0, 3 * _HOUR_US, 50 * _HOUR_US + 123], dtype=np.int64)
        hour, _, _ = _calendar_fields(ts, "UTC")
        assert hour.tolist() == ((ts % _DAY_US) // _HOUR_US).astype(np.int8).tolist()

    def test_local_zone_shifts_hour_and_day_boundary(self) -> None:
        # 2024-07-01 01:00 UTC → Berlin is +02:00 (CEST): 03:00 local, same day.
        ts = np.array([_epoch_us(dt.datetime(2024, 7, 1, 1, 0, tzinfo=dt.UTC))], dtype=np.int64)
        hu, _, _ = _calendar_fields(ts, "UTC")
        hb, _, domb = _calendar_fields(ts, "Europe/Berlin")
        assert int(hu[0]) == 1 and int(hb[0]) == 3 and int(domb[0]) == 1
        # 2024-06-30 23:30 UTC → Berlin 01:30 on Jul 1: the local day-of-month flips.
        ts2 = np.array([_epoch_us(dt.datetime(2024, 6, 30, 23, 30, tzinfo=dt.UTC))], dtype=np.int64)
        assert int(_calendar_fields(ts2, "UTC")[2][0]) == 30
        assert int(_calendar_fields(ts2, "Europe/Berlin")[2][0]) == 1

    def test_calendar_tz_changes_content_hash(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        d = tmp_path_factory.mktemp("byohash")
        generate(
            WorldConfig(n_users=30, months=14, n_merchants=200, mule_ring_count=1, seed=2,
                        eval_month_credit=2, eval_month_short=5),
            d / "raw", n_workers=0, write_report=False,
        )
        base = TokenizerConfig(target_vocab=2000, n_buckets=16, categorical_threshold=120)
        h_utc = PragmaTokenizer(base).fit(d / "raw").content_hash
        h_local = PragmaTokenizer(dataclasses.replace(base, calendar_tz="Europe/Berlin")).fit(d / "raw").content_hash
        assert h_utc != h_local  # from_pretrained must refuse a tz-mismatched tokenizer


class TestNullFieldValues:
    def test_null_text_value_maps_to_unk_not_crash(self, tok: PragmaTokenizer) -> None:
        text_keys = [k for k, kind in tok.field_kind.items() if kind == "text"]
        assert text_keys, "fixture must have a text/BPE field to exercise the crash path"
        rec = UserRecord(user_id="x", events=[(_epoch_us(dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
                                               "transaction", {text_keys[0]: None})])
        out = tok.encode(rec)  # None field value must not raise (→ '' → [UNK])
        assert UNK in out.value_ids.tolist()


class TestEventOrder:
    def test_out_of_order_events_warn_then_sort(self, tok: PragmaTokenizer, caplog) -> None:
        import logging

        tok._warned.clear()  # the warning channel dedups per tokenizer instance
        rec = UserRecord(user_id="u", events=[
            (200_000_000, "transaction", {"k": "v"}),
            (100_000_000, "transaction", {"k": "v"}),  # earlier ts after a later one
        ])
        with caplog.at_level(logging.WARNING, logger="pragmatiq.data.tokenizer"):
            out = tok.encode(rec)
        assert any("reorder" in r.getMessage() or "ascending" in r.getMessage()
                   for r in caplog.records)
        # still encoded in ascending ts order (the sort is applied)
        assert out.event_ts.tolist() == sorted(out.event_ts.tolist())


class TestNonAdjacentUser:
    def test_nonadjacent_user_id_raises(self) -> None:
        rows = [
            {"user_id": "u1", "ts": 0, "source": "transaction", "fields": {"k": "v"}},
            {"user_id": "u2", "ts": 1, "source": "transaction", "fields": {"k": "v"}},
            {"user_id": "u1", "ts": 2, "source": "transaction", "fields": {"k": "v"}},
        ]
        with pytest.raises(ValueError, match="non-adjacent"):
            list(_iter_records_from_batches([pa.RecordBatch.from_pylist(rows)], {}, None))

    def test_adjacent_user_groups_correctly(self) -> None:
        rows = [
            {"user_id": "u1", "ts": 0, "source": "transaction", "fields": {"k": "v"}},
            {"user_id": "u1", "ts": 1, "source": "transaction", "fields": {"k": "v"}},
            {"user_id": "u2", "ts": 2, "source": "transaction", "fields": {"k": "v"}},
        ]
        recs = list(_iter_records_from_batches([pa.RecordBatch.from_pylist(rows)], {}, None))
        assert [r.user_id for r in recs] == ["u1", "u2"]
        assert len(recs[0].events) == 2
