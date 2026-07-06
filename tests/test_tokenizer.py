"""Tokenizer tests.

Round-trip property tests, unseen key/value → [UNK] + warning, vocab-size
range, time encoding, save/load with content-hash verification.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from pragmatiq.core.schema import UserRecord
from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.data.tokenizer import (
    SPECIAL_TOKENS,
    UNK,
    PercentileBinner,
    PragmaTokenizer,
    TokenizerConfig,
    iter_user_records,
    time_encode,
    truncate_record,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tok_data")
    generate(
        WorldConfig(n_users=400, months=16, n_merchants=1500, mule_ring_count=1, seed=11,
                    eval_month_credit=4, eval_month_short=9),
        out, n_workers=0, write_report=False,
    )
    return out


@pytest.fixture(scope="module")
def tokenizer(dataset: Path) -> PragmaTokenizer:
    # categorical_threshold=200 pushes merchant (high-cardinality) onto the BPE
    # path so both the categorical and textual branches are exercised.
    return PragmaTokenizer(
        TokenizerConfig(target_vocab=6000, n_buckets=32, categorical_threshold=200, seed=0)
    ).fit(dataset)


class TestTimeEncode:
    def test_zero_and_monotone(self) -> None:
        assert time_encode(0.0) == 0.0
        xs = np.array([0, 1, 60, 3600, 86400, 86400 * 30], dtype=np.float64)
        ys = time_encode(xs)
        assert np.all(np.diff(ys) > 0)  # strictly increasing in Δt

    def test_formula(self) -> None:
        # 8 * ln(1 + Δ/8) at Δ=8s = 8 ln 2
        assert time_encode(8.0) == pytest.approx(8.0 * np.log(2.0))


class TestPercentileBinner:
    def test_zero_bucket_distinct(self) -> None:
        b = PercentileBinner(16).fit(np.abs(np.random.default_rng(0).normal(50, 20, 5000)) + 1)
        assert int(b.transform(np.array([0.0]))[0]) == 0
        assert int(b.transform(np.array([50.0]))[0]) >= 1

    def test_monotone_buckets(self) -> None:
        vals = np.random.default_rng(1).exponential(30, 10000)
        b = PercentileBinner(32).fit(vals)
        q = b.transform(np.array([1.0, 10.0, 50.0, 200.0]))
        assert np.all(np.diff(q) >= 0)

    def test_out_of_range_clips(self) -> None:
        b = PercentileBinner(8).fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert int(b.transform(np.array([1e9]))[0]) == b.n_bins - 1
        assert int(b.transform(np.array([-5.0]))[0]) >= 1


class TestVocab:
    def test_specials_first(self, tokenizer: PragmaTokenizer) -> None:
        assert SPECIAL_TOKENS == ("[PAD]", "[MASK]", "[UNK]", "[USR]", "[EVT]")
        # special ids occupy 0..4; real keys come after
        assert min(tokenizer.key_vocab.values()) >= len(SPECIAL_TOKENS)

    def test_vocab_size_in_range(self, tokenizer: PragmaTokenizer) -> None:
        # within budget and meaningfully populated
        assert 1000 <= tokenizer.vocab_size <= 6000

    def test_key_count_reasonable(self, tokenizer: PragmaTokenizer) -> None:
        # SPEC: ~60 key tokens; synthetic data uses fewer distinct keys
        assert 15 <= len(tokenizer.key_vocab) <= 80

    def test_field_kinds_present(self, tokenizer: PragmaTokenizer) -> None:
        kinds = set(tokenizer.field_kind.values())
        assert "numeric" in kinds  # amount
        assert "categorical" in kinds
        assert tokenizer.field_kind["amount"] == "numeric"
        assert tokenizer.field_kind["mcc"] == "categorical"  # codes, not quantities


class TestEncodeRoundTrip:
    def test_categorical_exact(self, tokenizer: PragmaTokenizer, dataset: Path) -> None:
        rec = next(iter_user_records(dataset, max_users=1))
        enc = tokenizer.encode(rec)
        assert enc.n_events == len(rec.events)
        assert enc.event_offsets[-1] == enc.n_tokens
        # find a transaction event and check categorical fields round-trip
        for i, (_, source, fields) in enumerate(rec.events):
            if source == "transaction":
                decoded = dict(tokenizer.decode_event(enc, i))
                assert decoded["currency"] == fields["currency"]
                assert decoded["merchant"] == fields["merchant"]
                assert decoded["txn_type"] == fields["txn_type"]
                break

    def test_numeric_bucket_contains_value(self, tokenizer: PragmaTokenizer, dataset: Path) -> None:
        rec = next(iter_user_records(dataset, max_users=1))
        enc = tokenizer.encode(rec)
        checked = 0
        for i, (_, source, fields) in enumerate(rec.events):
            if source != "transaction" or "amount" not in fields:
                continue
            decoded = dict(tokenizer.decode_event(enc, i))
            rep = decoded["amount"]  # half-open interval "[a,b)" or "0"
            amt = float(fields["amount"])
            if rep == "0":
                assert amt == 0.0
            else:
                lo, hi = (float(x) for x in rep[1:-1].split(","))
                assert lo <= amt < hi, f"amount {amt} not in {rep}"
            checked += 1
            if checked >= 8:
                break
        assert checked > 0

    def test_positions_within_field(self, tokenizer: PragmaTokenizer, dataset: Path) -> None:
        assert tokenizer.field_kind["merchant"] == "text"  # BPE path active in this fixture
        rec = next(iter_user_records(dataset, max_users=1))
        enc = tokenizer.encode(rec)
        # categorical/numeric tokens have position 0; text (merchant) BPE pieces increment
        assert enc.positions.min() == 0
        assert enc.positions.max() >= 1  # merchant names tokenize to multiple pieces

    def test_calendar_features_ranges(self, tokenizer: PragmaTokenizer, dataset: Path) -> None:
        rec = next(iter_user_records(dataset, max_users=1))
        enc = tokenizer.encode(rec)
        assert enc.hour.min() >= 0 and enc.hour.max() <= 23
        assert enc.dow.min() >= 0 and enc.dow.max() <= 6
        assert enc.dom.min() >= 1 and enc.dom.max() <= 31
        assert enc.time_log[-1] == pytest.approx(0.0)  # last event: Δt=0


class TestUnknownHandling:
    def test_unseen_key_maps_to_unk(self, tokenizer: PragmaTokenizer, caplog) -> None:
        rec = UserRecord(
            user_id="x", events=[(1_700_000_000_000_000, "transaction",
                                  {"totally_new_key": "weird", "amount": "12.50"})],
            attributes={}, lifelong=[],
        )
        with caplog.at_level(logging.WARNING):
            enc = tokenizer.encode(rec)
        assert UNK in enc.key_ids.tolist()  # unknown key -> [UNK]
        assert any("unseen" in r.message for r in caplog.records)

    def test_unseen_categorical_value_maps_to_unk(self, tokenizer: PragmaTokenizer, caplog) -> None:
        rec = UserRecord(
            user_id="x", events=[(1_700_000_000_000_000, "transaction",
                                  {"currency": "ZZZ_FAKE", "amount": "5.00"})],
            attributes={}, lifelong=[],
        )
        with caplog.at_level(logging.WARNING):
            enc = tokenizer.encode(rec)
        assert UNK in enc.value_ids.tolist()
        assert any("unseen" in r.message for r in caplog.records)

    def test_no_keyerror_on_unknown(self, tokenizer: PragmaTokenizer) -> None:
        rec = UserRecord(user_id="x", events=[(1_700_000_000_000_000, "alien_source",
                                              {"k1": "v1", "k2": "v2"})], attributes={"new_attr": "z"},
                         lifelong=[("new_milestone", 1_700_000_000_000_000)])
        tokenizer.encode(rec)  # must not raise


class TestSaveLoad:
    def test_round_trip_and_hash(self, tokenizer: PragmaTokenizer, tmp_path: Path, dataset: Path) -> None:
        tokenizer.save(tmp_path / "tok")
        loaded = PragmaTokenizer.load(tmp_path / "tok")
        assert loaded.content_hash == tokenizer.content_hash
        assert loaded.vocab_size == tokenizer.vocab_size
        rec = next(iter_user_records(dataset, max_users=1))
        e1, e2 = tokenizer.encode(rec), loaded.encode(rec)
        assert np.array_equal(e1.key_ids, e2.key_ids)
        assert np.array_equal(e1.value_ids, e2.value_ids)
        assert np.array_equal(e1.positions, e2.positions)

    def test_tampered_hash_rejected(self, tokenizer: PragmaTokenizer, tmp_path: Path) -> None:
        d = tmp_path / "tok2"
        tokenizer.save(d)
        (d / "tokenizer.hash").write_text("deadbeef")
        with pytest.raises(ValueError, match="hash mismatch"):
            PragmaTokenizer.load(d)


class TestDeterminism:
    def test_fit_is_deterministic(self, dataset: Path) -> None:
        a = PragmaTokenizer(TokenizerConfig(target_vocab=6000, n_buckets=32, seed=0)).fit(dataset)
        b = PragmaTokenizer(TokenizerConfig(target_vocab=6000, n_buckets=32, seed=0)).fit(dataset)
        assert a.content_hash == b.content_hash

    def test_fit_is_worker_count_invariant(self, tmp_path: Path, monkeypatch) -> None:
        """The fitted tokenizer is byte-identical serial vs. parallel for any
        worker count (global rule 2), even with the above-cap numeric reservoir
        active and users straddling row-group boundaries.

        Reuses the straddling BYO fixture from the parallel-tokenize suite: 7-row
        row groups cut users mid-run, and the tiny task constants make every row
        group a separate task — so workers fold disjoint, boundary-crossing
        slices and the parent must merge them back to the single-pass result.
        """
        import pragmatiq.data.parallel_tokenize as pt
        from tests.test_tokenize_parallel import _write_byo

        # Enough amount values (~3.2k) to far exceed max_numeric_sample below, so
        # the Algorithm-R reservoir replay — not just the prefix fill — runs.
        runs = [(f"u{i:03d}", 30 + (i * 13) % 40) for i in range(60)]
        runs.insert(25, ("whale", 200))
        ds = _write_byo(tmp_path / "byo", runs, rows_per_rg=7)
        monkeypatch.setattr(pt, "_MIN_TASK_ROWS", 1)
        monkeypatch.setattr(pt, "_MAX_TASK_ROWS", 4)
        tasks = pt._plan_tasks(ds / "events.parquet", 3)
        assert tasks is not None and len(tasks) > 5  # genuinely many parallel tasks

        def cfg() -> TokenizerConfig:
            return TokenizerConfig(n_buckets=16, max_numeric_sample=500, seed=0)

        serial = PragmaTokenizer(cfg()).fit(ds, n_workers=0)
        assert serial.field_kind["amount"] == "numeric"  # reservoir path exercised
        hashes = {0: serial.content_hash}
        for w in (2, 3, 5):
            hashes[w] = PragmaTokenizer(cfg()).fit(ds, n_workers=w).content_hash
        assert hashes[0] == hashes[2] == hashes[3] == hashes[5], hashes


class TestTruncationCaps:
    """Paper-faithful pre-training caps: tractable, stable training on long histories."""

    @staticmethod
    def _oversized(tokenizer: PragmaTokenizer) -> UserRecord:
        text_key = next((k for k, v in tokenizer.field_kind.items() if v == "text"), None)
        long_text = " ".join(f"zz{i}" for i in range(40))  # many BPE pieces → long event
        big_event = (1_700_000_000_000_000, "transaction",
                     {text_key: long_text} if text_key else {"amount": "1"})
        many = [(1_600_000_000_000_000 + i * 1_000_000, "transaction", {"amount": str(i)})
                for i in range(7000)]
        return UserRecord(user_id="x", events=[big_event] + many,
                          lifelong=[(f"ll{i}", 1_500_000_000_000_000) for i in range(300)])

    def test_caps_truncate_oversized_record(self, tokenizer: PragmaTokenizer) -> None:
        out = tokenizer.encode(self._oversized(tokenizer))  # default caps 24/200/6500
        per_event = np.diff(out.event_offsets)
        assert len(out.event_offsets) - 1 == 6500  # most-recent event subsample
        assert int(per_event.max()) <= 24  # per-event token cap
        assert out.prof_key_ids.size <= 200  # profile-state token cap

    def test_caps_off_keeps_everything(self, tokenizer: PragmaTokenizer) -> None:
        import copy
        import dataclasses

        off = copy.copy(tokenizer)  # shares fitted vocab; only the config differs
        off.config = dataclasses.replace(
            tokenizer.config, max_event_tokens=None, max_profile_tokens=None,
            max_events_per_user=None,
        )
        out = off.encode(self._oversized(tokenizer))
        assert len(out.event_offsets) - 1 == 7001  # nothing dropped
        assert int(np.diff(out.event_offsets).max()) > 24  # event not capped
        assert out.prof_key_ids.size == 300  # profile not capped

    def test_caps_fold_into_content_hash(self, dataset: Path) -> None:
        import dataclasses

        base = TokenizerConfig(target_vocab=2500, n_buckets=16, categorical_threshold=150)
        h1 = PragmaTokenizer(base).fit(dataset).content_hash
        h2 = PragmaTokenizer(dataclasses.replace(base, max_event_tokens=8)).fit(dataset).content_hash
        assert h1 != h2  # a different cap is a different tokenizer (from_pretrained refuses mismatch)


class TestEmbedTextMode:
    """PRAGMA+Nemotron variant: each event text field emits ONE sentinel token and
    carries its raw string for a frozen text encoder; text_values stays compact."""

    @staticmethod
    def _embed(tokenizer: PragmaTokenizer) -> PragmaTokenizer:
        import copy
        import dataclasses

        emb = copy.copy(tokenizer)  # share the fitted vocab; only the text path changes
        emb.config = dataclasses.replace(tokenizer.config, text_value_mode="embed")
        return emb

    @staticmethod
    def _text_keys(tokenizer: PragmaTokenizer) -> set[str]:
        return {k for k, v in tokenizer.field_kind.items() if v == "text"}

    @staticmethod
    def _values_in_token_order(rec: UserRecord, text_keys: set[str]) -> list[str]:
        # encode() sorts events by ts, then iterates source + fields.items() in order;
        # only field values whose key is text become sentinels (source is categorical).
        out: list[str] = []
        for _ts, _src, fields in sorted(rec.events, key=lambda e: e[0]):
            out.extend(fv for fk, fv in fields.items() if fk in text_keys)
        return out

    def _textful_record(self, dataset: Path, text_keys: set[str], min_text: int = 1) -> UserRecord:
        for r in iter_user_records(dataset):
            if sum(1 for _, _, fs in r.events for fk in fs if fk in text_keys) >= min_text:
                return r
        raise AssertionError("fixture has no record with text-field events")

    def test_sentinels_carry_compact_strings(self, tokenizer: PragmaTokenizer, dataset: Path) -> None:
        text_keys = self._text_keys(tokenizer)
        assert text_keys  # fixture exercises the text path (device_id, merchant)
        emb = self._embed(tokenizer)
        rec = self._textful_record(dataset, text_keys)
        out = emb.encode(rec)

        n_text = int(out.is_text.sum())
        assert n_text > 0
        assert out.is_text.shape == (out.n_tokens,)  # a per-token marker
        assert len(out.text_values) == n_text  # compact: only the text-token strings
        text_kids = {tokenizer.key_vocab[k] for k in text_keys}
        for i in np.flatnonzero(out.is_text):
            assert int(out.key_ids[i]) in text_kids  # marker sits on a text-field key
            assert int(out.value_ids[i]) == UNK  # value carried out-of-band, not as an id
            assert int(out.positions[i]) == 0  # one whole-string token (no BPE pieces)
        assert out.text_values == self._values_in_token_order(rec, text_keys)  # verbatim, in order

    def test_bpe_mode_has_no_text_state(self, tokenizer: PragmaTokenizer, dataset: Path) -> None:
        text_keys = self._text_keys(tokenizer)
        rec = self._textful_record(dataset, text_keys)
        bpe_out = tokenizer.encode(rec)  # default text_value_mode="bpe"
        emb_out = self._embed(tokenizer).encode(rec)
        assert int(bpe_out.is_text.sum()) == 0  # no markers on the BPE path
        assert bpe_out.text_values == []  # no per-token string overhead
        assert emb_out.n_tokens <= bpe_out.n_tokens  # one sentinel ≤ its BPE pieces

    def test_truncate_keeps_text_consistent(self, tokenizer: PragmaTokenizer, dataset: Path) -> None:
        text_keys = self._text_keys(tokenizer)
        emb = self._embed(tokenizer)
        out = emb.encode(self._textful_record(dataset, text_keys, min_text=2))
        cutoff = int(out.event_ts[len(out.event_ts) // 2])  # keep a strict event prefix
        tr = truncate_record(out, cutoff)
        n_text = int(tr.is_text.sum())
        assert tr.is_text.shape == (tr.n_tokens,)
        assert len(tr.text_values) == n_text  # compact invariant survives truncation
        assert tr.text_values == out.text_values[:n_text]  # kept strings are a prefix
