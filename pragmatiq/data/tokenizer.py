"""Key–value–time tokenizer.

Scheme:

- one **key token** per field key (~60 keys incl. profile/lifelong/source);
- numeric values → :class:`PercentileBinner` (``n_buckets`` percentile buckets
  + a dedicated zero bucket, fitted per key);
- string values: cardinality ≤ ``categorical_threshold`` → one categorical
  token per value; else textual → byte-level BPE (HF ``tokenizers``), vocab
  sized so the total lands ≈ ``target_vocab``;
- per token: ``key_id``, ``value_id`` (one shared vocab space, the model shares
  one embedding table), within-field ``position`` (0..n for BPE pieces);
- time: per event ``8·ln(1+Δt/8)`` log-seconds to the most recent event +
  calendar features (hour, day-of-week, day-of-month); profile lifelong items
  get log-seconds since their first occurrence; static attributes get 0;
- specials: ``[PAD] [MASK] [UNK] [USR] [EVT]``;
- ``fit()`` is one streaming pass over parquet; ``encode(UserRecord)`` returns
  a :class:`TokenizedRecord` (flat arrays + CSR offsets per event);
  ``save()/load()`` round-trip with a content hash.

Unseen keys/values at encode time map to ``[UNK]`` with a logged warning —
never a ``KeyError`` (global rule 4).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..progress import progress
from ..registry import get_value_encoder, register_value_encoder
from .schema import UserRecord

log = logging.getLogger(__name__)

PAD, MASK, UNK, USR, EVT = 0, 1, 2, 3, 4
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[MASK]", "[UNK]", "[USR]", "[EVT]")

DAY_US = 86_400_000_000
HOUR_US = 3_600_000_000


def time_encode(delta_seconds: np.ndarray | float) -> np.ndarray | float:
    """The paper's log-time transform ``8 · ln(1 + Δt/8)`` (Δt in seconds)."""
    return 8.0 * np.log1p(np.asarray(delta_seconds, dtype=np.float64) / 8.0)


@dataclass
class TokenizerConfig:
    """Knobs for :class:`PragmaTokenizer` (defaults per SPEC, GUESS where silent)."""

    n_buckets: int = 64  # GUESS: percentile buckets per numeric key
    categorical_threshold: int = 1000
    target_vocab: int = 28_000  # GUESS: ≈28k total vocab
    bpe_min_frequency: int = 2
    lowercase_text: bool = False
    max_numeric_sample: int = 100_000
    # GUESS: a float-parsing field is treated as a continuous numeric (percentile
    # binned) only above this distinct-value count; below it, low-cardinality
    # codes (MCC, version strings) stay categorical. None ⇒ 4 × n_buckets.
    numeric_min_cardinality: int | None = None
    # Per-key overrides for the field-kind heuristic (rule 8): force keys
    # categorical (e.g. an MCC/ZIP/BIN code that parses as a number) or numeric
    # (a magnitude with few distinct values). Empty ⇒ the heuristic decides.
    force_categorical: tuple[str, ...] = ()
    force_numeric: tuple[str, ...] = ()
    seed: int = 0
    # Numeric value encoder, resolved from the registry by name (rule 8) so a
    # config can swap in a custom @register_value_encoder without forking.
    value_encoder: str = "percentile_binner"
    # IANA timezone for deriving the calendar features (hour-of-day, day-of-week,
    # day-of-month) from each event's UTC instant. Default "UTC". Set a local zone
    # (e.g. "Europe/London") when timestamps are UTC instants but the behavioural
    # day/night, weekend and payday structure is local. Folded into the content
    # hash, so from_pretrained refuses a tokenizer with a mismatched calendar tz.
    calendar_tz: str = "UTC"
    # Pre-training sequence caps that keep training tractable and stable on real,
    # heavy-tailed histories (paper defaults). Each event is capped to the first
    # ``max_event_tokens`` tokens; the profile state to the first whole items
    # fitting ``max_profile_tokens`` tokens; a user with more than
    # ``max_events_per_user`` events keeps only the most recent ones. ``None``
    # disables a cap. At synthetic scale none of these bind, so output is
    # unchanged; they only act on large real-world records.
    max_event_tokens: int | None = 24
    max_profile_tokens: int | None = 200
    max_events_per_user: int | None = 6500
    # PRAGMA+Nemotron variant. "bpe" (default) tokenises text field values with the
    # BPE sub-word vocab. "embed" emits one sentinel token per text field and carries
    # the raw string, which a frozen text encoder maps to a vector at model time
    # (reconstructed via MSE during pre-training). Applies to event text fields;
    # profile text stays BPE. text_encoder names a registered frozen encoder.
    text_value_mode: str = "bpe"
    text_encoder: str = "hash"
    text_encoder_dim: int = 64

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenizerConfig:
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TokenizerConfig keys: {sorted(unknown)}")
        return cls(**d)


@register_value_encoder("percentile_binner")
class PercentileBinner:
    """Percentile bucketizer with a dedicated zero bucket (bucket 0).

    Buckets: 0 → exactly zero; ``1..n_edges`` → percentile intervals of the
    non-zero training values. Out-of-range values clip into the end buckets,
    so transform never fails on unseen magnitudes.
    """

    def __init__(self, n_buckets: int = 64) -> None:
        self.n_buckets = n_buckets
        self.edges: np.ndarray = np.zeros(0, dtype=np.float64)

    def fit(self, values: np.ndarray) -> PercentileBinner:
        """Learn bucket edges from a sample of this key's numeric values."""
        nz = np.asarray(values, dtype=np.float64)
        nz = nz[np.isfinite(nz) & (nz != 0.0)]
        if len(nz) == 0:
            self.edges = np.zeros(0, dtype=np.float64)
            return self
        qs = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
        self.edges = np.unique(np.quantile(nz, qs))
        return self

    @property
    def n_bins(self) -> int:
        """Total buckets incl. the zero bucket."""
        return len(self.edges) + 2  # zero bucket + len(edges)+1 intervals

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map numeric values to bucket indices (0 = zero bucket)."""
        x = np.asarray(values, dtype=np.float64)
        out = np.searchsorted(self.edges, x, side="right") + 1
        out[x == 0.0] = 0
        out[~np.isfinite(x)] = 0
        return out.astype(np.int64)

    def bucket_repr(self, bucket: int) -> str:
        """Half-open interval ``[lo, hi)`` a bucket covers (``transform`` uses
        ``searchsorted(side="right")``); bucket 0 is exactly zero."""
        if bucket <= 0:
            return "0"
        lo = self.edges[bucket - 2] if bucket >= 2 else float("-inf")
        hi = self.edges[bucket - 1] if bucket - 1 < len(self.edges) else float("inf")
        return f"[{lo:.6g},{hi:.6g})"

    def to_dict(self) -> dict[str, Any]:
        return {"n_buckets": self.n_buckets, "edges": self.edges.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PercentileBinner:
        b = cls(d["n_buckets"])
        b.edges = np.asarray(d["edges"], dtype=np.float64)
        return b


@dataclass
class TokenizedRecord:
    """One user's tokenized history: flat token arrays + CSR offsets per event.

    Token arrays cover the event stream; ``event_offsets[i]:event_offsets[i+1]``
    are event *i*'s tokens. Profile attributes + lifelong milestones live in the
    ``prof_*`` arrays with their own CSR (one slice per profile item).
    """

    user_id: str
    # events
    key_ids: np.ndarray  # int32[n_tokens]
    value_ids: np.ndarray  # int32[n_tokens]
    positions: np.ndarray  # int16[n_tokens] within-field position
    event_offsets: np.ndarray  # int64[n_events+1]
    event_ts: np.ndarray  # int64[n_events] µs
    time_log: np.ndarray  # float32[n_events] 8ln(1+Δ/8) to most recent event
    hour: np.ndarray  # int8[n_events]
    dow: np.ndarray  # int8[n_events]
    dom: np.ndarray  # int8[n_events]
    source_ids: np.ndarray  # int8[n_events] index into schema.SOURCES
    # profile
    prof_key_ids: np.ndarray  # int32
    prof_value_ids: np.ndarray  # int32
    prof_positions: np.ndarray  # int16
    prof_offsets: np.ndarray  # int64[n_items+1]
    prof_time_log: np.ndarray  # float32[n_items]
    prof_ts: np.ndarray  # int64[n_items] µs; -1 for static attributes
    # Nemotron variant (text_value_mode="embed"): ``is_text`` marks each event token
    # that carries a frozen text embedding (1) versus an ordinary id (0), and
    # ``text_values`` holds *only* those text tokens' raw strings, in token order, so
    # ``len(text_values) == int(is_text.sum())``. In BPE mode (the default) ``is_text``
    # is all-zero and ``text_values`` is empty — no per-token string overhead.
    is_text: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    text_values: list[str] = dataclasses.field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.event_ts)

    @property
    def n_tokens(self) -> int:
        return len(self.key_ids)


def truncate_record(rec: TokenizedRecord, cutoff_us: int) -> TokenizedRecord:
    """Return a copy of ``rec`` containing only history strictly before ``cutoff_us``.

    Drops events (and lifelong profile milestones) with ``ts >= cutoff_us`` and
    re-derives the time encodings against the new last event / snapshot moment,
    so a truncated record matches what tokenizing only pre-cutoff data would
    produce. Static profile attributes (``prof_ts == -1``) are kept. This keeps
    label outcome windows out of downstream embeddings — probes and fine-tunes
    must never see events from after a label's eval point.

    Truncation requires the ``prof_ts`` column so lifelong milestones can be cut
    at the eval point; a shard without it raises (re-tokenize to add prof_ts).
    """
    # searchsorted is only correct on an ascending array. encode() sorts events
    # by time, so shard-built records always satisfy this; the guard catches a
    # hand-built TokenizedRecord whose events are out of order (which would
    # otherwise truncate at a silently wrong position rather than raising).
    if rec.event_ts.size > 1 and not bool((np.diff(rec.event_ts) >= 0).all()):
        raise ValueError(
            f"truncate_record requires ascending event_ts for user {rec.user_id!r}"
        )
    n_keep = int(np.searchsorted(rec.event_ts, cutoff_us, side="left"))
    tok_end = int(rec.event_offsets[n_keep])
    ts = rec.event_ts[:n_keep]
    if n_keep:
        tlog = np.asarray(time_encode(np.maximum(ts[-1] - ts, 0) / 1e6), dtype=np.float32)
    else:
        tlog = np.zeros(0, dtype=np.float32)

    n_items = len(rec.prof_offsets) - 1
    prof_ts = rec.prof_ts
    if len(prof_ts) != n_items:
        # Without per-item timestamps we cannot tell which lifelong milestones
        # post-date the cutoff, so keeping them would leak the future. Fail loudly.
        raise ValueError(
            f"cannot truncate user {rec.user_id!r}: shard predates the prof_ts column, "
            "so lifelong milestones can't be cut at the eval point (would leak post-eval "
            "facts into the embedding). Re-tokenize the dataset to add prof_ts."
        )
    keep_items = np.flatnonzero((prof_ts < 0) | (prof_ts < cutoff_us))
    pk, pv, pp = [], [], []
    poff = [0]
    ptime: list[float] = []
    pts: list[int] = []
    for it in keep_items:
        lo, hi = int(rec.prof_offsets[it]), int(rec.prof_offsets[it + 1])
        pk.append(rec.prof_key_ids[lo:hi])
        pv.append(rec.prof_value_ids[lo:hi])
        pp.append(rec.prof_positions[lo:hi])
        poff.append(poff[-1] + (hi - lo))
        t = int(prof_ts[it]) if len(prof_ts) == n_items else -1
        if t < 0:
            ptime.append(float(rec.prof_time_log[it]))
        else:  # re-reference lifelong recency to the cutoff (the new snapshot)
            ptime.append(float(time_encode(max(cutoff_us - t, 0) / 1e6)))
        pts.append(t)

    def _cat(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    # Keep the text markers/strings for the surviving tokens (Nemotron variant).
    # text_values is compact, so retain the first int(is_text[:tok_end].sum()) of them.
    is_text = rec.is_text[:tok_end]
    text_values = rec.text_values[: int(is_text.sum())] if rec.text_values else []

    return TokenizedRecord(
        user_id=rec.user_id,
        key_ids=rec.key_ids[:tok_end], value_ids=rec.value_ids[:tok_end],
        positions=rec.positions[:tok_end], event_offsets=rec.event_offsets[:n_keep + 1].copy(),
        event_ts=ts, time_log=tlog,
        hour=rec.hour[:n_keep], dow=rec.dow[:n_keep], dom=rec.dom[:n_keep],
        source_ids=rec.source_ids[:n_keep],
        prof_key_ids=_cat(pk, np.int32), prof_value_ids=_cat(pv, np.int32),
        prof_positions=_cat(pp, np.int16), prof_offsets=np.asarray(poff, dtype=np.int64),
        prof_time_log=np.asarray(ptime, dtype=np.float32),
        prof_ts=np.asarray(pts, dtype=np.int64),
        is_text=is_text, text_values=text_values,
    )


class PragmaTokenizer:
    """Fits and applies the key–value–time vocabulary (see module docstring)."""

    FORMAT_VERSION = 1

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.key_vocab: dict[str, int] = {}
        self.field_kind: dict[str, str] = {}  # key -> categorical | numeric | text
        self.cat_vocab: dict[str, dict[str, int]] = {}
        self.binners: dict[str, PercentileBinner] = {}
        self.binner_base: dict[str, int] = {}  # key -> vocab id of bucket 0
        self.bpe: Any = None  # tokenizers.Tokenizer
        self.bpe_offset: int = 0
        self.vocab_size: int = len(SPECIAL_TOKENS)
        self._fitted = False
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ fit
    def fit(self, data_dir: str | Path, n_workers: int = 0) -> PragmaTokenizer:
        """Fit the vocabulary over ``events.parquet`` + ``profiles.parquet``.

        The fit is a fold into one :class:`_FitAccum`: with ``n_workers > 1`` the
        events are folded in parallel over row-group ranges and merged in task
        (file) order, otherwise in a single streaming pass. Both routes produce a
        byte-identical tokenizer for any worker count (global rule 2) because the
        accumulator is partition-independent and the binning sample is finalized
        once, in the parent, over the values in file order.
        """
        data_dir = Path(data_dir)
        if n_workers and n_workers > 1:
            from .parallel_tokenize import parallel_fit

            return parallel_fit(data_dir, self, n_workers)

        acc = _FitAccum()
        # events
        pf = pq.ParquetFile(data_dir / "events.parquet")
        n_ev_batches = -(-pf.metadata.num_rows // 65_536)
        for batch in progress(pf.iter_batches(columns=["source", "fields"], batch_size=65_536),
                              total=n_ev_batches, desc="tokenizer fit (events)", unit="batch"):
            acc.consume_events_batch(batch)
        self._consume_profiles(acc, data_dir)
        return self._finalize(acc)

    def _consume_profiles(self, acc: _FitAccum, data_dir: Path) -> None:
        """Fold ``profiles.parquet`` (attributes + lifelong) into ``acc``.

        Profiles are folded once, after the events, on whichever process owns the
        merged accumulator — the parent in the parallel route — so the events →
        profiles fold order matches the single-pass route exactly.
        """
        pfp = pq.ParquetFile(data_dir / "profiles.parquet")
        n_pr_batches = -(-pfp.metadata.num_rows // 16_384)
        for batch in progress(pfp.iter_batches(columns=["attributes", "lifelong"], batch_size=16_384),
                              total=n_pr_batches, desc="tokenizer fit (profiles)", unit="batch"):
            acc.consume_profiles_batch(batch)

    def _binning_sample(self, key: str, values: list[float]) -> np.ndarray:
        """The per-key sample the binner fits on, capped at ``max_numeric_sample``.

        Below the cap this is the whole finite stream (a prefix is order-stable).
        At or above it, a uniform reservoir (Algorithm R) keeps the sample
        unbiased even on time-/cohort-ordered exports rather than just the
        earliest rows. The reservoir is driven by a per-key stream seeded only
        from ``(seed, salt, key_hash)`` and the running count, and is replayed
        once over the values in file order — so the bin edges are a pure function
        of the data, independent of how the rows were partitioned across workers.
        """
        cap = self.config.max_numeric_sample
        if len(values) <= cap:
            return np.asarray(values, dtype=np.float64)
        res_rng = np.random.default_rng((self.config.seed, _RESERVOIR_SALT, _stable_key_hash(key)))
        samp = values[:cap]
        for m, x in enumerate(values[cap:], start=cap):
            r = int(res_rng.integers(0, m + 1))
            if r < cap:
                samp[r] = x
        return np.asarray(samp, dtype=np.float64)

    def _finalize(self, acc: _FitAccum) -> PragmaTokenizer:
        """Classify keys and build the vocab + binners + BPE from a merged accum."""
        thr = self.config.categorical_threshold
        all_keys = sorted(set(acc.counters) | set(acc.numeric_n))
        nid = len(SPECIAL_TOKENS)
        for k in all_keys:
            self.key_vocab[k] = nid
            nid += 1
        nb = self.config.n_buckets
        numeric_min_card = self.config.numeric_min_cardinality or 4 * nb
        text_keys: list[str] = []
        for k in all_keys:
            n_tot = acc.numeric_n[k]
            ratio = acc.numeric_ok[k] / max(n_tot, 1)
            distinct = len(acc.counters.get(k, ()))
            high_card = distinct > numeric_min_card
            # Numeric wins over a high-cardinality string: a genuinely continuous
            # quantity (amount/price/quantity/balance) parses as a float and is
            # high-cardinality — works for integer-valued money (cents, JPY) too.
            # Low-cardinality codes (MCC "5411", version "10.23") parse as floats
            # but are identifiers → categorical.
            # CAVEAT: this distinct-count gate is a GUESS — a high-cardinality
            # numeric *code* (real ISO-18245 MCC, ZIP, BIN with >numeric_min_card
            # distinct values) would be misrouted to the binner; use
            # force_categorical / force_numeric to override per key.
            sample = acc.numeric_sample.get(k, [])
            can_bin = len(sample) > 0
            looks_numeric = ratio >= 0.995 and n_tot >= 32 and high_card
            if k in self.config.force_categorical:
                looks_numeric = False
            elif k in self.config.force_numeric and can_bin:
                looks_numeric = True
            if looks_numeric and can_bin:
                self.field_kind[k] = "numeric"
                binner = get_value_encoder(self.config.value_encoder)(nb).fit(
                    self._binning_sample(k, sample))
                self.binners[k] = binner
                self.binner_base[k] = nid
                nid += binner.n_bins
            elif distinct > thr:  # too many distinct values for one-token-per-value
                self.field_kind[k] = "text"
                text_keys.append(k)
            else:
                self.field_kind[k] = "categorical"
                vocab = {}
                for v, _ in acc.counters[k].most_common():
                    vocab[v] = nid
                    nid += 1
                self.cat_vocab[k] = vocab

        # BPE over textual keys to fill the vocab to ≈ target. The per-key corpus
        # is the merged counter's values in most_common() order (frequency desc,
        # first-seen tie-break), capped — a deterministic, partition-independent
        # slice so the trained BPE is identical for any worker count.
        bpe_budget = max(self.config.target_vocab - nid, 0)
        if text_keys and bpe_budget >= 300:
            corpus: list[str] = []
            for k in text_keys:
                corpus.extend(_text_corpus(acc.counters[k]))
            if self.config.lowercase_text:
                corpus = [s.lower() for s in corpus]
            self.bpe = _train_bpe(corpus, vocab_size=bpe_budget,
                                  min_frequency=self.config.bpe_min_frequency)
            self.bpe_offset = nid
            nid += self.bpe.get_vocab_size()
        elif text_keys:
            raise ValueError(
                f"vocab budget exhausted before BPE ({nid} ids used, target "
                f"{self.config.target_vocab}); raise target_vocab"
            )
        self.vocab_size = nid
        self._fitted = True
        log.info("tokenizer fitted: %d keys, vocab=%d (%d bpe)", len(self.key_vocab),
                 self.vocab_size, self.bpe.get_vocab_size() if self.bpe else 0)
        return self

    # ------------------------------------------------------------------ encode
    def encode(self, record: UserRecord) -> TokenizedRecord:
        """Tokenize one user's raw history (unseen → [UNK] + warning)."""
        if not self._fitted:
            raise RuntimeError("tokenizer is not fitted; call fit() or load()")
        from .schema import SOURCES

        k_l: list[int] = []
        v_l: list[int] = []
        p_l: list[int] = []
        offsets: list[int] = [0]
        src_l: list[int] = []
        src_idx = {s: i for i, s in enumerate(SOURCES)}
        # Nemotron embed mode threads per-token text markers + strings; in BPE mode
        # these stay empty and the record's is_text/text_values are zero/empty.
        embed_text = self.config.text_value_mode == "embed"
        it_l: list[int] | None = [] if embed_text else None
        tx_l: list[str] | None = [] if embed_text else None

        # Events must be ascending by ts for the time encoding and eval-point
        # truncation to be correct; sort here so BYO / notebook records
        # (embed_records bypasses validate) always tokenize with well-defined time
        # positions and a correct truncation cut. Already-sorted input is unchanged;
        # out-of-order input is repaired with a one-time warning (the lenient
        # interactive path), since silent reordering is surprising.
        events = record.events
        if any(events[i][0] < events[i - 1][0] for i in range(1, len(events))):
            if "reorder" not in self._warned:
                self._warned.add("reorder")
                log.warning("events were not ascending by ts and were sorted before encoding; "
                            "pass pre-sorted events to avoid the reorder")
            events = sorted(events, key=lambda e: e[0])
        # Keep only the most recent events when a history is very long (preserves
        # recency); no-op below the cap.
        cap_n = self.config.max_events_per_user
        if cap_n is not None and len(events) > cap_n:
            events = events[-cap_n:]
        cap_tok = self.config.max_event_tokens
        for _ts, source, fields in events:
            start = len(k_l)
            self._encode_field("source", source, k_l, v_l, p_l, it_l, tx_l)
            for fk, fv in fields.items():
                self._encode_field(fk, fv, k_l, v_l, p_l, it_l, tx_l)
            if cap_tok is not None and len(k_l) - start > cap_tok:
                if it_l is not None and tx_l is not None:
                    # text_values is compact (text tokens only), so drop the strings
                    # whose sentinel tokens fall beyond the cap before cutting markers.
                    n_drop = int(sum(it_l[start + cap_tok:]))
                    del it_l[start + cap_tok:]
                    if n_drop:
                        del tx_l[len(tx_l) - n_drop:]
                del k_l[start + cap_tok:]
                del v_l[start + cap_tok:]
                del p_l[start + cap_tok:]
            offsets.append(len(k_l))
            sid = src_idx.get(source)
            if sid is None:
                self._warn_unseen(f"source:{source}")
                sid = 0
            src_l.append(sid)

        ts_arr = np.array([e[0] for e in events], dtype=np.int64)
        if len(ts_arr):
            # max() == ts_arr[-1] once sorted; clamp >= 0 so any residual
            # out-of-order delta can never reach log of a non-positive number.
            delta_s = np.maximum(ts_arr[-1] - ts_arr, 0) / 1e6  # to most recent event
            tlog = np.asarray(time_encode(delta_s), dtype=np.float32)
        else:
            tlog = np.zeros(0, dtype=np.float32)
        hour, dow, dom = _calendar_fields(ts_arr, self.config.calendar_tz)

        pk_l: list[int] = []
        pv_l: list[int] = []
        pp_l: list[int] = []
        poff: list[int] = [0]
        ptime: list[float] = []
        pts: list[int] = []
        for ak, av in sorted(record.attributes.items()):
            self._encode_field(ak, av, pk_l, pv_l, pp_l)
            poff.append(len(pk_l))
            ptime.append(0.0)  # static attributes get time 0
            pts.append(-1)
        if record.lifelong:
            # log-seconds *since* each milestone occurred, measured from the
            # profile snapshot (as_of), so the lifelong axis shares the event
            # stream's recency orientation (most-recent = small).
            ref = record.as_of
            if ref <= 0:
                ref = int(ts_arr[-1]) if len(ts_arr) else max(t for _, t in record.lifelong)
            for lk, lt in record.lifelong:
                self._encode_field(lk, lk, pk_l, pv_l, pp_l)  # presence token
                poff.append(len(pk_l))
                ptime.append(float(time_encode(max(ref - lt, 0) / 1e6)))
                pts.append(int(lt))

        # Cap the profile state to the first whole items fitting the token budget
        # (attributes precede lifelong items); no-op below the cap.
        cap_p = self.config.max_profile_tokens
        if cap_p is not None and len(pk_l) > cap_p:
            keep = 0
            while keep < len(ptime) and poff[keep + 1] <= cap_p:
                keep += 1
            cut = poff[keep]
            del pk_l[cut:]
            del pv_l[cut:]
            del pp_l[cut:]
            poff = poff[: keep + 1]
            ptime = ptime[:keep]
            pts = pts[:keep]

        return TokenizedRecord(
            user_id=record.user_id,
            key_ids=np.asarray(k_l, dtype=np.int32),
            value_ids=np.asarray(v_l, dtype=np.int32),
            positions=np.asarray(p_l, dtype=np.int16),
            event_offsets=np.asarray(offsets, dtype=np.int64),
            event_ts=ts_arr,
            time_log=tlog,
            hour=hour, dow=dow, dom=dom,
            source_ids=np.asarray(src_l, dtype=np.int8),
            prof_key_ids=np.asarray(pk_l, dtype=np.int32),
            prof_value_ids=np.asarray(pv_l, dtype=np.int32),
            prof_positions=np.asarray(pp_l, dtype=np.int16),
            prof_offsets=np.asarray(poff, dtype=np.int64),
            prof_time_log=np.asarray(ptime, dtype=np.float32),
            prof_ts=np.asarray(pts, dtype=np.int64),
            is_text=np.asarray(it_l, dtype=np.int8) if it_l is not None
            else np.zeros(len(k_l), dtype=np.int8),
            text_values=tx_l if tx_l is not None else [],
        )

    def _encode_field(self, key: str, value: str,
                      k_l: list[int], v_l: list[int], p_l: list[int],
                      is_text_l: list[int] | None = None,
                      text_l: list[str] | None = None) -> None:
        if value is None:
            # A null map value (common in BYO parquet) is treated as empty text and
            # falls through to the [UNK] + warning path — never a TypeError in the
            # text/BPE branch.
            value = ""
        kid = self.key_vocab.get(key)
        # Nemotron variant: a known text field emits ONE sentinel token and carries
        # its raw string; a frozen encoder maps it to a vector at model time. Only on
        # the event path (is_text_l provided); profile text stays BPE.
        if (is_text_l is not None and text_l is not None and kid is not None
                and self.field_kind.get(key) == "text"
                and self.config.text_value_mode == "embed"):
            k_l.append(kid)
            v_l.append(UNK)
            p_l.append(0)
            is_text_l.append(1)
            text_l.append(value)
            return
        before = len(k_l)
        if kid is None:
            self._warn_unseen(f"key:{key}")
            k_l.append(UNK)
            v_l.append(UNK)
            p_l.append(0)
        else:
            kind = self.field_kind[key]
            if kind == "numeric":
                x = _try_float(value)
                if not np.isfinite(x):
                    self._warn_unseen(f"numeric:{key}")
                    k_l.append(kid)
                    v_l.append(UNK)
                    p_l.append(0)
                else:
                    bucket = int(self.binners[key].transform(np.array([x]))[0])
                    k_l.append(kid)
                    v_l.append(self.binner_base[key] + bucket)
                    p_l.append(0)
            elif kind == "categorical":
                vid = self.cat_vocab[key].get(value)
                if vid is None:
                    self._warn_unseen(f"value:{key}={value[:40]}")
                    vid = UNK
                k_l.append(kid)
                v_l.append(vid)
                p_l.append(0)
            else:  # text → BPE pieces
                text = value.lower() if self.config.lowercase_text else value
                ids = self.bpe.encode(text).ids if self.bpe is not None else []
                if not ids:
                    # empty/untokenizable text value, or no BPE fitted → [UNK] + warn
                    self._warn_unseen(f"text:{key}={value[:40]!r}")
                    k_l.append(kid)
                    v_l.append(UNK)
                    p_l.append(0)
                else:
                    for pos, pid in enumerate(ids):
                        k_l.append(kid)
                        v_l.append(self.bpe_offset + pid)
                        p_l.append(min(pos, 32767))
        # Per-token text bookkeeping (event path only): mark every token this call
        # emitted as non-text. Text tokens are handled by the embed branch above,
        # which also appends their string to the compact text_values list.
        if is_text_l is not None:
            added = len(k_l) - before
            is_text_l.extend([0] * added)

    def _warn_unseen(self, what: str) -> None:
        if what not in self._warned:
            self._warned.add(what)
            log.warning("unseen at encode time -> [UNK]: %s", what)

    # ------------------------------------------------------------------ decode (round-trip)
    def decode_event(self, rec: TokenizedRecord, event_i: int) -> list[tuple[str, str]]:
        """Inverse mapping for tests: one event back to (key, value-ish) pairs.

        Categorical/text values reconstruct exactly; numeric values come back
        as bucket interval strings (binning is lossy by design).
        """
        inv_key = {v: k for k, v in self.key_vocab.items()}
        lo, hi = int(rec.event_offsets[event_i]), int(rec.event_offsets[event_i + 1])
        out: list[tuple[str, str]] = []
        i = lo
        while i < hi:
            kid = int(rec.key_ids[i])
            key = inv_key.get(kid, "[UNK]")
            kind = self.field_kind.get(key, "categorical")
            if kind == "text" and key in self.field_kind:
                pieces = []
                j = i
                while j < hi and int(rec.key_ids[j]) == kid and (j == i or rec.positions[j] > 0):
                    pieces.append(int(rec.value_ids[j]) - self.bpe_offset)
                    j += 1
                out.append((key, self.bpe.decode(pieces) if self.bpe else "[UNK]"))
                i = j
            else:
                vid = int(rec.value_ids[i])
                out.append((key, self._decode_value(key, vid)))
                i += 1
        return out

    def _decode_value(self, key: str, vid: int) -> str:
        if vid == UNK:
            return "[UNK]"
        kind = self.field_kind.get(key)
        if kind == "numeric":
            return self.binners[key].bucket_repr(vid - self.binner_base[key])
        if kind == "categorical":
            inv = {v: k for k, v in self.cat_vocab[key].items()}
            return inv.get(vid, "[UNK]")
        return "[UNK]"

    # ------------------------------------------------------------------ save / load / hash
    def _state(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "config": self.config.__dict__,
            "key_vocab": self.key_vocab,
            "field_kind": self.field_kind,
            "cat_vocab": self.cat_vocab,
            "binners": {k: b.to_dict() for k, b in self.binners.items()},
            "binner_base": self.binner_base,
            "bpe_offset": self.bpe_offset,
            "vocab_size": self.vocab_size,
        }

    @property
    def content_hash(self) -> str:
        """Stable sha256 over the full tokenizer state (vocab + binners + BPE)."""
        blob = json.dumps(self._state(), sort_keys=True).encode()
        if self.bpe is not None:
            blob += self.bpe.to_str().encode()
        return hashlib.sha256(blob).hexdigest()

    def save(self, out_dir: str | Path) -> Path:
        """Write tokenizer.json (+ bpe.json) + hash; returns the directory."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "tokenizer.json", "w") as f:
            json.dump(self._state(), f, sort_keys=True)
        if self.bpe is not None:
            (out / "bpe.json").write_text(self.bpe.to_str())
        (out / "tokenizer.hash").write_text(self.content_hash)
        return out

    @classmethod
    def load(cls, in_dir: str | Path) -> PragmaTokenizer:
        """Load a saved tokenizer; verifies the stored content hash."""
        in_dir = Path(in_dir)
        state = json.loads((in_dir / "tokenizer.json").read_text())
        if state.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(f"tokenizer format {state.get('format_version')} != {cls.FORMAT_VERSION}")
        tok = cls(TokenizerConfig.from_dict(state["config"]))
        tok.key_vocab = {str(k): int(v) for k, v in state["key_vocab"].items()}
        tok.field_kind = state["field_kind"]
        tok.cat_vocab = {k: {str(vk): int(vv) for vk, vv in d.items()} for k, d in state["cat_vocab"].items()}
        tok.binners = {k: PercentileBinner.from_dict(d) for k, d in state["binners"].items()}
        tok.binner_base = {str(k): int(v) for k, v in state["binner_base"].items()}
        tok.bpe_offset = int(state["bpe_offset"])
        tok.vocab_size = int(state["vocab_size"])
        bpe_path = in_dir / "bpe.json"
        if bpe_path.exists():
            from tokenizers import Tokenizer

            tok.bpe = Tokenizer.from_str(bpe_path.read_text())
        tok._fitted = True
        stored = (in_dir / "tokenizer.hash").read_text().strip() if (in_dir / "tokenizer.hash").exists() else None
        if stored and stored != tok.content_hash:
            raise ValueError("tokenizer content hash mismatch: files were modified")
        return tok


def _try_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _stable_key_hash(key: str) -> int:
    """A 64-bit hash of ``key`` that is stable across processes and runs.

    Python's built-in ``hash`` is salted per-process (``PYTHONHASHSEED``); the
    per-key reservoir stream is seeded from this hash, so it must be a pure
    function of the key text to keep fits byte-identical across workers and runs.
    """
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


# Reservoir stream salt: a fixed offset mixed into ``(seed, salt, key_hash)`` so
# the per-key numeric reservoir draws are independent of any other RNG use.
_RESERVOIR_SALT = 982_451_653
# Per-key cap on the textual-BPE corpus: bounds memory for high-cardinality text
# while keeping the corpus a deterministic, frequency-ordered slice.
_TEXT_CORPUS_CAP = 60_000


@dataclass
class _FitAccum:
    """Mergeable, picklable streaming-fit state for one slice of the dataset.

    The tokenizer fit is a fold over event/profile batches that is the same
    whether it runs in one process or many: a worker folds its row-group range
    into one ``_FitAccum`` and the parent merges the per-task accumulators in
    strict task (file) order. Every field is a monoid under :meth:`merge` whose
    result, for any split of the row stream, equals folding the whole stream —
    which is what makes the fitted tokenizer byte-identical across worker counts
    (global rule 2).

    Per-key state:

    - ``counters`` — value→count over categorical/text candidates; ``update``
      preserves first-seen (insertion) order so ``most_common()`` tie-breaks are
      stable;
    - ``numeric_ok`` / ``numeric_n`` — finite-parse and total counts driving the
      numeric-vs-categorical decision (a pure ratio, order-independent);
    - ``numeric_sample`` — every finite value in row order; the parent derives
      the binning sample from these (a prefix below the cap, else a reservoir
      replay — see :meth:`PragmaTokenizer.fit`), so binning never depends on how
      rows were partitioned.
    """

    counters: dict[str, Counter] = dataclasses.field(default_factory=dict)
    numeric_ok: Counter = dataclasses.field(default_factory=Counter)
    numeric_n: Counter = dataclasses.field(default_factory=Counter)
    numeric_sample: dict[str, list[float]] = dataclasses.field(default_factory=dict)

    def see(self, key: str, values: np.ndarray) -> None:
        """Fold one key's batch of raw (string) values into the accumulator."""
        vals = values[values != ""]
        if len(vals) == 0:
            return
        # Full-slice parse: numeric detection only needs the finite-parse ratio,
        # which is order-independent, so no subsampling RNG is involved.
        parsed = np.array([_try_float(v) for v in vals], dtype=np.float64)
        ok = np.isfinite(parsed)
        self.numeric_ok[key] += int(ok.sum())
        self.numeric_n[key] += len(vals)
        finite = parsed[ok]
        if len(finite):
            self.numeric_sample.setdefault(key, []).extend(finite.tolist())
        self.counters.setdefault(key, Counter()).update(vals.tolist())

    def consume_events_batch(self, batch: pa.RecordBatch) -> None:
        """Fold one ``events.parquet`` batch (``source`` + ``fields`` map)."""
        fields = batch.column("fields")
        flat_keys = fields.combine_chunks().keys if hasattr(fields, "combine_chunks") else fields.keys
        keys_np = np.asarray(flat_keys, dtype=object)
        items_np = np.asarray(fields.items, dtype=object)
        for k in np.unique(keys_np):
            self.see(str(k), items_np[keys_np == k])
        self.see("source", np.asarray(batch.column("source"), dtype=object))

    def consume_profiles_batch(self, batch: pa.RecordBatch) -> None:
        """Fold one ``profiles.parquet`` batch (attributes + lifelong keys)."""
        attrs = batch.column("attributes")
        keys_np = np.asarray(attrs.keys, dtype=object)
        items_np = np.asarray(attrs.items, dtype=object)
        for k in np.unique(keys_np):
            self.see(str(k), items_np[keys_np == k])
        ll = batch.column("lifelong")
        ll_keys = np.asarray(ll.flatten().field("key"), dtype=object)
        for k in np.unique(ll_keys):
            # lifelong items are key-presence tokens: value = key name
            self.see(str(k), np.full((ll_keys == k).sum(), str(k), dtype=object))

    def merge(self, other: _FitAccum) -> None:
        """Fold ``other`` (a later task's range) into ``self``, in task order."""
        for k, c in other.counters.items():
            self.counters.setdefault(k, Counter()).update(c)
        self.numeric_ok.update(other.numeric_ok)
        self.numeric_n.update(other.numeric_n)
        for k, vs in other.numeric_sample.items():
            self.numeric_sample.setdefault(k, []).extend(vs)


def _day_of_month(ts_us: np.ndarray) -> np.ndarray:
    if len(ts_us) == 0:
        return np.zeros(0, dtype=np.int64)
    days = ts_us.astype("datetime64[us]").astype("datetime64[D]")
    return (days - days.astype("datetime64[M]")).astype(np.int64) + 1


def _utc_offsets_us(ts_us: np.ndarray, tz: str) -> np.ndarray:
    """Per-instant UTC offset (µs) of timezone ``tz``, DST-correct."""
    import datetime as _dt
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(tz)
    offs = np.empty(ts_us.shape[0], dtype=np.int64)
    for i, t in enumerate(ts_us.tolist()):
        local = _dt.datetime.fromtimestamp(t / 1e6, tz=_dt.UTC).astimezone(zone)
        offs[i] = int((local.utcoffset() or _dt.timedelta()).total_seconds()) * 1_000_000
    return offs


def _calendar_fields(ts_us: np.ndarray, tz: str = "UTC") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(hour, day_of_week, day_of_month)`` from epoch-µs instants in ``tz``.

    ``"UTC"`` uses fast integer arithmetic (no per-row timezone math); a non-UTC
    zone shifts each instant by its DST-correct local offset first, so the
    calendar features reflect local wall-clock time. Monday is day-of-week 0
    (1970-01-01 was a Thursday).
    """
    local = ts_us if tz == "UTC" else ts_us + _utc_offsets_us(ts_us, tz)
    hour = ((local % DAY_US) // HOUR_US).astype(np.int8)
    days = local // DAY_US
    dow = ((days + 3) % 7).astype(np.int8)
    dom = _day_of_month(local).astype(np.int8)
    return hour, dow, dom


def _text_corpus(counter: Counter) -> list[str]:
    """A text key's BPE corpus: values frequency-weighted in ``most_common()``
    order, capped at ``_TEXT_CORPUS_CAP`` strings (deterministic, bounded)."""
    corpus: list[str] = []
    for value, count in counter.most_common():
        room = _TEXT_CORPUS_CAP - len(corpus)
        if room <= 0:
            break
        corpus.extend([str(value)] * min(count, room))
    return corpus


def _train_bpe(corpus: list[str], vocab_size: int, min_frequency: int) -> Any:
    """Train a byte-level BPE on the textual-field corpus (lossless decode)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok


def _load_profiles(data_dir: Path) -> dict[str, tuple[dict[str, str], list[tuple[str, int]], int]]:
    """Materialize profiles.parquet as ``{user_id: (attributes, lifelong, as_of_us)}``; last duplicate wins."""
    profiles: dict[str, tuple[dict[str, str], list[tuple[str, int]], int]] = {}
    pfp = pq.ParquetFile(data_dir / "profiles.parquet")
    for batch in pfp.iter_batches(batch_size=16_384):
        d = batch.to_pylist()
        for row in d:
            attrs = dict(row["attributes"])
            ll = [(e["key"], _ts_to_us(e["ts"])) for e in row["lifelong"]]
            profiles[row["user_id"]] = (attrs, ll, _ts_to_us(row["as_of"]))
    return profiles


def _iter_records_from_batches(
    batches: Iterator[pa.RecordBatch],
    profiles: dict[str, tuple[dict[str, str], list[tuple[str, int]], int]],
    max_users: int | None,
) -> Iterator[UserRecord]:
    """Group adjacent equal-uid rows into UserRecords.

    Cuts ONLY on adjacent-uid change; independent of batch segmentation
    (``cur_uid``/``cur_events`` carry across batches).
    """
    cur_uid: str | None = None
    cur_events: list[tuple[int, str, dict[str, str]]] = []
    seen: set[str] = set()
    yielded = 0
    for batch in batches:
        rows = batch.to_pylist()
        for row in rows:
            uid = row["user_id"]
            if uid != cur_uid:
                if cur_uid is not None:
                    seen.add(cur_uid)
                    yield _mk_record(cur_uid, cur_events, profiles)
                    yielded += 1
                    if max_users is not None and yielded >= max_users:
                        return
                if uid in seen:
                    # Non-adjacent rows for one user would fragment into multiple
                    # records that overwrite each other's index entry — refuse
                    # rather than silently corrupt the shards.
                    raise ValueError(
                        f"user_id {uid!r} appears in non-adjacent rows; events must be grouped "
                        "by user. Sort by (user_id, ts) before tokenizing."
                    )
                cur_uid = uid
                cur_events = []
            cur_events.append((_ts_to_us(row["ts"]), row["source"], dict(row["fields"])))
    if cur_uid is not None and (max_users is None or yielded < max_users):
        yield _mk_record(cur_uid, cur_events, profiles)


def iter_user_records(data_dir: str | Path, max_users: int | None = None) -> Iterator[UserRecord]:
    """Stream :class:`UserRecord` objects from a generated dataset directory.

    Events are user-major on disk, so this is a sequential merge of
    ``events.parquet`` row groups with ``profiles.parquet``.
    """
    data_dir = Path(data_dir)
    profiles = _load_profiles(data_dir)
    pf = pq.ParquetFile(data_dir / "events.parquet")
    yield from _iter_records_from_batches(pf.iter_batches(batch_size=65_536), profiles, max_users)


def _mk_record(uid: str, events: list, profiles: dict) -> UserRecord:
    attrs, ll, as_of = profiles.get(uid, ({}, [], 0))
    return UserRecord(user_id=uid, events=events, attributes=attrs, lifelong=ll, as_of=as_of)


def _ts_to_us(ts: Any) -> int:
    """Normalize pyarrow/pandas/py datetime scalars to µs since epoch."""
    if ts is None:
        raise ValueError(
            "event/lifelong ts is null; drop or impute null timestamps before tokenizing"
        )
    if isinstance(ts, (int, np.integer)):
        return int(ts)
    if hasattr(ts, "timestamp"):  # datetime.datetime
        import datetime as _dt

        if isinstance(ts, _dt.datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=_dt.UTC)
        # exact integer µs: float-seconds * 1e6 loses up to 1µs vs timestamp[us]
        return (ts - _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)) // _dt.timedelta(microseconds=1)
    raise TypeError(f"cannot convert {type(ts)} to µs")
