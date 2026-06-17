"""Shard writer + LMDB user index for tokenized records (Phase 3).

``ShardWriter`` streams :class:`TokenizedRecord` objects to parquet shards
partitioned by event-count band (so a batch sampler can pack users of similar
length cheaply), and builds a :class:`UserIndex` in LMDB mapping
``user_id -> {shard, row, band, n_events, n_tokens, profile stats}``.

Each record is stored as one parquet row whose columns are the record's flat
arrays as Arrow large-lists — lossless and fast to read back with
:func:`record_from_row`.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .tokenizer import TokenizedRecord

# Event-count band upper bounds; a record with n_events <= bound joins that band.
DEFAULT_BANDS: tuple[int, ...] = (8, 32, 128, 512, 2048, 8192, 1_000_000)

_ARRAY_FIELDS: tuple[tuple[str, Any], ...] = (
    ("key_ids", pa.int32()),
    ("value_ids", pa.int32()),
    ("positions", pa.int16()),
    ("event_offsets", pa.int64()),
    ("event_ts", pa.int64()),
    ("time_log", pa.float32()),
    ("hour", pa.int8()),
    ("dow", pa.int8()),
    ("dom", pa.int8()),
    ("source_ids", pa.int8()),
    ("prof_key_ids", pa.int32()),
    ("prof_value_ids", pa.int32()),
    ("prof_positions", pa.int16()),
    ("prof_offsets", pa.int64()),
    ("prof_time_log", pa.float32()),
    ("prof_ts", pa.int64()),
)

SHARD_SCHEMA = pa.schema(
    [pa.field("user_id", pa.string())] + [pa.field(n, pa.large_list(t)) for n, t in _ARRAY_FIELDS]
)

# The Nemotron variant (text_value_mode="embed") additionally stores per-token text
# markers + raw strings. These columns are written ONLY for bands that actually carry
# text, so BPE-mode shards are byte-identical to a build without the variant.
SHARD_SCHEMA_TEXT = (
    SHARD_SCHEMA.append(pa.field("is_text", pa.large_list(pa.int8())))
    .append(pa.field("text_values", pa.large_list(pa.large_string())))
)


@dataclass
class UserMeta:
    """Per-user index entry stored in LMDB (token stats + shard location)."""

    user_id: str
    band: int
    shard: int
    row: int
    n_events: int
    n_tokens: int
    n_prof_items: int
    n_prof_tokens: int

    def pack(self) -> bytes:
        """Serialize this index entry to compact JSON bytes for LMDB."""
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def unpack(cls, blob: bytes) -> UserMeta:
        """Reconstruct a :class:`UserMeta` from its LMDB JSON bytes."""
        return cls(**json.loads(blob.decode()))


def _record_to_arrays(rec: TokenizedRecord) -> dict[str, Any]:
    return {
        "key_ids": rec.key_ids, "value_ids": rec.value_ids, "positions": rec.positions,
        "event_offsets": rec.event_offsets, "event_ts": rec.event_ts, "time_log": rec.time_log,
        "hour": rec.hour, "dow": rec.dow, "dom": rec.dom, "source_ids": rec.source_ids,
        "prof_key_ids": rec.prof_key_ids, "prof_value_ids": rec.prof_value_ids,
        "prof_positions": rec.prof_positions, "prof_offsets": rec.prof_offsets,
        "prof_time_log": rec.prof_time_log, "prof_ts": rec.prof_ts,
        # Nemotron text path (see SHARD_SCHEMA_TEXT); written only when present.
        "is_text": np.asarray(rec.is_text, dtype=np.int8),
        "text_values": list(rec.text_values),
    }


def record_from_row(row: dict[str, Any]) -> TokenizedRecord:
    """Reconstruct a :class:`TokenizedRecord` from a parquet shard row."""
    def arr(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(row[name], dtype=dtype)

    return TokenizedRecord(
        user_id=row["user_id"],
        key_ids=arr("key_ids", np.int32), value_ids=arr("value_ids", np.int32),
        positions=arr("positions", np.int16), event_offsets=arr("event_offsets", np.int64),
        event_ts=arr("event_ts", np.int64), time_log=arr("time_log", np.float32),
        hour=arr("hour", np.int8), dow=arr("dow", np.int8), dom=arr("dom", np.int8),
        source_ids=arr("source_ids", np.int8),
        prof_key_ids=arr("prof_key_ids", np.int32), prof_value_ids=arr("prof_value_ids", np.int32),
        prof_positions=arr("prof_positions", np.int16), prof_offsets=arr("prof_offsets", np.int64),
        prof_time_log=arr("prof_time_log", np.float32),
        # A row missing the prof_ts column reads back as "no timestamps" (empty
        # array); truncate_record then requires it explicitly before cutting.
        prof_ts=np.asarray(row.get("prof_ts") if row.get("prof_ts") is not None else [],
                           dtype=np.int64),
        # Text columns exist only for embed-mode shards; absent → no text tokens.
        is_text=(np.asarray(row["is_text"], dtype=np.int8) if row.get("is_text") is not None
                 else np.zeros(len(row["key_ids"]), dtype=np.int8)),
        text_values=list(row["text_values"]) if row.get("text_values") is not None else [],
    )


def band_of(n_events: int, bands: tuple[int, ...] = DEFAULT_BANDS) -> int:
    """Index of the first band whose bound is >= ``n_events``."""
    for i, b in enumerate(bands):
        if n_events <= b:
            return i
    return len(bands) - 1


class ShardWriter:
    """Writes tokenized records to per-band parquet shards + an LMDB user index."""

    def __init__(
        self,
        out_dir: str | Path,
        tokenizer_hash: str,
        bands: tuple[int, ...] = DEFAULT_BANDS,
        rows_per_shard: int = 4096,
        map_size: int = 1 << 34,
    ) -> None:
        self.out = Path(out_dir)
        (self.out / "shards").mkdir(parents=True, exist_ok=True)
        self.bands = bands
        self.rows_per_shard = rows_per_shard
        self.tokenizer_hash = tokenizer_hash
        self._buffers: dict[int, list[dict[str, np.ndarray]]] = {b: [] for b in range(len(bands))}
        self._uids: dict[int, list[str]] = {b: [] for b in range(len(bands))}
        self._shard_counter: dict[int, int] = {b: 0 for b in range(len(bands))}
        self._index: list[UserMeta] = []
        self._profiles: dict[str, bytes] = {}
        self._map_size = map_size

    def add(self, rec: TokenizedRecord, profile: dict[str, Any] | None = None) -> None:
        """Buffer one record into its event-count band, flushing full shards.

        ``profile`` (raw attributes + lifelong + as_of) is stored verbatim in the
        LMDB index so notebooks/serving can fetch a user's profile blob without
        opening a shard (Phase 3: "user_id -> profile blob + token stats").
        """
        b = band_of(rec.n_events, self.bands)
        self._buffers[b].append(_record_to_arrays(rec))
        self._uids[b].append(rec.user_id)
        meta = UserMeta(
            user_id=rec.user_id, band=b, shard=self._shard_counter[b],
            row=len(self._buffers[b]) - 1, n_events=rec.n_events, n_tokens=rec.n_tokens,
            n_prof_items=len(rec.prof_offsets) - 1, n_prof_tokens=int(rec.prof_key_ids.size),
        )
        self._index.append(meta)
        if profile is not None:
            self._profiles[rec.user_id] = json.dumps(profile, separators=(",", ":")).encode()
        if len(self._buffers[b]) >= self.rows_per_shard:
            self._flush_band(b)

    def _flush_band(self, b: int) -> None:
        recs = self._buffers[b]
        if not recs:
            return
        cols: dict[str, list] = {"user_id": self._uids[b]}
        for name, _ in _ARRAY_FIELDS:
            cols[name] = [r[name].tolist() for r in recs]
        # Only carry the text columns when the band actually has text (embed mode),
        # so BPE-mode shards stay byte-identical to a build without the variant.
        has_text = any(int(r["is_text"].sum()) > 0 for r in recs)
        if has_text:
            cols["is_text"] = [r["is_text"].tolist() for r in recs]
            cols["text_values"] = [list(r["text_values"]) for r in recs]
        schema = SHARD_SCHEMA_TEXT if has_text else SHARD_SCHEMA
        table = pa.table(cols, schema=schema)
        path = self.out / "shards" / f"band{b}_shard{self._shard_counter[b]:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self._buffers[b] = []
        self._uids[b] = []
        self._shard_counter[b] += 1

    def close(self) -> dict[str, Any]:
        """Flush remaining shards, write the LMDB index + manifest; return stats."""
        for b in range(len(self.bands)):
            self._flush_band(b)
        self._write_index()
        manifest = {
            "tokenizer_hash": self.tokenizer_hash,
            "bands": list(self.bands),
            "n_users": len(self._index),
            "n_shards": sum(self._shard_counter.values()),
            "total_tokens": int(sum(m.n_tokens for m in self._index)),
            "total_events": int(sum(m.n_events for m in self._index)),
        }
        (self.out / "shard_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest

    def _write_index(self) -> None:
        import lmdb

        # Invalidate any cached read-only env at this path (shards rewritten).
        key = str((self.out / "user_index.lmdb").resolve())
        stale = _ENV_CACHE.pop(key, None)
        if stale is not None:
            stale.close()
        env = lmdb.open(str(self.out / "user_index.lmdb"), map_size=self._map_size, subdir=True)
        with env.begin(write=True) as txn:
            order = []
            seen_uids: set[str] = set()
            for m in self._index:
                if m.user_id in seen_uids:
                    raise ValueError(
                        f"duplicate user_id {m.user_id!r} in shard index — each user must map to "
                        "exactly one record; sort events by (user_id, ts) before tokenizing"
                    )
                seen_uids.add(m.user_id)
                txn.put(m.user_id.encode(), m.pack())
                order.append(m.user_id)
                blob = self._profiles.get(m.user_id)
                if blob is not None:
                    txn.put(b"prof:" + m.user_id.encode(), blob)
            # ordered user-id list + fast columnar tables for the sampler
            txn.put(b"__order__", json.dumps(order).encode())
            tok = np.array([m.n_tokens for m in self._index], dtype=np.int64)
            ptok = np.array([m.n_prof_tokens for m in self._index], dtype=np.int64)
            evs = np.array([m.n_events for m in self._index], dtype=np.int64)
            bnd = np.array([m.band for m in self._index], dtype=np.int16)
            txn.put(b"__tokens__", tok.tobytes())
            txn.put(b"__prof_tokens__", ptok.tobytes())
            txn.put(b"__events__", evs.tobytes())
            txn.put(b"__bands__", bnd.tobytes())
            txn.put(b"__count__", struct.pack("<q", len(self._index)))
        env.sync()
        env.close()


# Process-level cache of read-only LMDB environments. LMDB forbids opening the
# same env path twice in one process, so multiple UserIndex/ShardDataset handles
# over the same shards must share one environment.
_ENV_CACHE: dict[str, Any] = {}


def _open_env(path: Path) -> Any:
    import lmdb

    key = str(path.resolve())
    env = _ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(key, readonly=True, subdir=True, lock=False, max_readers=512)
        _ENV_CACHE[key] = env
    return env


class UserIndex:
    """Read-only view over the LMDB user index built by :class:`ShardWriter`.

    Environments are process-cached and shared, so opening several indices over
    the same shards is safe; ``close`` therefore does not unmap the env.
    """

    def __init__(self, shard_dir: str | Path) -> None:
        self.dir = Path(shard_dir)
        self.env = _open_env(self.dir / "user_index.lmdb")

        def _need(txn: Any, key: bytes) -> bytes:
            v = txn.get(key)
            if v is None:
                raise KeyError(f"corrupt user index: missing {key!r}")
            return bytes(v)

        with self.env.begin() as txn:
            self.order: list[str] = json.loads(_need(txn, b"__order__").decode())
            self.n_tokens = np.frombuffer(_need(txn, b"__tokens__"), dtype=np.int64)
            self.n_prof_tokens = np.frombuffer(_need(txn, b"__prof_tokens__"), dtype=np.int64)
            self.n_events = np.frombuffer(_need(txn, b"__events__"), dtype=np.int64)
            self.bands = np.frombuffer(_need(txn, b"__bands__"), dtype=np.int16)

    def __len__(self) -> int:
        return len(self.order)

    def meta(self, user_id: str) -> UserMeta:
        """Look up one user's index entry."""
        with self.env.begin() as txn:
            blob = txn.get(user_id.encode())
        if blob is None:
            raise KeyError(f"user {user_id!r} not in index")
        return UserMeta.unpack(bytes(blob))

    def profile(self, user_id: str) -> dict[str, Any] | None:
        """Return the stored raw profile blob (attributes + lifelong) or None."""
        with self.env.begin() as txn:
            blob = txn.get(b"prof:" + user_id.encode())
        return json.loads(bytes(blob).decode()) if blob is not None else None

    def close(self) -> None:
        """Drop this handle. The cached env stays open (shared, read-only)."""
        # Intentionally a no-op on the env: it is process-cached and may be
        # shared by other live handles. It is unmapped at process exit.
        return None
