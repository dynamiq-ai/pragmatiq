"""Input validation with actionable errors.

``validate_dataset`` checks a raw dataset directory against the data contract
(schema.py) and flags the problems that silently corrupt training: wrong dtypes,
non-monotonic timestamps per user, null ids, and pathological field cardinality.
Each issue is a human-readable string with the file and a concrete fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as _pc
import pyarrow.parquet as pq

from pragmatiq.core.schema import (
    EVENTS_SCHEMA,
    LABEL_TASKS,
    PROFILES_SCHEMA,
    SOURCES,
    TRANSFERS_SCHEMA,
    label_schema,
)

# pyarrow.compute builds its kernels at import time, so the stubs do not list them.
pc: Any = _pc


@dataclass
class ValidationReport:
    """Collected validation findings; ``ok`` is True iff there are no errors."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        out = [f"{'OK' if self.ok else 'FAILED'}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        out += [f"  ERROR: {e}" for e in self.errors]
        out += [f"  warn:  {w}" for w in self.warnings]
        return "\n".join(out)


def _check_schema(path: Path, expected: pa.Schema, report: ValidationReport) -> pa.Schema | None:
    if not path.exists():
        report.error(f"{path.name} is missing at {path}")
        return None
    schema = pq.read_schema(path)
    for f in expected:
        if f.name not in schema.names:
            report.error(f"{path.name}: missing column '{f.name}' (expected {f.type})")
            continue
        actual = schema.field(f.name).type
        if actual.equals(f.type):
            continue
        # A timezone-aware timestamp is a valid instant — accept it (calendar
        # localization is a tokenizer knob, calendar_tz). Never advise dropping
        # the zone, which would silently shift the instant to UTC.
        if pa.types.is_timestamp(f.type) and pa.types.is_timestamp(actual) and actual.unit == f.type.unit:
            continue
        report.error(f"{path.name}: column '{f.name}' has dtype {actual}, "
                     f"expected {f.type} — cast it before training")
    return schema


def _is_floatish(v: object) -> bool:
    try:
        float(v)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def _check_field_cardinality(data_dir: Path, report: ValidationReport,
                             sample_rows: int = 200_000, cap: int = 200_000) -> None:
    """Flag pathological field cardinality: a non-numeric
    event field that is (near-)unique per occurrence is almost always a per-event
    identifier — it explodes the categorical/BPE vocab and carries no learnable
    signal. Numeric magnitude fields are skipped (the tokenizer percentile-bins them)."""
    pf = pq.ParquetFile(data_dir / "events.parquet")
    key_parts: list[pa.Array] = []
    item_parts: list[pa.Array] = []
    n = 0
    for batch in pf.iter_batches(columns=["fields"], batch_size=65_536):
        f = batch.column("fields")
        key_parts.append(f.keys)
        item_parts.append(f.items)
        n += len(batch)
        if n >= sample_rows:
            break
    if not key_parts:
        return
    keys = pa.concat_arrays(key_parts)
    items = pa.concat_arrays(item_parts)
    # One grouped aggregation replaces a Python loop over every (key, value)
    # pair: occurrences, nulls, and distinct values per key.
    tbl = pa.table({"k": keys, "isnull": pc.cast(pc.is_null(items), pa.int64()), "v": items})
    agg = tbl.group_by("k").aggregate([
        ("isnull", "sum"), ("v", "count", pc.CountOptions(mode="all")), ("v", "count_distinct"),
    ])
    stats = {row["k"]: (int(row["isnull_sum"] or 0), int(row["v_count"]), int(row["v_count_distinct"]))
             for row in agg.to_pylist()}
    for k, (nulls, _o, _d) in stats.items():
        if nulls:
            report.warn(f"events.parquet: field '{k}' has {nulls} null values — tokenized as [UNK]; "
                        "impute or drop them if that is not intended")
    for k, (_nulls, o, d) in stats.items():
        d = min(d, cap)
        if not (o >= 1000 and d >= 0.9 * o):
            continue
        sample = pc.unique(pc.filter(items, pc.equal(keys, k))).slice(0, 500).to_pylist()
        numeric = sample and sum(_is_floatish(v) for v in sample) >= 0.9 * len(sample)
        if not numeric:
            report.warn(
                f"events.parquet: field '{k}' has ~{d} distinct values over {o} occurrences "
                "(near-unique per event) — likely a per-event identifier that will explode the "
                "tokenizer vocab or fall back to [UNK]/BPE. Drop, hash, or bucket it before tokenizing."
            )


def _check_transfers(data_dir: Path, report: ValidationReport) -> None:
    """Validate transfers.parquet if present (it feeds the AML graph): schema +
    null/self-loop ids. Part of the four-file data contract (README)."""
    path = data_dir / "transfers.parquet"
    if not path.exists():
        return
    if _check_schema(path, TRANSFERS_SCHEMA, report) is None:
        return
    t = pq.read_table(path, columns=["from_user", "to_user"])
    if pc.any(pc.is_null(t["from_user"])).as_py() or pc.any(pc.is_null(t["to_user"])).as_py():
        report.error("transfers.parquet: null from_user/to_user — drop or impute these edges")
    self_loops = int(pc.sum(pc.fill_null(pc.equal(t["from_user"], t["to_user"]), False)).as_py() or 0)
    if self_loops:
        report.warn(f"transfers.parquet: {self_loops} self-loop edges (from_user == to_user)")


def _check_labels(data_dir: Path, report: ValidationReport) -> None:
    """Validate known label tables under labels/ against their task schemas."""
    labels_dir = data_dir / "labels"
    if not labels_dir.exists():
        return
    known = set(LABEL_TASKS)
    for path in sorted(labels_dir.glob("*.parquet")):
        task = path.stem
        if task not in known:
            report.warn(f"labels/{path.name}: unknown label task — validate schema manually")
            continue
        expected = label_schema(task)
        schema = _check_schema(path, expected, report)
        if schema is None:
            continue
        extra = sorted(set(schema.names) - set(expected.names))
        if extra:
            report.warn(f"labels/{path.name}: extra column(s) {extra} beyond {expected.names}; "
                        "the trainer reads only the expected columns, so these are ignored")


def validate_dataset(data_dir: str | Path, max_rows: int | None = 2_000_000) -> ValidationReport:
    """Validate a raw dataset directory; returns a :class:`ValidationReport`."""
    data_dir = Path(data_dir)
    r = ValidationReport()

    _check_schema(data_dir / "events.parquet", EVENTS_SCHEMA, r)
    _check_schema(data_dir / "profiles.parquet", PROFILES_SCHEMA, r)
    if r.errors:  # schema problems make deeper checks meaningless
        return r

    ev = data_dir / "events.parquet"
    pf = pq.ParquetFile(ev)
    n_rows = 0
    null_ids = 0
    null_ts = 0
    bad_source: set[str] = set()
    out_of_order = 0
    closed: set[str] = set()  # uids whose run of adjacent rows has ended
    non_adjacent_flagged = False
    # Carry the previous batch's last row so runs and ordering are checked
    # across batch boundaries exactly as in a single row-by-row pass.
    carry_uid: str | None = None
    carry_ts: int | None = None
    have_carry = False
    for batch in pf.iter_batches(columns=["user_id", "ts", "source"], batch_size=131_072):
        n = batch.num_rows
        if n == 0:
            continue
        n_rows += n
        uids = batch.column("user_id")
        ts = batch.column("ts").cast(pa.int64())
        null_ids += int(pc.sum(pc.fill_null(pc.or_kleene(pc.is_null(uids), pc.equal(uids, "")), False)).as_py() or 0)
        null_ts += int(pc.sum(pc.is_null(ts)).as_py() or 0)
        bad_source.update(str(s) for s in pc.unique(batch.column("source")).to_pylist() if s not in SOURCES)
        if have_carry:
            uids_x = pa.concat_arrays([pa.array([carry_uid], type=pa.string()), uids])
            ts_x = pa.concat_arrays([pa.array([carry_ts], type=pa.int64()), ts])
        else:
            uids_x, ts_x = uids, ts
        m = len(uids_x)
        if m > 1:
            # Row j+1 continues row j's run when the user_id repeats; within a run an
            # event is out of order when its ts precedes the last non-null ts seen.
            same = np.asarray(pc.fill_null(pc.equal(uids_x.slice(1), uids_x.slice(0, m - 1)), False))
            ts_ff = pc.fill_null_forward(ts_x)
            earlier = np.asarray(pc.fill_null(pc.less(ts_x.slice(1), ts_ff.slice(0, m - 1)), False))
            out_of_order += int(np.count_nonzero(same & earlier))
            starts = np.flatnonzero(~same) + 1  # rows (in uids_x) that begin a new run
            if starts.size:
                prev_uids = uids_x.take(pa.array(starts - 1)).to_pylist()
                new_uids = uids_x.take(pa.array(starts)).to_pylist()
                for prev_uid, uid in zip(prev_uids, new_uids):
                    if prev_uid is not None:
                        closed.add(prev_uid)
                    if uid in closed and not non_adjacent_flagged:
                        r.error(f"events.parquet: user {uid!r} appears in non-adjacent rows — sort by "
                                "(user_id, ts) before tokenizing; otherwise the user fragments into "
                                "multiple records that overwrite each other's index entry")
                        non_adjacent_flagged = True
        else:
            ts_ff = pc.fill_null_forward(ts_x)
        carry_uid = uids_x[m - 1].as_py()
        carry_ts = ts_ff[m - 1].as_py()
        have_carry = True
        if max_rows is not None and n_rows >= max_rows:
            r.warn(f"events.parquet: stopped checking after {max_rows} rows")
            break

    if null_ids:
        r.error(f"events.parquet: {null_ids} rows with null/empty user_id — drop or impute them")
    if null_ts:
        r.error(f"events.parquet: {null_ts} rows with null ts — drop or impute them")
    if bad_source:
        r.error(f"events.parquet: unknown source(s) {sorted(bad_source)} — allowed: {list(SOURCES)}")
    if out_of_order:
        r.error(f"events.parquet: {out_of_order} events out of time order within a user — "
                "sort by (user_id, ts) before tokenizing")

    prof_uids = pq.read_table(data_dir / "profiles.parquet", columns=["user_id"]).column("user_id")
    if pc.any(pc.is_null(prof_uids)).as_py():
        r.error("profiles.parquet: null user_id present")
    if len(pc.unique(prof_uids)) < len(prof_uids):
        r.warn("profiles.parquet: duplicate user_id rows (the last one wins at tokenization)")

    _check_field_cardinality(data_dir, r)
    _check_transfers(data_dir, r)
    _check_labels(data_dir, r)

    return r
