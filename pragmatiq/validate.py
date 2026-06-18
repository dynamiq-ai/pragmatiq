"""Input validation with actionable errors.

``validate_dataset`` checks a raw dataset directory against the data contract
(schema.py) and flags the problems that silently corrupt training: wrong dtypes,
non-monotonic timestamps per user, null ids, and pathological field cardinality.
Each issue is a human-readable string with the file and a concrete fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .data.schema import EVENTS_SCHEMA, LABEL_TASKS, PROFILES_SCHEMA, SOURCES, TRANSFERS_SCHEMA, label_schema


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
    from collections import defaultdict

    pf = pq.ParquetFile(data_dir / "events.parquet")
    occ: dict[str, int] = defaultdict(int)
    distinct: dict[str, set] = defaultdict(set)
    nullv: dict[str, int] = defaultdict(int)
    n = 0
    for batch in pf.iter_batches(columns=["fields"], batch_size=65_536):
        f = batch.column("fields")
        keys = f.keys.to_pylist() if hasattr(f.keys, "to_pylist") else list(f.keys)
        items = f.items.to_pylist() if hasattr(f.items, "to_pylist") else list(f.items)
        for k, v in zip(keys, items):
            occ[k] += 1
            if v is None:
                nullv[k] += 1
            s = distinct[k]
            if len(s) <= cap:
                s.add(v)
        n += len(batch)
        if n >= sample_rows:
            break
    for k, c in nullv.items():
        report.warn(f"events.parquet: field '{k}' has {c} null values — tokenized as [UNK]; "
                    "impute or drop them if that is not intended")
    for k, s in distinct.items():
        d, o = len(s), occ[k]
        sample = list(s)[:500]
        numeric = sample and sum(_is_floatish(v) for v in sample) >= 0.9 * len(sample)
        if o >= 1000 and d >= 0.9 * o and not numeric:
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
    df = pq.read_table(path, columns=["from_user", "to_user"]).to_pandas()
    if df["from_user"].isna().any() or df["to_user"].isna().any():
        report.error("transfers.parquet: null from_user/to_user — drop or impute these edges")
    self_loops = int((df["from_user"] == df["to_user"]).sum())
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
    last_uid = None
    last_ts = None
    null_ids = 0
    null_ts = 0
    bad_source = set()
    out_of_order = 0
    closed: set[str] = set()  # uids whose run of adjacent rows has ended
    non_adjacent_flagged = False
    for batch in pf.iter_batches(columns=["user_id", "ts", "source"], batch_size=131_072):
        uids = batch.column("user_id").to_pylist()
        tss = batch.column("ts").cast(pa.int64()).to_pylist()
        srcs = batch.column("source").to_pylist()
        for uid, ts, src in zip(uids, tss, srcs):
            n_rows += 1
            if uid is None or uid == "":
                null_ids += 1
            if src not in SOURCES:
                bad_source.add(src)
            # events must be time-sorted within each user (null ts can't be compared)
            if ts is None:
                null_ts += 1
            elif uid == last_uid and last_ts is not None and ts < last_ts:
                out_of_order += 1
            if uid != last_uid:
                if last_uid is not None:
                    closed.add(last_uid)
                if uid in closed and not non_adjacent_flagged:
                    r.error(f"events.parquet: user {uid!r} appears in non-adjacent rows — sort by "
                            "(user_id, ts) before tokenizing; otherwise the user fragments into "
                            "multiple records that overwrite each other's index entry")
                    non_adjacent_flagged = True
            last_uid = uid
            if ts is not None:
                last_ts = ts
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

    prof = pq.read_table(data_dir / "profiles.parquet", columns=["user_id"]).to_pandas()
    if prof["user_id"].isna().any():
        r.error("profiles.parquet: null user_id present")
    if prof["user_id"].duplicated().any():
        r.warn("profiles.parquet: duplicate user_id rows (the last one wins at tokenization)")

    _check_field_cardinality(data_dir, r)
    _check_transfers(data_dir, r)
    _check_labels(data_dir, r)

    return r
