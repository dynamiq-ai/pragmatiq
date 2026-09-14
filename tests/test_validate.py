"""Dataset-validation tests: actionable errors."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pragmatiq.data.synthetic import WorldConfig, generate
from pragmatiq.validate import validate_dataset


@pytest.fixture(scope="module")
def good_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("valid")
    generate(WorldConfig(n_users=120, months=14, n_merchants=400, mule_ring_count=1, seed=2,
                         eval_month_credit=2, eval_month_short=8),
             out, n_workers=0, write_report=False)
    return out


def test_valid_dataset_passes(good_data: Path) -> None:
    report = validate_dataset(good_data)
    assert report.ok, report.summary()


def test_missing_file_flagged(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert any("missing" in e for e in report.errors)


def test_out_of_order_events_flagged(good_data: Path, tmp_path: Path) -> None:
    # build a dataset whose events are reverse-sorted within a user
    ev = pq.read_table(good_data / "events.parquet").to_pandas()
    one = ev[ev["user_id"] == ev["user_id"].iloc[0]].sort_values("ts", ascending=False)
    rest = ev[ev["user_id"] != ev["user_id"].iloc[0]]
    import pandas as pd

    bad = pd.concat([one, rest])
    (tmp_path).mkdir(exist_ok=True)
    from pragmatiq.core.schema import EVENTS_SCHEMA

    pq.write_table(pa.Table.from_pandas(bad, schema=EVENTS_SCHEMA, preserve_index=False),
                   tmp_path / "events.parquet")
    # copy a valid profiles file
    import shutil

    shutil.copy(good_data / "profiles.parquet", tmp_path / "profiles.parquet")
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert any("out of time order" in e for e in report.errors)


def test_summary_renders(good_data: Path) -> None:
    report = validate_dataset(good_data)
    s = report.summary()
    assert "OK" in s or "FAILED" in s


def test_null_ts_flagged_not_crashed(good_data: Path, tmp_path: Path) -> None:
    import shutil

    import pandas as pd

    from pragmatiq.core.schema import EVENTS_SCHEMA

    ev = pq.read_table(good_data / "events.parquet").to_pandas()
    ev.loc[ev.index[0], "ts"] = pd.NaT  # a null timestamp (Spark/pandas default-nullable)
    nullable_ts = pa.schema([pa.field(f.name, f.type, nullable=(f.name == "ts")) for f in EVENTS_SCHEMA])
    tmp_path.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pandas(ev, schema=nullable_ts, preserve_index=False),
                   tmp_path / "events.parquet")
    shutil.copy(good_data / "profiles.parquet", tmp_path / "profiles.parquet")
    report = validate_dataset(tmp_path)  # must not raise a TypeError
    assert not report.ok
    assert any("null ts" in e for e in report.errors), report.summary()


def test_tz_aware_ts_accepted(good_data: Path, tmp_path: Path) -> None:
    import shutil

    ev = pq.read_table(good_data / "events.parquet")
    idx = ev.schema.get_field_index("ts")
    ts_tz = ev.column("ts").cast(pa.timestamp("us", tz="UTC"))
    ev2 = ev.set_column(idx, pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False), ts_tz)
    tmp_path.mkdir(exist_ok=True)
    pq.write_table(ev2, tmp_path / "events.parquet")
    shutil.copy(good_data / "profiles.parquet", tmp_path / "profiles.parquet")
    report = validate_dataset(tmp_path)
    # a timezone-aware instant is valid; it must NOT trigger a 'cast it' dtype error
    assert not any("'ts'" in e and "dtype" in e for e in report.errors), report.summary()
    assert report.ok, report.summary()


def test_nonadjacent_user_is_error(good_data: Path, tmp_path: Path) -> None:
    import shutil

    import pandas as pd

    from pragmatiq.core.schema import EVENTS_SCHEMA

    ev = pq.read_table(good_data / "events.parquet").to_pandas().sort_values(["user_id", "ts"])
    a, b = list(dict.fromkeys(ev["user_id"]))[:2]
    ra, rb = ev[ev["user_id"] == a], ev[ev["user_id"] == b]
    bad = pd.concat([ra.iloc[:2], rb.iloc[:2], ra.iloc[2:3]])  # a, b, then a again (non-adjacent)
    tmp_path.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pandas(bad, schema=EVENTS_SCHEMA, preserve_index=False),
                   tmp_path / "events.parquet")
    shutil.copy(good_data / "profiles.parquet", tmp_path / "profiles.parquet")
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert any("non-adjacent" in e for e in report.errors), report.summary()


def test_pathological_cardinality_warns(good_data: Path, tmp_path: Path) -> None:
    import shutil

    # add a near-unique-per-event field (a per-event id) to the events
    ev = pq.read_table(good_data / "events.parquet")
    f = ev.column("fields").combine_chunks()
    keys, items, offs = f.keys.to_pylist(), f.items.to_pylist(), f.offsets.to_pylist()
    nk, nv, noff = [], [], [0]
    for i in range(len(ev)):
        lo, hi = offs[i], offs[i + 1]
        nk += keys[lo:hi] + ["txn_ref"]
        nv += items[lo:hi] + [f"ref_{i:07d}"]
        noff.append(len(nk))
    newfields = pa.MapArray.from_arrays(pa.array(noff, pa.int32()), pa.array(nk), pa.array(nv))
    ev2 = ev.set_column(ev.schema.get_field_index("fields"), "fields", newfields)
    pq.write_table(ev2, tmp_path / "events.parquet")
    shutil.copy(good_data / "profiles.parquet", tmp_path / "profiles.parquet")
    report = validate_dataset(tmp_path)
    assert any("txn_ref" in w for w in report.warnings), report.summary()


def test_transfers_self_loops_flagged(good_data: Path, tmp_path: Path) -> None:
    import shutil

    from pragmatiq.core.schema import TRANSFERS_SCHEMA

    shutil.copy(good_data / "events.parquet", tmp_path / "events.parquet")
    shutil.copy(good_data / "profiles.parquet", tmp_path / "profiles.parquet")
    tr = pq.read_table(good_data / "transfers.parquet").to_pandas()
    tr.loc[tr.index[:3], "to_user"] = tr.loc[tr.index[:3], "from_user"].to_numpy()  # self-loops
    pq.write_table(pa.Table.from_pandas(tr, schema=TRANSFERS_SCHEMA, preserve_index=False),
                   tmp_path / "transfers.parquet")
    report = validate_dataset(tmp_path)
    assert any("self-loop" in w for w in report.warnings), report.summary()


def test_forecast_label_missing_eval_ts_flagged(good_data: Path, tmp_path: Path) -> None:
    import shutil

    shutil.copytree(good_data, tmp_path, dirs_exist_ok=True)
    labels = pq.read_table(good_data / "labels" / "default_12m.parquet")
    idx = labels.schema.get_field_index("eval_ts")
    bad = labels.remove_column(idx)
    pq.write_table(bad, tmp_path / "labels" / "default_12m.parquet")
    report = validate_dataset(tmp_path)
    assert not report.ok
    assert any("default_12m.parquet" in e and "eval_ts" in e for e in report.errors), report.summary()


def test_unknown_label_table_warns(good_data: Path, tmp_path: Path) -> None:
    import shutil

    shutil.copytree(good_data, tmp_path, dirs_exist_ok=True)
    shutil.copy(good_data / "labels" / "aml.parquet", tmp_path / "labels" / "custom.parquet")
    report = validate_dataset(tmp_path)
    assert report.ok, report.summary()
    assert any("unknown label task" in w for w in report.warnings), report.summary()


def _reference_scan(events_path: Path, max_rows: int | None = None) -> dict:
    """The pre-1.1 row-by-row events scan, kept verbatim as the oracle."""
    from pragmatiq.core.schema import SOURCES

    pf = pq.ParquetFile(events_path)
    n_rows = 0
    last_uid = None
    last_ts = None
    null_ids = null_ts = out_of_order = 0
    bad_source: set = set()
    closed: set = set()
    non_adjacent: str | None = None
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
            if ts is None:
                null_ts += 1
            elif uid == last_uid and last_ts is not None and ts < last_ts:
                out_of_order += 1
            if uid != last_uid:
                if last_uid is not None:
                    closed.add(last_uid)
                if uid in closed and non_adjacent is None:
                    non_adjacent = uid
            last_uid = uid
            if ts is not None:
                last_ts = ts
        if max_rows is not None and n_rows >= max_rows:
            break
    return {"null_ids": null_ids, "null_ts": null_ts, "out_of_order": out_of_order,
            "bad_source": bad_source, "non_adjacent": non_adjacent}


@pytest.mark.parametrize("seed", range(12))
def test_vectorised_events_scan_matches_reference(good_data: Path, tmp_path: Path, seed: int) -> None:
    """Fuzz: shuffled runs, null ids/ts, unknown sources, non-adjacent users, tiny batches."""
    import numpy as np

    from pragmatiq.core.schema import EVENTS_SCHEMA

    rng = np.random.default_rng(seed)
    ev = pq.read_table(good_data / "events.parquet").to_pandas()
    ev = ev.head(int(rng.integers(50, 400)) if seed else 3).copy()
    # corrupt: reverse a user's run, blank some ids, null some ts, bad sources, duplicate a run later
    users = list(dict.fromkeys(ev["user_id"]))
    if seed % 3 == 0 and users:
        u = users[int(rng.integers(0, len(users)))]
        sel = ev["user_id"] == u
        ev.loc[sel, "ts"] = ev.loc[sel, "ts"].values[::-1]
    if seed % 4 == 1:
        ev.loc[ev.sample(frac=0.05, random_state=seed).index, "user_id"] = ""
    if seed % 2 == 1:
        ev.loc[ev.sample(frac=0.08, random_state=seed + 1).index, "ts"] = None
    if seed % 5 == 2:
        ev.loc[ev.sample(frac=0.03, random_state=seed + 2).index, "source"] = "bogus"
    if seed % 3 == 2 and len(users) > 2:
        ev = ev.iloc[list(range(len(ev))) + list(range(0, min(7, len(ev))))]
    tmp_path.mkdir(exist_ok=True)
    # The corrupted frame carries null ts on purpose; write it with a nullable copy of
    # the contract schema so the writer does not reject what the validator must catch.
    nullable = pa.schema([f.with_nullable(True) for f in EVENTS_SCHEMA])
    pq.write_table(pa.Table.from_pandas(ev, schema=nullable, preserve_index=False),
                   tmp_path / "events.parquet", row_group_size=int(rng.integers(3, 40)))
    import shutil

    shutil.copy(good_data / "profiles.parquet", tmp_path / "profiles.parquet")

    ref = _reference_scan(tmp_path / "events.parquet")
    report = validate_dataset(tmp_path)
    errors = "\n".join(report.errors)
    assert (f"{ref['null_ids']} rows with null/empty user_id" in errors) == bool(ref["null_ids"])
    assert (f"{ref['null_ts']} rows with null ts" in errors) == bool(ref["null_ts"])
    assert (f"{ref['out_of_order']} events out of time order" in errors) == bool(ref["out_of_order"])
    assert ("unknown source" in errors) == bool(ref["bad_source"])
    if ref["non_adjacent"] is not None:
        assert f"user {ref['non_adjacent']!r} appears in non-adjacent rows" in errors
    else:
        assert "non-adjacent" not in errors
