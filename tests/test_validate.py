"""Phase 8 tests: dataset validation with actionable errors."""

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
    from pragmatiq.data.schema import EVENTS_SCHEMA

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

    from pragmatiq.data.schema import EVENTS_SCHEMA

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

    from pragmatiq.data.schema import EVENTS_SCHEMA

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

    from pragmatiq.data.schema import TRANSFERS_SCHEMA

    shutil.copy(good_data / "events.parquet", tmp_path / "events.parquet")
    shutil.copy(good_data / "profiles.parquet", tmp_path / "profiles.parquet")
    tr = pq.read_table(good_data / "transfers.parquet").to_pandas()
    tr.loc[tr.index[:3], "to_user"] = tr.loc[tr.index[:3], "from_user"].to_numpy()  # self-loops
    pq.write_table(pa.Table.from_pandas(tr, schema=TRANSFERS_SCHEMA, preserve_index=False),
                   tmp_path / "transfers.parquet")
    report = validate_dataset(tmp_path)
    assert any("self-loop" in w for w in report.warnings), report.summary()
