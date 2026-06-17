"""Two-phase generation: build the world, then simulate users in parallel.

Phase A (``World.build``) is deterministic from ``cfg.seed``. Phase B simulates
each user with ``user_rng(seed, user_idx)`` (a disjoint namespace) — the per-user
stream is independent of worker count and block size, so output files are
byte-identical for a given (seed, pyarrow version), regardless of ``n_workers``
(CI-enforced).

Events/profiles are streamed to parquet block by block (one row group per
block, blocks in user order); labels stream to ``labels/*.parquet``;
transfers come straight from the world schedule. A ``realism_report.html``
plus ``manifest.json`` round things off.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ... import __version__
from ...progress import progress
from ..schema import EVENTS_SCHEMA, LABEL_TASKS, PROFILES_SCHEMA, SOURCES, TRANSFERS_SCHEMA, label_schema
from .config import WorldConfig
from .labels import LabelOracle, LabelRows
from .simulator import UserSimulator, UserTrace
from .world import DAY_US, World, user_rng

BLOCK_SIZE = 256
HOUR_US = 3_600_000_000

_WORKER_WORLD: World | None = None
_WORKER_CFG: dict[str, Any] | None = None


@dataclass
class ReportAggregates:
    """Streaming statistics for the realism report (mergeable across blocks)."""

    events_per_user: list[int] = field(default_factory=list)
    hour_hist: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.int64))
    dow_hist: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.int64))
    delta_log10s_hist: np.ndarray = field(default_factory=lambda: np.zeros(45, dtype=np.int64))
    merchant_counts: Counter = field(default_factory=Counter)
    amounts_by_mcc: dict[str, list[float]] = field(default_factory=dict)
    source_counts: Counter = field(default_factory=Counter)
    label_pos: Counter = field(default_factory=Counter)
    label_n: Counter = field(default_factory=Counter)

    def merge(self, other: ReportAggregates) -> None:
        """Fold another block's aggregates into this one."""
        self.events_per_user.extend(other.events_per_user)
        self.hour_hist += other.hour_hist
        self.dow_hist += other.dow_hist
        self.delta_log10s_hist += other.delta_log10s_hist
        self.merchant_counts.update(other.merchant_counts)
        for k, v in other.amounts_by_mcc.items():
            cur = self.amounts_by_mcc.setdefault(k, [])
            if len(cur) < 4000:
                cur.extend(v[: 4000 - len(cur)])
        self.source_counts.update(other.source_counts)
        self.label_pos.update(other.label_pos)
        self.label_n.update(other.label_n)


def _init_worker(cfg_dict: dict[str, Any]) -> None:
    """Pool initializer (spawn-safe): rebuild the world deterministically."""
    global _WORKER_WORLD, _WORKER_CFG
    _WORKER_CFG = cfg_dict
    _WORKER_WORLD = World.build(WorldConfig.from_dict(cfg_dict))


def _trace_block_aggregates(traces: list[UserTrace], rows: LabelRows) -> ReportAggregates:
    agg = ReportAggregates()
    for tr in traces:
        agg.events_per_user.append(len(tr.ts))
        if len(tr.ts):
            agg.hour_hist += np.bincount(((tr.ts % DAY_US) // HOUR_US).astype(np.int64), minlength=24)[:24]
            day_idx = (tr.ts // DAY_US + 3) % 7  # epoch was a Thursday
            agg.dow_hist += np.bincount(day_idx.astype(np.int64), minlength=7)[:7]
            if len(tr.ts) > 1:
                d = np.diff(tr.ts) / 1e6
                d = d[d > 0]
                if len(d):
                    bins = np.clip((np.log10(d) + 1) * 5, 0, 44).astype(np.int64)
                    agg.delta_log10s_hist += np.bincount(bins, minlength=45)[:45]
            src, cnt = np.unique(tr.source, return_counts=True)
            for s, c in zip(src.tolist(), cnt.tolist()):
                agg.source_counts[SOURCES[s]] += c
            for _sig, cols in tr.groups:
                if "merchant" in cols:
                    agg.merchant_counts.update(cols["merchant"].tolist())
                    for mc, am in zip(cols["mcc"].tolist()[:200], cols["amount"].tolist()[:200]):
                        lst = agg.amounts_by_mcc.setdefault(mc, [])
                        if len(lst) < 400:
                            lst.append(float(am))
        agg.label_n["fraud_users"] += 1
        agg.label_pos["fraud_users"] += int(len(tr.fraud_rows) > 0)
    for task in ("default_12m", "churn_6m", "ltv_positive", "aml"):
        rws = getattr(rows, task)
        agg.label_n[task] += len(rws)
        agg.label_pos[task] += sum(r[2] for r in rws)
    return agg


def _drop_fields_mask(uid: np.ndarray, ts: np.ndarray, offsets: np.ndarray,
                      total_pairs: int, seed: int, rate: float) -> np.ndarray:
    """Deterministic keep-mask over flattened (event, field) pairs.

    The decision for each pair is a pure function of ``(seed, user, event ts,
    field position within the event)`` via a splitmix64 mix, so it is identical
    regardless of how users are split into blocks/workers — preserving the
    byte-identical and worker-count-invariant guarantees (global rule 2).
    """
    event_of_pair = np.searchsorted(offsets, np.arange(total_pairs), side="right") - 1
    key_pos = (np.arange(total_pairs) - offsets[event_of_pair]).astype(np.uint64)
    uid_int = np.array([int(u.rsplit("_", 1)[1]) for u in uid], dtype=np.uint64)
    M = np.uint64(0xFFFFFFFFFFFFFFFF)
    # Python int for the scalar seed term so the 64-bit wraparound is explicit
    # (numpy would warn on a uint64 scalar overflow); array ops below wrap silently.
    seed_term = np.uint64((int(seed) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    x = np.full(total_pairs, seed_term, dtype=np.uint64)
    x = (x ^ (uid_int[event_of_pair] * np.uint64(0xBF58476D1CE4E5B9))) & M
    x = (x ^ (ts[event_of_pair].astype(np.uint64) * np.uint64(0x94D049BB133111EB))) & M
    x = (x ^ (key_pos + np.uint64(0x9E3779B9))) & M
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & M
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & M
    x = (x ^ (x >> np.uint64(31))) & M
    u = (x >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
    return u >= rate  # keep when the draw clears the drop rate


def _events_table(traces: list[UserTrace], missing_rate: float = 0.0, seed: int = 0) -> pa.Table:
    """Assemble one block's events into an Arrow table.

    Uses the traces' grouped-dense layout: per group the key set is constant,
    so map offsets/keys/values are built with fancy indexing only — no
    per-event Python loop. ``missing_rate`` > 0 omits that fraction of field
    entries (deterministically) to simulate sparse real-world feeds.
    """
    # 1) Unify groups across users by signature; concatenate group columns.
    sig_to_gid: dict[tuple[str, ...], int] = {}
    sig_list: list[tuple[str, ...]] = []
    col_parts: list[dict[str, list[np.ndarray]]] = []
    row_base: dict[tuple[int, int], int] = {}  # (trace_i, local_gid) -> row offset
    for ti, tr in enumerate(traces):
        for lg, (sig, cols) in enumerate(tr.groups):
            if sig not in sig_to_gid:
                sig_to_gid[sig] = len(sig_list)
                sig_list.append(sig)
                col_parts.append({k: [] for k in sig})
            g = sig_to_gid[sig]
            base = sum(len(a) for a in next(iter(col_parts[g].values()), []))
            row_base[(ti, lg)] = base
            for k in sig:
                col_parts[g][k].append(cols[k])
    group_cols: list[dict[str, np.ndarray]] = [
        {k: (np.concatenate(v) if v else np.zeros(0, dtype=object)) for k, v in parts.items()}
        for parts in col_parts
    ]

    # 2) Per-event arrays in (user, time) order.
    ts_parts, src_parts, uid_parts, gid_parts, row_parts = [], [], [], [], []
    for ti, tr in enumerate(traces):
        n = len(tr.ts)
        if n == 0:
            continue
        ts_parts.append(tr.ts)
        src_parts.append(tr.source)
        uid_parts.append(np.full(n, tr.user_id, dtype=object))
        local_to_global = np.array(
            [sig_to_gid[sig] for sig, _ in tr.groups] or [0], dtype=np.int64
        )
        local_base = np.array(
            [row_base[(ti, lg)] for lg in range(len(tr.groups))] or [0], dtype=np.int64
        )
        gid_parts.append(local_to_global[tr.group_of_event])
        row_parts.append(tr.row_of_event.astype(np.int64) + local_base[tr.group_of_event])
    if not ts_parts:
        return EVENTS_SCHEMA.empty_table()
    ts = np.concatenate(ts_parts)
    src = np.concatenate(src_parts)
    uid = np.concatenate(uid_parts)
    gid = np.concatenate(gid_parts)
    row = np.concatenate(row_parts)
    n = len(ts)

    # 3) Map offsets from per-group key counts; scatter keys/values per group.
    key_count = np.array([len(s) for s in sig_list], dtype=np.int64)
    counts = key_count[gid]
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total_pairs = int(offsets[-1])
    keys_flat = np.empty(total_pairs, dtype=object)
    vals_flat = np.empty(total_pairs, dtype=object)
    for g, sig in enumerate(sig_list):
        ev_pos = np.nonzero(gid == g)[0]
        if len(ev_pos) == 0:
            continue
        base_off = offsets[ev_pos]
        rows_g = row[ev_pos]
        for j, k in enumerate(sig):
            keys_flat[base_off + j] = k
            vals_flat[base_off + j] = group_cols[g][k][rows_g]
    if missing_rate > 0 and total_pairs:
        keep = _drop_fields_mask(uid, ts, offsets, total_pairs, seed, missing_rate)
        if not keep.all():
            ev_of_pair = np.searchsorted(offsets, np.arange(total_pairs), side="right") - 1
            new_counts = np.bincount(ev_of_pair[keep], minlength=n).astype(np.int64)
            offsets = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(new_counts, out=offsets[1:])
            keys_flat, vals_flat = keys_flat[keep], vals_flat[keep]
    fields = pa.MapArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(keys_flat, type=pa.string()),
        pa.array(vals_flat, type=pa.string()),
    )
    source_str = np.array(SOURCES, dtype=object)[src]
    return pa.Table.from_arrays(
        [
            pa.array(uid, type=pa.string()),
            pa.array(ts, type=pa.int64()).cast(pa.timestamp("us")),
            pa.array(source_str, type=pa.string()),
            fields,
        ],
        schema=EVENTS_SCHEMA,
    )


def _profiles_table(traces: list[UserTrace], as_of_us: int) -> pa.Table:
    uids = [tr.user_id for tr in traces]
    attrs = [list(tr.attributes.items()) for tr in traces]
    lifelong = [[{"key": k, "ts": t} for k, t in tr.lifelong] for tr in traces]
    return pa.Table.from_arrays(
        [
            pa.array(uids, type=pa.string()),
            pa.array([as_of_us] * len(uids), type=pa.int64()).cast(pa.timestamp("us")),
            pa.array(attrs, type=pa.map_(pa.string(), pa.string())),
            pa.array(
                lifelong,
                type=pa.list_(pa.struct([pa.field("key", pa.string()), pa.field("ts", pa.timestamp("us"))])),
            ),
        ],
        schema=PROFILES_SCHEMA,
    )


def _simulate_block(args: tuple[int, int]) -> tuple[bytes, bytes, LabelRows, ReportAggregates]:
    """Worker: simulate users [lo, hi), return serialized tables + labels + stats."""
    global _WORKER_WORLD
    if _WORKER_WORLD is None:  # spawn context without initializer — rebuild
        assert _WORKER_CFG is not None, "worker not initialized"
        _WORKER_WORLD = World.build(WorldConfig.from_dict(_WORKER_CFG))
    world = _WORKER_WORLD
    lo, hi = args
    sim = UserSimulator(world)
    oracle = LabelOracle(world)
    rows = LabelRows()
    traces: list[UserTrace] = []
    for user_idx in range(lo, hi):
        rng = user_rng(world.cfg.seed, user_idx)
        trace = sim.run(user_idx, rng)
        rows.extend(oracle.label_user(trace, rng))
        traces.append(trace)
    agg = _trace_block_aggregates(traces, rows)
    as_of = world.calendar.start_us() + world.calendar.n_days * DAY_US
    evt = _events_table(traces, missing_rate=world.cfg.missing_field_rate, seed=world.cfg.seed)
    prof = _profiles_table(traces, as_of)
    # Serialize to IPC bytes: cheap, and keeps pickling overhead predictable.
    return _ipc(evt), _ipc(prof), rows, agg


def _ipc(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    return sink.getvalue().to_pybytes()


def _from_ipc(buf: bytes) -> pa.Table:
    return pa.ipc.open_stream(pa.BufferReader(buf)).read_all()


def _label_tables(rows: LabelRows) -> dict[str, pa.Table]:
    """Convert one block's label rows into Arrow tables (schema per task)."""
    out: dict[str, pa.Table] = {}

    def tbl(task: str, cols: list[pa.Array]) -> pa.Table:
        return pa.Table.from_arrays(cols, schema=label_schema(task))

    def ts_arr(vals: list[int]) -> pa.Array:
        return pa.array(vals, type=pa.int64()).cast(pa.timestamp("us"))

    def col(rws: list, i: int) -> list:
        return [x[i] for x in rws]

    for task in ("default_12m", "churn_6m", "aml"):
        rws = getattr(rows, task)
        out[task] = tbl(task, [pa.array(col(rws, 0), pa.string()), ts_arr(col(rws, 1)),
                               pa.array(col(rws, 2), pa.int8())])
    ltv = rows.ltv_positive
    out["ltv_positive"] = tbl("ltv_positive", [pa.array(col(ltv, 0), pa.string()), ts_arr(col(ltv, 1)),
                                               pa.array(col(ltv, 2), pa.int8()), pa.array(col(ltv, 3), pa.float64())])
    fr = rows.fraud
    out["fraud"] = tbl("fraud", [pa.array(col(fr, 0), pa.string()), ts_arr(col(fr, 1)),
                                 pa.array(col(fr, 2), pa.int8())])
    rec = rows.recurring
    out["recurring"] = tbl("recurring", [pa.array(col(rec, 0), pa.string()), ts_arr(col(rec, 1)),
                                         pa.array(col(rec, 2), pa.string()), pa.array(col(rec, 3), pa.int8())])
    cu = rows.comm_uplift
    out["comm_uplift"] = tbl("comm_uplift", [pa.array(col(cu, 0), pa.string()), pa.array(col(cu, 1), pa.string()),
                                             ts_arr(col(cu, 2)), pa.array(col(cu, 3), pa.int8()),
                                             pa.array(col(cu, 4), pa.int8()), pa.array(col(cu, 5), pa.int8())])
    return out


def _write_transfers(world: World, path: Path) -> int:
    tg = world.transfers
    uid = np.array([f"u_{i:08d}" for i in range(world.cfg.n_users)], dtype=object)
    table = pa.Table.from_arrays(
        [
            pa.array(uid[tg.from_idx], type=pa.string()),
            pa.array(uid[tg.to_idx], type=pa.string()),
            pa.array(tg.ts_us, type=pa.int64()).cast(pa.timestamp("us")),
            pa.array(tg.amount, type=pa.float64()),
        ],
        schema=TRANSFERS_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd", row_group_size=1 << 20)
    return len(table)


def generate(
    cfg: WorldConfig,
    out_dir: str | Path,
    n_workers: int = 0,
    write_report: bool = True,
) -> dict[str, Any]:
    """Generate a full synthetic dataset under ``out_dir``; return the manifest.

    ``n_workers <= 1`` runs inline (no pool). Output bytes are independent of
    ``n_workers``. Files written: events.parquet, profiles.parquet,
    transfers.parquet, labels/*.parquet, manifest.json, realism_report.html.
    """
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)

    global _WORKER_WORLD, _WORKER_CFG
    world = World.build(cfg)
    _WORKER_WORLD = world
    _WORKER_CFG = cfg.to_dict()

    n = cfg.n_users
    blocks = [(lo, min(lo + BLOCK_SIZE, n)) for lo in range(0, n, BLOCK_SIZE)]

    ev_writer = pq.ParquetWriter(out / "events.parquet", EVENTS_SCHEMA, compression="zstd")
    pr_writer = pq.ParquetWriter(out / "profiles.parquet", PROFILES_SCHEMA, compression="zstd")
    lb_writers = {t: pq.ParquetWriter(out / "labels" / f"{t}.parquet", label_schema(t), compression="zstd") for t in LABEL_TASKS}

    agg = ReportAggregates()
    n_events = 0
    pool = None
    try:
        results: Iterable[tuple[bytes, bytes, LabelRows, ReportAggregates]]
        if n_workers and n_workers > 1:
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
            pool = ctx.Pool(n_workers, initializer=_init_worker, initargs=(cfg.to_dict(),))
            results = pool.imap(_simulate_block, blocks, chunksize=1)
        else:
            results = map(_simulate_block, blocks)
        results = progress(results, total=len(blocks),
                           desc=f"synth {n:,} users", unit="block")
        for evt_buf, prof_buf, rows, block_agg in results:
            evt = _from_ipc(evt_buf)
            ev_writer.write_table(evt, row_group_size=max(len(evt), 1))
            prof = _from_ipc(prof_buf)
            pr_writer.write_table(prof, row_group_size=max(len(prof), 1))
            for task, table in _label_tables(rows).items():
                if len(table):
                    lb_writers[task].write_table(table, row_group_size=max(len(table), 1))
            agg.merge(block_agg)
            n_events += len(evt)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        ev_writer.close()
        pr_writer.close()
        for wr in lb_writers.values():
            wr.close()

    n_transfers = _write_transfers(world, out / "transfers.parquet")
    elapsed = time.time() - t0

    # manifest.json is a pure function of (config, seed, pyarrow) so it is
    # byte-identical across runs (global rule 2). Wall-clock timing — which is
    # inherently non-deterministic — is written separately to timing.json and
    # NOT hashed by the determinism gate.
    manifest: dict[str, Any] = {
        "pragmatiq_version": __version__,
        "pyarrow_version": pa.__version__,
        "config": cfg.to_dict(),
        "n_users": n,
        "n_events": int(n_events),
        "n_transfers": int(n_transfers),
        "label_prevalence": {
            k: round(agg.label_pos[k] / max(agg.label_n[k], 1), 5) for k in sorted(agg.label_n)
        },
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    timing = {"elapsed_sec": round(elapsed, 2), "users_per_sec": round(n / elapsed, 2),
              "n_workers": n_workers}
    with open(out / "timing.json", "w") as f:
        json.dump(timing, f, indent=2, sort_keys=True)

    if write_report:
        from .report import write_realism_report

        write_realism_report(agg, {**manifest, **timing}, out / "realism_report.html")
    return {**manifest, **timing}
