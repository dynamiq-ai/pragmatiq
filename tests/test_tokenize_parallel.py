"""Tests for pragmatiq.data.parallel_tokenize — row-group-sharded parallel tokenize.

Rule 2: for ANY ``n_workers`` and ANY row-group layout the shard files, the
user index, and the manifest must be byte/content-identical to the inline
path. The matrix below fabricates BYO parquets whose users straddle row-group
boundaries (the stitching cases) on top of generator-shaped data.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import shutil
import threading
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pragmatiq.data.parallel_tokenize as pt
from pragmatiq import api
from pragmatiq.core.schema import EVENTS_SCHEMA, PROFILES_SCHEMA

# ---------------------------------------------------------------------------
# helpers


def _digests(out_dir: Path) -> dict[str, Any]:
    """Shard sha256s + manifest contents + every user-index entry/profile blob."""
    from pragmatiq.data.sharding import UserIndex

    d: dict[str, Any] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((out_dir / "shards").glob("*.parquet"))
    }
    d["__manifest__"] = json.loads((out_dir / "shard_manifest.json").read_text())
    idx = UserIndex(out_dir)
    d["__index__"] = [
        (uid, dataclasses.asdict(idx.meta(uid)), idx.profile(uid)) for uid in idx.order
    ]
    idx.close()
    return d


def _write_byo(
    tmp: Path,
    runs: list[tuple[str, int]],
    rows_per_rg: int | None = None,
    rg_sizes: list[int] | None = None,
) -> Path:
    """Fabricate a BYO dataset: one ``write_table`` call per slice = one row group.

    ``rows_per_rg`` not dividing run lengths forces users to straddle row-group
    boundaries; ``rg_sizes`` gives explicit per-row-group sizes (0 allowed).
    """
    ds = tmp
    ds.mkdir(parents=True, exist_ok=True)
    uids: list[str] = []
    tss: list[dt.datetime] = []
    srcs: list[str] = []
    fields: list[list[tuple[str, str]]] = []
    per_uid: dict[str, int] = {}  # ts stays increasing within each uid across runs
    g = 0
    base = dt.datetime(2024, 1, 1)
    for uid, n in runs:
        start = per_uid.get(uid, 0)
        for j in range(n):
            uids.append(uid)
            tss.append(base + dt.timedelta(minutes=start + j))
            srcs.append("transaction")
            fields.append([("amount", str(100 + g))])
            g += 1
        per_uid[uid] = start + n
    table = pa.table(
        {"user_id": uids, "ts": tss, "source": srcs, "fields": fields}, schema=EVENTS_SCHEMA
    )
    with pq.ParquetWriter(ds / "events.parquet", EVENTS_SCHEMA) as w:
        if rg_sizes is not None:
            assert sum(rg_sizes) == table.num_rows
            lo = 0
            for sz in rg_sizes:
                w.write_table(table.slice(lo, sz))
                lo += sz
        else:
            assert rows_per_rg is not None
            for lo in range(0, table.num_rows, rows_per_rg):
                w.write_table(table.slice(lo, rows_per_rg))
    prof_uids = list(dict.fromkeys(u for u, _ in runs))
    prof = pa.table(
        {
            "user_id": prof_uids,
            "as_of": [dt.datetime(2025, 1, 1)] * len(prof_uids),
            "attributes": [[("plan", "basic")]] * len(prof_uids),
            "lifelong": [[{"key": "account_opened", "ts": dt.datetime(2023, 1, 1)}]]
            * len(prof_uids),
        },
        schema=PROFILES_SCHEMA,
    )
    pq.write_table(prof, ds / "profiles.parquet")
    return ds


def _main_runs() -> list[tuple[str, int]]:
    """40 users x 5-13 events + one 40-event whale (spans >=5 of the 7-row RGs)."""
    runs = [(f"u{i:03d}", 5 + (i * 7) % 9) for i in range(40)]
    runs.insert(20, ("whale", 40))
    return runs


@pytest.fixture(scope="module")
def byo_main(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_byo(tmp_path_factory.mktemp("byo_main"), _main_runs(), rows_per_rg=7)


@pytest.fixture(scope="module")
def tokdir(tmp_path_factory: pytest.TempPathFactory, byo_main: Path) -> Path:
    """Fit the tokenizer ONCE so every parity check is over encoding, not fitting."""
    from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig

    out = tmp_path_factory.mktemp("tok") / "tokenizer"
    PragmaTokenizer(TokenizerConfig()).fit(byo_main).save(out)
    return out


@pytest.fixture
def small_tasks(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Shrink the task-size constants so tiny fixtures yield multiple tasks."""
    monkeypatch.setattr(pt, "_MIN_TASK_ROWS", 1)
    monkeypatch.setattr(pt, "_MAX_TASK_ROWS", 4)
    return pt


def _tokenize(ds: Path, out: Path, tokdir: Path, n_workers: int, **kw: Any) -> dict[str, Any]:
    return api.tokenize(ds, out, tokenizer_dir=tokdir, n_workers=n_workers, **kw)


# ---------------------------------------------------------------------------
# the matrix


class TestParallelTokenizeParity:
    def test_generator_data_true_parallel(self, tmp_path: Path, monkeypatch) -> None:
        """#1: 600 synth users => 3 row groups; digests equal for any n_workers."""
        ds = tmp_path / "ds"
        api.synthesize({"n_users": 600, "seed": 4}, out=ds, write_report=False)
        md = pq.ParquetFile(ds / "events.parquet").metadata
        assert md.num_row_groups == 3  # one RG per 256-user block
        monkeypatch.setattr(pt, "_MIN_TASK_ROWS", 1)
        monkeypatch.setattr(pt, "_MAX_TASK_ROWS", 1)  # one task per RG => 3 tasks
        from pragmatiq.data.tokenizer import PragmaTokenizer, TokenizerConfig

        tok = tmp_path / "tokenizer"
        PragmaTokenizer(TokenizerConfig()).fit(ds).save(tok)
        digests = {}
        for w in (0, 2, 3, 5):
            _tokenize(ds, tmp_path / f"out{w}", tok, n_workers=w)
            digests[w] = _digests(tmp_path / f"out{w}")
        assert digests[0] == digests[2] == digests[3] == digests[5]

    def test_users_straddling_row_groups(self, byo_main, tokdir, tmp_path, small_tasks) -> None:
        """#2: 7-row RGs cut users mid-run; the whale chains n_runs==1 tasks."""
        tasks = pt._plan_tasks(byo_main / "events.parquet", 3)
        assert tasks is not None and len(tasks) > 5  # true parallel, many boundaries
        m0 = _tokenize(byo_main, tmp_path / "o0", tokdir, n_workers=0)
        m3 = _tokenize(byo_main, tmp_path / "o3", tokdir, n_workers=3)
        assert m0 == m3
        assert _digests(tmp_path / "o0") == _digests(tmp_path / "o3")

    def test_single_row_group_falls_back_without_pool(
        self, tmp_path, tokdir, small_tasks, monkeypatch
    ) -> None:
        """#3: one RG => one task => inline fallback; the pool is never built."""
        ds = _write_byo(tmp_path / "byo", _main_runs(), rows_per_rg=10**6)
        assert pq.ParquetFile(ds / "events.parquet").metadata.num_row_groups == 1

        def boom(*a: Any, **kw: Any) -> None:
            raise AssertionError("ProcessPoolExecutor must not be constructed on fallback")

        monkeypatch.setattr(pt, "ProcessPoolExecutor", boom)
        m4 = _tokenize(ds, tmp_path / "o4", tokdir, n_workers=4)
        monkeypatch.undo()
        m0 = _tokenize(ds, tmp_path / "o0", tokdir, n_workers=0)
        assert m0 == m4
        assert _digests(tmp_path / "o0") == _digests(tmp_path / "o4")

    def test_more_workers_than_tasks(self, tmp_path, tokdir, small_tasks) -> None:
        """#4: 2 users / 2 RGs / 8 workers."""
        ds = _write_byo(tmp_path / "byo", [("a", 3), ("b", 3)], rows_per_rg=3)
        m0 = _tokenize(ds, tmp_path / "o0", tokdir, n_workers=0)
        m8 = _tokenize(ds, tmp_path / "o8", tokdir, n_workers=8)
        assert m0 == m8
        assert _digests(tmp_path / "o0") == _digests(tmp_path / "o8")

    def test_max_users_mid_stream(self, byo_main, tokdir, tmp_path, small_tasks) -> None:
        """#5: truncation inside interior, exactly on a stitched boundary user, 7, > total."""
        runs = _main_runs()
        # first user (1-based ordinal) whose rows straddle a 7-row RG boundary
        boundary_user = None
        acc = 0
        for i, (_, n) in enumerate(runs):
            if acc // 7 != (acc + n - 1) // 7:
                boundary_user = i + 1
                break
            acc += n
        assert boundary_user is not None
        for mu in sorted({4, boundary_user, 7, len(runs) + 9}):
            m0 = _tokenize(byo_main, tmp_path / f"i{mu}", tokdir, n_workers=0, max_users=mu)
            m2 = _tokenize(byo_main, tmp_path / f"p{mu}", tokdir, n_workers=2, max_users=mu)
            assert m0 == m2, f"manifest mismatch at max_users={mu}"
            assert _digests(tmp_path / f"i{mu}") == _digests(tmp_path / f"p{mu}"), (
                f"digest mismatch at max_users={mu}"
            )

    def test_empty_dataset(self, tmp_path, tokdir, small_tasks) -> None:
        """#6: 0-row events + empty profiles — parity, no crash, empty manifest."""
        ds = _write_byo(tmp_path / "byo", [], rg_sizes=[])
        m0 = _tokenize(ds, tmp_path / "o0", tokdir, n_workers=0)
        m4 = _tokenize(ds, tmp_path / "o4", tokdir, n_workers=4)
        assert m0 == m4
        assert m0["n_users"] == 0
        assert _digests(tmp_path / "o0") == _digests(tmp_path / "o4")

    def test_non_contiguous_uid_refused(self, tmp_path, tokdir, small_tasks) -> None:
        """#7: u1,u2,u1 is non-adjacent — tokenize refuses on BOTH the inline and the
        multi-task parallel path (the parent stitch detects the re-emitted uid), and
        validate reports it as an error, not a silent warning."""
        ds = _write_byo(tmp_path / "byo", [("u1", 4), ("u2", 3), ("u1", 2)], rows_per_rg=5)
        for nw, out in ((0, "o0"), (2, "o2")):
            with pytest.raises(ValueError, match="non-adjacent"):
                _tokenize(ds, tmp_path / out, tokdir, n_workers=nw)

        from pragmatiq.validate import validate_dataset

        report = validate_dataset(ds)
        assert any("non-adjacent" in e for e in report.errors)
        assert not report.ok

    def test_huge_row_group_falls_back_loudly(
        self, byo_main, tokdir, tmp_path, small_tasks, monkeypatch, caplog
    ) -> None:
        """#8: RG over the raw-payload cap => warning + inline parity."""
        monkeypatch.setattr(pt, "_MAX_SINGLE_RG_ROWS", 5)  # byo_main RGs have 7 rows
        with caplog.at_level(logging.WARNING, logger="pragmatiq.data.parallel_tokenize"):
            m4 = _tokenize(byo_main, tmp_path / "o4", tokdir, n_workers=4)
        assert any("row group" in r.message for r in caplog.records)
        monkeypatch.undo()
        m0 = _tokenize(byo_main, tmp_path / "o0", tokdir, n_workers=0)
        assert m0 == m4
        assert _digests(tmp_path / "o0") == _digests(tmp_path / "o4")

    def test_planning_failure_falls_back(
        self, byo_main, tokdir, tmp_path, monkeypatch, caplog
    ) -> None:
        """#9: a planner exception must degrade to a single process, not fail."""

        def boom(*a: Any, **kw: Any) -> None:
            raise RuntimeError("footer exploded")

        monkeypatch.setattr(pt, "_plan_tasks", boom)
        with caplog.at_level(logging.WARNING, logger="pragmatiq.data.parallel_tokenize"):
            m4 = _tokenize(byo_main, tmp_path / "o4", tokdir, n_workers=4)
        assert any("planning failed" in r.message for r in caplog.records)
        monkeypatch.undo()
        m0 = _tokenize(byo_main, tmp_path / "o0", tokdir, n_workers=0)
        assert m0 == m4
        assert _digests(tmp_path / "o0") == _digests(tmp_path / "o4")

    def test_determinism_same_worker_count(self, byo_main, tokdir, tmp_path, small_tasks) -> None:
        """#10: two n_workers=4 runs => identical digests."""
        _tokenize(byo_main, tmp_path / "a", tokdir, n_workers=4)
        _tokenize(byo_main, tmp_path / "b", tokdir, n_workers=4)
        assert _digests(tmp_path / "a") == _digests(tmp_path / "b")

    def test_worker_init_failure_raises_no_hang(
        self, byo_main, tokdir, tmp_path, small_tasks
    ) -> None:
        """#11: corrupted tokenizer hash in the child => BrokenProcessPool, never a hang."""
        from pragmatiq.data.tokenizer import PragmaTokenizer

        tok = PragmaTokenizer.load(tokdir)  # parent load from the GOOD dir
        bad = tmp_path / "bad_tok"
        shutil.copytree(tokdir, bad)
        (bad / "tokenizer.hash").write_text("0" * 64)  # child init must fail
        result: dict[str, Any] = {}

        def consume() -> None:
            try:
                list(pt.parallel_tokenize(byo_main, tok, bad, n_workers=2))
                result["exc"] = None
            except BaseException as e:  # noqa: B036 - we assert on the type below
                result["exc"] = e

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        t.join(timeout=60)
        assert not t.is_alive(), "parallel_tokenize hung on worker initializer failure"
        assert isinstance(result["exc"], BrokenProcessPool)

    def test_empty_interior_row_group_task(self, tmp_path, tokdir, monkeypatch) -> None:
        """#12: a 0-row RG between two RGs of the SAME uid; stitch across the empty task."""
        ds = _write_byo(tmp_path / "byo", [("x", 6), ("y", 2)], rg_sizes=[3, 0, 3, 2])
        monkeypatch.setattr(pt, "_plan_tasks", lambda *a, **kw: [(0, 1), (1, 2), (2, 3), (3, 4)])
        m2 = _tokenize(ds, tmp_path / "o2", tokdir, n_workers=2)
        monkeypatch.undo()
        m0 = _tokenize(ds, tmp_path / "o0", tokdir, n_workers=0)
        assert m0 == m2
        assert m0["n_users"] == 2  # x's two non-empty RGs stitch into ONE record
        assert _digests(tmp_path / "o0") == _digests(tmp_path / "o2")


def _shard_hashes(d):
    import hashlib

    return sorted(hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (d / "shards").glob("*.parquet"))


class TestSpawnUnsafeFallback:
    """From stdin / `python -c` / a notebook (unspawnable __main__) parallel_tokenize
    must still produce byte-identical output: via FORK where available (the common
    Linux/macOS case — gate heredocs included), else inline (never crash)."""

    def test_unspawnable_main_uses_fork_byte_identical(self, tmp_path, monkeypatch):
        import multiprocessing as _mp

        import pytest

        import pragmatiq.data.parallel_tokenize as P
        from pragmatiq import api

        if "fork" not in _mp.get_all_start_methods():
            pytest.skip("fork not available on this platform")
        ds = tmp_path / "ds"
        api.synthesize({"n_users": 600, "seed": 4}, out=ds, write_report=False)
        # spawn can't re-bootstrap __main__, but fork is available → still parallel
        monkeypatch.setattr(P, "_spawn_safe", lambda: False)
        m_par = api.tokenize(ds, tmp_path / "tokpar", n_workers=4)
        m_in = api.tokenize(ds, tmp_path / "tokin", n_workers=0)
        assert m_par == m_in
        assert _shard_hashes(tmp_path / "tokpar") == _shard_hashes(tmp_path / "tokin")

    def test_no_fork_and_unspawnable_falls_back_inline(self, tmp_path, monkeypatch):
        import pragmatiq.data.parallel_tokenize as P
        from pragmatiq import api

        ds = tmp_path / "ds"
        api.synthesize({"n_users": 600, "seed": 4}, out=ds, write_report=False)
        # spawn-only platform AND unspawnable __main__ → must run inline, not crash
        monkeypatch.setattr(P, "_spawn_safe", lambda: False)
        monkeypatch.setattr(P.mp, "get_all_start_methods", lambda: ["spawn"])

        def _boom(*a, **k):
            raise AssertionError("no pool should be built when unspawnable AND no fork")

        monkeypatch.setattr(P, "ProcessPoolExecutor", _boom)
        m_par = api.tokenize(ds, tmp_path / "tokpar", n_workers=4)
        m_in = api.tokenize(ds, tmp_path / "tokin", n_workers=0)
        assert m_par == m_in
        assert _shard_hashes(tmp_path / "tokpar") == _shard_hashes(tmp_path / "tokin")

    def test_threaded_unspawnable_host_runs_inline(self, tmp_path, monkeypatch):
        # A notebook kernel is unspawnable AND multi-threaded; forking it can
        # deadlock the child, so both fit and encode must run inline there.
        import multiprocessing as _mp
        import threading

        import pytest

        import pragmatiq.data.parallel_tokenize as P
        from pragmatiq import api

        if "fork" not in _mp.get_all_start_methods():
            pytest.skip("fork not available on this platform")
        monkeypatch.setattr(P, "_spawn_safe", lambda: False)  # notebook __main__

        def _boom(*a, **k):
            raise AssertionError("must not build a process pool in a threaded, unspawnable host")

        monkeypatch.setattr(P, "ProcessPoolExecutor", _boom)
        stop = threading.Event()
        worker = threading.Thread(target=stop.wait, daemon=True)
        worker.start()
        try:
            assert P._start_method() is None  # multi-threaded + fork-only-unspawnable -> inline
            ds = tmp_path / "ds"
            api.synthesize({"n_users": 400, "seed": 4}, out=ds, write_report=False)
            m_par = api.tokenize(ds, tmp_path / "tokpar", n_workers=4)  # must not fork
            m_in = api.tokenize(ds, tmp_path / "tokin", n_workers=0)
            assert m_par == m_in
            assert _shard_hashes(tmp_path / "tokpar") == _shard_hashes(tmp_path / "tokin")
        finally:
            stop.set()
            worker.join()
