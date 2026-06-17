"""Row-group-sharded parallel tokenizer fit and encode (work done in workers).

Both halves of tokenization parallelize over the same unit of work — contiguous
row-group ranges (tasks) planned from the parquet footer only — and both are
byte-identical to the single-process path for any worker count and any row-group
layout (global rule 2):

- :func:`parallel_fit` folds each task's events into a partition-independent
  :class:`pragmatiq.data.tokenizer._FitAccum`; the parent merges the per-task
  accumulators in strict task order, folds ``profiles.parquet`` once, and runs
  the classification / binning / BPE tail — the same fold the single-pass fit
  performs.
- :func:`parallel_tokenize` has each worker decode its range with the exact
  inline grouping code, encoding every run *closed on both sides within the
  task*. Only a task's first and last runs cross the wire raw (they may continue
  into a neighbour); the parent stitches those boundary runs in strict task
  order ("merge carry + head iff uids are equal" — the same adjacent-row rule
  the inline path applies at every row).

Both fall back to a single process (logged) when the file has fewer than two
usable tasks, an oversized row group, or no spawnable worker bootstrap.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import logging
import multiprocessing as mp
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pragmatiq.data.tokenizer import PragmaTokenizer, _FitAccum

logger = logging.getLogger(__name__)

_MIN_TASK_ROWS = 65_536
_MAX_TASK_ROWS = 1_048_576
_MAX_SINGLE_RG_ROWS = 4_000_000
_BATCH_SIZE = 65_536


def _plan_tasks(events_path: Path, n_workers: int) -> list[tuple[int, int]] | None:
    """Pack row-group indices into contiguous ``[lo, hi)`` tasks of ~target rows.

    ``None`` => caller falls back to inline with a warning (oversized row group).
    Footer-only: no data pages are read.
    """
    md = pq.ParquetFile(events_path).metadata
    rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    if max(rows, default=0) > _MAX_SINGLE_RG_ROWS:
        return None  # raw boundary payload could blow memory
    total = sum(rows)
    target = max(_MIN_TASK_ROWS, min(_MAX_TASK_ROWS, total // (n_workers * 4) or 1))
    tasks: list[tuple[int, int]] = []
    lo = acc = 0
    for i, r in enumerate(rows):
        acc += r
        if acc >= target:
            tasks.append((lo, i + 1))
            lo, acc = i + 1, 0
    if lo < len(rows):
        tasks.append((lo, len(rows)))
    return tasks


def _cap_arrow_threads() -> None:
    """K workers must not each spin Arrow's default 10 CPU + 8 IO threads."""
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


# --------------------------------------------------------------------- fit
# Fit worker state: the events ParquetFile, opened in the child (never pickled).
_FIT_PF: pq.ParquetFile | None = None


def _init_fit_worker(data_dir: str) -> None:
    """Fit-worker initializer: cap Arrow threads, open events.parquet."""
    _cap_arrow_threads()
    global _FIT_PF
    _FIT_PF = pq.ParquetFile(Path(data_dir) / "events.parquet")


def _run_fit_task(task: tuple[int, int]) -> _FitAccum:
    """Fold one contiguous row-group range's events into a ``_FitAccum``."""
    from pragmatiq.data.tokenizer import _FitAccum

    assert _FIT_PF is not None
    acc = _FitAccum()
    batches = _FIT_PF.iter_batches(
        columns=["source", "fields"],
        batch_size=_BATCH_SIZE,
        row_groups=list(range(task[0], task[1])),
        use_threads=False,
    )
    for batch in batches:
        acc.consume_events_batch(batch)
    return acc


def parallel_fit(
    data_dir: str | Path, tok: PragmaTokenizer, n_workers: int
) -> PragmaTokenizer:
    """Fit ``tok`` over ``data_dir`` using ``n_workers`` processes (rule 2).

    Workers fold their event row-group ranges into :class:`_FitAccum`; the parent
    merges them in strict task (file) order, folds ``profiles.parquet`` once, and
    runs the classification / binning / BPE tail. The result is byte-identical to
    the single-process fit for any worker count. Falls back to the single-process
    fit (logged) when there are fewer than two tasks, an oversized row group, or
    no spawnable worker bootstrap — exactly the conditions :func:`parallel_tokenize`
    falls back on.
    """
    from pragmatiq.data.tokenizer import _FitAccum

    data_dir = Path(data_dir)
    method = _start_method()
    if method is None:
        logger.warning(
            "parallel fit needs fork in a single-threaded process or a spawnable __main__ "
            "(a script, `python -m`, or the pragmatiq CLI); this process has neither "
            "(e.g. a multi-threaded notebook), so fitting in a single process."
        )
        return _inline_fit(data_dir, tok)
    try:
        tasks = _plan_tasks(data_dir / "events.parquet", n_workers)
    except Exception as exc:  # planning must never be fatal
        logger.warning("parallel fit planning failed (%s); using a single process", exc)
        tasks = []
    if tasks is None:
        logger.warning(
            "events.parquet has a row group >%d rows; fitting in a single process",
            _MAX_SINGLE_RG_ROWS,
        )
    if not tasks or len(tasks) < 2:
        return _inline_fit(data_dir, tok)

    n_workers = min(n_workers, len(tasks))
    ex = ProcessPoolExecutor(
        n_workers,
        mp_context=mp.get_context(method),
        initializer=_init_fit_worker,
        initargs=(str(data_dir),),
    )
    merged = _FitAccum()
    try:
        for acc in _ordered_bounded(ex, _run_fit_task, tasks, 2 * n_workers):
            merged.merge(acc)  # strict task order => stable counter / sample order
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    # Profiles fold + finalize happen once, in the parent, after the event merge —
    # the same events-then-profiles order the single-pass fit uses.
    tok._consume_profiles(merged, data_dir)
    return tok._finalize(merged)


def _inline_fit(data_dir: Path, tok: PragmaTokenizer) -> PragmaTokenizer:
    """Single-process fit (the fallback path)."""
    return tok.fit(data_dir, n_workers=0)


# ----------------------------------------------------------------- encode
# Worker-process state: (tokenizer, events ParquetFile, profiles dict).
_W: tuple[Any, pq.ParquetFile, dict[str, Any]] | None = None


def _init_worker(tok_dir: str, data_dir: str) -> None:
    """Worker initializer: load the tokenizer + profiles, open events.parquet."""
    _cap_arrow_threads()
    from pragmatiq.data.tokenizer import PragmaTokenizer, _load_profiles

    global _W
    d = Path(data_dir)
    _W = (
        PragmaTokenizer.load(tok_dir),  # content-hash verified => identical encodes
        pq.ParquetFile(d / "events.parquet"),  # opened in the child, never pickled
        _load_profiles(d),
    )


@dataclasses.dataclass
class _TaskResult:
    """One task's output: encoded interior runs + raw boundary runs."""

    n_runs: int
    head: tuple[str, list] | None = None  # (uid, RAW events) — may continue prev task
    interior: list = dataclasses.field(default_factory=list)  # [(TokenizedRecord, profile_payload)]
    tail: tuple[str, list] | None = None  # may continue into next task


def _encode_payload(rec: Any) -> tuple[Any, dict[str, Any]]:
    assert _W is not None
    return (
        _W[0].encode(rec),
        {"attributes": rec.attributes, "lifelong": rec.lifelong, "as_of": rec.as_of},
    )


def _run_task(task: tuple[int, int]) -> _TaskResult:
    """Decode + group + encode one contiguous row-group range in a worker."""
    from pragmatiq.data.tokenizer import _iter_records_from_batches

    assert _W is not None
    _tok, pf, profiles = _W
    batches = pf.iter_batches(
        batch_size=_BATCH_SIZE, row_groups=list(range(task[0], task[1])), use_threads=False
    )
    it = _iter_records_from_batches(batches, profiles, None)  # VERBATIM inline grouping code
    first = next(it, None)
    if first is None:
        return _TaskResult(n_runs=0)  # all-empty row groups (BYO)
    res = _TaskResult(n_runs=1, head=(first.user_id, first.events))
    pending = None  # one-behind: the last run must stay raw
    for rec in it:
        res.n_runs += 1
        if pending is not None:
            res.interior.append(_encode_payload(pending))  # closed on both sides within the task
        pending = rec
    if pending is not None:
        res.tail = (pending.user_id, pending.events)
    return res


_R = TypeVar("_R")


def _ordered_bounded(
    ex: ProcessPoolExecutor,
    fn: Callable[[tuple[int, int]], _R],
    tasks: Iterable[tuple[int, int]],
    k: int,
) -> Iterator[_R]:
    """Yield ``fn(task)`` results in strict submission order with ≤ ``k`` in flight."""
    pending: collections.deque[Future[_R]] = collections.deque()
    it = iter(tasks)
    for t in itertools.islice(it, k):
        pending.append(ex.submit(fn, t))
    while pending:
        fut = pending.popleft()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(ex.submit(fn, nxt))  # refill BEFORE blocking
        yield fut.result()  # strict submission order; BrokenProcessPool on worker death


def _inline_encoded(
    data_dir: Path, tok: PragmaTokenizer, max_users: int | None
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Single-process (encoded, profile_payload) stream — the fallback path."""
    from pragmatiq.data.tokenizer import iter_user_records

    for rec in iter_user_records(data_dir, max_users=max_users):
        yield (
            tok.encode(rec),
            {"attributes": rec.attributes, "lifelong": rec.lifelong, "as_of": rec.as_of},
        )


def _spawn_safe() -> bool:
    """True when ``spawn`` workers can re-bootstrap this process.

    The ``spawn`` start method re-imports the parent's ``__main__`` in each
    worker. That works for a real script, ``python -m pkg``, and console-script
    entrypoints (so the ``pragmatiq`` CLI is fine), but NOT when ``__main__``
    came from stdin (``python - <<EOF``), ``python -c``, or an interactive /
    notebook kernel — there the worker bootstrap dies with
    ``FileNotFoundError: <stdin>`` and the pool breaks. In those contexts we
    must fall back to a single process rather than crash.
    """
    import __main__

    if getattr(__main__, "__spec__", None) is not None:
        return True  # launched via -m: spec is importable
    main_file = getattr(__main__, "__file__", None)
    return isinstance(main_file, str) and Path(main_file).exists()


def _start_method() -> str | None:
    """Pick a safe multiprocessing start method, or ``None`` to run inline.

    ``fork`` inherits the parent, so it works from stdin / ``-c`` / gate heredocs
    where ``spawn`` cannot re-bootstrap ``__main__`` — but it is only safe in a
    single-threaded process: forking a multi-threaded host (e.g. a Jupyter
    kernel) can deadlock the child on locks held by threads that no longer exist
    after the fork. So ``fork`` is chosen only when this process has a single
    thread; otherwise ``spawn`` (when ``__main__`` is re-importable), and failing
    that, inline. All three produce byte-identical output.
    """
    if "fork" in mp.get_all_start_methods() and threading.active_count() == 1:
        return "fork"
    if _spawn_safe():
        return "spawn"
    return None


def parallel_tokenize(
    data_dir: str | Path,
    tok: PragmaTokenizer,
    tok_dir: str | Path,
    n_workers: int,
    max_users: int | None = None,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """(encoded, profile_payload) pairs in exactly ``iter_user_records`` order, for any file."""
    data_dir = Path(data_dir)
    # Single-threaded fork (inherits the parent — works from stdin/`-c`/gate
    # heredocs where spawn can't re-bootstrap __main__), else spawn when __main__
    # is re-importable, else inline. Fork is gated on a single thread because
    # forking a multi-threaded host (a notebook kernel) can deadlock the child.
    # Output is byte-identical across all three paths.
    method = _start_method()
    if method is None:
        logger.warning(
            "parallel tokenize needs fork in a single-threaded process or a spawnable __main__ "
            "(a script, `python -m`, or the pragmatiq CLI). This process has neither "
            "(e.g. a multi-threaded notebook), so tokenizing in a single process."
        )
        yield from _inline_encoded(data_dir, tok, max_users)
        return
    try:
        tasks = _plan_tasks(data_dir / "events.parquet", n_workers)
    except Exception as exc:  # planning must never be fatal
        logger.warning("parallel tokenize planning failed (%s); using a single process", exc)
        tasks = []
    if tasks is None:
        logger.warning(
            "events.parquet has a row group >%d rows; tokenizing in a single process",
            _MAX_SINGLE_RG_ROWS,
        )
    if not tasks or len(tasks) < 2:
        yield from _inline_encoded(data_dir, tok, max_users)
        return

    from pragmatiq.data.tokenizer import _load_profiles, _mk_record

    profiles = _load_profiles(data_dir)  # for boundary-run encodes
    n_workers = min(n_workers, len(tasks))

    def emit(uid: str, events: list) -> tuple[Any, dict[str, Any]]:
        rec = _mk_record(uid, events, profiles)  # same code path as inline
        return (
            tok.encode(rec),
            {"attributes": rec.attributes, "lifelong": rec.lifelong, "as_of": rec.as_of},
        )

    ex = ProcessPoolExecutor(
        n_workers,
        mp_context=mp.get_context(method),
        initializer=_init_worker,
        initargs=(str(tok_dir), str(data_dir)),
    )
    yielded = 0
    carry_uid: str | None = None
    carry: list = []
    seen: set[str] = set()

    def _check(uid: str) -> None:
        # A uid emitted from two different stitched runs means its rows were not
        # contiguous across tasks — refuse rather than emit it twice (same rule the
        # inline iterator enforces within a task).
        if uid in seen:
            raise ValueError(
                f"user_id {uid!r} appears in non-adjacent rows; events must be grouped "
                "by user. Sort by (user_id, ts) before tokenizing."
            )
        seen.add(uid)

    try:
        for res in _ordered_bounded(ex, _run_task, tasks, 2 * n_workers):
            if res.n_runs == 0:
                continue  # empty task: rows on either side stay adjacent
            assert res.head is not None
            huid, hevents = res.head
            if carry_uid is not None and huid == carry_uid:
                carry.extend(hevents)  # adjacent rows, same uid => same record
            else:
                if carry_uid is not None:
                    _check(carry_uid)
                    yield emit(carry_uid, carry)
                    yielded += 1
                    if max_users is not None and yielded >= max_users:
                        return
                carry_uid, carry = huid, list(hevents)
            if res.n_runs == 1:
                continue  # whole task one run; may chain across tasks
            _check(carry_uid)
            yield emit(carry_uid, carry)  # head closed by run 2 of this task
            yielded += 1
            if max_users is not None and yielded >= max_users:
                return
            for item in res.interior:
                _check(item[0].user_id)
                yield item
                yielded += 1
                if max_users is not None and yielded >= max_users:
                    return
            assert res.tail is not None
            carry_uid, carry = res.tail
            carry = list(carry)
        if carry_uid is not None and (max_users is None or yielded < max_users):
            _check(carry_uid)
            yield emit(carry_uid, carry)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)  # Ctrl-C / max_users / GeneratorExit safe
