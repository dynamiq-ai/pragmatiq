"""Progress reporting: tqdm bars when interactive, rate-limited logs otherwise.

Display-only by design: wrappers never touch RNG state, never reorder or
prefetch items from the underlying iterable, and write exclusively to stderr
(CLI stdout stays a single parseable JSON document; ``metrics.jsonl`` stays
line-JSON). This keeps the determinism and resume gates intact.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

log = logging.getLogger(__name__)

#: Cadence of the non-interactive fallback log lines, in seconds. Long enough
#: that CI/gate logs stay readable, short enough that a multi-minute phase is
#: visibly alive. Tests may monkeypatch this.
LOG_INTERVAL_S = 30.0


def _interactive() -> bool:
    """True when a live bar can render: a stderr TTY or a Jupyter kernel."""
    return sys.stderr.isatty() or "ipykernel" in sys.modules


def progress(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    enabled: bool = True,
) -> Iterator[T]:
    """Yield from ``iterable`` with progress reporting on stderr.

    Interactive sessions (terminal TTY or notebook) get a ``tqdm`` bar;
    non-interactive ones (CI, ``nohup``, pipes) get a rate-limited INFO log
    line every :data:`LOG_INTERVAL_S` seconds — and stay completely silent
    for phases that finish sooner. ``enabled=False`` is a transparent
    pass-through.
    """
    if not enabled:
        yield from iterable
        return
    if _interactive():
        from tqdm.auto import tqdm  # notebook-aware flavor; writes to stderr

        yield from tqdm(iterable, total=total, desc=desc, unit=unit)
        return
    t0 = time.time()
    last = t0
    n = 0
    name = desc or "progress"
    for item in iterable:
        yield item
        n += 1
        now = time.time()
        if now - last >= LOG_INTERVAL_S:
            last = now
            rate = n / max(now - t0, 1e-9)
            if total:
                log.info("%s: %d/%d %s (%.0f%%, %.1f %s/s)",
                         name, n, total, unit, 100.0 * n / total, rate, unit)
            else:
                log.info("%s: %d %s (%.1f %s/s)", name, n, unit, rate, unit)
    elapsed = time.time() - t0
    if n and elapsed >= LOG_INTERVAL_S:
        log.info("%s: done — %d %s in %.1fs", name, n, unit, elapsed)
