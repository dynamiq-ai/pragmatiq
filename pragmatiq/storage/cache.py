"""Download-to-local-cache helpers for storage backends that require a real path.

Some consumers (LMDB, ONNX Runtime) need a genuine local filesystem path rather
than a file-like object.  :func:`local_path` provides a context-manager that
transparently passes through local URLs and downloads remote ones to a stable
cache directory, so repeated accesses to the same remote URL do not re-download.

Cache directory resolution (in priority order):

1. ``PRAGMATIQ_CACHE_DIR`` environment variable (persistent — never deleted).
2. ``<tempdir>/pragmatiq-cache`` (persistent across calls within a process;
   only ad-hoc copies outside this dir are cleaned up on exit).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from pragmatiq.storage.fs import get_fs, is_local, ls

log = logging.getLogger(__name__)


def _cache_root() -> Path:
    """Return the root cache directory, honouring ``PRAGMATIQ_CACHE_DIR``."""
    env = os.environ.get("PRAGMATIQ_CACHE_DIR")
    if env:
        root = Path(env)
    else:
        root = Path(tempfile.gettempdir()) / "pragmatiq-cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _url_key(url: str) -> str:
    """Derive a stable short key from a URL for use as a cache subdirectory."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


@contextmanager
def local_path(url: str, *, name: str | None = None) -> Generator[str, None, None]:
    """Yield a **real local filesystem path** for *url*.

    - **Local** ``url``: yields the original path unchanged — no copy, no temp.
    - **Remote** ``url`` (single file): downloads to
      ``<cache_root>/<url_key>/<name>`` on first access and yields the local
      path; subsequent calls with the same URL reuse the cached file.

    The cache is **never deleted** when it resides under ``PRAGMATIQ_CACHE_DIR``
    or the default ``<tempdir>/pragmatiq-cache`` (it is meant to be reused).

    Args:
        url:  Source URL or local path.
        name: Override the filename used inside the cache directory.  Defaults
              to the basename of *url*.

    Yields:
        Absolute local filesystem path as a ``str``.
    """
    if is_local(url):
        # Normalise file:// prefix if present
        local = url[len("file://"):] if url.startswith("file://") else url
        yield local
        return

    # Remote: download to cache on first access
    fs, remote_path = get_fs(url)
    filename = name or os.path.basename(remote_path.rstrip("/")) or "data"
    cache_dir = _cache_root() / _url_key(url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_file = cache_dir / filename

    if not local_file.exists():
        log.debug("Downloading %s → %s", url, local_file)
        with fs.open(remote_path, "rb") as src, local_file.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        log.debug("Cache hit: %s → %s", url, local_file)

    yield str(local_file)


def materialize_dir(url: str, dest: str | Path) -> None:
    """Recursively copy a directory at *url* to a local *dest* path.

    Works for both local and remote sources.  Existing files at *dest* are
    overwritten.  This is the primary entry-point for materialising LMDB index
    directories from object storage before opening them with LMDB.

    Args:
        url:  Source directory URL (local or remote).
        dest: Local destination directory path.  Created if absent.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if is_local(url):
        src_root = Path(url[len("file://"):] if url.startswith("file://") else url)
        for src_file in src_root.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(src_root)
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, out)
        return

    # Remote: list recursively and download each file
    fs, remote_root = get_fs(url)
    remote_root = remote_root.rstrip("/")
    try:
        all_entries = fs.find(remote_root)
    except Exception:
        # Fall back to ls if find is not supported
        all_entries = ls(url)

    for entry in all_entries:
        if not fs.isfile(entry):
            continue
        # Compute relative path under remote_root
        rel = entry[len(remote_root):].lstrip("/")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        log.debug("Downloading %s → %s", entry, out)
        with fs.open(entry, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)


__all__: list[str] = [
    "local_path",
    "materialize_dir",
]
