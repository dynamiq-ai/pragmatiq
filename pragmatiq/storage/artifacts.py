"""Higher-level artifact helpers built on :mod:`pragmatiq.storage.fs`.

Provides JSON read/write, best-effort atomic writes (temp-then-commit), and
a :func:`pyarrow_filesystem` adapter for ``pyarrow.parquet`` read/write against
any fsspec-backed store.

Atomic write semantics
----------------------
- **Local paths**: write to a sibling ``*.tmp`` file, then rename via
  :func:`os.replace` (atomic on POSIX, best-effort on Windows).  If the context
  block raises an exception the ``.tmp`` file is removed and the target is left
  untouched.
- **Remote URLs**: write to a temp key on the same filesystem, then rename via
  ``fs.mv``.  Object-store rename is not atomic (it's a copy-then-delete under
  the hood), but readers are far less likely to observe a partial file this way
  than with a plain overwrite.

pyarrow filesystem integration
-------------------------------
:func:`pyarrow_filesystem` wraps an fsspec filesystem with
``pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs))``.  This approach works
with pyarrow ≥ 14 (confirmed against the installed pyarrow 24).  For local paths
``(None, local_path)`` is returned so pyarrow uses the OS filesystem directly
(no wrapper overhead).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from pragmatiq.storage.fs import get_fs, is_local

__all__: list[str] = [
    "read_json",
    "write_json",
    "atomic_write",
    "pyarrow_filesystem",
]


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def read_json(url: str) -> Any:
    """Read and deserialise a JSON file from *url*.

    Args:
        url: Local path or remote URL of the JSON file.

    Returns:
        The deserialised Python object.
    """
    fs, path = get_fs(url)
    with fs.open(path, "rb") as fh:
        return json.load(fh)


def write_json(url: str, obj: Any, *, indent: int = 2) -> None:
    """Serialise *obj* to JSON and write it to *url*.

    Args:
        url:    Target local path or remote URL.
        obj:    JSON-serialisable Python object.
        indent: Pretty-print indentation level (default ``2``).
    """
    data = json.dumps(obj, indent=indent).encode()
    fs, path = get_fs(url)
    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as fh:
        fh.write(data)


# --------------------------------------------------------------------------- #
# Atomic write
# --------------------------------------------------------------------------- #

@contextmanager
def atomic_write(url: str, mode: str = "wb") -> Generator[Any, None, None]:
    """Context manager for write-then-commit so readers never see a partial file.

    On **local** paths: writes to ``<target>.tmp`` then calls
    :func:`os.replace`.  On exception the ``.tmp`` is deleted; the target is
    unchanged.

    On **remote** URLs: writes to a temporary key on the same filesystem, then
    calls ``fs.mv`` to the final key.  Object stores do not guarantee atomic
    rename, but this avoids exposing partial content during the write phase.

    Args:
        url:  Final destination path or URL.
        mode: File mode (``"wb"`` or ``"w"``).

    Yields:
        A writable file-like object.  Commit happens automatically on clean exit.
    """
    fs, path = get_fs(url)

    if is_local(url):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, mode) as fh:
                yield fh
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise
    else:
        # Remote: write to a unique temp key, then mv
        parent = os.path.dirname(path)
        tmp_key = (parent + "/" if parent else "") + f".tmp-{uuid.uuid4().hex}"
        try:
            with fs.open(tmp_key, mode) as fh:
                yield fh
            # mv is copy + delete on most object stores — best-effort atomicity
            fs.mv(tmp_key, path)
        except BaseException:
            try:
                fs.rm(tmp_key)
            except Exception:
                pass
            raise


# --------------------------------------------------------------------------- #
# PyArrow filesystem adapter
# --------------------------------------------------------------------------- #

def pyarrow_filesystem(url: str) -> tuple[Any | None, str]:
    """Return ``(filesystem, path)`` for use with :mod:`pyarrow.parquet`.

    Suitable for::

        pq.read_table(path, filesystem=filesystem)
        pq.write_table(table, path, filesystem=filesystem)

    - **Local** *url*: returns ``(None, local_path)`` — pyarrow uses the OS
      filesystem directly (no wrapper needed or desired).
    - **Remote** *url*: wraps the fsspec fs with
      ``pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs))`` and returns the
      path component.  Tested against pyarrow 24 + fsspec 2026.4.

    Args:
        url: Local path or remote URL.

    Returns:
        ``(filesystem_or_None, path_string)`` pair.
    """
    if is_local(url):
        local = url[len("file://"):] if url.startswith("file://") else url
        return None, local

    import pyarrow.fs as pafs  # optional dep but pyarrow is in core deps

    fs, path = get_fs(url)
    pa_fs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
    return pa_fs, path
