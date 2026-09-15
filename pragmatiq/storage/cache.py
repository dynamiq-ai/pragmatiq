"""Directory materialisation between local disk and object storage.

Some consumers (LMDB, the serving runtime) need a genuine local directory rather
than a file-like object: :func:`materialize_dir` downloads a remote directory
tree to local disk and :func:`put_dir` uploads one, mirroring the tree.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pragmatiq.storage.fs import get_fs, is_local, ls

log = logging.getLogger(__name__)


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


def put_dir(local_dir: str | Path, url: str) -> None:
    """Recursively upload a local directory to *url*.

    - **Local** *url*: performs a local recursive copy (like :func:`materialize_dir`
      but in the opposite direction).
    - **Remote** *url*: uploads every file under *local_dir* to the remote
      filesystem rooted at *url*.

    Args:
        local_dir: Source directory on the local filesystem.
        url:       Destination directory URL (local or remote).
    """
    local_dir = Path(local_dir)
    if is_local(url):
        dest = Path(url[len("file://"):] if url.startswith("file://") else url)
        dest.mkdir(parents=True, exist_ok=True)
        for src_file in local_dir.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(local_dir)
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, out)
        return

    # Remote: upload each file
    fs, remote_root = get_fs(url)
    remote_root = remote_root.rstrip("/")
    for src_file in local_dir.rglob("*"):
        if src_file.is_file():
            rel = src_file.relative_to(local_dir)
            remote_path = remote_root + "/" + str(rel).replace("\\", "/")
            # ensure parent exists
            parent = "/".join(remote_path.split("/")[:-1])
            if parent:
                # makedirs is not on the base AbstractFileSystem spec but is
                # present on all backends we support (memory, s3, gcs, azure).
                fs.makedirs(parent, exist_ok=True)
            with src_file.open("rb") as src, fs.open(remote_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


__all__: list[str] = [
    "materialize_dir",
    "put_dir",
]
