"""Stage-in / stage-out context manager for the api boundary.

Allows api functions to accept remote URLs (s3://, gs://, memory://, etc.)
without touching the internal modules. Remote inputs are materialised to a
local temp directory; remote outputs are registered and uploaded on clean
exit. Local paths pass through unchanged (no temp, no upload) so all
existing tests are byte-identical.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pragmatiq.storage.cache import materialize_dir, put_dir
from pragmatiq.storage.fs import get_fs, is_local, write_bytes

_PathLike = str | Path | None


class Stage:
    """Tracks staged inputs/outputs for a single api call."""

    def __init__(self, work_root: str) -> None:
        self._work_root = Path(work_root)
        self._outputs: list[tuple[Path, str, bool]] = []  # (local_path, remote_url, is_dir)
        self._counter = 0

    def _next_slot(self) -> Path:
        self._counter += 1
        slot = self._work_root / f"slot_{self._counter}"
        return slot

    def input(self, url: _PathLike) -> _PathLike:
        """Remote url -> local materialized path (file or dir); local / None -> unchanged.

        For remote urls, auto-detects whether the target is a file or directory
        via ``fs.isdir()``, then downloads accordingly.

        Args:
            url: Remote URL, local path, or ``None``.

        Returns:
            A local :class:`~pathlib.Path` for remote urls, or the original
            value for local paths and ``None``.
        """
        if url is None or is_local(url):
            return url
        remote = str(url)
        fs, fpath = get_fs(remote)
        slot = self._next_slot()
        if fs.isdir(fpath):
            slot.mkdir(parents=True, exist_ok=True)
            materialize_dir(remote, slot)
            return slot
        else:
            # Single file: download to slot/filename
            slot.mkdir(parents=True, exist_ok=True)
            fname = fpath.rstrip("/").split("/")[-1] or "data"
            local_file = slot / fname
            with fs.open(fpath, "rb") as src, local_file.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            return local_file

    def output(self, url: _PathLike, *, is_dir: bool) -> _PathLike:
        """Remote url -> a fresh local temp path; local / None -> unchanged.

        The local path is registered for upload on clean exit.

        Args:
            url:    Remote destination URL (or local path / ``None``).
            is_dir: ``True`` if the output is a directory, ``False`` if a file.

        Returns:
            A local :class:`~pathlib.Path` for the caller to write to (remote
            case), or the original value for local paths and ``None``.
        """
        if url is None or is_local(url):
            return url
        remote = str(url)
        slot = self._next_slot()
        if is_dir:
            slot.mkdir(parents=True, exist_ok=True)
            local: Path = slot
        else:
            slot.mkdir(parents=True, exist_ok=True)
            local = slot / Path(remote).name
        self._outputs.append((local, remote, is_dir))
        return local

    def _upload_all(self) -> None:
        """Upload all registered outputs to their remote URLs."""
        for local, remote, is_dir in self._outputs:
            if is_dir:
                put_dir(local, remote)
            else:
                if local.exists():
                    write_bytes(remote, local.read_bytes())


@contextmanager
def staging() -> Iterator[Stage]:
    """Context manager that provides a :class:`Stage` for one api call.

    On normal exit, uploads all registered remote outputs.
    On exception, skips uploads and cleans up the temp directory.

    Yields:
        A :class:`Stage` instance for staging inputs and outputs.
    """
    work_root = tempfile.mkdtemp(prefix="pragmatiq-stage-")
    stage = Stage(work_root)
    try:
        yield stage
        stage._upload_all()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


__all__: list[str] = ["staging", "Stage"]
