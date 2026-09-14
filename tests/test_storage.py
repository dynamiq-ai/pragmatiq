"""Unit tests for pragmatiq.storage — fsspec abstraction layer.

Uses ``tmp_path`` (local filesystem) and ``memory://`` (fsspec in-memory fs,
simulates a remote store offline) to achieve real round-trips without requiring
any cloud credentials or backend packages.

All tests run with only the packages in the core + dev extras (i.e. NO s3fs,
gcsfs, adlfs), which allows testing the "missing backend" error path for real.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import pragmatiq.storage as storage
from pragmatiq.core.errors import MissingExtraError
from pragmatiq.storage.fs import get_fs, is_local, is_remote

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _mem_url(path: str) -> str:
    """Return a memory:// URL; ensure the path starts with /."""
    return f"memory://{path}" if path.startswith("/") else f"memory:///{path}"


# Wipe the in-memory filesystem between tests so state doesn't leak.
@pytest.fixture(autouse=True)
def _clean_memory_fs():
    """Clear the shared MemoryFileSystem before and after each test."""
    import fsspec
    mem = fsspec.filesystem("memory")
    mem.store.clear()
    yield
    mem.store.clear()


# --------------------------------------------------------------------------- #
# get_fs: scheme resolution
# --------------------------------------------------------------------------- #

class TestGetFs:
    def test_plain_path_gives_local_fs(self, tmp_path):
        fs, path = get_fs(str(tmp_path))
        from fsspec.implementations.local import LocalFileSystem
        assert isinstance(fs, LocalFileSystem)
        assert path == str(tmp_path)

    def test_file_scheme_gives_local_fs(self, tmp_path):
        url = f"file://{tmp_path}"
        fs, path = get_fs(url)
        from fsspec.implementations.local import LocalFileSystem
        assert isinstance(fs, LocalFileSystem)
        assert path == str(tmp_path)

    def test_pathlib_path_gives_local_fs(self, tmp_path):
        from pathlib import Path
        fs, path = get_fs(Path(tmp_path))
        from fsspec.implementations.local import LocalFileSystem
        assert isinstance(fs, LocalFileSystem)

    def test_memory_scheme_resolves(self):
        fs, path = get_fs("memory:///bucket/key")
        from fsspec.implementations.memory import MemoryFileSystem
        assert isinstance(fs, MemoryFileSystem)
        assert path == "/bucket/key"

    def test_s3_missing_raises_missing_extra(self, monkeypatch):
        """get_fs('s3://...') raises MissingExtraError naming pragmatiq[s3] and s3fs.

        The test simulates s3fs absence via monkeypatch so it is robust whether
        or not s3fs is installed in the current environment (CI installs the full
        extras; local dev does not).  Setting sys.modules['s3fs'] = None makes
        any ``import s3fs`` raise ImportError, which is the same signal that
        get_fs relies on to detect the missing extra.
        """
        monkeypatch.setitem(sys.modules, "s3fs", None)
        # Also block the s3fs subpackage so fsspec cannot import it via s3fs.core.
        monkeypatch.setitem(sys.modules, "s3fs.core", None)
        with pytest.raises(MissingExtraError) as exc_info:
            get_fs("s3://my-bucket/my-key")
        msg = str(exc_info.value)
        assert "pragmatiq[s3]" in msg
        assert "s3fs" in msg

    def test_s3_error_is_also_import_error(self, monkeypatch):
        """MissingExtraError must be catchable as ImportError (backward compat).

        Uses the same monkeypatch trick as test_s3_missing_raises_missing_extra
        so the test is env-independent (works with or without s3fs installed).
        """
        monkeypatch.setitem(sys.modules, "s3fs", None)
        monkeypatch.setitem(sys.modules, "s3fs.core", None)
        with pytest.raises(ImportError):
            get_fs("s3://bucket/key")


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #

class TestPredicates:
    def test_local_path_is_local(self, tmp_path):
        assert is_local(str(tmp_path))
        assert not is_remote(str(tmp_path))

    def test_file_scheme_is_local(self, tmp_path):
        assert is_local(f"file://{tmp_path}")

    def test_memory_is_remote(self):
        assert is_remote("memory:///x")
        assert not is_local("memory:///x")

    def test_s3_is_remote(self):
        assert is_remote("s3://bucket/key")


# --------------------------------------------------------------------------- #
# Round-trip: write_text / read_text, write_bytes / read_bytes — local
# --------------------------------------------------------------------------- #

class TestRoundTripLocal:
    def test_write_read_text(self, tmp_path):
        url = str(tmp_path / "hello.txt")
        storage.write_text(url, "hello world")
        assert storage.read_text(url) == "hello world"

    def test_write_read_bytes(self, tmp_path):
        url = str(tmp_path / "data.bin")
        storage.write_bytes(url, b"\x00\x01\x02")
        assert storage.read_bytes(url) == b"\x00\x01\x02"

    def test_exists_true_false(self, tmp_path):
        url = str(tmp_path / "f.txt")
        assert not storage.exists(url)
        storage.write_text(url, "x")
        assert storage.exists(url)

    def test_ls(self, tmp_path):
        for name in ("a.txt", "b.txt"):
            (tmp_path / name).write_text(name)
        entries = storage.ls(str(tmp_path))
        basenames = {os.path.basename(e) for e in entries}
        assert {"a.txt", "b.txt"} <= basenames

# --------------------------------------------------------------------------- #
# Round-trip: write_text / read_text, write_bytes / read_bytes — memory://
# --------------------------------------------------------------------------- #

class TestRoundTripMemory:
    def test_write_read_text(self):
        url = "memory:///test/hello.txt"
        storage.write_text(url, "hello remote")
        assert storage.read_text(url) == "hello remote"

    def test_write_read_bytes(self):
        url = "memory:///test/data.bin"
        storage.write_bytes(url, b"\xde\xad\xbe\xef")
        assert storage.read_bytes(url) == b"\xde\xad\xbe\xef"

    def test_exists(self):
        url = "memory:///test/ex.txt"
        assert not storage.exists(url)
        storage.write_text(url, "hi")
        assert storage.exists(url)

    def test_ls(self):
        storage.write_text("memory:///bucket/a.txt", "a")
        storage.write_text("memory:///bucket/b.txt", "b")
        entries = storage.ls("memory:///bucket")
        basenames = {os.path.basename(e) for e in entries}
        assert {"a.txt", "b.txt"} <= basenames

# --------------------------------------------------------------------------- #
# materialize_dir
# --------------------------------------------------------------------------- #

class TestMaterializeDir:
    def _write_tree(self):
        """Write a small directory tree into memory:// and return the root URL."""
        root = "memory:///lmdb_index"
        files = {
            "/lmdb_index/data.mdb": b"\x01" * 128,
            "/lmdb_index/lock.mdb": b"\x02" * 8,
            "/lmdb_index/meta.json": b'{"version": 1}',
        }
        import fsspec
        mem = fsspec.filesystem("memory")
        for path, data in files.items():
            mem.makedirs(os.path.dirname(path), exist_ok=True)
            with mem.open(path, "wb") as fh:
                fh.write(data)
        return root, files

    def test_materialize_all_files_present(self, tmp_path):
        root_url, expected = self._write_tree()
        dest = tmp_path / "materialized"
        storage.materialize_dir(root_url, str(dest))

        for remote_path, data in expected.items():
            rel = remote_path[len("/lmdb_index/"):]
            local_file = dest / rel
            assert local_file.exists(), f"Missing: {rel}"
            assert local_file.read_bytes() == data

    def test_materialize_local_to_local(self, tmp_path):
        """materialize_dir also works for local → local (copy)."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("aaa")
        (src / "b.txt").write_text("bbb")
        dest = tmp_path / "dest"
        storage.materialize_dir(str(src), str(dest))
        assert (dest / "a.txt").read_text() == "aaa"
        assert (dest / "b.txt").read_text() == "bbb"


# --------------------------------------------------------------------------- #
# torch-free import check
# --------------------------------------------------------------------------- #

def test_storage_is_torch_free():
    """import pragmatiq.storage must not pull in torch in a clean interpreter."""
    # Run in a subprocess so we start from a truly fresh sys.modules — no leakage
    # from other tests that may have imported torch earlier in the same process.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pragmatiq.storage, sys; "
                "assert not any("
                "    m == 'torch' or m.startswith('torch.')"
                "    for m in sys.modules"
                "), f'torch found in sys.modules: {[m for m in sys.modules if m==\"torch\" or m.startswith(\"torch.\")]}'; "
                "print('OK')"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"torch-free check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_storage_public_api_complete():
    """All names listed in __all__ are importable from pragmatiq.storage."""
    import pragmatiq.storage as s
    for name in s.__all__:
        assert hasattr(s, name), f"{name} in __all__ but not importable"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Parent-directory auto-creation in write_bytes / write_text / atomic_write
# --------------------------------------------------------------------------- #

class TestParentDirAutoCreate:
    """write_bytes and write_text must create missing parent dirs."""

    def test_write_bytes_local_creates_parent(self, tmp_path):
        target = str(tmp_path / "new_subdir" / "deep" / "file.bin")
        storage.write_bytes(target, b"hello")
        assert os.path.exists(target)
        assert open(target, "rb").read() == b"hello"

    def test_write_text_local_creates_parent(self, tmp_path):
        target = str(tmp_path / "sub" / "file.txt")
        storage.write_text(target, "world")
        assert os.path.exists(target)
        assert open(target).read() == "world"

    def test_write_bytes_memory_creates_parent(self):
        url = "memory:///new_parent/child/data.bin"
        storage.write_bytes(url, b"\x01\x02\x03")
        assert storage.read_bytes(url) == b"\x01\x02\x03"

    def test_write_text_memory_creates_parent(self):
        url = "memory:///new_parent2/nested/text.txt"
        storage.write_text(url, "remote text")
        assert storage.read_text(url) == "remote text"

