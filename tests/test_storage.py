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

import pyarrow as pa
import pyarrow.parquet as pq
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

    def test_s3_missing_raises_missing_extra(self):
        """s3fs is NOT installed — expect MissingExtraError naming pragmatiq[s3]."""
        with pytest.raises(MissingExtraError) as exc_info:
            get_fs("s3://my-bucket/my-key")
        msg = str(exc_info.value)
        assert "pragmatiq[s3]" in msg
        assert "s3fs" in msg

    def test_s3_error_is_also_import_error(self):
        """MissingExtraError must be catchable as ImportError (backward compat)."""
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

    def test_makedirs(self, tmp_path):
        url = str(tmp_path / "a" / "b" / "c")
        storage.makedirs(url)
        assert os.path.isdir(url)

    def test_makedirs_exist_ok(self, tmp_path):
        url = str(tmp_path / "dir")
        storage.makedirs(url)
        storage.makedirs(url, exist_ok=True)  # must not raise

    def test_ls(self, tmp_path):
        for name in ("a.txt", "b.txt"):
            (tmp_path / name).write_text(name)
        entries = storage.ls(str(tmp_path))
        basenames = {os.path.basename(e) for e in entries}
        assert {"a.txt", "b.txt"} <= basenames

    def test_open_file_rb(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"abc")
        with storage.open_file(str(p), "rb") as fh:
            assert fh.read() == b"abc"

    def test_open_file_wb(self, tmp_path):
        p = tmp_path / "out.bin"
        with storage.open_file(str(p), "wb") as fh:
            fh.write(b"xyz")
        assert p.read_bytes() == b"xyz"


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

    def test_makedirs(self):
        url = "memory:///deep/a/b"
        storage.makedirs(url)
        assert storage.exists(url)

    def test_ls(self):
        storage.write_text("memory:///bucket/a.txt", "a")
        storage.write_text("memory:///bucket/b.txt", "b")
        entries = storage.ls("memory:///bucket")
        basenames = {os.path.basename(e) for e in entries}
        assert {"a.txt", "b.txt"} <= basenames

    def test_open_file(self):
        url = "memory:///test/open.bin"
        with storage.open_file(url, "wb") as fh:
            fh.write(b"remote")
        with storage.open_file(url, "rb") as fh:
            assert fh.read() == b"remote"


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

class TestJsonHelpers:
    def test_round_trip_local(self, tmp_path):
        url = str(tmp_path / "cfg.json")
        obj = {"key": [1, 2, 3], "nested": {"a": True}}
        storage.write_json(url, obj)
        assert storage.read_json(url) == obj

    def test_round_trip_memory(self):
        url = "memory:///cfg/settings.json"
        obj = {"model": "pragma", "layers": 6}
        storage.write_json(url, obj)
        assert storage.read_json(url) == obj

    def test_indent_applied(self, tmp_path):
        url = str(tmp_path / "indented.json")
        storage.write_json(url, {"x": 1}, indent=4)
        raw = storage.read_text(url)
        assert "    " in raw  # 4-space indent


# --------------------------------------------------------------------------- #
# local_path
# --------------------------------------------------------------------------- #

class TestLocalPath:
    def test_local_yields_same_path(self, tmp_path):
        p = tmp_path / "file.bin"
        p.write_bytes(b"content")
        url = str(p)
        with storage.local_path(url) as lp:
            assert lp == url          # no copy, same path
            assert os.path.exists(lp)

    def test_memory_yields_real_local_path(self, tmp_path):
        url = "memory:///remote/data.bin"
        storage.write_bytes(url, b"payload")
        with storage.local_path(url) as lp:
            assert os.path.isabs(lp)
            assert os.path.exists(lp)
            assert open(lp, "rb").read() == b"payload"

    def test_memory_cached_on_second_call(self, tmp_path):
        """Second local_path call for the same URL must reuse the cached file."""
        url = "memory:///cache_test/obj.bin"
        storage.write_bytes(url, b"data")
        with storage.local_path(url) as lp1:
            mtime1 = os.path.getmtime(lp1)
        with storage.local_path(url) as lp2:
            mtime2 = os.path.getmtime(lp2)
        assert lp1 == lp2
        assert mtime1 == mtime2  # not re-downloaded


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
# atomic_write
# --------------------------------------------------------------------------- #

class TestAtomicWrite:
    def test_local_success(self, tmp_path):
        target = str(tmp_path / "out.bin")
        with storage.atomic_write(target, "wb") as fh:
            fh.write(b"final")
        assert open(target, "rb").read() == b"final"
        # no .tmp left over
        assert not os.path.exists(target + ".tmp")

    def test_local_exception_leaves_no_partial(self, tmp_path):
        target = str(tmp_path / "out.bin")
        with pytest.raises(ValueError, match="oops"):
            with storage.atomic_write(target, "wb") as fh:
                fh.write(b"partial")
                raise ValueError("oops")
        assert not os.path.exists(target)        # target must not exist
        assert not os.path.exists(target + ".tmp")  # tmp must be cleaned up

    def test_local_exception_preserves_existing(self, tmp_path):
        target = str(tmp_path / "out.bin")
        open(target, "wb").write(b"original")
        with pytest.raises(RuntimeError):
            with storage.atomic_write(target, "wb") as fh:
                fh.write(b"new-partial")
                raise RuntimeError("fail")
        assert open(target, "rb").read() == b"original"  # unchanged

    def test_memory_success(self):
        url = "memory:///atomic/result.bin"
        with storage.atomic_write(url, "wb") as fh:
            fh.write(b"committed")
        assert storage.read_bytes(url) == b"committed"

    def test_memory_exception_leaves_no_target(self):
        url = "memory:///atomic/fail.bin"
        with pytest.raises(ZeroDivisionError):
            with storage.atomic_write(url, "wb") as fh:
                fh.write(b"bad")
                raise ZeroDivisionError
        assert not storage.exists(url)


# --------------------------------------------------------------------------- #
# pyarrow_filesystem
# --------------------------------------------------------------------------- #

class TestPyarrowFilesystem:
    def _make_table(self) -> pa.Table:
        return pa.table({"id": [1, 2, 3], "label": ["a", "b", "c"]})

    def test_local_returns_none_fs(self, tmp_path):
        fs, path = storage.pyarrow_filesystem(str(tmp_path / "t.parquet"))
        assert fs is None
        assert str(tmp_path) in path

    def test_memory_returns_pa_filesystem(self):
        import pyarrow.fs as pafs
        fs, path = storage.pyarrow_filesystem("memory:///pa/t.parquet")
        assert isinstance(fs, pafs.PyFileSystem)
        assert path == "/pa/t.parquet"

    def test_parquet_round_trip_local(self, tmp_path):
        table = self._make_table()
        target = str(tmp_path / "t.parquet")
        fs, path = storage.pyarrow_filesystem(target)
        pq.write_table(table, path, filesystem=fs)
        t2 = pq.read_table(path, filesystem=fs)
        assert t2.equals(table)

    def test_parquet_round_trip_memory(self):
        """Write and read a parquet table via memory:// using pyarrow_filesystem."""
        table = self._make_table()
        url = "memory:///parquet/t.parquet"

        # Write via pyarrow_filesystem
        fs, path = storage.pyarrow_filesystem(url)
        # Ensure parent dir exists in memory fs
        storage.makedirs("memory:///parquet")
        pq.write_table(table, path, filesystem=fs)

        # Read back
        t2 = pq.read_table(path, filesystem=fs)
        assert t2.equals(table)


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
# PRAGMATIQ_CACHE_DIR env-var is honoured by local_path
# --------------------------------------------------------------------------- #

def test_cache_dir_env_var_honoured(tmp_path, monkeypatch):
    """PRAGMATIQ_CACHE_DIR must be used as the root for materialised remote files."""
    custom_cache = tmp_path / "my_cache"
    monkeypatch.setenv("PRAGMATIQ_CACHE_DIR", str(custom_cache))

    url = "memory:///env_test/payload.bin"
    data = b"\xca\xfe\xba\xbe"
    storage.write_bytes(url, data)

    with storage.local_path(url) as lp:
        # The local path must live under our custom cache dir
        assert os.path.abspath(lp).startswith(os.path.abspath(str(custom_cache))), (
            f"Expected path under {custom_cache}, got {lp}"
        )
        # And the bytes must match
        with open(lp, "rb") as fh:
            assert fh.read() == data


# --------------------------------------------------------------------------- #
# Parent-directory auto-creation in write_bytes / write_text / atomic_write
# --------------------------------------------------------------------------- #

class TestParentDirAutoCreate:
    """write_bytes, write_text, and atomic_write must create missing parent dirs."""

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

    def test_atomic_write_local_creates_parent(self, tmp_path):
        target = str(tmp_path / "atomic_sub" / "result.bin")
        with storage.atomic_write(target, "wb") as fh:
            fh.write(b"atomic")
        assert os.path.exists(target)
        assert open(target, "rb").read() == b"atomic"
        # No stray .tmp left over
        assert not os.path.exists(target + ".tmp")

    def test_atomic_write_memory_creates_parent(self):
        url = "memory:///atomic_parent/nested/out.bin"
        with storage.atomic_write(url, "wb") as fh:
            fh.write(b"remote-atomic")
        assert storage.read_bytes(url) == b"remote-atomic"
