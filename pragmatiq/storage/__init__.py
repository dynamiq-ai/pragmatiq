"""pragmatiq.storage — fsspec-backed storage abstraction.

Provides a unified I/O surface over local filesystems and object stores
(S3, GCS, Azure Blob) via `fsspec`.  Local paths always work without any
optional package; cloud backends require the matching extra
(``pragmatiq[s3]``, ``pragmatiq[gcs]``, ``pragmatiq[azure]``).

Quick start::

    import pragmatiq.storage as storage

    # Works for local paths and memory:// (offline testing) out of the box:
    storage.write_text("memory://my/file.txt", "hello")
    assert storage.read_text("memory://my/file.txt") == "hello"

    # Cloud (requires extra):
    storage.write_bytes("s3://bucket/key", b"data")   # needs pragmatiq[s3]

Public surface
--------------
Filesystem resolution:
    :func:`get_fs`, :func:`is_remote`, :func:`is_local`

Thin ops:
    :func:`exists`, :func:`makedirs`, :func:`ls`, :func:`remove`,
    :func:`read_bytes`, :func:`write_bytes`, :func:`read_text`,
    :func:`write_text`, :func:`open_file`

Cache / local materialisation:
    :func:`local_path`, :func:`materialize_dir`

Artifact helpers:
    :func:`read_json`, :func:`write_json`, :func:`atomic_write`,
    :func:`pyarrow_filesystem`
"""

from pragmatiq.storage.artifacts import (
    atomic_write,
    pyarrow_filesystem,
    read_json,
    write_json,
)
from pragmatiq.storage.cache import (
    local_path,
    materialize_dir,
)
from pragmatiq.storage.fs import (
    exists,
    get_fs,
    is_local,
    is_remote,
    ls,
    makedirs,
    open_file,
    read_bytes,
    read_text,
    remove,
    write_bytes,
    write_text,
)

__all__: list[str] = [
    # fs.py
    "get_fs",
    "is_remote",
    "is_local",
    "exists",
    "makedirs",
    "ls",
    "remove",
    "read_bytes",
    "write_bytes",
    "read_text",
    "write_text",
    "open_file",
    # cache.py
    "local_path",
    "materialize_dir",
    # artifacts.py
    "read_json",
    "write_json",
    "atomic_write",
    "pyarrow_filesystem",
]
