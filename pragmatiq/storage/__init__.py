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
    :func:`exists`, :func:`ls`, :func:`remove`, :func:`read_bytes`,
    :func:`write_bytes`, :func:`read_text`, :func:`write_text`

Local materialisation:
    :func:`materialize_dir`, :func:`put_dir`

Stage-in / stage-out:
    :func:`staging`, :class:`Stage`
"""

from pragmatiq.storage.cache import materialize_dir, put_dir
from pragmatiq.storage.fs import (
    exists,
    get_fs,
    is_local,
    is_remote,
    ls,
    read_bytes,
    read_text,
    remove,
    write_bytes,
    write_text,
)
from pragmatiq.storage.staging import Stage, staging

__all__: list[str] = [
    # fs.py
    "get_fs",
    "is_remote",
    "is_local",
    "exists",
    "ls",
    "remove",
    "read_bytes",
    "write_bytes",
    "read_text",
    "write_text",
    # cache.py
    "materialize_dir",
    "put_dir",
    # staging.py
    "staging",
    "Stage",
]
