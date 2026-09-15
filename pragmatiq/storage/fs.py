"""Filesystem resolution and thin I/O operations via fsspec.

:func:`get_fs` is the central resolver: given any URL or local path it returns
an ``(AbstractFileSystem, path)`` pair that all other helpers use.  Local paths
(no scheme, or ``file://``) always resolve to the local filesystem without
requiring any optional package.  Cloud schemes (``s3://``, ``gs://``, ``az://``
…) require the matching optional extra; a clear :class:`~pragmatiq.core.errors.MissingExtraError`
is raised when the extra is absent.
"""

from __future__ import annotations

import posixpath
from pathlib import Path

from fsspec import AbstractFileSystem
from fsspec import filesystem as _fsspec_filesystem
from fsspec.implementations.local import LocalFileSystem

from pragmatiq.core.errors import MissingExtraError

# --------------------------------------------------------------------------- #
# Scheme → (extra_name, package_name) mapping for clear error messages
# --------------------------------------------------------------------------- #
_SCHEME_TO_EXTRA: dict[str, tuple[str, str]] = {
    "s3": ("s3", "s3fs"),
    "s3a": ("s3", "s3fs"),
    "gs": ("gcs", "gcsfs"),
    "gcs": ("gcs", "gcsfs"),
    "az": ("azure", "adlfs"),
    "abfs": ("azure", "adlfs"),
    "abfss": ("azure", "adlfs"),
    "adl": ("azure", "adlfs"),
}

_LOCAL_SCHEMES = {"", "file"}


def _parse_scheme(url: str | Path) -> str:
    """Return the lowercased URI scheme, or '' for plain paths."""
    s = str(url)
    if "://" in s:
        return s.split("://", 1)[0].lower()
    return ""


def get_fs(url: str | Path) -> tuple[AbstractFileSystem, str]:
    """Resolve an fsspec filesystem and the path within it.

    For a plain path or ``file://`` URL the always-available
    :class:`~fsspec.implementations.local.LocalFileSystem` is returned.
    For cloud schemes the corresponding backend package must be installed;
    otherwise :exc:`~pragmatiq.core.errors.MissingExtraError` is raised with
    a ``pip install 'pragmatiq[<extra>]'`` remedy.

    Args:
        url: A local path (``str`` or :class:`~pathlib.Path`) or any URL
             understood by fsspec (``s3://…``, ``memory://…``, etc.).

    Returns:
        ``(fs, path)`` where *path* is the path component within the
        filesystem.

    Raises:
        MissingExtraError: When a cloud backend package is missing.
    """
    scheme = _parse_scheme(url)

    if scheme in _LOCAL_SCHEMES:
        # Strip file:// prefix if present so LocalFileSystem gets a plain path
        path = str(url)
        if path.startswith("file://"):
            path = path[len("file://"):]
        fs: AbstractFileSystem = LocalFileSystem()
        return fs, path

    # Check known cloud schemes and produce a clear error if backend absent
    if scheme in _SCHEME_TO_EXTRA:
        extra, pkg = _SCHEME_TO_EXTRA[scheme]
        try:
            fs = _fsspec_filesystem(scheme)
        except ImportError:
            raise MissingExtraError.for_extra(extra, pkg) from None
        # path is everything after scheme://
        path = str(url).split("://", 1)[1]
        # Normalise: include leading / so callers can join with os.path
        # For memory:// the path may be relative; keep as-is.
        return fs, "/" + path if not path.startswith("/") else path

    # Unknown scheme: let fsspec try; translate ImportError to MissingExtraError
    # if the scheme maps to one of our known extras.
    try:
        fs = _fsspec_filesystem(scheme)
    except ImportError as exc:
        # Check if it happens to map to one of our extras
        extra_info = _SCHEME_TO_EXTRA.get(scheme)
        if extra_info is not None:
            extra, pkg = extra_info
            raise MissingExtraError.for_extra(extra, pkg) from exc
        raise
    path = str(url).split("://", 1)[1]
    return fs, "/" + path if not path.startswith("/") else path


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #

def is_remote(url: str | Path) -> bool:
    """Return ``True`` when *url* has a non-local (cloud/network) scheme."""
    return _parse_scheme(url) not in _LOCAL_SCHEMES


def is_local(url: str | Path) -> bool:
    """Return ``True`` when *url* refers to the local filesystem."""
    return not is_remote(url)


# --------------------------------------------------------------------------- #
# Thin ops — each resolves via get_fs then delegates to the underlying fs
# --------------------------------------------------------------------------- #

def exists(url: str | Path) -> bool:
    """Return ``True`` when *url* exists on its filesystem."""
    fs, path = get_fs(url)
    return fs.exists(path)  # type: ignore[no-any-return]


def ls(url: str | Path) -> list[str]:
    """List the immediate children of *url* on its filesystem.

    Returns:
        A list of full paths/URLs (as strings) for the directory entries.
    """
    fs, path = get_fs(url)
    return fs.ls(path, detail=False)  # type: ignore[no-any-return]


def remove(url: str | Path, *, recursive: bool = False) -> None:
    """Delete *url* from its filesystem.

    Args:
        url:       Path or URL to remove.
        recursive: If ``True``, remove directories recursively.
    """
    fs, path = get_fs(url)
    fs.rm(path, recursive=recursive)


def read_bytes(url: str | Path) -> bytes:
    """Read and return the full contents of *url* as bytes."""
    fs, path = get_fs(url)
    return fs.read_bytes(path)  # type: ignore[no-any-return]


def write_bytes(url: str | Path, data: bytes) -> None:
    """Write *data* bytes to *url*, creating or overwriting the file.

    The parent directory is created automatically if it does not exist yet;
    this works for both local paths and remote/in-memory filesystems via
    the underlying fsspec ``makedirs`` call.
    """
    fs, path = get_fs(url)
    parent = posixpath.dirname(path)
    if parent and parent != path:
        fs.makedirs(parent, exist_ok=True)
    fs.write_bytes(path, data)


def read_text(url: str | Path, *, encoding: str = "utf-8") -> str:
    """Read and return the full contents of *url* as a ``str``."""
    return read_bytes(url).decode(encoding)


def write_text(url: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write *text* to *url*, creating or overwriting the file."""
    write_bytes(url, text.encode(encoding))


__all__: list[str] = [
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
]
