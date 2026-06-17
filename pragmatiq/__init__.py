"""pragmatiq: an open-source behavioral foundation model for banking event sequences.

Independent implementation inspired by the PRAGMA paper (arXiv 2604.08649);
not affiliated with or endorsed by Revolut.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pragmatiq")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.1.0b2"

__all__ = ["__version__"]
