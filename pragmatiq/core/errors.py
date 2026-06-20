"""Typed error hierarchy for pragmatiq.

All pragmatiq exceptions derive from :class:`PragmatiqError`.  Errors for
optional extras subclass :class:`MissingExtraError` (which is also an
``ImportError``) so existing ``except ImportError`` handlers keep working.
"""

from __future__ import annotations


class PragmatiqError(Exception):
    """Base exception for all pragmatiq errors."""


class MissingExtraError(ImportError):
    """Raised when an optional install extra is required but not installed.

    Subclasses ``ImportError`` so ``except ImportError`` handlers still catch
    it.  Use :meth:`for_extra` to construct with a clear remedy message.
    """

    @classmethod
    def for_extra(cls, extra: str, missing: str) -> "MissingExtraError":
        """Return a :class:`MissingExtraError` with a clear pip-install remedy.

        Args:
            extra:   The pragmatiq extras name, e.g. ``"train"``.
            missing: The missing package name, e.g. ``"lightning"``.

        Example::

            raise MissingExtraError.for_extra("train", "lightning")
            # message: "pragmatiq[train] is required for this feature:
            #           pip install 'pragmatiq[train]' (missing: lightning)"
        """
        msg = (
            f"pragmatiq[{extra}] is required for this feature: "
            f"pip install 'pragmatiq[{extra}]' (missing: {missing})"
        )
        return cls(msg)
