"""Typed error hierarchy for pragmatiq.

All pragmatiq exceptions derive from :class:`PragmatiqError`.  Errors for
optional extras subclass :class:`MissingExtraError` (which is also an
``ImportError``) so existing ``except ImportError`` handlers keep working.
"""

from __future__ import annotations


class PragmatiqError(Exception):
    """Base exception for all pragmatiq errors."""


class MissingExtraError(PragmatiqError, ImportError):
    """Raised when an optional install extra is required but not installed.

    Subclasses ``ImportError`` so ``except ImportError`` handlers still catch
    it.  Use :meth:`for_extra` to construct with a clear remedy message.
    """

    @classmethod
    def for_extra(cls, extra: str, missing: str) -> MissingExtraError:
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


class ConfigError(PragmatiqError, ValueError):
    """A configuration value is unknown, malformed, or inconsistent.

    Raised for unknown config keys, unknown model sizes, bad option values and
    an invalid ``resume`` request. Subclasses ``ValueError`` so callers that
    catch ``ValueError`` keep working.
    """


class DataContractError(PragmatiqError, ValueError):
    """Data on disk does not satisfy the contract pragmatiq expects.

    Raised for a missing shard / run / checkpoint directory, a label table
    without the required columns, and a tokenizer-hash mismatch between shards
    and a trained run. Subclasses ``ValueError`` so callers that catch
    ``ValueError`` keep working.
    """
