"""Component registries: configs reference heads/maskers/value-encoders by name.

Engineers extend pragmatiq without forking by registering a component under a
name and pointing the config at it::

    from pragmatiq.registry import register_head

    @register_head("my_ranking_head")
    class MyRankingHead(nn.Module): ...

Registries are plain dicts behind type-safe decorator factories; lookups raise
``KeyError`` with the list of known names so config typos fail loudly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class Registry:
    """A named string → class/callable registry with decorator registration."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a class decorator that registers the class under ``name``."""

        def deco(cls: type[T]) -> type[T]:
            if name in self._items and self._items[name] is not cls:
                raise ValueError(f"{self.kind} {name!r} already registered to {self._items[name]!r}")
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type:
        """Look up a registered component; raise with known names on miss."""
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        """Sorted list of registered names."""
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


HEADS = Registry("head")
MASKERS = Registry("masker")
VALUE_ENCODERS = Registry("value_encoder")
TEXT_ENCODERS = Registry("text_encoder")


def register_head(name: str) -> Callable[[type[T]], type[T]]:
    """Register a task head class under ``name`` (used by finetune configs)."""
    return HEADS.register(name)


def register_masker(name: str) -> Callable[[type[T]], type[T]]:
    """Register a masking strategy class under ``name`` (used by pretrain configs)."""
    return MASKERS.register(name)


def register_value_encoder(name: str) -> Callable[[type[T]], type[T]]:
    """Register a value-encoder class under ``name`` (used by tokenizer configs)."""
    return VALUE_ENCODERS.register(name)


def register_text_encoder(name: str) -> Callable[[type[T]], type[T]]:
    """Register a frozen text-embedding encoder under ``name`` (the Nemotron variant)."""
    return TEXT_ENCODERS.register(name)


def get_head(name: str) -> type:
    """Resolve a head by registered name."""
    return HEADS.get(name)


def get_masker(name: str) -> type:
    """Resolve a masker by registered name."""
    return MASKERS.get(name)


def get_value_encoder(name: str) -> type:
    """Resolve a value encoder by registered name."""
    return VALUE_ENCODERS.get(name)


def get_text_encoder(name: str) -> type:
    """Resolve a frozen text-embedding encoder by registered name."""
    return TEXT_ENCODERS.get(name)
