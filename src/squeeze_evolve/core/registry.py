"""Typed operator registry for Squeeze-Evolve.

Each operator family gets its own ``Registry`` instance.  Built-in operators
register at import time.  Extend by importing the registry and decorating::

    from squeeze_evolve import fitness

    @fitness.register("my_metric")
    def my_metric(scores):
        return float(max(scores) - min(scores))
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T", bound=Callable)


class Registry(Generic[T]):
    """A named mapping from string keys to callables."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._fns: dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator that registers *fn* under *key*."""
        def decorator(fn: T) -> T:
            if key in self._fns:
                raise KeyError(f"{self._name} operator {key!r} is already registered")
            self._fns[key] = fn
            return fn
        return decorator

    def get(self, key: str) -> T:
        """Return the function registered under *key*, or raise ``KeyError``."""
        try:
            return self._fns[key]
        except KeyError:
            available = ", ".join(sorted(self._fns)) or "(none)"
            raise KeyError(f"Unknown {self._name} operator: {key!r}. Available: {available}") from None

    def __contains__(self, key: str) -> bool:
        return key in self._fns

    def keys(self) -> list[str]:
        return list(self._fns)

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, keys={self.keys()})"
