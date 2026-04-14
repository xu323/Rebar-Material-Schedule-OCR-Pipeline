"""Pluggable shape classifier registry."""

from __future__ import annotations

from typing import Any

from .base import ShapeClassifier

_REGISTRY: dict[str, type[ShapeClassifier]] = {}


def register(name: str, cls: type[ShapeClassifier]) -> None:
    _REGISTRY[name] = cls


def get_classifier(name: str, **kwargs: Any) -> ShapeClassifier:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown shape classifier '{name}'. Available: {available}"
        )
    return _REGISTRY[name](**kwargs)
