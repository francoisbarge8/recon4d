"""Configuration helpers: nested dataclasses to and from dictionaries / YAML.

Every stage of the pipeline is configured by a plain dataclass. This module converts trees
of such dataclasses to dictionaries, builds them back with validation of the keys, and
applies command-line overrides of the form ``section.field=value``.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def to_dict(config: Any) -> Any:
    """Convert a (nested) dataclass to plain dictionaries, lists and scalars."""
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return {f.name: to_dict(getattr(config, f.name)) for f in dataclasses.fields(config)}
    if isinstance(config, (list, tuple)):
        return [to_dict(v) for v in config]
    if isinstance(config, dict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


def _candidates(hint: Any) -> list[Any]:
    """The concrete alternatives of a type hint (unwrapping ``X | None``)."""
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return [h for h in typing.get_args(hint) if h is not type(None)]
    return [hint]


def _coerce(value: Any, hint: Any) -> Any:
    if value is None:
        return None
    for candidate in _candidates(hint):
        if dataclasses.is_dataclass(candidate) and isinstance(value, dict):
            return from_dict(candidate, value)
        is_tuple = typing.get_origin(candidate) is tuple or candidate is tuple
        if is_tuple and isinstance(value, (list, tuple)):
            return tuple(value)
        if candidate is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
    return value


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build the dataclass ``cls`` from a dictionary, recursing into nested dataclasses.

    Missing keys take their default; unknown keys raise, so that a typo in a configuration
    file cannot be silently ignored.
    """
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown option(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**{name: _coerce(value, hints[name]) for name, value in data.items()})


def apply_overrides(config: T, overrides: list[str]) -> T:
    """Return a copy of ``config`` with ``"a.b.c=value"`` overrides applied.

    Values are parsed as YAML, so ``train.iterations=3000``, ``dynamic=false`` and
    ``train.crop=[96,72]`` all work.
    """
    data = to_dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override {override!r} is not of the form key=value")
        key, raw = override.split("=", 1)
        node = data
        *parents, leaf = key.strip().split(".")
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"unknown configuration section {part!r} in {key!r}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown configuration option {key!r}")
        node[leaf] = yaml.safe_load(raw)
    return from_dict(type(config), data)


def load_yaml(path: str | Path, cls: type[T]) -> T:
    """Read a configuration file; an empty file gives the defaults."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_dict(cls, data)


def save_yaml(path: str | Path, config: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_dict(config), sort_keys=False), encoding="utf-8")
