"""Small strict JSON validators for the user reference contract."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def fields(value: Any, names: str, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"{where}: expected fields {names}")
    return value


def string(value: Any, where: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{where}: expected a string")
    return value


def integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: expected an integer >= {minimum}")
    return value


def number(value: Any, where: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{where}: expected a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{where}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where}: expected a finite number")
    return result


def digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{where}: expected a lowercase SHA-256")
    return value


def array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected an array")
    return value


def strings(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a string mapping")
    return {string(k, where): string(v, where, empty=True) for k, v in value.items()}


def freeze(value: Any) -> Any:
    """Keep all metadata while preventing mutation of nested source records."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value
