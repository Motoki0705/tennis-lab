"""Decode the bytes whose identities are recorded by ingest."""

import json
import re
from typing import Any


def sha256_value(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label}: expected a lowercase SHA-256")
    return value


def decode_json(raw: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        obj = dict(items)
        if len(obj) != len(items):
            raise ValueError(f"{label}: duplicate JSON key")
        return obj

    def invalid_constant(value: str) -> None:
        raise ValueError(f"{label}: invalid JSON number {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)


def json_object(raw: bytes, label: str) -> dict[str, Any]:
    value = decode_json(raw, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected a JSON object")
    return value
