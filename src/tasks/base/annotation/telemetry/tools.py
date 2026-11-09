"""Read bounded nesting of JSON CLI envelopes in observed shell-tool output."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


def output_envelopes(
    text: str, required: tuple[str, ...], depth: int = 0
) -> Iterator[dict[str, Any]]:
    for line in text.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if all(key in value for key in required):
            yield value
        elif depth < 2:
            for key in ("output", "stdout"):
                if isinstance(value.get(key), str):
                    yield from output_envelopes(value[key], required, depth + 1)
