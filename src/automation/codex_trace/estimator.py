"""Explicit, model-independent token estimates for trace sub-segments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

TOKEN_METHOD = "compact_json_utf8_bytes_div_4"


@dataclass(frozen=True)
class TokenEstimate:
    """Estimated token count plus the exact serialized byte count behind it."""

    tokens: int
    byte_count: int
    method: str = TOKEN_METHOD


def compact_json(value: Any) -> str:
    """Serialize JSON-like input deterministically without ASCII expansion."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def estimate_value(value: Any) -> TokenEstimate:
    """Estimate tokens from compact JSON bytes.

    This intentionally does not claim tokenizer accuracy. Provider totals remain
    authoritative; the estimate is only used to attribute those totals.
    """

    encoded = compact_json(value).encode("utf-8")
    return TokenEstimate(tokens=math.ceil(len(encoded) / 4), byte_count=len(encoded))


def estimate_text(text: str) -> TokenEstimate:
    """Estimate tokens from raw UTF-8 text without JSON quote overhead."""

    encoded = text.encode("utf-8")
    return TokenEstimate(tokens=math.ceil(len(encoded) / 4), byte_count=len(encoded))
