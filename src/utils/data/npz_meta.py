"""Utilities for decoding NPZ metadata payloads."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def decode_meta(meta_raw: Any) -> dict[str, Any]:
    """Decode a meta payload from NPZ into a dictionary.

    Args:
        meta_raw: Raw meta field from numpy (bytes, str, dict, or scalar).

    Returns:
        Decoded metadata dictionary.
    """
    if hasattr(meta_raw, "item"):
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8")
    if isinstance(meta_raw, str):
        try:
            meta_raw = json.loads(meta_raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(meta_raw, np.generic):
        return {}
    return meta_raw if isinstance(meta_raw, dict) else {}
