"""Write pseudo-label processing metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.io.path_policy import ensure_writable
from src.tasks.ball_detection.data.type import PathPolicy


def write_metadata_json(path: str | Path, payload: dict[str, Any], *, policy: PathPolicy) -> None:
    """Write metadata as JSON with atomic rename."""
    dst = Path(path)
    ensure_writable(dst, policy=policy)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    tmp_path.replace(dst)
