"""Annotation reader adapting WASB Label.csv to internal typed records."""

from __future__ import annotations

from pathlib import Path

from src.ball_detection.data.type import LabelRecord
from src.wasb.tennis_format import load_label_csv


def read_label_csv(path: str | Path) -> list[LabelRecord]:
    """Read WASB-compatible `Label.csv` into typed `LabelRecord` objects."""
    rows = load_label_csv(path)
    return [
        LabelRecord(
            file_name=r.file_name,
            visibility=r.visibility if r.visibility in (0, 1, 2) else 0,
            x=float(r.x),
            y=float(r.y),
            status=int(r.status),
            score=float(r.score),
        )
        for r in rows
    ]
