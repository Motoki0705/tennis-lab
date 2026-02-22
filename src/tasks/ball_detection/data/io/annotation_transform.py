"""Pure transforms for annotation coordinates and frame alignment."""

from __future__ import annotations

from src.ball_detection.data.type import LabelRecord


def clamp_label_to_image(label: LabelRecord, *, width: int, height: int) -> LabelRecord:
    """Clamp coordinates into image bounds."""
    x = min(max(label.x, 0.0), float(max(width - 1, 0)))
    y = min(max(label.y, 0.0), float(max(height - 1, 0)))
    return LabelRecord(
        file_name=label.file_name,
        visibility=label.visibility,
        x=x,
        y=y,
        status=label.status,
        score=label.score,
    )
