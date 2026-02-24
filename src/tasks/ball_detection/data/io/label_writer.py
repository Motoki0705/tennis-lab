"""Write merged annotations to WASB-compatible Label.csv."""

from __future__ import annotations

from pathlib import Path

from src.tasks.ball_detection.data.io.path_policy import ensure_writable
from src.tasks.ball_detection.data.type import LabelRecord, PathPolicy
from src.tasks.wasb.tennis_format import TennisLabelRow, save_label_csv


def write_label_csv(path: str | Path, rows: list[LabelRecord], *, policy: PathPolicy) -> None:
    """Write labels atomically through a temporary file under policy guards."""
    dst = Path(path)
    ensure_writable(dst, policy=policy)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    tmp_rows = [
        TennisLabelRow(
            file_name=r.file_name,
            visibility=int(r.visibility),
            x=float(r.x),
            y=float(r.y),
            status=int(r.status),
            score=float(r.score),
        )
        for r in rows
    ]
    save_label_csv(tmp_path, tmp_rows)
    tmp_path.replace(dst)
