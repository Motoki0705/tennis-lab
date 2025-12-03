from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TennisLabelRow:
    file_name: str
    visibility: int
    x: float
    y: float
    status: int
    score: float


def load_label_csv(path: str | Path) -> list[TennisLabelRow]:
    p = Path(path)
    rows: list[TennisLabelRow] = []
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        header = [c.strip() for c in (reader.fieldnames or [])]
        base_header = [
            "file name",
            "visibility",
            "x-coordinate",
            "y-coordinate",
            "status",
        ]
        extended_header = base_header + ["score"]

        if header == base_header:
            has_score = False
        elif header == extended_header:
            has_score = True
        else:
            raise ValueError(f"Unexpected Label.csv header at {p}: {reader.fieldnames}")

        for line in reader:
            file_name = line["file name"].strip()
            visibility = int(line["visibility"]) if line["visibility"] != "" else 0
            status = int(line["status"]) if line["status"] != "" else 0
            x = float(line["x-coordinate"]) if line["x-coordinate"] != "" else 0.0
            y = float(line["y-coordinate"]) if line["y-coordinate"] != "" else 0.0
            if has_score:
                score_raw = line["score"]
                score = float(score_raw) if score_raw != "" else 0.0
            else:
                score = 0.0
            if visibility not in (0, 1, 2):
                raise ValueError(
                    f"visibility must be 0, 1, or 2, got {visibility} at {p}"
                )
            rows.append(TennisLabelRow(file_name, visibility, x, y, status, score))
    return rows


def save_label_csv(path: str | Path, rows: Iterable[TennisLabelRow]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file name",
        "visibility",
        "x-coordinate",
        "y-coordinate",
        "status",
        "score",
    ]
    sorted_rows = sorted(rows, key=lambda r: r.file_name)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_rows:
            if r.visibility not in (0, 1, 2):
                raise ValueError(f"Invalid visibility {r.visibility} for {r.file_name}")
            writer.writerow(
                {
                    "file name": r.file_name,
                    "visibility": int(r.visibility),
                    "x-coordinate": float(r.x),
                    "y-coordinate": float(r.y),
                    "status": int(r.status),
                    "score": float(r.score),
                }
            )


def make_empty_row(file_name: str) -> TennisLabelRow:
    return TennisLabelRow(
        file_name=file_name, visibility=0, x=0.0, y=0.0, status=0, score=0.0
    )


def row_from_detection(
    file_name: str, x: float, y: float, score: float
) -> TennisLabelRow:
    return TennisLabelRow(
        file_name=file_name, visibility=1, x=x, y=y, status=0, score=score
    )
