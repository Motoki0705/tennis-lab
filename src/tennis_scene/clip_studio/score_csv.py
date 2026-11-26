"""CSV interchange for saved score values, with atomic validation of imports."""

from __future__ import annotations

import csv
import io
from typing import Any

from src.tennis_scene.clip_studio.project import ClipScore, ClipStudioProject
from src.utils.tennis_scoring import TennisRules

COLUMNS = (
    "clip_name",
    "status",
    "sets_a",
    "sets_b",
    "games_a",
    "games_b",
    "points_a",
    "points_b",
    "server",
    "winner",
    "note",
)


def _pair(row: dict[str, str], name: str) -> list[int]:
    values = [row[f"{name}_a"], row[f"{name}_b"]]
    if any(not value.isascii() or not value.isdecimal() for value in values):
        raise ValueError(f"{name}_a / {name}_b に非負整数を指定してください。")
    return [int(value) for value in values]


def import_scores(project: ClipStudioProject, text: str) -> None:
    """Replace only named clips, after all rows pass validation."""
    match = project.match
    if match is None:
        raise ValueError("CSV を取り込む前に試合設定を保存してください。")
    clips = {clip.name: clip for clip in project.clips}
    scores: dict[str, ClipScore] = {}
    reader = csv.reader(
        io.StringIO(text.removeprefix("\ufeff"), newline=""), strict=True
    )
    try:
        if next(reader, None) != list(COLUMNS):
            raise ValueError(
                f"CSV のヘッダーは {','.join(COLUMNS)} を指定してください。"
            )
        for cells in reader:
            if len(cells) != len(COLUMNS):
                raise ValueError(f"{len(COLUMNS)} 列を指定してください。")
            row = dict(zip(COLUMNS, cells, strict=True))
            name = row["clip_name"]
            if name not in clips:
                raise ValueError(f"クリップ {name!r} が見つかりません。")
            if name in scores:
                raise ValueError(f"クリップ {name!r} の行が重複しています。")
            data: dict[str, Any] = {
                "status": row["status"],
                "before_source": "anchor",
                "server": row["server"],
                "server_source": "manual",
                "winner": row["winner"] or None,
                "note": row["note"],
            }
            sets_present = bool(row["sets_a"] or row["sets_b"])
            if row["status"] == "not_point":
                if any(row[column] for column in COLUMNS[2:8]):
                    raise ValueError(
                        "ポイント外の行のセット・ゲーム・ポイント列は空欄にしてください。"
                    )
            else:
                before = {name: _pair(row, name) for name in ("games", "points")}
                if isinstance(match.rules, TennisRules):
                    before["sets"] = _pair(row, "sets")
                elif sets_present:
                    raise ValueError(
                        "ソフトテニスでは sets_a / sets_b を空欄にしてください。"
                    )
                data["before"] = before
            score = ClipScore.from_dict(data)
            score.validate(match)
            scores[name] = score
    except (ValueError, csv.Error) as error:
        raise ValueError(f"CSV {max(reader.line_num, 1)} 行目: {error}") from error
    for name, score in scores.items():
        clips[name].score = score


def export_scores(project: ClipStudioProject) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(COLUMNS)
    for clip in sorted(project.clips, key=lambda item: item.start_sec):
        score = clip.score
        if score is None:
            continue
        before = score.before
        sets = (
            before.sets if before is not None and before.sets is not None else ("", "")
        )
        games = before.games if before is not None else ("", "")
        points = before.points if before is not None else ("", "")
        writer.writerow(
            (
                clip.name,
                score.status,
                *sets,
                *games,
                *points,
                score.server,
                score.winner,
                score.note,
            )
        )
    return output.getvalue()
