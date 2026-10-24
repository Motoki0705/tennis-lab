"""The small set of fields and views owned by the progress projection."""

from __future__ import annotations

from typing import Any

OWNER = "Motoki0705"
PROJECT_NUMBER = 3
STAGES = ("未着手", "着手可能", "作業中", "レビュー中", "受入待ち", "完了")
ACCEPTANCE = ("未確認", "受入待ち", "受入済み")
FIELDS = {
    "段階": STAGES,
    "待ち理由": (),
    "次の作業": (),
    "親の受入状態": ACCEPTANCE,
}
BUILTINS = (
    "Title",
    "Assignees",
    "Parent issue",
    "Sub-issues progress",
    "Linked pull requests",
    "Priority",
)
VIEWS = (
    {
        "name": "研究開発全体の一覧",
        "layout": "TABLE_LAYOUT",
        "filter": 'is:issue has:"親の受入状態"',
    },
    {
        "name": "子タスクのカンバン",
        "layout": "BOARD_LAYOUT",
        "filter": 'is:issue label:issue:child -reason:"not planned"',
    },
    {
        "name": "判断/ブロック一覧",
        "layout": "TABLE_LAYOUT",
        "filter": 'is:issue has:"待ち理由"',
    },
)
EVENTS = ("initial", "refresh", "start", "pr", "stop", "resume", "merge", "accept")
SUMMARY_PREFIX = "\n進捗（自動集計）: "


def named(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["name"]
        if name in result:
            raise ValueError(f"duplicate Project name: {name}")
        result[name] = row
    return result


def validate_fields(
    rows: list[dict[str, Any]], *, allow_missing: bool = False
) -> dict[str, dict[str, Any]]:
    fields = named(rows)
    for name in (*BUILTINS, "Status"):
        if name not in fields:
            raise ValueError(f"missing standard Project field: {name}")
    for name, options in FIELDS.items():
        field = fields.get(name)
        if field is None:
            if allow_missing:
                continue
            raise ValueError(f"missing Project field: {name}; run project-setup")
        expected = "SINGLE_SELECT" if options else "TEXT"
        if field["dataType"] != expected:
            raise ValueError(f"{name}: expected {expected} field")
        if options and tuple(o["name"] for o in field["options"]) != options:
            raise ValueError(f"{name}: options/order differ from {options}")
    return fields


def manual_steps(views: list[dict[str, Any]]) -> list[str]:
    board = named(views).get("子タスクのカンバン")
    if board is None:
        return []
    steps = []
    if board["columns"] != ["段階"]:
        steps.append(
            "子タスクのカンバン: View → Column field → 段階。6列を表示して保存する。"
        )
    if board["groups"] != ["Parent issue"]:
        steps.append(
            "子タスクのカンバン: View → Group by → Parent issue を選び保存する。"
        )
    return steps
