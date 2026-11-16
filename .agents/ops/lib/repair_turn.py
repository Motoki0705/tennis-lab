"""Explicit repair-turn contracts and strict caller-side final report parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from lib.agent_events import parse_json


def request(repair_id: str, pr: int) -> dict[str, Any]:
    value = {"schema_version": 1, "repair_id": repair_id, "pr": pr}
    validate_request(value)
    return value


def validate_request(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "repair_id", "pr"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["repair_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value["repair_id"])
        or type(value["pr"]) is not int
        or value["pr"] < 1
    ):
        raise ValueError("invalid repair turn request")


def validate_contract(contract: dict[str, Any]) -> None:
    validate_request(contract["repair"])
    settings = contract["settings"]
    if (
        contract.get("mode") != "write"
        or settings.get("agent") != "codex"
        or settings.get("role") != "implementation"
        or settings.get("profile") is not None
        or settings["selected"].get("effort") != "high"
    ):
        raise ValueError("repair turn requires write-mode implementation Codex/high")


REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "repair_id",
        "pr",
        "outcome",
        "head_sha",
        "summary",
        "verification",
        "parent_request",
    ],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "repair_id": {
            "type": "string",
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        },
        "pr": {"type": "integer", "minimum": 1},
        "outcome": {"enum": ["repaired", "needs_parent"]},
        "head_sha": {"type": "string", "pattern": "^[a-f0-9]{40}$", "maxLength": 40},
        "summary": {"type": "string", "pattern": r"\S"},
        "verification": {
            "type": "array",
            "items": {"type": "string", "pattern": r"\S"},
        },
        "parent_request": {"type": ["string", "null"], "pattern": r"\S"},
    },
}


def parse_report(text: str, *, repair_id: str, pr: int) -> dict[str, Any]:
    """Accept one JSON object, with no prose, fences or inferred success."""
    from jsonschema import Draft7Validator  # type: ignore[import-untyped]

    expected = request(repair_id, pr)
    try:
        value = parse_json(text)
        if not Draft7Validator(REPORT_SCHEMA).is_valid(value):
            raise ValueError("report schema mismatch")
        if (
            type(value["schema_version"]) is not int
            or type(value["pr"]) is not int
            or value["repair_id"] != expected["repair_id"]
            or value["pr"] != expected["pr"]
            or (value["outcome"] == "repaired" and value["parent_request"] is not None)
            or (value["outcome"] == "needs_parent" and value["parent_request"] is None)
        ):
            raise ValueError("report identity/outcome mismatch")
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("invalid repair final report") from exc
    return dict(value)


def instruction(repair: dict[str, Any], details: str) -> str:
    validate_request(repair)
    if not isinstance(details, str) or not details.strip():
        raise ValueError("repair requires failure/gate details")
    return (
        "## 自動進行の修正ターン\n\n"
        "このターンは同じモデル・同じsessionのeffort=highで行う修正。\n"
        "失敗情報を調べ、自分で直せるか親AIの判断が必要かを決める。\n"
        "直せる場合は実装・意味のある検証・pre-commit・commit・pushを行い、その直後にターンを終了する。\n"
        "CIの待機は呼出側CLIが担当する。workerはpush後にCIを待たない。\n"
        "親の判断が必要ならoutcomeをneeds_parentにし、parent_requestへ具体的な依頼を書く。\n"
        "最終報告は次のschemaを満たすJSONオブジェクト1つだけにする。\n"
        "repairedのparent_requestはnull、needs_parentでは非空文字列を必須にする。\n"
        "head_shaは報告時の完全な40桁commit。verificationには実施した検証と結果・未実施を記す。\n"
        f"修正の識別情報は {json.dumps(repair, ensure_ascii=False)}\n"
        f"報告schemaは {json.dumps(REPORT_SCHEMA, ensure_ascii=False)}\n\n" + details
    )
