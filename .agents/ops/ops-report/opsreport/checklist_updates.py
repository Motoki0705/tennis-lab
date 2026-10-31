"""Journaled checklist writes with post-PATCH edit-history verification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from issues import (
    CHILD_LABEL,
    TASKS_HEADING,
    Issue,
    parse_proposal_source,
    parse_task_metadata,
    validate_body,
)
from issues.content_edits import (
    BodyHistory,
    Checkpoint,
    intervening_edits,
    read_body_history,
)
from parent.github import GitHub
from parent.policy import replace_section


def validate_recovery(tx: dict[str, Any]) -> None:
    entries = tx.get("checklist_recovery", {})
    if not isinstance(entries, dict) or (tx["complete"] and entries):
        raise ValueError("invalid checklist recovery journal")
    for key, entry in entries.items():
        if (
            not isinstance(key, str)
            or not key.isascii()
            or not key.isdecimal()
            or int(key) < 1
            or not isinstance(entry, dict)
            or set(entry)
            != {"checkpoint", "written_body", "reapply_body", "observed_edit"}
        ):
            raise ValueError("invalid checklist recovery entry")
        point = entry["checkpoint"]
        if (
            not isinstance(point, dict)
            or set(point) != {"body", "edit_id", "edit_count", "created_at"}
            or not isinstance(point["body"], str)
            or not isinstance(point["created_at"], str)
            or type(point["edit_count"]) is not int
            or point["edit_count"] < 0
            or ((point["edit_id"] is None) != (point["edit_count"] == 0))
            or (point["edit_id"] is not None and not isinstance(point["edit_id"], str))
        ):
            raise ValueError("invalid checklist recovery checkpoint")
        if (
            not isinstance(entry["written_body"], str)
            or any(
                value is not None and not isinstance(value, str)
                for value in (entry["reapply_body"], entry["observed_edit"])
            )
            or ((entry["reapply_body"] is None) != (entry["observed_edit"] is None))
        ):
            raise ValueError("invalid checklist recovery body")


def _validate_body(parent: Issue, tx: dict[str, Any], body: str) -> None:
    errors = validate_body(body, is_parent=True, is_child=CHILD_LABEL in parent.labels)
    if errors:
        raise ValueError(f"#{parent.number}: invalid checklist recovery body: {errors}")
    for node in tx["plan"]["nodes"]:
        if node["role"] == "parent" and tx["numbers"].get(node["key"]) == parent.number:
            original = tx["drafts"][node["key"]]["body"]
            if parse_task_metadata(body) != parse_task_metadata(
                original
            ) or parse_proposal_source(body) != parse_proposal_source(original):
                raise ValueError("checklist recovery must preserve task provenance")


def _draft_body(tx: dict[str, Any], parent: int, body: str) -> None:
    for node in tx["plan"]["nodes"]:
        if node["role"] == "parent" and tx["numbers"].get(node["key"]) == parent:
            tx["drafts"][node["key"]]["body"] = body


def update_checklist(
    gh: GitHub,
    parent: Issue,
    tasks: str,
    tx: dict[str, Any],
    save: Callable[[], None],
    events: list[dict[str, Any]],
) -> None:
    """Apply only the checklist section; at most two PATCHes per invocation."""
    n = parent.number
    recovery = tx.setdefault("checklist_recovery", {})
    entry = recovery.get(str(n))
    if not entry and replace_section(parent.body, TASKS_HEADING, tasks) == parent.body:
        return

    def event(attempt: int, state: str, message: str) -> None:
        events.append(
            {"issue": n, "attempt": attempt, "state": state, "message": message}
        )

    def finish(history: BodyHistory) -> None:
        _draft_body(tx, n, history.checkpoint.body)
        recovery.pop(str(n), None)
        save()

    def check_written(
        record: dict[str, Any], history: BodyHistory, attempt: int
    ) -> bool:
        others = intervening_edits(
            Checkpoint(**record["checkpoint"]), history, record["written_body"]
        )
        if not others:
            finish(history)
            return True
        # The newest foreign revision survives in history even if our PATCH overwrote it.
        latest = others[0].body
        _validate_body(parent, tx, latest)
        record.update(reapply_body=latest, observed_edit=history.checkpoint.edit_id)
        _draft_body(tx, n, history.checkpoint.body)
        save()
        event(
            attempt,
            "interrupted",
            f"#{n}: 本文更新への割り込みをuserContentEditsで検出。"
            + (
                "再適用中の割り込みのため、再試行上限で停止する。"
                if attempt == 2
                else "最新の他編集へチェックリスト節を1回だけ再適用する。"
            ),
        )
        return False

    if entry and entry["reapply_body"] is None:
        history = read_body_history(gh, n)
        if asdict(history.checkpoint) == entry["checkpoint"]:
            event(
                0,
                "not_applied",
                f"#{n}: 前回PATCHが未反映であることを履歴で確認し、同じjournalから再開する。",
            )
            recovery.pop(str(n))
            save()
        elif check_written(entry, history, 0):
            event(
                0,
                "verified",
                f"#{n}: 前回PATCHを履歴で検証し、同じjournalから再開した。",
            )
        entry = recovery.get(str(n))

    for attempt in (1, 2):
        before = read_body_history(gh, n)
        entry = recovery.get(str(n))
        if not entry and attempt == 1 and before.checkpoint.body != parent.body:
            # A previously uncertain PATCH may have just been verified above.
            if (
                replace_section(before.checkpoint.body, TASKS_HEADING, tasks)
                == before.checkpoint.body
            ):
                return
            raise ValueError(
                f"#{n}: body changed before checklist PATCH; reconcile and retry triage"
            )
        source = (
            entry["reapply_body"]
            if entry
            and entry["reapply_body"] is not None
            and entry["observed_edit"] == before.checkpoint.edit_id
            else before.checkpoint.body
        )
        body = replace_section(source, TASKS_HEADING, tasks)
        _validate_body(parent, tx, body)
        if body == before.checkpoint.body:
            finish(before)
            return
        record = {
            "checkpoint": asdict(before.checkpoint),
            "written_body": body,
            "reapply_body": None,
            "observed_edit": None,
        }
        recovery[str(n)] = record
        save()  # Retain the checkpoint even if PATCH or the history response is lost.
        gh.write("PATCH", f"repos/{gh.repo}/issues/{n}", {"body": body})
        after = read_body_history(gh, n)
        if check_written(record, after, attempt):
            if attempt == 2 or entry:
                event(
                    attempt,
                    "reapplied",
                    f"#{n}: 最新の他編集へチェックリスト節を再適用し、PATCH後の履歴検査に成功した。",
                )
            return
        if attempt == 2:
            event(
                attempt,
                "retry_exhausted",
                f"#{n}: 再適用中にも割り込みが発生。復旧対象の本文とcheckpointをjournalへ保存して停止する。",
            )
            raise ValueError(
                f"#{n}: checklist PATCH interrupted twice; retry triage with the saved recovery journal"
            )
