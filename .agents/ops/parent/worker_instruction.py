"""The Codex implementation-worker instruction shared by every launch entry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).with_name("worker_prompt.md")


@dataclass(frozen=True)
class PreparedInstruction:
    text: str
    standard: bytes


def prepare_instruction(
    parent: int,
    child: int,
    worktree: Path,
    branch: str,
    issue_body: str,
    parent_instruction: str,
    turn: dict[str, Any] | None = None,
) -> PreparedInstruction:
    """Return the standard worker prompt followed by this delegation's specifics.

    Managed (``start-worker`` / ``continue-worker``) and manual (``run-worker``)
    launches both send this text to the common runner, so every worker receives
    the same repository rules whichever entry started its turn.
    """
    phase = ""
    if turn is not None:
        phase = (
            f"\n\n## このターンのフェーズ\n\n{turn['phase']}\n"
            f"turn_id: {turn['turn_id']}\ndecision_id: {turn['decision_id'] or 'なし'}\n"
        )
        if turn["phase"] == "planning":
            phase += "親が今回は調査を選択した。根拠・提案・未解決点を報告する。実装へ進む時点は親の具体指示に従う。\n"
        else:
            phase += "親の具体指示に基づいて作業し、成果と検証・未解決点を報告する。\n"
    standard = TEMPLATE.read_bytes()
    text = (
        standard.decode("utf-8")
        + phase
        + (
            f"\n\n親issue: #{parent}\n担当issue: #{child}\n"
            f"worktree: {worktree.resolve()}\nbranch: {branch}\n"
            "pytestは共有lockなしで他の親・子と並列実行できる。\n\n"
            f"## 担当issueの原文\n\n{issue_body}\n\n"
            f"## 親の具体的な指示\n\n{parent_instruction}\n"
        )
    )
    return PreparedInstruction(text, standard)


def worker_instruction(
    parent: int,
    child: int,
    worktree: Path,
    branch: str,
    issue_body: str,
    parent_instruction: str,
    turn: dict[str, Any] | None = None,
) -> str:
    return prepare_instruction(
        parent, child, worktree, branch, issue_body, parent_instruction, turn
    ).text
