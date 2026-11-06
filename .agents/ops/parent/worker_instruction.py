"""The Codex implementation-worker instruction shared by every launch entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).with_name("worker_prompt.md")


def worker_instruction(
    parent: int,
    child: int,
    worktree: Path,
    branch: str,
    issue_body: str,
    parent_instruction: str,
    turn: dict[str, Any] | None = None,
) -> str:
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
            phase += "調査と設計提案だけを行い、実装・commit・PR作成は行わず、必須6節の最終報告でターンを終える。\n"
        else:
            phase += "親の判断または理由付き省略に基づいて実装・検証・commit・PR作成を進める。\n"
    return (
        TEMPLATE.read_text(encoding="utf-8")
        + phase
        + (
            f"\n\n親issue: #{parent}\n担当issue: #{child}\n"
            f"worktree: {worktree.resolve()}\nbranch: {branch}\n"
            "pytestは共有lockなしで他の親・子と並列実行できる。\n\n"
            f"## 担当issueの原文\n\n{issue_body}\n\n"
            f"## 親の具体的な指示\n\n{parent_instruction}\n"
        )
    )
