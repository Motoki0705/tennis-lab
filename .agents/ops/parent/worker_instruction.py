"""The Codex implementation-worker instruction shared by every launch entry."""

from __future__ import annotations

from pathlib import Path

TEMPLATE = Path(__file__).with_name("worker_prompt.md")


def worker_instruction(
    parent: int,
    child: int,
    worktree: Path,
    branch: str,
    issue_body: str,
    parent_instruction: str,
) -> str:
    """Return the standard worker prompt followed by this delegation's specifics.

    Managed (``start-worker`` / ``continue-worker``) and manual (``run-worker``)
    launches both send this text to the common runner, so every worker receives
    the same repository rules whichever entry started its turn.
    """
    return TEMPLATE.read_text(encoding="utf-8") + (
        f"\n\n親issue: #{parent}\n担当issue: #{child}\n"
        f"worktree: {worktree.resolve()}\nbranch: {branch}\n"
        "pytestは共有lockなしで他の親・子と並列実行できる。\n\n"
        f"## 担当issueの原文\n\n{issue_body}\n\n"
        f"## 親の具体的な指示\n\n{parent_instruction}\n"
    )
