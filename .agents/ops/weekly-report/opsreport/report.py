"""Assemble the report issue (title + Japanese markdown body) from collected data
and the agent's analysis."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .agent import AgentReport
from .proposals import (
    STALE_PR_OPTIONS,
    STALE_PR_SECTION,
    Proposal,
    render_proposals,
    week_tag,
)

# GitHub rejects issue bodies above 65536 characters.
MAX_BODY_CHARS = 60_000


def report_title(day: date) -> str:
    iso = day.isocalendar()
    return f"週次運用レポート {iso.year}-W{iso.week:02d}"


def stale_prs(collected: dict[str, Any]) -> list[dict[str, Any]]:
    return [pr for pr in collected["github"]["open_prs"] if pr["stale"]]


def stale_pr_proposals(
    collected: dict[str, Any], notes: dict[int, str]
) -> list[Proposal]:
    props = []
    for pr in stale_prs(collected):
        draft = "・draft" if pr["draft"] else ""
        detail = (
            f"`{pr['branch']}`（{pr['author']}{draft}）/ 最終更新 {pr['updated_at'][:10]}"
            f"（{pr['idle_days']}日前）"
        )
        note = notes.get(int(pr["number"]))
        if note:
            detail += f"\n推奨: {note}"
        props.append(
            Proposal(
                section=STALE_PR_SECTION,
                title=f"#{pr['number']} {pr['title']}",
                detail=detail,
                options=STALE_PR_OPTIONS,
            )
        )
    return props


def _status_line(section: dict[str, Any], ok_text: str) -> str:
    if section["status"] == "ok":
        return ok_text
    return f"**{section['note']}**"


def _observations(c: dict[str, Any]) -> str:
    gh, git, ws = c["github"], c["git"], c["workspace"]
    ci = gh["ci_runs_in_period"]
    cleanup, memory, knowledge = c["cleanup"], c["memory"], c["knowledge"]
    lines = [
        f"- 期間内のcommit（`{c['meta']['base_ref']}`）: {git['commits_in_period']}件"
        f"（+{git['lines_added']} / -{git['lines_removed']}行）、merge済みPR {len(gh['merged_prs_in_period'])}件",
        f"- open PR: {len(gh['open_prs'])}件（うち{gh['stale_pr_days']}日以上更新なし"
        f" {sum(1 for p in gh['open_prs'] if p['stale'])}件）",
        f"- open issue: {gh['open_issues']['total']}件（`ai-proposed` {len(gh['open_issues']['ai_proposed'])}件）",
        f"- CI（期間内 {ci['total']} run）: "
        + (", ".join(f"{k} {v}" for k, v in ci["by_conclusion"].items()) or "なし"),
        f"- ローカル作業領域: worktree {ws['worktrees']}個、branch {ws['local_branches']}本"
        f"（base にmerge済み {ws['local_branches_merged_into_base']}本）",
        "- 掃除モジュール: "
        + _status_line(cleanup, f"`{cleanup.get('command')}` の結果を分析に使用"),
        "- 共有memory: "
        + _status_line(
            memory, f"{memory.get('entries_total')}件のエントリを分析に使用"
        ),
        "- knowledge: "
        + _status_line(
            knowledge,
            f"期間内に更新されたノード/レポート {knowledge.get('changed_in_period_total')}件",
        ),
        f"- コード規模: {git['code_files_total']}ファイル / {git['code_lines_total']}行、"
        f"TODO系マーカー {git['todo_markers_total']}件",
    ]
    largest = "\n".join(
        f"| `{f['path']}` | {f['lines']} |" for f in git["largest_code_files"][:10]
    )
    failures = "\n".join(
        f"- {f['workflow']} (`{f['branch']}`, {f['created_at'][:10]}) {f['url']}"
        for f in ci["failures"]
    )
    detail = [
        "<details><summary>行数の多いコードファイル（上位10）</summary>",
        "",
        "| path | lines |",
        "|---|---|",
        largest,
        "",
        "</details>",
    ]
    if failures:
        detail += [
            "",
            "<details><summary>期間内のCI失敗</summary>",
            "",
            failures,
            "",
            "</details>",
        ]
    return "\n".join(lines + [""] + detail)


def render_report(
    *,
    now: datetime,
    agent_name: str,
    collected: dict[str, Any],
    analysis: AgentReport,
) -> tuple[str, str, list[str]]:
    """Return ``(title, body, proposal_ids)``."""
    day = now.date()
    tag = week_tag(day)
    meta = collected["meta"]
    proposals = (
        stale_pr_proposals(collected, analysis.stale_pr_notes) + analysis.proposals
    )
    proposal_md, ids = render_proposals(proposals, tag)
    header = [
        f"<!-- ops-report:v1 week={tag} agent={agent_name} base={meta['base_sha']} -->",
        f"# {report_title(day)}",
        "",
        f"- 生成: {meta['generated_at']} / agent: `{agent_name}` / 対象: `{meta['base_ref']}`"
        f" @ `{meta['base_sha']}` / 期間: {meta['period_start'][:10]} 〜 {meta['generated_at'][:10]}",
    ]
    if not meta["checkout_matches_base"] or meta["checkout_modified_files"]:
        header.append(
            f"- **注意**: agentが読んだcheckoutは `{meta['checkout_branch']}` @ `{meta['checkout_sha']}`"
            f"（変更ファイル {meta['checkout_modified_files']}件）で、`{meta['base_ref']}` と一致しない。"
        )
    header += [
        "- 使い方: 子issue化したい提案にチェックを入れる。選択肢つきの提案は選択肢を**1つだけ**チェックする。"
        "日次triageが `ai-proposed` の子issueを作成し、該当行の末尾に `→ #番号` を追記する。"
        "不要な提案は放置でよい（このレポートをcloseするとtriage対象外になる）。",
    ]
    body = "\n".join(
        [
            *header,
            "",
            "## 概要",
            "",
            analysis.summary.strip(),
            "",
            "## 提案",
            "",
            proposal_md.rstrip() if ids else "（提案なし）",
            "",
            "## 観察データ（決定的に収集）",
            "",
            _observations(collected),
            "",
        ]
    )
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(f"report body is {len(body)} chars, above {MAX_BODY_CHARS}")
    return report_title(day), body, ids
