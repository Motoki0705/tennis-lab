"""Assemble the report issue (title + Japanese markdown body) from collected data
and the agent's analysis."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .agent import AgentReport, StalePrNote
from .audit import AREAS, Audit
from .proposals import (
    PROPOSALS_BEGIN,
    PROPOSALS_END,
    STALE_PR_OPTIONS,
    STALE_PR_SECTION,
    Proposal,
    neutralize,
    render_proposals,
    report_marker,
    slot_tag,
)

# GitHub rejects issue bodies above 65536 characters.
MAX_BODY_CHARS = 60_000


def report_title(day: date) -> str:
    return f"運用レポート {day.isoformat()}"


def stale_prs(collected: dict[str, Any]) -> list[dict[str, Any]]:
    return [pr for pr in collected["github"]["open_prs"] if pr["stale"]]


def stale_pr_proposals(
    collected: dict[str, Any], notes: dict[int, StalePrNote]
) -> list[Proposal]:
    props = []
    for pr in stale_prs(collected):
        if int(pr["number"]) not in notes:
            continue  # Unconfirmed stale PRs are listed as audit gaps instead.
        draft = "・draft" if pr["draft"] else ""
        detail = (
            f"`{pr['branch']}`（{pr['author']}{draft}）/ 最終更新 {pr['updated_at'][:10]}"
            f"（{pr['idle_days']}日前）"
        )
        note = notes[int(pr["number"])]
        detail += f"\n推奨: **{note.recommendation}** — {note.reason}"
        props.append(
            Proposal(
                section=STALE_PR_SECTION,
                title=f"#{pr['number']} {pr['title']}",
                detail=detail,
                evidence=f"親の放置PR確認 / github.open_prs[#{pr['number']}].detail / base={collected['meta']['base_sha']}",
                options=STALE_PR_OPTIONS,
                purpose=f"放置PR #{pr['number']} の対応方針を決める。",
                scope=f"PR #{pr['number']} と関連issue。採用した対応の詳細は担当親と確認する。",
                expected_effect="放置状態を解消し、継続する作業と終了する作業を明確にする。",
                acceptance=("選択した対応の結果をPRと関連issueに記録できる。",),
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


def _memory_status(collected: dict[str, Any]) -> list[str]:
    dedicated = collected["ops_memory"]
    lines = [
        f"- 専用memory: `origin/main` @ `{dedicated['base_sha']}`（最後の有効版）"
        if dedicated["status"] == "ok"
        else f"- 専用memory: {dedicated['note']}",
    ]
    for item in dedicated.get("persistence", []):
        if item["state"] in {"merged", "unchanged"}:
            continue
        detail = neutralize(str(item["error"] or item["stage"] or item["state"]))
        next_step = (
            "この更新は再適用せず、次回調査で新しい更新を作る。"
            if item["state"] == "failed"
            else "次回report/triageで再確認し、失敗・中断はREADMEの復旧手順に従う。"
        )
        lines.append(
            f"- **専用メモリ未反映**: 予定回 {item['slot']} / {item['pr_url'] or item['report_url'] or '公開未確認'} / {detail}。"
            f"上記main版を使用。{next_step}"
        )
    return lines


def render_report(
    *,
    slot: date,
    agent_name: str,
    collected: dict[str, Any],
    analysis: AgentReport,
    audit: Audit,
) -> tuple[str, str, list[str]]:
    """Return ``(title, body, proposal_ids)``."""
    day = slot
    tag = slot_tag(day)
    meta = collected["meta"]
    proposals = (
        stale_pr_proposals(collected, analysis.stale_pr_notes) + analysis.proposals
    )
    proposal_md, ids = render_proposals(proposals, tag)
    marker = report_marker(
        slot=tag,
        since=datetime.fromisoformat(meta["period_start"]),
        until=datetime.fromisoformat(meta["period_end"]),
        proposals=len(ids),
        agent=agent_name,
        base=meta["base_sha"],
        status="complete" if audit.complete else "incomplete",
        audit=collected["audit_input_sha256"],
    )
    fetched = (
        f"実行時に `{meta['base_fetched_from']}` をfetch"
        if meta["base_fetched_from"]
        else "ローカルref、fetchなし"
    )
    header = [
        marker,
        *(
            []
            if audit.complete
            else [
                "**未完了** — 確認済み結果のみを部分公開。予定回は未成功、専用メモリ更新は保留。",
                "",
            ]
        ),
        f"# {report_title(day)}",
        "",
        f"- 予定回: {day.isoformat()} 08:00 JST",
        f"- 生成: {meta['generated_at']} / agent: `{agent_name}` / 対象: `{meta['base_ref']}`"
        f" @ `{meta['base_sha'][:10]}`（{fetched}） / 集計期間: {meta['period_start']} 〜 {meta['period_end']}",
    ]
    if meta["checkout_diff_files_total"]:
        header.append(
            f"- **注意**: 呼び出し元checkoutは `{meta['checkout_branch']}` @ `{meta['checkout_sha']}`"
            f"（未commitの変更 {meta['checkout_modified_files']}件）で、`{meta['base_ref']}` と"
            f" {meta['checkout_diff_files_total']}ファイルの内容が異なる。"
            "調査担当は全員、上記baseの専用read-only checkoutを読んだ。",
        )
    header += [
        *_memory_status(collected),
        f"- 調査対象一覧: `{collected['audit_plan']['inventory_sha256']}` / 入力: `{collected['audit_input_sha256']}`",
        (
            "- 使い方: 未完了のため参考公開。完了までは提案にチェックを付けず、不足担当と統合を再開する。"
            if not audit.complete
            else "- 使い方: 方向性を採用する提案にチェックを入れる。選択肢は**1つだけ**選ぶ。"
            "採用は方向性の選択で、要件合意とは別。日次triageが未対応の採用案を整理して作業issueを生成・再利用する。"
            "関連親issueや担当親セッションの手動指定は不要。着手判断・親割当は08:00・20:00 JSTの予定回で行う。既存担当への受領記録は維持する。"
            "見送る場合は未チェックのまま該当提案の「見送り理由」を記入する。未記入は未判断。"
        ),
    ]
    body = "\n".join(
        [
            *header,
            "",
            "## 概要",
            "",
            neutralize(analysis.summary.strip()),
            "",
            "## 調査の分担と統合",
            "",
            _coverage(collected, audit),
            "",
            "## 提案",
            "",
            PROPOSALS_BEGIN,
            "",
            proposal_md.rstrip() if ids else "（提案なし）",
            "",
            PROPOSALS_END,
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


def _coverage(collected: dict[str, Any], audit: Audit) -> str:
    lines = (
        [
            "外部submodule内部は未調査（対象外）。このrepoで固定するgitlinkの版と利用箇所を調査する:",
            *[
                f"- `{link['path']}` @ `{link['sha']}`"
                for link in collected["audit_plan"]["gitlinks"]
            ],
            "",
        ]
        if collected["audit_plan"]["gitlinks"]
        else []
    )
    for assignment in collected["audit_plan"]["assignments"]:
        key = assignment["id"]
        result = audit.results.get(key)
        lines += [
            f"### {key}: {AREAS[key]}",
            "",
            f"対象: {len(assignment['targets'])}ファイル / 入力: {', '.join(assignment['inputs'])}",
        ]
        if result:
            lines += [
                neutralize(
                    f"- {result['status']} / scout {result['agent_id']} / {result['started_at']} 〜 {result['finished_at']}"
                ),
                neutralize(result["conclusion"]),
                *[neutralize(f"- 根拠: {e}") for e in result["evidence"]],
                *[neutralize(f"- 反証: {e}") for e in result["counterevidence"]],
                *[
                    neutralize(
                        f"- 未調査: {e['scope']} / 理由: {e['reason']} / 再開: {e['resume']}"
                    )
                    for e in result["unconfirmed"]
                ],
            ]
        else:
            lines += ["未調査: 有効な返却なし。再開時にこの担当を再起動する。"]
        lines += [""]
    for decision in audit.decisions:
        lines += [
            neutralize(
                f"- 統合 {', '.join(decision['sources'])}: {decision['outcome']} / {decision['reason']} / 根拠: {decision['evidence']}"
            )
        ]
    if not audit.complete:
        lines += [
            "",
            "### 未調査範囲・失敗理由・再開方法",
            "",
            *[neutralize(f"- {error}") for error in audit.errors],
            "",
            "次回起動または `report --if-due` で同じ予定回を再開する。固定入力と確認済み結果を再利用し、不足担当と親の統合を補ってこのissueを更新する。",
            "部分公開中の提案は参考情報。完了までチェックを付けない（triageは未完了レポートを処理しない）。",
        ]
    return "\n".join(lines)
