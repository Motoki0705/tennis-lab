"""Assemble the Japanese report from an explicitly ended audit ledger."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime
from typing import Any

from .handoff import evidence_text, excerpt, finding_proposals, issue_number
from .ledger import Ledger, LedgerError
from .proposals import PROPOSALS_BEGIN, PROPOSALS_END, render_proposals, report_marker

MAX_BODY_CHARS = 60_000


def report_title(day: date) -> str:
    return f"運用レポート {day.isoformat()}"


def render_ledger_report(
    *,
    ledger: Ledger,
    agent_name: str,
    since: datetime,
    until: datetime,
    input_sha256: str,
    ledger_reference: str,
    limit: int = 5,
) -> tuple[str, str, list[str]]:
    """Render one explicitly ended ledger; no fixed-area analysis is consumed.

    Prepare proposal mappings before end(). ``ledger_reference`` is a caller's
    public reference to the archived original, never a local Ledger.reference.
    The CLI switches to this entry point in the audit integration change.
    """
    state = ledger.snapshot
    if state["run"]["status"] == "running":
        raise LedgerError("end the ledger with a reason before rendering")
    if type(limit) is not int or not 1 <= limit <= 20:
        raise LedgerError("report summary limit must be between 1 and 20")
    if not ledger_reference.strip():
        raise LedgerError("a public reference to the original ledger is required")
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise LedgerError("input_sha256 must identify the pinned audit input")
    if since.tzinfo is None or until.tzinfo is None or since > until:
        raise LedgerError(
            "report collection period requires ordered timezone-aware times"
        )
    entries = finding_proposals(ledger)
    proposal_md, ids = render_proposals([e.proposal for e in entries], state["slot"])
    complete = state["run"]["status"] == "completed"
    day = date.fromisoformat(state["slot"])
    ending = state["run"]["endings"][-1]
    lines = [
        report_marker(
            slot=state["slot"],
            since=since,
            until=until,
            proposals=len(ids),
            agent=agent_name,
            base=state["base_sha"],
            status="complete" if complete else "incomplete",
            audit=input_sha256,
        ),
        *(
            []
            if complete
            else [
                "**未完了** — 確認済み結果のみを部分公開。提案へのチェックと専用memory更新は完了まで保留。",
                "",
            ]
        ),
        f"# {report_title(day)}",
        "",
        f"- 予定回: {day.isoformat()} 08:00 JST / agent: `{agent_name}`",
        f"- 調査基準版: `{state['base_sha']}` / 集計期間: {since.isoformat()} 〜 {until.isoformat()}",
        f"- 台帳原本: {excerpt(ledger_reference, 1000)}。全件の詳細・根拠・反証・履歴は原本のIDを参照。",
        "- 探索結果は正式レビュー・merge承認の証拠にはしない。",
        "",
        "## 調査回の終了と範囲",
        "",
        f"- 終了状態: {state['run']['status']} / 終了: {ending['finished_at']}",
        f"- 終了理由: {excerpt(ending['reason'])}",
        "- 外部submodule内部は対象外・未調査。gitlinkの版・利用箇所の確認範囲は台帳原本を参照。",
    ]
    for label, values in (
        ("確認範囲", ending["checked_scope"]),
        ("除外範囲", ending["excluded_scope"]),
        ("追加調査が必要な理由", ending["further_investigation"]),
    ):
        lines += _bounded_list(label, values, limit)
    gaps = [
        f"{gap['scope']} / 理由: {gap['reason']} / 再開: {gap['resume']}"
        for gap in ending["unverified_risks"]
    ]
    gaps += [
        f"{task['id']}: {gap['scope']} / 理由: {gap['reason']} / 再開: {gap['resume']}"
        for task in state["tasks"].values()
        for gap in task["unconfirmed"]
    ]
    lines += _bounded_list("重要な未検証リスク・未確認範囲", gaps, limit)
    lines += ["", "## 調査タスクと起点の連鎖", ""]
    tasks = list(state["tasks"].values())
    counts = Counter(t["status"] for t in tasks)
    origins = Counter(t["origin"]["kind"] for t in tasks)
    lines += [
        f"- タスク: {len(tasks)}件 / "
        + (
            ", ".join(f"{key} {value}件" for key, value in sorted(counts.items()))
            or "なし"
        ),
        "- 起点: "
        + (
            ", ".join(f"{key} {value}件" for key, value in sorted(origins.items()))
            or "なし"
        ),
    ]
    for task in tasks[-limit:]:
        lines.append(
            f"- {excerpt(_origin_chain(state, task), 500)} / {task['status']}: {excerpt(task['question'])}"
        )
    if len(tasks) > limit:
        lines.append(
            f"- 連鎖は末尾{limit}件の要約。残り{len(tasks) - limit}件は台帳原本を参照。"
        )
    lines += ["", "## 状態別の発見", ""]
    findings = list(state["findings"].values())
    for status in ("confirmed", "suspected", "unverified", "refuted"):
        group = [f for f in findings if f["status"] == status]
        lines += [f"### {status}（{len(group)}件）", ""]
        for finding in group[:limit]:
            lines += [
                f"- {excerpt(finding['id'])} / {excerpt(finding['kind'])} / タスク {excerpt(finding['task_id'])}: {excerpt(finding['conclusion'])}",
                f"  - 根拠: {evidence_text(state, finding['evidence'], 2)}",
            ]
            if finding["counterevidence"]:
                lines.append(
                    f"  - 反証: {evidence_text(state, finding['counterevidence'], 2)}"
                )
            if finding["kind"] in {"bug", "vulnerability"}:
                details = finding["details"]
                lines.append(
                    f"  - 対象版: {excerpt(details['target_version']) or '未確認'} / "
                    f"発生条件: {excerpt(' / '.join(details['conditions'])) or '未確認'} / "
                    f"到達可能性: {excerpt(details['reachability']) or '未確認'} / "
                    f"影響: {excerpt(details['impact']) or '未確認'}"
                )
                lines += _usage_lines(state, details["usage_paths"])
        if len(group) > limit:
            lines.append(f"- 残り{len(group) - limit}件は台帳原本を参照。")
        lines.append("")
    lines += ["## 不要コード候補の判断", ""]
    for decision, label in (
        ("delete-candidate", "削除候補"),
        ("keep", "維持"),
        ("unverified", "未確認"),
    ):
        group = [
            f
            for f in findings
            if f["kind"] == "unused-code"
            and (
                f["details"]["decision"] if f["status"] == "confirmed" else "unverified"
            )
            == decision
        ]
        lines += [f"### {label}（{len(group)}件）", ""]
        for finding in group[:limit]:
            lines.append(
                f"- {excerpt(finding['id'])} / {finding['status']}: {excerpt(finding['conclusion'])}"
            )
            lines += _usage_lines(state, finding["details"]["usage_paths"])
        if len(group) > limit:
            lines.append(f"- 残り{len(group) - limit}件は台帳原本を参照。")
        lines.append("")
    existing = [
        h
        for h in state["handoffs"].values()
        if h["kind"] in {"issue", "pr"}
        and state["findings"][h["finding_id"]]["status"] == "confirmed"
    ]
    lines += ["## 既存の修正先への参照", ""]
    for route in existing[:limit]:
        number = issue_number(route["kind"], route["reference"])
        lines.append(
            f"- 発見 {excerpt(route['finding_id'])} → {route['kind']} #{number}"
            f"（{excerpt(route['reference'])}） / 理由: {excerpt(route['reason'])}。新規提案にはしない。"
        )
    if not existing:
        lines.append("（既存issue/PRへの引渡しなし）")
    if len(existing) > limit:
        lines.append(f"- 残り{len(existing) - limit}件は台帳原本を参照。")
    lines += [
        "",
        "## 提案",
        "",
        "チェックは方向性の採用。要件合意・正式レビュー・merge gateは既存の手順で確認する。",
        "関連親付きの採用案は既存triageの受領経路へ渡し、親未指定の案は未割当とする。",
        PROPOSALS_BEGIN,
        "",
        proposal_md.rstrip() if ids else "（提案なし）",
        "",
        PROPOSALS_END,
        "",
    ]
    body = "\n".join(lines)
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(f"report body is {len(body)} chars, above {MAX_BODY_CHARS}")
    return report_title(day), body, ids


def _bounded_list(label: str, values: list[str], limit: int) -> list[str]:
    return [
        f"- {label}（{len(values)}件）: "
        + (" / ".join(excerpt(v) for v in values[:limit]) or "なし"),
        *(
            [f"  - 残り{len(values) - limit}件は台帳原本を参照。"]
            if len(values) > limit
            else []
        ),
    ]


def _usage_lines(state: dict[str, Any], paths: list[dict[str, Any]]) -> list[str]:
    return [
        f"  - 利用経路 {excerpt(path['kind'])}: {path['outcome']} / {excerpt(path['reason'])}"
        f" / 根拠: {evidence_text(state, path['evidence'], 1)}"
        for path in paths
    ] or ["  - 利用経路: 未確認（確認記録なし）"]


def _origin_chain(state: dict[str, Any], task: dict[str, Any]) -> str:
    chain = [f"task:{task['id']}"]
    origin = task["origin"]
    # A bounded excerpt of the real chain, not a limit on investigation depth.
    while len(chain) < 8:
        kind, key = origin["kind"], origin["id"]
        chain.append(f"{kind}:{key}")
        if kind == "goal":
            break
        if kind == "finding":
            origin = {"kind": "task", "id": state["findings"][key]["task_id"]}
        else:
            origin = state["tasks"][key]["origin"]
    else:
        chain.append("…（上流は台帳原本参照）")
    return " → ".join(reversed(chain))


def with_memory_status(body: str, dedicated: dict[str, Any]) -> str:
    """Keep persistence failures visible without copying observations into the report."""
    from .proposals import neutralize

    lines = [
        f"- 専用memory: origin/mainの最後の有効版 @ `{dedicated['base_sha']}`"
        if dedicated["status"] == "ok"
        else "- 専用memory: " + neutralize(dedicated["note"]),
    ]
    for item in dedicated.get("persistence", []):
        if item["state"] not in {"merged", "unchanged"}:
            lines.append(
                "- **専用メモリ未反映**: "
                + neutralize(
                    f"予定回 {item['slot']} / {item['state']} / {item['error'] or item['stage']}。次回report/triageで再確認する。"
                )
            )
    body = body.replace(
        "## 調査回の終了と範囲", "\n".join(lines) + "\n\n## 調査回の終了と範囲", 1
    )
    if len(body) > MAX_BODY_CHARS:
        raise ValueError("report body including memory status exceeds the GitHub limit")
    return body
