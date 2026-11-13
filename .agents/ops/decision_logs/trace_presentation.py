"""Bounded reader views of typed facts, never interpretation of summary text."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from worker_logs.safety import Guard

from .presentation import evidence_link, prose
from .schema import timestamp
from .turns import Turn, statistics

EXECUTION = {
    "ended": "終了確認",
    "running": "稼働中（観測時点）",
    "not_started": "起動前拒否",
    "unknown": "終了未確認",
}
FINAL = {
    "present": "最終報告あり",
    "absent": "最終報告なし（終了・欠落の根拠あり）",
    "unknown": "最終報告の有無は未確認",
}
PUBLICATION = {
    "main": "main保存の観測（照合時点）",
    "public_pr": "観測時点の公開PR",
    "pending": "公開保留（観測時点）",
    "unknown": "公開の所在は未確認",
}
TERMINATION = {
    "ended": "担当終了",
    "handoff_pending": "引継ぎ待ち",
    "handoff_accepted": "引継ぎ受領",
}


def url_link(url: str, label: str) -> str:
    Guard().check(url)
    return f"[{prose(label)}]({quote(url, safe='/:#?=&%+@,;~.-_')})"


def proofs(values: list[dict[str, Any]], limit: int = 4) -> str:
    lines = [evidence_link(e) for e in values[:limit]]
    if len(values) > limit:
        lines.append(f"ほか{len(values) - limit}件の根拠は記録本文を参照")
    return "<br>".join(lines)


def publication_revisions(value: dict[str, Any], repo: str) -> list[str]:
    return [
        url_link(f"https://github.com/{repo}/commit/{value[key]}", label)
        for key, label in (
            ("pr_head", "PR HEAD"),
            ("main_revision", "照合先main（観測時点）"),
        )
        if value[key] is not None
    ]


def turn_summary(items: list[Turn]) -> str:
    n = statistics(items)
    return (
        f"公開workerログ: {n['published']} ターン（分母）。"
        f"対応イベントあり: {n['matched']} ターン / 対応イベントなし: {n['unmatched']} ターン。"
        f"イベントのみ: {n['event_only']} ターン。和集合: {n['total']} ターン。"
        f"最終報告なし（終了・欠落の根拠あり）: {n['missing_final']} ターン。"
        f"稼働中: {n['running']} ターン。終了未確認: {n['end_unknown']} ターン。"
        f"最終報告あり・公開保留: {n['held_final']} ターン。"
    )


def turn_fields(item: Turn) -> tuple[str, str, str, str]:
    final = FINAL[item.final]
    if item.final == "present" and item.publication in {"pending", "public_pr"}:
        final = "最終報告あり・公開保留"
    if item.conflicting:
        final = "最終報告の観測が矛盾・要照合"
    public = PUBLICATION[item.publication]
    if item.directory is not None:
        public = "main / このHEADに公開3ファイルを確認"
    detail = item.publication_detail
    if item.directory is None and item.publication == "main":
        public = "main保存の観測あり（このHEADの3ファイルは未確認）"
    if detail:
        if detail["pr_url"]:
            public += "<br>" + url_link(detail["pr_url"], "公開PR")
        for revision in publication_revisions(detail, item.key[0]):
            public += "<br>" + revision
        if item.directory is None:
            for key, label in (("stage", "段階"), ("error", "理由")):
                if detail[key]:
                    public += "<br>" + label + ": " + prose(detail[key], 120)
    return EXECUTION[item.execution], final, public, proofs(item.evidence)


def observation_view(value: dict[str, Any], repo: str, *, brief: bool = False) -> str:
    if value["kind"] == "capture_gap":
        return (
            "要照合: "
            + prose(value["reason"], 500)
            + "<br>"
            + proofs(value["evidence"])
        )
    if value["kind"] == "worker_turn":
        lines = [
            EXECUTION[value["execution"]["state"]],
            FINAL[value["final"]["state"]],
            PUBLICATION[value["publication"]["state"]],
        ]
        if value["publication"]["pr_url"]:
            lines.append(url_link(value["publication"]["pr_url"], "公開PR"))
        lines.extend(publication_revisions(value["publication"], repo))
        return " / ".join(lines)
    lines = [
        TERMINATION[value["event"]]
        + f"（担当世代 {value['generation']} / 観測時点 {prose(value['observed_at'])}）",
        "最後の親の受入（明示記録時点）: " + prose(value["acceptance"]["state"]),
    ]
    if brief:
        return "<br>".join(lines)
    if value["main_revision"]:
        lines.append(
            url_link(
                f"https://github.com/{repo}/commit/{value['main_revision']}",
                "終了時の照合先main",
            )
        )
    lines += [
        proofs(value["acceptance"]["evidence"]),
        "終了・引継ぎの根拠: " + proofs(value["evidence"]),
    ]
    if value["remaining_status"] == "unknown":
        lines.append("公開残件の照合: 未確認（空配列を残件なしと扱わない）")
    else:
        lines.append(f"公開残件（観測時点）: {len(value['remaining'])} 件")
        for pending in value["remaining"][:6]:
            description = pending["kind"] + ": " + prose(pending["state"])
            if pending["kind"] == "worker_log":
                description += f" / 担当issue {pending['issue']} / {prose(pending['public_key'])} / turn {pending['sequence']}"
            else:
                description += f" / 未公開 {pending['unpublished_count']} 記録"
            for key in ("stage", "error"):
                if pending[key]:
                    description += " / " + prose(pending[key], 120)
            if pending["pr_number"]:
                description += " / " + url_link(
                    f"https://github.com/{repo}/pull/{pending['pr_number']}",
                    f"PR #{pending['pr_number']}",
                )
            lines.append(description)
        if len(value["remaining"]) > 6:
            lines.append("残りの公開作業は記録本文を参照")
    lines += [
        "公開の担当: "
        + (
            prose(value["publication_actor"])
            if value["publication_actor"]
            else "未確認"
        ),
        "再開先: "
        + (prose(value["resume_command"]) if value["resume_command"] else "未記録"),
    ]
    return "<br>".join(lines)


def latest_termination(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    choices = [
        r
        for r in rows
        if r["kind"] == "result"
        and r["data"].get("observation", {}).get("kind") == "termination"
    ]
    return max(
        choices,
        key=lambda r: (
            r["data"]["observation"]["generation"],
            timestamp(r["data"]["observation"]["observed_at"]),
            timestamp(r["time"]["recorded_at"]),
            r["id"],
        ),
        default=None,
    )
