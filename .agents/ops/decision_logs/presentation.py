"""Bounded, escaped reader fields for immutable captured records.

Some v1 adapters stored structured values inside the summary string. Decode only
literal containers in that display field; never reinterpret them as new facts.
The original string and every omitted field remain in the linked envelope.
"""

from __future__ import annotations

import ast
import html
import json
import re
from typing import Any
from urllib.parse import quote

from worker_logs.safety import Guard

LABELS = {
    "code": "コード",
    "requirements": "要件",
    "instruction": "指示",
    "conditions": "実行条件",
    "verified_scope": "確認済み範囲",
    "child": "担当issue",
    "parent": "親issue",
    "prs": "対象PR",
    "PRs": "対象PR",
    "pr": "対象PR",
    "exit_code": "終了値",
    "status": "状態",
    "state": "状態",
    "stage": "段階",
    "error": "エラー",
    "summary": "要約",
    "reason": "理由",
    "reasons": "理由",
    "review_state": "レビュー状態",
    "coverage": "coverage",
    "covered": "確認範囲の充足",
    "unresolved": "未解決",
    "progress": "進捗",
    "branch": "branch",
    "agent": "agent",
    "action": "操作",
    "delegated": "委任",
    "saved_to_main": "main保存確認",
    "turn": "ターン",
    "verdict": "判定",
    "exception": "例外",
    "previous_review_keys": "前回レビュー参照",
    "decision": "判断",
    "model": "モデル",
    "effort": "effort",
    "source": "根拠",
    "selected_model": "選択モデル",
    "effective_model": "実効モデル",
    "elapsed_seconds": "実行秒数",
    "wait_seconds": "待機秒数",
    "usage": "使用量",
    "specified_by": "指定元",
    "basis": "集計範囲",
    "counter_key": "counter参照",
    "input_tokens": "入力tokens",
    "cached_input_tokens": "キャッシュ入力tokens",
    "output_tokens": "出力tokens",
    "total_tokens": "合計tokens",
    "occurred_at": "発生時刻",
    "received_at": "受信時刻",
    "recorded_at": "記録時刻",
}
CHANGES = {"changed": "変更あり", "unchanged": "変更なし", "unconfirmed": "未確認"}
NOT_FINAL = "not a final-report observation"


def prose(value: Any, limit: int = 1400) -> str:
    """Captured text cannot insert Markdown, HTML, anchors or table rows."""
    value = str(value)
    checker = Guard()
    checker.check(value)
    if len(value) > limit:
        value = value[:limit] + "…（表示省略・記録本文を参照）"
        # Cutting text can create a credential/private-identifier boundary.
        checker.check(value)
    escaped = "".join(
        f"&#{ord(char)};" if char in "\\`*_{}[]()!#|" else html.escape(char, quote=True)
        for char in value
    )
    return escaped.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")


def checked_unescape(value: str) -> str:
    """Recheck entity-decoded display text before any clipping or re-escaping."""
    value = html.unescape(value)
    Guard().check(value)
    return value


def field_label(key: Any) -> str:
    # A mapping key alone may be safe while adding ': ' makes it a sensitive
    # assignment. Check both its original name and its translated display label.
    Guard().check(f"{key}: ")
    return prose(f"{LABELS.get(key, key)}: ", 80)


def evidence_link(item: dict[str, Any]) -> str:
    Guard().check(item)
    if item["kind"] == "unavailable":
        return "根拠非公開・記録不足: " + prose(item["reason"], 300)
    target = quote(item["url"], safe="/:#?=&%+@,;~.-_")
    suffix = "固定版" if item["revision"] else "内容識別あり・全文復元は未保証"
    return f"[根拠]({target})（{suffix}）"


def fields(value: Any, *, depth: int = 0, links: bool = True) -> str:
    """Render explicit nested fields without Python/JSON container notation."""
    if depth > 5:
        return "表示省略・記録本文を参照"
    if isinstance(value, dict):
        if (
            links
            and set(value) == {"status", "reason"}
            and value.get("status") == "unavailable"
        ):
            return "未取得: " + prose(value["reason"], 200)
        if (
            links
            and set(value) == {"status", "value"}
            and value.get("status") == "observed"
        ):
            return "観測: " + fields(value["value"], depth=depth + 1, links=links)
        if links and (
            value.get("kind") == "public"
            and set(value) == {"kind", "url", "revision", "content_sha256"}
            or value.get("kind") == "unavailable"
            and set(value) == {"kind", "reason"}
        ):
            return evidence_link(value)
        entries = list(value.items())
        return (
            " / ".join(
                field_label(k) + fields(v, depth=depth + 1, links=links)
                for k, v in entries[:12]
            )
            + (" / 表示省略・記録本文を参照" if len(entries) > 12 else "")
            if entries
            else "項目なし"
        )
    if isinstance(value, (list, tuple)):
        return (
            "、".join(fields(v, depth=depth + 1, links=links) for v in value[:12])
            + ("、表示省略・記録本文を参照" if len(value) > 12 else "")
            if value
            else "なし（空の一覧）"
        )
    if value is None:
        return "記録値なし"
    if type(value) is bool:
        return "はい" if value else "いいえ"
    return prose(value, 400)


def literal(value: str) -> tuple[bool, Any]:
    """Only bounded literal data; no code, expressions or unrestricted parsing."""
    if not value or value[0] not in "{[(" or len(value) > 16000:
        return False, None
    try:
        parsed = json.loads(value)
    except (ValueError, RecursionError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
            return False, None
    # Keep this outside the parse-error handlers: a public-check failure must
    # propagate its fixed error code instead of becoming an unparsed fallback.
    Guard().check(parsed)
    return isinstance(parsed, (dict, list, tuple)), parsed


def summary(value: str, *, compact: bool = False) -> str:
    """Recognize literal tails and key=value fields in legacy adapter summaries."""
    output = []
    for part in value.split("; "):
        part = part.strip()
        decoded, data = literal(part)
        if decoded:
            output.append(fields(data, links=False))
            continue
        match = re.fullmatch(r"([A-Za-z_][\w.-]*)=(.*)", part, re.DOTALL)
        if match:
            key, raw = match.groups()
            decoded, data = literal(raw)
            if decoded:
                output.append(field_label(key) + fields(data, links=False))
            elif raw in {"None", "True", "False"}:
                output.append(
                    field_label(key)
                    + fields({"None": None, "True": True, "False": False}[raw])
                )
            elif raw.startswith(("{", "[", "(")):
                output.append(
                    field_label(key) + "構造を解釈できない要約・記録本文を参照"
                )
            else:
                output.append(field_label(key) + prose(raw))
        elif part.startswith(("{", "[", "(")):
            output.append("構造を解釈できない要約・記録本文を参照")
        else:
            tail = re.fullmatch(r"(.*?)([\[{].*)", part, re.DOTALL)
            decoded, data = literal(tail[2]) if tail else (False, None)
            if decoded and tail:
                output.append(prose(tail[1]) + fields(data, links=False))
            elif tail and tail[2].startswith("{"):
                output.append(prose(tail[1]) + "構造を解釈できない要約・記録本文を参照")
            else:
                output.append(prose(part, 160 if compact else 1400))
    if compact:
        # Compact index cells have a fixed text budget regardless of the size
        # of a captured literal. Escape once more after flattening generated
        # field markup, so clipping never cuts an entity or an active link.
        return prose(
            checked_unescape(" / ".join(output[:3])).replace("<br>", " / "), 200
        )
    return "<br>".join(output)


def change_fields(data: dict[str, Any]) -> str:
    return " / ".join(f"{LABELS[k]}: {CHANGES[v]}" for k, v in data.items())


def final_report(data: dict[str, Any]) -> str:
    item = data["final_report"]
    if item["status"] == "observed":
        return "最終報告あり: " + evidence_link(item["value"])
    if item["reason"] == NOT_FINAL:
        return "最終報告の観測対象外（記録された理由による）"
    return "記録不足・最終報告: " + prose(item["reason"], 200)
