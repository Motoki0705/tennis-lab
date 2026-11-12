"""Canonical end-of-task candidates and human decisions, independent of runners.

GitHub comments are the business record. Labels, operational receipts and the
decision journal are projections, never substitutes for a human decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .format import _scan_markdown, parse_sections, raw_preamble, raw_section

EVENT_PREFIX = "<!-- parent-improvement:v1 "
SOURCE_PREFIX = "<!-- parent-improvement-source:"
PREAMBLE_KEY = "_preamble"
PROGRESS = frozenset(
    {
        "子タスクのチェックリスト",
        "進捗概要",
        "成果物・関連PR",
        "次の改善候補",
        "終了記録",
    }
)
SPEC_TEXT = frozenset(
    {
        "purpose",
        "observation",
        "evidence",
        "scope",
        "out_of_scope",
        "behavior",
        "stop_conditions",
        "judgment",
        "unknowns",
        "expected_effect",
    }
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def fields(value: Any, keys: set[str] | frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"improvement requires exactly {sorted(keys)}")
    return value


def text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("improvement requires nonempty text")
    return value


def positive(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("improvement requires a positive integer")
    return value


def key(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,63}", value
    ):
        raise ValueError("improvement key must be a stable lowercase token")
    return value


def candidate_key(value: str, parent: int) -> str:
    text(value)
    prefix = f"I-{parent}-"
    if value.startswith("I-"):
        if not value.startswith(prefix):
            raise ValueError("candidate belongs to a different source parent")
        value = value[len(prefix) :]
    return key(value)


def _sha(value: Any, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid improvement event reference")


@dataclass(frozen=True)
class ImprovementSource:
    parent: int
    key: str

    def __post_init__(self) -> None:
        positive(self.parent)
        key(self.key)

    @property
    def candidate_id(self) -> str:
        return f"I-{self.parent}-{self.key}"

    def marker(self) -> str:
        return (
            f"<!-- parent-improvement-source:v1 parent={self.parent} key={self.key} -->"
        )


def parse_source(body: str) -> ImprovementSource | None:
    _, markers = _scan_markdown(body, prefix=SOURCE_PREFIX)
    if not markers:
        return None
    position, line = markers[0]
    match = re.fullmatch(
        r"<!-- parent-improvement-source:v1 parent=([1-9][0-9]*) key=([a-z0-9][a-z0-9-]{0,63}) -->",
        line,
    )
    if len(markers) != 1 or position != 0 or match is None:
        raise ValueError("improvement source must occur exactly once on the first line")
    return ImprovementSource(int(match[1]), match[2])


def index_sources(rows: Iterable[tuple[int, str]]) -> dict[ImprovementSource, int]:
    result: dict[ImprovementSource, int] = {}
    for n, body in rows:
        source = parse_source(body)
        if source is not None:
            if source in result:
                raise ValueError(f"duplicate improvement source {source.candidate_id}")
            result[source] = positive(n)
    return result


def contract(body: str) -> dict[str, str]:
    """Only known progress and whitespace changes are presentation changes.

    Preamble, unknown sections, links, numbers, exclusions and stop conditions remain
    protected. This is not a natural-language equivalence classifier.
    """
    sections = parse_sections(body)
    if PREAMBLE_KEY in sections:
        raise ValueError("reserved approval contract heading: " + PREAMBLE_KEY)

    def normalize(content: str) -> str:
        # Keep fenced examples byte-sensitive: indentation/quoted spaces can be
        # executable requirements. Never drop them via the visible-line parser.
        parts: list[str] = []
        prose: list[str] = []
        fence: str | None = None
        for line in content.replace("\r\n", "\n").splitlines():
            marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if fence is not None or marker:
                if prose:
                    parts.append(re.sub(r"\s+", " ", "\n".join(prose)).strip())
                    prose = []
                parts.append(line)
                if fence and re.fullmatch(
                    rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*", line
                ):
                    fence = None
                elif fence is None and marker:
                    fence = marker[1]
            else:
                prose.append(re.sub(r"^(\s*[-*] \[)[xX](\])", r"\1 \2", line))
        if prose:
            parts.append(re.sub(r"\s+", " ", "\n".join(prose)).strip())
        return "\n".join(parts).strip()

    result = {
        heading: normalize(raw_section(body, heading))
        for heading in sections
        if heading not in PROGRESS
    }
    preamble = raw_preamble(body)
    if parse_source(body) is not None:
        # Only the source marker is metadata. Other text, including supplement
        # details and fenced examples, remains part of the approved requirements.
        preamble = preamble.partition("\n")[2]
    normalized = normalize(preamble)
    if normalized:
        result[PREAMBLE_KEY] = normalized
    return result


def execution_terms(requirements: Mapping[str, str]) -> dict[str, str]:
    """Project the approved snapshot, never the proposal's former scope.

    Additional protected sections and the preamble remain execution constraints;
    their meaning must not be guessed or lost while adapting to agreement fields.
    """
    sections = {
        "purpose": "目的・背景",
        "scope": "全体のスコープ",
        "behavior": "全体方針・合意事項",
        "acceptance": "全体の完了条件",
    }
    for heading in (*sections.values(), "依存関係・進め方"):
        content = requirements.get(heading)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"approval snapshot requires a nonempty section: {heading}"
            )
    terms = {field: requirements[heading] for field, heading in sections.items()}
    for heading in sorted(set(requirements) - set(sections.values())):
        label = "承認本文の前文" if heading == PREAMBLE_KEY else heading
        terms["behavior"] += f"\n\n## {label}\n\n{requirements[heading]}"
    return terms


def validate_spec(value: Any) -> dict[str, Any]:
    spec = fields(
        value, SPEC_TEXT | {"schema_version", "acceptance", "unmet_acceptance"}
    )
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("unknown improvement spec version")
    for field in SPEC_TEXT:
        text(spec[field])
    if not isinstance(spec["acceptance"], list) or not spec["acceptance"]:
        raise ValueError("improvement requires acceptance criteria")
    for item in spec["acceptance"]:
        text(item)
    if len(set(spec["acceptance"])) != len(spec["acceptance"]):
        raise ValueError("duplicate improvement acceptance criterion")
    if not isinstance(spec["unmet_acceptance"], list):
        raise ValueError("unmet_acceptance must be a list")
    for item in spec["unmet_acceptance"]:
        key(item)
    if len(set(spec["unmet_acceptance"])) != len(spec["unmet_acceptance"]):
        raise ValueError("duplicate unmet acceptance reference")
    return spec


def confirmation(value: Any) -> dict[str, Any]:
    result = fields(value, {"quote", "source"})
    text(result["quote"])
    text(result["source"])
    return result


def event(
    source: ImprovementSource,
    kind: str,
    revision: int,
    previous: str | None,
    data: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "parent": source.parent,
        "key": source.key,
        "event": kind,
        "revision": revision,
        "previous": previous,
        "data": data,
    }
    value["id"] = fingerprint(value)
    validate_event(value)
    return value


def validate_event(value: Any) -> dict[str, Any]:
    row = fields(
        value,
        {
            "schema_version",
            "parent",
            "key",
            "event",
            "revision",
            "previous",
            "data",
            "id",
        },
    )
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise ValueError("unknown improvement event version")
    ImprovementSource(row["parent"], row["key"])
    positive(row["revision"])
    _sha(row["previous"], nullable=True)
    if fingerprint({k: v for k, v in row.items() if k != "id"}) != row["id"]:
        raise ValueError("improvement event digest mismatch")
    if row["event"] == "candidate":
        data = fields(row["data"], {"source_issue", "spec", "legacy"})
        positive(data["source_issue"])
        if type(data["legacy"]) is not bool:
            raise ValueError("invalid candidate discovery kind")
        validate_spec(data["spec"])
    elif row["event"] == "published":
        data = fields(row["data"], {"issue", "mode", "draft_sha256"})
        positive(data["issue"])
        if data["mode"] not in ("created", "reference"):
            raise ValueError("invalid improvement publication mode")
        _sha(data["draft_sha256"])
    elif row["event"] == "decision":
        data = fields(
            row["data"],
            {
                "request_key",
                "request_sha256",
                "action",
                "issue",
                "human_confirmation",
                "reason",
                "requirements",
                "previous_decision",
            },
        )
        key(data["request_key"])
        _sha(data["request_sha256"])
        positive(data["issue"])
        if data["action"] not in ("approve", "hold", "decline", "revoke"):
            raise ValueError("invalid human improvement decision")
        confirmation(data["human_confirmation"])
        text(data["reason"])
        _sha(data["previous_decision"], nullable=True)
        if not isinstance(data["requirements"], dict) or not data["requirements"]:
            raise ValueError("decision requires the observed target requirements")
        for heading, content in data["requirements"].items():
            text(heading)
            text(content)
    else:
        raise ValueError("unknown improvement event")
    return row


def render_event(row: dict[str, Any]) -> str:
    validate_event(row)
    encoded = (
        json.dumps(row, ensure_ascii=True, sort_keys=True)
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
    )
    source = ImprovementSource(row["parent"], row["key"])
    data = row["data"]
    title = f"改善候補 {source.candidate_id} / 版 {row['revision']}"
    if row["event"] == "candidate":
        spec = data["spec"]
        lines = [
            title,
            f"発見元: #{data['source_issue']}",
            *[f"{k}: {spec[k]}" for k in sorted(SPEC_TEXT)],
            "受入条件:\n" + "\n".join(f"- {v}" for v in spec["acceptance"]),
            "残る必須条件: " + (", ".join(spec["unmet_acceptance"]) or "なし。"),
            "人間の実行判断: 未判断。この記録は実行承認ではありません。",
        ]
    elif row["event"] == "published":
        lines = [
            title,
            f"後継・集約先: #{data['issue']} ({data['mode']})",
            "起票・参照は実行開始ではありません。",
        ]
    else:
        lines = [
            title,
            f"人間の判断: {data['action']} / 対象 #{data['issue']}",
            f"理由: {data['reason']}",
            "人間の元発言:\n"
            + "\n".join(
                "> " + s for s in data["human_confirmation"]["quote"].splitlines()
            ),
            f"出典: {data['human_confirmation']['source']}",
            "実行状態は既存の担当・worker・成果記録を確認します。",
        ]
    return EVENT_PREFIX + encoded + " -->\n\n" + "\n\n".join(lines)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) != len({k for k, _ in pairs}):
        raise ValueError("duplicate improvement JSON key")
    return dict(pairs)


def read_candidates(
    comments: Iterable[Mapping[str, Any]], parent: int, publisher: str
) -> dict[str, dict[str, Any]]:
    """Validate a single append chain; identical POST replays are one event."""
    result: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for comment in sorted(comments, key=lambda c: positive(c.get("id"))):
        body = comment.get("body") or ""
        first = body.splitlines()[0] if body else ""
        if comment.get("user", {}).get("login") != publisher or not first.startswith(
            "<!-- parent-improvement:"
        ):
            continue
        if not first.startswith(EVENT_PREFIX) or not first.endswith(" -->"):
            raise ValueError("unknown or malformed improvement comment")
        row = validate_event(
            json.loads(first[len(EVENT_PREFIX) : -4], object_pairs_hook=_unique)
        )
        if row["parent"] != parent or render_event(row) != body:
            raise ValueError("improvement comment does not match its canonical record")
        if row["id"] in seen:
            continue
        candidate = result.get(row["key"])
        if row["previous"] != (candidate["last"]["id"] if candidate else None):
            raise ValueError(
                "improvement history has a missing event or conflicting branch"
            )
        if row["event"] == "candidate":
            if row["revision"] != (candidate["revision"] + 1 if candidate else 1):
                raise ValueError("candidate revision must advance once")
            if (
                candidate
                and row["data"]["source_issue"]
                != candidate["candidate"]["data"]["source_issue"]
            ):
                raise ValueError("candidate source issue changed")
            candidate = {
                **(candidate or {"publication": None, "decisions": []}),
                "revision": row["revision"],
                "candidate": row,
            }
            result[row["key"]] = candidate
        else:
            if candidate is None or row["revision"] != candidate["revision"]:
                raise ValueError("event has no matching candidate revision")
            if row["event"] == "published":
                if candidate["publication"] is not None:
                    raise ValueError("candidate cannot change its published target")
                candidate["publication"] = row
            else:
                pub = candidate["publication"]
                decisions = candidate["decisions"]
                if pub is None or row["data"]["issue"] != pub["data"]["issue"]:
                    raise ValueError("human decision has a different proposal target")
                if row["data"]["previous_decision"] != (
                    decisions[-1]["id"] if decisions else None
                ):
                    raise ValueError("human decision supersedes an unknown decision")
                if any(
                    d["data"]["request_key"] == row["data"]["request_key"]
                    for d in decisions
                ):
                    raise ValueError("conflicting human decision request")
                decisions.append(row)
        candidate["last"] = row
        candidate.setdefault("comments", {})[row["id"]] = positive(comment["id"])
        seen.add(row["id"])
    return result


def human_state(candidate: Mapping[str, Any]) -> str:
    decisions = candidate["decisions"]
    if not decisions or decisions[-1]["revision"] != candidate["revision"]:
        return "pending"
    return {
        "approve": "approved",
        "hold": "held",
        "decline": "declined",
        "revoke": "revoked",
    }[decisions[-1]["data"]["action"]]


def approved(
    candidate: Mapping[str, Any], source: ImprovementSource, issue: int, body: str
) -> dict[str, Any]:
    publication = candidate["publication"]
    if human_state(candidate) != "approved":
        raise ValueError(
            f"{source.candidate_id}: human execution approval is {human_state(candidate)}"
        )
    decision = candidate["decisions"][-1]
    if (
        publication is None
        or publication["data"]["issue"] != issue
        or publication["data"]["mode"] != "created"
        or decision["data"]["issue"] != issue
    ):
        raise ValueError(
            f"{source.candidate_id}: approval belongs to a different issue"
        )
    if candidate["candidate"]["data"]["spec"]["unmet_acceptance"]:
        raise ValueError(
            f"{source.candidate_id}: unfinished original acceptance is not a successor task"
        )
    if decision["data"]["requirements"] != contract(body):
        raise ValueError(
            f"{source.candidate_id}: requirements changed; inspect scope and record the human decision for the revised proposal"
        )
    return dict(decision)
