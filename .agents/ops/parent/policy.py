"""Decisions and public metadata, without filesystem, git, gh or agent I/O."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from issues import Issue, parse_sections, task_status
from issues.format import visible_lines

PREFIX = "<!-- parent-session:v1 "
PHASES = {
    "assigned": "担当指定のみ",
    "checking": "要件確認中",
    "agreed": "要件合意済み",
    "autonomous": "自律着手根拠あり",
}
REVIEW_STATES = (
    "targets_fixed",
    "instructed",
    "result_received",
    "fixes_verified",
    "complete",
)
REVIEW_LABELS = dict(
    zip(
        REVIEW_STATES,
        ("対象確定", "指示済み", "結果受領", "修正確認", "完了"),
        strict=True,
    )
)
ACTIVE = {"active", "handoff_pending"}


def object_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("expected a JSON object")
    return value


def nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: nonempty text required")
    return value.strip()


def number(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("positive issue/PR number required")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def issue_number(value: str, repo: str) -> int:
    if re.fullmatch(r"#?[1-9][0-9]*", value):
        return int(value.lstrip("#"))
    parsed = urlsplit(value)
    match = re.fullmatch(rf"/{re.escape(repo)}/issues/([1-9][0-9]*)", parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or not match
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "parent must be a number or an issue URL in the selected repository"
        )
    return int(match[1])


def marker(event: Mapping[str, Any]) -> str:
    data = json.dumps(dict(event), ensure_ascii=True, sort_keys=True)
    return PREFIX + data.replace("<", r"\u003c").replace(">", r"\u003e") + " -->"


def events(
    comments: Sequence[Mapping[str, Any]], login: str, parent: int
) -> list[dict[str, Any]]:
    result = []
    for comment in comments:
        if object_value(comment.get("user")).get("login") != login:
            continue
        body = comment.get("body") or ""
        if not isinstance(body, str) or not body:
            continue
        first = body.splitlines()[0]
        if first.startswith("<!-- parent-session:v") and not first.startswith(PREFIX):
            raise ValueError("unknown parent-session marker version")
        if not first.startswith(PREFIX):
            continue
        if not first.endswith(" -->"):
            raise ValueError("malformed parent-session marker")
        event = object_value(json.loads(first[len(PREFIX) : -4]))
        if event.get("parent") != parent:
            raise ValueError("parent-session marker refers to a different parent")
        if event.get("event") in {"agreed", "proposal_agreed"}:
            agreement = object_value(event.get("agreement"))
            if set(agreement) != {
                "purpose",
                "scope",
                "behavior",
                "acceptance",
                "human_confirmation",
            }:
                raise ValueError(
                    "agreement marker lacks requirements/human confirmation"
                )
            for key, value in agreement.items():
                nonempty(value, key)
            if event.get("agreement_sha256") != digest(agreement):
                raise ValueError("agreement marker digest mismatch")
        if "comment_inputs" in event or event.get("event") == "agreement_comments":
            from .agreement_comments import validate_inputs

            validate_inputs(event.get("comment_inputs"))
        if event.get("event") == "agreement_comments":
            from .execution import ignored_comment_map, sha256

            if set(event) != {
                "parent",
                "event",
                "agreement_sha256",
                "revision",
                "classification_revision",
                "comment_inputs",
                "ignored_comments",
            }:
                raise ValueError("malformed agreement comment classification")
            sha256(event.get("agreement_sha256"))
            number(event.get("revision"))
            number(event.get("classification_revision"))
            if not isinstance(event.get("ignored_comments"), list):
                raise ValueError("ignored comments must be a list")
            ignored_comment_map(event["ignored_comments"])
        if event.get("event") == "execution_basis":
            from .execution import validate_event

            validate_event(event)
        if event.get("event") == "basis_revoked":
            number(event.get("generation"))
            number(event.get("revision"))
            nonempty(event.get("reason"), "revocation reason")
        result.append({**event, "body": body})
    return result


def phase(history: Sequence[Mapping[str, Any]]) -> str:
    current = "assigned"
    for event in history:
        kind = event.get("event")
        if kind in {"checking", "agreed"}:
            current = str(kind)
        elif kind == "execution_basis":
            current = "autonomous"
        elif kind == "basis_revoked":
            current = "checking"
    return current


def require_agreement(history: Sequence[Mapping[str, Any]]) -> None:
    if phase(history) != "agreed":
        raise ValueError("親issueの要件合意記録がありません。分割・委任は拒否しました")


def review_state(
    history: Sequence[Mapping[str, Any]], request: Mapping[str, Any]
) -> str | None:
    current: str | None = None
    for event in history:
        if (
            event.get("event") != "review"
            or event.get("review_run_id") != request["review_run_id"]
        ):
            continue
        state = event.get("state")
        if state not in REVIEW_STATES or event.get("request_sha256") != digest(request):
            raise ValueError("review event has an invalid state or request digest")
        expected = 0 if current is None else REVIEW_STATES.index(current) + 1
        if state != current and REVIEW_STATES.index(state) != expected:
            raise ValueError("public review state skips or reverses a transition")
        current = str(state)
    return current


def registration(records: Mapping[str, Any], key: str, session: str) -> None:
    for other_key, row in records.items():
        if row["session"] == session and row["status"] in ACTIVE and other_key != key:
            raise ValueError(
                f"既に担当中: {other_key} ({row['status']})。解除または引継ぎが必要です"
            )
    row = records.get(key)
    if row and row["status"] in ACTIVE:
        if row["session"] != session or row["status"] != "active":
            raise ValueError(
                f"{key} は別セッション担当中または引継ぎ待ち。handoff が必要です"
            )
    elif row and row["session"] != session:
        raise ValueError(
            f"{key} は {row['status']}。明示的な handoff --accept が必要です"
        )


def dependencies(body: str) -> list[int]:
    return sorted(
        {
            int(n)
            for n in re.findall(r"#([1-9][0-9]*)", parse_sections(body)["依存関係"])
        }
    )


def progress(
    children: Mapping[int, Issue],
    tasks: Mapping[str, Any],
    overrides: Mapping[int, Sequence[int]] | None = None,
) -> dict[int, str]:
    deps = {
        n: list((overrides or {}).get(n, dependencies(i.body)))
        for n, i in children.items()
    }
    visiting: set[int] = set()
    result: dict[int, str] = {}

    def visit(n: int) -> str:
        if n in visiting:
            raise ValueError("cyclic child dependencies")
        if n in result:
            return result[n]
        visiting.add(n)
        if any(d not in children for d in deps[n]):
            raise ValueError(
                f"#{n}: dependency outside this parent; provide explicit dependencies"
            )
        upstream = [visit(d) for d in deps[n]]
        state = task_status(children[n])
        if state != "open":
            value = state
        elif child_stopped(
            children[n].body, tasks.get(str(n), {}).get("stopped", False)
        ):
            value = "stopped"
        elif any(s in {"stopped", "held", "cancelled"} for s in upstream):
            value = "held"
        else:
            value = "continue"
        visiting.remove(n)
        result[n] = value
        return str(value)

    for n in children:
        visit(n)
    return result


def closing_issues(body: str, repo: str) -> set[int]:
    # Require a closing instruction, not an incidental issue mention or branch name.
    result = set()
    for match in re.finditer(
        r"\b(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?)\s+(?:(https://github\.com/[^\s]+/issues/)([1-9][0-9]*)|#([1-9][0-9]*))\b",
        "\n".join(visible_lines(body)),
        re.I,
    ):
        if match[1] is None or match[1] == f"https://github.com/{repo}/issues/":
            result.add(int(match[2] or match[3]))
    return result


def select_prs(
    selected: Sequence[int],
    prs: Sequence[Mapping[str, Any]],
    child_numbers: set[int],
    repo: str,
) -> list[Mapping[str, Any]]:
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("review PR set is empty or contains duplicates")
    by_number = {p["number"]: p for p in prs}
    if len(by_number) != len(prs):
        raise ValueError("duplicate PR records from GitHub")
    related = {
        p["number"]
        for p in prs
        if closing_issues(p.get("body") or "", repo) & child_numbers
    }
    expected = {n for n in related if by_number[n]["state"] == "open"}
    for n in selected:
        if n not in by_number or n not in related:
            raise ValueError(f"PR #{n}: missing or unrelated to this parent")
        pr = by_number[n]
        if pr["state"] != "open" or pr.get("draft") is not False:
            raise ValueError(f"PR #{n}: closed/draft or unknown draft state")
        if pr["head"]["repo"]["full_name"].casefold() != repo.casefold():
            raise ValueError(f"PR #{n}: fork is not supported")
    if set(selected) != expected:
        raise ValueError(
            f"PR set has missing/extra entries: missing={sorted(expected - set(selected))}, extra={sorted(set(selected) - expected)}"
        )
    return [by_number[n] for n in sorted(selected)]


def section_span(body: str, heading: str) -> tuple[int, int] | None:
    """Use the shared Markdown parser as authority; refuse ambiguous raw editing."""
    sections = parse_sections(body)
    if heading not in sections:
        return None
    headings: list[tuple[str, re.Match[str]]] = []
    for match in re.finditer(r"(?m)^## (\S.*?)[ \t]*$", body):
        try:
            prefix = parse_sections(body[: match.end()])
        except ValueError as exc:
            if str(exc) in {"unclosed code fence", "unclosed HTML comment"}:
                continue  # Examples/comments do not define an editable section.
            raise
        name = next(reversed(prefix), None)
        if name and (not headings or headings[-1][0] != name):
            headings.append((name, match))
    if [name for name, _ in headings] != list(sections):
        raise ValueError("ambiguous raw Markdown section boundaries")
    index = list(sections).index(heading)
    return headings[index][1].end(), headings[index + 1][1].start() if index + 1 < len(
        headings
    ) else len(body)


def raw_section(body: str, heading: str) -> str:
    span = section_span(body, heading)
    return body[span[0] : span[1]].strip() if span else ""


def replace_section(body: str, heading: str, content: str) -> str:
    """Edit a validated section while preserving metadata, comments and other sections."""
    span = section_span(body, heading)
    if span is None:
        return body.rstrip() + f"\n\n## {heading}\n\n{content.rstrip()}\n"
    start, end = span
    return body[:start] + "\n\n" + content.rstrip() + "\n\n" + body[end:]


def child_stopped(body: str, default: bool = False) -> bool:
    section = raw_section(body, "作業状況・停止事項")
    last = section.splitlines()[-1] if section else ""
    record = re.fullmatch(
        r"<!-- parent-task-status:v1 state=(stopped|resumed) detail=[0-9a-f]{64} -->",
        last,
    )
    return record[1] == "stopped" if record else default
