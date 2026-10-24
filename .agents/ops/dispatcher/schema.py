"""Validate the complete judgment before allowing any publication or candidate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from parent.policy import (
    child_stopped,
    digest,
    events,
    nonempty,
    number,
    object_value,
    phase,
)

from .collect import is_container, route
from .records import owned_comment

DISPOSITIONS = {
    "startable",
    "information_needed",
    "dependency_wait",
    "existing_owner",
    "existing_parent",
    "out_of_scope",
    "human_hold",
}
REQUIREMENTS = {"purpose", "scope", "behavior", "acceptance"}
DECISION_KEYS = {
    "issue",
    "disposition",
    "target_parent",
    "priority",
    "requirements",
    "rationale",
    "facts",
    "missing_information",
    "resume_condition",
    "next_action",
    "sources",
    "dependencies",
}


class MechanicalGuard(TypedDict):
    disposition: str
    facts: list[str]


def strings(value: Any, field: str, *, empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(
            f"{field} must be a {'possibly empty ' if empty else 'nonempty '}list"
        )
    for item in value:
        nonempty(item, field)
    if len(set(value)) != len(value):
        raise ValueError(f"duplicate {field}")
    return value


def source_url(snapshot: dict[str, Any], source: str) -> str:
    base = f"https://github.com/{snapshot['repo']}"
    if source.startswith("issue:"):
        n = number(int(source.removeprefix("issue:")))
        if str(n) in snapshot["issues"] and source == f"issue:{n}":
            return f"{base}/issues/{n}"
    elif source.startswith("comment:"):
        parts = source.split(":")
        if len(parts) == 3:
            n, cid = number(int(parts[1])), number(int(parts[2]))
            for c in snapshot["issues"].get(str(n), {}).get("comments", []):
                if source == f"comment:{n}:{cid}" and c["id"] == cid:
                    if owned_comment(c, snapshot["publisher"], snapshot["repo"], n):
                        raise ValueError(
                            "controller display cannot serve as independent judgment evidence"
                        )
                    return f"{base}/issues/{n}#issuecomment-{cid}"
    elif source.startswith("pr:"):
        n = number(int(source.removeprefix("pr:")))
        if source == f"pr:{n}" and any(p["number"] == n for p in snapshot["pulls"]):
            return f"{base}/pull/{n}"
    elif source.startswith("code:"):
        _, commit, path = source.split(":", 2)
        if commit == snapshot["code"]["commit"] and any(
            f["path"] == path
            and f["kind"] == "blob"
            and f["mode"] in {"100644", "100755"}
            for f in snapshot["code"]["files"]
        ):
            # Keep an exact git reference, including paths containing URL punctuation.
            return source
    raise ValueError(f"source was not collected: {source}")


def guard(snapshot: dict[str, Any], n: int) -> MechanicalGuard | None:
    """Mechanical vetoes supplement, rather than replace, semantic agent judgment."""
    issues = snapshot["issues"]
    raw = issues[str(n)]
    target = route(issues, n)
    if is_container(raw) or is_container(issues[str(target)]):
        container = n if is_container(raw) else target
        return {
            "disposition": "out_of_scope",
            "facts": [f"#{container} はops-reportの報告容器である。"],
        }
    if raw["parent"] is not None:
        return {
            "disposition": "existing_parent",
            "facts": [
                f"#{n} の正式な親は #{raw['parent']}、最上位の管理先は #{target}。"
            ],
        }
    labels = {x["name"] for x in raw["labels"]}
    if "issue:child" in labels:
        return {
            "disposition": "human_hold",
            "facts": [f"#{n} はissue:childラベルがあるが正式な親関係がない。"],
        }
    history = events(raw["comments"], snapshot["publisher"], n)
    facts = []
    if "needs-human" in labels:
        facts.append(f"#{n} にneeds-humanラベルがある。")
    if child_stopped(raw["body"]):
        facts.append(f"#{n} の本文に子の停止記録がある。")
    current_phase = phase(history)
    if current_phase in {"checking", "agreed"}:
        facts.append(
            f"#{n} のpublisherによるparent-session記録の現在phaseは{current_phase}。"
        )
    if facts:
        return {"disposition": "human_hold", "facts": facts}
    family = {
        k for k in issues if route(issues, int(k)) == n and issues[k]["state"] == "open"
    }
    for k in sorted(family, key=int):
        if issues[k]["assignees"]:
            facts.append(f"#{k} にGitHub assigneeがある。")
    for k, owner in sorted(snapshot["owners"].items(), key=lambda item: int(item[0])):
        if k in family and owner["status"] not in {"released", "ended"}:
            facts.append(
                f"親 #{k} の担当状態は{owner['status']}、generationは{owner['generation']}。"
            )
        for task, info in sorted(owner["tasks"].items(), key=lambda item: int(item[0])):
            if task in family and info["delegated"]:
                facts.append(f"親 #{k} の記録に #{task} の委任予約がある。")
    for pull in snapshot["pulls"]:
        linked = sorted(set(pull["issues"]) & {int(k) for k in family})
        if linked:
            facts.append(
                f"PR #{pull['number']} にissue {linked} のclosing指示がある。"
                f"state={pull['state']}、merged_at={pull['merged_at']}。"
            )
    if facts:
        return {"disposition": "existing_owner", "facts": facts}
    for dep in raw["dependencies"]:
        issue = issues[str(dep)]
        if issue["state"] != "closed" or issue["state_reason"] != "completed":
            facts.append(
                f"先行依存 #{dep} はstate={issue['state']}、"
                f"state_reason={issue['state_reason']}で未充足。"
            )
    if facts:
        return {"disposition": "dependency_wait", "facts": facts}
    return None


def validate(
    value: Any, snapshot: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    data = object_value(value)
    if (
        set(data) != {"schema_version", "input_sha256", "decisions"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["input_sha256"] != snapshot["input_sha256"]
        or not isinstance(data["decisions"], list)
    ):
        raise ValueError("unknown/malformed judgement schema or input digest mismatch")
    decisions = []
    diagnostics = []
    seen: set[int] = set()
    priorities: set[int] = set()
    for raw in data["decisions"]:
        d = object_value(raw)
        if set(d) != DECISION_KEYS:
            raise ValueError("judgement decision has missing/unknown fields")
        n = number(d["issue"])
        if n not in snapshot["subjects"] or n in seen:
            raise ValueError("judgement returned an unknown/duplicate issue")
        seen.add(n)
        if d["disposition"] not in DISPOSITIONS:
            raise ValueError("unknown judgement disposition")
        issue = snapshot["issues"][str(n)]
        target = (
            None
            if is_container(issue)
            or is_container(snapshot["issues"][str(route(snapshot["issues"], n))])
            else route(snapshot["issues"], n)
        )
        if d["target_parent"] != target or (
            target is not None and type(d["target_parent"]) is not int
        ):
            raise ValueError("judgement routing differs from the formal ancestry")
        deps = d["dependencies"]
        if (
            not isinstance(deps, list)
            or any(type(x) is not int for x in deps)
            or deps != issue["dependencies"]
        ):
            raise ValueError("judgement omitted/changed collected dependencies")
        required = guard(snapshot, n)
        if required is not None and d["disposition"] == "startable":
            raise ValueError(
                f"#{n}: judgement contradicts mechanical guard {required['disposition']}"
            )
        for field in ("rationale", "resume_condition", "next_action"):
            nonempty(d[field], field)
        strings(d["facts"], "facts")
        strings(d["missing_information"], "missing_information", empty=True)
        sources = strings(d["sources"], "sources")
        for source in sources:
            source_url(snapshot, source)
        if not {
            f"issue:{k}"
            for k in [n, *deps, *issue["source_reports"], *([target] if target else [])]
        } <= set(sources):
            raise ValueError(
                "sources must include the issue, formal management target, and dependencies"
            )
        if d["disposition"] == "information_needed" and not d["missing_information"]:
            raise ValueError(
                "information_needed requires a concrete missing decision/input"
            )
        if d["disposition"] == "startable":
            priority = number(d["priority"])
            if priority in priorities:
                raise ValueError("duplicate candidate priority")
            priorities.add(priority)
            req = object_value(d["requirements"])
            if set(req) != REQUIREMENTS or d["missing_information"]:
                raise ValueError(
                    "startable requires complete requirements without unresolved choices"
                )
            for field in REQUIREMENTS:
                nonempty(req[field], field)
            if not any(source.startswith("code:") for source in sources):
                raise ValueError("startable requires inspected pinned code as evidence")
        elif d["priority"] is not None or d["requirements"] is not None:
            raise ValueError("held issues cannot carry execution requirements/priority")
        if required is not None and d["disposition"] != required["disposition"]:
            diagnostics.append(
                f"#{n}: non-startable disposition mismatch; "
                f"mechanical_guard={required['disposition']}; "
                f"agent_disposition={d['disposition']}; "
                f"recorded_disposition={required['disposition']}"
            )
            d = {
                **d,
                "disposition": required["disposition"],
                "rationale": (
                    f"機械的阻害条件に従い{required['disposition']}として記録する。"
                    f"agentの分類: {d['disposition']}。agentの理由: {d['rationale']}"
                ),
                "facts": list(
                    dict.fromkeys(
                        [
                            *d["facts"],
                            *(f"機械的阻害条件: {f}" for f in required["facts"]),
                        ]
                    )
                ),
            }
        decisions.append(d)
    if seen != set(snapshot["subjects"]):
        raise ValueError("judgement did not cover every collected open issue")
    return sorted(decisions, key=lambda d: d["issue"]), diagnostics


def execution_context(
    snapshot: dict[str, Any], issue: int, sources: Sequence[str] = ()
) -> dict[str, Any]:
    from .collect import material

    return {
        "dispatcher": {
            "schema_version": 1,
            "input_sha256": snapshot["input_sha256"],
            "target_sha256": digest(material(snapshot, issue, sources)),
            "code_commit": snapshot["code"]["commit"],
        }
    }
