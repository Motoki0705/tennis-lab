"""Validate the complete judgment before allowing any publication or candidate."""

from __future__ import annotations

from typing import Any

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


def guard(snapshot: dict[str, Any], n: int) -> str | None:
    """Mechanical vetoes supplement, rather than replace, semantic agent judgment."""
    issues = snapshot["issues"]
    raw = issues[str(n)]
    target = route(issues, n)
    if is_container(raw) or is_container(issues[str(target)]):
        return "out_of_scope"
    if raw["parent"] is not None:
        return "existing_parent"
    labels = {x["name"] for x in raw["labels"]}
    if "issue:child" in labels:
        return "human_hold"  # Label/body cannot manufacture a missing formal relation.
    history = events(raw["comments"], snapshot["publisher"], n)
    if (
        "needs-human" in labels
        or child_stopped(raw["body"])
        or phase(history) in {"checking", "agreed"}
    ):
        return "human_hold"
    family = {
        k for k in issues if route(issues, int(k)) == n and issues[k]["state"] == "open"
    }
    if any(issues[k]["assignees"] for k in family):
        return "existing_owner"
    for k, owner in snapshot["owners"].items():
        if k in family and owner["status"] not in {"released", "ended"}:
            return "existing_owner"
        if any(
            task in family and info["delegated"]
            for task, info in owner["tasks"].items()
        ):
            return "existing_owner"
    if any(set(p["issues"]) & {int(k) for k in family} for p in snapshot["pulls"]):
        return "existing_owner"
    if any(
        issues[str(dep)]["state"] != "closed"
        or issues[str(dep)]["state_reason"] != "completed"
        for dep in raw["dependencies"]
    ):
        return "dependency_wait"
    return None


def validate(value: Any, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
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
        if required is not None and d["disposition"] != required:
            raise ValueError(f"#{n}: judgement contradicts mechanical guard {required}")
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
        decisions.append(d)
    if seen != set(snapshot["subjects"]):
        raise ValueError("judgement did not cover every collected open issue")
    return sorted(decisions, key=lambda d: d["issue"])


def execution_context(snapshot: dict[str, Any], issue: int) -> dict[str, Any]:
    from .collect import material

    return {
        "dispatcher": {
            "schema_version": 1,
            "input_sha256": snapshot["input_sha256"],
            "target_sha256": digest(material(snapshot, issue)),
            "code_commit": snapshot["code"]["commit"],
        }
    }
