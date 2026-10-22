"""Explicit parent-review contracts and coverage decisions, without I/O."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from parent.execution import require_requirement_targets, validate_authorization
from parent.policy import digest, nonempty, number, object_value
from review_policy import parse_finding

PREFIX = "<!-- parent-review:v1 "
SNAPSHOT_KEYS = {
    "number",
    "head_sha",
    "base_sha",
    "base_ref",
    "merge_base_sha",
    "diff_sha256",
    "branch",
}
REQUEST_KEYS = {
    "schema_version",
    "review_run_id",
    "repo",
    "parent_issue",
    "reason",
    "prs",
}


def exact(value: Any, keys: set[str]) -> dict[str, Any]:
    data = object_value(value)
    if set(data) != keys:
        raise ValueError(f"expected exactly {sorted(keys)}, got {sorted(data)}")
    return data


def load_json(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        if len(pairs) != len({key for key, _ in pairs}):
            raise ValueError("duplicate JSON keys")
        return dict(pairs)

    return json.loads(text, object_pairs_hook=unique)


def hex_value(value: Any, length: int) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"expected {length}-digit lowercase hex identifier")
    return value


def snapshots(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("PR snapshots must be a nonempty list")
    result = []
    for raw in value:
        entry = exact(raw, SNAPSHOT_KEYS)
        number(entry["number"])
        for key in ("head_sha", "base_sha", "merge_base_sha"):
            hex_value(entry[key], 40)
        hex_value(entry["diff_sha256"], 64)
        nonempty(entry["base_ref"], "base_ref")
        nonempty(entry["branch"], "branch")
        result.append(entry)
    ids = [p["number"] for p in result]
    if ids != sorted(set(ids)):
        raise ValueError("PR snapshots must be unique and sorted by number")
    return result


def request_manifest(value: Any, *, instructed: bool = True) -> dict[str, Any]:
    version = object_value(value).get("schema_version")
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("unknown review-request schema")
    keys = REQUEST_KEYS | ({"review_state"} if instructed else set())
    if version == 2:
        keys |= {"authorization", "requirements"}
    data = exact(value, keys)
    if version == 2:
        validate_authorization(data["authorization"])
        requirements = object_value(data["requirements"])
        if str(data["parent_issue"]) not in requirements:
            raise ValueError("review requires the parent requirements digest")
        for n, sha in requirements.items():
            if not re.fullmatch(r"[1-9][0-9]*", n):
                raise ValueError("review requirements must name issue numbers")
            hex_value(sha, 64)
    parent = number(data["parent_issue"])
    if not re.fullmatch(rf"parent-{parent}-[0-9a-f]{{32}}", str(data["review_run_id"])):
        raise ValueError("invalid parent review run ID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(data["repo"])):
        raise ValueError("invalid repository")
    nonempty(data["reason"], "review reason")
    snapshots(data["prs"])
    if instructed and data["review_state"] != "instructed":
        raise ValueError("review has not been explicitly instructed")
    return {k: v for k, v in data.items() if k != "review_state"}


def path_value(value: Any) -> str:
    path = nonempty(value, "path")
    if (
        PurePosixPath(path).is_absolute()
        or any(p in {".", ".."} for p in path.split("/"))
        or "\\" in path
    ):
        raise ValueError("finding paths must be repository-relative POSIX paths")
    return path


def findings(value: Any, prefix: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("findings must be a list")
    result = []
    for i, raw in enumerate(value, 1):
        finding = asdict(parse_finding(raw, i))
        if finding["path"] is not None:
            finding["path"] = path_value(finding["path"])
        result.append({"id": f"{prefix}-{i}", **finding})
    return result


def parse_output(text: str, numbers: Sequence[int]) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\n(.*)\n```", stripped, re.DOTALL)
    data = exact(
        load_json(fence[1] if fence else stripped),
        {"verdict", "summary", "cross_findings", "prs"},
    )
    nonempty(data["summary"], "summary")
    if not isinstance(data["prs"], list) or not isinstance(
        data["cross_findings"], list
    ):
        raise ValueError("prs and cross_findings must be lists")
    cross = []
    for i, raw in enumerate(data["cross_findings"], 1):
        item = exact(raw, {"prs", "severity", "path", "line", "rule", "detail"})
        affected = item["prs"]
        if (
            not isinstance(affected, list)
            or not affected
            or any(type(n) is not int for n in affected)
            or affected != sorted(set(affected))
            or not set(affected) <= set(numbers)
        ):
            raise ValueError("cross finding must identify unique covered PRs")
        finding = findings([{k: v for k, v in item.items() if k != "prs"}], "cross")[0]
        cross.append({**finding, "id": f"cross-{i}", "prs": affected})
    prs = []
    for raw in data["prs"]:
        item = exact(raw, {"number", "coverage", "summary", "findings"})
        n = number(item["number"])
        nonempty(item["summary"], "PR summary")
        if item["coverage"] not in {"reviewed", "not_reviewed"}:
            raise ValueError("unknown PR coverage")
        parsed = findings(item["findings"], f"pr-{n}")
        if item["coverage"] == "not_reviewed" and parsed:
            raise ValueError("not_reviewed PR cannot carry findings")
        prs.append({**item, "findings": parsed})
    if sorted(p["number"] for p in prs) != sorted(numbers):
        raise ValueError("review output PR set differs from the request")
    partial = any(p["coverage"] != "reviewed" for p in prs)
    expected = (
        "incomplete"
        if partial
        else "request_changes"
        if cross or any(p["findings"] for p in prs)
        else "approve"
    )
    if data["verdict"] != expected:
        raise ValueError("overall verdict contradicts findings/coverage")
    return {
        **data,
        "cross_findings": cross,
        "prs": sorted(prs, key=lambda p: p["number"]),
    }


def marker(record: Mapping[str, Any]) -> str:
    data = json.dumps(
        dict(record), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return PREFIX + data.replace("<", r"\u003c").replace(">", r"\u003e") + " -->"


def records(
    comments: Sequence[Mapping[str, Any]], trusted_login: str
) -> list[dict[str, Any]]:
    """Only complete first-line records from the authenticated publisher are evidence."""
    result = []
    for comment in comments:
        if object_value(comment.get("user")).get("login") != trusted_login:
            continue
        line = (comment.get("body") or "").split("\n", 1)[0].rstrip("\r")
        if line.startswith("<!-- parent-review:v") and not line.startswith(PREFIX):
            raise ValueError("unknown parent-review marker version")
        if not line.startswith(PREFIX):
            continue
        if not line.endswith(" -->"):
            raise ValueError("malformed parent-review marker")
        record = object_value(load_json(line[len(PREFIX) : -4]))
        if (
            type(record.get("schema_version")) is not int
            or record.get("schema_version") != 1
            or record.get("kind")
            not in {
                "result",
                "pr",
                "resolution",
                "pending",
            }
        ):
            raise ValueError("unknown parent-review record")
        if record["kind"] == "result":
            validate_result(record)
        elif record["kind"] == "resolution":
            validate_resolution(record)
        result.append(record)
    return result


def pr_findings(result: Mapping[str, Any], n: int) -> list[dict[str, Any]]:
    pr = next(p for p in result["prs"] if p["number"] == n)
    return [*pr["findings"], *(f for f in result["cross_findings"] if n in f["prs"])]


def path_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("paths must be a list")
    paths = [path_value(path) for path in value]
    if paths != sorted(set(paths)):
        raise ValueError("paths must be unique and sorted")
    return paths


def evidence(value: Any) -> dict[str, Any]:
    data = exact(value, {"ci", "regression_tests", "resolution", "reviewed_paths"})
    for key in ("ci", "regression_tests", "resolution"):
        nonempty(data[key], key)
    path_list(data["reviewed_paths"])
    return data


def validate_result(value: Any) -> dict[str, Any]:
    data = exact(
        value,
        {
            "schema_version",
            "kind",
            "review_run_id",
            "request_sha256",
            "request",
            "reviewer",
            "profile",
            "started_at",
            "finished_at",
            "status",
            "verdict",
            "summary",
            "error",
            "exception",
            "cross_findings",
            "prs",
        },
    )
    request = request_manifest(data["request"], instructed=False)
    if (
        type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["kind"] != "result"
        or data["review_run_id"] != request["review_run_id"]
        or data["request_sha256"] != digest(request)
    ):
        raise ValueError("result identity/request digest mismatch")
    if data["reviewer"] != "codex" or data["profile"] != "review":
        raise ValueError("result must come from the Codex review profile")
    times = [datetime.fromisoformat(data[k]) for k in ("started_at", "finished_at")]
    if any(t.tzinfo is None for t in times) or times[1] < times[0]:
        raise ValueError("invalid run timestamps")
    if data["status"] not in {"succeeded", "failed"} or (
        data["status"] == "succeeded"
    ) != (data["verdict"] != "incomplete"):
        raise ValueError("result status/verdict mismatch")
    if data["status"] == "failed":
        nonempty(data["error"], "run error")
    elif data["error"] is not None:
        raise ValueError("successful run cannot carry an error")
    raw_cross = []
    for f in data["cross_findings"]:
        raw_cross.append({k: v for k, v in f.items() if k != "id"})
    raw_prs = []
    if [p["number"] for p in data["prs"]] != [p["number"] for p in request["prs"]]:
        raise ValueError("result PR order/set differs from the request")
    for p in data["prs"]:
        exact(p, {"number", "child_issues", "coverage", "summary", "findings"})
        children = p["child_issues"]
        if not isinstance(children, list) or children != sorted(set(children)):
            raise ValueError("invalid child issue assignment")
        for n in children:
            number(n)
        raw_prs.append(
            {
                **{k: v for k, v in p.items() if k not in {"child_issues", "findings"}},
                "findings": [
                    {k: v for k, v in f.items() if k != "id"} for f in p["findings"]
                ],
            }
        )
    require_requirement_targets(
        request,
        {n for p in data["prs"] for n in p["child_issues"]},
        exact=data["status"] == "succeeded",
    )
    parsed = parse_output(
        json.dumps(
            {
                "verdict": data["verdict"],
                "summary": data["summary"],
                "cross_findings": raw_cross,
                "prs": raw_prs,
            }
        ),
        [p["number"] for p in request["prs"]],
    )
    if parsed["cross_findings"] != data["cross_findings"] or any(
        a["findings"] != b["findings"]
        for a, b in zip(parsed["prs"], data["prs"], strict=True)
    ):
        raise ValueError("finding IDs/order are not canonical")
    if data["exception"] is not None:
        exception = exact(data["exception"], {"previous_runs", "reason"})
        if (
            not isinstance(exception["previous_runs"], list)
            or not exception["previous_runs"]
            or exception["reason"] != request["reason"]
        ):
            raise ValueError("invalid explicit review exception")
    return data


def validate_resolution(value: Any) -> dict[str, Any]:
    data = exact(
        value,
        {
            "schema_version",
            "kind",
            "review_run_id",
            "result_sha256",
            "parent_issue",
            "pr",
            "finding",
            "reviewed_head",
            "fixed_snapshot",
            "fix_diff_sha256",
            "changed_paths",
            "evidence",
        },
    )
    if (
        type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["kind"] != "resolution"
    ):
        raise ValueError("unknown resolution schema")
    n = number(data["pr"])
    number(data["parent_issue"])
    snapshots([data["fixed_snapshot"]])
    if data["fixed_snapshot"]["number"] != n or not re.fullmatch(
        rf"(?:cross|pr-{n})-[1-9][0-9]*", str(data["finding"])
    ):
        raise ValueError("invalid resolution PR/finding")
    hex_value(data["reviewed_head"], 40)
    hex_value(data["result_sha256"], 64)
    hex_value(data["fix_diff_sha256"], 64)
    paths = path_list(data["changed_paths"])
    if evidence(data["evidence"])["reviewed_paths"] != paths:
        raise ValueError("fix paths differ from the parent's reviewed paths")
    return data


@dataclass(frozen=True)
class BaseAdvance:
    """One first-parent step authenticated by GitHub's merged PR and real git.

    Construct with review_workspace.verify_base_advances, not PR-supplied data.
    A member of this run must explain *every* step between the two base tips.
    """

    pr: int
    base_ref: str
    before_sha: str
    after_sha: str


def base_advance_covered(
    original: Mapping[str, Any],
    latest: Mapping[str, Any],
    reviewed: Mapping[int, Mapping[str, Any]],
    advances: Sequence[BaseAdvance],
) -> bool:
    if original["base_ref"] != latest["base_ref"]:
        return False
    tip = original["base_sha"]
    seen: set[int] = set()
    while tip != latest["base_sha"]:
        steps = {
            step
            for step in advances
            if step.before_sha == tip and step.base_ref == original["base_ref"]
        }
        if len(steps) != 1:
            return False
        step = steps.pop()
        if (
            step.pr == original["number"]
            or step.pr in seen
            or step.pr not in reviewed
            or reviewed[step.pr]["base_ref"] != step.base_ref
            or step.before_sha == step.after_sha
        ):
            return False
        seen.add(step.pr)
        tip = step.after_sha
    return True


@dataclass(frozen=True)
class Coverage:
    covered: bool
    reasons: tuple[str, ...]
    unresolved: tuple[tuple[int, str], ...]


def evaluate_coverage(
    result: Mapping[str, Any],
    current: Sequence[Mapping[str, Any]],
    resolutions: Sequence[Mapping[str, Any]],
    changed_paths: Mapping[int, Sequence[str]],
    *,
    fix_diff_digests: Mapping[int, str],
    base_advances: Sequence[BaseAdvance] = (),
) -> Coverage:
    """Gate boundary: no I/O or reviewer calls. Every deviation fails closed.

    changed_paths is the *actual* reviewed-head..current-head git diff (including
    deletes/rename sources), and fix_diff_digests hashes that same full binary diff.
    Only verified peer merges may advance a snapshot's base tip; all other fields
    must match the reviewed or explicitly resolved snapshot. Unresolved findings
    are separate from coverage so the gate can report both.
    """
    validate_result(dict(result))
    request = request_manifest(result["request"], instructed=False)
    old = {p["number"]: p for p in request["prs"]}
    now = {p["number"]: p for p in snapshots(list(current))} if current else {}
    for step in base_advances:
        number(step.pr)
        nonempty(step.base_ref, "base_ref")
        hex_value(step.before_sha, 40)
        hex_value(step.after_sha, 40)
    valid_resolutions = [
        validate_resolution(dict(record))
        for record in resolutions
        if record.get("kind") == "resolution"
    ]
    reasons: list[str] = []
    unresolved: list[tuple[int, str]] = []
    if result.get("status") != "succeeded":
        reasons.append("run_failed_or_incomplete")
    if set(old) != set(now):
        reasons.append("pr_set_changed")
    for n, original in old.items():
        reviewed = next((p for p in result["prs"] if p["number"] == n), None)
        if reviewed is None or reviewed["coverage"] != "reviewed":
            reasons.append(f"pr_{n}_not_reviewed")
            continue
        if n not in now:
            continue
        latest = now[n]
        for key in ("base_ref", "merge_base_sha", "branch"):
            if original[key] != latest[key]:
                reasons.append(f"pr_{n}_{key}_changed")
        if not base_advance_covered(original, latest, old, base_advances):
            reasons.append(f"pr_{n}_base_sha_changed")
        all_findings = pr_findings(result, n)
        resolved = set()
        for record in valid_resolutions:
            fixed = record["fixed_snapshot"]
            if (
                record.get("review_run_id") != result["review_run_id"]
                or record.get("result_sha256") != digest(result)
                or record.get("pr") != n
                or record.get("parent_issue") != request["parent_issue"]
                or any(fixed[k] != latest[k] for k in SNAPSHOT_KEYS - {"base_sha"})
                or not base_advance_covered(fixed, latest, old, base_advances)
                or record.get("reviewed_head") != original["head_sha"]
                or record.get("finding") not in {f["id"] for f in all_findings}
            ):
                continue
            if record.get("changed_paths") != sorted(set(changed_paths.get(n, []))):
                continue
            if record["fix_diff_sha256"] != fix_diff_digests.get(n):
                continue
            resolved.add(record["finding"])
        unresolved.extend((n, f["id"]) for f in all_findings if f["id"] not in resolved)
        if not resolved and any(
            original[k] != latest[k] for k in ("head_sha", "diff_sha256")
        ):
            reasons.append(f"pr_{n}_unverified_head")
    return Coverage(not reasons, tuple(reasons), tuple(unresolved))
