"""Versioned execution evidence, distinct from human permission and runtime identity."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .judgement_comments import owned_comment
from .policy import digest, nonempty, number, object_value, phase, section_span

if TYPE_CHECKING:
    from .session import ParentSession


class ExecutionBasisStale(ValueError):
    """A successfully observed change/revocation, not a retrieval failure."""


POLICY_VERSION = "autonomous-start/v1"
BASIS_KEYS = {
    "schema_version",
    "kind",
    "purpose",
    "scope",
    "behavior",
    "acceptance",
    "rationale",
    "sources",
    "input_sha256",
    "policy_version",
}
AUTH_KEYS = {"generation", "execution_basis_sha256", "execution_basis_kind"}
LEGACY_AUTH_KEYS = {"generation", "agreement_sha256"}
PROGRESS_SECTIONS = {
    "子タスクのチェックリスト",
    "進捗概要",
    "成果物・関連PR",
    "次の改善候補",
}


def sha256(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def validate_basis(value: Any) -> dict[str, Any]:
    data = object_value(value)
    human = data.get("kind") == "human_agreement"
    keys = BASIS_KEYS | ({"human_confirmation"} if human else set())
    if (
        set(data) != keys
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
    ):
        raise ValueError("unknown or malformed execution basis")
    if data["kind"] not in {"human_agreement", "autonomous_decision"}:
        raise ValueError("unknown execution basis kind")
    policy = "human-agreement/v1" if human else POLICY_VERSION
    if data["policy_version"] != policy:
        raise ValueError("unsupported execution policy version")
    for key in ("purpose", "scope", "behavior", "acceptance", "rationale"):
        nonempty(data[key], key)
    if human:
        nonempty(data["human_confirmation"], "human_confirmation")
    if not isinstance(data["sources"], list) or not data["sources"]:
        raise ValueError("execution basis requires sources")
    for source in data["sources"]:
        nonempty(source, "source")
    if len(set(data["sources"])) != len(data["sources"]):
        raise ValueError("duplicate execution sources")
    sha256(data["input_sha256"])
    return data


def human_basis(event: Mapping[str, Any]) -> dict[str, Any]:
    agreement = object_value(event["agreement"])
    inputs = event["agreement_sha256"]
    if "requirements_sha256" in event:
        inputs = digest(
            {
                "agreement_sha256": inputs,
                "requirements_sha256": sha256(event["requirements_sha256"]),
            }
        )
    if event.get("improvement_approval") is not None:
        inputs = digest(
            {
                "agreement_inputs": inputs,
                "improvement_approval": event["improvement_approval"],
            }
        )
    if "acceptance_snapshot" in event:
        inputs = digest(
            {
                "agreement_inputs": inputs,
                "acceptance_snapshot": event["acceptance_snapshot"],
            }
        )
    return validate_basis(
        {
            "schema_version": 1,
            "kind": "human_agreement",
            **agreement,
            "rationale": "人間との要件合意",
            "sources": [f"parent-session:agreed:revision:{event.get('revision')}"],
            "input_sha256": inputs,
            "policy_version": "human-agreement/v1",
        }
    )


def validate_authorization(value: Any, *, legacy: bool = False) -> dict[str, Any]:
    data = object_value(value)
    number(data.get("generation"))
    if legacy and set(data) == LEGACY_AUTH_KEYS:
        sha256(data["agreement_sha256"])
    elif set(data) == AUTH_KEYS:
        sha256(data["execution_basis_sha256"])
        if data["execution_basis_kind"] not in {
            "human_agreement",
            "autonomous_decision",
        }:
            raise ValueError("unknown authorization basis kind")
    else:
        raise ValueError("malformed execution authorization")
    return data


def requirement_body(body: str) -> str:
    # Only named display sections are mutable without a new decision. Unknown
    # sections and stop/exclusion text remain material; timestamps are not inputs.
    spans = []
    for heading in PROGRESS_SECTIONS:
        span = section_span(body, heading)
        if span:
            start = body.rfind("\n## ", 0, span[0])
            if start < 0:
                start = 0
            spans.append((start, span[1]))
    for start, end in sorted(spans, reverse=True):
        body = body[:start] + "\n" + body[end:]
    body = re.sub(r"\n{3,}", "\n\n", body)
    for heading in ("Acceptance checklist", "全体の完了条件"):
        span = section_span(body, heading)
        if span:
            start, end = span
            body = (
                body[:start]
                + re.sub(r"(?m)^(\s*[-*] \[)[xX](\])", r"\1 \2", body[start:end])
                + body[end:]
            )
    return body.strip()


def comment_snapshot(comment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": number(comment.get("id")),
        "login": nonempty(
            object_value(comment.get("user")).get("login"), "comment login"
        ),
        "body": comment.get("body") or "",
    }


def instruction_comments(
    session: ParentSession, comments: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    from .history import instruction_comments as read_instructions

    return read_instructions(session.gh.repo, session.parent, session.login, comments)


def ignored_comment_map(
    ignored_comments: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    ignored = {}
    for raw in ignored_comments:
        entry = object_value(raw)
        if set(entry) != {"id", "sha256", "reason"}:
            raise ValueError("ignored comment requires id/sha256/reason")
        n = number(entry["id"])
        sha256(entry["sha256"])
        nonempty(entry["reason"], "irrelevance reason")
        if n in ignored:
            raise ValueError("duplicate ignored comment")
        ignored[n] = entry
    return ignored


def execution_input(
    session: ParentSession,
    ignored_comments: Sequence[Mapping[str, Any]] = (),
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect stable material. Only the trusted controller classifies prose."""
    ignored = ignored_comment_map(ignored_comments)
    raw = session.gh.issue(session.parent)
    if "ops-report" in {label["name"] for label in raw["labels"]}:
        raise ValueError("ops-report containers cannot receive an execution basis")
    raw_comments = session.gh.parent_comments(session.parent, session.login)
    # Older dispatcher records pinned the display's digest in ignored_comments.
    # Revalidate its complete current format before allowing a display update.
    for comment in raw_comments:
        if comment["id"] in ignored and owned_comment(
            comment, session.login, session.gh.repo, session.parent
        ):
            del ignored[comment["id"]]
    comments = []
    observed = set()
    for item in instruction_comments(session, raw_comments):
        observed.add(item["id"])
        if item["id"] in ignored:
            if digest(item) != ignored[item["id"]]["sha256"]:
                raise ValueError("ignored comment changed; classify it again")
        else:
            comments.append(
                {"id": item["id"], "login": item["login"], "sha256": digest(item)}
            )
    if set(ignored) - observed:
        raise ValueError("ignored comment missing from the current discussion")
    from .improvements import input_state

    improvement = input_state(session.gh, session.parent, session.login, session.store)
    return {
        "schema_version": 1,
        "repo": session.gh.repo.casefold(),
        "parent_issue": session.parent,
        "issue": {
            "title": raw["title"],
            "body_sha256": digest(requirement_body(raw.get("body") or "")),
            "needs_human": "needs-human" in {label["name"] for label in raw["labels"]},
            **({"improvement": improvement} if improvement is not None else {}),
        },
        "comments": sorted(comments, key=lambda c: c["id"]),
        "context": dict(context or {}),
    }


def validate_event(event: Mapping[str, Any]) -> None:
    required = {
        "event",
        "basis",
        "basis_sha256",
        "generation",
        "revision",
        "inputs",
        "ignored_comments",
    }
    if not required <= set(event) or set(event) - required - {
        "parent",
        "body",
        "improvement_approval",
        "acceptance_snapshot",
    }:
        raise ValueError("malformed execution basis event")
    basis = validate_basis(event.get("basis"))
    if basis["kind"] != "autonomous_decision" or event.get("basis_sha256") != digest(
        basis
    ):
        raise ValueError("autonomous execution basis digest/kind mismatch")
    number(event.get("generation"))
    number(event.get("revision"))
    inputs = object_value(event.get("inputs"))
    if (
        set(inputs)
        != {"schema_version", "repo", "parent_issue", "issue", "comments", "context"}
        or type(inputs["schema_version"]) is not int
        or inputs["schema_version"] != 1
    ):
        raise ValueError("unknown or malformed execution input schema")
    if basis["input_sha256"] != digest(inputs):
        raise ValueError("execution input digest mismatch")
    if not isinstance(event.get("ignored_comments"), list):
        raise ValueError("execution observations missing")


def _current_basis(session: ParentSession) -> dict[str, Any]:
    # A recorded agreement/decision is evidence of authority. Prose edits and
    # handoffs are observations for the parent, not revocations of that evidence.
    from .improvements import require as require_improvement

    binding = require_improvement(session)
    current = next(
        (
            e
            for e in reversed(session.history())
            if e["event"] in {"agreed", "execution_basis", "basis_revoked"}
        ),
        None,
    )
    if current is None or current["event"] == "basis_revoked":
        raise ExecutionBasisStale(
            "親issueの要件合意記録または有効な実行根拠がありません。"
        )
    if binding is not None and current.get("improvement_approval") != binding:
        raise ExecutionBasisStale(
            "current basis is not bound to the current human improvement approval"
        )
    if current["event"] == "agreed":
        return human_basis(current)
    validate_event(current)
    if "needs-human" in {
        label["name"] for label in session.gh.issue(session.parent)["labels"]
    }:
        raise ExecutionBasisStale("explicit human decision is pending")
    return object_value(current["basis"])


def current_basis(session: ParentSession) -> dict[str, Any]:
    session.owner()
    return _current_basis(session)


def observed_authorization(session: ParentSession) -> dict[str, Any]:
    """Validate the saved owner/basis without granting the viewer ownership."""
    return observed_review_authorization(session)


def observed_review_authorization(
    session: ParentSession, request: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Read saved authority, including the old review's representation of it."""
    row = session.row
    if not row:
        raise ValueError("worker freshness requires a saved parent owner")
    basis = _current_basis(session)
    return _basis_authorization(basis, row["generation"], request)


def validate_candidate(
    session: ParentSession, basis_sha256: str, generation: int
) -> dict[str, Any]:
    """Dispatcher calls this under Store.locked(), immediately before reservation."""
    sha256(basis_sha256)
    number(generation)
    if session.row and session.row["status"] not in {"released", "ended"}:
        raise ValueError("candidate already has an owner; do not replace it")
    if generation != session.row.get("generation", 0) + 1:
        raise ValueError("candidate owner generation changed")
    basis = _current_basis(session)
    if basis["kind"] != "autonomous_decision" or digest(basis) != basis_sha256:
        raise ValueError("candidate execution basis changed")
    # Before there is an active parent, the dispatcher has nobody who can
    # interpret new instructions. Recollect/rejudge changed admission inputs.
    decision = next(
        e for e in reversed(session.history()) if e["event"] == "execution_basis"
    )
    if (
        decision["generation"] != generation
        or execution_input(
            session, decision["ignored_comments"], decision["inputs"]["context"]
        )
        != decision["inputs"]
    ):
        raise ValueError("candidate inputs changed before parent admission")
    if session.gh.issue(session.parent)["state"] != "open":
        raise ValueError("candidate issue is no longer open")
    return {
        "generation": generation,
        "execution_basis_kind": basis["kind"],
        "execution_basis_sha256": basis_sha256,
    }


def _basis_authorization(
    basis: Mapping[str, Any], generation: int, request: Mapping[str, Any] | None
) -> dict[str, Any]:
    if request is not None and request["schema_version"] == 1:
        if basis["kind"] != "human_agreement":
            raise ValueError("legacy review requires a human agreement")
        agreement = {
            key: basis[key]
            for key in (
                "purpose",
                "scope",
                "behavior",
                "acceptance",
                "human_confirmation",
            )
        }
        return {"generation": generation, "agreement_sha256": digest(agreement)}
    return validate_authorization(
        {
            "generation": generation,
            "execution_basis_kind": basis["kind"],
            "execution_basis_sha256": digest(basis),
        }
    )


def authorization(
    session: ParentSession, request: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return _basis_authorization(
        current_basis(session), session.owner()["generation"], request
    )


def record_autonomous(session: ParentSession, value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"basis", "inputs", "ignored_comments"}:
        raise ValueError("basis record requires basis/inputs/ignored_comments")
    if session.row and session.row["status"] not in {"released", "ended"}:
        session.owner()  # A judgment job may not overwrite another active parent's basis.
    history = session.history()
    if phase(history) not in {"assigned", "autonomous"}:
        raise ValueError(
            "human checking/agreement/revocation cannot be overridden autonomously"
        )
    basis = validate_basis(value["basis"])
    from . import improvements

    binding = improvements.require(session)
    if binding is not None:
        permitted, _ = improvements.agreement(session, binding["decision_id"])
        if any(
            basis[k] != permitted[k]
            for k in ("purpose", "scope", "behavior", "acceptance")
        ):
            raise ValueError(
                "autonomous basis differs from the human-approved improvement scope"
            )
    row = session.row
    generation = row.get("generation", 0) + (row.get("status") != "active")
    event = {
        "event": "execution_basis",
        **value,
        "basis_sha256": digest(basis),
        "generation": generation,
        **({"improvement_approval": binding} if binding is not None else {}),
    }
    from .completion import snapshot as acceptance_snapshot

    event["acceptance_snapshot"] = acceptance_snapshot(
        session.gh.issue(session.parent).get("body") or ""
    )
    previous = next(
        (e for e in reversed(history) if e["event"] == "execution_basis"), {}
    )
    event["revision"] = (
        previous["revision"]
        if previous and all(previous.get(k) == v for k, v in event.items())
        else max(
            (e["revision"] for e in history if e["event"] == "execution_basis"),
            default=0,
        )
        + 1
    )
    validate_event(event)
    if session.gh.issue(session.parent)["state"] != "open":
        raise ValueError("cannot record an execution basis for a closed issue")
    inputs = execution_input(
        session, value["ignored_comments"], value["inputs"]["context"]
    )
    if inputs != value["inputs"] or inputs["issue"]["needs_human"]:
        raise ValueError("execution inputs changed or a human decision is pending")
    session.post(
        event,
        "## 自律着手の実行根拠\n\n"
        + "\n\n".join(
            f"{key}: {basis[key]}"
            for key in ("purpose", "scope", "behavior", "acceptance", "rationale")
        ),
    )
    if (
        execution_input(session, value["ignored_comments"], value["inputs"]["context"])
        != inputs
    ):
        raise ValueError(
            "execution inputs changed during publication; decision is stale"
        )
    return {
        "parent": session.parent,
        "basis_sha256": digest(basis),
        "generation": generation,
    }


def require_requirement_targets(
    request: Mapping[str, Any], targets: set[int], *, exact: bool = False
) -> None:
    version = request.get("schema_version")
    if type(version) is not int or version not in {1, 2, 3}:
        raise ValueError("unknown review request version for requirements")
    if version == 1:
        return
    expected = {str(n) for n in targets | {request["parent_issue"]}}
    actual = set(object_value(request["requirements"]))
    if expected - actual or (exact and actual != expected):
        raise ValueError(
            "review closing targets differ from frozen requirements: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}; "
            "prepare a new run with reason"
        )


def require_request_authorization(
    session: ParentSession,
    request: Mapping[str, Any],
    *,
    required_issues: set[int] | None = None,
) -> dict[str, Any]:
    auth = authorization(session, request)
    require_review_inputs(session, request, auth, required_issues=required_issues)
    return auth


def require_review_inputs(
    session: ParentSession,
    request: Mapping[str, Any],
    auth: Mapping[str, Any],
    *,
    required_issues: set[int] | None = None,
) -> None:
    # Preserve the actual target binding. The parent assesses changes in the
    # meaning of requirements; hashes remain immutable historical observations.
    if (
        request["repo"].casefold() != session.gh.repo.casefold()
        or request["parent_issue"] != session.parent
    ):
        raise ValueError("review belongs to another repository/parent")
    require_requirement_targets(request, required_issues or set())


def require_run_authorization(
    session: ParentSession,
    request: Mapping[str, Any],
    *,
    required_issues: set[int] | None = None,
) -> dict[str, Any]:
    auth = require_request_authorization(
        session, request, required_issues=required_issues
    )
    require_review_history(session, request, auth)
    return auth


def require_review_history(
    session: ParentSession, request: Mapping[str, Any], auth: Mapping[str, Any]
) -> None:
    for event in session.history():
        if (
            event.get("event") == "review"
            and event.get("review_run_id") == request["review_run_id"]
            and event.get("request_sha256") != digest(request)
        ):
            raise ValueError("review history refers to a different request")


def task_errors(
    task: Mapping[str, Any], auth: Mapping[str, Any], body: str
) -> tuple[str, ...]:
    from .worker_preflight import review_errors

    return review_errors(dict(task))


def input_changes(session: ParentSession) -> dict[str, Any]:
    """Read-only change notices; the parent decides meaning and affected work."""
    from .agreement_comments import _observations

    latest = next(
        (
            e
            for e in reversed(session.history())
            if e["event"] in {"agreed", "execution_basis"}
        ),
        None,
    )
    if latest is None:
        return {
            "body_changed": False,
            "comments": [],
            "assessment": "no_recorded_basis",
        }
    raw = session.gh.issue(session.parent)
    if latest["event"] == "agreed":
        _, changes = _observations(session, latest)
        expected = latest.get("requirements_sha256")
    else:
        expected = latest["inputs"]["issue"]["body_sha256"]
        before = {c["id"]: c["sha256"] for c in latest["inputs"]["comments"]}
        before.update({c["id"]: c["sha256"] for c in latest["ignored_comments"]})
        current = instruction_comments(
            session, session.gh.parent_comments(session.parent, session.login)
        )
        after = {c["id"]: digest(c) for c in current}
        changes = {
            n: "added" if n not in before else "deleted" if n not in after else "edited"
            for n in before.keys() | after.keys()
            if before.get(n) != after.get(n)
        }
    return {
        "body_changed": expected is None
        or expected != digest(requirement_body(raw.get("body") or "")),
        "comments": [
            {
                "id": n,
                "change": kind,
                "url": f"https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{n}",
            }
            for n, kind in sorted(changes.items())
        ],
        "assessment": "parent_judgement",
    }
