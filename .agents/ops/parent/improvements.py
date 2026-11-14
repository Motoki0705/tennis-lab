"""End-of-task proposal operations; no automatic assignment or label release."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from issues.improvement import (
    ImprovementSource,
    approved,
    candidate_key,
    confirmation,
    event,
    execution_terms,
    fields,
    fingerprint,
    human_state,
    key,
    parse_source,
    positive,
    read_candidates,
    render_event,
    text,
    validate_spec,
)
from issues.improvement_proposals import parse_extraction

from .improvement_receipts import bound_extraction
from .policy import raw_section, replace_section

if TYPE_CHECKING:
    from .session import ParentSession
    from .state import Store


def lock_key(repo: str) -> str:
    # The same lock protects candidate mutations and the final runtime admission.
    # It is always acquired before a parent operation/registry lock.
    return "improvements:" + repo.casefold()


def candidates(
    session: ParentSession, parent: int | None = None
) -> dict[str, dict[str, Any]]:
    return dict(
        read_candidates(
            session.gh.parent_comments(parent or session.parent, session.login),
            parent or session.parent,
            session.login,
        )
    )


def get(session: ParentSession, value: str) -> tuple[ImprovementSource, dict[str, Any]]:
    source = ImprovementSource(session.parent, candidate_key(value, session.parent))
    row = candidates(session).get(source.key)
    if row is None:
        raise ValueError(f"{source.candidate_id}: candidate is not recorded")
    return source, row


def append(session: ParentSession, row: dict[str, Any]) -> int:
    body = render_event(row)
    session.public(body)
    observed = candidates(session).get(row["key"])
    replay = observed and row["id"] in observed["comments"]
    if not replay and row["previous"] != (observed["last"]["id"] if observed else None):
        raise ValueError("candidate changed before recording; reread the current state")
    from .record_writer import write_record

    response = write_record(session, body)
    comment_id = positive(response.get("id"))
    from decision_logs.parent_sources import comment

    # Only the source's currently assigned parent captures source-side operations.
    # A later reader's human decisions do not write into the old parent's graph.
    if (
        session.row.get("status") == "active"
        and session.row.get("session") == session.session
    ):
        comment(session, row, body, response)
    return int(comment_id)


def comment_url(session: ParentSession, parent: int, comment_id: int) -> str:
    from .history import comment_url as public_url

    comments = session.gh.parent_comments(parent, session.login)
    matches = [c for c in comments if c["id"] == comment_id]
    if len(matches) != 1:
        raise ValueError("improvement evidence comment is missing or duplicated")
    return cast(str, public_url(session.gh.repo.casefold(), parent, matches[0]))


def record(
    session: ParentSession,
    child: int | None,
    value: dict[str, Any],
    fixed_key: str | None,
) -> dict[str, Any]:
    previous = candidates(session).get(fixed_key) if fixed_key is not None else None
    if session.row.get("status") not in {"ended", "released"} or previous is None:
        session.owner()
    parent, children = session.family()
    origin = (
        child
        if child is not None
        else previous["candidate"]["data"]["source_issue"]
        if previous
        else parent.number
    )
    if origin != parent.number and origin not in children:
        raise ValueError("improvement discovery requires the parent or a direct child")
    legacy = set(value) == {"purpose", "evidence", "scope"}
    if legacy:
        # Existing discovery callers get a stable initial key; editing that
        # candidate subsequently uses its returned key, never a new title key.
        fixed_key = (
            fixed_key or "discovery-" + fingerprint({"issue": origin, **value})[:16]
        )
        value = {
            "schema_version": 1,
            **value,
            "observation": value["evidence"],
            "out_of_scope": "未整理。起票前に対象外を記録する。",
            "behavior": "未整理。",
            "acceptance": ["未整理。起票前に検証可能な受入条件を記録する。"],
            "stop_conditions": "未整理。",
            "judgment": "既存の改善候補記録。切り出し判断は未確認。",
            "unknowns": "次の作業の境界は未整理。",
            "expected_effect": "未測定。",
            "unmet_acceptance": [],
        }
    spec = validate_spec(value)
    source = ImprovementSource(session.parent, key(fixed_key))
    previous = candidates(session).get(source.key)
    if previous is not None and previous["candidate"]["data"]["spec"] != spec:
        from .improvement_proposals import load_receipt

        receipt = load_receipt(session, f"proposal-{session.parent}")
        if (
            receipt
            and receipt["creation"]["attempted"]
            and source.key in receipt["members"]
            and previous["proposal"] is None
        ):
            raise ValueError(
                "recover the attempted proposal before revising its candidate"
            )
    data = {"source_issue": origin, "spec": spec, "legacy": legacy}
    if previous and previous["candidate"]["data"] == data:
        row = previous["candidate"]
    else:
        row = event(
            source,
            "candidate",
            previous["revision"] + 1 if previous else 1,
            previous["last"]["id"] if previous else None,
            data,
        )
    comment_id = append(session, row)
    from decision_logs.parent_sources import improvement_candidate

    trace = improvement_candidate(session, row, comment_id)
    # Keep the old discovery entry useful without a second editable candidate.
    if origin != session.parent:
        body = children[origin].body
        current = raw_section(body, "次の改善候補")
        ref = f"- [{source.candidate_id}]({comment_url(session, session.parent, comment_id)}) — {spec['purpose']}"
        lines = [
            line
            for line in current.splitlines()
            if not line.startswith(f"- [{source.candidate_id}]")
        ]
        session.update_body(
            origin,
            replace_section(body, "次の改善候補", "\n".join([*lines, ref]).strip()),
            body,
        )
    return {
        "candidate_id": source.candidate_id,
        "revision": row["revision"],
        "comment": comment_id,
        "recorded": True,
        "current_scope_changed": False,
        "child": origin,
        **trace,
    }


def _decision(
    session: ParentSession, candidate: str, action: str, value: dict[str, Any]
) -> tuple[ImprovementSource, dict[str, Any], bool]:
    from datetime import UTC, datetime

    from .improvement_proposals import candidate_requirements, deadline, proposal_issue

    source, row = get(session, candidate)
    fields(
        value,
        {
            "schema_version",
            "request_key",
            "revision",
            "previous_decision",
            "human_confirmation",
            "reason",
        },
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unknown improvement decision spec")
    key(value["request_key"])
    confirmation(value["human_confirmation"])
    text(value["reason"])
    requested = fingerprint({"action": action, "spec": value})
    replay = next(
        (
            d
            for d in row["decisions"]
            if d["data"]["request_key"] == value["request_key"]
        ),
        None,
    )
    if replay:
        if replay["data"]["request_sha256"] != requested:
            raise ValueError("human decision request key has different content")
        return source, replay, True
    if value["revision"] != row["revision"] or type(value["revision"]) is not int:
        raise ValueError("human decision refers to a different candidate revision")
    latest = row["decisions"][-1] if row["decisions"] else None
    if value["previous_decision"] != (latest["id"] if latest else None):
        raise ValueError(
            "human decision changed; inspect the latest decision before superseding it"
        )
    membership = row["proposal"]
    if membership is None:
        raise ValueError("propose the candidate before recording a human decision")
    raw = proposal_issue(session, membership)
    # A negative decision on an extracted candidate must still stop its parent.
    # Closed proposals otherwise cannot receive new decisions or reapprovals.
    stopping_extracted = row["extraction"] is not None and action != "approve"
    if not stopping_extracted and (
        raw["state"] != "open"
        or raw.get("state_reason") == "reopened"
        or deadline(raw) <= datetime.now(UTC)
    ):
        raise ValueError("closed or expired proposal cannot receive a human decision")
    if action == "approve":
        from .improvement_proposals import validate_candidate

        validate_candidate(row)
        if any(
            d["data"]["human_confirmation"] == value["human_confirmation"]
            for d in row["decisions"]
        ):
            raise ValueError(
                "a new approval needs a new human instruction, not replayed consent"
            )
    data = {
        "target": "candidate",
        "request_key": value["request_key"],
        "request_sha256": requested,
        "action": action,
        "issue": raw["number"],
        "human_confirmation": value["human_confirmation"],
        "reason": value["reason"],
        "requirements": candidate_requirements(row["candidate"]["data"]["spec"]),
        "previous_decision": value["previous_decision"],
    }
    return (
        source,
        event(source, "decision", row["revision"], row["last"]["id"], data),
        False,
    )


def decide(
    session: ParentSession,
    candidate: str | list[str],
    action: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    from .improvement_proposals import proposal_issue, refresh

    requested = [candidate] if isinstance(candidate, str) else candidate
    if not requested or len(
        {candidate_key(c, session.parent) for c in requested}
    ) != len(requested):
        raise ValueError("human decision requires unique candidates")
    specs = []
    for c in requested:
        if "candidates" in value:
            details = value["candidates"]
            if not isinstance(details, dict) or set(details) != set(requested):
                raise ValueError(
                    "decision spec must pin every requested candidate revision/previous_decision"
                )
            fields(details[c], {"revision", "previous_decision"})
            spec = {k: v for k, v in value.items() if k != "candidates"} | details[c]
        elif len(requested) == 1:
            spec = value
        else:
            raise ValueError(
                "multi-candidate decision requires candidates revision map"
            )
        specs.append(_decision(session, c, action, spec))
    results = []
    projections = {}
    for source, row_event, replay in specs:
        if not replay:
            append(session, row_event)
        observed = get(session, source.key)[1]
        stage_stop(session, source, observed)
        results.append(
            {
                "candidate_id": source.candidate_id,
                "decision_id": row_event["id"],
                "state": human_state(observed),
                "issue": row_event["data"]["issue"],
                "reused": replay,
            }
        )
        if observed["proposal"] is not None:
            projections[observed["proposal"]["data"]["issue"]] = observed["proposal"]
    for membership in projections.values():
        raw = proposal_issue(session, membership)
        refresh(
            session,
            raw,
            ended="期限切れ"
            if raw.get("state_reason") == "not_planned"
            else "提案終了"
            if raw["state"] == "closed"
            else None,
        )
    return results[0] if len(results) == 1 else {"decisions": results}


def stage_stop(
    session: ParentSession, source: ImprovementSource, row: dict[str, Any]
) -> None:
    """Commit stop intent before releasing the decision/admission lock."""
    publication = row["extraction"] or row["publication"]
    if (
        human_state(row) not in {"held", "declined", "revoked"}
        or publication is None
        or (row["extraction"] is None and publication["data"]["mode"] != "created")
    ):
        return
    target_key = f"{session.gh.repo.casefold()}#{publication['data']['issue']}"
    target = session.store.parents.get(target_key)
    if target is None:
        return
    decision_id = row["decisions"][-1]["id"]
    if target.get("improvement_stop", {}).get("decision_id") == decision_id:
        return
    session.store.focus(target_key)
    target["improvement_stop"] = {
        "candidate_id": source.candidate_id,
        "decision_id": decision_id,
        "state": "stopping",
    }
    session.store.save()
    session.store.focus(session.key)


def observe(
    gh: Any, parent: int, publisher: str, store: Store
) -> dict[str, Any] | None:
    raw = gh.issue(parent)
    from issues.evidence import require_work_issue

    require_work_issue(raw)
    source = parse_source(raw.get("body") or "")
    index = gh.improvement_extractions(store)
    extraction = bound_extraction(raw.get("body") or "", parent, index)
    if source is None and extraction is None:
        return None
    if source is not None:
        origin = source.parent
        sources = [source]
    else:
        assert extraction is not None
        origin = extraction.parent
        sources = [ImprovementSource(origin, k) for k in extraction.keys]
    rows = read_candidates(gh.parent_comments(origin, publisher), origin, publisher)
    entries = []
    for item in sources:
        candidate = rows.get(item.key)
        if candidate is None:
            raise ValueError(f"{item.candidate_id}: source candidate is unavailable")
        entries.append((item, candidate))
    return {
        "source": entries[0][0],
        "candidate": entries[0][1],
        "issue": raw,
        "approvals": entries,
        "extraction": extraction,
        "receipt": index.get(str(parent)),
    }


def require(
    session: ParentSession, target: int | None = None, *, fresh: bool = False
) -> dict[str, Any] | None:
    from .execution import ExecutionBasisStale

    n = target or session.parent
    index = session.gh.improvement_extractions(session.store)
    if not fresh:
        raw = session.gh.issue(n)
        if (
            parse_source(raw.get("body") or "") is None
            and bound_extraction(raw.get("body") or "", n, index) is None
        ):
            if n == session.parent and any(
                e.get("improvement_approval") for e in session.history()
            ):
                raise ExecutionBasisStale(
                    "improvement approval source marker is missing"
                )
            return None
    with session.gh.fresh_reads():
        observed = observe(session.gh, n, session.login, session.store)
        if observed is None:
            # Legacy publications also retain an existing agreement's binding.
            # Extracted parents have already been checked against the receipt index.
            if n == session.parent and any(
                e.get("improvement_approval") for e in session.history()
            ):
                raise ExecutionBasisStale(
                    "improvement approval source marker is missing"
                )
            if not fresh:
                raise ExecutionBasisStale(
                    "improvement source disappeared during authorization"
                )
            return None
        source, candidate, raw = (
            observed["source"],
            observed["candidate"],
            observed["issue"],
        )
        try:
            if observed["extraction"] is not None:
                extraction = observed["extraction"]
                extraction_rows = [c["extraction"] for _, c in observed["approvals"]]
                if any(e is None for e in extraction_rows):
                    raise ValueError(
                        "extraction record incomplete; recover improvement-extract before execution"
                    )
                data = extraction_rows[0]["data"]
                if (
                    any(e["data"] != data for e in extraction_rows)
                    or data["proposal"] != extraction.proposal
                    or set(data["decision_ids"]) != set(extraction.keys)
                    or data["decision_ids"] != observed["receipt"]["decision_ids"]
                ):
                    raise ValueError(
                        "extraction provenance differs from canonical approvals"
                    )
            decisions = [
                approved(c, item, n, raw.get("body") or "")
                for item, c in observed["approvals"]
            ]
            decision = decisions[0]
        except ValueError as exc:
            from decision_logs.capture import current

            capture = current()
            if capture is not None:
                try:
                    with capture.transaction():
                        capture.result(
                            {
                                "candidate_id": source.candidate_id,
                                "state": human_state(candidate),
                            },
                            artifact="improvement approval gate",
                            status="rejected",
                            summary=f"{source.candidate_id}: execution approval rejected; no implementation launch",
                        )
                except Exception:
                    capture.mark_failed(lost_source="parent.improvement-gate")
            # Ownership handoff already handles stale authority by transferring
            # discussion/control without importing permission or starting a job.
            raise ExecutionBasisStale(str(exc)) from exc
        owner = session.store.parents.get(f"{session.gh.repo.casefold()}#{n}", {})
        if owner.get("improvement_stop", {}).get("state") == "stopping":
            raise ExecutionBasisStale(
                f"{source.candidate_id}: previous runtime stop/cleanup is still pending; retry improvement-decide"
            )
        group = (
            {
                "proposal": observed["extraction"].proposal,
                "approvals": [
                    {"key": item.key, "revision": c["revision"], "decision_id": d["id"]}
                    for (item, c), d in zip(
                        observed["approvals"], decisions, strict=True
                    )
                ],
            }
            if observed["extraction"] is not None
            else {}
        )
        return {
            **group,
            "parent": source.parent,
            "key": source.key,
            "revision": candidate["revision"],
            "issue": n,
            "decision_id": decision["id"],
        }


def input_state(
    gh: Any, parent: int, publisher: str, store: Store
) -> dict[str, Any] | None:
    observed = observe(gh, parent, publisher, store)
    if observed is None:
        return None
    source, candidate = observed["source"], observed["candidate"]
    decisions = candidate["decisions"]
    result = {
        "parent": source.parent,
        "key": source.key,
        "revision": candidate["revision"],
        "state": human_state(candidate),
        "decision_id": decisions[-1]["id"] if decisions else None,
    }
    if observed["extraction"] is not None:
        result["approvals"] = [
            {
                "key": item.key,
                "revision": c["revision"],
                "state": human_state(c),
                "decision_id": c["decisions"][-1]["id"] if c["decisions"] else None,
            }
            for item, c in observed["approvals"]
        ]
    return result


def agreement(
    session: ParentSession, decision_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = require(session)
    if binding is None or decision_id not in [
        a["decision_id"] for a in binding.get("approvals", [binding])
    ]:
        raise ValueError(
            "agree requires a current improvement approval reference; all extracted approvals are imported"
        )
    recorded = candidates(session, binding["parent"])
    candidate = recorded[binding["key"]]
    decision = candidate["decisions"][-1]
    if "approvals" in binding:
        requirements = candidate["extraction"]["data"]["requirements"]
        confirmations = []
        for item in binding["approvals"]:
            row = recorded[item["key"]]
            if row["decisions"][-1]["id"] != item["decision_id"]:
                raise ValueError(
                    "improvement approval changed while deriving agreement"
                )
            confirmation = row["decisions"][-1]["data"]["human_confirmation"]
            confirmations.append(
                f"I-{binding['parent']}-{item['key']}: "
                + confirmation["quote"]
                + "\n\n出典: "
                + confirmation["source"]
            )
        human = "\n\n".join(confirmations)
    else:
        requirements = decision["data"]["requirements"]
        confirmation = decision["data"]["human_confirmation"]
        human = confirmation["quote"] + "\n\n出典: " + confirmation["source"]
    return {**execution_terms(requirements), "human_confirmation": human}, binding


def completion_applies(
    completion: dict[str, Any],
    source: ImprovementSource,
    candidate: dict[str, Any],
    target: int,
    body: str,
) -> bool:
    """A saved result belongs to the exact decision and its protected text.

    The immutable decision ID references the accepted requirements snapshot;
    candidate revision/issue alone cannot identify a later reapproval's scope.
    """
    decisions = candidate["decisions"]
    if not decisions:
        return False
    expected = {
        "parent": source.parent,
        "key": source.key,
        "revision": candidate["revision"],
        "issue": target,
        "decision_id": decisions[-1]["id"],
    }
    if candidate["extraction"] is not None:
        binding = completion.get("improvement_approval") or {}
        data = candidate["extraction"]["data"]
        extraction = parse_extraction(body)
        if (
            extraction is None
            or extraction.parent != source.parent
            or extraction.proposal != data["proposal"]
            or set(extraction.keys) != set(data["decision_ids"])
            or binding.get("parent") != source.parent
            or binding.get("issue") != target
            or {a["key"]: a["decision_id"] for a in binding.get("approvals", [])}
            != data["decision_ids"]
            or not any(
                a
                == {
                    "key": source.key,
                    "revision": candidate["revision"],
                    "decision_id": decisions[-1]["id"],
                }
                for a in binding.get("approvals", [])
            )
        ):
            return False
        try:
            approved(candidate, source, target, body)
        except ValueError:
            return False
        return True
    if (
        completion.get("improvement_approval") != expected
        or parse_source(body) != source
    ):
        return False
    try:
        approved(candidate, source, target, body)
    except ValueError:
        return False
    return True


def status(session: ParentSession) -> list[dict[str, Any]]:
    result = []
    from .improvement_proposals import load_receipt

    receipt = load_receipt(session, f"proposal-{session.parent}")
    for fixed_key, row in candidates(session).items():
        source = ImprovementSource(session.parent, fixed_key)
        publication = row["publication"]
        extraction = row["extraction"]
        proposal = row["proposal"]
        pending = bool(
            receipt
            and receipt["creation"]["attempted"]
            and fixed_key in receipt["members"]
            and proposal is None
        )
        target = (
            extraction["data"]["issue"]
            if extraction
            else publication["data"]["issue"]
            if publication
            else None
        )
        owner = session.store.parents.get(f"{session.gh.repo.casefold()}#{target}", {})
        completion = owner.get("completion", {})
        accepted = (
            target is not None
            and completion.get("state") == "completed"
            and completion_applies(
                completion,
                source,
                row,
                target,
                session.gh.issue(target).get("body") or "",
            )
        )
        if accepted and extraction is not None and target is not None:
            observed = observe(session.gh, target, session.login, session.store)
            accepted = observed is not None and all(
                completion_applies(
                    completion, item, c, target, observed["issue"].get("body") or ""
                )
                for item, c in observed["approvals"]
            )
        execution = execution_status(session, target, owner, accepted=accepted)
        decisions = row["decisions"]
        result.append(
            {
                "candidate_id": source.candidate_id,
                "revision": row["revision"],
                "purpose": row["candidate"]["data"]["spec"]["purpose"],
                "proposal": proposal["data"]["issue"] if proposal else None,
                "unpublished": publication is None and proposal is None,
                "publication": (
                    "extracted"
                    if extraction
                    else "proposed"
                    if proposal
                    else publication["data"]["mode"]
                    if publication
                    else "outcome_unknown"
                    if pending
                    else "not_requested"
                ),
                "issue": target,
                "human_decision": human_state(row),
                "decision_id": decisions[-1]["id"] if decisions else None,
                "execution": execution,
                "record": comment_url(
                    session, session.parent, row["comments"][row["last"]["id"]]
                ),
            }
        )
    return result


def execution_status(
    session: ParentSession, target: int | None, owner: dict[str, Any], *, accepted: bool
) -> str:
    if not owner or target is None:
        return "not_started"
    from worker_status import StatusError, collect_status

    try:
        observed = collect_status(
            parent=target,
            repo=session.gh.repo,
            state_dir=Path(session.store.directory),
            include_inactive=True,
            github=False,
        )
    except StatusError:
        return "unknown"  # Explicitly unavailable, never inferred success.
    states = {
        child["status"] for row in observed["parents"] for child in row["children"]
    }
    if "running" in states:
        return (
            "stopping"
            if owner.get("improvement_stop", {}).get("state") == "stopping"
            else "running"
        )
    if "unknown" in states:
        return "unknown"
    if "failed" in states or owner.get("dispatch", {}).get("phase") == "failed":
        return "failed"
    if accepted:
        return "accepted"
    if owner.get("improvement_stop"):
        return str(owner["improvement_stop"]["state"])
    if "not_started" in states:
        return "reserved"
    if "turn_completed" in states:
        return "awaiting_acceptance"
    return "not_started"
