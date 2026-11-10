"""End-of-task proposal operations; no automatic assignment or label release."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from issues.cli import render_spec
from issues.creation import create_issue_once
from issues.improvement import (
    ImprovementSource,
    approved,
    candidate_key,
    confirmation,
    contract,
    event,
    fields,
    fingerprint,
    human_state,
    index_sources,
    key,
    parse_source,
    positive,
    read_candidates,
    render_event,
    text,
    validate_spec,
)
from worker_logs.journal import atomic, decode

from .policy import raw_section, replace_section

if TYPE_CHECKING:
    from .session import ParentSession


def lock_key(repo: str) -> str:
    # The same lock protects candidate mutations and the final runtime admission.
    # It is always acquired before a parent operation/registry lock.
    return "improvements:" + repo.casefold()


def candidates(
    session: ParentSession, parent: int | None = None
) -> dict[str, dict[str, Any]]:
    return dict(
        read_candidates(
            session.gh.comments(parent or session.parent),
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
    if observed and row["id"] in observed["comments"]:
        return int(observed["comments"][row["id"]])
    if row["previous"] != (observed["last"]["id"] if observed else None):
        raise ValueError("candidate changed before recording; reread the current state")
    response = session.gh.write("POST", session.endpoint + "/comments", {"body": body})
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
    if previous and previous["publication"] is None:
        receipt = _receipt(session, source)
        if (
            receipt is not None
            and receipt["creation"]["attempted"]
            and previous["candidate"]["data"]["spec"] != spec
        ):
            raise ValueError(
                "recover the attempted publication before revising its candidate"
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
        ref = f"- [{source.candidate_id}](https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{comment_id}) — {spec['purpose']}"
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


def draft(
    session: ParentSession, source: ImprovementSource, row: dict[str, Any]
) -> dict[str, Any]:
    data = row["candidate"]["data"]
    spec = data["spec"]
    if data["legacy"]:
        raise ValueError(
            "candidate boundary is incomplete; update the same key before publishing"
        )
    if spec["unmet_acceptance"]:
        raise ValueError(
            "unfinished original acceptance must remain with the original task"
        )
    ref = f"https://github.com/{session.gh.repo}/issues/{source.parent}#issuecomment-{row['comments'][row['candidate']['id']]}"
    rendered = render_spec(
        {
            "role": "parent",
            "identifier": "継続改善",
            "summary": spec["purpose"],
            "children": [],
            "labels": ["enhancement", "needs-human"],
            "improvement_source": {"parent": source.parent, "key": source.key},
            "sections": {
                "目的・背景": spec["purpose"] + "\n\n" + spec["observation"],
                "全体のスコープ": "対象:\n"
                + spec["scope"]
                + "\n\n対象外:\n"
                + spec["out_of_scope"],
                "全体方針・合意事項": "人間の実行判断待ち。起票は実行承認ではありません。\n\n期待する動作:\n"
                + spec["behavior"],
                "全体の完了条件": "\n".join(
                    "- [ ] " + criterion for criterion in spec["acceptance"]
                ),
                "進捗概要": "未着手。人間の実行判断待ち。",
                "依存関係・進め方": "未分割。元親は出典であり先行依存ではありません。\n\n停止条件:\n"
                + spec["stop_conditions"],
                "改善の出典": f"[{source.candidate_id}]({ref})\n\n根拠:\n{spec['evidence']}\n\n切り出し判断:\n{spec['judgment']}",
                "未決事項・人間の判断が必要な点": spec["unknowns"],
            },
        }
    )
    return {k: rendered[k] for k in ("title", "body", "labels")}


def receipt_path(session: ParentSession, source: ImprovementSource) -> Path:
    return Path(
        Path(session.store.directory)
        / "improvement-publications"
        / str(fingerprint(session.gh.repo.casefold()))
        / f"{source.parent}-{source.key}.json"
    )


def _receipt(
    session: ParentSession, source: ImprovementSource
) -> dict[str, Any] | None:
    path = receipt_path(session, source)
    if path.is_symlink():
        raise ValueError("improvement receipt must not be a symlink")
    if not path.exists():
        return None
    value = decode(path.read_bytes())
    fields(
        value,
        {"schema_version", "repo", "parent", "key", "candidate_revision", "creation"},
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["repo"] != session.gh.repo.casefold()
        or value["parent"] != source.parent
        or value["key"] != source.key
    ):
        raise ValueError("improvement publication receipt identity changed")
    positive(value["candidate_revision"])
    fields(value["creation"], {"draft", "attempted", "number"})
    return dict(value)


def publish(
    session: ParentSession, candidate: str, existing: int | None
) -> dict[str, Any]:
    source, row = get(session, candidate)
    if session.row.get("status") in {"active", "handoff_pending"}:
        session.owner()
    # Always index all states. A partial/failed listing must never permit a POST.
    rows = session.gh.issue_rows()
    index = index_sources(
        (positive(raw["number"]), raw.get("body") or "") for raw in rows
    )
    receipt = _receipt(session, source)
    publication = row["publication"]
    if publication is not None:
        n = publication["data"]["issue"]
        if existing is not None and existing != n:
            raise ValueError("candidate already points to another issue")
        if publication["data"]["mode"] == "created" and index.get(source) != n:
            raise ValueError(
                "published proposal disappeared or lost its source; do not recreate it"
            )
        session.gh.issue(n)
        return {
            "candidate_id": source.candidate_id,
            "issue": n,
            "reused": True,
            "publication": publication["data"]["mode"],
        }
    if existing is not None:
        if receipt is not None and receipt["creation"]["attempted"]:
            raise ValueError(
                "creation was attempted; reconcile that issue before changing destination"
            )
        n = positive(existing)
        if n == source.parent:
            raise ValueError(
                "an improvement cannot be attached back to its source parent"
            )
        target = session.gh.issue(n)
        if source in index and index[source] != n:
            raise ValueError("candidate already has a different created proposal")
        # Reference an existing purpose without rewriting its scope, provenance,
        # owner, labels, close reason or human decision.
        row_event = event(
            source,
            "published",
            row["revision"],
            row["last"]["id"],
            {
                "issue": n,
                "mode": "reference",
                "draft_sha256": fingerprint(target.get("body") or ""),
            },
        )
    else:
        fixed_draft = draft(session, source, row)
        if receipt is None:
            receipt = {
                "schema_version": 1,
                "repo": session.gh.repo.casefold(),
                "parent": source.parent,
                "key": source.key,
                "candidate_revision": row["revision"],
                "creation": {"draft": None, "attempted": False, "number": None},
            }
        elif receipt["candidate_revision"] != row["revision"]:
            raise ValueError(
                "candidate changed during publication; recover the original issue first"
            )
        matches = [raw for raw in rows if raw["number"] == index.get(source)]
        session.public(fixed_draft)
        created = create_issue_once(
            session.gh,
            draft=fixed_draft,
            receipt=receipt["creation"],
            matches=matches,
            save=lambda: atomic(receipt_path(session, source), receipt),
        )
        n = positive(created["number"])
        with session.gh.fresh_reads():
            target = session.gh.issue(n)
            if parse_source(target.get("body") or "") != source:
                raise ValueError("created proposal source mismatch")
            labels = {label["name"] for label in target["labels"]}
            if "issue:parent" not in labels:
                raise ValueError("created proposal lost its parent role")
            if (
                session.gh.get(
                    f"repos/{session.gh.repo}/issues/{n}/parent", missing_parent=True
                )
                is not None
            ):
                raise ValueError("improvement proposal acquired a formal parent")
        row_event = event(
            source,
            "published",
            row["revision"],
            row["last"]["id"],
            {"issue": n, "mode": "created", "draft_sha256": fingerprint(fixed_draft)},
        )
    append(session, row_event)
    return {
        "candidate_id": source.candidate_id,
        "issue": n,
        "reused": bool(existing or matches),
        "publication": row_event["data"]["mode"],
    }


def decide(
    session: ParentSession, candidate: str, action: str, value: dict[str, Any]
) -> dict[str, Any]:
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
        # Returning the old receipt must not reinstate an approval after revoke.
        stage_stop(session, source, row)
        return {
            "candidate_id": source.candidate_id,
            "decision_id": replay["id"],
            "state": human_state(row),
            "issue": replay["data"]["issue"],
            "reused": True,
        }
    if value["revision"] != row["revision"] or type(value["revision"]) is not int:
        raise ValueError("human decision refers to a different candidate revision")
    latest = row["decisions"][-1] if row["decisions"] else None
    if value["previous_decision"] != (latest["id"] if latest else None):
        raise ValueError(
            "human decision changed; inspect the latest decision before superseding it"
        )
    publication = row["publication"]
    if publication is None:
        raise ValueError(
            "publish or link the candidate before recording the execution decision"
        )
    target = publication["data"]["issue"]
    if action == "approve":
        if publication["data"]["mode"] != "created":
            raise ValueError(
                "a reference does not extend an existing issue's execution authority; use its existing agreement"
            )
        if (
            latest
            and latest["data"]["human_confirmation"] == value["human_confirmation"]
        ):
            raise ValueError(
                "a new approval after a decision/revision needs a new human instruction, not replayed consent"
            )
        from .supplement import require_final

        raw = require_final(session.gh, target)
        if parse_source(raw.get("body") or "") != source:
            raise ValueError("approval target lost its improvement source")
        if row["candidate"]["data"]["spec"]["unmet_acceptance"]:
            raise ValueError(
                "unfinished source acceptance cannot be approved as a successor"
            )
    else:
        raw = session.gh.issue(target)
    data = {
        "request_key": value["request_key"],
        "request_sha256": requested,
        "action": action,
        "issue": target,
        "human_confirmation": value["human_confirmation"],
        "reason": value["reason"],
        "requirements": contract(raw.get("body") or ""),
        "previous_decision": value["previous_decision"],
    }
    row_event = event(source, "decision", row["revision"], row["last"]["id"], data)
    append(session, row_event)
    stage_stop(session, source, get(session, source.key)[1])
    return {
        "candidate_id": source.candidate_id,
        "decision_id": row_event["id"],
        "state": {
            "approve": "approved",
            "hold": "held",
            "decline": "declined",
            "revoke": "revoked",
        }[action],
        "issue": target,
        "reused": False,
    }


def stage_stop(
    session: ParentSession, source: ImprovementSource, row: dict[str, Any]
) -> None:
    """Commit stop intent before releasing the decision/admission lock."""
    publication = row["publication"]
    if (
        human_state(row) not in {"held", "declined", "revoked"}
        or publication is None
        or publication["data"]["mode"] != "created"
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


def observe(gh: Any, parent: int, publisher: str) -> dict[str, Any] | None:
    raw = gh.issue(parent)
    source = parse_source(raw.get("body") or "")
    if source is None:
        return None
    rows = read_candidates(gh.comments(source.parent), source.parent, publisher)
    candidate = rows.get(source.key)
    if candidate is None:
        raise ValueError(f"{source.candidate_id}: source candidate is unavailable")
    return {"source": source, "candidate": candidate, "issue": raw}


def require(
    session: ParentSession, target: int | None = None, *, fresh: bool = False
) -> dict[str, Any] | None:
    n = target or session.parent
    if not fresh:
        raw = session.gh.issue(n)
        if parse_source(raw.get("body") or "") is None:
            if n == session.parent and any(
                e.get("improvement_approval") for e in session.history()
            ):
                raise ValueError("improvement approval source marker is missing")
            return None
    with session.gh.fresh_reads():
        observed = observe(session.gh, n, session.login)
        if observed is None:
            # Removing a protected source marker does not unbind an existing
            # agreement. Initial proposals remain identifiable by their marker.
            if n == session.parent and any(
                e.get("improvement_approval") for e in session.history()
            ):
                raise ValueError("improvement approval source marker is missing")
            if not fresh:
                raise ValueError("improvement source disappeared during authorization")
            return None
        source, candidate, raw = (
            observed["source"],
            observed["candidate"],
            observed["issue"],
        )
        try:
            decision = approved(candidate, source, n, raw.get("body") or "")
        except ValueError:
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
            raise
        owner = session.store.parents.get(f"{session.gh.repo.casefold()}#{n}", {})
        if owner.get("improvement_stop", {}).get("state") == "stopping":
            raise ValueError(
                f"{source.candidate_id}: previous runtime stop/cleanup is still pending; retry improvement-decide"
            )
        return {
            "parent": source.parent,
            "key": source.key,
            "revision": candidate["revision"],
            "issue": n,
            "decision_id": decision["id"],
        }


def input_state(gh: Any, parent: int, publisher: str) -> dict[str, Any] | None:
    observed = observe(gh, parent, publisher)
    if observed is None:
        return None
    source, candidate = observed["source"], observed["candidate"]
    decisions = candidate["decisions"]
    return {
        "parent": source.parent,
        "key": source.key,
        "revision": candidate["revision"],
        "state": human_state(candidate),
        "decision_id": decisions[-1]["id"] if decisions else None,
    }


def agreement(
    session: ParentSession, decision_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = require(session)
    if binding is None or binding["decision_id"] != decision_id:
        raise ValueError("agree requires the current improvement approval reference")
    candidate = candidates(session, binding["parent"])[binding["key"]]
    spec = candidate["candidate"]["data"]["spec"]
    confirmation = candidate["decisions"][-1]["data"]["human_confirmation"]
    value = {
        "purpose": spec["purpose"],
        "scope": spec["scope"] + "\n\n対象外: " + spec["out_of_scope"],
        "behavior": spec["behavior"],
        "acceptance": "\n".join(spec["acceptance"])
        + "\n\n停止条件: "
        + spec["stop_conditions"],
        "human_confirmation": confirmation["quote"]
        + "\n\n出典: "
        + confirmation["source"],
    }
    return value, binding


def status(session: ParentSession) -> list[dict[str, Any]]:
    result = []
    for fixed_key, row in candidates(session).items():
        source = ImprovementSource(session.parent, fixed_key)
        publication = row["publication"]
        receipt = _receipt(session, source)
        pending = (
            receipt is not None
            and receipt["creation"]["attempted"]
            and publication is None
        )
        target = (
            publication["data"]["issue"]
            if publication
            else (receipt["creation"]["number"] if receipt else None)
        )
        owner = session.store.parents.get(f"{session.gh.repo.casefold()}#{target}", {})
        completion = owner.get("completion", {})
        binding = completion.get("improvement_approval", {})
        accepted = completion.get("state") == "completed" and all(
            binding.get(k) == v
            for k, v in {
                "parent": source.parent,
                "key": source.key,
                "revision": row["revision"],
                "issue": target,
            }.items()
        )
        execution = execution_status(session, target, owner, accepted=accepted)
        decisions = row["decisions"]
        result.append(
            {
                "candidate_id": source.candidate_id,
                "revision": row["revision"],
                "purpose": row["candidate"]["data"]["spec"]["purpose"],
                "publication": publication["data"]["mode"]
                if publication
                else "outcome_unknown"
                if pending
                else "not_requested",
                "issue": target,
                "human_decision": human_state(row),
                "decision_id": decisions[-1]["id"] if decisions else None,
                "execution": execution,
                "record": f"https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{row['comments'][row['last']['id']]}",
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
