"""Structured completion against the existing agreement's acceptance snapshot."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from issues.format import raw_section, visible_lines
from issues.improvement import candidate_key, fields, text

from .execution import authorization, requirement_body
from .policy import digest

if TYPE_CHECKING:
    from .session import ParentSession


def snapshot(body: str) -> dict[str, str]:
    return {
        heading: raw_section(body, heading)
        for heading in ("全体の完了条件", "Acceptance checklist")
        if raw_section(body, heading)
    }


def _criteria(section: str) -> list[dict[str, str]]:
    criteria: list[dict[str, str]] = []
    group: str | None = None
    groups: dict[str, int] = {}
    for line in visible_lines(section):
        heading = re.fullmatch(
            r"### (I-[1-9][0-9]*-[a-z0-9][a-z0-9-]{0,63})[ \t]*", line
        )
        if heading:
            if criteria and group is None:
                raise ValueError(
                    "acceptance snapshot mixes grouped and ungrouped criteria"
                )
            group = heading[1]
            if group in groups:
                raise ValueError("acceptance snapshot repeats a candidate group")
            groups[group] = 0
            continue
        match = re.fullmatch(r"[-*] \[[ xX]\] (\S.*)", line)
        if match:
            identifier = f"ac-{len(criteria) + 1}"
            criterion = match[1].strip()
            if group is not None:
                groups[group] += 1
                identifier = f"ac-{group}-{groups[group]}"
                criterion = f"[{group}] {criterion}"
            criteria.append({"id": identifier, "criterion": criterion})
        elif line.strip():
            # candidate_sections retains indentation within multiline criteria.
            if not criteria or (
                group is not None
                and (
                    groups[group] == 0
                    or not line[0].isspace()
                    or re.match(r"#{1,6}(?:\s|$)", line.lstrip())
                )
            ):
                raise ValueError("acceptance snapshot is not an explicit checklist")
            criteria[-1]["criterion"] += " " + line.strip()
    if any(count == 0 for count in groups.values()):
        raise ValueError("acceptance snapshot has an empty candidate group")
    if not criteria or len({item["criterion"] for item in criteria}) != len(criteria):
        raise ValueError("acceptance snapshot has no unique identifiable criteria")
    return criteria


def requirements(session: ParentSession) -> dict[str, Any]:
    history = session.history()
    basis = next(
        (
            row
            for row in reversed(history)
            if row["event"] in {"agreed", "execution_basis", "basis_revoked"}
        ),
        None,
    )
    if basis is None or basis["event"] == "basis_revoked":
        raise ValueError("completion requires an existing agreement/basis snapshot")
    accepted = basis.get("acceptance_snapshot")
    body = session.gh.issue(session.parent).get("body") or ""
    if accepted is None:
        # A byte-identical legacy observation is recoverable. Do not invent an
        # old acceptance baseline from an edited current issue or migrate all parents.
        expected = (
            basis.get("requirements_sha256")
            if basis["event"] == "agreed"
            else basis["inputs"]["issue"]["body_sha256"]
        )
        if expected != digest(requirement_body(body)):
            raise ValueError(
                "legacy acceptance snapshot is unavailable; do not infer the original criteria"
            )
        accepted = snapshot(body)
    if (
        not isinstance(accepted, dict)
        or set(accepted)
        - {
            "全体の完了条件",
            "Acceptance checklist",
        }
        or any(
            not isinstance(value, str) or not value.strip()
            for value in accepted.values()
        )
    ):
        raise ValueError("invalid acceptance snapshot")
    heading = (
        "全体の完了条件" if "全体の完了条件" in accepted else "Acceptance checklist"
    )
    if heading not in accepted:
        raise ValueError("agreement has no identifiable body acceptance snapshot")
    criteria = _criteria(accepted[heading])
    if _criteria(raw_section(body, heading)) != criteria:
        raise ValueError(
            "acceptance criteria changed since agreement; confirm the changed scope"
        )
    agreement = basis["agreement"] if basis["event"] == "agreed" else basis["basis"]
    items = [{"id": "agreement", "criterion": text(agreement["acceptance"])}]
    items.extend(criteria)
    return {
        "basis_ref": digest(
            {
                "kind": basis["event"],
                "revision": basis["revision"],
                "agreement": agreement,
                "acceptance_snapshot": accepted,
                "improvement_approval": basis.get("improvement_approval"),
            }
        ),
        "criteria": items,
    }


def status(session: ParentSession) -> dict[str, Any]:
    try:
        result = {"available": True, **requirements(session)}
    except ValueError as exc:
        result = {"available": False, "reason": str(exc)}
    applies = result.get("available") is True and session.row.get("completion", {}).get(
        "basis_ref"
    ) == result.get("basis_ref")
    if applies:
        from .improvements import completion_applies, observe

        current = observe(session.gh, session.parent, session.login, session.store)
        saved = session.row["completion"]
        if current is not None:
            applies = all(
                completion_applies(
                    saved,
                    source,
                    candidate,
                    session.parent,
                    current["issue"].get("body") or "",
                )
                for source, candidate in current["approvals"]
            )
        elif saved.get("improvement_approval"):
            applies = False
    return {
        "requirements": result,
        "record": session.row.get("completion"),
        "acceptance": session.row.get("completion", {}).get("state", "unverified")
        if applies
        else "unverified",
        "applies_to_current_basis": applies,
    }


def validate_finish(
    spec: dict[str, Any],
    required: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    parent: int,
) -> None:
    fields(
        spec,
        {
            "schema_version",
            "state",
            "basis_ref",
            "acceptance",
            "remaining_required",
            "improvements",
            "summary",
        },
    )
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["state"] not in ("completed", "incomplete")
    ):
        raise ValueError("invalid finish schema/version/state")
    text(spec["summary"])
    if spec["basis_ref"] != required["basis_ref"]:
        raise ValueError("finish refers to a different agreement/acceptance snapshot")
    if not isinstance(spec["acceptance"], list):
        raise ValueError("finish requires every acceptance result")
    ids = []
    for result in spec["acceptance"]:
        fields(result, {"id", "state", "evidence"})
        ids.append(text(result["id"]))
        text(result["evidence"])
        if result["state"] not in ("met", "unmet", "unverified"):
            raise ValueError("invalid acceptance result state")
    if sorted(ids) != sorted(item["id"] for item in required["criteria"]):
        raise ValueError(
            "finish must cover each agreed acceptance criterion exactly once"
        )
    if not isinstance(spec["remaining_required"], list):
        raise ValueError("remaining_required must be an explicit list")
    for item in spec["remaining_required"]:
        text(item)
    improvements = fields(spec["improvements"], {"reviewed", "candidates"})
    if improvements["reviewed"] is not True or not isinstance(
        improvements["candidates"], list
    ):
        raise ValueError(
            "finish must distinguish candidates reviewed with none from not reviewed"
        )
    keys = [candidate_key(value, parent) for value in improvements["candidates"]]
    if len(set(keys)) != len(keys) or set(keys) != set(candidates):
        raise ValueError(
            "finish must reference every recorded candidate once; no implicit candidate deletion"
        )
    if spec["state"] == "completed" and (
        spec["remaining_required"]
        or any(row["state"] != "met" for row in spec["acceptance"])
        or any(
            c["candidate"]["data"]["spec"]["unmet_acceptance"]
            for c in candidates.values()
        )
    ):
        raise ValueError(
            "unfinished acceptance cannot become completed by moving work to an improvement"
        )


def record(
    session: ParentSession, spec: dict[str, Any], *, prs: list[int] | None = None
) -> dict[str, Any]:
    session.owner()
    auth = authorization(session)
    required = requirements(session)
    from .improvements import candidates

    validate_finish(spec, required, candidates(session), session.parent)
    session.public(spec)
    sha = digest(spec)
    previous = session.row.get("completion")
    if spec["state"] == "completed" and prs is None:
        from .development import verify_completion

        # The shared resolver includes a childless parent's own work. Manual and
        # autonomous completion require the same execution/merge evidence.
        prs = verify_completion(session)
    if spec["state"] == "completed":
        from .notifications import completion as published_completion

        if published_completion(session) is None:
            raise ValueError(
                "completed finish requires a published completion notification"
            )
    if (
        previous
        and previous["finish_sha256"] == sha
        and previous["authorization"] == auth
    ):
        return dict(previous)
    receipt = {
        "schema_version": 1,
        "finish_sha256": sha,
        "state": spec["state"],
        "basis_ref": required["basis_ref"],
        "authorization": auth,
        "prs": prs or [],
        "candidate_ids": spec["improvements"]["candidates"],
    }
    from .improvements import require

    binding = require(session)
    if binding is not None:
        receipt["improvement_approval"] = binding
    description = [
        f"全体受入: {spec['state']}",
        spec["summary"],
        *[f"- {r['id']}: {r['state']} — {r['evidence']}" for r in spec["acceptance"]],
        "残る必須作業: " + (" / ".join(spec["remaining_required"]) or "なし。"),
        "次回改善: " + (", ".join(spec["improvements"]["candidates"]) or "なし。"),
        "改善の起票・人間判断・実行はstatusのimprovementsを参照。今回の成果とは別に扱います。",
    ]
    session.post(
        {"event": "completion", "generation": session.row["generation"], **receipt},
        "\n\n".join(description),
    )
    session.row["completion"] = receipt
    session.store.save()
    return receipt


def submit(
    session: ParentSession, submission: dict[str, Any], *, managed_report: bool
) -> dict[str, Any]:
    """Publish a completion and consume the same acceptance input on either path."""
    from .development import report, verify_completion
    from .improvements import candidates
    from .notifications import notify

    finish = dict(submission)
    evidence = finish.pop("evidence", None)
    artifacts = finish.pop("artifacts", [])
    resolves = finish.pop("resolves", [])
    finish["schema_version"] = 1
    if submission.get("schema_version") != 2 or finish.get("state") != "completed":
        raise ValueError("combined completion requires version 2 and state=completed")
    signature = digest(submission)
    if not managed_report and session.row.get("status") == "ended":
        if session.row.get("completion_submission") != signature:
            raise ValueError("ended parent has a different completion submission")
        return release(session, finish)
    session.owner()
    session.public(submission)
    auth = authorization(session)
    required = requirements(session)
    validate_finish(finish, required, candidates(session), session.parent)
    prs = verify_completion(session)
    from .notifications import strings

    urls = strings(artifacts, "artifacts", empty=True)
    urls = list(
        dict.fromkeys(
            [*urls, *(f"https://github.com/{session.gh.repo}/pull/{n}" for n in prs)]
        )
    )
    criteria = {item["id"]: item["criterion"] for item in required["criteria"]}
    notice = {
        "schema_version": 1,
        "key": "completion",
        "kind": "completion",
        "summary": finish["summary"],
        "evidence": evidence,
        "resolves": resolves,
        "details": {
            "achievements": [
                f"{criteria[item['id']]} — {item['evidence']}"
                for item in finish["acceptance"]
            ],
            "artifacts": urls,
            "remaining": finish["remaining_required"],
        },
    }
    published = notify(session, notice)
    if authorization(session) != auth:
        raise ValueError("completion authorization changed during notification")
    session.row["completion_submission"] = signature
    session.store.save()
    if managed_report:
        return report(
            session,
            {
                "schema_version": 2,
                "state": "completed",
                "reason": finish["summary"],
                "resume_condition": "全条件を受入済み",
                "evidence": published["url"],
                "finish": finish,
            },
            verified_prs=prs,
        )
    return release(session, finish, verified_prs=prs)


def release(
    session: ParentSession,
    spec: dict[str, Any],
    *,
    verified_prs: list[int] | None = None,
) -> dict[str, Any]:
    previous = session.row.get("completion", {})
    if session.row.get("status") == "ended" and previous.get("finish_sha256") == digest(
        spec
    ):
        # Replay repairs no execution and never revives the ended source owner.
        return dict(session.status())
    record(session, spec, prs=verified_prs)
    return dict(session.lifecycle("ended", spec["summary"]))


def require_report_receipt(
    row: dict[str, Any],
    job: dict[str, Any],
    response: dict[str, Any],
    current_authorization: dict[str, Any] | None = None,
) -> None:
    saved = row.get("development_result", {}).get("completion_receipt")
    completion = row.get("completion")
    if (
        not saved
        or not completion
        or completion.get("state") != "completed"
        or saved
        != {
            "job": job["id"],
            "generation": row["generation"],
            "authorization": current_authorization
            if current_authorization is not None
            else job["authorization"],
            "response_sha256": digest(response),
            "finish_sha256": completion["finish_sha256"],
        }
    ):
        raise ValueError(
            "completed final requires the matching workflow-report acceptance receipt"
        )
