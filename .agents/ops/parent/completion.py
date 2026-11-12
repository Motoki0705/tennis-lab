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


def _criteria(section: str) -> list[str]:
    criteria: list[str] = []
    for line in visible_lines(section):
        match = re.fullmatch(r"[-*] \[[ xX]\] (\S.*)", line)
        if match:
            criteria.append(match[1].strip())
        elif line.strip():
            if not criteria:
                raise ValueError("acceptance snapshot is not an explicit checklist")
            criteria[-1] += " " + line.strip()
    if not criteria or len(set(criteria)) != len(criteria):
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
    items.extend(
        {"id": f"ac-{i}", "criterion": criterion}
        for i, criterion in enumerate(criteria, 1)
    )
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


def release(session: ParentSession, spec: dict[str, Any]) -> dict[str, Any]:
    previous = session.row.get("completion", {})
    if session.row.get("status") == "ended" and previous.get("finish_sha256") == digest(
        spec
    ):
        # Replay repairs no execution and never revives the ended source owner.
        return dict(session.status())
    record(session, spec)
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
