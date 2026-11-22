"""Register the parent's evidence and targets for a records-only merge."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lib.ops_review import knowledge_records
from lib.ops_review.merge_gate_impl import registered, state_path
from lib.ops_review.progression import KNOWLEDGE_MERGE, new_state, save
from parent.execution import authorization
from parent.policy import closing_issues, dependencies, nonempty, object_value
from parent.review_handoff import require_followup_ended, runner_ended
from parent.work_targets import require_separate_self_prs

if TYPE_CHECKING:
    from lib.ops_review.progression_backend import LiveBackend
    from parent.session import ParentSession


def register(
    session: ParentSession, numbers: list[int], evidence: str
) -> dict[str, Any]:
    evidence = nonempty(evidence, "knowledge PR acceptance and execution evidence")
    session.gh.targeted_history = True
    row = session.agreed()
    auth = authorization(session)
    if previous := registered(session, numbers, kind=KNOWLEDGE_MERGE):
        return previous
    repo = session.gh.repo
    parent, items = session.work_items()
    entries = []
    for n in sorted(numbers):
        pr = object_value(session.gh.get(f"repos/{repo}/pulls/{n}"))
        if (
            pr.get("number") != n
            or pr.get("state") != "open"
            or pr.get("draft") is not False
            or pr["base"]["ref"] != "main"
        ):
            raise ValueError(
                f"PR #{n}: knowledge merge requires an open ready PR to main"
            )
        require_separate_self_prs(parent, [pr], repo)
        with session.gh.io_scope():
            scope, comparison = knowledge_records.assess(Path.cwd(), repo, pr)
        if comparison is None:
            raise ValueError(
                f"PR #{n}: {scope.reason}; use the formal review path for ordinary PRs"
            )
        body = pr.get("body") or ""
        closing = closing_issues(body, repo)
        if closing - items.keys():
            raise ValueError(
                f"PR #{n}: closing Issue is outside the current work targets"
            )
        implementation_pr = None
        related = closing
        if not closing:
            refs = knowledge_records.references(body, repo)
            implementation = [
                p
                for p in session.gh.pulls(refs & items.keys())
                if p["number"] != n
                and p["number"] in refs
                and (p.get("state") == "open" or p.get("merged") is True)
            ]
            if len(implementation) != 1:
                raise ValueError(
                    f"PR #{n}: reference the work Issue and its implementation PR"
                )
            implementation_pr = implementation[0]["number"]
            related = (
                closing_issues(implementation[0].get("body") or "", repo)
                & refs
                & items.keys()
            )
            if not related:
                raise ValueError(
                    f"PR #{n}: implementation PR does not close a referenced work Issue"
                )
        for child in sorted(closing):
            require_followup_ended(row, child)
            runner_ended(session, child)
        deps = sorted(
            {
                d
                for child in closing
                for d in row["tasks"]
                .get(str(child), {})
                .get("dependencies", dependencies(items[child].body))
            }
            - closing
        )
        entries.append(
            {
                "number": n,
                "issues": sorted(closing),
                "related_issues": sorted(related),
                "implementation_pr": implementation_pr,
                "dependencies": [{"issue": d} for d in deps],
                "worker": None,
                "head_sha": comparison.head,
                "branch": pr["head"]["ref"],
                "base_ref": pr["base"]["ref"],
            }
        )
    closing_targets = sorted({n for entry in entries for n in entry["issues"]})
    if closing_targets:
        session.record_results(closing_targets, evidence)
    if authorization(session) != auth:
        raise ValueError(
            "knowledge merge execution authority changed during registration"
        )
    if previous := registered(session, numbers, kind=KNOWLEDGE_MERGE):
        return previous
    identifier = f"parent-{session.parent}-{uuid.uuid4().hex}"
    request = {
        "schema_version": 4,
        "kind": KNOWLEDGE_MERGE,
        "review_run_id": identifier,
        "repo": repo,
        "parent_issue": session.parent,
        "authorization": auth,
        "evidence": evidence,
        "prs": entries,
    }
    state = new_state(request, owner=row["session"], generation=row["generation"])
    state.update(checkout=str(Path.cwd()), options={})
    path = state_path(session.store, identifier)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(path, state)
    row.setdefault("knowledge_merges", []).append(identifier)
    row["latest_progression_id"] = identifier
    session.store.save()
    from decision_logs.review_capture import bind_request

    bind_request(identifier)
    return {"review_run_id": identifier, "reused": False, "status": "requested"}


def refresh_evidence(
    backend: LiveBackend, state: dict[str, Any], prs: list[int] | None, evidence: str
) -> None:
    for target in state["targets"].values():
        n = target["number"]
        stage = (
            target.get("resume_stage")
            if target["stage"] == "parent_wait"
            else target["stage"]
        )
        if stage not in {"ci_wait", "dependency_wait"} or (
            prs is not None and n not in prs
        ):
            continue
        pr = backend.pr(n)
        head = pr["head"]["sha"]
        if head == target.get("evidence_head", target["head_sha"]):
            continue
        if reason := backend.shape(target, pr):
            raise ValueError(f"PR #{n}: {reason}")
        scope, comparison = knowledge_records.assess(
            Path(state["checkout"]), backend.repo, pr
        )
        if comparison is None:
            raise ValueError(f"PR #{n}: {scope.reason}")
        if head not in evidence:
            raise ValueError(
                f"PR #{n}: include new HEAD {head} and its evidence in the continuation"
            )
        target["evidence_head"] = head
