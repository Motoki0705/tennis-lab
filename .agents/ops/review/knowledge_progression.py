"""Register the parent's evidence and targets for a records-only merge."""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lib.ops_review import knowledge_evidence, knowledge_records
from lib.ops_review.merge_gate_impl import (
    controller_lock,
    knowledge_states,
    registered,
    state_path,
)
from lib.ops_review.progression import KNOWLEDGE_MERGE, new_state, read, save
from lib.ops_review.review_locks import pr_lease
from parent.execution import authorization
from parent.policy import closing_issues, dependencies, object_value
from parent.review_handoff import require_followup_ended, runner_ended
from parent.work_targets import require_separate_self_prs

if TYPE_CHECKING:
    from lib.ops_review.progression_backend import LiveBackend
    from parent.github import GitHub
    from parent.session import ParentSession

PRE_MERGE_STAGES = {"ci_wait", "dependency_wait"}


def related_targets(
    gh: GitHub, pr: dict[str, Any], closing: set[int], work: set[int]
) -> tuple[set[int], int | None]:
    n = pr["number"]
    if closing - work:
        raise ValueError(
            f"PR #{n}: closing Issues outside the current work targets: {sorted(closing - work)}"
        )
    refs = knowledge_records.references(pr.get("body") or "", gh.repo)
    linked = (refs | closing) & work
    mentioned_prs = refs - work - {n}
    implementations = []
    if linked and mentioned_prs:
        for candidate in gh.pulls(linked):
            if candidate["number"] in mentioned_prs:
                issues = closing_issues(candidate.get("body") or "", gh.repo) & linked
                if issues:
                    implementations.append((candidate, issues))
    if closing:
        if implementations:
            raise ValueError(f"PR #{n}: a companion knowledge PR cannot close Issues")
        return closing, None
    if len(implementations) != 1:
        raise ValueError(f"PR #{n}: reference the work Issue and its implementation PR")
    implementation, issues = implementations[0]
    if (
        implementation.get("state") != "open"
        and implementation.get("merged") is not True
    ):
        raise ValueError(
            f"PR #{n}: referenced implementation PR is neither open nor merged"
        )
    return issues, implementation["number"]


def transfer_parked(session: ParentSession, entries: list[dict[str, Any]]) -> None:
    """Retire reservations for parked targets that now require formal review."""
    selected = {str(entry["number"]): entry for entry in entries}
    transfers = {
        state["id"]: active
        for state in knowledge_states(session.store, session.row)
        if (
            active := [
                n
                for n, target in state["targets"].items()
                if n in selected and target["stage"] != "transferred"
            ]
        )
    }
    if not transfers:
        return
    repo = session.gh.repo
    gh = session.gh.detached()
    states = {}
    with ExitStack() as locks:
        # Match controller startup order. All acquisitions are nonblocking.
        for identifier in sorted(transfers):
            path = state_path(session.store, identifier)
            locks.enter_context(controller_lock(path))
            states[identifier] = read(path)
        for identifier, numbers in transfers.items():
            for n in numbers:
                target = states[identifier]["targets"][n]
                receipt = locks.enter_context(
                    pr_lease(session.store.directory, repo, int(n))
                )
                if (
                    target["stage"] != "parent_wait"
                    or target.get("resume_stage") not in PRE_MERGE_STAGES
                    or {"merge_head", "merge_base", "merge_error", "merge_commit"}
                    & target.keys()
                    or (receipt.exists() and read(receipt)["progression"] != identifier)
                ):
                    raise ValueError(
                        f"PR #{n}: only a parked knowledge merge with no merge intent "
                        "can transfer to formal review"
                    )
        with session.gh.io_scope():
            for identifier, numbers in transfers.items():
                for n in numbers:
                    pr = object_value(gh.get(f"repos/{repo}/pulls/{n}"))
                    target = states[identifier]["targets"][n]
                    if (
                        pr.get("number") != int(n)
                        or pr.get("state") != "open"
                        or pr.get("merged") is not False
                        or pr.get("merged_at") is not None
                        or pr["head"]["ref"] != target["branch"]
                        or pr["head"]["sha"] != selected[n]["head_sha"]
                        or pr["base"]["ref"] != target["base_ref"]
                        or not set(selected[n]["issues"])
                        <= closing_issues(pr.get("body") or "", repo)
                    ):
                        raise ValueError(
                            f"PR #{n}: open unmerged PR changed during transfer"
                        )
                    scope, _ = knowledge_records.assess(Path.cwd(), repo, pr)
                    if scope.records_only:
                        raise ValueError(
                            f"PR #{n}: knowledge records only; continue its dedicated merge"
                        )
        for identifier, numbers in transfers.items():
            state = states[identifier]
            for n in numbers:
                target = state["targets"][n]
                target.pop("resume_stage")
                target.update(
                    stage="transferred", reason="transferred_to_formal_review"
                )
            stages = {t["stage"] for t in state["targets"].values()}
            if stages == {"transferred"}:
                state["status"] = "transferred"
            elif stages <= {"completed", "transferred"}:
                state["status"] = "completed"
            save(state_path(session.store, identifier), state)


def register(
    session: ParentSession, numbers: list[int], evidence: str
) -> dict[str, Any]:
    proofs = knowledge_evidence.read_evidence(evidence, numbers)
    session.gh.targeted_history = True
    row = session.agreed()
    auth = authorization(session)
    if previous := registered(session, numbers, kind=KNOWLEDGE_MERGE):
        return previous
    repo = session.gh.repo
    root = Path.cwd()
    gh = session.gh.detached()
    parent, items = session.work_items()
    work = set(items)
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
            scope, comparison = knowledge_records.assess(root, repo, pr)
            if comparison is None:
                raise ValueError(
                    f"PR #{n}: {scope.reason}; use the formal review path for ordinary PRs"
                )
            verified = knowledge_evidence.bind(root, pr, comparison, proofs[n])
            closing = knowledge_records.closing_targets(gh, root, pr, comparison)
        related, implementation_pr = related_targets(session.gh, pr, closing, work)
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
                "verified_evidence": verified,
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
    state.update(checkout=str(root), options={})
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
    pending = []
    for target in state["targets"].values():
        n = target["number"]
        stage = (
            target.get("resume_stage")
            if target["stage"] == "parent_wait"
            else target["stage"]
        )
        if stage in PRE_MERGE_STAGES and (prs is None or n in prs):
            pending.append(target)
    if not pending:
        return

    proofs = knowledge_evidence.read_evidence(evidence, [t["number"] for t in pending])
    root = Path(state["checkout"])
    for target in pending:
        n = target["number"]
        pr = backend.pr(n)
        if reason := backend.shape(target, pr):
            raise ValueError(f"PR #{n}: {reason}")
        scope, comparison = knowledge_records.assess(root, backend.repo, pr)
        if comparison is None:
            raise ValueError(f"PR #{n}: {scope.reason}")
        verified = knowledge_evidence.bind(root, pr, comparison, proofs[n])
        closing = knowledge_records.closing_targets(backend.gh, root, pr, comparison)
        if closing != set(target["issues"]):
            raise ValueError(f"PR #{n}: GitHub closing targets changed")
        related, implementation = related_targets(
            backend.gh, pr, closing, set(target["related_issues"])
        )
        if related != set(target["related_issues"]) or implementation != target.get(
            "implementation_pr"
        ):
            raise ValueError(f"PR #{n}: knowledge PR references changed")
        target.update(evidence_head=comparison.head, verified_evidence=verified)
