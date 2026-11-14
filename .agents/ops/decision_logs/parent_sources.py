"""References to canonical parent comments and explicit verification records."""

from __future__ import annotations

from typing import Any

from worker_logs.schema import sha

from .capture import current, source


def comment(
    session: Any, event: dict[str, Any], body: str, response: dict[str, Any]
) -> None:
    op = current()
    if op is None:
        return
    try:
        evidence = []
        if type(response.get("id")) is int:
            from parent.history import comment_url

            evidence.append(
                {
                    "kind": "public",
                    "url": comment_url(session.gh.repo, session.parent, response),
                    "revision": None,
                    "content_sha256": sha(body.encode()),
                }
            )
        with op.transaction():
            op.result(
                {"event": event, "body_sha256": sha(body.encode())},
                artifact="parent issue comment",
                status="success",
                summary=f"parent comment recorded: event={event.get('event')}; state={event.get('state')}",
                evidence=evidence,
            )
    except Exception:
        op.mark_failed(lost_source="parent.comment")


def verification(session: Any, child: int) -> None:
    op = current()
    if op is None:
        return
    try:
        proof = session.row["tasks"][str(child)]["verification"]
        targets = op.row.get("verification_targets", [])
        if not targets:
            from pathlib import Path

            from .parent_commands import capture_for_runtime

            task = session.row["tasks"][str(child)]
            runtime = None
            if task.get("worker_runs"):
                runtime = str(
                    (
                        session.store.directory
                        / Path(task["worker_runs"][-1]["run_dir"])
                    ).resolve()
                )
            elif task.get("preflight", {}).get("turns"):
                runtime = task["preflight"]["turns"][-1].get("source")
            if runtime:
                targets = [
                    r["id"]
                    for r in capture_for_runtime(
                        session.store.directory,
                        session.gh.repo,
                        session.parent,
                        runtime,
                    )
                    if r["data"]["status"] != "unknown"
                ]
        with op.transaction():
            op.result(
                proof,
                artifact="saved parent verification",
                status="success",
                observer="parent_verification",
                verifies=targets,
                summary=f"parent.record-result: PRs={proof['prs']}; "
                + proof["evidence"].splitlines()[0],
                evidence=[
                    source(
                        "parent.record-result",
                        "registry task verification and child issue evidence",
                        proof,
                    )
                ],
            )
    except Exception:
        op.mark_failed(lost_source="parent.verification")


def improvement_candidate(
    session: Any, row: dict[str, Any], comment_id: int
) -> dict[str, Any]:
    """Capture the observation and reference the single recorded splitting reason."""
    op = current()
    if op is None:
        return {}
    try:
        from issues.improvement import ImprovementSource, render_event
        from parent.improvements import comment_url

        from .journal import Journal
        from .parent_commands import decide_value

        url = comment_url(session, session.parent, comment_id)
        evidence = {
            "kind": "public",
            "url": url,
            "revision": None,
            "content_sha256": sha(render_event(row).encode()),
        }
        recorded = [
            e["record"]
            for e in Journal(
                session.store.directory, session.gh.repo, session.parent
            ).read()["entries"]
            if e["event"] == "capture"
        ]
        decisions = [
            r
            for r in recorded
            if r["kind"] == "decision" and evidence in r["data"]["source"]["evidence"]
        ]
        if decisions:
            saved = decisions[-1]
            return {
                "decision_id": saved["id"],
                "plan_ids": [p["id"] for p in saved["data"]["next_actions"]],
            }
        observations = [
            r
            for r in recorded
            if r["kind"] == "result" and evidence in r["data"]["evidence"]
        ]
        with op.transaction():
            result_id = (
                observations[0]["id"]
                if observations
                else op.result(
                    {"candidate_event": row["id"]},
                    artifact="improvement observation",
                    status="success",
                    summary=f"Observed improvement candidate I-{session.parent}-{row['key']}; execution not authorized",
                    evidence=[evidence],
                )
            )
        candidate_id = ImprovementSource(session.parent, row["key"]).candidate_id
        return decide_value(
            session,
            {
                "inputs": [result_id],
                "evaluation": "handoff",
                "reason_ref": {
                    "kind": "improvement",
                    "key": row["key"],
                    "event_id": row["id"],
                },
                "next_actions": []
                if row["data"]["legacy"]
                else [
                    {
                        "operation": "publish",
                        "summary": "改善提案Issueへ列挙して人間判断へ渡す",
                        "actor": "parent",
                        "target": candidate_id,
                        "instruction": "記録済み候補を改善提案Issueへ列挙する。",
                        "conditions": "元親の必須作業ではないことを確認する。",
                        "expected_result": "人間判断待ちの改善提案Issue。",
                    }
                ],
                "source_kind": "parent_choice",
            },
        )
    except Exception:
        op.mark_failed(lost_source="parent.improvement-decision")
        return {}


def improvement_approval(session: Any, binding: dict[str, Any]) -> None:
    """The successor observes approval; never publish into an ended source's graph."""
    op = current()
    if op is None:
        return
    try:
        from issues.improvement import render_event
        from parent.improvements import candidates, comment_url

        recorded = candidates(session, binding["parent"])
        evidence = []
        for approval in binding.get("approvals", [binding]):
            candidate = recorded[approval["key"]]
            decision = candidate["decisions"][-1]
            comment_id = candidate["comments"][decision["id"]]
            evidence.append(
                {
                    "kind": "public",
                    "url": comment_url(session, binding["parent"], comment_id),
                    "revision": None,
                    "content_sha256": sha(render_event(decision).encode()),
                }
            )
        with op.transaction():
            op.result(
                binding,
                artifact="observed human improvement approval",
                status="success",
                summary=f"Human approval observed for source #{binding['parent']} ({len(evidence)} candidates); execution not started",
                evidence=evidence,
            )
    except Exception:
        op.mark_failed(lost_source="parent.improvement-approval")
