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
            evidence.append(
                {
                    "kind": "public",
                    "url": f"https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{response['id']}",
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
