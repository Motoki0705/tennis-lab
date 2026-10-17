"""Real parent edit services and synthetic response streams for both label tasks."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, FrameRange
from src.tasks.base.annotation.editing.contracts import EditReceipt, Revision
from src.tasks.base.annotation.handoff import read_handoff
from src.tasks.base.annotation.telemetry.capture import (
    capture,
    iter_responses,
    verify_report,
)
from src.tasks.base.annotation.telemetry.contracts import ResponsePage
from src.tasks.base.annotation.telemetry.storage import retained_report
from src.tasks.base.annotation.workers.contracts import ExitReceipt, Launch
from src.tasks.base.annotation.workers.review.contracts import HandoffRequest
from tests.support.tasks.ball_detection.review_worker import (
    ReviewSession as BallSession,
)
from tests.support.tasks.player_detection.review_worker import (
    ReviewSession as PlayerSession,
)


def event(kind: str, **payload: Any) -> dict[str, Any]:
    return {"type": kind, "timestamp": "2026-01-01T00:00:00Z", "payload": payload}


@pytest.fixture(params=["ball", "player"])
def session(
    tmp_path: Path, request: pytest.FixtureRequest
) -> BallSession | PlayerSession:
    return (
        BallSession(tmp_path / "a")
        if request.param == "ball"
        else PlayerSession(tmp_path / "a")
    )


def receipt_reference(
    session: BallSession | PlayerSession, receipt: EditReceipt
) -> ArtifactRef:
    return annotation_reference(
        session.env.layout,
        session.env.layout.artifact_path(receipt.command).parent / "receipt.json",
    )


def test_response_links_exact_image_edit_and_handoff_without_loading_rollout_again(
    session: BallSession | PlayerSession,
) -> None:
    env, worker = session.env, session.worker
    image = session.image(1)
    edit = session.confirm(image, 1, 2)
    edit_ref = receipt_reference(session, edit)
    handoff_ref = session.call(
        "review_handoff", HandoffRequest(scope=FrameRange(start=1, stop=2))
    )
    handoff = read_handoff(env.layout, handoff_ref)
    launch = read_document(env.layout, env.actor, Launch)
    execution = write_document(
        env.layout,
        session.directory / "exit.json",
        ExitReceipt(
            launch_digest=env.actor.digest,
            exit_code=0,
            status="succeeded",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:01Z",
            elapsed_seconds=1.0,
            events=env.blob,
            errors=env.blob,
            thread_id="11111111-1111-1111-1111-111111111111",
            reason=None,
        ),
    )
    rows = [
        event(
            "session_meta",
            id="11111111-1111-1111-1111-111111111111",
            cwd=str(session.directory / "work"),
        ),
        event("turn_context", model=env.plan.agent_model),
    ]
    rows.extend(
        [
            event(
                "response_item",
                type="custom_tool_call",
                call_id="image",
                name="view_image",
                input="view",
            ),
            event(
                "response_item",
                type="custom_tool_call_output",
                call_id="image",
                output=[
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(
                            env.layout.artifact_path(image.image).read_bytes()
                        ).decode(),
                    }
                ],
            ),
            event(
                "token_usage_record",
                response_id="seen-image",
                usage={"input_tokens": 100, "output_tokens": 20},
            ),
        ]
    )
    for category, reference, document in (
        ("review_edit", edit_ref, edit),
        ("review_handoff", handoff_ref, handoff),
    ):
        output = json.dumps(
            {
                "operation": category,
                "receipt": reference.model_dump(mode="json"),
                "result": document.model_dump(mode="json"),
            }
        )
        rows.extend(
            [
                event(
                    "response_item",
                    type="function_call",
                    call_id=category,
                    name="exec_command",
                    arguments=f"python -m src.tasks.base.annotation.workers.review --operation {category}",
                ),
                event(
                    "response_item",
                    type="function_call_output",
                    call_id=category,
                    output=json.dumps(
                        {"output": "Process exited with code 0\nOutput:\n" + output}
                    ),
                ),
            ]
        )
    rows.append(
        event(
            "token_usage_record",
            response_id="received-changes",
            usage={"input_tokens": 200, "output_tokens": 20},
        )
    )
    rollout = session.directory / "rollout.jsonl"
    rollout.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ref = capture(
        env.layout,
        env.actor,
        execution,
        (image,),
        rollout=rollout,
        classifier=worker.classify_tool,
        classifier_id=f"{worker.name}.{worker.version}",
        evidence_resolver=lambda category, text: worker.tool_evidence(
            env.layout, env.actor, category, text
        ),
    )
    records = list(iter_responses(env.layout, ref))
    assert records[0].tool_outputs[0].images[0].image == image.image
    assert records[0].tool_outputs[0].evidence is None
    assert records[1].response_id == "received-changes"
    linked = [output.evidence for output in records[1].tool_outputs]
    assert all(item is not None for item in linked)
    assert [item.artifacts for item in linked if item is not None] == [
        (edit_ref,),
        (handoff_ref,),
    ]
    retained = retained_report(env.layout, ref)
    before = read_document(env.layout, edit.previous, Revision)
    after = read_document(env.layout, edit.revision, Revision)
    assert {
        edit_ref,
        edit.command,
        edit.previous,
        edit.revision,
        before.payload,
        after.payload,
        handoff_ref,
    }.issubset(retained)
    report = verify_report(env.layout, ref)
    page = read_document(env.layout, report.pages[0], ResponsePage)
    html = env.layout.artifact_path(page.entries[1].page).read_text()
    assert edit_ref.digest in html and handoff_ref.digest in html
    rollout.unlink()
    assert list(iter_responses(env.layout, ref)) == records
    assert launch.request.payload == handoff.context
    # A delayed exec reply is classified by its polling call, not the initial CLI.
    delayed = worker.tool_evidence(
        env.layout,
        env.actor,
        "inspection",
        json.dumps(
            {
                "operation": "review_handoff",
                "receipt": handoff_ref.model_dump(mode="json"),
                "result": handoff.model_dump(mode="json"),
            }
        ),
    )
    assert delayed is not None and delayed.artifacts == (handoff_ref,)
    assert after.payload in delayed.dependencies
    env.layout.artifact_path(before.payload).write_text("modified old annotations")
    with pytest.raises(ValueError, match="changed"):
        verify_report(env.layout, ref)


def test_failed_or_truncated_edit_output_is_explicitly_unlinked(
    session: BallSession | PlayerSession,
) -> None:
    result = session.worker.tool_evidence(
        session.env.layout,
        session.env.actor,
        "review_edit",
        "Traceback: RevisionConflict",
    )
    assert result is not None and result.artifacts == ()
    assert result.unavailable_reason is not None


@pytest.mark.parametrize("damage", ["actor", "summary", "uncommitted"])
def test_edit_link_rejects_foreign_actor_forged_summary_and_uncommitted_edit(
    session: BallSession | PlayerSession,
    damage: str,
) -> None:
    env = session.env
    image = session.image(1)
    edit = session.confirm(image, 1, 2)
    edit_ref = receipt_reference(session, edit)
    result = edit.model_dump(mode="json")
    actor = env.actor
    if damage == "actor":
        actor = env.new_actor("different-attempt")
    elif damage == "summary":
        result["summary"]["changed_rows"] += 1
    else:
        with env.store.registry.transaction() as db:
            db.execute("DELETE FROM annotation_edits")
    with pytest.raises(ValueError, match="attempt|summary|committed"):
        session.worker.tool_evidence(
            env.layout,
            actor,
            "review_edit",
            json.dumps(
                {
                    "operation": "review_edit",
                    "receipt": edit_ref.model_dump(mode="json"),
                    "result": result,
                }
            ),
        )
