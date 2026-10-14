from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, save_plan
from src.tasks.base.annotation.telemetry.capture import (
    capture,
    iter_responses,
    locate_rollout,
    presented_images,
    verify_report,
)
from src.tasks.base.annotation.telemetry.contracts import (
    RenderedImage,
    ResponsePage,
)
from src.tasks.base.annotation.workers.contracts import (
    ExitReceipt,
    Launch,
    PreparedWorker,
    WorkRequest,
)


def event(kind: str, **payload: Any) -> dict[str, Any]:
    return {"type": kind, "timestamp": "2026-10-14T01:00:00Z", "payload": payload}


def usage(response: str | None, **changes: Any) -> dict[str, Any]:
    values = dict(
        input_tokens=100,
        cached_input_tokens=40,
        output_tokens=10,
        reasoning_output_tokens=6,
        total_tokens=110,
    )
    values.update(changes)
    return event("token_usage_record", response_id=response, usage=values)


def picture(color: str = "red") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (10, 8), color).save(output, "PNG")
    return output.getvalue()


def image_block(data: bytes) -> dict[str, str]:
    return {
        "type": "input_image",
        "detail": "original",
        "image_url": "data:image/png;base64," + base64.b64encode(data).decode(),
    }


@dataclass
class Fixture:
    layout: AnnotationLayout
    launch: ArtifactRef
    execution: ArtifactRef
    rollout: Path
    catalog: tuple[RenderedImage, ...]
    prefix: list[dict[str, Any]]

    def run(self, *items: dict[str, Any], **options: Any) -> ArtifactRef:
        self.rollout.write_text(
            "".join(json.dumps(item) + "\n" for item in [*self.prefix, *items])
        )
        return capture(
            self.layout,
            self.launch,
            self.execution,
            self.catalog,
            rollout=self.rollout,
            **options,
        )

    def tool(
        self, call: str, *, images: tuple[bytes, ...] = ()
    ) -> list[dict[str, Any]]:
        return [
            event(
                "response_item",
                type="custom_tool_call",
                call_id=call,
                name="exec",
                input="render or edit",
            ),
            event(
                "response_item",
                type="custom_tool_call_output",
                call_id=call,
                output=[
                    {"type": "input_text", "text": "表示済"},
                    *[image_block(data) for data in images],
                ],
            ),
        ]


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    ref = write_asset(
        layout, layout.path("campaigns", "inputs", "input.txt"), b"fixture"
    )
    raw = ref.model_dump()
    plan = CampaignPlan.model_validate_json(
        json.dumps(
            {
                "campaign_id": "fixture",
                "phase": "preview",
                "annotation_root": str(layout.root),
                "counts": {
                    "channels": 1,
                    "videos_per_channel": 1,
                    "max_clips_per_video": 1,
                },
                "selection": raw,
                "preparation": raw,
                "code_commit": "a" * 40,
                "code_diff_digest": "b" * 64,
                "agent_model": "gpt-6.1-sol",
                "agent_effort": "high",
                "resources": {
                    "codex_workers": 1,
                    "additional_storage_bytes": 1024**3,
                    "minimum_free_bytes": 1,
                    "total_pss_bytes": 1024**3,
                    "gpu_seconds": 60,
                    "stages": [
                        {
                            "stage": "review",
                            "kind": "codex",
                            "parallel": 1,
                            "threads_per_process": 1,
                        }
                    ],
                },
                "tasks": [
                    {
                        "task": "fixture",
                        "model": {
                            "task": "fixture",
                            "version": "v1",
                            "training_weights": raw,
                            "inference_weights": raw,
                            "configuration": raw,
                        },
                        "inference": raw,
                        "review": None,
                        "learning": None,
                        "products": [
                            {
                                "product": "fixture",
                                "dataset_task": "fixture",
                                "dataset_version": "v1",
                                "component": "labels",
                                "base_snapshot": None,
                                "generation_policy": raw,
                            }
                        ],
                    }
                ],
            }
        )
    )
    plan_ref = annotation_reference(layout, save_plan(layout, plan))
    directory = layout.path("campaigns", "fixture", "workers", "attempt-1")
    launch = Launch(
        work_id="c" * 64,
        attempt="attempt-1",
        plan=plan_ref,
        request=WorkRequest(
            adapter="fixture",
            adapter_version="v1",
            stage="review",
            subject_id="c" * 64,
            payload=ref,
            inputs=(ref,),
        ),
        prepared=PreparedWorker(prompt=ref),
        directory=str(directory),
        code_root=str(tmp_path),
        codex_binary="codex",
        codex_home=str(tmp_path / "codex"),
        timeout_seconds=60,
    )
    launch_ref = write_document(layout, directory / "launch.json", launch)
    execution = ExitReceipt(
        launch_digest=launch_ref.digest,
        exit_code=0,
        status="succeeded",
        started_at="2026-10-14T00:00:00Z",
        ended_at="2026-10-14T00:01:00Z",
        elapsed_seconds=60.0,
        events=ref,
        errors=ref,
        thread_id="11111111-1111-1111-1111-111111111111",
        reason=None,
    )
    execution_ref = write_document(layout, directory / "exit.json", execution)
    catalog = (
        RenderedImage(
            image=write_asset(layout, directory / "original.png", picture()),
            metadata=write_asset(
                layout,
                directory / "render.json",
                b'{"frame":10,"pts":0.4,"annotation_revision":"v1"}\n',
            ),
        ),
    )
    prefix = [
        event(
            "session_meta",
            id=execution.thread_id,
            cwd=str(directory / "work"),
            cli_version="fixture",
            model_provider="openai",
        ),
        event("turn_context", model=plan.agent_model),
    ]
    return Fixture(
        layout, launch_ref, execution_ref, tmp_path / "rollout.jsonl", catalog, prefix
    )


def test_distinct_responses_receive_preceding_outputs_and_usage_is_not_notifications(
    fixture: Fixture,
) -> None:
    ref = fixture.run(
        event("event_msg", type="token_count", info={"model_context_window": 400000}),
        usage("response-1"),
        *fixture.tool("view", images=(picture(),)),
        usage("response-1"),
        event("event_msg", type="token_count", info={"model_context_window": 400000}),
        usage("response-2"),
        usage("response-2"),
        classifier=lambda name, raw: "render" if name == "exec" else "unknown",
        classifier_id="fixture.v1",
    )
    report = verify_report(fixture.layout, ref)
    first, second = tuple(iter_responses(fixture.layout, ref))
    assert not first.tool_outputs
    output = second.tool_outputs[0]
    assert output.category == "render" and output.text_chars == 3
    assert output.images[0].source == fixture.catalog[0]
    assert second.context_window == 400000 and second.context_proxy_tokens == 110
    assert report.responses == 2 and report.duplicate_usage_records == 2
    assert report.token_notifications == 2 and report.totals["input_tokens"] == 200
    assert (
        report.totals["output_tokens"] == 20
        and report.totals["reasoning_output_tokens"] == 12
    )
    assert presented_images(fixture.layout, ref) == (fixture.catalog[0].image,)
    assert report.prompt_chars == 7 and report.elapsed_seconds == 60
    assert (
        capture(
            fixture.layout,
            fixture.launch,
            fixture.execution,
            fixture.catalog,
            rollout=fixture.rollout,
            classifier=lambda *_: "render",
            classifier_id="fixture.v1",
        )
        == ref
    )
    with pytest.raises(ValueError, match="input changed"):
        capture(
            fixture.layout,
            fixture.launch,
            fixture.execution,
            fixture.catalog,
            rollout=fixture.rollout,
        )


def test_missing_ids_or_usage_are_unknown_and_images_do_not_jump_to_later_response(
    fixture: Fixture,
) -> None:
    ref = fixture.run(
        *fixture.tool("view", images=(picture(),)),
        usage(None),
        event("token_usage_record", response_id="has-id", usage=None),
        usage("partial", output_tokens=None),
    )
    report = verify_report(fixture.layout, ref)
    assert (
        report.responses == 2
        and report.missing_response_ids == 1
        and report.missing_usage == 1
    )
    assert (
        report.orphan_output_count == 1
        and report.known_subtotals["input_tokens"] == 100
    )
    assert all(value is None for value in report.totals.values())
    assert not presented_images(fixture.layout, ref)
    assert all(
        not response.tool_outputs for response in iter_responses(fixture.layout, ref)
    )


def test_conflicting_duplicate_is_visible_and_cannot_prove_presentation(
    fixture: Fixture,
) -> None:
    ref = fixture.run(
        *fixture.tool("view", images=(picture(),)),
        usage("a"),
        usage("a", input_tokens=101),
    )
    report = verify_report(fixture.layout, ref)
    assert (
        report.conflicting_usage_records == 1
        and report.known_subtotals["input_tokens"] == 100
    )
    assert report.totals["input_tokens"] is None
    with pytest.raises(ValueError, match="conflicting"):
        presented_images(fixture.layout, ref)


def test_archive_uses_logged_bytes_not_overwritten_work_image_and_has_no_reasoning_export(
    fixture: Fixture,
) -> None:
    launch = read_document(fixture.layout, fixture.launch, Launch)
    current = Path(launch.directory) / "work" / "mutable.png"
    current.parent.mkdir(parents=True)
    current.write_bytes(picture("blue"))
    # No catalog entry for this mutable path: the evidence still archives the old red bytes.
    fixture.catalog = ()
    tool = fixture.tool("view", images=(picture(),))
    ref = fixture.run(
        tool[0],
        event(
            "event_msg",
            type="item_completed",
            item={"type": "ImageView", "path": current.as_uri()},
        ),
        tool[1],
        event("response_item", type="reasoning", summary="PRIVATE REASONING FIXTURE"),
        usage("a"),
    )
    response = next(iter_responses(fixture.layout, ref))
    image = response.tool_outputs[0].images[0]
    assert fixture.layout.artifact_path(image.image).read_bytes() == picture()
    assert image.binding == "unbound" and image.source is None
    report_dir = fixture.layout.artifact_path(ref).parent
    assert all(
        "PRIVATE REASONING FIXTURE" not in path.read_text()
        for path in report_dir.iterdir()
        if path.is_file()
    )
    current.write_bytes(picture("green"))
    assert (
        capture(
            fixture.layout,
            fixture.launch,
            fixture.execution,
            (),
            rollout=fixture.rollout,
        )
        == ref
    )


def test_identical_images_do_not_use_batch_order_to_guess_metadata(
    fixture: Fixture,
) -> None:
    original = fixture.catalog[0]
    second = RenderedImage(
        image=write_asset(
            fixture.layout, fixture.layout.path("cache", "same.png"), picture()
        ),
        metadata=write_asset(
            fixture.layout, fixture.layout.path("cache", "other.json"), b'{"frame":11}'
        ),
    )
    fixture.catalog = (original, second)
    tool = fixture.tool("view", images=(picture(), picture()))
    ref = fixture.run(
        tool[0],
        event(
            "event_msg",
            type="item_completed",
            item={
                "type": "ImageView",
                "path": fixture.layout.artifact_path(original.image).as_uri(),
            },
        ),
        event(
            "event_msg",
            type="item_completed",
            item={
                "type": "ImageView",
                "path": fixture.layout.artifact_path(second.image).as_uri(),
            },
        ),
        tool[1],
        usage("a"),
    )
    assert not presented_images(fixture.layout, ref)
    assert verify_report(fixture.layout, ref).unbound_image_inputs == 2


def test_compaction_does_not_claim_discarded_context_images_were_seen(
    fixture: Fixture,
) -> None:
    ref = fixture.run(
        *fixture.tool("view", images=(picture(),)),
        event("compacted", message="replacement"),
        usage("a"),
    )
    assert verify_report(fixture.layout, ref).compactions == 1
    assert not presented_images(fixture.layout, ref)


@pytest.mark.parametrize(
    "field,value", [("id", "other-thread"), ("cwd", "/other/attempt")]
)
def test_rollout_identity_is_exact(fixture: Fixture, field: str, value: str) -> None:
    fixture.prefix[0]["payload"][field] = value
    with pytest.raises(ValueError, match="identity/CWD"):
        fixture.run(usage("a"))


def test_model_mismatch_cannot_support_adoption(fixture: Fixture) -> None:
    fixture.prefix[1]["payload"]["model"] = "different-model"
    ref = fixture.run(usage("a"))
    with pytest.raises(ValueError, match="model"):
        presented_images(fixture.layout, ref)


def test_pages_load_one_response_and_nested_tampering_is_detected_on_recovery(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = fixture.run(*(usage(f"r-{i}") for i in range(101)))
    report = verify_report(fixture.layout, ref)
    assert len(report.pages) == 2
    page = read_document(fixture.layout, report.pages[-1], ResponsePage)
    assert len(page.entries) == 1
    # First-response traversal does not read later pages, even if one is damaged.
    later = fixture.layout.artifact_path(page.entries[0].record)
    later.write_text("tampered")
    assert next(iter_responses(fixture.layout, ref)).response_id == "r-0"
    with pytest.raises(ValueError, match="changed"):
        capture(
            fixture.layout,
            fixture.launch,
            fixture.execution,
            fixture.catalog,
            rollout=fixture.rollout,
        )
    index = fixture.layout.artifact_path(report.index_page).read_text()
    assert "data:image" not in index and "response-00101.html" in index


def test_changed_raw_rollout_cannot_replace_frozen_report(fixture: Fixture) -> None:
    fixture.run(usage("a"))
    with fixture.rollout.open("a") as handle:
        handle.write(json.dumps(usage("b")) + "\n")
    with pytest.raises(ValueError, match="input changed"):
        capture(
            fixture.layout,
            fixture.launch,
            fixture.execution,
            fixture.catalog,
            rollout=fixture.rollout,
        )


def test_rollout_locator_never_chooses_latest_or_another_thread(
    fixture: Fixture,
) -> None:
    launch = read_document(fixture.layout, fixture.launch, Launch)
    execution = read_document(fixture.layout, fixture.execution, ExitReceipt)
    root = Path(launch.codex_home) / "sessions" / "2026" / "10" / "14"
    root.mkdir(parents=True)
    (root / "rollout-newer-unrelated.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="found 0"):
        locate_rollout(launch, execution)
    target = root / f"rollout-2026-{execution.thread_id}.jsonl"
    target.write_text("{}")
    assert locate_rollout(launch, execution) == target
    (root / f"rollout-duplicate-{execution.thread_id}.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="found 2"):
        locate_rollout(launch, execution)


def test_usage_preserves_turn_totals_cache_write_and_thread_identity(
    fixture: Fixture,
) -> None:
    execution = read_document(fixture.layout, fixture.execution, ExitReceipt)
    record = usage("a", cache_write_input_tokens=12)
    record["payload"].update(
        thread_id=execution.thread_id,
        session_id=execution.thread_id,
        turn_id="turn-2",
        root_turn_id="turn-1",
        turn_token_usage={"input_tokens": 500},
        thread_token_usage={"input_tokens": 800},
    )
    ref = fixture.run(record)
    response = next(iter_responses(fixture.layout, ref))
    assert response.logged_thread_usage is not None
    assert response.logged_turn_usage is not None
    assert response.usage is not None
    assert response.logged_thread_usage.input_tokens == 800
    assert response.logged_turn_usage.input_tokens == 500
    assert (
        response.usage.input_tokens == 100
        and response.usage.cache_write_input_tokens == 12
    )
    assert response.turn_id == "turn-2" and response.root_turn_id == "turn-1"
    assert verify_report(fixture.layout, ref).totals["cache_write_input_tokens"] == 12


def test_foreign_usage_record_is_rejected_even_with_correct_session_header(
    fixture: Fixture,
) -> None:
    record = usage("a")
    record["payload"]["thread_id"] = "other-thread"
    with pytest.raises(ValueError, match="another thread/session"):
        fixture.run(record)
