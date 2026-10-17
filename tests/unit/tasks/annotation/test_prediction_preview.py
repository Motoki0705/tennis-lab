from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import pytest
from PIL import Image

from src.tasks.annotation.contracts import (
    OperationInput,
    OperationReceipt,
    PredictionPreviewOperation,
    Runtime,
    StageCost,
)
from src.tasks.annotation.jobs import Operations
from src.tasks.annotation.operations import execute
from src.tasks.annotation.prediction_preview import preview_prediction
from src.tasks.ball_detection.annotation.contracts import BallPredictions
from src.tasks.ball_detection.annotation.inference import BallPredictionAdapter
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.execution.contracts import QueueConfiguration
from src.tasks.base.annotation.images.contracts import (
    Box,
    Crop,
    ImageTile,
    PageView,
    Point,
    RenderReceipt,
    RenderRequest,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import file_digest, save_plan
from src.tasks.base.annotation.predictions import PredictionContext, PredictionReceipt
from src.tasks.base.annotation.resources import BudgetLimits, ResourceManager
from src.tasks.player_detection.annotation.contracts import (
    Box as PlayerBox,
)
from src.tasks.player_detection.annotation.contracts import (
    PlayerFrameProposals,
    PlayerPredictions,
    PlayerProposal,
)
from src.tasks.player_detection.annotation.inference import (
    PlayerPredictionAdapter,
    observation_id,
)
from tests.support.tasks.annotation_predictions import Fixture, example
from tests.support.tasks.ball_detection.annotation import ProposalFixture


class PlayerFixture(PlayerPredictionAdapter):
    def __init__(self, count: int = 2) -> None:
        super().__init__()
        self.count = count

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef:
        return write_document(
            layout,
            directory / "boxes.json",
            PlayerPredictions(
                context_digest=context.sha256,
                rows=tuple(
                    PlayerFrameProposals(
                        frame_index=f.frame_index,
                        source_frame_index=f.source_frame_index,
                        source_pts=f.source_pts,
                        clip_pts=f.clip_pts,
                        observations=tuple(
                            PlayerProposal(
                                observation_id=observation_id(
                                    context, f.frame_index, rank
                                ),
                                candidate_index=rank,
                                confidence=0.8,
                                box=PlayerBox(x1=float(rank), y1=0.0, x2=64.0, y2=32.0),
                            )
                            for rank in range(
                                self.count if f.frame_index % 2 == 0 else 0
                            )
                        ),
                    )
                    for f in manifest.frames
                ),
            ),
        )


def prepared(
    tmp_path: Path,
    task: Literal["ball", "player"],
    *,
    count: int = 2,
) -> tuple[Fixture, ArtifactRef, ArtifactRef]:
    env = example(tmp_path, 32, task=task)
    prediction = env.service.run(
        env.reference, ProposalFixture() if task == "ball" else PlayerFixture(count)
    )
    plan = annotation_reference(env.layout, save_plan(env.layout, env.plan))
    return env, prediction, plan


def operation(
    prediction: ArtifactRef,
    task: Literal["ball", "player"],
    *,
    drawing: Literal["plain", "prediction"] = "prediction",
    frames: tuple[int, ...] = (0, 1),
    right: int = 64,
) -> PredictionPreviewOperation:
    return PredictionPreviewOperation(
        task=task,
        prediction=prediction,
        drawing=drawing,
        view=PageView(
            tiles=tuple(
                ImageTile(
                    frame_index=i, crop=Crop(left=0, top=0, right=right, bottom=32)
                )
                for i in frames
            ),
            columns=1,
            tile_width=128,
            tile_height=64,
            reason="initial model preview",
        ),
    )


def forbidden(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("preview must neither load nor run a model")


@pytest.mark.parametrize("task", ["ball", "player"])
@pytest.mark.parametrize("drawing", ["plain", "prediction"])
def test_saved_predictions_draw_native_geometry_without_state_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: Literal["ball", "player"],
    drawing: Literal["plain", "prediction"],
) -> None:
    env, prediction, _ = prepared(tmp_path, task)
    monkeypatch.setattr(BallPredictionAdapter, "produce", forbidden)
    monkeypatch.setattr(PlayerPredictionAdapter, "produce", forbidden)
    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.MDDPredictor", forbidden
    )
    monkeypatch.setattr(
        "src.tasks.player_detection.annotation.inference.DinoPredictor", forbidden
    )
    op = operation(prediction, task, drawing=drawing)
    reference = preview_prediction(env.layout, env.plan, op)
    receipt = read_document(env.layout, reference, RenderReceipt)
    request = read_document(env.layout, receipt.request, RenderRequest)
    assert request.annotation == prediction and request.manifest == env.reference
    assert receipt.tiles[1].source_pts == env.manifest.frames[1].source_pts
    assert request.drawing == drawing
    if drawing == "plain":
        assert all(not t.drawings for t in request.tiles)
    elif task == "ball":
        assert request.tiles[0].drawings == (Point(x=16.0, y=12.0),)
    else:
        assert request.tiles[0].drawings == (
            Box(left=0.0, top=0.0, right=64.0, bottom=32.0),
            Box(left=1.0, top=0.0, right=64.0, bottom=32.0),
        )
        assert not request.tiles[1].drawings  # Empty proposals remain unreviewed.
    with Image.open(env.layout.artifact_path(receipt.image)) as image:
        assert image.size == (receipt.width, receipt.height)
        if drawing == "prediction" and task == "ball":
            # The yellow marker is present in actual encoded pixels, not just metadata.
            assert any(r > b + 80 and g > b + 80 for r, g, b in image.getdata())
    path = env.layout.artifact_path(receipt.image)
    identity = (path.stat().st_ino, path.stat().st_mtime_ns)
    assert preview_prediction(env.layout, env.plan, op) == reference
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == identity
    with env.service.registry.transaction() as db:
        assert (
            db.execute("SELECT state FROM annotation_tasks").fetchone()[0]
            == "predicted"
        )


@pytest.mark.parametrize("state", ["reviewing", "adopted", "pending"])
def test_preview_rejects_non_preview_states(tmp_path: Path, state: str) -> None:
    env, prediction, _ = prepared(tmp_path, "ball")
    with env.service.registry.transaction() as db:
        db.execute("UPDATE annotation_tasks SET state=?", (state,))
    with pytest.raises(ValueError, match="unreviewed"):
        preview_prediction(env.layout, env.plan, operation(prediction, "ball"))


def test_preview_requires_exact_campaign_task_and_receipt(tmp_path: Path) -> None:
    env, prediction, _ = prepared(tmp_path, "ball")
    receipt = read_document(env.layout, prediction, PredictionReceipt)
    copy = write_document(
        env.layout, env.layout.path("campaigns", "copied.json"), receipt
    )
    with pytest.raises(ValueError, match="not registered"):
        preview_prediction(env.layout, env.plan, operation(copy, "ball"))
    with pytest.raises(ValueError, match="absent"):
        preview_prediction(env.layout, env.plan, operation(prediction, "player"))
    with pytest.raises(ValueError, match="preview campaign"):
        preview_prediction(
            env.layout,
            env.plan.model_copy(update={"phase": "production"}),
            operation(prediction, "ball"),
        )
    with env.service.registry.transaction() as db:
        db.execute("UPDATE annotation_tasks SET model_digest=?", ("e" * 64,))
    with pytest.raises(ValueError, match="frozen campaign"):
        preview_prediction(env.layout, env.plan, operation(prediction, "ball"))


@pytest.mark.parametrize("field,value", [("x", 64.0), ("source_pts", -1)])
def test_task_payload_is_validated_even_for_plain_image(
    tmp_path: Path, field: str, value: float
) -> None:
    env, reference, _ = prepared(tmp_path, "ball")
    receipt = read_document(env.layout, reference, PredictionReceipt)
    payload = read_document(env.layout, receipt.prediction.artifact, BallPredictions)
    rows = (payload.rows[0].model_copy(update={field: value}), *payload.rows[1:])
    payload_ref = write_document(
        env.layout,
        env.layout.path("campaigns", "altered.json"),
        payload.model_copy(update={"rows": rows}),
    )
    altered = receipt.model_copy(
        update={
            "prediction": receipt.prediction.model_copy(
                update={"artifact": payload_ref}
            )
        }
    )
    altered_ref = write_document(
        env.layout, env.layout.path("campaigns", "altered-receipt.json"), altered
    )
    with env.service.registry.transaction() as db:
        db.execute(
            "UPDATE annotation_predictions SET receipt=?",
            (altered_ref.canonical_bytes().decode(),),
        )
    with pytest.raises(ValueError):
        preview_prediction(
            env.layout, env.plan, operation(altered_ref, "ball", drawing="plain")
        )
    assert not env.layout.path("annotations", "rendered").exists()


@pytest.mark.parametrize("frames,right", [((32,), 64), ((0,), 65)])
def test_clip_bounds_are_not_clamped(
    tmp_path: Path, frames: tuple[int, ...], right: int
) -> None:
    env, reference, _ = prepared(tmp_path, "ball")
    with pytest.raises(ValueError, match="outside"):
        preview_prediction(
            env.layout,
            env.plan,
            operation(reference, "ball", frames=frames, right=right),
        )


def test_crowded_player_proposals_are_not_silently_truncated(tmp_path: Path) -> None:
    env, reference, _ = prepared(tmp_path, "player", count=33)
    with pytest.raises(ValueError, match="at most 32"):
        preview_prediction(env.layout, env.plan, operation(reference, "player"))


def test_operation_input_caps_and_disallows_caller_drawings(tmp_path: Path) -> None:
    _env, reference, plan = prepared(tmp_path, "ball")
    op = operation(reference, "ball")
    assert OperationInput(plan=plan, operation=op).stage == "render"
    value = op.model_dump()
    value["view"]["tiles"][0]["drawings"] = [{"kind": "point", "x": 0.0, "y": 0.0}]
    with pytest.raises(ValueError, match="Extra inputs"):
        PredictionPreviewOperation.model_validate(value)
    view = op.view.model_dump()
    view.update(tiles=view["tiles"] * 8, columns=4, tile_width=1024, tile_height=1024)
    with pytest.raises(ValueError, match="pixel budget"):
        PageView.model_validate(view)


def test_formal_operation_requires_owner_and_reuses_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, prediction, plan = prepared(tmp_path, "ball")
    data = OperationInput(plan=plan, operation=operation(prediction, "ball"))
    reference = write_document(
        env.layout,
        env.layout.path(
            "campaigns",
            env.plan.campaign_id,
            "jobs",
            data.sha256 + "-1",
            "operation.json",
        ),
        data,
    )
    with pytest.raises(ValueError, match="resource-accounted"):
        execute(env.layout, reference)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", reference.digest)
    result = execute(env.layout, reference)
    assert read_document(env.layout, result, OperationReceipt).status == "complete"
    monkeypatch.setattr(
        "src.tasks.annotation.prediction_preview.preview_prediction", forbidden
    )
    assert execute(env.layout, reference) == result


def test_preview_runs_in_resource_owned_cpu_subprocess(tmp_path: Path) -> None:
    env, prediction, plan = prepared(tmp_path, "player")
    resources = ResourceManager.create(
        env.layout,
        scope_id="preview",
        limits=BudgetLimits.from_budget(env.plan.resources),
        roots=(env.layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0,
        max_pending_per_stage=2,
    )
    script = env.layout.repository / "queue.sh"
    script.write_text("# fixture --idempotency-key\n")
    runtime = Runtime(
        plan=plan,
        scope=resources.reference,
        queue=QueueConfiguration(
            script=str(script),
            script_digest=file_digest(script),
            directory=str(env.layout.repository / ".training_queue"),
            session="fixture",
            issue=1088,
        ),
        codex_binary="codex",
        codex_home=str(env.layout.path("cache", "codex")),
        costs=(
            StageCost(
                stage="render",
                pss_bytes=2 * 1024**3,
                storage_bytes=1024**2,
                timeout_seconds=30.0,
                gpu_seconds=0.0,
            ),
        ),
    )
    jobs = Operations(env.layout, runtime, env.plan, resources)
    data = OperationInput(plan=plan, operation=operation(prediction, "player"))
    deadline = time.monotonic() + 40
    state = jobs.operation(data)
    while (
        state["state"] in {"pending", "running", "draining", "waiting"}
        and time.monotonic() < deadline
    ):
        time.sleep(0.1)
        state = jobs.operation(data)
    assert state["state"] == "succeeded", state
    receipt = jobs.receipt(state)
    assert read_document(env.layout, receipt.results[0], RenderReceipt).tiles
    assert jobs.operation(data) == state
    sample = resources.sample()
    assert (
        sample.gpu_charged_seconds
        == sample.gpu_active_seconds
        == sample.gpu_reserved_seconds
        == 0
    )
