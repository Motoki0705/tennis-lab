"""Read-only pilot images from registered predictions, with task-owned geometry."""

from collections.abc import Callable

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.images import render_page
from src.tasks.base.annotation.images.contracts import Primitive, RenderRequest, Tile
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan
from src.tasks.base.annotation.predictions import (
    PredictionAdapter,
    PredictionReceipt,
    Predictions,
)

from .contracts import PredictionPreviewOperation


def preview_prediction(
    layout: AnnotationLayout, plan: CampaignPlan, operation: PredictionPreviewOperation
) -> ArtifactRef:
    if plan.phase != "preview" or layout.review_namespace is not None:
        raise ValueError("prediction preview requires a preview campaign")
    adapter: PredictionAdapter
    draw: Callable[
        [AnnotationLayout, PredictionReceipt, tuple[int, ...]],
        tuple[tuple[Primitive, ...], ...],
    ]
    if operation.task == "ball":
        from src.tasks.ball_detection.annotation.inference import BallPredictionAdapter
        from src.tasks.ball_detection.annotation.preview import (
            preview_drawings as ball_drawings,
        )

        adapter = BallPredictionAdapter()
        draw = ball_drawings
    else:
        from src.tasks.player_detection.annotation.inference import (
            PlayerPredictionAdapter,
        )
        from src.tasks.player_detection.annotation.preview import (
            preview_drawings as player_drawings,
        )

        adapter = PlayerPredictionAdapter()
        draw = player_drawings
    receipt = Predictions(layout, plan).require_completed(
        operation.prediction, adapter, unreviewed=True
    )
    manifest = read_document(layout, receipt.context.manifest, ClipManifest)
    frames = tuple(tile.frame_index for tile in operation.view.tiles)
    if any(index >= len(manifest.frames) for index in frames):
        raise ValueError("preview frame is outside the registered clip")
    drawings: tuple[tuple[Primitive, ...], ...] = (
        draw(layout, receipt, frames)
        if operation.drawing == "prediction"
        else tuple(() for _ in frames)
    )
    request = RenderRequest(
        manifest=receipt.context.manifest,
        annotation=operation.prediction,
        annotation_kind="prediction",
        drawing=operation.drawing,
        tiles=tuple(
            Tile(frame_index=tile.frame_index, crop=tile.crop, drawings=markers)
            for tile, markers in zip(operation.view.tiles, drawings, strict=True)
        ),
        **operation.view.model_dump(exclude={"tiles"}),
    )
    result: ArtifactRef = render_page(layout, request).metadata
    return result
