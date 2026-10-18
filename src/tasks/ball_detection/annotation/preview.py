"""Map unreviewed proposals to native markers; predicted absence has no point."""

from src.tasks.base.annotation.images.contracts import Point
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionReceipt

from .contracts import BallProposal, HybridProposal, read_predictions


def proposal_drawings(row: BallProposal | HybridProposal) -> tuple[Point, ...]:
    return () if row.center is None else (Point(x=row.center[0], y=row.center[1]),)


def preview_drawings(
    layout: AnnotationLayout, receipt: PredictionReceipt, frames: tuple[int, ...]
) -> tuple[tuple[Point, ...], ...]:
    prediction = read_predictions(layout, receipt.prediction.artifact)
    return tuple(
        proposal_drawings(prediction.rows[index])
        for index in frames
    )
