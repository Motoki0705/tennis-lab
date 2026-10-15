"""Map validated, unreviewed MDD coordinates to native-pixel preview markers."""

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.images.contracts import Point
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionReceipt

from .contracts import BallPredictions


def preview_drawings(
    layout: AnnotationLayout, receipt: PredictionReceipt, frames: tuple[int, ...]
) -> tuple[tuple[Point, ...], ...]:
    prediction = read_document(layout, receipt.prediction.artifact, BallPredictions)
    return tuple(
        (Point(x=prediction.rows[index].x, y=prediction.rows[index].y),)
        for index in frames
    )
