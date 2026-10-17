"""Map validated DINO proposals to native edge-coordinate preview boxes."""

from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.images.contracts import Box
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionReceipt

from .contracts import PlayerPredictions


def preview_drawings(
    layout: AnnotationLayout, receipt: PredictionReceipt, frames: tuple[int, ...]
) -> tuple[tuple[Box, ...], ...]:
    prediction = read_document(layout, receipt.prediction.artifact, PlayerPredictions)
    return tuple(
        tuple(
            Box(left=p.box.x1, top=p.box.y1, right=p.box.x2, bottom=p.box.y2)
            for p in prediction.rows[index].observations
        )
        for index in frames
    )
