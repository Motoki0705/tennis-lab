"""Only the supervisor-owned attempt may train or evaluate; adoption stays in base."""

from pathlib import Path

from src.tasks.base.annotation.artifacts import artifact_lock
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.learning.execution import stage_request
from src.tasks.base.annotation.paths import AnnotationLayout

from .evaluation import evaluate
from .train import train


def execute(
    layout: AnnotationLayout, reference: ArtifactRef, result: Path
) -> ArtifactRef:
    request, intent = stage_request(layout, reference, result, task="player")
    with artifact_lock(layout, "player-learning-" + intent.sha256, blocking=False):
        value: ArtifactRef = (train if request.stage == "train" else evaluate)(
            layout, reference, request, intent, result
        )
        return value
