"""Supervisor-owned learning stage boundaries shared by concrete task CLIs."""

import os
from pathlib import Path

from ..artifacts import read_document, verify_reference
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from .contracts import CampaignCompletion, LearningIntent, StageInput
from .validation import check_completion


def stage_request(
    layout: AnnotationLayout, reference: ArtifactRef, result: Path, *, task: str
) -> tuple[StageInput, LearningIntent]:
    if os.environ.get("TENNIS_ANNOTATION_STAGE_OPERATION") != reference.digest:
        raise ValueError("learning requires the supervisor-owned stage input")
    request = read_document(layout, reference, StageInput)
    intent = read_document(layout, request.intent, LearningIntent)
    if intent.task != task:
        raise ValueError("learning stage belongs to a different task")
    completion = read_document(layout, intent.completion, CampaignCompletion)
    plan = check_completion(layout, completion)
    directory = layout.path(
        "campaigns",
        plan.campaign_id,
        "learning",
        task,
        request.stage,
        str(request.attempt),
    )
    if (
        verify_reference(layout, reference) != directory / "request.json"
        or result != directory / "result.json"
    ):
        raise ValueError("learning request/result is outside its stage attempt")
    return request, intent
