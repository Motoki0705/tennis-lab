"""Deterministic Court semantic decisions before renderer execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from src.synthetic_data_generation.dataset.court.components.labels import (
    AmbiguousCameraRelativeNearFarError,
    MultiCourtProjectionAny,
    project_court_semantics_for_version,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.scene_contract import MultiCourtLayout, SceneCamera

COURT_SEMANTIC_PHASE_DISPOSITION_DIGEST_SCHEMA = (
    "court_semantic_phase_disposition_digest_v1"
)
INSUFFICIENT_PRE_RENDER_SEMANTIC_COVERAGE_REASON = (
    "insufficient_pre_render_semantic_coverage"
)


@dataclass(frozen=True, slots=True)
class CourtSemanticPreRenderDecision:
    """One camera's complete semantic disposition before rendering."""

    camera_id: str
    projection: MultiCourtProjectionAny | None
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.camera_id, str)
            or not self.camera_id
            or self.camera_id != self.camera_id.strip()
        ):
            raise TypeError("camera_id must be a non-empty trimmed string.")
        if self.projection is not None and self.projection.camera_id != self.camera_id:
            raise ValueError(
                "Semantic projection camera_id disagrees with the decision."
            )
        reasons = tuple(self.rejection_reasons)
        if len(reasons) != len(set(reasons)) or any(
            not isinstance(reason, str) or not reason or reason != reason.strip()
            for reason in reasons
        ):
            raise ValueError(
                "Semantic rejection reasons must be unique non-empty strings."
            )
        if self.projection is None and not reasons:
            raise ValueError("An accepted semantic decision requires a projection.")
        object.__setattr__(self, "rejection_reasons", reasons)

    @property
    def accepted(self) -> bool:
        """Return whether the camera may proceed to renderer execution."""
        return not self.rejection_reasons

    @property
    def disposition(self) -> str:
        """Return the canonical accepted/rejected disposition token."""
        return "accepted" if self.accepted else "rejected"


@dataclass(frozen=True, slots=True)
class CourtSemanticFrameDisposition:
    """Bind one semantic decision to its trajectory-local frame geometry."""

    trajectory_frame_index: int
    camera: SceneCamera
    decision: CourtSemanticPreRenderDecision

    def __post_init__(self) -> None:
        if (
            isinstance(self.trajectory_frame_index, bool)
            or not isinstance(self.trajectory_frame_index, int)
            or self.trajectory_frame_index < 0
        ):
            raise TypeError("trajectory_frame_index must be a non-negative integer.")
        if not isinstance(self.camera, SceneCamera):
            raise TypeError("camera must be a SceneCamera.")
        if not isinstance(self.decision, CourtSemanticPreRenderDecision):
            raise TypeError("decision must be a CourtSemanticPreRenderDecision.")
        if self.camera.camera_id != self.decision.camera_id:
            raise ValueError("Semantic decision camera_id disagrees with its camera.")


def evaluate_court_semantic_pre_render(
    camera: SceneCamera,
    layout: MultiCourtLayout,
    *,
    schema_version: CourtDatasetSchemaVersion,
) -> CourtSemanticPreRenderDecision:
    """Return the sole deterministic per-camera Court semantic disposition."""
    try:
        projection = project_court_semantics_for_version(
            camera,
            layout,
            schema_version=schema_version,
        )
    except AmbiguousCameraRelativeNearFarError as error:
        return CourtSemanticPreRenderDecision(
            camera_id=camera.camera_id,
            projection=None,
            rejection_reasons=(error.reason,),
        )
    in_frame_points = sum(court.in_frame_point_count for court in projection.courts)
    reasons = (
        (INSUFFICIENT_PRE_RENDER_SEMANTIC_COVERAGE_REASON,)
        if in_frame_points < 4
        else ()
    )
    return CourtSemanticPreRenderDecision(
        camera_id=camera.camera_id,
        projection=projection,
        rejection_reasons=reasons,
    )


def court_semantic_phase_disposition_digest(
    dispositions: Sequence[CourtSemanticFrameDisposition],
    *,
    schema_version: CourtDatasetSchemaVersion,
    trajectory_group_id: str,
    phase_index: int,
    phase_count: int,
) -> str:
    """Hash one phase by trajectory frame order, excluding incidental sample IDs."""
    if not isinstance(schema_version, CourtDatasetSchemaVersion):
        raise TypeError("schema_version must be a CourtDatasetSchemaVersion.")
    if (
        not isinstance(trajectory_group_id, str)
        or not trajectory_group_id
        or trajectory_group_id != trajectory_group_id.strip()
    ):
        raise TypeError("trajectory_group_id must be a non-empty trimmed string.")
    if (
        isinstance(phase_count, bool)
        or not isinstance(phase_count, int)
        or phase_count <= 0
    ):
        raise TypeError("phase_count must be a positive integer.")
    if (
        isinstance(phase_index, bool)
        or not isinstance(phase_index, int)
        or not 0 <= phase_index < phase_count
    ):
        raise ValueError("phase_index must be an integer in [0, phase_count).")
    values = tuple(dispositions)
    if not values or any(
        not isinstance(item, CourtSemanticFrameDisposition) for item in values
    ):
        raise TypeError("dispositions must contain typed semantic frame decisions.")
    ordered = sorted(values, key=lambda item: item.trajectory_frame_index)
    frame_indices = tuple(item.trajectory_frame_index for item in ordered)
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("A semantic phase cannot repeat a trajectory frame index.")
    payload = {
        "schema": COURT_SEMANTIC_PHASE_DISPOSITION_DIGEST_SCHEMA,
        "schema_version": schema_version.value,
        "trajectory_group_id": trajectory_group_id,
        "phase_index": phase_index,
        "phase_count": phase_count,
        "frames": [_canonical_frame_payload(item) for item in ordered],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_frame_payload(
    disposition: CourtSemanticFrameDisposition,
) -> dict[str, object]:
    camera = disposition.camera
    decision = disposition.decision
    projection: dict[str, object] | None = None
    if decision.projection is not None:
        projection = decision.projection.to_dict()
        # Generated sample IDs are deliberately outside semantic phase identity.
        projection.pop("camera_id")
    return {
        "trajectory_frame_index": disposition.trajectory_frame_index,
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "intrinsics": list(camera.intrinsics),
            "camera_to_scene": camera.camera_to_scene.to_list(),
        },
        "semantic_result": {
            "disposition": decision.disposition,
            "rejection_reasons": list(decision.rejection_reasons),
            "projection": projection,
        },
    }


__all__ = [
    "COURT_SEMANTIC_PHASE_DISPOSITION_DIGEST_SCHEMA",
    "INSUFFICIENT_PRE_RENDER_SEMANTIC_COVERAGE_REASON",
    "CourtSemanticFrameDisposition",
    "CourtSemanticPreRenderDecision",
    "court_semantic_phase_disposition_digest",
    "evaluate_court_semantic_pre_render",
]
