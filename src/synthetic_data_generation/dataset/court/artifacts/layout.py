"""Versioned multi-court layout derived from an accepted alignment artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.artifacts.court_geometry import (
    load_court_geometry_artifact,
)
from src.synthetic_data_generation.scene_contract import (
    SceneContract,
    SimilarityTransform,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d

_MATRIX_ATOL = 1.0e-6


@dataclass(frozen=True)
class CourtInstance:
    """One physical court with its own immutable scene transform."""

    court_instance_id: str
    candidate_id: str
    scene_from_court: SimilarityTransform
    court_from_scene: SimilarityTransform
    template_score: float

    def __post_init__(self) -> None:
        if not self.court_instance_id:
            raise ValueError("court_instance_id must not be empty.")
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty.")
        if not np.isfinite(self.template_score) or self.template_score <= 0.0:
            raise ValueError("template_score must be finite and positive.")
        round_trip = self.court_from_scene.matrix() @ self.scene_from_court.matrix()
        if not np.allclose(
            round_trip,
            np.eye(4),
            atol=_MATRIX_ATOL,
            rtol=0.0,
        ):
            raise ValueError("Court transforms are not mutual inverses.")

    def keypoints_scene(self) -> NDArray[np.float64]:
        """Return CourtKP20 in provider scene coordinates."""
        points = np.asarray(
            court_keypoints_3d(STANDARD_COURT_CONFIG).numpy(),
            dtype=np.float64,
        )
        return np.asarray(
            self.scene_from_court.apply(points),
            dtype=np.float64,
        )

    def center_in(
        self,
        reference: CourtInstance,
    ) -> NDArray[np.float64]:
        """Return this court origin in another court's metric coordinates."""
        origin_scene = self.scene_from_court.apply(np.zeros((1, 3), dtype=np.float64))
        return cast(
            NDArray[np.float64],
            reference.court_from_scene.apply(origin_scene)[0],
        )


@dataclass(frozen=True)
class MultiCourtLayout:
    """N physical courts with stable instance identity."""

    geometry_artifact_fingerprint: str
    reference_court_instance_id: str
    courts: tuple[CourtInstance, ...]

    def __post_init__(self) -> None:
        if len(self.geometry_artifact_fingerprint) != 64:
            raise ValueError("geometry_artifact_fingerprint must be SHA-256.")
        courts = tuple(self.courts)
        if not courts:
            raise ValueError("MultiCourtLayout must contain at least one court.")
        instance_ids = [court.court_instance_id for court in courts]
        candidate_ids = [court.candidate_id for court in courts]
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("court_instance_id values must be unique.")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_id values must be unique.")
        if self.reference_court_instance_id not in instance_ids:
            raise ValueError("reference_court_instance_id is absent.")
        object.__setattr__(self, "courts", courts)

    @property
    def reference(self) -> CourtInstance:
        """Return the layout's metric reference court."""
        return next(
            court
            for court in self.courts
            if court.court_instance_id == self.reference_court_instance_id
        )

    def centers_in_reference(self) -> NDArray[np.float64]:
        """Return all physical court centres in the reference court frame."""
        return np.stack([court.center_in(self.reference) for court in self.courts])


def load_multi_court_layout(
    geometry_artifact_path: Path,
    scene_contract: SceneContract,
    *,
    candidate_ids: tuple[str, ...],
) -> MultiCourtLayout:
    """Load N court candidates and bind court-0 to the accepted scene contract.

    The selected court in the accepted scene contract remains the metric
    reference. Additional candidates are accepted only from the same
    fingerprint-verified geometry artifact; no inferred offset or fallback
    court is manufactured.
    """
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique.")
    payload = load_court_geometry_artifact(geometry_artifact_path)
    fingerprint = payload.get("artifact_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("Court geometry artifact has no fingerprint.")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Court geometry candidates are missing.")
    by_id = {
        candidate.get("candidate_id"): candidate
        for candidate in raw_candidates
        if isinstance(candidate, dict)
    }
    missing = set(candidate_ids).difference(by_id)
    if missing:
        raise ValueError(f"Unknown court candidates: {sorted(missing)}.")

    selected_candidate = scene_contract.alignment.selected_court_cluster
    if selected_candidate not in candidate_ids:
        raise ValueError(
            "candidate_ids must include the accepted selected court cluster."
        )
    result = []
    for index, candidate_id in enumerate(candidate_ids):
        candidate = by_id[candidate_id]
        scene_from_court = _similarity_from_matrix(
            candidate.get("scene_from_court"),
            name=f"{candidate_id}.scene_from_court",
        )
        court_from_scene = _similarity_from_matrix(
            candidate.get("court_from_scene"),
            name=f"{candidate_id}.court_from_scene",
        )
        score = candidate.get("template_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{candidate_id}.template_score is invalid.")
        instance = CourtInstance(
            court_instance_id=f"court_{index}",
            candidate_id=candidate_id,
            scene_from_court=scene_from_court,
            court_from_scene=court_from_scene,
            template_score=float(score),
        )
        if candidate_id == selected_candidate:
            _require_accepted_reference(instance, scene_contract)
        result.append(instance)

    reference_id = result[candidate_ids.index(selected_candidate)].court_instance_id
    return MultiCourtLayout(
        geometry_artifact_fingerprint=fingerprint,
        reference_court_instance_id=reference_id,
        courts=tuple(result),
    )


def _similarity_from_matrix(value: object, *, name: str) -> SimilarityTransform:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix.")
    if not np.allclose(
        matrix[3],
        (0.0, 0.0, 0.0, 1.0),
        atol=_MATRIX_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must have a homogeneous bottom row.")
    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise ValueError(f"{name} must have positive determinant.")
    scale = float(np.cbrt(determinant))
    return SimilarityTransform(
        scale=scale,
        rotation=tuple(float(item) for item in (linear / scale).ravel()),
        translation=tuple(float(item) for item in matrix[:3, 3]),
    )


def _require_accepted_reference(
    instance: CourtInstance,
    scene_contract: SceneContract,
) -> None:
    expected = scene_contract.alignment
    errors = (
        float(
            np.max(
                np.abs(
                    instance.scene_from_court.matrix()
                    - expected.scene_from_court.matrix()
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    instance.court_from_scene.matrix()
                    - expected.court_from_scene.matrix()
                )
            )
        ),
    )
    if max(errors) > _MATRIX_ATOL:
        raise ValueError(
            "Selected multi-court reference differs from accepted alignment: "
            f"max error {max(errors):.3g}."
        )
