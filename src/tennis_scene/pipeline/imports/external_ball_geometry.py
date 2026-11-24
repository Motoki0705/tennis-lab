"""Bind archived camera parameters to a pinned ball-based court side."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tennis_scene.pipeline.components.camera_geometry import (
    CameraGeometryResult,
    describe_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideInput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.imports.external_ball_positions import (
    ExternalBallPositions,
)
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.tennis_scene.pipeline.video_court.diagnostics import (
    DriftReference,
    load_video_drift,
)
from src.tennis_scene.pipeline.video_court.store import (
    VideoCourtArtifact,
    VideoCourtStore,
)
from src.tennis_scene.pipeline.video_court_side.artifact import (
    VideoCourtSideArtifact,
    require_accepted_courts,
)
from src.tennis_scene.pipeline.video_court_side.prepare import calibration_digest
from src.utils.geometry.triangulation import PinholeCamera

SAVED_GEOMETRY = "saved_camera_alignment"
SHARED_GEOMETRY = "calibration_and_shared_ball_court_side"


@dataclass(frozen=True)
class PinnedBallCourt:
    side: VideoCourtSideArtifact
    court: VideoCourtArtifact

    @classmethod
    def load(
        cls,
        side_store: Path,
        side_digest: str,
        court_store: Path,
        drift_store: Path | None,
    ) -> PinnedBallCourt:
        side = VideoCourtSideArtifact.load(side_store, side_digest)
        if side.decision.evidence != "ball":
            raise ValueError(
                "External ball import requires a ball-based shared court side"
            )
        side.output()
        court = VideoCourtStore(court_store).load(side.decision.court_digest)
        drift = side.decision.court_drift_digest
        if (drift_store is None) != (drift is None):
            raise ValueError("Drift store must match the pinned ball court side")
        if drift_store is not None and drift is not None:
            court = load_video_drift(court, DriftReference(drift_store, drift))
        side.check_court(court, "warn")
        return cls(side, court)

    def bind(
        self,
        source: ClipSource,
        calibration: CourtCalibrationOutput,
        positions: ExternalBallPositions,
        saved: CameraGeometryResult | None,
    ) -> tuple[CameraGeometryResult, dict[str, Any]]:
        decision = self.side.decision
        self.court.check_source(source)
        if (
            source.clip_id not in self.side.clips
            or source.camera_ids != decision.camera_ids
            or source.camera_ids != positions.camera_ids
            or calibration.calibration.camera_ids != source.camera_ids
            or calibration.calibration.excluded
        ):
            raise ValueError(
                "Shared ball court side requires the complete calibrated camera axis"
            )
        require_accepted_courts(self.court, (source.clip_id,), "warn")
        self.side.require_clip(source.clip_id, "warn")
        # Use the same calibration signature as the builder of the pinned side.
        # The imported observations are never scored to choose a new side.
        usable = positions.usable[None, :, :, None]
        grouped = GroupedObservations(
            np.array([0], dtype=np.int64),
            positions.uv_px.astype(np.float32)[None, :, :, None, :],
            usable.astype(np.float32),
            usable,
            np.where(usable[..., 0], 0, -1).astype(np.int64),
        )
        if (
            calibration_digest(CourtSideInput(source, calibration, grouped), self.court)
            != decision.calibration_digest
        ):
            raise ValueError(
                "Calibration disagrees with the pinned shared ball court side"
            )
        side = self.side.output()
        turns, reference = side.view_half_turns, side.reference_camera
        cameras = tuple(
            view.camera.half_turned(turn)
            for view, turn in zip(calibration.calibration.views, turns, strict=True)
        )
        provenance = {
            "origin": SAVED_GEOMETRY if saved is not None else SHARED_GEOMETRY,
            "player_validation": "inherited_from_saved_alignment"
            if saved is not None
            else "not_run",
            "shared_side_digest": self.side.digest,
            "shared_side_diagnostics_digest": decision.diagnostics_digest,
            "shared_court_digest": decision.court_digest,
            "shared_court_drift_digest": decision.court_drift_digest,
            "calibration_digest": decision.calibration_digest,
            "reference_camera": reference,
            "view_half_turns": list(turns),
            "shared_side_clip_verdict": self.side.clip_verdict(source.clip_id, "warn"),
        }
        if saved is not None:
            require_matching_alignment(saved, reference, turns, cameras)
            return saved, provenance
        geometry = describe_camera_geometry(
            calibration.calibration, reference, turns, cameras, provenance
        )
        return geometry, provenance


def require_matching_alignment(
    saved: CameraGeometryResult,
    reference: str,
    turns: tuple[bool, ...],
    cameras: tuple[PinholeCamera, ...],
) -> None:
    """Reject conflicting saved axes, gauge or matrices before touching publication."""
    ids = tuple(camera.camera_id for camera in cameras)
    if (
        (saved.camera_ids, saved.reference_camera, saved.view_half_turns)
        != (ids, reference, turns)
        or tuple(camera.camera_id for camera in saved.cameras) != ids
        or any(
            not np.array_equal(getattr(actual, field), getattr(expected, field))
            for actual, expected in zip(saved.cameras, cameras, strict=True)
            for field in ("intrinsic", "rotation", "translation")
        )
    ):
        raise ValueError(
            "Saved camera_alignment conflicts with pinned shared court orientation "
            f"(saved {saved.reference_camera}/{saved.view_half_turns}, pinned {reference}/{turns})"
        )
