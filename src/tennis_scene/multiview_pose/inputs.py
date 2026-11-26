"""Read only the raw person observations and the pinned ball-based court gauge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
    require_alignment,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.people_artifacts import stored_people_models
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
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
    shared_calibration_digest,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera

_Value = TypeVar("_Value")


@dataclass(frozen=True)
class PoseInputs:
    source: ClipSource
    cameras: tuple[PinholeCamera, ...]
    detections: tuple[PersonDetectionOutput, ...]
    poses: tuple[PersonPoseFeaturesOutput, ...]

    def __post_init__(self) -> None:
        if (
            not 2 <= len(self.cameras) <= 5
            or tuple(c.camera_id for c in self.cameras) != self.source.camera_ids
        ):
            raise ValueError(
                "Calibrated pose views must cover the ordered source cameras (2–5)"
            )
        for video, detection, pose in zip(
            self.source.videos, self.detections, self.poses, strict=True
        ):
            rows = require_alignment(video, detection, pose)
            if detection.duplicate_merges or not np.array_equal(
                rows, np.arange(len(rows))
            ):
                raise ValueError(
                    f"Every original detector row is required: {video.camera_id}"
                )
            if (detection.boxes_xyxy[:, 2:] <= detection.boxes_xyxy[:, :2]).any():
                raise ValueError(f"Detector rows require positive boxes: {video.camera_id}")


@dataclass(frozen=True)
class PinnedCourtSide:
    """Verified video artifact snapshot reused across clips of that video."""

    _side: VideoCourtSideArtifact
    _court: VideoCourtArtifact

    @classmethod
    def load(
        cls,
        side_directory: Path,
        digest: str,
        court_directory: Path,
        drift_directory: Path | None = None,
    ) -> PinnedCourtSide:
        side = VideoCourtSideArtifact.load(side_directory, digest)
        if side.decision.evidence != "ball":
            raise ValueError(
                "Pose reconstruction requires a ball-based shared court side"
            )
        side.output()  # Reject an undecided pinned video up front.
        court = VideoCourtStore(court_directory).load(side.decision.court_digest)
        drift = side.decision.court_drift_digest
        if (drift is None) != (drift_directory is None):
            raise ValueError(
                "Shared court drift store must match the pinned side decision"
            )
        if drift is not None and drift_directory is not None:
            court = load_video_drift(court, DriftReference(drift_directory, drift))
        side.check_court(court, "warn")
        return cls(side, court)

    @property
    def clip_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._side.clips))

    def bind(
        self, source: ClipSource, calibration: CourtCalibrationOutput
    ) -> tuple[tuple[PinholeCamera, ...], dict[str, Any]]:
        decision = self._side.decision
        if (
            source.clip_id not in self._side.clips
            or source.camera_ids != decision.camera_ids
        ):
            raise ValueError(
                "Clip or camera order differs from the pinned video court side"
            )
        self._court.check_source(source)
        require_accepted_courts(self._court, (source.clip_id,), "warn")
        self._side.require_clip(source.clip_id, "warn")
        if (
            calibration.calibration.camera_ids != source.camera_ids
            or calibration.calibration.excluded
            or shared_calibration_digest(source, calibration, self._court)
            != decision.calibration_digest
        ):
            raise ValueError("Camera matrices differ from the pinned video court side")
        for video in source.videos:
            if not video.path.is_file() or dual_sha256(video.path) != video.sha256:
                raise ValueError(
                    f"Source video SHA-256 mismatch: {source.clip_id} {video.camera_id}"
                )
        turns = self._side.output().view_half_turns
        cameras = tuple(
            view.camera.half_turned(turn)
            for view, turn in zip(calibration.calibration.views, turns, strict=True)
        )
        return cameras, {
            "digest": self._side.digest,
            "diagnostics_digest": decision.diagnostics_digest,
            "dataset_id": decision.dataset_id,
            "video_id": decision.video_id,
            "evidence": decision.evidence,
            "court_digest": decision.court_digest,
            "drift_digest": decision.court_drift_digest,
            "calibration_digest": decision.calibration_digest,
            "reference_camera": decision.reference_camera,
            "view_half_turns": list(turns),
            "clip_verdict": self._side.clip_verdict(source.clip_id, "warn"),
        }


@dataclass(frozen=True)
class SavedPoseInputs:
    observations: PoseInputs
    references: dict[str, ArtifactRef]
    court_side: dict[str, Any]


def _descriptor(
    root: Path, index: dict[str, Any], node: str, schema: str, version: int
) -> tuple[ArtifactRef, dict[str, Any]]:
    raw = index["artifacts"].get(node)
    if not isinstance(raw, dict) or (raw.get("schema"), raw.get("version")) != (
        schema,
        version,
    ):
        raise ValueError(f"{node} requires {schema} v{version}")
    ref = ArtifactRef(**raw)
    doc = read_component_descriptor(
        root, raw, node=node, source_sha256=index["source_sha256"]
    )
    if doc.get("status") != "complete" or doc.get("execution_key") != ref.execution_key:
        raise ValueError(f"Incomplete or inconsistent input artifact: {node}")
    return ref, doc


def _value(
    root: Path, ref: ArtifactRef, document: dict[str, Any], output: type[_Value]
) -> _Value:
    value: _Value = ArtifactCodec(output).load(
        document["payload"], (root / ref.path).parent, document["arrays"]
    )
    return value


def read_saved_inputs(directory: Path, side: PinnedCourtSide) -> SavedPoseInputs:
    """Load the complete dependency closure through its explicitly named ports."""
    root = directory.resolve()
    index = json.loads((root / "scene.json").read_text())
    if index.get("schema") != "tennis_scene_index_v1":
        raise ValueError("Expected a tennis scene input store")
    source = restore_type(index["source"], ClipSource)
    if document_digest(source) != index["source_sha256"]:
        raise ValueError("Input store source digest mismatch")
    court_node = "court_calibration"
    court_ref, court_doc = _descriptor(
        root, index, court_node, "local_court_calibration", 1
    )
    refs = {court_node: court_ref}
    court_dependencies = {}
    detections = []
    poses = []
    model_selection = None
    for camera in source.camera_ids:
        court_observation = f"court_detection/{camera}"
        detection = f"person_detection/{camera}"
        pose = f"person_pose_features/{camera}"
        c_ref, c_doc = _descriptor(
            root, index, court_observation, "court_observations", 3
        )
        d_ref, d_doc = _descriptor(root, index, detection, "person_detections", 2)
        p_ref, p_doc = _descriptor(root, index, pose, "person_pose_features", 2)
        if (
            c_doc["dependencies"] != {}
            or d_doc["dependencies"] != {}
            or p_doc["dependencies"] != {"detections": index["artifacts"][detection]}
        ):
            raise ValueError(
                f"Unexpected raw detection, pose or court dependencies: {camera}"
            )
        models = stored_people_models(d_doc["identity"].get("settings"), p_doc["identity"].get("settings"),
                                      camera=camera, require_unmerged=True)
        if model_selection is not None and model_selection != models:
            raise ValueError(f"People model selection changes between cameras: {camera}")
        model_selection = models
        refs.update({court_observation: c_ref, detection: d_ref, pose: p_ref})
        court_dependencies[camera] = index["artifacts"][court_observation]
        detections.append(_value(root, d_ref, d_doc, PersonDetectionOutput))
        poses.append(_value(root, p_ref, p_doc, PersonPoseFeaturesOutput))
    if court_doc["dependencies"] != court_dependencies:
        raise ValueError("Unexpected court calibration dependencies")
    calibration = _value(root, court_ref, court_doc, CourtCalibrationOutput)
    cameras, provenance = side.bind(source, calibration)
    return SavedPoseInputs(
        PoseInputs(source, cameras, tuple(detections), tuple(poses)), refs, provenance
    )
