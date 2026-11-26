"""Prepare a single fixed-camera decision from the complete set of pinned clip inputs."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_side.hypothesis import (
    CourtSideConfig,
    PlayerCourtSideConfig,
    SideConfig,
    SideEvidence,
    collect_side_evidence_with_frames,
    evidence_kind,
)
from src.tasks.court_side.player import PlayerFootpoints, collect_player_side_evidence
from src.tasks.court_side.video import pool_side_evidence
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.components.court_side import (
    CourtSideInput,
    PlayerCourtSideInput,
)
from src.tennis_scene.pipeline.video_artifact import (
    VideoArtifactConfig,
    publish_document,
    store_lock,
)
from src.tennis_scene.pipeline.video_court.diagnostics import drift_identity
from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact
from src.utils.configuration import PathResolver

from .artifact import (
    DIAGNOSTICS_SCHEMA,
    SCHEMA,
    ClipMeasurement,
    SideAssessment,
    VideoCourtSideArtifact,
    VideoSideDecision,
    require_accepted_courts,
    shared_calibration_digest,
)
from .inputs import CourtSideBuildRequest, PinnedSideInputs, selected_evidence_clips


def calibration_digest(inputs: CourtSideInput | PlayerCourtSideInput, court: VideoCourtArtifact) -> str:
    return shared_calibration_digest(inputs.source, inputs.calibration, court)


def observations_digest(indices: NDArray[np.int64], uv: NDArray[np.float32], visible: NDArray[np.bool_]) -> str:
    digest: str = document_digest({"source_frames": indices, "uv_px": uv, "visible": visible})
    return digest


def player_observations_digest(indices: NDArray[np.int64], views: tuple[PlayerFootpoints, ...]) -> str:
    points = [[sorted(map(tuple, frame[valid].tolist())) for frame, valid in zip(view.xy_m, view.observed, strict=True)]
              for view in views]
    digest: str = document_digest({"source_frames": indices, "player_ground_xy_m": points})
    return digest


def measure_clip_side(inputs: CourtSideInput | PlayerCourtSideInput, thresholds: SideConfig,
                      max_frames: int) -> tuple[ClipMeasurement, SideEvidence]:
    calibration = inputs.calibration
    if isinstance(inputs, PlayerCourtSideInput) and isinstance(thresholds, PlayerCourtSideConfig):
        indices, footpoints = inputs.sample(max_frames, thresholds)
        single, scored = collect_player_side_evidence(calibration.calibration.camera_ids, calibration.reference_camera,
                                                      footpoints, thresholds)
        digest = player_observations_digest(indices, footpoints)
    elif isinstance(inputs, CourtSideInput) and isinstance(thresholds, CourtSideConfig):
        cameras = tuple(v.camera for v in calibration.calibration.views)
        indices, uv, visible = inputs.sample(max_frames)
        config = inputs.scaled_config(thresholds)
        single, scored = collect_side_evidence_with_frames(cameras, calibration.reference_camera, uv, visible, config)
        digest = observations_digest(indices, uv, visible)
    else:
        raise ValueError("Court side observations and thresholds must use the same evidence")
    return ClipMeasurement(len(indices), tuple(indices[scored].tolist()), digest, SideAssessment.judge(single, thresholds)), single


def build_video_court_side(
    stores: Mapping[str, PinnedSideInputs], *, court: VideoCourtArtifact, directory: Path,
    thresholds: SideConfig, max_frames: int, code_identity: str,
    evidence_clip_ids: Sequence[str] | None = None, rebuild: bool = False, court_policy: str = "warn",
) -> VideoCourtSideArtifact:
    """Build from saved calibration and the selected observations. By default every clip contributes.

    All members always receive their own diagnostic. The sorted selection, revisions,
    geometry and numerical settings identify repeat builds across processes.
    """
    court.verify()
    members = set(stores)
    if not members or any(set(camera["sources"]) != members for camera in court.cameras.values()):
        raise ValueError("Shared court side requires exactly all clips of the shared court video")
    selected = selected_evidence_clips(members, evidence_clip_ids)
    require_accepted_courts(court, selected, court_policy)
    kind = evidence_kind(thresholds)
    for clip, pin in stores.items():
        if pin.source.clip_id != clip or set(pin.source.camera_ids) != set(court.cameras):
            raise ValueError("Court side input clip/camera membership mismatch")
        if pin.evidence != kind:
            raise ValueError(f"Pinned court side evidence differs from the build settings: {clip}")
        court.check_source(pin.source)
    identity: dict[str, Any] = dict(schema=SCHEMA, dataset_id=court.manifest["dataset_id"], video_id=court.manifest["video_id"],
                    court_digest=court.digest, court_drift_digest=drift_identity(court).get("artifact_digest"),
                    thresholds=json_value(thresholds), max_frames=max_frames,
                    implementation=code_identity, input_revisions={clip: pin.digest for clip, pin in stores.items()},
                    evidence_clip_ids=list(selected))
    if kind == "player":
        identity["evidence"] = kind
    cache = directory / "requests" / f"{document_digest(identity)}.json"
    with store_lock(directory, ".video-court-side.lock"):
        if cache.exists() and not rebuild:
            cached = VideoCourtSideArtifact.load(directory, json.loads(cache.read_text())["artifact_digest"])
            if any(json_value(getattr(cached.decision, key)) != value for key, value in identity.items()):
                raise ValueError("Shared court side request identity mismatch")
            for pin in stores.values():
                pin.assert_current()
            return cached
        per_clip, evidence = {}, {}
        geometry = None
        for clip, pin in sorted(stores.items()):
            inputs = pin.assemble()
            signature = calibration_digest(inputs, court)
            if geometry is not None and geometry != signature:
                raise ValueError("Shared court side calibration changed between clips")
            geometry = signature
            per_clip[clip], single = measure_clip_side(inputs, thresholds, max_frames)
            if clip in selected:
                evidence[clip] = single
        pooled = pool_side_evidence(evidence)
        assert geometry is not None
        court.verify()
        for pin in stores.values():
            pin.assert_current()
        diagnostics = publish_document(directory, {"schema": DIAGNOSTICS_SCHEMA, "clips": json_value(per_clip)}, repair=rebuild)
        decision = VideoSideDecision(**identity, calibration_digest=geometry, camera_ids=pooled.camera_ids,
                                     reference_camera=pooled.reference_camera, shared=SideAssessment.judge(pooled, thresholds),
                                     diagnostics_digest=diagnostics)
        digest = publish_document(directory, decision.document(), repair=rebuild)
        artifact = VideoCourtSideArtifact.load(directory, digest)
        write_json_atomic(cache, {"artifact_digest": digest})
        return artifact


def verify_prepared_court_side(
    artifact: VideoCourtSideArtifact, config: VideoArtifactConfig, resolver: PathResolver, *,
    code_identity: str,
) -> None:
    """Bind a new plan or run to its requested, still-readable evidence."""
    if config.action == "load":
        # The verified object at this digest includes both the selection and all input revisions.
        if artifact.digest != config.artifact_digest:
            raise ValueError("Execution plan court side digest disagrees with configuration")
    else:
        request = CourtSideBuildRequest.read(config, resolver, evidence=artifact.decision.evidence)
        if artifact.decision.evidence_clip_ids != request.evidence_clip_ids:
            raise ValueError("Prepared court side evidence clips disagree with the build request")
        if artifact.decision.input_revisions != request.input_revisions:
            raise ValueError("Prepared court side input revisions disagree with the build request")
        if artifact.decision.implementation != code_identity:
            raise ValueError("Prepared court side implementation disagrees with the requested build")
        for pin in request.pin_inputs().values():
            pin.assert_current()
    artifact.verify()


def prepare_video_court_side(
    config: VideoArtifactConfig, resolver: PathResolver, *, court: VideoCourtArtifact, thresholds: SideConfig,
    max_frames: int, code_identity: str, court_policy: str = "warn",
) -> VideoCourtSideArtifact:
    if (config.dataset_id, config.video_id) != (court.manifest["dataset_id"], court.manifest["video_id"]):
        raise ValueError("Shared court and side dataset/video must match")
    if config.action == "load":
        assert config.artifact_digest is not None
        artifact = VideoCourtSideArtifact.load(config.directory, config.artifact_digest)
    else:
        request = CourtSideBuildRequest.read(config, resolver, evidence=evidence_kind(thresholds))
        artifact = build_video_court_side(request.pin_inputs(), court=court, directory=config.directory, thresholds=thresholds,
                                          max_frames=max_frames, code_identity=code_identity,
                                          evidence_clip_ids=request.evidence_clip_ids, rebuild=config.rebuild, court_policy=court_policy)
    artifact.check_settings(court, thresholds, max_frames, court_policy=court_policy)
    return artifact
