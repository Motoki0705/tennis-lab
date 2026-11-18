"""Prepare a single fixed-camera decision from the complete set of pinned clip inputs."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.court_side.hypothesis import (
    CourtSideConfig,
    collect_side_evidence_with_frames,
)
from src.tasks.court_side.video import pool_side_evidence
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideInput
from src.tennis_scene.pipeline.video_artifact import (
    VideoArtifactConfig,
    publish_document,
    store_lock,
)
from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact
from src.utils.configuration import PathResolver, PathRole

from .artifact import (
    DIAGNOSTICS_SCHEMA,
    SCHEMA,
    ClipMeasurement,
    SideAssessment,
    VideoCourtSideArtifact,
    VideoSideDecision,
)
from .inputs import PinnedSideInputs


def calibration_digest(inputs: CourtSideInput, court: VideoCourtArtifact) -> str:
    calibration = inputs.calibration
    diagnostics = calibration.court.diagnostics or {}
    refs = diagnostics.get("shared_courts", [])
    scope = court.manifest
    expected = [(scope["dataset_id"], scope["video_id"], camera, scope["cameras"][camera]) for camera in inputs.source.camera_ids]
    if (diagnostics.get("temporal_policy") != "static_video_shared_broadcast"
            or [(r["dataset_id"], r["video_id"], r["camera_id"], r["artifact_digest"]) for r in refs] != expected):
        raise ValueError("Court side inputs require calibration from the pinned video_shared court")
    for view in calibration.calibration.views:
        if not 0 <= view.source_index < len(expected) or inputs.source.camera_ids[view.source_index] != view.camera.camera_id:
            raise ValueError("Calibrated court side cameras disagree with the source view order")
    digest: str = document_digest({"cameras": [v.camera for v in calibration.calibration.views], "reference": calibration.reference_camera})
    return digest


def observations_digest(indices: NDArray[np.int64], uv: NDArray[np.float32], visible: NDArray[np.bool_]) -> str:
    digest: str = document_digest({"source_frames": indices, "uv_px": uv, "visible": visible})
    return digest


def build_video_court_side(
    stores: Mapping[str, PinnedSideInputs], *, court: VideoCourtArtifact, directory: Path,
    thresholds: CourtSideConfig, max_frames: int, code_identity: str,
    evidence_clip_ids: Sequence[str] | None = None, rebuild: bool = False,
) -> VideoCourtSideArtifact:
    """Build from saved calibration/ball points. By default every clip contributes.

    All members always receive their own diagnostic. The sorted selection, revisions,
    geometry and numerical settings identify repeat builds across processes.
    """
    court.verify()
    members = set(stores)
    if not members or any(set(camera["sources"]) != members for camera in court.cameras.values()):
        raise ValueError("Shared court side requires exactly all clips of the shared court video")
    selected = sorted(members if evidence_clip_ids is None else evidence_clip_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= members:
        raise ValueError("Empty, repeated or unknown court side evidence clip")
    for clip, pin in stores.items():
        if pin.source.clip_id != clip or set(pin.source.camera_ids) != set(court.cameras):
            raise ValueError("Court side input clip/camera membership mismatch")
        court.check_source(pin.source)
    identity: dict[str, Any] = dict(schema=SCHEMA, dataset_id=court.manifest["dataset_id"], video_id=court.manifest["video_id"],
                    court_digest=court.digest, thresholds=json_value(thresholds), max_frames=max_frames,
                    implementation=code_identity, input_revisions={clip: pin.digest for clip, pin in stores.items()},
                    evidence_clip_ids=selected)
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
            cameras = tuple(v.camera for v in inputs.calibration.calibration.views)
            indices, uv, visible = inputs.sample(max_frames)
            config = inputs.scaled_config(thresholds)
            single, scored = collect_side_evidence_with_frames(cameras, inputs.calibration.reference_camera, uv, visible, config)
            per_clip[clip] = ClipMeasurement(len(indices), tuple(indices[scored].tolist()), observations_digest(indices, uv, visible),
                                             SideAssessment.judge(single, thresholds))
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
        digest = publish_document(directory, json_value(decision), repair=rebuild)
        artifact = VideoCourtSideArtifact.load(directory, digest)
        write_json_atomic(cache, {"artifact_digest": digest})
        return artifact


def prepare_video_court_side(
    config: VideoArtifactConfig, resolver: PathResolver, *, court: VideoCourtArtifact, thresholds: CourtSideConfig,
    max_frames: int, code_identity: str,
) -> VideoCourtSideArtifact:
    if (config.dataset_id, config.video_id) != (court.manifest["dataset_id"], court.manifest["video_id"]):
        raise ValueError("Shared court and side dataset/video must match")
    if config.action == "load":
        court.verify()
        assert config.artifact_digest is not None
        artifact = VideoCourtSideArtifact.load(config.directory, config.artifact_digest)
    else:
        assert config.request is not None
        raw = config.request.read_bytes()
        if sha256(raw).hexdigest() != config.request_sha256:
            raise ValueError("Shared court side build request changed")
        request: dict[str, Any] = json.loads(raw)
        if (set(request) != {"schema", "clips", "evidence_clip_ids"} or request["schema"] != "video_court_side_inputs_v1"
                or not isinstance(request["clips"], dict)):
            raise ValueError("Invalid shared court side build request")
        pins = {}
        for clip, record in request["clips"].items():
            if set(record) != {"role", "directory", "inputs_digest"} or record["role"] not in {"data", "artifact"}:
                raise ValueError("Court side build requires pinned data/artifact component stores")
            directory = resolver.resolve(PathRole(record["role"]), record["directory"])
            pin = PinnedSideInputs.from_store(directory)
            if pin.digest != record["inputs_digest"]:
                raise ValueError(f"Pinned court side inputs changed: {clip}")
            pins[clip] = pin
        artifact = build_video_court_side(pins, court=court, directory=config.directory, thresholds=thresholds,
                                          max_frames=max_frames, code_identity=code_identity,
                                          evidence_clip_ids=request["evidence_clip_ids"], rebuild=config.rebuild)
    artifact.check_settings(court, thresholds, max_frames)
    return artifact
