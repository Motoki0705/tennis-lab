"""Read explicitly indexed camera-local context, never guess a cached revision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import ClipRecord
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.utils.checksum import dual_sha256

POSE_JOINTS = (7, 8, 9, 10)
COURT_KEYPOINTS = 14


@dataclass(frozen=True)
class PoseContext:
    uv: NDArray[np.float32]  # T,D,4,2; source endpoint normalized
    confidence: NDArray[np.float32]  # T,D,4
    valid: NDArray[np.bool_]


@dataclass(frozen=True)
class CourtContext:
    uv: NDArray[np.float32]  # 14,2; source endpoint normalized
    confidence: NDArray[np.float32]  # 14
    valid: NDArray[np.bool_]


@dataclass(frozen=True)
class PipelineContext:
    """Missing execution is None; executed but undetected points use masks."""

    pose: PoseContext | None
    court: CourtContext | None
    provenance: dict[str, Any]

    def require_complete(self) -> tuple[PoseContext, CourtContext]:
        if self.pose is None or self.court is None:
            raise ValueError(f"Context was not generated: {self.provenance}")
        return self.pose, self.court


def meiji_scene_index(clip: ClipRecord, root: Path) -> Path:
    parts: list[str] = clip.clip_id.split("/")
    if (len(parts) != 4 or parts[0] != "meiji" or parts[1] != clip.group_id
            or parts[3] != clip.camera_id or any(p in {"", ".", ".."} for p in parts)):
        raise ValueError(f"Unexpected Meiji clip identity: {clip.clip_id}")
    return root / parts[1] / parts[2] / "scene.json"


def load_pipeline_context(
    clip: ClipRecord, scene_index: Path, *, pose_threshold: float,
) -> PipelineContext:
    """Verify media hash, size, complete frame timeline, source and lineage.

    The pipeline's per-frame pose uses the exact video's dense frame index;
    source hash + full frame count bind it to the store's real PTS. No FPS
    resampling or nearest-time matching is performed. Only frame 0 court is
    accepted. Missing nodes are an audit result, not all-zero observations;
    malformed, superseded or mismatched nodes fail immediately.
    """
    if not np.isfinite(pose_threshold) or not 0 <= pose_threshold <= 1:
        raise ValueError("pose_threshold must be in [0,1]")
    if clip.camera_id is None:
        raise ValueError("Pipeline context requires an explicit camera ID")
    if min(clip.source_width, clip.source_height) <= 1:
        raise ValueError("Source size must support endpoint normalization")
    index = scene_index.resolve()
    provenance: dict[str, Any] = {
        "scene_index": str(index), "pose_threshold": pose_threshold,
        "pose_status": "missing_scene", "court_status": "missing_scene",
    }
    if not index.is_file():
        return PipelineContext(None, None, provenance)
    document = json.loads(index.read_text())
    source = document["source"]
    if document["schema"] != "tennis_scene_index_v1" or document_digest(source) != document["source_sha256"]:
        raise ValueError(f"Invalid pipeline source digest: {index}")
    videos = [video for video in source["videos"] if video["camera_id"] == clip.camera_id]
    if len(videos) != 1:
        raise ValueError(f"Pipeline must contain exactly one source camera {clip.camera_id}")
    video = videos[0]
    if clip.source == "meiji" and source["clip_id"] != "/".join(clip.clip_id.split("/")[1:3]):
        raise ValueError(f"Pipeline clip identity mismatch: {clip.clip_id}")
    expected = (clip.media_sha256, clip.source_width, clip.source_height, clip.frame_count)
    actual = (video["sha256"], video["width"], video["height"], video["num_frames"])
    if actual != expected or not np.isclose(video["fps"], float(Fraction(clip.fps)), rtol=1e-6):
        raise ValueError(f"Pipeline media/size/timeline mismatch: {clip.clip_id}")
    provenance.update(scene_index_sha256=dual_sha256(index), source_sha256=document["source_sha256"],
                      timeline="exact_media_dense_frame_index_bound_to_store_pts")
    artifacts = document["artifacts"]
    pose_node, court_node = f"pose_estimation/{clip.camera_id}", f"court_detection/{clip.camera_id}"
    selected = {name: artifacts[name] for name in (pose_node, court_node) if name in artifacts}
    if selected:
        assert_current_component_lineage(document, index.parent, selected)

    def descriptor(node: str, schema: str, version: int) -> dict[str, Any] | None:
        reference = selected.get(node)
        if reference is None:
            return None
        if (reference["schema"], reference["version"]) != (schema, version):
            raise ValueError(f"Unsupported pipeline context schema: {node}")
        provenance[node] = reference
        result: dict[str, Any] = read_component_descriptor(index.parent, reference, node=node, source_sha256=document["source_sha256"])
        return result

    pose: PoseContext | None = None
    item = descriptor(pose_node, "person_poses", 1)
    provenance["pose_status"] = "not_generated"
    if item is not None:
        observations = ArtifactCodec(ObjectObservations).load(
            item["payload"], (index.parent / selected[pose_node]["path"]).parent, item["arrays"],
        )
        if (observations.camera_ids != (clip.camera_id,) or observations.num_frames != clip.frame_count
                or observations.size != (clip.source_width, clip.source_height)
                or observations.uv_px.shape[-2] != 17
                or not np.isclose(observations.fps, float(Fraction(clip.fps)), rtol=1e-6)):
            raise ValueError(f"Pose axes/timeline mismatch: {clip.clip_id}")
        uv = np.take(observations.uv_px[0], POSE_JOINTS, axis=2)
        raw_confidence = np.take(observations.confidence[0], POSE_JOINTS, axis=2)
        # ViTPose emits nonnegative heatmap peaks, not calibrated probabilities.
        # The refiner's bounded feature contract uses an explicit saturation.
        confidence = np.minimum(raw_confidence, np.float32(1))
        provenance.update(pose_confidence_transform="nonnegative_heatmap_peak_saturate_at_one.v1",
                          pose_confidence_saturated_slots=int((raw_confidence > 1).sum()),
                          pose_confidence_raw_max=float(raw_confidence.max(initial=0)))
        # Keep finite outside-image context, unlike downstream visibility gates.
        valid = observations.observed[0, ..., None] & (confidence >= pose_threshold) & (confidence > 0)
        denominator = np.asarray((clip.source_width - 1, clip.source_height - 1), np.float32)
        pose = PoseContext(uv / denominator, confidence, valid)
        provenance.update(pose_status="loaded", pose_frames=int(valid.any(axis=(1, 2)).sum()))
    court: CourtContext | None = None
    item = descriptor(court_node, "court_observations", 2)
    provenance["court_status"] = "not_generated"
    if item is not None:
        observation = ArtifactCodec(CourtKPResult).load(
            item["payload"], (index.parent / selected[court_node]["path"]).parent, item["arrays"],
        )
        ok, errors = observation.validate()
        diagnostics = observation.diagnostics
        if (not ok or observation.keypoints.shape != (1, 1, COURT_KEYPOINTS, 2)
                or observation.frame_indices.dtype != np.int32
                or observation.frame_indices.tolist() != [0] or diagnostics is None
                or diagnostics.get("temporal_policy") != "static_first_frame"
                or diagnostics.get("source_frame_count") != clip.frame_count
                or diagnostics.get("output_keypoint_contract") != "camera_view_v2"
                or [c["camera_id"] for c in diagnostics.get("cameras", [])] != [clip.camera_id]):
            raise ValueError(f"Court frame-0/schema contract mismatch: {clip.clip_id}: {errors}")
        uv = observation.keypoints[0, 0]
        confidence = observation.visibility[0, 0]
        if uv.dtype != np.float32 or confidence.dtype != np.float32:
            raise ValueError("Court context must be float32")
        if ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("Court confidence must be in [0,1]")
        court = CourtContext(uv, confidence, confidence > 0)
        provenance.update(court_status="loaded", court_keypoints=int(court.valid.sum()))
    return PipelineContext(pose, court, provenance)
