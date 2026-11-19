"""ViTPose on declared track boxes; no detection or association side effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import Pose2DRequest, TrackResult, ViTPosePose2D
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.tennis_scene.pipeline.model_resources import vitpose_model
from src.tennis_scene.pipeline.observation_types import ObjectObservations


@dataclass(frozen=True)
class PoseEstimationInput:
    video: SourceVideo
    tracks: PersonTrackingOutput


class PoseEstimationModule(InferenceComponent[PoseEstimationInput, ObjectObservations]):
    io = ComponentIO("pose_estimation", PoseEstimationInput, ObjectObservations,
        {"selection": InputPort("selected_player_tracks", 3)}, "person_poses")

    def __init__(self, config: PeopleModelConfig, *, require_evidence: bool = False, enabled: bool = True) -> None:
        self.config = config
        self.require_evidence = require_evidence
        self.enabled = enabled

    def model_specs(self) -> dict[str, ModelSpec[ViTPosePose2D]]:
        return {} if self.require_evidence or not self.enabled else {"pose": vitpose_model(self.config, factory=ViTPosePose2D, show_progress=True)}

    def process_with_models(self, inputs: PoseEstimationInput, models: ModelBorrower) -> ObjectObservations:
        video, tracks = inputs.video, inputs.tracks
        if tracks.camera_id != video.camera_id or tracks.observed.shape[1] != video.num_frames:
            raise ValueError("Pose input camera/timeline mismatch")
        ids = tracks.track_ids.tolist()
        if ids and not self.enabled:
            raise ValueError('Disabled pose estimation cannot consume person tracks')
        tracked = TrackResult({i: torch.from_numpy(np.array(tracks.boxes_xyxy[p], copy=True)) for p, i in enumerate(ids)}, video.num_frames,
            {i: torch.from_numpy(np.array(tracks.observed[p], copy=True)) for p, i in enumerate(ids)})
        shape = (1, video.num_frames, len(ids), 17)
        uv: NDArray[np.float32] = np.zeros((*shape, 2), np.float32)
        confidence: NDArray[np.float32] = np.zeros(shape, np.float32)
        boxes: NDArray[np.float32] = np.zeros((*shape[:3], 3), np.float32)
        config = self.config
        if tracks.evidence is not None:
            for p, track_id in enumerate(ids):
                boxes[0, :, p] = tracked.bbx_xys(track_id, base_enlarge=config.runtime.tracking.bbox_enlarge).numpy()
            uv[:] = tracks.evidence.require_poses()[..., :2].transpose(1, 0, 2, 3)[None]
            confidence[:] = tracks.evidence.require_poses()[..., 2].transpose(1, 0, 2)[None]
            return ObjectObservations((video.camera_id,), (video.width, video.height), video.fps,
                uv, confidence, tracks.observed.T[None], tracks.track_ids[None], boxes)
        if self.require_evidence:
            raise ValueError('Feature tracker lost per-detection pose evidence; regenerate upstream artifacts')
        pose: ViTPosePose2D | None = None
        for p, track_id in enumerate(ids):
            indices = tracked.observed_mask(track_id).nonzero().flatten()
            track_boxes = tracked.bbx_xys(track_id, base_enlarge=config.runtime.tracking.bbox_enlarge)
            boxes[0, :, p] = track_boxes.numpy()
            if not len(indices):
                continue
            if pose is None:
                pose = models.borrow("pose")
            assert pose is not None
            result = pose.predict(Pose2DRequest(video.path, track_boxes[indices], frame_indices=indices))
            if result.keypoints.shape != (len(indices), 17, 3) or not bool(torch.isfinite(result.keypoints).all()):
                raise ValueError("ViTPose output does not preserve observed frame IDs")
            uv[0, indices.numpy(), p] = result.keypoints[..., :2].numpy()
            confidence[0, indices.numpy(), p] = result.keypoints[..., 2].numpy()
        return ObjectObservations((video.camera_id,), (video.width, video.height), video.fps,
            uv, confidence, tracks.observed.T[None], tracks.track_ids[None], boxes)
