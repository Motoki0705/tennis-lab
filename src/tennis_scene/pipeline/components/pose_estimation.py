"""ViTPose on declared track boxes; no detection or association side effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import Pose2DRequest, TrackResult, ViTPosePose2D
from src.tennis_scene.pipeline.components.base import release_inference_memory
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.tennis_scene.pipeline.observation_types import ObjectObservations


@dataclass(frozen=True)
class PoseEstimationInput:
    video: SourceVideo
    tracks: PersonTrackingOutput


class PoseEstimationModule:
    io = ComponentIO("pose_estimation", PoseEstimationInput, ObjectObservations,
        {"tracks": InputPort("person_tracks")}, "person_poses")

    def __init__(self, config: PeopleModelConfig) -> None:
        self.config = config

    def process(self, inputs: PoseEstimationInput) -> ObjectObservations:
        video, tracks = inputs.video, inputs.tracks
        if tracks.camera_id != video.camera_id or tracks.observed.shape[1] != video.num_frames:
            raise ValueError("Pose input camera/timeline mismatch")
        ids = tracks.track_ids.tolist()
        tracked = TrackResult({i: torch.from_numpy(np.array(tracks.boxes_xyxy[p], copy=True)) for p, i in enumerate(ids)}, video.num_frames,
            {i: torch.from_numpy(np.array(tracks.observed[p], copy=True)) for p, i in enumerate(ids)})
        shape = (1, video.num_frames, len(ids), 17)
        uv: NDArray[np.float32] = np.zeros((*shape, 2), np.float32)
        confidence: NDArray[np.float32] = np.zeros(shape, np.float32)
        boxes: NDArray[np.float32] = np.zeros((*shape[:3], 3), np.float32)
        config = self.config
        pose = ViTPosePose2D(config.vitpose_checkpoint, device=config.runtime.device,
            flip_test=config.runtime.vitpose.flip_test, batch_size=config.runtime.vitpose.batch_size, head_config=config.runtime.vitpose.head)
        try:
            for p, track_id in enumerate(ids):
                indices = tracked.observed_mask(track_id).nonzero().flatten()
                track_boxes = tracked.bbx_xys(track_id, base_enlarge=config.runtime.tracking.bbox_enlarge)
                boxes[0, :, p] = track_boxes.numpy()
                if not len(indices):
                    continue
                result = pose.predict(Pose2DRequest(video.path, track_boxes[indices], frame_indices=indices))
                if result.keypoints.shape != (len(indices), 17, 3) or not bool(torch.isfinite(result.keypoints).all()):
                    raise ValueError("ViTPose output does not preserve observed frame IDs")
                uv[0, indices.numpy(), p] = result.keypoints[..., :2].numpy()
                confidence[0, indices.numpy(), p] = result.keypoints[..., 2].numpy()
        finally:
            pose.unload()
            release_inference_memory(config.runtime.device)
        return ObjectObservations((video.camera_id,), (video.width, video.height), video.fps,
            uv, confidence, tracks.observed.T[None], tracks.track_ids[None], boxes)
