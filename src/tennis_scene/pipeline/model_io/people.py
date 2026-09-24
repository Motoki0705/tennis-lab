"""Separate 2D people observations from the expensive body-recovery chain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.configuration import BundledModelAssetPaths, SubmoduleRuntimeConfig
from src.submodules.models import Pose2DRequest, TrackRequest, TrackResult
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.model_io.gvhmr import (
    GVHMRChainAdapter,
    build_gvhmr_chain,
)
from src.tennis_scene.pipeline.model_io.observations import ObjectObservations
from src.utils.video import VideoInfo


@dataclass(frozen=True)
class PeopleModelConfig:
    detector: str
    dino_checkpoint: Path
    dino_repository: Path
    yolo_checkpoint: Path
    vitpose_checkpoint: Path
    hmr2_checkpoint: Path
    gvhmr_checkpoint: Path
    body_models_dir: Path
    bundled_assets: BundledModelAssetPaths
    runtime: SubmoduleRuntimeConfig

    def __post_init__(self) -> None:
        if self.detector != "dino":
            raise ValueError("Automatic scene people observations use the DINO detector")
        if not self.runtime.static_cam:
            raise ValueError("Automatic reconstruction currently requires static cameras")


class PersonObservationAdapter:
    """The component sees observations; only this adapter knows submodel layouts."""

    def __init__(self, chain: GVHMRChainAdapter, *, bbox_enlarge: float) -> None:
        self.chain = chain
        self.bbox_enlarge = bbox_enlarge

    def observe(
        self,
        video_paths: Sequence[Path],
        video_infos: Sequence[VideoInfo],
        camera_ids: tuple[str, ...],
        *,
        num_frames: int,
        polygons: Sequence[tuple[tuple[float, float], ...] | None],
    ) -> ObjectObservations:
        if not (len(video_paths) == len(video_infos) == len(camera_ids) == len(polygons)):
            raise ValueError("People observation camera metadata disagree")
        tracks: list[TrackResult] = []
        try:
            for path, polygon in zip(video_paths, polygons, strict=True):
                result = self.chain.tracker.predict(TrackRequest(path, None, False, footpoint_polygon_px=polygon, max_frames=num_frames))
                if result.num_frames != num_frames:
                    raise ValueError("Tracker did not preserve the requested timeline")
                masks = [result.observed_mask(track_id) for track_id in result.track_ids]
                if masks and bool((torch.stack(masks).sum(0) > 4).any()):
                    raise ReconstructionUnavailable("person_capacity_exceeded", "More than four observed persons in the target court")
                tracks.append(result)
        finally:
            self.chain.tracker.unload()
        carriers = max((len(t.track_ids) for t in tracks), default=0)
        views = len(camera_ids)
        uv: NDArray[np.float32] = np.zeros((views, num_frames, carriers, 17, 2), np.float32)
        confidence = np.zeros(uv.shape[:-1], np.float32)
        observed: NDArray[np.bool_] = np.zeros((views, num_frames, carriers), bool)
        boxes: NDArray[np.float32] = np.zeros((views, num_frames, carriers, 3), np.float32)
        local_ids: NDArray[np.int64] = np.full((views, carriers), -1, np.int64)
        try:
            for view, (path, result) in enumerate(zip(video_paths, tracks, strict=True)):
                for carrier, track_id in enumerate(result.track_ids):
                    mask = result.observed_mask(track_id)
                    indices = mask.nonzero().flatten()
                    track_boxes = result.bbx_xys(track_id, base_enlarge=self.bbox_enlarge)
                    local_ids[view, carrier] = track_id
                    observed[view, :, carrier] = mask.numpy()
                    boxes[view, :, carrier] = track_boxes.numpy()
                    if not len(indices):
                        continue
                    pose = self.chain.pose_model.predict(Pose2DRequest(path, track_boxes[indices], frame_indices=indices))
                    if pose.keypoints.shape != (len(indices), 17, 3) or not bool(torch.isfinite(pose.keypoints).all()):
                        raise ValueError("ViTPose result does not match actual detector frames")
                    uv[view, indices.numpy(), carrier] = pose.keypoints[..., :2].numpy()
                    confidence[view, indices.numpy(), carrier] = pose.keypoints[..., 2].numpy()
        finally:
            self.chain.pose_model.unload()
        info = video_infos[0]
        return ObjectObservations(camera_ids, (info.width, info.height), info.fps, uv, confidence, observed, local_ids, boxes)


def build_people_chain(config: PeopleModelConfig) -> GVHMRChainAdapter:
    return build_gvhmr_chain(config)
