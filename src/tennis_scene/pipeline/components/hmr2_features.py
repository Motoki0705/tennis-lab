"""Persist HMR2 image evidence with explicit clip/person/segment/frame identity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import Hmr2FeatureExtractor, ImageFeatureRequest
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig


@dataclass(frozen=True)
class HMR2FeatureSegment:
    segment_index: int
    source_frames: NDArray[np.int64]
    features: NDArray[np.float32]  # (L,1024), CPU float32

    def __post_init__(self) -> None:
        frames = self.source_frames
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError("HMR2 segment index must be a nonnegative integer")
        if frames.ndim != 1 or frames.dtype != np.int64 or len(frames) < 2 or (frames < 0).any() or (np.diff(frames) <= 0).any():
            raise ValueError("HMR2 features require ordered source frames")
        if self.features.shape != (len(frames), 1024) or self.features.dtype != np.float32 or not np.isfinite(self.features).all():
            raise ValueError("HMR2 features must be finite float32 (L,1024)")


@dataclass(frozen=True)
class HMR2BodyFeatures:
    person_id: int
    camera_id: str
    video_sha256: str
    segments: tuple[HMR2FeatureSegment, ...]

    def __post_init__(self) -> None:
        if type(self.person_id) is not int or not self.camera_id or not self.video_sha256:
            raise ValueError("HMR2 features require person/camera/source identity")
        if tuple(s.segment_index for s in self.segments) != tuple(range(len(self.segments))):
            raise ValueError("HMR2 segment indices must preserve selection order")
        if any(a.source_frames[-1] >= b.source_frames[0] for a, b in zip(self.segments, self.segments[1:], strict=False)):
            raise ValueError("HMR2 segments must have disjoint ordered source frames")


@dataclass(frozen=True)
class HMR2FeaturesOutput:
    clip_id: str
    bodies: tuple[HMR2BodyFeatures, ...]

    def __post_init__(self) -> None:
        if not self.clip_id or len({b.person_id for b in self.bodies}) != len(self.bodies):
            raise ValueError("HMR2 features require a clip and unique person identities")


@dataclass(frozen=True)
class HMR2FeaturesInput:
    source: ClipSource
    selection: BodyViewSelectionOutput


def validate_body_selection(source: ClipSource, selection: BodyViewSelectionOutput) -> None:
    """Reject mismatched sources before any image or temporal inference starts."""
    if len({s.person_id for s in selection.selections}) != len(selection.selections):
        raise ValueError("Body selection contains duplicate person identities")
    for selected in selection.selections:
        if selected.camera_id not in source.camera_ids:
            raise ValueError("Body selection camera does not belong to this clip")
        video = source.video(selected.camera_id)
        if len(selected.requests) != len(selected.observed_samples):
            raise ValueError("Body selection segment/observation counts disagree")
        previous = -1
        for request, observed in zip(selected.requests, selected.observed_samples, strict=True):
            if request.video_path != video.path or request.size != source.size:
                raise ValueError("Body selection video/size does not match this clip")
            if request.source_frames[0] <= previous or request.source_frames[-1] >= video.num_frames:
                raise ValueError("Body selection source frames overlap or exceed this clip")
            if type(observed) is not int or not 0 <= observed <= len(request.source_frames):
                raise ValueError("Body selection observed count is invalid")
            previous = int(request.source_frames[-1])


class HMR2FeaturesModule(InferenceComponent[HMR2FeaturesInput, HMR2FeaturesOutput]):
    io = ComponentIO("hmr2_features", HMR2FeaturesInput, HMR2FeaturesOutput,
        {"selection": InputPort("body_view_selection")}, "hmr2_features")

    def __init__(self, config: PeopleModelConfig, *, enabled: bool = True) -> None:
        self.config, self.enabled = config, enabled

    def model_specs(self) -> dict[str, ModelSpec[Hmr2FeatureExtractor]]:
        config = self.config
        return {} if not self.enabled else {
            "features": ModelSpec("hmr2", {"checkpoint": config.hmr2_checkpoint, "mean_params": config.smpl_assets.hmr2_mean_params},
                "hmr2_vit_huge", config.runtime.device, "float32", {"runtime": config.runtime.hmr2},
                lambda: Hmr2FeatureExtractor(config.hmr2_checkpoint, device=config.runtime.device,
                    batch_size=config.runtime.hmr2.batch_size, mean_params_path=config.smpl_assets.hmr2_mean_params),
                initialize=lambda model: model.load(), release=lambda model: model.unload()),
        }

    def process_with_models(self, inputs: HMR2FeaturesInput, models: ModelBorrower) -> HMR2FeaturesOutput:
        if not self.enabled:
            return HMR2FeaturesOutput(inputs.source.clip_id, ())
        validate_body_selection(inputs.source, inputs.selection)
        bodies = []
        for selected in inputs.selection.selections:
            segments = []
            for index, request in enumerate(selected.requests):
                model: Hmr2FeatureExtractor = models.borrow("features")
                result = model.predict(ImageFeatureRequest(request.video_path,
                    torch.from_numpy(np.array(request.boxes_xys, copy=True)),
                    frame_indices=torch.from_numpy(np.array(request.source_frames, copy=True))))
                segments.append(HMR2FeatureSegment(index, request.source_frames,
                    result.features.detach().cpu().numpy()))
            bodies.append(HMR2BodyFeatures(selected.person_id, selected.camera_id,
                inputs.source.video(selected.camera_id).sha256, tuple(segments)))
        return HMR2FeaturesOutput(inputs.source.clip_id, tuple(bodies))
