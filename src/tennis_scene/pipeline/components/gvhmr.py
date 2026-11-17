"""GVHMR temporal inference over persisted, source-checked HMR2 image features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import GvhmrMeshRecovery, GvhmrRequest
from src.tennis_scene.pipeline.body_types import BodyParameters
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.hmr2_features import (
    HMR2FeaturesOutput,
    validate_body_selection,
)
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig


@dataclass(frozen=True)
class RecoveredBodySegment:
    source_frames: NDArray[np.int64]
    parameters: BodyParameters
    observed_samples: int


@dataclass(frozen=True)
class RecoveredBody:
    person_id: int
    camera_id: str
    segments: tuple[RecoveredBodySegment, ...]


@dataclass(frozen=True)
class GVHMROutput:
    bodies: tuple[RecoveredBody, ...]


@dataclass(frozen=True)
class GVHMRInput:
    source: ClipSource
    selection: BodyViewSelectionOutput
    features: HMR2FeaturesOutput

    def validate(self) -> None:
        validate_body_selection(self.source, self.selection)
        if self.features.clip_id != self.source.clip_id or len(self.features.bodies) != len(self.selection.selections):
            raise ValueError("HMR2 features do not match the selected clip/persons")
        for selected, body in zip(self.selection.selections, self.features.bodies, strict=True):
            video = self.source.video(selected.camera_id)
            if (body.person_id, body.camera_id, body.video_sha256) != (selected.person_id, selected.camera_id, video.sha256):
                raise ValueError("HMR2 features do not match the selected person/camera/source")
            if len(body.segments) != len(selected.requests):
                raise ValueError("HMR2 features do not match the selected segments")
            for index, (request, segment) in enumerate(zip(selected.requests, body.segments, strict=True)):
                if segment.segment_index != index or not np.array_equal(segment.source_frames, request.source_frames):
                    raise ValueError("HMR2 features do not match the selected segment/source frames")


class GVHMRModule(InferenceComponent[GVHMRInput, GVHMROutput]):
    io = ComponentIO("gvhmr", GVHMRInput, GVHMROutput,
        {"selection": InputPort("body_view_selection"), "features": InputPort("hmr2_features")}, "body_parameters", version=2)

    def __init__(self, config: PeopleModelConfig, *, enabled: bool = True) -> None:
        self.config, self.enabled = config, enabled

    def model_specs(self) -> dict[str, ModelSpec[Any]]:
        config = self.config
        if not self.enabled:
            return {}
        return {
            "mesh": ModelSpec("gvhmr", {k: v for k, v in config.body_assets().items() if k != "hmr2"},
                "gvhmr", config.runtime.device, "float32", {"static_cam": config.runtime.static_cam},
                lambda: GvhmrMeshRecovery(config.gvhmr_checkpoint, config.body_models_dir,
                    device=config.runtime.device, smpl_assets=config.smpl_assets),
                initialize=lambda model: model.load(), release=lambda model: model.unload()),
        }

    def process_with_models(self, inputs: GVHMRInput, models: ModelBorrower) -> GVHMROutput:
        if not self.enabled:
            return GVHMROutput(())
        inputs.validate()
        bodies: list[RecoveredBody] = []
        for selected, image in zip(inputs.selection.selections, inputs.features.bodies, strict=True):
            segments: list[RecoveredBodySegment] = []
            for request, observed_samples, segment in zip(selected.requests, selected.observed_samples, image.segments, strict=True):
                boxes = torch.from_numpy(np.array(request.boxes_xys, copy=True))
                model: GvhmrMeshRecovery = models.borrow("mesh")
                result = model.predict(GvhmrRequest(kp2d=torch.from_numpy(np.array(request.keypoints, copy=True)),
                    bbx_xys=boxes, f_imgseq=torch.from_numpy(np.array(segment.features, copy=True)), width=request.size[0], height=request.size[1],
                    static_cam=True, K_fullimg=torch.from_numpy(request.intrinsic.astype(np.float32))))
                params = result.smpl_params_incam
                parameters = BodyParameters(**{name: params[name].detach().float().cpu().numpy() for name in ("body_pose", "global_orient", "betas", "transl")})
                segments.append(RecoveredBodySegment(request.source_frames, parameters, observed_samples))
            bodies.append(RecoveredBody(selected.person_id, selected.camera_id, tuple(segments)))
        return GVHMROutput(tuple(bodies))
