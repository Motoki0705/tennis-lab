"""HMR image features and GVHMR only; detections and view choices are inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import (
    GvhmrMeshRecovery,
    GvhmrRequest,
    Hmr2FeatureExtractor,
    ImageFeatureRequest,
)
from src.tennis_scene.pipeline.body_types import BodyParameters
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort
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
    selection: BodyViewSelectionOutput


class GVHMRModule(InferenceComponent[GVHMRInput, GVHMROutput]):
    io = ComponentIO("gvhmr", GVHMRInput, GVHMROutput,
        {"selection": InputPort("body_view_selection")}, "body_parameters")

    def __init__(self, config: PeopleModelConfig, *, enabled: bool = True) -> None:
        self.config, self.enabled = config, enabled

    def model_specs(self) -> dict[str, ModelSpec[Any]]:
        config = self.config
        if not self.enabled:
            return {}
        return {
            "features": ModelSpec("hmr2", {"checkpoint": config.hmr2_checkpoint, "mean_params": config.smpl_assets.hmr2_mean_params},
                "hmr2_vit_huge", config.runtime.device, "float32", {"runtime": config.runtime.hmr2},
                lambda: Hmr2FeatureExtractor(config.hmr2_checkpoint, device=config.runtime.device,
                    batch_size=config.runtime.hmr2.batch_size, mean_params_path=config.smpl_assets.hmr2_mean_params),
                initialize=lambda model: model.load(), release=lambda model: model.unload()),
            "mesh": ModelSpec("gvhmr", {k: v for k, v in config.body_assets().items() if k != "hmr2"},
                "gvhmr", config.runtime.device, "float32", {"static_cam": config.runtime.static_cam},
                lambda: GvhmrMeshRecovery(config.gvhmr_checkpoint, config.body_models_dir,
                    device=config.runtime.device, smpl_assets=config.smpl_assets),
                initialize=lambda model: model.load(), release=lambda model: model.unload()),
        }

    def process_with_models(self, inputs: GVHMRInput, models: ModelBorrower) -> GVHMROutput:
        if not self.enabled or not inputs.selection.selections:
            return GVHMROutput(())
        features: Hmr2FeatureExtractor = models.borrow("features")
        model: GvhmrMeshRecovery = models.borrow("mesh")
        bodies: list[RecoveredBody] = []
        for selected in inputs.selection.selections:
            segments: list[RecoveredBodySegment] = []
            for request, observed_samples in zip(selected.requests, selected.observed_samples, strict=True):
                boxes = torch.from_numpy(np.array(request.boxes_xys, copy=True))
                image = features.predict(ImageFeatureRequest(request.video_path, boxes,
                    frame_indices=torch.from_numpy(np.array(request.source_frames, copy=True))))
                result = model.predict(GvhmrRequest(kp2d=torch.from_numpy(np.array(request.keypoints, copy=True)),
                    bbx_xys=boxes, f_imgseq=image.features, width=request.size[0], height=request.size[1],
                    static_cam=True, K_fullimg=torch.from_numpy(request.intrinsic.astype(np.float32))))
                params = result.smpl_params_incam
                parameters = BodyParameters(**{name: params[name].detach().float().cpu().numpy() for name in ("body_pose", "global_orient", "betas", "transl")})
                segments.append(RecoveredBodySegment(request.source_frames, parameters, observed_samples))
            bodies.append(RecoveredBody(selected.person_id, selected.camera_id, tuple(segments)))
        return GVHMROutput(tuple(bodies))
