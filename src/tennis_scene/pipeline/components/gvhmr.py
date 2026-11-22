"""GVHMR temporal inference over persisted, source-checked HMR2 image features."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby, islice
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
from src.tennis_scene.pipeline.model_resources import gvhmr_tensorrt
from src.tennis_scene.pipeline.tensorrt_assets import tensorrt_identity
from src.utils.models.tensorrt_artifacts import EXPORT_MANIFEST, ExportManifest


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
        {"selection": InputPort("body_view_selection"), "features": InputPort("hmr2_features")}, "body_parameters", version=2,
        after_all_inputs=("features",))

    def __init__(self, config: PeopleModelConfig, *, enabled: bool = True) -> None:
        self.config, self.enabled = config, enabled

    def model_specs(self) -> dict[str, ModelSpec[Any]]:
        config = self.config
        if not self.enabled:
            return {}
        runtime = config.runtime.gvhmr
        assets = {k: v for k, v in config.body_assets().items() if k != "hmr2"}
        settings: dict[str, Any] = {"static_cam": config.runtime.static_cam}
        if runtime.batch_size != 1:
            settings["batch_size"] = runtime.batch_size
        dtype = "float32"
        kwargs = {}
        if runtime.backend.engines:
            root = runtime.backend.engines[0]
            manifest = ExportManifest.model_validate_json((root / EXPORT_MANIFEST).read_bytes())
            engine = gvhmr_tensorrt(config, legacy="attention_mask" not in manifest.settings).bind(
                root, device=config.runtime.device, batch_size=runtime.batch_size)
            engine_assets, identity = tensorrt_identity((engine,), batch_size=runtime.batch_size)
            assets.update(engine_assets)
            settings.update(identity)
            dtype = "float16"
            kwargs["engine"] = engine
        return {
            "mesh": ModelSpec("gvhmr", assets, "gvhmr", config.runtime.device, dtype, settings,
                lambda: GvhmrMeshRecovery(config.gvhmr_checkpoint, config.body_models_dir,
                    device=config.runtime.device, smpl_assets=config.smpl_assets, **kwargs),
                initialize=lambda model: model.load(), release=lambda model: model.unload()),
        }

    def process_with_models(self, inputs: GVHMRInput, models: ModelBorrower) -> GVHMROutput:
        if not self.enabled:
            return GVHMROutput(())
        inputs.validate()
        selections = inputs.selection.selections
        segments: list[list[RecoveredBodySegment]] = [[] for _ in selections]
        jobs = (
            (i, request, observed, feature)
            for i, (selected, image) in enumerate(zip(selections, inputs.features.bodies, strict=True))
            for request, observed, feature in zip(selected.requests, selected.observed_samples, image.segments, strict=True)
        )
        limit = self.config.runtime.gvhmr.batch_size
        for _, compatible in groupby(jobs, key=lambda job: len(job[1].source_frames)):
            while batch := list(islice(compatible, limit)):
                requests = [GvhmrRequest(
                    kp2d=torch.from_numpy(np.array(request.keypoints, copy=True)),
                    bbx_xys=torch.from_numpy(np.array(request.boxes_xys, copy=True)),
                    f_imgseq=torch.from_numpy(np.array(feature.features, copy=True)),
                    width=request.size[0], height=request.size[1], static_cam=True,
                    K_fullimg=torch.from_numpy(request.intrinsic.astype(np.float32)),
                ) for _, request, _, feature in batch]
                model: GvhmrMeshRecovery = models.borrow("mesh")
                results = [model.predict(requests[0])] if len(requests) == 1 else model.predict_batch(requests)
                for (i, request, observed, _), result in zip(batch, results, strict=True):
                    params = result.smpl_params_incam
                    parameters = BodyParameters(**{name: params[name].detach().float().cpu().numpy()
                        for name in ("body_pose", "global_orient", "betas", "transl")})
                    segments[i].append(RecoveredBodySegment(request.source_frames, parameters, observed))
        return GVHMROutput(tuple(
            RecoveredBody(selected.person_id, selected.camera_id, tuple(rows))
            for selected, rows in zip(selections, segments, strict=True)
        ))
