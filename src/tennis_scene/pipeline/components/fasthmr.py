"""Recover native SMPL parameters for the fixed source views and timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from torch import Tensor

from src.submodules.models.fasthmr.assets import (
    CHECKPOINT_REVISION,
    IMAGE_SIZE,
    MERGE_RATIO,
    MODEL_ARCHITECTURE,
    UPSTREAM_COMMIT,
    WINDOW_LENGTH,
    FastHMRConfig,
)
from src.submodules.models.fasthmr.inference import FastHMR, FastHMRPrediction
from src.submodules.models.fasthmr.upstream import OfficialSMPL
from src.tennis_scene.pipeline.body_types import (
    BodyGeometry,
    BodyParameters,
    NativeSMPLParameters,
)
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.gvhmr import (
    RecoveredBody,
    RecoveredBodySegment,
)
from src.tennis_scene.pipeline.components.hmr2_features import validate_body_selection
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.recipes import FASTHMR_RECIPE


@dataclass(frozen=True)
class FastHMRInput:
    source: ClipSource
    selection: BodyViewSelectionOutput


@dataclass(frozen=True)
class NativeSMPLSegment(RecoveredBodySegment):
    parameters: NativeSMPLParameters


@dataclass(frozen=True)
class FastHMRBody(RecoveredBody):
    segments: tuple[NativeSMPLSegment, ...]
    video_sha256: str


@dataclass(frozen=True)
class FastHMROutput:
    clip_id: str
    bodies: tuple[FastHMRBody, ...]

    def validate_source(self, source: ClipSource) -> None:
        if self.clip_id != source.clip_id:
            raise ValueError("FastHMR output belongs to a different clip")
        for body in self.bodies:
            if body.camera_id not in source.camera_ids or body.video_sha256 != source.video(body.camera_id).sha256:
                raise ValueError("FastHMR output belongs to a different source camera or video")

    def validate_selection(self, inputs: FastHMRInput) -> None:
        validate_body_selection(inputs.source, inputs.selection)
        self.validate_source(inputs.source)
        for body, selected in zip(self.bodies, inputs.selection.selections, strict=True):
            if (body.person_id, body.camera_id) != (selected.person_id, selected.camera_id):
                raise ValueError("FastHMR output disagrees with the selected person or camera")
            for segment, request, observed in zip(body.segments, selected.requests, selected.observed_samples, strict=True):
                if (len(segment.parameters.transl) != len(request.source_frames)
                        or not np.array_equal(segment.source_frames, request.source_frames)
                        or segment.observed_samples != observed):
                    raise ValueError("FastHMR output disagrees with the selected frames or observation count")


def native_smpl_parameters(prediction: FastHMRPrediction, length: int) -> NativeSMPLParameters:
    """Keep every SMPL joint and the root rotation in OpenCV camera axes."""
    rotations = prediction.rotations
    if rotations.shape != (length, 24, 3, 3) or not np.isfinite(rotations).all():
        raise ValueError("FastHMR HybrIK must return finite SMPL rotations (T,24,3,3)")
    if (not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=2e-3)
            or not np.allclose(np.linalg.det(rotations), 1, atol=2e-3)):
        raise ValueError("FastHMR HybrIK matrices must be proper rotations")
    angles = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().astype(np.float32).reshape(length, 24, 3)
    return NativeSMPLParameters(angles[:, 1:].reshape(length, 69), angles[:, 0], prediction.betas, prediction.translation)


class NativeSMPLGeometry(OfficialSMPL):
    @property
    def root_regressor(self) -> NDArray[np.float64]:
        if self.layer is None:
            raise RuntimeError("FastHMR SMPL geometry has not been loaded")
        regressor: Tensor = self.layer.J_regressor
        return regressor[0].detach().cpu().double().numpy()

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry:
        if not isinstance(parameters, NativeSMPLParameters):
            raise TypeError("Native SMPL geometry requires native SMPL parameters")
        if self.layer is None or self.coco17 is None:
            raise RuntimeError("FastHMR SMPL geometry has not been loaded")
        values = {key: value.to(self.config.device) for key, value in parameters.tensors().items()}
        vertices: Tensor = self.layer(
            pose_axis_angle=values["body_pose"], betas=values["betas"],
            global_orient=values["global_orient"], transl=values["transl"],
        ).vertices
        joints = torch.einsum("jv,tvc->tjc", self.coco17, vertices)
        return BodyGeometry(vertices.cpu().numpy(), joints.cpu().numpy())


def native_smpl_spec(config: FastHMRConfig) -> ModelSpec[NativeSMPLGeometry]:
    return ModelSpec(
        "body_geometry", {**config.geometry_weights, **config.verify_repository()}, "fasthmr_native_smpl_coco17",
        config.device, "float32", {"upstream_commit": UPSTREAM_COMMIT}, lambda: NativeSMPLGeometry(config),
        initialize=lambda body: body.load(), release=lambda body: body.unload(),
    )


class FastHMRModule(InferenceComponent[FastHMRInput, FastHMROutput]):
    io = ComponentIO(
        FASTHMR_RECIPE, FastHMRInput, FastHMROutput,
        {"selection": InputPort("body_view_selection")}, "body_parameters", version=3,
    )

    def __init__(self, config: FastHMRConfig) -> None:
        self.config = config

    def model_specs(self) -> dict[str, ModelSpec[Any]]:
        config = self.config
        geometry = native_smpl_spec(config)
        return {
            "mesh": ModelSpec(
                FASTHMR_RECIPE, {**geometry.weights, **config.inference_weights, "config": config.demo_config},
                MODEL_ARCHITECTURE, config.device, "float16" if config.backend == "pt16" else "float32",
                {"upstream_commit": UPSTREAM_COMMIT, "release": CHECKPOINT_REVISION, "backend": config.backend,
                 "feature_batch_size": config.feature_batch_size, "diffusion_seed": config.seed,
                 "window_length": WINDOW_LENGTH, "image_size": IMAGE_SIZE, "token_merging_ratio": MERGE_RATIO,
                 "crop": "source_square_cv2", "camera": "humanfov_condition_diagonal_focal_translation",
                 "flip_test": True}, lambda: FastHMR(config),
                initialize=lambda model: model.load(), release=lambda model: model.unload(),
            ),
            "body": geometry,
        }

    def estimate_working_bytes(self, inputs: FastHMRInput) -> int:
        source, selections = inputs.source, inputs.selection.selections
        rows = max((sum(len(r.source_frames) for r in b.requests) for b in selections), default=0)
        batch = min(rows, self.config.feature_batch_size)
        return int(batch * (source.size[0] * source.size[1] * 3 + IMAGE_SIZE**2 * 32) + rows * (2 * 1024 + 512) * 4)

    def process_with_models(self, inputs: FastHMRInput, models: ModelBorrower) -> FastHMROutput:
        validate_body_selection(inputs.source, inputs.selection)
        bodies = []
        for selected in inputs.selection.selections:
            video = inputs.source.video(selected.camera_id)
            segments = []
            if selected.requests:
                model: FastHMR = models.borrow("mesh")
                body: NativeSMPLGeometry = models.borrow("body")
                predictions = model.predict(
                    video.path, inputs.source.size, [(r.source_frames, r.boxes_xys) for r in selected.requests], body,
                )
                for request, count, prediction in zip(selected.requests, selected.observed_samples, predictions, strict=True):
                    samples = request.source_frames
                    segments.append(NativeSMPLSegment(samples, native_smpl_parameters(prediction, len(samples)), count))
            bodies.append(FastHMRBody(selected.person_id, selected.camera_id, tuple(segments), video.sha256))
        result = FastHMROutput(inputs.source.clip_id, tuple(bodies))
        result.validate_selection(inputs)
        return result
