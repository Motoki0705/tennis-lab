"""Typed, indexed body recovery from already associated 2D observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import (
    GvhmrRequest,
    ImageFeatureRequest,
    SmplCoco17Reconstructor,
)
from src.tennis_scene.pipeline.model_io.gvhmr import GVHMRChainAdapter


@dataclass(frozen=True)
class BodyParameters:
    body_pose: NDArray[np.float32]
    global_orient: NDArray[np.float32]
    betas: NDArray[np.float32]
    transl: NDArray[np.float32]

    def __post_init__(self) -> None:
        count = len(self.body_pose)
        for name, width in (("body_pose", 63), ("global_orient", 3), ("betas", 10), ("transl", 3)):
            value = getattr(self, name)
            if value.shape != (count, width) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid body parameter {name}")

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: torch.from_numpy(getattr(self, name)) for name in ("body_pose", "global_orient", "betas", "transl")}


@dataclass(frozen=True)
class BodyRecoveryRequest:
    video_path: Path
    source_frames: NDArray[np.int64]
    keypoints: NDArray[np.float32]  # (L,17,3) pixel/confidence
    boxes_xys: NDArray[np.float32]  # (L,3) pixel boxes
    size: tuple[int, int]
    intrinsic: NDArray[np.float64]


@dataclass(frozen=True)
class BodyGeometry:
    vertices: NDArray[np.float32]
    coco17: NDArray[np.float32]


class BodyRecoveryAdapter:
    def __init__(self, chain: GVHMRChainAdapter, coco: SmplCoco17Reconstructor, root_regressor_path: Path) -> None:
        self.chain, self.coco = chain, coco
        self.root_regressor_path = root_regressor_path
        self._root_regressor: NDArray[np.float64] | None = None

    @property
    def root_regressor(self) -> NDArray[np.float64]:
        if self._root_regressor is None:
            loaded = torch.load(self.root_regressor_path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, torch.Tensor) or loaded.ndim not in (1, 2) or not bool(torch.isfinite(loaded).all()):
                raise ValueError("Invalid SMPL root regressor")
            self._root_regressor = loaded.double().numpy()
        return self._root_regressor

    def recover(self, request: BodyRecoveryRequest) -> BodyParameters:
        count = len(request.source_frames)
        if count < 2 or request.keypoints.shape != (count, 17, 3) or request.boxes_xys.shape != (count, 3):
            raise ValueError("Body recovery requires at least two aligned pose/box samples")
        boxes = torch.from_numpy(request.boxes_xys)
        features = self.chain.feature_model.predict(ImageFeatureRequest(request.video_path, boxes, frame_indices=torch.from_numpy(request.source_frames)))
        result = self.chain.mesh_model.predict(GvhmrRequest(
            kp2d=torch.from_numpy(request.keypoints), bbx_xys=boxes, f_imgseq=features.features,
            width=request.size[0], height=request.size[1], static_cam=True,
            K_fullimg=torch.from_numpy(request.intrinsic.astype(np.float32)),
        ))
        params = result.smpl_params_incam
        return BodyParameters(**{name: params[name].detach().float().cpu().numpy() for name in ("body_pose", "global_orient", "betas", "transl")})

    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry:
        tensors = parameters.tensors()
        vertices = self.chain.vertex_reconstructor.reconstruct(tensors)
        joints = self.coco.reconstruct(tensors)
        return BodyGeometry(vertices.numpy(), joints.numpy())

    def unload(self) -> None:
        self.chain.vertex_reconstructor.unload()
        self.coco.unload()
        self.chain.feature_model.unload()
        self.chain.mesh_model.unload()
