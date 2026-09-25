"""Place stored GVHMR parameters against triangulated joints without image inference."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import SmplCoco17Reconstructor, SmplVertexReconstructor
from src.tennis_scene.motion_alignment.temporal import TemporalPlacementConfig
from src.tennis_scene.pipeline.body_types import BodyGeometry, BodyParameters
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.gvhmr import GVHMROutput
from src.tennis_scene.pipeline.components.player_reconstruction import (
    ReconstructedPlayers,
    reconstruct_player_bodies,
)
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationOutput
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)


class SmplGeometry:
    """Deterministic body-model conversion, independent of HMR/GVHMR inference."""

    def __init__(self, config: PeopleModelConfig) -> None:
        self.config = config
        self.vertices = SmplVertexReconstructor(config.body_models_dir, device=config.runtime.device, bundled_assets=config.bundled_assets)
        self.coco = SmplCoco17Reconstructor(config.body_models_dir, device=config.runtime.device, bundled_assets=config.bundled_assets)
        self._root: NDArray[np.float64] | None = None

    @property
    def root_regressor(self) -> NDArray[np.float64]:
        if self._root is None:
            value = torch.load(self.config.bundled_assets.smpl_neutral_joint_regressor, map_location="cpu", weights_only=True)
            if not isinstance(value, torch.Tensor) or value.ndim not in (1, 2) or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid SMPL root regressor")
            self._root = value.double().numpy()
        return self._root

    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry:
        tensors = parameters.tensors()
        return BodyGeometry(self.vertices.reconstruct(tensors).numpy(), self.coco.reconstruct(tensors).numpy())

    def unload(self) -> None:
        self.vertices.unload()
        self.coco.unload()


@dataclass(frozen=True)
class BodyPlacementInput:
    source: ClipSource
    alignment: CameraAlignmentOutput
    raw: ObjectObservations
    grouped: GroupedObservations
    skeleton: PlayerTriangulationOutput
    recovered: GVHMROutput


@dataclass(frozen=True)
class BodyPlacementOutput:
    players: ReconstructedPlayers | None


class BodyPlacementModule:
    def __init__(self, camera_ids: tuple[str, ...], models: PeopleModelConfig, config: TemporalPlacementConfig,
                 *, reprojection_px: float, enabled: bool = True) -> None:
        self.models, self.config, self.reprojection_px, self.enabled = models, config, reprojection_px, enabled
        self.io = ComponentIO("body_placement", BodyPlacementInput, BodyPlacementOutput,
            {"alignment": InputPort("aligned_cameras"), "calibration": InputPort("local_court_calibration"),
             "identities": InputPort("person_identities"), "skeleton": InputPort("player_skeletons"),
             "recovered": InputPort("body_parameters"), **{f"pose_{c}": InputPort("person_poses") for c in camera_ids}}, "placed_bodies")

    def process(self, inputs: BodyPlacementInput) -> BodyPlacementOutput:
        geometry, skeleton = inputs.alignment.geometry, inputs.skeleton.skeleton
        if geometry is None or skeleton is None:
            return BodyPlacementOutput(None)
        scale = float(np.hypot(*inputs.source.size) / np.hypot(1920, 1080))
        config = replace(self.config, max_reprojection_rms_px=self.config.max_reprojection_rms_px * scale,
            reprojection_weight_sigma_px=self.config.reprojection_weight_sigma_px * scale)
        body = SmplGeometry(self.models) if self.enabled and inputs.recovered.bodies else None
        return BodyPlacementOutput(reconstruct_player_bodies(inputs.grouped, inputs.raw, skeleton,
            geometry.cameras, inputs.recovered, body=body, reprojection_px=self.reprojection_px * scale, placement_config=config))
