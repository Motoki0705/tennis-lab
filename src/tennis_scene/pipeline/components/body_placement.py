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
from src.tennis_scene.pipeline.components.identity import IDENTITIES_PORT
from src.tennis_scene.pipeline.components.player_reconstruction import (
    ReconstructedPlayers,
    reconstruct_player_bodies,
)
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationOutput
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.utils.inference_recording import model_event


class SmplGeometry:
    """Deterministic body-model conversion, independent of HMR/GVHMR inference."""

    def __init__(self, config: PeopleModelConfig) -> None:
        self.config = config
        self.vertices = SmplVertexReconstructor(config.body_models_dir, device=config.runtime.device, smpl_assets=config.smpl_assets)
        self.coco = SmplCoco17Reconstructor(config.body_models_dir, device=config.runtime.device, smpl_assets=config.smpl_assets)
        self._root: NDArray[np.float64] | None = None

    @property
    def root_regressor(self) -> NDArray[np.float64]:
        if self._root is None:
            with model_event("load", "SMPL-root-regressor", weights={"regressor": self.config.smpl_assets.smpl_neutral_joint_regressor},
                             device="cpu", dtype="float64", instance=f"{id(self)}:root"):
                value = torch.load(self.config.smpl_assets.smpl_neutral_joint_regressor, map_location="cpu", weights_only=True)
            if not isinstance(value, torch.Tensor) or value.ndim not in (1, 2) or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid SMPL root regressor")
            self._root = value.double().numpy()
        return self._root

    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry:
        tensors = parameters.tensors()
        return BodyGeometry(self.vertices.reconstruct(tensors).numpy(), self.coco.reconstruct(tensors).numpy())

    def load(self) -> None:
        self.vertices.load()
        self.coco.load()
        _ = self.root_regressor

    def unload(self) -> None:
        self.vertices.unload()
        self.coco.unload()
        self._root = None


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


class BodyPlacementModule(InferenceComponent[BodyPlacementInput, BodyPlacementOutput]):
    def __init__(self, camera_ids: tuple[str, ...], models: PeopleModelConfig, config: TemporalPlacementConfig,
                 *, reprojection_px: float, enabled: bool = True) -> None:
        self.models, self.config, self.reprojection_px, self.enabled = models, config, reprojection_px, enabled
        self.io = ComponentIO("body_placement", BodyPlacementInput, BodyPlacementOutput,
            {"alignment": InputPort("aligned_cameras"), "calibration": InputPort("local_court_calibration"),
             "identities": IDENTITIES_PORT, "skeleton": InputPort("player_skeletons"),
             "recovered": InputPort("body_parameters", version=2), **{f"pose_{c}": InputPort("person_poses", 2) for c in camera_ids}}, "placed_bodies")

    def model_specs(self) -> dict[str, ModelSpec[SmplGeometry]]:
        weights = {k: v for k, v in self.models.body_assets().items() if k not in {"hmr2", "gvhmr"}}
        return {} if not self.enabled else {"body": ModelSpec("body_geometry", weights, "smplx_supermotion_smpl_coco17",
            self.models.runtime.device, "float32", {}, lambda: SmplGeometry(self.models),
            initialize=lambda body: body.load(), release=lambda body: body.unload())}

    def process_with_models(self, inputs: BodyPlacementInput, models: ModelBorrower) -> BodyPlacementOutput:
        geometry, skeleton = inputs.alignment.geometry, inputs.skeleton.skeleton
        if geometry is None or skeleton is None:
            return BodyPlacementOutput(None)
        scale = inputs.source.pixel_threshold_scale
        config = replace(self.config, max_reprojection_rms_px=self.config.max_reprojection_rms_px * scale,
            reprojection_weight_sigma_px=self.config.reprojection_weight_sigma_px * scale)
        body = models.borrow("body") if self.enabled and inputs.recovered.bodies else None
        return BodyPlacementOutput(reconstruct_player_bodies(inputs.grouped, inputs.raw, skeleton,
            geometry.cameras, inputs.recovered, body=body, reprojection_px=self.reprojection_px * scale, placement_config=config))
