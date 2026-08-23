"""Explicit production-device boundary for the PLCS stage handler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import torch

from src.synthetic_data_generation.composition.contracts import GaussianAsset
from src.synthetic_data_generation.composition.gaussians import GaussianTensorSet
from src.synthetic_data_generation.dataset.plcs.articulation import (
    MotionArticulationReport,
)
from src.synthetic_data_generation.dataset.plcs.components.avatar_asset import (
    AvatarGaussianAsset,
)
from src.synthetic_data_generation.dataset.plcs.composition import (
    AvatarAppearance,
    PLCSAvatarFrameTensors,
    prepare_avatar,
)
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCSSourceSupportPlane,
)
from src.synthetic_data_generation.dataset.plcs.rendering.contracts import (
    PLCSForegroundCompositor,
)
from src.synthetic_data_generation.dataset.plcs.smplh import (
    SMPLHDeviceClip,
    SMPLHDeviceModel,
    SMPLHModelData,
    initial_smplh_surface_min_z,
    load_smplh_model,
    upload_motion_clip,
    upload_smplh_model,
)
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    ForegroundDelta,
)
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.plcs.generate_dataset.sampling.motion_source import PLCSMotionClip


class PLCSPreparedAvatar(Protocol):
    """Device-agnostic avatar surface consumed by the real stage loop."""

    clip: PLCSMotionClip
    surface_asset: AvatarGaussianAsset
    semantic_asset: GaussianAsset
    articulation: MotionArticulationReport

    def frame_tensors_batch(
        self,
        source_frame_indices: tuple[int, ...],
    ) -> dict[int, PLCSAvatarFrameTensors]:
        """Evaluate the requested intact source frames."""


class PLCSExecutionBackend(Protocol):
    """Constructor-injected numerical/device operations used by the handler.

    Canonical application composition uses :class:`CUDAPLCSExecutionBackend`.
    Tests may explicitly inject an independently implemented backend whose
    ``execution_device`` is exactly ``"test-cpu-oracle"``. There is no runtime
    selection or fallback from CUDA to that oracle.
    """

    @property
    def execution_device(self) -> str:
        """Return the exact device identifier persisted in diagnostics."""

    @property
    def torch_device(self) -> torch.device:
        """Return the tensor device used by explicit appearance inputs."""

    @property
    def cuda_peak_bytes(self) -> int:
        """Return the measured CUDA allocation, or zero for a test oracle."""

    @property
    def background_upload_count(self) -> int:
        """Return exact prepared background count for this attempt."""

    def reset_stage(
        self,
        *,
        configured_device: str,
        compositor: PLCSForegroundCompositor,
    ) -> None:
        """Initialize one explicit stage attempt without fallback."""

    def prepare_source(
        self,
        *,
        clip: PLCSMotionClip,
        model: object,
    ) -> None:
        """Retain one complete motion/model source on the execution device."""

    def load_model(self, *, model_root: Path, gender: str) -> object:
        """Load one gender model through the explicit execution dependency."""

    def initial_support_plane(
        self,
        *,
        clip: PLCSMotionClip,
        model: object,
    ) -> PLCSSourceSupportPlane:
        """Evaluate explicit frame-zero full-surface support provenance."""

    def prepare_avatar(
        self,
        *,
        asset_id: str,
        clip: PLCSMotionClip,
        model: object,
        appearance: AvatarAppearance,
        gaussian_count: int,
        seed: int,
    ) -> PLCSPreparedAvatar:
        """Build one articulated avatar from the complete source clip."""

    def prepare_background(
        self,
        *,
        compositor: PLCSForegroundCompositor,
        background: BackgroundArrays,
    ) -> None:
        """Prepare one validated static camera background exactly once."""

    def compose_delta(
        self,
        *,
        compositor: PLCSForegroundCompositor,
        frame_index: int,
        camera: SceneCamera,
        gaussians_scene: GaussianTensorSet,
        expected_instance_ids: tuple[int, ...],
    ) -> tuple[ForegroundDelta, dict[int, int]]:
        """Compose one logical sparse foreground sample."""


@dataclass(slots=True)
class CUDAPLCSExecutionBackend:
    """Default production implementation with stage-scoped CUDA buffers."""

    _device: torch.device | None = field(default=None, init=False, repr=False)
    _device_models: dict[str, SMPLHDeviceModel] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _device_clips: dict[str, SMPLHDeviceClip] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _background_upload_count: int = field(default=0, init=False, repr=False)

    @property
    def execution_device(self) -> str:
        """Return the initialized CUDA identifier."""
        return str(self.torch_device)

    @property
    def torch_device(self) -> torch.device:
        """Return the initialized CUDA tensor device."""
        if self._device is None:
            raise RuntimeError("PLCS CUDA backend has not been initialized.")
        return self._device

    @property
    def cuda_peak_bytes(self) -> int:
        """Return current stage CUDA peak allocation."""
        return int(torch.cuda.max_memory_allocated(self.torch_device))

    @property
    def background_upload_count(self) -> int:
        """Return exact prepared background count."""
        return self._background_upload_count

    def reset_stage(
        self,
        *,
        configured_device: str,
        compositor: PLCSForegroundCompositor,
    ) -> None:
        """Require CUDA and clear all buffers for one attempt."""
        device = torch.device(configured_device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PLCS production execution requires available CUDA.")
        self._device = device
        self._device_models.clear()
        self._device_clips.clear()
        self._background_upload_count = 0
        compositor.reset_stage()
        # Peak-stat reset rejects a CUDA device whose allocator has not yet been
        # initialized, which is the normal ordering for a fresh PLCS-only run.
        initialization_probe = torch.empty(
            1,
            dtype=torch.uint8,
            device=device,
        )
        del initialization_probe
        torch.cuda.reset_peak_memory_stats(device)

    def load_model(self, *, model_root: Path, gender: str) -> SMPLHModelData:
        """Load the strict licensed SMPL-H model for one gender."""
        return load_smplh_model(model_root, gender=gender)

    def prepare_source(
        self,
        *,
        clip: PLCSMotionClip,
        model: object,
    ) -> None:
        """Upload one gender model and one complete clip exactly once."""
        model_data = _model_data(model)
        if clip.gender not in self._device_models:
            self._device_models[clip.gender] = upload_smplh_model(
                model_data,
                device=self.torch_device,
            )
        if clip.source_path not in self._device_clips:
            self._device_clips[clip.source_path] = upload_motion_clip(
                clip,
                model_data,
                device=self.torch_device,
            )

    def initial_support_plane(
        self,
        *,
        clip: PLCSMotionClip,
        model: object,
    ) -> PLCSSourceSupportPlane:
        """Evaluate frame-zero full SMPL-H surface support on CUDA."""
        model_data = _model_data(model)
        if model_data.gender != clip.gender:
            raise ValueError("PLCS support clip/model gender is inconsistent.")
        try:
            device_model = self._device_models[clip.gender]
            device_clip = self._device_clips[clip.source_path]
        except KeyError as error:
            raise RuntimeError(
                "PLCS support evaluation requires retained source buffers."
            ) from error
        local_min_z = initial_smplh_surface_min_z(device_model, device_clip)
        return PLCSSourceSupportPlane.from_surface_minimum(
            initial_root_translation_z_m=float(clip.root_translation_m[0, 2]),
            support_local_z_m=local_min_z,
        )

    def prepare_avatar(
        self,
        *,
        asset_id: str,
        clip: PLCSMotionClip,
        model: object,
        appearance: AvatarAppearance,
        gaussian_count: int,
        seed: int,
    ) -> PLCSPreparedAvatar:
        """Prepare the production SMPL-H Gaussian avatar."""
        try:
            device_model = self._device_models[clip.gender]
            device_clip = self._device_clips[clip.source_path]
        except KeyError as error:
            raise RuntimeError(
                "PLCS source was not retained by production preflight."
            ) from error
        return cast(
            PLCSPreparedAvatar,
            prepare_avatar(
                asset_id=asset_id,
                clip=clip,
                model=_model_data(model),
                device_model=device_model,
                device_clip=device_clip,
                appearance=appearance,
                gaussian_count=gaussian_count,
                seed=seed,
            ),
        )

    def prepare_background(
        self,
        *,
        compositor: PLCSForegroundCompositor,
        background: BackgroundArrays,
    ) -> None:
        """Upload one static background through the CUDA compositor."""
        compositor.prepare_background(background, device=self.torch_device)
        self._background_upload_count += 1

    def compose_delta(
        self,
        *,
        compositor: PLCSForegroundCompositor,
        frame_index: int,
        camera: SceneCamera,
        gaussians_scene: GaussianTensorSet,
        expected_instance_ids: tuple[int, ...],
    ) -> tuple[ForegroundDelta, dict[int, int]]:
        """Delegate one sparse sample to the CUDA compositor."""
        result: tuple[ForegroundDelta, dict[int, int]] = compositor.compose_delta(
            frame_index=frame_index,
            camera=camera,
            gaussians_scene=gaussians_scene,
            expected_instance_ids=expected_instance_ids,
        )
        return result


def _model_data(value: object) -> SMPLHModelData:
    if not isinstance(value, SMPLHModelData):
        raise TypeError("PLCS production execution requires SMPLHModelData.")
    return value


__all__ = [
    "CUDAPLCSExecutionBackend",
    "PLCSExecutionBackend",
    "PLCSPreparedAvatar",
]
