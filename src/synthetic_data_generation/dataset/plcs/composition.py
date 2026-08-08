"""CUDA-resident articulated PLCS avatars for bounded scene composition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.synthetic_data_generation.composition.contracts import (
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinates,
)
from src.synthetic_data_generation.composition.gaussians import GaussianTensorSet
from src.synthetic_data_generation.dataset.plcs.articulation import (
    MotionArticulationReport,
    articulation_probe_indices,
    validate_articulated_motion,
)
from src.synthetic_data_generation.dataset.plcs.components.avatar_asset import (
    AvatarGaussianAsset,
)
from src.synthetic_data_generation.dataset.plcs.smplh import (
    SMPLHDeviceClip,
    SMPLHDeviceGaussianAsset,
    SMPLHDeviceModel,
    SMPLHModelData,
    build_smplh_surface_asset,
    skin_gaussian_batch,
    upload_gaussian_asset,
)
from src.synthetic_data_generation.rendering.foreground import (
    RGB_APPEARANCE_MODEL,
    RGB_APPEARANCE_SPACE,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import PLCSMotionClip


@dataclass(frozen=True, slots=True)
class AvatarAppearance:
    """Renderer-compatible per-Gaussian RGB from an explicit source."""

    features: Tensor
    appearance_model: str
    appearance_space: str

    def __post_init__(self) -> None:
        if not isinstance(self.features, Tensor) or self.features.ndim != 2:
            raise ValueError("Avatar appearance features must have shape [N,F].")
        if self.features.shape[0] <= 0 or self.features.shape[1] != 3:
            raise ValueError("Avatar appearance features must have shape [N,3].")
        if self.features.dtype not in {torch.float32, torch.float64}:
            raise TypeError("Avatar appearance features must use float32 or float64.")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("Avatar appearance features contain NaN or infinity.")
        if bool((self.features < 0.0).any()) or bool((self.features > 1.0).any()):
            raise ValueError("Avatar RGB appearance must lie in the closed unit range.")
        if self.appearance_model != RGB_APPEARANCE_MODEL:
            raise ValueError(
                f"Avatar appearance_model must be {RGB_APPEARANCE_MODEL!r}."
            )
        if self.appearance_space != RGB_APPEARANCE_SPACE:
            raise ValueError(
                f"Avatar appearance_space must be {RGB_APPEARANCE_SPACE!r}."
            )


@dataclass(frozen=True, slots=True)
class PreparedAvatar:
    """Stage-scoped device buffers for one complete source clip."""

    clip: PLCSMotionClip
    surface_asset: AvatarGaussianAsset
    device_model: SMPLHDeviceModel
    device_clip: SMPLHDeviceClip
    device_asset: SMPLHDeviceGaussianAsset
    appearance: AvatarAppearance
    semantic_asset: GaussianAsset
    articulation: MotionArticulationReport

    def __post_init__(self) -> None:
        count = self.surface_asset.gaussian_count
        if self.device_clip.frame_count != self.clip.frame_count:
            raise ValueError("Prepared avatar device clip does not preserve source T.")
        if self.device_asset.gaussian_count != count:
            raise ValueError(
                "Prepared avatar device shell differs from its surface asset."
            )
        if self.appearance.features.shape != (count, 3):
            raise ValueError("Avatar appearance and surface Gaussian counts differ.")
        if self.appearance.features.device != self.device_model.device:
            raise ValueError("Avatar appearance must reside on the stage CUDA device.")
        if self.appearance.features.dtype != torch.float32:
            raise TypeError("Production avatar appearance must use float32.")
        if self.semantic_asset.gaussian_count != count:
            raise ValueError(
                "Semantic avatar asset and surface Gaussian counts differ."
            )
        if self.articulation.frame_count != self.clip.frame_count:
            raise ValueError("Articulation report does not cover the full clip.")

    def frame_tensors_batch(
        self,
        source_frame_indices: tuple[int, ...],
    ) -> dict[int, GaussianTensorSet]:
        """Deform one bounded batch and retain it only for its current chunk."""
        batch = skin_gaussian_batch(
            self.device_model,
            self.device_clip,
            self.device_asset,
            source_frame_indices=source_frame_indices,
        )
        return {
            frame_index: GaussianTensorSet(
                means=batch.means_m[index],
                quaternions_wxyz=batch.quaternions_wxyz[index],
                log_scales=batch.log_scales_m[index],
                opacity_logits=self.device_asset.opacity_logits,
                features=self.appearance.features,
                instance_ids=torch.zeros(
                    self.device_asset.gaussian_count,
                    dtype=torch.int64,
                    device=self.device_model.device,
                ),
                coordinates=GaussianCoordinates.asset_local_metres(),
                appearance_model=self.appearance.appearance_model,
                appearance_space=self.appearance.appearance_space,
            )
            for index, frame_index in enumerate(source_frame_indices)
        }


def prepare_avatar(
    *,
    asset_id: str,
    clip: PLCSMotionClip,
    model: SMPLHModelData,
    device_model: SMPLHDeviceModel,
    device_clip: SMPLHDeviceClip,
    appearance: AvatarAppearance,
    gaussian_count: int,
    seed: int,
) -> PreparedAvatar:
    """Prepare one avatar without retaining full-frame geometry on CPU or CUDA."""
    if device_model.device.type != "cuda" or appearance.features.device.type != "cuda":
        raise ValueError("PLCS production avatar preparation requires CUDA buffers.")
    if device_model.gender != model.gender or device_model.gender != clip.gender:
        raise ValueError("Prepared avatar host/device model gender is inconsistent.")
    surface_asset = build_smplh_surface_asset(
        model,
        clip,
        gaussian_count=gaussian_count,
        seed=seed,
    )
    if appearance.features.shape[0] != surface_asset.gaussian_count:
        raise ValueError(
            "Explicit avatar appearance count must equal the configured surface count."
        )
    device_asset = upload_gaussian_asset(surface_asset, device=device_model.device)
    probe_indices = articulation_probe_indices(clip)
    probes = skin_gaussian_batch(
        device_model,
        device_clip,
        device_asset,
        source_frame_indices=probe_indices,
    )
    articulation = validate_articulated_motion(clip, device_clip, probes)
    semantic_asset = GaussianAsset(
        asset_id=asset_id,
        asset_class="smplh-player",
        role=GaussianAssetRole.MOVABLE,
        coordinates=GaussianCoordinates.asset_local_metres(),
        gaussian_count=surface_asset.gaussian_count,
        feature_dim=3,
        floating_dtype="float32",
        appearance_model=appearance.appearance_model,
        appearance_space=appearance.appearance_space,
    )
    return PreparedAvatar(
        clip=clip,
        surface_asset=surface_asset,
        device_model=device_model,
        device_clip=device_clip,
        device_asset=device_asset,
        appearance=appearance,
        semantic_asset=semantic_asset,
        articulation=articulation,
    )


__all__ = ["AvatarAppearance", "PreparedAvatar", "prepare_avatar"]
