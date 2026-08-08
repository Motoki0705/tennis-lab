"""CUDA-resident articulated PLCS avatars for bounded scene composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from src.synthetic_data_generation.composition.contracts import (
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinates,
    GaussianForegroundComposition,
)
from src.synthetic_data_generation.composition.gaussians import (
    GaussianTensorSet,
    _quaternion_multiply,
    _rotation_matrix_to_quaternion,
)
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


def compose_prevalidated_frame_gaussians(
    composition: GaussianForegroundComposition,
    *,
    frame_index: int,
    object_tensors: Mapping[str, GaussianTensorSet],
) -> GaussianTensorSet:
    """Fuse validated PLCS placement and concatenation into one tensor contract.

    The foreground composition validates stable assets, identities, transforms,
    and the complete timeline once. Prepared-avatar batches validate their
    deformation outputs once. This path preserves those checks while avoiding
    the several short-lived, fully rescanned ``GaussianTensorSet`` objects that
    the generic composition helper creates for every object on every frame.
    """
    frame = composition.frame(frame_index)
    expected_object_ids = tuple(instance.object_id for instance in frame.instances)
    if set(object_tensors) != set(expected_object_ids):
        raise ValueError(
            f"Foreground frame {frame_index} tensor objects differ from its plan."
        )
    if not frame.instances:
        raise ValueError(
            f"Foreground frame {frame_index} has no visible object candidates."
        )

    means: list[Tensor] = []
    quaternions: list[Tensor] = []
    log_scales: list[Tensor] = []
    opacity_logits: list[Tensor] = []
    features: list[Tensor] = []
    instance_ids: list[Tensor] = []
    reference: GaussianTensorSet | None = None
    for instance in frame.instances:
        scene_object = composition.scene_object(instance.object_id)
        asset = composition.asset(scene_object.asset_id)
        local = object_tensors[instance.object_id]
        _validate_prepared_local_tensors(local, asset=asset)
        if bool(torch.count_nonzero(local.instance_ids)):
            raise ValueError(
                "PLCS prepared avatar tensors must retain canonical instance ID zero."
            )
        if reference is None:
            reference = local
        elif (
            local.means.dtype != reference.means.dtype
            or local.means.device != reference.means.device
            or local.feature_dim != reference.feature_dim
            or local.appearance_model != reference.appearance_model
            or local.appearance_space != reference.appearance_space
        ):
            raise ValueError("PLCS prepared avatar tensor contracts are incompatible.")

        transform = instance.scene_from_asset
        rotation = torch.as_tensor(
            transform.rotation,
            dtype=local.means.dtype,
            device=local.means.device,
        ).reshape(3, 3)
        translation = torch.as_tensor(
            transform.translation,
            dtype=local.means.dtype,
            device=local.means.device,
        )
        scale = local.means.new_tensor(transform.scale)
        transform_quaternion = _rotation_matrix_to_quaternion(rotation)
        means.append(scale * (local.means @ rotation.T) + translation)
        quaternions.append(
            torch.nn.functional.normalize(
                _quaternion_multiply(
                    transform_quaternion.expand_as(local.quaternions_wxyz),
                    local.quaternions_wxyz,
                ),
                dim=-1,
            )
        )
        log_scales.append(local.log_scales + torch.log(scale))
        opacity_logits.append(local.opacity_logits)
        features.append(local.features)
        instance_ids.append(
            torch.full_like(local.instance_ids, scene_object.instance_id)
        )

    assert reference is not None
    return GaussianTensorSet(
        means=torch.cat(means, dim=0),
        quaternions_wxyz=torch.cat(quaternions, dim=0),
        log_scales=torch.cat(log_scales, dim=0),
        opacity_logits=torch.cat(opacity_logits, dim=0),
        features=torch.cat(features, dim=0),
        instance_ids=torch.cat(instance_ids, dim=0),
        coordinates=GaussianCoordinates.scene(),
        appearance_model=reference.appearance_model,
        appearance_space=reference.appearance_space,
    )


def _validate_prepared_local_tensors(
    tensors: GaussianTensorSet,
    *,
    asset: GaussianAsset,
) -> None:
    if (
        tensors.gaussian_count != asset.gaussian_count
        or tensors.feature_dim != asset.feature_dim
        or tensors.floating_dtype != asset.floating_dtype
        or tensors.coordinates != asset.coordinates
        or tensors.appearance_model != asset.appearance_model
        or tensors.appearance_space != asset.appearance_space
    ):
        raise ValueError(
            f"Prepared PLCS tensors disagree with asset {asset.asset_id!r}."
        )


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


__all__ = [
    "AvatarAppearance",
    "PreparedAvatar",
    "compose_prevalidated_frame_gaussians",
    "prepare_avatar",
]
