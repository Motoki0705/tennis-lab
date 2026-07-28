"""Validated tensor operations for native all-Gaussian scene composition."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from src.synthetic_data_generation.scene_contract import SimilarityTransform

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROTATION_ATOL = 1.0e-5


@dataclass(frozen=True)
class GaussianTensorSet:
    """Raw NHT Gaussian parameters plus exact appearance and instance identity."""

    means: Tensor
    quats: Tensor
    log_scales: Tensor
    opacity_logits: Tensor
    features: Tensor
    instance_ids: Tensor
    appearance_space_sha256: str

    def __post_init__(self) -> None:
        count = self.means.shape[0] if self.means.ndim == 2 else -1
        expected_shapes = {
            "means": (count, 3),
            "quats": (count, 4),
            "log_scales": (count, 3),
            "opacity_logits": (count,),
            "instance_ids": (count,),
        }
        tensors = {
            "means": self.means,
            "quats": self.quats,
            "log_scales": self.log_scales,
            "opacity_logits": self.opacity_logits,
            "instance_ids": self.instance_ids,
        }
        if count <= 0:
            raise ValueError("Gaussian tensor sets must contain at least one Gaussian.")
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shapes[name]}."
                )
        if self.features.ndim != 2 or self.features.shape[0] != count:
            raise ValueError("features must have shape (gaussian_count, feature_dim).")
        if self.features.shape[1] <= 0:
            raise ValueError("features must have a positive feature dimension.")

        floating = (
            self.means,
            self.quats,
            self.log_scales,
            self.opacity_logits,
            self.features,
        )
        if not all(tensor.is_floating_point() for tensor in floating):
            raise TypeError("Gaussian parameters and features must be floating point.")
        dtype = self.means.dtype
        device = self.means.device
        if any(tensor.dtype != dtype for tensor in floating):
            raise TypeError("All floating Gaussian tensors must have the same dtype.")
        if any(tensor.device != device for tensor in floating):
            raise ValueError("All floating Gaussian tensors must share one device.")
        if self.instance_ids.dtype != torch.int64:
            raise TypeError("instance_ids must use torch.int64.")
        if self.instance_ids.device != device:
            raise ValueError("instance_ids must be on the Gaussian tensor device.")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in floating):
            raise ValueError("Gaussian tensors must contain only finite values.")
        if bool((torch.linalg.vector_norm(self.quats, dim=-1) <= 1.0e-12).any()):
            raise ValueError("Gaussian quaternions must have non-zero norm.")
        if bool((self.instance_ids < 0).any()):
            raise ValueError("instance_ids must be non-negative.")
        digest = self.appearance_space_sha256.lower()
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("appearance_space_sha256 must be a full SHA-256 digest.")
        object.__setattr__(self, "appearance_space_sha256", digest)

    @property
    def gaussian_count(self) -> int:
        """Return the number of Gaussian primitives."""
        return int(self.means.shape[0])

    @property
    def feature_dim(self) -> int:
        """Return the raw NHT feature dimension."""
        return int(self.features.shape[1])


def transform_gaussians(
    gaussians: GaussianTensorSet,
    scene_from_asset: SimilarityTransform,
) -> GaussianTensorSet:
    """Apply ``x_scene = scale * R * x_asset + t`` to anisotropic Gaussians."""
    dtype = gaussians.means.dtype
    device = gaussians.means.device
    rotation = torch.tensor(
        scene_from_asset.rotation,
        dtype=dtype,
        device=device,
    ).reshape(3, 3)
    translation = torch.tensor(
        scene_from_asset.translation,
        dtype=dtype,
        device=device,
    )
    scale = gaussians.means.new_tensor(scene_from_asset.scale)
    _validate_rotation(rotation)

    transform_quaternion = _rotation_matrix_to_quaternion(rotation)
    normalized_quaternions = F.normalize(gaussians.quats, dim=-1)
    rotated_quaternions = F.normalize(
        _quaternion_multiply(
            transform_quaternion.expand_as(normalized_quaternions),
            normalized_quaternions,
        ),
        dim=-1,
    )
    return GaussianTensorSet(
        means=scale * (gaussians.means @ rotation.T) + translation,
        quats=rotated_quaternions,
        log_scales=gaussians.log_scales + torch.log(scale),
        opacity_logits=gaussians.opacity_logits.clone(),
        features=gaussians.features.clone(),
        instance_ids=gaussians.instance_ids.clone(),
        appearance_space_sha256=gaussians.appearance_space_sha256,
    )


def compose_gaussians(
    background: GaussianTensorSet,
    instances: Sequence[GaussianTensorSet],
) -> GaussianTensorSet:
    """Concatenate one background and movable instances before rasterization."""
    instance_tuple = tuple(instances)
    if not instance_tuple:
        raise ValueError("At least one movable Gaussian instance is required.")
    if not bool((background.instance_ids == 0).all()):
        raise ValueError("Background Gaussians must use reserved instance_id 0.")

    seen_instance_ids: set[int] = set()
    for index, instance in enumerate(instance_tuple):
        _validate_compatible(background, instance, index=index)
        ids = {int(value) for value in torch.unique(instance.instance_ids).cpu()}
        if len(ids) != 1 or next(iter(ids)) <= 0:
            raise ValueError(
                "Each movable tensor set must contain exactly one positive instance_id."
            )
        overlap = seen_instance_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Duplicate movable instance ids: {sorted(overlap)}.")
        seen_instance_ids.update(ids)

    parts = (background, *instance_tuple)
    return GaussianTensorSet(
        means=torch.cat([part.means for part in parts], dim=0),
        quats=torch.cat([part.quats for part in parts], dim=0),
        log_scales=torch.cat([part.log_scales for part in parts], dim=0),
        opacity_logits=torch.cat(
            [part.opacity_logits for part in parts],
            dim=0,
        ),
        features=torch.cat([part.features for part in parts], dim=0),
        instance_ids=torch.cat([part.instance_ids for part in parts], dim=0),
        appearance_space_sha256=background.appearance_space_sha256,
    )


def _validate_compatible(
    background: GaussianTensorSet,
    instance: GaussianTensorSet,
    *,
    index: int,
) -> None:
    if instance.appearance_space_sha256 != background.appearance_space_sha256:
        raise ValueError(
            f"Instance {index} uses a different NHT appearance space. "
            "Independently trained deferred features cannot be concatenated."
        )
    if instance.feature_dim != background.feature_dim:
        raise ValueError(f"Instance {index} has a different feature dimension.")
    if instance.means.dtype != background.means.dtype:
        raise TypeError(f"Instance {index} has a different floating dtype.")
    if instance.means.device != background.means.device:
        raise ValueError(f"Instance {index} is on a different device.")


def _validate_rotation(rotation: Tensor) -> None:
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    error = torch.max(torch.abs(rotation @ rotation.T - identity))
    determinant = torch.linalg.det(rotation)
    if float(error) > _ROTATION_ATOL or not torch.isclose(
        determinant,
        determinant.new_tensor(1.0),
        atol=_ROTATION_ATOL,
        rtol=0.0,
    ):
        raise ValueError("scene_from_asset rotation must be a proper rotation.")


def _rotation_matrix_to_quaternion(rotation: Tensor) -> Tensor:
    values = torch.stack(
        (
            rotation.trace(),
            rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
            rotation[1, 1] - rotation[0, 0] - rotation[2, 2],
            rotation[2, 2] - rotation[0, 0] - rotation[1, 1],
        )
    )
    largest_index = int(torch.argmax(values))
    largest = torch.sqrt(values[largest_index] + 1.0) * 0.5
    multiplier = 0.25 / largest
    if largest_index == 0:
        quaternion = torch.stack(
            (
                largest,
                (rotation[2, 1] - rotation[1, 2]) * multiplier,
                (rotation[0, 2] - rotation[2, 0]) * multiplier,
                (rotation[1, 0] - rotation[0, 1]) * multiplier,
            )
        )
    elif largest_index == 1:
        quaternion = torch.stack(
            (
                (rotation[2, 1] - rotation[1, 2]) * multiplier,
                largest,
                (rotation[1, 0] + rotation[0, 1]) * multiplier,
                (rotation[0, 2] + rotation[2, 0]) * multiplier,
            )
        )
    elif largest_index == 2:
        quaternion = torch.stack(
            (
                (rotation[0, 2] - rotation[2, 0]) * multiplier,
                (rotation[1, 0] + rotation[0, 1]) * multiplier,
                largest,
                (rotation[2, 1] + rotation[1, 2]) * multiplier,
            )
        )
    else:
        quaternion = torch.stack(
            (
                (rotation[1, 0] - rotation[0, 1]) * multiplier,
                (rotation[0, 2] + rotation[2, 0]) * multiplier,
                (rotation[2, 1] + rotation[1, 2]) * multiplier,
                largest,
            )
        )
    quaternion = F.normalize(quaternion, dim=0)
    return torch.where(quaternion[0] < 0, -quaternion, quaternion)


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    left_w, left_x, left_y, left_z = left.unbind(dim=-1)
    right_w, right_x, right_y, right_z = right.unbind(dim=-1)
    return torch.stack(
        (
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ),
        dim=-1,
    )
