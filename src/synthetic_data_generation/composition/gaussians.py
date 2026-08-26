"""Validated tensor operations for semantic all-Gaussian composition."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from src.synthetic_data_generation.composition.contracts import (
    GaussianAsset,
    GaussianCoordinates,
    GaussianDeformationKind,
    GaussianForegroundComposition,
    GaussianSceneComposition,
    GaussianTransform,
)

_ROTATION_ATOL = 1.0e-5
_QUATERNION_ATOL = 1.0e-5
_SUPPORTED_DTYPES = {torch.float32: "float32", torch.float64: "float64"}


@dataclass(frozen=True, slots=True)
class GaussianTensorSet:
    """Raw Gaussian parameters with explicit coordinates and appearance semantics."""

    means: Tensor
    quaternions_wxyz: Tensor
    log_scales: Tensor
    opacity_logits: Tensor
    features: Tensor
    instance_ids: Tensor
    coordinates: GaussianCoordinates
    appearance_model: str
    appearance_space: str

    def __post_init__(self) -> None:
        for name, tensor in (
            ("means", self.means),
            ("quaternions_wxyz", self.quaternions_wxyz),
            ("log_scales", self.log_scales),
            ("opacity_logits", self.opacity_logits),
            ("features", self.features),
            ("instance_ids", self.instance_ids),
        ):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
        if not isinstance(self.coordinates, GaussianCoordinates):
            raise TypeError("coordinates must be GaussianCoordinates.")
        count = self.means.shape[0] if self.means.ndim == 2 else -1
        expected_shapes = {
            "means": (count, 3),
            "quaternions_wxyz": (count, 4),
            "log_scales": (count, 3),
            "opacity_logits": (count,),
            "instance_ids": (count,),
        }
        tensors = {
            "means": self.means,
            "quaternions_wxyz": self.quaternions_wxyz,
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
            self.quaternions_wxyz,
            self.log_scales,
            self.opacity_logits,
            self.features,
        )
        if not all(tensor.is_floating_point() for tensor in floating):
            raise TypeError("Gaussian parameters and features must be floating point.")
        dtype = self.means.dtype
        device = self.means.device
        if dtype not in _SUPPORTED_DTYPES:
            raise TypeError("Gaussian floating tensors must use float32 or float64.")
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

        quaternion_norms = torch.linalg.vector_norm(self.quaternions_wxyz, dim=-1)
        if not bool(
            torch.allclose(
                quaternion_norms,
                torch.ones_like(quaternion_norms),
                atol=_QUATERNION_ATOL,
                rtol=0.0,
            )
        ):
            raise ValueError("Gaussian quaternions must be normalized in wxyz order.")
        scales = torch.exp(self.log_scales)
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0.0).any()):
            raise ValueError("Gaussian scales must be finite and strictly positive.")
        variances = torch.exp(2.0 * self.log_scales)
        if not bool(torch.isfinite(variances).all()) or bool((variances <= 0.0).any()):
            raise ValueError("Gaussian covariance variances must be finite and positive.")
        if bool((self.instance_ids < 0).any()):
            raise ValueError("instance_ids must be non-negative.")
        appearance_model = _trimmed_string(
            self.appearance_model,
            name="appearance_model",
        )
        appearance_space = _trimmed_string(
            self.appearance_space,
            name="appearance_space",
        )
        object.__setattr__(self, "appearance_model", appearance_model)
        object.__setattr__(self, "appearance_space", appearance_space)

    @property
    def gaussian_count(self) -> int:
        """Return the number of Gaussian primitives."""
        return int(self.means.shape[0])

    @property
    def feature_dim(self) -> int:
        """Return the renderer feature dimension."""
        return int(self.features.shape[1])

    @property
    def floating_dtype(self) -> str:
        """Return the canonical semantic dtype name."""
        return _SUPPORTED_DTYPES[self.means.dtype]


@dataclass(frozen=True, slots=True)
class GaussianDeformationReport:
    """Observable non-rigid local-deformation evidence for one object."""

    object_id: str
    frame_count: int
    deformed_frame_indices: tuple[int, ...]
    max_mean_residual: float
    max_covariance_residual: float


def gaussian_covariances(gaussians: GaussianTensorSet) -> Tensor:
    """Return finite positive-definite covariance matrices with shape ``[N,3,3]``."""
    rotations = _quaternions_to_rotation_matrices(gaussians.quaternions_wxyz)
    variances = torch.exp(2.0 * gaussians.log_scales)
    covariances = rotations @ torch.diag_embed(variances) @ rotations.transpose(-1, -2)
    if not bool(torch.isfinite(covariances).all()):
        raise ValueError("Gaussian covariance matrices must be finite.")
    eigenvalues = torch.linalg.eigvalsh(covariances)
    if bool((eigenvalues <= 0.0).any()):
        raise ValueError("Gaussian covariance matrices must be positive definite.")
    return covariances


def validate_asset_tensors(asset: GaussianAsset, tensors: GaussianTensorSet) -> None:
    """Validate tensor data against all declared semantic asset metadata."""
    _validate_asset_tensor_metadata(asset, tensors)
    if not bool((tensors.instance_ids == 0).all()):
        raise ValueError(
            f"Gaussian asset {asset.asset_id!r} tensor contract mismatch: "
            "canonical asset instance_ids must all be 0."
        )
    gaussian_covariances(tensors)


def validate_identified_asset_tensors(
    asset: GaussianAsset,
    tensors: GaussianTensorSet,
    *,
    instance_id: int,
) -> None:
    """Validate one movable asset already carrying its positive scene identity."""
    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        raise ValueError("instance_id must be a positive integer.")
    _validate_asset_tensor_metadata(asset, tensors)
    if not bool((tensors.instance_ids == instance_id).all()):
        raise ValueError(
            f"Gaussian asset {asset.asset_id!r} must carry only instance_id "
            f"{instance_id}."
        )
    gaussian_covariances(tensors)


def _validate_asset_tensor_metadata(
    asset: GaussianAsset,
    tensors: GaussianTensorSet,
) -> None:
    mismatches: list[str] = []
    if tensors.gaussian_count != asset.gaussian_count:
        mismatches.append(
            f"gaussian_count={tensors.gaussian_count} (declared {asset.gaussian_count})"
        )
    if tensors.feature_dim != asset.feature_dim:
        mismatches.append(
            f"feature_dim={tensors.feature_dim} (declared {asset.feature_dim})"
        )
    if tensors.floating_dtype != asset.floating_dtype:
        mismatches.append(
            f"floating_dtype={tensors.floating_dtype} (declared {asset.floating_dtype})"
        )
    if tensors.coordinates != asset.coordinates:
        mismatches.append("coordinate convention")
    if tensors.appearance_model != asset.appearance_model:
        mismatches.append("appearance model")
    if tensors.appearance_space != asset.appearance_space:
        mismatches.append("appearance space")
    if mismatches:
        raise ValueError(
            f"Gaussian asset {asset.asset_id!r} tensor contract mismatch: "
            + ", ".join(mismatches)
            + "."
        )


def assign_instance_id(
    gaussians: GaussianTensorSet,
    instance_id: int,
) -> GaussianTensorSet:
    """Assign one positive, stable object identity to canonical asset tensors."""
    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        raise ValueError("instance_id must be a positive integer.")
    if not bool((gaussians.instance_ids == 0).all()):
        raise ValueError("Only canonical instance_id 0 tensors may be instantiated.")
    return _replace_tensor_set(
        gaussians,
        instance_ids=torch.full_like(gaussians.instance_ids, instance_id),
    )


def transform_gaussians(
    gaussians: GaussianTensorSet,
    scene_from_asset: GaussianTransform,
) -> GaussianTensorSet:
    """Map metric asset-local means and covariance into canonical scene space."""
    if gaussians.coordinates != GaussianCoordinates.asset_local_metres():
        raise ValueError(
            "transform_gaussians requires right-handed asset-local metre coordinates."
        )
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
    rotated_quaternions = F.normalize(
        _quaternion_multiply(
            transform_quaternion.expand_as(gaussians.quaternions_wxyz),
            gaussians.quaternions_wxyz,
        ),
        dim=-1,
    )
    result = GaussianTensorSet(
        means=scale * (gaussians.means @ rotation.T) + translation,
        quaternions_wxyz=rotated_quaternions,
        log_scales=gaussians.log_scales + torch.log(scale),
        opacity_logits=gaussians.opacity_logits.clone(),
        features=gaussians.features.clone(),
        instance_ids=gaussians.instance_ids.clone(),
        coordinates=GaussianCoordinates.scene(),
        appearance_model=gaussians.appearance_model,
        appearance_space=gaussians.appearance_space,
    )
    gaussian_covariances(result)
    return result


def compose_gaussians(
    background: GaussianTensorSet,
    instances: Sequence[GaussianTensorSet],
) -> GaussianTensorSet:
    """Concatenate a scene background and zero or more unique movable objects."""
    instance_tuple = tuple(instances)
    if background.coordinates != GaussianCoordinates.scene():
        raise ValueError("Background Gaussians must use canonical scene coordinates.")
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
    result = GaussianTensorSet(
        means=torch.cat([part.means for part in parts], dim=0),
        quaternions_wxyz=torch.cat(
            [part.quaternions_wxyz for part in parts],
            dim=0,
        ),
        log_scales=torch.cat([part.log_scales for part in parts], dim=0),
        opacity_logits=torch.cat([part.opacity_logits for part in parts], dim=0),
        features=torch.cat([part.features for part in parts], dim=0),
        instance_ids=torch.cat([part.instance_ids for part in parts], dim=0),
        coordinates=GaussianCoordinates.scene(),
        appearance_model=background.appearance_model,
        appearance_space=background.appearance_space,
    )
    gaussian_covariances(result)
    return result


def compose_foreground_gaussians(
    instances: Sequence[GaussianTensorSet],
) -> GaussianTensorSet:
    """Concatenate only positive-identity scene-space movable Gaussians."""
    instance_tuple = tuple(instances)
    if not instance_tuple:
        raise ValueError("Foreground composition requires at least one movable instance.")
    reference = instance_tuple[0]
    seen_instance_ids: set[int] = set()
    for index, instance in enumerate(instance_tuple):
        if instance.coordinates != GaussianCoordinates.scene():
            raise ValueError(
                f"Foreground instance {index} is not in canonical scene coordinates."
            )
        if index > 0:
            _validate_compatible(reference, instance, index=index)
        ids = {int(value) for value in torch.unique(instance.instance_ids).cpu()}
        if len(ids) != 1 or next(iter(ids)) <= 0:
            raise ValueError(
                "Each foreground tensor set must contain exactly one positive instance_id."
            )
        overlap = seen_instance_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Duplicate foreground instance ids: {sorted(overlap)}.")
        seen_instance_ids.update(ids)

    result = GaussianTensorSet(
        means=torch.cat([part.means for part in instance_tuple], dim=0),
        quaternions_wxyz=torch.cat(
            [part.quaternions_wxyz for part in instance_tuple],
            dim=0,
        ),
        log_scales=torch.cat([part.log_scales for part in instance_tuple], dim=0),
        opacity_logits=torch.cat(
            [part.opacity_logits for part in instance_tuple],
            dim=0,
        ),
        features=torch.cat([part.features for part in instance_tuple], dim=0),
        instance_ids=torch.cat([part.instance_ids for part in instance_tuple], dim=0),
        coordinates=GaussianCoordinates.scene(),
        appearance_model=reference.appearance_model,
        appearance_space=reference.appearance_space,
    )
    gaussian_covariances(result)
    return result


def compose_frame_gaussians(
    composition: GaussianSceneComposition,
    *,
    frame_index: int,
    background_tensors: GaussianTensorSet,
    object_tensors: Mapping[str, GaussianTensorSet],
) -> GaussianTensorSet:
    """Validate and compose one exact frame from semantic object-local tensors."""
    validate_asset_tensors(composition.background, background_tensors)
    frame = composition.frame(frame_index)
    expected_object_ids = {instance.object_id for instance in frame.instances}
    actual_object_ids = set(object_tensors)
    if actual_object_ids != expected_object_ids:
        raise ValueError(
            f"Frame {frame_index} tensor objects differ; "
            f"missing={sorted(expected_object_ids - actual_object_ids)}, "
            f"unexpected={sorted(actual_object_ids - expected_object_ids)}."
        )

    transformed: list[GaussianTensorSet] = []
    for instance in frame.instances:
        scene_object = composition.scene_object(instance.object_id)
        asset = composition.asset(scene_object.asset_id)
        local = object_tensors[instance.object_id]
        validate_asset_tensors(asset, local)
        identified = assign_instance_id(local, scene_object.instance_id)
        transformed.append(transform_gaussians(identified, instance.scene_from_asset))
    return compose_gaussians(background_tensors, transformed)


def compose_foreground_frame_gaussians(
    composition: GaussianForegroundComposition,
    *,
    frame_index: int,
    object_tensors: Mapping[str, GaussianTensorSet],
) -> GaussianTensorSet:
    """Validate and place every positive-identity movable foreground object."""
    frame = composition.frame(frame_index)
    expected_object_ids = {instance.object_id for instance in frame.instances}
    actual_object_ids = set(object_tensors)
    if actual_object_ids != expected_object_ids:
        raise ValueError(
            f"Foreground frame {frame_index} tensor objects differ; "
            f"missing={sorted(expected_object_ids - actual_object_ids)}, "
            f"unexpected={sorted(actual_object_ids - expected_object_ids)}."
        )
    if not frame.instances:
        raise ValueError(
            f"Foreground frame {frame_index} has no visible object candidates."
        )

    transformed: list[GaussianTensorSet] = []
    for instance in frame.instances:
        scene_object = composition.scene_object(instance.object_id)
        asset = composition.asset(scene_object.asset_id)
        local = object_tensors[instance.object_id]
        validate_identified_asset_tensors(
            asset,
            local,
            instance_id=scene_object.instance_id,
        )
        transformed.append(transform_gaussians(local, instance.scene_from_asset))
    return compose_foreground_gaussians(transformed)


def validate_articulated_deformation(
    composition: GaussianSceneComposition | GaussianForegroundComposition,
    *,
    object_id: str,
    frame_tensors: Mapping[int, GaussianTensorSet],
    mean_atol: float = 1.0e-5,
    covariance_atol: float = 1.0e-5,
) -> GaussianDeformationReport:
    """Reject a declared articulated object that is only rigidly transformed.

    Tensors are compared in asset-local coordinates.  A best-fit proper rigid
    transform is removed from each frame before both labelled means and Gaussian
    covariance are compared with the first active frame.
    """
    scene_object = composition.scene_object(object_id)
    if scene_object.deformation_kind != GaussianDeformationKind.ARTICULATED:
        raise ValueError(f"Object {object_id!r} is not declared articulated.")
    expected_indices = composition.active_frame_indices(object_id)
    actual_indices = tuple(sorted(frame_tensors))
    if actual_indices != expected_indices:
        raise ValueError(
            f"Object {object_id!r} deformation frames differ; "
            f"expected {expected_indices}, got {actual_indices}."
        )
    if len(expected_indices) < 2:
        raise ValueError("Articulated deformation requires at least two frames.")
    mean_tolerance = _positive_tolerance(mean_atol, name="mean_atol")
    covariance_tolerance = _positive_tolerance(
        covariance_atol,
        name="covariance_atol",
    )

    asset = composition.asset(scene_object.asset_id)
    for frame_index in expected_indices:
        validate_asset_tensors(asset, frame_tensors[frame_index])
    reference = frame_tensors[expected_indices[0]]
    reference_covariances = gaussian_covariances(reference)
    deformed: list[int] = []
    max_mean_residual = 0.0
    max_covariance_residual = 0.0
    for frame_index in expected_indices[1:]:
        current = frame_tensors[frame_index]
        rotation, translation = _best_fit_rigid_transform(
            reference.means,
            current.means,
        )
        aligned_means = reference.means @ rotation.T + translation
        mean_residual = float(torch.max(torch.abs(aligned_means - current.means)))
        aligned_covariances = (
            rotation @ reference_covariances @ rotation.transpose(-1, -2)
        )
        covariance_residual = float(
            torch.max(
                torch.abs(aligned_covariances - gaussian_covariances(current))
            )
        )
        max_mean_residual = max(max_mean_residual, mean_residual)
        max_covariance_residual = max(
            max_covariance_residual,
            covariance_residual,
        )
        if mean_residual > mean_tolerance or covariance_residual > covariance_tolerance:
            deformed.append(frame_index)

    if not deformed:
        raise ValueError(
            f"Articulated object {object_id!r} is rigid-only across all frames."
        )
    return GaussianDeformationReport(
        object_id=object_id,
        frame_count=len(expected_indices),
        deformed_frame_indices=tuple(deformed),
        max_mean_residual=max_mean_residual,
        max_covariance_residual=max_covariance_residual,
    )


def _validate_compatible(
    background: GaussianTensorSet,
    instance: GaussianTensorSet,
    *,
    index: int,
) -> None:
    if instance.coordinates != GaussianCoordinates.scene():
        raise ValueError(f"Instance {index} is not in canonical scene coordinates.")
    if instance.appearance_model != background.appearance_model:
        raise ValueError(f"Instance {index} uses a different appearance model.")
    if instance.appearance_space != background.appearance_space:
        raise ValueError(f"Instance {index} uses a different appearance space.")
    if instance.feature_dim != background.feature_dim:
        raise ValueError(f"Instance {index} has a different feature dimension.")
    if instance.means.dtype != background.means.dtype:
        raise TypeError(f"Instance {index} has a different floating dtype.")
    if instance.means.device != background.means.device:
        raise ValueError(f"Instance {index} is on a different device.")


def _replace_tensor_set(
    gaussians: GaussianTensorSet,
    *,
    instance_ids: Tensor,
) -> GaussianTensorSet:
    return GaussianTensorSet(
        means=gaussians.means,
        quaternions_wxyz=gaussians.quaternions_wxyz,
        log_scales=gaussians.log_scales,
        opacity_logits=gaussians.opacity_logits,
        features=gaussians.features,
        instance_ids=instance_ids,
        coordinates=gaussians.coordinates,
        appearance_model=gaussians.appearance_model,
        appearance_space=gaussians.appearance_space,
    )


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


def _best_fit_rigid_transform(source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if source.shape != target.shape:
        raise ValueError("Deformation frames must preserve Gaussian count and ordering.")
    source_centroid = source.mean(dim=0)
    target_centroid = target.mean(dim=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    left, _, right_transpose = torch.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if float(torch.linalg.det(rotation)) < 0.0:
        right_transpose = right_transpose.clone()
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    _validate_rotation(rotation)
    translation = target_centroid - source_centroid @ rotation.T
    return rotation, translation


def _quaternions_to_rotation_matrices(quaternions: Tensor) -> Tensor:
    normalized = F.normalize(quaternions, dim=-1)
    w, x, y, z = normalized.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


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


def _positive_tolerance(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _trimmed_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


__all__ = [
    "GaussianDeformationReport",
    "GaussianTensorSet",
    "assign_instance_id",
    "compose_foreground_frame_gaussians",
    "compose_foreground_gaussians",
    "compose_frame_gaussians",
    "compose_gaussians",
    "gaussian_covariances",
    "transform_gaussians",
    "validate_articulated_deformation",
    "validate_asset_tensors",
    "validate_identified_asset_tensors",
]
