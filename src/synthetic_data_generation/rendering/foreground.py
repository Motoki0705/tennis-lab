"""Deterministic foreground-only Gaussian rasterization and NHT depth composition.

This module consumes only semantic scene-space Gaussian tensors, validated
``SceneCamera`` geometry, and the public arrays named by ``NHTRenderRecord``.
It does not import a renderer backend or inspect reconstruction state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.synthetic_data_generation.composition.contracts import GaussianCoordinates
from src.synthetic_data_generation.composition.gaussians import (
    GaussianTensorSet,
    gaussian_covariances,
)
from src.synthetic_data_generation.rendering.nht.contracts import NHTRenderRecord
from src.synthetic_data_generation.rendering.nht.depth import nht_depth_to_metric
from src.synthetic_data_generation.scene_contract import SceneCamera

RGB_APPEARANCE_MODEL = "rgb"
RGB_APPEARANCE_SPACE = "linear_rgb"
_RASTER_TILE_SIZE = 32
_MAXIMUM_BATCH_ELEMENTS = 8_388_608


@dataclass(frozen=True, slots=True)
class ForegroundRenderResult:
    """Straight RGB, alpha, nearest-visible depth, and foreground instance mask."""

    camera_id: str
    rgb: NDArray[np.float32]
    alpha: NDArray[np.float32]
    depth: NDArray[np.float32]
    instance_mask: NDArray[np.int32]
    instance_ids: tuple[int, ...]
    visibility_threshold: float

    def __post_init__(self) -> None:
        threshold = _positive_float(
            self.visibility_threshold,
            name="visibility_threshold",
        )
        instance_ids = _positive_unique_ids(self.instance_ids, name="instance_ids")
        rgb, alpha, depth, mask = _validate_render_arrays(
            rgb=self.rgb,
            alpha=self.alpha,
            depth=self.depth,
            instance_mask=self.instance_mask,
        )
        actual_ids = {int(value) for value in np.unique(mask)}
        unexpected = actual_ids.difference({0, *instance_ids})
        if unexpected:
            raise ValueError(
                f"Foreground mask contains unexpected instance IDs: {sorted(unexpected)}."
            )
        visible_alpha = alpha[..., 0] > threshold
        if np.any(visible_alpha & (mask <= 0)):
            raise ValueError(
                "Every visible foreground pixel must have a positive instance ID."
            )
        if np.any((mask > 0) & (depth[..., 0] <= 0.0)):
            raise ValueError("Every foreground instance pixel must have positive depth.")
        invisible = [
            instance_id
            for instance_id in instance_ids
            if not np.any(mask == instance_id)
        ]
        if invisible:
            raise ValueError(
                f"Foreground instances are not renderer-visible: {invisible}."
            )
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "instance_mask", mask)
        object.__setattr__(self, "instance_ids", instance_ids)
        object.__setattr__(self, "visibility_threshold", threshold)

    @property
    def height(self) -> int:
        """Return output height."""
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        """Return output width."""
        return int(self.rgb.shape[1])


@dataclass(frozen=True, slots=True)
class CompositedRenderResult:
    """Foreground-over-NHT arrays with foreground instance visibility."""

    camera_id: str
    rgb: NDArray[np.float32]
    alpha: NDArray[np.float32]
    depth: NDArray[np.float32]
    instance_mask: NDArray[np.int32]
    foreground_instance_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        instance_ids = _positive_unique_ids(
            self.foreground_instance_ids,
            name="foreground_instance_ids",
        )
        rgb, alpha, depth, mask = _validate_render_arrays(
            rgb=self.rgb,
            alpha=self.alpha,
            depth=self.depth,
            instance_mask=self.instance_mask,
        )
        actual_ids = {int(value) for value in np.unique(mask)}
        unexpected = actual_ids.difference({0, *instance_ids})
        if unexpected:
            raise ValueError(
                f"Composite mask contains unexpected instance IDs: {sorted(unexpected)}."
            )
        if np.any((mask > 0) & (depth[..., 0] <= 0.0)):
            raise ValueError("Visible foreground composite pixels require positive depth.")
        invisible = [
            instance_id
            for instance_id in instance_ids
            if not np.any(mask == instance_id)
        ]
        if invisible:
            raise ValueError(
                f"Foreground instances are fully occluded after depth composition: {invisible}."
            )
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "instance_mask", mask)
        object.__setattr__(self, "foreground_instance_ids", instance_ids)


@dataclass(frozen=True, slots=True)
class TorchGaussianForegroundRasterizer:
    """Small deterministic Torch rasterizer for explicit RGB Gaussian foregrounds."""

    sigma_extent: float
    minimum_pixel_variance: float
    near_plane: float
    visibility_threshold: float
    maximum_alpha: float

    def __post_init__(self) -> None:
        sigma_extent = _positive_float(self.sigma_extent, name="sigma_extent")
        minimum_pixel_variance = _positive_float(
            self.minimum_pixel_variance,
            name="minimum_pixel_variance",
        )
        near_plane = _positive_float(self.near_plane, name="near_plane")
        visibility_threshold = _positive_float(
            self.visibility_threshold,
            name="visibility_threshold",
        )
        maximum_alpha = _positive_float(self.maximum_alpha, name="maximum_alpha")
        if maximum_alpha >= 1.0:
            raise ValueError("maximum_alpha must be less than 1.")
        object.__setattr__(self, "sigma_extent", sigma_extent)
        object.__setattr__(self, "minimum_pixel_variance", minimum_pixel_variance)
        object.__setattr__(self, "near_plane", near_plane)
        object.__setattr__(self, "visibility_threshold", visibility_threshold)
        object.__setattr__(self, "maximum_alpha", maximum_alpha)

    def render(
        self,
        *,
        camera: SceneCamera,
        gaussians_scene: GaussianTensorSet,
    ) -> ForegroundRenderResult:
        """Rasterize positive-identity scene Gaussians on their Torch device."""
        _validate_raster_input(camera=camera, gaussians_scene=gaussians_scene)
        device = gaussians_scene.means.device
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Foreground rasterization supports only CPU and CUDA tensors.")
        instance_ids = tuple(
            sorted(int(value) for value in torch.unique(gaussians_scene.instance_ids).cpu())
        )
        with torch.no_grad():
            rgb, alpha, depth, mask = self._render_tensors(
                camera=camera,
                gaussians_scene=gaussians_scene,
            )
        return ForegroundRenderResult(
            camera_id=camera.camera_id,
            rgb=_float_tensor_numpy(rgb),
            alpha=_float_tensor_numpy(alpha[..., None]),
            depth=_float_tensor_numpy(depth[..., None]),
            instance_mask=_int_tensor_numpy(mask),
            instance_ids=instance_ids,
            visibility_threshold=self.visibility_threshold,
        )

    def _render_tensors(
        self,
        *,
        camera: SceneCamera,
        gaussians_scene: GaussianTensorSet,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = gaussians_scene.means.device
        dtype = torch.float32
        height = camera.height
        width = camera.width
        means_scene = gaussians_scene.means.to(dtype=dtype)
        covariances_scene = gaussian_covariances(gaussians_scene).to(dtype=dtype)
        features = gaussians_scene.features.to(dtype=dtype)
        opacities = torch.sigmoid(gaussians_scene.opacity_logits.to(dtype=dtype))
        instance_ids = gaussians_scene.instance_ids

        camera_from_scene = camera.camera_to_scene.inverse().matrix()
        rotation = torch.tensor(
            camera_from_scene[:3, :3],
            dtype=dtype,
            device=device,
        )
        translation = torch.tensor(
            camera_from_scene[:3, 3],
            dtype=dtype,
            device=device,
        )
        means_camera = means_scene @ rotation.T + translation
        depths = means_camera[:, 2]
        if not bool(torch.isfinite(means_camera).all()) or bool(
            (depths <= self.near_plane).any()
        ):
            raise ValueError(
                "Every foreground Gaussian mean must have finite positive camera depth."
            )
        covariances_camera = (
            rotation @ covariances_scene @ rotation.transpose(-1, -2)
        )

        intrinsics = torch.tensor(
            np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3),
            dtype=dtype,
            device=device,
        )
        homogeneous = means_camera @ intrinsics.T
        denominator = homogeneous[:, 2]
        pixels = homogeneous[:, :2] / denominator[:, None]
        jacobian = (
            intrinsics[:2][None, :, :] * denominator[:, None, None]
            - homogeneous[:, :2, None] * intrinsics[2][None, None, :]
        ) / denominator[:, None, None].square()
        covariances_pixel = (
            jacobian @ covariances_camera @ jacobian.transpose(-1, -2)
        )
        identity = torch.eye(2, dtype=dtype, device=device)
        covariances_pixel = (
            covariances_pixel + self.minimum_pixel_variance * identity
        )
        if not bool(torch.isfinite(pixels).all()) or not bool(
            torch.isfinite(covariances_pixel).all()
        ):
            raise ValueError("Projected foreground Gaussian geometry is non-finite.")
        eigenvalues = torch.linalg.eigvalsh(covariances_pixel)
        if bool((eigenvalues <= 0.0).any()):
            raise ValueError("Projected foreground covariance must be positive definite.")

        order = torch.argsort(depths, stable=True)
        support_squared = self.sigma_extent * self.sigma_extent
        inverse_covariances = torch.linalg.inv(covariances_pixel)
        tile_candidates = _tile_candidate_indices(
            pixels=pixels,
            radii=self.sigma_extent * torch.sqrt(eigenvalues[:, -1]),
            order=order,
            width=width,
            height=height,
            tile_size=_RASTER_TILE_SIZE,
        )
        rgb, alpha, depth, mask = _rasterize_tiles(
            pixels=pixels,
            inverse_covariances=inverse_covariances,
            depths=depths,
            opacities=opacities,
            features=features,
            instance_ids=instance_ids,
            tile_candidates=tile_candidates,
            width=width,
            height=height,
            tile_size=_RASTER_TILE_SIZE,
            support_squared=support_squared,
            visibility_threshold=self.visibility_threshold,
            maximum_alpha=self.maximum_alpha,
        )
        if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(depth).all()):
            raise ValueError("Foreground rasterization produced non-finite output.")
        return rgb, alpha, depth, mask


def _tile_candidate_indices(
    *,
    pixels: Tensor,
    radii: Tensor,
    order: Tensor,
    width: int,
    height: int,
    tile_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Bin projected Gaussians into tiles in stable front-to-back order."""
    tile_columns = math.ceil(width / tile_size)
    tile_rows = math.ceil(height / tile_size)
    candidates: list[list[int]] = [[] for _ in range(tile_columns * tile_rows)]
    geometry = torch.cat((pixels, radii[:, None]), dim=1).detach().cpu().numpy()
    for raw_index in order.detach().cpu().tolist():
        index = int(raw_index)
        centre_x, centre_y, radius = (float(value) for value in geometry[index])
        x_min = max(0, math.floor(centre_x - radius))
        x_max = min(width, math.ceil(centre_x + radius) + 1)
        y_min = max(0, math.floor(centre_y - radius))
        y_max = min(height, math.ceil(centre_y + radius) + 1)
        if x_min >= x_max or y_min >= y_max:
            continue
        for tile_y in range(y_min // tile_size, (y_max - 1) // tile_size + 1):
            row_offset = tile_y * tile_columns
            for tile_x in range(x_min // tile_size, (x_max - 1) // tile_size + 1):
                candidates[row_offset + tile_x].append(index)
    return tuple(tuple(indices) for indices in candidates)


def _rasterize_tiles(
    *,
    pixels: Tensor,
    inverse_covariances: Tensor,
    depths: Tensor,
    opacities: Tensor,
    features: Tensor,
    instance_ids: Tensor,
    tile_candidates: tuple[tuple[int, ...], ...],
    width: int,
    height: int,
    tile_size: int,
    support_squared: float,
    visibility_threshold: float,
    maximum_alpha: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Alpha-composite tile batches without synchronizing once per Gaussian."""
    device = pixels.device
    dtype = pixels.dtype
    tile_columns = math.ceil(width / tile_size)
    tile_rows = math.ceil(height / tile_size)
    tile_count = tile_columns * tile_rows
    rgb_tiles = torch.zeros(
        (tile_count, tile_size, tile_size, 3), dtype=dtype, device=device
    )
    alpha_tiles = torch.zeros(
        (tile_count, tile_size, tile_size), dtype=dtype, device=device
    )
    depth_tiles = torch.zeros_like(alpha_tiles)
    mask_tiles = torch.zeros(
        (tile_count, tile_size, tile_size), dtype=torch.int32, device=device
    )

    buckets: dict[int, list[int]] = {}
    for tile_id, candidates in enumerate(tile_candidates):
        if candidates:
            bucket = (len(candidates) - 1).bit_length()
            buckets.setdefault(bucket, []).append(tile_id)

    offsets = torch.arange(tile_size, dtype=dtype, device=device)
    for bucket in sorted(buckets):
        bucket_tiles = buckets[bucket]
        maximum_candidates = max(len(tile_candidates[tile_id]) for tile_id in bucket_tiles)
        elements_per_tile = maximum_candidates * tile_size * tile_size
        batch_size = max(
            1,
            min(64, _MAXIMUM_BATCH_ELEMENTS // max(1, elements_per_tile)),
        )
        for batch_start in range(0, len(bucket_tiles), batch_size):
            batch_tile_ids = bucket_tiles[batch_start : batch_start + batch_size]
            batch_count = len(batch_tile_ids)
            padded_indices: NDArray[np.int64] = np.zeros(
                (batch_count, maximum_candidates), dtype=np.int64
            )
            candidate_valid: NDArray[np.bool_] = np.zeros(
                (batch_count, maximum_candidates), dtype=np.bool_
            )
            for batch_index, tile_id in enumerate(batch_tile_ids):
                indices = tile_candidates[tile_id]
                padded_indices[batch_index, : len(indices)] = indices
                candidate_valid[batch_index, : len(indices)] = True

            gaussian_indices = torch.from_numpy(padded_indices).to(device=device)
            valid_gaussians = torch.from_numpy(candidate_valid).to(device=device)
            tile_ids = torch.tensor(batch_tile_ids, dtype=torch.int64, device=device)
            tile_x = torch.remainder(tile_ids, tile_columns)
            tile_y = torch.div(tile_ids, tile_columns, rounding_mode="floor")
            grid_x = tile_x[:, None, None] * tile_size + offsets[None, None, :]
            grid_y = tile_y[:, None, None] * tile_size + offsets[None, :, None]
            valid_pixels = (grid_x < width) & (grid_y < height)

            centres = pixels[gaussian_indices]
            delta_x = grid_x[:, None] - centres[:, :, 0, None, None]
            delta_y = grid_y[:, None] - centres[:, :, 1, None, None]
            inverse = inverse_covariances[gaussian_indices]
            mahalanobis = (
                inverse[:, :, 0, 0, None, None] * delta_x.square()
                + (inverse[:, :, 0, 1, None, None] + inverse[:, :, 1, 0, None, None])
                * delta_x
                * delta_y
                + inverse[:, :, 1, 1, None, None] * delta_y.square()
            )
            support = mahalanobis <= support_squared
            support &= valid_gaussians[:, :, None, None]
            support &= valid_pixels[:, None]
            gaussian_alpha = torch.where(
                support,
                opacities[gaussian_indices, None, None] * torch.exp(-0.5 * mahalanobis),
                torch.zeros_like(mahalanobis),
            ).clamp(max=maximum_alpha)
            transmittance_before = torch.cumprod(
                torch.cat(
                    (
                        torch.ones(
                            (batch_count, 1, tile_size, tile_size),
                            dtype=dtype,
                            device=device,
                        ),
                        1.0 - gaussian_alpha[:, :-1],
                    ),
                    dim=1,
                ),
                dim=1,
            )
            contribution = transmittance_before * gaussian_alpha
            batch_alpha = 1.0 - torch.prod(1.0 - gaussian_alpha, dim=1)
            batch_rgb_premultiplied = torch.sum(
                contribution[..., None] * features[gaussian_indices, None, None, :],
                dim=1,
            )
            winner_contribution, winner_position = torch.max(contribution, dim=1)
            winner_index = torch.gather(
                gaussian_indices,
                1,
                winner_position.reshape(batch_count, -1),
            ).reshape(batch_count, tile_size, tile_size)
            visible = batch_alpha > visibility_threshold
            has_winner = winner_contribution > visibility_threshold
            batch_rgb = torch.where(
                visible[..., None],
                batch_rgb_premultiplied
                / batch_alpha.clamp_min(visibility_threshold)[..., None],
                torch.zeros_like(batch_rgb_premultiplied),
            ).clamp(0.0, 1.0)
            batch_depth = torch.where(
                visible & has_winner,
                depths[winner_index],
                torch.zeros_like(batch_alpha),
            )
            batch_mask = torch.where(
                visible & has_winner,
                instance_ids[winner_index].to(torch.int32),
                torch.zeros_like(winner_index, dtype=torch.int32),
            )
            rgb_tiles[tile_ids] = batch_rgb
            alpha_tiles[tile_ids] = batch_alpha
            depth_tiles[tile_ids] = batch_depth
            mask_tiles[tile_ids] = batch_mask

    def untile(values: Tensor) -> Tensor:
        trailing = values.shape[3:]
        tiled = values.reshape(tile_rows, tile_columns, tile_size, tile_size, *trailing)
        permutation = (0, 2, 1, 3, *range(4, tiled.ndim))
        return tiled.permute(permutation).reshape(
            tile_rows * tile_size,
            tile_columns * tile_size,
            *trailing,
        )[:height, :width]

    return untile(rgb_tiles), untile(alpha_tiles), untile(depth_tiles), untile(mask_tiles)


def composite_foreground_over_nht(
    *,
    background: NHTRenderRecord,
    foreground: ForegroundRenderResult,
    nht_scene_units_per_metre: float,
) -> CompositedRenderResult:
    """Depth-test metric foreground against explicitly scaled public NHT depth."""
    if not isinstance(background, NHTRenderRecord):
        raise TypeError("background must be an NHTRenderRecord.")
    if background.camera_id != foreground.camera_id:
        raise ValueError("Foreground and NHT background camera IDs differ.")
    if (background.height, background.width) != (foreground.height, foreground.width):
        raise ValueError("Foreground and NHT background resolutions differ.")
    shape = (background.height, background.width)
    background_rgb = _load_public_nht_array(
        background.rgb_path,
        shape=(*shape, 3),
        name="NHT background RGB",
        unit_range=True,
    )
    background_alpha = _load_public_nht_array(
        background.alpha_path,
        shape=(*shape, 1),
        name="NHT background alpha",
        unit_range=True,
    )
    background_depth = nht_depth_to_metric(
        _load_public_nht_array(
            background.depth_path,
            shape=(*shape, 1),
            name="NHT background depth",
            nonnegative=True,
        ),
        nht_scene_units_per_metre=nht_scene_units_per_metre,
    )

    foreground_present = foreground.instance_mask > 0
    foreground_depth = foreground.depth[..., 0]
    nht_depth = background_depth[..., 0]
    foreground_in_front = foreground_present & (
        (nht_depth <= 0.0) | (foreground_depth < nht_depth)
    )
    foreground_alpha = np.where(
        foreground_in_front[..., None],
        foreground.alpha,
        np.float32(0.0),
    )
    rgb = (
        foreground.rgb * foreground_alpha
        + background_rgb * (np.float32(1.0) - foreground_alpha)
    )
    alpha = foreground_alpha + background_alpha * (
        np.float32(1.0) - foreground_alpha
    )
    depth = np.where(
        foreground_in_front[..., None],
        foreground.depth,
        background_depth,
    )
    mask = np.where(
        foreground_in_front,
        foreground.instance_mask,
        np.int32(0),
    )
    return CompositedRenderResult(
        camera_id=foreground.camera_id,
        rgb=np.asarray(rgb, dtype=np.float32),
        alpha=np.asarray(alpha, dtype=np.float32),
        depth=np.asarray(depth, dtype=np.float32),
        instance_mask=np.asarray(mask, dtype=np.int32),
        foreground_instance_ids=foreground.instance_ids,
    )


def _validate_raster_input(
    *,
    camera: SceneCamera,
    gaussians_scene: GaussianTensorSet,
) -> None:
    if not isinstance(camera, SceneCamera):
        raise TypeError("camera must be a SceneCamera.")
    if not isinstance(gaussians_scene, GaussianTensorSet):
        raise TypeError("gaussians_scene must be a GaussianTensorSet.")
    if gaussians_scene.coordinates != GaussianCoordinates.scene():
        raise ValueError("Foreground rasterization requires scene-space Gaussians.")
    if gaussians_scene.appearance_model != RGB_APPEARANCE_MODEL or (
        gaussians_scene.appearance_space != RGB_APPEARANCE_SPACE
    ):
        raise ValueError(
            "Foreground rasterization supports only explicit linear RGB features."
        )
    if gaussians_scene.feature_dim != 3:
        raise ValueError("Foreground RGB features must have shape [N,3].")
    if bool((gaussians_scene.features < 0.0).any()) or bool(
        (gaussians_scene.features > 1.0).any()
    ):
        raise ValueError("Foreground RGB features must stay in [0,1].")
    if bool((gaussians_scene.instance_ids <= 0).any()):
        raise ValueError("Foreground Gaussians require positive instance IDs; 0 is empty.")


def _validate_render_arrays(
    *,
    rgb: object,
    alpha: object,
    depth: object,
    instance_mask: object,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    if not all(isinstance(value, np.ndarray) for value in (rgb, alpha, depth, instance_mask)):
        raise TypeError("Render outputs must be numpy arrays.")
    rgb_array = np.asarray(rgb)
    alpha_array = np.asarray(alpha)
    depth_array = np.asarray(depth)
    mask_array = np.asarray(instance_mask)
    if rgb_array.dtype != np.dtype(np.float32) or rgb_array.ndim != 3:
        raise TypeError("RGB must be a float32 [H,W,3] array.")
    if rgb_array.shape[2] != 3 or rgb_array.shape[0] <= 0 or rgb_array.shape[1] <= 0:
        raise ValueError("RGB must be a non-empty [H,W,3] array.")
    height, width = rgb_array.shape[:2]
    if alpha_array.dtype != np.dtype(np.float32) or alpha_array.shape != (
        height,
        width,
        1,
    ):
        raise TypeError("Alpha must be a float32 [H,W,1] array.")
    if depth_array.dtype != np.dtype(np.float32) or depth_array.shape != (
        height,
        width,
        1,
    ):
        raise TypeError("Depth must be a float32 [H,W,1] array.")
    if mask_array.dtype != np.dtype(np.int32) or mask_array.shape != (height, width):
        raise TypeError("Instance mask must be an int32 [H,W] array.")
    if not np.isfinite(rgb_array).all() or not np.isfinite(alpha_array).all() or (
        not np.isfinite(depth_array).all()
    ):
        raise ValueError("Render RGB, alpha, and depth must contain only finite values.")
    if (
        np.any(rgb_array < 0.0)
        or np.any(rgb_array > 1.0)
        or np.any(alpha_array < 0.0)
        or np.any(alpha_array > 1.0)
        or np.any(depth_array < 0.0)
    ):
        raise ValueError("Render RGB/alpha/depth violate their numeric ranges.")
    return (
        _readonly_float_array(cast(NDArray[np.float32], rgb_array)),
        _readonly_float_array(cast(NDArray[np.float32], alpha_array)),
        _readonly_float_array(cast(NDArray[np.float32], depth_array)),
        _readonly_int_array(cast(NDArray[np.int32], mask_array)),
    )


def _load_public_nht_array(
    path: Path,
    *,
    shape: tuple[int, ...],
    name: str,
    unit_range: bool = False,
    nonnegative: bool = False,
) -> NDArray[np.float32]:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} is not an ordinary file: {path}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.dtype(np.float32) or value.shape != shape:
        raise ValueError(f"{name} must be float32 with shape {shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    if unit_range and (np.any(value < 0.0) or np.any(value > 1.0)):
        raise ValueError(f"{name} must stay in [0,1].")
    if nonnegative and np.any(value < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return cast(NDArray[np.float32], np.asarray(value, dtype=np.float32))


def _positive_unique_ids(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError(f"{name} must be a non-empty tuple.")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{name} must contain only positive integers.")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must be unique.")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _float_tensor_numpy(tensor: Tensor) -> NDArray[np.float32]:
    return cast(
        NDArray[np.float32],
        np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32),
    )


def _int_tensor_numpy(tensor: Tensor) -> NDArray[np.int32]:
    return cast(
        NDArray[np.int32],
        np.asarray(tensor.detach().cpu().numpy(), dtype=np.int32),
    )


def _readonly_float_array(value: NDArray[np.float32]) -> NDArray[np.float32]:
    result = np.ascontiguousarray(value, dtype=np.float32)
    result.setflags(write=False)
    return result


def _readonly_int_array(value: NDArray[np.int32]) -> NDArray[np.int32]:
    result = np.ascontiguousarray(value, dtype=np.int32)
    result.setflags(write=False)
    return result


__all__ = [
    "RGB_APPEARANCE_MODEL",
    "RGB_APPEARANCE_SPACE",
    "CompositedRenderResult",
    "ForegroundRenderResult",
    "TorchGaussianForegroundRasterizer",
    "composite_foreground_over_nht",
]
