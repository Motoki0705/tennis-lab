"""CUDA PLCS foreground raster/composition to compact published deltas."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.synthetic_data_generation.composition.gaussians import (
    GaussianTensorSet,
    gaussian_covariances,
)
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    ForegroundDelta,
    RenderSampleKey,
)
from src.synthetic_data_generation.rendering.foreground import (
    _RASTER_TILE_SIZE,
    TorchGaussianForegroundRasterizer,
    _rasterize_tiles,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True, slots=True)
class _DeviceBackground:
    camera_id: str
    rgb: Tensor
    alpha: Tensor
    depth: Tensor


@dataclass(frozen=True, slots=True)
class _DeviceCamera:
    """One immutable generated camera uploaded once for the whole stage."""

    camera: SceneCamera
    rotation: Tensor
    translation: Tensor
    intrinsics: Tensor


@dataclass(frozen=True, slots=True)
class PLCSForegroundCompositor:
    """Keep full raster/composition arrays on CUDA and download sparse deltas."""

    sigma_extent: float
    minimum_pixel_variance: float
    near_plane: float
    visibility_threshold: float
    maximum_alpha: float
    _rasterizer: TorchGaussianForegroundRasterizer = field(init=False, repr=False)
    _backgrounds: dict[str, _DeviceBackground] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _cameras: dict[str, _DeviceCamera] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_rasterizer",
            TorchGaussianForegroundRasterizer(
                sigma_extent=self.sigma_extent,
                minimum_pixel_variance=self.minimum_pixel_variance,
                near_plane=self.near_plane,
                visibility_threshold=self.visibility_threshold,
                maximum_alpha=self.maximum_alpha,
            ),
        )

    @property
    def background_upload_count(self) -> int:
        """Return exact camera buffers uploaded in this stage-scoped compositor."""
        return len(self._backgrounds)

    def reset_stage(self) -> None:
        """Discard all prior-attempt GPU background buffers explicitly."""
        self._backgrounds.clear()
        self._cameras.clear()

    def prepare_background(
        self,
        background: BackgroundArrays,
        *,
        device: str | torch.device,
    ) -> None:
        """Upload one validated static camera background exactly once."""
        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PLCS production background composition requires CUDA.")
        if background.camera_id in self._backgrounds:
            raise ValueError(
                f"PLCS background {background.camera_id!r} was uploaded more than once."
            )
        self._backgrounds[background.camera_id] = _DeviceBackground(
            camera_id=background.camera_id,
            rgb=torch.as_tensor(
                np.array(background.rgb, copy=True),
                dtype=torch.float32,
                device=target,
            ),
            alpha=torch.as_tensor(
                np.array(background.alpha, copy=True),
                dtype=torch.float32,
                device=target,
            ),
            depth=torch.as_tensor(
                np.array(background.depth, copy=True),
                dtype=torch.float32,
                device=target,
            ),
        )

    def compose_delta(
        self,
        *,
        frame_index: int,
        camera: SceneCamera,
        gaussians_scene: GaussianTensorSet,
        expected_instance_ids: tuple[int, ...],
    ) -> tuple[ForegroundDelta, dict[int, int]]:
        """Rasterize and depth-compose on CUDA, then transfer visible pixels only."""
        if frame_index < 0:
            raise ValueError("PLCS frame_index must be non-negative.")
        if gaussians_scene.means.device.type != "cuda":
            raise ValueError(
                "PLCS production foreground tensors must be CUDA-resident."
            )
        if len(expected_instance_ids) != len(set(expected_instance_ids)) or any(
            value <= 0 for value in expected_instance_ids
        ):
            raise ValueError("Expected PLCS instance IDs must be positive and unique.")
        try:
            background = self._backgrounds[camera.camera_id]
        except KeyError as error:
            raise KeyError(
                f"PLCS camera background is not prepared: {camera.camera_id!r}."
            ) from error
        if background.rgb.device != gaussians_scene.means.device:
            raise ValueError(
                "PLCS background and foreground must share one CUDA device."
            )
        device_camera = self._device_camera(
            camera,
            device=gaussians_scene.means.device,
        )
        with torch.inference_mode():
            foreground_rgb, foreground_alpha, foreground_depth, foreground_ids = (
                _render_plcs_tensors(
                    rasterizer=self._rasterizer,
                    camera=camera,
                    device_camera=device_camera,
                    gaussians_scene=gaussians_scene,
                )
            )
            if foreground_rgb.shape != background.rgb.shape:
                raise ValueError("PLCS raster and background resolutions differ.")
            present = foreground_ids > 0
            background_depth = background.depth[..., 0]
            in_front = present & (
                (background_depth <= 0.0) | (foreground_depth < background_depth)
            )
            flat_visible = in_front.reshape(-1)
            pixel_indices = torch.nonzero(flat_visible, as_tuple=False).reshape(-1)
            flat_ids = foreground_ids.reshape(-1)
            visible_ids = flat_ids.index_select(0, pixel_indices)
            flat_foreground_alpha = foreground_alpha.reshape(-1)
            selected_foreground_alpha = flat_foreground_alpha.index_select(
                0, pixel_indices
            )
            selected_rgb = foreground_rgb.reshape(-1, 3).index_select(
                0, pixel_indices
            ) * selected_foreground_alpha[:, None] + background.rgb.reshape(
                -1, 3
            ).index_select(0, pixel_indices) * (
                1.0 - selected_foreground_alpha[:, None]
            )
            selected_alpha = selected_foreground_alpha + background.alpha.reshape(
                -1
            ).index_select(0, pixel_indices) * (1.0 - selected_foreground_alpha)
            selected_depth = foreground_depth.reshape(-1).index_select(0, pixel_indices)
            integer_payload = (
                torch.stack(
                    (pixel_indices, visible_ids),
                    dim=1,
                )
                .to(dtype=torch.int32)
                .cpu()
                .numpy()
            )
            floating_payload = (
                torch.cat(
                    (
                        selected_rgb,
                        selected_alpha[:, None],
                        selected_depth[:, None],
                    ),
                    dim=1,
                )
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
        instance_ids = integer_payload[:, 1]
        actual_ids, actual_counts = np.unique(instance_ids, return_counts=True)
        visible_counts = {instance_id: 0 for instance_id in expected_instance_ids}
        for raw_instance_id, raw_count in zip(
            actual_ids,
            actual_counts,
            strict=True,
        ):
            instance_id = int(raw_instance_id)
            if instance_id not in visible_counts:
                raise ValueError(
                    "PLCS renderer produced an undeclared instance; "
                    f"unexpected={[instance_id]}."
                )
            visible_counts[instance_id] = int(raw_count)
        delta = ForegroundDelta(
            key=RenderSampleKey(frame_index, camera.camera_id),
            pixel_indices=integer_payload[:, 0],
            rgb=floating_payload[:, :3],
            alpha=floating_payload[:, 3],
            depth=floating_payload[:, 4],
            instance_ids=instance_ids,
        )
        return delta, visible_counts

    def _device_camera(
        self,
        camera: SceneCamera,
        *,
        device: torch.device,
    ) -> _DeviceCamera:
        try:
            cached = self._cameras[camera.camera_id]
        except KeyError:
            camera_from_scene = camera.camera_to_scene.inverse().matrix()
            cached = _DeviceCamera(
                camera=camera,
                rotation=torch.tensor(
                    camera_from_scene[:3, :3],
                    dtype=torch.float32,
                    device=device,
                ),
                translation=torch.tensor(
                    camera_from_scene[:3, 3],
                    dtype=torch.float32,
                    device=device,
                ),
                intrinsics=torch.tensor(
                    np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            self._cameras[camera.camera_id] = cached
        if cached.camera != camera:
            raise ValueError(
                "PLCS camera geometry changed after its stage-scoped upload."
            )
        if cached.rotation.device != device:
            raise ValueError("PLCS generated cameras must share one CUDA device.")
        return cached


def _render_plcs_tensors(
    *,
    rasterizer: TorchGaussianForegroundRasterizer,
    camera: SceneCamera,
    device_camera: _DeviceCamera,
    gaussians_scene: GaussianTensorSet,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Rasterize with cached camera tensors and vectorized CPU tile binning."""
    device = gaussians_scene.means.device
    dtype = torch.float32
    means_scene = gaussians_scene.means.to(dtype=dtype)
    covariances_scene = gaussian_covariances(gaussians_scene).to(dtype=dtype)
    features = gaussians_scene.features.to(dtype=dtype)
    opacities = torch.sigmoid(gaussians_scene.opacity_logits.to(dtype=dtype))
    instance_ids = gaussians_scene.instance_ids

    rotation = device_camera.rotation
    translation = device_camera.translation
    intrinsics = device_camera.intrinsics
    means_camera = means_scene @ rotation.T + translation
    depths = means_camera[:, 2]
    if not bool(torch.isfinite(means_camera).all()) or bool(
        (depths <= rasterizer.near_plane).any()
    ):
        raise ValueError(
            "Every foreground Gaussian mean must have finite positive camera depth."
        )
    covariances_camera = rotation @ covariances_scene @ rotation.transpose(-1, -2)
    homogeneous = means_camera @ intrinsics.T
    denominator = homogeneous[:, 2]
    pixels = homogeneous[:, :2] / denominator[:, None]
    jacobian = (
        intrinsics[:2][None, :, :] * denominator[:, None, None]
        - homogeneous[:, :2, None] * intrinsics[2][None, None, :]
    ) / denominator[:, None, None].square()
    covariances_pixel = jacobian @ covariances_camera @ jacobian.transpose(-1, -2)
    identity = torch.eye(2, dtype=dtype, device=device)
    covariances_pixel = covariances_pixel + rasterizer.minimum_pixel_variance * identity
    if not bool(torch.isfinite(pixels).all()) or not bool(
        torch.isfinite(covariances_pixel).all()
    ):
        raise ValueError("Projected foreground Gaussian geometry is non-finite.")
    eigenvalues = torch.linalg.eigvalsh(covariances_pixel)
    if bool((eigenvalues <= 0.0).any()):
        raise ValueError("Projected foreground covariance must be positive definite.")

    order = torch.argsort(depths, stable=True)
    tile_candidates = _tile_candidate_indices_vectorized(
        pixels=pixels,
        radii=rasterizer.sigma_extent * torch.sqrt(eigenvalues[:, -1]),
        order=order,
        width=camera.width,
        height=camera.height,
        tile_size=_RASTER_TILE_SIZE,
    )
    rgb, alpha, depth, mask = _rasterize_tiles(
        pixels=pixels,
        inverse_covariances=torch.linalg.inv(covariances_pixel),
        depths=depths,
        opacities=opacities,
        features=features,
        instance_ids=instance_ids,
        tile_candidates=tile_candidates,
        width=camera.width,
        height=camera.height,
        tile_size=_RASTER_TILE_SIZE,
        support_squared=rasterizer.sigma_extent * rasterizer.sigma_extent,
        visibility_threshold=rasterizer.visibility_threshold,
        maximum_alpha=rasterizer.maximum_alpha,
    )
    if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(depth).all()):
        raise ValueError("Foreground rasterization produced non-finite output.")
    return rgb, alpha, depth, mask


def _tile_candidate_indices_vectorized(
    *,
    pixels: Tensor,
    radii: Tensor,
    order: Tensor,
    width: int,
    height: int,
    tile_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Bin Gaussians exactly in stable depth order without per-Gaussian Python."""
    tile_columns = math.ceil(width / tile_size)
    tile_rows = math.ceil(height / tile_size)
    tile_count = tile_columns * tile_rows
    geometry = (
        torch.cat((pixels, radii[:, None]), dim=1)
        .index_select(0, order)
        .detach()
        .cpu()
        .numpy()
    )
    ordered_indices = order.detach().cpu().numpy().astype(np.int64, copy=False)
    x_min = np.maximum(0, np.floor(geometry[:, 0] - geometry[:, 2])).astype(np.int64)
    x_max = np.minimum(
        width,
        np.ceil(geometry[:, 0] + geometry[:, 2]).astype(np.int64) + 1,
    )
    y_min = np.maximum(0, np.floor(geometry[:, 1] - geometry[:, 2])).astype(np.int64)
    y_max = np.minimum(
        height,
        np.ceil(geometry[:, 1] + geometry[:, 2]).astype(np.int64) + 1,
    )
    valid = (x_min < x_max) & (y_min < y_max)
    if not bool(np.any(valid)):
        return tuple(() for _ in range(tile_count))

    ordered_indices = ordered_indices[valid]
    tile_x_min = x_min[valid] // tile_size
    tile_x_count = (x_max[valid] - 1) // tile_size - tile_x_min + 1
    tile_y_min = y_min[valid] // tile_size
    tile_y_count = (y_max[valid] - 1) // tile_size - tile_y_min + 1
    pair_counts = tile_x_count * tile_y_count
    gaussian_positions: NDArray[np.int64] = np.repeat(
        np.arange(len(ordered_indices), dtype=np.int64),
        pair_counts,
    )
    pair_starts: NDArray[np.int64] = np.repeat(
        np.cumsum(pair_counts) - pair_counts,
        pair_counts,
    )
    pair_offsets = np.arange(int(pair_counts.sum()), dtype=np.int64) - pair_starts
    tile_x = (
        tile_x_min[gaussian_positions] + pair_offsets % tile_x_count[gaussian_positions]
    )
    tile_y = (
        tile_y_min[gaussian_positions]
        + pair_offsets // tile_x_count[gaussian_positions]
    )
    tile_ids = tile_y * tile_columns + tile_x
    stable_order = np.argsort(tile_ids, kind="stable")
    sorted_tile_ids = tile_ids[stable_order]
    sorted_gaussian_indices = ordered_indices[gaussian_positions[stable_order]]
    boundaries: NDArray[np.int64] = np.asarray(
        np.searchsorted(
            sorted_tile_ids,
            np.arange(tile_count + 1, dtype=np.int64),
        ),
        dtype=np.int64,
    )
    return tuple(
        tuple(map(int, sorted_gaussian_indices[int(start) : int(stop)]))
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


__all__ = ["PLCSForegroundCompositor"]
