"""CUDA PLCS foreground raster/composition to compact published deltas."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from src.synthetic_data_generation.composition.gaussians import GaussianTensorSet
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    ForegroundDelta,
    RenderSampleKey,
)
from src.synthetic_data_generation.rendering.foreground import (
    TorchGaussianForegroundRasterizer,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


@dataclass(frozen=True, slots=True)
class _DeviceBackground:
    camera_id: str
    rgb: Tensor
    alpha: Tensor
    depth: Tensor


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
        with torch.inference_mode():
            foreground_rgb, foreground_alpha, foreground_depth, foreground_ids = (
                self._rasterizer._render_tensors(  # noqa: SLF001
                    camera=camera,
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
            actual_ids = {
                int(value)
                for value in torch.unique(visible_ids).detach().cpu().tolist()
            }
            expected_ids = set(expected_instance_ids)
            if not actual_ids.issubset(expected_ids):
                raise ValueError(
                    "PLCS renderer produced an undeclared instance; "
                    f"unexpected={sorted(actual_ids - expected_ids)}."
                )
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
            visible_counts = {
                instance_id: int((visible_ids == instance_id).sum().item())
                for instance_id in expected_instance_ids
            }
        delta = ForegroundDelta(
            key=RenderSampleKey(frame_index, camera.camera_id),
            pixel_indices=pixel_indices.to(dtype=torch.int32).cpu().numpy(),
            rgb=selected_rgb.to(dtype=torch.float32).cpu().numpy(),
            alpha=selected_alpha.to(dtype=torch.float32).cpu().numpy(),
            depth=selected_depth.to(dtype=torch.float32).cpu().numpy(),
            instance_ids=visible_ids.to(dtype=torch.int32).cpu().numpy(),
        )
        return delta, visible_counts


__all__ = ["PLCSForegroundCompositor"]
