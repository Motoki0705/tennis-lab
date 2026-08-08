"""Tests for foreground-only Torch rasterization and public NHT composition."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.composition.contracts import GaussianCoordinates
from src.synthetic_data_generation.composition.gaussians import GaussianTensorSet
from src.synthetic_data_generation.rendering.foreground import (
    RGB_APPEARANCE_MODEL,
    RGB_APPEARANCE_SPACE,
    ForegroundRenderResult,
    TorchGaussianForegroundRasterizer,
    composite_foreground_over_nht,
)
from src.synthetic_data_generation.rendering.nht.contracts import NHTRenderRecord
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def _camera(*, width: int = 17, height: int = 17) -> SceneCamera:
    return SceneCamera(
        camera_id="camera-1",
        source_frame_index=0,
        width=width,
        height=height,
        intrinsics=(20.0, 0.0, 8.0, 0.0, 20.0, 8.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="request-only",
    )


def _gaussians(
    *,
    means: list[list[float]],
    colours: list[list[float]],
    instance_ids: list[int],
    device: torch.device | str = "cpu",
) -> GaussianTensorSet:
    count = len(means)
    return GaussianTensorSet(
        means=torch.tensor(means, dtype=torch.float32, device=device),
        quaternions_wxyz=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count,
            dtype=torch.float32,
            device=device,
        ),
        log_scales=torch.log(
            torch.full((count, 3), 0.05, dtype=torch.float32, device=device)
        ),
        opacity_logits=torch.full(
            (count,),
            6.0,
            dtype=torch.float32,
            device=device,
        ),
        features=torch.tensor(colours, dtype=torch.float32, device=device),
        instance_ids=torch.tensor(instance_ids, dtype=torch.int64, device=device),
        coordinates=GaussianCoordinates.scene(),
        appearance_model=RGB_APPEARANCE_MODEL,
        appearance_space=RGB_APPEARANCE_SPACE,
    )


def _rasterizer() -> TorchGaussianForegroundRasterizer:
    return TorchGaussianForegroundRasterizer(
        sigma_extent=3.0,
        minimum_pixel_variance=0.25,
        near_plane=1.0e-4,
        visibility_threshold=1.0e-4,
        maximum_alpha=0.999,
    )


def test_cpu_rasterizer_projects_rgb_gaussian_with_exact_output_contract() -> None:
    camera = _camera()
    gaussians = _gaussians(
        means=[[0.0, 0.0, 2.0]],
        colours=[[1.0, 0.0, 0.0]],
        instance_ids=[4],
    )
    rasterizer = _rasterizer()

    first = rasterizer.render(camera=camera, gaussians_scene=gaussians)
    second = rasterizer.render(camera=camera, gaussians_scene=gaussians)

    assert first.rgb.dtype == np.dtype(np.float32)
    assert first.alpha.dtype == np.dtype(np.float32)
    assert first.depth.dtype == np.dtype(np.float32)
    assert first.instance_mask.dtype == np.dtype(np.int32)
    assert first.rgb.shape == (17, 17, 3)
    assert first.alpha.shape == (17, 17, 1)
    assert first.depth.shape == (17, 17, 1)
    assert first.instance_mask.shape == (17, 17)
    assert first.instance_mask[8, 8] == 4
    assert first.depth[8, 8, 0] == pytest.approx(2.0)
    assert first.rgb[8, 8, 0] > 0.99
    np.testing.assert_array_equal(first.rgb, second.rgb)
    np.testing.assert_array_equal(first.instance_mask, second.instance_mask)


def test_rasterizer_front_to_back_order_selects_nearest_depth() -> None:
    result = _rasterizer().render(
        camera=_camera(),
        gaussians_scene=_gaussians(
            means=[[0.0, 0.0, 4.0], [0.0, 0.0, 2.0]],
            colours=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            instance_ids=[3, 3],
        ),
    )

    assert result.instance_mask[8, 8] == 3
    assert result.depth[8, 8, 0] == pytest.approx(2.0)
    assert result.rgb[8, 8, 0] > result.rgb[8, 8, 2]


def test_rasterizer_preserves_multiple_positive_object_identities() -> None:
    result = _rasterizer().render(
        camera=_camera(),
        gaussians_scene=_gaussians(
            means=[[-0.3, 0.0, 2.0], [0.3, 0.0, 2.0]],
            colours=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            instance_ids=[6, 7],
        ),
    )

    assert set(np.unique(result.instance_mask)) == {0, 6, 7}
    assert result.instance_ids == (6, 7)


def test_rasterizer_preserves_support_across_internal_tile_boundaries() -> None:
    result = _rasterizer().render(
        camera=_camera(width=65, height=33),
        gaussians_scene=_gaussians(
            means=[[2.4, 0.8, 2.0]],
            colours=[[0.0, 1.0, 0.0]],
            instance_ids=[9],
        ),
    )

    assert result.instance_mask[16, 31] == 9
    assert result.instance_mask[16, 32] == 9
    assert result.depth[16, 31, 0] == pytest.approx(2.0)
    assert result.depth[16, 32, 0] == pytest.approx(2.0)


def test_rasterizer_rejects_unsupported_appearance_and_invisible_geometry() -> None:
    camera = _camera()
    valid = _gaussians(
        means=[[0.0, 0.0, 2.0]],
        colours=[[1.0, 0.0, 0.0]],
        instance_ids=[1],
    )

    with pytest.raises(ValueError, match="only explicit linear RGB"):
        _rasterizer().render(
            camera=camera,
            gaussians_scene=replace(valid, appearance_model="deferred"),
        )
    with pytest.raises(ValueError, match="positive camera depth"):
        _rasterizer().render(
            camera=camera,
            gaussians_scene=replace(valid, means=valid.means.new_tensor([[0.0, 0.0, -1.0]])),
        )
    with pytest.raises(ValueError, match="not renderer-visible"):
        _rasterizer().render(
            camera=camera,
            gaussians_scene=replace(valid, means=valid.means.new_tensor([[100.0, 0.0, 2.0]])),
        )


def _nht_record(tmp_path: Path, *, depth: np.ndarray) -> NHTRenderRecord:
    rgb_path = tmp_path / "rgb.npy"
    alpha_path = tmp_path / "alpha.npy"
    depth_path = tmp_path / "depth.npy"
    np.save(rgb_path, np.broadcast_to(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (2, 2, 3)))
    np.save(alpha_path, np.ones((2, 2, 1), dtype=np.float32))
    np.save(depth_path, depth)
    return NHTRenderRecord(
        camera_id="camera-1",
        request_source="arbitrary",
        width=2,
        height=2,
        rgb_path=rgb_path,
        rgb_preview_path=tmp_path / "rgb.png",
        alpha_path=alpha_path,
        alpha_preview_path=tmp_path / "alpha.png",
        depth_path=depth_path,
    )


def _foreground_result() -> ForegroundRenderResult:
    return ForegroundRenderResult(
        camera_id="camera-1",
        rgb=np.broadcast_to(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            (2, 2, 3),
        ).copy(),
        alpha=np.full((2, 2, 1), 0.5, dtype=np.float32),
        depth=np.full((2, 2, 1), 2.0, dtype=np.float32),
        instance_mask=np.full((2, 2), 5, dtype=np.int32),
        instance_ids=(5,),
        visibility_threshold=1.0e-4,
    )


def test_depth_composite_uses_only_public_nht_arrays(tmp_path: Path) -> None:
    background = _nht_record(
        tmp_path,
        depth=np.asarray([[[1.0], [3.0]], [[1.0], [3.0]]], dtype=np.float32),
    )

    result = composite_foreground_over_nht(
        background=background,
        foreground=_foreground_result(),
        nht_scene_units_per_metre=1.0,
    )

    np.testing.assert_array_equal(result.instance_mask, [[0, 5], [0, 5]])
    np.testing.assert_array_equal(result.depth[..., 0], [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(
        result.rgb[:, 0],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(
        result.rgb[:, 1],
        [[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]],
    )


def test_depth_composite_rejects_invalid_or_fully_occluded_input(tmp_path: Path) -> None:
    invalid = _nht_record(
        tmp_path,
        depth=np.ones((2, 2, 1), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="NHT background depth must be float32"):
        composite_foreground_over_nht(
            background=invalid,
            foreground=_foreground_result(),
            nht_scene_units_per_metre=1.0,
        )

    np.save(invalid.depth_path, np.ones((2, 2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="fully occluded"):
        composite_foreground_over_nht(
            background=invalid,
            foreground=_foreground_result(),
            nht_scene_units_per_metre=1.0,
        )


def test_depth_composite_converts_public_nht_depth_to_metric_units(
    tmp_path: Path,
) -> None:
    background = _nht_record(
        tmp_path,
        depth=np.ones((2, 2, 1), dtype=np.float32),
    )

    result = composite_foreground_over_nht(
        background=background,
        foreground=_foreground_result(),
        nht_scene_units_per_metre=0.25,
    )

    np.testing.assert_array_equal(result.instance_mask, np.full((2, 2), 5))
    np.testing.assert_array_equal(result.depth, np.full((2, 2, 1), 2.0))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_depth_composite_rejects_invalid_scene_scale(
    tmp_path: Path,
    scale: float,
) -> None:
    background = _nht_record(
        tmp_path,
        depth=np.ones((2, 2, 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="nht_scene_units_per_metre"):
        composite_foreground_over_nht(
            background=background,
            foreground=_foreground_result(),
            nht_scene_units_per_metre=scale,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rasterizer_accepts_cuda_scene_tensors() -> None:
    result = _rasterizer().render(
        camera=_camera(),
        gaussians_scene=_gaussians(
            means=[[0.0, 0.0, 2.0]],
            colours=[[0.0, 1.0, 0.0]],
            instance_ids=[2],
            device="cuda",
        ),
    )

    assert result.instance_mask[8, 8] == 2
