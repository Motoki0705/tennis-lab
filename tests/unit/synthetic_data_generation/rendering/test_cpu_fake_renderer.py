"""Geometry and occlusion tests for the deterministic CPU sphere renderer."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import src.synthetic_data_generation.rendering.cpu_fake_renderer as fake_module
import src.synthetic_data_generation.rendering.renderer_port as port_module
from src.synthetic_data_generation.rendering.cpu_fake_renderer import (
    CpuSceneFrame,
    DeterministicCpuSphereRenderer,
)
from src.synthetic_data_generation.rendering.renderer_port import (
    RendererPort,
    RenderRequest,
    SpherePrimitive,
    VisibilityState,
)


def _renderer(scene_camera, depth: np.ndarray) -> DeterministicCpuSphereRenderer:
    rgb = np.zeros((scene_camera.height, scene_camera.width, 3), dtype=np.uint8)
    return DeterministicCpuSphereRenderer(
        scene_fingerprint="a" * 64,
        frames={
            scene_camera.camera_id: CpuSceneFrame(
                rgb=rgb,
                depth=depth.astype(np.float32),
            )
        },
    )


def _request(scene_camera, sphere: SpherePrimitive) -> RenderRequest:
    return RenderRequest(
        scene_fingerprint="a" * 64,
        frame_index=7,
        camera=scene_camera,
        spheres=(sphere,),
        supersampling=4,
    )


def test_centered_sphere_is_green_and_fully_visible(scene_camera) -> None:
    sphere = SpherePrimitive(
        primitive_id="ball",
        center_scene=(0.0, 0.0, 4.0),
        radius_scene_units=0.2,
        color_rgb=(32, 224, 64),
    )
    renderer = _renderer(
        scene_camera,
        np.full((scene_camera.height, scene_camera.width), np.inf),
    )

    first = renderer.render(_request(scene_camera, sphere))
    second = renderer.render(_request(scene_camera, sphere))
    evidence = first.spheres[0]

    assert isinstance(renderer, RendererPort)
    np.testing.assert_array_equal(first.rgb, second.rgb)
    np.testing.assert_array_equal(first.alpha, first.coverage)
    assert evidence.projected_center_xy == pytest.approx((31.5, 23.5))
    assert evidence.apparent_diameter_px == pytest.approx(5.0)
    assert evidence.visibility == VisibilityState.FULLY_VISIBLE
    assert evidence.visible_pixel_fraction == pytest.approx(1.0)
    assert evidence.covered_pixel_equivalent > 15.0
    assert first.alpha.max() > 0.0
    green_pixels = first.rgb[first.alpha > 0.99]
    assert len(green_pixels) > 0
    np.testing.assert_array_equal(green_pixels[0], (32, 224, 64))
    assert first.metadata.frame_index == 7
    assert not first.rgb.flags.writeable


def test_static_scene_depth_fully_occludes_sphere(scene_camera) -> None:
    sphere = SpherePrimitive(
        primitive_id="ball",
        center_scene=(0.0, 0.0, 4.0),
        radius_scene_units=0.2,
    )
    renderer = _renderer(
        scene_camera,
        np.full((scene_camera.height, scene_camera.width), 2.0),
    )

    result = renderer.render(_request(scene_camera, sphere))
    evidence = result.spheres[0]

    assert evidence.visibility == VisibilityState.FULLY_OCCLUDED
    assert evidence.in_frame
    assert evidence.covered_pixel_equivalent > 0.0
    assert evidence.visible_pixel_equivalent == 0.0
    assert np.max(result.alpha) == 0.0
    assert np.max(result.coverage) > 0.0
    assert np.isfinite(result.sphere_depth).any()


def test_half_plane_scene_depth_partially_occludes_sphere(scene_camera) -> None:
    depth = np.full((scene_camera.height, scene_camera.width), np.inf)
    depth[:, : scene_camera.width // 2] = 2.0
    sphere = SpherePrimitive(
        primitive_id="ball",
        center_scene=(0.0, 0.0, 4.0),
        radius_scene_units=0.3,
    )

    result = _renderer(scene_camera, depth).render(_request(scene_camera, sphere))
    evidence = result.spheres[0]

    assert evidence.visibility == VisibilityState.PARTIALLY_OCCLUDED
    assert 0.35 < evidence.visible_pixel_fraction < 0.65
    assert 0.0 < evidence.visible_pixel_equivalent < evidence.covered_pixel_equivalent


def test_sphere_outside_camera_is_out_of_frame(scene_camera) -> None:
    sphere = SpherePrimitive(
        primitive_id="ball",
        center_scene=(100.0, 0.0, 4.0),
        radius_scene_units=0.2,
    )

    result = _renderer(
        scene_camera,
        np.full((scene_camera.height, scene_camera.width), np.inf),
    ).render(_request(scene_camera, sphere))
    evidence = result.spheres[0]

    assert evidence.visibility == VisibilityState.OUT_OF_FRAME
    assert not evidence.in_frame
    assert evidence.covered_pixel_equivalent == 0.0
    assert np.max(result.coverage) == 0.0


def test_nearer_sphere_occludes_farther_sphere(scene_camera) -> None:
    near = SpherePrimitive(
        primitive_id="near",
        center_scene=(0.0, 0.0, 3.0),
        radius_scene_units=0.25,
    )
    far = SpherePrimitive(
        primitive_id="far",
        center_scene=(0.0, 0.0, 4.0),
        radius_scene_units=0.2,
    )
    renderer = _renderer(
        scene_camera,
        np.full((scene_camera.height, scene_camera.width), np.inf),
    )
    request = RenderRequest(
        scene_fingerprint="a" * 64,
        frame_index=0,
        camera=scene_camera,
        spheres=(near, far),
        supersampling=4,
    )

    result = renderer.render(request)

    assert result.spheres[0].visibility == VisibilityState.FULLY_VISIBLE
    assert result.spheres[1].visibility == VisibilityState.FULLY_OCCLUDED


def test_empty_sphere_request_preserves_negative_frame(scene_camera) -> None:
    rgb = np.full(
        (scene_camera.height, scene_camera.width, 3),
        (12, 34, 56),
        dtype=np.uint8,
    )
    renderer = DeterministicCpuSphereRenderer(
        scene_fingerprint="a" * 64,
        frames={
            scene_camera.camera_id: CpuSceneFrame(
                rgb=rgb,
                depth=np.full(
                    (scene_camera.height, scene_camera.width),
                    np.inf,
                    dtype=np.float32,
                ),
            )
        },
    )
    request = RenderRequest(
        scene_fingerprint="a" * 64,
        frame_index=8,
        camera=scene_camera,
        spheres=(),
        supersampling=4,
    )

    result = renderer.render(request)

    np.testing.assert_array_equal(result.rgb, rgb)
    assert np.max(result.alpha) == 0.0
    assert np.max(result.coverage) == 0.0
    assert np.isinf(result.sphere_depth).all()
    assert result.spheres == ()


def test_conservative_roi_matches_full_frame_sampling(
    scene_camera,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = np.full((scene_camera.height, scene_camera.width), np.inf)
    depth[16:30, 25:35] = 3.5
    spheres = (
        SpherePrimitive(
            primitive_id="partly-occluded",
            center_scene=(-0.2, 0.05, 4.0),
            radius_scene_units=0.25,
        ),
        SpherePrimitive(
            primitive_id="edge",
            center_scene=(2.4, -1.2, 4.0),
            radius_scene_units=0.3,
        ),
    )
    renderer = _renderer(scene_camera, depth)
    request = RenderRequest(
        scene_fingerprint="a" * 64,
        frame_index=9,
        camera=scene_camera,
        spheres=spheres,
        supersampling=4,
    )

    roi_result = renderer.render(request)
    monkeypatch.setattr(
        fake_module,
        "_conservative_render_region",
        lambda **_: (0, scene_camera.width, 0, scene_camera.height),
    )
    full_result = renderer.render(request)

    np.testing.assert_array_equal(roi_result.rgb, full_result.rgb)
    np.testing.assert_array_equal(roi_result.alpha, full_result.alpha)
    np.testing.assert_array_equal(roi_result.coverage, full_result.coverage)
    np.testing.assert_array_equal(roi_result.sphere_depth, full_result.sphere_depth)
    assert roi_result.spheres == full_result.spheres


def test_renderer_contract_has_no_gsplat_or_task_imports() -> None:
    imported: set[str] = set()
    for module in (port_module, fake_module):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)

    forbidden = ("gsplat", "src.tasks")
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported
        for prefix in forbidden
    )
