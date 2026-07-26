"""Unit tests for the renderer-independent sphere result contract."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.rendering.renderer_port import (
    RenderRequest,
    SpherePrimitive,
    SphereRenderEvidence,
    VisibilityState,
)


def test_sphere_primitive_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        SpherePrimitive(
            primitive_id="ball",
            center_scene=(0.0, 0.0, 1.0),
            radius_scene_units=0.0,
        )


def test_render_request_rejects_duplicate_primitive_ids(scene_camera) -> None:
    sphere = SpherePrimitive(
        primitive_id="ball",
        center_scene=(0.0, 0.0, 3.0),
        radius_scene_units=0.1,
    )

    with pytest.raises(ValueError, match="ids must be unique"):
        RenderRequest(
            scene_fingerprint="a" * 64,
            frame_index=0,
            camera=scene_camera,
            spheres=(sphere, sphere),
        )


def test_evidence_rejects_visibility_inconsistent_with_pixels() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        SphereRenderEvidence(
            primitive_id="ball",
            projected_center_xy=(10.0, 10.0),
            apparent_diameter_px=2.0,
            centre_depth_scene_units=3.0,
            in_frame=True,
            covered_pixel_equivalent=2.0,
            visible_pixel_equivalent=0.0,
            visible_pixel_fraction=0.0,
            visibility=VisibilityState.FULLY_VISIBLE,
        )
