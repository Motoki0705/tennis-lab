"""Tests for CUDA compact PLCS composition and one public NHT rig call."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.composition.contracts import GaussianCoordinates
from src.synthetic_data_generation.composition.gaussians import GaussianTensorSet
from src.synthetic_data_generation.dataset.plcs.rendering.contracts import (
    PLCSForegroundCompositor,
)
from src.synthetic_data_generation.dataset.plcs.rendering.nht import NHTPLCSRenderer
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    RenderSession,
    materialize_logical_sample,
)
from src.synthetic_data_generation.rendering.nht.client import NHTRenderClient
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderArrays,
    NHTRenderCommandRequest,
    NHTRenderRecord,
    NHTRenderResult,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def _camera(
    *, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> SceneCamera:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = translation
    return SceneCamera(
        camera_id="camera-1",
        source_frame_index=0,
        width=17,
        height=17,
        intrinsics=(20.0, 0.0, 8.0, 0.0, 20.0, 8.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path="request-only",
    )


def _background() -> BackgroundArrays:
    return BackgroundArrays(
        camera_id="camera-1",
        rgb=np.zeros((17, 17, 3), dtype=np.float32),
        alpha=np.ones((17, 17, 1), dtype=np.float32),
        depth=np.full((17, 17, 1), 10.0, dtype=np.float32),
    )


def _foreground(device: str) -> GaussianTensorSet:
    return GaussianTensorSet(
        means=torch.tensor(((0.0, 0.0, 2.0),), dtype=torch.float32, device=device),
        quaternions_wxyz=torch.tensor(
            ((1.0, 0.0, 0.0, 0.0),), dtype=torch.float32, device=device
        ),
        log_scales=torch.log(
            torch.full((1, 3), 0.05, dtype=torch.float32, device=device)
        ),
        opacity_logits=torch.full((1,), 6.0, dtype=torch.float32, device=device),
        features=torch.tensor(((1.0, 0.0, 0.0),), dtype=torch.float32, device=device),
        instance_ids=torch.tensor((4,), dtype=torch.int64, device=device),
        coordinates=GaussianCoordinates.scene(),
        appearance_model="rgb",
        appearance_space="linear_rgb",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compositor_downloads_only_compact_visible_delta() -> None:
    compositor = PLCSForegroundCompositor(
        sigma_extent=3.0,
        minimum_pixel_variance=0.25,
        near_plane=1.0e-4,
        visibility_threshold=1.0e-4,
        maximum_alpha=0.999,
    )
    background = _background()
    compositor.prepare_background(background, device="cuda:0")

    delta, visibility = compositor.compose_delta(
        frame_index=7,
        camera=_camera(),
        gaussians_scene=_foreground("cuda:0"),
        expected_instance_ids=(4,),
    )
    logical = materialize_logical_sample(background, delta)

    assert compositor.background_upload_count == 1
    assert 0 < len(delta.pixel_indices) < 17 * 17
    assert visibility == {4: len(delta.pixel_indices)}
    assert set(np.unique(logical.instance_ids)) == {0, 4}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_present_instance_may_be_fully_occluded_in_one_camera() -> None:
    compositor = PLCSForegroundCompositor(
        sigma_extent=3.0,
        minimum_pixel_variance=0.25,
        near_plane=1.0e-4,
        visibility_threshold=1.0e-4,
        maximum_alpha=0.999,
    )
    background = BackgroundArrays(
        camera_id="camera-1",
        rgb=np.zeros((17, 17, 3), dtype=np.float32),
        alpha=np.ones((17, 17, 1), dtype=np.float32),
        depth=np.full((17, 17, 1), 1.0, dtype=np.float32),
    )
    compositor.prepare_background(background, device="cuda:0")

    delta, visibility = compositor.compose_delta(
        frame_index=7,
        camera=_camera(),
        gaussians_scene=_foreground("cuda:0"),
        expected_instance_ids=(4,),
    )

    assert len(delta.pixel_indices) == 0
    assert visibility == {4: 0}


class _CapturingClient:
    def __init__(self) -> None:
        self.request: NHTRenderCommandRequest | None = None

    def render(
        self,
        request: NHTRenderCommandRequest,
        *,
        environment: object = None,
        timeout_seconds: float | None = None,
    ) -> NHTRenderResult:
        del environment, timeout_seconds
        self.request = request
        request.output_directory.mkdir(parents=True)
        assert request.arbitrary_cameras is not None
        records = []
        for camera in request.arbitrary_cameras.cameras:
            root = request.output_directory / camera.camera_id
            root.mkdir()
            rgb = root / "rgb.npy"
            alpha = root / "alpha.npy"
            depth = root / "depth.npy"
            np.save(rgb, np.zeros((camera.height, camera.width, 3), dtype=np.float32))
            np.save(alpha, np.ones((camera.height, camera.width, 1), dtype=np.float32))
            np.save(
                depth, np.full((camera.height, camera.width, 1), 10.0, dtype=np.float32)
            )
            record = NHTRenderRecord(
                camera_id=camera.camera_id,
                request_source="arbitrary",
                width=camera.width,
                height=camera.height,
                rgb_path=rgb,
                rgb_preview_path=root / "rgb.png",
                alpha_path=alpha,
                alpha_preview_path=root / "alpha.png",
                depth_path=depth,
            )
            record._bind_arrays(  # noqa: SLF001
                NHTRenderArrays(
                    rgb=np.zeros((camera.height, camera.width, 3), dtype=np.float32),
                    alpha=np.ones((camera.height, camera.width, 1), dtype=np.float32),
                    depth=np.full(
                        (camera.height, camera.width, 1), 10.0, dtype=np.float32
                    ),
                )
            )
            records.append(record)
        return NHTRenderResult(
            scene_id="B00",
            output_directory=request.output_directory,
            records=tuple(records),
        )


def test_nht_adapter_invokes_once_and_creates_one_shared_store(tmp_path: Path) -> None:
    scene = tmp_path / "reconstruction" / "export" / "scene.json"
    scene.parent.mkdir(parents=True)
    scene.touch()
    staging = tmp_path / "datasets" / "plcs" / "staging"
    staging.mkdir(parents=True)
    metric_camera = _camera(translation=(1.0, 2.0, 3.0))
    similarity = np.eye(4, dtype=np.float64)
    similarity[:3, :3] *= 0.25
    adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(similarity)
    client = _CapturingClient()
    renderer = NHTPLCSRenderer(
        client=cast(NHTRenderClient, client),
        compositor=PLCSForegroundCompositor(
            sigma_extent=3.0,
            minimum_pixel_variance=0.25,
            near_plane=1.0e-4,
            visibility_threshold=1.0e-4,
            maximum_alpha=0.999,
        ),
        executable="nht-render",
        environment={},
        timeout_seconds=60.0,
    )
    session = RenderSession(
        domain="plcs",
        attempt_token="B00-plcs",
        execution_device="cuda:0",
    )

    store = renderer.render_background_store(
        scene_path=scene,
        cameras=(metric_camera,),
        metric_adapter=adapter,
        staging_directory=staging,
        session=session,
    )

    assert client.request is not None
    assert client.request.arbitrary_cameras is not None
    nht_camera = client.request.arbitrary_cameras.cameras[0]
    np.testing.assert_allclose(
        nht_camera.camera_to_scene.matrix()[:3, 3],
        (0.25, 0.5, 0.75),
    )
    assert nht_camera.intrinsics == metric_camera.intrinsics
    assert session.nht_invocations == 1
    assert session.background_cache_misses == 1
    assert store.camera_ids == ("camera-1",)
