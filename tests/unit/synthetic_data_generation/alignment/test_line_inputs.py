"""Tests for explicit captured and NHT-rendered alignment line inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from src.synthetic_data_generation.alignment.line_inputs import (
    CourtLineInputSource,
    NHTRenderedAlignmentLineInputSource,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCommandRequest,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


@dataclass
class _RenderClient:
    color: int = 173
    error: Exception | None = None
    request: NHTRenderCommandRequest | None = None
    environment: dict[str, str] | None = None
    timeout_seconds: float | None = None

    def render(
        self,
        request: NHTRenderCommandRequest,
        *,
        environment: dict[str, str],
        timeout_seconds: float,
    ) -> Any:
        self.request = request
        self.environment = dict(environment)
        self.timeout_seconds = timeout_seconds
        if self.error is not None:
            raise self.error
        records = []
        for camera_id in request.observed_camera_ids:
            output = request.output_directory / camera_id
            output.mkdir(parents=True)
            preview = output / "rgb.png"
            Image.fromarray(
                np.full((4, 6, 3), self.color, dtype=np.uint8),
                mode="RGB",
            ).save(preview)
            records.append(
                SimpleNamespace(
                    camera_id=camera_id,
                    request_source="observed",
                    width=6,
                    height=4,
                    rgb_preview_path=preview,
                )
            )
        return SimpleNamespace(scene_id="scene-a", records=tuple(records))


def test_nht_rendered_inputs_use_observed_cameras_and_public_rgb(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    client = _RenderClient()
    source = NHTRenderedAlignmentLineInputSource(
        client=cast(Any, client),
        executable=_executable(tmp_path),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        timeout_seconds=19.0,
    )

    batch = source.load(scene, scene.cameras)

    assert batch.source is CourtLineInputSource.NHT_RENDERED_RGB
    assert batch.camera_ids == ("camera-0", "camera-1")
    assert all(np.all(view.image_rgb == 173) for view in batch.views)
    assert all(not view.image_rgb.flags.writeable for view in batch.views)
    assert client.request is not None
    assert client.request.observed_camera_ids == batch.camera_ids
    assert client.request.arbitrary_cameras is None
    assert client.environment == {"CUDA_VISIBLE_DEVICES": "0"}
    assert client.timeout_seconds == 19.0
    identity = batch.cache_identity()
    assert identity["source"] == "nht_rendered_rgb"
    provenance = cast(dict[str, Any], identity["provenance"])
    assert provenance["camera_ids"] == ["camera-0", "camera-1"]
    assert cast(dict[str, Any], provenance["checkpoint"])["sha256"]
    assert cast(dict[str, Any], provenance["renderer"])["request_source"] == (
        "observed"
    )


def test_nht_render_failure_propagates_without_captured_rgb_fallback(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    source = NHTRenderedAlignmentLineInputSource(
        client=cast(Any, _RenderClient(error=RuntimeError("render failed"))),
        executable=_executable(tmp_path),
        environment={},
        timeout_seconds=10.0,
    )

    with pytest.raises(RuntimeError, match="render failed"):
        source.load(scene, scene.cameras)


def test_nht_render_source_rejects_camera_geometry_not_owned_by_scene(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    foreign = replace(
        scene.cameras[0],
        camera_to_scene=RigidTransform.from_matrix(
            np.asarray(
                (
                    (1.0, 0.0, 0.0, 1.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            )
        ),
    )
    source = NHTRenderedAlignmentLineInputSource(
        client=cast(Any, _RenderClient()),
        executable=_executable(tmp_path),
        environment={},
        timeout_seconds=10.0,
    )

    with pytest.raises(ValueError, match="exact StandardSceneExport record"):
        source.preflight(scene, (foreign, scene.cameras[1]))


def _scene(tmp_path: Path) -> StandardSceneExport:
    export = (tmp_path / "reconstruction/export").resolve()
    images = export / "images"
    model = export / "model"
    images.mkdir(parents=True)
    model.mkdir()
    scene_path = export / "scene.json"
    scene_path.write_text('{"schema":"test"}\n', encoding="utf-8")
    (export / "cameras.json").write_text('{"schema":"test"}\n', encoding="utf-8")
    checkpoint = model / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    runtime_config = model / "runtime-config.json"
    runtime_config.write_text('{"schema":"test"}\n', encoding="utf-8")
    cameras = []
    for index in range(2):
        image_path = images / f"camera-{index}.png"
        Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), mode="RGB").save(
            image_path
        )
        cameras.append(
            SceneCamera(
                camera_id=f"camera-{index}",
                source_frame_index=index,
                width=6,
                height=4,
                intrinsics=(4.0, 0.0, 3.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0),
                camera_to_scene=RigidTransform.identity(),
                image_path=str(image_path),
            )
        )
    return StandardSceneExport(
        scene_id="scene-a",
        export_root=export,
        scene_path=scene_path,
        cameras=tuple(cameras),
        points_scene=np.zeros((4, 6), dtype=np.float32),
        scene_from_sfm=tuple(float(value) for value in np.eye(4).ravel()),
        sfm_from_scene=tuple(float(value) for value in np.eye(4).ravel()),
        checkpoint_path=checkpoint,
        runtime_config_path=runtime_config,
    )


def _executable(tmp_path: Path) -> Path:
    path = (tmp_path / "bin/nht-render").resolve()
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path
