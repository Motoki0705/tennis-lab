from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter, load_scene
from src.plcs.generate_dataset.scene_generator import CameraData, SceneData

TCallable = TypeVar("TCallable", bound=Callable[..., object])


def _typed_mark(mark: Any) -> Callable[[TCallable], TCallable]:
    return cast(Callable[[TCallable], TCallable], mark)


def _make_dummy_scene(*, scene_id: str = "scene_000001") -> SceneData:
    num_frames = 2
    num_joints = 5
    num_cameras = 2

    meta = {
        "scene_id": scene_id,
        "motion_source": "unit_test",
        "motion_category": "serve",
        "gender": "neutral",
        "fps": 30,
        "num_frames": num_frames,
        "initial_position": [0.1, -0.2],
        "initial_yaw": 0.0,
        "num_cameras_sampled": num_cameras,
    }

    position = np.arange(num_frames * 3, dtype=np.float32).reshape(num_frames, 3) / 10
    rotation = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    canonical_pose_3d = (
        np.arange(num_frames * num_joints * 3, dtype=np.float32).reshape(
            num_frames, num_joints, 3
        )
        / 100
    )

    cameras: list[CameraData] = []
    for camera_idx in range(num_cameras):
        camera_params = {
            "camera_idx": camera_idx,
            "K": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            "image_size": [1280, 720],
        }
        human_kp_uv: NDArray[np.float32] = (
            np.arange(num_frames * 17 * 2, dtype=np.float32).reshape(num_frames, 17, 2)
            + camera_idx
        )
        court_kp_uv: NDArray[np.float32] = (
            np.arange(num_frames * 20 * 2, dtype=np.float32).reshape(num_frames, 20, 2)
            + camera_idx
        )
        human_kp_visible: NDArray[np.bool_] = np.ones((num_frames, 17), dtype=bool)
        court_kp_visible: NDArray[np.bool_] = np.ones((num_frames, 20), dtype=bool)

        cameras.append(
            CameraData(
                camera_params=camera_params,
                human_kp_uv=human_kp_uv,
                court_kp_uv=court_kp_uv,
                human_kp_visible=human_kp_visible,
                court_kp_visible=court_kp_visible,
                human_visibility_ratio=1.0,
                court_visibility_count=20.0,
            )
        )

    return SceneData(
        meta=meta,
        position=position,
        rotation=rotation,
        canonical_pose_3d=canonical_pose_3d,
        cameras=cameras,
    )


@_typed_mark(pytest.mark.unit)
def test_dataset_io_save_scene_and_load_scene_roundtrip(tmp_path: Path) -> None:
    writer = PLCSDatasetWriter(output_dir=tmp_path)
    scene = _make_dummy_scene(scene_id="scene_000001")

    scene_path = writer.save_scene(scene)
    assert scene_path.exists()
    assert scene_path.name == "scene_000001.npz"

    loaded = load_scene(scene_path)
    assert isinstance(loaded["meta"], dict)
    assert loaded["meta"]["scene_id"] == "scene_000001"
    assert loaded["num_cameras"] == len(scene.cameras)
    assert len(loaded["cameras"]) == len(scene.cameras)

    np.testing.assert_allclose(loaded["position"], scene.position)
    np.testing.assert_allclose(loaded["rotation"], scene.rotation)
    np.testing.assert_allclose(loaded["canonical_pose_3d"], scene.canonical_pose_3d)

    for cam_idx, (loaded_cam, original_cam) in enumerate(
        zip(loaded["cameras"], scene.cameras, strict=True)
    ):
        assert isinstance(loaded_cam["params"], dict)
        assert loaded_cam["params"]["camera_idx"] == cam_idx
        np.testing.assert_allclose(loaded_cam["human_kp_uv"], original_cam.human_kp_uv)
        np.testing.assert_array_equal(
            loaded_cam["human_kp_visible"], original_cam.human_kp_visible
        )
        np.testing.assert_allclose(loaded_cam["court_kp_uv"], original_cam.court_kp_uv)
        np.testing.assert_array_equal(
            loaded_cam["court_kp_visible"], original_cam.court_kp_visible
        )
        assert loaded_cam["human_visibility_ratio"] == pytest.approx(
            original_cam.human_visibility_ratio
        )
        assert loaded_cam["court_visibility_count"] == pytest.approx(
            original_cam.court_visibility_count
        )
