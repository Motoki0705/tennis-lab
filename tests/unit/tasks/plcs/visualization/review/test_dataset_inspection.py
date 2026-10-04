"""Observation review must expose missing/malformed data and teacher correspondence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.visualization.review.court import court_keypoints
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.tasks.plcs.visualization.review.dataset_inspection import split_inventory
from src.tasks.plcs.visualization.review.dataset_service import PLCSDatasetReviewService
from src.tasks.plcs.visualization.review.dataset_web import create_dataset_app
from src.utils.projection.camera_projector import (
    Camera,
    make_look_at_camera,
    project_points,
)
from src.utils.schema.court_normalization import normalize_court_position


def _project(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uv, front = project_points(camera, torch.from_numpy(points.reshape(-1, 3)))
    uv = uv.numpy().reshape(*points.shape[:-1], 2) / np.asarray(
        [camera.w, camera.h], dtype=np.float32
    )
    visible = front.numpy().reshape(points.shape[:-1]) & ((uv >= 0) & (uv <= 1)).all(
        axis=-1
    )
    return uv, visible


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, PLCSDatasetReviewService]:
    root = tmp_path / "plcs" / "single_object"
    contract = resolve_court_keypoint_contract("physical_v1")
    writer = PLCSDatasetWriter(root, court_keypoint_contract=contract)
    writer.save_meta_json(
        config={
            "generation": {"mode": "single_object"},
            "court_keypoints": {"selector": "physical_v1"},
        }
    )
    camera = make_look_at_camera(
        (0, -18, 5), look_at=(0, 0, 0), image_size=(1280, 720), hfov_deg=60
    )
    world: np.ndarray = np.zeros((2, 17, 3), dtype=np.float32)
    world[..., 2] = 1.0
    world[1, :, 0] = 100.0  # A genuine out-of-frame observation in the test input.
    uv, visible = _project(camera, world)
    court_uv, court_vis = _project(camera, court_keypoints(None))
    camera_data = CameraData(
        camera_params={
            "C": camera.C.tolist(),
            "R": camera.R.tolist(),
            "f": camera.f,
            "cx": camera.cx,
            "cy": camera.cy,
            "w": camera.w,
            "h": camera.h,
        },
        human_kp_uv=uv,
        human_kp_vis=visible,
        court_kp_uv=np.repeat(court_uv[None], 2, axis=0),
        court_kp_vis=np.repeat(court_vis[None], 2, axis=0),
        human_visibility_ratio=0.5,
        court_visibility_count=float(court_vis.sum()),
        court_view=build_court_view_record(
            camera_id="camera_0",
            camera_center_court_m=camera.C.tolist(),
            contract=contract,
        ),
    )
    scene = SceneData(
        meta={
            "scene_id": "scene_000000",
            "motion_source": "source-a.npz",
            "motion_category": "walking",
            "gender": "neutral",
            "fps": 30,
            "num_frames": 2,
            "initial_position": [1, 2],
            "initial_yaw": 0,
            "num_cameras_sampled": 1,
        },
        position=normalize_court_position(
            np.array([[1, 2, 1], [2, 3, 1]], dtype=np.float32)
        ),
        rotation=np.array([[1, 0], [0, 1]], dtype=np.float32),
        canonical_pose_3d=world,
        cameras=[camera_data],
        num_persons=1,
        human_kp_3d=world,
        court_keypoint_contract=contract,
    )
    writer.save_scene(scene)
    scene.meta["scene_id"] = "scene_000001"
    writer.save_scene(scene)
    writer.save_meta_json(
        config={
            "generation": {"mode": "single_object"},
            "court_keypoints": {"selector": "physical_v1"},
        }
    )
    (root / "train.txt").write_text("scene_000000\n", encoding="utf-8")
    (root / "val.txt").write_text("scene_000001\n", encoding="utf-8")
    (root / "test.txt").write_text("", encoding="utf-8")
    return root, PLCSDatasetReviewService(tmp_path)


def _query(client: TestClient) -> dict[str, str]:
    scene = client.get("/api/scene?form=single_object&scene=scene_000000").json()
    assert "revision" in scene, scene
    return {
        "form": "single_object",
        "scene": "scene_000000",
        "revision": scene["revision"],
    }


def _field(buffer: bytes, document: dict[str, Any], name: str) -> np.ndarray:
    field = document["buffer_fields"][name]
    shape = tuple(int(value) for value in field["shape"])
    values: np.ndarray = np.frombuffer(
        buffer,
        dtype="<f4",
        offset=int(field["byte_offset"]),
        count=int(np.prod(field["shape"])),
    )
    return values.reshape(shape)


def test_buffer_reproduces_saved_observations_and_restores_root_units(
    dataset: tuple[Path, PLCSDatasetReviewService],
) -> None:
    root, service = dataset
    client = TestClient(create_dataset_app(service))
    query = _query(client)
    document = client.get("/api/scene/inspection", params=query).json()
    buffer = client.get("/api/scene/observations", params=query).content
    assert len(buffer) == document["byte_length"]
    np.testing.assert_allclose(
        _field(buffer, document, "root_m"), [[1, 2, 1], [2, 3, 1]], atol=1e-6
    )
    np.testing.assert_array_equal(
        _field(buffer, document, "cam_0_human_uv"),
        np.load(root / "scenes/scene_000000/cam_0_human_kp_uv.npy"),
    )
    assert _field(buffer, document, "cam_0_human_vis")[1].sum() == 0
    assert document["cameras"][0]["human"]["empty_frames"] == 1
    assert document["cameras"][0]["human"]["max_error_px"] == pytest.approx(0)
    assert document["cameras"][0]["human"]["visibility_mismatch_count"] == 0
    assert document["source_splits"] == ["train", "val"]
    assert document["splits"] == ["train"]
    assert document["canonical_shape"] == [2, 17, 3]
    assert document["rgb_available"] is False


def test_saved_uv_and_visibility_corruption_are_reported(
    dataset: tuple[Path, PLCSDatasetReviewService],
) -> None:
    root, service = dataset
    scene = root / "scenes/scene_000000"
    uv = np.load(scene / "cam_0_human_kp_uv.npy")
    uv[0, 0, 0] += 0.01
    np.save(scene / "cam_0_human_kp_uv.npy", uv)
    visible = np.load(scene / "cam_0_human_kp_vis.npy")
    visible[1, 0] = True
    np.save(scene / "cam_0_human_kp_vis.npy", visible)
    client = TestClient(create_dataset_app(service))
    document = client.get("/api/scene/inspection", params=_query(client)).json()
    report = document["cameras"][0]["human"]
    assert report["max_error_px"] == pytest.approx(12.8, abs=1e-3)
    assert report["visibility_mismatch_count"] == 1
    assert report["visible_outside_count"] == 1


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "shape",
        "nonfinite",
        "invalid_visibility",
        "heading",
        "canonical",
        "source",
    ],
)
def test_input_errors_are_explicit(
    dataset: tuple[Path, PLCSDatasetReviewService], problem: str
) -> None:
    root, service = dataset
    scene = root / "scenes/scene_000000"
    uv_path = scene / "cam_0_human_kp_uv.npy"
    if problem == "missing":
        uv_path.unlink()
    elif problem == "shape":
        np.save(uv_path, np.zeros((1, 17, 2), dtype=np.float32))
    elif problem == "nonfinite":
        uv = np.load(uv_path)
        uv[0, 0, 0] = np.nan
        np.save(uv_path, uv)
    elif problem == "invalid_visibility":
        np.save(scene / "cam_0_human_kp_vis.npy", np.full((2, 17), 2, dtype=np.int8))
    elif problem == "heading":
        np.save(scene / "rotation.npy", np.zeros((2, 2), dtype=np.float32))
    elif problem == "canonical":
        np.save(scene / "canonical_pose_3d.npy", np.zeros((2, 17, 3), dtype=np.int32))
    else:
        meta_path = scene / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["motion_source"] = {"unknown": True}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    client = TestClient(create_dataset_app(service))
    result = client.get("/api/scene/inspection", params=_query(client))
    assert result.status_code == (404 if problem == "missing" else 422)


def test_optional_canonical_absence_is_reported_without_replacing_world_teacher(
    dataset: tuple[Path, PLCSDatasetReviewService],
) -> None:
    root, service = dataset
    (root / "scenes/scene_000000/canonical_pose_3d.npy").unlink()
    client = TestClient(create_dataset_app(service))
    document = client.get("/api/scene/inspection", params=_query(client)).json()
    assert document["canonical_shape"] is None
    assert document["cameras"][0]["human"]["max_error_px"] == pytest.approx(0)


def test_stale_revision_never_mixes_teacher_and_observations(
    dataset: tuple[Path, PLCSDatasetReviewService],
) -> None:
    root, service = dataset
    client = TestClient(create_dataset_app(service))
    query = _query(client)
    np.save(
        root / "scenes/scene_000000/cam_0_human_kp_uv.npy",
        np.zeros((2, 17, 2), dtype=np.float32),
    )
    assert client.get("/api/scene/inspection", params=query).status_code == 409
    assert client.get("/api/scene/observations", params=query).status_code == 409


def test_split_integrity_reports_unknown_duplicate_and_unassigned_scenes(
    dataset: tuple[Path, PLCSDatasetReviewService],
) -> None:
    root, _ = dataset
    (root / "train.txt").write_text(
        "scene_000000\nscene_000000\nghost\n", encoding="utf-8"
    )
    (root / "val.txt").unlink()
    report = split_inventory(root)
    assert report["missing_files"] == ["val.txt"]
    assert report["unknown_scenes"] == ["train:ghost"]
    assert report["duplicate_entries"] == ["train:scene_000000"]
    assert report["unassigned_scenes"] == ["scene_000001"]
