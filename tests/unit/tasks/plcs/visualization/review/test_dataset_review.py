"""PLCS review data must reproduce the stored 2D projections for all forms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.tasks.base.visualization.review.camera import parse_cameras
from src.tasks.base.visualization.review.court import court_keypoints
from src.tasks.plcs.visualization.review.dataset_service import (
    PLCSDatasetReviewService,
)
from src.tasks.plcs.visualization.review.dataset_web import create_dataset_app
from src.utils.projection.camera_projector import project_points

DATA_ROOT = Path("/home/kamimura/projects/tennis-lab/data")
SCENE = "scene_000000"
FORMS = (
    "single_object",
    "multi_object",
    "single_object_broadcast",
    "multi_object_broadcast",
    "multi_object_camera_view_v2",
    "single_object_camera_view_v2",
)

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "plcs").is_dir(), reason="PLCS dataset is unavailable"
)


def scene_dir(form: str) -> Path:
    return DATA_ROOT / "plcs" / form / "scenes" / SCENE


def load_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def project(record: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = record.to_camera()
    uv, in_front = project_points(
        camera, torch.as_tensor(np.asarray(points), dtype=torch.float32)
    )
    uv = (uv / torch.tensor([camera.w, camera.h], dtype=torch.float32)).numpy()
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)
    )
    return uv, (in_front.numpy() & in_bounds)


@pytest.mark.parametrize("form", FORMS)
def test_court_keypoints_reproject_exactly(form: str) -> None:
    directory = scene_dir(form)
    meta = load_json(directory / "meta.json")
    cameras = parse_cameras(load_json(directory / "scalars.json"))
    keypoints = court_keypoints(None)
    for index, record in enumerate(cameras):
        uv, visible = project(record, keypoints)
        # PLCS stores the (static) court projection once per frame.
        saved = np.load(directory / f"cam_{index}_court_kp_uv.npy")[0]
        saved_vis = np.load(directory / f"cam_{index}_court_kp_vis.npy")[0]
        order = np.asarray(
            meta["court_keypoint_views"][index]["semantic_to_physical"]
        )
        physical = np.empty_like(uv)
        physical[order] = saved
        physical_vis = np.empty_like(visible)
        physical_vis[order] = saved_vis
        assert np.abs(physical - uv).max() < 1e-5
        assert np.array_equal(physical_vis, visible)


@pytest.mark.parametrize("form", FORMS)
def test_human_keypoints_reproject_exactly(form: str) -> None:
    directory = scene_dir(form)
    cameras = parse_cameras(load_json(directory / "scalars.json"))
    joints = np.load(directory / "human_kp_3d.npy")
    multi = joints.ndim == 4
    presence = np.load(directory / "person_present.npy") if multi else None
    points = joints.reshape(-1, 3)
    for index, record in enumerate(cameras):
        uv, visible = project(record, points)
        saved = np.load(directory / f"cam_{index}_human_kp_uv.npy")
        saved_vis = np.load(directory / f"cam_{index}_human_kp_vis.npy")
        uv = uv.reshape(saved.shape)
        vis = visible.reshape(saved_vis.shape)
        if presence is not None:
            vis = vis & presence[..., None]
        assert np.array_equal(vis, saved_vis)
        assert np.abs(uv[vis] - saved[vis]).max() < 1e-5


def test_api_shapes_and_errors() -> None:
    service = PLCSDatasetReviewService(
        DATA_ROOT, forms=["single_object", "multi_object"]
    )
    client = TestClient(create_dataset_app(service))

    catalog = client.get("/api/catalog").json()
    assert catalog["task"] == "plcs"
    assert catalog["entity"] == "player"
    assert catalog["skeleton"]["names"][0] == "nose"
    assert len(catalog["skeleton"]["edges"]) == 16

    document = client.get(
        "/api/scene", params={"form": "single_object", "scene": SCENE}
    ).json()
    assert document["entity"]["kind"] == "player"
    assert document["entity"]["joint_count"] == 17
    assert document["entity"]["orientation"] is True
    assert document["entity"]["presence"] is False
    assert document["fps"] == pytest.approx(120.0)

    frames = document["entity"]["frames"]
    buffer = client.get(
        "/api/scene/buffer",
        params={"form": "single_object", "scene": SCENE, "revision": document["revision"]},
    ).content
    assert len(buffer) == frames * 17 * 3 * 4 + frames * 2 * 4

    multi = client.get(
        "/api/scene", params={"form": "multi_object", "scene": SCENE}
    ).json()
    slots = multi["entity"]["slots"]
    frames = multi["entity"]["frames"]
    assert slots == 10
    assert multi["entity"]["presence"] is True
    multi_buffer = client.get(
        "/api/scene/buffer",
        params={"form": "multi_object", "scene": SCENE, "revision": multi["revision"]},
    ).content
    assert len(multi_buffer) == (
        slots * frames * 17 * 3 * 4 + slots * frames * 2 * 4 + slots * frames
    )

    stale = client.get(
        "/api/scene",
        params={"form": "single_object", "scene": SCENE, "revision": "0" * 20},
    )
    assert stale.status_code == 409
    escaped = client.get(
        "/api/scene", params={"form": "single_object", "scene": "../escape"}
    )
    assert escaped.status_code in {404, 422}
