"""SLCS review preserves canonical pseudo labels and rejects stale artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.tasks.slcs.data.annotation import load_slcs_annotation, slcs_annotation_dir
from src.tasks.slcs.visualization.review.dataset_service import SLCSDatasetReviewService
from src.tasks.slcs.visualization.review.dataset_web import create_dataset_app
from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


@pytest.fixture(scope="module")
def source_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = Path(tmp_path_factory.mktemp("slcs-review-source"))
    index = build_slcs_dataset_fixture(
        root,
        SLCSFixtureDatasetConfig(
            videos=("video_000", "video_001"),
            num_frames=5,
            num_cameras=3,
        ),
    )
    for record in index.clips:
        manifest = ClipManifest.load(index.clip_dir(record))
        scene = load_slcs_annotation(manifest)
        # Save far first, near second, and leave explicit quality gaps.
        scene.player_position[0] = [2, 6, 0.8]
        scene.player_position[1] = [-1, -7, 0.9]
        scene.player_yaw[0] = np.pi / 2
        scene.player_yaw[1] = np.pi
        assert scene.human_kp_vis is not None
        scene.human_kp_vis[:] = 1
        scene.human_kp_vis[1, :, 1] = 0
        scene.player_position[1, 1] = np.nan
        scene.player_yaw[1, 1] = np.nan
        assert scene.ball_vis is not None
        scene.ball_vis[:] = True
        scene.ball_vis[:, 2] = False
        assert scene.ball_3d is not None
        scene.ball_3d[:] = [1, 3, 2]
        scene.ball_3d[2] = np.nan
        save_scene_result(scene, slcs_annotation_dir(manifest.clip_dir) / "scene.npz")
        # DINO and split artifacts must not be prerequisites for a review.
        shutil.rmtree(manifest.clip_dir / "annotations" / "dino_v3")
    return root


@pytest.fixture
def dataset(source_dataset: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(source_dataset, tmp_path / "dataset"))


def clip_dir(root: Path) -> Path:
    return root / "videos" / "video_000" / "clips" / "clip_000"


def test_catalog_is_lazy_and_groups_manifest_clips_by_video(dataset: Path) -> None:
    slcs_annotation_dir(clip_dir(dataset)).joinpath("scene.npz").unlink()
    service = SLCSDatasetReviewService(dataset)
    catalog = service.catalog()
    assert catalog["task"] == "slcs"
    assert [f["name"] for f in catalog["forms"]] == ["video_000", "video_001"]
    assert [f["scene_count"] for f in catalog["forms"]] == [1, 1]
    assert service.scenes("video_000")["scenes"] == ["clip_000"]
    selected = SLCSDatasetReviewService(dataset, video_ids=["video_001"])
    assert len(selected.catalog()["forms"]) == 1
    with pytest.raises(ValueError, match="Unknown video"):
        selected.scenes("video_000")
    with pytest.raises(ValueError, match="Unknown video"):
        SLCSDatasetReviewService(dataset, video_ids=["unknown"])


def test_mixed_payload_keeps_metres_yaw_order_and_label_gaps(dataset: Path) -> None:
    service = SLCSDatasetReviewService(dataset)
    payload = service.payload("video_000", "clip_000")
    document, data = payload.document, payload.binary
    assert document["fps"] == 30
    assert document["frame_count"] == 5
    assert document["units"] == "m"
    assert document["cameras"] == []
    assert document["source_camera_ids"] == ["cam0", "cam1", "cam2"]
    assert [entity["kind"] for entity in document["entities"]] == ["player", "ball"]
    assert document["quality"]["player_valid_frames"] == [4, 5]
    assert document["quality"]["ball_valid_frames"] == 4
    positions = np.frombuffer(data, dtype="<f4", count=30).reshape(2, 5, 3)
    np.testing.assert_allclose(positions[0, 0], [-1, -7, 0.9], atol=1e-6)
    np.testing.assert_allclose(positions[1, 0], [2, 6, 0.8], atol=1e-6)
    assert (positions[0, 1] == 0).all()
    heading = np.frombuffer(data, dtype="<f4", count=20, offset=120).reshape(2, 5, 2)
    np.testing.assert_allclose(heading[:, 0], [[-1, 0], [0, 1]], atol=1e-6)
    np.testing.assert_array_equal(heading[0, 1], [0, 0])
    presence = np.frombuffer(data, dtype=np.uint8, count=10, offset=200).reshape(2, 5)
    np.testing.assert_array_equal(presence[0], [1, 0, 1, 1, 1])
    # Ten mask bytes require two padding bytes before the float32 ball block.
    assert data[210:212] == b"\x00\x00"
    balls = np.frombuffer(data, dtype="<f4", count=15, offset=212).reshape(5, 3)
    np.testing.assert_allclose(balls[0], [1, 3, 2], atol=1e-6)
    np.testing.assert_array_equal(balls[2], [0, 0, 0])
    assert list(data[272:]) == [1, 1, 0, 1, 1]


def test_api_assets_and_read_only_errors(dataset: Path) -> None:
    client = TestClient(create_dataset_app(SLCSDatasetReviewService(dataset)))
    params = {"form": "video_000", "scene": "clip_000"}
    assert client.get("/api/catalog").status_code == 200
    document = client.get("/api/scene", params=params)
    assert document.status_code == 200
    response = client.get(
        "/api/scene/buffer", params={**params, "revision": document.json()["revision"]}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-task"] == "slcs"
    for asset in ("/", "/static/model.mjs", "/static/app.js", "/shared/scene3d.mjs"):
        assert client.get(asset).status_code == 200
    assert client.get("/static/dataset_service.py").status_code == 404
    assert client.post("/api/scene", params=params).status_code == 405
    assert (
        client.get("/api/scene", params={**params, "scene": "../escape"}).status_code
        == 422
    )
    assert client.get("/api/scenes", params={"form": "unknown"}).status_code == 422
    assert (
        client.get(
            "/api/scene/buffer", params={**params, "revision": "stale"}
        ).status_code
        == 409
    )


@pytest.mark.parametrize(
    "name", ["annotation.json", "scene.npz", "scene.metadata.json"]
)
def test_missing_annotation_artifacts_are_not_silently_accepted(
    dataset: Path, name: str
) -> None:
    slcs_annotation_dir(clip_dir(dataset)).joinpath(name).unlink()
    client = TestClient(create_dataset_app(SLCSDatasetReviewService(dataset)))
    assert (
        client.get(
            "/api/scene", params={"form": "video_000", "scene": "clip_000"}
        ).status_code
        == 404
    )


def test_revision_covers_sidecar_and_rejects_stale_cached_buffer(dataset: Path) -> None:
    service = SLCSDatasetReviewService(dataset)
    first = service.scene("video_000", "clip_000")
    sidecar = slcs_annotation_dir(clip_dir(dataset)) / "scene.metadata.json"
    sidecar.write_text(sidecar.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        service.buffer("video_000", "clip_000", first["revision"])
    assert service.scene("video_000", "clip_000")["revision"] != first["revision"]


def test_manifest_digest_and_marker_shapes_are_validated(dataset: Path) -> None:
    marker = slcs_annotation_dir(clip_dir(dataset)) / "annotation.json"
    original = json.loads(marker.read_text())
    modified = {**original, "clip_manifest_sha256": "bad-digest"}
    marker.write_text(json.dumps(modified))
    service = SLCSDatasetReviewService(dataset)
    with pytest.raises(ValueError, match="different clip.json"):
        service.scene("video_000", "clip_000")
    original["arrays"]["ball_3d"]["shape"] = [5, 99]
    marker.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="marker-recorded shape"):
        service.scene("video_000", "clip_000")


def test_symlink_escape_is_rejected(dataset: Path, tmp_path: Path) -> None:
    archive = slcs_annotation_dir(clip_dir(dataset)) / "scene.npz"
    outside = tmp_path / "outside.npz"
    archive.rename(outside)
    archive.symlink_to(outside)
    service = SLCSDatasetReviewService(dataset)
    with pytest.raises(ValueError, match="escapes"):
        service.scene("video_000", "clip_000")
