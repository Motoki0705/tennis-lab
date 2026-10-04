"""SLCS review preserves canonical pseudo labels and rejects stale artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.tasks.slcs.data.annotation import load_slcs_annotation, slcs_annotation_dir
from src.tasks.slcs.visualization.review.dataset_service import SLCSDatasetReviewService
from src.tasks.slcs.visualization.review.dataset_web import create_dataset_app
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.schema import SceneResult
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)
from tests.support.tennis_scene.annotations import publish_dataset_annotations


def _review_scene(scene: SceneResult) -> SceneResult:
    """Far player first, near second, with explicit v2 gaps (invalid mask + zero)."""
    assert scene.player_valid is not None and scene.player_heading_valid is not None
    assert scene.player_observed is not None and scene.player_rejection_code is not None
    scene.player_position[0] = [2, 6, 0.8]
    scene.player_position[1] = [-1, -7, 0.9]
    scene.player_yaw[0] = np.pi / 2
    scene.player_yaw[1] = np.pi
    assert scene.human_kp_vis is not None
    scene.human_kp_vis[:] = 1
    scene.human_kp_vis[1, :, 1] = 0
    scene.player_position[1, 1] = 0
    scene.player_yaw[1, 1] = 0
    scene.player_valid[1, 1] = scene.player_heading_valid[1, 1] = False
    scene.player_rejection_code[1, 1] = 1
    assert scene.ball_vis is not None and scene.ball_3d is not None
    assert scene.ball_3d_valid is not None and scene.ball_rejection_code is not None
    scene.ball_vis[:] = True
    scene.ball_vis[:, 2] = False
    scene.ball_3d[:] = [1, 3, 2]
    scene.ball_3d[2] = 0
    scene.ball_3d_valid[:] = True
    scene.ball_3d_valid[2] = False
    scene.ball_rejection_code[:] = (~scene.ball_3d_valid).astype(np.uint8)
    return scene


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
    scenes = {}
    for record in index.clips:
        manifest = ClipManifest.load(index.clip_dir(record))
        scenes[manifest.clip_id] = _review_scene(load_slcs_annotation(manifest))
        # DINO and split artifacts must not be prerequisites for a review.
        shutil.rmtree(manifest.clip_dir / "annotations" / "dino_v3")
    publish_dataset_annotations(index.root, scenes, overwrite=True)
    return root


@pytest.fixture
def dataset(source_dataset: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(source_dataset, tmp_path / "dataset"))


def clip_dir(root: Path) -> Path:
    return root / "videos" / "video_000" / "clips" / "clip_000"


def published_scene(root: Path) -> Path:
    """The immutable export the clip's marker publishes."""
    annotation = slcs_annotation_dir(clip_dir(root))
    marker = json.loads((annotation / "annotation.json").read_text())
    return Path(annotation / marker["scene_result"])


def test_catalog_is_lazy_and_groups_manifest_clips_by_video(dataset: Path) -> None:
    published_scene(dataset).unlink()
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
    ("name", "status"),
    # No marker: nothing is published (404). A file the scene index records
    # is missing: the publication is broken (422 from the checksum check).
    [("annotation.json", 404), ("scene.npz", 422), ("scene.metadata.json", 422)],
)
def test_missing_annotation_artifacts_are_not_silently_accepted(
    dataset: Path, name: str, status: int
) -> None:
    folder = (
        slcs_annotation_dir(clip_dir(dataset))
        if name == "annotation.json"
        else published_scene(dataset).parent
    )
    folder.joinpath(name).unlink()
    client = TestClient(create_dataset_app(SLCSDatasetReviewService(dataset)))
    assert (
        client.get(
            "/api/scene", params={"form": "video_000", "scene": "clip_000"}
        ).status_code
        == status
    )


def test_revision_follows_republication_and_rejects_stale_cached_buffer(
    dataset: Path,
) -> None:
    service = SLCSDatasetReviewService(dataset)
    first = service.scene("video_000", "clip_000")
    manifest = ClipManifest.load(clip_dir(dataset))
    scene = load_slcs_annotation(manifest)
    scene.player_yaw[0, 0] = 0.5
    publish_dataset_annotations(dataset, {manifest.clip_id: scene}, overwrite=True)
    with pytest.raises(RuntimeError, match="changed"):
        service.buffer("video_000", "clip_000", first["revision"])
    assert service.scene("video_000", "clip_000")["revision"] != first["revision"]


def test_tampered_export_is_rejected(dataset: Path) -> None:
    sidecar = published_scene(dataset).with_suffix(".metadata.json")
    sidecar.write_text(sidecar.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        SLCSDatasetReviewService(dataset).scene("video_000", "clip_000")


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
    archive = published_scene(dataset)
    outside = tmp_path / "outside.npz"
    archive.rename(outside)
    archive.symlink_to(outside)
    service = SLCSDatasetReviewService(dataset)
    with pytest.raises(ValueError, match="escapes"):
        service.scene("video_000", "clip_000")


def test_inspection_keeps_source_order_and_separates_observation_from_teacher(
    dataset: Path,
) -> None:
    ClipManifest.load(clip_dir(dataset)).media_path("cam0").unlink()
    service = SLCSDatasetReviewService(dataset)
    scene = service.scene("video_000", "clip_000")
    inspection = scene["inspection"]
    assert inspection["is_ground_truth"] is False
    assert inspection["player_source_slots"] == [1, 0]
    assert inspection["calibration_status"] == "unavailable"
    assert inspection["source_cameras"][0]["media_available"] is False
    frame = service.inspection_frame(
        "video_000", "clip_000", "cam2", 1, scene["revision"]
    )
    assert frame["camera_id"] == "cam2" and frame["frame"] == 1
    near, far = frame["players"]
    assert near["source_slot"] == 1 and far["source_slot"] == 0
    assert near["position_m"] is None and near["yaw_rad"] is None
    assert near["label_valid"] is False and near["weight"] == 0
    assert near["pose_uv"] == [None] * 17
    assert far["label_valid"] is True
    np.testing.assert_allclose(far["position_m"], [2, 6, 0.8], atol=1e-6)
    assert far["projection"] == {"state": "calibration_unavailable", "uv": None}
    assert frame["ball"]["label_valid"] is True
    frame = service.inspection_frame(
        "video_000", "clip_000", "cam0", 2, scene["revision"]
    )
    assert frame["ball"]["observation_state"] == "unobserved"
    assert frame["ball"]["uv"] is None and frame["ball"]["position_m"] is None


def test_saved_pinhole_projects_height_and_invalid_teacher_never_projects(
    dataset: Path,
) -> None:
    manifest = ClipManifest.load(clip_dir(dataset))
    raw = load_slcs_annotation(manifest)
    reference = raw.metadata["court_reference"]
    reference["camera_fits"] = [
        {
            "K": [[100, 0, 100], [0, 100, 50], [0, 0, 1]],
            "R": np.eye(3).tolist(),
            "t": [0, 0, 10],
            "calibration": "approximate single-plane pinhole; no distortion correction",
        }
        for _ in manifest.camera_ids
    ]
    # A visible 2D observation must not re-enable a hard-rejected 3D teacher.
    assert raw.ball_vis is not None
    raw.ball_vis[:, 2] = True
    raw.ball_uv[0, 0] = [1.1, 0.5]
    publish_dataset_annotations(dataset, {manifest.clip_id: raw}, overwrite=True)
    service = SLCSDatasetReviewService(dataset)
    scene = service.scene("video_000", "clip_000")
    assert scene["inspection"]["calibration_status"] == "saved_pinhole"
    result = service.inspection_frame(
        "video_000", "clip_000", "cam0", 0, scene["revision"]
    )
    # Z=2 participates in K[R|t] projection. A ground-only homography gives a different denominator.
    np.testing.assert_allclose(
        result["ball"]["projection"]["uv"],
        [108.3333333 / manifest.width, 75 / manifest.height],
    )
    assert result["ball"]["observation_state"] == "out_of_frame"
    rejected = service.inspection_frame(
        "video_000", "clip_000", "cam0", 2, scene["revision"]
    )["ball"]
    assert rejected["observed_cameras"] == 3
    assert rejected["label_valid"] is False
    assert rejected["projection"] == {"state": "teacher_invalid", "uv": None}
    assert rejected["reasons"] == ["INSUFFICIENT_VIEWS"]


def test_inspection_api_rejects_invalid_queries_stale_media_and_writes(
    dataset: Path, tmp_path: Path
) -> None:
    ClipManifest.load(clip_dir(dataset)).media_path("cam0").unlink()
    client = TestClient(create_dataset_app(SLCSDatasetReviewService(dataset)))
    query = {"form": "video_000", "scene": "clip_000"}
    revision = client.get("/api/scene", params=query).json()["revision"]
    params = {**query, "revision": revision, "camera": "cam0", "frame": 0}
    assert client.get("/api/inspection/frame", params=params).status_code == 200
    assert client.get("/api/inspection/image", params=params).status_code == 404
    assert (
        client.get(
            "/api/inspection/frame", params={**params, "camera": "../escape"}
        ).status_code
        == 422
    )
    assert (
        client.get("/api/inspection/frame", params={**params, "frame": 5}).status_code
        == 422
    )
    assert (
        client.get("/api/inspection/frame", params={**params, "frame": -1}).status_code
        == 422
    )
    assert (
        client.get(
            "/api/inspection/frame", params={**params, "revision": "stale"}
        ).status_code
        == 409
    )
    assert client.post("/api/inspection/frame", params=params).status_code == 405
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not a video")
    media = clip_dir(dataset) / "media/cam0.mp4"
    media.symlink_to(outside)
    assert client.get("/api/inspection/image", params=params).status_code == 422


def test_source_jpeg_is_the_requested_frame_and_changes_invalidate_revision(
    dataset: Path,
) -> None:
    manifest = ClipManifest.load(clip_dir(dataset))
    media = manifest.media_path("cam0", must_exist=False)
    writer = cv2.VideoWriter(
        str(media),
        cv2.VideoWriter.fourcc(*"mp4v"),
        manifest.fps,
        (manifest.width, manifest.height),
    )
    assert writer.isOpened()
    for frame in range(manifest.num_frames):
        writer.write(
            np.full((manifest.height, manifest.width, 3), frame * 40, np.uint8)
        )
    writer.release()
    client = TestClient(create_dataset_app(SLCSDatasetReviewService(dataset)))
    query = {"form": "video_000", "scene": "clip_000"}
    scene = client.get("/api/scene", params=query).json()
    params = {**query, "revision": scene["revision"], "camera": "cam0", "frame": 3}
    response = client.get("/api/inspection/image", params=params)
    assert response.status_code == 200
    assert response.headers["x-slcs-frame"] == "3"
    bgr = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    assert bgr.shape == (manifest.height, manifest.width, 3)
    assert 110 < bgr.mean() < 125
    # Reuse is tied to the RGB media too, not just the pseudo-label archive.
    media.touch()
    assert client.get("/api/inspection/image", params=params).status_code == 409


def test_task_extension_assets_are_packaged_and_served(dataset: Path) -> None:
    client = TestClient(create_dataset_app(SLCSDatasetReviewService(dataset)))
    html = client.get("/").text
    assert html.index("slcs-inspection.mjs") < html.index("/static/app.js")
    for name in ("slcs-inspection.css", "slcs-inspection.mjs", "slcs-overlay.mjs"):
        assert client.get(f"/static/{name}").status_code == 200
