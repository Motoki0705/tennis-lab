"""Review the same proposal masks as training on real JPEG/store timelines."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.tasks.ball_detection.visualization.inference.service import DetectionService
from src.tasks.base.visualization.detection.web import create_detection_app
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def client(service: DetectionService) -> TestClient:
    return TestClient(create_detection_app(
        service, task="ball_detection", mode="review", service_config={},
        play_intervals=service.play_intervals,
    ))


def test_timeline_and_images_use_same_clip_without_writing_assets(tmp_path: Path) -> None:
    root = write_store_clip(tmp_path / "data/ball_detection/test", "clip", [
        frame(i, ball()) if 20 <= i < 80 and not 45 <= i < 50 else frame(i)
        for i in range(100)
    ])
    before = {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}
    with client(DetectionService(tmp_path)) as web:
        assert web.get("/api/catalog").json()["play_intervals_available"]
        scene = "store/test::clip"
        response = web.get("/api/play-intervals", params={"scene": scene})
        assert response.status_code == 200
        data = response.json()
        assert data["scene"] == scene and data["frames"] == 100
        assert data["config"]["window_length"] == 32
        assert data["play"] == [[20, 80]]
        assert data["excluded"] == [[0, 20], [80, 100]]
        assert data["training"] == [[20, 80]]
        assert data["bridged"] == [[45, 50]]
        assert data["presence"] == [[20, 45], [50, 80]]
        assert data["windows"] == [20, 36, 48]
        assert data["timestamps"][80] == pytest.approx(80 / 30)
        assert data["pose_approved"] is False
        preview = web.get("/api/preview", params={"scene": scene, "start": 45}).json()
        assert preview["items"][0]["gt"]["points"] == []
        assert web.get("/api/image", params={"scene": scene, "frame": 45}).status_code == 200
        assert web.get("/api/play-intervals", params={"scene": "/etc/passwd"}).status_code == 422
    assert before == {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}


def test_play_without_coordinate_teachers_and_short_clips(tmp_path: Path) -> None:
    root = tmp_path / "data/ball_detection/test"
    write_store_clip(root, "unknown", [frame(i, ball("unresolved", None)) for i in range(40)])
    write_store_clip(root, "short", [frame(i, ball()) for i in range(5)])
    service = DetectionService(tmp_path)
    unknown = service.play_intervals("store/test::unknown")
    assert unknown["play"] == ((0, 40),)
    assert not unknown["training"] and not unknown["windows"]
    short = service.play_intervals("store/test::short")
    assert not short["play"] and short["excluded"] == ((0, 5),)


def test_annotation_visibility_and_selection_evidence_are_distinct(tmp_path: Path) -> None:
    frames = [
        replace(frame(0, ball()), is_target=False),
        replace(frame(1, ball("unresolved", None)), is_target=False),
        frame(2, ball("unresolved", None)),
        frame(3, ball("interpolated")),
        frame(4, ball("occlusion_estimated")),
        frame(5, ball("out_of_frame", None)),
        frame(6, ball(), ball(track="b002")),
        frame(7, ball(), annotated=False),
        frame(8),
        frame(9, annotated=False),
    ]
    write_store_clip(tmp_path / "data/ball_detection/test", "clip", frames)
    service = DetectionService(tmp_path)
    response = service.play_intervals("store/test::clip")
    annotations = response["annotations"]
    assert [a["located_count"] for a in annotations] == [1, 0, 0, 1, 1, 0, 2, 1, 0, 0]
    assert [a["evidence"] for a in annotations] == [False, False, True, True, True, False, False, False, False, False]
    assert [a["exclusion_reasons"] for a in annotations] == [
        ["reference_only"], ["reference_only"], [], [], [], ["out_of_frame"],
        ["multiple_balls"], ["unreviewed"], ["no_ball"], ["unreviewed", "no_ball"],
    ]
    assert annotations[2]["kinds"] == ["unresolved"]
    assert annotations[3]["kinds"] == ["interpolated"]
    preview = service.preview("store/test::clip", count=10)
    # Exact parity with the viewer's visiblePoints filter, without eligibility masking.
    assert [sum(p["visible"] for p in item["gt"]["points"]) for item in preview["items"]] == [a["located_count"] for a in annotations]


def test_one_frame_coordinate_gap_is_preserved_even_when_evidence_is_continuous(tmp_path: Path) -> None:
    write_store_clip(tmp_path / "data/ball_detection/test", "clip", [
        frame(i, ball("unresolved", None) if i == 58 else ball()) for i in range(100)
    ])
    response = DetectionService(tmp_path).play_intervals("store/test::clip")
    assert response["presence"] == ((0, 100),)
    assert response["play"] == ((0, 100),)
    assert [response["annotations"][i]["located_count"] for i in (57, 58, 59)] == [1, 0, 1]
    assert response["annotations"][58]["exclusion_reasons"] == []


@pytest.fixture
def pose_source(tmp_path: Path) -> tuple[DetectionService, Path, Path]:
    snapshot = tmp_path / "outputs/input_snapshot/ball"
    for name in ("approved", "skipped"):
        write_store_clip(snapshot, name, [frame(i, ball()) for i in range(40)])
    # The live version is larger/different and must not replace the snapshot.
    write_store_clip(tmp_path / "data/ball_detection/live", "approved", [frame(i) for i in range(80)])
    poses = tmp_path / "data/ball_detection/poses"
    poses.mkdir()
    manifest = poses / "manifest.json"
    manifest.write_text(json.dumps(dict(
        schema="ball_detection_player_poses.v1", coordinate_system="stored_jpeg_pixels",
        ball_store=dict(directory=str(snapshot), hashes={
            name: dual_sha256(snapshot / name) for name in ("metadata.json", "index.npz")}),
        clips=[dict(clip_id="approved", pose_status="approved"),
               dict(clip_id="skipped", pose_status="skipped")],
    )))
    return DetectionService(tmp_path, play_poses=poses), manifest, snapshot


def test_pose_subset_uses_verified_snapshot_and_refreshes_approvals(pose_source) -> None:
    service, manifest, _ = pose_source
    dataset = "pose-approved/poses"
    assert service.scenes(dataset)["total"] == 1
    proposal = service.play_intervals(f"{dataset}::approved")
    assert proposal["frames"] == 40 and proposal["pose_approved"]
    assert proposal["play"] == ((0, 40),)
    assert service.preview(f"{dataset}::approved")["frames"] == 40
    overview = service.dataset_catalog.overview(dataset)
    assert overview["clips"] == 1 and overview["counts"]["frames"] == 40
    assert overview["point_counts"]["observed"] == 40
    assert overview["selection"]["parent_clips"] == 2
    assert service.scenes(dataset, source="tracknet", review_state="observed")["total"] == 1
    assert service.review(f"{dataset}::approved")["positions"]["observed"] == list(range(40))
    assert service.play_intervals("store/live::approved")["frames"] == 80
    with pytest.raises(ValueError, match="Unknown scene"):
        service.play_intervals(f"{dataset}::skipped")
    data = json.loads(manifest.read_text())
    data["clips"][1]["pose_status"] = "approved"
    manifest.write_text(json.dumps(data))
    assert service.scenes(dataset)["total"] == 1
    service.catalog()
    assert service.scenes(dataset)["total"] == 2


@pytest.mark.parametrize("corruption", ["hash", "escape", "duplicate", "missing_hash", "missing_index", "shard_escape"])
def test_invalid_pose_snapshot_is_unavailable_without_live_fallback(pose_source, tmp_path, corruption) -> None:
    service, manifest, snapshot = pose_source
    data = json.loads(manifest.read_text())
    if corruption == "hash":
        data["ball_store"]["hashes"]["index.npz"] = "wrong"
    elif corruption == "escape":
        data["ball_store"]["directory"] = str(tmp_path.parent / "untrusted")
    elif corruption == "duplicate":
        data["clips"].append(data["clips"][0])
    elif corruption == "missing_hash":
        del data["ball_store"]["hashes"]["index.npz"]
    elif corruption == "missing_index":
        (snapshot / "index.npz").unlink()
    else:
        shard = next((snapshot / "shards").iterdir())
        external = tmp_path / "external.jpg"
        external.write_bytes(shard.read_bytes())
        shard.unlink()
        shard.symlink_to(external)
    manifest.write_text(json.dumps(data))
    for _ in range(2):
        catalog = service.catalog()
        entry = next(d for d in catalog["datasets"] if d["id"] == "pose-approved/poses")
        assert not entry["available"] and entry["reason"]
        with pytest.raises(ValueError):
            service.scenes("pose-approved/poses")
        with pytest.raises(ValueError, match="Unknown scene"):
            service.play_intervals("pose-approved/poses::approved")
