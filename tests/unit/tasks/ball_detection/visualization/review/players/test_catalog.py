from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.visualization.inference.service import DetectionService
from src.tasks.ball_detection.visualization.review.players import PlayerCatalog
from src.tennis_scene.chat_annotation.player_pose.storage import (
    digest,
    write_json,
    write_npz,
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.fixture
def assets(tmp_path: Path) -> dict[str, Any]:
    data = tmp_path / "data"
    campaign = tmp_path / "outputs" / "campaign"
    snapshot = campaign / "snapshot" / "ball-v2"
    labels = [replace(frame(i, ball()), segment_break=i == 4) for i in range(6)]
    for name in ("approved", "held", "waiting", "unbuilt", "skipped"):
        write_store_clip(snapshot, name, labels)
    metadata = json.loads((snapshot / "metadata.json").read_text())
    metadata["version"] = "ball-v2"
    write_json(snapshot / "metadata.json", metadata)
    public = data / "ball_detection" / "ball-v2"
    shutil.copytree(snapshot, public)
    write_store_clip(public, "new", labels)
    store = BallFrameStore(snapshot)
    poses = data / "ball_detection" / "ball-v2-player-pose-v1"
    entries = []
    for clip in store.clips:
        entry: dict[str, Any] = {
            "index": clip.index,
            "clip_id": clip.clip_id,
            "pose_status": "approved"
            if clip.clip_id == "approved"
            else "skipped"
            if clip.clip_id == "skipped"
            else "pending",
        }
        if clip.clip_id in {"approved", "held", "waiting"}:
            root = campaign / "clips" / f"clip-{clip.index:05d}"
            raw = dict(
                track_ids=np.array([7, 19]),
                frame_index=np.arange(6, dtype=np.int32),
                pts=np.arange(6, dtype=np.int64),
                boxes=np.tile([1.0, 2.0, 20.0, 30.0], (2, 6, 1)).astype(np.float32),
                keypoints=np.full((2, 6, 17, 3), 1.7, dtype=np.float32),
                detection_rows=np.tile(np.arange(6), (2, 1)),
            )
            raw["detection_rows"][0, 2] = -1
            # Even a nonzero interpolated box/pose must not become an observation.
            write_npz(root / "tracks.npz", **raw)
            write_json(
                root / "input.json",
                {
                    "clip_id": clip.clip_id,
                    "shard_sha256": digest(
                        snapshot / "shards" / shard_name(clip.index)
                    ),
                },
            )
            write_json(
                root / "generation.json",
                {
                    "status": "complete",
                    "clip_id": clip.clip_id,
                    "frame_count": 6,
                    "raw_tracks": 2,
                    "files": {
                        name: digest(root / name)
                        for name in ("tracks.npz", "input.json")
                    },
                },
            )
            if clip.clip_id == "held":
                write_json(root / "review_status.json", {"status": "needs_review"})
            if clip.clip_id == "approved":
                observed = raw["detection_rows"][:1].T >= 0
                pose = dict(
                    frame_index=raw["frame_index"],
                    pts=raw["pts"],
                    player_ids=np.array(["player_1"]),
                    boxes_xyxy=raw["boxes"][:1].transpose(1, 0, 2).copy(),
                    keypoints=raw["keypoints"][:1].transpose(1, 0, 2, 3).copy(),
                    observed=observed,
                    raw_track_ids=np.where(observed, 7, -1),
                    detection_rows=raw["detection_rows"][:1].T,
                )
                pose["boxes_xyxy"][~observed] = 0
                pose["keypoints"][~observed] = 0
                write_npz(poses / "clips" / "a.npz", **pose)
                raw_hash = digest(root / "tracks.npz")
                write_json(
                    poses / "reviews" / "a.json",
                    {
                        "status": "approved",
                        "clip_id": clip.clip_id,
                        "raw_tracks_sha256": raw_hash,
                    },
                )
                entry.update(
                    file="clips/a.npz",
                    sha256=digest(poses / "clips/a.npz"),
                    review_file="reviews/a.json",
                    review_sha256=digest(poses / "reviews/a.json"),
                    raw_tracks_sha256=raw_hash,
                )
        entries.append(entry)
    write_json(
        poses / "manifest.json",
        {
            "schema": "ball_detection_player_poses.v1",
            "coordinate_system": "stored_jpeg_pixels",
            "ball_store": {
                "directory": str(snapshot),
                "hashes": {
                    name: digest(snapshot / name)
                    for name in ("metadata.json", "index.npz")
                },
            },
            "campaign": str(campaign),
            "clips": entries,
        },
    )
    catalog = PlayerCatalog(data, tmp_path)
    return dict(
        catalog=catalog,
        public=BallFrameStore(public),
        snapshot=snapshot,
        poses=poses,
        campaign=campaign,
        identity="players/ball-v2-player-pose-v1",
        root=tmp_path,
    )


def preview(
    a: dict[str, Any], clip: str = "approved", mode: str = "reviewed"
) -> dict[str, Any]:
    catalog: PlayerCatalog = a["catalog"]
    result: dict[str, Any] = catalog.preview(
        a["identity"], a["public"], clip, 0, 6, mode
    )
    return result


def test_appended_store_join_mask_raw_ids_and_gap_breaks(
    assets: dict[str, Any],
) -> None:
    result = preview(assets)
    assert result["available"] and result["status"] == "approved"
    people = result["items"]
    assert people[0]["people"][0]["id"] == "player_1"
    assert people[0]["people"][0]["raw_track_id"] == 7
    assert people[0]["people"][0]["keypoints"][0][2] == pytest.approx(
        1.7
    )  # never clamp scores
    assert people[2]["people"] == []
    assert len(people[3]["people"][0]["trail"]) == 1  # missing frame 2
    assert len(people[5]["people"][0]["trail"]) == 2  # segment break at 4
    raw = preview(assets, mode="raw")
    assert [p["id"] for p in raw["items"][0]["people"]] == ["raw_7", "raw_19"]
    assert [p["id"] for p in raw["items"][2]["people"]] == ["raw_19"]


@pytest.mark.parametrize(
    "clip,status,raw",
    [
        ("held", "needs_review", True),
        ("waiting", "review_pending", True),
        ("unbuilt", "not_generated", False),
        ("new", "not_generated", False),
        ("skipped", "skipped", False),
    ],
)
def test_absence_is_explicit_not_an_empty_approved_result(
    assets: dict[str, Any], clip: str, status: str, raw: bool
) -> None:
    result = preview(assets, clip)
    assert result["status"] == status
    assert not result["available"]
    assert result["raw_available"] is raw
    assert all(not item["people"] for item in result["items"])


def test_status_filter_and_raw_api_use_stable_clip_id(assets: dict[str, Any]) -> None:
    from fastapi.testclient import TestClient

    from src.tasks.base.visualization.detection.web import create_detection_app

    service = DetectionService(assets["root"])
    client = TestClient(
        create_detection_app(
            service, task="ball_detection", mode="review", service_config={}
        )
    )
    result = client.get(
        "/api/scenes",
        params={
            "dataset": "store/ball-v2",
            "player_dataset": assets["identity"],
            "player_status": "needs_review",
        },
    )
    assert result.status_code == 200
    assert [i["id"] for i in result.json()["items"]] == ["store/ball-v2::held"]
    result = client.get(
        "/api/players",
        params={
            "scene": "store/ball-v2::held",
            "dataset": assets["identity"],
            "mode": "raw",
            "count": 6,
        },
    )
    assert result.status_code == 200
    assert result.json()["items"][0]["people"][0]["id"] == "raw_7"
    assert (
        client.get(
            "/api/players",
            params={"scene": "store/ball-v2::held", "dataset": "../../escape"},
        ).status_code
        == 422
    )


def test_replaced_artifact_is_not_served_from_cache(assets: dict[str, Any]) -> None:
    preview(assets)
    (assets["poses"] / "clips/a.npz").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="checksum"):
        preview(assets)


@pytest.mark.parametrize(
    "field,value",
    [("clip_id", "held"), ("raw_tracks_sha256", "0" * 64)],
)
def test_review_identity_is_checked_after_artifact_checksum(
    assets: dict[str, Any], field: str, value: str
) -> None:
    review_path = assets["poses"] / "reviews/a.json"
    decision = json.loads(review_path.read_text())
    decision[field] = value
    write_json(review_path, decision)
    manifest_path = assets["poses"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(c for c in manifest["clips"] if c["clip_id"] == "approved")
    entry["review_sha256"] = digest(review_path)
    write_json(manifest_path, manifest)
    assets["catalog"] = PlayerCatalog(assets["root"] / "data", assets["root"])
    with pytest.raises(ValueError, match="review identity/status mismatch"):
        preview(assets)


@pytest.mark.parametrize("kind", ["image", "pts"])
def test_matching_clip_name_is_insufficient_for_join(
    assets: dict[str, Any], kind: str
) -> None:
    if kind == "image":
        path = assets["public"].directory / "shards/clip-00000.bin"
        data = bytearray(path.read_bytes())
        data[30] ^= 1
        path.write_bytes(data)
    else:
        assets["public"].frames["pts"][0] += 1
    with pytest.raises(ValueError, match="different JPEG|correspondence"):
        preview(assets)


def test_raw_requires_original_input_image_hash(
    assets: dict[str, Any],
) -> None:
    root = assets["campaign"] / "clips/clip-00000"
    write_json(root / "input.json", {"clip_id": "approved", "shard_sha256": "wrong"})
    record = json.loads((root / "generation.json").read_text())
    record["files"]["input.json"] = digest(root / "input.json")
    write_json(root / "generation.json", record)
    with pytest.raises(ValueError, match="different JPEG"):
        preview(assets, mode="raw")


@pytest.mark.parametrize(
    "change, message",
    [
        ("pts", "time axes"),
        ("keypoints", "axis mismatch"),
        ("detection_rows", "observation mask"),
    ],
)
def test_raw_axes_and_mask_are_validated_after_checksum(
    assets: dict[str, Any], change: str, message: str
) -> None:
    root = assets["campaign"] / "clips/clip-00000"
    with np.load(root / "tracks.npz") as archive:
        raw = {name: archive[name] for name in archive.files}
    if change == "pts":
        raw[change][0] += 1
    elif change == "keypoints":
        raw[change] = raw[change].transpose(1, 0, 2, 3)
    else:
        raw[change][0, 0] = -2
    write_npz(root / "tracks.npz", **raw)
    record = json.loads((root / "generation.json").read_text())
    record["files"]["tracks.npz"] = digest(root / "tracks.npz")
    write_json(root / "generation.json", record)
    with pytest.raises(ValueError, match=message):
        preview(assets, mode="raw")


def test_campaign_failure_is_distinct_from_not_started(assets: dict[str, Any]) -> None:
    write_json(assets["campaign"] / "generation_status.json", {"failed": [3]})
    assert preview(assets, "unbuilt")["status"] == "failed"


def test_pose_artifact_symlink_cannot_escape_root(
    assets: dict[str, Any], tmp_path: Path
) -> None:
    path = assets["poses"] / "clips/a.npz"
    outside = tmp_path / "outside.npz"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        preview(assets)


def test_manifest_refresh_is_required_and_corruption_is_visible(
    assets: dict[str, Any],
) -> None:
    preview(assets)
    path = assets["poses"] / "manifest.json"
    path.write_text("invalid")
    with pytest.raises(ValueError, match="refresh"):
        preview(assets)
    catalog = PlayerCatalog(assets["root"] / "data", assets["root"])
    entries = catalog.discover()
    assert len(entries) == 1 and not entries[0]["available"] and entries[0]["error"]
