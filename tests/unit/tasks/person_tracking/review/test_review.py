"""Missing tracking states, current data bindings and read-only HTTP boundaries."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    INDEX_FILE,
    METADATA_FILE,
    SHARDS_DIR,
    BallFrameStore,
)
from src.tasks.ball_detection.generate_dataset.frame_store.builder import build_dataset
from src.tasks.person_tracking.review.model import ReviewSequence, runs
from src.tasks.person_tracking.review.service import TrackingReviewService
from src.tasks.person_tracking.review.sources import (
    BallPoseCampaign,
    CheckedFile,
    ComponentStore,
    VideoFrames,
)
from src.tasks.person_tracking.review.web import create_app
from src.tasks.person_tracking.scripts.review_dataset import PATH_BOUNDARY
from src.tasks.player_association.evaluation.labels import (
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.tennis_scene.chat_annotation.player_pose.reviews import validate_and_remap
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES
from tests.support.tasks.ball_detection.frame_sources import (
    TRACKNET_SOURCE,
    SourceRoots,
    write_meiji,
    write_tracknet,
)


class SolidFrames:
    def read(self, frame: int) -> NDArray[np.uint8]:
        return np.full((48, 64, 3), frame * 30, np.uint8)


def sample(*, synthetic: bool = False) -> ReviewSequence:
    observed = np.array([[True, True, False, True]], bool)
    # Garbage outside the real-observation mask must never become a visible box.
    boxes = np.tile(np.array([1, 2, 13, 27], np.float32), (1, 4, 1))
    selected = observed.copy()
    interval = {
        "raw_track_id": 3,
        "start_frame": 0,
        "stop_frame": 4,
        "role": "player",
        "player_id": "player_1",
        "evidence_frames": [0, 3],
        "reason": "A reviewed interval spanning a saved observation gap",
    }
    return ReviewSequence(
        "sample",
        "clip",
        "cam0",
        64,
        48,
        np.arange(4, dtype=np.int32),
        np.array([100, 110, 140, 150], np.int64),
        "1/1000",
        np.array([3], np.int64),
        boxes,
        observed,
        np.array([[0, 1, -1, 2]], np.int64),
        ((3,),),
        boxes.copy() if synthetic else None,
        ~observed if synthetic else None,
        selected,
        None,
        (interval,),
        None,
        {},
        SolidFrames(),
    )


def resolver(root: Path) -> PathResolver:
    return PathResolver(
        RuntimePathRoots(
            project_root=root,
            data_root=root / "data",
            artifact_root=root / "outputs",
            checkpoint_root=root / "outputs",
            output_root=root / "outputs",
            cache_root=root / "outputs",
            external_asset_root=root / "data",
        )
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def campaign_fixture(tmp_path: Path) -> tuple[BallPoseCampaign, Path, Path]:
    """Only unit-test fixture data; never used as screenshot/review evidence."""
    roots = SourceRoots(tmp_path)
    write_tracknet(roots.data / "tennis/tracknet")
    original = build_dataset(roots.config({"tracknet": TRACKNET_SOURCE}))
    campaign, dataset = tmp_path / "outputs/campaign", tmp_path / "data/poses"
    snapshot = campaign / "snapshot"
    shutil.copytree(original, snapshot)
    store = BallFrameStore(snapshot)
    plans = [
        {
            "index": clip.index,
            "clip_id": clip.clip_id,
            "frame_count": clip.frame_count,
            "width": clip.width,
            "height": clip.height,
        }
        for clip in store.clips
    ]
    config = {
        "store": str(snapshot),
        "store_hashes": {
            name: dual_sha256(snapshot / name) for name in (METADATA_FILE, INDEX_FILE)
        },
    }
    write_json(campaign / "config.json", config)
    write_json(
        campaign / "plan.json",
        {"schema": "ball_store_player_pose_plan.v1", "clips": plans},
    )
    write_json(
        campaign / "identity.json",
        {
            "config_sha256": dual_sha256(campaign / "config.json"),
            "plan_sha256": dual_sha256(campaign / "plan.json"),
        },
    )
    clip = store.clips[0]
    raw_root = campaign / "clips/clip-00000"
    raw_root.mkdir(parents=True)
    observations = np.array(
        [[True, True, False, False], [False, False, False, True]], bool
    )
    raw = {
        "frame_index": store.frames["frame_index"][:4],
        "pts": store.frames["pts"][:4],
        "track_ids": np.array([3, 8], np.int64),
        "detection_rows": np.array([[0, 1, -1, -1], [-1, -1, -1, 2]], np.int64),
        "boxes": np.tile(np.array([2, 2, 15, 20], np.float32), (2, 4, 1)),
        "keypoints": np.zeros((2, 4, 17, 3), np.float32),
    }
    raw["boxes"][~observations] = 0
    np.savez(raw_root / "tracks.npz", **raw)
    write_json(
        raw_root / "input.json",
        {
            "clip_id": clip.clip_id,
            "shard_sha256": dual_sha256(snapshot / SHARDS_DIR / "clip-00000.bin"),
        },
    )
    raw_hash = dual_sha256(raw_root / "tracks.npz")
    write_json(
        raw_root / "generation.json",
        {
            "status": "complete",
            "clip_id": clip.clip_id,
            "frame_count": 4,
            "tracking_profile": {"method": "strongsort_pp_pose"},
            "source_track_ids": [[3], [8]],
            "link_candidates": [],
            "files": {
                "tracks.npz": raw_hash,
                "input.json": dual_sha256(raw_root / "input.json"),
            },
        },
    )
    review = {
        "clip_id": clip.clip_id,
        "raw_tracks_sha256": raw_hash,
        "status": "approved",
        "reviewed_sheets": ["frames-000000-000004.jpg"],
        "notes": "Unit-test interval assignments",
        "segments": [
            {
                "raw_track_id": track,
                "start_frame": start,
                "stop_frame": stop,
                "role": "player",
                "player_id": "player_1",
                "evidence_frames": [start],
                "reason": "Same person in a saved observation",
            }
            for track, start, stop in [(3, 0, 2), (8, 3, 4)]
        ],
    }
    remapped = validate_and_remap(
        raw,
        review,
        clip_id=clip.clip_id,
        raw_hash=raw_hash,
        required_sheets=review["reviewed_sheets"],
    )
    dataset.mkdir(parents=True)
    np.savez(dataset / "clip.npz", **remapped)
    write_json(dataset / "review.json", review)
    entries = [
        {"clip_id": item.clip_id, "pose_status": "pending"} for item in store.clips
    ]
    entries[0].update(
        pose_status="approved",
        file="clip.npz",
        sha256=dual_sha256(dataset / "clip.npz"),
        review_file="review.json",
        review_sha256=dual_sha256(dataset / "review.json"),
        raw_tracks_sha256=raw_hash,
    )
    write_json(
        dataset / "manifest.json",
        {
            "schema": "ball_detection_player_poses.v1",
            "campaign": str(campaign),
            "ball_store": {
                "directory": str(snapshot),
                "hashes": config["store_hashes"],
            },
            "clips": entries,
        },
    )
    return BallPoseCampaign(campaign, dataset, resolver(tmp_path)), campaign, dataset


def test_missing_frame_has_no_box_crop_or_id_assignment() -> None:
    sequence = sample()
    frame = sequence.frame_info(2)["tracks"][0]
    assert frame["state"] == "missing" and frame["box"] is None
    assert frame["assignment"] is None and frame["detection_row"] is None
    assert frame["previous_observed"] == 1 and frame["next_observed"] == 3
    assert sequence.summary()["labelled_observations"] == 3
    assert sequence.summary()["saved_synthetic"] is None
    assert sequence.summary()["events"] == [
        {
            "kind": "observation_gap",
            "track_id": 3,
            "start": 2,
            "stop": 3,
            "previous": 1,
            "next": 3,
            "synthetic_frames": 0,
        }
    ]


def test_saved_synthetic_box_stays_outside_observations_selection_and_labels() -> None:
    sequence = sample(synthetic=True)
    frame = sequence.frame_info(2)["tracks"][0]
    assert frame["state"] == "interpolated" and frame["box"] == [1, 2, 13, 27]
    assert frame["assignment"] is None and frame["selected"] is False
    assert (
        sequence.summary()["observations"]
        == sequence.summary()["labelled_observations"]
        == 3
    )
    assert sequence.summary()["saved_synthetic"] == 1
    assert sequence.summary()["tracks"][0]["synthetic_runs"] == [[2, 3]]
    with pytest.raises(ValueError, match="Synthetic boxes require"):
        replace(sequence, reconstructed_boxes=None)
    with pytest.raises(ValueError, match="separate"):
        replace(sequence, interpolated=np.ones((1, 4), bool))


@pytest.mark.parametrize("change", ["rows", "pts", "selected"])
def test_axis_or_observation_mask_conflicts_fail_closed(change: str) -> None:
    sequence = sample()
    with pytest.raises(ValueError):
        if change == "rows":
            replace(sequence, detection_rows=np.array([[0, 1, 2, 3]], np.int64))
        elif change == "pts":
            replace(sequence, pts=np.array([0, 1, 1, 3], np.int64))
        else:
            replace(sequence, selected=np.ones((1, 4), bool))


def test_run_intervals_preserve_single_frame_gaps_and_empty_masks() -> None:
    assert runs(np.array([False, True, False, True, True, False], bool)) == [
        [1, 2],
        [3, 5],
    ]
    assert runs(np.zeros(4, bool)) == []


def test_campaign_uses_snapshot_and_preserves_the_reviewed_raw_fragments(
    campaign_fixture: tuple[BallPoseCampaign, Path, Path],
) -> None:
    source, _, _ = campaign_fixture
    sequence = source.load("ball:0")
    assert source.catalog()["available"] == 1 and source.catalog()["clips"] == 3
    assert (
        sequence.summary()["observations"]
        == sequence.summary()["selected_observations"]
        == 3
    )
    assert [item["track_id"] for item in sequence.summary()["tracks"]] == [3, 8]
    assert sequence.frame_info(2)["tracks"][0]["box"] is None
    assert sequence.images.read(3).shape == (48, 64, 3)
    with pytest.raises(ValueError, match="未生成"):
        source.load("ball:1")


def test_changed_rgb_shard_is_rejected_after_load(
    campaign_fixture: tuple[BallPoseCampaign, Path, Path],
) -> None:
    source, campaign, _ = campaign_fixture
    sequence = source.load("ball:0")
    path = campaign / "snapshot" / SHARDS_DIR / "clip-00000.bin"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        sequence.images.read(1)


def test_raw_pts_cannot_be_rebound_by_rehashing_the_archive(
    campaign_fixture: tuple[BallPoseCampaign, Path, Path],
) -> None:
    source, campaign, _ = campaign_fixture
    raw_path = campaign / "clips/clip-00000/tracks.npz"
    with np.load(raw_path) as archive:
        raw = {key: archive[key] for key in archive.files}
    raw["pts"][1] += 100
    np.savez(raw_path, **raw)
    generation_path = raw_path.with_name("generation.json")
    generation = json.loads(generation_path.read_text())
    generation["files"]["tracks.npz"] = dual_sha256(raw_path)
    write_json(generation_path, generation)
    with pytest.raises(ValueError, match="frame/PTS"):
        source.load("ball:0")


def test_adopted_review_must_reference_the_exact_raw_archive(
    campaign_fixture: tuple[BallPoseCampaign, Path, Path],
) -> None:
    _, campaign, dataset = campaign_fixture
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["clips"][0]["raw_tracks_sha256"] = "different"
    write_json(dataset / "manifest.json", manifest)
    source = BallPoseCampaign(campaign, dataset, resolver(campaign.parent.parent))
    with pytest.raises(ValueError, match="different raw"):
        source.load("ball:0")


@pytest.fixture
def component_fixture(tmp_path: Path) -> tuple[ComponentStore, Path, PathResolver]:
    write_meiji(tmp_path / "data/meiji", videos=("video_000",))
    media = tmp_path / "data/meiji/videos/video_000/clips/clip_000/media/cam0.mp4"
    source = {
        "clip_id": "video_000/clip_000",
        "videos": [
            {
                "camera_id": "cam0",
                "path": str(media),
                "sha256": dual_sha256(media),
                "num_frames": 5,
                "fps": 59.94006,
                "width": 96,
                "height": 64,
            }
        ],
    }
    store = ClipStore(tmp_path / "outputs/store", source)
    boxes = np.tile(np.array([1, 2, 20, 42], np.float32), (1, 5, 1))
    observed = np.array([[True, True, False, True, True]], bool)
    payload = {
        "camera_id": "cam0",
        "track_ids": np.array([3], np.int64),
        "boxes_xyxy": boxes,
        "observed": observed,
        "source_track_ids": [[3]],
        "tracklet_links": [],
        "evidence": {"detection_rows": np.array([[0, 1, -1, 2, 3]], np.int64)},
        "reconstruction": {
            "boxes": boxes.copy(),
            "observed": observed.copy(),
            "interpolated": ~observed,
        },
        "offline_link_candidates": [],
    }
    store.publish(
        "person_tracking/cam0",
        payload,
        ArtifactCodec(dict),
        schema="person_tracks",
        version=5,
        identity={"settings": {"profile": {"method": "strongsort_pp_pose"}}},
        dependencies={},
        provenance={"origin": "test"},
    )
    return (
        ComponentStore(store.root, resolver(tmp_path), {}),
        store.root,
        resolver(tmp_path),
    )


def test_current_component_has_real_rgb_pts_and_separate_gsi(
    component_fixture: tuple[ComponentStore, Path, PathResolver],
) -> None:
    source, _, _ = component_fixture
    sequence = source.load(next(iter(source.records)))
    assert sequence.summary()["saved_synthetic"] == 1
    assert sequence.frame_info(2)["tracks"][0]["state"] == "interpolated"
    assert sequence.images.read(3).shape == (64, 96, 3)
    assert isinstance(sequence.images, VideoFrames)
    assert sequence.frame_info(3)["pts"] == int(sequence.images.pts[3])


def test_source_video_checksum_and_frame_count_must_match(
    component_fixture: tuple[ComponentStore, Path, PathResolver],
) -> None:
    source, _, paths = component_fixture
    declared = source.source_videos["cam0"]
    with pytest.raises(ValueError, match="checksum"):
        VideoFrames({**declared, "sha256": "wrong"}, paths)
    with pytest.raises(ValueError, match="frame count"):
        VideoFrames({**declared, "num_frames": 4}, paths)


def test_old_component_contract_is_catalogued_without_implicit_upgrade(
    component_fixture: tuple[ComponentStore, Path, PathResolver],
) -> None:
    _, root, paths = component_fixture
    index = json.loads((root / "scene.json").read_text())
    index["artifacts"]["person_tracking/cam0"]["version"] = 3
    write_json(root / "scene.json", index)
    source = ComponentStore(root, paths, {})
    record = next(iter(source.records.values()))
    assert record["available"] is False and record["status"] == "historical v3"
    with pytest.raises(ValueError, match="暗黙変換"):
        source.load(record["key"])


def test_changed_component_array_is_rejected(
    component_fixture: tuple[ComponentStore, Path, PathResolver],
) -> None:
    source, root, _ = component_fixture
    record = next(iter(source.records.values()))
    reference = source.document["artifacts"][record["node"]]
    array = (root / reference["path"]).parent / "array_0000.npy"
    array.write_bytes(b"corrupted array")
    with pytest.raises(ValueError, match="checksum"):
        source.load(record["key"])


@pytest.mark.parametrize("invent_synthetic_selection", [False, True])
def test_selection_groups_retain_raw_origins_and_never_include_gsi(
    component_fixture: tuple[ComponentStore, Path, PathResolver],
    invent_synthetic_selection: bool,
) -> None:
    source, root, paths = component_fixture
    raw, _ = source.payload("person_tracking/cam0")
    store = ClipStore(root, source.source)
    selected = raw["observed"].copy()
    if invent_synthetic_selection:
        selected[0, 2] = True
    groups = {**raw, "track_ids": np.array([0], np.int64), "source_track_ids": [[0]]}
    selection = {
        "camera_id": "cam0",
        "raw_track_ids": raw["track_ids"],
        "selected": selected,
        "tracks": groups,
        "origin_rows": np.where(raw["observed"], 0, -1).astype(np.int64),
        "diagnostics": {},
    }
    reference = store.active("person_tracking/cam0")
    assert reference is not None
    store.publish(
        "player_selection/cam0",
        selection,
        ArtifactCodec(dict),
        schema="selected_player_tracks",
        version=2,
        identity={},
        dependencies={"tracks": reference},
        provenance={"origin": "test"},
    )
    updated = ComponentStore(root, paths, {})
    key = next(iter(updated.records))
    if invent_synthetic_selection:
        with pytest.raises(ValueError, match="only saved real observations"):
            updated.load(key)
    else:
        sequence = updated.load(key)
        assert sequence.frame_info(1)["tracks"][0]["group_id"] == 0
        assert sequence.frame_info(1)["tracks"][0]["track_id"] == 3
        assert sequence.frame_info(2)["tracks"][0]["group_id"] is None
        assert sequence.summary()["selected_observations"] == 4


def test_retired_dependency_cannot_be_adopted_by_rehashing_the_descriptor(
    component_fixture: tuple[ComponentStore, Path, PathResolver],
) -> None:
    source, root, paths = component_fixture
    index = json.loads((root / "scene.json").read_text())
    reference = index["artifacts"]["person_tracking/cam0"]
    descriptor_path = root / reference["path"]
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["dependencies"] = {"detections": {"artifact_id": "retired"}}
    write_json(descriptor_path, descriptor)
    reference["sha256"] = dual_sha256(descriptor_path)
    write_json(root / "scene.json", index)
    updated = ComponentStore(root, paths, {})
    with pytest.raises(ValueError, match="superseded"):
        updated.load(next(iter(updated.records)))


def test_partial_reference_is_separate_from_raw_identity_assignments(
    component_fixture: tuple[ComponentStore, Path, PathResolver],
    tmp_path: Path,
) -> None:
    source, root, paths = component_fixture
    labels = ClipLabels(
        "video_000/clip_000",
        5,
        (LabelledPerson("A", "player", "Reviewed person"),),
        {
            "cam0": CameraLabels(
                np.array([0, 3], np.int64),
                np.array([0, 0], np.int64),
                np.tile([1.0, 2.0, 20.0, 42.0], (2, 1)),
            )
        },
        {"scope": "partial saved boxes"},
    )
    file = tmp_path / "data/reference.json"
    labels.save(file)
    updated = ComponentStore(
        root, paths, {labels.clip_id: (labels, CheckedFile.open(file))}
    )
    sequence = updated.load(next(iter(updated.records)))
    assert sequence.frame_info(3)["reference_boxes"][0]["person"] == "A"
    assert sequence.frame_info(3)["tracks"][0]["assignment"] is None
    assert sequence.summary()["labelled_observations"] == 0
    assert sequence.metadata["reference"]["boxes"] == 2
    bad = replace(labels, num_frames=6)
    mismatched = ComponentStore(
        root, paths, {bad.clip_id: (bad, CheckedFile.open(file))}
    )
    with pytest.raises(ValueError, match="camera/frame"):
        mismatched.load(next(iter(mismatched.records)))


def test_http_is_read_only_and_never_exposes_a_missing_box() -> None:
    sequence = sample()
    service = SimpleNamespace(
        load=lambda key: (
            sequence if key == "sample" else (_ for _ in ()).throw(KeyError(key))
        ),
        catalog=lambda: {"read_only": True},
    )
    with TestClient(create_app(cast(TrackingReviewService, service))) as client:
        assert client.get("/api/catalog").json()["read_only"] is True
        assert (
            client.get("/api/sequences/sample/frames/2").json()["tracks"][0]["box"]
            is None
        )
        image = client.get("/api/sequences/sample/frames/2/image")
        decoded = cv2.imdecode(np.frombuffer(image.content, np.uint8), cv2.IMREAD_COLOR)
        assert image.status_code == 200 and decoded is not None
        assert decoded.shape == (48, 64, 3)
        assert client.get("/api/sequences/sample/frames/4/image").status_code == 404
        assert client.get("/api/sequences/unknown").status_code == 404
        assert client.post("/api/sequences/sample").status_code == 405
        assert client.get("/static/%2e%2e/sources.py").status_code == 404


def test_cli_has_registered_path_boundary_and_rejects_relative_or_escaped_inputs(
    tmp_path: Path,
) -> None:
    paths = resolver(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs/store").mkdir(parents=True)
    arguments: dict[str, Any] = {
        "data_root": tmp_path / "data",
        "artifact_root": tmp_path / "outputs",
        "stores": [tmp_path / "outputs/store"],
    }
    boundary = next(
        item
        for item in EXPECTED_RUNTIME_BOUNDARIES
        if item.module == "src.tasks.person_tracking.scripts.review_dataset"
    )
    assert boundary.validator_key == PATH_BOUNDARY.name
    assert (
        len(PATH_BOUNDARY.validate(arguments, resolver=paths).declared_many("stores"))
        == 1
    )
    with pytest.raises(ValueError, match="absolute"):
        PATH_BOUNDARY.validate(
            {**arguments, "stores": [Path("outputs/store")]}, resolver=paths
        )
    with pytest.raises(ValueError):
        PATH_BOUNDARY.validate(
            {**arguments, "stores": [tmp_path / "data"]}, resolver=paths
        )


def test_campaign_pair_and_unused_references_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="together"):
        TrackingReviewService(resolver=resolver(tmp_path), campaign=tmp_path)
    with pytest.raises(ValueError, match="Supply"):
        TrackingReviewService(resolver=resolver(tmp_path))


def test_checked_file_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text("{}")
    checked = CheckedFile.open(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        checked.unchanged()
