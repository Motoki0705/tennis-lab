from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.tasks.ball_detection.data.store import POINT_KIND_CODES, BallFrameStore
from src.tasks.ball_detection.generate_dataset.frame_store.builder import build_dataset
from src.tennis_scene.chat_annotation.player_pose import runner
from src.tennis_scene.chat_annotation.player_pose.dataset import PlayerPoseStore
from src.tennis_scene.chat_annotation.player_pose.generation import (
    _cached_chunk,
    _save_chunk,
)
from src.tennis_scene.chat_annotation.player_pose.reviews import (
    accept_review,
    validate_and_remap,
)
from src.tennis_scene.chat_annotation.player_pose.selection import (
    initialize,
    presence_counts,
)
from src.tennis_scene.chat_annotation.player_pose.storage import (
    clip_root,
    digest,
    read_json,
    write_json,
    write_npz,
)
from tests.support.tasks.ball_detection.frame_sources import (
    TRACKNET_SOURCE,
    SourceRoots,
    write_tracknet,
)


def raw_tracks() -> dict:
    return {
        "track_ids": np.array([3, 8], np.int64),
        "frame_index": np.arange(4, dtype=np.int32),
        "pts": np.arange(4, dtype=np.int64),
        "detection_rows": np.array([[0, 1, -1, -1], [-1, -1, 2, 3]], np.int64),
        "boxes": np.ones((2, 4, 4), np.float32) * np.array([2, 2, 15, 20], np.float32),
        "keypoints": np.stack(
            [np.ones((4, 17, 3), np.float32), np.ones((4, 17, 3), np.float32) * 2]
        ),
    }


def segment(
    track: int, start: int, stop: int, player: str | None, role: str = "player"
) -> dict:
    return {
        "raw_track_id": track,
        "start_frame": start,
        "stop_frame": stop,
        "role": role,
        "player_id": player,
        "evidence_frames": [start],
        "reason": "Same clothing and continuous motion in viewed frames",
    }


def decision(segments: list[dict]) -> dict:
    return {
        "clip_id": "clip",
        "raw_tracks_sha256": "hash",
        "status": "approved",
        "reviewed_sheets": ["frames.jpg"],
        "segments": segments,
        "notes": "",
    }


def remap(raw: dict, review: dict) -> dict:
    result: dict = validate_and_remap(
        raw, review, clip_id="clip", raw_hash="hash", required_sheets=["frames.jpg"]
    )
    return result


def test_fragment_merge_reuses_pose_and_keeps_missing_frames() -> None:
    raw = raw_tracks()
    raw["detection_rows"][1, 2] = -1
    output = remap(
        raw, decision([segment(3, 0, 2, "player_1"), segment(8, 3, 4, "player_1")])
    )
    assert output["player_ids"].tolist() == ["player_1"]
    assert output["observed"][:, 0].tolist() == [True, True, False, True]
    assert output["raw_track_ids"][:, 0].tolist() == [3, 3, -1, 8]
    assert (output["keypoints"][2] == 0).all()
    assert (output["keypoints"][3, 0] == 2).all()


def test_id_switch_splits_one_track_and_excludes_non_player() -> None:
    raw = raw_tracks()
    raw["detection_rows"][0] = [0, 1, 4, 5]
    output = remap(
        raw,
        decision(
            [
                segment(3, 0, 2, "player_1"),
                segment(3, 2, 4, "player_2"),
                segment(8, 2, 4, None, "non_player"),
            ]
        ),
    )
    assert output["observed"].tolist() == [
        [True, False],
        [True, False],
        [False, True],
        [False, True],
    ]
    assert output["detection_rows"][:, 1].tolist() == [-1, -1, 4, 5]


def test_same_frame_identity_collision_requires_explicit_duplicate() -> None:
    raw = raw_tracks()
    raw["detection_rows"][0] = [0, 1, 4, 5]
    with pytest.raises(ValueError, match="Two raw tracks"):
        remap(
            raw, decision([segment(3, 0, 4, "player_1"), segment(8, 2, 4, "player_1")])
        )
    output = remap(
        raw,
        decision(
            [segment(3, 0, 4, "player_1"), segment(8, 2, 4, "player_1", "duplicate")]
        ),
    )
    assert (output["keypoints"][:, 0] == 1).all()
    with pytest.raises(ValueError, match="duplicate needs"):
        remap(
            raw_tracks(),
            decision(
                [
                    segment(3, 0, 2, "player_1"),
                    segment(8, 2, 4, "player_1", "duplicate"),
                ]
            ),
        )


@pytest.mark.parametrize(
    "change, message",
    [
        ("omit", "exactly one"),
        ("overlap", "overlapping"),
        ("wrong_hash", "identity"),
        ("unknown", "unresolved"),
        ("unseen_sheet", "All frame"),
        ("invent_track", "Invalid track"),
    ],
)
def test_incomplete_or_invented_reviews_are_rejected(change: str, message: str) -> None:
    review = decision([segment(3, 0, 2, "player_1"), segment(8, 2, 4, "player_1")])
    if change == "omit":
        review["segments"].pop()
    elif change == "overlap":
        review["segments"].append(segment(3, 0, 1, "player_1"))
    elif change == "wrong_hash":
        review["raw_tracks_sha256"] = "other"
    elif change == "unknown":
        review["segments"][0].update(role="unknown", player_id=None)
    elif change == "unseen_sheet":
        review["reviewed_sheets"] = []
    else:
        review["segments"][0]["raw_track_id"] = 99
    with pytest.raises(ValueError, match=message):
        remap(raw_tracks(), review)


@pytest.fixture
def store(tmp_path: Path) -> BallFrameStore:
    roots = SourceRoots(tmp_path)
    write_tracknet(roots.data / "tennis/tracknet")
    return BallFrameStore(build_dataset(roots.config({"tracknet": TRACKNET_SOURCE})))


def test_presence_protects_unknowns_excludes_context_and_respects_boundary(
    store: BallFrameStore,
) -> None:
    clip = store.clips[0]
    rows = store.clip_rows(clip)
    a = int(store.frames["inst_start"][rows[0]])
    store.instances["point_kind"][a] = POINT_KIND_CODES["unresolved"]
    result = presence_counts(store, 0.4)[0]
    assert result["presence_rate"] == 0.75 and result["selected"]
    assert result["unresolved_frames"] == 1
    assert not presence_counts(store, 0.75)[0]["selected"]
    store.frames["is_target"][rows[-1]] = False
    assert presence_counts(store, 0.4)[0]["presence_rate"] == 2 / 3
    store.frames["annotated"][rows[0]] = False
    assert (
        presence_counts(store, 1.0)[0]["reason"] == "unreviewed_target_frames_retained"
    )


def test_committed_chunk_corruption_is_not_silently_reused(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    _save_chunk(path, value=np.arange(3))
    assert _cached_chunk(path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Artifact changed"):
        _cached_chunk(path)


@pytest.mark.parametrize("service_tier", [None, "fast"])
def test_real_images_fake_codex_and_publication(
    store: BallFrameStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service_tier: str | None,
) -> None:
    import tomllib

    import torch

    from src.submodules.models import Pose2DResult
    from src.tennis_scene.chat_annotation.player_pose import selected_pose
    from src.tennis_scene.chat_annotation.player_pose.publication import publish

    fake = tmp_path / "fake-codex"
    fake.write_text("""#!/usr/bin/env python3
import sys,json,re
from pathlib import Path
Path('captured-argv.json').write_text(json.dumps(sys.argv))
prompt=sys.stdin.read()
packet=json.loads(Path(re.search(r"packet: ([^\\n]+)",prompt).group(1)).read_text())
segments=[]
for t in packet['tracks']:
    segments.append({'raw_track_id':t['raw_track_id'],'start_frame':t['first_frame'],'stop_frame':t['last_frame']+1,'role':'player','player_id':'player_1','evidence_frames':[t['first_frame']],'reason':'reviewed clothing and motion'})
result={'clip_id':packet['clip']['clip_id'],'raw_tracks_sha256':packet['raw_tracks_sha256'],'status':'approved','reviewed_sheets':packet['required_frame_sheets'],'segments':segments,'notes':''}
Path(sys.argv[sys.argv.index('-o')+1]).write_text(json.dumps(result))
print(json.dumps({'type':'thread.started','thread_id':'fake-test'}))
print(json.dumps({'type':'turn.completed'}))
""")
    fake.chmod(0o755)
    campaign = tmp_path / "campaign"
    asset = tmp_path / "weight"
    asset.write_bytes(b"fake fixture")
    config = {
        "store": str(store.directory),
        "dataset": str(tmp_path / "poses"),
        "project_root": str(tmp_path),
        "presence_threshold": 0.4,
        "assets": {"vitpose": str(asset)},
        "cuda_memory_fraction": 0.5,
        "chunk_frames": 2,
        "codex_binary": str(fake),
        "codex_home": str(tmp_path),
        "model": "gpt-6.1-sol",
        "effort": "max",
        "review_timeout_seconds": 30,
        "review_attempts": 2,
        "review_parallel": 1,
    }
    if service_tier is not None:
        config["service_tier"] = service_tier
    plan = initialize(campaign, config)
    root = clip_root(campaign, 0)
    root.mkdir(parents=True)
    raw = raw_tracks()
    raw.pop("keypoints")
    write_npz(root / "tracks.npz", **raw)
    from src.tasks.ball_detection.data.store import SHARDS_DIR, shard_name

    write_json(
        root / "input.json",
        {"shard_sha256": digest(store.directory / SHARDS_DIR / shard_name(0))},
    )
    write_json(
        root / "generation.json", {"files": {"tracks.npz": digest(root / "tracks.npz")}}
    )
    result = runner.review_clip(campaign, 0)
    assert result["status"] == "approved"
    argv = read_json(root / "review/attempt-000/captured-argv.json")
    overrides = {}
    for flag, value in zip(argv, argv[1:], strict=False):
        if flag == "-c":
            overrides.update(tomllib.loads(value))
    assert overrides["model_reasoning_effort"] == "max"
    assert overrides.get("service_tier") == service_tier
    enabled = [
        value for flag, value in zip(argv, argv[1:], strict=False) if flag == "--enable"
    ]
    assert ("fast_mode" in enabled) == (service_tier == "fast")
    with pytest.raises(RuntimeError, match="not approved"):
        PlayerPoseStore(tmp_path / "poses").read_clip(plan["clips"][0]["clip_id"])
    with pytest.raises(FileNotFoundError):
        publish(campaign, 0)

    class Pose:
        def predict(self, request: object) -> Pose2DResult:
            return Pose2DResult(torch.ones(1, 17, 3))

        def unload(self) -> None:
            pass

    monkeypatch.setenv("TENNIS_RUN_ID", "fixture")
    monkeypatch.setenv("TENNIS_GPU_RESOURCE", "all")
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda _: None)
    monkeypatch.setattr(selected_pose, "pose_model", lambda _: Pose())
    raw_hash = digest(root / "tracks.npz")
    selected_pose.generate_selected_pose(campaign, 0)
    publish(campaign, 0)
    assert digest(root / "tracks.npz") == raw_hash
    loaded = PlayerPoseStore(tmp_path / "poses").read_clip(plan["clips"][0]["clip_id"])
    assert loaded is not None and loaded["player_ids"].tolist() == ["player_1"]
    assert loaded["observed"].all()
    launch = read_json(root / "review/attempt-000/launch.json")
    assert launch["model"] == "gpt-6.1-sol"
    assert launch["effort"] == "max"
    assert launch["service_tier"] == service_tier
    assert len(list((root / "evidence").glob("frames-*.jpg"))) == 1
    path = root / "review/attempt-000/result.json"
    assert (
        accept_review(campaign, 0, path, ["frames-000000-000004.jpg"])["status"]
        == "approved"
    )
    held = read_json(path)
    held["status"] = "needs_review"
    held["segments"][0].update(role="unknown", player_id=None)
    write_json(root / "held.json", held)
    with pytest.raises(ValueError, match="immutable"):
        accept_review(campaign, 0, root / "held.json", ["frames-000000-000004.jpg"])
    assert (
        PlayerPoseStore(tmp_path / "poses").read_clip(plan["clips"][0]["clip_id"])
        is not None
    )


def test_queue_resume_does_not_duplicate_an_owned_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tennis_scene.chat_annotation.player_pose import orchestrator

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    root = clip_root(campaign, 7)
    write_json(root / "queue-attempt-00.json", {"job": "owned.job"})
    config = {"queue_dir": str(tmp_path / "queue"), "generation_attempts": 2}
    plan = {
        "clips": [{"selected": True, "index": 7, "clip_id": "clip", "frame_count": 4}]
    }
    monkeypatch.setattr(orchestrator, "load_campaign", lambda _: (config, plan, None))
    monkeypatch.setattr(
        orchestrator,
        "enqueue_clip",
        lambda *_: pytest.fail("existing job must be resumed"),
    )

    def finished(queue: Path, name: str) -> str:
        assert name == "owned.job"
        artifact = root / "tracks.npz"
        write_npz(artifact, test=np.arange(2))
        write_json(
            root / "generation.json", {"files": {"tracks.npz": digest(artifact)}}
        )
        return "done"

    monkeypatch.setattr(orchestrator, "queue_state", finished)
    orchestrator.generate_through_queue(campaign)
    assert read_json(campaign / "generation_status.json")["status"] == "complete"


def test_queue_stops_after_three_clip_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tennis_scene.chat_annotation.player_pose import orchestrator

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = {"queue_dir": str(tmp_path / "queue"), "generation_attempts": 2}
    plan = {
        "clips": [
            {"selected": True, "index": i, "clip_id": str(i), "frame_count": 4}
            for i in range(4)
        ]
    }
    calls = []
    monkeypatch.setattr(orchestrator, "load_campaign", lambda _: (config, plan, None))

    def enqueue(_: dict, __: Path, index: int, attempt: int) -> str:
        calls.append((index, attempt))
        return f"owned-{index}-{attempt}.job"

    monkeypatch.setattr(orchestrator, "enqueue_clip", enqueue)
    monkeypatch.setattr(orchestrator, "queue_state", lambda *_: "failed")
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Three consecutive"):
        orchestrator.generate_through_queue(campaign)
    assert calls == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
