import shutil
from pathlib import Path

import numpy as np
import pytest

from src.tasks.ball_detection.data.store import SHARDS_DIR, BallFrameStore, shard_name
from src.tennis_scene.chat_annotation.player_pose.dataset import (
    PlayerPoseStore,
    update_entry,
)
from src.tennis_scene.chat_annotation.player_pose.expansion import (
    plan_expansion,
    reuse_approved,
)
from src.tennis_scene.chat_annotation.player_pose.reviews import validate_and_remap
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
from tests.unit.tennis_scene.chat_annotation.player_pose.test_campaign import (
    decision,
    raw_tracks,
    segment,
)
from tests.unit.tennis_scene.chat_annotation.player_pose.test_campaign import (
    store as store,
)


def legacy_fixture(store: BallFrameStore, tmp_path: Path) -> Path:
    source = tmp_path / "legacy-poses"
    campaign = tmp_path / "legacy-campaign"
    initialize(
        campaign,
        {
            "store": str(store.directory),
            "dataset": str(source),
            "project_root": str(tmp_path),
            "presence_threshold": 0.4,
            "assets": {},
        },
    )
    root = clip_root(campaign, 0)
    raw = raw_tracks()
    write_npz(root / "tracks.npz", **raw)
    write_json(
        root / "input.json",
        {"shard_sha256": digest(store.directory / SHARDS_DIR / shard_name(0))},
    )
    write_json(
        root / "generation.json",
        {"files": {name: digest(root / name) for name in ("tracks.npz", "input.json")}},
    )
    write_json(root / "evidence/packet.json", {"required_frame_sheets": ["frames.jpg"]})
    review = decision([segment(3, 0, 2, "player_1"), segment(8, 2, 4, "player_1")])
    review.update(
        clip_id=store.clips[0].clip_id, raw_tracks_sha256=digest(root / "tracks.npz")
    )
    arrays = validate_and_remap(
        raw,
        review,
        clip_id=store.clips[0].clip_id,
        raw_hash=review["raw_tracks_sha256"],
        required_sheets=["frames.jpg"],
    )
    write_npz(source / "clips/approved.npz", **arrays)
    write_json(source / "reviews/approved.json", review)
    update_entry(
        source,
        0,
        pose_status="approved",
        file="clips/approved.npz",
        sha256=digest(source / "clips/approved.npz"),
        review_file="reviews/approved.json",
        review_sha256=digest(source / "reviews/approved.json"),
        raw_tracks_sha256=review["raw_tracks_sha256"],
    )
    return source


def test_expansion_compares_identity_and_republishes_verified_legacy_bytes(
    store: BallFrameStore, tmp_path: Path
) -> None:
    source = legacy_fixture(store, tmp_path)
    original_hash = digest(source / "manifest.json")
    target = tmp_path / "expanded-store"
    shutil.copytree(store.directory, target)
    metadata = read_json(target / "metadata.json")
    metadata["clips"][1]["annotation_sha256"] = "changed-annotation"
    metadata["clips"][2]["clip_id"] = "added-clip"
    write_json(target / "metadata.json", metadata)
    campaign = tmp_path / "expansion"
    dataset = tmp_path / "expanded-poses"
    plan = initialize(
        campaign,
        {
            "store": str(target),
            "dataset": str(dataset),
            "project_root": str(tmp_path),
            "presence_threshold": 0.4,
            "assets": {},
            "reuse_dataset": str(source),
        },
    )
    assert plan["expansion"] == {
        "existing": 1,
        "changed": 1,
        "added": 1,
        "reuse": 1,
        "generate": 2,
        "skip": 0,
    }
    reuse_approved(campaign)
    reuse_approved(campaign)
    result = PlayerPoseStore(dataset)
    clip_id = store.clips[0].clip_id
    arrays = result.read_clip(clip_id)
    assert arrays is not None
    legacy = PlayerPoseStore(source).read_clip(clip_id)
    assert legacy is not None
    assert all(np.array_equal(arrays[k], legacy[k]) for k in arrays)
    assert result.manifest["ball_store"]["hashes"]["metadata.json"] == digest(
        target / "metadata.json"
    )
    assert (
        result.clips[clip_id]["provenance"]["generation_mode"]
        == "legacy_pose_before_review.v1"
    )
    assert digest(source / "manifest.json") == original_hash
    with pytest.raises(RuntimeError, match="not approved"):
        result.read_clip("added-clip")


@pytest.mark.parametrize("change", ["pts", "pixels", "review"])
def test_legacy_reuse_rejects_changed_timeline_pixels_or_review(
    store: BallFrameStore, tmp_path: Path, change: str
) -> None:
    source = legacy_fixture(store, tmp_path)
    target = tmp_path / "new-store"
    shutil.copytree(store.directory, target)
    new_store = BallFrameStore(target)
    if change == "pts":
        new_store.frames["pts"][0] = -1
    elif change == "pixels":
        shard = target / SHARDS_DIR / shard_name(0)
        shard.write_bytes(shard.read_bytes() + b"changed")
    else:
        (source / "reviews/approved.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch|changed"):
        plan_expansion(
            {"reuse_dataset": str(source)}, presence_counts(new_store, 0.4), new_store
        )


def test_presence_skip_is_preserved_even_for_approved_legacy(
    store: BallFrameStore, tmp_path: Path
) -> None:
    source = legacy_fixture(store, tmp_path)
    entries = presence_counts(store, 0.75)
    summary = plan_expansion({"reuse_dataset": str(source)}, entries, store)
    assert summary["reuse"] == 0 and summary["skip"] == 3
