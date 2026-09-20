import json
from typing import TypedDict

import numpy as np
import pytest

from src.tennis_scene.dataset_pipeline.legacy import (
    copy_legacy_broadcast,
    filter_saved_ball,
)
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.reference_pipeline.observations import sha256


def test_spike_filter_requires_both_neighbours_and_preserves_real_displacement():
    uv = np.array(
        [[[0.1, 0.2], [0.11, 0.2], [0.8, 0.8], [0.12, 0.2], [0.4, 0.4], [0.5, 0.5]]],
        np.float32,
    )
    valid, report = filter_saved_ball(
        uv, np.ones((1, 6), bool), isolated_jump=0.12, neighbor_distance=0.08
    )
    np.testing.assert_array_equal(valid, [[True, True, False, True, True, True]])
    assert report["rejected_spikes"] == 1
    missing: np.ndarray = np.ones((1, 6), bool)
    missing[0, 1] = False
    valid, _ = filter_saved_ball(
        uv, missing, isolated_jump=0.12, neighbor_distance=0.08
    )
    assert valid[0, 2]


def test_legacy_import_is_separate_version_and_rejects_changed_sources(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    clip = source / "clips/compilation/clip_000"
    (clip / "media").mkdir(parents=True)
    (clip / "media/cam0.mp4").write_bytes(b"unchanged source media")
    annotation = clip / "annotations/tennis_scene"
    annotation.mkdir(parents=True)
    raw = {
        "version": 1,
        "clip_id": "compilation/clip_000",
        "recording_id": "compilation",
        "clip_name": "clip_000",
        "fps": 30.0,
        "num_frames": 4,
        "width": 640,
        "height": 360,
        "global_start_sec": 0.0,
        "global_end_sec": 4 / 30,
        "camera_ids": ["cam0"],
        "video_paths": ["media/cam0.mp4"],
        "cameras": [],
        "sync_source": "test",
        "exported_at": "test",
    }
    (clip / "clip.json").write_text(json.dumps(raw))
    (annotation / "annotation.json").write_text(
        json.dumps({"clip_manifest_sha256": sha256(clip / "clip.json")})
    )
    np.savez(
        annotation / "scene.npz",
        ball_uv=np.full((1, 4, 2), 0.4, np.float32),
        ball_vis=np.ones((1, 4), bool),
    )
    (source / "dataset.json").write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "test",
                "updated_at": "test",
                "clips": [
                    {
                        "clip_id": "compilation/clip_000",
                        "path": "clips/compilation/clip_000",
                    }
                ],
            }
        )
    )
    class ImportOptions(TypedDict):
        dataset_id: str
        video_groups: dict[str, str]
        isolated_jump: float
        neighbor_distance: float

    kwargs: ImportOptions = {
        "dataset_id": "broadcast_v2",
        "video_groups": {"compilation/clip_000": "match_0"},
        "isolated_jump": 0.12,
        "neighbor_distance": 0.08,
    }
    original = (clip / "clip.json").read_bytes()
    copy_legacy_broadcast(source, destination, **kwargs)
    index = load_dataset_manifest(destination)
    imported = ClipManifest.load(destination / index.clips["match_0/clip_000"].path)
    assert imported.video_id == "match_0"
    assert (clip / "clip.json").read_bytes() == original
    assert imported.media_path("cam0").samefile(clip / "media/cam0.mp4")
    assert not (imported.clip_dir / "annotations/tennis_scene/scene.npz").exists()
    copy_legacy_broadcast(source, destination, **kwargs)
    np.savez(
        annotation / "scene.npz",
        ball_uv=np.full((1, 4, 2), 0.5, np.float32),
        ball_vis=np.ones((1, 4), bool),
    )
    with pytest.raises(ValueError, match="recipe changed"):
        copy_legacy_broadcast(source, destination, **kwargs)
