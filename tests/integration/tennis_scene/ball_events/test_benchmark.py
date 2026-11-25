"""Read a real scene index/archive and write the downstream prediction JSON."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.ball_events import detect_scene_events
from src.tennis_scene.ball_events.__main__ import run_benchmark
from src.tennis_scene.schema import SceneResult
from tests.support.tennis_scene.annotations import publish_scene_to_clip_store


def _doubles_scene() -> SceneResult:
    players, frames, cameras, joints = 4, 3, 2, 17
    xyz = np.array([[0, -4.977, 1.4], [0, -5, 1.4], [0, -4.78, 1.4]], np.float32)
    wrists: NDArray[np.float32] = np.zeros((players, frames, joints, 3), np.float32)
    wrists[3, 1, 10] = [0.4, -5, 1.4]
    joints_valid: NDArray[np.bool_] = np.zeros((players, frames, joints), bool)
    joints_valid[3, 1, 10] = True
    return SceneResult(
        num_frames=frames,
        fps=30,
        width=640,
        height=360,
        court_kp=np.zeros((cameras, frames, 14, 2), np.float32),
        court_vis=np.ones((cameras, frames, 14), np.float32),
        player_position=np.zeros((players, frames, 3), np.float32),
        player_yaw=np.zeros((players, frames), np.float32),
        ball_uv=np.zeros((cameras, frames, 2), np.float32),
        ball_vis=np.ones((cameras, frames), bool),
        ball_3d=xyz,
        human_kp_2d=np.zeros((players, cameras, frames, joints, 2), np.float32),
        human_kp_vis=np.ones((players, cameras, frames, joints), np.float32),
        player_track_ids=np.array([12, 3, 24, 9], np.int32),
        player_kp_3d=wrists,
        metadata={"scene_schema_version": 2, "court_reference": {}},
        player_observed=np.ones((players, frames), bool),
        player_valid=np.zeros((players, frames), bool),
        player_heading_valid=np.zeros((players, frames), bool),
        player_kp_3d_vis=joints_valid,
        player_smpl_valid=np.zeros((players, frames), bool),
        ball_3d_valid=np.ones(frames, bool),
        player_rejection_code=np.ones((players, frames), np.uint8),
        player_kp_3d_rejection_code=(~joints_valid).astype(np.uint8),
        ball_rejection_code=np.zeros(frames, np.uint8),
    )


def test_player_masks_distinguish_wrist_root_and_ball_only_shot() -> None:
    scene = _doubles_scene()
    [wrist] = detect_scene_events(scene)
    assert (wrist.kind, wrist.frame, wrist.flags) == ("shot", 1, ())
    assert wrist.features["player_track_id"] == 9
    assert wrist.features["player_landmark"] == "right_wrist"
    assert wrist.features["player_distance_m"] == pytest.approx(0.4)

    assert scene.player_kp_3d_vis is not None
    scene.player_kp_3d_vis[:] = False
    assert detect_scene_events(scene) == []  # Gentle impulse needs a nearby player.

    assert scene.player_valid is not None
    scene.player_valid[2, 1] = True
    scene.player_position[2, 1] = [0, -4.5, 0.95]
    [root] = detect_scene_events(scene)
    assert root.features["player_track_id"] == 24
    assert root.features["player_landmark"] == "root"

    assert scene.ball_3d is not None
    scene.ball_3d[2, 1] = -4.4
    scene.player_position[2, 1] = [0, 4.5, 0.95]
    [distant] = detect_scene_events(scene)
    assert distant.flags == ("player_far",)
    assert distant.features["player_track_id"] == 24


def test_benchmark_records_success_and_pinned_scene_failure(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    first_id, second_id = "video_000/clip_003", "video_000/clip_004"
    export = publish_scene_to_clip_store(root / first_id, first_id, _doubles_scene())
    scene_index = export.parents[2] / "scene.json"
    scene_sha = sha256(scene_index.read_bytes()).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "tennis_multiview_benchmark_v1",
                "benchmark_id": "example",
                "totals": {"clips": 2},
                "clips": [
                    {
                        "clip_id": clip_id,
                        "frames": 3,
                        "fps": 30,
                        "scene": {
                            "path": str(scene_index.relative_to(root)),
                            "sha256": digest,
                        },
                    }
                    for clip_id, digest in (
                        (first_id, scene_sha),
                        (second_id, "0" * 64),
                    )
                ],
            }
        )
    )
    manifest_sha = sha256(manifest.read_bytes()).hexdigest()
    out = tmp_path / "outputs"
    method_version = "a" * 40
    with pytest.raises(ValueError, match="manifest SHA-256"):
        run_benchmark(root, "f" * 64, out, method_version=method_version)
    assert not out.exists()
    assert not run_benchmark(root, manifest_sha, out, method_version=method_version)

    prediction = json.loads((out / f"predictions/{first_id}.json").read_text())
    assert prediction["schema"] == "ball_event_predictions.v1"
    assert prediction["clip_id"] == first_id
    assert prediction["status"] == "ok" and prediction["reason"] is None
    assert (
        prediction["method"] == "rule"
        and prediction["method_version"] == method_version
    )
    assert prediction["inputs"] == {"scene_sha256": scene_sha}
    [shot] = prediction["events"]
    assert (shot["kind"], shot["frame"], shot["flags"]) == ("shot", 1, [])
    assert shot["time_s"] == pytest.approx(1 / 30)
    assert shot["features"]["player_track_id"] == 9

    failed = json.loads((out / f"predictions/{second_id}.json").read_text())
    assert failed["status"] == "failed"
    assert "Scene index SHA-256 mismatch" in failed["reason"]
    assert failed["events"] == []
    progress = json.loads((out / "progress.json").read_text())
    assert (
        progress["state"],
        progress["total"],
        progress["ok"],
        progress["failed"],
        progress["pending"],
    ) == (
        "failed",
        2,
        1,
        1,
        0,
    )
    assert progress["clips"][first_id]["events"] == {"shot": 1, "bounce": 0, "net": 0}
    assert progress["clips"][second_id]["reason"] == failed["reason"]
