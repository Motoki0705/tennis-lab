"""Real JPEG/store + frozen pose + play selection + coordinate model boundary."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.data.play_intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.play_manifest import build_play_manifest
from src.tasks.ball_detection.data.pose_windows import (
    PoseWindowDataset,
    collate_pose_windows,
    coordinate_loss,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDPoseDetector
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.fixture
def frozen(tmp_path: Path) -> tuple[Path, Path]:
    root = write_store_clip(tmp_path / "ball", "tracknet/game1/clip", [
        frame(i, ball(xy=(8., 8.))) if not 10 <= i < 15 else frame(i)
        for i in range(40)
    ], size=(32, 32))
    store = BallFrameStore(root)
    poses = tmp_path / "poses"
    poses.mkdir()
    xy: NDArray[np.float32] = np.full((40, 2, 17, 3), .9, np.float32)
    xy[..., :2] = 12
    valid: NDArray[np.bool_] = np.ones((40, 2), bool)
    valid[10:15, 1] = False
    xy[~valid] = 0
    file = poses / "clip.npz"
    np.savez(file, keypoints=xy, observed=valid, player_ids=np.array(["p1", "p2"]),
             frame_index=np.arange(40), pts=store.frames["pts"])
    manifest = dict(schema="ball_detection_player_poses.v1", coordinate_system="stored_jpeg_pixels",
                    ball_store=dict(directory=str(root), hashes={name: dual_sha256(root / name)
                                    for name in ("metadata.json", "index.npz")}),
                    clips=[dict(clip_id=store.clips[0].clip_id, pose_status="approved",
                                file="clip.npz", sha256=dual_sha256(file))])
    (poses / "manifest.json").write_text(json.dumps(manifest))
    report = build_play_manifest(poses, PlayIntervalConfig())
    frozen_manifest = tmp_path / "windows.json"
    frozen_manifest.write_text(json.dumps(report))
    return frozen_manifest, poses


def test_frozen_dataset_keeps_resolution_alignment_and_masks(frozen: tuple[Path, Path]) -> None:
    manifest, poses = frozen
    dataset = PoseWindowDataset(manifest, split="train")
    # New approvals/changes to the live campaign do not change a frozen experiment.
    (poses / "manifest.json").write_text("{}")
    sample = dataset[0]
    assert sample["rgb"].shape == (32, 3, 32, 32) and sample["rgb"].dtype == torch.uint8
    assert "mdd" not in sample and not RGBToMDD()(sample["rgb"][None])[:, :, 0].any()
    torch.testing.assert_close(sample["uv"][0], torch.tensor([8 / 31, 8 / 31]))
    torch.testing.assert_close(sample["pose"][0, 0, 0], torch.tensor([12 / 31, 12 / 31]))
    assert not sample["position_valid"][10:15].any()
    assert not sample["pose_valid"][10:15, 1].any()
    batch = collate_pose_windows([sample])
    model = MDDPoseDetector(MDDPoseConfig("conv2d", "attention", "query", 32, (4, 4, 8, 8), (8, 8), 16, 2, 1, 0., 10000.))
    output = model(batch["rgb"], batch["pose"], batch["pose_valid"], batch["timestamps"])
    loss = coordinate_loss(output, batch["uv"], batch["position_valid"])
    loss.backward()
    assert torch.isfinite(loss)


def test_changed_pose_bytes_are_rejected(frozen: tuple[Path, Path]) -> None:
    manifest, poses = frozen
    dataset = PoseWindowDataset(manifest, split="train")
    with (poses / "clip.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity"):
        dataset[0]


def test_no_windows_does_not_fall_back_to_another_split(frozen: tuple[Path, Path]) -> None:
    with pytest.raises(ValueError, match="No accepted windows"):
        PoseWindowDataset(frozen[0], split="val")
