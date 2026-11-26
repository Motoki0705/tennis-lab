"""Score the exact frame identities admitted by a checksum-pinned cohort."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.evaluation.player_pose.bundle import (
    MANIFEST_SCHEMA,
    load_reference_frames,
)
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey
from src.utils.checksum import dual_sha256
from src.utils.schema.player import COCO_KP_NAMES


def test_open_keeps_empty_images_and_rejects_changed_files(tmp_path: Path) -> None:
    path = tmp_path / "pseudo_gt.npz"
    uv = np.full((1, 1, 3, 17, 2), np.nan)
    uv[0, 0, 0, :2] = [(5.0, 10.0), (15.0, 30.0)]
    np.savez(
        path,
        keypoints_px=uv,
        boxes_xyxy=np.array([[[[4, 8, 16, 32], [0, 0, 1, 1], [0, 0, 1, 1]]]]),
        in_image=np.isfinite(uv).all(-1),
        box_reasons=np.array([[[0, 1, 2]]]),
        joint_reasons=np.where(np.isfinite(uv).all(-1), 0, 8),
        player_ids=[21],
        camera_ids=["cam0"],
        evaluation_frames=[0, 2],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "joint_order": COCO_KP_NAMES,
                "reference_policy": {"min_box_joints": 2},
                "clips": [
                    {
                        "clip_id": "video/clip",
                        "camera_ids": ["cam0"],
                        "player_ids": [21],
                        "evaluation_frames": [0, 2],
                        "source_frames": 3,
                        "width": 100,
                        "height": 80,
                        "pseudo_ground_truth": {
                            "path": path.name,
                            "sha256": dual_sha256(path),
                        },
                    }
                ],
            }
        )
    )
    digest = dual_sha256(manifest)
    frames = load_reference_frames(tmp_path, expected_sha256=digest, box_margin=0.2)
    first, last = FrameKey("video/clip", "cam0", 0), FrameKey("video/clip", "cam0", 2)
    assert list(frames) == [first, last]
    assert frames[first].player_ids.tolist() == [21]
    assert frames[first].boxes_xyxy[0] == pytest.approx([3, 6, 17, 34])
    assert int(frames[first].joint_valid.sum()) == 2
    assert len(frames[last].player_ids) == 0
    with pytest.raises(ValueError, match="SHA-256"):
        load_reference_frames(tmp_path, expected_sha256="0" * 64)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="pseudo_gt.npz"):
        load_reference_frames(tmp_path, expected_sha256=digest)
