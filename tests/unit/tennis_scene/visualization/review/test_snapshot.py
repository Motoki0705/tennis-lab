"""Stored masks/lineage, rather than nonzero coordinates, authorize display."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.schema import SceneResult
from src.tennis_scene.visualization.review.gallery import Review
from src.tennis_scene.visualization.review.snapshot import (
    build_snapshot,
    scene_snapshot,
)


def test_snapshot_keeps_3d_masks_and_does_not_publish_meshes(v2_scene: SceneResult) -> None:
    scene = v2_scene
    scene.metadata.update(camera_ids=["cam0", "cam1"], status="partial")
    value = {item.name: getattr(scene, item.name) for item in fields(scene)}
    snapshot = scene_snapshot(value)
    assert snapshot["arrays"]["ball_3d_valid"] == scene.ball_3d_valid.tolist()
    assert snapshot["arrays"]["player_valid"] == scene.player_valid.tolist()
    assert "smpl_vertices_local" not in snapshot["arrays"]
    assert all(gap["end"] > gap["start"] for gap in snapshot["gaps"])


def test_snapshot_rejects_a_reason_that_disagrees_with_validity(v2_scene: SceneResult) -> None:
    scene = v2_scene
    scene.metadata.update(camera_ids=["cam0", "cam1"], status="partial")
    value = {item.name: getattr(scene, item.name) for item in fields(scene)}
    value["ball_rejection_code"] = np.zeros_like(scene.ball_rejection_code)
    with pytest.raises(ValueError, match="zero reason must mean valid"):
        scene_snapshot(value)


def test_source_checksum_mismatch_cannot_be_shown_with_saved_2d(tmp_path: Path) -> None:
    video = tmp_path / "cam0.mp4"
    video.write_bytes(b"changed source")
    review = Review.__new__(Review)
    review.source = {"videos": [{"camera_id": "cam0", "sha256": "wrong"}]}
    review.videos = {"cam0": video}
    with pytest.raises(ValueError, match="Source video checksum mismatch"):
        build_snapshot(review, {}, online=False)


def test_missing_scene_and_rgb_are_explicit_without_fabricated_masks(tmp_path: Path) -> None:
    review = Review.__new__(Review)
    review.document = {"exports": {}}
    review.source = {"clip_id": "clip", "videos": [{"camera_id": "cam0", "path": str(tmp_path / "missing.mp4"), "sha256": "hash", "fps": 30.}]}
    review.index_path = tmp_path / "scene.json"
    review.index_path.write_text("{}")
    review.source_sha256 = "hash"
    review.frame_count = 3
    review.videos = {"cam0": tmp_path / "missing.mp4"}
    review.references = {}
    review.samples = lambda: (0, 2)  # type: ignore[method-assign]
    snapshot = build_snapshot(review, {"scene_assembly": {"status": "missing"}}, online=False)
    assert snapshot["scene"] is None
    assert snapshot["export"]["status"] == "missing"
    assert snapshot["sources"][0]["available"] is False
    assert snapshot["samples"] == {"cam0": {}}
    assert "未生成" in snapshot["scene_reason"]
