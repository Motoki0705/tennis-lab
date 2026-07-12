"""Tests for src/tennis_scene/clip_studio/paths.py."""

from pathlib import Path

import pytest

from src.tennis_scene.clip_studio.paths import standard_clip_studio_paths


def test_standard_clip_studio_paths_follow_dataset_layout() -> None:
    paths = standard_clip_studio_paths(Path("data/tennis_multivew"), "match01")

    assert paths.video_paths == (
        Path("data/tennis_multivew/raw/match01/cam0.mp4"),
        Path("data/tennis_multivew/raw/match01/cam1.mp4"),
        Path("data/tennis_multivew/raw/match01/cam2.mp4"),
    )
    assert paths.project_path == Path(
        "data/tennis_multivew/processed/match01/project.json"
    )
    assert paths.dataset_dir == Path(
        "data/tennis_multivew/processed/match01/dataset"
    )


@pytest.mark.parametrize("match_id", ["", ".", "..", "group/match01"])
def test_standard_clip_studio_paths_reject_invalid_match_id(match_id: str) -> None:
    with pytest.raises(ValueError, match="single path component"):
        standard_clip_studio_paths(Path("data/tennis_multivew"), match_id)
