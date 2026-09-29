from pathlib import Path

import pytest

from src.tasks.player_association.evaluation.dataset_labels import (
    discover_labels,
    label_path,
)
from src.tasks.player_association.evaluation.labels import ClipLabels


def test_dataset_owned_labels_are_discovered_and_misplaced_ids_rejected(tmp_path: Path) -> None:
    path = label_path(tmp_path, "video_000/clip_001")
    labels = ClipLabels("video_000/clip_001", 10, (), {}, {})
    labels.save(path)
    assert discover_labels(tmp_path) == (path,)
    labels.save(label_path(tmp_path, "video_000/clip_002"))
    with pytest.raises(ValueError, match="disagrees"):
        discover_labels(tmp_path)


def test_empty_dataset_does_not_fall_back_to_repository_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No player association labels"):
        discover_labels(tmp_path)


@pytest.mark.parametrize("clip_id", ["../clip", "video/..", "/absolute", "video/clip/extra"])
def test_clip_id_cannot_escape_dataset(tmp_path: Path, clip_id: str) -> None:
    with pytest.raises(ValueError, match="video/clip"):
        label_path(tmp_path, clip_id)
