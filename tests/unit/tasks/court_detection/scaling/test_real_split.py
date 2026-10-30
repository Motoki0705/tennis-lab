"""Video leakage, source immutability and registry drift rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tasks.court_detection.configuration import TennisCourtEvaluationSourceConfig
from src.tasks.court_detection.scaling.real_split import (
    TennisCourtEvaluationInput,
    make_registry,
    real_source,
    video_id,
)
from tests.unit.tasks.court_detection.data.inputs.fixtures import write_tennis_records

pytestmark = pytest.mark.unit


def test_video_registry_unites_old_splits_without_modifying_store(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    write_tennis_records(
        root,
        [
            {"id": sample, "split": split, "kps": [[x, x] for x in range(14)]}
            for sample, split in [
                ("video_a_1", "train"),
                ("video_a_2", "val"),
                ("video_b_1", "train"),
                ("video_c_1", "val"),
                ("video_d_1", "train"),
                ("QszoUKyCOHo_600", "train"),
            ]
        ],
    )
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    source = real_source(root)
    registry = make_registry(source, version="real-video-v1", seed=42)
    assert registry == make_registry(source, version="real-video-v1", seed=42)
    assert registry["video_overlap"] == 0
    assert sum(split["images"] for split in registry["summary"].values()) == 5
    assert [split["videos"] for split in registry["summary"].values()] == [2, 2]
    a_split = next(
        split for split, ids in registry["splits"].items() if "video_a_1" in ids
    )
    assert "video_a_2" in registry["splits"][a_split]
    assert registry["summary"][a_split]["original_split_images"]["train"] >= 1
    assert registry["summary"][a_split]["original_split_images"]["val"] >= 1
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    evaluation = TennisCourtEvaluationInput(
        TennisCourtEvaluationSourceConfig("tennis_court_evaluation", root, path)
    )
    assert evaluation.available_splits == ("val", "test")
    for split in evaluation.available_splits:
        for record in evaluation.records(split):
            assert record.split == split
            assert evaluation.load(record).image.size == (32, 24)
    with pytest.raises(ValueError, match="never"):
        evaluation.records("train")
    assert before == {file: file.read_bytes() for file in before}
    assert len(source.records("train")) == 3 and len(source.records("val")) == 2
    registry["splits"]["test"].append(registry["splits"]["val"][0])
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="registry"):
        TennisCourtEvaluationInput(
            TennisCourtEvaluationSourceConfig("tennis_court_evaluation", root, path)
        )


@pytest.mark.parametrize("sample", ["no_frame", "_123", "video_-1", "video_１２"])
def test_video_id_rejects_ambiguous_sample_ids(sample: str) -> None:
    with pytest.raises(ValueError, match="video ID"):
        video_id(sample)
