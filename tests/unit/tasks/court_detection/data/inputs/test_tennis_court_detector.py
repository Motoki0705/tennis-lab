"""Tests for the strict TennisCourtDetector input adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest
from PIL import Image

from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)
from src.utils.schema.court import GROUND_COURT_KP_NAMES

pytestmark = pytest.mark.unit


def _write_source(root: Path, record: dict[str, object]) -> None:
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(root / "images" / "sample.png")
    Image.new("RGB", (32, 24)).save(root / "images" / "validation.png")
    validation = {**record, "id": "validation"}
    (root / "data_train.json").write_text(json.dumps([record]), encoding="utf-8")
    (root / "data_val.json").write_text(json.dumps([validation]), encoding="utf-8")


def _input(
    root: Path,
    *,
    excluded_sample_ids: tuple[str, ...] = (),
) -> TennisCourtDetectorInput:
    return TennisCourtDetectorInput(
        TennisCourtDetectorSourceConfig(
            kind="tennis_court_detector",
            root=root,
            split_mapping=MappingProxyType(
                {"train": "train", "val": "val", "test": None}
            ),
            excluded_sample_ids=excluded_sample_ids,
        ),
        target_store=CourtDerivedTargetStore(root.parent / "derived"),
    )


def _record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "sample",
        "kps": [[float(index + 1), float(index + 2)] for index in range(14)],
        "metric": 0.25,
    }
    record.update(updates)
    return record


def test_real_annotation_metadata_preserves_canonical_kp14_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "court"
    _write_source(root, _record())

    input_layer = _input(root)
    record = input_layer.records("train")[0]
    sample = input_layer.load(record)

    assert input_layer.spec.keypoint_channel_names == GROUND_COURT_KP_NAMES
    assert record.payload["annotation_metric"] == 0.25
    assert sample.keypoint_channels is not None
    assert sample.keypoint_channels.channel_names == GROUND_COURT_KP_NAMES
    assert sample.metadata.provenance["annotation_metric"] == 0.25


@pytest.mark.parametrize("metric", [True, "0.25", -0.1, float("inf"), None])
def test_annotation_metric_must_be_finite_non_negative_number(
    tmp_path: Path,
    metric: object,
) -> None:
    root = tmp_path / "court"
    _write_source(root, _record(metric=metric))

    with pytest.raises(ValueError, match="metric must be"):
        _input(root)


def test_annotation_rejects_unknown_record_keys(tmp_path: Path) -> None:
    root = tmp_path / "court"
    _write_source(root, _record(unexpected="value"))

    with pytest.raises(ValueError, match="only optional metric"):
        _input(root)


def test_configured_sample_quarantine_must_match_exactly_one_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "court"
    _write_source(root, _record())

    input_layer = _input(root, excluded_sample_ids=("sample",))

    assert input_layer.records("train") == ()
    assert len(input_layer.records("val")) == 1

    with pytest.raises(ValueError, match="must match exactly one"):
        _input(root, excluded_sample_ids=("missing",))
