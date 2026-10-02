"""Tests for the strict TennisCourtDetector input adapter."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.utils.schema.court import GROUND_COURT_KP_NAMES
from tests.unit.tasks.court_detection.data.inputs.fixtures import write_tennis_records

pytestmark = pytest.mark.unit


def _write_source(root: Path, record: dict[str, object]) -> None:
    write_tennis_records(
        root,
        [{**record, "split": "train"}, {**record, "id": "validation", "split": "val"}],
    )


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


@pytest.mark.parametrize("metric", [True, "0.25", -0.1])
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

    with pytest.raises(ValueError, match="Invalid TennisCourtDetector sparse record"):
        _input(root)


@pytest.mark.parametrize(
    "sample_id",
    [
        "/tmp/outside",
        "../outside",
        "a/b",
        "a\\b",
        "sample.png",
        ".",
        "..",
        " sample",
        "sample ",
        "sample\x00suffix",
    ],
)
def test_annotation_id_must_be_a_portable_filename_stem(
    tmp_path: Path,
    sample_id: str,
) -> None:
    root = tmp_path / "court"
    _write_source(root, _record(id=sample_id))

    with pytest.raises(ValueError, match="portable filename stem"):
        _input(root)


@pytest.mark.parametrize(
    "sample_id", ["sample", "-0M6ixK7aIU_1050", "_vnC7WQazMM_3950"]
)
def test_annotation_accepts_portable_filename_stems(
    tmp_path: Path,
    sample_id: str,
) -> None:
    root = tmp_path / "court"
    _write_source(root, _record(id=sample_id))

    input_layer = _input(root)

    assert input_layer.records("train")[0].sample_id == sample_id


def test_annotation_ids_must_be_unique_across_configured_splits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "court"
    record = _record()
    write_tennis_records(
        root, [{**record, "split": "train"}, {**record, "split": "val"}]
    )

    with pytest.raises(ValueError, match="unique across configured splits"):
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
