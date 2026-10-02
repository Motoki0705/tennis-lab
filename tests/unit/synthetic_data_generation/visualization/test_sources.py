"""Selection and source-order tests for canonical visualization readers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import src.synthetic_data_generation.visualization.sources as sources_module
from src.synthetic_data_generation.visualization.sources import CourtVisualizationSource


def _write_court_fixture(
    root: Path,
    *,
    indices: tuple[int, ...],
    dataset_schema: str = "canonical_court_dataset_v1",
    label_schema: str | None = "canonical_court_sample_v1",
) -> None:
    records = []
    for sample_index, frame_index in enumerate(indices):
        directory = root / "samples" / f"sample-{sample_index}"
        directory.mkdir(parents=True)
        rgb: NDArray[np.float32] = np.full(
            (32, 48, 3),
            fill_value=sample_index / 10.0,
            dtype=np.float32,
        )
        np.save(directory / "rgb.npy", rgb, allow_pickle=False)
        projection: dict[str, object] = {"courts": []}
        labels = {
            "sample_id": f"sample-{sample_index}",
            "view_id": "view-0",
            "trajectory_frame_index": frame_index,
            "projection": projection,
        }
        if label_schema is not None:
            labels["schema"] = label_schema
        (directory / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        records.append(
            {
                "sample_id": f"sample-{sample_index}",
                "trajectory_id": "orbit-0",
                "view_id": "view-0",
                "trajectory_frame_index": frame_index,
                "width": 48,
                "height": 32,
                "rgb": f"samples/sample-{sample_index}/rgb.npy",
                "labels": f"samples/sample-{sample_index}/labels.json",
                "projection": projection,
            }
        )
    payload = {
        "schema": dataset_schema,
        "scene_id": "scene-0",
        "trajectory_groups": [
            {
                "trajectory": {"trajectory_id": "orbit-0"},
                "views": [{"view_id": "view-0"}],
                "sample_count": 2,
            }
        ],
        "samples": records,
    }
    (root / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")


def test_court_source_streams_selected_trajectory_in_exact_frame_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_court_fixture(tmp_path, indices=(0, 1))
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    source = CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")
    frames = tuple(source.frames())

    assert tuple(frame.trajectory_frame_index for frame in frames) == (0, 1)
    assert tuple(frame.sample_id for frame in frames) == ("sample-0", "sample-1")
    assert frames[1].rgb[0, 0, 0] == pytest.approx(0.1)


def test_court_source_fails_closed_on_unknown_id_or_reordered_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_court_fixture(tmp_path, indices=(1, 0))
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    with pytest.raises(KeyError, match="Unknown Court trajectory_id"):
        CourtVisualizationSource(tmp_path, trajectory_id="missing")
    with pytest.raises(ValueError, match="source-frame ordering"):
        CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")


@pytest.mark.parametrize(
    "label_schema",
    [
        None,
        "canonical_court_sample_v1",
        "canonical_court_sample_v2",
        "canonical_court_sample_v3",
    ],
)
def test_v2_court_source_rejects_missing_or_mixed_sample_schema_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label_schema: str | None,
) -> None:
    _write_court_fixture(
        tmp_path,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v2",
        label_schema=label_schema,
    )
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )
    source = CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")

    if label_schema == "canonical_court_sample_v2":
        assert tuple(source.frames())
    else:
        with pytest.raises(ValueError, match="labels schema changed"):
            tuple(source.frames())


def test_court_source_rejects_unknown_dataset_schema_without_shape_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_court_fixture(
        tmp_path,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v4",
        label_schema="canonical_court_sample_v2",
    )
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    with pytest.raises(
        ValueError,
        match=r"^Unknown Court dataset schema: 'canonical_court_dataset_v4'\.$",
    ):
        CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")
