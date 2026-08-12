"""Selection and source-order tests for canonical visualization readers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import src.synthetic_data_generation.visualization.sources as sources_module
from src.synthetic_data_generation.dataset.blcs.contracts import BLCSSampleRecord
from src.synthetic_data_generation.dataset.runtime import (
    ChunkWriter,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
)
from src.synthetic_data_generation.visualization.sources import (
    BLCSVisualizationSource,
    CourtVisualizationSource,
)


def _write_court_fixture(root: Path, *, indices: tuple[int, ...]) -> None:
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
        "schema": "canonical_court_dataset_v1",
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


def test_blcs_stream_rejects_chunk_replaced_by_a_foreign_attempt(
    tmp_path: Path,
) -> None:
    writer = ChunkWriter(
        tmp_path / "chunks",
        attempt_token="foreign-attempt",
        camera_ids=("camera-0",),
        width=2,
        height=2,
    )
    chunk = writer.write(
        ForegroundDeltaBatch(
            chunk_id="chunk-000000",
            deltas=(
                ForegroundDelta(
                    key=RenderSampleKey(0, "camera-0"),
                    pixel_indices=np.asarray([0], dtype=np.int32),
                    rgb=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                    alpha=np.asarray([1.0], dtype=np.float32),
                    depth=np.asarray([1.0], dtype=np.float32),
                    instance_ids=np.asarray([1], dtype=np.int32),
                ),
            ),
            metadata=({},),
        )
    )
    source = object.__new__(BLCSVisualizationSource)
    source.root = tmp_path
    source.logical_scene_id = "trajectory-0"
    source.camera_id = "camera-0"
    source._attempt_token = "selected-attempt"
    source._records = (
        BLCSSampleRecord(
            trajectory_id="trajectory-0",
            split="train",
            global_frame_index=0,
            source_frame_index=0,
            chunk_index=0,
            camera_id="camera-0",
            background_store="backgrounds",
            foreground_chunk=chunk.directory.relative_to(tmp_path).as_posix(),
            chunk_sample_index=0,
        ),
    )

    with pytest.raises(ValueError, match="another stage attempt"):
        next(source.frames())
