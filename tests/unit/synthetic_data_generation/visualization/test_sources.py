"""Selection and source-order tests for canonical visualization readers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import src.synthetic_data_generation.visualization.sources as sources_module
from src.synthetic_data_generation.dataset.blcs.contracts import BLCSSampleRecord
from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.runtime import (
    ChunkWriter,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
)
from src.synthetic_data_generation.visualization.sources import (
    BLCSVisualizationSource,
    CourtVisualizationSource,
    PLCSVisualizationSource,
)
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


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


def test_plcs_visualization_rejects_v4_before_reading_any_payload(
    tmp_path: Path,
) -> None:
    for directory in ("backgrounds", "scenes", "diagnostics"):
        (tmp_path / directory).mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "tennis_plcs_compact_dataset_v4",
                "scene_id": "B00",
                "domain": "plcs",
                "frame_inventory": {},
                "target_courts": [],
                "metadata": {},
                "diagnostics": [],
                "storage": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported canonical compact PLCS"):
        PLCSVisualizationSource(
            tmp_path,
            logical_scene_id="B00",
            camera_id="camera-0",
        )


@pytest.mark.parametrize("mutation", ["missing", "malformed", "unknown", "mismatched"])
def test_plcs_visualization_rejects_invalid_normalization_before_payloads(
    tmp_path: Path,
    mutation: str,
) -> None:
    for directory in ("backgrounds", "scenes", "diagnostics"):
        (tmp_path / directory).mkdir()
    contract: object = court_coordinate_normalization_metadata()
    if mutation == "malformed":
        contract = "isotropic_half_length"
    elif mutation in {"unknown", "mismatched"}:
        assert isinstance(contract, dict)
        contract = deepcopy(contract)
        if mutation == "unknown":
            contract["identity"] = "anisotropic"
        else:
            contract["scale_xyz_m"] = [5.485, 11.885, 1.07]
    metadata = {
        "coordinate_contract": {},
        "court_coordinate_normalization": contract,
        "seed": 0,
        "logical_scene_count": 1,
        "aggregate_global_frame_count": 1,
        "aggregate_source_frame_count": 1,
        "required_motion_categories": [],
        "accepted_court_instance_ids": [],
        "logical_scenes": [],
    }
    if mutation == "missing":
        del metadata["court_coordinate_normalization"]
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "schema": PLCS_DATASET_SCHEMA,
                "scene_id": "B00",
                "domain": "plcs",
                "frame_inventory": {},
                "target_courts": [],
                "metadata": metadata,
                "diagnostics": [],
                "storage": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible|unknown|mismatched"):
        PLCSVisualizationSource(
            tmp_path,
            logical_scene_id="B00",
            camera_id="camera-0",
        )
