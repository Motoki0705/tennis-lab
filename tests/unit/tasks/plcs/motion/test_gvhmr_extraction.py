"""Configuration tests for foreground-player GVHMR extraction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from src.submodules.configuration import GvhmrDemoConfig
from src.tasks.plcs.motion.gvhmr_extraction import (
    COLLECTION_SCHEMA_VERSION,
    GvhmrMotionExtractor,
    GvhmrSelectionConfig,
    normalize_vitpose_confidence,
    require_extraction_space,
)


def test_storage_guard_retains_existing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "completed.motion.npz"
    artifact.write_bytes(b"completed")
    monkeypatch.setattr(
        "src.tasks.plcs.motion.gvhmr_extraction.shutil.disk_usage",
        lambda path: SimpleNamespace(free=1024**3),
    )
    with pytest.raises(OSError, match="Insufficient GVHMR output storage"):
        require_extraction_space(tmp_path, 1000)
    assert artifact.read_bytes() == b"completed"
    monkeypatch.setattr(
        "src.tasks.plcs.motion.gvhmr_extraction.shutil.disk_usage",
        lambda path: SimpleNamespace(free=2 * 1024**3),
    )
    require_extraction_space(tmp_path, 1000)


def test_meiji_selection_assigns_one_distinct_near_player_per_camera() -> None:
    path = (
        Path(__file__).parents[5]
        / "src/tasks/plcs/configs/gvhmr_motion/meiji_3cam.yaml"
    )
    config = GvhmrSelectionConfig.load(path)

    assert config.dataset_id == "meiji_3cam"
    assert [camera.camera_id for camera in config.cameras] == ["cam1", "cam2"]
    assert [camera.player_role for camera in config.cameras] == [
        "near_player_cam1",
        "near_player_cam2",
    ]
    assert all(
        len(camera.footpoint_polygon_normalized) == 4 for camera in config.cameras
    )
    assert [camera.camera_id for camera in config.select_cameras(("cam2",))] == ["cam2"]
    with pytest.raises(ValueError, match="Unknown requested camera IDs"):
        config.select_cameras(("cam0",))


def test_selection_rejects_duplicate_camera_ids(tmp_path: Path) -> None:
    path = tmp_path / "selection.yaml"
    path.write_text(
        """schema_version: plcs_gvhmr_selection_v1
dataset_id: fixture
cameras:
  - camera_id: cam1
    player_role: first
    footpoint_polygon_normalized: [[0, 0], [1, 0], [0, 1]]
  - camera_id: cam1
    player_role: second
    footpoint_polygon_normalized: [[0, 0], [1, 0], [0, 1]]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="camera_id values must be unique"):
        GvhmrSelectionConfig.load(path)


def test_vitpose_confidence_clip_is_explicit_and_counted() -> None:
    raw = np.asarray([[0.2, 1.0, 1.019]], dtype=np.float32)

    normalized, clipped_count = normalize_vitpose_confidence(raw)

    np.testing.assert_array_equal(
        normalized,
        np.asarray([[0.2, 1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(raw, np.asarray([[0.2, 1.0, 1.019]], np.float32))
    assert clipped_count == 1


def test_collection_manifest_rejects_previous_detector_pipeline(
    tmp_path: Path,
) -> None:
    selection_path = (
        Path(__file__).parents[5]
        / "src/tasks/plcs/configs/gvhmr_motion/meiji_3cam.yaml"
    )
    selection = GvhmrSelectionConfig.load(selection_path)
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    dataset_root.mkdir()
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "dataset_id": selection.dataset_id,
                "selection_config_sha256": "selection-digest",
                "extraction_pipeline_version": "plcs_gvhmr_yolo_v1",
                "selection": selection.to_dict(),
                "updated_at": "2026-09-14T00:00:00+00:00",
                "records": [],
                "reproducibility_sha256": "test-repro",
            }
        ),
        encoding="utf-8",
    )
    extractor = GvhmrMotionExtractor(
        dataset_root=dataset_root,
        output_root=output_root,
        selection=selection,
        selection_digest="selection-digest",
        model_runtime=cast(GvhmrDemoConfig, object()),
        max_frames=None,
        overwrite=False,
        write_preview=True,
        reproducibility_digest="test-repro",
        seed=42,
        deterministic=True,
    )

    with pytest.raises(RuntimeError, match="manifest is incompatible"):
        extractor._load_collection_manifest()
