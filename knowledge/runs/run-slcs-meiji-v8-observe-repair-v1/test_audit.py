"""CPU checks for repair comparison semantics and schema rejection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def load_audit() -> Any:
    spec = importlib.util.spec_from_file_location(
        "meiji_repair_audit", Path(__file__).with_name("audit.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_records_float_mask_and_dtype_changes(tmp_path: Path) -> None:
    audit = load_audit()
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(
        old,
        values=np.array([1.0, 2.0], dtype=np.float32),
        mask=np.array([True, False]),
        ids=np.array([1], dtype=np.int32),
    )
    np.savez(
        new,
        values=np.array([1.0, 2.25], dtype=np.float32),
        mask=np.array([False, False]),
        ids=np.array([1], dtype=np.int64),
    )
    result = audit.compare_npz(old, new)
    assert not result["equal"]
    assert result["arrays"]["values"]["different_elements"] == 1
    assert result["arrays"]["values"]["float_max_abs"] == 0.25
    assert result["arrays"]["mask"]["different_elements"] == 1
    assert result["arrays"]["ids"]["different_elements"] == 0
    assert not result["arrays"]["ids"]["equal"]
    assert audit.compare_npz(old, old)["equal"]


def test_compare_records_missing_keys_and_shape_changes(tmp_path: Path) -> None:
    audit = load_audit()
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(old, values=np.zeros(2), removed=np.zeros(1))
    np.savez(new, values=np.zeros(3), added=np.zeros(1))
    result = audit.compare_npz(old, new)
    assert not result["equal"]
    assert result["arrays"]["values"]["different_elements"] is None
    assert result["arrays"]["removed"]["new_shape"] is None
    assert result["arrays"]["added"]["old_shape"] is None


def test_nested_metadata_changes_preserve_missing_vs_null() -> None:
    audit = load_audit()
    changes = audit.metadata_diff(
        {"settings": {"stride": 4}, "gone": None}, {"settings": {"stride": 8}}
    )
    assert changes == [
        {
            "field": "gone",
            "old_present": True,
            "new_present": False,
            "old": None,
            "new": None,
        },
        {
            "field": "settings.stride",
            "old_present": True,
            "new_present": True,
            "old": 4,
            "new": 8,
        },
    ]


def fixture_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    people = {
        "boxes": np.ones((2, 2, 4), np.float32),
        "track_ids": np.arange(2),
        "keypoints": np.ones((2, 2, 17, 3), np.float32),
        "observed_masks": np.ones((2, 2), bool),
        "pose_supported_mask": np.ones((2, 2), bool),
        "source_detection_ids": np.array([[0, 2], [1, 3]]),
        "detection_frame_indices": np.arange(2),
    }
    raw = {
        "frame_indices": np.arange(2),
        "offsets": np.array([0, 2, 4]),
        "boxes": np.ones((4, 4), np.float32),
        "scores": np.ones(4, np.float32),
    }
    return people, raw


def test_masks_and_source_ids_are_validated() -> None:
    audit = load_audit()
    people, raw = fixture_arrays()
    audit.validate_arrays(people, raw, 2)
    people["observed_masks"] = people["observed_masks"].astype(np.uint8)
    with pytest.raises(ValueError, match="boolean mask dtype"):
        audit.validate_arrays(people, raw, 2)
    people, raw = fixture_arrays()
    people["source_detection_ids"][0, 1] = 0
    with pytest.raises(ValueError, match="outside frame"):
        audit.validate_arrays(people, raw, 2)


def test_production_receipts_reject_metadata_disagreement(tmp_path: Path) -> None:
    import json

    from src.tennis_scene.dataset_pipeline.people import validate_people_receipts
    from src.utils.checksum import FileIntegrityError

    (tmp_path / "cam0_people.metadata.json").write_text(
        json.dumps({"detector_sha256": "a" * 64, "pose_sha256": "b" * 64})
    )
    (tmp_path / "cam0_detections.metadata.json").write_text(
        json.dumps({"checkpoint_sha256": "c" * 64})
    )
    with pytest.raises(FileIntegrityError, match="mismatch"):
        validate_people_receipts(["cam0"], tmp_path)


def test_failure_is_recorded_and_existing_output_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys

    audit = load_audit()
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--cache",
            str(tmp_path),
            "--inventory",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
        ],
    )
    assert audit.main() == 1
    result = json.loads(output.read_text())
    assert result["status"] == "failed"
    assert result["errors"][0]["type"] == "FileIntegrityError"
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.main()
    assert output.read_bytes() == original
