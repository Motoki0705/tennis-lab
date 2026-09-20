# mypy: disallow_untyped_decorators=False
"""CPU checks for the explicit observation migration boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pytest


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "meiji_reuse", Path(__file__).with_name("reuse.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_selection(driver: ModuleType, h: np.ndarray) -> dict[str, np.ndarray]:
    raw = {
        "frame_indices": np.arange(3, dtype=np.int64),
        "offsets": np.arange(0, 7, 2, dtype=np.int64),
        "boxes": np.tile([[-1, -9, 1, -5], [-1, 1, 1, 5]], (3, 1)).astype(np.float32),
        "scores": np.ones(6, np.float32),
    }
    driver.validate_raw(raw, total=3, stride=1)
    selected = driver.selection(
        raw,
        h,
        total=3,
        fps=30.0,
        camera="cam0",
        settings={
            "selection_policy": "temporal_continuity",
            "association": {
                "min_iou": 0.0,
                "max_center_distance": 1.0,
                "max_prediction_frames": 60,
                "max_prediction_distance": 1.0,
            },
            "court_half_width_m": 6.5,
            "court_half_length_m": 18.0,
            "min_sample_coverage": 0.7,
            "max_gap_seconds": 1.0,
            "long_gap_policy": "mask",
        },
    )

    return cast(dict[str, np.ndarray], selected)


def test_changed_h_changes_production_selection(driver: ModuleType) -> None:
    old = fixture_selection(driver, np.eye(3))
    new = fixture_selection(driver, np.diag([1.0, -1.0, 1.0]))
    differences = driver.exact_differences(old, new)
    assert "boxes" in differences and "source_detection_ids" in differences
    assert "observed_masks" not in differences


def test_changed_h_can_preserve_exact_pose_inputs(driver: ModuleType) -> None:
    old = fixture_selection(driver, np.eye(3))
    new = fixture_selection(driver, np.array([[1.0, 0, 0.1], [0, 1.0, 0], [0, 0, 1.0]]))
    assert driver.exact_differences(old, new) == {}


@pytest.mark.parametrize(
    "key",
    [
        "track_ids",
        "observed_masks",
        "pose_supported_mask",
        "source_detection_ids",
        "detection_frame_indices",
    ],
)
def test_identical_boxes_do_not_override_mask_or_identity_changes(
    driver: ModuleType, key: str
) -> None:
    old = fixture_selection(driver, np.eye(3))
    new = {k: v.copy() for k, v in old.items()}
    new[key].flat[0] = not new[key].flat[0] if new[key].dtype == np.bool_ else -1
    assert np.array_equal(old["boxes"], new["boxes"])
    assert set(driver.exact_differences(old, new)) == {key}


def test_same_values_with_different_dtype_are_rejected(driver: ModuleType) -> None:
    old = fixture_selection(driver, np.eye(3))
    new = {**old, "boxes": old["boxes"].astype(np.float64)}
    assert set(driver.exact_differences(old, new)) == {"boxes"}


def test_validated_copy_is_exact_and_refuses_overwrite(
    driver: ModuleType, tmp_path: Path
) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"validated immutable source")
    digest = driver.dual_sha256(source)
    driver.copy_new(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        driver.copy_new(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="Source changed"):
        driver.copy_new(source, tmp_path / "other", "0" * 64)
    assert not (tmp_path / "other").exists()


def test_receipt_publication_refuses_existing_file(
    driver: ModuleType, tmp_path: Path
) -> None:
    receipt = tmp_path / "receipt.json"
    driver.write_new(receipt, {"homography": "new"})
    with pytest.raises(FileExistsError):
        driver.write_new(receipt, {"homography": "replacement"})
    assert "replacement" not in receipt.read_text()


def test_hash_audit_detects_source_change(driver: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"before")
    audit = driver.Inputs()
    audit.digest(path)
    assert audit.after() == audit.before
    path.write_bytes(b"after")
    assert audit.after() != audit.before
