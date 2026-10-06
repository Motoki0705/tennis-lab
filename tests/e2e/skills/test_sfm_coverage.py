from __future__ import annotations

import pytest

from experiments.sfm_comparison.plot_coverage import common_coverage


def test_rejects_equal_counts_from_different_inputs() -> None:
    first = {"manifest_sha256": "first", "input_images": 90}
    second = {"manifest_sha256": "second", "input_images": 90}
    with pytest.raises(ValueError, match="different input manifests"):
        common_coverage(first, second)


def test_coverage_preserves_missing_pose_gaps() -> None:
    first = {
        "manifest_sha256": "same",
        "input_images": 3,
        "registered_images": [
            "frame_000000.jpg",
            "frame_000001.jpg",
            "frame_000002.jpg",
        ],
        "missing_images": [],
    }
    second = {
        "manifest_sha256": "same",
        "input_images": 3,
        "registered_images": ["frame_000000.jpg", "frame_000002.jpg"],
        "missing_images": ["frame_000001.jpg"],
    }
    assert common_coverage(first, second) == ([0, 1, 2], [0, 2])
