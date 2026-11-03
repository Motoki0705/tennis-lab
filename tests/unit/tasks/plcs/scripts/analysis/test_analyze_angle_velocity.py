"""Feature-specific denominators in angular-velocity statistics."""

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.tasks.base.generate_dataset import (
    build_physical_court_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.court_keypoint_contract import court_keypoint_contract_document
from src.tasks.plcs.scripts.analysis.analyze_angle_velocity import (
    ChannelStats,
    _normalize_weights,
    analyze_angle_velocity,
)
from tests.support.geometry.skeleton import coco_pose_m


def test_channel_stats_ignore_missing_observations_in_each_denominator() -> None:
    stats = ChannelStats(2)
    stats.update(
        np.array([[1.0, 10.0], [3.0, 999.0]]), np.array([[True, True], [True, False]])
    )
    stats.update(np.array([[5.0, 14.0]]), np.array([[True, True]]))
    np.testing.assert_array_equal(stats.count(), [3, 2])
    np.testing.assert_allclose(stats.mean_abs(), [3, 12])
    np.testing.assert_allclose(stats.std(), [np.sqrt(8 / 3), 2])


def test_never_observed_channel_is_missing_and_all_missing_family_raises() -> None:
    stats = ChannelStats(2)
    stats.update(np.array([[2.0, np.nan]]), np.array([[True, False]]))
    assert np.isnan(stats.std()[1])
    assert _normalize_weights(stats.std()) == [1.0, None]
    with pytest.raises(ValueError, match="No valid"):
        ChannelStats(2).std()


class _SceneSamples(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        n = 3 if index == 0 else 2
        pose = coco_pose_m().repeat(n, 1, 1)
        if index == 0:
            pose[2, 9] = pose[2, 7]  # Missing left forearm in final frame.
        return {
            "court_keypoint_metadata": court_keypoint_contract_document(
                resolve_court_keypoint_contract("physical_v1")
            ),
            "court_reference_provenance": build_physical_court_provenance(),
            "selected_camera_ids": ("camera_0",),
            "human_kp": torch.zeros(1, n, 17, 2),
            "human_vis": torch.ones(1, n, 17),
            "court_kp": torch.zeros(1, n, 14, 2),
            "court_vis": torch.ones(1, n, 14),
            "padding_mask": torch.zeros(1, n, dtype=torch.bool),
            "position": torch.zeros(n, 3),
            "rotation": torch.tensor([1.0, 0.0]).repeat(n, 1),
            "human_kp_3d": pose,
            "camera_R": torch.eye(3).unsqueeze(0),
            "camera_C": torch.zeros(1, 3),
        }


def test_analysis_counts_both_endpoint_masks_and_excludes_collator_padding() -> None:
    # Real collator adds one padded zero pose; it must not enter geometry.
    result = analyze_angle_velocity(_SceneSamples(), "test", 2, 0, None)
    assert result["joint_angles"]["sample_count"] == [2] + [3] * 11
    assert result["torsion_angles"]["sample_count"] == [2, 3, 3, 3]
    assert result["torso_twist"]["sample_count"] == 3
    assert result["joint_angles"]["mean_abs"] == [0.0] * 12
