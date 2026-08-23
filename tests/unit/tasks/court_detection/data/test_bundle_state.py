"""Versioned query checkpoint supervision identity contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from src.tasks.court_detection.data.bundle_state import (
    deserialize_query_checkpoint_state,
    serialize_query_checkpoint_state,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.geometry.pose import POSE10D_RAW_ORDER, POSE10D_SCHEMA


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema=(
                    "synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1"
                ),
                output_channels=14,
                channel_names=tuple(f"kp_{index}" for index in range(14)),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )


def test_query_checkpoint_records_pose_schema_order_and_supervision_subset() -> None:
    snapshot = serialize_query_checkpoint_state(
        _bundle(),
        loss_config_name="query_pose_v1",
        pose_supervision=True,
    )
    restored = deserialize_query_checkpoint_state(snapshot)

    assert snapshot["pose_schema"] == POSE10D_SCHEMA
    assert snapshot["pose_raw_order"] == list(POSE10D_RAW_ORDER)
    assert restored.loss_config_name == "query_pose_v1"
    assert restored.supervision_subset == ("kp", "pose")
    assert restored.pose_supervision


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_family", "court_hierarchical"),
        ("pose_schema", "pose_v0"),
        ("pose_raw_order", list(reversed(POSE10D_RAW_ORDER))),
    ],
)
def test_query_checkpoint_identity_mismatch_is_rejected(
    field: str,
    value: object,
) -> None:
    snapshot = serialize_query_checkpoint_state(
        _bundle(),
        loss_config_name="query_pose_v1",
        pose_supervision=True,
    )
    corrupted = deepcopy(snapshot)
    corrupted[field] = value

    with pytest.raises(ValueError, match="identity|schema"):
        deserialize_query_checkpoint_state(corrupted)


def test_query_checkpoint_supervision_cannot_masquerade_as_dense_only() -> None:
    snapshot = serialize_query_checkpoint_state(
        _bundle(),
        loss_config_name="query_pose_v1",
        pose_supervision=True,
    )
    supervision = snapshot["supervision"]
    assert isinstance(supervision, dict)
    supervision["subset"] = ["kp"]

    with pytest.raises(ValueError, match="subset"):
        deserialize_query_checkpoint_state(snapshot)
