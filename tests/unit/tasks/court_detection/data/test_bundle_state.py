"""Versioned query checkpoint supervision identity contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from src.tasks.court_detection.configuration import (
    CourtQueryConsistencyLossConfig,
)
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


def _enabled_consistency() -> CourtQueryConsistencyLossConfig:
    return CourtQueryConsistencyLossConfig(
        enabled=True,
        weight=1.0,
        temperature=1.0,
        huber_delta=0.01,
        min_depth_m=0.1,
        depth_scale_m=1.0,
        cheirality_weight=0.1,
        warmup_fraction=0.1,
        gradient_flow="both",
    )


def _disabled_consistency() -> CourtQueryConsistencyLossConfig:
    return CourtQueryConsistencyLossConfig.legacy_disabled()


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
    assert restored.consistency is None


def test_enabled_query_consistency_uses_v2_and_round_trips_exact_identity() -> None:
    consistency = _enabled_consistency()
    snapshot = serialize_query_checkpoint_state(
        _bundle(),
        loss_config_name="query_joint_both_v1",
        pose_supervision=True,
        consistency=consistency,
    )
    restored = deserialize_query_checkpoint_state(snapshot)

    assert snapshot["schema"] == "court_query_checkpoint_v2"
    assert restored.consistency == consistency


def test_explicit_direct_only_consistency_preserves_v1_checkpoint_schema() -> None:
    snapshot = serialize_query_checkpoint_state(
        _bundle(),
        loss_config_name="query_direct_all_v1",
        pose_supervision=True,
        consistency=_disabled_consistency(),
    )

    assert snapshot["schema"] == "court_query_checkpoint_v1"
    assert "consistency" not in snapshot
    assert deserialize_query_checkpoint_state(snapshot).consistency is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight", 2.0),
        ("temperature", 0.5),
        ("huber_delta", 0.02),
        ("min_depth_m", 0.2),
        ("depth_scale_m", 2.0),
        ("cheirality_weight", 0.2),
        ("warmup_fraction", 0.2),
        ("gradient_flow", "stopgrad_pose"),
    ],
)
def test_enabled_query_consistency_checkpoint_fields_are_identity(
    field: str,
    value: object,
) -> None:
    snapshot = serialize_query_checkpoint_state(
        _bundle(),
        loss_config_name="query_joint_both_v1",
        pose_supervision=True,
        consistency=_enabled_consistency(),
    )
    changed = deepcopy(snapshot)
    raw_consistency = changed["consistency"]
    assert isinstance(raw_consistency, dict)
    raw_consistency[field] = value

    assert (
        deserialize_query_checkpoint_state(changed)
        != deserialize_query_checkpoint_state(snapshot)
    )


def test_v1_cannot_carry_consistency_and_v2_requires_enabled_consistency() -> None:
    v1 = serialize_query_checkpoint_state(
        _bundle(),
        loss_config_name="query_pose_v1",
        pose_supervision=True,
    )
    v1["consistency"] = {
        "enabled": True,
    }
    with pytest.raises(ValueError, match="fields changed"):
        deserialize_query_checkpoint_state(v1)

    v2 = serialize_query_checkpoint_state(
        _bundle(),
        loss_config_name="query_joint_both_v1",
        pose_supervision=True,
        consistency=_enabled_consistency(),
    )
    raw = v2["consistency"]
    assert isinstance(raw, dict)
    raw["enabled"] = False
    raw["weight"] = 0.0
    raw["cheirality_weight"] = 0.0
    raw["warmup_fraction"] = 0.0
    with pytest.raises(ValueError, match="v2 requires enabled"):
        deserialize_query_checkpoint_state(v2)


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
