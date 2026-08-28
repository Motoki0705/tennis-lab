"""Strict serializable snapshots for resolved Court target bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch

from src.tasks.court_detection.configuration import (
    CourtConsistencyLossConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
)
from src.tasks.court_detection.geometry.pose import POSE10D_RAW_ORDER, POSE10D_SCHEMA

_BUNDLE_SCHEMA = "court_target_bundle_v1"
_POSE_CHECKPOINT_SCHEMA_V1 = "court_hierarchical_pose_checkpoint_v1"
_POSE_CHECKPOINT_SCHEMA_V2 = "court_hierarchical_pose_checkpoint_v2"
# Read old additive snapshots but never emit the retired model-family marker.
_QUERY_CHECKPOINT_SCHEMA_V1 = "court_query_checkpoint_v1"
_QUERY_CHECKPOINT_SCHEMA_V2 = "court_query_checkpoint_v2"
_DTYPE_TO_NAME = {
    torch.float32: "float32",
    torch.long: "int64",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}
_RECORD_KEYS = {
    "kind",
    "schema",
    "output_channels",
    "channel_names",
    "target_dtype",
    "precomputed",
}


def serialize_target_bundle(bundle: CourtTargetBundleSpec) -> dict[str, object]:
    """Return a pickle-safe, order-preserving bundle snapshot."""
    return {
        "schema": _BUNDLE_SCHEMA,
        "targets": [
            {
                "kind": spec.kind,
                "schema": spec.schema,
                "output_channels": spec.output_channels,
                "channel_names": list(spec.channel_names),
                "target_dtype": _DTYPE_TO_NAME[spec.target_dtype],
                "precomputed": spec.precomputed,
            }
            for spec in bundle.targets.values()
        ],
    }


def deserialize_target_bundle(value: object) -> CourtTargetBundleSpec:
    """Reconstruct and validate an exact bundle snapshot."""
    if not isinstance(value, Mapping) or set(value) != {"schema", "targets"}:
        raise ValueError("Court target bundle snapshot fields changed.")
    if value["schema"] != _BUNDLE_SCHEMA:
        raise ValueError(
            f"Court target bundle snapshot must use schema {_BUNDLE_SCHEMA!r}."
        )
    raw_targets = value["targets"]
    if (
        not isinstance(raw_targets, Sequence)
        or isinstance(raw_targets, (str, bytes))
        or not raw_targets
    ):
        raise ValueError("Court target bundle snapshot requires target records.")
    targets: dict[CourtTargetKind, CourtTargetSpec] = {}
    for raw in raw_targets:
        if not isinstance(raw, Mapping) or set(raw) != _RECORD_KEYS:
            raise ValueError("Court target bundle record fields changed.")
        kind = raw["kind"]
        if kind not in {"kp", "seg", "line"} or kind in targets:
            raise ValueError("Court target bundle kinds must be valid and unique.")
        schema = raw["schema"]
        output_channels = raw["output_channels"]
        channel_names = raw["channel_names"]
        dtype_name = raw["target_dtype"]
        precomputed = raw["precomputed"]
        if not isinstance(schema, str):
            raise ValueError("Court target schema snapshot must be a string.")
        if type(output_channels) is not int:
            raise ValueError(
                "Court target output_channels snapshot must be an integer."
            )
        if (
            not isinstance(channel_names, Sequence)
            or isinstance(channel_names, (str, bytes))
            or any(type(name) is not str for name in channel_names)
        ):
            raise ValueError(
                "Court target channel_names snapshot must contain strings."
            )
        if dtype_name not in _NAME_TO_DTYPE:
            raise ValueError("Court target dtype snapshot is unsupported.")
        if type(precomputed) is not bool:
            raise ValueError("Court target precomputed snapshot must be boolean.")
        selected_kind = cast(CourtTargetKind, kind)
        targets[selected_kind] = CourtTargetSpec(
            kind=selected_kind,
            schema=schema,
            output_channels=output_channels,
            channel_names=tuple(cast(Sequence[str], channel_names)),
            target_dtype=_NAME_TO_DTYPE[cast(str, dtype_name)],
            precomputed=precomputed,
        )
    return CourtTargetBundleSpec(targets)


@dataclass(frozen=True, slots=True)
class CourtQueryCheckpointState:
    target_bundle: CourtTargetBundleSpec
    loss_config_name: str
    supervision_subset: tuple[str, ...]
    pose_supervision: bool
    consistency: CourtConsistencyLossConfig | None


def serialize_query_checkpoint_state(
    bundle: CourtTargetBundleSpec,
    *,
    loss_config_name: str,
    pose_supervision: bool,
    consistency: CourtConsistencyLossConfig | None = None,
) -> dict[str, object]:
    """Snapshot query family, target order, and supervision identity."""
    if not loss_config_name or loss_config_name != loss_config_name.strip():
        raise ValueError("Query loss_config_name must be non-empty and trimmed.")
    subset = (*bundle.kinds, *(("pose",) if pose_supervision else ()))
    if consistency is not None and consistency.enabled and not pose_supervision:
        raise ValueError(
            "Enabled query checkpoint consistency requires pose supervision."
        )
    enabled_consistency = (
        consistency if consistency is not None and consistency.enabled else None
    )
    snapshot: dict[str, object] = {
        "schema": (
            _POSE_CHECKPOINT_SCHEMA_V2
            if enabled_consistency is not None
            else _POSE_CHECKPOINT_SCHEMA_V1
        ),
        "model_family": "court_hierarchical",
        "pose_schema": POSE10D_SCHEMA,
        "pose_raw_order": list(POSE10D_RAW_ORDER),
        "target_bundle": serialize_target_bundle(bundle),
        "supervision": {
            "loss_config_name": loss_config_name,
            "subset": list(subset),
            "pose_enabled": pose_supervision,
        },
    }
    if enabled_consistency is not None:
        snapshot["consistency"] = {
            "enabled": enabled_consistency.enabled,
            "weight": enabled_consistency.weight,
            "temperature": enabled_consistency.temperature,
            "huber_delta": enabled_consistency.huber_delta,
            "min_depth_m": enabled_consistency.min_depth_m,
            "depth_scale_m": enabled_consistency.depth_scale_m,
            "cheirality_weight": enabled_consistency.cheirality_weight,
            "warmup_fraction": enabled_consistency.warmup_fraction,
            "gradient_flow": enabled_consistency.gradient_flow,
        }
    return snapshot


def deserialize_query_checkpoint_state(value: object) -> CourtQueryCheckpointState:
    """Revalidate an exact versioned query checkpoint snapshot."""
    common_fields = {
        "schema",
        "model_family",
        "pose_schema",
        "pose_raw_order",
        "target_bundle",
        "supervision",
    }
    if not isinstance(value, Mapping):
        raise ValueError("Court query checkpoint snapshot fields changed.")
    schema = value.get("schema")
    expected = (
        common_fields | {"consistency"}
        if schema in {_POSE_CHECKPOINT_SCHEMA_V2, _QUERY_CHECKPOINT_SCHEMA_V2}
        else common_fields
    )
    if set(value) != expected:
        raise ValueError("Court query checkpoint snapshot fields changed.")
    if (
        schema not in {
            _POSE_CHECKPOINT_SCHEMA_V1,
            _POSE_CHECKPOINT_SCHEMA_V2,
            _QUERY_CHECKPOINT_SCHEMA_V1,
            _QUERY_CHECKPOINT_SCHEMA_V2,
        }
        or value["model_family"] not in {"court_hierarchical", "court_query_encoder"}
        or value["pose_schema"] != POSE10D_SCHEMA
        or value["pose_raw_order"] != list(POSE10D_RAW_ORDER)
    ):
        raise ValueError("Court query checkpoint identity/schema changed.")
    bundle = deserialize_target_bundle(value["target_bundle"])
    supervision = value["supervision"]
    if not isinstance(supervision, Mapping) or set(supervision) != {
        "loss_config_name",
        "subset",
        "pose_enabled",
    }:
        raise ValueError("Court query supervision snapshot fields changed.")
    name = supervision["loss_config_name"]
    subset = supervision["subset"]
    pose_enabled = supervision["pose_enabled"]
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("Court query loss config identity is invalid.")
    if (
        not isinstance(subset, Sequence)
        or isinstance(subset, (str, bytes))
        or any(not isinstance(item, str) for item in subset)
        or type(pose_enabled) is not bool
    ):
        raise ValueError("Court query supervision subset is invalid.")
    expected_subset = [*bundle.kinds, *(("pose",) if pose_enabled else ())]
    if list(subset) != expected_subset:
        raise ValueError("Court query supervision subset disagrees with bundle/pose.")
    consistency: CourtConsistencyLossConfig | None = None
    if schema in {_POSE_CHECKPOINT_SCHEMA_V2, _QUERY_CHECKPOINT_SCHEMA_V2}:
        try:
            consistency = CourtConsistencyLossConfig.from_mapping(
                value["consistency"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Court query checkpoint consistency identity is invalid."
            ) from error
        if not consistency.enabled or not pose_enabled:
            raise ValueError(
                "Court query checkpoint v2 requires enabled consistency and pose."
            )
    return CourtQueryCheckpointState(
        target_bundle=bundle,
        loss_config_name=name,
        supervision_subset=tuple(cast(Sequence[str], subset)),
        pose_supervision=pose_enabled,
        consistency=consistency,
    )


CourtPoseCheckpointState = CourtQueryCheckpointState
serialize_pose_checkpoint_state = serialize_query_checkpoint_state
deserialize_pose_checkpoint_state = deserialize_query_checkpoint_state


__all__ = [
    "CourtQueryCheckpointState",
    "CourtPoseCheckpointState",
    "deserialize_query_checkpoint_state",
    "deserialize_target_bundle",
    "serialize_target_bundle",
    "serialize_query_checkpoint_state",
    "serialize_pose_checkpoint_state",
    "deserialize_pose_checkpoint_state",
]
