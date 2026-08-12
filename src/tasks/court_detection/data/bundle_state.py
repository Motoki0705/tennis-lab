"""Strict serializable snapshots for resolved Court target bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch

from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
)

_BUNDLE_SCHEMA = "court_target_bundle_v1"
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


__all__ = [
    "deserialize_target_bundle",
    "serialize_target_bundle",
]
