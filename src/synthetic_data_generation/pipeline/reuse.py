"""Typed semantic gates for reusing completed scene-stage publications."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCSCoordinateContract,
    PLCSSourceSupportPlane,
)
from src.utils.schema.court_normalization import (
    validate_court_coordinate_normalization,
)


@dataclass(frozen=True, slots=True)
class RequiredOutputsReusablePublicationValidator:
    """Preserve the established reuse policy for non-PLCS stages."""

    def validate(self, owner_path: Path) -> None:
        """Require the existing fixed owner after required-output validation."""
        if not owner_path.is_dir() or owner_path.is_symlink():
            raise ValueError("Reusable stage owner must be an ordinary directory.")


@dataclass(frozen=True, slots=True)
class PLCSV5ReusablePublicationValidator:
    """Reject completed PLCS owners without exact current semantic provenance."""

    def validate(self, owner_path: Path) -> None:
        """Validate only the metadata needed to decide v5 reuse eligibility."""
        if not owner_path.is_dir() or owner_path.is_symlink():
            raise ValueError("Reusable PLCS owner must be an ordinary directory.")
        manifest_path = owner_path.joinpath("dataset.json")
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise FileNotFoundError("Reusable PLCS manifest is missing.")
        manifest = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            name="PLCS dataset manifest",
        )
        _require_keys(
            manifest,
            {"schema", "domain", "metadata"},
            name="PLCS dataset manifest",
        )
        if manifest["schema"] != PLCS_DATASET_SCHEMA or manifest["domain"] != "plcs":
            raise ValueError("Completed PLCS publication is not exact v5/plcs.")
        metadata = _mapping(manifest["metadata"], name="PLCS metadata")
        validate_court_coordinate_normalization(
            metadata,
            artifact="Reusable PLCS publication",
        )
        _require_keys(
            metadata,
            {"coordinate_contract", "logical_scenes"},
            name="PLCS metadata",
        )
        PLCSCoordinateContract.from_dict(metadata["coordinate_contract"])
        logical_scenes = _sequence(
            metadata["logical_scenes"], name="PLCS logical_scenes"
        )
        if not logical_scenes:
            raise ValueError("Reusable PLCS publication has no logical scenes.")
        for scene_index, raw_scene in enumerate(logical_scenes):
            scene_name = f"PLCS logical_scenes[{scene_index}]"
            scene = _mapping(raw_scene, name=scene_name)
            _require_keys(scene, {"tracks"}, name=scene_name)
            tracks = _sequence(
                scene["tracks"],
                name=f"{scene_name}.tracks",
            )
            if not tracks:
                raise ValueError("Reusable PLCS logical scene has no tracks.")
            for track_index, raw_track in enumerate(tracks):
                track = _mapping(
                    raw_track,
                    name=(f"PLCS logical_scenes[{scene_index}].tracks[{track_index}]"),
                )
                _require_keys(track, {"support_plane"}, name="PLCS track")
                PLCSSourceSupportPlane.from_dict(track["support_plane"])


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _require_keys(
    value: Mapping[str, object],
    required: set[str],
    *,
    name: str,
) -> None:
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{name} is missing required keys: {sorted(missing)}.")


__all__ = [
    "PLCSV5ReusablePublicationValidator",
    "RequiredOutputsReusablePublicationValidator",
]
