"""Tests for shared semantic dataset manifest contracts."""

from __future__ import annotations

import copy
from typing import cast

import pytest

from src.synthetic_data_generation.dataset.contracts import (
    DatasetDomain,
    DatasetManifest,
    FrameInventory,
    TargetCourtBinding,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def _manifest() -> DatasetManifest:
    indices = tuple(range(3))
    return DatasetManifest(
        scene_id="scene-a",
        domain=DatasetDomain.BLCS,
        schema="blcs_dataset_v1",
        frame_inventory=FrameInventory(3, indices, indices, indices),
        target_courts=(
            TargetCourtBinding(
                court_instance_id="court-0",
                candidate_id="candidate-0",
                scene_from_court=RigidTransform.identity(),
                selection_seed=695,
            ),
        ),
        metadata={"camera_profile": "default"},
        diagnostics=("diagnostics/continuity.json",),
    )


def test_dataset_manifest_round_trip_recomputes_frame_equality() -> None:
    manifest = _manifest()

    parsed = DatasetManifest.from_dict(manifest.to_dict())

    assert parsed == manifest


@pytest.mark.parametrize("field", ["planned", "rendered", "labelled"])
def test_dataset_manifest_rejects_incomplete_persisted_frame_summary(field: str) -> None:
    payload = copy.deepcopy(_manifest().to_dict())
    raw_inventory = payload["frame_inventory"]
    assert isinstance(raw_inventory, dict)
    inventory = cast(dict[str, object], raw_inventory)
    inventory[field] = 2

    with pytest.raises(ValueError, match="exact 0..T-1 equality"):
        DatasetManifest.from_dict(payload)


def test_dataset_manifest_rejects_unknown_fields() -> None:
    payload = _manifest().to_dict()
    payload["fingerprint"] = "forbidden"

    with pytest.raises(ValueError, match=r"unknown=\['fingerprint'\]"):
        DatasetManifest.from_dict(payload)


def test_dataset_manifest_rejects_duplicate_target_courts() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="unique court_instance_id"):
        DatasetManifest(
            scene_id=manifest.scene_id,
            domain=manifest.domain,
            schema=manifest.schema,
            frame_inventory=manifest.frame_inventory,
            target_courts=manifest.target_courts * 2,
            metadata=manifest.metadata,
            diagnostics=manifest.diagnostics,
        )
