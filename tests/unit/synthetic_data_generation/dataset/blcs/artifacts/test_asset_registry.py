"""Tests for explicit and reproducible tennis-ball Gaussian inventories."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.synthetic_data_generation.dataset.blcs.artifacts.asset_registry import (
    BallAssetRegistry,
    load_ball_asset_registry,
    select_ball_asset,
    verify_local_ball_asset_registry,
    write_ball_asset_registry,
)


def test_registry_round_trip_verifies_bytes_and_refuses_overwrite(
    tmp_path: Path,
    ball_registry: BallAssetRegistry,
) -> None:
    path = tmp_path / "registry.json"

    write_ball_asset_registry(path, ball_registry)
    loaded = load_ball_asset_registry(path)

    assert loaded == ball_registry
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_ball_asset_registry(path, ball_registry)


def test_selection_is_stable_across_registry_input_order(
    ball_registry: BallAssetRegistry,
) -> None:
    reversed_registry = BallAssetRegistry.create(
        registry_id=ball_registry.registry_id,
        appearance_space_sha256=ball_registry.appearance_space_sha256,
        entries=tuple(reversed(ball_registry.entries)),
    )

    first = select_ball_asset(
        ball_registry,
        seed=1729,
        selection_key="scene-7:object:1",
    )
    second = select_ball_asset(
        reversed_registry,
        seed=1729,
        selection_key="scene-7:object:1",
    )

    assert reversed_registry.registry_fingerprint == ball_registry.registry_fingerprint
    assert second.selection_sha256 == first.selection_sha256
    assert second.entry.variant_id == first.entry.variant_id
    assert second.entry_index == first.entry_index


def test_registry_rejects_mismatched_appearance_space(
    ball_registry: BallAssetRegistry,
) -> None:
    mismatched_entry = replace(
        ball_registry.entries[0],
        asset=replace(
            ball_registry.entries[0].asset,
            appearance_space_sha256="f" * 64,
        ),
    )

    with pytest.raises(ValueError, match="do not share"):
        BallAssetRegistry.create(
            registry_id="mismatched",
            appearance_space_sha256=ball_registry.appearance_space_sha256,
            entries=(mismatched_entry,),
        )


def test_local_verification_rejects_tampered_asset(
    ball_registry: BallAssetRegistry,
) -> None:
    tensor_uri = ball_registry.entries[0].asset.tensors.uri
    tensor_path = Path(tensor_uri.removeprefix("file://"))
    tensor_path.write_bytes(b"changed bytes with another length")

    with pytest.raises(ValueError, match="size mismatch"):
        verify_local_ball_asset_registry(ball_registry)


def test_parser_rejects_unknown_registry_fields(
    ball_registry: BallAssetRegistry,
) -> None:
    payload = json.loads(json.dumps(ball_registry.to_dict()))
    payload["fallback_asset"] = "ball-felt-green"

    with pytest.raises(ValueError, match="fallback_asset"):
        BallAssetRegistry.from_dict(payload)
