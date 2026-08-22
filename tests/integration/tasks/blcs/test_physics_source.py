"""Real-physics regression for the canonical BLCS source seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.tasks.blcs.generate_dataset.source_api import (
    BLCSPhysicsSourceSettings,
    BLCSPhysicsTrajectorySource,
    build_blcs_generator_configuration,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_PRODUCTION_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "src/synthetic_data_generation/configs/dataset/blcs/production.yaml"
)
_SCENE_ID = "B00-canonical-final-real-20260810-blcs-000000"


def _production_source() -> BLCSPhysicsTrajectorySource:
    raw = OmegaConf.load(_PRODUCTION_CONFIG)
    raw.generator.targeted_velocity.gravity = raw.generator.physics.gravity
    generator = OmegaConf.to_container(raw.generator, resolve=True)
    source_settings = OmegaConf.to_container(raw.trajectory_source, resolve=True)
    assert isinstance(generator, dict)
    assert isinstance(source_settings, dict)
    return BLCSPhysicsTrajectorySource(
        generator_config=build_blcs_generator_configuration(generator),
        settings=BLCSPhysicsSourceSettings.from_mapping(
            {
                "timeline": source_settings["timeline"],
                "maximum_physics_attempts_per_object": source_settings[
                    "maximum_physics_attempts_per_object"
                ],
                "device": source_settings["device"],
            }
        ),
    )


def test_production_seed_695_is_complete_and_deterministic_after_rejection() -> None:
    source = _production_source()

    first = source.generate(scene_id=_SCENE_ID, seed=695)
    repeated = source.generate(scene_id=_SCENE_ID, seed=695)

    assert first.frame_count == 1024
    assert first.object_count == 8
    assert any(
        "requested-side tolerance" in rejection.reason
        for diagnostic in first.proposal_diagnostics
        for rejection in diagnostic.rejected_attempts
    )
    np.testing.assert_array_equal(first.positions_court_m, repeated.positions_court_m)
    np.testing.assert_array_equal(first.velocities_court_mps, repeated.velocities_court_mps)
    np.testing.assert_array_equal(first.present, repeated.present)
    assert first.to_metadata() == repeated.to_metadata()
