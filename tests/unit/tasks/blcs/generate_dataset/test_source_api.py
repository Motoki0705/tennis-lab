"""Tests for the public BLCS physics trajectory source boundary."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.blcs import generate_dataset as generate_dataset_package
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
    GeneratorConfig,
)
from src.tasks.blcs.generate_dataset.source_api import (
    BLCSPhysicsProposalExhausted,
    BLCSPhysicsProposalRejected,
    BLCSPhysicsSourceSettings,
    BLCSPhysicsTrajectorySource,
    BLCSSourceScene,
    BLCSTimelineSpec,
    build_blcs_generator_configuration,
)
from src.utils.projection.camera_projector import CameraConfig
from src.utils.schema.court import NET_POST_OFFSET_X, CourtConfig
from src.utils.schema.court_normalization import (
    normalize_court_position,
    normalize_court_velocity,
)

_CONFIG_ROOT = Path(__file__).resolve().parents[5] / "src/tasks/blcs/configs"


def _timeline_mapping() -> dict[str, object]:
    return {
        "num_frames": 4,
        "min_tracks": 2,
        "max_tracks": 2,
        "max_concurrent": 2,
        "min_reuse_gap_frames": 0,
        "start_index_range": [0, 0],
        "min_active_frames": 4,
        "overlap_probability": 1.0,
        "min_gap_frames": 0,
        "max_gap_frames": 0,
    }


def _settings(*, maximum_attempts: int = 3) -> BLCSPhysicsSourceSettings:
    return BLCSPhysicsSourceSettings.from_mapping(
        {
            "timeline": _timeline_mapping(),
            "maximum_physics_attempts_per_object": maximum_attempts,
            "device": "cpu",
        }
    )


def _plain_yaml(relative_path: str) -> dict[str, object]:
    value = OmegaConf.to_container(
        OmegaConf.load(_CONFIG_ROOT / relative_path),
        resolve=False,
    )
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {str(key): item for key, item in value.items()}


def _resolved_generator_mapping() -> dict[str, object]:
    physics = _plain_yaml("physics/default.yaml")
    targeted = _plain_yaml("targeted_velocity/default.yaml")
    targeted["gravity"] = physics["gravity"]
    generator = _plain_yaml("generator/default.yaml")
    court = generator["court"]
    assert isinstance(court, dict)
    return {
        "physics": physics,
        "rally": _plain_yaml("rally/default.yaml"),
        "camera": _plain_yaml("camera/default.yaml"),
        "targeted_velocity": targeted,
        "court": court,
    }


def _camera_config() -> CameraConfig:
    return CameraConfig(
        z_min=3.0,
        z_max=5.0,
        hfov_deg=60.0,
        image_size=(1280, 720),
        fixed_look_at=(0.0, 0.0, 0.0),
        fixed_baseline_clear_extra=0.0,
        fixed_position_noise_radius=0.0,
        fixed_look_at_xy_radius=0.0,
        layout="fixed",
        broadcast_setback=20.0,
        broadcast_height=7.0,
        broadcast_hfov_deg=35.0,
        broadcast_look_at_y=0.0,
        broadcast_look_at_height=0.5,
        broadcast_position_noise_radius=1.0,
        broadcast_look_at_xy_radius=1.0,
        broadcast_hfov_jitter_deg=2.0,
        broadcast_setback_range=None,
        broadcast_height_range=None,
        broadcast_court_width_frac_range=None,
    )


class _RetryingPhysicsGenerator:
    """Task-internal shaped stub with one explicit rejection per object."""

    def __init__(self, *, config: GeneratorConfig, device: str) -> None:
        del config
        assert device == "cpu"
        self.config = SimpleNamespace(
            camera=_camera_config(),
            court=CourtConfig(
                net_post_offset_x=NET_POST_OFFSET_X,
                net_post_offset_x_range=None,
            ),
        )
        self.calls: dict[str, int] = {}

    @staticmethod
    def sample_from_cell() -> int:
        return int(torch.randint(0, 4, ()).item())

    @staticmethod
    def sample_side() -> str:
        return "near" if random.random() < 0.5 else "far"

    def generate_scene(
        self,
        from_cell: int,
        side: str,
        scene_id: str,
    ) -> BLCSSceneData:
        attempt = self.calls.get(scene_id, 0) + 1
        self.calls[scene_id] = attempt
        if attempt == 1:
            raise RuntimeError(
                "Full-physics targeted-velocity refinement produced no valid landing; "
                "no gravity-only retry fallback is defined."
            )
        offset = random.random() + float(np.random.random()) + float(torch.rand(()))
        trajectory = torch.tensor(
            [
                [offset, -2.0, 1.0],
                [offset + 0.1, -1.0, 1.5],
                [offset + 0.2, 0.0, 1.2],
                [offset + 0.3, 1.0, 0.8],
                [offset + 0.4, 2.0, 0.1],
            ],
            dtype=torch.float32,
        )
        return BLCSSceneData(
            scene_id=scene_id,
            initial_from_cell=from_cell,
            initial_from_side=side,
            rally_length=1,
            end_reason="double_bounce",
            winner_side=None,
            shots=[{"shot_index": 0, "t_start": 0, "t_bounce1": 4}],
            ball_pos_world=trajectory,
            ball_pos_norm=normalize_court_position(trajectory),
            ball_vel_world=torch.ones_like(trajectory),
            ball_vel_norm=normalize_court_velocity(torch.ones_like(trajectory)),
            cameras=[],
            num_cameras_sampled=0,
            fps_out=30,
            sim_fps=120,
            physics_config_dict={
                "gravity": -9.81,
                "wind": [0.0, 0.0, 0.0],
            },
            court_config_dict={"net_post_offset_x": NET_POST_OFFSET_X},
            num_balls=1,
        )


class _ExhaustingPhysicsGenerator(_RetryingPhysicsGenerator):
    total_calls = 0

    def generate_scene(
        self,
        from_cell: int,
        side: str,
        scene_id: str,
    ) -> BLCSSceneData:
        del from_cell, side, scene_id
        type(self).total_calls += 1
        raise BLCSPhysicsProposalRejected("no valid full-physics landing")


class _UnexpectedFailureGenerator(_RetryingPhysicsGenerator):
    total_calls = 0

    def generate_scene(
        self,
        from_cell: int,
        side: str,
        scene_id: str,
    ) -> BLCSSceneData:
        del from_cell, side, scene_id
        type(self).total_calls += 1
        raise RuntimeError("unexpected implementation failure")


def _source() -> BLCSPhysicsTrajectorySource:
    return BLCSPhysicsTrajectorySource(
        generator_config=object.__new__(GeneratorConfig),
        settings=_settings(),
    )


def test_source_settings_are_strict_explicit_and_multi_object() -> None:
    settings = _settings()

    assert generate_dataset_package.BLCSSourceScene is BLCSSourceScene
    assert (
        generate_dataset_package.build_blcs_generator_configuration
        is build_blcs_generator_configuration
    )
    assert settings.timeline == BLCSTimelineSpec.from_mapping(_timeline_mapping())
    assert settings.maximum_physics_attempts_per_object == 3
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        BLCSPhysicsSourceSettings.from_mapping(
            {
                "timeline": _timeline_mapping(),
                "maximum_physics_attempts_per_object": 3,
                "device": "cpu",
                "unexpected": True,
            }
        )
    invalid_timeline = {**_timeline_mapping(), "min_tracks": 1}
    with pytest.raises(ValueError, match="at least two tracks"):
        BLCSTimelineSpec.from_mapping(invalid_timeline)
    with pytest.raises(ValueError, match="CPU execution"):
        BLCSPhysicsSourceSettings(
            timeline=settings.timeline,
            maximum_physics_attempts_per_object=3,
            device="cuda",
        )


def test_public_factory_builds_the_hidden_generator_config_from_resolved_mapping() -> (
    None
):
    resolved = _resolved_generator_mapping()

    configuration = build_blcs_generator_configuration(resolved)

    assert isinstance(configuration, GeneratorConfig)
    assert configuration.physics.gravity == 9.81
    assert configuration.rally.output_fps == 30
    assert configuration.rally.sim_fps == 240
    assert configuration.camera.layout == "fixed"
    assert configuration.camera.image_size == (1280, 720)
    assert configuration.targeted_velocity.gravity == 9.81
    assert configuration.court.net_post_offset_x == NET_POST_OFFSET_X


def test_public_factory_rejects_unknown_unresolved_and_invalid_values() -> None:
    unknown = deepcopy(_resolved_generator_mapping())
    physics = unknown["physics"]
    assert isinstance(physics, dict)
    physics["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        build_blcs_generator_configuration(unknown)

    unresolved = deepcopy(_resolved_generator_mapping())
    targeted = unresolved["targeted_velocity"]
    assert isinstance(targeted, dict)
    targeted["gravity"] = "${physics.gravity}"
    with pytest.raises(TypeError, match="targeted_velocity.gravity must be numeric"):
        build_blcs_generator_configuration(unresolved)

    invalid_bool = deepcopy(_resolved_generator_mapping())
    physics = invalid_bool["physics"]
    assert isinstance(physics, dict)
    physics["use_drag"] = 1
    with pytest.raises(TypeError, match="physics.use_drag must be a boolean"):
        build_blcs_generator_configuration(invalid_bool)

    invalid_fps = deepcopy(_resolved_generator_mapping())
    rally = invalid_fps["rally"]
    assert isinstance(rally, dict)
    rally["sim_fps"] = 239
    with pytest.raises(ValueError, match="integer multiple"):
        build_blcs_generator_configuration(invalid_fps)


def test_source_is_deterministic_complete_and_independent_of_internal_scenes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_module = __import__(
        "src.tasks.blcs.generate_dataset.source_api",
        fromlist=["source_api"],
    )
    monkeypatch.setattr(
        source_module,
        "BLCSSceneGenerator",
        _RetryingPhysicsGenerator,
    )
    source = _source()

    first = source.generate(scene_id="B00-blcs-000000", seed=17)
    repeated = source.generate(scene_id="B00-blcs-000000", seed=17)
    changed = source.generate(scene_id="B00-blcs-000000", seed=18)

    assert isinstance(first, BLCSSourceScene)
    assert first.frame_indices == (0, 1, 2, 3)
    assert first.frame_count == 4
    assert first.object_count == 2
    assert first.positions_court_m.shape == (4, 2, 3)
    assert first.velocities_court_mps.shape == (4, 2, 3)
    assert first.present.shape == (4, 2)
    assert first.present.all()
    assert not first.positions_court_m.flags.writeable
    assert not first.present.flags.writeable
    assert [track.object_id for track in first.tracks] == [
        "ball-001",
        "ball-002",
    ]
    assert all(
        track.source_frame_indices in {(0, 1, 2, 3), (1, 2, 3, 4)}
        for track in first.tracks
    )
    assert all(item.accepted_attempt == 2 for item in first.proposal_diagnostics)
    assert all(len(item.rejected_attempts) == 1 for item in first.proposal_diagnostics)
    assert all(item.maximum_attempts == 3 for item in first.proposal_diagnostics)
    assert all(item.source_frame_count == 5 for item in first.physics_provenance)
    assert all(item.output_fps == 30.0 for item in first.physics_provenance)
    assert all(item.simulation_fps == 120.0 for item in first.physics_provenance)
    metadata = first.to_metadata()
    assert metadata["source_frame_count"] == 4
    assert isinstance(metadata["tracks"], list)
    assert isinstance(metadata["physics_sources"], list)
    assert isinstance(metadata["physics_proposals"], list)
    assert len(metadata["tracks"]) == 2
    assert len(metadata["physics_sources"]) == 2
    assert len(metadata["physics_proposals"]) == 2
    np.testing.assert_array_equal(first.positions_court_m, repeated.positions_court_m)
    np.testing.assert_array_equal(first.present, repeated.present)
    assert first.to_metadata() == repeated.to_metadata()
    assert not np.array_equal(first.positions_court_m, changed.positions_court_m)


def test_proposal_exhaustion_is_bounded_and_carries_all_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_module = __import__(
        "src.tasks.blcs.generate_dataset.source_api",
        fromlist=["source_api"],
    )
    monkeypatch.setattr(
        source_module,
        "BLCSSceneGenerator",
        _ExhaustingPhysicsGenerator,
    )
    _ExhaustingPhysicsGenerator.total_calls = 0
    source = BLCSPhysicsTrajectorySource(
        generator_config=object.__new__(GeneratorConfig),
        settings=_settings(maximum_attempts=2),
    )

    with pytest.raises(BLCSPhysicsProposalExhausted) as raised:
        source.generate(scene_id="B00-blcs-000000", seed=17)

    diagnostic = raised.value.diagnostic
    assert diagnostic.accepted_attempt is None
    assert diagnostic.maximum_attempts == 2
    assert [item.attempt for item in diagnostic.rejected_attempts] == [1, 2]
    assert _ExhaustingPhysicsGenerator.total_calls == 2


def test_unrecognized_runtime_failure_is_not_retried_or_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_module = __import__(
        "src.tasks.blcs.generate_dataset.source_api",
        fromlist=["source_api"],
    )
    monkeypatch.setattr(
        source_module,
        "BLCSSceneGenerator",
        _UnexpectedFailureGenerator,
    )
    _UnexpectedFailureGenerator.total_calls = 0

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        _source().generate(scene_id="B00-blcs-000000", seed=17)

    assert _UnexpectedFailureGenerator.total_calls == 1
