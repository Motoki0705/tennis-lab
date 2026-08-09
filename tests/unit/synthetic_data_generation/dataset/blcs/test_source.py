"""Tests for the canonical adapter over the public BLCS source API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from src.synthetic_data_generation.dataset.blcs.source import (
    BLCSTrajectorySourceSettings,
    PhysicsBLCSTrajectoryProvider,
)
from src.tasks.blcs.generate_dataset.source_api import (
    BLCSGeneratorConfiguration,
    BLCSPhysicsProvenance,
    BLCSPhysicsSourceSettings,
    BLCSProposalDiagnostic,
    BLCSSourceScene,
    BLCSSourceTrack,
    BLCSTimelineSpec,
)


def _timeline_mapping(*, min_tracks: int = 2) -> dict[str, object]:
    return {
        "num_frames": 4,
        "min_tracks": min_tracks,
        "max_tracks": 2,
        "max_concurrent": 2,
        "min_reuse_gap_frames": 0,
        "start_index_range": [0, 0],
        "min_active_frames": 2,
        "overlap_probability": 1.0,
        "min_gap_frames": 0,
        "max_gap_frames": 0,
    }


def _settings() -> BLCSTrajectorySourceSettings:
    return BLCSTrajectorySourceSettings.from_mapping(
        {
            "scene_count": 3,
            "split_scene_counts": {"train": 1, "validation": 1, "test": 1},
            "multi_object": True,
            "maximum_physics_attempts_per_object": 4,
            "timeline": _timeline_mapping(),
            "device": "cpu",
        }
    )


def _source_scene(*, scene_id: str, seed: int, maximum_attempts: int) -> BLCSSourceScene:
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(4, 2, 3)).astype(np.float64)
    positions[:, :, 2] = np.abs(positions[:, :, 2]) + 0.5
    velocities = rng.normal(size=(4, 2, 3)).astype(np.float64)
    present: npt.NDArray[np.bool_] = np.ones((4, 2), dtype=np.bool_)
    source_ids = tuple(f"{scene_id}-source-{index}" for index in range(2))
    tracks = tuple(
        BLCSSourceTrack(
            object_id=f"ball-{index + 1:03d}",
            source_trajectory_id=source_id,
            source_frame_indices=(0, 1, 2, 3),
        )
        for index, source_id in enumerate(source_ids)
    )
    provenance = tuple(
        BLCSPhysicsProvenance(
            source_trajectory_id=source_id,
            source_frame_count=4,
            initial_from_cell=index,
            initial_from_side="near" if index == 0 else "far",
            rally_length=1,
            end_reason="double_bounce",
            winner_side=None,
            output_fps=30.0,
            simulation_fps=240.0,
            physics_parameters={"gravity": 9.81},
            court_parameters={"net_post_offset_x": 0.914},
            shot_events=({"shot_index": 0},),
        )
        for index, source_id in enumerate(source_ids)
    )
    diagnostics = tuple(
        BLCSProposalDiagnostic(
            source_trajectory_id=source_id,
            accepted_attempt=1,
            maximum_attempts=maximum_attempts,
            rejected_attempts=(),
        )
        for source_id in source_ids
    )
    return BLCSSourceScene(
        scene_id=scene_id,
        seed=seed,
        frame_indices=(0, 1, 2, 3),
        fps=30.0,
        simulation_fps=240.0,
        positions_court_m=positions,
        velocities_court_mps=velocities,
        present=present,
        tracks=tracks,
        physics_provenance=provenance,
        proposal_diagnostics=diagnostics,
    )


class _FakePublicPhysicsSource:
    """Public-source-shaped deterministic fixture, not a task generator stub."""

    def __init__(
        self,
        *,
        generator_config: BLCSGeneratorConfiguration,
        settings: BLCSPhysicsSourceSettings,
    ) -> None:
        self.generator_config = generator_config
        self.settings = settings

    @staticmethod
    def preflight(*, scene_id: str, seed: int) -> None:
        if not scene_id or seed < 0:
            raise ValueError("invalid public source request")

    def generate(self, *, scene_id: str, seed: int) -> BLCSSourceScene:
        self.preflight(scene_id=scene_id, seed=seed)
        return _source_scene(
            scene_id=scene_id,
            seed=seed,
            maximum_attempts=self.settings.maximum_physics_attempts_per_object,
        )


def test_source_settings_fail_closed_on_counts_mode_and_unknown_fields() -> None:
    settings = _settings()

    assert settings.split_sequence() == ("train", "validation", "test")
    assert settings.physics_settings() == BLCSPhysicsSourceSettings(
        timeline=BLCSTimelineSpec.from_mapping(_timeline_mapping()),
        maximum_physics_attempts_per_object=4,
        device="cpu",
    )
    with pytest.raises(ValueError, match="sum exactly"):
        replace(settings, scene_count=4)
    with pytest.raises(ValueError, match="must be multi-object"):
        replace(settings, multi_object=False)
    with pytest.raises(ValueError, match="at least two tracks"):
        BLCSTrajectorySourceSettings.from_mapping(
            {
                "scene_count": 3,
                "split_scene_counts": {
                    "train": 1,
                    "validation": 1,
                    "test": 1,
                },
                "multi_object": True,
                "maximum_physics_attempts_per_object": 4,
                "timeline": _timeline_mapping(min_tracks=1),
                "device": "cpu",
            }
        )
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        BLCSTrajectorySourceSettings.from_mapping(
            {
                "scene_count": 3,
                "split_scene_counts": {
                    "train": 1,
                    "validation": 1,
                    "test": 1,
                },
                "multi_object": True,
                "maximum_physics_attempts_per_object": 4,
                "timeline": _timeline_mapping(),
                "device": "cpu",
                "unexpected": 1,
            }
        )


def test_provider_uses_only_public_source_and_preserves_every_semantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_module = __import__(
        "src.synthetic_data_generation.dataset.blcs.source", fromlist=["source"]
    )
    monkeypatch.setattr(
        source_module,
        "BLCSPhysicsTrajectorySource",
        _FakePublicPhysicsSource,
    )
    provider = PhysicsBLCSTrajectoryProvider(
        generator_config=object.__new__(BLCSGeneratorConfiguration),
        settings=_settings(),
    )

    provider.preflight(scene_id="B00", seed=17)
    first = provider.load(scene_id="B00", seed=17)
    repeated = provider.load(scene_id="B00", seed=17)
    changed = provider.load(scene_id="B00", seed=18)

    assert [trajectory.split for trajectory in first] == [
        "train",
        "validation",
        "test",
    ]
    assert [trajectory.trajectory_id for trajectory in first] == [
        "B00-blcs-000000",
        "B00-blcs-000001",
        "B00-blcs-000002",
    ]
    assert all(trajectory.frame_count == 4 for trajectory in first)
    assert all(trajectory.object_count == 2 for trajectory in first)
    assert all(trajectory.fps == 30.0 for trajectory in first)
    assert all(trajectory.present.all() for trajectory in first)
    for trajectory in first:
        metadata = trajectory.source_metadata
        assert metadata["source_frame_count"] == 4
        assert metadata["fps"] == 30.0
        assert metadata["simulation_fps"] == 240.0
        assert metadata["seed"] in {17, 18, 19}
        assert isinstance(metadata["physics_sources"], list)
        proposals = metadata["physics_proposals"]
        assert isinstance(proposals, list)
        accepted_attempts: list[int] = []
        for record in proposals:
            assert isinstance(record, Mapping)
            accepted_attempt = record["accepted_attempt"]
            assert isinstance(accepted_attempt, int)
            accepted_attempts.append(accepted_attempt)
        assert accepted_attempts == [1, 1]
        assert [track.object_id for track in trajectory.tracks] == [
            "ball-001",
            "ball-002",
        ]
        assert all(
            track.source_frame_indices == (0, 1, 2, 3)
            for track in trajectory.tracks
        )
    for left, right in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(left.positions_court_m, right.positions_court_m)
        np.testing.assert_array_equal(left.present, right.present)
        assert left.source_metadata == right.source_metadata
    assert not np.array_equal(
        first[0].positions_court_m,
        changed[0].positions_court_m,
    )
