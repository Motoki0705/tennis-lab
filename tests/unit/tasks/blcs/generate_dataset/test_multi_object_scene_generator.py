"""Tests for bounded full-physics proposal selection."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData
from src.tasks.blcs.generate_dataset.simulation.errors import (
    FullPhysicsLandingError,
    FullPhysicsProposalError,
)


def _timeline() -> TimelineConfig:
    return TimelineConfig(
        num_frames=4,
        min_tracks=2,
        max_tracks=2,
        max_concurrent=2,
        min_reuse_gap_frames=0,
        start_index_range=(0, 0),
        min_active_frames=2,
        overlap_probability=1.0,
        min_gap_frames=0,
        max_gap_frames=0,
    )


class _ProposalSource:
    config = object()

    def __init__(self, *, failures: int) -> None:
        self.failures = failures
        self.attempts = 0

    def sample_from_cell(self) -> int:
        return 0

    def sample_side(self) -> str:
        return "near"

    def generate_scene(
        self,
        from_cell: int,
        side: str,
        scene_id: str,
    ) -> BLCSSceneData:
        del from_cell, side
        self.attempts += 1
        if self.attempts <= self.failures:
            raise FullPhysicsLandingError(f"rejected proposal {self.attempts}")
        positions = torch.zeros((2, 3), dtype=torch.float32)
        return BLCSSceneData(
            scene_id=scene_id,
            initial_from_cell=0,
            initial_from_side="near",
            rally_length=1,
            end_reason="test",
            winner_side=None,
            shots=[],
            ball_pos_world=positions,
            ball_pos_norm=positions.clone(),
            ball_vel_world=positions.clone(),
            cameras=[],
            num_cameras_sampled=0,
            fps_out=30,
            sim_fps=120,
            physics_config_dict={},
            court_config_dict={},
            num_balls=1,
        )


def test_full_physics_proposals_retry_only_typed_rejections() -> None:
    source = _ProposalSource(failures=2)
    generator = MultiBallSceneGenerator(
        source,
        timeline=_timeline(),
        maximum_physics_attempts_per_object=4,
    )

    scene, diagnostic = generator._generate_ball("scene-ball-1")

    assert scene.scene_id == "scene-ball-1"
    assert source.attempts == 3
    assert diagnostic["accepted_attempt"] == 3
    rejected_attempts = diagnostic["rejected_attempts"]
    assert isinstance(rejected_attempts, list)
    attempts: list[int] = []
    for record in rejected_attempts:
        assert isinstance(record, dict)
        attempt = record.get("attempt")
        assert isinstance(attempt, int)
        attempts.append(attempt)
    assert attempts == [1, 2]


def test_full_physics_proposals_fail_closed_after_explicit_bound() -> None:
    source = _ProposalSource(failures=3)
    generator = MultiBallSceneGenerator(
        source,
        timeline=_timeline(),
        maximum_physics_attempts_per_object=2,
    )

    with pytest.raises(FullPhysicsProposalError, match="exhausted 2 attempts"):
        generator._generate_ball("scene-ball-2")

    assert source.attempts == 2
