"""Tests for the deterministic production BLCS physics trajectory source."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.dataset.blcs.source import (
    BLCSTrajectorySourceSettings,
    PhysicsBLCSTrajectoryProvider,
)
from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig


def _timeline(*, min_tracks: int = 2) -> TimelineConfig:
    return TimelineConfig(
        num_frames=4,
        min_tracks=min_tracks,
        max_tracks=2,
        max_concurrent=2,
        min_reuse_gap_frames=0,
        start_index_range=(0, 0),
        min_active_frames=2,
        overlap_probability=1.0,
        min_gap_frames=0,
        max_gap_frames=0,
    )


def _settings() -> BLCSTrajectorySourceSettings:
    return BLCSTrajectorySourceSettings.from_mapping(
        {
            "scene_count": 3,
            "split_scene_counts": {"train": 1, "validation": 1, "test": 1},
            "multi_object": True,
            "maximum_physics_attempts_per_object": 4,
            "timeline": {
                "num_frames": 4,
                "min_tracks": 2,
                "max_tracks": 2,
                "max_concurrent": 2,
                "min_reuse_gap_frames": 0,
                "start_index_range": [0, 0],
                "min_active_frames": 2,
                "overlap_probability": 1.0,
                "min_gap_frames": 0,
                "max_gap_frames": 0,
            },
            "device": "cpu",
        }
    )


class _FakeBaseGenerator:
    def __init__(self, *, config: GeneratorConfig, device: str) -> None:
        self.config = config
        self.device = device


class _FakeMultiGenerator:
    def __init__(
        self,
        base: _FakeBaseGenerator,
        *,
        timeline: TimelineConfig,
        maximum_physics_attempts_per_object: int,
        rng: random.Random,
    ) -> None:
        self.base = base
        self.timeline = timeline
        self.maximum_physics_attempts_per_object = maximum_physics_attempts_per_object
        self.rng = rng

    def generate_scene(self, scene_id: str) -> SimpleNamespace:
        offset = self.rng.random() + random.random() + float(np.random.random())
        offset += float(torch.rand(()).item())
        positions = torch.zeros((4, 2, 3), dtype=torch.float32)
        positions[:, :, 0] = offset
        positions[:, :, 2] = 1.5
        present = torch.ones((4, 2), dtype=torch.bool)
        return SimpleNamespace(
            scene_id=scene_id,
            ball_pos_world=positions,
            ball_vel_world=torch.zeros_like(positions),
            ball_present=present,
            num_balls=2,
            fps_out=30,
            track_instances=[
                {
                    "track_id": index,
                    "source_scene_id": f"{scene_id}-source-{index}",
                    "source_start": 0,
                    "source_end": 4,
                    "birth_frame": 0,
                    "death_frame": 4,
                }
                for index in range(2)
            ],
            physics_proposal_diagnostics=[
                {
                    "source_scene_id": f"{scene_id}-source-{index}",
                    "accepted_attempt": 1,
                    "maximum_attempts": self.maximum_physics_attempts_per_object,
                    "rejected_attempts": [],
                }
                for index in range(2)
            ],
        )


def test_source_settings_fail_closed_on_counts_mode_and_unknown_fields() -> None:
    settings = _settings()

    assert settings.split_sequence() == ("train", "validation", "test")
    with pytest.raises(ValueError, match="sum exactly"):
        replace(settings, scene_count=4)
    with pytest.raises(ValueError, match="must be multi-object"):
        replace(settings, multi_object=False)
    with pytest.raises(ValueError, match="multiple objects"):
        replace(settings, timeline=_timeline(min_tracks=1))
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
                "timeline": _timeline(),
                "device": "cpu",
                "unexpected": 1,
            }
        )


def test_physics_provider_is_seeded_and_preserves_full_multi_object_scenes(
    monkeypatch,
) -> None:
    source_module = __import__(
        "src.synthetic_data_generation.dataset.blcs.source", fromlist=["source"]
    )
    monkeypatch.setattr(source_module, "BLCSSceneGenerator", _FakeBaseGenerator)
    monkeypatch.setattr(source_module, "MultiBallSceneGenerator", _FakeMultiGenerator)
    generator_config = object.__new__(GeneratorConfig)
    provider = PhysicsBLCSTrajectoryProvider(
        generator_config=generator_config,
        settings=_settings(),
    )

    first = provider.load(scene_id="B00", seed=17)
    second = provider.load(scene_id="B00", seed=17)
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
    for trajectory in first:
        proposals = trajectory.source_metadata["physics_proposals"]
        assert isinstance(proposals, list)
        accepted_attempts: list[int] = []
        for record in proposals:
            assert isinstance(record, Mapping)
            accepted_attempt = record["accepted_attempt"]
            assert isinstance(accepted_attempt, int)
            accepted_attempts.append(accepted_attempt)
        assert accepted_attempts == [1, 1]
    assert all(
        trajectory.tracks[0].source_frame_indices == (0, 1, 2, 3)
        for trajectory in first
    )
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            left.positions_court_m,
            right.positions_court_m,
        )
    assert not np.array_equal(
        first[0].positions_court_m,
        changed[0].positions_court_m,
    )
