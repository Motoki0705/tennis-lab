"""Adversarial checks for the experimental plan, with portable CPU fixtures."""

# Match the repository tests.* policy for this archived test module.
# mypy: disallow_untyped_decorators=False

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)

probe = importlib.import_module(
    "knowledge.runs.run-court-nested-plan-cpu-20261022.prototype"
)


@pytest.fixture
def scene() -> Any:
    court = CourtInstance(
        "court-0",
        "candidate-0",
        RigidTransform.identity(),
        RigidTransform.identity(),
        "accepted",
        {},
        "accepted",
        {},
    )
    layout = MultiCourtLayout(
        (court,), (-40.0, -40.0, -2.0, 40.0, 40.0, 10.0), "court-0"
    )
    cameras = []
    for i, (x, y) in enumerate(
        [(-20.0, -30.0), (20.0, -30.0), (20.0, 30.0), (-20.0, 30.0)]
    ):
        transform = np.eye(4)
        transform[:3, 3] = (x, y, 2.0)
        cameras.append(
            SceneCamera(
                f"capture-{i}",
                i,
                640,
                360,
                (400.0, 0.0, 319.5, 0.0, 400.0, 179.5, 0.0, 0.0, 1.0),
                RigidTransform.from_matrix(transform),
                f"images/{i}.png",
            )
        )
    return SimpleNamespace(
        sid="fixture", cameras=tuple(cameras), alignment=SimpleNamespace(layout=layout)
    )


@pytest.mark.parametrize("level", [1, 2, 4])
def test_density_retains_camera_target_and_teacher(scene: Any, level: int) -> None:
    # Non-planar rectangle includes corners, vertical motion and per-point jitter.
    track = probe.make_track(scene, 6)
    coarse = probe.plan(track, 0)
    fine = probe.plan(track, level)
    assert len(fine) == len(coarse) * 2**level
    assert probe.compare(coarse, fine) == len(coarse)
    assert len({point["id"] for point in fine}) == len(fine)
    positions = np.array(list(track.points(level).values()))
    steps = np.linalg.norm(
        np.diff(np.vstack((positions, positions[:1])), axis=0), axis=1
    )
    assert steps.max() <= probe.STEP_M / 2**level + 1e-5


def test_track_append_and_generation_order_preserve_existing_geometry(
    scene: Any,
) -> None:
    before = probe.plan(probe.make_track(scene, 0), 0)
    probe.make_track(scene, 50)
    after = probe.plan(probe.make_track(scene, 0), 0)
    assert before == after
    assert (
        probe.make_track(scene, 0, seed=probe.SEED + 1).family
        != probe.make_track(scene, 0).family
    )


def test_fixed_evaluation_ignores_training_seed_and_keeps_disjoint_families(
    scene: Any,
) -> None:
    evaluation = probe.freeze_evaluation(scene)
    before = [point for track in evaluation for point in probe.plan(track, 0)]
    probe.plan(probe.make_track(scene, 1, seed=probe.SEED + 1), 2)
    after = [
        point
        for track in probe.freeze_evaluation(scene)
        for point in probe.plan(track, 0)
    ]
    assert before == after
    assert evaluation[0].family != evaluation[1].family
    assert probe.separated(evaluation[0], evaluation[1])
    assert not probe.separated(evaluation[0], evaluation[0])
    assert {point["split"] for point in before} == {"validation", "test"}


def test_selection_is_order_independent_and_shortage_is_explicit() -> None:
    points = [{"id": str(i), "family": str(i % 3), "reason": "ok"} for i in range(60)]
    selected = probe.take_balanced(points, 12)
    assert selected == probe.take_balanced(list(reversed(points)), 12)
    assert {point["family"] for point in selected} == {"0", "1", "2"}
    with pytest.raises(ValueError, match="target=100, available="):
        probe.take_balanced(points, 100)


def test_density_birth_shortage_cannot_be_filled_by_coarse_surplus() -> None:
    # Plenty of unused coarse points do not cure an empty new-density layer.
    coarse_ids = {f"coarse-{i}" for i in range(100)}
    dense_points = [{"id": sid, "family": "one", "reason": "ok"} for sid in coarse_ids]
    new = [point for point in dense_points if point["id"] not in coarse_ids]
    with pytest.raises(ValueError, match="target=24, available=0"):
        probe.take_balanced(new, 24)


def test_output_inside_input_root_is_rejected_without_creating_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    output = tmp_path / "forbidden-output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prototype.py", "--scenes-root", str(tmp_path), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="read-only input scenes"):
        probe.main()
    assert not output.exists()
