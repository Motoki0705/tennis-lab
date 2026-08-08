"""CPU integration of config, alignment layout, planning, sampling, and assignment."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)


def test_court_domain_resolves_production_quantities_and_balanced_courts() -> None:
    layout = _layout()
    configuration = CourtDatasetConfiguration.from_mapping(
        OmegaConf.to_container(
            OmegaConf.load(
                Path("src/synthetic_data_generation/configs/dataset/court/train.yaml")
            ),
            resolve=True,
        )
    )
    plan = build_court_dataset_plan(
        scene_id="B00",
        profile="train",
        cameras=_captured_cameras(),
        layout=layout,
        configuration=configuration,
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
    )
    assert len(plan.groups) >= 24
    assert 2_000 <= plan.proposal_count <= 5_000
    assert max(group.maximum_adjacent_step_m for group in plan.groups) <= 1.05
    global_counts = Counter(
        group.target_court.court_instance_id for group in plan.groups
    )
    assert set(global_counts) == {court.court_instance_id for court in layout.courts}
    assert max(global_counts.values()) - min(global_counts.values()) <= 1
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for group in plan.groups:
        by_split[group.split.value][group.target_court.court_instance_id] += 1
    assert all(
        max(counts.values()) - min(counts.values()) <= 1
        for counts in by_split.values()
    )
    assert all(
        group.trajectory.center_court_instance_id is None
        or group.trajectory.center_court_instance_id
        == group.target_court.court_instance_id
        for group in plan.groups
    )


def _layout() -> MultiCourtLayout:
    courts = []
    for index, x in enumerate((-8.0, 8.0)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = x
        scene_from_court = RigidTransform.from_matrix(matrix)
        courts.append(
            CourtInstance(
                court_instance_id=f"court-{index}",
                candidate_id=f"candidate-{index}",
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit_status="accepted",
                fit_metrics={"rms_error_m": 0.01},
                holdout_status="accepted",
                holdout_metrics={"rms_error_m": 0.02},
            )
        )
    return MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
        primary_court_instance_id="court-0",
    )


def _captured_cameras() -> tuple[SceneCamera, ...]:
    result = []
    for index, angle in enumerate(
        np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    ):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (
            24.0 * math.cos(angle),
            30.0 * math.sin(angle),
            6.0 + 2.0 * math.sin(angle),
        )
        result.append(
            SceneCamera(
                camera_id=f"captured-{index}",
                source_frame_index=index,
                width=64,
                height=48,
                intrinsics=(
                    100.0,
                    0.0,
                    31.5,
                    0.0,
                    100.0,
                    23.5,
                    0.0,
                    0.0,
                    1.0,
                ),
                camera_to_scene=RigidTransform.from_matrix(matrix),
                image_path=f"images/{index}.png",
            )
        )
    return tuple(result)
