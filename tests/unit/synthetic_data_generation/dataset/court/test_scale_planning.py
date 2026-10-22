from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
)
from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    TrajectoryMethod,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    build_candidate_plan,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def inputs(method, captured_cameras, multi_court_layout):
    cameras = list(captured_cameras)
    for i, (x, y) in enumerate(
        ((-4, -4), (0, -4), (4, -4), (-4, 0), (0, 0), (4, 0), (-4, 4), (0, 4), (4, 4))
    ):
        matrix = np.eye(4)
        matrix[:3, 3] = [x, y, 2.0]
        cameras.append(
            replace(
                cameras[0],
                camera_id=f"inside-{i}",
                camera_to_scene=RigidTransform.from_matrix(matrix),
            )
        )
    config = None
    if method is TrajectoryMethod.GRID_BUDGET_REPLAY:
        config_dir = Path("src/synthetic_data_generation/configs").resolve()
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(
                config_name="run_scene_pipeline",
                overrides=["dataset/court=sfm_bounded"],
            )
        config = CourtDatasetConfiguration.from_mapping(
            OmegaConf.to_container(cfg.dataset.court, resolve=True)
        )
    return dict(
        scene_id="fixture",
        method=method,
        recipe=GeometryRecipe(base_step_m=12, minimum_points=4),
        seed=24,
        cameras=cameras,
        layout=multi_court_layout,
        replay_configuration=config,
    )


@pytest.mark.parametrize("method", list(TrajectoryMethod))
def test_every_method_retains_common_camera_target_and_projection(
    method, captured_cameras, multi_court_layout
):
    kwargs = inputs(method, captured_cameras, multi_court_layout)
    coarse = build_candidate_plan(**kwargs, trajectory_count=2, density_level=0)
    # Reversing captured input order is part of the same plan, not a new asset.
    kwargs["cameras"] = list(reversed(kwargs["cameras"]))
    larger = build_candidate_plan(**kwargs, trajectory_count=3, density_level=1)
    before = {
        s["sample_id"]: s for g in coarse["trajectory_groups"] for s in g["samples"]
    }
    after = {
        s["sample_id"]: s for g in larger["trajectory_groups"] for s in g["samples"]
    }
    assert before
    assert before.keys() <= after.keys()
    assert all(value == after[key] for key, value in before.items())
    assert len(after) >= 2 * len(before)
    assert all(s["split"] == "train" for s in after.values())
    changed_seed = build_candidate_plan(
        **{**kwargs, "seed": 25}, trajectory_count=2, density_level=0
    )
    assert (
        coarse["trajectory_groups"][0]["trajectory_id"]
        != changed_seed["trajectory_groups"][0]["trajectory_id"]
    )
    if method is TrajectoryMethod.GRID_BUDGET_REPLAY:
        original = build_court_dataset_plan(
            scene_id="fixture",
            profile="test",
            cameras=kwargs["cameras"],
            layout=multi_court_layout,
            configuration=kwargs["replay_configuration"],
            metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
                np.eye(4)
            ),
        )
        by_id = {g.trajectory.trajectory_id: g for g in original.groups}
        for candidate in coarse["trajectory_groups"]:
            group = by_id[candidate["provenance"]["saved_trajectory"]["trajectory_id"]]
            assert (
                candidate["provenance"]["saved_trajectory"]
                == group.trajectory.to_dict()
            )
            assert candidate["provenance"]["saved_view"] == group.views[0].to_dict()


def test_method_requires_explicit_selection_and_no_fallback(
    captured_cameras, multi_court_layout
):
    kwargs = inputs(
        TrajectoryMethod.STRATIFIED_COUNTER, captured_cameras, multi_court_layout
    )
    with pytest.raises(ValueError, match="explicit"):
        build_candidate_plan(
            **{**kwargs, "method": "unknown"}, trajectory_count=1, density_level=0
        )
    with pytest.raises(ValueError, match="exactly"):
        build_candidate_plan(
            **{**kwargs, "method": TrajectoryMethod.GRID_BUDGET_REPLAY},
            trajectory_count=1,
            density_level=0,
        )
    # No admissible open pairs: fail as an exhausted sequence, never borrow an orbit.
    with pytest.raises(ValueError, match="exhausted"):
        build_candidate_plan(
            **{
                **kwargs,
                "method": TrajectoryMethod.OPEN_BEZIER,
                "cameras": captured_cameras,
            },
            trajectory_count=1,
            density_level=0,
        )
