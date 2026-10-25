from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitSamplingPolicy,
    OrbitShape,
)
from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    TrajectoryMethod,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    build_candidate_plan,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def inputs(method, captured_cameras, multi_court_layout, preset="sfm_bounded"):
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
                overrides=[f"dataset/court={preset}"],
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


@pytest.mark.parametrize(
    "method",
    [m for m in TrajectoryMethod if m is not TrajectoryMethod.SAVED_TRAIN_TRAJECTORIES],
)
def test_every_method_retains_common_camera_target_and_projection(
    method, captured_cameras, multi_court_layout
):
    kwargs = inputs(method, captured_cameras, multi_court_layout)
    prefix = len(kwargs["cameras"]) if method is TrajectoryMethod.CAPTURE_LOCAL else 2
    coarse = build_candidate_plan(**kwargs, trajectory_count=prefix, density_level=0)
    # Reversing captured input order is part of the same plan, not a new asset.
    kwargs["cameras"] = list(reversed(kwargs["cameras"]))
    larger = build_candidate_plan(
        **kwargs, trajectory_count=prefix + 1, density_level=1
    )
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
            assert (
                candidate["provenance"]["replay_rule_version"]
                == "saved_preset_parameterization_v2"
            )


@pytest.mark.parametrize("preset", ["v3", "sfm_bounded"])
def test_grid_replay_preserves_nonplanar_circle_xyz_at_every_arc_point(
    preset, captured_cameras, multi_court_layout, identity_metric_adapter
):
    kwargs = inputs(
        TrajectoryMethod.GRID_BUDGET_REPLAY,
        captured_cameras,
        multi_court_layout,
        preset,
    )
    config = kwargs["replay_configuration"]
    # Both legacy vertical frequencies must have a non-trivial saved phase.
    config = replace(config, sampling=replace(config.sampling, seed=1))
    kwargs["replay_configuration"] = config
    original = build_court_dataset_plan(
        scene_id="fixture",
        profile="test",
        cameras=kwargs["cameras"],
        layout=multi_court_layout,
        configuration=config,
        metric_adapter=identity_metric_adapter,
    )
    policy = OrbitSamplingPolicy.from_configuration(config.sampling)
    # Match the legacy arc table and counts so every stored camera is comparable.
    kwargs["recipe"] = replace(
        kwargs["recipe"],
        arc_intervals=4096,
        base_step_m=policy.max_arc_step_m,
        minimum_points=policy.minimum_sample_count,
        count_multiple=policy.sample_count_multiple,
    )
    replay = build_candidate_plan(
        **kwargs, trajectory_count=len(original.groups), density_level=0
    )
    by_id = {g.trajectory.trajectory_id: g for g in original.groups}
    checked = set()
    for candidate in replay["trajectory_groups"]:
        group = by_id[candidate["provenance"]["saved_trajectory"]["trajectory_id"]]
        spec = group.trajectory
        if spec.shape is not OrbitShape.CIRCLE or spec.vertical_amplitude_m == 0:
            continue
        if spec.orientation_radians == 0 or spec.vertical_phase_radians == 0:
            continue
        old_path = sample_uniform_arc_length(spec, group.center, policy)
        positions = np.asarray(
            [s["camera"]["camera_to_scene"] for s in candidate["samples"]]
        ).reshape(-1, 4, 4)[:, :3, 3]
        np.testing.assert_allclose(
            positions, old_path.points_scene_m, rtol=0, atol=1e-9
        )
        checked.add(spec.vertical_cycles)
    assert checked


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
