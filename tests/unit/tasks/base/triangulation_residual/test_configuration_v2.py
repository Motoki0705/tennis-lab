"""Historical recipes stay explicit, and the v2 fit cannot mix camera perturbations."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.base.triangulation_residual.configuration import validate_config


def recipe(task="blcs"):
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / f"src/tasks/{task}/configs"), version_base="1.3"
    ):
        return compose(config_name="train_triangulation_residual_v2")


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_v2_recipe_has_six_candidates_and_three_evaluation_views(task):
    config = validate_config(recipe(task))
    assert config.v2 is not None
    assert config.v2.camera_preset == "four_corners_front_pair"
    assert config.data.max_views == 6
    assert config.v2.evaluation_views == 3
    assert config.model.name.endswith("_v2")


@pytest.mark.parametrize(
    "key,value",
    [
        ("camera_center_std_m", 0.1),
        ("camera_rotation_std_deg", 0.1),
        ("camera_focal_log_std", 0.1),
    ],
)
def test_v2_rejects_independent_calibration_perturbation(key, value):
    config = recipe()
    config.corruption[key] = value
    with pytest.raises(ValueError, match="independent camera"):
        validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("camera_preset", "broadcast"),
        ("evaluation_views", 7),
        ("loss_mode", "relative_gt_normalized"),
        ("persistent_min_seconds", 2.0),
    ],
)
def test_v2_rejects_ambiguous_or_invalid_policy(key, value):
    config = recipe()
    config.v2[key] = value
    with pytest.raises(ValueError):
        validate_config(config)
