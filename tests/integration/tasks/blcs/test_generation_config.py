from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.blcs.configuration import (
    parse_court_keypoint_contract,
    parse_generation_run,
    validate_generation_boundary,
    validate_training_boundary,
)
from src.utils.configuration import ConfigurationTypeError, SemanticConfigurationError

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()

_TRAINING_PROFILES = (
    (
        "multiview_sequence",
        "train",
        ("data=multiview_sequence",),
        "blcs/single_object",
        "default",
        (3, 5),
        "physical_v1",
        "blcs_multiview_axial",
    ),
    (
        "multiview_chunked_sequence",
        "train_chunked",
        (),
        "blcs/single_object",
        "chunked",
        (3, 5),
        "physical_v1",
        "blcs_multiview_axial",
    ),
)


def test_generation_default_uses_canonical_single_object_path() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="generate_dataset")

    runtime, _resolver = parse_generation_run(config)

    assert (
        runtime.output_dir
        == (_CONFIG_DIR.parents[3] / "data/blcs/single_object").resolve()
    )


@pytest.mark.parametrize(
    ("generation", "camera", "output_dir", "camera_layout"),
    (("single_object", "default", "blcs/single_object", "fixed"),),
)
def test_generation_variants_resolve_canonical_dataset_paths(
    tmp_path: Path,
    generation: str,
    camera: str,
    output_dir: str,
    camera_layout: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=[
                f"paths.data_root={tmp_path.as_posix()}",
                f"generation={generation}",
                f"camera={camera}",
                f"run.output_dir={output_dir}",
            ],
        )

    runtime, _resolver = parse_generation_run(config)

    assert runtime.output_dir == tmp_path / output_dir
    assert config.court_keypoints.selector == "physical_v1"
    assert config.generation.mode == generation
    assert config.camera.layout == camera_layout


@pytest.mark.parametrize(
    ("config_name", "scene_dir", "chunks_dir"),
    (
        ("train", "blcs/single_object", None),
        ("train_chunked", "blcs/single_object", "blcs/single_object/chunks"),
    ),
)
def test_training_profiles_use_canonical_dataset_paths(
    config_name: str,
    scene_dir: str,
    chunks_dir: str | None,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name=config_name)

    validate_training_boundary(config)

    assert config.data.scene_dir == scene_dir
    if chunks_dir is None:
        assert "chunk" not in config.data
    else:
        assert config.data.chunk.chunks_dir == chunks_dir


@pytest.mark.parametrize(
    ("config_name", "reprojection_weight"),
    (
        ("train", 0.0),
        ("train_chunked", 0.1),
        ("train_chunked_gan", 0.0),
    ),
)
def test_standard_training_profiles_select_loss_without_training_loss_fields(
    config_name: str,
    reprojection_weight: float,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name=config_name)

    validate_training_boundary(config)

    assert config.loss.reprojection_weight == reprojection_weight
    assert "position_weight" in config.loss
    assert config.training.trainer.log_every_n_steps == 100
    assert set(config.training).isdisjoint(
        {
            "position_loss_weight",
            "position_axis_weights",
            "reprojection_loss_weight",
            "smoothness_loss_weight",
            "gravity_loss_weight",
            "smoothness_order",
            "smoothness_beta",
            "gravity_beta",
            "smoothness_axis_weights",
        }
    )


@pytest.mark.parametrize(
    (
        "profile",
        "config_name",
        "overrides",
        "scene_dir",
        "backend",
        "views",
        "court_selector",
        "model_name",
    ),
    _TRAINING_PROFILES,
    ids=[row[0] for row in _TRAINING_PROFILES],
)
def test_all_public_data_profiles_compose_and_validate(
    profile: str,
    config_name: str,
    overrides: tuple[str, ...],
    scene_dir: str,
    backend: str,
    views: tuple[int, int],
    court_selector: str,
    model_name: str,
) -> None:
    """The profile matrix is the executable data/model/court contract."""
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name=config_name, overrides=list(overrides))

    validate_training_boundary(config)

    assert config.data.scene_dir == scene_dir
    assert config.data.backend == backend
    assert tuple(config.data.num_views_range) == views
    assert config.court_keypoints.selector == court_selector
    assert config.model.name == model_name


def test_blcs_public_data_profiles_cover_all_datasets() -> None:
    profiles = {
        path.stem
        for path in (_CONFIG_DIR / "data").glob("*.yaml")
        if not path.name.startswith("_")
    }
    assert profiles == {"multiview_sequence", "multiview_chunked_sequence"}
    assert profiles == {row[0] for row in _TRAINING_PROFILES}
    assert {row[3] for row in _TRAINING_PROFILES} == {"blcs/single_object"}


@pytest.mark.parametrize(
    ("selector", "contract_id"),
    [("physical_v1", "physical_courtkp20_v1")],
)
def test_generation_composes_explicit_court_keypoint_selector(
    selector: str,
    contract_id: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=[f"court_keypoints={selector}"],
        )
    validate_generation_boundary(config)
    assert parse_court_keypoint_contract(config).contract_id == contract_id


def test_generation_default_court_keypoints_remains_physical_v1() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="generate_dataset")
    assert config.court_keypoints.selector == "physical_v1"


def test_single_object_generation_has_explicit_bounded_physics_budget() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="generate_dataset")

    assert config.generation.maximum_physics_attempts_per_scene == 64
    validate_generation_boundary(config)


@pytest.mark.parametrize("maximum_attempts", (0, -1))
def test_single_object_generation_rejects_nonpositive_physics_budget(
    maximum_attempts: int,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=[
                (f"generation.maximum_physics_attempts_per_scene={maximum_attempts}"),
            ],
        )

    with pytest.raises(
        SemanticConfigurationError,
        match="maximum_physics_attempts_per_scene must be positive",
    ):
        validate_generation_boundary(config)


def test_generation_rejects_unknown_or_untyped_court_selector() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="generate_dataset")
    for invalid in ("v2", 2):
        with open_dict(config.court_keypoints):
            config.court_keypoints.selector = invalid
        with pytest.raises((SemanticConfigurationError, ConfigurationTypeError)):
            validate_generation_boundary(config)
