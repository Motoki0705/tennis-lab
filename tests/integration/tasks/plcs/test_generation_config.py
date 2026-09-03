from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.utils.configuration import UnknownConfigurationKeyError

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()
_DATA_CONFIG_DIR = _CONFIG_DIR / "data"

# Keep this table as the executable catalogue of the ten public PLCS data
# profiles.  Each row documents the root training boundary needed to validate
# the profile and its externally visible dataset/view/model contracts.
_TRAINING_PROFILES = (
    # profile, root config, extra overrides, scene, backend, views, court, model
    (
        "singleview_frame",
        "train",
        ("model=frame",),
        "plcs/single_object",
        "default",
        (1, 1),
        "physical_v1",
        "plcs",
    ),
    (
        "singleview_sequence",
        "train",
        ("model=frame",),
        "plcs/single_object",
        "default",
        (1, 1),
        "physical_v1",
        "plcs",
    ),
    (
        "multiview_sequence",
        "train",
        (),
        "plcs/single_object",
        "default",
        (3, 5),
        "physical_v1",
        "plcs_multiview_axial",
    ),
    (
        "chunked_multiview_sequence",
        "train_chunked",
        (),
        "plcs/single_object",
        "chunked",
        (3, 5),
        "physical_v1",
        "plcs_multiview_axial",
    ),
    (
        "tracking",
        "train_tracking",
        (),
        "plcs/multi_object",
        "default",
        (3, 5),
        "physical_v1",
        "plcs_track_query",
    ),
    (
        "tracking_chunked",
        "train_tracking_chunked",
        (),
        "plcs/multi_object",
        "chunked",
        (3, 5),
        "physical_v1",
        "plcs_track_query",
    ),
    (
        "singleview_sequence_broadcast",
        "train",
        ("model=frame",),
        "plcs/single_object_broadcast",
        "default",
        (1, 1),
        "physical_v1",
        "plcs",
    ),
    (
        "multiview_sequence_broadcast",
        "train",
        (),
        "plcs/single_object_broadcast",
        "default",
        (2, 2),
        "physical_v1",
        "plcs_multiview_axial",
    ),
    (
        "tracking_broadcast",
        "train_tracking",
        (),
        "plcs/multi_object_broadcast",
        "default",
        (2, 2),
        "physical_v1",
        "plcs_track_query",
    ),
    (
        "tracking_camera_view_v2",
        "train_tracking",
        (),
        "plcs/multi_object_camera_view_v2",
        "default",
        (3, 5),
        "camera_view_v2",
        "plcs_track_query_reference",
    ),
)


def _compose_training_profile(
    config_name: str,
    profile: str,
    extra_overrides: tuple[str, ...],
):
    """Compose and validate one public profile through its train boundary."""
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name=config_name,
            overrides=[f"data={profile}", *extra_overrides],
        )
    runtime = PLCSTrainingConfig.from_config(config)
    return config, runtime


def test_public_data_profile_catalogue_is_exactly_ten() -> None:
    profiles = sorted(
        path.stem
        for path in _DATA_CONFIG_DIR.glob("*.yaml")
        if not path.stem.startswith("_")
    )
    assert profiles == sorted(row[0] for row in _TRAINING_PROFILES)


@pytest.mark.parametrize(
    (
        "profile",
        "config_name",
        "extra_overrides",
        "scene_dir",
        "backend",
        "views",
        "court_selector",
        "model_name",
    ),
    _TRAINING_PROFILES,
    ids=[row[0] for row in _TRAINING_PROFILES],
)
def test_public_data_profiles_compose_and_validate_contracts(
    profile: str,
    config_name: str,
    extra_overrides: tuple[str, ...],
    scene_dir: str,
    backend: str,
    views: tuple[int, int],
    court_selector: str,
    model_name: str,
) -> None:
    config, runtime = _compose_training_profile(config_name, profile, extra_overrides)

    assert runtime.model.name == model_name
    assert runtime.court_keypoint_contract.selector == court_selector
    assert config.data.scene_dir == scene_dir
    assert config.data.backend == backend
    assert tuple(config.data.num_views_range) == views
    assert config.data.camera_mode == "random"
    assert runtime.data.scene_dir == runtime.paths.resolver.roots.data_root / scene_dir

    if views[0] > 1:
        # The random 3-5-view profiles need two cameras as a lower bound;
        # broadcast profiles are exactly two views and therefore also use 2.
        assert config.data.min_cameras == 2
    if profile in {"tracking_broadcast", "tracking_camera_view_v2"}:
        assert config.data.evaluation_reference_camera_id == "camera_1"


def test_public_profiles_cover_each_plcs_dataset_once_or_more() -> None:
    scene_dirs = {row[3] for row in _TRAINING_PROFILES}
    assert scene_dirs == {
        "plcs/single_object",
        "plcs/multi_object",
        "plcs/single_object_broadcast",
        "plcs/multi_object_broadcast",
        "plcs/multi_object_camera_view_v2",
    }


def test_camera_view_profile_selects_reference_contract_without_overrides() -> None:
    config, runtime = _compose_training_profile(
        "train_tracking", "tracking_camera_view_v2", ()
    )

    assert config.model.name == "plcs_track_query_reference"
    assert config.court_keypoints.selector == "camera_view_v2"
    assert runtime.model.name == "plcs_track_query_reference"
    assert runtime.court_keypoint_contract.selector == "camera_view_v2"


@pytest.mark.parametrize(
    ("generation", "camera", "output_dir", "camera_layout"),
    (
        ("single_object", "default", "plcs/single_object", "fixed"),
        ("multi_object", "default", "plcs/multi_object", "fixed"),
        (
            "single_object",
            "broadcast",
            "plcs/single_object_broadcast",
            "broadcast",
        ),
        (
            "multi_object",
            "broadcast",
            "plcs/multi_object_broadcast",
            "broadcast",
        ),
    ),
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

    runtime = PLCSGenerationConfig.from_config(config)

    assert runtime.output_dir == tmp_path / output_dir
    assert config.court_keypoints.selector == "physical_v1"
    assert config.generation.mode == generation
    assert config.camera.layout == camera_layout


@pytest.mark.parametrize(
    ("config_name", "scene_dir", "chunks_dir"),
    (
        ("train", "plcs/single_object", None),
        (
            "train_chunked",
            "plcs/single_object",
            "plcs/single_object/chunks",
        ),
        ("train_tracking", "plcs/multi_object", None),
        (
            "train_tracking_chunked",
            "plcs/multi_object",
            "plcs/multi_object/chunks",
        ),
    ),
)
def test_training_profiles_use_canonical_dataset_paths(
    config_name: str,
    scene_dir: str,
    chunks_dir: str | None,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name=config_name)

    runtime = PLCSTrainingConfig.from_config(config)

    assert config.data.scene_dir == scene_dir
    assert runtime.data.scene_dir == runtime.paths.resolver.roots.data_root / scene_dir
    if chunks_dir is None:
        assert "chunk" not in config.data
    else:
        assert config.data.chunk.chunks_dir == chunks_dir


@pytest.mark.parametrize("court_selector", ["physical_v1", "camera_view_v2"])
def test_generation_composes_court_keypoint_contract(
    court_selector: str,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=[
                f"court_keypoints={court_selector}",
                "run.device=cpu",
                "run.num_workers=1",
                "simulation.num_scenes=1",
            ],
        )
    runtime = PLCSGenerationConfig.from_config(config)
    assert runtime.court_keypoint_contract.selector == court_selector


def test_generation_rejects_unknown_typed_selector() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=["court_keypoints.selector=unknown"],
        )
    with pytest.raises(ValueError, match="Unknown court keypoint selector"):
        PLCSGenerationConfig.from_config(config)


def test_generation_rejects_unknown_selector_field() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="generate_dataset",
            overrides=["+court_keypoints.fallback=true"],
        )
    with pytest.raises(UnknownConfigurationKeyError):
        PLCSGenerationConfig.from_config(config)


@pytest.mark.parametrize("config_name", ["train", "train_tracking"])
@pytest.mark.parametrize("court_selector", ["physical_v1", "camera_view_v2"])
def test_training_composes_court_keypoint_contract(
    config_name: str,
    court_selector: str,
) -> None:
    overrides = [f"court_keypoints={court_selector}"]
    if config_name == "train_tracking" and court_selector == "camera_view_v2":
        overrides.append("model=tracking_query_reference")
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name=config_name,
            overrides=overrides,
        )

    runtime = PLCSTrainingConfig.from_config(config)
    assert runtime.court_keypoint_contract.selector == court_selector
