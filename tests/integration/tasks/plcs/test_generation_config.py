from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.utils.configuration import UnknownConfigurationKeyError

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


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
        overrides.append("model=track_query_reference")
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name=config_name,
            overrides=overrides,
        )

    runtime = PLCSTrainingConfig.from_config(config)
    assert runtime.court_keypoint_contract.selector == court_selector
