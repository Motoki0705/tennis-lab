"""Dataset defaults and explicit overrides reach readable v2 stores via Hydra."""

from dataclasses import replace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.player_detection.configuration import (
    EvaluateConfig,
    GenerateDatasetConfig,
    PlayerTrainingConfig,
    PreviewConfig,
)
from src.tasks.player_detection.data.store import PlayerFrameStore
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from tests.support.tasks.player_detection.chat_root import (
    HEIGHT,
    WIDTH,
    make_synthetic_root,
)

pytestmark = pytest.mark.integration
_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/player_detection/configs"
_DEFAULT_VERSION = "chat-player-v2"
_OVERRIDE_VERSION = "alternate-player-v2"


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory: Path = tmp_path_factory.mktemp("player-datasets")
    source = make_synthetic_root(directory)
    data_root = directory / "input-data"
    for version in (_DEFAULT_VERSION, _OVERRIDE_VERSION):
        build_dataset(
            replace(
                source.config,
                dataset_dir=data_root / "player_detection" / version,
                version=version,
            )
        )
    assert not (data_root / "player_detection/chat-player-v1").exists()
    return data_root


@pytest.mark.parametrize(
    "config_name, section, config_type",
    [
        ("train", "data", PlayerTrainingConfig),
        ("evaluate", "evaluate", EvaluateConfig),
        ("preview_dataset", "preview", PreviewConfig),
    ],
)
@pytest.mark.parametrize(
    "dataset_override", [None, _OVERRIDE_VERSION], ids=["default", "override"]
)
def test_dataset_selection_resolves_under_data_root_and_reads_v2(
    config_name: str,
    section: str,
    config_type: type[PlayerTrainingConfig]
    | type[EvaluateConfig]
    | type[PreviewConfig],
    dataset_override: str | None,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    version = dataset_override or _DEFAULT_VERSION
    relative_dir = f"player_detection/{version}"
    overrides = [f"paths.project_root={tmp_path}", f"paths.data_root={dataset_root}"]
    if dataset_override is not None:
        overrides.append(f"{section}.dataset_dir={relative_dir}")
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        config = compose(config_name=config_name, overrides=overrides)

    assert OmegaConf.select(config, f"{section}.dataset_dir") == relative_dir
    runtime = config_type.from_config(config)
    directory = (
        runtime.data.dataset_dir
        if isinstance(runtime, PlayerTrainingConfig)
        else runtime.dataset_dir
    )
    assert directory == dataset_root / relative_dir
    assert directory.is_absolute()

    store = PlayerFrameStore(directory)
    assert store.schema_version == "player_detection_frames.v2"
    assert store.metadata["version"] == version
    assert store.read_bgr(0).shape == (HEIGHT, WIDTH, 3)


@pytest.mark.parametrize(
    "version_override", [None, _OVERRIDE_VERSION], ids=["default", "override"]
)
def test_generation_destination_resolves_under_data_root_and_refuses_existing_store(
    version_override: str | None,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    version = version_override or _DEFAULT_VERSION
    overrides = [f"paths.project_root={tmp_path}", f"paths.data_root={dataset_root}"]
    if version_override is not None:
        overrides.append(f"dataset.version={version_override}")
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        config = compose(config_name="generate_dataset", overrides=overrides)

    assert config.run.output_dir == f"player_detection/{version}"
    runtime = GenerateDatasetConfig.from_config(config)
    assert runtime.version == version
    assert runtime.dataset_dir == dataset_root / "player_detection" / version
    with pytest.raises(FileExistsError, match="Dataset version already exists"):
        build_dataset(runtime)
