"""PLCS configuration ownership and import-graph regressions."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

import src.tasks.plcs.configuration as runtime_configuration
import src.tasks.plcs.configuration_contracts as configuration_contracts
import src.tasks.plcs.generate_dataset.config as generation_configuration
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.utils.configuration import SemanticConfigurationError
from src.utils.paths import PROJECT_ROOT


def _generation_config() -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="generate_dataset")


def test_generation_config_defaults_to_physical_v1_court_keypoints() -> None:
    config = _generation_config()

    assert config.court_keypoints.selector == "physical_v1"


@pytest.mark.parametrize(
    "modules",
    [
        (
            "src.tasks.plcs.configuration",
            "src.tasks.plcs.configuration_contracts",
            "src.tasks.plcs.generate_dataset.config",
        ),
        (
            "src.tasks.plcs.generate_dataset.config",
            "src.tasks.plcs.configuration_contracts",
            "src.tasks.plcs.configuration",
        ),
        (
            "src.tasks.plcs.configuration_contracts",
            "src.tasks.plcs.configuration",
            "src.tasks.plcs.generate_dataset.config",
        ),
    ],
)
def test_plcs_configuration_modules_import_cold_in_either_order(
    modules: tuple[str, ...],
) -> None:
    imports = "; ".join(f"import {module}" for module in modules)
    result = subprocess.run(
        [sys.executable, "-c", imports],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_plcs_contracts_have_one_canonical_module() -> None:
    assert configuration_contracts.PLCSPathConfig.__module__ == (
        "src.tasks.plcs.configuration_contracts"
    )
    assert configuration_contracts.PLCSGenerationComponents.__module__ == (
        "src.tasks.plcs.configuration_contracts"
    )
    assert not hasattr(runtime_configuration, "PLCSPathConfig")
    assert not hasattr(generation_configuration, "PLCSPathConfig")
    assert not hasattr(generation_configuration, "validate_generation_components")


def test_plcs_configuration_consumers_have_no_reverse_import_edge() -> None:
    runtime_source = (
        PROJECT_ROOT / "src/tasks/plcs/configuration.py"
    ).read_text(encoding="utf-8")
    generation_source = (
        PROJECT_ROOT / "src/tasks/plcs/generate_dataset/config.py"
    ).read_text(encoding="utf-8")
    contracts_source = (
        PROJECT_ROOT / "src/tasks/plcs/configuration_contracts.py"
    ).read_text(encoding="utf-8")

    assert "src.tasks.plcs.generate_dataset.config" not in runtime_source
    assert "src.tasks.plcs.configuration import" not in generation_source
    assert "src.tasks.plcs.configuration" not in contracts_source
    assert "src.tasks.plcs.generate_dataset" not in contracts_source


def test_shared_components_do_not_own_generation_run_validation() -> None:
    config = deepcopy(_generation_config())
    with open_dict(config):
        config.run.train_ratio = 0.5
        config.run.val_ratio = 0.5
        config.run.test_ratio = 0.5

    components = configuration_contracts.PLCSGenerationComponents.from_config(
        config
    )

    assert components.mode == "single_object"
    with pytest.raises(SemanticConfigurationError, match="sum to 1"):
        PLCSGenerationConfig.from_config(config)


def test_generation_boundary_validates_shared_components_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = configuration_contracts.PLCSGenerationComponents.from_config

    def tracked_from_config(
        cls: type[configuration_contracts.PLCSGenerationComponents],
        value: object,
    ) -> configuration_contracts.PLCSGenerationComponents:
        nonlocal calls
        del cls
        calls += 1
        return original(value)

    monkeypatch.setattr(
        configuration_contracts.PLCSGenerationComponents,
        "from_config",
        classmethod(tracked_from_config),
    )

    PLCSGenerationConfig.from_config(_generation_config())

    assert calls == 1


def test_explicit_unavailable_generation_cuda_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_generation_config())
    with open_dict(config):
        config.run.device = "cuda"
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(SemanticConfigurationError, match="not an available device"):
        PLCSGenerationConfig.from_config(config)
