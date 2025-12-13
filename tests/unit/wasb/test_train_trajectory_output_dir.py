from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.wasb.utils.config import load_config, resolve_model_name


def test_resolve_model_name_from_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "src" / "wasb" / "configs" / "trajectory.yaml"
    config = load_config(config_path)

    assert resolve_model_name(config, config_path) == str(config.model.name)


def test_resolve_model_name_from_wasb_default_config() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "src" / "wasb" / "configs" / "train.yaml"
    config = load_config(config_path)

    assert resolve_model_name(config, config_path) == str(config.model.name)


def test_resolve_model_name_fallback_to_config_stem() -> None:
    config = OmegaConf.create({})

    assert resolve_model_name(config, "outputs/my_config.yaml") == "my_config"


def test_resolve_model_name_empty_name_falls_back_to_config_stem() -> None:
    config = OmegaConf.create({"model": {"name": "  "}})

    assert resolve_model_name(config, "some/path/trajectory.yaml") == "trajectory"
