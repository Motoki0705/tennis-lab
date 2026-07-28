"""Integration tests for composing every synthetic-dataset Hydra config."""

from __future__ import annotations

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.synthetic_data_generation.dataset.registry import get_dataset_pipeline
from src.utils.paths import PROJECT_ROOT


def test_all_dataset_configs_compose_to_valid_pipeline_plans() -> None:
    config_dir = PROJECT_ROOT / "src/synthetic_data_generation/configs/dataset"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        for dataset in ("blcs", "plcs", "court"):
            config = compose(
                config_name="pipeline",
                overrides=[f"domain={dataset}"],
            )
            domain = OmegaConf.to_container(config.domain, resolve=True)
            assert isinstance(domain, dict)
            plan = get_dataset_pipeline(dataset).build_plan(domain)
            assert plan.dataset == dataset
            assert [command.stage for command in plan.commands] == ["runtime_probe"]
