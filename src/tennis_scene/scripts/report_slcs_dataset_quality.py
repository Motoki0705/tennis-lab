"""Audit generated SLCS teachers and feature provenance on CPU."""

from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.data.quality import QualityConfig
from src.tennis_scene.dataset_pipeline.quality_report import (
    validate_quality_report_config,
    write_quality_report,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.paths import PROJECT_ROOT

register_boundary_validator(
    "tennis_scene.report_slcs_dataset_quality", validate_quality_report_config
)


@hydra_main(
    config_path="../configs",
    config_name="report_slcs_dataset_quality",
    version_base="1.3",
    validation_boundary="tennis_scene.report_slcs_dataset_quality",
)
def main(cfg: DictConfig) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(dict(cfg.paths), repository_root=PROJECT_ROOT)
    )
    output = resolver.resolve(PathRole.OUTPUT, cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    report = write_quality_report(
        resolver.resolve(PathRole.DATA, cfg.dataset_directory),
        resolver.resolve(PathRole.DATA, cfg.expected_dataset_directory),
        [
            resolver.resolve(PathRole.OUTPUT, path)
            for path in cfg.generation_directories
        ],
        [
            resolver.resolve(PathRole.OUTPUT, path)
            for path in cfg.observation_directories
        ],
        output,
        quality=QualityConfig(**dict(cfg.quality)),
        excluded_clips=dict(cfg.excluded_clips),
        allow_incomplete=cfg.allow_incomplete,
    )
    print(f"Dataset audit: {report['status']} {report['counts']}")


if __name__ == "__main__":
    main()
