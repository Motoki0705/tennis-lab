"""Assemble completed Meiji/broadcast annotations with fixed, disjoint splits."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.dataset_pipeline.assemble import (
    assemble_dataset,
    validate_assembly_config,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.paths import PROJECT_ROOT

register_boundary_validator(
    "tennis_scene.assemble_slcs_dataset", validate_assembly_config
)


@hydra_main(
    config_path="../configs",
    config_name="assemble_slcs_dataset",
    version_base="1.3",
    validation_boundary="tennis_scene.assemble_slcs_dataset",
)
def main(cfg: DictConfig) -> None:
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(dict(cfg.paths), repository_root=PROJECT_ROOT)
    )
    result = assemble_dataset(
        [
            resolver.resolve(PathRole.DATA, str(source))
            for source in cfg.source_datasets
        ],
        resolver.resolve(PathRole.DATA, str(cfg.dataset_directory)),
        dataset_id=str(cfg.dataset_id),
        video_splits=dict(cfg.video_splits),
        seed=int(cfg.seed),
        clip_ids=None if cfg.clip_ids is None else list(cfg.clip_ids),
    )
    output = resolver.resolve(PathRole.OUTPUT, str(cfg.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    print(f"Published {result['num_clips']} clips with fixed recording/venue splits")


if __name__ == "__main__":
    main()
