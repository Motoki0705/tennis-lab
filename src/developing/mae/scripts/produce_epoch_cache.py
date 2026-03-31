"""Generate cached MAE batches for a single epoch using Hydra-managed configuration.

Usage:
    python -m src.developing.mae.scripts.produce_epoch_cache
    python -m src.developing.mae.scripts.produce_epoch_cache task.epoch=5
    python -m src.developing.mae.scripts.produce_epoch_cache task.split=val

Notes:
    - Configuration is loaded from `src/developing/mae/configs/produce_epoch_cache.yaml`.
    - Hydra handles runtime overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.developing.mae.data.cache.paths import EpochCachePaths
from src.developing.mae.data.catalog import VideoCatalog
from src.developing.mae.data.planning import split_video_paths
from src.developing.mae.data.producer import (
    CacheProducerConfig,
    PreprocessConfig,
    build_epoch_plan,
    produce_epoch_cache,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="produce_epoch_cache", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    pp_cfg = PreprocessConfig(**OmegaConf.to_container(cfg.producer.preprocess, resolve=True))
    producer_cfg = CacheProducerConfig(
        **{
            **OmegaConf.to_container(cfg.producer, resolve=True),
            "preprocess": pp_cfg,
        }
    )
    split = str(cfg.task.split)
    epoch = int(cfg.task.epoch)

    full = VideoCatalog.from_video_dir(
        producer_cfg.video_dir,
        use_decord=producer_cfg.use_decord,
        min_frames=producer_cfg.min_frames,
    )
    train_paths, val_paths = split_video_paths(
        full,
        val_split=producer_cfg.val_split,
        seed=producer_cfg.seed,
    )
    catalog = full.filter_by_paths(train_paths if split == "train" else val_paths)

    cache_paths = EpochCachePaths(cache_root=Path(producer_cfg.cache_root), split=split)
    plan = build_epoch_plan(cfg=producer_cfg, catalog=catalog, epoch=epoch, split=split)
    manifest = produce_epoch_cache(
        cfg=producer_cfg,
        split=split,
        epoch=epoch,
        plan=plan,
        catalog=catalog,
        cache_paths=cache_paths,
    )
    log.info("Wrote manifest: %s", manifest)


if __name__ == "__main__":
    main()
