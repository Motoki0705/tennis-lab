"""Epoch cache callback for MAE cached-batch training.

Creates a double-buffered cache of preprocessed batches:
  - epoch e cache is used for training
  - epoch e+1 cache is produced in the background
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

from src.developing.mae.data.cache.paths import EpochCachePaths
from src.developing.mae.data.catalog import VideoCatalog
from src.developing.mae.data.planning import split_video_paths
from src.developing.mae.data.producer import (
    CacheProducerConfig,
    build_epoch_plan,
    ensure_current_pointer,
    produce_epoch_cache,
)


@dataclass(frozen=True)
class EpochCacheCallbackConfig:
    producer: CacheProducerConfig
    poll_interval_s: float = 2.0


def _produce_worker(
    producer_cfg: CacheProducerConfig,
    split: str,
    epoch: int,
    catalog: VideoCatalog,
) -> None:
    cache_paths = EpochCachePaths(cache_root=Path(producer_cfg.cache_root), split=split)
    plan = build_epoch_plan(cfg=producer_cfg, catalog=catalog, epoch=epoch, split=split)
    produce_epoch_cache(
        cfg=producer_cfg,
        split=split,
        epoch=epoch,
        plan=plan,
        catalog=catalog,
        cache_paths=cache_paths,
    )


class MAEEpochCacheCallback(pl.Callback):
    def __init__(self, cfg: EpochCacheCallbackConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._train_catalog: Optional[VideoCatalog] = None
        self._val_catalog: Optional[VideoCatalog] = None
        self._next_proc = None
        self._next_epoch: Optional[int] = None

    def _wait_for_done(self, cache_paths: EpochCachePaths, epoch: int) -> None:
        done_path = cache_paths.done_path(epoch)
        while not done_path.exists():
            time.sleep(self.cfg.poll_interval_s)

    def _wait_for_val_done(self, cache_paths: EpochCachePaths) -> None:
        while not cache_paths.val_done_path().exists():
            time.sleep(self.cfg.poll_interval_s)

    def _ensure_epoch_cache(self, split: str, epoch: int) -> Path:
        cache_paths = EpochCachePaths(cache_root=Path(self.cfg.producer.cache_root), split=split)
        manifest_path = cache_paths.manifest_path(epoch)
        if manifest_path.exists() and cache_paths.done_path(epoch).exists():
            return manifest_path

        plan = build_epoch_plan(cfg=self.cfg.producer, catalog=self._train_catalog if split == "train" else self._val_catalog, epoch=epoch, split=split)  # type: ignore[arg-type]
        return produce_epoch_cache(
            cfg=self.cfg.producer,
            split=split,
            epoch=epoch,
            plan=plan,
            catalog=self._train_catalog if split == "train" else self._val_catalog,  # type: ignore[arg-type]
            cache_paths=cache_paths,
        )

    def _kick_next(self, trainer: pl.Trainer) -> None:
        max_epochs = int(trainer.max_epochs) if trainer.max_epochs is not None else None
        current = int(trainer.current_epoch)
        if max_epochs is not None and current + 1 >= max_epochs:
            return

        epoch = current + 1
        if self._next_proc is not None and self._next_proc.is_alive():
            return

        cache_paths = EpochCachePaths(cache_root=Path(self.cfg.producer.cache_root), split="train")
        if cache_paths.done_path(epoch).exists():
            return

        ctx = get_context("spawn")
        self._next_proc = ctx.Process(
            target=_produce_worker,
            args=(self.cfg.producer, "train", epoch, self._train_catalog),
            daemon=True,
        )
        self._next_epoch = epoch
        self._next_proc.start()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        cache_root = Path(self.cfg.producer.cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)

        if trainer.is_global_zero:
            full = VideoCatalog.from_video_dir(
                self.cfg.producer.video_dir,
                use_decord=self.cfg.producer.use_decord,
                min_frames=self.cfg.producer.min_frames,
            )
            train_paths, val_paths = split_video_paths(
                full,
                val_split=self.cfg.producer.val_split,
                seed=self.cfg.producer.seed,
            )
            self._train_catalog = full.filter_by_paths(train_paths)
            self._val_catalog = full.filter_by_paths(val_paths)

            if self.cfg.producer.static_val and len(self._val_catalog.videos) > 0:
                val_cache_paths = EpochCachePaths(cache_root=Path(self.cfg.producer.cache_root), split="val")
                if not val_cache_paths.val_done_path().exists():
                    val_plan = build_epoch_plan(
                        cfg=self.cfg.producer,
                        catalog=self._val_catalog,
                        epoch=0,
                        split="val",
                    )
                    produce_epoch_cache(
                        cfg=self.cfg.producer,
                        split="val",
                        epoch=0,
                        plan=val_plan,
                        catalog=self._val_catalog,
                        cache_paths=val_cache_paths,
                    )

            self._ensure_epoch_cache("train", 0)
            train_cache_paths = EpochCachePaths(cache_root=Path(self.cfg.producer.cache_root), split="train")
            ensure_current_pointer(train_cache_paths, train_cache_paths.manifest_path(0))
            self._kick_next(trainer)

        else:
            train_cache_paths = EpochCachePaths(cache_root=Path(self.cfg.producer.cache_root), split="train")
            while not train_cache_paths.current_pointer_path().exists():
                time.sleep(self.cfg.poll_interval_s)

            if self.cfg.producer.static_val:
                val_cache_paths = EpochCachePaths(cache_root=Path(self.cfg.producer.cache_root), split="val")
                if val_cache_paths.val_dir().exists():
                    self._wait_for_val_done(val_cache_paths)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = int(trainer.current_epoch)
        train_cache_paths = EpochCachePaths(cache_root=Path(self.cfg.producer.cache_root), split="train")

        if trainer.is_global_zero:
            if self._next_proc is not None and self._next_epoch == epoch:
                self._next_proc.join()
                self._next_proc = None
                self._next_epoch = None
            else:
                self._wait_for_done(train_cache_paths, epoch)

            ensure_current_pointer(train_cache_paths, train_cache_paths.manifest_path(epoch))
            self._kick_next(trainer)
        else:
            while True:
                try:
                    manifest_path = train_cache_paths.current_pointer_path().read_text(encoding="utf-8").strip()
                except FileNotFoundError:
                    time.sleep(self.cfg.poll_interval_s)
                    continue
                if f"epoch_{epoch:04d}" in manifest_path:
                    break
                time.sleep(self.cfg.poll_interval_s)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._next_proc is not None and self._next_proc.is_alive():
            self._next_proc.terminate()
            self._next_proc.join(timeout=5)
        self._next_proc = None
        self._next_epoch = None


if __name__ == "__main__":  # pragma: no cover
    cfg = EpochCacheCallbackConfig(producer=CacheProducerConfig(samples_per_video=1))
    print(cfg)

