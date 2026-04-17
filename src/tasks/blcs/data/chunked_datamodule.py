"""PyTorch Lightning DataModule for chunked BLCS training.

The training set is backed by :class:`ChunkManager` which generates scene
chunks in a background thread.  Validation and test sets are fixed NPZ
datasets loaded from ``data/blcs/``.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.blcs.data.chunk_manager import ChunkManager
from src.tasks.blcs.data.dataset import (
    BallTrajectoryDataset,
    collate_and_adapt_blcs_batch,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig

logger = logging.getLogger(__name__)


class ChunkedBLCSDataModule(pl.LightningDataModule):
    """DataModule that swaps train chunks generated in the background.

    Val/test splits are loaded once from the fixed ``scene_dir``.
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        *,
        generator_config: GeneratorConfig,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.generator_config = generator_config

        data_cfg = self.config.get("data", {})
        self.batch_size = int(data_cfg.get("batch_size", 2))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.scene_dir = Path(data_cfg.get("scene_dir", "data/blcs"))

        # Chunk settings
        chunk_cfg = data_cfg.get("chunk", {})
        self.scenes_per_chunk = int(chunk_cfg.get("scenes_per_chunk", 1000))
        self.epochs_per_chunk = int(chunk_cfg.get("epochs_per_chunk", 3))
        self.prefetch_chunks = int(chunk_cfg.get("prefetch_chunks", 1))
        self.chunks_dir = Path(chunk_cfg.get("chunks_dir", "data/blcs/chunks"))
        self.generation_workers = int(chunk_cfg.get("generation_workers", 0))
        self.generator_device = str(data_cfg.get("generator_device", "cpu"))

        self.input_profile = str(self.config["model"]["io"]["input_profile"])
        self.collate_fn = partial(
            collate_and_adapt_blcs_batch,
            input_profile=self.input_profile,
        )

        self.chunk_manager: ChunkManager | None = None
        self.train_dataset: BallTrajectoryDataset | None = None
        self.val_dataset: BallTrajectoryDataset | None = None
        self.test_dataset: BallTrajectoryDataset | None = None

        # Track chunk usage for epoch-based rotation
        self._current_chunk_id: int | None = None
        self._epochs_on_current_chunk = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:  # noqa: D401
        if stage == "fit" or stage is None:
            # Load initial training chunk synchronously to avoid waiting in the first epoch
            train_split = self.scene_dir / "train.txt"
            if not train_split.exists():
                raise RuntimeError(f"Missing required split file: {train_split}")
            self.train_dataset = BallTrajectoryDataset(
                scene_dir=self.scene_dir,
                split_file="train.txt",
                config=self.config,
                augment=True,
            )
            # Fixed val dataset
            val_split = self.scene_dir / "val.txt"
            if val_split.exists():
                self.val_dataset = BallTrajectoryDataset(
                    scene_dir=self.scene_dir,
                    split_file="val.txt",
                    config=self.config,
                    augment=False,
                )

            # Start background chunk generation
            self.chunk_manager = ChunkManager(
                chunks_dir=self.chunks_dir,
                generator_config=self.generator_config,
                scenes_per_chunk=self.scenes_per_chunk,
                epochs_per_chunk=self.epochs_per_chunk,
                prefetch_chunks=self.prefetch_chunks,
                generator_device=self.generator_device,
                generation_workers=self.generation_workers,
            )
            self.chunk_manager.start()

        if stage == "test" or stage is None:
            test_split = self.scene_dir / "test.txt"
            if not test_split.exists():
                raise RuntimeError(f"Missing required split file: {test_split}")
            self.test_dataset = BallTrajectoryDataset(
                scene_dir=self.scene_dir,
                split_file="test.txt",
                config=self.config,
                augment=False,
            )

    def teardown(self, stage: str | None = None) -> None:
        if self.chunk_manager is not None:
            self.chunk_manager.stop()
            self.chunk_manager = None

    # ------------------------------------------------------------------
    # Chunk rotation
    # ------------------------------------------------------------------

    def on_train_epoch_end(self) -> None:
        """Called by the training loop callback to rotate chunks."""
        self._epochs_on_current_chunk += 1
        if self._epochs_on_current_chunk >= self.epochs_per_chunk:
            self._rotate_chunk()

    def _load_next_chunk(self) -> None:
        """Wait for a ready chunk and replace the training dataset."""
        assert self.chunk_manager is not None
        chunk = self.chunk_manager.wait_for_ready_chunk()
        if chunk is None:
            raise RuntimeError("ChunkManager returned no ready chunk.")
        logger.info(
            "Loading training chunk %d from %s", chunk.chunk_id, chunk.path,
        )
        self.train_dataset = BallTrajectoryDataset(
            scene_dir=chunk.path,
            split_file="train.txt",
            config=self.config,
            augment=True,
        )
        self._current_chunk_id = chunk.chunk_id
        self._epochs_on_current_chunk = 0

    def _rotate_chunk(self) -> None:
        """Mark current chunk as used and switch to the next ready one."""
        assert self.chunk_manager is not None
        old_id = self._current_chunk_id
        if old_id is not None:
            self.chunk_manager.mark_used(old_id)
            logger.info("Chunk %d marked as used.", old_id)
        self._load_next_chunk()

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _build_loader(
        self, dataset: BallTrajectoryDataset, *, train: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=train,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")
        return self._build_loader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader().")
        return self._build_loader(self.val_dataset, train=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader().")
        return self._build_loader(self.test_dataset, train=False)
