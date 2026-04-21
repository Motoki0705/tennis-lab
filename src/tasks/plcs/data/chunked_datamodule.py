"""PyTorch Lightning DataModule for chunked PLCS training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.tasks.plcs.data.chunk_manager import ChunkManager
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.data.dataset import SceneDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ChunkedPLCSDataModule(PLCSDataModule):
    """PLCS datamodule that rotates background-generated training chunks."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        data_cfg = self.config.get("data", {}) or {}
        chunk_cfg = data_cfg.get("chunk", {}) or {}
        run_cfg = self.config.get("run", {}) or {}

        self.scenes_per_chunk = int(chunk_cfg.get("scenes_per_chunk", 1000))
        self.epochs_per_chunk = int(chunk_cfg.get("epochs_per_chunk", 3))
        self.prefetch_chunks = int(chunk_cfg.get("prefetch_chunks", 1))
        self.chunks_dir = Path(chunk_cfg.get("chunks_dir", "data/plcs/chunks"))
        self.generation_workers = int(chunk_cfg.get("generation_workers", 0))
        self.max_attempt_factor = int(chunk_cfg.get("max_attempt_factor", 10))
        self.generator_device = str(data_cfg.get("generator_device", run_cfg.get("device", "auto")))
        category = run_cfg.get("category")
        self.generator_category = None if category is None else str(category)

        self.chunk_manager: ChunkManager | None = None
        self._current_chunk_id: int | None = None
        self._epochs_on_current_chunk = 0

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)

        if (stage == "fit" or stage is None) and self.chunk_manager is None:
            self.chunk_manager = ChunkManager(
                chunks_dir=self.chunks_dir,
                config=self.config,
                scenes_per_chunk=self.scenes_per_chunk,
                epochs_per_chunk=self.epochs_per_chunk,
                prefetch_chunks=self.prefetch_chunks,
                generator_device=self.generator_device,
                generation_workers=self.generation_workers,
                category=self.generator_category,
                max_attempt_factor=self.max_attempt_factor,
            )
            self.chunk_manager.start()

    def teardown(self, stage: str | None = None) -> None:
        if self.chunk_manager is not None:
            self.chunk_manager.stop()
            self.chunk_manager = None

    def on_train_epoch_end(self) -> None:
        self._epochs_on_current_chunk += 1
        if self._epochs_on_current_chunk >= self.epochs_per_chunk:
            self._rotate_chunk()

    def _load_next_chunk(self) -> None:
        assert self.chunk_manager is not None
        chunk = self.chunk_manager.wait_for_ready_chunk()
        if chunk is None:
            raise RuntimeError("ChunkManager returned no ready chunk.")
        logger.info(
            "Loading PLCS training chunk %d from %s",
            chunk.chunk_id,
            chunk.path,
        )
        self.train_dataset = SceneDataset(
            scene_dir=chunk.path,
            split_file="train.txt",
            config=self.config,
            augment=True,
        )
        self._current_chunk_id = chunk.chunk_id
        self._epochs_on_current_chunk = 0

    def _rotate_chunk(self) -> None:
        assert self.chunk_manager is not None
        old_id = self._current_chunk_id
        if old_id is not None:
            self.chunk_manager.mark_used(old_id)
            logger.info("PLCS chunk %d marked as used.", old_id)
        self._load_next_chunk()