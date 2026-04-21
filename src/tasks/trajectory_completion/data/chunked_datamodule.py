"""PyTorch Lightning DataModule for chunked trajectory completion training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.tasks.blcs.data.chunk_manager import ChunkManager
from src.tasks.trajectory_completion.data.datamodule import (
    TrajectoryCompletionDataModule,
)
from src.tasks.trajectory_completion.data.dataset import (
    BLCSUVTrajectoryCompletionDataset,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig

logger = logging.getLogger(__name__)


class ChunkedTrajectoryCompletionDataModule(TrajectoryCompletionDataModule):
    """Trajectory-completion datamodule that rotates BLCS-generated chunks."""

    def __init__(self, config: DictConfig, *, generator_config: GeneratorConfig) -> None:
        super().__init__(config)

        data_cfg = self.config.get("data", {}) or {}
        chunk_cfg = data_cfg.get("chunk", {}) or {}

        self.generator_config = generator_config
        self.scenes_per_chunk = int(chunk_cfg.get("scenes_per_chunk", 1000))
        self.epochs_per_chunk = int(chunk_cfg.get("epochs_per_chunk", 3))
        self.prefetch_chunks = int(chunk_cfg.get("prefetch_chunks", 1))
        self.chunks_dir = Path(chunk_cfg.get("chunks_dir", "data/blcs/chunks"))
        self.generation_workers = int(chunk_cfg.get("generation_workers", 0))
        self.generator_device = str(data_cfg.get("generator_device", "cpu"))

        self.chunk_manager: ChunkManager | None = None
        self._current_chunk_id: int | None = None
        self._epochs_on_current_chunk = 0

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)

        if (stage == "fit" or stage is None) and self.chunk_manager is None:
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
            "Loading trajectory completion training chunk %d from %s",
            chunk.chunk_id,
            chunk.path,
        )
        self.train_dataset = BLCSUVTrajectoryCompletionDataset(
            scene_dir=chunk.path,
            split_file=self.train_file,
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
            logger.info("Trajectory completion chunk %d marked as used.", old_id)
        self._load_next_chunk()