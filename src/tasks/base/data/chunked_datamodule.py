"""Shared chunk-rotation lifecycle for chunked DataModules.

Extracts the background chunk-rotation machinery duplicated between
:class:`ChunkedPLCSDataModule` and :class:`ChunkedBLCSDataModule`: parsing the
``data.chunk`` config block, starting/stopping a chunk manager, and rotating
training chunks across epochs.

Subclasses provide the chunk manager construction (:meth:`_build_chunk_manager`)
and a default chunks directory (:meth:`_default_chunks_dir`).  The training
dataset for each chunk is rebuilt through the inherited
:meth:`SceneDirectoryDataModule._build_dataset` hook.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from src.tasks.base.configuration import (
    ChunkDataConfig,
    as_config_mapping,
    require_config_mapping,
)
from src.tasks.base.data.datamodule import SceneDirectoryDataModule
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)


class BaseChunkedDataModule(SceneDirectoryDataModule):
    """DataModule that rotates background-generated training chunks.

    Validation/test splits are loaded once from the fixed ``scene_dir`` by the
    parent :class:`SceneDirectoryDataModule`.
    """

    def __init__(self, config: object) -> None:
        super().__init__(config)
        root = as_config_mapping(config, path="configuration")
        resolver = PathResolver(
            RuntimePathRoots.from_mapping(
                require_config_mapping(root, "paths", path="configuration"),
                repository_root=PROJECT_ROOT,
            )
        )
        chunk_config = ChunkDataConfig.from_validated_task_mapping(
            require_config_mapping(root, "data", path="configuration"),
            resolver=resolver,
        )

        self.scenes_per_chunk = chunk_config.scenes_per_chunk
        self.epochs_per_chunk = chunk_config.epochs_per_chunk
        self.prefetch_chunks = chunk_config.prefetch_chunks
        self.chunks_dir = chunk_config.chunks_dir
        self.generation_workers = chunk_config.generation_workers
        self.generator_device = chunk_config.generator_device

        self.chunk_manager: Any | None = None
        self._current_chunk_id: int | None = None
        self._epochs_on_current_chunk = 0

    # -- abstract hooks --------------------------------------------------------

    @abstractmethod
    def _build_chunk_manager(self) -> Any:
        """Construct the task-specific (unstarted) chunk manager."""

    # -- shared lifecycle ------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)

        if (stage == "fit" or stage is None) and self.chunk_manager is None:
            self.chunk_manager = self._build_chunk_manager()
            self.chunk_manager.start()
            self._load_next_chunk()

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
            "Loading %s training chunk %d from %s",
            self._dataset_name(),
            chunk.chunk_id,
            chunk.path,
        )
        self.train_dataset = self._build_dataset(
            scene_dir=chunk.path,
            split_file="train.txt",
            augment=True,
            seed=self._dataset_seed(chunk.path, "train.txt"),
        )
        self._current_chunk_id = chunk.chunk_id
        self._epochs_on_current_chunk = 0

    def _rotate_chunk(self) -> None:
        assert self.chunk_manager is not None
        old_id = self._current_chunk_id
        if old_id is not None:
            self.chunk_manager.mark_used(old_id)
            logger.info("%s chunk %d marked as used.", self._dataset_name(), old_id)
        self._load_next_chunk()
