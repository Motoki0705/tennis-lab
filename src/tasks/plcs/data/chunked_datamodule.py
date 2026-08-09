"""PyTorch Lightning DataModule for chunked PLCS training."""

from __future__ import annotations

from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule
from src.tasks.plcs.data.chunk_manager import PLCSChunkManager
from src.tasks.plcs.data.datamodule import PLCSDataModule


class ChunkedPLCSDataModule(BaseChunkedDataModule, PLCSDataModule):
    """PLCS datamodule that rotates background-generated training chunks."""

    def _build_chunk_manager(self) -> PLCSChunkManager:
        return PLCSChunkManager(
            chunks_dir=self.chunks_dir,
            config=self.plcs_runtime.raw,
            scenes_per_chunk=self.scenes_per_chunk,
            epochs_per_chunk=self.epochs_per_chunk,
            prefetch_chunks=self.prefetch_chunks,
            generator_device=self.generator_device,
            generation_workers=self.generation_workers,
        )
