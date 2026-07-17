from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tasks.base.data.scene_chunk_manager import SceneChunkManager


class _Writer:
    def __init__(self, output_dir: Path) -> None:
        self.scenes_dir = output_dir / "scenes"
        self.scenes_dir.mkdir()

    def save_scene(self, scene: dict[str, Any]) -> None:
        (self.scenes_dir / scene["scene_id"]).mkdir()


def test_scene_chunk_manager_materializes_canonical_scene_directories(tmp_path) -> None:
    manager = SceneChunkManager(
        chunks_dir=tmp_path,
        writer_factory=_Writer,
        scene_factory=lambda scene_id: {"scene_id": scene_id},
        scenes_per_chunk=2,
        epochs_per_chunk=1,
        prefetch_chunks=0,
    )
    manager.start()
    try:
        chunk = manager.wait_for_ready_chunk(timeout=5.0)
        assert chunk is not None
        names = (chunk.path / "train.txt").read_text().splitlines()
        assert names == ["scene_train_000000000", "scene_train_000000001"]
    finally:
        manager.stop()
