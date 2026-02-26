"""Filesystem layout for cached-batch MAE data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpochCachePaths:
    cache_root: Path
    split: str

    def split_dir(self) -> Path:
        return self.cache_root / self.split

    def epoch_dir(self, epoch: int) -> Path:
        return self.split_dir() / f"epoch_{epoch:04d}"

    def epoch_tmp_dir(self, epoch: int) -> Path:
        return self.split_dir() / f"epoch_{epoch:04d}.tmp"

    def current_pointer_path(self) -> Path:
        return self.split_dir() / "CURRENT"

    def manifest_path(self, epoch: int) -> Path:
        return self.epoch_dir(epoch) / "manifest.json"

    def done_path(self, epoch: int) -> Path:
        return self.epoch_dir(epoch) / "DONE"

    def val_dir(self) -> Path:
        return self.split_dir() / "val_static"

    def val_manifest_path(self) -> Path:
        return self.val_dir() / "manifest.json"

    def val_done_path(self) -> Path:
        return self.val_dir() / "DONE"


if __name__ == "__main__":  # pragma: no cover
    p = EpochCachePaths(cache_root=Path("data/mae/cache"), split="train")
    print(p.epoch_dir(3))

