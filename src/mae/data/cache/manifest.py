"""Cache manifest types for MAE cached-batch pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchEntry:
    batch_id: int
    resolution: int
    batch_size: int
    path: str


@dataclass(frozen=True)
class EpochCacheManifest:
    epoch: int
    split: str
    batches: tuple[BatchEntry, ...]

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EpochCacheManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        batches = tuple(BatchEntry(**b) for b in data["batches"])
        return cls(epoch=int(data["epoch"]), split=str(data["split"]), batches=batches)


if __name__ == "__main__":  # pragma: no cover
    m = EpochCacheManifest(epoch=0, split="train", batches=())
    print(m)

