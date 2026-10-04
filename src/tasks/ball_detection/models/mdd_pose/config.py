from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MDDPoseConfig:
    compression: str
    pose_pooling: str
    readout: str
    frames: int
    channels: tuple[int, int, int]
    dim: int
    heads: int
    layers: int
    dropout: float
    rope_base: float

    @classmethod
    def load(cls, path: Path) -> MDDPoseConfig:
        raw: dict[str, Any] = yaml.safe_load(path.read_text())
        if raw.pop("name", None) != "mdd_pose":
            raise ValueError("Expected the MDD+pose coordinate model configuration")
        if set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("MDD+pose config requires every field and no unknown fields")
        raw["channels"] = tuple(raw["channels"])
        return cls(**raw)

    def __post_init__(self) -> None:
        if self.compression not in {"conv3d", "average", "unshuffle", "haar"}:
            raise ValueError("Unknown MDD compression")
        if self.pose_pooling not in {"deepsets", "attention", "hierarchical", "gnn"}:
            raise ValueError("Unknown pose pooling")
        if self.readout not in {"query", "pose"}:
            raise ValueError("Unknown coordinate readout")
        if self.frames != 32:
            raise ValueError("This review architecture uses 32 real frames")
        if len(self.channels) != 3 or min(self.channels) < 1:
            raise ValueError("Three positive encoder widths are required")
        if self.heads < 1 or self.dim < 4 or self.dim % (2 * self.heads) or self.layers < 1:
            raise ValueError("Attention head dimensions must be positive and even")
        if not 0 <= self.dropout < 1 or not self.rope_base > 1:
            raise ValueError("Invalid dropout or RoPE base")
