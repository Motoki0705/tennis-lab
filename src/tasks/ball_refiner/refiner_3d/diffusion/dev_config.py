"""Complete bounded configuration for one synthetic train/val bring-up."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from .losses import LossConfig
from .model import ModelConfig


@dataclass(frozen=True)
class DevConfig:
    seed: int
    updates: int
    batch_size: int
    frames: int
    stride: int
    learning_rate: float
    weight_decay: float
    evaluate_updates: tuple[int, ...]
    samples: int
    steps: int
    maximum_seconds: int
    allocator_limit_gib: float
    maximum_device_bytes: int
    expected_counts: dict[str, int]
    model: ModelConfig
    loss: LossConfig

    def __post_init__(self) -> None:
        integers = ('seed', 'updates', 'batch_size', 'frames', 'stride', 'samples', 'steps', 'maximum_seconds', 'maximum_device_bytes')
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in integers):
            raise ValueError('Dev dimensions/budgets must be positive integers')
        if self.frames < 4 or not 1 <= self.stride <= self.frames or self.samples < 2:
            raise ValueError('Invalid dev windows/sampling')
        if (not self.evaluate_updates or any(type(n) is not int for n in self.evaluate_updates)
                or tuple(sorted(set(self.evaluate_updates))) != self.evaluate_updates
                or self.evaluate_updates[0] != 0 or self.evaluate_updates[-1] != self.updates
                or self.updates > 20000 or self.maximum_seconds > 5100):
            raise ValueError('Need bounded updates and a final scheduled validation')
        if not 0 < self.allocator_limit_gib <= 6 or not 0 < self.maximum_device_bytes <= 10_000_000_000:
            raise ValueError('Dev run exceeds the GPU grant')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid optimizer settings')
        if set(self.expected_counts) != {'train', 'val', 'test'} or any(type(n) is not int or n < 1 for n in self.expected_counts.values()):
            raise ValueError('Require explicit train/val/test counts')


def load_config(path: Path) -> DevConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {f.name for f in fields(DevConfig)}:
        raise ValueError('Need complete dev configuration without extra fields')
    raw['model'] = ModelConfig(**raw['model'])
    raw['loss'] = LossConfig(**raw['loss'])
    if not isinstance(raw['evaluate_updates'], list):
        raise ValueError('Need explicit validation updates')
    raw['evaluate_updates'] = tuple(raw['evaluate_updates'])
    return DevConfig(**raw)
