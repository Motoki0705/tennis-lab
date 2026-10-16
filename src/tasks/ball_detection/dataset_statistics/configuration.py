"""The shipped YAML is the sole source of initial statistics settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class StatisticsConfig:
    strides: tuple[int, ...]
    scope_stride: int
    grid_x: int
    grid_y: int
    pose_min_score: float
    pose_jump_speeds: tuple[float, ...]
    pose_jump_primary: float
    pose_spike_threshold: float
    pose_bone_change_threshold: float

    @classmethod
    def shipped(cls) -> StatisticsConfig:
        text = files('src.tasks.ball_detection').joinpath('configs/dataset_statistics.yaml').read_text()
        return cls.from_mapping(yaml.safe_load(text))

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> StatisticsConfig:
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError('Statistics settings require exactly the documented fields')
        values = dict(raw)
        if any(not isinstance(values[key], (list, tuple)) for key in ('strides', 'pose_jump_speeds')):
            raise ValueError('Strides and pose thresholds must be sequences')
        values['strides'] = tuple(values['strides'])
        values['pose_jump_speeds'] = tuple(values['pose_jump_speeds'])
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if not self.strides or len(set(self.strides)) != len(self.strides) or any(type(s) is not int or not 1 <= s <= 32 for s in self.strides):
            raise ValueError('Distinct integer strides must be between 1 and 32')
        if type(self.scope_stride) is not int or self.scope_stride not in self.strides:
            raise ValueError('Scope stride must be one of the compared strides')
        if any(type(n) is not int or not 1 <= n <= 16 for n in (self.grid_x, self.grid_y)):
            raise ValueError('Spatial grid axes must be integers between 1 and 16')
        if not self.pose_jump_speeds or len(self.pose_jump_speeds) > 10:
            raise ValueError('Provide 1–10 pose speed thresholds')
        positive = (*self.pose_jump_speeds, self.pose_jump_primary, self.pose_spike_threshold, self.pose_bone_change_threshold)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError('Pose thresholds must be finite and positive')
        if self.pose_jump_primary not in self.pose_jump_speeds or len(set(self.pose_jump_speeds)) != len(self.pose_jump_speeds):
            raise ValueError('Distinct speed thresholds must contain the primary threshold')
        if isinstance(self.pose_min_score, bool) or not isinstance(self.pose_min_score, (int, float)) or not np.isfinite(self.pose_min_score):
            raise ValueError('Pose score threshold must be finite (scores are not probabilities)')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
