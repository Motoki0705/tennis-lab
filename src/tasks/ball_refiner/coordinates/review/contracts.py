"""Strict, serializable review requests; no model features are added here."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.tasks.ball_refiner.coordinates.config import CorruptionConfig, parse_section

SCHEMA = "ball_refiner.coordinate_review.v1"


class Augmentation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    event_probability: float = Field(ge=0, le=1)
    isolated_probability: float = Field(ge=0, le=1)
    gap_min: int = Field(ge=1, le=120)
    gap_max: int = Field(ge=2, le=120)
    noise_p95_px: float = Field(ge=0, le=2000)
    jitter_sigma_px: float = Field(ge=0, le=100)
    outlier_probability: float = Field(ge=0, lt=1)
    triangulation_steps: int = Field(ge=0, le=10)

    def config(self) -> CorruptionConfig:
        return parse_section(CorruptionConfig, self.model_dump())


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rally: str = Field(min_length=1, max_length=100)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_2d: str | None = None
    checkpoint_3d: str | None = None
    augmentation: Augmentation
    augmentation_seed: int = Field(ge=0, le=2**32 - 1)
    flow_seed: int = Field(ge=0, le=2**32 - 1)
    missing_enabled: bool = True
    noise_enabled: bool = True
    device: Literal["cpu", "cuda"] = "cpu"
    checkpoint_hashes: dict[str, str] = Field(default_factory=dict)

    def corruption(self) -> CorruptionConfig:
        config = self.augmentation.config()
        return config if self.missing_enabled else replace(config, event_probability=0.0, isolated_probability=0.0)

    def profile(self) -> dict[str, Any]:
        return {"augmentation": asdict(self.augmentation.config()), "augmentation_seed": self.augmentation_seed,
                "flow_seed": self.flow_seed, "missing_enabled": self.missing_enabled, "noise_enabled": self.noise_enabled}


def evaluation_profile(config: dict[str, Any]) -> dict[str, Any]:
    corruption = parse_section(CorruptionConfig, config["corruption"])
    corruption = replace(corruption, event_probability=float(config["data"]["evaluation_event_probability"]))
    seed = int(config["data"]["evaluation_seed"])
    return {"augmentation": asdict(corruption), "augmentation_seed": seed, "flow_seed": seed,
            "missing_enabled": True, "noise_enabled": True}
