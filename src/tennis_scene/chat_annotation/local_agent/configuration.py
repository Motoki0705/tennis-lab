"""Explicit, validated paths and control settings for a local annotation campaign."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["local_annotation_campaign.v1"] = (
        "local_annotation_campaign.v1"
    )
    annotation_root: Path
    campaign_dir: Path
    project_root: Path
    python_executable: Path
    codex_binary: str = "codex"
    codex_home: Path
    ball_checkpoint: Path | None = None
    ball_checkpoint_sha256: str | None = None

    @field_validator(
        "annotation_root", "campaign_dir", "project_root", "codex_home"
    )
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("campaign paths must be absolute")
        return value.resolve()

    @field_validator('python_executable')
    @classmethod
    def python_launcher(cls, value: Path) -> Path:
        if not value.is_absolute() or not value.is_file() or not os.access(value, os.X_OK):
            raise ValueError('python_executable must name an existing absolute executable')
        return value.parent.resolve() / value.name

    @field_validator("ball_checkpoint")
    @classmethod
    def optional_absolute_path(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("optional paths must be absolute")
        return value.resolve() if value is not None else None

    @model_validator(mode="after")
    def disjoint_outputs(self) -> CampaignConfig:
        protected = [
            self.annotation_root / name
            for name in ("annotated", "videos", "done", "sources", "_preparation")
        ]
        for path in protected:
            if (
                self.campaign_dir == path
                or self.campaign_dir.is_relative_to(path)
                or path.is_relative_to(self.campaign_dir)
            ):
                raise ValueError(
                    "campaign_dir must be separate from annotation data and source videos"
                )
        if (self.ball_checkpoint is None) != (self.ball_checkpoint_sha256 is None):
            raise ValueError("checkpoint path and SHA-256 must be configured together")
        return self

    @property
    def annotated(self) -> Path:
        return self.annotation_root / "annotated"

    @property
    def tasks(self) -> Path:
        return self.campaign_dir / "tasks"

    @property
    def logs(self) -> Path:
        return self.campaign_dir / "logs"

    @property
    def locks(self) -> Path:
        return self.campaign_dir / "cache" / "locks"

    @property
    def state(self) -> Path:
        return self.campaign_dir / "state.json"

    @property
    def control(self) -> Path:
        return self.campaign_dir / "control.json"

    @property
    def events(self) -> Path:
        return self.logs / "events.log"

    def checkpoint(self) -> Path:
        if self.ball_checkpoint is None:
            raise ValueError(
                "candidate generation requires init --ball-checkpoint; visual annotation works without a model"
            )
        if file_sha256(self.ball_checkpoint) != self.ball_checkpoint_sha256:
            raise ValueError("ball checkpoint changed since campaign initialization")
        return self.ball_checkpoint


class AdaptiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    enabled: bool = False
    min: int = Field(default=1, ge=1)
    max: int = Field(default=30, ge=1)
    step: int = Field(default=5, ge=1)
    interval_minutes: float = Field(default=5, gt=0)
    decrease_factor: float = Field(default=0.8, gt=0, lt=1)
    cooldown_minutes: float = Field(default=30, ge=0)
    ceiling_minutes: float = Field(default=120, ge=0)
    min_available_mb: float = Field(default=10000, ge=0)
    max_load_per_core: float = Field(default=1.5, gt=0)
    min_disk_free_gb: float = Field(default=20, ge=0)

    @model_validator(mode="after")
    def ordered_limits(self) -> AdaptiveConfig:
        if self.min > self.max:
            raise ValueError("adaptive minimum exceeds maximum")
        return self


class VariantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    weight: float = Field(default=1, gt=0)
    codex_config: list[str] = Field(default_factory=list)

    @field_validator("codex_config")
    @classmethod
    def permitted_overrides(cls, value: list[str]) -> list[str]:
        for item in value:
            if item.partition("=")[0] != "model_auto_compact_token_limit":
                raise ValueError(
                    "variant overrides may only change model_auto_compact_token_limit"
                )
        return value


class ControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    mode: Literal["pilot", "run", "drain", "stop"] = "run"
    targets: list[Literal["ball"]] = Field(default_factory=lambda: ["ball"])
    max_parallel: int = Field(default=1, ge=1, le=30)
    model: str = Field(default="gpt-6.1-sol", min_length=1)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "max"
    context_stop_fraction: float = Field(default=0.5, gt=0, le=1)
    poll_seconds: float = Field(default=30, gt=0)
    pilot_tasks: list[str] = Field(default_factory=list)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    variants: dict[str, VariantConfig] = Field(
        default_factory=lambda: {"base": VariantConfig()}, min_length=1,
    )
    max_launch_per_tick: int = Field(default=3, ge=1)
    quota_stop_percent: float | None = Field(default=None, gt=0, le=100)
    slow_seconds: float = Field(default=10800, gt=0)
    timeout_seconds: float = Field(default=21600, gt=0)
    termination_grace_seconds: float = Field(default=5, gt=0)

    @field_validator("targets")
    @classmethod
    def ball_target(cls, value: list[Literal["ball"]]) -> list[Literal["ball"]]:
        if value != ["ball"]:
            raise ValueError(
                "this workflow supports exactly one ball annotation target"
            )
        return value


_CURRENT: ContextVar[CampaignConfig] = ContextVar("local_annotation_campaign")


def paths() -> CampaignConfig:
    try:
        return _CURRENT.get()
    except LookupError as error:
        raise RuntimeError(
            "campaign is not configured; use local_agent --campaign PATH"
        ) from error


@contextmanager
def campaign_context(config: CampaignConfig) -> Iterator[None]:
    token = _CURRENT.set(config)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def load_config(campaign_dir: Path) -> CampaignConfig:
    directory = campaign_dir.resolve()
    config: CampaignConfig = CampaignConfig.model_validate_json(
        (directory / "campaign.json").read_text()
    )
    if config.campaign_dir.resolve() != directory:
        raise ValueError("campaign.json belongs to another campaign directory")
    return config


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    from ..runtime.contracts import read_json

    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value
