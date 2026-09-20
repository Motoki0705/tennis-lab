"""Strict persisted contracts for appearance experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class OpenAIImageConfig(Record):
    model: Literal[
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-sunburst-2026-09-08",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-flare-2026-09-08",
    ] = "gpt-image-2.5-sunburst-2026-09-08"
    quality: Literal["low", "medium", "high", "xhigh", "max", "auto"] = "high"
    api_key_file: Path = Field(
        default_factory=lambda: Path.home() / ".config/tennis-lab/openai.env"
    )
    timeout_seconds: float = Field(default=600.0, gt=0, le=3600)


class VariantConfig(Record):
    source_workspace: Path
    output_root: Path
    scene_id: str = "B00-clay-v001"
    first_index: int = Field(default=0, ge=0)
    last_index: int = Field(default=248, ge=0)
    sample_count: int = Field(default=50, ge=2)
    frame_indices: list[int] | None = None
    reference_index: int = Field(default=124, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    max_steps: int = Field(default=7000, gt=3000)
    nht_executable: str = "nht-reconstruct"
    nht_source_root: Path
    training_python: Path
    generation_size: tuple[int, int] | None = None
    generation_provider: Literal["builtin_imagegen", "openai_api"] = "builtin_imagegen"
    api: OpenAIImageConfig | None = None

    @model_validator(mode="after")
    def validate_paths_and_range(self) -> VariantConfig:
        for name in (
            "source_workspace",
            "output_root",
            "nht_source_root",
            "training_python",
        ):
            value = getattr(self, name)
            if not value.is_absolute():
                raise ValueError(f"{name} must be absolute")
        if self.generation_size is not None and min(self.generation_size) <= 0:
            raise ValueError("generation_size must contain positive dimensions")
        if self.generation_provider == "openai_api":
            if self.api is None or self.generation_size is None:
                raise ValueError(
                    "API generation requires explicit api settings and generation_size"
                )
            width, height = self.generation_size
            if width % 16 or height % 16 or max(width, height) / min(width, height) > 3:
                raise ValueError(
                    "API dimensions must be multiples of 16 with aspect ratio at most 3:1"
                )
            if max(width, height) > 3840 or not 655360 <= width * height <= 8294400:
                raise ValueError("API dimensions exceed supported pixel/edge limits")
            if not self.api.api_key_file.is_absolute():
                raise ValueError("api.api_key_file must be absolute")
        source, output = self.source_workspace.resolve(), self.output_root.resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Variant and source workspaces must be separate")
        if self.last_index - self.first_index + 1 < self.sample_count:
            raise ValueError(
                "Sampling interval must contain sample_count distinct frames"
            )
        if self.frame_indices is not None and (
            len(self.frame_indices) != self.sample_count
            or self.frame_indices != sorted(set(self.frame_indices))
            or self.frame_indices[0] != self.first_index
            or self.frame_indices[-1] != self.last_index
        ):
            raise ValueError(
                "frame_indices must be sorted, unique, match sample_count and include both endpoints"
            )
        if not self.scene_id or Path(self.scene_id).name != self.scene_id:
            raise ValueError("scene_id must be a single nonempty path component")
        return self


class FrameRecord(Record):
    name: str
    source_index: int
    original_sorted_index: int
    split: Literal["train", "validation"]
    source_sha256: str
    accepted_attempt: str | None = None
    attempts: list[str] = Field(default_factory=list)


class GenerationRequest(Record):
    workflow_revision: int = 1
    request_id: str
    target: str
    attempt: int
    prompt: str
    prompt_sha256: str
    referenced_image_paths: list[str]
    input_sha256: list[str]
    input_geometry: dict[str, Any] | None = None
    provider: Literal["builtin_imagegen", "openai_api"] = "builtin_imagegen"
    api_parameters: dict[str, Any] | None = None

    def tool_arguments(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "referenced_image_paths": self.referenced_image_paths,
        }


class AttemptRecord(Record):
    request: GenerationRequest
    raw_path: str
    raw_sha256: str
    raw_size: tuple[int, int]
    accepted: bool
    review_notes: str
    normalized_path: str | None = None
    normalized_sha256: str | None = None
    normalized_size: tuple[int, int] | None = None
    processing: str | None = None
    tool_metadata: dict[str, Any] = Field(default_factory=dict)


class Manifest(Record):
    workflow_revision: int = 1
    provider_session: str | None = None
    schema_version: Literal["appearance_variant_v1"] = "appearance_variant_v1"
    config: VariantConfig
    config_sha256: str
    source_files: dict[str, str]
    prompt_sha256: dict[str, str]
    image_size: tuple[int, int]
    frames: list[FrameRecord]
    reference_source_sha256: str
    reference_attempts: list[str] = Field(default_factory=list)
    reference_accepted_attempt: str | None = None
    reference_import: dict[str, Any] | None = None
    pending: GenerationRequest | None = None
    status: Literal[
        "generating", "ready", "finalized", "training", "complete", "failed"
    ] = "generating"
