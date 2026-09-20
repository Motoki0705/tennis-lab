"""Typed Hydra boundary for explicit appearance-workflow paths and actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import AfterValidator, Field, StrictBool, StrictInt, model_validator

from src.utils.hydra import register_boundary_validator

from .contracts import Record, VariantConfig

APPEARANCE_BOUNDARY = "synthetic.appearance_variant"


def require_absolute_path(path: Path) -> Path:
    """Accept explicitly selected absolute paths, preserving executable symlinks."""
    if not path.is_absolute() or "\x00" in str(path) or ".." in path.parts:
        raise ValueError(
            "Appearance paths must be explicit absolute paths without '..'"
        )
    return path


AbsolutePath = Annotated[Path, AfterValidator(require_absolute_path)]


class ResultConfig(Record):
    request_id: str | None
    path: AbsolutePath | None
    accepted: StrictBool | None
    notes: str


class ComparisonConfig(Record):
    output_root: AbsolutePath
    reference: AbsolutePath
    target_index: StrictInt = Field(ge=0)


class DeriveConfig(Record):
    parent_root: AbsolutePath


class ReportConfig(Record):
    roots: list[AbsolutePath] = Field(min_length=1)
    output_root: AbsolutePath


class BatchConfig(Record):
    concurrency: StrictInt = Field(ge=1, le=4)
    start_interval_seconds: float = Field(ge=0, allow_inf_nan=False)
    indices: list[Annotated[StrictInt, Field(ge=0)]] | None


class AppearanceRuntimeConfig(Record):
    action: Literal[
        "prepare",
        "derive",
        "report",
        "compare_models",
        "configure_api_key",
        "check_api",
        "generate",
        "generate_batch",
        "next_request",
        "record_result",
        "finalize",
        "train",
        "execute_training",
    ]
    api_retry: StrictBool
    training_retry: StrictBool
    appearance_data_root: AbsolutePath
    variant: VariantConfig
    result: ResultConfig
    comparison: ComparisonConfig
    derive: DeriveConfig
    report: ReportConfig
    batch: BatchConfig

    @classmethod
    def from_config(cls, config: DictConfig) -> AppearanceRuntimeConfig:
        runtime: AppearanceRuntimeConfig = cls.model_validate_json(
            json.dumps(OmegaConf.to_container(config, resolve=True)), strict=True
        )
        return runtime

    @model_validator(mode="after")
    def validate_action(self) -> AppearanceRuntimeConfig:
        # Persisted records retain backwards-compatible defaults; the live CLI
        # instead requires every variant/API field from composed configuration.
        records: list[Record] = [self.variant]
        if self.variant.api is not None:
            records.append(self.variant.api)
        for record in records:
            missing = set(type(record).model_fields) - record.model_fields_set
            if missing:
                raise ValueError(
                    f"Composed configuration is missing fields: {sorted(missing)}"
                )
        for path in (
            self.variant.source_workspace,
            self.variant.output_root,
            self.variant.nht_source_root,
            self.variant.training_python,
        ):
            require_absolute_path(path)
        if self.variant.api is not None:
            require_absolute_path(self.variant.api.api_key_file)
        if self.action == "configure_api_key" and self.variant.api is None:
            raise ValueError("configure_api_key requires variant.api settings")
        if self.action == "record_result" and (
            not self.result.request_id
            or self.result.path is None
            or self.result.accepted is None
            or not self.result.notes.strip()
        ):
            raise ValueError(
                "record_result requires request_id, path, explicit accepted and review notes"
            )
        if self.batch.indices is not None and len(set(self.batch.indices)) != len(
            self.batch.indices
        ):
            raise ValueError("batch.indices must not contain duplicate frame indices")
        if len(set(self.report.roots)) != len(self.report.roots):
            raise ValueError("report.roots must not contain duplicates")
        return self


def validate_appearance_boundary(config: DictConfig) -> None:
    AppearanceRuntimeConfig.from_config(config)


register_boundary_validator(APPEARANCE_BOUNDARY, validate_appearance_boundary)
