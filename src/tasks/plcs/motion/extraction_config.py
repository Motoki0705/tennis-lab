"""Hydra-owned configuration for reusable GVHMR motion extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.submodules.configuration import GvhmrDemoConfig
from src.tasks.plcs.configuration_contracts import PLCSPathConfig
from src.tasks.plcs.motion.gvhmr_extraction import (
    GvhmrSelectionConfig,
    load_model_runtime,
)
from src.utils.configuration import ConfigField, PathRole, StrictConfigSchema
from src.utils.hydra import register_boundary_validator

EXTRACTION_DATASET_SCHEMA = StrictConfigSchema(
    name="plcs.extraction.dataset",
    fields={key: ConfigField.of(str) for key in ("name", "root", "selection_config")},
)
EXTRACTION_MODELS_SCHEMA = StrictConfigSchema(
    name="plcs.extraction.models",
    fields={
        "config": ConfigField.of(str),
        "checkpoint_root": ConfigField.of(str),
        "dino_checkpoint": ConfigField.of(str),
        "runtime_overrides": ConfigField.of(dict),
    },
)
EXTRACTION_RUN_SCHEMA = StrictConfigSchema(
    name="plcs.extraction.run",
    fields={
        "output_dir": ConfigField.of(str),
        "seed": ConfigField.of(int),
        "deterministic": ConfigField.of(bool),
        "clip_ids": ConfigField.of(list, type(None)),
        "camera_ids": ConfigField.of(list, type(None)),
        "max_clips": ConfigField.of(int, type(None)),
        "max_frames": ConfigField.of(int, type(None)),
        "overwrite": ConfigField.of(bool),
        "write_preview": ConfigField.of(bool),
    },
)
EXTRACTION_SCHEMA = StrictConfigSchema(
    name="plcs.extraction",
    fields={
        "paths": ConfigField.of(dict),
        "dataset": ConfigField.mapping(EXTRACTION_DATASET_SCHEMA),
        "models": ConfigField.mapping(EXTRACTION_MODELS_SCHEMA),
        "run": ConfigField.mapping(EXTRACTION_RUN_SCHEMA),
    },
)


@dataclass(frozen=True)
class ExtractionConfig:
    """Validated inputs; selected settings never come from CLI parser defaults."""

    resolved: dict[str, Any]
    dataset_root: Path
    output_root: Path
    selection: GvhmrSelectionConfig
    model_runtime: GvhmrDemoConfig

    @classmethod
    def from_config(cls, value: DictConfig) -> ExtractionConfig:
        raw = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
        if not isinstance(raw, dict):
            raise TypeError("Extraction configuration must be a mapping.")
        validated = EXTRACTION_SCHEMA.validate(raw)
        dataset = cast(dict[str, Any], validated["dataset"])
        models = cast(dict[str, Any], validated["models"])
        run = cast(dict[str, Any], validated["run"])
        if run["seed"] < 0 or run["seed"] >= 2**32:
            raise ValueError("run.seed must be within [0, 2**32).")
        for name in ("max_clips", "max_frames"):
            if run[name] is not None and run[name] <= 0:
                raise ValueError(f"run.{name} must be positive or null.")
        for name in ("clip_ids", "camera_ids"):
            items = run[name]
            if items is not None and (
                not items
                or any(type(item) is not str or not item.strip() for item in items)
                or len(items) != len(set(items))
            ):
                raise ValueError(f"run.{name} must be unique nonempty strings or null.")
        resolver = PLCSPathConfig.from_config(value).resolver
        dataset_root = resolver.resolve(PathRole.DATA, dataset["root"])
        output_root = resolver.resolve(PathRole.DATA, run["output_dir"])
        if not dataset_root.is_dir():
            raise FileNotFoundError(dataset_root)
        if output_root == dataset_root or dataset_root in output_root.parents:
            raise ValueError("Extraction output must be outside the input dataset.")
        selection = GvhmrSelectionConfig.load(
            resolver.resolve(PathRole.PROJECT, dataset["selection_config"])
        )
        if dataset["name"] != selection.dataset_id:
            raise ValueError("dataset.name and selection dataset_id disagree.")
        selection.select_cameras(
            None if run["camera_ids"] is None else tuple(run["camera_ids"])
        )
        model_runtime = load_model_runtime(
            resolver.resolve(PathRole.PROJECT, models["config"]),
            repository_root=resolver.roots.project_root,
            checkpoint_root=resolver.resolve(
                PathRole.EXTERNAL_ASSET, models["checkpoint_root"]
            ),
            dino_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, models["dino_checkpoint"]
            ),
            runtime_overrides=models["runtime_overrides"],
            asset_roots=resolver.roots,
        )
        raw["paths"] = {key: str(path) for key, path in asdict(resolver.roots).items()}
        return cls(
            cast(dict[str, Any], raw),
            dataset_root,
            output_root,
            selection,
            model_runtime,
        )


def validate_extraction_boundary(config: DictConfig) -> None:
    ExtractionConfig.from_config(config)


register_boundary_validator("plcs.extract_gvhmr_motions", validate_extraction_boundary)
