"""Explicitly convert the selected historical PLCS Court14/raw weights.

This is an inference/weight-initialization conversion, not a training-resume
conversion. Historical v1, BLCS, balanced-loss and asinh experiments remain
reproducible at their recorded commits rather than adding runtime branches.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.model_io.residual_checkpoint import (
    checkpoint_contract,
    validate_residual_checkpoint,
)
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import validate_court_coordinate_normalization

PATH_BOUNDARY = NonHydraPathBoundary(
    name="plcs.residual_checkpoint_migration",
    fields=(
        BoundaryPathField(
            "source",
            PathRole.CHECKPOINT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "destination", PathRole.ARTIFACT, PathDirection.OUTPUT, PathKind.FILE
        ),
    ),
)


def convert_checkpoint(source: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a copy, keeping numerical weights and source provenance intact."""
    validate_court_coordinate_normalization(
        source, artifact="historical PLCS residual checkpoint"
    )
    marker = source.get("geometric_residual_contract")
    if (
        not isinstance(marker, dict)
        or type(marker.get("schema_version")) is not int
        or marker["schema_version"] not in (1, 2)
    ):
        raise ValueError(
            "Only historical schema 1/2 PLCS Court14/raw checkpoints can be converted"
        )
    parameters = source.get("hyper_parameters")
    if not isinstance(parameters, Mapping) or "config" not in parameters:
        raise ValueError("Checkpoint requires its original config")
    embedded = parameters["config"]
    if isinstance(embedded, DictConfig):
        embedded = OmegaConf.to_container(embedded, resolve=True)
    old = deepcopy(embedded)
    if (
        not isinstance(old, dict)
        or old.get("task") != "plcs"
        or old.get("model", {}).get("name") != "plcs_triangulation_residual_v2"
    ):
        raise ValueError("Only the PLCS Court14 training recipe is supported")
    source_config = deepcopy(old)
    expected_keys = {
        "task",
        "model",
        "data",
        "initializer",
        "corruption",
        "loss",
        "training",
        "run",
        "paths",
        "court_keypoints",
        "v2",
    }
    if "features" in old:
        expected_keys.add("features")
    if set(old) != expected_keys:
        raise ValueError("Unknown historical configuration keys")
    features = old.pop("features", {"residual_encoding": "raw", "residual_scale": 1.0})
    if features != {"residual_encoding": "raw", "residual_scale": 1.0}:
        raise ValueError(
            "Only raw-input checkpoints preserve the selected PLCS semantics"
        )
    if marker["schema_version"] == 2 and (
        "features" not in source_config or "ffn_type" not in old["model"]
    ):
        raise ValueError("Schema 2 requires explicit feature/FFN config")
    old["model"].setdefault("ffn_type", "swiglu")
    if marker["schema_version"] == 1 and old["model"]["ffn_type"] != "swiglu":
        raise ValueError("Schema 1 requires SwiGLU")
    old.pop("task")
    old["model"]["name"] = "plcs_triangulation_residual"
    augmentation = old.pop("corruption")
    for name in (
        "camera_rotation_std_deg",
        "camera_center_std_m",
        "camera_focal_log_std",
        "camera_principal_std_px",
    ):
        if augmentation.pop(name) != 0:
            raise ValueError(
                "Court14 calibration cannot use independent camera perturbations"
            )
    extra = old.pop("v2")
    if extra.pop("loss_mode") != "legacy":
        raise ValueError(
            "Only the selected componentwise Smooth L1 objective is supported"
        )
    for name in ("regret_weight", "regret_tolerance_m", "balanced_world_weight"):
        extra.pop(name)
    if augmentation.keys() & extra.keys():
        raise ValueError("Ambiguous historical augmentation keys")
    old["augmentation"] = {**augmentation, **extra}
    # The artifact is for inference or explicit initialization, never silent resume.
    old["run"]["resume"] = None
    old["run"]["init_weights"] = None
    config = OmegaConf.create(old)
    parsed = validate_residual_config(config)
    expected_old = checkpoint_contract(parsed)
    expected_old["family"] = "plcs_triangulation_residual_v2"
    expected_old["schema_version"] = marker["schema_version"]
    if marker["schema_version"] == 1:
        del expected_old["ffn_type"], expected_old["features"]
    if marker != expected_old:
        raise ValueError("Historical checkpoint contract disagrees with its config")
    result = {
        key: value
        for key, value in source.items()
        if key not in {"optimizer_states", "lr_schedulers", "loops", "callbacks"}
    }
    result["hyper_parameters"] = {
        **parameters,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    result["geometric_residual_contract"] = checkpoint_contract(parsed)
    result["geometric_residual_migration"] = {
        "source_contract": deepcopy(marker),
        "source_config": source_config,
        "weights_only": True,
        "conversion": "PLCS Court14/raw to package-by-layer schema 3",
    }
    return validate_residual_checkpoint(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not args.source.is_absolute() or not args.destination.is_absolute():
        raise ValueError("Use explicit absolute source and destination paths")
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=args.source.parent.resolve(),
        artifact_root=args.destination.parent.resolve(),
        output_root=args.destination.parent.resolve(),
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    paths = PATH_BOUNDARY.validate(
        {"source": args.source, "destination": args.destination},
        resolver=PathResolver(roots),
    )
    args.source = paths.declared("source").path
    args.destination = paths.declared("destination").path
    if args.source.resolve() == args.destination.resolve() or args.destination.exists():
        raise FileExistsError(
            "Choose a new destination; source checkpoints are never overwritten"
        )
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping):
        raise ValueError("Checkpoint must be a mapping")
    converted = convert_checkpoint(source)
    # Verify the exact architecture and tensor shapes before writing the artifact.
    from src.tasks.plcs.training.residual_lightning_module import (
        ResidualLightningModule,
    )

    module = ResidualLightningModule(
        OmegaConf.create(converted["hyper_parameters"]["config"])
    )
    module.load_state_dict(converted["state_dict"], strict=True)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("xb") as stream:
        torch.save(converted, stream)


if __name__ == "__main__":
    main()
