"""Tests for historical Court checkpoint inference compatibility."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.court_detection.configuration import LINE_TARGET_SCHEMA
from src.tasks.court_detection.inference.checkpoint_compat import (
    extract_court_checkpoint_dense_head_config,
    load_court_inference_config_override,
    migrate_court_inference_config,
)

_EXPECTED_LOCAL_STORE = {
    "mode": "local",
    "remote": None,
    "remote_root": None,
    "sync_interval_seconds": None,
}


def _legacy_targets() -> list[dict[str, str | float]]:
    return [
        {"kind": "kp", "sigma_ratio": 0.01},
        {"kind": "seg", "target_schema": "court_cell_segmentation_v1"},
        {
            "kind": "line",
            "target_schema": "court_line_binary_75mm_150mm_v2",
        },
    ]


def _residual_dense_head() -> dict[str, Any]:
    return {
        "name": "residual",
        "normalization_groups": 32,
        "kp": {"hidden_channels": 256, "depth": 2},
        "seg": {"hidden_channels": 256, "depth": 2},
        "line": {"hidden_channels": 256, "depth": 2},
    }


def _runtime_roots() -> dict[str, str]:
    return {
        "project_root": "/workspace",
        "data_root": "/tennis-lab/data",
        "checkpoint_root": "/tennis-lab/ckpt",
        "artifact_root": "/workspace/assets",
        "output_root": "/tennis-lab/outputs",
        "cache_root": "/tennis-lab/.cache",
        "external_asset_root": "/tennis-lab/third_party",
    }


def _legacy_config() -> dict[str, Any]:
    return {
        "paths": {
            "project_root": ".",
            "data_root": "/old-host/tennis-lab/data",
            "checkpoint_root": "/old-host/tennis-lab/outputs",
            "artifact_root": "assets",
            "output_root": "/old-host/tennis-lab/outputs",
            "cache_root": "/old-host/tennis-lab/.cache",
            "external_asset_root": "/old-host/tennis-lab/third_party",
        },
        "run": {"output_dir": "outputs/test"},
        "data": {"processing": {"targets": _legacy_targets()}},
        "model": {
            "name": "court_hierarchical",
            "dense_head": _residual_dense_head(),
        },
        "mixed": {"batch_size": 8},
    }


def test_mapping_migration_adds_inference_defaults_without_mutation() -> None:
    legacy = _legacy_config()

    migrated = migrate_court_inference_config(legacy)

    assert migrated is not None
    assert migrated["run"]["artifact_store"] == _EXPECTED_LOCAL_STORE
    assert migrated["data"]["processing"]["targets"][2]["target_schema"] == (
        LINE_TARGET_SCHEMA
    )
    assert "dense_head" not in migrated["model"]
    assert "mixed" not in migrated

    assert "artifact_store" not in legacy["run"]
    assert legacy["data"]["processing"]["targets"][2]["target_schema"] == (
        "court_line_binary_75mm_150mm_v2"
    )
    assert legacy["model"]["dense_head"] == _residual_dense_head()
    assert legacy["mixed"] == {"batch_size": 8}


def test_runtime_roots_replace_serialized_machine_paths_without_mutation() -> None:
    legacy = _legacy_config()

    migrated = migrate_court_inference_config(
        legacy,
        runtime_path_roots=_runtime_roots(),
    )

    assert migrated is not None
    assert migrated["paths"] == _runtime_roots()
    assert legacy["paths"]["external_asset_root"] == (
        "/old-host/tennis-lab/third_party"
    )


def test_runtime_roots_require_the_complete_contract() -> None:
    roots = _runtime_roots()
    roots.pop("external_asset_root")

    with pytest.raises(ValueError, match="missing=.*external_asset_root"):
        migrate_court_inference_config(
            _legacy_config(),
            runtime_path_roots=roots,
        )


def test_dictconfig_migration_preserves_interpolation() -> None:
    legacy = OmegaConf.create(
        {
            "paths": {"output_root": "outputs"},
            "run": {"output_dir": "${paths.output_root}/test"},
            "data": {"processing": {"targets": _legacy_targets()}},
            "model": {
                "name": "court_hierarchical",
                "dense_head": _residual_dense_head(),
            },
            "mixed": {"batch_size": 8},
        }
    )

    migrated = migrate_court_inference_config(legacy)

    assert migrated is not None
    assert OmegaConf.select(migrated, "run.artifact_store.mode") == "local"
    assert OmegaConf.select(migrated, "run.output_dir") == "outputs/test"
    assert (
        OmegaConf.select(
            migrated,
            "data.processing.targets[2].target_schema",
        )
        == LINE_TARGET_SCHEMA
    )
    assert OmegaConf.select(migrated, "model.dense_head") is None
    assert OmegaConf.select(migrated, "mixed") is None
    assert OmegaConf.select(legacy, "run.artifact_store") is None
    assert OmegaConf.select(legacy, "model.dense_head.name") == "residual"


def test_dense_head_metadata_is_extracted_without_mutation() -> None:
    legacy = OmegaConf.create(_legacy_config())

    extracted = extract_court_checkpoint_dense_head_config(legacy)

    assert extracted == _residual_dense_head()
    assert OmegaConf.select(legacy, "model.dense_head.name") == "residual"


def test_current_config_requires_no_override() -> None:
    current = {
        "run": {
            "output_dir": "outputs/test",
            "artifact_store": dict(_EXPECTED_LOCAL_STORE),
        },
        "data": {
            "processing": {
                "targets": [
                    {"kind": "line", "target_schema": LINE_TARGET_SCHEMA}
                ]
            }
        },
        "model": {"name": "court_hierarchical"},
    }

    assert migrate_court_inference_config(current) is None
    assert extract_court_checkpoint_dense_head_config(current) is None


def test_checkpoint_metadata_loader_returns_migrated_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "hyper_parameters": {"config": _legacy_config()},
            "state_dict": {"weight": torch.ones(1)},
        },
        checkpoint_path,
    )

    migrated = load_court_inference_config_override(
        checkpoint_path,
        runtime_path_roots=_runtime_roots(),
    )

    assert migrated is not None
    assert migrated["paths"] == _runtime_roots()
    assert migrated["run"]["artifact_store"] == _EXPECTED_LOCAL_STORE
    assert migrated["data"]["processing"]["targets"][2]["target_schema"] == (
        LINE_TARGET_SCHEMA
    )
    assert "dense_head" not in migrated["model"]
    assert "mixed" not in migrated
