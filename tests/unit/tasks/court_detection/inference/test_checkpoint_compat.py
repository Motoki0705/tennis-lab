"""Tests for historical Court checkpoint inference compatibility."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.tasks.court_detection.configuration import LINE_TARGET_SCHEMA
from src.tasks.court_detection.inference.checkpoint_compat import (
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


def test_mapping_migration_adds_inference_defaults_without_mutation() -> None:
    legacy = {
        "run": {"output_dir": "outputs/test"},
        "data": {"processing": {"targets": _legacy_targets()}},
        "model": {"name": "test"},
    }

    migrated = migrate_court_inference_config(legacy)

    assert migrated is not None
    assert migrated["run"]["artifact_store"] == _EXPECTED_LOCAL_STORE
    assert migrated["data"]["processing"]["targets"][2]["target_schema"] == (
        LINE_TARGET_SCHEMA
    )
    assert "artifact_store" not in legacy["run"]
    assert legacy["data"]["processing"]["targets"][2]["target_schema"] == (
        "court_line_binary_75mm_150mm_v2"
    )
    assert migrated["model"] == legacy["model"]


def test_dictconfig_migration_preserves_interpolation() -> None:
    legacy = OmegaConf.create(
        {
            "paths": {"output_root": "outputs"},
            "run": {"output_dir": "${paths.output_root}/test"},
            "data": {"processing": {"targets": _legacy_targets()}},
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
    assert OmegaConf.select(legacy, "run.artifact_store") is None


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
    }

    assert migrate_court_inference_config(current) is None


def test_checkpoint_metadata_loader_returns_migrated_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "config": {
                    "run": {"output_dir": "outputs/test"},
                    "data": {"processing": {"targets": _legacy_targets()}},
                }
            },
            "state_dict": {"weight": torch.ones(1)},
        },
        checkpoint_path,
    )

    migrated = load_court_inference_config_override(checkpoint_path)

    assert migrated is not None
    assert migrated["run"]["artifact_store"] == _EXPECTED_LOCAL_STORE
    assert migrated["data"]["processing"]["targets"][2]["target_schema"] == (
        LINE_TARGET_SCHEMA
    )
