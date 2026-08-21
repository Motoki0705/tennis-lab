"""Artifact-contract tests for the SLCS prediction analysis entry point."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.scripts.analyze_predictions import run

_savez_compressed = cast(Callable[..., None], np.savez_compressed)


def _arrays() -> dict[str, np.ndarray]:
    player_mask = np.array([[[True, False]]])
    ball_mask = np.array([[True, False]])
    return {
        "player_pos_error_m": np.array([[[0.25, 99.0]]], dtype=np.float32),
        "player_ang_error_deg": np.array([[[5.0, 99.0]]], dtype=np.float32),
        "ball_pos_error_m": np.array([[0.5, 99.0]], dtype=np.float32),
        "player_mask": player_mask,
        "ball_mask": ball_mask,
        "padding_mask": np.array([[False, True]]),
        "player_observed": player_mask.copy(),
        "ball_observed": ball_mask.copy(),
        "player_sigma_m": np.ones((1, 1, 2), dtype=np.float32),
        "player_rot_sigma_deg": np.ones((1, 1, 2), dtype=np.float32),
        "ball_sigma_m": np.ones((1, 2), dtype=np.float32),
    }


def _config(tmp_path: Path, *, arrays: str, output_dir: str) -> DictConfig:
    return OmegaConf.create(
        {
            "paths": {
                "project_root": str(tmp_path),
                "data_root": "data",
                "checkpoint_root": "checkpoints",
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": "cache",
                "external_asset_root": "external-assets",
            },
            "analysis": {
                "arrays": arrays,
                "calibration_bins": 2,
                "output_dir": output_dir,
            },
        }
    )


def test_analysis_consumes_true_means_padding_artifact(tmp_path: Path) -> None:
    arrays_path = tmp_path / "outputs" / "evaluation" / "eval_arrays.npz"
    arrays_path.parent.mkdir(parents=True)
    _savez_compressed(arrays_path, **_arrays())

    run(
        _config(
            tmp_path,
            arrays="evaluation/eval_arrays.npz",
            output_dir="analysis",
        )
    )

    report = json.loads((tmp_path / "outputs" / "analysis" / "analysis.json").read_text())
    assert report["num_windows"] == 1
    assert report["label_missing_rate_player"] == pytest.approx(0.0)
    assert report["label_missing_rate_ball"] == pytest.approx(0.0)
    assert report["player_pos_error_m"]["mean"] == pytest.approx(0.25)
    assert report["ball_pos_error_m"]["mean"] == pytest.approx(0.5)


def test_analysis_rejects_legacy_frame_mask_artifact(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays["frame_mask"] = ~arrays.pop("padding_mask")
    arrays_path = tmp_path / "outputs" / "evaluation" / "legacy_arrays.npz"
    arrays_path.parent.mkdir(parents=True)
    _savez_compressed(arrays_path, **arrays)

    with pytest.raises(KeyError, match="padding_mask"):
        run(
            _config(
                tmp_path,
                arrays="evaluation/legacy_arrays.npz",
                output_dir="legacy-analysis",
            )
        )
