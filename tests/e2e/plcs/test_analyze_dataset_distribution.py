"""Entry-point contract tests for PLCS dataset distribution analysis."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Protocol, cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

import src.tasks.plcs.scripts.analysis.analyze_dataset_distribution as analysis_script
from src.utils.schema.court_normalization import (
    COURT_COORDINATE_NORMALIZATION_KEY,
    CourtCoordinateContractError,
    court_coordinate_normalization_metadata,
)

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "tasks" / "plcs" / "configs"


class _WrappedEntrypoint(Protocol):
    __wrapped__: Callable[[DictConfig], int]


@pytest.mark.parametrize("mutation", ["missing", "malformed", "unknown", "mismatched"])
def test_analysis_rejects_invalid_scene_contract_before_loading_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    scene = tmp_path / "plcs" / "scenes" / "scene_000"
    scene.mkdir(parents=True)
    contract: object = court_coordinate_normalization_metadata()
    if mutation == "malformed":
        contract = "isotropic_half_length"
    elif mutation in {"unknown", "mismatched"}:
        assert isinstance(contract, dict)
        contract = deepcopy(contract)
        if mutation == "unknown":
            contract["identity"] = "anisotropic"
        else:
            contract["scale_xyz_m"] = [5.485, 11.885, 1.07]
    metadata = {
        "scene_id": "scene_000",
        "num_frames": 1,
        COURT_COORDINATE_NORMALIZATION_KEY: contract,
    }
    if mutation == "missing":
        del metadata[COURT_COORDINATE_NORMALIZATION_KEY]
    (scene / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    def fail_array_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("normalized arrays must not load before validation")

    monkeypatch.setattr(analysis_script.np, "load", fail_array_load)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="analyze_dataset_distribution",
            overrides=[
                f"paths.data_root={tmp_path}",
                f"paths.output_root={tmp_path / 'outputs'}",
                "data.scene_dir=plcs",
                "run.output_dir=analysis",
                "plots.enabled=false",
            ],
        )
    entrypoint = cast(_WrappedEntrypoint, analysis_script.main).__wrapped__

    with pytest.raises(CourtCoordinateContractError, match="incompatible|mismatched"):
        entrypoint(config)
