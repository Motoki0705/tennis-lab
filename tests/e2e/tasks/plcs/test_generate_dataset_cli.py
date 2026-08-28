from __future__ import annotations

import inspect
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.tasks.plcs.scripts import generate_dataset as generate_dataset_script


@pytest.mark.parametrize("selector", ["physical_v1", "camera_view_v2"])
@pytest.mark.parametrize("camera", ["default", "broadcast"])
def test_cli_can_publish_resolved_selector_without_loading_assets(
    selector: str,
    camera: str,
) -> None:
    root = Path(__file__).resolve().parents[4]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.plcs.scripts.generate_dataset",
            "--cfg",
            "job",
            f"court_keypoints={selector}",
            f"camera={camera}",
            "run.device=cpu",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"selector: {selector}" in result.stdout
    assert "court_coordinate_normalization:" not in result.stdout


def test_cli_keeps_physical_v1_as_the_public_default() -> None:
    root = Path(__file__).resolve().parents[4]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.plcs.scripts.generate_dataset",
            "--cfg",
            "job",
            "run.device=cpu",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "court_keypoints:\n  selector: physical_v1" in result.stdout
    assert "court_coordinate_normalization:" not in result.stdout


@pytest.mark.parametrize("selector", ["physical_v1", "camera_view_v2"])
def test_cli_publishes_root_contract_before_first_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
) -> None:
    contract = resolve_court_keypoint_contract(selector)
    runtime = SimpleNamespace(
        config=OmegaConf.create({"court_keypoints": {"selector": selector}}),
        court_keypoint_contract=contract,
        output_dir=tmp_path,
        device="cpu",
        seed=1,
        num_workers=1,
        num_scenes=1,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    writer = MagicMock()

    def _runtime_from_config(value: object) -> SimpleNamespace:
        del value
        return runtime

    def _writer_factory(*args: object, **kwargs: object) -> MagicMock:
        assert args == (tmp_path,)
        assert kwargs == {"court_keypoint_contract": contract}
        return writer

    scene = SimpleNamespace(
        meta={
            "scene_id": "scene_000000",
            "motion_category": "fixture",
            "num_frames": 1,
        },
        cameras=[],
    )

    def _one_scene(**kwargs: object) -> Iterator[object]:
        del kwargs
        assert [method_call[0] for method_call in writer.method_calls] == [
            "save_meta_json"
        ]
        yield scene

    monkeypatch.setattr(
        PLCSGenerationConfig,
        "from_config",
        _runtime_from_config,
    )
    monkeypatch.setattr(
        generate_dataset_script,
        "PLCSDatasetWriter",
        _writer_factory,
    )
    monkeypatch.setattr(
        generate_dataset_script,
        "generate_parallel_scenes",
        _one_scene,
    )
    entrypoint = cast(
        "Callable[[DictConfig], int]",
        inspect.unwrap(generate_dataset_script.main),
    )

    assert entrypoint(runtime.config) == 0
    assert [method_call[0] for method_call in writer.method_calls] == [
        "save_meta_json",
        "save_scene",
        "save_meta_json",
        "save_split_info",
    ]
