"""Regression checks for the issued experiment commands and evidence files."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from src.tennis_scene.configuration import parse_video_batch_config
from src.utils.hydra import HYDRA_PATH_RESOLVER
from src.utils.paths import PROJECT_ROOT


def load_campaign() -> ModuleType:
    spec = importlib.util.spec_from_file_location("player_video_experiment", Path(__file__).with_name("run.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hydra_and_video_batch_resolve_the_same_progress_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = load_campaign()
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(campaign, "OUTPUTS", output_root)
    run = output_root / "i2363/test/host"
    dataset = tmp_path / "input/dataset"
    engines = {name: f"tensorrt/{name}/fixture" for name in campaign.CONFIG["models"]}
    args = campaign.reconstruction_arguments(
        run, dataset, engines, synchronize=False, clip="video_000/clip_037",
    )
    assert OmegaConf.has_resolver(HYDRA_PATH_RESOLVER)
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")):
        cfg = compose(config_name="run_video_batch", overrides=args, return_hydra_config=True)
        assert Path(OmegaConf.select(cfg, "hydra.run.dir")) == run / "progress/hydra"
        with open_dict(cfg):
            del cfg["hydra"]
        runtime = parse_video_batch_config(cfg)
    assert runtime.progress_directory == run / "progress"
    assert runtime.generation.dataset_directory == dataset
    assert runtime.record_cuda and not runtime.record_cuda_synchronize


def test_command_receipt_preserves_the_report_created_by_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = load_campaign()
    log = tmp_path / "qualification.log"
    report = log.with_suffix(".json")
    original = {"status": "complete", "shapes": {"B4_L177": {"tensorrt_finite": True}}}

    def child(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        assert command[-1] == str(report)
        report.write_text(json.dumps(original))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(campaign.subprocess, "run", child)
    result = campaign.run_module("fixture.qualification", [str(report)], log)
    assert json.loads(report.read_text()) == original
    assert json.loads(log.with_suffix(".command.json").read_text()) == result
    assert result["exit_code"] == 0
