"""One small real subprocess per fixed-length MDD training entry."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from src.tasks.ball_detection.checkpoints.pretraining import transfer_encoder
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
from tests.integration.tasks.ball_detection.mdd_training_contracts import (
    encoder_arguments,
    pretrained_run,
    read_json,
    tiny_hybrid_recipe,
    tiny_inputs,
    training_arguments,
    weights,
)
from tests.support.tasks.ball_detection.learning import (
    full_model_arguments,
    full_model_checkpoint,
    training_manifests,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("entry", ["dpt_pretrain", "encoder", "full_model", "hybrid"])
def test_training_entry_commits_its_stage(tmp_path: Path, entry: str) -> None:
    output = tmp_path / "run"
    if entry == "dpt_pretrain":
        source, inputs = tiny_inputs(tmp_path)
        args = [
            "--stage", entry, *training_arguments(inputs.manifest, output),
            "--model-config", str(inputs.model_file), "--epochs", "1",
            "--input-verification", "upfront", "--preview-clips", "1",
        ]
    elif entry == "encoder":
        inputs, parent = pretrained_run(tmp_path, channels=5)
        args = [
            "--stage", "coordinates", "--initialize", entry,
            *encoder_arguments(inputs.manifest, parent, output),
        ]
    elif entry == "full_model":
        parent = full_model_checkpoint(tmp_path / "parent")
        train, validation = training_manifests(tmp_path)
        args = [
            "--stage", "coordinates", "--initialize", entry,
            *full_model_arguments(parent, train, validation, output),
            "--cpu-threads", "1",
        ]
    else:
        cfg = tiny_hybrid_recipe(tmp_path)
        cfg.run.output_dir = output.name
        recipe = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(recipe, dict)
        recipe["hydra"] = dict(run=dict(dir=str(output / "hydra")), output_subdir=None, job=dict(chdir=False))
        (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(recipe))
        args = ["--stage", entry, "--config-path", str(tmp_path), "--config-name", "recipe"]

    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-m", "src.tasks.ball_detection.scripts.train", *args],
        env=environment,
        text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected_step = 1 if entry == "dpt_pretrain" else 2
    completed = read_json(output / "COMPLETED.json")
    assert completed["global_step"] == expected_step
    checkpoint = weights(output / f"epoch-{expected_step - 1:03d}.pt")
    stage = {"dpt_pretrain": "heatmap_pretraining", "hybrid": "hybrid"}.get(entry, "query_posttraining")
    assert checkpoint["stage"] == completed["stage"] == stage
    assert checkpoint["recipe"]["test_usage"] == "none"
    assert set(checkpoint["validation"]["scopes"]) == {"full", "common"}
    if entry == "dpt_pretrain":
        assert completed["automatic_next_stage"] is False
        assert checkpoint["validation"]["scopes"]["full"]["macro_heatmap_loss"] > 0
        assert list((output / "previews").rglob("*.gif"))
        target = DeepMDDQueryDetector(source.config)
        assert transfer_encoder(output / "epoch-000.pt", target)["global_step"] == 1
    elif entry == "hybrid":
        assert (output / "hydra" / "runner.log").is_file()
        evaluated = tmp_path / "evaluation.json"
        result = subprocess.run(
            [sys.executable, "-m", "src.tasks.ball_detection.scripts.evaluate", "--mode", "hybrid",
             "--checkpoint", str(output / "epoch-001.pt"), "--manifest", str(output / "data_manifest.json"),
             "--output", str(evaluated), "--split", "val", "--device", "cpu"],
            env=environment, text=True, capture_output=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert read_json(evaluated)["scopes"] == checkpoint["validation"]["scopes"]
    else:
        assert checkpoint["recipe"]["initialization"] == dict(mode=entry, seed=42)
