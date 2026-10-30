"""Minimal real inputs shared by the MDD runner and public-entry contracts."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from omegaconf import DictConfig

from src.tasks.ball_detection.checkpoints.provenance import data_identity
from tests.integration.tasks.ball_detection.contract_loops import _pretrain
from tests.integration.tasks.ball_detection.model_contracts import (
    FAMILIES,
    LoopInputs,
    _query_parent,
)
from tests.support.model_contracts.lifecycle import assert_tree_equal
from tests.support.tasks.ball_detection.hybrid import hybrid_recipe


def invoke_main(entry: Callable[[], None], arguments: list[str]) -> None:
    """Call a real runner/evaluator, leaving process dispatch to the e2e tests."""
    with patch.object(sys, "argv", ["mdd-contract", *arguments]):
        entry()


def tiny_inputs(root: Path, *, channels: int = 2) -> tuple[Any, LoopInputs]:
    root.mkdir(parents=True, exist_ok=True)
    family = FAMILIES[
        {2: "mdd_dpt_pretrain", 5: "mdd_dpt_rgb_mdd"}[channels]
    ]
    model, inputs = family.build(family.compose(root, False))
    return model, inputs


def pretrained_run(root: Path, *, channels: int = 2) -> tuple[LoopInputs, Path]:
    model, inputs = tiny_inputs(root, channels=channels)
    output = root / "pretrained"
    _pretrain(model, inputs, output)
    return inputs, output


def training_arguments(manifest: Path, output: Path) -> list[str]:
    return [
        "--manifest", str(manifest), "--output", str(output),
        "--epochs", "2", "--windows-per-epoch", "1",
        "--learning-rate", ".001", "--warmup-updates", "0", "--seed", "42",
        "--device", "cpu", "--precision", "fp32", "--jpeg-decoder", "opencv",
        "--input-verification", "lazy", "--num-workers", "0",
        "--cpu-threads", "1", "--compile-mode", "off",
        "--preview-clips", "0", "--log-every", "1",
    ]


def encoder_arguments(manifest: Path, parent: Path, output: Path) -> list[str]:
    augmentation = Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml").resolve()
    return [
        *training_arguments(manifest, output),
        "--pretraining-run", str(parent), "--freeze-epochs", "1",
        "--augmentation-config", str(augmentation),
    ]


def tiny_hybrid_recipe(root: Path) -> DictConfig:
    source, inputs = tiny_inputs(root, channels=5)
    parent = _query_parent(root, source.config, inputs.manifest)
    saved = torch.load(parent, weights_only=True)
    saved["recipe"]["data_identity"] = data_identity(inputs.manifest)
    # Descendants must be portable even when their recorded grandparent is gone.
    saved["recipe"]["parent"]["checkpoint"] = str(root / "unavailable-dpt.pt")
    torch.save(saved, parent)
    cfg = hybrid_recipe(inputs.manifest, parent, root / "hybrid")
    cfg.model = dict(name="mdd_hybrid", **asdict(source.config))
    cfg.windows_per_epoch, cfg.warmup_updates, cfg.runtime.cpu_threads = 1, 1, 1
    cfg.absence.window_weights = dict(play=1., absent=0., nonplay=0.)
    return cfg


def weights(path: Path) -> dict[str, Any]:
    saved: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return saved


def omit_explicit_channels(payload: dict[str, Any]) -> None:
    """The supported historical format identifies 2ch from its saved contract."""
    assert payload["model_config"].pop("input_channels") == 2
    assert payload["recipe"]["model"].pop("input_channels") == 2


def assert_replay(expected: Path, actual: Path) -> None:
    a, b = weights(expected), weights(actual)
    for key in (
        "recipe", "state_dict", "optimizer", "epoch", "global_step",
        "torch_rng", "cuda_rng", "validation", "best_epoch", "best_score",
    ):
        assert_tree_equal(a[key], b[key], f"resume.{key}")


def diagnostic_prefix(completed: Path, interrupted: Path) -> None:
    """Keep the committed first epoch; the real resume truncates later logs."""
    import shutil

    shutil.copytree(completed, interrupted)
    (interrupted / "epoch-001.pt").unlink()
    (interrupted / "COMPLETED.json").unlink()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
