"""Observe the real Ball loops; no model/loss/backward/step implementation is fake."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.training.coordinates import joint, scratch
from src.tasks.ball_detection.training.dpt_pretrain import runner as pretrain
from src.tasks.ball_detection.training.hybrid import runner as hybrid
from src.tasks.ball_detection.training.runtime import execution
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal import runner as temporal
from tests.integration.tasks.ball_detection.model_contracts import (
    LoopInputs,
    parameter_contract,
)
from tests.support.model_contracts.steps import StepAudit
from tests.support.tasks.ball_detection.hybrid import hybrid_recipe
from tests.support.tasks.ball_detection.temporal_heatmap import update_budget


@contextmanager
def _optimizer_audit(model: Any, kind: str) -> Iterator[StepAudit]:
    def phase(step: int) -> Any:
        parameters, contracts = parameter_contract(
            model,
            kind=kind,
            frozen_phase=kind in {"deep_query", "hybrid"} and step == 0,
        )
        return parameters, contracts[0]

    audit = StepAudit(model, phase)
    original = torch.optim.AdamW
    loss_module, loss_name = {
        "coordinate": (execution, "coordinate_loss"),
        "deep_query": (execution, "coordinate_loss"),
        "pretrain": (pretrain, "heatmap_objective"),
        "hybrid": (hybrid, "hybrid_objective"),
        "temporal": (temporal, "supervised_loss_sum"),
    }[kind]
    objective = getattr(loss_module, loss_name)

    def record_loss(*args: Any, **kwargs: Any) -> Any:
        result = objective(*args, **kwargs)
        audit.record_loss(result[0] if isinstance(result, tuple) else result)
        return result

    def create(*args: Any, **kwargs: Any) -> Any:
        # Fixed .01 decay in legacy loops would hide a zero-loss head. Keep the
        # actual AdamW implementation, with decay explicitly disabled for audit.
        kwargs["weight_decay"] = 0.0
        optimizer = original(*args, **kwargs)
        excluded: frozenset[int] = getattr(
            model, "_contract_optimizer_exclusions", frozenset()
        )
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in excluded]
        audit.attach(optimizer)
        return optimizer

    try:
        with (
            patch("torch.optim.AdamW", side_effect=create),
            patch.object(loss_module, loss_name, side_effect=record_loss),
        ):
            yield audit
    finally:
        audit.close()


def _same_model(model: Any) -> Any:
    def factory(config: Any, **kwargs: Any) -> Any:
        assert config == model.config, "runner selected a different model configuration"
        return model

    return factory


def _scratch(model: Any, inputs: LoopInputs, output: Path) -> None:
    argv = [
        "contract",
        "--manifest",
        str(inputs.manifest),
        "--model-config",
        str(inputs.model_file),
        "--output",
        str(output),
        "--epochs",
        "1",
        "--windows-per-epoch",
        "1",
        "--learning-rate",
        ".001",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--precision",
        "fp32",
        "--cpu-threads",
        "1",
        "--jpeg-decoder",
        "opencv",
        "--compile-mode",
        "off",
        "--input-verification",
        "lazy",
    ]
    constructor = (
        "MDDPoseDetector" if model.config.requires_pose else "MDDQueryDetector"
    )
    with (
        patch.object(scratch, constructor, side_effect=_same_model(model)),
        patch.object(sys, "argv", argv),
    ):
        scratch.main()


def _pretrain(model: Any, inputs: LoopInputs, output: Path) -> None:
    args = SimpleNamespace(
        manifest=inputs.manifest,
        model_config=inputs.model_file,
        output=output,
        resume=None,
        epochs=1,
        windows_per_epoch=1,
        learning_rate=0.001,
        warmup_updates=0,
        seed=42,
        device="cpu",
        precision="fp32",
        batch_size=1,
        num_workers=0,
        prefetch_factor=1,
        cpu_threads=1,
        pin_memory=False,
        jpeg_decoder="opencv",
        input_verification="lazy",
        image_prefetch=False,
        compile_mode="off",
        selection_scope="common",
        sigma_ratio=0.012,
        focal_gamma=2.0,
        log_every=1,
        preview_clips=0,
        stop_after_epoch=None,
    )
    with (
        patch.object(pretrain, "arguments", return_value=args),
        patch.object(pretrain, "MDDDPTDetector", side_effect=_same_model(model)),
    ):
        pretrain.main()


def _deep_query(model: Any, inputs: LoopInputs, output: Path) -> None:
    augmentation = output.parent / "augmentation.yaml"
    raw = OmegaConf.load(
        Path("src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml")
    )
    raw.camera_probability, raw.occlusion_probability = 0.0, 0.0
    OmegaConf.save(raw, augmentation)
    args = SimpleNamespace(
        manifest=inputs.manifest,
        output=output,
        resume=None,
        augmentation_config=augmentation,
        epochs=2,
        freeze_epochs=1,
        windows_per_epoch=1,
        learning_rate=0.001,
        encoder_lr_ratio=0.1,
        warmup_updates=0,
        seed=42,
        device="cpu",
        precision="fp32",
        batch_size=1,
        num_workers=0,
        prefetch_factor=1,
        cpu_threads=1,
        pin_memory=False,
        jpeg_decoder="opencv",
        input_verification="lazy",
        image_prefetch=False,
        compile_mode="off",
        log_every=1,
        preview_clips=0,
        stop_after_epoch=None,
        selection_scope="common",
        stress_evaluation=False,
    )
    joint.run_finetuning(
        args,
        validation_manifest=inputs.manifest,
        initialize=lambda: (model, inputs.initialization),
    )


def _hybrid(model: Any, inputs: LoopInputs, output: Path) -> None:
    assert inputs.parent is not None
    config = hybrid_recipe(inputs.manifest, inputs.parent, output)
    config.model = dict(name="mdd_hybrid", **asdict(model.config))
    config.windows_per_epoch, config.warmup_updates, config.runtime.cpu_threads = (
        1,
        0,
        1,
    )
    config.absence.window_weights = dict(play=1.0, absent=0.0, nonplay=0.0)

    def initialize(path: Path, spec: Any, **kwargs: Any) -> Any:
        assert path == inputs.parent and spec == model.config
        return model, inputs.initialization

    with patch.object(hybrid, "initialize_hybrid", side_effect=initialize):
        hybrid.train_hybrid(config)


def _temporal(model: Any, inputs: LoopInputs, output: Path) -> None:
    architecture = {
        "mdd_dpt_heatmap": "dpt",
        "mdd_heatmap_hybrid": "hybrid",
        "mdd_temporal_query": "query",
    }[inputs.config.model.name]
    with patch.object(
        temporal, "build_temporal_model", side_effect=_same_model(model)
    ):
        temporal.train_temporal_model(
            architecture=architecture,
            phase="pretrain",
            manifest=inputs.manifest,
            evaluation_manifest=inputs.manifest.parent / "evaluation_T32.json",
            model_config=model.config,
            output=output,
            budget=update_budget(
                updates=1, warmup_updates=0, weight_decay=0.0, evaluation_interval=1
            ),
            runtime=CoordinateRuntime(cpu_threads=1),
            device=torch.device("cpu"),
            augmentation=None,
            distance_px=20.0,
        )


def fit_loop(
    model: Any,
    inputs: LoopInputs,
    parameters: Any,
    contracts: Any,
    output: Path,
    *,
    gan: bool,
) -> StepAudit:
    assert not gan
    runners = dict(
        coordinate=_scratch,
        pretrain=_pretrain,
        deep_query=_deep_query,
        hybrid=_hybrid,
        temporal=_temporal,
    )
    with _optimizer_audit(model, inputs.kind) as audit:
        runners[inputs.kind](model, inputs, output)
    assert audit.updates == (2 if inputs.kind in {"deep_query", "hybrid"} else 1)
    assert audit.losses
    parameters.validate(model)  # staged loops must end with every group thawed
    return audit
