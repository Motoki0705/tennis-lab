"""Checkpoint and interrupted-training contracts over the registered real tasks.

CPU comparisons are exact, including NaN locations, dtype and nested schemas.
The adapters own physical-unit decoding; this engine never selects output fields.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.tasks.base.training.gan_transition_callback import GANTransitionCallback
from tests.support.model_contracts.registry import ModelFamily


def assert_tree_equal(expected: Any, actual: Any, path: str = "output") -> None:
    """Compare every leaf, rejecting missing fields and implicit dtype conversions."""
    assert type(actual) is type(expected), (
        f"{path}: type {type(actual)} != {type(expected)}"
    )
    if isinstance(expected, torch.Tensor):
        assert actual.device == expected.device, f"{path}: device"
        assert actual.dtype == expected.dtype, f"{path}: dtype"
        assert actual.shape == expected.shape, f"{path}: shape"
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, equal_nan=True, msg=path
        )
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, f"{path}: dtype"
        np.testing.assert_array_equal(actual, expected, err_msg=path, strict=True)
    elif OmegaConf.is_config(expected):
        assert_tree_equal(
            OmegaConf.to_container(expected, resolve=True),
            OmegaConf.to_container(actual, resolve=True),
            path,
        )
    elif is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            assert_tree_equal(
                getattr(expected, field.name),
                getattr(actual, field.name),
                f"{path}.{field.name}",
            )
    elif isinstance(expected, Mapping):
        assert expected.keys() == actual.keys(), (
            f"{path}: fields {expected.keys()} != {actual.keys()}"
        )
        for key, value in expected.items():
            assert_tree_equal(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual), f"{path}: length"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_tree_equal(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def rng_state() -> dict[str, Any]:
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
    )


def snapshot_tree(value: Any) -> Any:
    """Own mutable state while retaining schemas of frozen metadata records."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type(value).__qualname__,
            **{
                field.name: snapshot_tree(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {key: snapshot_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot_tree(item) for item in value)
    return copy.deepcopy(value)


def _fit(
    trainer: pl.Trainer, module: Any, data: Any, checkpoint: Path | None = None
) -> None:
    kwargs = {"ckpt_path": checkpoint, "weights_only": False}
    if isinstance(data, DataLoader):
        trainer.fit(module, train_dataloaders=data, **kwargs)
    else:
        trainer.fit(module, datamodule=data, **kwargs)


def _trainer(
    root: Path, callbacks: list[pl.Callback], *, epochs: int, validation: bool
) -> pl.Trainer:
    return pl.Trainer(
        default_root_dir=root,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=epochs,
        limit_train_batches=1,
        limit_val_batches=int(validation),
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
    )


def exercise_checkpoint(
    family: ModelFamily,
    root: Path,
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Train, save to disk, strict-load a fresh object, compare all public fields."""
    pl.seed_everything(219, workers=True)
    config = family.compose(root, False)
    module, data = family.build(config)
    path = root / "roundtrip.ckpt"
    if isinstance(module, pl.LightningModule):

        class SaveEpoch(pl.Callback):
            def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Any) -> None:
                family.save(trainer, path)
                trainer.should_stop = True

        trainer = _trainer(
            root,
            [SaveEpoch()],
            epochs=int(config.training.trainer.max_epochs),
            validation=family.name.startswith("refiner/"),
        )
        _fit(trainer, module, data)
    else:
        assert family.fit is not None
        parameters, optimizers = family.parameters(module)
        audit = family.fit(
            module, data, parameters, optimizers, root / "fit", gan=False
        )
        family.save(audit.artifact, path)
    batch = family.batch(config)
    module.eval()
    with torch.no_grad():
        pl.seed_everything(927, workers=True)
        expected = family.predict(module, batch)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if mutate is not None:
        mutate(saved)
        torch.save(saved, path)
    calls = []
    load = torch.nn.Module.load_state_dict

    def strict_load(
        instance: Any, state: Any, strict: bool = True, **kwargs: Any
    ) -> Any:
        calls.append(strict)
        return load(instance, state, strict=strict, **kwargs)

    with patch.object(torch.nn.Module, "load_state_dict", strict_load):
        restored = family.restore(config, path)
    assert restored is not module, "restore must construct a fresh model"
    assert calls and all(calls), "restore must load strictly"
    assert_tree_equal(module.config, restored.config, "config")
    # No storage may be shared with the source model.
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(module.parameters(), restored.parameters(), strict=True)
    )
    restored.eval()
    with torch.no_grad():
        pl.seed_everything(927, workers=True)
        actual = family.predict(restored, batch)
    assert_tree_equal(expected, actual)
    if family.predict_artifact is not None:
        with torch.no_grad():
            pl.seed_everything(927, workers=True)
            deployed = family.predict_artifact(path, batch)
        assert_tree_equal(expected, deployed, "inference_artifact")


class _Trace(pl.Callback):
    def __init__(self, stop: int | None, checkpoint: Path, family: ModelFamily) -> None:
        self.stop = stop
        self.checkpoint = checkpoint
        self.family = family
        self.records: list[dict[str, Any]] = []

    def on_train_batch_end(
        self, trainer: pl.Trainer, module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self.records.append(
            snapshot_tree(
                dict(
                    rng=rng_state(),
                    data=trainer.datamodule.state_dict()
                    if trainer.datamodule is not None
                    else None,
                    batch=batch,
                    scheduler=[
                        config.scheduler.state_dict()
                        for config in trainer.lr_scheduler_configs
                    ],
                    optimizer=[
                        optimizer.state_dict() for optimizer in trainer.optimizers
                    ],
                    model=module.state_dict(),
                    position=(trainer.current_epoch, batch_idx, trainer.global_step),
                    task_state=self.family.training_state(module)
                    if self.family.training_state is not None
                    else None,
                )
            )
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, module: Any) -> None:
        if self.stop is not None and trainer.current_epoch + 1 == self.stop:
            self.family.save(trainer, self.checkpoint)
            trainer.should_stop = True


def exercise_resume(
    family: ModelFamily,
    root: Path,
    *,
    gan: bool = False,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    interrupt_epoch: int = 1,
) -> None:
    """N=2 updates/epochs plus one extra update; interrupt after the first epoch.

    The schedule horizon stays fixed. Model, optimizer, scheduler, global RNG,
    actual batches, data RNG and Lightning's epoch/batch/global-step are compared
    at N and N+1, including manual D/G updates when the family enables GAN.
    """
    config = family.compose(root, gan)
    epochs = interrupt_epoch + 2
    if family.name.startswith("refiner/"):
        config.training.steps = epochs
    else:
        config.training.trainer.max_epochs = epochs
    if "dropout" in config.model:
        config.model.dropout = 0.1
    # Build once: mutable modules are fresh, generated immutable datasets are shared.
    pl.seed_everything(219, workers=True)
    initial, data = family.build(config)
    assert isinstance(initial, pl.LightningModule), (
        "non-Lightning families need their task-owned resume driver"
    )
    assert family.rebuild is not None, (
        "resume needs a registered fresh module/data factory"
    )
    interrupted, interrupted_data = family.rebuild(config)
    interrupted.load_state_dict(initial.state_dict(), strict=True)
    _, resumed_data = family.rebuild(config)

    def run(
        module: Any, run_data: Any, *, stop: int | None, checkpoint: Path | None = None
    ) -> tuple[pl.Trainer, _Trace]:
        trace = _Trace(stop, root / "interrupted.ckpt", family)
        callbacks: list[pl.Callback] = [trace]
        if gan and not family.name.startswith("refiner/"):
            callbacks.insert(0, GANTransitionCallback(config))
        trainer = _trainer(
            root,
            callbacks,
            epochs=epochs,
            validation=family.name.startswith("refiner/"),
        )
        # Rebuild datasets before every fresh run; restore overwrites their state.
        _fit(trainer, module, run_data, checkpoint)
        return trainer, trace

    pl.seed_everything(815, workers=True)
    _, continuous = run(initial, data, stop=None)
    pl.seed_everything(815, workers=True)
    _, first = run(interrupted, interrupted_data, stop=interrupt_epoch)
    assert_tree_equal(continuous.records[:interrupt_epoch], first.records, "prefix")
    checkpoint = root / "interrupted.ckpt"
    assert checkpoint.is_file()
    if mutate is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        mutate(payload)
        torch.save(payload, checkpoint)
    pl.seed_everything(731, workers=True)  # a new process starts with unrelated RNG
    resumed = family.restore(config, checkpoint)
    _, remainder = run(resumed, resumed_data, stop=None, checkpoint=checkpoint)
    assert len(continuous.records) == epochs and len(remainder.records) == 2
    for index in range(interrupt_epoch, epochs):
        assert_tree_equal(
            continuous.records[index],
            remainder.records[index - interrupt_epoch],
            f"resume[{index + 1}]",
        )


def omit_head(path: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        keys = [
            key
            for key in payload["state_dict"]
            if key == path or key.startswith(path + ".")
        ]
        assert keys, f"unknown injected head {path}"
        for key in keys:
            del payload["state_dict"][key]

    return mutate


def resume_fault(kind: str) -> Callable[[dict[str, Any]], None]:
    """Simulate omitted state while keeping the remaining checkpoint valid."""

    def mutate(payload: dict[str, Any]) -> None:
        if kind == "optimizer":
            for optimizer in payload["optimizer_states"]:
                assert optimizer["state"], "fault needs initialized optimizer moments"
                optimizer["state"] = {}
        elif kind == "rng":
            unrelated = torch.Generator().manual_seed(731).get_state()
            if "training_random_state" in payload:
                payload["training_random_state"]["torch"] = unrelated
            if "torch_rng" in payload:
                payload["torch_rng"] = unrelated
                payload["flow_rng"] = unrelated
        elif kind == "data":
            keys = [
                key
                for key, value in payload.items()
                if isinstance(value, dict) and "loader_rng" in value
            ]
            assert len(keys) == 1, "fault needs a checkpointed data stream"
            del payload[keys[0]]
        else:
            raise ValueError(f"Unknown resume fault {kind}")

    return mutate
