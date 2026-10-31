"""Resume adapters for Ball's five real non-Lightning loop families."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytorch_lightning as pl
import torch

from tests.integration.tasks.ball_detection import contract_loops as loops
from tests.support.model_contracts.lifecycle import assert_tree_equal, snapshot_tree
from tests.support.model_contracts.registry import ModelFamily


class _Interrupted(Exception):
    """An interruption immediately after committing an epoch checkpoint."""


def _artifact(output: Path, kind: str, completed: int) -> Path:
    return output / (
        f"checkpoint-update-{completed:08d}.pt"
        if kind == "temporal"
        else f"epoch-{completed - 1:03d}.pt"
    )


def _state(path: Path, kind: str) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if kind == "temporal":
        return {
            key: payload[key]
            for key in (
                "model_artifact",
                "optimizer",
                "schedule",
                "rng",
                "updates_completed",
                "sample_position",
            )
        }
    step = (
        payload["training_state"]["global_step"]
        if kind == "coordinate"
        else payload["global_step"]
    )
    # These loops implement LR directly from recipe + epoch + completed updates.
    # The optimizer param groups contain the LR used by the next update.
    return dict(
        model=payload["state_dict"],
        optimizer=payload["optimizer"],
        torch_rng=payload["torch_rng"],
        cuda_rng=payload["cuda_rng"]
        if kind != "coordinate"
        else payload["training_state"]["cuda_rng"],
        epoch=payload["epoch"],
        global_step=step,
    )


@contextmanager
def _batches(kind: str) -> Iterator[list[Any]]:
    batches: list[Any] = []
    if kind in {"coordinate", "deep_query"}:
        original = loops.execution.prediction_batch

        def prepare(batch: Any, *args: Any, **kwargs: Any) -> Any:
            batches.append(snapshot_tree(batch))
            return original(batch, *args, **kwargs)

        with patch.object(loops.execution, "prediction_batch", side_effect=prepare):
            yield batches
    else:
        module = {
            "pretrain": loops.pretrain,
            "hybrid": loops.hybrid,
            "temporal": loops.temporal,
        }[kind]
        iterator = module.coordinate_batches

        def observe(*args: Any, **kwargs: Any) -> Iterator[Any]:
            for batch in iterator(*args, **kwargs):
                batches.append(snapshot_tree(batch))
                yield batch

        with patch.object(module, "coordinate_batches", side_effect=observe):
            yield batches


def exercise_mdd_resume(
    family: ModelFamily, root: Path, *, fault: str | None = None,
    epochs: int = 3, mutate: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Compare a committed prefix and subsequent updates across a real restart."""
    assert epochs >= 2
    pl.seed_everything(219, workers=True)
    config = family.compose(root, False)
    config.model.dropout = 0.1
    model, inputs = family.build(config)
    kind = inputs.kind
    run_loop = {
        "coordinate": loops._scratch,
        "pretrain": loops._pretrain,
        "deep_query": loops._deep_query,
        "hybrid": loops._hybrid,
        "temporal": loops._temporal,
    }[kind]

    def run(
        output: Path, *, stop: bool = False, resume: Path | None = None
    ) -> list[Any]:
        fresh = copy.deepcopy(model)
        with _batches(kind) as batches:
            if stop and kind == "coordinate":
                save = loops.scratch.save_coordinate_checkpoint

                def save_and_interrupt(*args: Any, **kwargs: Any) -> None:
                    save(*args, **kwargs)
                    raise _Interrupted

                with (
                    patch.object(
                        loops.scratch,
                        "save_coordinate_checkpoint",
                        side_effect=save_and_interrupt,
                    ),
                    suppress(_Interrupted),
                ):
                    run_loop(fresh, inputs, output, epochs=epochs)
            else:
                run_loop(
                    fresh,
                    inputs,
                    output,
                    epochs=epochs,
                    stop=0 if stop else None,
                    resume=resume,
                )
        return batches

    continuous = root / "continuous"
    interrupted = root / "interrupted"
    expected_batches = run(continuous)
    first_batches = run(interrupted, stop=True)
    assert not (interrupted / "COMPLETED.json").exists(), "prefix reported completion"
    checkpoint = _artifact(interrupted, kind, 1)
    assert_tree_equal(
        _state(_artifact(continuous, kind, 1), kind), _state(checkpoint, kind), "prefix"
    )
    assert_tree_equal(
        expected_batches[: len(first_batches)], first_batches, "prefix.batch"
    )
    if mutate is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        mutate(payload)
        torch.save(payload, checkpoint)
    pl.seed_everything(731, workers=True)
    if fault is None:
        remaining = run(interrupted, resume=checkpoint)
    elif fault == "optimizer":
        with patch.object(torch.optim.Optimizer, "load_state_dict", return_value=None):
            remaining = run(interrupted, resume=checkpoint)
    elif fault == "rng":
        with patch("torch.set_rng_state", return_value=None):
            remaining = run(interrupted, resume=checkpoint)
    elif fault == "data":
        assert kind == "temporal", (
            "only shared-pack temporal training has a within-stream pack cursor"
        )
        sampler = loops.temporal.PackSampler

        def reset_position(*args: Any, **kwargs: Any) -> Any:
            kwargs["position"] = 0
            return sampler(*args, **kwargs)

        with patch.object(loops.temporal, "PackSampler", side_effect=reset_position):
            remaining = run(interrupted, resume=checkpoint)
    else:
        raise ValueError(f"Unknown resume fault {fault}")
    assert_tree_equal(expected_batches[len(first_batches) :], remaining, "resume.batch")
    for completed in range(2, epochs + 1):
        assert_tree_equal(
            _state(_artifact(continuous, kind, completed), kind),
            _state(_artifact(interrupted, kind, completed), kind),
            f"resume[{completed}]",
        )
