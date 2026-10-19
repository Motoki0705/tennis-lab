"""Validated, measured evidence for the two completed ball learning stages."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.visualization.presentation import CurveSeries
from src.utils.checksum import dual_sha256


@dataclass(frozen=True)
class StageRun:
    metrics: tuple[dict[str, Any], ...]
    training: tuple[dict[str, Any], ...]
    recipe: dict[str, Any]
    best: dict[str, Any]
    common: CurveSeries
    loss: CurveSeries
    learning_rate: CurveSeries
    encoder_learning_rate: CurveSeries | None
    encoder_frozen: NDArray[np.bool_]
    offset: int

    @property
    def steps(self) -> int:
        return int(self.learning_rate.steps[-1]) - self.offset

    @property
    def best_point(self) -> tuple[int, float, str]:
        return (
            self.offset + self.best["global_step"],
            self.best["selection_error_px"],
            "best",
        )

    def frozen_at(self, local_step: float) -> bool:
        # A summary at step N describes the updates since the previous summary,
        # not the updates after N. Thus epoch 0's last update remains frozen.
        index = int(np.searchsorted(self.learning_rate.steps - self.offset, local_step))
        return bool(self.encoder_frozen[min(index, len(self.encoder_frozen) - 1)])


@dataclass(frozen=True)
class StageLogs:
    pretraining: StageRun
    posttraining: StageRun
    input_files: dict[str, str]

    @property
    def parent_step(self) -> int:
        return int(self.posttraining.recipe["parent"]["global_step"])


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty JSONL objects: {path}")
    return rows


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _series(
    rows: tuple[dict[str, Any], ...], key: str, offset: int, name: str
) -> CurveSeries:
    return CurveSeries(
        name,
        np.array(
            [_integer(row["global_step"], "global_step") + offset for row in rows],
            dtype=np.int64,
        ),
        np.array([_number(row[key], key) for row in rows], dtype=np.float64),
        "line",
    )


def trailing_mean(values: NDArray[np.float64], window: int = 20) -> NDArray[np.float64]:
    """Causal moving mean; no future loss samples leak ahead of the cursor."""
    totals = np.concatenate(([0.0], np.cumsum(values)))
    stops = np.arange(1, len(values) + 1)
    starts = np.maximum(0, stops - window)
    return np.asarray(
        (totals[stops] - totals[starts]) / (stops - starts), dtype=np.float64
    )


def _load_run(path: Path, *, schema: str, offset: int, posttraining: bool) -> StageRun:
    metrics, training = (
        _rows(path / "metrics.jsonl"),
        _rows(path / "train-summary.jsonl"),
    )
    recipe, best = _object(path / "config.json")["recipe"], _object(path / "best.json")
    if (
        any(row.get("schema") != schema for row in metrics)
        or best.get("schema") != schema
    ):
        raise ValueError(f"Unexpected evaluation schema in {path}; expected {schema}")
    if posttraining and any(row["profile"] != "clean" for row in metrics):
        raise ValueError(f"Only clean Common evaluation is supported: {path}")
    for row in metrics:
        for scope in ("common", "full"):
            _number(row["scopes"][scope]["macro_mean_error_px"], f"{scope} error")
    common = _series(
        tuple(
            dict(
                global_step=row["global_step"],
                error=row["scopes"]["common"]["macro_mean_error_px"],
            )
            for row in metrics
        ),
        "error",
        offset,
        "Common",
    )
    loss = _series(
        training, "train_loss", offset, "heatmap" if not posttraining else "座標"
    )
    smoothed = trailing_mean(loss.values)
    scale = float(smoothed.max())
    loss = CurveSeries(
        loss.name, loss.steps, smoothed / scale if scale else smoothed, "line"
    )
    lr = _series(training, "learning_rate", offset, "head")
    if lr.steps[0] <= offset or lr.steps[-1] != common.steps[-1]:
        raise ValueError(f"Training/evaluation step ranges disagree: {path}")
    epochs = _integer(recipe["epochs"], "epochs")
    if len(metrics) != epochs or [row["epoch"] for row in metrics] != list(
        range(epochs)
    ):
        raise ValueError(f"Expected one evaluation at the end of each epoch: {path}")
    for row in training:
        epoch = _integer(row["epoch"], "epoch")
        if epoch >= epochs or not (
            (0 if epoch == 0 else common.steps[epoch - 1] - offset)
            < row["global_step"]
            <= common.steps[epoch] - offset
        ):
            raise ValueError(
                f"Training summaries disagree with evaluation epochs: {path}"
            )
    best_step = _integer(best["global_step"], "best.global_step")
    best_error = _number(best["selection_error_px"], "best.selection_error_px")
    matches = common.values[common.steps == best_step + offset]
    if (
        recipe["selection_scope"] != "common"
        or len(matches) != 1
        or not math.isclose(float(matches[0]), best_error)
        or not math.isclose(best_error, float(common.values.min()))
        or not isinstance(best["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", best["sha256"]) is None
    ):
        raise ValueError(
            f"Best checkpoint does not match measured Common metrics: {path}"
        )
    frozen: NDArray[np.bool_] = np.zeros(len(training), dtype=np.bool_)
    encoder_lr = None
    if posttraining:
        if not isinstance(recipe["augmentation"], dict) or not recipe["augmentation"]:
            raise ValueError(
                "Posttraining augmentation must be an explicit nonempty mapping"
            )
        freeze_epochs = _integer(recipe["freeze_epochs"], "freeze_epochs")
        ratio = _number(recipe["encoder_lr_ratio"], "encoder_lr_ratio")
        if freeze_epochs >= epochs or not 0 < ratio <= 1:
            raise ValueError(
                "Posttraining needs a finite freeze period and encoder LR ratio in (0, 1]"
            )
        for row in training:
            if type(row["encoder_frozen"]) is not bool or row["encoder_frozen"] != (
                row["epoch"] < freeze_epochs
            ):
                raise ValueError("encoder_frozen disagrees with recipe.freeze_epochs")
        frozen = np.array([row["encoder_frozen"] for row in training], dtype=np.bool_)
        encoder_lr = _series(training, "encoder_learning_rate", offset, "CNN")
        if not np.allclose(
            encoder_lr.values, np.where(frozen, 0, lr.values * ratio), rtol=1e-9, atol=0
        ):
            raise ValueError(
                "encoder_learning_rate disagrees with encoder_frozen/LR ratio"
            )
    return StageRun(
        metrics, training, recipe, best, common, loss, lr, encoder_lr, frozen, offset
    )


def load_stage_logs(pretraining: Path, posttraining: Path) -> StageLogs:
    """Read both bundles and reject mismatched lineage or evaluation contracts.

    The checkpoint hashes are recorded evidence, not a request to open weights.
    Hash all consumed files before/after reading so a changing bundle cannot
    produce a manifest for evidence different from the plotted values.
    """
    files = [
        path / name
        for path in (pretraining, posttraining)
        for name in ("metrics.jsonl", "train-summary.jsonl", "config.json", "best.json")
    ]
    hashes = {str(path): dual_sha256(path) for path in files}
    first = _load_run(
        pretraining,
        schema="mdd_heatmap_pretraining_evaluation.v1",
        offset=0,
        posttraining=False,
    )
    second = _load_run(
        posttraining,
        schema="mdd_coordinate_evaluation.v1",
        offset=first.steps,
        posttraining=True,
    )
    parent = second.recipe["parent"]
    if parent["sha256"] != first.best["sha256"]:
        raise ValueError(
            "Phase 2 parent SHA-256 does not match Phase 1 best checkpoint"
        )
    if parent["global_step"] != first.best["global_step"]:
        raise ValueError(
            "Phase 2 parent global_step does not match Phase 1 best checkpoint"
        )
    if first.recipe["manifest_sha256"] != second.recipe["manifest_sha256"]:
        raise ValueError("The Common curves must use the same fixed manifest")
    if first.recipe["input_contract"] != parent["input_contract"]:
        raise ValueError("Phase 2 parent input_contract does not match Phase 1")
    if hashes != {str(path): dual_sha256(path) for path in files}:
        raise ValueError("Stage evidence changed while reading")
    return StageLogs(first, second, hashes)
