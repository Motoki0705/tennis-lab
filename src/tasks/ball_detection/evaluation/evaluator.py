"""Execute one checkpoint/dataset/split evaluation job."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, SequentialSampler

from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.tasks.ball_detection.evaluation.adapters import (
    BallPredictionAdapter,
    LightningBallPredictionAdapter,
)
from src.tasks.ball_detection.evaluation.configuration import (
    build_evaluation_config,
    read_checkpoint_config,
    validate_checkpoint_model_name,
)
from src.tasks.ball_detection.evaluation.contracts import (
    DatasetSpec,
    EvaluationManifest,
    ModelSpec,
)
from src.tasks.ball_detection.evaluation.dataset_provenance import (
    SequentialSourceResolver,
    build_split_provenance,
    sha256_file,
)
from src.tasks.ball_detection.evaluation.metrics import StratifiedBallMetrics


class JobEvaluator(Protocol):
    """Injectable interface used by the resumable pipeline."""

    def evaluate(
        self,
        *,
        model: ModelSpec,
        dataset: DatasetSpec,
        split: str,
        manifest: EvaluationManifest,
    ) -> dict[str, Any]:
        """Evaluate one job and return a JSON-compatible payload."""
        ...


class DefaultJobEvaluator:
    """Load registered Lightning models and evaluate fixed val/test splits."""

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self._checkpoint_configs: dict[Path, DictConfig] = {}
        self._checkpoint_hashes: dict[Path, str] = {}
        self._adapter_key: tuple[Path, bool, bool] | None = None
        self._adapter: LightningBallPredictionAdapter | None = None

    def evaluate(
        self,
        *,
        model: ModelSpec,
        dataset: DatasetSpec,
        split: str,
        manifest: EvaluationManifest,
    ) -> dict[str, Any]:
        checkpoint_config = self._checkpoint_config(model.checkpoint)
        validate_checkpoint_model_name(
            checkpoint_config,
            expected_model_name=model.expected_model_name,
        )
        evaluation_config = build_evaluation_config(
            checkpoint_config=checkpoint_config,
            dataset_spec=dataset,
            metrics_spec=manifest.metrics,
            resolver=manifest.resolver,
        )
        datamodule = build_ball_detection_datamodule(evaluation_config)
        dataloader = build_evaluation_dataloader(datamodule, split)
        if not isinstance(dataloader.sampler, SequentialSampler):
            raise RuntimeError(
                "Evaluation dataloaders must use sequential sampling to preserve "
                "source provenance."
            )

        adapter = self._prediction_adapter(model)
        split_result = evaluate_dataloader(
            adapter=adapter,
            dataloader=dataloader,
            data_config=evaluation_config.data,
            split=split,
            manifest=manifest,
        )
        checkpoint_hash = self._checkpoint_hash(model.checkpoint)
        split_result["provenance"] = {
            "command": " ".join(sys.argv),
            "git_revision": _git_revision(),
            "git_dirty": _git_dirty(),
            "checkpoint_path": str(model.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "model_name": model.expected_model_name,
            "resolved_config": OmegaConf.to_container(
                evaluation_config,
                resolve=True,
            ),
            "dataset": split_result.pop("dataset_provenance"),
        }
        return split_result

    def _checkpoint_config(self, path: Path) -> DictConfig:
        if path not in self._checkpoint_configs:
            self._checkpoint_configs[path] = read_checkpoint_config(path)
        return self._checkpoint_configs[path]

    def _checkpoint_hash(self, path: Path) -> str:
        if path not in self._checkpoint_hashes:
            self._checkpoint_hashes[path] = sha256_file(path)
        return self._checkpoint_hashes[path]

    def _prediction_adapter(
        self,
        model: ModelSpec,
    ) -> LightningBallPredictionAdapter:
        key = (model.checkpoint, model.strict, model.weights_only)
        if self._adapter is None or self._adapter_key != key:
            self._adapter = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._adapter = LightningBallPredictionAdapter.load(
                model.checkpoint,
                device=self.device,
                strict=model.strict,
                weights_only=model.weights_only,
            )
            self._adapter_key = key
        return self._adapter


def build_evaluation_dataloader(
    datamodule: pl.LightningDataModule,
    split: str,
) -> DataLoader[Any]:
    """Construct only the requested val/test dataset; never initialize train."""
    if split == "val":
        datamodule.setup(stage="validate")
        loader = datamodule.val_dataloader()
        if not isinstance(loader, DataLoader):
            raise TypeError("Validation dataloader must be a DataLoader.")
        return loader
    if split == "test":
        datamodule.setup(stage="test")
        loader = datamodule.test_dataloader()
        if not isinstance(loader, DataLoader):
            raise TypeError("Test dataloader must be a DataLoader.")
        return loader
    raise ValueError(f"Evaluation split must be val or test, got {split!r}.")


def evaluate_dataloader(
    *,
    adapter: BallPredictionAdapter,
    dataloader: DataLoader[Any],
    data_config: Any,
    split: str,
    manifest: EvaluationManifest,
) -> dict[str, Any]:
    """Evaluate one sequential dataloader with metrics and inference timing."""
    dataset = dataloader.dataset
    source_resolver = SequentialSourceResolver(
        dataset,
        default_source=str(data_config.source),
    )
    metrics = StratifiedBallMetrics(manifest.metrics)
    timings: list[float] = []
    processed_frames = 0
    processed_batches = 0
    max_batches = manifest.performance.max_batches_per_split

    iterator = iter(dataloader)
    try:
        first_batch = next(iterator)
    except StopIteration as error:
        raise RuntimeError(f"Evaluation split {split!r} is empty.") from error

    with torch.inference_mode():
        for _ in range(manifest.performance.warmup_batches):
            _predict_batch(adapter, first_batch)
        _reset_peak_memory(adapter.device)

        for batch_index, batch in enumerate(_prepend(first_batch, iterator)):
            if max_batches is not None and batch_index >= max_batches:
                break
            _synchronize(adapter.device)
            start = time.perf_counter()
            pred_heatmaps = _predict_batch(adapter, batch)
            _synchronize(adapter.device)
            timings.append(time.perf_counter() - start)

            target_coords = _tensor(batch, "coords").to(
                adapter.device,
                non_blocking=True,
            )
            target_visibility = _tensor(batch, "visibility").to(
                adapter.device,
                non_blocking=True,
            )
            original_size = _tensor(batch, "original_size").to(
                adapter.device,
                non_blocking=True,
            )
            sources = source_resolver.next(pred_heatmaps.shape[0])
            metrics.update(
                pred_heatmaps,
                target_coords,
                target_visibility,
                original_size,
                sources=sources,
            )
            processed_frames += int(pred_heatmaps.shape[0] * pred_heatmaps.shape[1])
            processed_batches += 1

    total_seconds = sum(timings)
    performance = {
        "batches": processed_batches,
        "frames": processed_frames,
        "total_inference_seconds": total_seconds,
        "latency_ms_per_batch": (
            None if not timings else 1000.0 * total_seconds / len(timings)
        ),
        "throughput_frames_per_second": (
            None if total_seconds <= 0 else processed_frames / total_seconds
        ),
        "peak_vram_mb": _peak_vram_mb(adapter.device),
        "warmup_batches": manifest.performance.warmup_batches,
    }
    return {
        "split": split,
        "metrics": metrics.compute(),
        "performance": performance,
        "dataset_provenance": build_split_provenance(
            data_config=data_config,
            split=split,
            dataset=dataset,
            resolver=manifest.resolver,
        ),
    }


def _predict_batch(
    adapter: BallPredictionAdapter,
    batch: dict[str, Any],
) -> Tensor:
    images = _tensor(batch, "images")
    target_heatmaps = _tensor(batch, "heatmaps")
    return adapter.predict_heatmaps(
        images,
        target_size_hw=(
            int(target_heatmaps.shape[-2]),
            int(target_heatmaps.shape[-1]),
        ),
    )


def _tensor(batch: dict[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Evaluation batch field {key!r} must be a Tensor.")
    return value


def _prepend(
    first: dict[str, Any],
    iterator: Any,
) -> Any:
    yield first
    yield from iterator


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


__all__ = [
    "DefaultJobEvaluator",
    "JobEvaluator",
    "build_evaluation_dataloader",
    "evaluate_dataloader",
]
