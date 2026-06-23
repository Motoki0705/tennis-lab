"""Contracts for manifest-driven ball-detector evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.tasks.ball_detection.evaluation.adapters import LightningBallPredictionAdapter
from src.tasks.ball_detection.evaluation.configuration import (
    build_evaluation_config,
    read_checkpoint_config,
    validate_checkpoint_model_name,
)
from src.tasks.ball_detection.evaluation.contracts import (
    DatasetSpec,
    EvaluationManifest,
    MetricsSpec,
    ModelSpec,
    PerformanceSpec,
    load_evaluation_manifest,
)
from src.tasks.ball_detection.evaluation.evaluator import (
    build_evaluation_dataloader,
    evaluate_dataloader,
)
from src.tasks.ball_detection.evaluation.metrics import StratifiedBallMetrics
from src.tasks.ball_detection.evaluation.runner import EvaluationPipeline


def _write_manifest(
    path: Path,
    *,
    checkpoint: Path,
    splits: str = "[val, test]",
    data_config: str = "rgb_sequence",
) -> None:
    path.write_text(
        f"""
schema: ball_detection_evaluation_manifest_v1
output_dir: {path.parent / "output"}
device: cpu
resume: true
fail_fast: false
metrics:
  peak_threshold: 0.5
  ball_distance_threshold: 2.0
  nms_kernel: 3
  max_predictions_per_frame: 2
performance:
  warmup_batches: 0
  max_batches_per_split: 1
datasets:
  tracknet:
    config: {data_config}
    splits: {splits}
    overrides:
      num_workers: 0
models:
  - id: baseline
    category: architecture-controlled
    checkpoint: {checkpoint}
    expected_model_name: conv_next_unet
    datasets: [tracknet]
""",
        encoding="utf-8",
    )


def _success_payload() -> dict[str, Any]:
    return {
        "metrics": {
            "aggregate": {
                "precision": 0.8,
                "recall": 0.75,
                "f1": 0.774,
                "mean_distance_px": 1.25,
                "negative_frame_fpr": 0.1,
            },
            "by_source": {
                "tracknet": {
                    "precision": 0.8,
                    "recall": 0.75,
                    "f1": 0.774,
                    "mean_distance_px": 1.25,
                    "negative_frame_fpr": 0.1,
                }
            },
        },
        "performance": {
            "throughput_frames_per_second": 100.0,
            "latency_ms_per_batch": 10.0,
            "peak_vram_mb": None,
        },
        "provenance": {"git_revision": "test"},
    }


class _RecordingEvaluator:
    def __init__(self, *, fail_once_split: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_once_split = fail_once_split
        self.failed = False

    def evaluate(
        self,
        *,
        model: ModelSpec,
        dataset: DatasetSpec,
        split: str,
        manifest: EvaluationManifest,
    ) -> dict[str, Any]:
        del model, dataset, manifest
        self.calls.append(split)
        if split == self.fail_once_split and not self.failed:
            self.failed = True
            raise RuntimeError("synthetic failure")
        return _success_payload()


def test_manifest_forbids_train_split(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, checkpoint=checkpoint, splits="[train]")
    with pytest.raises(ValueError, match="forbidden splits"):
        load_evaluation_manifest(manifest_path)


def test_pipeline_reuses_success_and_reruns_only_failure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    val_split = tmp_path / "val.txt"
    test_split = tmp_path / "test.txt"
    val_split.write_text("game-val\n", encoding="utf-8")
    test_split.write_text("game-test\n", encoding="utf-8")
    data_config = tmp_path / "evaluation_data.yaml"
    data_config.write_text(
        "\n".join(
            [
                "source: tracknet",
                f"data_dir: {tmp_path}",
                "batch_size: 1",
                "split:",
                f"  val_file: {val_split}",
                f"  test_file: {test_split}",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        checkpoint=checkpoint,
        data_config=str(data_config),
    )
    manifest = load_evaluation_manifest(manifest_path)
    evaluator = _RecordingEvaluator(fail_once_split="val")

    first = EvaluationPipeline(manifest, evaluator=evaluator).run()
    assert first == {"jobs": 2, "executed": 2, "reused": 0, "failed": 1}
    assert evaluator.calls == ["val", "test"]

    second = EvaluationPipeline(manifest, evaluator=evaluator).run()
    assert second == {"jobs": 2, "executed": 1, "reused": 1, "failed": 0}
    assert evaluator.calls == ["val", "test", "val"]
    comparison = (manifest.output_dir / "comparison.md").read_text(
        encoding="utf-8"
    )
    assert "## Architecture-controlled" in comparison
    assert "## Full strategy" in comparison

    data_config.write_text(
        data_config.read_text(encoding="utf-8") + "\npin_memory: false\n",
        encoding="utf-8",
    )
    third = EvaluationPipeline(manifest, evaluator=evaluator).run()
    assert third == {"jobs": 2, "executed": 2, "reused": 0, "failed": 0}
    assert evaluator.calls == ["val", "test", "val", "val", "test"]


def test_stratified_metrics_include_negative_fpr_and_sources() -> None:
    accumulator = StratifiedBallMetrics(
        MetricsSpec(
            peak_threshold=0.5,
            ball_distance_threshold=2.0,
            nms_kernel=3,
            max_predictions_per_frame=2,
        )
    )
    pred_heatmaps = torch.zeros(2, 1, 8, 8)
    pred_heatmaps[0, 0, 3, 3] = 1.0
    pred_heatmaps[1, 0, 5, 5] = 1.0
    coords = torch.zeros(2, 1, 1, 2)
    coords[0, 0, 0] = torch.tensor([3.0, 3.0])
    visibility = torch.zeros(2, 1, 1)
    visibility[0, 0, 0] = 1.0
    original_size = torch.tensor([[8.0, 8.0], [8.0, 8.0]])

    accumulator.update(
        pred_heatmaps,
        coords,
        visibility,
        original_size,
        sources=["source-a", "source-b"],
    )
    result = accumulator.compute()
    aggregate = result["aggregate"]
    by_source = result["by_source"]
    assert isinstance(aggregate, dict)
    assert isinstance(by_source, dict)
    assert aggregate["negative_frame_fpr"] == 1.0
    assert by_source["source-b"]["negative_frame_fpr"] == 1.0
    assert set(by_source) == {"source-a", "source-b"}


def test_checkpoint_model_name_mismatch_fails_early(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save(
        {
            "hyper_parameters": {
                "config": {"model": {"name": "conv_next_unet"}}
            }
        },
        checkpoint_path,
    )
    checkpoint_config = read_checkpoint_config(checkpoint_path)
    with pytest.raises(ValueError, match="Checkpoint/model config mismatch"):
        validate_checkpoint_model_name(
            checkpoint_config,
            expected_model_name="dinov3_rope",
        )


def test_evaluation_config_keeps_checkpoint_model_and_replaces_dataset() -> None:
    checkpoint_config = {
        "model": {
            "name": "conv_next_unet",
            "num_frames": 8,
            "input_mode": "mdd",
            "input_layout": "bcthw",
        },
        "training": {"matmul_precision": "high"},
    }
    config = build_evaluation_config(
        checkpoint_config=OmegaConf.create(checkpoint_config),
        dataset_spec=DatasetSpec(
            id="web",
            config="web_frames",
            splits=("test",),
            overrides={
                "num_workers": 0,
                "sampling": {"mode": "temporal"},
            },
        ),
        metrics_spec=MetricsSpec(),
    )
    assert config.model.name == "conv_next_unet"
    assert config.model.num_frames == 8
    assert config.data.source == "web"
    assert config.data.num_workers == 0
    assert config.data.sampling.mode == "temporal"
    assert config.data.sampling.temporal.frame_step == 1
    assert config.data.augmentation.normalize_imagenet.enabled is True


def test_prediction_adapter_rejects_wrong_output_shape() -> None:
    class WrongShapeModel(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs[:, :1, 0]

    class FakeModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = WrongShapeModel()
            self.config = {
                "model": {
                    "input_mode": "rgb",
                    "input_layout": "bcthw",
                }
            }

    adapter = LightningBallPredictionAdapter(FakeModule(), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="output shape mismatch"):
        adapter.predict_heatmaps(
            torch.zeros(1, 2, 3, 8, 8),
            target_size_hw=(8, 8),
        )


def test_validation_loader_does_not_initialize_train_dataset() -> None:
    class RecordingDataModule:
        def __init__(self) -> None:
            self.stage: str | None = None
            self.dataset = TensorDataset(torch.zeros(2, 1))

        def setup(self, stage: str | None = None) -> None:
            self.stage = stage

        def val_dataloader(self) -> DataLoader[Any]:
            return DataLoader(self.dataset, batch_size=1, shuffle=False)

    datamodule = RecordingDataModule()
    dataloader = build_evaluation_dataloader(datamodule, "val")
    assert datamodule.stage == "validate"
    assert isinstance(dataloader.sampler, torch.utils.data.SequentialSampler)


def test_evaluate_dataloader_records_metrics_performance_and_split(
    tmp_path: Path,
) -> None:
    class DictDataset(Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
            return {
                "images": torch.zeros(1, 3, 8, 8),
                "heatmaps": torch.zeros(1, 8, 8),
                "coords": torch.tensor([[[3.0, 3.0]]]),
                "visibility": torch.ones(1, 1),
                "original_size": torch.tensor([8.0, 8.0]),
                "heatmap_size": torch.tensor([8.0, 8.0]),
            }

    class FakeAdapter:
        device = torch.device("cpu")

        def predict_heatmaps(
            self,
            images: torch.Tensor,
            *,
            target_size_hw: tuple[int, int],
        ) -> torch.Tensor:
            heatmaps = torch.zeros(
                images.shape[0],
                images.shape[1],
                *target_size_hw,
            )
            heatmaps[:, :, 3, 3] = 1.0
            return heatmaps

    split_file = tmp_path / "test.txt"
    split_file.write_text("game1\n", encoding="utf-8")
    dataloader = DataLoader(DictDataset(), batch_size=1, shuffle=False)
    manifest = EvaluationManifest(
        schema="ball_detection_evaluation_manifest_v1",
        output_dir=tmp_path / "output",
        device="cpu",
        resume=True,
        fail_fast=False,
        metrics=MetricsSpec(
            peak_threshold=0.5,
            ball_distance_threshold=2.0,
            nms_kernel=3,
            max_predictions_per_frame=2,
        ),
        performance=PerformanceSpec(
            warmup_batches=1,
            max_batches_per_split=None,
        ),
        datasets={},
        models=(),
    )
    result = evaluate_dataloader(
        adapter=FakeAdapter(),
        dataloader=dataloader,
        data_config={
            "source": "tracknet",
            "data_dir": str(tmp_path),
            "split": {"test_file": str(split_file)},
        },
        split="test",
        manifest=manifest,
    )

    assert result["metrics"]["aggregate"]["f1"] == 1.0
    assert result["metrics"]["by_source"]["tracknet"]["f1"] == 1.0
    assert result["performance"]["frames"] == 1
    assert result["dataset_provenance"]["split"] == "test"
