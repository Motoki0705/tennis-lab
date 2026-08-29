"""Integration coverage for sequence-weighted tracking metric aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.tasks.base.training.metric_logging import (
    ScalarMetricStatistic,
    evaluation_only_metric_logging_contract,
)
from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
    TrackingStepResult,
)
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig

_COUNT_KEYS = (
    "query_reuse_count",
    "illegal_overlap_count",
    "id_switches",
    "duplicate_active_tracks",
    "missed_gt_frames",
    "inactive_query_false_positives",
)


class _SequenceCountDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, counts: Tensor) -> None:
        self.counts = counts

    def __len__(self) -> int:
        return int(self.counts.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "count": self.counts[index],
            "target_position": torch.zeros(1, 1, 3),
        }


class _AggregationModule(TrackingLightningModule[Tensor]):
    metric_logging_contract = evaluation_only_metric_logging_contract(
        "aggregation fixture",
        headline_keys=("id_switches",),
        progress_bar_keys=(),
    )

    def compute_tracking_step(
        self,
        batch: dict[str, Tensor],
        *,
        compute_metrics: bool,
    ) -> TrackingStepResult[Tensor]:
        per_batch_sequence_mean = batch["count"].float().mean()
        denominator = per_batch_sequence_mean.new_tensor(
            float(batch["count"].shape[0])
        )
        numerator = batch["count"].float().sum()
        metrics = (
            {name: per_batch_sequence_mean for name in _COUNT_KEYS}
            if compute_metrics
            else {}
        )
        return TrackingStepResult(
            losses={
                "total": per_batch_sequence_mean * 0.0,
                "auxiliary": per_batch_sequence_mean,
            },
            metrics=metrics,
            prediction=batch["target_position"],
            statistics=(
                {
                    name: ScalarMetricStatistic(numerator, denominator)
                    for name in _COUNT_KEYS
                }
                if compute_metrics
                else None
            ),
        )

    def tracking_prediction_result(self, prediction: Tensor) -> dict[str, Any]:
        return {"prediction": prediction}


def _tracking_config() -> Any:
    config_dir = Path("src/tasks/plcs/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="train_tracking")


@pytest.mark.parametrize("task", ["blcs", "plcs"])
@pytest.mark.parametrize("config_name", ["train_tracking", "train_tracking_chunked"])
def test_tracking_configs_share_the_same_strict_metric_contract(
    task: str,
    config_name: str,
) -> None:
    config_dir = Path(f"src/tasks/{task}/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name=config_name)

    assert set(config.tracking_metrics) == {
        "presence_threshold",
        "duplicate_distance",
        "id_switch_distance",
    }
    assert TrackingMetricConfig.from_mapping(config.tracking_metrics) == (
        TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
            id_switch_distance=0.05,
        )
    )


@pytest.mark.parametrize("batch_size", [1, 2], ids=["unit", "uneven-final-batch"])
def test_epoch_count_metrics_are_invariant_to_batch_partition(
    tmp_path: Path,
    batch_size: int,
) -> None:
    counts = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    module = _AggregationModule(_tracking_config())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )

    results = trainer.validate(
        module,
        dataloaders=DataLoader(_SequenceCountDataset(counts), batch_size=batch_size),
        verbose=False,
    )

    expected = counts.mean().item()
    assert len(results) == 1
    assert results[0]["val/id_switches"] == pytest.approx(expected)
    assert set(results[0]) == {"val/loss", "val/id_switches"}


def test_test_epoch_logs_only_headline_and_saves_full_disjoint_artifacts(
    monkeypatch: Any,
) -> None:
    module = _AggregationModule(_tracking_config())
    captured: dict[str, dict[str, float]] = {}
    logged: list[str] = []
    monkeypatch.setattr(
        module,
        "log",
        lambda name, value, **kwargs: logged.append(name),
    )
    monkeypatch.setattr(module, "_reset_test_prediction_buffer", lambda: None)
    monkeypatch.setattr(module, "collect_test_predictions", lambda *args: None)
    monkeypatch.setattr(
        module,
        "save_test_predictions",
        lambda **kwargs: captured.update(kwargs),
    )
    module.on_test_epoch_start()

    module.test_step(
        {
            "count": torch.tensor([1.0, 3.0]),
            "target_position": torch.zeros(2, 1, 1, 3),
        },
        0,
    )
    module.test_step(
        {
            "count": torch.tensor([8.0]),
            "target_position": torch.zeros(1, 1, 1, 3),
        },
        1,
    )
    assert set(logged) == {"test/loss"}
    module.on_test_epoch_end()

    assert set(captured["metrics"]) == {"id_switches"}
    assert captured["metrics"]["id_switches"] == pytest.approx(4.0)
    assert set(captured["diagnostic_metrics"]) == (
        set(_COUNT_KEYS) - {"id_switches"}
    ) | {"loss_auxiliary"}
    assert captured["diagnostic_metrics"]["loss_auxiliary"] == pytest.approx(4.0)
    assert set(captured["metrics"]).isdisjoint(captured["diagnostic_metrics"])
    assert logged.count("test/id_switches") == 1
    assert set(logged) == {"test/loss", "test/id_switches"}
