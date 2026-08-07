"""Cross-task ownership tests for the canonical tracking lifecycle."""

from __future__ import annotations

from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
)
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)

_SHARED_LIFECYCLE_METHODS = (
    "training_step",
    "validation_step",
    "test_step",
    "on_test_epoch_end",
)


def test_tracking_tasks_inherit_one_shared_lightning_lifecycle() -> None:
    for task_module in (
        BLCSTrackingLightningModule,
        PLCSTrackingLightningModule,
    ):
        assert issubclass(task_module, TrackingLightningModule)
        for method_name in _SHARED_LIFECYCLE_METHODS:
            assert method_name not in task_module.__dict__
            assert getattr(task_module, method_name) is getattr(
                TrackingLightningModule, method_name
            )


def test_tracking_tasks_keep_only_task_specific_step_and_payload_hooks() -> None:
    for task_module in (
        BLCSTrackingLightningModule,
        PLCSTrackingLightningModule,
    ):
        assert "compute_tracking_step" in task_module.__dict__
        assert "tracking_prediction_result" in task_module.__dict__
        assert "test_prediction_payload" in task_module.__dict__
