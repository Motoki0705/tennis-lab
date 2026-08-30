"""Unit tests for PLCS Lightning persistence payloads."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from src.tasks.base.model_io import ModelCall
from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.plcs.model_io import PLCSDecodedPrediction, PLCSPreparedBatch
from src.tasks.plcs.training.lightning_module import (
    PLCS_TRAJECTORY_METRIC_CONTRACT,
    PLCSLightningModule,
    build_plcs_metric_logging_contract,
)
from src.tasks.plcs.training.metrics import (
    CANONICAL_POSE_DIAGNOSTIC_KEYS,
    CANONICAL_POSE_HEADLINE_KEYS,
    PLCSMetrics,
)
from src.utils.geometry.court_pose import canonical_pose_to_world_pose


def test_canonical_test_payload_persists_prediction_and_physical_target() -> None:
    position = torch.tensor([[[0.25, -0.10, 0.20], [-0.15, 0.05, 0.30]]])
    rotation = torch.tensor([[[0.0, 1.0], [0.6, 0.8]]])
    canonical_target = torch.randn(1, 2, 17, 3)
    world_pose = canonical_pose_to_world_pose(
        canonical_target,
        position,
        rotation,
    )
    prediction = torch.randn_like(world_pose)
    prepared = PLCSPreparedBatch(
        call=ModelCall(),
        target_position=position,
        target_rotation=rotation,
        target_human_kp_3d=world_pose,
    )
    result = {
        "outputs": PLCSDecodedPrediction(
            position=position,
            rotation=rotation,
            canonical_pose=prediction,
        ),
        "prepared": prepared,
    }
    module = cast("PLCSLightningModule", object())

    payload = PLCSLightningModule.test_prediction_payload(module, {}, result)

    torch.testing.assert_close(payload["pred_canonical_pose"], prediction)
    torch.testing.assert_close(payload["target_canonical_pose"], canonical_target)


def test_canonical_pose_metric_inputs_use_physical_world_target() -> None:
    position = torch.tensor([[[0.25, -0.10, 0.20], [-0.15, 0.05, 0.30]]])
    rotation = torch.tensor([[[0.0, 1.0], [0.6, 0.8]]])
    canonical_target = torch.randn(1, 2, 17, 3)
    world_pose = canonical_pose_to_world_pose(canonical_target, position, rotation)
    prediction = torch.randn_like(canonical_target)
    prepared = PLCSPreparedBatch(
        call=ModelCall(),
        target_position=position,
        target_rotation=rotation,
        target_human_kp_3d=world_pose,
    )

    class _Module:
        io_adapter = type("_Adapter", (), {"predict_canonical_pose": True})()

    pred_pose, target_pose = PLCSLightningModule._canonical_pose_metric_inputs(
        cast("PLCSLightningModule", _Module()),
        PLCSDecodedPrediction(
            position=position,
            rotation=rotation,
            canonical_pose=prediction,
        ),
        prepared,
        position,
        rotation,
    )

    torch.testing.assert_close(pred_pose, prediction)
    torch.testing.assert_close(target_pose, canonical_target)


def test_canonical_pose_metric_inputs_fail_when_output_or_target_is_missing() -> None:
    position = torch.zeros(1, 1, 3)
    rotation = torch.tensor([[[1.0, 0.0]]])

    class _Module:
        io_adapter = type("_Adapter", (), {"predict_canonical_pose": True})()

    module = cast("PLCSLightningModule", _Module())
    with pytest.raises(ValueError, match="must produce canonical_pose"):
        PLCSLightningModule._canonical_pose_metric_inputs(
            module,
            PLCSDecodedPrediction(position=position, rotation=rotation),
            PLCSPreparedBatch(
                call=ModelCall(),
                target_position=position,
                target_rotation=rotation,
                target_human_kp_3d=torch.zeros(1, 1, 17, 3),
            ),
            position,
            rotation,
        )

    with pytest.raises(ValueError, match="require target_human_kp_3d"):
        PLCSLightningModule._canonical_pose_metric_inputs(
            module,
            PLCSDecodedPrediction(
                position=position,
                rotation=rotation,
                canonical_pose=torch.zeros(1, 1, 17, 3),
            ),
            PLCSPreparedBatch(
                call=ModelCall(),
                target_position=position,
                target_rotation=rotation,
            ),
            position,
            rotation,
        )


def test_noncanonical_metric_inputs_remain_absent() -> None:
    position = torch.zeros(1, 3)
    rotation = torch.tensor([[1.0, 0.0]])

    class _Module:
        io_adapter = type("_Adapter", (), {"predict_canonical_pose": False})()

    result = PLCSLightningModule._canonical_pose_metric_inputs(
        cast("PLCSLightningModule", _Module()),
        PLCSDecodedPrediction(position=position, rotation=rotation),
        PLCSPreparedBatch(
            call=ModelCall(),
            target_position=position,
            target_rotation=rotation,
            target_human_kp_3d=torch.zeros(1, 17, 3),
        ),
        position,
        rotation,
    )

    assert result == (None, None)


def test_metric_logging_contract_is_conditional_on_canonical_capability() -> None:
    trajectory_keys = {
        "position_error_m",
        "angular_error_deg",
        "position_accuracy_0.5m",
        "angle_accuracy_15deg",
    }
    for enabled in (False, True):
        contract = build_plcs_metric_logging_contract(predict_canonical_pose=enabled)
        expected = trajectory_keys | (
            set(CANONICAL_POSE_HEADLINE_KEYS) if enabled else set()
        )
        for stage in ("train", "val", "test"):
            stage_contract = contract.for_stage(stage)
            assert set(stage_contract.headline_keys) == expected
            assert not set(CANONICAL_POSE_DIAGNOSTIC_KEYS).intersection(
                stage_contract.headline_keys
            )
            assert ("canonical_mpjpe_m" in stage_contract.progress_bar_keys) is (
                enabled and stage != "test"
            )


@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_canonical_epoch_logging_emits_only_conditional_headlines(stage: str) -> None:
    tracker = PLCSMetrics(
        position_threshold_m=0.5,
        angle_threshold_deg=15.0,
        predict_canonical_pose=True,
    )
    position = torch.zeros(1, 1, 3)
    rotation = torch.tensor([[[1.0, 0.0]]])
    pose = torch.zeros(1, 1, 17, 3)
    tracker.update(
        position,
        rotation,
        position,
        rotation,
        pred_canonical_pose=pose,
        target_canonical_pose=pose,
    )

    class _Recorder(ManualGANSupportMixin):
        metric_logging_contract = build_plcs_metric_logging_contract(
            predict_canonical_pose=True
        )

        def __init__(self) -> None:
            self.logged: dict[str, bool] = {}

        def _metric_tracker_for_stage(self, requested_stage: str) -> PLCSMetrics:
            assert requested_stage == stage
            return tracker

        def log(self, name: str, value: Any, **kwargs: Any) -> None:
            del value
            self.logged[name] = bool(kwargs["prog_bar"])

    recorder = _Recorder()
    recorder._flush_stage_metrics(stage)

    expected_names = {
        f"{stage}/position_error_m",
        f"{stage}/angular_error_deg",
        f"{stage}/position_accuracy_0.5m",
        f"{stage}/angle_accuracy_15deg",
        f"{stage}/canonical_mpjpe_m",
        f"{stage}/canonical_pck_0.1m",
    }
    assert set(recorder.logged) == expected_names
    assert not any(
        name.removeprefix(f"{stage}/") in CANONICAL_POSE_DIAGNOSTIC_KEYS
        for name in recorder.logged
    )
    assert recorder.logged[f"{stage}/canonical_mpjpe_m"] is (stage != "test")
    assert not recorder.logged[f"{stage}/canonical_pck_0.1m"]


def test_stage_logging_omits_batch_metric_aliases_and_eval_loss_components() -> None:
    class _Recorder:
        metric_logging_contract = PLCS_TRAJECTORY_METRIC_CONTRACT

        def __init__(self) -> None:
            self.names: list[str] = []
            self.gan_enabled = True

        def log(self, name: str, value: Any, **kwargs: Any) -> None:
            del value, kwargs
            self.names.append(name)

        def _log_gan_metrics(self, stage: str, metrics: dict[str, Any]) -> None:
            raise AssertionError(
                f"PLCS stage logging bypassed its contract: {stage=}, {metrics=}"
            )

    recorder = _Recorder()
    PLCSLightningModule._log_stage_metrics(
        cast("PLCSLightningModule", recorder),
        "val",
        torch.tensor(1.0),
        {
            "position_error_m": 0.1,
            "angular_error_deg": 2.0,
            "loss_position": 0.5,
        },
    )

    assert recorder.names == ["val/loss"]
    assert "val/pos_error_m" not in recorder.names
    assert "val/ang_error_deg" not in recorder.names
    assert "val/loss_position" not in recorder.names

    recorder.names.clear()
    PLCSLightningModule._log_stage_metrics(
        cast("PLCSLightningModule", recorder),
        "train",
        torch.tensor(1.0),
        {
            "position_error_m": 0.1,
            "angular_error_deg": 2.0,
            "loss_position": 0.5,
            "loss_gan_generator": 0.2,
            "loss_gan_discriminator": 0.3,
            "gan_weight": 0.1,
            "gan_phase_active": 1.0,
        },
    )
    assert recorder.names == ["train/loss"]

    recorder.names.clear()
    PLCSLightningModule._log_stage_metrics(
        cast("PLCSLightningModule", recorder),
        "test",
        torch.tensor(1.0),
        {
            "position_error_m": 0.1,
            "angular_error_deg": 2.0,
            "loss_position": 0.5,
        },
    )
    assert recorder.names == ["test/loss"]


def test_enabled_mcmc_injects_noise_without_bypassing_logging_contract() -> None:
    class _Injector:
        def __init__(self) -> None:
            self.calls = 0

        def inject(self, *args: Any, **kwargs: Any) -> float:
            del args, kwargs
            self.calls += 1
            return 0.125

    class _Recorder:
        def __init__(self) -> None:
            self.mcmc_injector = _Injector()
            self.model = torch.nn.Linear(1, 1)
            self.global_step = 3
            self.current_epoch = 1
            self.names: list[str] = []

        def optimizers(self) -> Any:
            return type("_Optimizer", (), {"param_groups": [{"lr": 1e-3}]})()

        def _estimate_total_steps(self) -> int:
            return 10

        def log(self, name: str, value: Any, **kwargs: Any) -> None:
            del value, kwargs
            self.names.append(name)

    recorder = _Recorder()

    PLCSLightningModule.on_train_batch_end(
        cast("PLCSLightningModule", recorder),
        outputs=None,
        batch=None,
        batch_idx=0,
    )

    assert recorder.mcmc_injector.calls == 1
    assert recorder.names == []
