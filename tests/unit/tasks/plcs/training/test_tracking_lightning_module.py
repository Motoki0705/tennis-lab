"""Unit tests for PLCS tracking fine-tuning and persistence payloads."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn
from torch.nn import functional as F

from src.tasks.plcs.models.plcs_track_query_reference_ablation_model import (
    PLCSTrackQueryReferenceAblationModel,
)
from src.tasks.plcs.models.plcs_track_query_reference_model import (
    PLCSTrackQueryReferenceModel,
)
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
    _require_independent_presence_head,
    _require_presence_competition,
)
from src.utils.geometry.court_pose import canonical_pose_to_world_pose
from src.utils.paths import PROJECT_ROOT


def _fine_tune_module() -> PLCSTrackingLightningModule:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="train_tracking_pose_presence_head",
            overrides=[
                "run.init_weights=source.ckpt",
                "model.hidden_dim=32",
                "model.num_heads=4",
                "model.ffn_dim=64",
                "model.rope_dim=4",
                "model.mhc.coefficient_dim=16",
                "training.trainer.max_epochs=2",
                "training.steps_per_epoch=1",
                "training.warmup_steps=0",
                "training.compile.enabled=false",
            ],
        )
    return PLCSTrackingLightningModule(config)


def _all_mode_module() -> PLCSTrackingLightningModule:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="train_tracking_pose",
            overrides=[
                "model.hidden_dim=32",
                "model.num_heads=4",
                "model.ffn_dim=64",
                "model.rope_dim=4",
                "model.mhc.coefficient_dim=16",
                "training.compile.enabled=false",
            ],
        )
    return PLCSTrackingLightningModule(config)


def _competition_fine_tune_module(
    *,
    model_profile: str = "track_query",
    competition_mode: Literal["deepsets", "deepsets_centered"] = "deepsets",
) -> PLCSTrackingLightningModule:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    overrides = [
        "run.init_weights=source.ckpt",
        f"model={model_profile}",
        "model.hidden_dim=32",
        "model.num_heads=4",
        "model.ffn_dim=64",
        "model.rope_dim=4",
        "model.mhc.coefficient_dim=16",
        "training.trainer.max_epochs=2",
        "training.steps_per_epoch=1",
        "training.warmup_steps=0",
        "training.compile.enabled=false",
    ]
    if model_profile in {
        "track_query_reference",
        "track_query_ablation_d_v2_selector",
    }:
        overrides.extend(
            [
                "court_keypoints=camera_view_v2",
                "model.rope_dim=6",
            ]
        )
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name=(
                "train_tracking_pose_presence_competition_centered"
                if competition_mode == "deepsets_centered"
                else "train_tracking_pose_presence_competition"
            ),
            overrides=overrides,
        )
    return PLCSTrackingLightningModule(config)


def _model_inputs(module: PLCSTrackingLightningModule) -> dict[str, torch.Tensor]:
    prefix = (1, 2, 2)
    num_queries = module.plcs_runtime.model.integer("num_queries")
    return {
        "human_kp": torch.rand(*prefix, num_queries, 17, 2),
        "human_vis": torch.ones(
            *prefix,
            num_queries,
            17,
            dtype=torch.bool,
        ),
        "court_kp": torch.rand(*prefix, 14, 2),
        "court_vis": torch.ones(*prefix, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(*prefix, dtype=torch.bool),
    }


def _model_forward(
    module: PLCSTrackingLightningModule,
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return cast("dict[str, torch.Tensor]", module.model(**inputs))


def test_presence_head_fine_tune_has_exact_trainable_parameter_names() -> None:
    module = _fine_tune_module()

    trainable_names = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }

    assert trainable_names == {
        "model.presence_head.weight",
        "model.presence_head.bias",
    }
    presence_head = _require_independent_presence_head(module.model)
    groups = module.optimizer_param_groups()
    assert groups is not None
    grouped_parameters = cast("list[nn.Parameter]", groups[0]["params"])
    assert {id(parameter) for parameter in grouped_parameters} == {
        id(parameter) for parameter in presence_head.parameters()
    }


def test_presence_head_fine_tune_keeps_trunk_eval_only_during_training() -> None:
    module = _fine_tune_module()
    presence_head = _require_independent_presence_head(module.model)

    module.train()

    assert module.training
    assert not module.model.training
    assert presence_head.training
    assert all(
        not child.training
        for child in module.model.modules()
        if child is not presence_head
    )

    module.eval()

    assert not module.training
    assert all(not child.training for child in module.model.modules())


def test_presence_head_optimizer_step_preserves_pose_weights_outputs_and_buffers() -> None:
    torch.manual_seed(31)
    module = _fine_tune_module()
    module.train()
    inputs = _model_inputs(module)
    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    buffer_before = {
        name: buffer.detach().clone() for name, buffer in module.named_buffers()
    }
    output_before = {
        name: value.detach().clone()
        for name, value in _model_forward(module, inputs).items()
    }
    optimizer_config = cast("dict[str, Any]", module.configure_optimizers())
    optimizer = cast("torch.optim.Optimizer", optimizer_config["optimizer"])

    optimizer.zero_grad(set_to_none=True)
    output = _model_forward(module, inputs)
    loss = F.binary_cross_entropy_with_logits(
        output["presence_logits"],
        torch.zeros_like(output["presence_logits"]),
    )
    loss.backward()
    optimizer.step()

    parameter_after = dict(module.named_parameters())
    presence_parameter_names = {
        "model.presence_head.weight",
        "model.presence_head.bias",
    }
    assert any(
        not torch.equal(parameter_before[name], parameter_after[name])
        for name in presence_parameter_names
    )
    for name, before in parameter_before.items():
        if name not in presence_parameter_names:
            torch.testing.assert_close(parameter_after[name], before, rtol=0.0, atol=0.0)
            assert parameter_after[name].grad is None
    for name, before in buffer_before.items():
        torch.testing.assert_close(
            dict(module.named_buffers())[name],
            before,
            rtol=0.0,
            atol=0.0,
        )

    output_after = _model_forward(module, inputs)
    for name in ("position", "rotation", "canonical_pose"):
        torch.testing.assert_close(output_after[name], output_before[name])
    assert not torch.equal(
        output_after["presence_logits"], output_before["presence_logits"]
    )


def test_unsupported_model_without_independent_presence_head_is_rejected() -> None:
    with pytest.raises(ValueError, match="independent registered nn.Module"):
        _require_independent_presence_head(nn.Sequential(nn.Linear(2, 1)))


def test_presence_competition_has_exact_trainable_parameter_names() -> None:
    module = _competition_fine_tune_module()
    branch = _require_presence_competition(module.model)

    trainable_names = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }

    assert trainable_names == {
        "model.presence_competition.feature_projection.weight",
        "model.presence_competition.feature_projection.bias",
        "model.presence_competition.output_projection.weight",
        "model.presence_competition.output_projection.bias",
    }
    assert not module.model.presence_head.weight.requires_grad
    assert not module.model.presence_head.bias.requires_grad
    groups = module.optimizer_param_groups()
    assert groups is not None
    grouped_parameters = cast("list[nn.Parameter]", groups[0]["params"])
    assert {id(parameter) for parameter in grouped_parameters} == {
        id(parameter) for parameter in branch.parameters()
    }


def test_centered_presence_competition_has_exact_bias_free_optimizer_state() -> None:
    module = _competition_fine_tune_module(
        competition_mode="deepsets_centered",
    )
    branch = _require_presence_competition(module.model)
    trainable_names = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }

    assert trainable_names == {
        "model.presence_competition.feature_projection.weight",
        "model.presence_competition.feature_projection.bias",
        "model.presence_competition.output_projection.weight",
    }
    assert branch.output_projection.bias is None
    assert "output_projection.bias" not in dict(branch.named_parameters())
    assert "output_projection.bias" not in branch.state_dict()
    groups = module.optimizer_param_groups()
    assert groups is not None
    grouped_parameters = cast("list[nn.Parameter]", groups[0]["params"])
    grouped_ids = {id(parameter) for parameter in grouped_parameters}
    assert grouped_ids == {id(parameter) for parameter in branch.parameters()}
    assert grouped_ids == {
        id(parameter)
        for name, parameter in module.named_parameters()
        if name in trainable_names
    }


def test_presence_competition_keeps_only_branch_in_train_mode() -> None:
    module = _competition_fine_tune_module()
    branch = _require_presence_competition(module.model)

    module.train()

    assert module.training
    assert not module.model.training
    assert branch.training
    branch_modules = set(branch.modules())
    assert all(
        not child.training
        for child in module.model.modules()
        if child not in branch_modules
    )

    module.eval()

    assert not module.training
    assert all(not child.training for child in module.model.modules())


def test_presence_competition_step_preserves_legacy_heads_pose_and_buffers() -> None:
    torch.manual_seed(37)
    module = _competition_fine_tune_module()
    module.train()
    inputs = _model_inputs(module)
    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    buffer_before = {
        name: buffer.detach().clone() for name, buffer in module.named_buffers()
    }
    output_before = {
        name: value.detach().clone()
        for name, value in _model_forward(module, inputs).items()
    }
    optimizer_config = cast("dict[str, Any]", module.configure_optimizers())
    optimizer = cast("torch.optim.Optimizer", optimizer_config["optimizer"])

    optimizer.zero_grad(set_to_none=True)
    output = _model_forward(module, inputs)
    loss = F.binary_cross_entropy_with_logits(
        output["presence_logits"],
        torch.zeros_like(output["presence_logits"]),
    )
    loss.backward()
    optimizer.step()

    parameter_after = dict(module.named_parameters())
    branch_prefix = "model.presence_competition."
    assert any(
        not torch.equal(parameter_before[name], parameter_after[name])
        for name in parameter_before
        if name.startswith(branch_prefix)
    )
    for name, before in parameter_before.items():
        if not name.startswith(branch_prefix):
            torch.testing.assert_close(
                parameter_after[name], before, rtol=0.0, atol=0.0
            )
            assert parameter_after[name].grad is None
    for name, before in buffer_before.items():
        torch.testing.assert_close(
            dict(module.named_buffers())[name],
            before,
            rtol=0.0,
            atol=0.0,
        )

    output_after = _model_forward(module, inputs)
    for name in ("position", "rotation", "canonical_pose"):
        torch.testing.assert_close(output_after[name], output_before[name])
    assert not torch.equal(
        output_after["presence_logits"], output_before["presence_logits"]
    )


def test_presence_competition_rejects_model_without_registered_branch() -> None:
    with pytest.raises(ValueError, match="DeepSetsPresenceResidual"):
        _require_presence_competition(nn.Sequential(nn.Linear(2, 1)))


@pytest.mark.parametrize(
    ("model_profile", "expected_model_type"),
    [
        ("track_query_reference", PLCSTrackQueryReferenceModel),
        (
            "track_query_ablation_d_v2_selector",
            PLCSTrackQueryReferenceAblationModel,
        ),
    ],
)
def test_reference_competition_fine_tune_lifecycle_is_branch_only(
    model_profile: str,
    expected_model_type: type[nn.Module],
) -> None:
    module = _competition_fine_tune_module(model_profile=model_profile)
    branch = _require_presence_competition(module.model)

    assert type(module.model) is expected_model_type
    trainable_names = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    expected_trainable_names = {
        f"model.presence_competition.{name}"
        for name, _parameter in branch.named_parameters()
    }
    assert trainable_names == expected_trainable_names
    groups = module.optimizer_param_groups()
    assert groups is not None
    grouped_parameters = cast("list[nn.Parameter]", groups[0]["params"])
    assert {id(parameter) for parameter in grouped_parameters} == {
        id(parameter) for parameter in branch.parameters()
    }

    module.train()

    branch_modules = set(branch.modules())
    assert not module.model.training
    assert all(child.training for child in branch_modules)
    assert all(
        not child.training
        for child in module.model.modules()
        if child not in branch_modules
    )


def test_all_mode_preserves_legacy_train_and_optimizer_behavior() -> None:
    module = _all_mode_module()
    presence_head = _require_independent_presence_head(module.model)

    assert module.fine_tune_mode == "all"
    assert all(parameter.requires_grad for parameter in module.parameters())
    assert module.optimizer_param_groups() is None

    module.train()

    assert module.model.training
    assert presence_head.training


def test_canonical_payload_persists_prediction_and_derived_target() -> None:
    target_position = torch.tensor([[[[0.2, -0.1, 0.03]]]])
    target_rotation = torch.tensor([[[[0.6, 0.8]]]])
    target_canonical = torch.linspace(-0.5, 0.8, 17 * 3).reshape(
        1, 1, 1, 17, 3
    )
    target_world = canonical_pose_to_world_pose(
        target_canonical,
        target_position,
        target_rotation,
    )
    prediction = torch.randn_like(target_canonical)
    batch = {
        "target_position": target_position,
        "target_rotation": target_rotation,
        "target_human_kp_3d": target_world,
        "target_presence": torch.ones(1, 1, 1, dtype=torch.bool),
        "target_instance_id": torch.ones(1, 1, 1, dtype=torch.int64),
        "padding_mask": torch.zeros(1, 1, 1, dtype=torch.bool),
    }
    result = {
        "position": target_position,
        "rotation": target_rotation,
        "presence_logits": torch.zeros(1, 1, 1),
        "canonical_pose": prediction,
    }
    module = object.__new__(PLCSTrackingLightningModule)

    payload = module.test_prediction_payload(batch, result)

    np.testing.assert_allclose(payload["pred_canonical_pose"], prediction.numpy())
    np.testing.assert_allclose(
        payload["target_canonical_pose"], target_canonical.numpy(), atol=1e-6
    )
