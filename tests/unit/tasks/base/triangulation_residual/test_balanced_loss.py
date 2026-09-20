"""Properties of the opt-in v2 objective and the untouched legacy objective."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.base.triangulation_residual.configuration import LossConfig, V2Config
from src.tasks.base.triangulation_residual.losses import residual_loss


def loss_config(task: str, *, auxiliaries: bool = True) -> LossConfig:
    return LossConfig(
        root_weight=1.0,
        relative_weight=1.0 if task == "plcs" else 0.0,
        world_weight=0.5,
        reprojection_weight=0.2 if auxiliaries else 0.0,
        velocity_weight=0.05 if auxiliaries else 0.0,
        bone_weight=0.05 if task == "plcs" and auxiliaries else 0.0,
        huber_delta_m=0.1,
    )


def v2_config() -> V2Config:
    return V2Config(
        camera_preset="four_corners_front_pair",
        evaluation_views=6,
        true_camera_position_jitter_m=0.0,
        true_camera_height_jitter_m=0.0,
        calibration_min_points=6,
        court_noise_px=1.0,
        court_bias_px=1.0,
        court_dropout_probability=0.0,
        court_outlier_probability=0.0,
        court_outlier_sigma_px=1.0,
        persistent_rate_per_view_second=0.1,
        persistent_min_seconds=0.1,
        persistent_max_seconds=0.5,
        persistent_offset_scale=1.0,
        persistent_high_confidence_probability=0.5,
        error_mode="mixed",
        loss_mode="balanced_regret",
        regret_weight=0.2,
        regret_tolerance_m=0.002,
        balanced_world_weight=0.25,
    )


def batch_fixture(task: str, *, samples: int = 1, frames: int = 4) -> dict[str, Tensor]:
    joints = 17 if task == "plcs" else 1
    generator = torch.Generator().manual_seed(23)
    target = torch.randn(samples, frames, joints, 3, generator=generator) * 0.2
    target[..., 2] += 5.0
    initial = target + torch.randn(target.shape, generator=generator) * 0.15
    indices = [11, 12] if task == "plcs" else [0]
    root = initial[:, :, indices].mean(dim=2)
    relative = initial - root[:, :, None]
    projection = (
        torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
        .expand(samples, 2, -1, -1)
        .clone()
    )
    return {
        "features": torch.zeros(samples, 2, frames, 18 * joints + 61),
        "root_init": root,
        "relative_init": relative,
        "init_world": root[:, :, None] + relative,
        "init_valid": torch.ones(samples, frames, joints, dtype=torch.bool),
        "target_world": target,
        "frame_valid": torch.ones(samples, frames, dtype=torch.bool),
        "true_projection": projection,
        "clean_uv": (target[..., :2] / target[..., 2:3])[:, None]
        .expand(-1, 2, -1, -1, -1)
        .clone(),
        "clean_visible": torch.ones(samples, 2, frames, joints, dtype=torch.bool),
        "fps": torch.full((samples, 1), 30.0),
        "severity": torch.ones(samples, 1),
    }


def identity_output(batch: dict[str, Tensor], task: str) -> dict[str, Tensor]:
    result = {
        "root_residual" if task == "plcs" else "position_residual": torch.zeros_like(
            batch["root_init"]
        )
    }
    if task == "plcs":
        result["relative_residual"] = torch.zeros_like(batch["relative_init"])
    return result


def target_output(batch: dict[str, Tensor], task: str) -> dict[str, Tensor]:
    target = batch["target_world"]
    indices = [11, 12] if task == "plcs" else [0]
    root = target[:, :, indices].mean(dim=2)
    result = {
        "root_residual" if task == "plcs" else "position_residual": root
        - batch["root_init"]
    }
    if task == "plcs":
        result["relative_residual"] = target - root[:, :, None] - batch["relative_init"]
    return result


def position_fixture(
    initial_errors: list[float], final_errors: list[float], *, frames: int = 4
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    batch = batch_fixture("blcs", samples=len(initial_errors), frames=frames)
    batch["target_world"].zero_()
    batch["target_world"][..., 2] = 5.0
    batch["clean_uv"].zero_()
    batch["root_init"] = batch["target_world"][:, :, 0].clone()
    batch["root_init"][..., 0] = torch.tensor(initial_errors)[:, None]
    batch["relative_init"].zero_()
    batch["init_world"] = batch["root_init"][:, :, None].clone()
    output = identity_output(batch, "blcs")
    output["position_residual"][..., 0] = (
        torch.tensor(final_errors) - torch.tensor(initial_errors)
    )[:, None]
    return batch, output


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_ground_truth_residuals_have_zero_balanced_loss(task):
    batch = batch_fixture(task)
    output = {
        key: value.requires_grad_() for key, value in target_output(batch, task).items()
    }
    total, parts, world = residual_loss(
        output, batch, loss_config(task), task, v2=v2_config()
    )
    torch.testing.assert_close(world, batch["target_world"], atol=1e-6, rtol=1e-6)
    assert total < 1e-8
    assert set(parts) == {
        "root",
        "relative",
        "world",
        "reprojection",
        "velocity",
        "bone",
        "regret",
    }
    assert all(value < 1e-8 for value in parts.values())
    total.backward()
    assert all(
        value.grad is not None and value.grad.isfinite().all()
        for value in output.values()
    )


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_identity_has_zero_regret_even_for_filled_initial_points(task):
    batch = batch_fixture(task, samples=3)
    batch["severity"] = torch.tensor([0.0, 1.0, 3.0])
    batch["init_valid"][:, 1:3] = False
    _, parts, world = residual_loss(
        identity_output(batch, task), batch, loss_config(task), task, v2=v2_config()
    )
    torch.testing.assert_close(world, batch["init_world"])
    assert parts["regret"] == 0
    assert parts["root"] > 0


@pytest.mark.parametrize(
    "initial,final,expected",
    [
        (0.2, 0.3, 0.098),
        (2.0, 2.1, 0.098),
        (0.0, 0.1, 0.098),
        (0.2, 0.1, 0.0),
        (0.2, 0.201, 0.0),
        (0.2, 0.203, 0.001),
    ],
)
def test_paired_regret_uses_absolute_metres_and_tolerance(initial, final, expected):
    batch, output = position_fixture([initial], [final])
    v2 = v2_config()
    config = loss_config("blcs", auxiliaries=False)
    total, parts, _ = residual_loss(output, batch, config, "blcs", v2=v2)
    torch.testing.assert_close(
        parts["regret"], torch.tensor(expected), atol=2e-7, rtol=1e-5
    )
    torch.testing.assert_close(
        total, parts["root"] + v2.regret_weight * parts["regret"]
    )


def test_sample_length_and_stratum_frequency_do_not_change_main_weighting():
    batch, output = position_fixture([0.0] * 4, [0.1, 0.3, 0.5, 0.7])
    batch["severity"] = torch.tensor([0.0, 1.0, 1.0, 3.0])
    batch["frame_valid"] = torch.arange(4)[None] < torch.tensor([4, 1, 3, 2])[:, None]
    output["position_residual"][~batch["frame_valid"]] = 1000.0
    _, parts, _ = residual_loss(
        output, batch, loss_config("blcs"), "blcs", v2=v2_config()
    )
    # Clean: .05, normal: mean(.25, .45), hard: .65. Each stratum is one vote.
    torch.testing.assert_close(parts["root"], torch.tensor(0.35))


@pytest.mark.parametrize("severity", [[1.0, 1.0, 1.0], [0.0, 1.0, 3.0]])
def test_empty_samples_and_absent_strata_do_not_dilute_loss(severity):
    batch, output = position_fixture([0.0] * 3, [0.2, 0.4, 100.0])
    batch["severity"] = torch.tensor(severity)
    batch["frame_valid"] = torch.arange(4)[None] < torch.tensor([1, 3, 0])[:, None]
    _, parts, _ = residual_loss(
        output, batch, loss_config("blcs"), "blcs", v2=v2_config()
    )
    torch.testing.assert_close(parts["root"], torch.tensor(0.25))


def test_auxiliary_terms_use_the_same_sample_and_stratum_balance():
    batch = batch_fixture("plcs", samples=4)
    batch["severity"] = torch.tensor([0.0, 1.0, 1.0, 3.0])
    batch["frame_valid"] = torch.arange(4)[None] < torch.tensor([4, 1, 3, 2])[:, None]
    batch["clean_visible"][1] = False
    output = identity_output(batch, "plcs")
    _, parts, _ = residual_loss(
        output, batch, loss_config("plcs"), "plcs", v2=v2_config()
    )
    individual = []
    for sample in range(4):
        _, sample_parts, _ = residual_loss(
            {key: value[sample : sample + 1] for key, value in output.items()},
            {key: value[sample : sample + 1] for key, value in batch.items()},
            loss_config("plcs"),
            "plcs",
            v2=v2_config(),
        )
        individual.append(sample_parts)
    for term in ("bone", "reprojection", "velocity"):
        # Sample 1 has neither a clean UV target nor adjacent valid frames.
        normal = (
            (individual[1][term] + individual[2][term]) / 2
            if term == "bone"
            else individual[2][term]
        )
        expected = (individual[0][term] + normal + individual[3][term]) / 3
        torch.testing.assert_close(parts[term], expected)


def test_blcs_has_only_one_position_term():
    batch, output = position_fixture([0.0], [0.2])
    config = replace(loss_config("blcs", auxiliaries=False), world_weight=100.0)
    v2 = replace(v2_config(), balanced_world_weight=200.0, regret_weight=0.0)
    total, parts, _ = residual_loss(output, batch, config, "blcs", v2=v2)
    torch.testing.assert_close(total, torch.tensor(0.15))
    assert parts["world"] == 0
    assert parts["relative"] == 0
    assert parts["bone"] == 0


def test_plcs_keeps_direct_heads_and_uses_v2_world_coupling_weight():
    batch = batch_fixture("plcs")
    config = replace(loss_config("plcs", auxiliaries=False), world_weight=100.0)
    v2 = replace(v2_config(), regret_weight=0.0)
    total, parts, _ = residual_loss(
        identity_output(batch, "plcs"), batch, config, "plcs", v2=v2
    )
    assert all(parts[key] > 0 for key in ("root", "relative", "world"))
    torch.testing.assert_close(
        total,
        config.root_weight * parts["root"]
        + config.relative_weight * parts["relative"]
        + v2.balanced_world_weight * parts["world"],
    )


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_main_3d_terms_are_rotation_invariant(task):
    batch = batch_fixture(task)
    output = identity_output(batch, task)
    config = loss_config(task, auxiliaries=False)
    v2 = v2_config()
    total, parts, _ = residual_loss(output, batch, config, task, v2=v2)
    rotation = torch.tensor([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    rotated = {key: value.clone() for key, value in batch.items()}
    for key in ("root_init", "relative_init", "init_world", "target_world"):
        rotated[key] = rotated[key] @ rotation.T
    rotated_total, rotated_parts, _ = residual_loss(
        {key: value @ rotation.T for key, value in output.items()},
        rotated,
        config,
        task,
        v2=v2,
    )
    for term in ("root", "relative", "world", "regret"):
        torch.testing.assert_close(
            parts[term], rotated_parts[term], atol=1e-6, rtol=1e-5
        )
    torch.testing.assert_close(total, rotated_total, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_zero_distance_has_finite_gradients(task):
    batch = batch_fixture(task)
    # Use the represented seed as GT so the zero-error norm is exact.
    batch["target_world"] = batch["init_world"].clone()
    batch["clean_uv"] = (
        (batch["target_world"][..., :2] / batch["target_world"][..., 2:3])[:, None]
        .expand(-1, 2, -1, -1, -1)
        .clone()
    )
    output = {
        key: value.requires_grad_()
        for key, value in identity_output(batch, task).items()
    }
    total, _, _ = residual_loss(output, batch, loss_config(task), task, v2=v2_config())
    assert total < 1e-8
    total.backward()
    for value in output.values():
        assert value.grad is not None
        assert value.grad.isfinite().all()
        torch.testing.assert_close(
            value.grad, torch.zeros_like(value.grad), atol=1e-5, rtol=0
        )


@pytest.mark.parametrize("task", ["plcs", "blcs"])
@pytest.mark.parametrize("fully_empty", [False, True])
def test_nan_padding_is_ignored_and_has_zero_gradient(task, fully_empty):
    batch = batch_fixture(task)
    batch["frame_valid"][:, 0 if fully_empty else 2 :] = False
    output = identity_output(batch, task)
    total, parts, _ = residual_loss(
        output, batch, loss_config(task), task, v2=v2_config()
    )
    for key in ("root_init", "relative_init", "init_world", "target_world"):
        batch[key][~batch["frame_valid"]] = torch.nan
    for value in output.values():
        value[~batch["frame_valid"]] = torch.nan
        value.requires_grad_()
    padded_total, padded_parts, _ = residual_loss(
        output, batch, loss_config(task), task, v2=v2_config()
    )
    torch.testing.assert_close(total, padded_total)
    for key in parts:
        torch.testing.assert_close(parts[key], padded_parts[key])
    if fully_empty:
        assert padded_total == 0
    padded_total.backward()
    for value in output.values():
        assert value.grad is not None
        assert value.grad.isfinite().all()
        assert (value.grad[~batch["frame_valid"]] == 0).all()


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_bfloat16_predictions_keep_loss_and_camera_math_in_fp32(task):
    batch = batch_fixture(task)
    output = {
        key: value.to(torch.bfloat16).requires_grad_()
        for key, value in target_output(batch, task).items()
    }
    config, v2 = loss_config(task), v2_config()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        total, parts, world = residual_loss(output, batch, config, task, v2=v2)
    reference, reference_parts, reference_world = residual_loss(
        {key: value.float() for key, value in output.items()},
        batch,
        config,
        task,
        v2=v2,
    )
    assert total.dtype == world.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in parts.values())
    torch.testing.assert_close(total, reference, atol=0, rtol=0)
    torch.testing.assert_close(world, reference_world, atol=0, rtol=0)
    for key in parts:
        torch.testing.assert_close(parts[key], reference_parts[key], atol=0, rtol=0)
    total.backward()
    assert all(
        value.grad is not None and value.grad.isfinite().all()
        for value in output.values()
    )


@pytest.mark.parametrize(
    "field", ["prediction", "target_world", "clean_uv", "init_world"]
)
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_valid_supervision_fails_instead_of_dropping_the_sample(
    field, nonfinite
):
    batch = batch_fixture("blcs")
    output = identity_output(batch, "blcs")
    if field == "prediction":
        output["position_residual"][0, 0, 0] = nonfinite
    else:
        batch[field].flatten()[0] = nonfinite
    with pytest.raises(ValueError, match="Non-finite"):
        residual_loss(output, batch, loss_config("blcs"), "blcs", v2=v2_config())


@pytest.mark.parametrize("severity", [float("nan"), float("inf"), -1.0, 0.5])
def test_invalid_severity_cannot_silently_remove_a_sample(severity):
    batch = batch_fixture("blcs")
    batch["severity"].fill_(severity)
    with pytest.raises(ValueError, match="Severity"):
        residual_loss(
            identity_output(batch, "blcs"),
            batch,
            loss_config("blcs"),
            "blcs",
            v2=v2_config(),
        )


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_v2_legacy_dispatch_is_exactly_the_v1_loss(task):
    batch = batch_fixture(task, samples=3)
    # Legacy mode retains its original dependencies; no severity or seed error.
    del batch["severity"], batch["init_world"]
    output = identity_output(batch, task)
    config = loss_config(task)
    total, parts, world = residual_loss(output, batch, config, task)
    legacy, legacy_parts, legacy_world = residual_loss(
        output, batch, config, task, v2=replace(v2_config(), loss_mode="legacy")
    )
    torch.testing.assert_close(total, legacy, atol=0, rtol=0)
    torch.testing.assert_close(world, legacy_world, atol=0, rtol=0)
    assert set(parts) == set(legacy_parts)
    assert "regret" not in parts
    for key in parts:
        torch.testing.assert_close(parts[key], legacy_parts[key], atol=0, rtol=0)


def test_legacy_blcs_preserves_componentwise_huber_and_world_weight():
    batch, output = position_fixture([0.0], [0.2])
    config = loss_config("blcs", auxiliaries=False)
    total, parts, world = residual_loss(output, batch, config, "blcs")
    expected = F.smooth_l1_loss(world, batch["target_world"], beta=config.huber_delta_m)
    torch.testing.assert_close(parts["root"], expected)
    torch.testing.assert_close(parts["world"], expected)
    torch.testing.assert_close(
        total, (config.root_weight + config.world_weight) * expected
    )
