"""CPU model/adapter/loss integration, including gradient and leakage checks."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallCandidates,
)
from src.tasks.ball_refiner.refiner_2d import (
    Refiner2DConfig,
    Refiner2DInput,
    Refiner2DTarget,
    build_ball_refiner_2d,
    refiner_2d_nll,
)
from src.tasks.ball_refiner.refiner_2d.model import Refiner2DModel
from src.tasks.ball_refiner.refiner_2d.model_io import Refiner2DAdapter


def config(**changes) -> Refiner2DConfig:
    path = Path(__file__).resolve().parents[4] / "src/tasks/ball_refiner/configs/model/refiner_2d.yaml"
    values = dict(OmegaConf.load(path))
    values.update(hidden_dim=16, attention_heads=2, court_keypoints=5, patch_size=3, dropout=0.0, pose_dropout=0.0)
    return replace(
        Refiner2DConfig(**values),
        **changes,
    )


def test_yaml_is_the_complete_config_authority() -> None:
    from src.utils.configuration.contracts import inspect_typed_adapter

    path = Path(__file__).resolve().parents[4] / "src/tasks/ball_refiner/configs/model/refiner_2d.yaml"
    values = dict(OmegaConf.load(path))
    cfg = Refiner2DConfig(**values)
    inspect_typed_adapter(Refiner2DConfig)  # rejects Python defaults
    assert cfg.court_keypoints == 14  # pipeline camera_view_v2
    del values["use_court"]
    with pytest.raises(TypeError, match="use_court"):
        Refiner2DConfig(**values)


def inputs(*, people: int = 2) -> Refiner2DInput:
    b, t, n = 2, 5, 3
    patches = torch.rand(b, t, n, 3, 3)
    return Refiner2DInput(
        candidates=BallCandidates(
            coords=torch.rand(b, t, n, 2),
            scores=patches[..., 1, 1].clone(),
            valid=torch.ones(b, t, n, dtype=torch.bool),
            cells=torch.zeros(b, t, n, 2, dtype=torch.int64),
            patches=patches,
            patch_valid=torch.ones_like(patches, dtype=torch.bool),
            config=BallCandidateConfig(max_candidates=n, patch_size=3, nms_kernel=3),
        ),
        timestamps_seconds=torch.arange(t).float().expand(b, -1) / 30,
        pose_uv=torch.rand(b, t, people, 4, 2),
        pose_confidence=torch.rand(b, t, people, 4),
        pose_valid=torch.ones(b, t, people, 4, dtype=torch.bool),
        court_uv=torch.rand(b, 5, 2),
        court_confidence=torch.rand(b, 5),
        court_valid=torch.ones(b, 5, dtype=torch.bool),
    )


def raw(
    model: Refiner2DModel, adapter: Refiner2DAdapter, batch: Refiner2DInput
) -> torch.Tensor:
    call = adapter.build_call(batch)
    return model(*call.args, **dict(call.kwargs))


def test_zero_gates_equal_detector_temporal_model_and_can_learn_context() -> None:
    cfg, batch = config(), inputs()
    model, adapter = Refiner2DModel(cfg), Refiner2DAdapter(cfg)
    model.eval()
    baseline_adapter = Refiner2DAdapter(replace(cfg, use_pose=False, use_court=False))
    torch.testing.assert_close(
        raw(model, adapter, batch), raw(model, baseline_adapter, batch), rtol=0, atol=0
    )
    teacher = Refiner2DTarget(
        uv=torch.rand(2, 5, 2),
        position_valid=torch.ones(2, 5, dtype=torch.bool),
        presence=torch.ones(2, 5, dtype=torch.bool),
        presence_valid=torch.ones(2, 5, dtype=torch.bool),
        weight=torch.ones(2, 5),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = refiner_2d_nll(adapter.decode_output(raw(model, adapter, batch)), teacher)
    result.loss.backward()
    for gate in (model.pose_context.gate, model.court_context.gate):
        assert gate.grad is not None and gate.grad.abs() > 0
    assert torch.count_nonzero(model.pose_encoder.weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad()
    refiner_2d_nll(
        adapter.decode_output(raw(model, adapter, batch)), teacher
    ).loss.backward()
    assert torch.count_nonzero(model.pose_encoder.weight.grad) > 0
    assert torch.count_nonzero(model.court_encoder.weight.grad) > 0
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert not torch.equal(
        raw(model, adapter, batch), raw(model, baseline_adapter, batch)
    )


def test_all_missing_evidence_and_zero_people_still_produce_valid_distribution() -> (
    None
):
    pair = build_ball_refiner_2d(config())
    pair.model.eval()
    batch = inputs(people=0)
    c = batch.candidates
    empty = replace(
        c,
        coords=torch.zeros_like(c.coords),
        scores=torch.zeros_like(c.scores),
        patches=torch.zeros_like(c.patches),
        valid=torch.zeros_like(c.valid),
        patch_valid=torch.zeros_like(c.patch_valid),
    )
    batch = replace(
        batch, candidates=empty, court_valid=torch.zeros_like(batch.court_valid)
    )
    result = pair.run(batch)
    assert result.means.shape == (2, 5, 4, 2)
    assert result.covariance.shape == (2, 5, 4, 2, 2)
    assert torch.isfinite(result.means).all()
    assert torch.linalg.eigvalsh(result.covariance).min() > 0
    torch.testing.assert_close(result.weights.sum(-1), torch.ones(2, 5))
    assert (
        (result.presence_probability >= 0) & (result.presence_probability <= 1)
    ).all()


def test_candidate_and_person_permutations_do_not_change_prediction() -> None:
    cfg, batch = config(), inputs()
    model, adapter = Refiner2DModel(cfg), Refiner2DAdapter(cfg)
    model.eval()
    with torch.no_grad():
        model.pose_context.gate.fill_(0.5)
    c = batch.candidates
    changed = replace(
        batch,
        candidates=replace(
            c,
            coords=c.coords.flip(2),
            scores=c.scores.flip(2),
            valid=c.valid.flip(2),
            cells=c.cells.flip(2),
            patches=c.patches.flip(2),
            patch_valid=c.patch_valid.flip(2),
        ),
        pose_uv=batch.pose_uv.flip(2),
        pose_confidence=batch.pose_confidence.flip(2),
        pose_valid=batch.pose_valid.flip(2),
    )
    torch.testing.assert_close(
        raw(model, adapter, batch), raw(model, adapter, changed), rtol=1e-5, atol=1e-6
    )


def test_camera_batch_rows_do_not_exchange_information() -> None:
    cfg, batch = config(), inputs()
    model, adapter = Refiner2DModel(cfg), Refiner2DAdapter(cfg)
    model.eval()
    with torch.no_grad():
        model.pose_context.gate.fill_(1)
        model.court_context.gate.fill_(1)
    # Mutate only the second camera-window; the first output must stay fixed.
    coords = batch.candidates.coords.clone()
    pose, court = batch.pose_uv.clone(), batch.court_uv.clone()
    coords[1] = 1 - coords[1]
    pose[1] += 1
    court[1] -= 1
    changed = replace(
        batch,
        candidates=replace(batch.candidates, coords=coords),
        pose_uv=pose,
        court_uv=court,
    )
    before, after = raw(model, adapter, batch), raw(model, adapter, changed)
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert not torch.equal(before[1], after[1])


def test_context_only_has_no_detector_leakage_and_missing_pose_has_no_effect() -> None:
    cfg, batch = config(use_detector=False), inputs()
    model, adapter = Refiner2DModel(cfg), Refiner2DAdapter(cfg)
    model.eval()
    with torch.no_grad():
        model.pose_context.gate.fill_(1)
        model.court_context.gate.fill_(1)
    changed = replace(batch, candidates=inputs().candidates)
    torch.testing.assert_close(
        raw(model, adapter, batch), raw(model, adapter, changed), rtol=0, atol=0
    )
    missing = replace(batch, pose_valid=torch.zeros_like(batch.pose_valid))
    poisoned = replace(missing, pose_uv=torch.full_like(missing.pose_uv, 1000))
    torch.testing.assert_close(
        raw(model, adapter, missing), raw(model, adapter, poisoned), rtol=0, atol=0
    )


def test_pose_dropout_drops_entire_window_and_eval_restores_context() -> None:
    cfg, batch = config(pose_dropout=1.0), inputs()
    model, adapter = Refiner2DModel(cfg), Refiner2DAdapter(cfg)
    with torch.no_grad():
        model.pose_context.gate.fill_(1)
    no_pose = Refiner2DAdapter(replace(cfg, use_pose=False))
    model.train()
    torch.testing.assert_close(
        raw(model, adapter, batch), raw(model, no_pose, batch), rtol=0, atol=0
    )
    model.eval()
    assert not torch.equal(raw(model, adapter, batch), raw(model, no_pose, batch))


def test_real_time_intervals_affect_output_but_absolute_origin_does_not() -> None:
    cfg, batch = config(), inputs()
    model, adapter = Refiner2DModel(cfg), Refiner2DAdapter(cfg)
    model.eval()
    original = raw(model, adapter, batch)
    shifted = replace(batch, timestamps_seconds=batch.timestamps_seconds + 1)
    torch.testing.assert_close(
        original, raw(model, adapter, shifted), atol=3e-6, rtol=1e-5
    )
    slower = replace(batch, timestamps_seconds=batch.timestamps_seconds * 2)
    assert not torch.equal(original, raw(model, adapter, slower))


def test_amp_decoding_keeps_covariance_and_likelihood_float32() -> None:
    pair, batch = build_ball_refiner_2d(config()), inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = pair.run(batch)
        covariance = result.covariance
        log_prob = result.log_prob(torch.full((2, 5, 2), 0.5))
    assert result.means.dtype == torch.float32
    assert covariance.dtype == torch.float32
    assert log_prob.dtype == torch.float32
    assert torch.linalg.eigvalsh(covariance).min() > 0


@pytest.mark.parametrize(
    "change",
    [
        {"timestamps_seconds": torch.zeros(2, 5)},
        {"timestamps_seconds": torch.full((2, 5), float("nan"))},
        {"pose_uv": torch.ones(2, 5, 2, 17, 2)},  # Only elbows/wrists at this boundary.
        {"pose_confidence": torch.full((2, 5, 2, 4), 1.1)},
        {"court_uv": torch.zeros(2, 5, 5, 2)},  # Static prior cannot have a time axis.
        {"pose_valid": torch.zeros(2, 5, 2, 4)},  # Float mask rejected.
    ],
)
def test_bad_context_or_timeline_fails_at_adapter(change) -> None:
    pair = build_ball_refiner_2d(config())
    with pytest.raises(ValueError):
        pair.build_call(replace(inputs(), **change))


def test_corrupt_detector_evidence_is_not_silently_repaired() -> None:
    batch, adapter = inputs(), Refiner2DAdapter(config())
    c = batch.candidates
    for broken in (
        replace(c, scores=torch.zeros_like(c.scores)),
        replace(c, patch_valid=torch.zeros_like(c.patch_valid)),
        replace(c, coords=torch.full_like(c.coords, float("nan"))),
        replace(c, coords=c.coords.double()),
    ):
        with pytest.raises(ValueError):
            adapter.build_call(replace(batch, candidates=broken))


def test_upstream_detector_contract_covers_weak_boundary_and_empty_peaks() -> None:
    heatmaps = torch.zeros(2, 5, 7, 9)
    heatmaps[:, 0, 0, 0] = 0.01  # Weak corner peak with an out-of-image patch.
    heatmaps[:, 1, 3, 4] = 0.8
    heatmaps[:, 2] = 0.2  # Flat map: no contrastive peaks.
    candidates = decode_candidates(
        heatmaps,
        config=BallCandidateConfig(max_candidates=3, patch_size=3, nms_kernel=3),
        subpixel_refine=True,
    )
    assert candidates.valid[:, 0, 0].all()
    assert not candidates.patch_valid[:, 0, 0].all()
    assert not candidates.valid[:, 2:].any()
    pair = build_ball_refiner_2d(config())
    pair.model.eval()
    prediction = pair.run(replace(inputs(), candidates=candidates))
    assert prediction.means.shape[:2] == heatmaps.shape[:2]
    assert torch.isfinite(prediction.covariance).all()


def test_extreme_head_values_remain_positive_definite_and_differentiable() -> None:
    cfg = config()
    raw_output = (
        torch.linspace(-100, 100, cfg.components * 6 + 1)
        .reshape(1, 1, -1)
        .requires_grad_()
    )
    prediction = Refiner2DAdapter(cfg).decode_output(raw_output)
    assert torch.linalg.eigvalsh(prediction.covariance).min() > 0
    loss = -prediction.log_prob(torch.tensor([[[0.2, 0.7]]])).sum()
    loss.backward()
    assert torch.isfinite(raw_output.grad).all()
    with pytest.raises(ValueError):
        Refiner2DAdapter(cfg).decode_output(torch.full_like(raw_output, float("nan")))


def test_state_dict_roundtrip_preserves_distribution(tmp_path) -> None:
    cfg, batch = config(), inputs()
    original = build_ball_refiner_2d(cfg)
    original.model.eval()
    path = tmp_path / "model.pt"
    torch.save(original.model.state_dict(), path)
    restored = build_ball_refiner_2d(cfg)
    restored.model.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored.model.eval()
    a, b = original.run(batch), restored.run(batch)
    for left, right in (
        (a.means, b.means),
        (a.covariance, b.covariance),
        (a.weights, b.weights),
        (a.presence_probability, b.presence_probability),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
