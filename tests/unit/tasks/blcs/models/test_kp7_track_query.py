from __future__ import annotations

from typing import cast

import torch

from src.tasks.base.data.court_peaks import CourtObservationProfile
from src.tasks.blcs.configuration import TrackQueryModelConfig
from src.tasks.blcs.model_io import TrackQueryModelIOAdapter
from src.tasks.blcs.models.blcs_track_query_model import BLCSTrackQueryModel
from src.tasks.blcs.models.components.observation_fusion import (
    KP7TrackObservationFusion,
    LinearTrackObservationFusion,
)


def _config(
    profile: CourtObservationProfile = "kp7_reference",
    *,
    kp7_camera_rope_enabled: bool = False,
) -> TrackQueryModelConfig:
    return TrackQueryModelConfig(
        name="blcs_track_query",
        hidden_dim=32,
        num_heads=4,
        num_stages=1,
        ffn_dim=64,
        num_queries=3,
        rope_dim=8,
        dropout=0.0,
        role_rope_enabled=True,
        mask_invisible_observations=True,
        invisible_init_std=0.02,
        observation_fusion="linear",
        point_fusion=None,
        court_observation_profile=profile,
        kp7_camera_rope_enabled=kp7_camera_rope_enabled,
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(719)
    shape = (1, 2, 2, 2)
    return {
        "ball_uv": torch.rand(*shape, 2),
        "ball_score": torch.rand(*shape),
        "ball_visible": torch.ones(*shape, dtype=torch.bool),
        "candidate_mask": torch.ones(*shape, dtype=torch.bool),
        "frame_mask": torch.ones(1, 2, dtype=torch.bool),
        "view_mask": torch.ones(1, 2, dtype=torch.bool),
        "reference_view_index": torch.tensor([1]),
        "court_peak_uv": torch.rand(1, 2, 2, 7, 3, 2),
        "court_peak_score": torch.rand(1, 2, 2, 7, 3),
        "court_peak_covariance": torch.eye(2)
        .reshape(1, 1, 1, 1, 1, 2, 2)
        .expand(1, 2, 2, 7, 3, 2, 2)
        * 1.0e-4,
        "court_peak_valid": torch.ones(1, 2, 2, 7, 3, dtype=torch.bool),
    }


def _run(
    model: BLCSTrackQueryModel, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=3,
        presence_threshold=0.5,
        mask_invisible_observations=True,
        court_observation_profile="kp7_reference",
    )
    return cast(
        "dict[str, torch.Tensor]",
        model(**dict(adapter.build_call(batch).kwargs)),
    )


def test_kp7_model_is_invariant_to_class_internal_peak_order() -> None:
    model = BLCSTrackQueryModel(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        expected = _run(model, batch)

        peak_permuted = {key: value.clone() for key, value in batch.items()}
        permutation = torch.tensor([2, 0, 1])
        peak_permuted["court_peak_uv"] = peak_permuted["court_peak_uv"][
            ..., permutation, :
        ]
        peak_permuted["court_peak_score"] = peak_permuted["court_peak_score"][
            ..., permutation
        ]
        peak_permuted["court_peak_covariance"] = peak_permuted[
            "court_peak_covariance"
        ][..., permutation, :, :]
        peak_permuted["court_peak_valid"] = peak_permuted["court_peak_valid"][
            ..., permutation
        ]
        peak_actual = _run(model, peak_permuted)

    for key in expected:
        torch.testing.assert_close(peak_actual[key], expected[key], atol=2e-6, rtol=2e-6)


def test_kp7_model_has_no_detection_index_shortcut() -> None:
    model = BLCSTrackQueryModel(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        expected = _run(model, batch)
        reordered = {key: value.clone() for key, value in batch.items()}
        swap = torch.tensor([1, 0])
        for key in ("ball_uv", "ball_score", "ball_visible", "candidate_mask"):
            reordered[key] = reordered[key].index_select(3, swap)
        actual = _run(model, reordered)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-6)


def test_kp7_view_permutation_transforms_reference_and_erases_only_camera_rope() -> None:
    model = BLCSTrackQueryModel(_config()).eval()
    assert not model.kp7_camera_rope_enabled
    batch = _batch()
    with torch.no_grad():
        expected = _run(model, batch)
        reordered = {key: value.clone() for key, value in batch.items()}
        swap = torch.tensor([1, 0])
        for key in ("ball_uv", "ball_score", "ball_visible", "candidate_mask"):
            reordered[key] = reordered[key].index_select(1, swap)
        for key in (
            "court_peak_uv",
            "court_peak_score",
            "court_peak_covariance",
            "court_peak_valid",
        ):
            reordered[key] = reordered[key].index_select(1, swap)
        reordered["reference_view_index"] = torch.tensor([0])
        reorder_actual = _run(model, reordered)
    for key in expected:
        torch.testing.assert_close(reorder_actual[key], expected[key], atol=2e-6, rtol=2e-6)


def test_kp7_camera_rope_policy_has_an_explicit_ablation_switch() -> None:
    assert not BLCSTrackQueryModel(_config()).kp7_camera_rope_enabled
    assert BLCSTrackQueryModel(
        _config(kp7_camera_rope_enabled=True)
    ).kp7_camera_rope_enabled


def test_kp7_reference_role_changes_without_changing_camera_identity() -> None:
    model = BLCSTrackQueryModel(_config()).eval()
    batch = _batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["reference_view_index"] = torch.tensor([0])
    with torch.no_grad():
        expected = _run(model, batch)
        actual = _run(model, changed)
    assert not torch.equal(actual["position"], expected["position"])


def test_reference_context_survives_fully_missing_reference_and_backward() -> None:
    model = BLCSTrackQueryModel(_config()).train()
    batch = _batch()
    batch["ball_visible"][:, 1] = False
    batch["ball_visible"][:, 0, :, 0] = False
    adapter = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=3,
        presence_threshold=0.5,
        mask_invisible_observations=True,
        court_observation_profile="kp7_reference",
    )
    call = adapter.build_call(batch)
    state = call.kwargs["observation_state_valid"]
    assert isinstance(state, torch.Tensor)
    assert bool(state[:, :, 1, 0].all())
    assert not bool(state[:, :, 1, 1:].any())
    assert not bool(state[:, :, 0, 0].any())
    assert bool(state[:, :, 0, 1].all())
    assert state.shape[-1] == batch["ball_uv"].shape[3]

    output = model(**dict(call.kwargs))
    (output["position"].square().mean() + output["presence_logits"].square().mean()).backward()
    assert model.reference_conditioning is not None
    assert model.reference_conditioning.reference_delta.grad is not None


def test_kp7_tokens_execute_existing_spatial_and_temporal_blocks() -> None:
    model = BLCSTrackQueryModel(_config()).eval()
    batch = _batch()
    calls: dict[str, list[tuple[int, ...]]] = {"spatial": [], "temporal": []}

    spatial_handle = model.spatial_blocks[0].register_forward_pre_hook(
        lambda _module, args: calls["spatial"].append(tuple(args[0].shape))
    )
    temporal_handle = model.temporal_blocks[0].register_forward_pre_hook(
        lambda _module, args: calls["temporal"].append(tuple(args[0].shape))
    )
    try:
        with torch.no_grad():
            _run(model, batch)
    finally:
        spatial_handle.remove()
        temporal_handle.remove()

    assert calls == {
        "spatial": [(2, 7, 32)],  # B*T, Q+V*D, H
        "temporal": [(3, 2, 32)],  # B*Q, T, H
    }


def test_no_reference_profile_omits_reference_value_parameter() -> None:
    model = BLCSTrackQueryModel(_config("kp7_no_reference"))
    assert model.reference_conditioning is None


def test_observation_architecture_is_selected_once_at_blcs_construction() -> None:
    kp7_model = BLCSTrackQueryModel(_config("kp7_reference"))
    baseline_model = BLCSTrackQueryModel(_config("kp14_reference_baseline"))

    assert isinstance(kp7_model.observation_encoder, KP7TrackObservationFusion)
    assert isinstance(baseline_model.observation_encoder, LinearTrackObservationFusion)
    assert kp7_model.num_court_tokens is None
    assert baseline_model.num_court_tokens == 14
