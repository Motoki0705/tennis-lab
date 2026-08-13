from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf

from src.tasks.base.data.court_peaks import CourtObservationProfile
from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.model_io import PLCSTrackQueryIOAdapter
from src.tasks.plcs.models.components.observation_fusion import (
    KP7PlayerObservationFusion,
    KP14PlayerObservationFusion,
)
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel


def _model(
    profile: CourtObservationProfile = "kp7_reference",
    *,
    kp7_camera_rope_enabled: bool = False,
) -> PLCSTrackQueryModel:
    raw = OmegaConf.load(Path("src/tasks/plcs/configs/model/track_query.yaml"))
    raw.court_observation_profile = profile
    raw.kp7_camera_rope_enabled = kp7_camera_rope_enabled
    config = PLCSModelConfig.from_mapping(
        cast("dict[str, object]", OmegaConf.to_container(raw, resolve=True))
    )
    return PLCSTrackQueryModel(config)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(719)
    human = torch.rand(1, 2, 2, 2, 17, 2)
    return {
        "human_kp": human,
        "joint_visibility": torch.ones_like(human[..., 0], dtype=torch.bool),
        "detection_score": torch.ones(1, 2, 2, 2),
        "detection_mask": torch.ones(1, 2, 2, 2, dtype=torch.bool),
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


def _adapter(
    profile: CourtObservationProfile = "kp7_reference",
) -> PLCSTrackQueryIOAdapter:
    return PLCSTrackQueryIOAdapter(
        model_type=PLCSTrackQueryModel,
        num_queries=4,
        num_court_tokens=14,
        num_joints=17,
        mask_invisible_observations=True,
        court_observation_profile=profile,
    )


def test_kp7_player_model_forward_backward_and_missing_reference_context() -> None:
    model = _model().train()
    batch = _batch()
    batch["detection_mask"][:, 1] = False
    batch["joint_visibility"][:, 1] = False
    batch["detection_score"][:, 1] = 0
    batch["detection_mask"][:, 0, :, 0] = False
    batch["joint_visibility"][:, 0, :, 0] = False
    batch["detection_score"][:, 0, :, 0] = 0
    call = _adapter().build_call(batch)
    state = call.kwargs["camera_state_valid"]
    assert isinstance(state, torch.Tensor)
    assert bool(state[:, :, 1, 0].all())
    assert not bool(state[:, :, 1, 1:].any())
    assert not bool(state[:, :, 0, 0].any())
    assert bool(state[:, :, 0, 1].all())
    assert state.shape[-1] == batch["human_kp"].shape[3]

    output = model(**dict(call.kwargs))
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()
    assert model.reference_conditioning is not None
    assert model.reference_conditioning.reference_delta.grad is not None


def test_visibility_aware_player_geometry_ignores_hidden_joint_coordinates() -> None:
    model = _model().eval()
    batch = _batch()
    batch["joint_visibility"][..., 0] = False
    changed = {key: value.clone() for key, value in batch.items()}
    changed["human_kp"][..., 0, :] = 1.0

    with torch.no_grad():
        expected = model(**dict(_adapter().build_call(batch).kwargs))
        actual = model(**dict(_adapter().build_call(changed).kwargs))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_kp7_player_model_is_invariant_to_class_internal_peak_order() -> None:
    model = _model().eval()
    batch = _batch()
    reordered = {key: value.clone() for key, value in batch.items()}
    permutation = torch.tensor([2, 0, 1])
    reordered["court_peak_uv"] = reordered["court_peak_uv"][
        ..., permutation, :
    ]
    reordered["court_peak_score"] = reordered["court_peak_score"][..., permutation]
    reordered["court_peak_covariance"] = reordered["court_peak_covariance"][
        ..., permutation, :, :
    ]
    reordered["court_peak_valid"] = reordered["court_peak_valid"][..., permutation]
    with torch.no_grad():
        expected = model(**dict(_adapter().build_call(batch).kwargs))
        actual = model(**dict(_adapter().build_call(reordered).kwargs))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-6)


def test_kp7_player_model_has_no_detection_index_shortcut() -> None:
    model = _model().eval()
    batch = _batch()
    reordered = {key: value.clone() for key, value in batch.items()}
    swap = torch.tensor([1, 0])
    for key in ("human_kp", "joint_visibility", "detection_score", "detection_mask"):
        reordered[key] = reordered[key].index_select(3, swap)
    with torch.no_grad():
        expected = model(**dict(_adapter().build_call(batch).kwargs))
        actual = model(**dict(_adapter().build_call(reordered).kwargs))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=5e-6, rtol=5e-6)


def test_kp7_player_view_permutation_transforms_reference_index() -> None:
    model = _model().eval()
    assert not model.kp7_camera_rope_enabled
    batch = _batch()
    reordered = {key: value.clone() for key, value in batch.items()}
    swap = torch.tensor([1, 0])
    for key in ("human_kp", "joint_visibility", "detection_score", "detection_mask"):
        reordered[key] = reordered[key].index_select(1, swap)
    for key in (
        "court_peak_uv",
        "court_peak_score",
        "court_peak_covariance",
        "court_peak_valid",
    ):
        reordered[key] = reordered[key].index_select(1, swap)
    reordered["reference_view_index"] = torch.tensor([0])
    with torch.no_grad():
        expected = model(**dict(_adapter().build_call(batch).kwargs))
        actual = model(**dict(_adapter().build_call(reordered).kwargs))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=5e-6, rtol=5e-6)


def test_kp7_player_camera_rope_policy_has_an_explicit_ablation_switch() -> None:
    assert not _model().kp7_camera_rope_enabled
    assert _model(kp7_camera_rope_enabled=True).kp7_camera_rope_enabled


def test_kp7_player_reference_role_changes_independently_of_view_slot() -> None:
    model = _model().eval()
    batch = _batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["reference_view_index"] = torch.tensor([0])
    with torch.no_grad():
        expected = model(**dict(_adapter().build_call(batch).kwargs))
        actual = model(**dict(_adapter().build_call(changed).kwargs))
    assert not torch.equal(actual["position"], expected["position"])


def test_kp7_player_tokens_execute_existing_spatial_and_temporal_blocks() -> None:
    model = _model().eval()
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
            model(**dict(_adapter().build_call(batch).kwargs))
    finally:
        spatial_handle.remove()
        temporal_handle.remove()

    assert calls == {
        "spatial": [(2, 8, 64)],  # B*T, Q+V*D, H
        "temporal": [(4, 2, 64)],  # B*Q, T, H
    }


def test_no_reference_player_profile_has_no_reference_conditioning() -> None:
    assert _model("kp7_no_reference").reference_conditioning is None


def test_observation_architecture_is_selected_once_at_plcs_construction() -> None:
    kp7_model = _model("kp7_reference")
    baseline_model = _model("kp14_reference_baseline")

    assert isinstance(kp7_model.observation_encoder, KP7PlayerObservationFusion)
    assert isinstance(baseline_model.observation_encoder, KP14PlayerObservationFusion)
    assert kp7_model.num_court_tokens is None
    assert baseline_model.num_court_tokens == 14
