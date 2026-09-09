"""Reference readout, selector frequencies and padding isolation tests."""

from __future__ import annotations

import pytest
import torch

from src.tasks.plcs.model_io.attention_masks import prepare_axial_attention_masks
from src.tasks.plcs.models.components.heads import TemporalDecomposedCanonicalPoseHead
from src.tasks.plcs.models.plcs_multiview_axial_reference_model import (
    PLCSMultiViewAxialReferenceModel,
)
from src.utils.models import precompute_freqs_cis_nd


def _model(layers: int = 1) -> PLCSMultiViewAxialReferenceModel:
    model = PLCSMultiViewAxialReferenceModel(
        hidden_dim=32,
        num_layers=layers,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        rope_dim=8,
        rope_theta_time=1000.0,
        rope_theta_camera=1000.0,
        ffn_type="swiglu",
        predict_canonical_pose=True,
        max_views=4,
        max_seq_len=8,
        invisible_init_std=0.02,
        num_court_tokens=14,
        canonical_pose_readout="temporal_decomposition",
    )
    model.eval()
    return model


def _inputs() -> dict[str, torch.Tensor]:
    mask = torch.zeros(2, 4, 8, dtype=torch.bool)
    mask[0, 3] = True
    mask[1, :, 6:] = True
    camera, time = prepare_axial_attention_masks(mask)
    return dict(
        human_kp=torch.rand(2, 4, 8, 17, 2),
        court_kp=torch.rand(2, 4, 8, 14, 2),
        human_vis=torch.ones(2, 4, 8, 17),
        court_vis=torch.ones(2, 4, 8, 14),
        padding_mask=mask,
        camera_attention_mask=camera,
        time_attention_mask=time,
        reference_view_index=torch.tensor([2, 1]),
    )


def test_selector_frequencies_match_explicit_coordinates() -> None:
    model = _model()
    for reference in range(4):
        positions = model._build_token_positions(seq_len=8, n_cams=4)
        selector = (torch.arange(4) != reference)[None, :, None].expand(8, 4, 1)
        expected = precompute_freqs_cis_nd(
            8,
            torch.cat((positions, selector.long()), -1),
            base=(1000.0, 1000.0, 1000.0),
        )
        torch.testing.assert_close(model.token_freqs_cis[reference], expected)


def test_readout_uses_each_sample_reference_not_first_camera() -> None:
    model = _model(layers=0)
    assert isinstance(model.canonical_pose_head, TemporalDecomposedCanonicalPoseHead)
    batch = _inputs()
    observed: list[torch.Tensor] = []
    hook = model.final_norm.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone())
    )
    result = model(**batch)
    hook.remove()
    for b, ref in enumerate([2, 1]):
        expected = model.group_embed(
            batch["court_kp"][b, ref],
            batch["human_kp"][b, ref],
            ~batch["padding_mask"][b, ref],
        )
        torch.testing.assert_close(observed[0][b], expected)
    assert result["canonical_pose"].shape == (2, 8, 17, 3)


def test_padded_camera_cannot_change_valid_reference_prediction() -> None:
    model = _model()
    inputs = _inputs()
    expected = model(**inputs)
    inputs["human_kp"][0, 3] = 1000
    inputs["court_kp"][0, 3] = -1000
    actual = model(**inputs)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key])
    loss = sum(value.square().mean() for value in actual.values())
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_reference_marker_rejects_physical_state() -> None:
    model = _model()
    state = model.state_dict()
    del state["_axial_reference_contract_marker"]
    with pytest.raises(RuntimeError, match="_axial_reference_contract_marker"):
        model.load_state_dict(state)


def test_three_view_forward_can_compile_without_selector_graph_breaks() -> None:
    model = _model()
    inputs = _inputs()
    for key in ("human_kp", "court_kp", "human_vis", "court_vis", "padding_mask"):
        inputs[key] = inputs[key][:, :3]
    camera, time = prepare_axial_attention_masks(inputs["padding_mask"])
    inputs.update(camera_attention_mask=camera, time_attention_mask=time)
    expected = model(**inputs)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(**inputs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
