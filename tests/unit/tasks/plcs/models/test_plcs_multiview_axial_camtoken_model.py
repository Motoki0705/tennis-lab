"""Unit tests for ``PLCSMultiViewAxialCamTokenModel`` (issue #576).

The model keeps the shared trunk of ``PLCSMultiViewAxialModel`` and only changes
the readout: the pose head reads the camera-0 token and the rotation head reads
the camera-1 token. These CPU tests pin that routing and the output shapes.
"""

from __future__ import annotations

import torch

from src.tasks.plcs.models import PLCSMultiViewAxialCamTokenModel
from src.utils.schema.player import NUM_HUMAN_KP


def _make_model(
    *, predict_canonical_pose: bool = False
) -> PLCSMultiViewAxialCamTokenModel:
    torch.manual_seed(0)
    return PLCSMultiViewAxialCamTokenModel(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        max_views=4,
        max_seq_len=16,
        predict_canonical_pose=predict_canonical_pose,
    )


def _make_inputs(model: PLCSMultiViewAxialCamTokenModel, *, n_cams: int):
    torch.manual_seed(1)
    b, t = 2, 5
    human_kp = torch.randn(b, n_cams, t, NUM_HUMAN_KP, 2)
    court_kp = torch.randn(b, n_cams, t, model.num_court_tokens, 2)
    return human_kp, court_kp


def test_output_shapes() -> None:
    model = _make_model(predict_canonical_pose=True).eval()
    human_kp, court_kp = _make_inputs(model, n_cams=3)
    with torch.no_grad():
        out = model(human_kp, court_kp)
    b, t = human_kp.shape[0], human_kp.shape[2]
    assert out["position"].shape == (b, t, 3)
    assert out["rotation"].shape == (b, t, 2)
    # Rotation head returns a unit-normalized (cos, sin).
    norms = out["rotation"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert out["canonical_pose"].shape == (b, t, NUM_HUMAN_KP, 3)


def _capture_head_inputs(model: PLCSMultiViewAxialCamTokenModel, human_kp, court_kp):
    captured: dict[str, torch.Tensor] = {}

    def _hook(name):
        def fn(_module, inputs, _output):
            captured[name] = inputs[0].detach().clone()

        return fn

    handles = [
        model.position_head.register_forward_hook(_hook("pose")),
        model.rotation_head.register_forward_hook(_hook("rot")),
    ]
    try:
        with torch.no_grad():
            model(human_kp, court_kp)
    finally:
        for h in handles:
            h.remove()
    return captured


def test_pose_and_rotation_read_distinct_camera_tokens() -> None:
    """With >=2 views the two heads must receive different readout tokens."""
    model = _make_model().eval()
    human_kp, court_kp = _make_inputs(model, n_cams=3)
    captured = _capture_head_inputs(model, human_kp, court_kp)
    assert not torch.allclose(captured["pose"], captured["rot"])


def test_single_view_falls_back_to_shared_token() -> None:
    """With a single view the rotation head reuses the camera-0 token."""
    model = _make_model().eval()
    human_kp, court_kp = _make_inputs(model, n_cams=1)
    captured = _capture_head_inputs(model, human_kp, court_kp)
    assert torch.allclose(captured["pose"], captured["rot"])
