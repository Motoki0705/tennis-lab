import pytest
import torch

from src.tasks.base.model_io.association_decoding import decode_association
from src.tasks.base.models.view_association import (
    ViewQueryAssociationModel,
    ViewQueryModelConfig,
)
from src.tasks.base.training.association_losses import association_loss
from src.tennis_scene.pipeline.association_state import stitch_overlap_ids


def inputs(joints=17):
    torch.manual_seed(1)
    return dict(
        object_uv=torch.rand(2, 3, 7, 4, joints, 2),
        object_vis=torch.ones(2, 3, 7, 4, joints, dtype=torch.bool),
        court_kp=torch.rand(2, 3, 7, 14, 2),
        court_vis=torch.ones(2, 3, 7, 14, dtype=torch.bool),
        padding_mask=torch.zeros(2, 3, 7, dtype=torch.bool),
        reference_view_index=torch.tensor([0, 2]),
    )


def model(j=17):
    return ViewQueryAssociationModel(
        ViewQueryModelConfig(
            hidden_dim=48,
            num_heads=4,
            ffn_dim=96,
            num_stages=2,
            rope_dim=12,
            dropout=0.0,
        ),
        num_keypoints=j,
    )


@pytest.mark.parametrize("joints", [1, 17])
def test_forward_backward_and_temporal_query_spatial_width(joints):
    m = model(joints)
    widths = []

    def hook(module, args):
        widths.append(args[0].shape[1])

    handle = m.stages[0].spatial.register_forward_pre_hook(hook)
    x = inputs(joints)
    out = m(**x)
    handle.remove()
    assert widths == [3]  # Only V; no spatial Q.
    assert out["side_logits"].shape == (2, 3)
    assert out["object_id_logits"].shape == (2, 3, 7, 4, 11)
    assert not any(
        "position_head" in name or "slot_embeddings" in name
        for name, _ in m.named_parameters()
    )
    (
        out["side_logits"].square().mean() + out["object_id_logits"].square().mean()
    ).backward()
    assert m.view_query.grad is not None and m.view_query.grad.abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


def test_local_slot_permutation_equivariance_and_missing_value_invariance():
    m = model().eval()
    x = inputs()
    p = torch.tensor([2, 0, 3, 1])
    a = m(**x)
    y = {
        **x,
        "object_uv": x["object_uv"][:, :, :, p],
        "object_vis": x["object_vis"][:, :, :, p],
    }
    b = m(**y)
    torch.testing.assert_close(a["side_logits"], b["side_logits"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        a["object_id_logits"][:, :, :, p], b["object_id_logits"], atol=1e-5, rtol=1e-5
    )
    x["object_vis"][:, :, :2] = False
    x["court_vis"][:, :, :2] = False
    a = m(**x)
    x["object_uv"][:, :, :2] = 999
    x["court_kp"][:, :, :2] = -999
    b = m(**x)
    for k in a:
        torch.testing.assert_close(a[k], b[k])


def test_global_matching_penalizes_cross_view_id_swaps_and_ignores_unobserved():
    target = torch.tensor([[[[0, 1]], [[0, 1]]]])
    observed = torch.ones_like(target, dtype=torch.bool)
    good = torch.tensor(
        [
            [
                [[[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]]],
                [[[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]]],
            ]
        ]
    )
    side = torch.tensor([[False, True]])

    def loss(logits, ids=target):
        return association_loss(
            {"object_id_logits": logits, "side_logits": torch.tensor([[-9.0, 9.0]])},
            ids,
            observed,
            side,
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([0]),
        )["loss"]

    assert loss(good) < 0.001
    torch.testing.assert_close(loss(good), loss(good, 1 - target))
    swapped = good.clone()
    swapped[:, 1] = swapped[:, 1].flip(-2)
    assert loss(swapped) > 4
    observed[:, :, 0, 1] = False
    assert loss(good) < 0.001


def test_unique_decoding_rejects_non_targets_and_fixes_reference_side():
    output = {
        "object_id_logits": torch.tensor(
            [[[[[10.0, 0.0, -10.0], [9.0, 8.0, -10.0], [-1.0, -2.0, 10.0]]]]]
        ),
        "side_logits": torch.tensor([[9.0]]),
    }
    decoded = decode_association(
        output,
        observed=torch.ones(1, 1, 1, 3, dtype=torch.bool),
        reference=torch.tensor([0]),
        view_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    assert decoded["object_ids"].tolist() == [[[[0, 1, -1]]]]
    assert not decoded["view_half_turns"].any()


def test_overlap_matches_once_across_all_views():
    old = torch.tensor([[[5, 8]], [[5, 8]]])
    new = torch.tensor([[[1, 0]], [[1, 0]]])
    mapping, next_id = stitch_overlap_ids(old, new, next_identity=9)
    assert mapping.tolist() == [8, 5] and next_id == 9


def test_false_positive_and_empty_observation_losses_are_finite():
    prediction = {
        "object_id_logits": torch.randn(1, 2, 2, 4, 11, requires_grad=True),
        "side_logits": torch.randn(1, 2, requires_grad=True),
    }
    identity = torch.full((1, 2, 2, 4), -1, dtype=torch.long)
    observed = torch.ones_like(identity, dtype=torch.bool)
    losses = association_loss(
        prediction,
        identity,
        observed,
        torch.tensor([[False, True]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([0]),
    )
    losses["loss"].backward()
    assert torch.isfinite(prediction["object_id_logits"].grad).all()
    losses = association_loss(
        prediction,
        identity,
        observed & False,
        torch.tensor([[False, True]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([0]),
    )
    assert losses["identity_loss"] == 0 and torch.isfinite(losses["loss"])


def test_downstream_groups_ids_without_assuming_shared_local_slots():
    import numpy as np

    from src.tennis_scene.pipeline.components.view_association import (
        ViewAssociationResult,
    )

    ids = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
    uv: np.ndarray = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 1, 2)
    vis = np.ones(uv.shape[:-1], dtype=np.bool_)
    result = ViewAssociationResult(
        ("left", "right"), "left", (False, True), ids, np.array([-1.0, 1.0], np.float32)
    )
    unique, grouped, mask = result.group_observations(uv, vis)
    assert unique.tolist() == [0, 1]
    np.testing.assert_array_equal(grouped[0, 1, 0], uv[1, 0, 1])
    assert mask.all()
