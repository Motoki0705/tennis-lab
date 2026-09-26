"""Identity purity, permutation symmetry, metric learning and global matching."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.tasks.plcs.data.tracked_slots import FixedTrackRegistry
from src.tasks.plcs.model_io.track_matching import match_track_embeddings
from src.tasks.plcs.models.components.track_query import keep_mask
from src.tasks.plcs.models.court_side_model import CourtSideModel
from src.tasks.plcs.models.person_tokens import PersonModelConfig
from src.tasks.plcs.models.player_reid_model import PlayerReIDModel
from src.tasks.plcs.training.reid_losses import reid_loss


def inputs():
    torch.manual_seed(4)
    return dict(human_kp=torch.rand(2, 3, 7, 4, 17, 2), human_vis=torch.ones(2, 3, 7, 4, 17, dtype=torch.bool),
        court_kp=torch.rand(2, 3, 7, 14, 2), court_vis=torch.ones(2, 3, 7, 14, dtype=torch.bool),
        padding_mask=torch.zeros(2, 3, 7, dtype=torch.bool))


def model_config():
    return PersonModelConfig(hidden_dim=32, num_heads=4, ffn_dim=64, num_stages=2, rope_dim=8, dropout=0.)


def test_slots_keep_id_across_missingness_reentry_and_windows():
    registry = FixedTrackRegistry(("a",), num_slots=4)
    points = torch.rand(1, 3, 2, 17, 2)
    ids = torch.tensor([[[80, 4], [-1, 80], [4, 80]]])
    visible = ids.ge(0)[..., None].expand(1, 3, 2, 17)
    out = registry.pack(points, visible, ids)
    assert out.local_track_ids.tolist() == [[4, 80, -1, -1]]
    torch.testing.assert_close(out.keypoints[0, 0, 0], points[0, 0, 1])
    torch.testing.assert_close(out.keypoints[0, 2, 0], points[0, 2, 0])
    assert not out.visibility[0, 1, 0].any()
    for identity, slot in ((12, 2), (600, 3), (4, 0)):
        next_out = registry.pack(points[:, :1, :1], torch.ones(1, 1, 1, 17, dtype=torch.bool), torch.tensor([[identity]]))
        assert next_out.local_track_ids[0, slot] == identity
    with pytest.raises(ValueError, match="cumulative"):
        registry.pack(points[:, :1, :1], torch.ones(1, 1, 1, 17, dtype=torch.bool), torch.tensor([[900]]))
    assert len(registry.assignments[0]) == 4


def test_capacity_is_per_camera_and_same_number_is_camera_local():
    points = torch.rand(2, 2, 4, 17, 2)
    visible = torch.ones(points.shape[:-1], dtype=torch.bool)
    out = FixedTrackRegistry(("a", "b")).pack(points, visible, torch.tensor([[1, 2, 3, 4], [4, 1, 99, 8]]))
    assert (out.local_track_ids >= 0).sum() == 8
    with pytest.raises(ValueError, match="multiple detections"):
        FixedTrackRegistry(("a", "b")).pack(points, visible, torch.ones(2, 4, dtype=torch.int64))


def test_attention_axes_feedback_and_finite_empty_tracks():
    model = PlayerReIDModel(model_config())
    seen = []
    hooks = [model.stages[0].temporal.register_forward_pre_hook(lambda m, a: seen.append(a[0].shape)),
             model.stages[0].queries.register_forward_pre_hook(lambda m, a: seen.append(a[0].shape))]
    batch = inputs()
    output = model(**batch)
    for hook in hooks:
        hook.remove()
    assert seen == [torch.Size([24, 8, 32]), torch.Size([2, 12, 32])]
    loss = reid_loss(output, torch.arange(4)[None, None].expand(2, 3, 4), temperature=.1, margin=.5)["loss"]
    loss.backward()
    assert model.track_query.grad is not None and model.track_query.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    batch["human_vis"].zero_()
    empty = model(**batch)
    assert not empty["track_valid"].any() and not empty["track_embedding"].any()
    assert set(empty) == {"track_embedding", "track_valid"}
    assert not hasattr(model, "player_head")


def test_invalid_row_safety_preserves_visible_attention_and_gradients():
    valid = torch.tensor([[True, False, True], [False, False, False]])
    torch.manual_seed(17)
    query = torch.randn(2, 1, 3, 8, requires_grad=True)
    dense_mask = valid[:, :, None] & valid[:, None, :]
    ordinary = F.scaled_dot_product_attention(query, query, query, attn_mask=dense_mask[:, None])
    safe = F.scaled_dot_product_attention(query, query, query, attn_mask=keep_mask(valid)[:, None])
    safe = safe.masked_fill(~valid[:, None, :, None], 0)
    torch.testing.assert_close(safe, ordinary)
    original_gradient = torch.autograd.grad(ordinary.square().sum(), query, retain_graph=True)[0]
    safe_gradient = torch.autograd.grad(safe.square().sum(), query)[0]
    torch.testing.assert_close(safe_gradient, original_gradient)
    assert keep_mask(valid).any(-1).all()


def test_camera_and_independent_slot_permutations_and_missing_values():
    model = PlayerReIDModel(model_config()).eval()
    x = inputs()
    x["human_vis"][:, :, :2] = False
    original = model(**x)
    x["human_kp"][:, :, :2] = float("nan")
    for key, value in model(**x).items():
        torch.testing.assert_close(value, original[key])
    orders = torch.tensor([[1, 3, 0, 2], [2, 0, 3, 1], [3, 1, 2, 0]])
    moved = dict(x)
    for name in ("human_kp", "human_vis"):
        moved[name] = torch.stack([x[name][:, view, :, order] for view, order in enumerate(orders)], dim=1)
    output = model(**moved)
    for view, order in enumerate(orders):
        torch.testing.assert_close(output["track_embedding"][:, view], original["track_embedding"][:, view, order], atol=2e-6, rtol=2e-5)
    camera_order = torch.tensor([2, 0, 1])
    reordered = model(**{name: value[:, camera_order] for name, value in x.items()})
    torch.testing.assert_close(reordered["track_embedding"], original["track_embedding"][:, camera_order], atol=2e-6, rtol=2e-5)


def test_side_is_a_separate_model_with_permutation_invariant_person_pooling():
    reid = PlayerReIDModel(model_config())
    side = CourtSideModel(model_config()).eval()
    assert not {id(p) for p in side.parameters()} & {id(p) for p in reid.parameters()}
    x = inputs()
    reference = torch.tensor([0, 1])
    a = side(**x, reference_view_index=reference)
    perm = torch.tensor([3, 1, 0, 2])
    y = {**x, "human_kp": x["human_kp"][:, :, :, perm], "human_vis": x["human_vis"][:, :, :, perm]}
    b = side(**y, reference_view_index=reference)
    assert set(a) == {"side_logits"}
    torch.testing.assert_close(a["side_logits"], b["side_logits"], atol=1e-5, rtol=1e-5)


def test_pair_loss_uses_correspondence_not_numeric_labels():
    z = torch.eye(2)[None, None].expand(1, 3, 2, 2).clone().requires_grad_()
    output = dict(track_embedding=z, track_valid=torch.ones(1, 3, 2, dtype=torch.bool))
    identities = torch.tensor([[[12, 99], [12, 99], [12, 99]]])
    def loss(ids):
        return reid_loss(output, ids, temperature=.1, margin=.5)["loss"]
    assert loss(identities) < .01
    torch.testing.assert_close(loss(identities), loss(identities * 7))
    bad = identities.clone()
    bad[:, 1] = bad[:, 1].flip(-1)
    assert loss(bad) > 2
    unlabeled = identities.clone()
    unlabeled[:, 0, 0] = -1
    with pytest.raises(ValueError, match="person identity"):
        loss(unlabeled)
    output["track_valid"].zero_()
    empty = loss(identities)
    assert empty == 0
    empty.backward()
    assert torch.isfinite(z.grad).all()


def test_matching_camera_exclusivity_transitivity_and_singletons():
    z = torch.stack((torch.eye(3)[:2], torch.eye(3)[[1, 0]], torch.eye(3)[[0, 2]]))
    ids = match_track_embeddings(z, torch.ones(3, 2, dtype=torch.bool), threshold=.5)
    assert ids[0, 0] == ids[1, 1] == ids[2, 0]
    assert ids[0, 1] == ids[1, 0]
    assert ids[2, 1] not in ids[:2].unique()
    for camera in ids:
        assert len(camera.unique()) == 2
    angles = torch.tensor([0., 40., 80.]) * math.pi / 180
    triangle = torch.stack((angles.cos(), angles.sin()), -1)[:, None]
    result = match_track_embeddings(triangle, torch.ones(3, 1, dtype=torch.bool), threshold=.6)
    assert len(result.unique()) == 2  # AB, BC supported but AC is not: no three-way merge.
    independent = torch.eye(8).reshape(2, 4, 8)
    result = match_track_embeddings(independent, torch.ones(2, 4, dtype=torch.bool), threshold=.5)
    assert len(result.unique()) == 8  # No scene-wide four-person limit.


def test_matching_result_equivariance_uses_partitions_not_label_numbers():
    z = F.normalize(torch.tensor([[[1., 0.], [0., 1.]], [[.9, .1], [.1, .9]], [[1., 0.], [0., 1.]]]), dim=-1)
    valid = torch.ones(3, 2, dtype=torch.bool)
    a = match_track_embeddings(z, valid, threshold=.5)
    b = match_track_embeddings(z.flip(0).flip(1), valid, threshold=.5).flip(0).flip(1)
    assert torch.equal(a.flatten()[:, None] == a.flatten()[None], b.flatten()[:, None] == b.flatten()[None])
